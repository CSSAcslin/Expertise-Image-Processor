"""Fixed-tau window scan through the real module editor and result plot."""
from pathlib import Path
import csv
import io
import json
import os
import subprocess
import sys
import time
import zipfile
import httpx
import numpy as np
import tifffile
from playwright.sync_api import expect,sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/".test-output"/f"correlation-window-{int(time.time())}"
    folder.mkdir(parents=True)
    t=np.arange(31)[:,None,None];grid=np.arange(144).reshape(12,12)
    values=np.concatenate((20+grid+3*t,50+grid+2*t+t*t),axis=2).astype(np.uint16)
    tifffile.imwrite(folder/"source.tif",values,photometric="minisblack",metadata={"axes":"TYX"})
    url="http://127.0.0.1:8910"
    with (folder/"server.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,str(ROOT/"run_web.py"),"--port","8910","--no-browser","--workspace",str(folder)],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.1)
            else:raise AssertionError("isolated service did not start")
            with httpx.Client(base_url=url,timeout=45) as api,sync_playwright() as pw:
                state=api.get("/api/state").json();api.headers["X-EIP-Token"]=state["session_token"]
                response=api.post("/api/sources/open",json=dict(path=str(folder/"source.tif"),interval=.01));response.raise_for_status()
                browser=pw.chromium.launch(channel="msedge" if os.name=="nt" else "chromium",headless=True)
                page=browser.new_page(viewport=dict(width=1600,height=1000),accept_downloads=True)
                errors,computes=[],[]
                page.on("pageerror",lambda e:errors.append(str(e)))
                page.on("request",lambda r:computes.append(r.url) if r.method=="POST" and r.url.endswith("/api/jobs") else None)
                page.goto(url);expect(page.locator("#frame-info")).to_contain_text("31")
                page.locator("#flow-clear").click()
                def add(algorithm):
                    page.locator("#open-module-library").click();page.locator("[data-category-filter=all]").click()
                    page.locator(f"[data-add-module='{algorithm}']").click()
                def parameter(key,value):
                    control=page.locator(f"#general-module-fields [data-key='{key}']")
                    control.fill(str(value));control.press("Tab")
                add("alignment");add("point_extract");parameter("x",2);parameter("y",3)
                add("correlation_window")
                parameter("lag",-2);parameter("center",16);parameter("window_list","2,3,6,11,21")
                expect(page.locator("#general-module-title")).to_contain_text("平均窗长扫描")
                expect(page.locator("#general-module-fields [data-key=window_max]")).not_to_be_visible()
                page.locator("#job-name").fill("固定 τ 窗长曲线")
                with page.expect_response(lambda r:r.request.method=="POST" and r.url.endswith("/api/jobs")) as received:
                    page.locator("#run").click()
                assert received.value.status==200,received.value.text()
                identifier=received.value.json()["id"]
                for _ in range(300):
                    job=api.get(f"/api/jobs/{identifier}").json()
                    if job["status"] in {"succeeded","failed"}:break
                    time.sleep(.05)
                assert job["status"]=="succeeded",job
                expect(page.locator("#curve-toolbar")).to_be_visible()
                expect(page.locator("#split-curve")).to_contain_text("平均窗长 W")
                expect(page.locator("#curve-axis [value=time]")).to_be_disabled()
                windows=[2,3,6,11,21];expected=[]
                for window in windows:
                    start=15-(window-1)//2;stop=start+window
                    left,right=values[start:stop,3,2].astype(float),values[start-2:stop-2,3,14].astype(float)
                    expected.append(np.mean(left*right)/(left.mean()*right.mean()))
                comparison=api.post("/api/curves/compare",json=dict(items=[dict(job_id=identifier,source=0)],x="samples"));comparison.raise_for_status()
                assert comparison.json()["xs"]==windows
                np.testing.assert_allclose(comparison.json()["curves"][0]["values"],expected)
                with page.expect_download() as download:page.locator("#curve-export").click()
                download.value.save_as(str(folder/"comparison.zip"))
                with zipfile.ZipFile(folder/"comparison.zip") as archive:
                    rows=list(csv.reader(io.StringIO(archive.read("values.csv").decode("utf-8-sig"))))
                    assert [int(float(row[0])) for row in rows[1:]]==windows
                    np.testing.assert_allclose([float(row[1]) for row in rows[1:]],expected)
                    assert json.loads(archive.read("comparison.json"))["items"][0]["contract"]["sequence_axis"]=="W"
                page.locator("#curve-fit").click()
                page.screenshot(path=str(folder/"desktop.png"),animations="disabled")
                page.set_viewport_size(dict(width=1024,height=768));page.wait_for_timeout(150)
                expect(page.locator("#curve-toolbar")).to_be_visible()
                assert page.locator("#split-curve").bounding_box()["height"]>=150
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                page.screenshot(path=str(folder/"small.png"),animations="disabled")
                assert not errors,errors
                assert len(computes)==1,computes
                browser.close()
            print(json.dumps(dict(status="passed",output=str(folder),checks="module editor, signed tau, window coordinates, exact g2, full curve export, responsive plot, no hidden jobs")))
        finally:
            process.terminate()
            try:process.wait(10)
            except subprocess.TimeoutExpired:process.kill();process.wait(5)


if __name__=="__main__":main()
