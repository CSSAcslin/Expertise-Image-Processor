"""Focused 0.8.4 browser acceptance; uses real CPU/auto, never fabricates CUDA."""
from pathlib import Path
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
from playwright.sync_api import expect, sync_playwright

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from eip import __version__


def main():
    output=ROOT/".test-output"/f"gpu-pipeline-{int(time.time())}"
    output.mkdir(parents=True)
    values=(17*np.arange(121)[:,None,None]+np.arange(128).reshape(8,16)).astype(np.uint16)
    tifffile.imwrite(output/"source.tif",values,photometric="minisblack",metadata={"axes":"TYX"})
    url="http://127.0.0.1:8908"
    with (output/"server.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,str(ROOT/"run_web.py"),"--port","8908","--no-browser","--workspace",str(output)],
            cwd=ROOT,stdout=log,stderr=log,env=dict(os.environ,PYTHONIOENCODING="utf-8"),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.1)
            else:raise AssertionError("isolated server failed")
            with httpx.Client(base_url=url,timeout=45) as api,sync_playwright() as pw:
                state=api.get("/api/state").json()
                assert state["version"]==__version__
                api.headers["X-EIP-Token"]=state["session_token"]
                r=api.post("/api/sources/open",json=dict(path=str(output/"source.tif"),interval=.01));r.raise_for_status()
                browser=pw.chromium.launch(channel="msedge" if os.name=="nt" else "chromium",headless=True)
                page=browser.new_page(viewport=dict(width=1600,height=1000),accept_downloads=True)
                errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
                page.goto(url)
                expect(page.locator("#version")).to_have_text("v"+__version__)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 121")
                page.locator("#open-run-settings").click()
                for key,value in dict(memory_mb=256,preview_mb=32,cache_mb=16,gpu_mb=128).items():
                    page.locator(f"#resource-{key}").fill(str(value))
                page.locator("#resource-backend").select_option("auto")
                page.locator("#resource-save").click()
                expect(page.locator("#toast")).to_contain_text("已保存")
                page.screenshot(path=str(output/"resources.png"))
                page.locator("#run-settings-dialog [data-close-dialog]").click()
                page.locator("#flow-clear").click()

                def add(a):
                    page.locator("#open-module-library").click()
                    page.locator("[data-category-filter=all]").click()
                    page.locator(f"[data-add-module='{a}']").click()
                    expect(page.locator("[data-execution=backend] option[value=gpu]")).to_have_count(1)
                def parameter(k,v):
                    control=page.locator(f"#general-module-fields [data-key='{k}']")
                    if control.evaluate("e=>e.tagName")=="SELECT":control.select_option(str(v))
                    else:control.fill(str(v));control.press("Tab")
                add("time_smooth");parameter("window",9)
                add("time_statistics");parameter("mode","quantile");parameter("quantile",.3)
                add("global_statistics");parameter("mode","variance")
                page.locator("#job-name").fill("0.8.4 时序与全域统计")
                expect(page.locator("#module-status")).to_have_text("连接有效")
                page.locator("#run").click()
                for _ in range(450):
                    jobs=api.get("/api/jobs").json()
                    if jobs and jobs[0]["status"] in {"succeeded","failed","completed_with_errors"}:break
                    page.wait_for_timeout(100)
                assert jobs[0]["status"]=="succeeded",jobs
                job=api.get("/api/jobs/"+jobs[0]["id"]).json()
                item=job["run"]["results"][0]
                assert job["run"]["app_version"]==__version__
                assert item["contract"]["full_axes"]=="N"
                assert item["modules"]["resource_plan"][0]["temporal_strategy"]=="compensated_rolling_reduction"
                r=api.post(f"/api/jobs/{job['id']}/download-link",json=dict(file="array"));r.raise_for_status()
                numeric=api.get(r.json()["url"]);numeric.raise_for_status()
                np.testing.assert_allclose(np.load(io.BytesIO(numeric.content)),[307496.])
                page.locator("#open-data-library").click()
                page.locator(f"[data-manage-result='{job['id']}']").click()
                expect(page.locator("#split-curve svg")).to_be_visible()
                page.screenshot(path=str(output/"workbench-desktop.png"),full_page=True)
                with page.expect_download() as received:
                    page.locator("#managed-export").click()
                received.value.save_as(str(output/"run.zip"))
                with zipfile.ZipFile(output/"run.zip") as archive:
                    prefix=f"run_{job['id']}/"
                    record=json.loads(archive.read(prefix+"run.json"))
                    np.testing.assert_allclose(np.load(io.BytesIO(archive.read(prefix+"source_000/values.npy"))),[307496.])
                    assert record["spec"]["resources"]["backend"]=="auto"
                    assert "device_execution" in record["results"][0]["modules"]
                page.set_viewport_size(dict(width=1024,height=768));page.wait_for_timeout(200)
                page.screenshot(path=str(output/"workbench-small.png"),full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                assert not errors,errors
                devices=item["modules"]["devices"]
                report=dict(version=__version__,gpu_available=state["split_devices"]["gpu"],actual_devices=devices,
                    expected_variance=307496.,preview=True,numeric_export=True,package_export=True,javascript_errors=errors)
                (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
                print(json.dumps(dict(output=str(output),**report),ensure_ascii=False))
                browser.close()
        finally:
            process.terminate()
            try:process.wait(10)
            except subprocess.TimeoutExpired:process.kill();process.wait(5)


if __name__=="__main__":main()
