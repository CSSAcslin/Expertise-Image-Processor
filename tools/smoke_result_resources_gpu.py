"""Isolated real-browser lifecycle and resource UI acceptance checks."""
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
from playwright.sync_api import expect,sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/".test-output"/f"result-resources-{int(time.time())}"
    output.mkdir(parents=True)
    values=(np.arange(9*24*40).reshape(9,24,40)+20).astype(np.uint16)
    tifffile.imwrite(output/"raw.tif",values,photometric="minisblack",metadata={"axes":"TYX"})
    url="http://127.0.0.1:8903"
    with (output/"server.log").open("w",encoding="utf-8") as log:
        server=subprocess.Popen([sys.executable,str(ROOT/"run_web.py"),"--port","8903","--no-browser","--workspace",str(output)],cwd=ROOT,stdout=log,stderr=log,env=dict(os.environ,PYTHONIOENCODING="utf-8"),creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.1)
            else:raise AssertionError("isolated server failed")
            with httpx.Client(base_url=url,timeout=60) as api,sync_playwright() as pw:
                state=api.get("/api/state").json();api.headers["X-EIP-Token"]=state["session_token"]
                opened=api.post("/api/sources/open",json=dict(path=str(output/"raw.tif"),interval=.01));opened.raise_for_status();source=opened.json()["source"]
                browser=pw.chromium.launch(channel="msedge" if os.name=="nt" else "chromium",headless=True)
                page=browser.new_page(viewport=dict(width=1600,height=1000),accept_downloads=True)
                errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
                page.goto(url);page.wait_for_timeout(800);assert not errors,errors;expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 9")
                page.locator("#open-run-settings").click()
                expect(page.locator("#resource-memory_mb")).not_to_have_value("")
                for key,value in dict(memory_mb=192,preview_mb=32,cache_mb=16,gpu_mb=64).items():page.locator(f"#resource-{key}").fill(str(value))
                page.locator("#resource-backend").select_option("auto");page.locator("#resource-save").click()
                expect(page.locator("#toast")).to_contain_text("已保存")
                expect(page.locator("#resource-readout")).to_contain_text("处理预算 160 MiB")
                page.screenshot(path=str(output/"resources-desktop.png"))
                page.set_viewport_size(dict(width=1024,height=768));page.screenshot(path=str(output/"resources-small.png"))
                assert page.locator("#resource-save").evaluate("e=>e.getBoundingClientRect().right<=innerWidth")
                page.locator("#run-settings-dialog [data-close-dialog]").click();page.set_viewport_size(dict(width=1600,height=1000))
                page.locator("#open-module-library").click();page.locator("[data-category-filter='all']").click();page.locator("[data-add-module='gain_offset']").click()
                gain=page.locator("#general-module-fields [data-key='gain']");gain.fill("2");gain.press("Tab")
                expect(page.locator("#general-module-fields [data-execution='backend']")).to_have_value("inherit")
                page.locator("#open-module-library").click();page.locator("[data-add-module='data_tap']").click()
                destination=page.locator("#general-module-fields [data-key='destination']");destination.select_option("file")
                page.locator("#job-name").fill("资源测试结果")
                before={j["id"] for j in api.get("/api/jobs").json()};page.locator("#run").click()
                job=None
                for _ in range(300):
                    fresh=[j for j in api.get("/api/jobs").json() if j["id"] not in before and j["status"] in {"succeeded","failed","completed_with_errors"}]
                    if fresh:job=api.get("/api/jobs/"+fresh[0]["id"]).json();break
                    page.wait_for_timeout(100)
                assert job and job["status"]=="succeeded",job
                page.locator("#manage-results").click()
                expect(page.locator(f"[data-manage-result='{job['id']}']")).to_be_visible()
                page.locator(f"[data-manage-result='{job['id']}']").click();expect(page.locator("#managed-name")).to_have_value("资源测试结果")
                side=job["run"]["results"][0]["side_outputs"][0];page.locator("#managed-output").select_option(f"0|{side['id']}")
                page.screenshot(path=str(output/"results-desktop.png"))
                page.set_viewport_size(dict(width=1024,height=768));page.screenshot(path=str(output/"results-small.png"))
                assert page.locator("#managed-promote").evaluate("e=>e.getBoundingClientRect().right<=innerWidth")
                with page.expect_download() as received:page.locator("#managed-export").click()
                downloaded=received.value;downloaded.save_as(str(output/"run-package.zip"))
                with zipfile.ZipFile(output/"run-package.zip") as z:
                    prefix=f"run_{job['id']}/"
                    np.testing.assert_array_equal(np.load(io.BytesIO(z.read(prefix+"source_000/values.npy"))),values*2)
                    record=json.loads(z.read(prefix+"run.json"));assert record["spec"]["resources"]["backend"]=="auto"
                    assert any(d["actual"]=="cpu" and d["reason"] for d in record["results"][0]["modules"]["devices"].values())
                page.locator("#managed-promote").click();expect(page.locator("#result-manager-dialog")).not_to_be_visible()
                expect(page.locator("#steps .flow-node")).to_have_count(0)
                inputs=[s for s in api.get("/api/state").json()["sources"] if s["role"]=="input"]
                derived=next(s for s in inputs if s.get("lineage"));assert derived["lineage"][-1]["side_output"]==side["id"]
                page.locator("#manage-results").click();page.locator("#managed-delete").click();expect(page.locator("#result-delete-files")).to_be_disabled()
                page.screenshot(path=str(output/"delete-dependencies.png"));page.locator("#result-delete-confirm").click()
                expect(page.locator("#result-delete-dialog")).not_to_be_visible();assert Path(job["directory"]).is_dir()
                assert any(s["id"]==derived["id"] for s in api.get("/api/state").json()["sources"])
                page.locator("#open-run-settings").click();page.locator("#resource-clear").click();expect(page.locator("#toast")).to_contain_text("释放")
                expect(page.locator("#resource-readout")).to_contain_text("原值缓存 0.0 MiB")
                assert not errors,errors
                browser.close()
            print(json.dumps(dict(status="passed",output=str(output)),ensure_ascii=False))
        finally:
            server.terminate()
            try:server.wait(10)
            except subprocess.TimeoutExpired:server.kill();server.wait(5)


if __name__=="__main__":main()
