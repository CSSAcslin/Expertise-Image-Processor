"""Real browser: draw paired selections, compare signals and export full values."""
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
    output=ROOT/".test-output"/f"paired-comparison-{int(time.time())}";output.mkdir(parents=True)
    grid=np.arange(144).reshape(12,12);t=np.arange(201)[:,None,None]
    values=np.concatenate((grid+5*t,100+grid+10*t),axis=2).astype(np.uint16)
    tifffile.imwrite(output/"source.tif",values,photometric="minisblack",metadata={"axes":"TYX"})
    url="http://127.0.0.1:8909"
    with (output/"server.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,str(ROOT/"run_web.py"),"--port","8909","--no-browser","--workspace",str(output)],cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.1)
            else:raise AssertionError("isolated server did not start")
            with httpx.Client(base_url=url,timeout=45) as api,sync_playwright() as pw:
                state=api.get("/api/state").json();assert state["version"]=="0.8.5"
                api.headers["X-EIP-Token"]=state["session_token"]
                opened=api.post("/api/sources/open",json=dict(path=str(output/"source.tif"),interval=.01));opened.raise_for_status()
                browser=pw.chromium.launch(channel="msedge" if os.name=="nt" else "chromium",headless=True)
                page=browser.new_page(viewport=dict(width=1600,height=1000),accept_downloads=True)
                errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
                page.goto(url);expect(page.locator("#frame-info")).to_contain_text("201")
                page.locator("#flow-clear").click()
                def add(a):
                    page.locator("#open-module-library").click();page.locator("[data-category-filter=all]").click();page.locator(f"[data-add-module='{a}']").click()
                def parameter(k,v):
                    control=page.locator(f"#general-module-fields [data-key='{k}']")
                    if control.evaluate("e=>e.tagName")=="SELECT":control.select_option(str(v))
                    else:control.fill(str(v));control.press("Tab")
                def canvas_point(x,y):
                    box=page.locator("#input-canvas").bounding_box();scale=min((box["width"]-40)/12,(box["height"]-50)/12)
                    return box["x"]+box["width"]/2+(x-6)*scale,box["y"]+box["height"]/2+(y-6)*scale
                add("alignment");add("paired_trace")
                page.locator('[data-canvas-tool="point"]').click();expect(page.locator("#axis-stage-caption")).to_contain_text("P:2")
                expect(page.locator("#inspection-state")).to_be_hidden()
                page.mouse.click(*canvas_point(3.2,5.2));expect(page.locator('[data-key="x"]')).to_have_value("3");expect(page.locator('[data-key="y"]')).to_have_value("5")
                assert api.get("/api/jobs").json()==[]
                add("paired_trace");parameter("selection","region");parameter("mode","max")
                page.locator('[data-canvas-tool="roi-new"]').click();expect(page.locator("#axis-stage-caption")).to_contain_text("P:2")
                expect(page.locator("#inspection-state")).to_be_hidden()
                page.mouse.move(*canvas_point(1,2));page.mouse.down();page.mouse.move(*canvas_point(5,6),steps=6);page.mouse.up()
                expect(page.locator('[data-key="width"]')).to_have_value("4");expect(page.locator('[data-key="height"]')).to_have_value("4")
                page.screenshot(path=str(output/"selection-desktop.png"))
                page.locator("#axis-restore-raw").click()
                page.locator("#job-name").fill("对应点与区域时序")
                expect(page.locator("#module-status")).to_have_text("连接有效");page.locator("#run").click()
                for _ in range(400):
                    jobs=api.get("/api/jobs").json()
                    if jobs and jobs[0]["status"] in {"succeeded","failed","completed_with_errors"}:break
                    page.wait_for_timeout(100)
                assert jobs[0]["status"]=="succeeded",jobs
                job=api.get("/api/jobs/"+jobs[0]["id"]).json();sides=job["run"]["results"][0]["side_outputs"]
                point,region=sides
                # Inspecting geometry deliberately holds the result preview; open the saved side explicitly.
                page.locator("#open-result-tools").click();page.locator("#result-tools-dialog").get_by_role("button",name="关闭结果导出").click()
                page.locator("#open-data-library").click();page.locator(f"[data-manage-result='{job['id']}']").click()
                expect(page.locator(".comparison-series")).to_have_count(2)
                expect(page.locator("#curve-list")).to_be_visible()
                page.locator("#curve-add").click()
                # The two side candidates have the same module title; source order is deterministic.
                page.locator("#curve-candidate").select_option("0");page.locator("#curve-add-confirm").click()
                expect(page.locator(".comparison-series")).to_have_count(4)
                page.locator('[data-curve-focus="3"]').check();expect(page.locator('[data-curve-row="3"]')).to_have_class("curve-row is-focus")
                page.locator('[data-curve-visible="0"]').uncheck();expect(page.locator(".comparison-series")).to_have_count(3)
                page.locator('[data-curve-visible="0"]').check();expect(page.locator(".comparison-series")).to_have_count(4)
                page.locator('[data-curve-name="3"]').fill("点信号 · 右路");page.locator('[data-curve-name="3"]').press("Tab")
                page.locator("#curve-axis").select_option("time");expect(page.locator("#sequence-readout")).to_contain_text("真实时间")
                box=page.locator("#split-curve").bounding_box();page.mouse.move(box["x"]+box["width"]*.6,box["y"]+box["height"]*.55)
                initial=page.locator("#split-curve svg").get_attribute("data-x-max");page.mouse.wheel(0,-300);page.wait_for_timeout(100)
                assert float(page.locator("#split-curve svg").get_attribute("data-x-max"))<float(initial)
                page.mouse.move(box["x"]+15,box["y"]+box["height"]*.5);old=page.locator("#split-curve svg").get_attribute("data-y-max");page.mouse.wheel(0,-300);page.wait_for_timeout(100)
                assert page.locator("#split-curve svg").get_attribute("data-y-max")!=old
                page.locator("#curve-fit").click();page.screenshot(path=str(output/"comparison-desktop.png"))
                with page.expect_download() as downloaded:page.locator("#curve-export").click()
                downloaded.value.save_as(str(output/"comparison.zip"))
                with zipfile.ZipFile(output/"comparison.zip") as archive:
                    recipe=json.loads(archive.read("comparison.json"));assert recipe["full_values"] and len(recipe["items"])==4
                    records=list(csv.reader(io.StringIO(archive.read("values.csv").decode("utf-8-sig"))))
                    assert len(records)==202
                    np.testing.assert_allclose(np.array(records[1:]).astype(float)[:,1:],[np.r_[grid[2:6,1:5].max()+5*i,100+grid[2:6,1:5].max()+10*i,grid[5,3]+5*i,100+grid[5,3]+10*i] for i in range(201)])
                # Draft reload preserves independent visibility, names and focus.
                page.wait_for_timeout(1000);page.reload();expect(page.locator(".comparison-series")).to_have_count(4,timeout=15000)
                expect(page.locator('[data-curve-name="3"]')).to_have_value("点信号 · 右路")
                expect(page.locator('[data-curve-focus="3"]')).to_be_checked()
                page.set_viewport_size(dict(width=1024,height=768));page.wait_for_timeout(200);page.screenshot(path=str(output/"comparison-small.png"))
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                page.locator("#curve-list-toggle").click();expect(page.locator("#curve-list")).to_be_visible()
                page.locator('[data-curve-remove="3"]').click();expect(page.locator(".comparison-series")).to_have_count(3)
                assert len(api.get("/api/jobs").json())==1
                assert not errors,errors
                (output/"report.json").write_text(json.dumps(dict(version=state["version"],status="passed",checks=["draw_point","draw_roi","four_curves","visibility","focus","zoom_xy","export_all","draft_restore","small_window","remove_without_file_delete"],javascript_errors=errors),ensure_ascii=False,indent=2),encoding="utf-8")
                print(output);browser.close()
        finally:
            process.terminate()
            try:process.wait(10)
            except subprocess.TimeoutExpired:process.kill();process.wait(5)


if __name__=="__main__":main()
