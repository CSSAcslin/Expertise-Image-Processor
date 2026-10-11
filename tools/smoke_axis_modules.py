"""Real-browser checks for spatial/time modules and transparent shape connections."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import httpx
import numpy as np
import tifffile
from playwright.sync_api import sync_playwright,expect


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/".test-output"/f"axis-modules-{int(time.time())}"
    out.mkdir(parents=True)
    yy,xx = np.mgrid[:48,:64]
    halves = [(30+xx+yy+i*2).astype(np.uint16) for i in range(17)]
    values = np.stack([np.concatenate([a,a+i%3],axis=1) for i,a in enumerate(halves)])
    tifffile.imwrite(out/"sample.tif",values,photometric="minisblack",metadata={"axes":"TYX"})
    url = "http://127.0.0.1:8891"
    with (out/"server.log").open("w",encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable,str(root/"run_web.py"),"--port","8891","--no-browser","--workspace",str(out)],cwd=root,stdout=log,stderr=log,env=dict(os.environ,PYTHONIOENCODING="utf-8"),creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:
                        break
                except httpx.HTTPError:
                    time.sleep(.1)
            with httpx.Client(base_url=url,timeout=45) as api,sync_playwright() as pw:
                api.headers["X-EIP-Token"] = api.get("/api/state").json()["session_token"]
                api.post("/api/sources/open",json=dict(path=str(out/"sample.tif"),interval=.01)).raise_for_status()
                browser = pw.chromium.launch(channel="msedge" if sys.platform=="win32" else "chromium",headless=True)
                context = browser.new_context(viewport=dict(width=1600,height=1000))
                page = context.new_page()
                errors = []
                page.on("pageerror",lambda e:errors.append(str(e)))
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 17")
                expect(page.locator('#steps [data-algorithm="time_crop"]')).to_have_count(1)
                expect(page.locator("#scope")).to_be_hidden()
                def add(a):
                    page.locator("#algorithm").select_option(a)
                    page.locator("#add-step").click()
                def param(key,value,split=False):
                    p=page.locator(("#split-module-fields" if split else "#general-module-fields")+f' [data-key="{key}"]')
                    if p.evaluate("e=>e.tagName")=="SELECT":
                        p.select_option(value)
                    else:
                        p.fill(str(value));p.press("Tab")
                def completed(before):
                    deadline=time.monotonic()+35
                    while time.monotonic()<deadline:
                        jobs=api.get("/api/jobs").json()
                        fresh=[j for j in jobs if j["id"] not in before]
                        terminal=[j for j in fresh if j["status"] in {"succeeded","failed","completed_with_errors"}]
                        if terminal:
                            job=api.get("/api/jobs/"+terminal[0]["id"]).json()
                            assert job["status"]=="succeeded",job
                            return job
                        page.wait_for_timeout(100)
                    raise AssertionError("UI job timeout")
                add("spatial_roi")
                for key,value in (("x",5),("y",4),("width",20),("height",15)):
                    param(key,value)
                expect(page.locator('#steps [data-algorithm="spatial_roi"] .flow-type')).to_contain_text("Y:15 × X:20")
                before={j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#run").click();matrix=completed(before)
                expect(page.locator("#sequence-controls")).to_be_visible()
                assert matrix["run"]["results"][0]["contract"]["shape"]==[15,20]
                page.locator("#sequence-index").fill("3");page.locator("#sequence-index").dispatch_event("change")
                expect(page.locator("#sequence-readout")).to_contain_text("样本 4/17")
                page.locator(".display-panel > summary").click()
                page.locator("#display-target").select_option("result");page.locator("#colormap").select_option("heat")
                expect(page.locator("#sequence-controls")).to_be_visible()
                page.locator("#sequence-index").fill("4");page.locator("#sequence-index").dispatch_event("change")
                expect(page.locator("#sequence-readout")).to_contain_text("样本 5/17")
                expect(page.locator("#colormap")).to_have_value("heat")
                page.locator("#display-target").select_option("input")
                param("mode","mean")
                expect(page.locator("#general-module-inspector .module-contract")).to_contain_text("输出 T:17")
                before={j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#run").click();signal=completed(before)
                expect(page.locator("#split-curve svg")).to_be_visible()
                assert signal["run"]["results"][0]["contract"]["shape"]==[]
                add("gaussian")
                expect(page.locator("#module-status")).to_contain_text("Y/X")
                page.locator("#seek-value").fill("7");page.locator("#seek").click()
                expect(page.locator("#frame-info")).to_contain_text("第 7 帧 / 17")
                page.locator('#steps [data-algorithm="gaussian"] [data-delete]').click()
                page.locator('#steps [data-algorithm="spatial_roi"] [data-edit]').click()
                param("mode","pixels")
                before={j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#general-module-inspector [data-roi-module]").click();completed(before)
                expect(page.locator("#axis-stage-note")).to_be_visible()
                expect(page.locator("#input-label")).to_have_text("模块阶段检查")
                page.locator("#display-gamma").fill("2");page.locator("#display-gamma").press("Tab")
                expect(page.locator("#axis-stage-note")).to_be_visible()
                canvas=page.locator("#input-canvas").bounding_box()
                page.mouse.move(canvas["x"]+canvas["width"]*.4,canvas["y"]+canvas["height"]*.4)
                page.mouse.down();page.mouse.move(canvas["x"]+canvas["width"]*.55,canvas["y"]+canvas["height"]*.58);page.mouse.up()
                x=page.locator('#general-module-fields [data-key="x"]')
                before_x=int(x.input_value());direction=-1 if before_x else 1
                page.keyboard.press("ArrowLeft" if direction<0 else "ArrowRight")
                expect(x).to_have_value(str(before_x+direction))
                page.keyboard.press("Control+z")
                expect(x).to_have_value(str(before_x))
                page.locator("#axis-restore-raw").click()
                expect(page.locator("#input-label")).to_have_text("原始数据")
                page.locator("#analysis-mode").select_option("split_correlation")
                expect(page.locator("#split-run-context")).to_be_hidden()
                page.locator("#split-accept").click()
                expect(page.locator("#split-status")).not_to_contain_text("请检查并接受")
                page.locator("#split-module-type").select_option("spatial_roi")
                page.locator("#split-module-position").select_option("3")
                page.locator("#split-module-add").click()
                param("mode","mean",True)
                expect(page.locator("#split-module-inspector .module-contract")).to_contain_text("输出 T:17 × P:2")
                page.locator('#split-flow-list [data-algorithm="correlation"] [data-edit]').click()
                param("analysis","curve",True)
                before={j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#split-curve-run").click();curve=completed(before)
                expect(page.locator("#split-curve svg")).to_be_visible()
                assert curve["run"]["results"][0]["contract"]["full_axes"]=="Q"
                page.locator("#sequence-controls summary").click();page.locator("#sequence-x").select_option("time");page.locator("#sequence-controls summary").click()
                expect(page.locator("#split-curve")).to_contain_text("实际配对时差")
                page.wait_for_timeout(450);page.reload()
                expect(page.locator("#sequence-controls")).to_be_visible()
                expect(page.locator("#split-curve")).to_contain_text("实际配对时差")
                expect(page.locator("#split-flow-list [data-algorithm=spatial_roi]")).to_have_count(1)
                expect(page.locator("#split-module-inspector .module-contract")).to_contain_text("输出 Q:")
                page.screenshot(path=str(out/"desktop.png"))
                for width,height in ((1100,800),(960,760)):
                    page.set_viewport_size(dict(width=width,height=height));page.wait_for_timeout(100)
                    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                    page.screenshot(path=str(out/f"layout-{width}.png"))
                page.set_viewport_size(dict(width=1600,height=1000))
                page.wait_for_timeout(450)
                # A second page reads the legacy draft without overwriting it on unload.
                page.evaluate("""()=>{const key=Object.keys(localStorage).find(k=>k.startsWith('eip-workbench-view-3:')),v=JSON.parse(localStorage.getItem(key)),r=v.split_mode.records[v.active_source],c=r.config;
                  c.workflow=c.workflow.filter(n=>!['time_crop','spatial_roi','time_sample','time_smooth','time_bin'].includes(n.algorithm));c.workflow_version=2;c.start=2;c.end=15;c.center=8;c.roi={x:3,y:2,width:5,height:4};c.points=[[5,5]];
                  const corr=c.workflow.find(n=>n.algorithm==='correlation');corr.parameters.analysis='map';corr.parameters.roi_estimator='intensity_sum';localStorage.setItem(key,JSON.stringify(v));}""")
                migrated=page.context.new_page();migrated.on("pageerror",lambda e:errors.append(str(e)));migrated.goto(url)
                expect(migrated.locator('#split-flow-list [data-algorithm="spatial_roi"]')).to_have_count(1)
                migrated.locator('#split-flow-list [data-algorithm="spatial_roi"] [data-edit]').click()
                expect(migrated.locator('#split-module-fields [data-key="mode"]')).to_have_value("pixels")
                expect(migrated.locator(".module-migration")).to_contain_text("统计改为")
                migrated.wait_for_timeout(450)
                migrated.evaluate("""()=>{const key=Object.keys(localStorage).find(k=>k.startsWith('eip-workbench-view-3:')),v=JSON.parse(localStorage.getItem(key)),c=v.split_mode.records[v.active_source].config;c.workflow=c.workflow.filter(n=>n.algorithm!=='spatial_roi');c.workflow_version=2;c.roi=null;c.points=[[5,5]];localStorage.setItem(key,JSON.stringify(v));}""")
                migrated_again=page.context.new_page();migrated_again.goto(url)
                expect(migrated_again.locator('#split-flow-list [data-algorithm="time_crop"]')).to_have_count(1)
                expect(migrated_again.locator('#split-flow-list [data-algorithm="spatial_roi"]')).to_have_count(0)
                expect(migrated_again.locator("#frame-info")).to_contain_text("第 7 帧")
                assert not errors,errors
                report=dict(matrix=matrix["id"],signal=signal["id"],signed_curve=curve["id"],shape_errors_leave_raw_preview_usable=True,roi_is_instance_owned=True,roi_keyboard_undo=True,result_display_independent=True,result_draft_restored=True,legacy_map_roi_preserved=True,execution_settings_per_instance=True,javascript_errors=errors)
                (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
                print(json.dumps(dict(output=str(out),**report),ensure_ascii=False))
                browser.close()
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill()


if __name__=="__main__":
    main()
