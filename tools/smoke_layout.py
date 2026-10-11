"""Browser checks for horizontal dock columns, real splitter drags and restore."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import httpx
import numpy as np
import tifffile
from playwright.sync_api import sync_playwright, expect


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/".test-output"/f"layout-ui-{int(time.time())}"
    output.mkdir(parents=True)
    sample=output/"sample.tif"
    yy,xx=np.mgrid[:72,:168]
    tifffile.imwrite(sample,np.stack([(xx+yy+i*3).astype(np.uint16) for i in range(15)]),photometric="minisblack",metadata={"axes":"TYX"})
    url="http://127.0.0.1:8886"
    with (output/"server.log").open("w",encoding="utf-8") as log:
        server=subprocess.Popen([sys.executable,str(root/"run_web.py"),"--port","8886","--no-browser","--workspace",str(output)],cwd=root,stdout=log,stderr=log,env=dict(os.environ,PYTHONIOENCODING="utf-8"),creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:break
                except httpx.HTTPError:time.sleep(.1)
            with httpx.Client(base_url=url,timeout=30) as api,sync_playwright() as pw:
                api.headers["X-EIP-Token"]=api.get("/api/state").json()["session_token"]
                api.post("/api/sources/open",json={"path":str(sample),"interval":.00001}).raise_for_status()
                browser=pw.chromium.launch(channel="msedge" if sys.platform=="win32" else "chromium",headless=True)
                page=browser.new_page(viewport={"width":1440,"height":960})
                errors=[]
                page.on("pageerror",lambda e:errors.append(str(e)))
                page.goto(url)
                expect(page.locator("#connection")).to_have_text("本机已连接")
                expect(page.locator("#version")).to_have_text("v0.5.1")

                def box(selector):return page.locator(selector).bounding_box()
                def drag(key,dx=0,dy=0):
                    b=box(f'[data-resize="{key}"]');assert b
                    x,y=b["x"]+b["width"]/2,b["y"]+b["height"]/2
                    page.mouse.move(x,y);page.mouse.down();page.mouse.move(x+dx,y+dy,steps=8);page.mouse.up()
                    page.wait_for_timeout(50)
                def toggle(mode,key):return page.locator(f'[data-dock-stack="{mode}"] [data-dock-toggle="{key}"]')
                def columns(mode):
                    flow=box(f'[data-dock-stack="{mode}"] [data-dock-panel="flow"]')
                    parameters=box(f'[data-dock-stack="{mode}"] [data-dock-panel="parameters"]')
                    assert abs(flow["y"]-parameters["y"])<1 and abs(flow["height"]-parameters["height"])<1
                    assert parameters["x"]>=flow["x"]+flow["width"],(flow,parameters)
                    return flow,parameters
                def draft():return page.evaluate("Object.entries(localStorage).map(([k,v])=>k.startsWith('eip-workbench-view-3:')?JSON.parse(v):null).find(Boolean)")
                def in_screen(selector):
                    b=box(selector);assert b and b["width"]>0 and b["height"]>0
                    viewport=page.viewport_size
                    assert b["x"]>=-1 and b["y"]>=0 and b["x"]+b["width"]<=viewport["width"]+1 and b["y"]+b["height"]<=viewport["height"]+1,(selector,b)

                page.locator("#algorithm").select_option("affine");page.locator("#add-step").click()
                expect(page.locator('#steps [data-param]')).to_have_count(0)
                fields=page.locator("#general-module-fields")
                fields.locator('[data-key="gain"]').fill("2");fields.locator('[data-key="gain"]').dispatch_event("change")
                original_id=page.locator('#steps [data-instance]').get_attribute("data-instance")
                page.locator('#steps [data-duplicate]').click()
                fields.locator('[data-key="gain"]').fill("3");fields.locator('[data-key="gain"]').dispatch_event("change")
                page.locator(f'#steps [data-instance="{original_id}"] [data-edit]').click()
                expect(fields.locator('[data-key="gain"]')).to_have_value("2")
                toggle("general","parameters").click()
                expect(toggle("general","parameters")).to_have_attribute("aria-expanded","false")
                expect(page.locator("#general-flow-body")).to_be_visible()
                page.locator('#steps [data-edit]').last.click()
                expect(toggle("general","parameters")).to_have_attribute("aria-expanded","true")
                expect(fields.locator('[data-key="gain"]')).to_have_value("3")
                columns("general")
                lib=box(".library")["width"];drag("library",dx=35)
                assert abs(box(".library")["width"]-lib-35)<2
                dock=box(".processing")["width"];drag("dock",dx=-45)
                assert abs(box(".processing")["width"]-dock-45)<2
                flow=columns("general")[0]["width"]
                drag("general-panels",dx=40)
                assert abs(columns("general")[0]["width"]-flow-40)<2
                tasks=box(".tasks")["height"];drag("tasks",dy=-25)
                assert abs(box(".tasks")["height"]-tasks-25)<2
                page.locator('[data-view="compare"]').click()
                left=box("#input-pane")["width"];drag("compare",dx=40)
                assert abs(box("#input-pane")["width"]-left-40)<2
                # Keyboard sizing and reset are usable without dragging.
                handle=page.locator('[data-resize="library"]');handle.focus();handle.press("ArrowRight")
                assert abs(box(".library")["width"]-lib-45)<2
                expanded_dock=box(".processing")["width"];expanded_flow,expanded_parameters=columns("general")
                preview=box(".preview-workspace")["width"]
                toggle("general","flow").click()
                expect(page.locator("#general-flow-body")).not_to_be_visible()
                expect(page.locator("#general-parameters-body")).to_be_visible()
                dock_box=box(".processing");footer=box(".run-footer")
                flow,parameters=columns("general")
                assert abs(flow["width"]-34)<1 and abs(parameters["width"]-expanded_parameters["width"])<1
                assert abs(flow["width"]+parameters["width"]+1-dock_box["width"])<2
                assert abs(parameters["height"]+footer["height"]-dock_box["height"])<2
                assert abs(box(".preview-workspace")["width"]-preview-expanded_dock+dock_box["width"])<2
                page.wait_for_timeout(450)
                before=draft();page.reload()
                expect(toggle("general","flow")).to_have_attribute("aria-expanded","false")
                expect(fields.locator('[data-key="gain"]')).to_have_value("3")
                assert abs(box(".processing")["width"]-dock_box["width"])<2
                assert before["web_docks"]["general"]["flow"] is True
                assert before["web_docks"]["schema"]==2
                # Resizing the remaining open column preserves the closed column's width.
                drag("dock",dx=-20)
                assert abs(columns("general")[1]["width"]-expanded_parameters["width"]-20)<2
                expanded_dock+=20
                # Either column can become a narrow, vertical tab; both recover preview space.
                toggle("general","parameters").click()
                assert abs(box(".processing")["width"]-69)<1
                expect(page.locator(".run-footer")).not_to_be_visible()
                expect(page.locator('[data-resize="dock"]')).not_to_be_visible()
                for key in["flow","parameters"]:
                    button=toggle("general",key);button.focus();button.press("Enter")
                    expect(button).to_have_attribute("aria-expanded","true")
                assert abs(box(".processing")["width"]-expanded_dock)<2
                assert abs(columns("general")[0]["width"]-expanded_flow["width"])<2
                toggle("general","flow").click()
                page.screenshot(path=str(output/"01-general-docks.png"))

                page.locator("#analysis-mode").select_option("split_correlation")
                expect(page.locator("#split-canvas")).to_be_visible()
                expect(page.locator("#split-flow-list [data-param]")).to_have_count(0)
                expect(toggle("split","flow")).to_have_attribute("aria-expanded","true")
                split_dock=box(".split-dock")["width"];drag("dock",dx=-30)
                assert abs(box(".split-dock")["width"]-split_dock-30)<2
                left=box(".split-data-window")["width"];drag("splitCompare",dx=30)
                assert abs(box(".split-data-window")["width"]-left-30)<2
                flow=columns("split")[0]["width"];drag("split-panels",dx=-45)
                assert abs(columns("split")[0]["width"]-flow+45)<2
                toggle("split","parameters").click()
                expect(page.locator("#split-parameters-body")).not_to_be_visible()
                page.locator('#split-flow-list [data-algorithm="alignment"] [data-edit]').click()
                expect(toggle("split","parameters")).to_have_attribute("aria-expanded","true")
                page.locator("#split-module-type").select_option("affine");page.locator("#split-module-add").click()
                expect(page.locator('#split-module-fields [data-key="gain"]')).to_have_value("1")
                page.locator('#split-module-fields [data-key="gain"]').fill("4");page.locator('#split-module-fields [data-key="gain"]').dispatch_event("change")
                toggle("split","flow").click()
                split_collapsed=box(".split-dock")["width"]
                page.wait_for_timeout(450);page.reload()
                expect(page.locator("#analysis-mode")).to_have_value("split_correlation")
                expect(toggle("split","flow")).to_have_attribute("aria-expanded","false")
                expect(page.locator('#split-module-fields [data-key="gain"]')).to_have_value("4")
                assert abs(box(".split-dock")["width"]-split_collapsed)<2
                page.locator("#save-project").click()
                page.locator("#filename").fill("layout.eip.json")
                page.locator("#dialog-submit").click()
                expect(page.locator("#file-dialog")).not_to_be_visible()
                saved=json.loads((output/"layout.eip.json").read_text(encoding="utf-8"))
                assert saved["web_docks"]["split"]["flow"] is True
                assert saved["web_docks"]["schema"]==2
                assert saved["web_selected_module"]!=original_id
                toggle("split","flow").click()
                page.screenshot(path=str(output/"02-split-docks.png"))
                # Every primary pane remains inside the viewport after responsive clamping.
                for width,height in[(1100,760),(960,760),(1440,960)]:
                    page.set_viewport_size({"width":width,"height":height});page.wait_for_timeout(100)
                    for selector in[".library",".split-dock",".split-data-window",".split-result-window",".split-run-bar",".tasks",".split-input-timeline"]:in_screen(selector)
                    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                    columns("split")
                    workspace=box(".split-workspace");bar=box(".split-run-bar");tasks_box=box(".tasks")
                    assert abs(workspace["y"]+workspace["height"]-bar["y"]-bar["height"])<2
                    assert abs(bar["y"]+bar["height"]+6-tasks_box["y"])<2
                    timeline=box(".split-input-timeline")
                    assert timeline["y"]+timeline["height"]<=bar["y"]+1,(timeline,bar)
                    page.screenshot(path=str(output/f"03-split-{width}.png"))
                page.locator("#split-layout").select_option("data")
                expect(page.locator('[data-resize="splitCompare"]')).not_to_be_visible()
                page.locator("#split-layout").select_option("both")
                page.locator("#toggle-processing").click()
                expect(page.locator(".split-dock")).not_to_be_visible()
                page.locator("#split-open-context").click()
                expect(page.locator(".split-dock")).to_be_visible()
                expect(page.locator("#split-run-context")).to_have_attribute("open","")
                page.locator("#reset-layout").click()
                expect(toggle("split","flow")).to_have_attribute("aria-expanded","true")
                expect(toggle("split","parameters")).to_have_attribute("aria-expanded","true")
                assert abs(box(".split-dock")["width"]-556)<2
                page.locator("#analysis-mode").select_option("general")
                expect(toggle("general","flow")).to_have_attribute("aria-expanded","true")
                page.locator("#focus-preview").click()
                expect(page.locator(".processing")).not_to_be_visible()
                expect(page.locator(".tasks")).not_to_be_visible()
                page.locator("#focus-preview").click()
                expect(page.locator(".processing")).to_be_visible()
                # Loading the saved project restores both layout and the inspector's instance.
                page.locator(f'#steps [data-instance="{original_id}"] [data-edit]').click()
                expect(fields.locator('[data-key="gain"]')).to_have_value("2")
                page.locator("#load-project").click()
                page.locator("#file-list .file-row").filter(has_text="layout.eip.json").click()
                page.locator("#dialog-submit").click()
                expect(page.locator("#file-dialog")).not_to_be_visible()
                expect(page.locator("#analysis-mode")).to_have_value("split_correlation")
                expect(toggle("split","flow")).to_have_attribute("aria-expanded","false")
                expect(page.locator('#split-module-fields [data-key="gain"]')).to_have_value("4")
                assert abs(box(".split-dock")["width"]-split_collapsed)<2
                page.locator("#analysis-mode").select_option("general")
                expect(fields.locator('[data-key="gain"]')).to_have_value("3")
                # Stacked layout drafts migrate without treating a 340px total as two columns.
                page.wait_for_timeout(450)
                page.add_init_script("""const entry=Object.entries(localStorage).find(([k])=>k.startsWith('eip-workbench-view-3:'));if(entry){const [key,text]=entry;const value=JSON.parse(text);value.web_docks={...value.web_docks,schema:1,general:{width:340,share:.46,flow:false,parameters:false}};localStorage.setItem(key,JSON.stringify(value));}""")
                page.reload()
                expect(page.locator("#connection")).to_have_text("本机已连接")
                expect(fields.locator('[data-key="gain"]')).to_have_value("3")
                assert abs(box(".processing")["width"]-556)<2
                columns("general")
                assert not errors,errors
                report={"independent_columns":True,"horizontal_collapse_and_width_restore":True,"shared_parameter_inspector":True,"real_drag_dividers":["library","dock","general-panels","split-panels","tasks","compare","splitCompare"],"restored_layout_and_instances":True,"legacy_stacked_layout_migration":True,"project_roundtrip":True,"viewports":[1440,1100,960],"javascript_errors":errors}
                (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
                print(json.dumps({"output":str(output),**report},ensure_ascii=False))
                browser.close()
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill()


if __name__=="__main__":main()
