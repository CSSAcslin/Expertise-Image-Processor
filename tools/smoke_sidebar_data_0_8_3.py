"""Sidebar hierarchy, preview space and independent input/result workflows."""
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
sys.path.insert(0,str(ROOT/"src"))
from eip.module_registry import new_node

def main():
    folder=ROOT/".test-output"/f"sidebar-data-{int(time.time())}"
    folder.mkdir(parents=True)
    values=np.arange(9*24*40,dtype=np.uint16).reshape(9,24,40)+10
    for name,offset in [("输入 A",0),("输入 B",10),("输入 C",20)]:
        tifffile.imwrite(folder/(name+".tif"),values+offset,photometric="minisblack",metadata={"axes":"TYX"})
    url="http://127.0.0.1:8906"
    with (folder/"server.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,str(ROOT/"run_web.py"),"--port","8906","--no-browser","--workspace",str(folder)],cwd=ROOT,stdout=log,stderr=log,env=dict(os.environ,PYTHONIOENCODING="utf-8"),creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:break
                except httpx.HTTPError:pass
                time.sleep(.1)
            else:raise AssertionError("isolated service failed")
            with httpx.Client(base_url=url,timeout=45) as api,sync_playwright() as pw:
                api.headers["X-EIP-Token"]=api.get("/api/state").json()["session_token"]
                sources=[api.post("/api/sources/open",json=dict(path=str(folder/(name+".tif")),interval=.01)).json()["source"] for name in ("输入 A","输入 B","输入 C")]
                def wait(identifier):
                    for _ in range(300):
                        job=api.get(f"/api/jobs/{identifier}").json()
                        if job["status"] in ("succeeded","failed"):
                            assert job["status"]=="succeeded",job
                            time.sleep(.25);return job
                        time.sleep(.05)
                    raise AssertionError("worker timeout")
                def run(name,gain):
                    node=new_node("gain_offset");node["parameters"]["gain"]=gain
                    response=api.post("/api/jobs",json=dict(contract_version=3,source_ids=[sources[0]["id"]],name=name,steps=[node]))
                    response.raise_for_status();return wait(response.json()["id"])
                first,second=run("参考输出",1),run("增强输出",2)
                browser=pw.chromium.launch(channel="msedge" if os.name=="nt" else "chromium",headless=True)
                page=browser.new_page(viewport=dict(width=1600,height=1000),accept_downloads=True)
                errors,computes=[],[]
                page.on("pageerror",lambda error:errors.append(str(error)))
                page.on("request",lambda request:computes.append(request.post_data_json) if request.method=="POST" and request.url.endswith("/api/jobs") else None)
                page.goto(url)
                expect(page.locator("#source-count")).to_have_text("3")
                expect(page.locator("#result-count")).to_have_text("2")
                expect(page.locator("#library-data")).to_be_visible()
                assert page.locator("[data-library-tab]").count()==2
                assert page.locator(".library [data-library-tab=data]").count()==1
                assert page.locator("#library-data [data-data-group]").count()==2
                assert page.locator(".workspace > #data-management").count()==0
                def check_layout(small=False):
                    library=page.locator(".library").bounding_box()
                    preview=page.locator(".preview-workspace").bounding_box()
                    task=page.locator(".tasks").bounding_box()
                    assert library["x"]==0 and 244<=library["width"]<=420,library
                    assert library["y"]==60 and library["height"]>=700,library
                    assert abs(preview["x"]-library["width"])<=1,(library,preview)
                    assert abs(task["x"]-preview["x"])<=1 and abs(task["width"]-preview["width"])<=1,(task,preview)
                    assert task["height"]<=29,task
                    assert page.locator("#result-canvas").bounding_box()["height"]>=(180 if small else 320)
                    for toolbar in page.locator(".data-tools").all():
                        assert toolbar.evaluate("e=>e.scrollWidth<=e.clientWidth+1"),toolbar.evaluate("e=>[e.scrollWidth,e.clientWidth]")
                    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
                    source_group=page.locator("[data-data-group=sources]").bounding_box()
                    result_group=page.locator("[data-data-group=results]").bounding_box()
                    assert source_group["y"]+source_group["height"]<=result_group["y"]+1
                    assert page.locator("#library-data").bounding_box()["width"]<=library["width"]
                expect(page.locator("#open-data-library")).to_have_attribute("aria-selected","true")
                source_check=lambda source:page.locator(f"[data-source-check='{source['id']}']")
                result_check=lambda job:page.locator(f"[data-result-check='{job['id']}']")
                row=lambda job:page.locator(f"[data-result-id='{job['id']}']")
                for source in sources[:2]:source_check(source).check()
                result_check(first).check()
                expect(page.locator("#source-selection-count")).to_have_text("2 已选")
                expect(page.locator("#result-selection-count")).to_have_text("1 已选")
                assert not computes
                row(first).click()
                expect(page.locator("#sequence-index")).to_have_attribute("max","8")
                page.locator("#sequence-index").press("ArrowRight")
                expect(page.locator("#sequence-index")).to_have_value("1")
                page.locator("#open-module-library").click()
                expect(page.locator("#library-data")).not_to_be_visible()
                expect(page.locator("#library-modules")).to_be_visible()
                page.locator("#open-module-library").focus()
                page.locator("#open-module-library").press("ArrowLeft")
                expect(page.locator("#library-data")).to_be_visible()
                expect(page.locator("#sequence-index")).to_have_value("1")
                expect(page.locator("#source-selection-count")).to_have_text("2 已选")
                expect(page.locator("#result-selection-count")).to_have_text("1 已选")
                check_layout()
                page.locator("#toggle-tasks").click()
                assert 96<=page.locator(".tasks").bounding_box()["height"]<=160
                assert page.locator("#result-canvas").bounding_box()["height"]>=250
                page.locator("#toggle-tasks").click()
                for key in ("sources","results"):
                    other="results" if key=="sources" else "sources"
                    previous=page.locator(f"[data-data-group={other}]").bounding_box()["height"]
                    page.locator("#toggle-data-"+key).click()
                    expect(page.locator("#library-"+key)).not_to_be_visible()
                    expect(page.locator("#library-"+other)).to_be_visible()
                    assert page.locator(f"[data-data-group={other}]").bounding_box()["height"]>previous
                    page.locator("#toggle-data-"+key).click()
                page.screenshot(path=str(folder/"desktop.png"),animations="disabled")
                page.locator("#toggle-data-sources").click()
                page.locator("#toggle-data-results").click()
                expect(page.locator("#toggle-data-sources")).to_be_visible()
                expect(page.locator("#toggle-data-results")).to_be_visible()
                assert page.locator(".tasks").bounding_box()["height"]<=29
                page.wait_for_timeout(650);page.reload()
                expect(page.locator("#toggle-data-sources")).to_have_attribute("aria-expanded","false")
                expect(page.locator("#toggle-data-results")).to_have_attribute("aria-expanded","false")
                page.locator("#toggle-data-sources").click();page.locator("#toggle-data-results").click()
                for source in sources[:2]:source_check(source).check()
                page.locator("#job-name").fill("所选来源输出")
                with page.expect_response(lambda r:r.request.method=="POST" and r.url.endswith("/api/jobs")) as received:
                    page.locator("#source-batch-run").click()
                batch=wait(received.value.json()["id"])
                assert set(computes[-1]["source_ids"])=={s["id"] for s in sources[:2]}
                result_check(first).check();result_check(second).check()
                expect(page.locator("#result-selection-count")).to_have_text("2 已选")
                with page.expect_download() as received:page.locator("#result-batch-export").click()
                received.value.save_as(str(folder/"selected-results.zip"))
                with zipfile.ZipFile(folder/"selected-results.zip") as archive:
                    for job,multiplier in [(first,1),(second,2)]:
                        prefix=Path(job["directory"]).name
                        assert json.loads(archive.read(prefix+"/run.json"))["run_id"]==job["id"]
                        np.testing.assert_array_equal(np.load(io.BytesIO(archive.read(prefix+"/source_000/values.npy"))),values*multiplier)
                row(first).click()
                page.locator("#source-batch-remove").click()
                expect(page.locator("#source-count")).to_have_text("1")
                expect(page.locator("#result-selection-count")).to_have_text("2 已选")
                row(first).click()
                expect(page.locator("#result-label")).to_have_text("参考输出")
                page.wait_for_function("()=>{const c=document.getElementById('result-canvas'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]===d[i+1]&&d[i]===d[i+2]&&d[i+3])n++;return n>100;}")
                page.locator("#result-batch-remove").click()
                expect(page.locator("#result-batch-summary")).to_have_text("2 个结果 · 保留文件")
                assert page.locator("#result-batch-files").count()==0
                page.locator("#result-batch-confirm").click()
                expect(page.locator("#result-count")).to_have_text("1")
                assert all(Path(job["directory"]).is_dir() for job in (first,second))
                assert all((folder/(name+".tif")).is_file() for name in ("输入 A","输入 B","输入 C"))
                row(batch).click()
                page.set_viewport_size(dict(width=1024,height=768))
                page.wait_for_timeout(150)
                assert page.locator("#result-canvas").bounding_box()["height"]>=75
                assert page.locator("#result-canvas").evaluate("c=>{const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]===d[i+1]&&d[i]===d[i+2]&&d[i+3])n++;return n;}")>100
                check_layout(small=True)
                page.screenshot(path=str(folder/"small.png"),animations="disabled")
                handle=page.locator("[data-resize=library]")
                handle.focus();before=handle.get_attribute("aria-valuenow");handle.press("ArrowRight")
                assert int(handle.get_attribute("aria-valuenow"))>int(before)
                page.locator("#result-pane [data-window-action=maximize]").click()
                panel=page.locator("#result-pane").bounding_box()
                library=page.locator(".library").bounding_box()
                assert panel["x"]>=library["x"]+library["width"]
                expect(page.locator("#toggle-data-results")).to_be_visible()
                page.locator("#result-pane [data-window-action=maximize]").click()
                page.locator("#toggle-data-sources").click()
                page.wait_for_timeout(600)
                page.reload()
                expect(page.locator("#toggle-data-sources")).to_have_attribute("aria-expanded","false")
                expect(page.locator("#open-data-library")).to_have_attribute("aria-selected","true")
                for schema in (4,5):
                    page.evaluate("""schema=>{const key=Object.keys(localStorage).find(k=>k.startsWith('eip-workbench-view-3:'));const draft=JSON.parse(localStorage[key]);draft.web_docks.schema=schema;draft.web_docks.libraryHidden=true;draft.web_docks.dataHeight=230;draft.web_docks.tasksCollapsed=false;localStorage[key]=JSON.stringify(draft);}""",schema)
                    page.reload()
                    expect(page.locator(".library")).to_be_visible()
                    expect(page.locator("#library-data")).to_be_visible()
                    assert page.locator(".tasks").bounding_box()["height"]<=29
                page.locator("#toggle-data-sources").click()
                expect(page.locator("#library-sources")).to_be_visible()
                for width in (1200,1600,1200,1024):
                    page.set_viewport_size(dict(width=width,height=768))
                    if width<1280:
                        expect(page.locator(".processing")).not_to_be_visible()
                    else:
                        expect(page.locator(".processing")).to_be_visible()
                    assert page.locator(".preview-workspace").bounding_box()["width"]>=700
                page.locator(".window-menu summary").click()
                page.locator("[data-show-window=inspector]").click()
                expect(page.locator(".processing")).to_be_visible()
                assert not errors,errors
                assert len(computes)==1,computes
                browser.close()
            print(json.dumps(dict(status="passed",output=str(folder),checks="two sidebar tabs, vertically folded inputs/results, compact center tasks, retained preview position, independent selections, exact exports, selected run, reference removal, responsive canvas, saved layout migration"),ensure_ascii=False))
        finally:
            process.terminate()
            try:process.wait(10)
            except subprocess.TimeoutExpired:process.kill();process.wait(5)

if __name__=="__main__":main()
