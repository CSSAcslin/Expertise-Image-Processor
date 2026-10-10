"""Isolated browser validation for split windows and the editable module workflow."""
from pathlib import Path
import csv
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
    output=root/".test-output"/f"split-workflow-ui-{int(time.time())}"
    output.mkdir(parents=True)
    rng=np.random.default_rng(177)
    array=np.zeros((31,72,168),dtype=np.uint16)
    texture=rng.uniform(100,600,(56,64))
    for i in range(31):
        signal=texture+20*np.sin(i*.4)+rng.normal(0,3,texture.shape)
        array[i,8:64,12:76]=signal.astype(np.uint16)
        array[i,8:64,94:158]=(np.roll(signal,1,axis=1)*.7+8).astype(np.uint16)
    path=output/"sample.tif"
    tifffile.imwrite(path,array,photometric="minisblack",metadata={"axes":"TYX"})
    port=8885
    url=f"http://127.0.0.1:{port}"
    with (output/"server.log").open("w",encoding="utf-8") as log:
        server=subprocess.Popen([sys.executable,str(root/"run_web.py"),"--port",str(port),"--no-browser","--workspace",str(output)],cwd=root,stdout=log,stderr=log,env=dict(os.environ,PYTHONIOENCODING="utf-8"),creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:break
                except httpx.HTTPError:time.sleep(.1)
            with httpx.Client(base_url=url,timeout=30) as api,sync_playwright() as pw:
                api.headers["X-EIP-Token"]=api.get("/api/state").json()["session_token"]
                api.post("/api/sources/open",json={"path":str(path),"interval":.00001}).raise_for_status()
                browser=pw.chromium.launch(channel="msedge" if sys.platform=="win32" else "chromium",headless=True)
                page=browser.new_page(viewport={"width":1440,"height":960},accept_downloads=True)
                errors=[]
                page.on("pageerror",lambda e:errors.append(str(e)))
                page.goto(url)
                expect(page.locator("#connection")).to_have_text("本机已连接")
                page.locator("#analysis-mode").select_option("split_correlation")
                expect(page.locator('[data-split-rect="left"] [data-key="width"]')).to_have_value("64")
                page.locator("#split-calibrate").click()
                expect(page.locator("#split-apply-proposal")).to_be_enabled()
                page.locator("#split-apply-proposal").click()
                expect(page.locator("#split-dx")).to_have_value("1")
                page.locator("#split-accept").click()
                expect(page.locator("#split-photo-curve")).to_have_value("linear")
                expect(page.locator("#split-normalization")).to_have_value("none")
                page.locator("#split-photo-curve").select_option("sqrt")
                expect(page.locator("#split-lut-formula")).to_contain_text("改变强度涨落")
                page.locator("#split-photo-curve").select_option("linear")
                page.locator("#split-statistics").click()
                expect(page.locator("#split-stat-status")).to_contain_text("完整全源统计已保存",timeout=25000)
                expect(page.locator(".split-data-window")).to_be_visible()
                expect(page.locator("#split-result-canvas")).to_be_visible()
                expect(page.locator("#result-list .split-result-card")).to_have_count(1)
                page.locator('[data-split-stage="trial"]').click()
                for id,value in [("split-window",5),("split-stride",2),("split-map-lag",-2),("split-center",16)]:
                    page.locator("#"+id).fill(str(value));page.locator("#"+id).dispatch_event("change")
                page.locator("#split-view").select_option("left")
                page.locator("#split-roi-draw").click()
                expect(page.locator("#split-tool")).to_have_value("roi")
                box=page.locator("#split-canvas").bounding_box()
                rw,rh=map(int,page.locator("#split-view-caption").inner_text().split(" · ")[0].split(" × "))
                scale=min((box["width"]-40)/rw,(box["height"]-50)/rh)
                def screen(x,y):return (box["x"]+box["width"]/2+(x-rw/2)*scale,box["y"]+box["height"]/2+(y-rh/2)*scale)
                x,y=screen(5,5);xx,yy=screen(20,18)
                page.mouse.move(x,y);page.mouse.down();page.mouse.move(xx,yy,steps=8);page.mouse.up()
                expect(page.locator("#split-roi-status")).to_contain_text("分析 ROI")
                details=page.locator("#split-roi-x").locator("xpath=ancestor::details")
                details.locator("summary").click()
                for key,value in [("x",5),("y",5),("width",15),("height",13)]:page.locator("#split-roi-"+key).fill(str(value))
                page.locator("#split-apply-roi").click()
                page.locator("#split-job-name").fill("样本 A · 空间图 τ−2")
                page.locator("#split-trial").click()
                expect(page.locator("#split-result-caption")).to_contain_text("有效像素",timeout=25000)
                expect(page.locator("#split-result-lag option")).to_have_count(1)
                expect(page.locator("#split-result-title")).to_have_text("样本 A · 空间图 τ−2")
                expect(page.locator("#split-view")).to_have_value("left")
                expect(page.locator("#split-roi-status")).to_contain_text("15 × 13")
                with page.expect_download() as download:page.locator('[data-split-export="image"]').click()
                download.value.save_as(output/"space.tif")
                assert tifffile.imread(output/"space.tif").shape==(13,15)
                page.screenshot(path=str(output/"01-two-windows.png"))
                page.locator('[data-split-stage="trial"]').click()
                page.locator('[data-split-analysis="curve"]').click()
                expect(page.locator("#split-map-lag-field")).to_be_hidden()
                expect(page.locator("#split-run")).to_be_hidden()
                for id,value in [("split-lag-min",-3),("split-lag-max",3),("split-point-x",8),("split-point-y",8)]:
                    page.locator("#"+id).fill(str(value));page.locator("#"+id).dispatch_event("change")
                page.locator("#split-set-point").click()
                page.locator("#split-job-name").fill("样本 A · τ 扫描")
                page.locator("#split-curve-run").click()
                expect(page.locator("#split-curve svg")).to_be_visible(timeout=25000)
                expect(page.locator("#split-result-lag option")).to_have_count(7)
                expect(page.locator("#split-view")).to_have_value("left")
                page.locator("#split-result-name").fill("ROI A · W5 · 延迟扫描")
                page.locator("#split-result-rename").click()
                expect(page.locator("#split-result-title")).to_have_text("ROI A · W5 · 延迟扫描")
                with page.expect_download() as download:page.locator('[data-split-export="csv"]').click()
                download.value.save_as(output/"curve.csv")
                with (output/"curve.csv").open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
                assert sorted(set(int(r["lag_frames"]) for r in rows))==list(range(-3,4))
                page.screenshot(path=str(output/"02-curve.png"))
                page.locator('[data-split-stage="trial"]').click()
                page.locator("#split-roi-estimator").select_option("intensity_sum")
                page.locator("#split-job-name").fill("ROI A · 总强度曲线")
                page.locator("#split-curve-run").click()
                expect(page.locator("#split-curve-readout")).to_contain_text("先汇总强度再算 g²",timeout=25000)
                page.locator('[data-split-stage="flow"]').click()
                expect(page.locator(".flow-node")).to_have_count(4)
                page.locator("#split-module-add").click()
                expect(page.locator(".flow-node")).to_have_count(5)
                node=page.locator('.flow-node[data-node="2"]')
                node.locator('[data-param="gain"]').fill("2");node.locator('[data-param="gain"]').dispatch_event("change")
                node.locator('[data-duplicate]').click()
                expect(page.locator(".flow-node")).to_have_count(6)
                node=page.locator('.flow-node[data-node="3"]')
                node.locator('[data-param="gain"]').fill("1");node.locator('[data-param="gain"]').dispatch_event("change")
                node.locator('[data-param="offset"]').fill("3");node.locator('[data-param="offset"]').dispatch_event("change")
                node.locator('[data-move="-1"]').click()
                expect(page.locator('.flow-node[data-node="2"] [data-param="offset"]')).to_have_value("3")
                # Exercise actual HTML drag / drop as well as accessible arrow controls.
                page.locator('.flow-node[data-node="2"] .flow-grip').drag_to(page.locator('.flow-node[data-node="3"]'))
                expect(page.locator('.flow-node[data-node="2"] [data-param="gain"]')).to_have_value("2")
                expect(page.locator("#split-flow-status")).to_contain_text("按列表顺序执行")
                page.screenshot(path=str(output/"03-module-flow.png"))
                page.locator('[data-split-stage="trial"]').click()
                page.locator('[data-split-analysis="map"]').click()
                page.locator("#split-job-name").fill("模块流程 · 空间序列")
                page.locator("#split-run").click()
                expect(page.locator("#split-result-title")).to_have_text("模块流程 · 空间序列",timeout=25000)
                jobs=api.get("/api/jobs").json()
                job=next(j for j in jobs if j["name"]=="模块流程 · 空间序列")
                full=api.get(f"/api/jobs/{job['id']}").json()
                assert full["run"]["lags"]==[-2]
                assert [n["parameters"].get("offset") for n in full["run"]["config"]["workflow"] if n["algorithm"]=="affine"]==[0,3]
                page.locator(f'[data-detail="{job["id"]}"]').click()
                expect(page.locator("#job-detail")).to_contain_text("分析 ROI")
                expect(page.locator("#job-detail")).to_contain_text("最少有效样本")
                page.locator("#job-rename-input").fill("可复现实验 · 空间序列")
                page.locator("#job-rename-save").click()
                expect(page.locator("#result-list")).to_contain_text("可复现实验 · 空间序列")
                page.locator("#job-close").click()
                page.reload()
                expect(page.locator("#analysis-mode")).to_have_value("split_correlation")
                page.locator('[data-split-stage="flow"]').click()
                expect(page.locator(".flow-node")).to_have_count(6)
                page.set_viewport_size({"width":1100,"height":760})
                page.wait_for_timeout(150)
                bottom=page.locator(".split-run-bar").bounding_box()["y"]
                for selector in (".split-input-timeline","#split-result-nav"):
                    box=page.locator(selector).bounding_box()
                    assert box["y"]+box["height"]<=bottom+1,(selector,box,bottom)
                page.screenshot(path=str(output/"04-small-window.png"))
                page.locator('[data-split-stage="result"]').click()
                page.locator("#split-promote").click()
                expect(page.locator("#analysis-mode")).to_have_value("general")
                promoted=next(s for s in api.get("/api/state").json()["sources"] if s["source"]["kind"]=="correlation")
                assert promoted["lineage"][-1]["lag_frames"]==-2
                expect(page.locator("#source-list")).to_contain_text("派生 g²")
                page.screenshot(path=str(output/"05-next-stage.png"))
                assert not errors,errors
                (output/"report.json").write_text(json.dumps({"status":"passed","job":job["id"],"browser_errors":errors},ensure_ascii=False,indent=2),encoding="utf-8")
                browser.close()
                print(output)
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=="__main__":main()
