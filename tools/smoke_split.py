"""Exercise the split mode in an isolated Edge instance and local workspace."""
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
from playwright.sync_api import sync_playwright,expect


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/".test-output"/f"split-ui-{int(time.time())}"
    output.mkdir(parents=True)
    rng=np.random.default_rng(43)
    n,h,w=31,72,168
    texture=rng.uniform(60,120,(56,64))
    array=np.zeros((n,h,w),dtype=np.float32)
    for i in range(n):
        signal=texture+8*np.sin(i*.4)+rng.normal(0,1.8,texture.shape)
        array[i,8:64,12:76]=signal
        array[i,8:64,94:158]=np.roll(signal,1,axis=1)*.7+7+rng.normal(0,1,texture.shape)
    path=output/"split_sample.tif"
    tifffile.imwrite(path,array,photometric="minisblack",metadata={"axes":"TYX"})
    port=8884
    url=f"http://127.0.0.1:{port}"
    environment=dict(os.environ,PYTHONIOENCODING="utf-8")
    with (output/"server.log").open("w",encoding="utf-8") as log:
        server=subprocess.Popen([sys.executable,str(root/"run_web.py"),"--port",str(port),"--no-browser","--workspace",str(output)],cwd=root,stdout=log,stderr=log,env=environment,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        try:
            for _ in range(80):
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:break
                except httpx.HTTPError:time.sleep(.1)
            with httpx.Client(base_url=url,timeout=30) as api,sync_playwright() as pw:
                api.headers["X-EIP-Token"]=api.get("/api/state").json()["session_token"]
                opened=api.post("/api/sources/open",json={"path":str(path),"interval":.01})
                opened.raise_for_status()
                browser=pw.chromium.launch(channel="msedge" if sys.platform=="win32" else "chromium",headless=True)
                page=browser.new_page(viewport={"width":1440,"height":960},accept_downloads=True)
                errors=[]
                page.on("pageerror",lambda error:errors.append(str(error)))
                page.goto(url)
                expect(page.locator("#connection")).to_have_text("本机已连接")
                page.locator("#analysis-mode").select_option("split_correlation")
                expect(page.locator('[data-split-rect="left"] [data-key="width"]')).to_have_value("64")
                expect(page.locator("#split-line-top")).to_have_value("84")
                page.locator("#split-calibrate").click()
                expect(page.locator("#split-apply-proposal")).to_be_enabled()
                page.locator("#split-apply-proposal").click()
                expect(page.locator("#split-dx")).to_have_value("1")
                page.screenshot(path=str(output/"01-calibration.png"))
                page.locator("#split-accept").click()
                expect(page.locator('[data-split-stage="preprocess"]')).to_have_class("selected")
                page.locator("#split-semantics").select_option("linear_intensity")
                page.locator("#split-statistics").click()
                expect(page.locator("#split-stat-status")).to_contain_text("完整全源统计已保存",timeout=20000)
                page.locator('[data-split-stage="trial"]').click()
                for id,value in [("split-window",5),("split-stride",2),("split-lag-min",-3),("split-lag-max",3),("split-center",16),("split-point-x",4),("split-point-y",7)]:
                    page.locator("#"+id).fill(str(value));page.locator("#"+id).dispatch_event("change")
                page.locator("#split-set-point").click()
                page.locator("#split-curve-run").click()
                expect(page.locator("#split-curve svg")).to_be_visible(timeout=20000)
                expect(page.locator("#split-curve-readout")).to_contain_text("点 (4, 7)")
                page.locator("#split-curve-axis").select_option("time")
                page.locator("#split-result-lag").select_option("0")
                expect(page.locator("#split-curve-readout")).to_contain_text("选定 q=-3")
                page.screenshot(path=str(output/"02-point-curve.png"))
                with page.expect_download() as download:
                    page.locator('[data-split-export="csv"]').click()
                csv_path=output/"point.csv";download.value.save_as(csv_path)
                with csv_path.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
                assert [int(r["lag_frames"]) for r in rows]==list(range(-3,4))
                assert all(int(r["center_index"])==15 for r in rows)
                page.locator('[data-split-stage="trial"]').click()
                page.locator("#split-tool").select_option("roi")
                box=page.locator("#split-canvas").bounding_box()
                shape=page.locator("#split-view-caption").inner_text().split(" · ")[0].split(" × ")
                rw,rh=map(int,shape)
                scale=min((box["width"]-40)/rw,(box["height"]-50)/rh)
                def screen(x,y):return (box["x"]+box["width"]/2+(x-rw/2)*scale,box["y"]+box["height"]/2+(y-rh/2)*scale)
                x,y=screen(5,5);x2,y2=screen(20,18)
                page.mouse.move(x,y);page.mouse.down();page.mouse.move(x2,y2,steps=8);page.mouse.up()
                expect(page.locator("#split-roi-status")).to_contain_text("分析 ROI")
                page.locator("#split-roi-x").locator("xpath=ancestor::details").locator("summary").click()
                for key,value in [("x",5),("y",5),("width",15),("height",13)]:page.locator("#split-roi-"+key).fill(str(value))
                page.locator("#split-apply-roi").click()
                expect(page.locator("#split-roi-status")).to_contain_text("15 × 13")
                page.locator("#split-clear-points").click()
                page.locator("#split-curve-run").click()
                expect(page.locator("#split-curve-readout")).to_contain_text("逐像素 g² 空间平均",timeout=20000)
                page.screenshot(path=str(output/"03-roi-curve.png"))
                page.locator('[data-split-stage="trial"]').click()
                page.locator("#split-roi-estimator").select_option("intensity_sum")
                page.locator("#split-curve-run").click()
                expect(page.locator("#split-curve-readout")).to_contain_text("先汇总强度再算 g²",timeout=20000)
                page.locator('[data-split-stage="trial"]').click()
                page.locator("#split-trial").click()
                expect(page.locator("#split-view-caption")).to_contain_text("有效像素",timeout=20000)
                page.locator("#split-result-lag").select_option("0")
                expect(page.locator("#split-window-note")).to_contain_text("右窗 11～15")
                box=page.locator("#split-canvas").bounding_box()
                page.mouse.move(box["x"]+box["width"]/2,box["y"]+box["height"]/2)
                expect(page.locator("#split-pixel")).to_contain_text("g²")
                with page.expect_download() as download:page.locator('[data-split-export="image"]').click()
                image=output/"selected.tif";download.value.save_as(image)
                assert tifffile.imread(image).dtype==np.float64
                page.screenshot(path=str(output/"04-map-negative-lag.png"))
                page.locator('[data-split-stage="trial"]').click()
                page.locator("#split-run").click()
                expect(page.locator("#split-result-time")).to_have_attribute("max","10",timeout=20000)
                page.locator("#split-result-lag").select_option("1")
                page.locator("#split-result-time").fill("10");page.locator("#split-result-time").dispatch_event("change")
                expect(page.locator("#split-view-caption")).to_contain_text("中心帧 26")
                page.reload()
                expect(page.locator("#analysis-mode")).to_have_value("split_correlation")
                expect(page.locator("#split-result-lag")).to_have_value("1")
                expect(page.locator("#split-result-time")).to_have_value("10")
                expect(page.locator("#split-view-caption")).to_contain_text("中心帧 26")
                page.set_viewport_size({"width":1100,"height":760})
                page.screenshot(path=str(output/"05-small-window.png"))
                assert page.evaluate("document.body.scrollWidth<=window.innerWidth"),"horizontal overflow"
                page.locator("#split-promote").click()
                expect(page.locator("#analysis-mode")).to_have_value("general")
                expect(page.locator("#preview-title")).to_contain_text("派生 g²")
                page.screenshot(path=str(output/"06-chained-result.png"))
                promoted=next(s for s in api.get("/api/state").json()["sources"] if s["source"]["kind"]=="correlation")
                assert promoted["lineage"][-1]["lag_frames"]==-2
                page.locator(f'[data-info="{promoted["id"]}"]').click()
                expect(page.locator("#source-detail")).to_contain_text("延迟：-2 帧")
                page.locator("#source-dialog").evaluate("dialog=>dialog.close()")
                assert not errors,errors
                summary={"ui_errors":errors,"source_count":len(api.get("/api/state").json()["sources"]),"checks":["static calibration","full-source statistics","signed point curve","both ROI curve definitions","negative map and TIFF export","time-lag navigation","small window layout","lag sequence chaining"]}
                (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
                browser.close()
                print(str(output))
        finally:
            server.terminate()
            try:server.wait(timeout=8)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=3)


if __name__=="__main__":main()
