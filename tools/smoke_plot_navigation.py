"""Focused browser verification of dock boundaries and display-only 1D axis zoom."""
from pathlib import Path
import json
import math
import os
import re
import subprocess
import sys
import time

import httpx
import numpy as np
import tifffile
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eip.module_registry import new_node


def main():
    output = ROOT / ".test-output" / f"plot-navigation-{int(time.time())}"
    output.mkdir(parents=True)
    pixels = (np.arange(60*4*32, dtype=np.uint16).reshape(60, 4, 32)%128)*60 + np.arange(60, dtype=np.uint16)[:, None, None]%16
    tifffile.imwrite(output / "source.tif", pixels, photometric="minisblack", metadata={"axes": "TYX"})
    url = "http://127.0.0.1:8897"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, str(ROOT / "run_web.py"), "--port", "8897", "--no-browser", "--workspace", str(output)],
            cwd=ROOT, stdout=log, stderr=log, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0,
        )
        try:
            for _ in range(100):
                try:
                    if httpx.get(url + "/api/state", timeout=.5).status_code==200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise AssertionError("isolated server did not start")
            with httpx.Client(base_url=url, timeout=30) as api, sync_playwright() as pw:
                api.headers["X-EIP-Token"] = api.get("/api/state").json()["session_token"]
                opened = api.post("/api/sources/open", json=dict(path=str(output / "source.tif"), interval=.01))
                opened.raise_for_status()
                source = opened.json()["source"]

                def run(algorithm, **parameters):
                    node = new_node(algorithm)
                    node["parameters"].update(parameters)
                    response = api.post("/api/jobs", json=dict(contract_version=3, mode="run", source_ids=[source["id"]], steps=[node], output_dir=str(output / algorithm)))
                    response.raise_for_status()
                    identifier = response.json()["id"]
                    for _ in range(150):
                        job = api.get("/api/jobs/" + identifier).json()
                        if job["status"] in {"succeeded", "failed", "cancelled", "completed_with_errors"}:
                            break
                        time.sleep(.1)
                    assert job["status"]=="succeeded", job
                    return identifier

                point = run("point_extract", x=2, y=1, output="main")
                histogram = run("distribution", bins=32, minimum=0, maximum=7680, output="main")
                profile = run("line_extract", mode="segment", x0=0, x1=31, y0=1, y1=1, samples=32, output="main")
                browser = pw.chromium.launch(channel="msedge" if sys.platform=="win32" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1600, height=1000))
                errors, calculations = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda r: calculations.append(r.url) if r.method=="POST" and (r.url.endswith("/api/jobs") or r.url.endswith("/api/modules/inspect") or r.url.endswith("/sequence")) else None)
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 60")
                page.locator("[data-window='inspector'] [data-window-action='hide']").click()
                page.locator("[data-library-tab='sources']").click()
                chart = page.locator("#split-curve svg")

                def drag_separator(key, dx=0, dy=0):
                    separator = page.locator(f"[data-resize='{key}']")
                    r = separator.bounding_box()
                    x, y = r["x"]+r["width"]/2, r["y"]+r["height"]/2
                    page.mouse.move(x, y);page.mouse.down()
                    page.mouse.move(x+dx, y+dy, steps=8);page.mouse.up()
                    assert page.locator(".workspace").get_attribute("data-resizing") is None

                width = lambda: page.locator(".library").bounding_box()["width"]
                height = lambda: page.locator(".tasks").bounding_box()["height"]
                initial_preview = page.locator(".preview-workspace").bounding_box()
                drag_separator("library", dx=90)
                assert width()==315, width()
                page.locator("#open-module-library").click()
                assert width()==315
                page.locator("[data-library-tab='sources']").click()
                drag_separator("tasks", dy=-130)
                assert height()==172, height()
                assert page.locator(".preview-workspace").bounding_box()["height"]<initial_preview["height"]
                page.locator("#toggle-tasks").click()
                assert height()==42
                page.locator("#toggle-tasks").click()
                assert height()==172, "expanding must retain the chosen height"
                page.wait_for_timeout(450)
                page.reload()
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 60")
                assert width()==315 and height()==172, "dock sizes must survive a page reload"
                drag_separator("library", dx=900)
                drag_separator("tasks", dy=-450)
                assert width()<=520
                assert height()<=float(page.locator("[data-resize='tasks']").get_attribute("aria-valuemax"))
                page.set_viewport_size(dict(width=1100, height=760))
                page.wait_for_timeout(180)
                task_rect = page.locator(".tasks").bounding_box()
                assert task_rect["y"]+task_rect["height"]<=760, task_rect
                assert page.locator(".preview-workspace").bounding_box()["height"]>=300
                assert page.locator(".timeline").bounding_box()["y"]+page.locator(".timeline").bounding_box()["height"]<=page.locator(".workbench-runbar").bounding_box()["y"]
                page.screenshot(path=str(output / "dock-boundaries-compact.png"), full_page=True)
                page.set_viewport_size(dict(width=1600, height=1000))
                page.locator("[data-resize='library']").dblclick()
                page.locator("[data-resize='tasks']").dblclick()
                assert width()==225 and height()==42
                page.locator("[data-resize='library']").press("ArrowRight")
                page.locator("[data-resize='tasks']").press("ArrowUp")
                assert width()==235 and height()==100
                page.locator("[data-resize='tasks']").press("ArrowDown")
                assert height()==42
                page.locator("[data-resize='library']").dblclick()
                panels = dict(library_drag=True, tasks_drag=True, folded_drag_expands=True, module_tab_retains_width=True,
                              collapse_retains_height=True, reload_preserves_sizes=True, compact_bounds=True, keyboard=True, double_click_reset=True)

                def domain():
                    return chart.evaluate("e=>[Number(e.dataset.xMin),Number(e.dataset.xMax)]")

                def y_domain():
                    return chart.evaluate("e=>[Number(e.dataset.yMin),Number(e.dataset.yMax)]")

                def zoom_y():
                    x_before, before = domain(), y_domain()
                    point = chart.evaluate("e=>{const r=e.querySelector('clipPath rect'),p=new DOMPoint(+r.getAttribute('x')-22,+r.getAttribute('y')+.55*(+r.getAttribute('height'))).matrixTransform(e.getScreenCTM());return {x:p.x,y:p.y};}")
                    page.mouse.move(**point)
                    chart.evaluate("e=>e.addEventListener('wheel',v=>{const p=new DOMPoint(v.clientX,v.clientY).matrixTransform(e.getScreenCTM().inverse()),r=e.querySelector('clipPath rect');window.plotTestYAnchor=1-(p.y-Number(r.getAttribute('y')))/Number(r.getAttribute('height'));},{once:true,capture:true})")
                    page.mouse.wheel(0, -650)
                    page.wait_for_function("old=>{const e=document.querySelector('#split-curve svg');return Number(e.dataset.yMax)-Number(e.dataset.yMin)<old;}", arg=before[1]-before[0])
                    after, anchor = y_domain(), page.evaluate("window.plotTestYAnchor")
                    assert domain()==x_before, "Y-axis wheel must not change X"
                    assert math.isclose(before[0]+anchor*(before[1]-before[0]), after[0]+anchor*(after[1]-after[0]), rel_tol=1e-9, abs_tol=1e-9), (before, after)
                    page.mouse.wheel(0, 650)
                    page.wait_for_function("old=>{const e=document.querySelector('#split-curve svg');return Number(e.dataset.yMax)-Number(e.dataset.yMin)>old;}", arg=after[1]-after[0])
                    assert math.isclose(y_domain()[1]-y_domain()[0], before[1]-before[0], rel_tol=1e-9)
                    page.mouse.wheel(0, -650)
                    page.wait_for_function("old=>{const e=document.querySelector('#split-curve svg');return Number(e.dataset.yMax)-Number(e.dataset.yMin)<old;}", arg=before[1]-before[0])
                    return before, y_domain()

                def coords(fraction=.4):
                    return chart.evaluate("(e,f)=>{const r=e.querySelector('clipPath rect'),p=new DOMPoint(+r.getAttribute('x')+f*(+r.getAttribute('width')),+r.getAttribute('y')+.5*(+r.getAttribute('height'))).matrixTransform(e.getScreenCTM());return {x:p.x,y:p.y};}", fraction)

                def zoom():
                    before = domain()
                    page.mouse.move(**coords())
                    chart.evaluate("e=>e.addEventListener('wheel',v=>{const p=new DOMPoint(v.clientX,v.clientY).matrixTransform(e.getScreenCTM().inverse()),r=e.querySelector('clipPath rect');window.plotTestAnchor=(p.x-Number(r.getAttribute('x')))/Number(r.getAttribute('width'));},{once:true,capture:true})")
                    page.mouse.wheel(0, -650)
                    page.wait_for_function("old=>{const e=document.querySelector('#split-curve svg');return Number(e.dataset.xMax)-Number(e.dataset.xMin)<old;}", arg=before[1]-before[0])
                    after = domain()
                    anchor = page.evaluate("window.plotTestAnchor")
                    assert math.isclose(before[0]+anchor*(before[1]-before[0]), after[0]+anchor*(after[1]-after[0]), rel_tol=1e-9, abs_tol=1e-9), (before, after)
                    return before, after

                def pan():
                    before = domain()
                    p = coords(.5)
                    page.mouse.move(**p);page.mouse.down()
                    page.mouse.move(p["x"]-55, p["y"], steps=7);page.mouse.up()
                    page.wait_for_function("old=>Number(document.querySelector('#split-curve svg').dataset.xMin)>old", arg=before[0])
                    after = domain()
                    assert math.isclose(after[1]-after[0], before[1]-before[0], rel_tol=1e-9)
                    assert page.locator("#split-curve").evaluate("e=>!e.classList.contains('plot-dragging')")
                    return after

                def reset(full):
                    page.mouse.dblclick(**coords())
                    page.wait_for_function("full=>{const e=document.querySelector('#split-curve svg');return Number(e.dataset.xMin)===full[0]&&Number(e.dataset.xMax)===full[1];}", arg=full)

                report = {"panels":panels}
                for identifier, name in ((point, "point"), (histogram, "histogram"), (profile, "profile")):
                    print(f"checking {name}", flush=True)
                    page.locator("#manage-results").click()
                    page.locator(f"[data-sequence-result='{identifier}']").click()
                    expect(page.locator("#result-meta")).to_contain_text({"point":"T:60", "histogram":"B:32", "profile":"R:32"}[name])
                    expect(chart).to_be_visible()
                    if name=="profile":
                        page.locator("#open-result-tools").click()
                        page.locator("#sequence-plot-axis").select_option("component")
                        page.locator("#result-tools-dialog [data-close-dialog]").click()
                        expect(chart.locator("text").last).to_contain_text("剖面距离")
                    request_count = len(calculations)
                    signature = chart.evaluate("e=>e.querySelectorAll('.histogram-bin').length?Array.from(e.querySelectorAll('.histogram-bin'),b=>b.dataset.frequency):null")
                    full, zoomed = zoom()
                    full_y, zoomed_y = zoom_y()
                    moved = pan()
                    assert y_domain()==zoomed_y, "horizontal dragging must preserve Y"
                    if name=="point":
                        readout = page.locator("#sequence-readout").inner_text()
                        sample = int(re.search(r"样本 (\d+)", readout)[1])-1
                        assert float(readout.split(" · ")[-1])==float(pixels[sample, 1, 2]), readout
                    page.screenshot(path=str(output / f"{name}-zoomed.png"), full_page=True)
                    assert len(calculations)==request_count, "navigation must not fetch/recompute data"
                    if signature is not None:
                        assert signature==chart.evaluate("e=>Array.from(e.querySelectorAll('.histogram-bin'),b=>b.dataset.frequency)")
                    # Window changes must keep the same scientific domain and viewport.
                    page.set_viewport_size(dict(width=1100, height=760))
                    page.wait_for_timeout(180)
                    assert domain()==moved, (domain(), moved)
                    assert y_domain()==zoomed_y, "window resizing must preserve Y"
                    page.screenshot(path=str(output / f"{name}-compact.png"), full_page=True)
                    if name in {"histogram", "profile"}:
                        page.locator("#sequence-index").fill("1")
                        page.locator("#sequence-index").press("Tab")
                        if name=="histogram":
                            expect(page.locator("#sequence-readout")).to_contain_text("样本 2")
                        else:
                            page.wait_for_timeout(180)
                        assert domain()==moved, "frame navigation must preserve horizontal view"
                        assert y_domain()==zoomed_y, "frame navigation must retain manual Y limits"
                    reset(full)
                    assert math.isclose(y_domain()[1]-y_domain()[0], full_y[1]-full_y[0], rel_tol=1e-9), "double click must fit both axes"
                    page.set_viewport_size(dict(width=1600, height=1000))
                    page.wait_for_timeout(180)
                    zoom()
                    page.locator("[data-view='result']").click()
                    page.locator("#fit").click()
                    page.wait_for_function("full=>{const e=document.querySelector('#split-curve svg');return Number(e.dataset.xMin)===full[0]&&Number(e.dataset.xMax)===full[1];}", arg=full)
                    expect(page.locator("#zoom-value")).to_have_text("100%")
                    # Repeated wheel input has bounded zoom and readable, distinct ticks.
                    page.mouse.move(**coords())
                    for _ in range(7):
                        page.mouse.wheel(0, -1000)
                        page.wait_for_timeout(30)
                    page.wait_for_timeout(120)
                    tight = domain()
                    assert math.isclose((full[1]-full[0])/(tight[1]-tight[0]), 1000, rel_tol=1e-9)
                    ticks = chart.locator(".plot-x-tick").all_text_contents()
                    assert len(set(ticks))==len(ticks), ticks
                    reset(full)
                    page.set_viewport_size(dict(width=1600, height=1000))
                    report[name] = dict(full=full, zoomed=zoomed, panned=moved, full_y=full_y, zoomed_y=zoomed_y, independent_y_zoom=True, y_anchor_preserved=True, anchor_preserved=True, resize_preserved=True, reset=True, display_only=True)
                # Historical signed-delay curves share exactly the same navigation.
                # Only their existing result payload is supplied; no correlation run is needed.
                lags = list(range(-10, 11))
                refs = [dict(center_index=15, lag_frames=q, lag_seconds=q*.01, left_start=13, left_end=18, right_start=13+q, right_end=18+q) for q in lags]
                legacy = dict(directory=str(output), run=dict(task="curve", display_name="历史延迟曲线", config=dict(window=5), lags=lags),
                              curve_result=dict(window_refs=refs, config=dict(window=5), coefficient_reference=1,
                                                curves=[dict(name="点信号", estimator="point", pixel_count=1, g2=[1+q*q*.001 for q in lags], counts_min=[5]*21, counts_max=[5]*21)]))
                page.route("**/api/split/jobs/plot-navigation-legacy", lambda route: route.fulfill(json=legacy))
                page.locator("#split-result-select").evaluate("e=>{e.add(new Option('历史延迟曲线','plot-navigation-legacy'));e.value='plot-navigation-legacy';e.dispatchEvent(new Event('change'));}")
                expect(chart).to_have_attribute("aria-label", "g² 随正负延迟变化曲线")
                full, zoomed = zoom()
                full_y, zoomed_y = zoom_y()
                page.mouse.click(**coords(.7))
                selected = page.locator("#split-result-lag").input_value()
                moved = pan()
                assert page.locator("#split-result-lag").input_value()==selected, "drag must not select a different delay"
                reset(full)
                page.screenshot(path=str(output / "legacy-delay.png"), full_page=True)
                page.locator("#split-curve-axis").select_option("time", force=True)
                assert domain()==[-.1, .1], domain()
                page.locator("#manage-results").click()
                page.locator(f"[data-sequence-result='{point}']").click()
                expect(page.locator("#result-meta")).to_contain_text("T:60")
                expect(chart.locator("text").last).to_have_text("编辑后样本号")
                page.set_viewport_size(dict(width=1100, height=760))
                page.wait_for_timeout(180)
                expect(chart.locator("text").last).to_have_text("编辑后样本号")
                report["legacy_delay"] = dict(full=full, zoomed=zoomed, panned=moved, full_y=full_y, zoomed_y=zoomed_y, independent_y_zoom=True, click_preserved=True, drag_does_not_select=True, axis_change_resets=True, return_to_module_plot=True)
                assert len(api.get("/api/jobs").json())==3
                assert not errors, errors
                report["javascript_errors"] = errors
                (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(dict(output=str(output), **report), ensure_ascii=False))
                browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill();server.wait(timeout=5)


if __name__=="__main__":
    main()
