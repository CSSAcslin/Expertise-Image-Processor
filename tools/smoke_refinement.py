"""Real-browser regression checks in a disposable workspace and server."""
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
    root = Path(__file__).resolve().parents[1]
    output = root / ".test-output" / f"refinement-{int(time.time())}"
    output.mkdir(parents=True)
    tifffile.imwrite(output / "good.tif", np.arange(120 * 160, dtype=np.uint16).reshape(120, 160) * 2)
    (output / "broken.tif").write_text("not an image")
    sequence = output / "sequence"
    sequence.mkdir()
    for i in range(3):
        tifffile.imwrite(sequence / f"frame{i}.tif", np.full((80, 100), i * 100, dtype=np.uint16))
    url = "http://127.0.0.1:8881"
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, str(root / "run_web.py"), "--port", "8881", "--no-browser", "--workspace", str(output)], cwd=root, stdout=log, stderr=log, env=environment, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        try:
            for _ in range(60):
                try:
                    if httpx.get(url + "/api/state", timeout=.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(.1)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge" if sys.platform == "win32" else "chromium", headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 960})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                expect(page.locator("#connection")).to_have_text("本机已连接")
                page.locator("#import").click()
                expect(page.locator("#dialog-submit")).to_be_disabled()
                page.locator("#file-select-all").check()
                page.locator("#dialog-submit").click()
                expect(page.locator("#source-count")).to_have_text("1")
                expect(page.locator("#import-report")).to_contain_text("broken.tif")
                expect(page.locator("#import-report")).to_contain_text("good.tif")
                expect(page.locator("#file-dialog")).to_be_visible()
                page.screenshot(path=str(output / "mixed-import-1440.png"))
                page.locator("#browse-path").fill(str(output / "missing.tif"))
                page.locator("#browse-go").click()
                expect(page.locator("#toast")).to_contain_text("路径不存在")
                page.locator("#dialog-close").click()
                page.locator("#import").click()
                page.locator("#browse-path").fill(str(sequence));page.locator("#browse-go").click()
                expect(page.locator("#file-list .file-row")).to_have_count(3)
                page.locator("#import-mode-sequence").click()
                page.locator("#interval").fill("0.25")
                page.locator("#dialog-submit").click()
                expect(page.locator("#file-dialog")).not_to_be_visible()
                expect(page.locator("#source-count")).to_have_text("2")
                page.locator("#next").click()
                expect(page.locator("#frame-info")).to_contain_text("0.2500 s")
                frame_requests = []
                page.on("request", lambda request: frame_requests.append(request.url) if request.url.endswith("/frame") else None)
                page.locator("#play").click()
                expect(page.locator("#frame-info")).to_contain_text("0.5000 s")
                expect(page.locator("#play")).to_have_attribute("aria-label", "播放")
                request_count = len(frame_requests)
                page.wait_for_timeout(400)
                assert len(frame_requests) == request_count
                expect(page.locator("#next")).to_be_disabled()
                page.locator("#source-list .source-card").filter(has_text="good.tif").click()
                expect(page.locator("#preview-title")).to_have_text("good.tif")
                expect(page.locator("#seek")).to_be_enabled()
                page.locator("#tool-roi").click()
                canvas = page.locator("#input-canvas").bounding_box()
                fit = min((canvas["width"] - 40) / 160, (canvas["height"] - 50) / 120)

                def point(x, y):
                    return (canvas["x"] + canvas["width"] / 2 + (x - 80) * fit, canvas["y"] + canvas["height"] / 2 + (y - 60) * fit)

                page.mouse.move(*point(20, 20));page.mouse.down();page.mouse.move(*point(60, 50), steps=6);page.mouse.up()
                expect(page.locator("#roi-x")).to_have_value("20")
                expect(page.locator("#roi-width")).to_have_value("40")
                page.mouse.move(*point(40, 35));page.mouse.down();page.mouse.move(*point(50, 40), steps=5);page.mouse.up()
                expect(page.locator("#roi-x")).to_have_value("30")
                expect(page.locator("#roi-y")).to_have_value("25")
                page.mouse.move(*point(70, 55));page.mouse.down();page.mouse.move(*point(80, 60), steps=5);page.mouse.up()
                expect(page.locator("#roi-width")).to_have_value("50")
                expect(page.locator("#roi-height")).to_have_value("35")
                page.locator("#roi-x").fill("35");page.locator("#roi-x").press("Tab")
                expect(page.locator("#roi-info")).to_contain_text("x 35")
                page.locator("#roi-undo").click();expect(page.locator("#roi-x")).to_have_value("30")
                page.locator("#roi-redo").click();expect(page.locator("#roi-x")).to_have_value("35")
                expected_mean = float((np.arange(120 * 160, dtype=np.uint16).reshape(120, 160) * 2)[25:60, 35:85].mean())
                expect(page.locator("#roi-statistics")).to_contain_text(f"{expected_mean:.5g}")
                page.locator("#roi-width").fill("200");page.locator("#roi-width").press("Tab")
                expect(page.locator("#roi-width")).to_have_value("50")
                page.locator("#source-list .source-card").filter(has_text="sequence").click()
                expect(page.locator("#preview-title")).to_have_text("sequence")
                expect(page.locator("#roi-info")).to_have_text("全幅图像")
                page.locator("#scrubber").evaluate("el => { for (const v of [0, 1, 2]) { el.value = v; el.dispatchEvent(new Event('input')); } }")
                expect(page.locator("#frame-info")).to_contain_text("0.5000 s")
                page.locator("#scrubber").evaluate("el => { el.value = 0; el.dispatchEvent(new Event('input')); [...document.querySelectorAll('#source-list .source-card')].find(e => e.textContent.includes('good.tif')).click(); }")
                expect(page.locator("#preview-title")).to_have_text("good.tif")
                expect(page.locator("#shape-badge")).to_have_text("160 × 120")
                page.locator("#source-list .source-card").filter(has_text="good.tif").click()
                expect(page.locator("#roi-x")).to_have_value("35")
                page.locator("#algorithm").select_option("intensity");page.locator("#add-step").click()
                page.locator('[data-key="gain"]').fill("2");page.locator('[data-key="gain"]').press("Tab")
                page.locator("#scope").select_option("range")
                page.locator("#range-start").fill("1");page.locator("#range-end").fill("2")
                page.locator("#toggle-library").click()
                page.reload()
                expect(page.locator("#roi-x")).to_have_value("35")
                expect(page.locator('[data-key="gain"]')).to_have_value("2")
                expect(page.locator("#scope")).to_have_value("range")
                expect(page.locator("#range-end")).to_have_value("2")
                expect(page.locator("#toggle-library")).to_have_attribute("aria-pressed", "true")
                page.locator("#toggle-library").click()
                page.locator("#scope").select_option("current")
                page.locator("#trial").click()
                expect(page.locator("#result-label")).to_have_text("当前帧精确试算", timeout=15000)
                page.locator(".display-panel summary").click()
                page.locator("#display-target").select_option("result")
                result_high = float(page.locator("#display-high").input_value())
                page.locator("#display-target").select_option("input")
                input_high = float(page.locator("#display-high").input_value())
                assert result_high != input_high
                page.screenshot(path=str(output / "roi-comparison-1440.png"))
                page.locator("#run").click()
                expect(page.locator("#result-count")).to_have_text("1", timeout=15000)
                page.locator("#result-list .source-card").click()
                expect(page.locator("#promote-result")).to_be_enabled()
                page.locator("#promote-result").click()
                expect(page.locator("#source-count")).to_have_text("3")
                expect(page.locator("#step-count")).to_have_text("0")
                expect(page.locator("#shape-badge")).to_have_text("50 × 35")
                expect(page.locator("#roi-info")).to_have_text("全幅图像")
                expect(page.locator("#flow-summary")).to_contain_text("阶段 2")
                page.locator("#source-list .source-card.selected [data-info]").click()
                expect(page.locator("#source-detail")).to_contain_text("上游阶段 1")
                page.screenshot(path=str(output / "lineage-1440.png"))
                page.locator("#source-close").click()
                page.set_viewport_size({"width": 1024, "height": 768})
                page.screenshot(path=str(output / "workbench-1024.png"))
                assert page.locator("#run").bounding_box()["y"] < 768
                width_before = page.locator("#viewers").bounding_box()["width"]
                page.locator("#focus-preview").click()
                expect(page.locator("#focus-preview")).to_have_attribute("aria-pressed", "true")
                assert page.locator("#viewers").bounding_box()["width"] > width_before * 1.5
                assert page.locator("#viewers").bounding_box()["height"] > 350
                page.screenshot(path=str(output / "focused-preview-1024.png"))
                page.locator("#focus-preview").click()
                expect(page.locator("#run")).to_be_visible()
                page.reload()
                expect(page.locator("#job-count")).to_have_text("2")
                page.wait_for_timeout(1100)
                assert "试算已完成" not in page.locator("#toast").text_content()
                assert "处理完成" not in page.locator("#toast").text_content()
                assert not errors, errors
                print(json.dumps({"status":"passed", "browser_errors":errors,"screenshots":str(output),"checks":["mixed import", "missing path", "sequence interval and EOF", "ROI move/resize/numeric/undo/statistics", "per-source ROI", "refresh draft and layout", "separate display", "promoted result", "lineage", "small viewport and focus"]},ensure_ascii=False))
                browser.close()
        finally:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                server.terminate()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
