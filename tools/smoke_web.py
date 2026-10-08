"""Exercise the real local Web workbench in an isolated headless browser.

Start run_web.py --no-browser first. No user browser profile is accessed.
"""
from pathlib import Path
import argparse
import json
import sys

from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, help="截图及导出验证目录")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root / ".test-output" / "web-ui").resolve()
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge" if sys.platform == "win32" else "chromium", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.locator("#connection")).to_have_text("本机已连接")
        if page.locator("#source-count").inner_text() == "0":
            page.screenshot(path=str(output / "empty-1440.png"))
            page.locator("#demo").click()
        expect(page.locator("#source-list .source-card")).to_have_count(2, timeout=20000)
        page.locator("#source-list .source-card").filter(has_text="粒子运动").click()
        previous_results = int(page.locator("#result-count").inner_text())
        expect(page.locator("#preview-title")).to_contain_text("粒子运动")
        expect(page.locator("#seek")).to_be_enabled()
        page.locator("#seek-value").fill("0.8")
        page.locator("#seek").click()
        expect(page.locator("#frame-info")).to_contain_text("0.8333 s")
        page.locator("#next").click()
        expect(page.locator("#frame-info")).to_contain_text("0.9167 s")
        page.locator("#previous").click()
        expect(page.locator("#frame-info")).to_contain_text("0.8333 s")
        page.locator("#algorithm").select_option("box_blur")
        page.locator("#add-step").click()
        radius = page.locator('[data-key="radius"]')
        radius.fill("2"); radius.press("Tab")
        page.locator("#tool-roi").click()
        box = page.locator("#input-canvas").bounding_box()
        page.mouse.move(box["x"] + box["width"]*.38, box["y"] + box["height"]*.36)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"]*.64, box["y"] + box["height"]*.63, steps=8)
        page.mouse.up()
        expect(page.locator("#roi-info")).to_contain_text("px")
        page.locator("#trial").click()
        expect(page.locator("#viewers")).to_have_attribute("data-mode", "compare", timeout=15000)
        expect(page.locator("#result-label")).to_have_text("当前帧精确试算")
        page.screenshot(path=str(output / "trial-1440.png"))
        radius.fill("3"); radius.press("Tab")
        expect(page.locator("#stale")).to_be_visible()
        page.locator("#trial").click()
        expect(page.locator("#stale")).to_be_hidden(timeout=15000)
        page.locator(".display-panel summary").click()
        page.locator("#display-gamma").fill("1.8"); page.locator("#display-gamma").press("Tab")
        page.locator("#colormap").select_option("heat")
        page.locator("#scope").select_option("range")
        page.locator("#range-start").fill("0.8");page.locator("#range-end").fill("1.2")
        page.locator("#run").click()
        expect(page.locator("#result-count")).to_have_text(str(previous_results + 1), timeout=20000)
        page.locator("#result-list .source-card").last.click()
        expect(page.locator("#result-label")).to_contain_text("正式结果")
        page.locator("#seek-value").fill("0.9");page.locator("#seek").click()
        expect(page.locator("#result-label")).to_contain_text("输出第 2 帧")
        page.screenshot(path=str(output / "workbench-1440.png"))
        page.locator("#job-list [data-detail]").first.click()
        with page.expect_download() as download_info:
            page.locator('[data-download="csv"]').first.click()
        download_info.value.save_as(str(output / "browser-export.csv"))
        page.locator("#job-close").click()
        page.locator("#save-project").click()
        page.locator("#browse-path").fill(str(output));page.locator("#browse-go").click()
        expect(page.locator("#browse-path")).to_have_value(str(output))
        page.locator("#filename").fill("browser-project.json");page.locator("#dialog-submit").click()
        expect(page.locator("#file-dialog")).not_to_be_visible()
        value = json.loads((output / "browser-project.json").read_text(encoding="utf-8"))
        assert value["web_roi"] and value["steps"][0]["parameters"]["radius"] == 3
        page.locator("#display-target").select_option("input")
        page.locator("#display-gamma").fill("0.5");page.locator("#display-gamma").press("Tab")
        page.locator("#clear-roi").click()
        page.locator("#load-project").click()
        page.locator("#browse-path").fill(str(output / "browser-project.json"));page.locator("#browse-go").click()
        expect(page.locator("#selection-info")).to_contain_text("browser-project.json")
        page.locator("#dialog-submit").click()
        expect(page.locator("#file-dialog")).not_to_be_visible()
        expect(page.locator("#roi-info")).to_contain_text("px")
        expect(page.locator("#range-start")).to_have_value("0.8")
        expect(page.locator("#range-end")).to_have_value("1.2")
        expect(page.locator("#display-gamma")).to_have_value("1.8")
        expect(page.locator("#colormap")).to_have_value("heat")
        saved_time = value["web_views"][value["active_source"]]["ref"]["time"]
        expect(page.locator("#seek-value")).to_have_value(f"{saved_time:.4f}")
        page.locator("#plugin").click()
        page.locator("#browse-path").fill(str(root / "examples" / "plugins" / "invert.json"));page.locator("#browse-go").click()
        expect(page.locator("#selection-info")).to_contain_text("invert.json")
        page.locator("#dialog-submit").click()
        expect(page.locator("#file-dialog")).not_to_be_visible()
        page.locator("#add-step").click()
        expect(page.locator("#step-count")).to_have_text("2")
        page.locator("#trial").click()
        expect(page.locator("#result-label")).to_have_text("当前帧精确试算", timeout=15000)
        page.locator('[data-view="result"]').click()
        page.locator("#actual").click()
        expect(page.locator("#zoom-value")).to_have_text("100%")
        page.locator('[data-view="compare"]').click()
        page.locator("#fit").click()
        page.screenshot(path=str(output / "plugin-1440.png"))
        page.set_viewport_size({"width":1024,"height":768})
        page.screenshot(path=str(output / "workbench-1024.png"))
        box = page.locator("#run").bounding_box()
        assert box["y"] + box["height"] <= 768 and box["x"] + box["width"] <= 1024
        assert not errors, errors
        print(json.dumps({"browser_errors": errors, "screenshots": str(output), "checks": ["video seek/step", "ROI", "trial freshness", "display", "formal export", "result mapping", "CSV download", "project restore", "external plugin", "small viewport"]}, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    main()
