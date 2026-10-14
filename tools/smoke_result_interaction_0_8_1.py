"""Interaction acceptance: shared canvas, keyboard, icons and result lifecycle."""
from pathlib import Path
import io
import json
import os
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
    folder = ROOT / ".test-output" / f"result-interaction-{int(time.time())}"
    folder.mkdir(parents=True)
    values = np.arange(9 * 24 * 40, dtype=np.uint16).reshape(9, 24, 40) + 10
    tifffile.imwrite(folder / "raw.tif", values, photometric="minisblack", metadata={"axes":"TYX"})
    url = "http://127.0.0.1:8905"
    with (folder / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(ROOT / "run_web.py"), "--port", "8905", "--no-browser", "--workspace", str(folder)], cwd=ROOT, stdout=log, stderr=log, env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url + "/api/state", timeout=.5).status_code == 200: break
                except httpx.HTTPError: pass
                time.sleep(.1)
            else: raise AssertionError("isolated service failed")
            with httpx.Client(base_url=url, timeout=45) as api, sync_playwright() as pw:
                api.headers["X-EIP-Token"] = api.get("/api/state").json()["session_token"]
                response = api.post("/api/sources/open", json=dict(path=str(folder / "raw.tif"), interval=.01))
                response.raise_for_status(); source = response.json()["source"]

                def run(name, nodes, **extra):
                    response = api.post("/api/jobs", json=dict(contract_version=3, source_ids=[source["id"]], name=name, steps=nodes, **extra))
                    response.raise_for_status(); identifier = response.json()["id"]
                    for _ in range(300):
                        job = api.get(f"/api/jobs/{identifier}").json()
                        if job["status"] in {"succeeded", "failed", "completed_with_errors"}:
                            assert job["status"] == "succeeded", job
                            time.sleep(.25); return job
                        time.sleep(.04)
                    raise AssertionError("worker timeout")

                original = run("原始强度", [new_node("gain_offset")])
                gain, tap, after = [new_node(name) for name in ("gain_offset", "data_tap", "gain_offset")]
                gain["parameters"]["gain"] = 2; after["parameters"]["gain"] = 3
                tap["parameters"].update(name="中间 2 倍", destination="file")
                doubled = run("双倍过程 · 六倍输出", [gain, tap, after])
                trial = run("当前帧试算", [gain], mode="trial", sample_index=2)
                browser = pw.chromium.launch(channel="msedge" if os.name == "nt" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1600, height=1000), accept_downloads=True)
                errors, computes = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: computes.append(request.url) if request.method == "POST" and request.url.endswith("/api/jobs") else None)
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 9")
                page.locator("#manage-results").click()
                expect(page.locator("#library-results")).to_be_visible()
                assert page.locator("dialog[open]").count() == 0
                assert page.locator("#result-manager-dialog").evaluate("e=>e.tagName") == "SECTION"
                expect(page.locator("#result-list [data-result-id]")).to_have_count(3)

                row = lambda job: page.locator(f"[data-result-id='{job['id']}']")
                row(doubled).click(); expect(page.locator("#managed-name")).to_have_value(doubled["name"])
                expect(page.locator("#result-label")).to_have_text(doubled["name"])
                expect(page.locator("#sequence-index")).to_be_visible()
                page.locator("#sequence-index").press("ArrowRight"); page.locator("#sequence-index").press("ArrowRight")
                expect(page.locator("#sequence-readout")).to_contain_text("样本 3/9")
                page.locator("#result-manager-shortcut").click()
                expect(page.locator("#sequence-index")).to_have_value("2")
                assert "run_" not in page.locator("#result-manager-dialog").inner_text()
                assert not page.locator("#managed-metadata").evaluate("e=>e.open")

                page.locator("#result-search").fill("双倍")
                expect(page.locator("#result-list [data-result-id]")).to_have_count(1)
                page.locator("#result-search").fill("不存在")
                expect(page.locator("#result-manager-empty")).to_have_text("无匹配结果")
                page.locator("#result-search").press("Escape")
                expect(page.locator("#result-list [data-result-id]")).to_have_count(3)
                page.locator("[data-result-filter='trial']").click()
                expect(page.locator("#result-list [data-result-id]")).to_have_count(1)
                row(trial).click(); expect(page.locator("#managed-promote")).to_be_disabled()
                expect(page.locator("#managed-promote")).to_have_attribute("title", "试算不能作为正式输入")
                page.locator("[data-result-filter='all']").click()
                row(trial).focus(); row(trial).press("ArrowDown")
                expect(row(doubled)).to_have_attribute("aria-pressed", "true")
                expect(row(doubled)).to_be_focused()
                assert page.locator("#play").get_attribute("aria-label") == "播放"

                page.locator("#managed-edit").click()
                page.locator("#managed-name").fill("复用实验 <A>")
                page.locator("#managed-name").press("Enter")
                expect(page.locator("#managed-name")).to_have_attribute("readonly", "")
                expect(row(doubled).locator("strong")).to_have_text("复用实验 <A>")
                side = doubled["run"]["results"][0]["side_outputs"][0]
                page.locator("#managed-output").select_option(f"0|{side['id']}")
                expect(page.locator("#result-label")).to_have_text("中间 2 倍")
                page.locator("#managed-formats summary").click()
                with page.expect_download() as received: page.locator("[data-result-export='array']").click()
                received.value.save_as(str(folder / "selected.npy"))
                np.testing.assert_array_equal(np.load(folder / "selected.npy"), values * 2)
                assert not page.locator("#managed-formats").evaluate("e=>e.open")
                for button in page.locator(".result-action-bar > button,.managed-export-group > button").all():
                    assert button.get_attribute("aria-label") and button.get_attribute("title")
                    assert button.locator("svg").count() == 1 and not button.inner_text().strip()
                page.screenshot(path=str(folder / "desktop.png"))
                page.set_viewport_size(dict(width=1024, height=768))
                expect(page.locator(".processing")).not_to_be_visible()
                expect(page.locator("#result-canvas")).to_be_visible()
                page.screenshot(path=str(folder / "small.png"))
                assert page.locator("#managed-delete").evaluate("e=>e.getBoundingClientRect().bottom<innerHeight && e.getBoundingClientRect().right<innerWidth")

                row(doubled).locator("[data-result-action='promote']").click()
                expect(page.locator("#library-sources")).to_be_visible()
                expect(page.locator("#steps .flow-node")).to_have_count(0)
                derived = next(item for item in api.get("/api/state").json()["sources"] if item.get("lineage"))
                assert derived["lineage"][-1]["side_output"] == side["id"]
                page.locator("#manage-results").click(); row(doubled).click()
                page.locator("#managed-delete").click()
                expect(page.locator("#result-delete-files")).to_be_disabled()
                page.screenshot(path=str(folder / "delete-protected.png"))
                page.locator("#result-delete-confirm").click()
                expect(row(doubled)).to_have_count(0)
                assert Path(doubled["directory"]).is_dir()

                row(original).click(); page.locator("#managed-delete").click()
                page.locator("#result-delete-files").check()
                expect(page.locator("#result-delete-confirm")).to_have_text("永久删除")
                expect(page.locator("#result-delete-path")).to_be_visible()
                page.screenshot(path=str(folder / "delete-files.png"), animations="disabled")
                page.locator("#result-delete-confirm").click()
                expect(row(original)).to_have_count(0)
                assert not Path(original["directory"]).exists() and (folder / "raw.tif").is_file()

                page.locator("#result-restore-open").click(); page.locator("#result-restore-pick").click()
                expect(page.locator("#dialog-title")).to_have_text("打开历史结果")
                page.locator("#browse-path").fill(doubled["directory"]); page.locator("#browse-go").click()
                page.locator("#file-list .file-row").filter(has=page.locator("span", has_text="run.json")).click()
                page.locator("#dialog-submit").click()
                expect(page.locator("#result-restore-dialog")).not_to_be_visible()
                expect(page.locator("#managed-name")).to_have_value("复用实验 <A>")
                expect(row(doubled)).to_have_count(1)
                assert not computes, computes
                assert not errors, errors
                browser.close()
            print(json.dumps(dict(status="passed", output=str(folder), checks="shared_preview, icons, search, keyboard, rename, output_export, reuse, delete, restore"), ensure_ascii=False))
        finally:
            process.terminate()
            try: process.wait(10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(5)


if __name__ == "__main__": main()
