"""Unified telemetry browser regression, isolated from the user's live project."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time

import httpx
import numpy as np
import tifffile
from playwright.sync_api import expect, sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / ".test-output" / f"device-monitor-{int(time.time())}"
    folder.mkdir(parents=True)
    fixture = folder / "task-fixture.tif"
    tifffile.imwrite(fixture, np.arange(3*8*12, dtype=np.uint16).reshape(3, 8, 12), photometric="minisblack", metadata={"axes": "TYX"})
    url = "http://127.0.0.1:8915"
    with (folder / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, str(root / "run_web.py"), "--no-browser", "--port", "8915", "--workspace", str(folder)], cwd=root,
            stdout=log, stderr=log, env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url + "/api/state", timeout=.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise AssertionError("isolated service did not start")
            # Seed two real completed tasks so their bottom-row buttons must
            # remain clickable when the device panel is collapsed or open.
            sys.path.insert(0, str(root / "src"))
            from eip.module_registry import new_node
            with httpx.Client(base_url=url, timeout=30) as client:
                client.headers["X-EIP-Token"] = client.get("/api/state").json()["session_token"]
                opened = client.post("/api/sources/open", json=dict(path=str(fixture), interval=.01))
                opened.raise_for_status(); source = opened.json()["source"]
                for name in ("设备布局回归 A", "设备布局回归 B"):
                    response = client.post("/api/jobs", json=dict(contract_version=3, source_ids=[source["id"]], name=name, steps=[new_node("gain_offset")]))
                    response.raise_for_status(); identifier = response.json()["id"]
                    for _ in range(200):
                        job = client.get(f"/api/jobs/{identifier}").json()
                        if job["status"] in ("succeeded", "failed"):
                            assert job["status"] == "succeeded", job
                            break
                        time.sleep(.05)
                    else:
                        raise AssertionError("fixture task did not complete")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(channel="msedge" if os.name == "nt" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1600, height=1000))
                samples, errors, unexpected = [], [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: unexpected.append(request.url) if request.url.endswith("/api/resources/gpu") or request.method == "POST" and request.url.endswith("/api/jobs") else None)
                mode = dict(gpu=True, failure=False)

                def telemetry(route):
                    samples.append(time.monotonic())
                    if mode["failure"]:
                        route.fulfill(status=503, json=dict(detail="simulated unavailable")); return
                    count = len(samples)
                    now, gib, mib = 1700000000 + count * 2, 1024**3, 1024**2
                    cards = [dict(index=i, uuid=f"GPU-test-{i}", name="NVIDIA GeForce RTX 5070 Ti" if i == 0 else "Secondary CUDA GPU", driver_model="WDDM",
                        utilization_percent=30 + count % 5 * 10 + i, total_bytes=16*gib, used_bytes=(2+i)*gib, free_bytes=(14-i)*gib,
                        application_bytes=None if i == 0 else 512*mib, application_reason="Windows WDDM 未提供本程序显存" if i == 0 else None) for i in range(2)]
                    route.fulfill(json=dict(sampled_at=now, interval_seconds=2,
                        cpu=dict(system_percent=20+count % 4*10, application_percent=5+count, logical_processors=32, reason=None),
                        memory=dict(total_bytes=64*gib, used_bytes=(16+count % 3)*gib, available_bytes=46*gib, application_bytes=512*mib),
                        gpu=dict(available=mode["gpu"], cards=cards if mode["gpu"] else [], sampled_at=now, reason=None if mode["gpu"] else "GPU unavailable in test"),
                        budgets=dict(memory_mb=10240, gpu_mb=10240, processing_mb=8960, job_mb=8704, inspection_mb=256, preview_mb=1280, cache_mb=2048),
                        cache=dict(frame_cache_bytes=count*mib, frame_cache_count=count, frame_cache_limit_bytes=640*mib),
                        processes=[dict(name="工作台", pid=1, resident_bytes=400*mib), dict(name="数据读取", pid=2, resident_bytes=112*mib)],
                        activity=dict(busy=True, pending=0, job_name="示例流程", status="running")))

                page.route("**/api/resources/telemetry", telemetry)
                page.goto(url)
                expect(page.locator("#connection")).to_have_text("本机已连接")
                expect(page.locator("[data-device-state]")).to_have_text("实时")
                expect(page.locator(".device-metric")).to_have_count(4)
                expect(page.locator('[data-device-metric="vram"]')).to_contain_text("本程序 未知")
                resource_details = page.locator("#device-monitor .device-resource-details")
                resource_details.locator("summary").click()
                expect(resource_details).to_contain_text("处理预算 8960 MiB")
                expect(resource_details).to_contain_text("工作台 400.0 MiB")
                expect(resource_details).to_contain_text("数据读取 112.0 MiB")
                resource_before = page.locator("#resource-readout").inner_text()
                expect(page.locator("#resource-readout")).not_to_have_text(resource_before, timeout=5000)
                resource_details.scroll_into_view_if_needed()
                resource_details.screenshot(path=str(folder / "resource-details.png"))
                resource_details.locator("summary").click()
                expect(page.locator('[data-series="cpu-system"]')).to_have_attribute("d", re.compile(r"M.*L.*L"), timeout=7000)
                assert page.locator('[data-series="gpu-GPU-test-0-app"]').get_attribute("d") == ""
                box = lambda selector: page.locator(selector).bounding_box()

                def no_overlap():
                    panel, canvas, inspector, task = box("#device-monitor"), box("#result-canvas") or box("#input-canvas"), box(".processing"), box(".tasks")
                    assert panel and canvas and canvas["height"] > 30, (panel, canvas)
                    assert panel["x"] >= canvas["x"]+canvas["width"]-1 or panel["y"] >= canvas["y"]+canvas["height"]-1, (panel, canvas)
                    assert task and task["height"] <= 29, task
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    if inspector and page.viewport_size["width"] >= 1280:
                        assert abs(inspector["y"]+inspector["height"]-panel["y"]) <= 1, (inspector, panel)

                no_overlap()
                page.locator("#device-monitor").screenshot(path=str(folder / "desktop-panel.png"))
                page.screenshot(path=str(folder / "desktop.png"))
                page.locator("[data-device-gpu]").select_option("GPU-test-1")
                expect(page.locator('[data-device-metric="vram"]')).to_contain_text("本程序 512 MiB")
                page.locator("[data-device-pause]").click()
                expect(page.locator("[data-device-state]")).to_have_text("已暂停")
                paused_readout = page.locator("#resource-readout").inner_text()
                paused = len(samples); page.wait_for_timeout(2300)
                assert len(samples) == paused
                assert page.locator("#resource-readout").inner_text() == paused_readout
                page.locator("[data-device-pause]").click()
                expect(page.locator("[data-device-state]")).to_have_text("实时")
                expect(page.locator('[data-series="cpu-system"]')).to_have_attribute("d", re.compile(r"M.*L.*M"))
                # Exercise the visibility event handler without depending on
                # headless Chromium's foreground-tab visibility policy.
                page.evaluate('Object.defineProperty(document,"hidden",{configurable:true,get:()=>true});document.dispatchEvent(new Event("visibilitychange"))')
                expect(page.locator("[data-device-state]")).to_have_text("已挂起")
                hidden_count = len(samples); page.wait_for_timeout(2300)
                assert len(samples) == hidden_count
                page.evaluate('delete document.hidden;document.dispatchEvent(new Event("visibilitychange"))')
                expect(page.locator("[data-device-state]")).to_have_text("实时")
                before = box("#device-monitor")["height"]
                page.locator('[data-resize="devices"]').focus(); page.keyboard.press("ArrowUp")
                assert box("#device-monitor")["height"] >= before+9
                page.wait_for_timeout(500)
                page.reload(); expect(page.locator("[data-device-state]")).to_have_text("实时")
                assert box("#device-monitor")["height"] >= before+9
                page.locator("#open-run-settings").click()
                expect(page.locator("#run-settings-dialog .gpu-memory-monitor")).to_have_count(0)
                expect(page.locator("#run-settings-dialog #resource-readout")).to_have_count(0)
                expect(page.locator("#run-settings-dialog .resource-runtime")).to_have_count(1)
                expect(page.locator("#resource-gpu_mb")).to_have_value("512")
                page.locator("#resource-gpu_mb").fill("1024"); page.wait_for_timeout(2200)
                expect(page.locator("#resource-gpu_mb")).to_have_value("1024")
                page.locator("#run-settings-dialog [data-close-dialog]").click()
                page.locator("#toggle-device-monitor").click()
                expect(page.locator("#device-monitor-body")).to_be_hidden()
                collapsed = len(samples); page.wait_for_timeout(2300)
                assert len(samples) == collapsed
                page.reload(); expect(page.locator("#device-monitor-body")).to_be_hidden()
                page.locator("#toggle-device-monitor").click()
                expect(page.locator("[data-device-state]")).to_have_text("实时")
                page.set_viewport_size(dict(width=1024, height=768))
                expect(page.locator("#device-monitor")).to_have_class(re.compile("is-launcher"))
                expect(page.locator("#device-monitor")).to_be_hidden()
                expect(page.locator(".job-row")).to_have_count(2)
                page.locator("#toggle-tasks").click()

                def task_buttons_clickable():
                    for button in page.locator(".job-row [data-detail]").all():
                        button.scroll_into_view_if_needed()
                        assert button.evaluate("e=>{const r=e.getBoundingClientRect();const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);return hit===e||e.contains(hit)}"), "device area blocks a task detail button"
                        button.click()
                        expect(page.locator("#job-dialog")).to_be_visible()
                        page.locator("#job-close").click()

                task_buttons_clickable()
                page.locator("#open-device-monitor-toolbar").click()
                expect(page.locator("[data-device-state]")).to_have_text("实时")
                task_buttons_clickable()
                page.locator("#toggle-tasks").click()
                no_overlap(); page.screenshot(path=str(folder / "small-side.png"))
                page.locator("#toggle-device-monitor").click()
                page.set_viewport_size(dict(width=800, height=768))
                page.locator("#open-device-monitor-toolbar").click()
                expect(page.locator("#device-monitor")).to_have_class(re.compile("is-bottom"))
                no_overlap(); page.screenshot(path=str(folder / "small-bottom.png"))
                mode["gpu"] = False
                expect(page.locator('[data-device-metric="gpu"] .device-metric-value')).to_have_text("未知", timeout=5000)
                expect(page.locator('[data-device-metric="cpu"] .device-metric-value')).not_to_have_text("未知")
                mode["failure"] = True
                expect(page.locator("[data-device-state]")).to_have_text("更新失败", timeout=5000)
                assert "stale" in page.locator("#device-monitor").get_attribute("class")
                mode["failure"] = False; mode["gpu"] = True
                expect(page.locator("[data-device-state]")).to_have_text("实时", timeout=5000)
                page.unroute("**/api/resources/telemetry", telemetry)
                expect(page.locator('[data-device-metric="cpu"] .device-metric-value')).not_to_have_text("未知", timeout=9000)
                page.wait_for_timeout(4500)
                expect(page.locator("[data-device-state]")).to_have_text("实时", timeout=5000)
                hardware = page.locator("#device-monitor").inner_text()
                (folder / "hardware-readout.txt").write_text(hardware, encoding="utf-8")
                page.set_viewport_size(dict(width=1600, height=1000))
                page.screenshot(path=str(folder / "hardware.png"))
                assert not errors and not unexpected, (errors, unexpected)
                browser.close()
            print(json.dumps(dict(status="passed", output=str(folder), source="simulated telemetry plus actual hardware in real browser", checks="four curves; WDDM unknown; GPU selection; pause/collapse/hidden; resize persistence; draft preservation; responsive no overlap; missing GPU; stale recovery; actual hardware; no UI processing or legacy monitor calls; two real task detail buttons unobstructed")))
        finally:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(5)


if __name__ == "__main__":
    main()
