"""Inspect GPU memory controls with simulated driver samples, without CUDA."""
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time

import httpx
from playwright.sync_api import expect, sync_playwright


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/".test-output"/f"gpu-memory-{int(time.time())}"
    folder.mkdir(parents=True)
    url = "http://127.0.0.1:8913"
    with (folder/"server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, str(root/"run_web.py"), "--no-browser", "--port", "8913", "--workspace", str(folder)],
            cwd=root, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state", timeout=.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise AssertionError("isolated service did not start")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(channel="msedge" if os.name == "nt" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1440, height=1000))
                samples, errors, jobs = [], [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: jobs.append(request.url) if request.method == "POST" and request.url.endswith("/api/jobs") else None)
                def memory(route):
                    samples.append(time.monotonic())
                    used = [1536, 4096, 3072][(len(samples)-1) % 3]
                    # Accelerated synthetic timestamps make the two-minute
                    # history legible without waiting for a two-minute test.
                    data = dict(available=True, sampled_at=1700000000+len(samples)*40, interval_seconds=2, reason=None, budget_mb=512,
                        cards=[dict(index=0, uuid="GPU-test", name="NVIDIA GeForce RTX 5070 Ti", driver_model="WDDM", total_bytes=16303*1048576,
                                    used_bytes=used*1048576, free_bytes=(16303-used)*1048576, application_bytes=None, application_reason="Windows 驱动未提供进程显存")])
                    route.fulfill(json=data)
                page.route("**/api/resources/gpu", memory)
                page.goto(url)
                expect(page.locator("#connection")).to_have_text("本机已连接")
                assert not samples
                page.locator("#open-run-settings").click()
                expect(page.locator("[data-gpu-state]")).to_have_text("实时")
                expect(page.locator(".gpu-memory-name")).to_contain_text("RTX 5070 Ti")
                expect(page.locator(".gpu-memory-metrics")).to_contain_text("本程序 未知")
                page.locator("#resource-gpu_mb").fill("1024")
                expect(page.locator(".gpu-memory-line")).to_have_attribute("d", re.compile(r"M.*L.*L"), timeout=8000)
                expect(page.locator("#resource-gpu_mb")).to_have_value("1024")
                page.locator("[data-gpu-pause]").click()
                expect(page.locator("[data-gpu-state]")).to_have_text("已暂停")
                paused_count = len(samples)
                page.wait_for_timeout(2300)
                assert len(samples) == paused_count
                page.locator(".gpu-memory-monitor").scroll_into_view_if_needed()
                page.locator("#run-settings-dialog").screenshot(path=str(folder/"desktop.png"))
                page.set_viewport_size(dict(width=1024, height=768))
                page.locator(".gpu-memory-monitor").scroll_into_view_if_needed()
                page.locator("#run-settings-dialog").screenshot(path=str(folder/"small.png"))
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.locator("[data-gpu-pause]").click()
                expect(page.locator("[data-gpu-state]")).to_have_text("实时")
                expect(page.locator(".gpu-memory-line")).to_have_attribute("d", re.compile(r"M.*L.*M"))
                assert len(samples) > paused_count
                page.locator("#run-settings-dialog [data-close-dialog]").click()
                closed_count = len(samples)
                page.wait_for_timeout(2300)
                assert len(samples) == closed_count
                assert not errors and not jobs, (errors, jobs)
                browser.close()
            print(json.dumps(dict(status="passed", output=str(folder), source="simulated driver samples", checks="live history, WDDM unknown, unchanged draft, pause/resume, stop on close, responsive layout, no jobs")))
        finally:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(5)


if __name__ == "__main__":
    main()
