"""Inspect 0.9.0 timings and W progress in an isolated CPU-only Web run.

Creates only synthetic data below .test-output. Run explicitly after benchmarks;
the script does not attach to or reuse an existing service on its requested port.
"""
from pathlib import Path
import argparse
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import numpy as np
import tifffile
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def check_metrics(item):
    assert item["progress_axis"] == "W", item
    assert item["progress_unit"] == "窗长", item
    performance = item["performance"]
    assert performance["schema_version"] == 1
    assert performance["output_samples"] == 5
    assert performance["input_read_samples"] > 0
    assert performance["input_read_bytes"] > 0
    assert performance["elapsed_seconds"] > 0
    assert performance["output_per_second"] > 0
    assert performance["input_reads_per_second"] > 0
    assert performance["gpu_execution_seconds"] is None
    assert abs(sum(performance["host_seconds"].values())-performance["elapsed_seconds"]) < 1e-6
    assert "not unique" in performance["semantics"]["input_read_samples"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8911)
    args = parser.parse_args()
    # Check before launching: never inspect another task's service by accident.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise SystemExit(f"Port {args.port} is already in use; choose --port with an unused port.") from error
    folder = ROOT/".test-output"/f"acceleration-0.9.0-{int(time.time())}"
    folder.mkdir(parents=True)
    t = np.arange(31)[:, None, None]
    grid = np.arange(144).reshape(12, 12)
    values = np.concatenate((20+grid+3*t, 50+grid+2*t+t*t), axis=2).astype(np.uint16)
    tifffile.imwrite(folder/"source.tif", values, photometric="minisblack", metadata={"axes": "TYX"})
    url = f"http://127.0.0.1:{args.port}"
    with (folder/"server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, str(ROOT/"run_web.py"), "--port", str(args.port), "--no-browser", "--workspace", str(folder)],
            cwd=ROOT, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise AssertionError(f"isolated service exited; see {folder/'server.log'}")
                try:
                    if httpx.get(url+"/api/state", timeout=.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise AssertionError("isolated service did not start")
            with httpx.Client(base_url=url, timeout=45) as api, sync_playwright() as pw:
                state = api.get("/api/state").json()
                api.headers["X-EIP-Token"] = state["session_token"]
                resources = api.get("/api/resources").json()["settings"]
                resources["backend"] = "cpu"
                response = api.post("/api/resources", json=resources)
                response.raise_for_status()
                response = api.post("/api/sources/open", json=dict(path=str(folder/"source.tif"), interval=.01))
                response.raise_for_status()
                browser = pw.chromium.launch(channel="msedge" if os.name == "nt" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1600, height=1000), accept_downloads=True)
                errors, computes = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: computes.append(request.url)
                    if request.method == "POST" and request.url.endswith("/api/jobs") else None)
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("31")
                page.locator("#flow-clear").click()

                def add(algorithm):
                    page.locator("#open-module-library").click()
                    page.locator("[data-category-filter=all]").click()
                    page.locator(f"[data-add-module='{algorithm}']").click()

                def parameter(key, value):
                    control = page.locator(f"#general-module-fields [data-key='{key}']")
                    control.fill(str(value))
                    control.press("Tab")

                add("alignment")
                add("point_extract")
                parameter("x", 2)
                parameter("y", 3)
                add("correlation_window")
                parameter("lag", -2)
                parameter("center", 16)
                parameter("window_list", "2,3,6,11,21")
                page.locator("#job-name").fill("0.9.0 加速与耗时检查")
                with page.expect_response(lambda response: response.request.method == "POST" and response.url.endswith("/api/jobs")) as received:
                    page.locator("#run").click()
                assert received.value.status == 200, received.value.text()
                identifier = received.value.json()["id"]
                for _ in range(300):
                    job = api.get(f"/api/jobs/{identifier}").json()
                    if job["status"] in {"succeeded", "completed_with_errors", "failed", "cancelled"}:
                        break
                    time.sleep(.05)
                assert job["status"] == "succeeded", job
                assert job["progress_unit"] == "窗长"
                item = job["run"]["results"][0]
                check_metrics(item)
                assert all(device["actual"] == "cpu" for device in item["modules"]["devices"].values())
                manifest = Path(job["directory"])/"run.json"
                assert manifest.resolve().is_relative_to(folder.resolve())
                check_metrics(json.loads(manifest.read_text(encoding="utf-8"))["results"][0])
                response = api.get(f"/api/jobs/{identifier}/download?file=manifest")
                response.raise_for_status()
                check_metrics(response.json()["results"][0])

                # The task strip remains a small part of the workbench. Expand
                # only its existing compact record area to inspect the unit.
                assert page.locator(".tasks").bounding_box()["height"] <= 45
                page.locator("#toggle-tasks").click()
                row = page.locator(".job-row").filter(has=page.locator(f"[data-detail='{identifier}']"))
                expect(row.locator(".job-progress")).to_contain_text("窗长")
                assert "帧" not in row.locator(".job-progress").inner_text()
                assert page.locator(".tasks").bounding_box()["height"] <= 161
                page.screenshot(path=str(folder/"desktop-workbench.png"), animations="disabled")
                row.locator("[data-detail]").click()
                dialog = page.locator("#job-dialog")
                expect(dialog).to_be_visible()
                timings = dialog.locator("details").filter(has=page.locator("summary", has_text="运行耗时与吞吐"))
                expect(timings).to_have_count(1)
                expect(timings).to_have_js_property("open",False)
                timings.locator("summary").click()
                expect(timings).to_contain_text("输出吞吐")
                expect(timings).to_contain_text("输入读取吞吐")
                expect(timings).to_contain_text("窗长/秒")
                expect(timings).to_contain_text("样本读取/秒")
                expect(timings).to_contain_text("GPU 内核耗时")
                expect(timings.locator("dd").filter(has_text="未采集")).to_have_count(1)
                timings.scroll_into_view_if_needed()
                page.screenshot(path=str(folder/"desktop-details.png"), animations="disabled")
                with page.expect_download() as downloaded:
                    dialog.locator("[data-download=manifest]").click()
                downloaded.value.save_as(str(folder/"downloaded-run.json"))
                check_metrics(json.loads((folder/"downloaded-run.json").read_text(encoding="utf-8"))["results"][0])
                page.locator("#job-close").click()
                page.set_viewport_size(dict(width=1024, height=768))
                page.wait_for_timeout(150)
                expect(row.locator(".job-progress")).to_contain_text("窗长")
                assert page.locator(".tasks").bounding_box()["height"] <= 161
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(folder/"small-workbench.png"), animations="disabled")
                row.locator("[data-detail]").click()
                timings = dialog.locator("details").filter(has=page.locator("summary", has_text="运行耗时与吞吐"))
                # showModal precedes the asynchronous fresh detail render.
                expect(timings).to_have_js_property("open",False)
                timings.locator("summary").click()
                timings.scroll_into_view_if_needed()
                box = dialog.bounding_box()
                assert box["x"] >= 0 and box["x"]+box["width"] <= 1025
                assert dialog.evaluate("element => element.scrollWidth <= element.clientWidth+1")
                page.screenshot(path=str(folder/"small-details.png"), animations="disabled")
                assert not errors, errors
                assert len(computes) == 1, computes
                jobs = api.get("/api/jobs").json()
                assert len(jobs) == 1 and jobs[0]["id"] == identifier, jobs
                (folder/"checks.json").write_text(json.dumps(dict(status="passed",job_id=identifier,requests=len(computes),javascript_errors=errors,performance=item["performance"]),ensure_ascii=False,indent=2),encoding="utf-8")
                browser.close()
            print(json.dumps(dict(status="passed", output=str(folder), checks="one formal CPU W scan, compact tasks, W units, collapsed timing detail, input/output throughput, unknown GPU kernel timing, persisted/downloaded performance, responsive layouts, no hidden jobs")))
        finally:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(5)


if __name__ == "__main__":
    main()
