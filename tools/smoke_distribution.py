"""Focused browser check: histogram port, counts, frame navigation and export."""
from pathlib import Path
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


def main():
    output = ROOT / ".test-output" / f"distribution-{int(time.time())}"
    output.mkdir(parents=True)
    values = np.array([
        [[0, 1, 2, 3], [4, 5, 6, 9]],
        [[0, 0, 1, 2], [2, 3, 4, 9]],
        [[-2, -2, -2, -2], [-2, -2, -2, -2]],
    ], dtype=np.int16)
    tifffile.imwrite(output / "source.tif", values, photometric="minisblack", metadata={"axes": "TYX"})
    url = "http://127.0.0.1:8896"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, str(ROOT / "run_web.py"), "--port", "8896", "--no-browser", "--workspace", str(output)],
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
                api.post("/api/sources/open", json=dict(path=str(output / "source.tif"), interval=.01)).raise_for_status()
                browser = pw.chromium.launch(channel="msedge" if sys.platform=="win32" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1600, height=1000))
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 3")
                page.locator("#open-module-library").click()
                page.locator("[data-category-filter='all']").click()
                page.locator("[data-add-module='distribution']").click()
                for key, value in dict(bins=2, minimum=0, maximum=4).items():
                    field = page.locator(f"#general-module-fields [data-key='{key}']")
                    field.fill(str(value))
                    field.press("Tab")
                expect(page.locator("#steps .flow-type").filter(has_text="直方图")).to_contain_text("TB")
                page.locator("#stage-sample").fill("2")
                page.locator("#stage-sample").press("Tab")
                page.locator("#inspect-output").click()
                chart = page.locator('#split-curve svg[aria-label="强度—频数直方图"]')
                expect(chart).to_be_visible()
                expect(page.locator("#sequence-readout")).to_contain_text("样本 2")

                def counts():
                    return chart.locator(".histogram-bin").evaluate_all("es=>es.map(e=>Number(e.dataset.frequency))")

                assert counts()==[3, 4], counts()
                assert {"强度", "频数"}.issubset(set(chart.locator("text").all_text_contents()))
                assert api.get("/api/jobs").json()==[], "inspection must not enter the main job queue"
                page.locator("#run-current").click()
                for _ in range(150):
                    jobs = api.get("/api/jobs").json()
                    if jobs and jobs[0]["status"] in {"succeeded", "failed", "completed_with_errors", "cancelled"}:
                        break
                    page.wait_for_timeout(100)
                assert jobs[0]["status"]=="succeeded", jobs
                job = api.get("/api/jobs/" + jobs[0]["id"]).json()
                side = job["run"]["results"][0]["side_outputs"][0]
                expect(page.locator("#result-artifact")).to_have_value(side["id"])
                expect(page.locator("#result-meta")).to_contain_text("B:2")
                expect(page.locator("#sequence-readout")).to_contain_text("样本 1")
                assert counts()==[2, 3]
                for index, expected in ((1, [3, 4]), (2, [0, 0]), (0, [2, 3])):
                    page.locator("#sequence-index").fill(str(index))
                    page.locator("#sequence-index").press("Tab")
                    expect(page.locator("#sequence-readout")).to_contain_text(f"样本 {index+1}")
                    assert counts()==expected, counts()
                page.locator("#open-result-tools").click()
                expect(page.locator("#sequence-plot-axis")).to_be_hidden()
                page.locator("#artifact-export-format").select_option("svg")
                expect(page.locator("#artifact-export-format option[value='svg']")).to_have_text("首样本直方图 SVG")
                with page.expect_download() as download:
                    page.locator("#artifact-export").click()
                target = output / "histogram.svg"
                download.value.save_as(target)
                assert all(text in target.read_text(encoding="utf-8") for text in ("Intensity", "Frequency", "Histogram: sample 1"))
                page.locator("#result-tools-dialog [data-close-dialog]").click()
                original_job = job["id"]
                bound = lambda key: page.locator(f"#general-module-fields [data-histogram-bound='{key}']")
                bound("maximum").select_option("auto")
                expect(page.locator("#general-module-fields [data-key='maximum']")).to_be_disabled()
                for lower_auto, upper_auto, expected, edges in (
                    (False, True, [7, 1], [0, 4.5, 9]),
                    (True, False, [2, 5], [-2, 1, 4]),
                    (True, True, [6, 2], [-2, 3.5, 9]),
                ):
                    bound("minimum").select_option("auto" if lower_auto else "manual")
                    bound("maximum").select_option("auto" if upper_auto else "manual")
                    if not upper_auto:
                        expect(page.locator("#general-module-fields [data-key='maximum']")).to_have_value("4")
                    with page.expect_response(lambda response: response.url.endswith("/api/modules/inspect") and response.request.method=="POST") as inspected:
                        page.locator("#inspect-output").click()
                    data = inspected.value.json()
                    assert data["mappings"][side["node_id"]]["edges"]==edges, data
                    expect(chart.locator(".histogram-bin").nth(0)).to_have_attribute("data-frequency", str(expected[0]))
                    assert counts()==expected, counts()
                expect(page.locator("#general-module-fields [data-key='minimum']")).to_be_disabled()
                page.locator("#run-current").click()
                for _ in range(150):
                    jobs = [j for j in api.get("/api/jobs").json() if j["id"]!=original_job]
                    if jobs and jobs[0]["status"] in {"succeeded", "failed", "completed_with_errors", "cancelled"}:
                        break
                    page.wait_for_timeout(100)
                assert jobs[0]["status"]=="succeeded", jobs
                auto_job = api.get("/api/jobs/" + jobs[0]["id"]).json()
                auto_side = auto_job["run"]["results"][0]["side_outputs"][0]
                assert auto_side["mappings"][side["node_id"]]["edges"]==[-2, 3.5, 9]
                assert auto_side["module"]["parameters"]["minimum"] is None
                assert auto_side["module"]["parameters"]["maximum"] is None
                expect(page.locator("#sequence-readout")).to_contain_text("样本 1")
                expect(chart.locator(".histogram-bin").nth(0)).to_have_attribute("data-frequency", "4")
                for index, expected in ((1, [6, 2]), (2, [8, 0]), (0, [4, 4])):
                    page.locator("#sequence-index").fill(str(index))
                    page.locator("#sequence-index").press("Tab")
                    expect(page.locator("#sequence-readout")).to_contain_text(f"样本 {index+1}")
                    assert counts()==expected, counts()
                page.locator("#open-result-tools").click()
                page.locator("#artifact-export-format").select_option("svg")
                with page.expect_download() as auto_download:
                    page.locator("#artifact-export").click()
                auto_target = output / "auto-histogram.svg"
                auto_download.value.save_as(auto_target)
                assert all(text in auto_target.read_text(encoding="utf-8") for text in ("Intensity", "Frequency", "Histogram: sample 1"))
                page.locator("#result-tools-dialog [data-close-dialog]").click()
                page.screenshot(path=str(output / "desktop.png"), full_page=True)
                page.set_viewport_size(dict(width=1100, height=760))
                expect(chart).to_be_visible()
                page.screenshot(path=str(output / "compact.png"), full_page=True)
                assert not errors, errors
                report = dict(histogram_port=side["id"], inspection_counts=[3, 4], saved_counts=[[2, 3], [3, 4], [0, 0]],
                              axes="TB", intensity_frequency_axes=True, default_histogram_result=True,
                              frame_navigation=True, histogram_export=True, automatic_bounds=[-2, 9],
                              independent_auto_bounds=True, manual_value_restored=True, javascript_errors=errors)
                (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(dict(output=str(output), **report), ensure_ascii=False))
                browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__=="__main__":
    main()
