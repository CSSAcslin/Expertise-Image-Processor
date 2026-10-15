"""Isolated browser/export checks limited to the 0.7.4 module revision."""
from pathlib import Path
import csv
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
sys.path.insert(0,str(ROOT/"src"))
from eip import __version__


def main():
    output = ROOT / ".test-output" / f"module-revision-{int(time.time())}"
    output.mkdir(parents=True)
    pixels = np.repeat(np.arange(20, 110, 10, dtype=np.uint16)[:, None, None], 4, axis=1)
    pixels = np.repeat(pixels, 8, axis=2)
    tifffile.imwrite(output / "source.tif", pixels, photometric="minisblack", metadata={"axes": "TYX"})
    port = 8899
    url = f"http://127.0.0.1:{port}"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, str(ROOT / "run_web.py"), "--port", str(port), "--no-browser", "--workspace", str(output)],
            cwd=ROOT, stdout=log, stderr=log, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        try:
            for _ in range(100):
                try:
                    if httpx.get(url + "/api/state", timeout=.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise AssertionError("isolated server did not start")
            with httpx.Client(base_url=url, timeout=45) as api, sync_playwright() as pw:
                state = api.get("/api/state").json()
                assert state["version"] == __version__
                api.headers["X-EIP-Token"] = state["session_token"]
                response = api.post("/api/sources/open", json=dict(path=str(output / "source.tif"), interval=.01))
                response.raise_for_status()
                source = response.json()["source"]
                browser = pw.chromium.launch(channel="msedge" if sys.platform == "win32" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1600, height=1000))
                errors, computations = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda r: computations.append(r.url) if r.method == "POST" and r.url.endswith("/api/jobs") else None)
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 9")
                page.locator("#flow-clear").click()
                page.locator("#open-module-library").click()
                for old in ("box_blur", "gaussian", "intensity", "affine", "region_reduce", "sequence_statistics"):
                    assert page.locator(f"[data-add-module='{old}']").count() == 0
                for category, identifiers in {
                    "modifier": ("spatial_smooth", "gain_offset", "curve", "fixed_normalize", "time_statistics", "global_statistics"),
                    "intermediate": ("parameter_binding", "point_extract", "line_extract", "axis_slice"),
                }.items():
                    page.locator(f"[data-category-filter='{category}']").click()
                    for identifier in identifiers:
                        expect(page.locator(f"[data-add-module='{identifier}']")).to_be_visible()

                def add(identifier):
                    page.locator("#open-module-library").click()
                    page.locator("[data-category-filter='all']").click()
                    page.locator(f"[data-add-module='{identifier}']").click()
                    expect(page.locator(f"#steps [data-algorithm='{identifier}']").last).to_be_visible()

                def parameter(key, value):
                    field = page.locator(f"#general-module-fields [data-key='{key}']")
                    expect(field).to_be_visible()
                    if field.evaluate("e=>e.tagName") == "SELECT":
                        field.select_option(str(value))
                    else:
                        field.fill(str(value))
                        field.press("Tab")

                add("gain_offset")
                parameter("gain", .5)
                parameter("offset", .25)
                page.locator(".module-execution summary").click()
                page.locator("[data-execution='precision']").select_option("float64")
                add("spatial_smooth")
                expect(page.locator("[data-key='sigma']")).to_have_count(0)
                parameter("method", "gaussian")
                expect(page.locator("[data-key='sigma']")).to_be_visible()
                parameter("method", "mean")
                expect(page.locator("[data-key='sigma']")).to_have_count(0)
                parameter("method", "gaussian")
                add("time_statistics")
                choices = page.locator("[data-key='mode'] option").evaluate_all("options=>options.map(o=>o.value)")
                assert {"min", "max", "variance", "quantile"}.issubset(choices)
                parameter("mode", "max")
                expect(page.locator("[data-key='quantile']")).to_have_count(0)
                add("global_statistics")
                parameter("mode", "quantile")
                parameter("quantile", .25)
                page.locator("[data-inspector-tab='io']").click()
                expect(page.locator(".module-contract")).to_contain_text("N:1")
                expect(page.locator("#module-status")).to_have_text("连接有效")
                assert not computations
                page.locator("#run").click()
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    jobs = api.get("/api/jobs").json()
                    if jobs and jobs[0]["status"] in {"succeeded", "failed", "completed_with_errors"}:
                        break
                    page.wait_for_timeout(100)
                assert jobs[0]["status"] == "succeeded", jobs
                saved = api.get("/api/jobs/" + jobs[0]["id"]).json()
                item = saved["run"]["results"][0]
                assert saved["run"]["app_version"] == __version__
                assert item["contract"]["full_axes"] == "N" and item["contract"]["length"] == 1
                numeric = api.get(f"/api/jobs/{saved['id']}/download", params=dict(file="array"))
                numeric.raise_for_status()
                np.testing.assert_allclose(np.load(io.BytesIO(numeric.content)), [20.25])
                table = api.get(f"/api/jobs/{saved['id']}/download", params=dict(file="csv"))
                table.raise_for_status()
                rows = list(csv.DictReader(io.StringIO(table.content.decode("utf-8-sig"))))
                assert len(rows) == 1 and abs(float(rows[0]["value_1"]) - 20.25) < 1e-10
                assert rows[0]["time_seconds"] == ""
                assert [n["algorithm"] for n in saved["run"]["spec"]["steps"]] == ["gain_offset", "spatial_smooth", "time_statistics", "global_statistics"]
                page.locator("#open-data-library").click()
                if not page.locator("#library-sources").is_visible(): page.locator("#toggle-data-sources").click()
                page.locator("#manage-results").click()
                page.locator(f"[data-sequence-result='{saved['id']}']").click()
                expect(page.locator("#split-curve svg")).to_be_visible()
                page.screenshot(path=str(output / "statistics-result.png"), full_page=True)

                page.locator("#flow-clear").click()
                expect(page.locator("#steps [data-instance]")).to_have_count(0)
                expect(page.locator("#flow-clear")).to_be_disabled()
                page.locator("#flow-undo").click()
                expect(page.locator("#steps [data-instance]")).to_have_count(4)
                page.locator("#flow-redo").click()
                expect(page.locator("#steps [data-instance]")).to_have_count(0)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 9")
                page.wait_for_timeout(450)
                page.reload()
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 9")
                expect(page.locator("#steps [data-instance]")).to_have_count(0)
                for template, algorithm in (("split_correlation", "correlation_map"), ("split_curve", "correlation_curve")):
                    page.locator("#analysis-mode").select_option(template)
                    page.locator("#apply-template").click()
                    if template == "split_curve":
                        page.locator("#confirm-template").click()
                    expect(page.locator("#steps [data-instance]")).to_have_count(5)
                    assert page.locator("#steps [data-algorithm='sequence_statistics']").count() == 0
                    assert page.locator(f"#steps [data-algorithm='{algorithm}']").count() == 1
                    expect(page.locator("#module-status")).to_have_text("连接有效")
                page.locator("#flow-clear").click()
                add("global_statistics")
                parameter("mode", "max")
                parameter("output", "side")
                add("parameter_binding")
                page.locator("[data-bind='value']").click()
                page.locator("[data-bind-port]").select_option("value")
                page.locator("[data-bind-apply]").click()
                expect(page.locator(".bound-value")).to_contain_text("全域统计")
                expect(page.locator("#module-status")).to_have_text("连接有效")
                hide_inspector = page.locator("[data-window='inspector'] [data-window-action='hide']")
                if hide_inspector.is_visible():
                    hide_inspector.click()
                page.set_viewport_size(dict(width=1100, height=760))
                page.wait_for_timeout(180)
                expect(page.locator("#flow-clear")).to_be_visible()
                assert page.locator("#flow-clear").bounding_box()["x"] >= page.locator("#add-step").bounding_box()["x"]
                page.screenshot(path=str(output / "modules-compact.png"), full_page=True)
                assert len(computations) == 1 and len(api.get("/api/jobs").json()) == 1
                assert not errors, errors
                report = dict(version=__version__, categories=True, merged_selectors=True, precision=True,
                              statistics_export=20.25, output_axes="N", clear_undo_redo=True,
                              clear_persisted=True, templates_without_statistics=True, statistic_binding=True,
                              javascript_errors=errors)
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


if __name__ == "__main__":
    main()
