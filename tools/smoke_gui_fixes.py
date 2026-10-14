"""Browser regression for axes, prefix execution, results, logs and busy tools."""
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
    output = ROOT / ".test-output" / f"gui-fixes-{int(time.time())}"
    output.mkdir(parents=True)
    pixels = np.arange(60*32*64, dtype=np.uint16).reshape(60, 32, 64)
    tifffile.imwrite(output/"sample.tif", pixels, photometric="minisblack", metadata={"axes": "TYX"})
    (output/"slow.py").write_text(
        "import time\nclass Slow:\n"
        " def prepare(self, parameters, context): context.log('GUI-regression-log-marker')\n"
        " def process(self, pixels, context):\n  time.sleep(.3)\n  context.check_cancelled()\n  return pixels.copy()\n"
        " def close(self): pass\n", encoding="utf-8")
    (output/"slow.json").write_text(json.dumps(dict(schema_version=1, id="slow", name="Slow", entry_point="slow.py:Slow")), encoding="utf-8")
    url = "http://127.0.0.1:8895"
    with (output/"server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, str(ROOT/"run_web.py"), "--port", "8895", "--no-browser", "--workspace", str(output)], cwd=ROOT, stdout=log, stderr=log, env=dict(os.environ, PYTHONIOENCODING="utf-8"), creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+"/api/state", timeout=.5).status_code==200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise AssertionError("isolated server did not start")
            with httpx.Client(base_url=url, timeout=30) as api, sync_playwright() as pw:
                api.headers["X-EIP-Token"] = api.get("/api/state").json()["session_token"]
                source = api.post("/api/sources/open", json=dict(path=str(output/"sample.tif"), interval=.01)).json()["source"]
                browser = pw.chromium.launch(channel="msedge" if sys.platform=="win32" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1600, height=1000))
                errors, inspections = [], []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("request", lambda r: inspections.append(r.url) if r.method=="POST" and r.url.endswith("/api/modules/inspect") else None)
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 60")

                def add(identifier):
                    page.locator("#open-module-library").click()
                    page.locator("#module-search").fill("")
                    page.locator("[data-category-filter='all']").click()
                    page.locator(f"[data-add-module='{identifier}']").click()

                def parameter(key, value):
                    field = page.locator(f"#general-module-fields [data-key='{key}']")
                    if field.evaluate("e=>e.tagName")=="SELECT":
                        field.select_option(str(value))
                    else:
                        field.fill(str(value));field.press("Tab")

                def template():
                    page.locator("#analysis-mode").select_option("general")
                    page.locator("#apply-template").click()
                    page.locator("#confirm-template").click()

                def completed(before):
                    for _ in range(150):
                        jobs = api.get("/api/jobs").json()
                        fresh = [j for j in jobs if j["id"] not in before and j["status"] in {"succeeded", "failed", "cancelled", "completed_with_errors"}]
                        if fresh:
                            job = api.get("/api/jobs/"+fresh[0]["id"]).json()
                            assert job["status"]=="succeeded", job
                            return job
                        page.wait_for_timeout(100)
                    raise AssertionError("prefix job did not finish")

                def has_pixels():
                    for _ in range(50):
                        painted = page.locator("#result-canvas").evaluate("c=>{if(!c.width||!c.height)return false;const a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;const colors=new Set();for(let i=3;i<a.length;i+=4)if(a[i]){n++;colors.add(a[i-3]);}return n>1000&&colors.size>16;}")
                        if painted:
                            return True
                        page.wait_for_timeout(30)
                    return False

                def coordinate(x, y, width, height):
                    box = page.locator("#input-canvas").bounding_box()
                    scale = min((box["width"]-40)/width, (box["height"]-50)/height)
                    return dict(x=box["x"]+box["width"]/2+(x-width/2)*scale, y=box["y"]+box["height"]/2+(y-height/2)*scale)

                parameter("first", 6);parameter("last", 15)
                expect(page.locator("#steps [data-algorithm='time_crop'] .flow-type")).to_have_text("TYX")
                assert "输入：T:60" in page.locator("#steps .flow-type").first.get_attribute("title")
                add("affine");parameter("gain", 2)
                add("point_extract");parameter("output", "main")
                expect(page.locator("#steps [data-algorithm='point_extract'] .flow-type")).to_have_text("T")
                add("gaussian")
                expect(page.locator("#module-status")).to_have_class("connection-status error")
                page.locator("#steps [data-algorithm='affine']").click()
                before = {j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#run-current").click()
                saved = completed(before)
                assert [n["algorithm"] for n in saved["run"]["spec"]["steps"]]==["time_crop", "affine"]
                assert saved["run"]["spec"]["execution_scope"]["kind"]=="prefix"
                actual = np.load(Path(saved["directory"])/"source_000/values.npy", allow_pickle=False)
                np.testing.assert_array_equal(actual, pixels[5:15].astype(np.float64)*2)

                page.locator("[data-library-tab='sources']").click()
                page.locator("#manage-results").click()
                page.locator(f"[data-sequence-result='{saved['id']}']").click()
                expect(page.locator("#sequence-readout")).to_contain_text("来源帧 6")
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 60")
                assert has_pixels()
                page.locator("#result-pane [data-window-action='hide']").click()
                expect(page.locator("#result-pane")).to_be_hidden()
                page.locator("[data-view='compare']").click()
                expect(page.locator("#result-pane")).to_be_visible()
                assert has_pixels()
                page.locator("#input-pane [data-window-action='hide']").click()
                page.locator("[data-view='compare']").click()
                expect(page.locator("#input-pane")).to_be_visible()
                page.locator("#result-pane [data-window-action='float']").click()
                page.locator("#result-pane [data-window-action='hide']").click()
                expect(page.locator("#result-pane")).to_be_hidden()
                page.locator("[data-view='compare']").click()
                expect(page.locator("#result-pane")).to_be_visible()
                page.locator("#result-pane [data-window-action='dock']").click()
                page.locator("[data-view='result']").click()
                page.locator("#sequence-index").focus();page.locator("#sequence-index").press("ArrowRight")
                expect(page.locator("#sequence-readout")).to_contain_text("样本 2/10")
                expect(page.locator("#viewers")).to_have_attribute("data-mode", "result")
                assert has_pixels()
                page.locator("#result-pane [data-window-action='hide']").click()
                page.reload()
                expect(page.locator("#sequence-readout")).to_contain_text("样本 2/10")
                expect(page.locator("#result-pane")).to_be_visible()
                assert has_pixels()
                page.locator("[data-view='compare']").click()
                page.screenshot(path=str(output/"results-visible.png"))

                template()
                spec = api.post("/api/plugins/register", json=dict(path=str(output/"slow.json"))).json()
                slow = dict(id="slow", algorithm="slow", parameters={}, enabled=True, version=spec["version"], manifest=spec["manifest"], code_hash=spec["code_hash"])
                slow_id = api.post("/api/jobs", json=dict(contract_version=3, mode="run", source_ids=[source["id"]], steps=[slow])).json()["id"]
                all_jobs = {j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#tool-roi").click()
                expect(page.locator("#axis-stage-caption")).to_contain_text("ROI", timeout=15000)
                start, end = coordinate(5.2, 4.2, 64, 32), coordinate(25.2, 18.2, 64, 32)
                page.mouse.move(**start);page.mouse.down();page.mouse.move(**end, steps=6);page.mouse.up()
                expect(page.locator("#general-module-fields [data-key='x']")).to_have_value("5")
                width = int(page.locator("#general-module-fields [data-key='width']").input_value())
                height = int(page.locator("#general-module-fields [data-key='height']").input_value())
                page.locator("#tool-point").click()
                expect(page.locator("#axis-stage-caption")).to_contain_text("点", timeout=15000)
                page.mouse.click(**coordinate(3.2, 4.2, width, height))
                expect(page.locator("#general-module-fields [data-key='x']")).to_have_value("3")
                expect(page.locator("#general-module-fields [data-key='y']")).to_have_value("4")
                parameter("output", "side")
                expect(page.locator("#steps [data-algorithm='point_extract'] .flow-type")).to_have_text("TYX · 旁路 T")
                assert {j["id"] for j in api.get("/api/jobs").json()}==all_jobs
                assert api.get(f"/api/jobs/{slow_id}").json()["status"]=="running"
                page.screenshot(path=str(output/"point-while-processing.png"))
                parameter("output", "main")
                page.locator("#inspect-output").click()
                expect(page.locator("#result-label")).to_contain_text("输出检查")
                expect(page.locator("#split-curve svg")).to_be_visible()
                page.locator("#seek-value").fill("2");page.locator("#seek").click()
                expect(page.locator("#frame-info")).to_contain_text("第 2 帧 / 60")
                assert api.get(f"/api/jobs/{slow_id}").json()["status"]=="running"

                page.locator("#toggle-tasks").click()
                page.locator(f"[data-detail='{slow_id}']").click()
                log_details = page.locator("#job-detail details").filter(has=page.locator("summary", has_text="任务日志"))
                expect(log_details).to_have_count(1)
                expect(page.locator("#job-detail pre", has_text="GUI-regression-log-marker")).to_have_count(1)
                log_details.locator("summary").click()
                expect(log_details.locator("pre")).to_be_hidden()
                page.screenshot(path=str(output/"collapsible-log.png"))
                page.locator("#job-close").click()
                api.post(f"/api/jobs/{slow_id}/cancel", json={})
                for _ in range(100):
                    if api.get(f"/api/jobs/{slow_id}").json()["status"]=="cancelled":
                        break
                    page.wait_for_timeout(100)
                else:
                    raise AssertionError("cancelled worker did not exit")
                template();add("axis_slice");parameter("output", "main")
                expect(page.locator("#steps [data-algorithm='axis_slice'] .flow-type")).to_have_text("YX")
                assert inspections and not errors, errors
                report = dict(version=api.get("/api/state").json()["version"], full_axes=True,
                              prefix_run=saved["id"], invalid_tail_skipped=True,
                              actual_canvas_pixels=True, hidden_and_floating_results_restored=True,
                              result_navigation_preserves_view=True, result_restored_after_reload=True,
                              busy_job_tools=True, toolbar_point=True, local_coordinates=True,
                              checks_never_enqueued=True, single_collapsible_log=True,
                              inspection_requests=len(inspections), javascript_errors=errors)
                (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
