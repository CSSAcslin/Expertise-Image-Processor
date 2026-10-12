"""Exercise the module workbench in an isolated service and real browser.

Writes only to .test-output; it never loads the user's live project or data.
Run with the development environment (requires Playwright and local Edge).
"""
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
PORT = 8894


def main():
    output = ROOT / ".test-output" / f"workbench-{int(time.time())}"
    output.mkdir(parents=True)
    yy, xx = np.mgrid[:32, :64]
    pixels = np.stack([(yy * 4 + xx + i * 3 + 20).astype(np.uint16) for i in range(25)])
    tifffile.imwrite(output / "sample.tif", pixels, photometric="minisblack", metadata={"axes": "TYX"})
    url = f"http://127.0.0.1:{PORT}"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, str(ROOT / "run_web.py"), "--port", str(PORT), "--no-browser", "--workspace", str(output)],
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
                raise AssertionError("isolated workbench did not start")
            with httpx.Client(base_url=url, timeout=45) as api, sync_playwright() as pw:
                api.headers["X-EIP-Token"] = api.get("/api/state").json()["session_token"]
                response = api.post("/api/sources/open", json=dict(path=str(output / "sample.tif"), interval=.01))
                response.raise_for_status()
                source = response.json()["source"]
                browser = pw.chromium.launch(channel="msedge" if sys.platform == "win32" else "chromium", headless=True)
                context = browser.new_context(viewport=dict(width=1600, height=1000))
                page = context.new_page()
                errors, geometry, computations = [], [], []
                page.on("pageerror", lambda error: errors.append(str(error)))

                def requested(request):
                    if request.method == "POST" and request.url.endswith("/api/jobs"):
                        computations.append(request.url)
                    if request.method == "POST" and request.url.endswith("/api/split/geometry"):
                        geometry.append(request.url)

                page.on("request", requested)
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 25")
                assert page.locator("#input-canvas").count() == 1
                assert page.locator("#steps").count() == 1
                page.locator("#open-module-library").click()

                def add(identifier):
                    page.locator("#open-module-library").click()
                    page.locator("#module-search").fill("")
                    page.locator("[data-category-filter='all']").click()
                    button = page.locator(f"[data-add-module='{identifier}']")
                    expect(button).to_be_visible()
                    button.click()
                    expect(page.locator(f"#steps [data-algorithm='{identifier}']").last).to_be_visible()

                def parameter(key, value):
                    field = page.locator(f"#general-module-fields [data-key='{key}']")
                    expect(field).to_be_visible()
                    if field.evaluate("e=>e.tagName") == "SELECT":
                        field.select_option(str(value))
                    else:
                        field.fill(str(value))
                        field.press("Tab")

                def completed(before):
                    deadline = time.monotonic() + 45
                    while time.monotonic() < deadline:
                        fresh = [j for j in api.get("/api/jobs").json() if j["id"] not in before]
                        done = [j for j in fresh if j["status"] in {"succeeded", "completed_with_errors", "failed", "cancelled"}]
                        if done:
                            value = api.get("/api/jobs/" + done[0]["id"]).json()
                            assert value["status"] == "succeeded", value
                            return value
                        page.wait_for_timeout(100)
                    raise AssertionError("workbench job timed out")

                # Classification and search are a library concern, never a new mode.
                page.locator("[data-category-filter='modifier']").click()
                expect(page.locator("[data-add-module='data_tap']")).to_be_visible()
                expect(page.locator("[data-add-module='roi_crop']")).to_be_hidden()
                page.locator("[data-category-filter='all']").click()
                page.locator("#module-search").fill("增益")
                expect(page.locator("[data-add-module='affine']")).to_be_visible()
                expect(page.locator("[data-add-module='data_tap']")).to_be_hidden()
                add("roi_crop")
                for key, value in (("x", 4), ("y", 3), ("width", 20), ("height", 12)):
                    parameter(key, value)
                assert page.locator("#general-module-fields [data-key='mode']").count() == 0
                add("affine")
                parameter("gain", 2)
                page.locator("[data-inspector-tab='formula']").click()
                expect(page.locator("#general-module-inspector")).to_contain_text("公式")
                expect(page.locator("#general-module-fields math")).to_be_visible()
                page.locator("[data-inspector-tab='io']").click()
                page.locator("[data-inspector-tab='parameters']").click()
                add("data_tap")
                parameter("name", "归一前截流")
                parameter("destination", "both")
                add("affine")
                parameter("gain", 3)

                count = page.locator("#steps [data-instance]").count()
                selected = page.locator("#steps [data-algorithm='affine']").last
                selected.locator("[data-node-menu]").click()
                selected.locator("[data-node-action='duplicate']").click()
                expect(page.locator("#steps [data-instance]")).to_have_count(count + 1)
                page.locator("#flow-undo").click()
                expect(page.locator("#steps [data-instance]")).to_have_count(count)
                page.locator("#flow-redo").click()
                expect(page.locator("#steps [data-instance]")).to_have_count(count + 1)
                page.locator("#flow-undo").click()

                boxes = page.locator("#steps [data-instance]").evaluate_all("els=>els.map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y}})")
                assert max(b["y"] for b in boxes)-min(b["y"] for b in boxes)<5, boxes
                assert all(a["x"]<b["x"] for a,b in zip(boxes,boxes[1:])), boxes

                # Selection, formula viewing, layout and ordinary playback must not run algorithms.
                assert not computations and not geometry, (computations, geometry)
                page.locator("#seek-value").fill("5")
                page.locator("#seek").click()
                expect(page.locator("#frame-info")).to_contain_text("第 5 帧 / 25")
                page.locator("#play").click()
                page.wait_for_timeout(350)
                page.locator("#play").click()
                assert not computations and not geometry, (computations, geometry)

                before = {j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#trial").click()
                trial = completed(before)
                before = {j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#run").click()
                saved = completed(before)
                page.locator("[data-library-tab='sources']").click()
                page.locator(f"[data-sequence-result='{saved['id']}']").click()
                page.locator("#open-result-tools").click()
                expect(page.locator("#sequence-controls")).to_be_visible()
                result = saved["run"]["results"][0]
                assert result["contract"]["shape"] == [12, 20]
                assert result["contract"]["length"] == 25
                numeric = api.get(f"/api/jobs/{saved['id']}/download", params=dict(file="array"))
                numeric.raise_for_status()
                np.testing.assert_array_equal(np.load(io.BytesIO(numeric.content), allow_pickle=False), pixels[:, 3:15, 4:24] * 6.)
                # Saved side-output behavior is checked by the API regression tests;
                # this browser test verifies the actual workflow can create it.
                side = result.get("side_outputs", [])
                assert side, "data tap did not produce a captured result"
                capture = next(s for s in side if s.get("kind") == "data_tap")
                assert capture["status"] == "succeeded" and not capture["temporary"]
                captured = api.get(f"/api/jobs/{saved['id']}/download", params=dict(file="array", side_output=capture["id"]))
                captured.raise_for_status()
                np.testing.assert_array_equal(np.load(io.BytesIO(captured.content), allow_pickle=False), pixels[:, 3:15, 4:24] * 2.)

                page.locator("#result-artifact").select_option(capture["id"])
                expect(page.locator("#result-label")).to_have_text("归一前截流")
                page.locator("#artifact-export-format").select_option("array")
                with page.expect_download() as delivery:
                    page.locator("#artifact-export").click()
                delivered=delivery.value
                assert delivered.suggested_filename.endswith(".npy")
                delivered.save_as(str(output / "browser-capture.npy"))
                np.testing.assert_array_equal(np.load(output / "browser-capture.npy"),pixels[:,3:15,4:24]*2.)
                page.locator("#sequence-promote").click()
                expect(page.locator("#frame-info")).to_contain_text("第 1 帧 / 25")
                page.locator("#result-tools-dialog [data-close-dialog]").click()
                expect(page.locator("#steps [data-instance]")).to_have_count(0)
                page.locator("#flow-undo").click()
                expect(page.locator("#steps [data-instance]")).to_have_count(count)
                page.locator("[data-library-tab='sources']").click()
                page.locator(f"#source-list [data-source='{source['id']}']").click()
                page.locator(f"[data-sequence-result='{saved['id']}']").click()
                expect(page.locator("#result-label")).to_have_text(saved["name"])
                before_jobs = len(api.get("/api/jobs").json())
                inspector = page.locator("[data-window='inspector']")
                inspector.locator("[data-window-action='float']").click()
                expect(inspector).to_be_visible()
                inspector.locator("[data-window-action='dock']").click()
                expect(page.locator("#run")).to_be_visible()
                page.wait_for_timeout(300)
                screenshots = []
                for width, height in ((1600, 1000), (1366, 768), (1100, 800), (960, 760)):
                    page.set_viewport_size(dict(width=width, height=height))
                    page.wait_for_timeout(150)
                    expect(page.locator("#run")).to_be_visible()
                    expect(page.locator("#play")).to_be_visible()
                    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth"), (width, height)
                    for canvas in ("#input-canvas","#result-canvas"):
                        box=page.locator(canvas).bounding_box()
                        assert box and box["height"]>=100, (width,height,canvas,box)
                    path = output / f"workbench-{width}.png"
                    page.screenshot(path=str(path))
                    screenshots.append(str(path))
                assert len(api.get("/api/jobs").json()) == before_jobs
                page.reload()
                expect(page.locator("#steps [data-instance]")).to_have_count(count)
                page.set_viewport_size(dict(width=1600,height=1000))

                def template(kind):
                    page.locator("#analysis-mode").select_option(kind)
                    page.locator("#apply-template").click()
                    expect(page.locator("#template-dialog")).to_be_visible()
                    page.locator("#confirm-template").click()
                    page.locator("[data-inspector-tab='parameters']").click()

                # The binding is selected in the UI and changes real numerical output.
                template("general")
                add("parameter_binding")
                provider=page.locator("#steps [data-algorithm='parameter_binding']").get_attribute("data-instance")
                parameter("value",2.5)
                add("affine")
                page.locator("[data-bind='gain']").click()
                page.locator("[data-bind-source]").select_option(provider)
                page.locator("[data-bind-port]").select_option("value")
                page.locator("[data-bind-apply]").click()
                expect(page.locator(".bound-value")).to_contain_text("value")
                before={j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#trial").click()
                bound=completed(before)
                arr=api.get(f"/api/jobs/{bound['id']}/download",params=dict(file="array"))
                arr.raise_for_status()
                np.testing.assert_array_equal(np.load(io.BytesIO(arr.content))[0],pixels[0]*2.5)
                provider_card=page.locator(f"#steps [data-instance='{provider}']")
                provider_card.locator("input[type=checkbox]").uncheck()
                expect(page.locator("#module-status")).to_have_class("connection-status error")
                before_jobs=len(api.get("/api/jobs").json())
                page.locator("#seek-value").fill("7");page.locator("#seek").click()
                expect(page.locator("#frame-info")).to_contain_text("第 7 帧 / 25")
                assert len(api.get("/api/jobs").json())==before_jobs
                provider_card.locator("input[type=checkbox]").check()
                provider_card.click();provider_card.press("Alt+ArrowRight")
                expect(page.locator("#module-status")).to_have_class("connection-status error")
                page.locator("#flow-undo").click()
                expect(page.locator("#module-status")).not_to_have_class("connection-status error")
                provider_card.locator("[data-node-menu]").click()
                provider_card.locator("[data-node-action='delete']").click()
                expect(page.locator("#module-status")).to_have_class("connection-status error")
                page.locator("#flow-undo").click()

                # Register a real local plugin through the one custom-module entry.
                page.locator("#open-module-library").click();page.locator("#plugin").click()
                page.locator("#browse-path").fill(str(ROOT/"examples/plugins/invert.json"))
                page.locator("#browse-go").click();page.locator("#dialog-submit").click()
                expect(page.locator("#file-dialog")).to_be_hidden()
                add("example.invert")
                page.locator("[data-inspector-tab='formula']").click()
                expect(page.locator("#general-module-fields math")).to_be_visible()
                page.screenshot(path=str(output/"custom-formula.png"))

                # The split preset is an editable flow with a readable g2 definition.
                template("split_curve")
                expect(page.locator("#steps [data-instance]")).to_have_count(6)
                page.locator("#steps [data-algorithm='correlation_curve']").click()
                page.locator("[data-inspector-tab='formula']").click()
                expect(page.locator("#general-module-fields math")).to_be_visible()
                page.locator("[data-formula-expand]").click()
                expect(page.locator("#formula-dialog .symbol-table")).to_be_visible()
                page.screenshot(path=str(output/"correlation-formula.png"))
                page.locator("#formula-dialog [data-close-dialog]").click()
                template("general")

                def canvas_point(x,y,w,h):
                    box=page.locator("#input-canvas").bounding_box()
                    scale=min((box["width"]-40)/w,(box["height"]-50)/h)
                    return dict(x=box["x"]+box["width"]/2+(x-w/2)*scale,y=box["y"]+box["height"]/2+(y-h/2)*scale)

                add("roi_crop")
                page.locator("[data-canvas-tool='roi-new']").click()
                expect(page.locator("#axis-stage-note")).to_be_visible(timeout=15000)
                start,end=canvas_point(8.2,6.2,64,32),canvas_point(20.2,16.2,64,32)
                page.mouse.move(**start);page.mouse.down();page.mouse.move(**end,steps=8);page.mouse.up()
                expect(page.locator("#general-module-fields [data-key='x']")).to_have_value("8")
                rw=int(page.locator("#general-module-fields [data-key='width']").input_value())
                rh=int(page.locator("#general-module-fields [data-key='height']").input_value())
                page.locator("#axis-restore-raw").click()
                add("point_extract");parameter("output","side")
                page.locator("[data-canvas-tool='point']").click()
                expect(page.locator("#axis-stage-caption")).to_contain_text("点")
                # All point and line coordinates are local to the cropped stage.
                point=canvas_point(3.2,4.2,rw,rh)
                page.mouse.click(**point)
                expect(page.locator("#general-module-fields [data-key='x']")).to_have_value("3")
                expect(page.locator("#general-module-fields [data-key='y']")).to_have_value("4")
                page.locator("#axis-restore-raw").click()
                add("line_extract");parameter("mode","segment");parameter("output","side");parameter("samples",6)
                page.locator("[data-canvas-tool='line']").click()
                expect(page.locator("#axis-stage-caption")).to_contain_text("线")
                start,end=canvas_point(1.2,2.2,rw,rh),canvas_point(9.2,7.2,rw,rh)
                page.mouse.move(**start);page.mouse.down();page.mouse.move(**end,steps=8);page.mouse.up()
                assert 8<float(page.locator("#general-module-fields [data-key='x1']").input_value())<10
                page.locator("#axis-restore-raw").click()
                add("distribution");parameter("output","side");parameter("bins",8)
                before={j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#run").click();extracted=completed(before)
                page.locator("[data-library-tab='sources']").click()
                page.locator(f"[data-sequence-result='{extracted['id']}']").click()
                page.locator("#open-result-tools").click()
                extraction=extracted["run"]["results"][0]
                line=next(a for a in extraction["side_outputs"] if a["kind"]=="line_extract")
                page.locator("#result-artifact").select_option(line["id"])
                page.locator("#sequence-plot-axis").select_option("component")
                page.locator("#result-tools-dialog [data-close-dialog]").click()
                expect(page.locator("#split-curve svg")).to_be_visible()
                expect(page.locator("#split-curve")).to_contain_text("剖面距离")
                page.locator("#sequence-index").fill("5");page.locator("#sequence-index").dispatch_event("change")
                expected=api.post(f"/api/jobs/{extracted['id']}/sequence",json=dict(side_output=line["id"],index=5,profile=True)).json()
                expect(page.locator("#sequence-readout")).to_contain_text(format(expected["values"][0][0],".7g"))
                page.screenshot(path=str(output/"line-profile.png"))
                page.locator("#open-result-tools").click()
                dist=next(a for a in extraction["side_outputs"] if a["kind"]=="distribution")
                page.locator("#result-artifact").select_option(dist["id"])
                page.locator("#artifact-export-format").select_option("csv")
                with page.expect_download() as csv_delivery:page.locator("#artifact-export").click()
                csv_delivery.value.save_as(str(output/"distribution.csv"))
                assert "B0" in (output/"distribution.csv").read_text(encoding="utf-8-sig")
                page.locator("#result-tools-dialog [data-close-dialog]").click()
                expect(page.locator("#split-curve")).to_contain_text("分布箱中心")
                page.screenshot(path=str(output/"distribution.png"))
                page.reload()
                expect(page.locator("#split-curve")).to_contain_text("分布箱中心")
                # A Q curve keeps signed lag semantics in the actual result viewer.
                template("split_curve")
                page.locator("#steps [data-algorithm='correlation_curve']").click()
                add("point_extract");parameter("x",3);parameter("y",4);parameter("output","main")
                before={j["id"] for j in api.get("/api/jobs").json()}
                page.locator("#trial").click();qjob=completed(before)
                page.locator("[data-library-tab='sources']").click()
                page.locator(f"[data-sequence-result='{qjob['id']}']").click()
                expect(page.locator("#split-curve")).to_contain_text("有符号 τ")
                expect(page.locator("#sequence-play")).to_be_hidden()
                qdata=api.post(f"/api/jobs/{qjob['id']}/sequence",json={}).json()
                assert qdata["contract"]["sequence_axis"]=="Q"
                assert qjob["run"]["spec"]["steps"][1]["parameters"]["curve"]=="linear"
                assert [r["correlation"]["lag_samples"] for r in qdata["references"]]==list(range(-5,6))
                page.screenshot(path=str(output/"signed-lag-curve.png"))

                # Migrate a 0.6 browser draft through the real bootstrap path.
                import uuid
                catalog=api.get("/api/modules").json()
                legacy=[]
                for identifier in ("time_crop","spatial_roi","statistics"):
                    spec=catalog[identifier]
                    legacy.append(dict(id=uuid.uuid4().hex,algorithm=identifier,parameters=spec["defaults"],enabled=True,target="both",version=spec["version"],code_hash=spec["code_hash"],manifest=spec["manifest"]))
                legacy[1]["parameters"].update(mode="mean",x=4,y=3,width=20,height=12)
                draft=dict(web_module_contract=3,steps=legacy,active_source=source["id"])
                draft_key="eip-workbench-view-3:"+str(output)
                page.evaluate("([key,draft])=>localStorage.setItem(key,JSON.stringify(draft))",[draft_key,draft])
                # Prevent pagehide from replacing this deliberately old fixture.
                migrated_page=context.new_page()
                migrated_page.on("pageerror",lambda error:errors.append(str(error)))
                migrated_page.goto(url)
                expect(migrated_page.locator("#steps [data-instance]")).to_have_count(5)
                order=migrated_page.locator("#steps [data-instance]").evaluate_all("els=>els.map(e=>e.dataset.algorithm)")
                assert order==["time_crop","roi_crop","region_reduce","sequence_statistics","fixed_normalize"],order
                expect(migrated_page.locator("#frame-info")).to_contain_text("25")
                assert not errors, errors
                report = dict(
                    version=api.get("/api/state").json()["version"], source_count=1,
                    shared_raw_canvas=True, categories_and_search=True, roi_has_no_reduction=True,
                    formula_and_contract_tabs=True, independent_instances=True, undo_redo=True,
                    preview_did_not_compute=True, trial=trial["id"], saved=saved["id"],
                    tap_side_output=True, tap_browser_download_and_promotion=True, horizontal_flow=True, main_result_numeric_verified=True,
                    float_and_dock=True, responsive_sizes=screenshots, restored_flow=True,
                    parameter_binding_and_dependency_errors=True, custom_module_registered=True,
                    editable_split_template=True, roi_point_line_canvas_tools=True, exact_line_profile=True,
                    distribution_csv=True, restored_side_output=True, signed_q_curve=True, migrated_browser_draft=True,
                    javascript_errors=errors,
                )
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
