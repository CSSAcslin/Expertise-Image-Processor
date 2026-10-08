import base64
import io
from pathlib import Path
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
import tifffile

from eip.web.app import create_app
from eip.web.runtime import FrameStore


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        client.headers["X-EIP-Token"] = client.get("/api/state").json()["session_token"]
        yield client


def opened(client, tmp_path, array):
    path = tmp_path / "raw.tif"
    tifffile.imwrite(path, array, photometric="minisblack")
    response = client.post("/api/sources/open", json={"path": str(path)})
    assert response.status_code == 200, response.text
    return response.json()


def wait_job(client, identifier):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{identifier}").json()
        if job["status"] in {"succeeded", "failed", "cancelled", "completed_with_errors"}:
            return job
        time.sleep(0.05)
    pytest.fail("Web job did not finish")


def submit(client, source, frame, mode="trial", **extra):
    value = {"source_ids": [source["id"]], "mode": mode, "steps": [{"algorithm": "intensity", "parameters": {"gain": 2, "offset": 7}}], "selection": {"mode": "current", "current": frame["ref"], "roi": {"x": 2, "y": 1, "width": 3, "height": 2}}, **extra}
    response = client.post("/api/jobs", json=value)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def test_local_session_and_static_assets(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/jobs", headers={"X-EIP-Token": "wrong"}).status_code == 403
    assert client.post("/api/demo", headers={"Origin": "https://untrusted.invalid"}, json={}).status_code == 403
    assert client.get("/api/state", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.get("/api/state", headers={"Host": "untrusted.invalid"}).status_code == 400


def test_display_and_true_pixels_remain_separate(client, tmp_path):
    array = np.arange(48, dtype=np.uint16).reshape(6, 8) * 1000
    data = opened(client, tmp_path, array)
    frame = data["frame"]
    assert frame["dtype"] == "uint16"
    png = Image.open(io.BytesIO(base64.b64decode(frame["image"].split(",")[1])))
    assert png.mode == "RGB" and png.size == (8, 6)
    rendered = client.post(f'/api/frames/{frame["token"]}/render', json={"display": {"low": 0, "high": 65535, "gamma": 2, "colormap": "heat"}})
    assert rendered.status_code == 200
    assert rendered.json()["image"] != frame["image"]
    pixel = client.get(f'/api/frames/{frame["token"]}/pixel?x=7&y=5').json()
    assert pixel["value"] == 47000 and pixel["dtype"] == "uint16"
    assert client.get(f'/api/frames/{frame["token"]}/pixel?x=8&y=5').status_code == 400
    assert client.post(f'/api/sources/{data["source"]["id"]}/frame', json={"action": "index", "index": 0.5}).status_code == 400
    assert client.post("/api/jobs", json={"mode": "run", "source_ids": [data["source"]["id"]], "steps": [], "selection": {"mode": "range", "start": 0, "end": 0.5}}).status_code == 400
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "raw.tif"), array)


def test_trial_export_result_mapping_and_project(client, tmp_path):
    array = np.arange(48, dtype=np.uint16).reshape(6, 8) * 100
    data = opened(client, tmp_path, array)
    identifier = submit(client, data["source"], data["frame"])
    trial = wait_job(client, identifier)
    assert trial["status"] == "succeeded", trial
    result = trial["result"]
    assert result["origin"] == [2, 1] and result["shape"] == [2, 3]
    pixel = client.get(f'/api/frames/{result["token"]}/pixel?x=2&y=1').json()
    assert pixel["value"] == int(array[1, 2]) * 2 + 7
    identifier = submit(client, data["source"], data["frame"], mode="run")
    formal = wait_job(client, identifier)
    assert formal["status"] == "succeeded", formal
    tiff = client.get(f"/api/jobs/{identifier}/download?file=image").content
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(tiff)), array[1:3, 2:5] * 2 + 7)
    assert client.get(f"/api/jobs/{identifier}/download?file=csv").status_code == 200
    assert client.get(f"/api/jobs/{identifier}/download?file=manifest").json()["app_version"] == "0.2.0"
    result_entry = next(s for s in client.get("/api/state").json()["sources"] if s["role"] == "result")
    mapped = client.post(f'/api/sources/{result_entry["id"]}/frame', json={"action": "match", "reference": {"index": 0, "time": None}}).json()
    assert mapped["origin"] == [2, 1] and mapped["base_shape"] == [6, 8]
    project = tmp_path / "project.json"
    assert client.post("/api/project/save", json={"path": str(project), "view": {"steps": [], "web_roi": {"x": 2, "y": 1, "width": 3, "height": 2}}}).status_code == 200
    # The terminal status can precede worker exit by a fraction of a second.
    deadline = time.monotonic() + 3
    while True:
        restored = client.post("/api/project/load", json={"path": str(project)})
        if restored.status_code == 200 or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    assert restored.status_code == 200, restored.text
    restored_result = next(s for s in restored.json()["state"]["sources"] if s["role"] == "result")
    assert restored_result["base_shape"] == [6, 8]
    assert client.post("/api/project/save", json={"path": str(tmp_path / "raw.tif"), "view": {"steps": []}}).status_code == 400


def test_external_plugin_runs_in_worker(client, tmp_path):
    data = opened(client, tmp_path, np.full((6, 8), 12, dtype=np.uint16))
    manifest = Path(__file__).resolve().parents[1] / "examples" / "plugins" / "invert.json"
    spec = client.post("/api/plugins/register", json={"path": str(manifest)}).json()
    step = {"algorithm": spec["id"], "parameters": {p["key"]: p["default"] for p in spec["parameters"]}, "manifest": spec["manifest"], "code_hash": spec["code_hash"], "version": spec["version"]}
    job = wait_job(client, submit(client, data["source"], data["frame"], steps=[step]))
    assert job["status"] == "succeeded", job
    assert job["result"]["dtype"] == "float32"
    assert client.post("/api/jobs", json={"mode": "trial", "source_ids": [data["source"]["id"]], "steps": [{**step, "code_hash": "outdated"}], "selection": {"mode": "current", "current": data["frame"]["ref"]}}).status_code == 400


def test_queue_cancellation(client, tmp_path):
    import json
    (tmp_path / "slow.py").write_text("import time\nclass Slow:\n def prepare(self, parameters, context):\n  time.sleep(.7)\n def process(self, pixels, context):\n  context.check_cancelled()\n  return pixels.copy()\n def close(self): pass\n", encoding="utf-8")
    manifest = tmp_path / "slow.json"
    manifest.write_text(json.dumps({"schema_version": 1, "id": "slow", "name": "Slow", "entry_point": "slow.py:Slow"}), encoding="utf-8")
    spec = client.post("/api/plugins/register", json={"path": str(manifest)}).json()
    step = {"algorithm": "slow", "parameters": {}, "manifest": spec["manifest"], "code_hash": spec["code_hash"], "version": spec["version"]}
    data = opened(client, tmp_path, np.ones((6, 8), dtype=np.uint16))
    running = submit(client, data["source"], data["frame"], steps=[step])
    queued = submit(client, data["source"], data["frame"], steps=[step])
    assert client.post(f"/api/jobs/{queued}/cancel", json={}).status_code == 200
    assert wait_job(client, queued)["status"] == "cancelled"
    assert client.post(f"/api/jobs/{running}/cancel", json={}).status_code == 200
    assert wait_job(client, running)["status"] == "cancelled"


def test_frame_cache_is_bounded_and_roi_pixel_uses_origin():
    store = FrameStore()
    oldest = store.add(np.ones((5, 5), dtype=np.uint16), {"index": 0}, [7, 8], [20, 20], "image")
    assert store.pixel(oldest, 7, 8)["value"] == 1
    for i in range(8):
        store.add(np.ones((5, 5), dtype=np.uint16), {"index": i}, [0, 0], [5, 5], "image")
    assert len(store.frames) == 6
    with pytest.raises(ValueError):
        store.get(oldest)


def test_web_video_reopen_steps_and_formal_vfr_mapping(client, vfr_video):
    path, stamps = vfr_video
    data = client.post("/api/sources/open", json={"path": str(path)}).json()
    identifier = data["source"]["id"]
    located = client.post(f"/api/sources/{identifier}/frame", json={"action": "time", "time": 0.08}).json()
    assert located["ref"]["time"] == pytest.approx(0.11)
    client.post(f"/api/sources/{identifier}/frame", json={"action": "open"})
    stepped = client.post(f"/api/sources/{identifier}/frame", json={"action": "next", "time": 0}).json()
    assert stepped["ref"]["time"] == pytest.approx(0.04)
    job = wait_job(client, submit(client, data["source"], located, mode="run", selection={"mode": "range", "start": 0.08, "end": 0.3, "roi": None}))
    assert job["status"] == "succeeded"
    result = next(s for s in client.get("/api/state").json()["sources"] if s["role"] == "result")
    mapped = client.post(f'/api/sources/{result["id"]}/frame', json={"action": "match", "reference": {"index": None, "time": 0.16}}).json()
    assert mapped["ref"]["index"] == 1 and mapped["ref"]["time"] == pytest.approx(0.16)
    assert mapped["dtype"] == "uint16"
    assert client.get(f'/api/frames/{mapped["token"]}/pixel?x=0&y=0').json()["value"] == 8007
    missing = client.post(f'/api/sources/{result["id"]}/frame', json={"action": "match", "reference": {"index": 1, "time": 0.04}}).json()
    assert missing == {"unavailable": True}
