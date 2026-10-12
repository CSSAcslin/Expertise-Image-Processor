from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import deque
import io
import json
import time
import threading

import numpy as np
import pytest

from eip.module_registry import new_node
from eip.web.runtime import Workbench
from test_sequence_web import client, open_tiff, wait


def inspect(client, source, nodes, **values):
    response = client.post("/api/modules/inspect", json=dict(source_id=source["id"], steps=nodes, **values))
    assert response.status_code == 200, response.text
    return response.json()


def slow_plugin(client, tmp_path, *, prepare=0, process=0):
    marker = tmp_path / "entered.txt"
    (tmp_path / "slow.py").write_text(
        "import time\nfrom pathlib import Path\nclass Slow:\n"
        " def prepare(self, parameters, context):\n"
        f"  Path({str(marker)!r}).write_text('entered')\n"
        f"  time.sleep({prepare})\n"
        "  context.log('inspection-regression-marker')\n"
        " def process(self, pixels, context):\n"
        f"  time.sleep({process})\n"
        "  context.check_cancelled()\n  return pixels.copy()\n"
        " def close(self): pass\n", encoding="utf-8")
    manifest = tmp_path / "slow.json"
    manifest.write_text(json.dumps(dict(schema_version=1, id="slow", name="Slow", entry_point="slow.py:Slow")), encoding="utf-8")
    spec = client.post("/api/plugins/register", json=dict(path=str(manifest))).json()
    node = dict(id="slow-instance", algorithm=spec["id"], parameters={}, enabled=True,
                manifest=spec["manifest"], code_hash=spec["code_hash"], version=spec["version"])
    return node, marker


def until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    pytest.fail("interactive worker did not reach expected state")


def test_prefix_run_skips_invalid_tail_and_records_boundary(client, tmp_path):
    values = np.arange(7*4*6, dtype=np.uint16).reshape(7, 4, 6)
    source = open_tiff(client, tmp_path, values)
    crop, affine, point, invalid = [new_node(a) for a in ("time_crop", "affine", "point_extract", "gaussian")]
    crop["parameters"].update(first=3, last=6)
    affine["parameters"].update(gain=3, offset=7)
    point["parameters"].update(output="main")
    nodes = [crop, affine, point, invalid]
    payload = dict(contract_version=3, mode="run", source_ids=[source["id"]], steps=nodes)
    assert client.post("/api/jobs", json=payload).status_code == 400
    response = client.post("/api/jobs", json=dict(payload, stop_after=affine["id"]))
    assert response.status_code == 200, response.text
    job = wait(client, response.json()["id"])
    assert job["status"] == "succeeded", job
    run = job["run"]
    assert [n["id"] for n in run["spec"]["steps"]] == [crop["id"], affine["id"]]
    assert run["spec"]["execution_scope"] == dict(kind="prefix", stop_after=affine["id"], full_step_count=4)
    actual = np.load(io.BytesIO(client.get(f"/api/jobs/{job['id']}/download?file=array").content), allow_pickle=False)
    np.testing.assert_array_equal(actual, values[2:6]*3+7)
    assert client.post("/api/jobs", json=dict(payload, stop_after="missing")).status_code == 400


def test_inspection_preserves_axes_coordinates_and_never_enqueues(client, tmp_path):
    values = np.arange(9*4*6, dtype=np.uint16).reshape(9, 4, 6)
    source = open_tiff(client, tmp_path, values)
    crop, sample, roi, point = [new_node(a) for a in ("time_crop", "time_sample", "roi_crop", "point_extract")]
    crop["parameters"].update(first=3, last=9)
    roi["parameters"].update(x=1, y=1, width=3, height=2)
    prefix = [crop, sample, roi]
    data = inspect(client, source, prefix, sample_index=1)
    assert data["contract"]["full_axes"] == "TYX"
    assert data["contract"]["length"] == 4
    assert data["reference"]["index"] == 4
    frame = data["frame"]
    assert frame["shape"] == [2, 3] and frame["origin"] == [0, 0]
    pixel = client.get(f"/api/frames/{frame['token']}/pixel?x=2&y=1").json()
    assert pixel["value"] == int(values[4, 2, 3])
    point["parameters"].update(x=2, y=1, output="main")
    data = inspect(client, source, prefix+[point], sample_index=1)
    assert data["kind"] == "signals" and data["contract"]["full_axes"] == "T"
    assert data["contract"]["shape"] == []
    assert data["values"] == [[float(values[4, 2, 3])]]
    section = new_node("axis_slice")
    section["parameters"].update(axis="T", index=2, output="main")
    static = inspect(client, source, prefix+[section])
    assert static["contract"]["sequence_axis"] == "N" and static["contract"]["axes"] == "YX"
    assert static["reference"]["index"] == 6
    assert client.get("/api/jobs").json() == []
    assert not (tmp_path / "outputs").exists()
    np.testing.assert_array_equal(__import__("tifffile").imread(tmp_path/"source.tif"), values)


def test_main_job_does_not_block_interactive_or_raw_preview(client, tmp_path):
    values = np.arange(30*4*6, dtype=np.uint16).reshape(30, 4, 6)
    source = open_tiff(client, tmp_path, values)
    slow, marker = slow_plugin(client, tmp_path, process=.2)
    response = client.post("/api/jobs", json=dict(contract_version=3, mode="run", source_ids=[source["id"]], steps=[slow]))
    assert response.status_code == 200, response.text
    identifier = response.json()["id"]
    until(marker.exists)
    roi = new_node("roi_crop")
    roi["parameters"].update(x=1, y=1, width=3, height=2)
    data = inspect(client, source, [roi], sample_index=3)
    assert data["frame"]["statistics"]["mean"] == values[3, 1:3, 1:4].mean()
    assert client.get(f"/api/jobs/{identifier}").json()["status"] == "running"
    raw = client.post(f"/api/sources/{source['id']}/frame", json=dict(action="index", index=5))
    assert raw.status_code == 200 and raw.json()["ref"]["index"] == 5
    assert len(client.get("/api/jobs").json()) == 1
    client.post(f"/api/jobs/{identifier}/cancel", json={})
    assert wait(client, identifier)["status"] == "cancelled"


def test_slow_inspection_is_replaceable_and_cancel_does_not_touch_raw(client, tmp_path):
    source = open_tiff(client, tmp_path, np.arange(3*4*6, dtype=np.uint16).reshape(3, 4, 6))
    slow, marker = slow_plugin(client, tmp_path, prepare=10)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/api/modules/inspect", json=dict(source_id=source["id"], steps=[slow], inspection_id="old"))
        until(marker.exists)
        started = time.monotonic()
        raw = client.post(f"/api/sources/{source['id']}/frame", json=dict(action="index", index=2))
        assert raw.status_code == 200 and raw.json()["ref"]["index"] == 2
        fresh = inspect(client, source, [], inspection_id="new")
        assert fresh["frame"]["statistics"]["min"] == 0
        assert time.monotonic()-started < 8
        old = pending.result(timeout=4)
        assert old.status_code == 400 and "替代" in old.text
        marker.unlink()
        pending = pool.submit(client.post, "/api/modules/inspect", json=dict(source_id=source["id"], steps=[slow], inspection_id="cancel-me"))
        until(marker.exists)
        assert client.post("/api/modules/inspect/cancel", json=dict(inspection_id="unrelated")).json() == dict(cancelled=False)
        assert client.post("/api/modules/inspect/cancel", json=dict(inspection_id="cancel-me")).json() == dict(cancelled=True)
        assert pending.result(timeout=4).status_code == 400
    assert client.get("/api/jobs").json() == []
    assert inspect(client, source, [], sample_index=1)["reference"]["index"] == 1


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled", "completed_with_errors"])
def test_late_cancel_keeps_terminal_status(status):
    # A completed worker may still be exiting when a stale cancel click arrives.
    workbench = object.__new__(Workbench)
    workbench.lock = threading.RLock()
    workbench.pending = deque()
    workbench.jobs = {"done": dict(status=status)}
    workbench.active = dict(id="done", terminal=True, cancel=threading.Event())
    workbench.cancel("done")
    assert workbench.jobs["done"]["status"] == status
    assert not workbench.active["cancel"].is_set()
