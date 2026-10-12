"""Saved side ports retain numerical meaning and never replace the main flow."""
from pathlib import Path
import io
import json
import time
import zipfile

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from eip.module_registry import new_node
from eip.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as session:
        session.headers["X-EIP-Token"] = session.get("/api/state").json()["session_token"]
        yield session


def source(client, tmp_path, values):
    path = tmp_path / "source.tif"
    tifffile.imwrite(path, values, photometric="minisblack", metadata={"axes": "TYX"})
    response = client.post("/api/sources/open", json=dict(path=str(path), interval=.01))
    assert response.status_code == 200, response.text
    return response.json()["source"]


def run(client, entry, nodes, **options):
    response = client.post("/api/jobs", json=dict(contract_version=3, source_ids=[entry["id"]], steps=nodes, **options))
    assert response.status_code == 200, response.text
    identifier = response.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{identifier}").json()
        if job["status"] in {"succeeded", "failed", "completed_with_errors", "cancelled"}:
            return job
        time.sleep(.03)
    pytest.fail("side-output worker timeout")


def test_tap_captures_direct_input_and_auto_previews(client, tmp_path):
    values = np.arange(4 * 2 * 4, dtype=np.uint16).reshape(4, 2, 4)
    entry = source(client, tmp_path, values)
    before, tap, after = [new_node(a) for a in ("affine", "data_tap", "affine")]
    before["parameters"]["gain"] = 2
    after["parameters"]["gain"] = 3
    tap["parameters"].update(name="双倍中间数据", destination="both", export_format="npy")
    job = run(client, entry, [before, tap, after])
    assert job["status"] == "succeeded", job
    item = job["run"]["results"][0]
    side = item["side_outputs"][0]
    assert side["status"] == "succeeded" and not side["temporary"]
    np.testing.assert_array_equal(np.load(Path(job["directory"]) / item["output_directory"] / "values.npy"), values * 6)
    url = f"/api/jobs/{job['id']}/download?side_output={side['id']}&file=export"
    response = client.get(url)
    assert response.status_code == 200, response.text
    np.testing.assert_array_equal(np.load(io.BytesIO(response.content)), values * 2)
    frame = client.post(f"/api/jobs/{job['id']}/sequence", json=dict(side_output=side["id"], index=2)).json()
    assert frame["kind"] == "image" and frame["reference"]["index"] == 2
    captured = next(s for s in client.get("/api/state").json()["sources"] if s.get("side_output") == side["id"])
    assert captured["role"] == "input" and captured["display_name"] == "双倍中间数据"
    assert captured["lineage"][-1]["node_id"] == tap["id"]
    follow = run(client, captured, [new_node("affine")])
    assert follow["status"] == "succeeded", follow


def test_csv_signal_and_dataset_package(client, tmp_path):
    values = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    entry = source(client, tmp_path, values)
    reduce, csv_tap, archive_tap = [new_node(a) for a in ("region_reduce", "data_tap", "data_tap")]
    csv_tap["parameters"].update(destination="file", export_format="csv")
    archive_tap["parameters"].update(destination="file", export_format="dataset")
    job = run(client, entry, [reduce, csv_tap, archive_tap])
    assert job["status"] == "succeeded", job
    first, second = job["run"]["results"][0]["side_outputs"]
    assert first["contract"]["axes"] == ""
    response = client.get(f"/api/jobs/{job['id']}/download", params=dict(file="export", side_output=first["id"]))
    assert "output_index,time_seconds,lag_samples,value" in response.content.decode("utf-8-sig")
    signals = client.post(f"/api/jobs/{job['id']}/sequence", json=dict(side_output=first["id"])).json()
    np.testing.assert_allclose(np.array(signals["values"])[:, 0], values.mean(axis=(1, 2)))
    response = client.get(f"/api/jobs/{job['id']}/download", params=dict(file="export", side_output=second["id"]))
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert {"dataset.json", "delivery.json", "values.npy", "references.jsonl"}.issubset(archive.namelist())
        assert json.loads(archive.read("delivery.json"))["status"] == "succeeded"
    assert not any(s.get("side_output") for s in client.get("/api/state").json()["sources"])
    response = client.post(f"/api/jobs/{job['id']}/sequence/promote", json=dict(side_output=first["id"]))
    assert response.status_code == 200, response.text
    assert response.json()["source"]["role"] == "input"


def test_trial_is_temporary_and_not_promotable(client, tmp_path):
    values = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    entry = source(client, tmp_path, values)
    tap = new_node("data_tap")
    tap["parameters"].update(destination="preview")
    job = run(client, entry, [tap], mode="trial", sample_index=1)
    assert job["status"] == "succeeded", job
    side = job["run"]["results"][0]["side_outputs"][0]
    assert side["temporary"] and side["contract"]["length"] == 1
    frame = client.post(f"/api/jobs/{job['id']}/sequence", json=dict(side_output=side["id"])).json()
    assert frame["temporary"] and frame["reference"]["index"] == 1
    response = client.post(f"/api/jobs/{job['id']}/sequence/promote", json=dict(side_output=side["id"]))
    assert response.status_code == 400 and "临时" in response.text
    assert len(client.get("/api/state").json()["sources"]) == 1


def test_tiff_retains_both_aligned_planes(client, tmp_path):
    values = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    entry = source(client, tmp_path, values)
    align, tap = new_node("alignment"), new_node("data_tap")
    align["parameters"].update(left=dict(x=0, y=0, width=2, height=2), right=dict(x=2, y=0, width=2, height=2), midline=[2, 2])
    tap["parameters"].update(destination="file", export_format="tiff")
    job = run(client, entry, [align, tap])
    assert job["status"] == "succeeded", job
    item = job["run"]["results"][0]
    side = item["side_outputs"][0]
    path = Path(job["directory"]) / item["output_directory"] / side["output_directory"] / "export.tif"
    with tifffile.TiffFile(path) as image:
        assert len(image.series) == 3
        np.testing.assert_array_equal(image.series[1].asarray(), np.stack((values[1, :, :2], values[1, :, 2:])))
        assert image.series[1].axes == "PYX"


def test_side_identity_survives_project_reload(client, tmp_path):
    values = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    entry = source(client, tmp_path, values)
    tap = new_node("data_tap")
    tap["parameters"].update(destination="file", export_format="npy")
    job = run(client, entry, [tap])
    side = job["run"]["results"][0]["side_outputs"][0]
    response = client.post(f"/api/jobs/{job['id']}/sequence/promote", json=dict(side_output=side["id"], target="result"))
    assert response.status_code == 200, response.text
    project = tmp_path / "project.json"
    assert client.post("/api/project/save", json=dict(path=str(project), view=dict(steps=[tap], web_module_contract=3))).status_code == 200
    response = client.post("/api/project/load", json=dict(path=str(project)))
    assert response.status_code == 200, response.text
    captured = next(s for s in response.json()["state"]["sources"] if s.get("side_output") == side["id"])
    assert captured["source_position"] == 0
    response = client.post(f"/api/sources/{captured['id']}/promote", json={})
    assert response.status_code == 200, response.text
    assert response.json()["source"]["lineage"][-1]["port"] == side["port"]


def test_spatial_csv_is_rejected_before_computing(client, tmp_path):
    entry = source(client, tmp_path, np.ones((3, 2, 4), dtype=np.uint16))
    tap = new_node("data_tap")
    tap["parameters"].update(destination="file", export_format="csv")
    response = client.post("/api/jobs", json=dict(contract_version=3, source_ids=[entry["id"]], steps=[tap]))
    assert response.status_code == 400 and "CSV" in response.text
    assert client.get("/api/jobs").json() == []


@pytest.mark.parametrize("normalization", ["none", "fixed_rms"])
def test_old_project_migration_keeps_source_and_g2(client, tmp_path, normalization):
    values = (np.arange(13 * 2 * 4).reshape(13, 2, 4) + 1).astype(np.uint16)
    entry = source(client, tmp_path, values)
    align, stats, roi, corr = [new_node(a) for a in ("alignment", "statistics", "spatial_roi", "correlation")]
    align["parameters"].update(left=dict(x=0, y=0, width=2, height=2), right=dict(x=2, y=0, width=2, height=2), midline=[2, 2])
    stats["parameters"]["normalization"] = normalization
    roi["parameters"]["mode"] = "mean"
    corr["parameters"].update(analysis="curve", window=3, lag_list=[-1, 0, 1], center=7)
    legacy = [align, stats, roi, corr]
    old_job = run(client, entry, legacy)
    assert old_job["status"] == "succeeded", old_job
    old_values = client.post(f"/api/jobs/{old_job['id']}/sequence", json={}).json()
    project = tmp_path / "legacy.eip.json"
    assert client.post("/api/project/save", json=dict(path=str(project), view=dict(steps=legacy, web_module_contract=3))).status_code == 200
    original = project.read_bytes()
    response = client.post("/api/project/load", json=dict(path=str(project), migrate=True))
    assert response.status_code == 200, response.text
    migrated = response.json()
    assert project.read_bytes() == original
    assert migrated["view"]["migration"]["report"]
    assert Path(migrated["view"]["migration"]["path"]).is_file()
    assert Path(migrated["state"]["project_path"]) != project
    steps = migrated["view"]["steps"]
    assert [n["algorithm"] for n in steps] == ["alignment", "sequence_statistics", "fixed_normalize", "roi_crop", "region_reduce", "correlation_curve"]
    new_job = run(client, entry, steps)
    assert new_job["status"] == "succeeded", new_job
    new_values = client.post(f"/api/jobs/{new_job['id']}/sequence", json={}).json()
    assert new_values["contract"]["dtype"] == old_values["contract"]["dtype"] == "float64"
    np.testing.assert_allclose(new_values["values"], old_values["values"], rtol=1e-13)
    np.testing.assert_allclose([r["correlation"]["lag_seconds"] for r in new_values["references"]], [-.01, 0, .01])
    response = client.post("/api/project/save", json=dict(path=str(project), view=dict(steps=steps, web_module_contract=3)))
    assert response.status_code == 400
    assert project.read_bytes() == original


def test_changed_source_revokes_completed_side_carrier(tmp_path, monkeypatch):
    from dataclasses import asdict, replace
    import eip.sequence_processing as processing
    from eip.domain import SourceSpec
    from eip.readers import open_source

    values = np.arange(3 * 2 * 4, dtype=np.uint16).reshape(3, 2, 4)
    path = tmp_path / "changing.tif"
    tifffile.imwrite(path, values, photometric="minisblack", metadata={"axes": "TYX"})
    spec = SourceSpec(str(path), interval=.01)
    reader = open_source(spec)
    revision = reader.meta.revision
    reader.close()
    original_open = processing.open_source
    count = 0

    def changing_open(source_spec):
        nonlocal count
        count += 1
        opened = original_open(source_spec)
        if count > 1:
            opened.meta = replace(opened.meta, revision="changed-during-run")
        return opened

    monkeypatch.setattr(processing, "open_source", changing_open)
    events = []
    payload = dict(id="revision-test", display_name="revision-test", mode="run", output_dir=str(tmp_path), sources=[asdict(spec)], revisions=[revision], steps=[new_node("data_tap")])
    processing.process_sequence_job(payload, events.append, lambda: False)
    complete = next(event for event in events if event["type"] == "sequence_completed")
    assert complete["run"]["status"] == "failed"
    item = complete["run"]["results"][0]
    side = item["side_outputs"][0]
    assert side["status"] == "incomplete"
    assert not (Path(complete["directory"]) / item["output_directory"] / side["output_directory"] / "dataset.json").exists()
