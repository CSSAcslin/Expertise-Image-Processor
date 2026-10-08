from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pytest
import tifffile

from eip.domain import FrameRef, ROI, Selection, SourceSpec, StepSpec
from eip.processing import process_job
from eip.readers import open_source


def payload_for(source, output, selection=None):
    reader = open_source(source)
    try:
        revision = reader.meta.revision
    finally:
        reader.close()
    return {"id": "test-run", "mode": "run", "sources": [asdict(source)], "revisions": [revision], "steps": [asdict(StepSpec("intensity", {"gain": 1.5, "offset": 10}))], "selection": asdict(selection or Selection()), "output_dir": str(output), "config_revision": 1}


def test_trial_formal_export_and_reopen_are_identical(scientific_image, tmp_path):
    path, pixels = scientific_image
    selection = Selection("current", current=FrameRef(0, None), roi=ROI(2, 1, 3, 2))
    payload = payload_for(SourceSpec(str(path)), tmp_path / "out", selection)
    trial_events, formal_events = [], []
    process_job({**payload, "mode": "trial"}, trial_events.append, lambda: False)
    process_job(payload, formal_events.append, lambda: False)
    trial = next(event for event in trial_events if event["type"] == "trial")
    final = formal_events[-1]
    assert final["type"] == "completed" and final["run"]["status"] == "succeeded"
    output_file = Path(final["directory"]) / "source_001/frame_00000000.tif"
    actual = tifffile.imread(output_file)
    np.testing.assert_array_equal(actual, trial["pixels"])
    expected = np.clip(np.rint(pixels.astype(float) * 1.5 + 10), 0, 65535).astype(np.uint16)[1:3, 2:5]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint16
    assert final["run"]["results"][0]["origin"] == [2, 1]
    record = json.loads((Path(final["directory"]) / "run.json").read_text(encoding="utf-8"))
    assert record["content_sha256"] is None and record["fingerprint_kind"] == "path_size_mtime_ns"
    assert record["spec"]["steps"][0]["parameters"]["gain"] == 1.5
    assert (Path(final["directory"]) / "source_001/measurements.csv").exists()


def test_batch_preserves_per_file_failure(scientific_image, tmp_path):
    path, _ = scientific_image
    source = SourceSpec(str(path))
    payload = payload_for(source, tmp_path / "out")
    payload["sources"].append(asdict(SourceSpec(str(tmp_path / "missing.tif"))))
    payload["revisions"].append("missing")
    events = []
    process_job(payload, events.append, lambda: False)
    final = events[-1]
    assert final["run"]["status"] == "completed_with_errors"
    assert [item["status"] for item in final["run"]["results"]] == ["succeeded", "failed"]
    assert Path(final["directory"]).name == "test-run"


def test_cancellation_keeps_partial_output_unpublished(tmp_path):
    sequence = tmp_path / "input"
    sequence.mkdir()
    for index in range(3):
        tifffile.imwrite(sequence / f"{index}.tif", np.full((10, 10), index, dtype=np.uint16))
    payload = payload_for(SourceSpec(str(sequence), "sequence", 0.1), tmp_path / "out")
    events = []
    stop = [False]
    def emit(event):
        events.append(event)
        if event["type"] == "progress":
            stop[0] = True
    process_job(payload, emit, lambda: stop[0])
    assert events[-1]["type"] == "cancelled"
    assert not (tmp_path / "out/test-run").exists()
    staging = tmp_path / "out/test-run.partial"
    assert staging.exists()
    record = json.loads((staging / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "cancelled" and record["results"][0]["completed_frames"] == 1


def test_changed_input_is_not_silently_processed(scientific_image, tmp_path):
    path, pixels = scientific_image
    payload = payload_for(SourceSpec(str(path)), tmp_path / "out")
    tifffile.imwrite(path, pixels + 1)
    events = []
    process_job(payload, events.append, lambda: False)
    final = events[-1]
    assert final["run"]["status"] == "failed"
    assert "变化" in final["run"]["results"][0]["error"]


def test_exported_video_sequence_reopens_with_real_timestamps(vfr_video, tmp_path):
    path, stamps = vfr_video
    payload = payload_for(SourceSpec(str(path)), tmp_path / "out")
    events = []
    process_job(payload, events.append, lambda: False)
    reader = open_source(SourceSpec(str(Path(events[-1]["directory"]) / "source_001"), "sequence"))
    try:
        assert reader.meta.timestamp_source == "run_record"
        assert [frame.ref.time for frame in reader.iter_frames(Selection())] == pytest.approx(stamps)
    finally:
        reader.close()
