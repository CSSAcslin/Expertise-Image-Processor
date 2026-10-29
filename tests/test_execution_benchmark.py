"""Saved-run replay verifies complete values and exact source semantics."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from eip.domain import SourceSpec
from eip.module_registry import new_node
from eip.readers import open_source
from eip.sequence_processing import process_sequence_job


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "benchmark_execution_time.py"


def test_reference_comparison_tolerates_only_accumulated_lag_mean():
    spec = importlib.util.spec_from_file_location("execution_benchmark", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = dict(index=12, time=.012, correlation=dict(
        lag_samples=-2, lag_seconds=-.002, lag_seconds_min=-.0021,
        lag_seconds_max=-.0019, known_pairs=1001, left_start=0, left_stop=1001))
    actual = json.loads(json.dumps(expected))
    actual["correlation"]["lag_seconds"] += 4e-16
    assert module.reference_difference(expected, actual, atol=1e-12) is None
    actual["correlation"]["lag_seconds_min"] += 4e-16
    assert module.reference_difference(expected, actual, atol=1e-12) == "correlation.lag_seconds_min"
    actual = json.loads(json.dumps(expected))
    actual["index"] = 12.0
    assert module.reference_difference(expected, actual, atol=1e-12) == "index"


@pytest.mark.parametrize("tamper_last_value", [False, True])
def test_saved_multisource_replay_checks_last_value_and_preserves_originals(tmp_path, tamper_last_value):
    sources, revisions = [], []
    for index in range(2):
        path = tmp_path / f"source-{index}.tif"
        pixels = np.arange(7 * 4 * 8, dtype=np.uint16).reshape(7, 4, 8) + index * 100
        tifffile.imwrite(path, pixels, photometric="minisblack", metadata={"axes": "TYX"})
        source = SourceSpec(str(path), interval=.01)
        reader = open_source(source)
        revisions.append(reader.meta.revision)
        reader.close()
        sources.append(asdict(source))
    crop, point, tap = [new_node(a) for a in ("time_crop", "point_extract", "data_tap")]
    crop["parameters"].update(first=2, last=6)
    point["parameters"].update(x=2, y=1)
    point.setdefault("execution", {})["backend"] = "cpu"
    payload = dict(id="baseline", mode="run", contract_version=3, display_name="Baseline",
                   sources=sources, revisions=revisions, steps=[crop, point, tap], seed=0,
                   output_dir=str(tmp_path / "original"), resources=dict(
                       memory_mb=256, preview_mb=16, cache_mb=0, gpu_mb=128, backend="cpu"))
    events = []
    process_sequence_job(payload, events.append, lambda: False)
    event = next(e for e in events if e["type"] == "sequence_completed")
    assert event["run"]["status"] == "succeeded"
    baseline_path = Path(event["directory"]) / "run.json"
    if tamper_last_value:
        values_path = baseline_path.parent / "source_001" / "values.npy"
        values = np.load(values_path, allow_pickle=False)
        values[-1] += 1
        np.save(values_path, values, allow_pickle=False)
    original_manifest = baseline_path.read_bytes()
    original_sources = [Path(s["path"]).read_bytes() for s in sources]
    result = subprocess.run([sys.executable, str(TOOL), "--baseline", str(baseline_path),
                             "--output-dir", str(tmp_path / "replays"), "--timeout", "30"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert result.returncode == (1 if tamper_last_value else 0), result.stdout + result.stderr
    report_path = next((tmp_path / "replays").glob("execution-*/report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = report["repeats"][0]["validation"]
    assert validation["passed"] is (not tamper_last_value)
    assert len(validation["sources"]) == 2
    assert validation["sources"][0]["values"]["checked_values"] == 5
    assert validation["sources"][0]["references"]["byte_equal"]
    assert len(validation["sources"][0]["side_outputs"]) == 1
    assert baseline_path.read_bytes() == original_manifest
    assert [Path(s["path"]).read_bytes() for s in sources] == original_sources
