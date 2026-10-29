"""Benchmark policy and validation must not hide scientific differences."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "tools" / "benchmark_direct_flow.py"
spec = importlib.util.spec_from_file_location("direct_benchmark_tool", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("policy,expected", [("cpu", ["cpu", "cpu"]), ("mixed", ["inherit", "cpu"]), ("gpu", ["gpu", "gpu"])])
def test_explicit_device_boundary_preserved_or_explicitly_changed(tmp_path, policy, expected):
    baseline = dict(spec=dict(id="original", output_dir="original", resources=dict(backend="cpu", memory_mb=18320),
                              steps=[dict(algorithm="calibration", execution=dict(backend="inherit")),
                                     dict(algorithm="alignment", execution=dict(backend="cpu"))]))
    original = deepcopy(baseline)
    result = benchmark.payload_for(baseline, tmp_path, policy)
    assert baseline == original
    assert [node["execution"]["backend"] for node in result["steps"]] == expected
    assert result["resources"]["backend"] == ("cpu" if policy == "cpu" else "gpu")
    assert result["id"] != baseline["spec"]["id"]
    assert result["resources"]["memory_mb"] == 18320


def test_csv_value_tolerance_does_not_hide_coordinate_changes(tmp_path):
    left, right = tmp_path / "expected.csv", tmp_path / "actual.csv"
    left.write_text("output_index,source_frame,value_1\n0,500,1.1\n", encoding="utf-8")
    right.write_text("output_index,source_frame,value_1\n0,500,1.10000000000001\n", encoding="utf-8")
    assert benchmark.compare_csv(left, right, rtol=2e-13, atol=2e-13)["passed"]
    right.write_text("output_index,source_frame,value_1\n0,501,1.1\n", encoding="utf-8")
    result = benchmark.compare_csv(left, right, rtol=2e-13, atol=2e-13)
    assert not result["passed"]
    assert result["first_mismatches"] == [dict(row=0, field="source_frame")]
