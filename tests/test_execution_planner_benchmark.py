"""The general benchmark must preserve timing and validation boundaries."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1]/"tools"/"benchmark_execution_planner.py"
spec = importlib.util.spec_from_file_location("planner_benchmark_tool", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def artifacts(folder, values, *, axis="T", references=None):
    folder.mkdir()
    np.save(folder/"values.npy", values, allow_pickle=False)
    benchmark.write_json(folder/"summary.json", dict(source_sha256="same-synthetic-source",
                          contract=dict(axes="", sequence_axis=axis), output_shape=list(values.shape), dtype=str(values.dtype)))
    benchmark.write_json(folder/"references.json", references or [dict(index=0, time=0., pts=0)])


def test_case_ids_and_device_policies_do_not_change_source_references():
    for case in benchmark.CASES:
        cpu = benchmark.case_nodes(case, 24, 32, "cpu")
        gpu = benchmark.case_nodes(case, 24, 32, "gpu")
        assert [node["id"] for node in cpu] == [node["id"] for node in gpu]
        assert [node["parameters"] for node in cpu] == [node["parameters"] for node in gpu]
        assert all(node["execution"]["backend"] == "cpu" for node in cpu)
        assert all(node["execution"]["backend"] == "gpu" for node in gpu)
        assert all(node["algorithm"] not in {"correlation", "correlation_map", "correlation_curve", "correlation_window"}
                   for node in cpu)


@pytest.mark.parametrize("change", ["finite_mask", "signed_infinity", "axis", "reference", "dtype"])
def test_complete_validation_rejects_semantic_and_nonfinite_changes(tmp_path, change):
    original = np.array([1., np.nan, np.inf, -np.inf])
    actual = original.copy()
    if change == "finite_mask":
        actual[1] = 0.
    elif change == "signed_infinity":
        actual[2] = -np.inf
    elif change == "dtype":
        actual = actual.astype(np.float32)
    artifacts(tmp_path/"reference", original)
    artifacts(tmp_path/"actual", actual, axis="Q" if change == "axis" else "T",
              references=[dict(index=1, time=0., pts=0)] if change == "reference" else None)
    with pytest.raises(AssertionError):
        benchmark.compare_trial(tmp_path/"reference", tmp_path/"actual")


def test_all_valid_values_are_checked_beyond_validation_chunk(tmp_path):
    original = np.arange(131080, dtype=np.float64)
    actual = original.copy()
    artifacts(tmp_path/"reference", original)
    artifacts(tmp_path/"actual", actual)
    assert benchmark.compare_trial(tmp_path/"reference", tmp_path/"actual")["checked_values"] == len(original)
    actual[-1] += .1
    np.save(tmp_path/"actual"/"values.npy", actual, allow_pickle=False)
    with pytest.raises(AssertionError):
        benchmark.compare_trial(tmp_path/"reference", tmp_path/"actual")
