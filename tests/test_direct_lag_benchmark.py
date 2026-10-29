"""A shortened Q diagnostic must retain its saved centre and device boundary."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "tools/benchmark_direct_lag_scan.py"
spec = importlib.util.spec_from_file_location("direct_lag_benchmark", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("policy", ["cpu", "mixed"])
def test_sampled_q_keeps_window_fixed_center_and_explicit_cpu_boundary(tmp_path, policy):
    baseline = dict(spec=dict(resources=dict(backend="gpu", memory_mb=1024), steps=[
        dict(id="align", algorithm="alignment", execution=dict(backend="cpu")),
        dict(id="curve", algorithm="correlation_curve", parameters=dict(window=10001, center=None), execution=dict(backend="inherit")),
        dict(id="off", algorithm="correlation_window", enabled=False),
    ]), results=[dict(mappings=dict(curve=dict(lags=[-10, 0, 10, 20], centers=dict(start=10326))))])
    unchanged = deepcopy(baseline)
    payload = benchmark.diagnostic_payload(baseline, tmp_path, policy, [0, 2, 3])
    assert baseline == unchanged
    assert payload["steps"][1]["parameters"] == dict(window=10001, center=10327, lag_list=[-10, 10, 20])
    assert payload["steps"][0]["execution"]["backend"] == "cpu"
    assert payload["steps"][1]["execution"]["backend"] == ("cpu" if policy == "cpu" else "inherit")
    assert payload["steps"][2] == unchanged["spec"]["steps"][2]
    assert payload["resources"]["backend"] == ("cpu" if policy == "cpu" else "gpu")


def test_full_validation_preserves_all_original_lags_and_automatic_center(tmp_path):
    baseline=dict(spec=dict(id="original",resources=dict(backend="gpu"),steps=[
        dict(id="align",algorithm="alignment",execution=dict(backend="cpu")),
        dict(id="curve",algorithm="correlation_curve",parameters=dict(window=10001,center=None,lag_min=0,lag_max=30000,lag_step=10,lag_list=None)),
    ]))
    unchanged=deepcopy(baseline)
    payload=benchmark.full_application_payload(baseline,tmp_path,"mixed")
    assert baseline==unchanged
    assert payload["steps"]==unchanged["spec"]["steps"]
    assert payload["id"]!="original"
