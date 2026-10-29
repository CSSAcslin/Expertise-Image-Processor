"""Real CUDA coverage of every visible module, including side outputs.

The ordinary suite skips these tests without CUDA. --require-cuda makes missing
or unusable CUDA a preflight failure rather than a successful CPU-only run.
"""
import numpy as np
import pytest

from eip.module_registry import catalog
from test_gpu_pipeline_0_8_4 import make, node


CASES = [
    ("threshold", dict(threshold=120)),
    ("curve", dict(curve="gamma", gamma=1.8, input_max=400)),
    ("calibration", dict(curve="linear", input_max=400, black_left=2, black_right=3)),
    ("alignment", dict(dx=.25, dy=.2, interpolation="bilinear")),
    ("select_channel", dict(channel="right")),
    ("time_crop", dict(first=2, last=17)),
    ("time_sample", dict(step=3, offset=1)),
    ("time_smooth", dict(window=5, mode="mean")),
    ("time_smooth", dict(window=5, mode="quantile", quantile=.73)),
    ("time_bin", dict(size=3, mode="sum")),
    ("paired_trace", dict(selection="region", width=2, height=2, mode="variance")),
    ("spatial_smooth", dict(method="mean", radius=1)),
    ("spatial_smooth", dict(method="gaussian", radius=2, sigma=.7)),
    ("gain_offset", dict(gain=1.3, offset=-5)),
    ("time_statistics", dict(mode="variance")),
    ("global_statistics", dict(mode="quantile", quantile=.73)),
    ("roi_crop", dict(x=1, y=1, width=3, height=2)),
    ("fixed_normalize", dict(scale=3.5)),
    ("parameter_binding", dict(value=2.5)),
    ("grayscale", dict(method="luma")),
    ("color_convert", dict(mode="bgr")),
    ("point_extract", dict(x=2, y=1)),
    ("line_extract", dict(mode="segment", x0=0, y0=0, x1=6, y1=3,
                          samples=13, interpolation="bilinear", output="side")),
    ("axis_slice", dict(axis="T", index=3, output="side")),
    ("distribution", dict(minimum=None, maximum=None, bins=9, output="side")),
    ("data_tap", dict(destination="both")),
    ("correlation_map", dict(window=5, stride=2, map_lag=-2)),
    ("correlation_curve", dict(window=5, lag_list=[-2, 0, 3], center=11)),
    ("correlation_window", dict(window_list=[2, 3, 7, 12], lag=-2, center=11)),
]


@pytest.fixture(scope="module")
def real_cuda():
    from eip.modes.split_correlation.backend import devices
    availability = devices()
    if not availability["gpu"]:
        pytest.skip(availability["reason"])
    import cupy
    return cupy


def test_cuda_cases_cover_visible_catalog():
    assert {a for a, _ in CASES} == {a for a, s in catalog().items() if not s.get("legacy")}


@pytest.mark.parametrize("algorithm,parameters", CASES,
                         ids=[f"{a}-{i}" for i, (a, _) in enumerate(CASES)])
def test_real_cuda_module_and_side_outputs_match_cpu(tmp_path, real_cuda, algorithm, parameters):
    values = np.random.default_rng(2710).uniform(10, 350, (21, 4, 8))
    if algorithm == "grayscale":
        values = np.stack([values, values * .7, values * 1.2], axis=-1)
    if algorithm in {"time_statistics", "global_statistics", "paired_trace", "distribution"}:
        values[3, 1, 2] = np.nan
        values[5, 2, 6] = np.inf
    paired = algorithm in {"select_channel", "paired_trace", "correlation_map",
                           "correlation_curve", "correlation_window"}
    nodes = ([node("alignment")] if paired else []) + [node(algorithm, **parameters)]
    if algorithm == "calibration":
        # Distinct channel dark levels require an explicit downstream split.
        nodes.append(node("alignment"))
    cpu = make(tmp_path / "cpu", values, nodes, False)
    gpu = None
    try:
        gpu = make(tmp_path / "gpu", values, nodes, True)
        assert cpu.output.record() == gpu.output.record()
        assert all(d["actual"] == "gpu" for d in gpu.device_records.values())
        # Visit every result, then seek backwards to exercise retained states.
        indices = [*range(cpu.output.length), cpu.output.length - 1, 0]
        for index in indices:
            expected, actual = cpu.get(len(nodes), index), gpu.get(len(nodes), index)
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11, equal_nan=True)
        sides_cpu, sides_gpu = cpu.side_outputs(), gpu.side_outputs()
        assert len(sides_cpu) == len(sides_gpu)
        for left, right in zip(sides_cpu, sides_gpu):
            assert left["contract"].record() == right["contract"].record()
            for index in range(left["contract"].length):
                expected = cpu.get_side_output(left["id"], index)
                actual = gpu.get_side_output(right["id"], index)
                assert expected.dtype == actual.dtype
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11, equal_nan=True)
        # Verify that this exercised CuPy arrays, not the NumPy test adapter.
        value = gpu.device.get(len(nodes), 0, gpu.backends[len(nodes)-1])
        assert isinstance(value, real_cuda.ndarray)
        assert gpu.device.record()["cache_peak_bytes"] <= gpu.device.record()["cache_budget_bytes"]
    finally:
        cpu.close()
        if gpu is not None:
            gpu.close()
