"""The same editable module chains with range execution enabled and disabled.

The host device adapter checks scheduling and semantic parity, not CUDA speed.
Real CUDA numerical coverage is exercised separately by the CUDA suite.
"""
from copy import deepcopy

import numpy as np
import pytest

from eip.algorithms import Cancelled, RunContext
from eip.module_contracts import Contract
from eip.module_registry import new_node
from eip.sequence_engine import SequenceEngine
from test_gpu_pipeline_0_8_4 import HostDevice, Reader


@pytest.fixture(params=["cpu", "host_gpu"])
def backend(request, monkeypatch):
    monkeypatch.setattr("eip.modes.split_correlation.backend.Backend", HostDevice)
    return request.param


def node(algorithm, **parameters):
    n = new_node(algorithm, shape=(12, 20), frames=67)
    n["parameters"].update(parameters)
    return n


def alignment(interpolation="nearest", dx=0., dy=0., mirror_x=False, mirror_y=False):
    return node("alignment", left=dict(x=0, y=0, width=10, height=12),
                right=dict(x=10, y=0, width=10, height=12), midline=[10., 10.],
                interpolation=interpolation, dx=dx, dy=dy,
                mirror_x=mirror_x, mirror_y=mirror_y)


def make(tmp_path, values, steps, backend, optimize, *, contract=None, resources=None, context=None):
    steps = deepcopy(steps)
    for n in steps:
        n.setdefault("execution", {}).setdefault("backend", "gpu" if backend == "host_gpu" else "cpu")
    return SequenceEngine(Reader(values, contract), steps, context,
                          workdir=tmp_path, optimize=optimize,
                          resources=resources or dict(memory_mb=256, preview_mb=16, cache_mb=0,
                                                      gpu_mb=128, backend="cpu"))


def compare(tmp_path, values, steps, backend, *, contract=None, indices=None, inspect_stages=()):
    original = values.copy()
    engines = [make(tmp_path/str(flag), values, steps, backend, flag, contract=contract)
               for flag in (False, True)]
    try:
        old, new = engines
        assert old.output.record() == new.output.record()
        requested = list(range(old.output.length)) if indices is None else indices
        for index in requested:
            expected = old.get(len(steps), index)
            actual = new.get(len(steps), index)
            assert expected.dtype == actual.dtype
            np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13, equal_nan=True)
            assert old.ref(len(steps), index) == new.ref(len(steps), index)
        # A preceding local request must not poison a full-stage cache lookup.
        for stage in inspect_stages:
            np.testing.assert_allclose(new.get(stage, 0), old.get(stage, 0),
                                       rtol=2e-13, atol=2e-13, equal_nan=True)
            assert new.get(stage, 0).shape == new.contracts[stage].shape
        np.testing.assert_array_equal(values, original)
        return new.spatial.record()
    finally:
        for engine in engines:
            engine.close()


def test_generic_pointwise_multiple_crops_and_label_output(tmp_path, backend):
    values = np.random.default_rng(201).integers(0, 3000, (7, 12, 20), dtype=np.uint16)
    gain = node("gain_offset", gain=1.125, offset=-7.25)
    gain["execution"] = dict(precision="float32")
    steps = [gain, node("curve", curve="square", input_max=4095),
             node("roi_crop", x=2, y=1, width=10, height=8),
             node("threshold", threshold=.2), node("roi_crop", x=2, y=3, width=3, height=2)]
    metrics = compare(tmp_path, values, steps, backend, inspect_stages=(1, 2, 3))
    assert metrics["region_requests"] > 0


def test_color_conversion_and_grayscale_keep_components_and_precision(tmp_path, backend):
    values = np.random.default_rng(202).integers(0, 255, (5, 12, 20, 3), dtype=np.uint8)
    gain = node("gain_offset", gain=1.3, offset=.6)
    gain["execution"] = dict(precision="float64")
    steps = [node("color_convert", mode="bgr"), gain, node("grayscale", method="luma"),
             node("roi_crop", x=3, y=4, width=4, height=3), node("point_extract", x=2, y=1)]
    compare(tmp_path, values, steps, backend, inspect_stages=(1, 2, 3))


def test_grayscale_expansion_and_native_rounding_are_not_fused_away(tmp_path, backend):
    values = np.arange(4*12*20, dtype=np.uint16).reshape(4, 12, 20)
    steps = [node("gain_offset", gain=.5, offset=.25),
             node("gain_offset", gain=2., offset=-.5), node("color_convert", mode="rgb"),
             node("roi_crop", x=2, y=1, width=3, height=4)]
    compare(tmp_path, values, steps, backend)


@pytest.mark.parametrize("interpolation,dx,dy,mirror_x,mirror_y", [
    ("nearest", 2., -1., False, False),
    ("nearest", -.5, .5, True, False),
    ("nearest", -1.25, -.25, False, True),
    ("bilinear", .25, -.5, False, False),
    ("bilinear", -.25, .125, True, True),
])
def test_fixed_geometry_crop_keeps_mapping_and_nan_weights(tmp_path, backend, interpolation, dx, dy, mirror_x, mirror_y):
    values = np.random.default_rng(203).uniform(50., 500., (5, 12, 20))
    values[1, 4, 13] = np.nan
    values[2, 5, 4] = np.inf
    steps = [node("calibration", curve="linear", input_max=4095),
             alignment(interpolation, dx, dy, mirror_x, mirror_y),
             node("roi_crop", x=1, y=2, width=4, height=4),
             node("time_statistics", mode="mean", validity="independent")]
    compare(tmp_path, values, steps, backend, indices=[0, 3, 1, 4, 0, 2], inspect_stages=(1, 2))


@pytest.mark.parametrize("selector", [
    dict(algorithm="point_extract", x=3, y=2),
    dict(algorithm="line_extract", mode="axis", axis="X", index=3, output="main"),
    dict(algorithm="line_extract", mode="axis", axis="Y", index=4, output="main"),
    dict(algorithm="line_extract", mode="segment", x0=1.5, y0=2.5, x1=8.5, y1=9.5,
         interpolation="nearest", samples=17, output="main"),
    dict(algorithm="line_extract", mode="segment", x0=1.25, y0=2., x1=8.5, y1=9.25,
         interpolation="bilinear", samples=19, output="main"),
    dict(algorithm="axis_slice", axis="Y", index=3, output="main"),
    dict(algorithm="axis_slice", axis="X", index=4, output="main"),
])
def test_selection_modules_work_without_correlation_or_alignment(tmp_path, backend, selector):
    values = np.random.default_rng(204).normal(100., 2., (6, 12, 20))
    values[1, 3, 5] = np.nan
    parameters = dict(selector)
    algorithm = parameters.pop("algorithm")
    steps = [node("curve", curve="sqrt", input_max=2048), node(algorithm, **parameters)]
    compare(tmp_path, values, steps, backend, indices=[3, 0, 5, 1, 2, 0])


@pytest.mark.parametrize("x,y,w,h", [(0, 0, 3, 4), (17, 9, 3, 3), (6, 4, 5, 4)])
def test_gaussian_halo_uses_original_image_edges_and_validity(tmp_path, backend, x, y, w, h):
    values = np.random.default_rng(205).normal(size=(4, 12, 20))
    values[:, 1, 1] = np.nan
    values[2, 8, 18] = np.inf
    steps = [node("spatial_smooth", method="gaussian", radius=2, sigma=.7),
             node("roi_crop", x=x, y=y, width=w, height=h)]
    compare(tmp_path, values, steps, backend, inspect_stages=(1,))


@pytest.mark.parametrize("temporal", [
    dict(algorithm="time_sample", step=3, offset=1),
    dict(algorithm="time_smooth", window=5, mode="mean"),
    dict(algorithm="time_smooth", window=5, mode="quantile", quantile=.37),
    dict(algorithm="time_bin", size=4, mode="mean"),
    dict(algorithm="time_bin", size=4, mode="sum"),
])
def test_time_edits_with_spatial_requests_keep_windows_and_random_access(tmp_path, backend, temporal):
    values = np.random.default_rng(206).normal(1e5, .4, (67, 12, 20))
    values[4, 3, 6] = np.nan
    parameters = dict(temporal)
    algorithm = parameters.pop("algorithm")
    steps = [node("time_crop", first=3, last=63), node("curve", curve="linear", input_max=4095),
             node(algorithm, **parameters), node("roi_crop", x=3, y=2, width=5, height=4)]
    metrics = compare(tmp_path, values, steps, backend, indices=[0, 1, 2, 7, 3, 0, 8, 1])
    assert metrics["temporal_peak_bytes"] <= metrics["temporal_limit_bytes"]


def test_batch_entry_matches_per_sample_and_does_not_pollute_logical_axes(tmp_path, backend):
    values = np.random.default_rng(207).normal(100., 2., (35, 12, 20))
    steps = [node("curve", curve="square", input_max=1024),
             node("roi_crop", x=4, y=3, width=5, height=4),
             node("time_statistics", mode="mean", validity="independent")]
    old = make(tmp_path/"old", values, steps, backend, False)
    new = make(tmp_path/"new", values, steps, backend, True)
    try:
        expected = np.stack([old.get(3, j) for j in range(3, 35)])
        actual = new.spatial.compute(3, range(3, 35))
        assert actual.shape == (32,)
        assert new.output.axes == old.output.axes == ""
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        assert new.spatial.record()["blocks"] == 1
    finally:
        old.close()
        new.close()


def test_target_single_side_and_node_precision_survive_domain_fusion(tmp_path, backend):
    values = np.random.default_rng(208).uniform(100., 300., (5, 12, 20))
    gain = node("gain_offset", gain=1.7, offset=2.3)
    gain.update(target="left", execution=dict(precision="float32"))
    steps = [alignment(), gain, node("curve", curve="square", input_max=4095),
             node("roi_crop", x=2, y=3, width=4, height=3), node("select_channel", channel="right")]
    compare(tmp_path, values, steps, backend, inspect_stages=(2,))


def test_distinct_pre_alignment_dark_levels_keep_sensor_midline_rule(tmp_path, backend):
    values = np.random.default_rng(209).uniform(100., 300., (4, 12, 20))
    steps = [node("calibration", curve="linear", input_max=4095, black_left=1., black_right=2.),
             alignment("bilinear", .25, .25), node("point_extract", x=2, y=3)]
    metrics = compare(tmp_path, values, steps, backend)
    assert any("midline" in key for key in metrics["full_barriers"])


def test_data_tap_and_modifier_side_keep_complete_main_and_full_stage(tmp_path, backend):
    values = np.random.default_rng(210).uniform(50., 500., (7, 12, 20))
    steps = [node("gain_offset", gain=1.2), node("data_tap"),
             node("point_extract", x=2, y=3, output="side"),
             node("roi_crop", x=1, y=2, width=4, height=5)]
    old = make(tmp_path/"old", values, steps, backend, False)
    new = make(tmp_path/"new", values, steps, backend, True)
    try:
        assert old.output.record() == new.output.record()
        for j in (0, 5, 1):
            np.testing.assert_array_equal(new.get(4, j), old.get(4, j))
            for side in new.side_outputs():
                np.testing.assert_array_equal(new.get_side_output(side["id"], j),
                                              old.get_side_output(side["id"], j))
        np.testing.assert_array_equal(new.get(1, 0), old.get(1, 0))
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("algorithm", ["box_blur", "spatial_smooth"])
def test_mean_filter_still_rejects_nonfinite_outside_final_roi(tmp_path, backend, algorithm):
    values = np.ones((3, 12, 20))
    values[:, 11, 19] = np.nan
    steps = [node(algorithm, radius=1), node("roi_crop", x=0, y=0, width=2, height=2)]
    for optimize in (False, True):
        engine = make(tmp_path/str(optimize), values, steps, backend, optimize)
        try:
            with pytest.raises(ValueError, match="有限像素"):
                engine.get(2, 0)
        finally:
            engine.close()


def test_lut_still_rejects_out_of_range_code_outside_final_roi(tmp_path, backend):
    lut = tmp_path/"short-lut.npy"
    np.save(lut, np.arange(16, dtype=np.float64))
    values = np.ones((3, 12, 20), dtype=np.uint16)
    values[:, 11, 19] = 25
    steps = [node("calibration", lut_path=str(lut)), node("roi_crop", x=0, y=0, width=2, height=2)]
    for optimize in (False, True):
        engine = make(tmp_path/str(optimize), values, steps, backend, optimize)
        try:
            with pytest.raises(ValueError, match="LUT"):
                engine.get(2, 0)
        finally:
            engine.close()


def test_cancel_before_regional_work_leaves_source_unmodified(tmp_path, backend):
    values = np.ones((3, 12, 20), dtype=np.uint16)
    context = RunContext()
    engine = make(tmp_path, values, [node("curve"), node("roi_crop", width=2, height=2)],
                  backend, True, context=context)
    context.cancelled = lambda: True
    try:
        with pytest.raises(Cancelled):
            engine.get(2, 0)
        assert not engine.source.reader.reads
        np.testing.assert_array_equal(values, np.ones_like(values))
    finally:
        engine.close()


def test_local_cache_and_temporal_state_stay_within_tight_budget(tmp_path, backend):
    values = np.random.default_rng(211).normal(size=(39, 12, 20))
    steps = [node("time_smooth", window=7), node("roi_crop", x=2, y=3, width=3, height=2)]
    resources = dict(memory_mb=128, preview_mb=16, cache_mb=0, gpu_mb=32, backend="cpu")
    engine = make(tmp_path, values, steps, backend, True, resources=resources)
    try:
        # Exercise eviction without an unnecessarily large fixture. These are
        # smaller shares of the admitted task budget, not an extra allocation.
        engine.spatial.raw_limit = values[0].nbytes
        engine.spatial.temporal_limit = 3*2*32
        for index in (0, 1, 2, 15, 0, 23, 1):
            expected = values[index:index+7, 3:5, 2:5].mean(axis=0)
            np.testing.assert_allclose(engine.get(2, index), expected, rtol=2e-13, atol=2e-13)
        other = engine.spatial.compute(1, [0, 1], (5, 6, 3, 2))
        expected = np.stack([values[j:j+7, 6:8, 5:8].mean(axis=0) for j in (0, 1)])
        np.testing.assert_allclose(other, expected, rtol=2e-13, atol=2e-13)
        assert len(engine.spatial.temporal) <= 1
        record = engine.spatial.record()
        assert record["raw_peak_bytes"] <= record["raw_limit_bytes"]
        assert record["temporal_peak_bytes"] <= record["temporal_limit_bytes"]
    finally:
        engine.close()


@pytest.mark.parametrize("mode", ["mean", "quantile"])
def test_nested_temporal_states_are_not_evicted_while_in_use(tmp_path, backend, monkeypatch, mode):
    import eip.window_statistics as windows

    active = set()
    for cls in (windows.RollingReduction, windows.WindowQuantile):
        original_get = cls.get
        original_close = getattr(cls, "close", None)

        def guarded_get(self, *args, _get=original_get, **kwargs):
            active.add(id(self))
            try:
                return _get(self, *args, **kwargs)
            finally:
                active.remove(id(self))

        def guarded_close(self, _close=original_close):
            # Check before touching an mmap so a regression fails clearly,
            # rather than trying to use a closed ring buffer afterwards.
            assert id(self) not in active, "evicted an active temporal state"
            if _close:
                _close(self)

        monkeypatch.setattr(cls, "get", guarded_get)
        monkeypatch.setattr(cls, "close", guarded_close, raising=False)
    values = np.random.default_rng(212).normal(size=(11, 140, 140))
    steps = [node("time_smooth", window=3, mode=mode, quantile=.37) for _ in range(3)]
    steps.append(node("roi_crop", x=0, y=0, width=128, height=128))
    resources = dict(memory_mb=128, preview_mb=16, cache_mb=0, gpu_mb=128, backend="cpu")
    old = make(tmp_path/"old", values, steps, backend, False, resources=resources)
    new = make(tmp_path/"new", values, steps, backend, True, resources=resources)
    try:
        for index in (0, 1, 0):
            np.testing.assert_allclose(new.get(4, index), old.get(4, index),
                                       rtol=2e-13, atol=2e-13, equal_nan=True)
        record = new.spatial.record()
        assert record["temporal_peak_bytes"] <= record["temporal_limit_bytes"]
        assert any("active_nested_temporal_state_budget" in key for key in record["full_barriers"])
        assert not active and not new.spatial.temporal_active
    finally:
        old.close()
        new.close()


def test_long_exact_quantile_uses_canonical_budget_instead_of_new_failure(tmp_path, backend):
    size = 12001
    values = np.arange(size, dtype=np.float64)[:, None, None] + np.arange(6).reshape(1, 2, 3)
    steps = [node("time_smooth", window=size, mode="quantile", quantile=.37),
             node("roi_crop", x=1, y=0, width=1, height=1)]
    resources = dict(memory_mb=128, preview_mb=16, cache_mb=0, gpu_mb=128, backend="cpu")
    old = make(tmp_path/"old", values, steps, backend, False, resources=resources)
    new = make(tmp_path/"new", values, steps, backend, True, resources=resources)
    try:
        expected = old.get(2, 0)
        np.testing.assert_allclose(new.get(2, 0), expected, rtol=0, atol=0)
        assert expected[0, 0] == np.quantile(values[:, 0, 1], .37)
        assert any("regional_quantile_scratch_budget" in key
                   for key in new.spatial.record()["full_barriers"])
    finally:
        old.close()
        new.close()


def test_side_modifier_temporary_metadata_does_not_grow_with_sequence(tmp_path, backend):
    values = np.random.default_rng(213).normal(size=(89, 12, 20))
    steps = [node("curve", curve="square", input_max=2.),
             node("point_extract", x=4, y=3, output="side")]
    engine = make(tmp_path, values, steps, backend, True)
    try:
        side = engine.side_outputs()[0]
        for index in range(len(values)):
            expected = (values[index, 3, 4]/2.)**2
            np.testing.assert_allclose(engine.get_side_output(side["id"], index), expected,
                                       rtol=2e-13, atol=2e-13)
            assert not engine.spatial.evaluated
            assert not engine.spatial.raw and engine.spatial.raw_bytes == 0
        assert engine.spatial.record()["raw_peak_bytes"] <= engine.spatial.raw_limit
    finally:
        engine.close()


def test_lut_full_input_barrier_reuses_decode_with_zero_numeric_cache(tmp_path, backend):
    values = np.random.default_rng(214).integers(1, 1024, (67, 12, 20), dtype=np.uint16)
    path = tmp_path/"linear-lut.npy"
    np.save(path, np.arange(1024, dtype=np.float64)/4095.)
    steps = [node("calibration", lut_path=str(path)), alignment(),
             node("roi_crop", x=2, y=3, width=3, height=2),
             node("time_statistics", mode="mean", validity="independent"),
             node("correlation_map", window=5, stride=1, map_lag=0)]
    engine = make(tmp_path, values, steps, backend, True)
    try:
        for index, actual in engine.iter_outputs(range(engine.output.length)):
            left = (values[index:index+5, 3:5, 2:5]/4095.).mean(axis=(1, 2))
            right = (values[index:index+5, 3:5, 12:15]/4095.).mean(axis=(1, 2))
            expected = np.mean(left*right)/(left.mean()*right.mean())
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        assert len(engine.source.reader.reads) == len(values)
        assert engine.cache.size == 0
        assert engine.spatial.record()["raw_peak_bytes"] <= engine.spatial.raw_limit
    finally:
        engine.close()


def test_spatial_gpu_grayscale_preserves_uint64_rejection(tmp_path, backend):
    values = np.arange(2*12*20*3, dtype=np.uint64).reshape(2, 12, 20, 3) + np.uint64(2**60)
    steps = [node("grayscale", method="luma"), node("roi_crop", x=1, y=2, width=3, height=2)]
    for optimize in (False, True):
        engine = make(tmp_path/str(optimize), values, steps, backend, optimize)
        try:
            if backend == "host_gpu":
                with pytest.raises(ValueError, match="64.*整数"):
                    engine.get(2, 0)
            else:
                expected = np.sum(values[0, 2:4, 1:4].astype(np.float64)*[.2126, .7152, .0722], axis=-1)
                np.testing.assert_array_equal(engine.get(2, 0), expected)
        finally:
            engine.close()
