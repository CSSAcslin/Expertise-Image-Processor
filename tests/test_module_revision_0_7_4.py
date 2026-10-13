"""Numerical and compatibility coverage for the 0.7.4 module changes only."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from eip.algorithms import Cancelled, RunContext
from eip.module_contracts import Contract, compile_flow
from eip.module_registry import catalog, migrate_workbench_nodes, new_node
from eip.sequence_engine import SequenceEngine
from test_axis_modules import Reader


def node(algorithm, precision="native", **parameters):
    n = new_node(algorithm)
    n["parameters"].update(parameters)
    n["execution"] = dict(precision=precision)
    return n


def engine(tmp_path, values, nodes, contract=None, **options):
    reader = Reader(values)
    if contract is not None:
        reader.meta = replace(reader.meta, contract=contract.record())
    return SequenceEngine(reader, nodes, workdir=tmp_path, **options)


def reference(values, mode, q=.25):
    finite = np.asarray(values)[np.isfinite(values)]
    if mode == "count":
        return len(finite)
    if not len(finite):
        return np.nan
    return np.quantile(finite, q) if mode == "quantile" else getattr(np, "var" if mode == "variance" else mode)(finite)


def test_visible_catalog_merges_and_reclassifies_modules():
    specs = catalog()
    for old in ("intensity", "affine", "box_blur", "gaussian", "region_reduce", "sequence_statistics"):
        assert specs[old]["legacy"]
    for key in ("gain_offset", "spatial_smooth", "curve", "fixed_normalize", "time_statistics", "global_statistics"):
        assert not specs[key]["legacy"] and specs[key]["category"] == "modifier"
        assert specs[key]["formula"]["latex"]
    for key in ("parameter_binding", "point_extract", "line_extract", "axis_slice"):
        assert specs[key]["category"] == "intermediate"


@pytest.mark.parametrize("precision", ["native", "float32", "float64"])
def test_merged_gain_precision_controls_quantization(tmp_path, precision):
    data = np.array([[[0, 1, 255]]], dtype=np.uint8)
    e = engine(tmp_path, data, [node("gain_offset", precision=precision, gain=.5, offset=.25)])
    try:
        expected = [0, 1, 128] if precision == "native" else [.25, .75, 127.75]
        np.testing.assert_array_equal(e.get(1, 0), [expected])
        assert e.output.dtype == ("uint8" if precision == "native" else precision)
        assert str(e.get(1, 0).dtype) == e.output.dtype
    finally:
        e.close()


@pytest.mark.parametrize("method", ["mean", "gaussian"])
def test_merged_spatial_filter_has_floating_precision_before_quantization(tmp_path, method):
    data = np.zeros((1, 3, 3), dtype=np.uint16)
    data[0, 1, 1] = 9
    e = engine(tmp_path, data, [node("spatial_smooth", method=method, radius=1, sigma=1., precision="float64")])
    try:
        center = 1. if method == "mean" else 9 / (1 + 2*np.exp(-.5))**2
        assert e.get(1, 0)[1, 1] == pytest.approx(center)
        assert e.output.axes == "YX" and e.output.dtype == "float64"
    finally:
        e.close()


@pytest.mark.parametrize("mode", ["mean", "sum", "min", "max", "variance", "std", "quantile", "count"])
def test_time_and_global_statistics_use_distinct_real_axes(tmp_path, mode):
    data = np.array([[[1., 2., np.nan], [4., np.inf, 8.]], [[3., 7., 9.], [11., 13., 17.]]])
    time = engine(tmp_path, data, [node("time_statistics", mode=mode, quantile=.25)])
    global_ = engine(tmp_path, data, [node("global_statistics", mode=mode, quantile=.25)])
    try:
        assert time.output.record()["full_axes"] == "T" and time.output.length == 2
        assert global_.output.record()["full_axes"] == "N" and global_.output.length == 1
        for i in range(2):
            np.testing.assert_allclose(time.get(1, i), reference(data[i], mode))
        np.testing.assert_allclose(global_.get(1, 0), reference(data, mode))
        assert global_.ref(1, 0)["time"] is None
        assert global_.ref(1, 0)["contribution"]["upstream_stop"] == 2
        assert global_.output.dtype == ("uint64" if mode == "count" else "float64")
    finally:
        time.close()
        global_.close()


@pytest.mark.parametrize("mode", ["min", "max", "variance", "quantile", "count"])
def test_all_invalid_statistics_have_explicit_empty_support(tmp_path, mode):
    data = np.full((2, 2, 3), np.nan)
    for algorithm in ("time_statistics", "global_statistics"):
        e = engine(tmp_path, data, [node(algorithm, mode=mode)])
        try:
            value = e.get(1, 0)
            assert value == 0 if mode == "count" else np.isnan(value)
        finally:
            e.close()


@pytest.mark.parametrize("scope", ["samples", "space_time", "all"])
@pytest.mark.parametrize("mode", ["mean", "variance", "quantile"])
def test_global_scope_and_exact_quantile_with_unequal_support(tmp_path, scope, mode):
    data = np.arange(3*2*2*3, dtype=float).reshape(3, 2, 2, 3)
    data[0, 0, 0, :2] = np.nan
    data[:, 1, 1, 2] = np.nan
    contract = Contract("PYX", (2, 2, 3), 3)
    e = engine(tmp_path, data, [node("global_statistics", scope=scope, mode=mode, quantile=.25)], contract)
    try:
        result = e.get(1, 0)
        if scope == "samples":
            assert e.output.axes == "PYX"
            for p, y, x in np.ndindex(2, 2, 3):
                np.testing.assert_allclose(result[p, y, x], reference(data[:, p, y, x], mode), equal_nan=True)
        elif scope == "space_time":
            assert e.output.axes == "P"
            np.testing.assert_allclose(result, [reference(data[:, p], mode) for p in range(2)])
        else:
            assert e.output.axes == ""
            np.testing.assert_allclose(result, reference(data, mode))
        assert not list(tmp_path.glob("quantile-*")), "temporary quantile files must close and disappear"
    finally:
        e.close()


def test_stats_side_output_is_passthrough_and_can_feed_scalar_parameter(tmp_path):
    data = np.arange(1, 13, dtype=np.float64).reshape(2, 2, 3)
    stats = node("global_statistics", mode="max", output="side")
    gain = node("gain_offset", gain={"ref": dict(node_id=stats["id"], port="value")})
    e = engine(tmp_path, data, [stats, gain])
    try:
        side = e.side_outputs()[0]
        assert side["contract"].record()["full_axes"] == "N"
        assert e.get_side_output(side["id"], 0) == 12
        np.testing.assert_array_equal(e.get(1, 1), data[1])
        np.testing.assert_array_equal(e.get(2, 1), data[1]*12)
        assert e.side_ref(side["id"], 0)["time"] is None
        assert e.plan["resolved_bindings"][gain["id"]]["gain"]["value"] == 12
    finally:
        e.close()
    matrix_stats = node("global_statistics", scope="samples", output="side", mode="max")
    matrix_gain = node("gain_offset", gain={"ref": dict(node_id=matrix_stats["id"], port="value")})
    with pytest.raises(ValueError, match="标量"):
        engine(tmp_path, data, [matrix_stats, matrix_gain])


def test_global_uses_edited_range_and_preserves_q_axis_meaning(tmp_path):
    data = np.arange(6*2*3, dtype=float).reshape(6, 2, 3)
    e = engine(tmp_path, data, [node("time_crop", first=2, last=5), node("time_sample", step=2), node("global_statistics", mode="max")])
    try:
        assert e.get(3, 0) == data[3].max()
        assert e.ref(3, 0)["contribution"]["upstream_stop"] == 2
    finally:
        e.close()
    e = engine(tmp_path, data, [node("global_statistics", mode="mean")], Contract("YX", (2, 3), 6, sequence_axis="Q"))
    try:
        assert e.get(1, 0) == data.mean()
        assert e.ref(1, 0)["statistic"]["input_axes"] == "QYX"
    finally:
        e.close()


def test_time_statistics_accepts_scalar_signal_without_fake_spatial_axes(tmp_path):
    values = np.array([2., np.nan, 7.])
    e = engine(tmp_path, values, [node("time_statistics", mode="variance")], Contract("", (), 3))
    try:
        assert e.output.record()["full_axes"] == "T"
        assert e.get(1, 0) == 0 and np.isnan(e.get(1, 1))
    finally:
        e.close()


def test_global_per_channel_std_can_be_used_as_fixed_scale(tmp_path):
    values = np.arange(3*2*2*3, dtype=float).reshape(3, 2, 2, 3)+1
    stats = node("global_statistics", scope="space_time", mode="std", output="side")
    normalize = node("fixed_normalize", scale={"ref": dict(node_id=stats["id"], port="value")})
    e = engine(tmp_path, values, [stats, normalize], Contract("PYX", (2, 2, 3), 3))
    try:
        scales = np.array([values[:, p].std() for p in range(2)])[:, None, None]
        np.testing.assert_allclose(e.get(2, 1), values[1]/scales)
    finally:
        e.close()


def test_global_after_correlation_trial_reads_complete_upstream(tmp_path):
    data = np.arange(26, dtype=float).reshape(13, 1, 2)+1
    nodes = [node("alignment", left=dict(x=0,y=0,width=1,height=1), right=dict(x=1,y=0,width=1,height=1), midline=[1,1]),
             node("correlation_map", window=3, stride=1), node("global_statistics")]
    e = engine(tmp_path, data, nodes, single=True)
    try:
        expected = []
        for center in range(1, 12):
            left, right = data[center-1:center+2, 0, 0], data[center-1:center+2, 0, 1]
            expected.append(np.mean(left*right)/(left.mean()*right.mean()))
        assert e.contracts[2].length == 11
        assert e.get(3, 0) == pytest.approx(np.mean(expected))
    finally:
        e.close()


def test_merge_migration_preserves_legacy_quantization_and_fixed_scale(tmp_path):
    old = [node("affine", gain=.5, offset=.25), node("gaussian"), node("intensity", gain=.5, precision="float32")]
    original = deepcopy(old)
    migrated, report = migrate_workbench_nodes(old)
    assert old == original and report
    assert [n["algorithm"] for n in migrated] == ["gain_offset", "spatial_smooth", "intensity"]
    assert migrated[0]["execution"]["precision"] == "float64"
    assert migrated[1]["execution"]["precision"] == "float64"
    data = np.arange(18, dtype=np.uint16).reshape(2, 3, 3)
    first, second = engine(tmp_path, data, old), engine(tmp_path, data, migrated)
    try:
        np.testing.assert_array_equal(first.get(3, 1), second.get(3, 1))
    finally:
        first.close()
        second.close()


def test_quantile_cancellation_and_disk_failure_cleanup(tmp_path, monkeypatch):
    data = np.ones((5, 2, 3))
    e = engine(tmp_path, data, [node("global_statistics", mode="quantile")])
    e.context.cancelled = lambda: True
    try:
        with pytest.raises(Cancelled):
            e.get(1, 0)
    finally:
        e.close()
    e = engine(tmp_path, data, [node("global_statistics", mode="quantile")])
    monkeypatch.setattr("eip.module_statistics.shutil.disk_usage", lambda _: SimpleNamespace(free=1))
    try:
        with pytest.raises(ValueError, match="磁盘空间不足"):
            e.get(1, 0)
        assert not list(tmp_path.glob("quantile-*"))
    finally:
        e.close()
