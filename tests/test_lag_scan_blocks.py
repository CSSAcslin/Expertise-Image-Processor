"""Production Q batches retain paired finite rules and ordered float64 sums."""
from types import SimpleNamespace

import numpy as np
import pytest

from eip.algorithms import Cancelled
from eip.lag_scan_blocks import LagScanBlocks, lag_engine_blocks, required_intervals
from eip.modes.split_correlation.correlation import RollingMoments


def reference(signals, window, center, lags, min_mean=1e-9, min_samples=2):
    backend = SimpleNamespace(xp=np, host=np.asarray)
    p = SimpleNamespace(min_mean=min_mean, min_samples=min_samples)
    base = np.nan_to_num(signals[center], nan=0., posinf=0., neginf=0.)
    first = center-(window-1)//2
    values, counts = [], []
    for lag in lags:
        moments = RollingMoments(base, p, backend)
        for j in range(first, first+window):
            moments.add(signals[j, 0], signals[j+lag, 1])
        value, count = moments.result()
        values.append(value)
        counts.append(count)
    return np.asarray(values), np.asarray(counts)


def fixture_signal(shape=(), *, invalid=True):
    values = np.random.default_rng(5129).uniform(.1, 5, (173, 2, *shape))
    if invalid:
        values[[42, 47, 66, 84], 0] = np.nan
        values[[35, 55, 69, 105], 1] = np.inf
        values[82] = [np.nan, -np.inf] if not shape else np.nan
    return values


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("window", [17, 18])
def test_cpu_signed_random_repeated_Q_blocks_match_ordered_reference(shape, window):
    values = fixture_signal(shape)
    original = values.copy()
    lags = [-27, 0, 19, -5, 19, -1, 12]
    expected, counts = reference(values, window, 82, lags)
    runner = LagScanBlocks(values, window=window, center=82, lags=lags, max_output_block=4)
    for iteration, (start, stop) in enumerate([(3, 7), (0, 4), (4, 6), (0, 1), (3, 7)]):
        actual, samples = runner.compute(start, stop, count=True)
        np.testing.assert_array_equal(actual, expected[start:stop])
        np.testing.assert_array_equal(samples, counts[start:stop])
        if iteration == 0:
            assert runner.record()["state"] == "active"
            assert runner.record()["unique_output_samples"] == 4
    np.testing.assert_array_equal(values, original)
    assert runner.record()["pair_updates"] == (4+4+2+1+4)*window
    assert runner.record()["kernel_calls"] == 0
    assert runner.record()["state"] == "complete"
    assert runner.record()["unique_output_samples"] == 7
    runner.close()
    assert runner.record()["state"] == "complete" and runner.record()["closed"]
    with pytest.raises(RuntimeError):
        runner.compute(0, 1)


@pytest.mark.parametrize("case", ["zero", "invalid", "few", "tiny_fluctuations", "overflow"])
def test_cpu_validity_thresholds_and_nonfinite_results(case):
    values = fixture_signal(invalid=False)
    if case == "zero":
        values[:, 0] = 0
    if case == "invalid":
        values[:, 1] = np.nan
    if case == "few":
        values[::2, 0] = np.inf
    if case == "tiny_fluctuations":
        values = 1.e12+values*.002
    if case == "overflow":
        values[70, 0], values[75, 1] = 1.e308, 1.e308
    lags = [-5, 0, 7]
    with np.errstate(over="ignore", invalid="ignore"):
        expected, counts = reference(values, 31, 82, lags, min_samples=20)
    runner = LagScanBlocks(values, window=31, center=82, lags=lags, min_samples=20)
    actual, samples = runner.compute(0, 3, count=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(samples, counts)
    if case in {"zero", "invalid", "few"}:
        assert np.isnan(actual).all()


def test_rejects_unsupported_work_and_input_before_allocation(monkeypatch):
    import eip.lag_scan_blocks as blocks
    values = np.ones((173, 2))
    spec = dict(window=17, center=82, lags=[-30, 0, 19])
    for signals in (values.astype(np.float32), values[::2], values[:, :1]):
        with pytest.raises(ValueError):
            LagScanBlocks(signals, **spec)
    for overrides in (dict(window=0), dict(center=2), dict(lags=[200]), dict(lags=[]),
                      dict(lags=[True]), dict(lags=[1.5]), dict(min_mean=0),
                      dict(min_samples=0), dict(max_output_block=257), dict(window=65537)):
        with pytest.raises(ValueError):
            LagScanBlocks(values, **dict(spec, **overrides))
    with pytest.raises(ValueError, match="components"):
        LagScanBlocks(np.ones((3, 2, 4097)), window=1, center=1, lags=[0])
    runner = LagScanBlocks(np.ones((17, 2, 4096)), window=17, center=8, lags=[0], max_output_block=256)
    assert runner.max_output_block == 1
    monkeypatch.setattr(blocks, "MAX_LAUNCH_PAIR_UPDATES", 17*4096-1)
    with pytest.raises(ValueError, match="pair work"):
        LagScanBlocks(np.ones((17, 2, 4096)), window=17, center=8, lags=[0], max_output_block=256)


def test_invalid_and_cancelled_blocks_do_not_publish_metrics():
    calls = [0]
    def check():
        calls[0] += 1
        if calls[0] == 3:
            raise Cancelled()
    values = fixture_signal()
    runner = LagScanBlocks(values, window=17, center=82, lags=[-5, 0, 9],
                           max_output_block=2, check=check)
    for first, stop in [(0, 0), (-1, 1), (2, 4), (0, 3), (True, 2)]:
        with pytest.raises(ValueError):
            runner.compute(first, stop)
    with pytest.raises(Cancelled):
        runner.compute(0, 2)
    assert runner.metrics["output_blocks"] == 0
    actual, count = runner.compute(0, 2, count=True)
    expected, samples = reference(values, 17, 82, [-5, 0])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(count, samples)


def test_required_ranges_keep_sparse_gaps_and_even_window_rule():
    assert required_intervals(10326, 10001, range(0, 30001, 10)) == [[5326, 45327]]
    assert required_intervals(10326, 10001, [0, 30000]) == [[5326, 15327], [35326, 45327]]
    assert required_intervals(9, 4, [-4, 3]) == [[4, 15]]


def engine_fixture(tmp_path, *, kind="point", gpu=False, trial=False, point_backend=None):
    from test_spatial_execution import node, alignment, make
    values = np.random.default_rng(777).uniform(1, 19, (601, 12, 20))
    selector = node("point_extract", x=3, y=4) if kind == "point" else node("roi_crop", x=2, y=3, width=2, height=3)
    steps = [alignment(), selector, node("correlation_curve", window=31, lag_list=[-30, 0, 51], center=301)]
    if gpu:
        for n in steps:
            n["execution"] = dict(backend="gpu", precision="native", memory_mb=None)
    if point_backend:
        steps[1]["execution"]["backend"] = point_backend
    engine = make(tmp_path, values, steps, "cpu", True)
    engine._single = trial
    return engine


@pytest.mark.parametrize("kind", ["point", "roi"])
@pytest.mark.parametrize("trial", [False, True])
def test_factory_shares_checkpoint_prepares_only_required_union_and_matches_get(tmp_path, kind, trial):
    engine = engine_fixture(tmp_path, kind=kind, trial=trial)
    runner = None
    try:
        runner = lag_engine_blocks(engine, 3)
        assert runner is not None
        assert runner.signals is engine.main_store.values
        assert not engine.main_store.ready.all()
        assert len(engine.source.reader.reads) < 200
        assert len(set(engine.source.reader.reads)) == len(engine.source.reader.reads)
        actual, _ = runner.compute(0, engine.output.length)
        expected = np.asarray([engine.get(3, j) for j in range(engine.output.length)])
        np.testing.assert_array_equal(actual, expected)
        row = engine.execution_metrics["lag_output_blocks"]["3"]
        assert row["required_intervals"] == [[255, 316], [336, 367]]
        assert row["required_samples"] == 92
        assert row["additional_reserved_bytes"] < engine.resource_plan[2]["limit_bytes"]
    finally:
        if runner:
            runner.close()
        engine.close()


@pytest.mark.parametrize("case,reason", [
    ("disk", "direct_input_has_no_native_resident_checkpoint"),
    ("dtype", "checkpoint_conversion_is_not_implicit"),
    ("budget", "insufficient_budget_for_state_and_one_output"),
    ("inspection", "local_inspection_uses_demand_execution"),
    ("map", "requires_fixed_center_Q_outputs"),
    ("work", "window_pair_work_exceeds_bounded_block_limit"),
])
def test_factory_rejects_before_input_work(tmp_path, monkeypatch, case, reason):
    import eip.lag_scan_blocks as blocks
    engine = engine_fixture(tmp_path)
    try:
        if case == "disk":
            engine.main_store.mode = "disk"
        elif case == "dtype":
            engine.main_store.dtype = np.dtype("float32")
        elif case == "budget":
            engine.budget = 0
        elif case == "inspection":
            engine.inspection = True
        elif case == "map":
            engine.operations[-1]["parameters"]["analysis"] = "map"
        elif case == "work":
            monkeypatch.setattr(blocks, "MAX_SERIAL_PAIR_UPDATES", 30)
        assert lag_engine_blocks(engine, 3) is None
        assert not engine.source.reader.reads
        assert engine.execution_metrics["lag_output_blocks"]["3"]["reason"] == reason
    finally:
        engine.close()


def test_factory_propagates_actual_preparation_error_without_replay(tmp_path):
    engine = engine_fixture(tmp_path)
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise ValueError("real input fault")
    engine._stored_range = fail
    try:
        with pytest.raises(ValueError, match="real input fault"):
            lag_engine_blocks(engine, 3)
        assert len(calls) == 1
        assert engine.execution_metrics["lag_output_blocks"]["3"]["state"] == "failed"
    finally:
        engine.close()


@pytest.fixture(scope="module")
def cuda():
    from eip.modes.split_correlation.backend import devices
    available = devices()
    if not available["gpu"]:
        pytest.skip(available["reason"])
    import cupy as cp
    return cp


@pytest.mark.parametrize("shape,window", [((), 31), ((3,), 32), ((2, 3), 17)])
def test_cuda_signed_repeated_Qs_match_cpu_and_counts(cuda, shape, window):
    values = fixture_signal(shape)
    spec = dict(window=window, center=82, lags=[-27, 0, 19, -5, 19, -1, 12], max_output_block=4)
    expected, counts = reference(values, window, 82, spec["lags"])
    backend = SimpleNamespace(xp=cuda)
    runner = LagScanBlocks(cuda.asarray(values), backend, **spec)
    for start, stop in [(3, 7), (0, 4), (4, 6), (0, 1)]:
        actual, samples = runner.compute(start, stop, count=True)
        np.testing.assert_array_equal(cuda.asnumpy(actual), expected[start:stop])
        np.testing.assert_array_equal(cuda.asnumpy(samples), counts[start:stop])
    np.testing.assert_array_equal(cuda.asnumpy(runner.signals), values)
    assert runner.metrics["kernel_calls"] == 4
    runner.close()


def test_cuda_factory_and_timer_keep_native_checkpoint_and_bounded_Qs(tmp_path, cuda):
    from eip.gpu_timing import GpuEventTimer
    from eip.lag_scan_blocks import KERNEL_NAME
    engine = engine_fixture(tmp_path, kind="roi", gpu=True, point_backend="auto")
    runner = None
    try:
        backend = engine.backends[2]
        assert backend is not engine.main_store.backend
        timer = GpuEventTimer(cuda, initial_calls=2, sample_every=1)
        backend.gpu_timer = timer
        runner = lag_engine_blocks(engine, 3)
        assert runner.signals is engine.main_store.values
        actual, count = runner.compute(0, 3, count=True)
        repeated, _ = runner.compute(0, 3, count=True)
        expected = np.asarray([engine.get(3, j) for j in range(3)])
        np.testing.assert_array_equal(cuda.asnumpy(actual), expected)
        np.testing.assert_array_equal(cuda.asnumpy(repeated), expected)
        assert (cuda.asnumpy(count) == 31).all()
        timer.finish(raise_errors=True)
        row = timer.snapshot()["kernels"][KERNEL_NAME]
        assert row["calls"] == 2 and row["timed_calls"] == 1
        assert not engine.main_store.ready.all()
    finally:
        if runner:
            runner.close()
        engine.close()
