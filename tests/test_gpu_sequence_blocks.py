"""Real CUDA sequence blocks compared with the established rolling estimator."""
from types import SimpleNamespace

import numpy as np
import pytest

from eip.gpu_sequence_blocks import GpuCorrelationBlocks, KERNEL_NAME, engine_blocks
from eip.modes.split_correlation.correlation import RollingMoments, direct_g2


@pytest.fixture(scope="module")
def cuda():
    from eip.modes.split_correlation.backend import devices
    availability = devices()
    if not availability["gpu"]:
        pytest.skip(availability["reason"])
    import cupy as cp
    return cp


def reference(signals, *, window, lag, first_center, stride, output_count,
              min_mean=1e-9, min_samples=2):
    backend = SimpleNamespace(xp=np, host=np.asarray)
    config = SimpleNamespace(min_mean=min_mean, min_samples=min_samples)
    values, counts = [], []
    moments = None
    old_start = old_stop = None
    for ordinal in range(output_count):
        center = first_center+ordinal*stride
        start = center-(window-1)//2
        stop = start+window
        if moments is None or stride >= window or ordinal % 256 == 0:
            baseline = np.nan_to_num(signals[center], nan=0., posinf=0., neginf=0.)
            moments = RollingMoments(baseline, config, backend)
            changes = [(start, stop, 1)]
        else:
            changes = [(old_start, start, -1), (old_stop, stop, 1)]
        for first, last, sign in changes:
            for index in range(first, last):
                moments.add(signals[index, 0], signals[index+lag, 1], sign)
        result, count = moments.result()
        values.append(result.copy())
        counts.append(count.copy())
        old_start, old_stop = start, stop
    return np.asarray(values), np.asarray(counts)


def schedule(length, window, lag, stride):
    return dict(window=window, lag=lag, stride=stride,
                first_center=(window-1)//2+max(0, -lag),
                output_count=(length-window-abs(lag))//stride+1)


def collect(runner, cuda, chunks, *, count=True):
    values, counts = [], []
    part = 0
    while runner.next_index < runner.output_count:
        first = runner.next_index
        stop = min(first+chunks[part % len(chunks)], runner.output_count)
        actual, samples = runner.compute(first, stop, count=count)
        values.append(cuda.asnumpy(actual))
        if count:
            counts.append(cuda.asnumpy(samples))
        else:
            assert samples is None
        part += 1
    return np.concatenate(values), np.concatenate(counts) if count else None


def test_requires_real_device_array_without_optional_cuda():
    with pytest.raises(ValueError, match="real CuPy"):
        GpuCorrelationBlocks(np.ones((10, 2)), SimpleNamespace(xp=np),
                             **schedule(10, 3, 0, 1))


@pytest.mark.parametrize("lag,stride,window", [(-5, 1, 17), (0, 2, 18),
                                             (7, 3, 23), (-3, 7, 7),
                                             (4, 11, 5)])
def test_signed_lag_stride_and_global_rebuild_match_reference(cuda, lag, stride, window):
    rng = np.random.default_rng(2940)
    signals = rng.uniform(1, 19, (1901, 2, 2, 3))
    signals[7, 0, 0, 0] = np.nan
    signals[29, 1, 1, 2] = np.inf
    signals[256, 0, 1, 1] = -np.inf
    spec = schedule(len(signals), window, lag, stride)
    expected, counts = reference(signals, **spec)
    resident = cuda.asarray(signals)
    runner = GpuCorrelationBlocks(resident, SimpleNamespace(xp=cuda), **spec)
    actual, actual_counts = collect(runner, cuda, [63, 129, 1, 17, 256])
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13, equal_nan=True)
    np.testing.assert_array_equal(actual_counts, counts)
    np.testing.assert_array_equal(cuda.asnumpy(resident), signals)
    record = runner.record()
    assert record["state_bytes"] == 6*6*8
    assert record["largest_output_block"] <= 256
    assert record["output_samples"] == spec["output_count"]
    rebuilds = ((spec["output_count"]-1)//256+1 if stride < window else spec["output_count"])
    assert record["periodic_rebuilds"] == rebuilds
    assert record["pair_updates"] == rebuilds*window+(spec["output_count"]-rebuilds)*2*stride
    runner.close()


def test_chunk_boundaries_do_not_change_accumulation_or_rebuilds(cuda):
    signals = np.random.default_rng(44).uniform(2, 8, (777, 2))
    spec = schedule(len(signals), 31, -2, 1)
    expected, _ = reference(signals, **spec)
    resident = cuda.asarray(signals)
    results = []
    for chunks in ([1], [256], [127, 255, 11, 3]):
        runner = GpuCorrelationBlocks(resident, SimpleNamespace(xp=cuda), **spec)
        actual, _ = collect(runner, cuda, chunks, count=False)
        assert actual.dtype == np.float64 and actual.shape == (spec["output_count"],)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        results.append(actual)
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])


def test_nonzero_tiny_fluctuations_baseline_invalid_and_zero_mean(cuda):
    rng = np.random.default_rng(2901)
    signals = np.empty((621, 2, 4))
    signals[:, :, 0] = 1e10+rng.normal(0, .01, (621, 2))
    signals[:, :, 1] = rng.uniform(10, 20, (621, 2))
    signals[:, :, 2] = 0
    signals[:, :, 3] = rng.uniform(-2, -1, (621, 2))
    signals[15, 0, 1] = np.nan  # First baseline center is 15.
    signals[271, 1, 1] = np.inf  # Periodic baseline center at ordinal 256.
    spec = schedule(len(signals), 31, 0, 1)
    expected, counts = reference(signals, **spec)
    runner = GpuCorrelationBlocks(cuda.asarray(signals), SimpleNamespace(xp=cuda), **spec)
    actual, actual_counts = collect(runner, cuda, [111, 145, 78])
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13, equal_nan=True)
    np.testing.assert_array_equal(actual_counts, counts)
    assert np.isfinite(actual[:, :2]).all()
    assert np.isnan(actual[:, 2:]).all()
    # An independent raw g² formula verifies an ordinary nonzero component.
    for index in (0, 12, 255, 256, 590):
        direct, _ = direct_g2(signals[index:index+31, 0, 1], signals[index:index+31, 1, 1])
        np.testing.assert_allclose(actual[index, 1], direct, rtol=2e-13, atol=2e-13)


def test_schedule_validation_random_access_reset_and_cancellation(cuda):
    signals = cuda.ones((33, 2), dtype=cuda.float64)
    spec = schedule(33, 5, 0, 1)
    runner = GpuCorrelationBlocks(signals, SimpleNamespace(xp=cuda), **spec, max_output_block=7)
    for first, stop in ((1, 3), (0, 8), (0, 0), (0, 30)):
        with pytest.raises(ValueError):
            runner.compute(first, stop)
    assert runner.next_index == 0 and runner.record()["output_blocks"] == 0
    first, _ = runner.compute(0, 7)
    expected = cuda.asnumpy(first)
    with pytest.raises(ValueError):
        runner.compute(2, 3)
    runner.reset()
    again, _ = runner.compute(0, 7)
    np.testing.assert_array_equal(cuda.asnumpy(again), expected)
    assert runner.record()["resets"] == 1
    def cancel():
        raise RuntimeError("cancelled before launch")
    runner.check = cancel
    with pytest.raises(RuntimeError, match="cancelled"):
        runner.compute(7, 8)
    assert runner.next_index == 7 and runner.record()["output_blocks"] == 2
    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.compute(7, 8)
    with pytest.raises(RuntimeError, match="closed"):
        runner.reset()


def test_input_and_complete_schedule_validation_do_not_convert(cuda):
    spec = schedule(31, 5, 0, 1)
    backend = SimpleNamespace(xp=cuda)
    for signals in (cuda.ones((31, 2), cuda.float32),
                    cuda.ones((31, 4), cuda.float64)[:, ::2]):
        with pytest.raises(ValueError, match="contiguous float64"):
            GpuCorrelationBlocks(signals, backend, **spec)
    signals = cuda.ones((31, 2), cuda.float64)
    for change in (dict(first_center=0), dict(lag=1), dict(window=0),
                   dict(stride=0), dict(output_count=100), dict(min_mean=0),
                   dict(max_output_block=4097), dict(window=3.1)):
        with pytest.raises(ValueError):
            GpuCorrelationBlocks(signals, backend, **(spec | change))
    with pytest.raises(ValueError, match="components"):
        GpuCorrelationBlocks(cuda.ones((31, 2, 4097), cuda.float64), backend, **spec)


def test_timer_wraps_new_kernel_without_repeating_numerical_execution(cuda):
    class Timer:
        def __init__(self):
            self.names = []
        def launch(self, name, function, *args):
            self.names.append(name)
            return function(*args)
    timer = Timer()
    spec = schedule(41, 7, -1, 1)
    runner = GpuCorrelationBlocks(cuda.ones((41, 2), cuda.float64),
                                 SimpleNamespace(xp=cuda, gpu_timer=timer), **spec)
    result, _ = runner.compute(0, 13)
    np.testing.assert_array_equal(cuda.asnumpy(result), np.ones(13))
    assert timer.names == [KERNEL_NAME]
    assert runner.record()["output_blocks"] == 1


def test_engine_factory_prepares_checkpoint_without_extra_input_copy(tmp_path, cuda):
    from test_gpu_pipeline_0_8_4 import make, node
    nodes = [node("alignment"), node("point_extract", x=1, y=1),
             node("correlation_map", window=17, stride=1, map_lag=-3)]
    values = np.random.default_rng(719).uniform(2, 40, (661, 4, 8))
    cpu = make(tmp_path/"cpu", values, nodes, False)
    gpu = make(tmp_path/"gpu", values, nodes, True)
    runner = None
    try:
        expected = np.asarray([cpu.get(3, j) for j in range(cpu.output.length)])
        assert gpu.main_store.mode == "device" and gpu.main_store.values is None
        runner = engine_blocks(gpu, 3)
        assert runner is not None
        assert runner.signals is gpu.main_store.values
        assert gpu.main_store.ready.all()
        actual, _ = collect(runner, cuda, [runner.max_output_block], count=False)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
        assert gpu.execution_metrics["gpu_output_blocks"]["3"]["output_blocks"] == 3
        record = runner.record()
        assert record["state"] == "complete"
        assert record["additional_reserved_bytes"] >= record["state_bytes"]+runner.max_output_block*8
        assert record["preparation_seconds"] >= 0
        assert gpu.device.cache_budget <= gpu.device.budget//4
        assert len(gpu.source.reader.reads) == len(values)
    finally:
        if runner:
            runner.close()
        gpu.close()
        cpu.close()


def test_engine_factory_rejects_ineligible_or_tight_budget_before_input_execution(tmp_path, cuda):
    from test_gpu_pipeline_0_8_4 import make, node
    values = np.random.default_rng(19).uniform(2, 40, (33, 4, 8))
    nodes = [node("alignment"), node("point_extract"), node("correlation_map", window=7, stride=1)]
    gpu = make(tmp_path/"gpu", values, nodes, True)
    try:
        assert engine_blocks(gpu, 2) is None
        assert gpu.execution_metrics["gpu_output_blocks"]["2"]["reason"] == "not_a_fixed_window_correlation"
        gpu.inspection = True
        assert engine_blocks(gpu, 3) is None
        assert gpu.execution_metrics["gpu_output_blocks"]["3"]["reason"] == "local_inspection_uses_demand_execution"
        gpu.inspection = False
        gpu.device.budget = 64
        assert engine_blocks(gpu, 3) is None
        assert gpu.execution_metrics["gpu_output_blocks"]["3"]["reason"] == "insufficient_budget_for_state_and_one_output"
        assert gpu.main_store.values is None
        assert not gpu.source.reader.reads
    finally:
        gpu.close()


def test_engine_factory_does_not_swallow_input_errors(tmp_path, cuda):
    from test_gpu_pipeline_0_8_4 import make, node
    values = np.ones((33, 4, 8), np.float64)
    nodes = [node("alignment"), node("point_extract"), node("correlation_map", window=7, stride=1)]
    gpu = make(tmp_path/"gpu", values, nodes, True)
    def broken(*args):
        raise ValueError("invalid original source pixels")
    gpu._stored_main = broken
    try:
        with pytest.raises(ValueError, match="original source pixels"):
            engine_blocks(gpu, 3)
        record = gpu.execution_metrics["gpu_output_blocks"]["3"]
        assert record["state"] == "failed"
        assert "original source pixels" in record["error"]
    finally:
        gpu.close()


def test_sparse_windows_keep_existing_demand_execution(tmp_path,cuda):
    from test_gpu_pipeline_0_8_4 import make,node
    values = np.ones((173,4,8),np.float64)
    steps = [node("alignment"),node("point_extract"),node("correlation_map",window=7,stride=31)]
    engine = make(tmp_path,values,steps,True)
    try:
        assert engine_blocks(engine,3) is None
        assert engine.execution_metrics["gpu_output_blocks"]["3"]["reason"] == "sparse_windows_keep_demand_loading"
        assert not engine.source.reader.reads
    finally:
        engine.close()


def test_registered_cuda_event_timer_reports_block_kernel_coverage(cuda):
    from eip.gpu_timing import GpuEventTimer
    timer = GpuEventTimer(cuda)
    signals = np.random.default_rng(918).uniform(3, 30, (521, 2))
    spec = schedule(len(signals), 11, 2, 1)
    runner = GpuCorrelationBlocks(cuda.asarray(signals),
                                 SimpleNamespace(xp=cuda, gpu_timer=timer), **spec,
                                 max_output_block=64)
    actual, _ = collect(runner, cuda, [64], count=False)
    timer.finish(raise_errors=True)
    expected, _ = reference(signals, **spec)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    coverage = timer.snapshot()["kernels"][KERNEL_NAME]
    assert coverage["calls"] == runner.record()["output_blocks"] == 8
    assert coverage["timed_calls"] > 0 and coverage["seconds"] > 0
    assert coverage["failed_calls"] == 0


def test_engine_factory_rejects_large_rebuild_work_before_preparing_input(tmp_path, cuda):
    from test_gpu_pipeline_0_8_4 import make, node
    from eip.gpu_sequence_blocks import MAX_SERIAL_PAIR_UPDATES
    # A broadcast reader fixture represents a long sequence without allocating
    # all its source frames. The production limit is exercised without reading
    # or preparing that sequence at all.
    window = MAX_SERIAL_PAIR_UPDATES+1
    values = np.broadcast_to(np.ones((4, 8), np.float64), (window+3, 4, 8))
    nodes = [node("alignment"), node("point_extract"),
             node("correlation_map", window=window, stride=1)]
    gpu = make(tmp_path/"gpu", values, nodes, True)
    try:
        assert gpu.main_store is not None and gpu.main_store.mode == "device"
        assert engine_blocks(gpu, 3) is None
        row = gpu.execution_metrics["gpu_output_blocks"]["3"]
        assert row["reason"] == "rebuild_pair_work_exceeds_bounded_launch_limit"
        assert gpu.main_store.values is None
        assert not gpu.source.reader.reads
    finally:
        gpu.close()


def test_engine_factory_rejects_rolling_work_that_exceeds_rebuild_work(tmp_path, cuda):
    from test_gpu_pipeline_0_8_4 import make, node
    from eip.gpu_sequence_blocks import MAX_SERIAL_PAIR_UPDATES
    window = MAX_SERIAL_PAIR_UPDATES-1
    stride = window-1
    values = np.broadcast_to(np.ones((4, 8), np.float64), (window+stride+2, 4, 8))
    nodes = [node("alignment"), node("point_extract"),
             node("correlation_map", window=window, stride=stride)]
    gpu = make(tmp_path/"gpu", values, nodes, True)
    try:
        assert engine_blocks(gpu, 3) is None
        row = gpu.execution_metrics["gpu_output_blocks"]["3"]
        assert row["reason"] == "rolling_pair_work_exceeds_bounded_launch_limit"
        assert gpu.main_store.values is None
        assert not gpu.source.reader.reads
    finally:
        gpu.close()


@pytest.mark.parametrize("stride,components,serial_limit,launch_limit,expected_block", [
    (1, 1, 95, 950, 40),
    # A no-rebuild block costs 2*stride per result, more than rebuilding W.
    (16, 1, 95, 950, 2),
    (17, 1, 95, 950, 5),
    (1, 4, 95, 96, 4),
])
def test_engine_factory_bounds_serial_and_total_launch_work(
        tmp_path, cuda, monkeypatch, stride, components, serial_limit, launch_limit, expected_block):
    from test_gpu_pipeline_0_8_4 import make, node
    import eip.gpu_sequence_blocks as blocks
    monkeypatch.setattr(blocks, "MAX_SERIAL_PAIR_UPDATES", serial_limit)
    monkeypatch.setattr(blocks, "MAX_LAUNCH_PAIR_UPDATES", launch_limit)
    selector = node("point_extract") if components == 1 else node("roi_crop", width=2, height=2)
    nodes = [node("alignment"), selector, node("correlation_map", window=17, stride=stride)]
    values = np.random.default_rng(8002).uniform(2, 20, (173, 4, 8))
    cpu = make(tmp_path/"cpu", values, nodes, False)
    gpu = make(tmp_path/"gpu", values, nodes, True)
    runner = None
    try:
        expected = np.stack([cpu.get(3, j) for j in range(cpu.output.length)])
        runner = engine_blocks(gpu, 3)
        assert runner is not None and runner.max_output_block == expected_block
        actual = []
        previous_pairs = 0
        while runner.next_index < runner.output_count:
            start = runner.next_index
            stop = min(start+runner.max_output_block, runner.output_count)
            value, _ = runner.compute(start, stop)
            actual.append(cuda.asnumpy(value))
            pairs = runner.record()["pair_updates"]
            assert pairs-previous_pairs <= serial_limit
            assert (pairs-previous_pairs)*components <= launch_limit
            previous_pairs = pairs
        np.testing.assert_allclose(np.concatenate(actual), expected, rtol=2e-13, atol=2e-13)
        row = runner.record()
        assert row["estimated_pair_updates_per_component"] <= serial_limit
        assert row["estimated_pair_updates_per_component"]*components <= launch_limit
    finally:
        if runner:
            runner.close()
        gpu.close()
        cpu.close()


@pytest.mark.parametrize("mixed", [False, True])
def test_engine_output_blocks_keep_postprocessing_precision_and_explicit_devices(tmp_path, cuda, mixed):
    from copy import deepcopy
    from eip.sequence_engine import SequenceEngine
    from test_gpu_pipeline_0_8_4 import Reader, node
    correlation = node("correlation_map", window=17, stride=1, map_lag=-3)
    correlation["execution"] = dict(backend="gpu", precision="float32")
    gain = node("gain_offset", gain=1.123456789, offset=-.234567891)
    gain["execution"] = dict(backend="cpu" if mixed else "gpu", precision="float32")
    curve = node("curve", curve="square", input_max=1.7)
    curve["execution"] = dict(backend="gpu", precision="float32")
    suffix = [gain, curve]
    if not mixed:
        gain2 = node("gain_offset", gain=.987654321, offset=.123456789)
        gain2["execution"] = dict(backend="gpu", precision="float64")
        suffix.append(gain2)
    nodes = [node("alignment"), node("point_extract", x=1, y=1), correlation, *suffix]
    values = np.random.default_rng(2190).uniform(2, 18, (661, 4, 8))
    def engine(where, optimize):
        return SequenceEngine(Reader(values), deepcopy(nodes), workdir=where,
                              optimize=optimize,
                              resources=dict(memory_mb=256, preview_mb=16, cache_mb=0,
                                             gpu_mb=128, backend="gpu"))
    old, new = engine(tmp_path/"old", False), engine(tmp_path/"new", True)
    try:
        stage = len(nodes)
        expected = [old.get(stage, j).copy() for j in range(old.output.length)]
        actual = [value.copy() for _, value in new.iter_outputs(range(new.output.length))]
        assert old.output.record() == new.output.record()
        assert len(actual) == len(expected)
        for j, (left, right) in enumerate(zip(expected, actual)):
            assert left.shape == right.shape == ()
            assert left.dtype == right.dtype == np.dtype(new.output.dtype)
            np.testing.assert_array_equal(np.isfinite(right), np.isfinite(left))
            np.testing.assert_allclose(right, left, rtol=0., atol=0.)
            assert new.ref(stage, j) == old.ref(stage, j)
        row = new.execution_metrics["gpu_output_blocks"]["3"]
        assert row["output_blocks"] == 3
        assert row["state"] == "complete"
        assert row["suffix_stage_count"] == len(suffix)
        assert row["suffix_budget_policy"] == "validated_forward_chain"
        assert row["suffix_device_working_bytes_per_sample"] > 0
        gain_record = new.device_records[nodes[3]["id"]]
        assert gain_record["actual"] == ("cpu" if mixed else "gpu")
        assert new.device_records[nodes[4]["id"]]["actual"] == "gpu"
        if mixed:
            # Each batch crosses GPU→CPU for gain and CPU→GPU for curve, then
            # crosses GPU→host for delivery. No per-result synchronization.
            assert new.device.metrics["downloads"] == 2*row["output_blocks"]
            assert new.spatial.record()["downloaded_bytes"] > 0
            assert new.spatial.record()["uploaded_bytes"] > 0
            assert row["suffix_host_working_bytes_per_sample"] > 0
            assert row["host_bytes_per_output"] > row["suffix_host_working_bytes_per_sample"]
        else:
            assert new.device.metrics["downloads"] == row["output_blocks"]
        for suffix_stage in range(4, stage+1):
            assert new.execution_metrics["stage_evaluations"][str(suffix_stage)] == new.output.length
    finally:
        new.close()
        old.close()


@pytest.mark.parametrize("enough_for_suffix", [False, True])
def test_engine_factory_budgets_expanded_suffix_batches_before_checkpoint_preparation(
        tmp_path, cuda, enough_for_suffix):
    from copy import deepcopy
    from eip.sequence_engine import SequenceEngine
    from test_gpu_pipeline_0_8_4 import Reader, node
    nodes = [node("alignment"), node("roi_crop", width=2, height=2),
             node("correlation_map", window=17, stride=1),
             node("color_convert", mode="rgb"),
             node("gain_offset", gain=1.234567, offset=.123456)]
    nodes[4]["execution"] = dict(precision="float32", backend="gpu")
    values = np.random.default_rng(512).uniform(5, 40, (661, 4, 8))
    gpu = SequenceEngine(Reader(values), deepcopy(nodes), workdir=tmp_path/"gpu", optimize=True,
                         resources=dict(memory_mb=256, preview_mb=16, cache_mb=0,
                                        gpu_mb=128, backend="gpu"))
    runner = None
    try:
        plans = [r for r in gpu.resource_plan if "device_working_bytes" in r]
        base = (sum(r.get("device_retained_bytes", 0) for r in plans)
                +max(r["device_working_bytes"] for r in plans)+gpu.main_store.capacity_bytes)
        state_bytes = 6*4*8
        # With only enough space for the original four-value correlation
        # buffer, RGB expansion and float64 modifier work cannot be hidden.
        output_space = 4*8*(256 if enough_for_suffix else 1)
        gpu.device.budget = int(np.ceil((base+state_bytes+output_space)/.75))
        assert gpu.main_store.values is None
        runner = engine_blocks(gpu, 3)
        if not enough_for_suffix:
            assert runner is None
            assert gpu.main_store.values is None and not gpu.source.reader.reads
            assert gpu.execution_metrics["gpu_output_blocks"]["3"]["reason"] == "insufficient_budget_for_state_and_one_output"
        else:
            assert runner is not None
            row = runner.record()
            assert 1 <= runner.max_output_block < 256
            assert row["suffix_device_working_bytes_per_sample"] >= gpu.resource_plan[4]["working_bytes"]
            assert row["device_bytes_per_output"] > 4*8
            assert row["additional_reserved_bytes"] <= state_bytes+output_space+1
            result, _ = runner.compute(0, runner.max_output_block)
            final = gpu.spatial.forward(3, len(nodes), result, gpu.device.backend(3))
            assert final.shape == (runner.max_output_block, 2, 2, 3)
            assert final.dtype == np.float32
            assert np.isfinite(final).all()
    finally:
        if runner:
            runner.close()
        gpu.close()
