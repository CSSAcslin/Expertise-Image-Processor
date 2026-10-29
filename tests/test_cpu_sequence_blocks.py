"""Scalar host batches preserve the established array estimator exactly."""
from types import SimpleNamespace

import numpy as np
import pytest

from eip.algorithms import Cancelled
from eip.cpu_sequence_blocks import CpuCorrelationBlocks, cpu_engine_blocks
from eip.modes.split_correlation.correlation import RollingMoments, direct_g2


def schedule(length, window, lag, stride):
    return dict(window=window, lag=lag, stride=stride,
                first_center=(window-1)//2+max(0, -lag),
                output_count=(length-window-abs(lag))//stride+1)


def reference(signals, *, window, lag, first_center, stride, output_count,
              min_mean=1e-9, min_samples=2):
    config = SimpleNamespace(min_mean=min_mean, min_samples=min_samples)
    backend = SimpleNamespace(xp=np, host=np.asarray)
    values, counts = [], []
    moments = None
    old_start = old_stop = None
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        for ordinal in range(output_count):
            center = first_center+ordinal*stride
            start = center-(window-1)//2
            stop = start+window
            if moments is None or stride >= window or ordinal % 256 == 0:
                baseline = np.nan_to_num(signals[center], nan=0., posinf=0., neginf=0.)
                moments = RollingMoments(baseline, config, backend)
                changes = ((start, stop, 1),)
            else:
                changes = ((old_start, start, -1), (old_stop, stop, 1))
            for first, last, sign in changes:
                for index in range(first, last):
                    moments.add(signals[index, 0], signals[index+lag, 1], sign)
            result, count = moments.result()
            values.append(result.copy())
            counts.append(count.copy())
            old_start, old_stop = start, stop
    return np.asarray(values), np.asarray(counts)


def collect(runner, chunks, *, count=True):
    values, counts = [], []
    part = 0
    while runner.next_index < runner.output_count:
        first = runner.next_index
        stop = min(first+chunks[part % len(chunks)], runner.output_count)
        actual, samples = runner.compute(first, stop, count=count)
        values.append(actual)
        if count:
            counts.append(samples)
        else:
            assert samples is None
        part += 1
    return np.concatenate(values), np.concatenate(counts) if count else None


@pytest.mark.parametrize("lag,stride,window",[(-5,1,17),(0,2,18),(7,3,23),(-3,7,7),(4,11,5)])
def test_signed_lag_stride_and_global_rebuild_match_bitwise(lag,stride,window):
    signals = np.random.default_rng(2940).uniform(1,19,(1901,2))
    signals[7,0],signals[29,1],signals[256,0] = np.nan,np.inf,-np.inf
    spec = schedule(len(signals),window,lag,stride)
    expected, counts = reference(signals,**spec)
    original = signals.copy()
    runner = CpuCorrelationBlocks(signals,**spec)
    actual, actual_counts = collect(runner,[63,129,1,17,256])
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual_counts,counts)
    np.testing.assert_array_equal(signals,original)
    assert runner.signals is signals
    rebuilds = (spec["output_count"]-1)//256+1 if stride<window else spec["output_count"]
    assert runner.record()["periodic_rebuilds"]==rebuilds
    assert runner.record()["pair_updates"]==rebuilds*window+(spec["output_count"]-rebuilds)*2*stride


def test_arbitrary_chunks_preserve_global_order_and_invalid_baseline():
    signals = np.random.default_rng(44).uniform(2,8,(777,2))
    signals[15,0],signals[271,1] = np.nan,np.inf
    spec = schedule(len(signals),31,0,1)
    expected,_ = reference(signals,**spec)
    for chunks in ([1],[256],[127,255,11,3]):
        runner = CpuCorrelationBlocks(signals,**spec)
        actual,_ = collect(runner,chunks,count=False)
        np.testing.assert_array_equal(actual,expected)
        assert runner.record()["largest_output_block"]<=256


@pytest.mark.parametrize("mode",["tiny_fluctuations","zero","negative","overflow","underflow"])
def test_nonfinite_and_extreme_values_match_array_estimator(mode):
    rng = np.random.default_rng(2901)
    signals = np.ascontiguousarray(1e10+rng.normal(0,.01,(621,2)))
    min_mean = 1e-9
    if mode=="zero":
        signals[:,0] = 0
    elif mode=="negative":
        signals *= -1
    elif mode=="overflow":
        signals = np.ascontiguousarray(rng.choice([-1.,1.],(621,2))*1e308)
    elif mode=="underflow":
        signals = np.full((621,2),1e-200)
        min_mean = 1e-250
    signals[17,0],signals[35,1] = np.nan,-np.inf
    spec = dict(schedule(len(signals),31,-2,1),min_mean=min_mean)
    expected,counts = reference(signals,**spec)
    runner = CpuCorrelationBlocks(signals,**spec)
    actual,actual_counts = collect(runner,[111,145,78])
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual_counts,counts)
    if mode=="tiny_fluctuations":
        assert np.isfinite(actual).all()
        for index in (0,12,255,256,580):
            spec_center = spec["first_center"]+index
            begin = spec_center-15
            direct,_ = direct_g2(signals[begin:begin+31,0],signals[begin-2:begin+29,1])
            np.testing.assert_allclose(actual[index],direct,rtol=2e-13,atol=2e-13)


def test_rejects_conversion_random_access_and_closed_state():
    signals = np.ones((33,2),dtype=np.float64)
    spec = schedule(33,5,0,1)
    for bad in (signals.astype(np.float32),signals[::-1],signals.tolist(),signals[:,:,None]):
        with pytest.raises(ValueError):
            CpuCorrelationBlocks(bad,**spec)
    runner = CpuCorrelationBlocks(signals,**spec,max_output_block=7)
    for first,stop in ((1,3),(0,8),(0,0),(0,30)):
        with pytest.raises(ValueError):
            runner.compute(first,stop)
    assert runner.next_index==0 and runner.record()["output_blocks"]==0
    values,_ = runner.compute(0,7)
    assert np.equal(values,1).all()
    runner.reset()
    repeated,_ = runner.compute(0,7)
    np.testing.assert_array_equal(repeated,values)
    runner.close()
    with pytest.raises(RuntimeError):
        runner.compute(7,8)
    with pytest.raises(RuntimeError):
        runner.reset()


def test_mid_block_cancellation_does_not_commit_partial_state():
    calls = [0]
    def check():
        calls[0] += 1
        if calls[0]==5:
            raise Cancelled()
    signals = np.random.default_rng(77).uniform(1,10,(2001,2))
    spec = schedule(len(signals),1001,0,1)
    runner = CpuCorrelationBlocks(signals,**spec,check=check)
    with pytest.raises(Cancelled):
        runner.compute(0,5)
    assert runner.next_index==0 and runner.record()["output_blocks"]==0
    actual,_ = runner.compute(0,5)
    expected,_ = reference(signals,**dict(spec,output_count=5))
    np.testing.assert_array_equal(actual,expected)


def test_engine_factory_prepares_and_shares_ram_checkpoint(tmp_path):
    from test_spatial_execution import node,alignment,make
    values = np.random.default_rng(81).uniform(1,20,(601,12,20))
    steps = [alignment(),node("point_extract",x=3,y=4),
             node("correlation_map",window=17,map_lag=-3,stride=1)]
    engine = make(tmp_path,values,steps,"cpu",True)
    try:
        assert engine.main_store.mode=="ram"
        runner = cpu_engine_blocks(engine,3)
        assert runner is not None and runner.signals is engine.main_store.values
        actual,_ = collect(runner,[256,113])
        expected = np.asarray([engine.get(3,index) for index in range(engine.output.length)])
        np.testing.assert_array_equal(actual,expected)
        assert engine.main_store.ready.all()
        row = engine.execution_metrics["cpu_output_blocks"]["3"]
        assert row["output_samples"]==engine.output.length
        assert row["additional_reserved_bytes"]<=engine.resource_plan[2]["limit_bytes"]
        runner.close()
    finally:
        engine.close()


@pytest.mark.parametrize("unsupported",["disk","dtype","budget","inspection","curve"])
def test_engine_factory_rejects_unsupported_before_input_preparation(tmp_path,unsupported):
    from test_spatial_execution import node,alignment,make
    values = np.ones((17,12,20),dtype=np.float64)
    steps = [alignment(),node("point_extract",x=0,y=0),node("correlation_map",window=5,stride=1)]
    engine = make(tmp_path,values,steps,"cpu",True)
    try:
        if unsupported=="disk":
            engine.main_store.mode="disk"
            reason="direct_input_has_no_RAM_checkpoint"
        elif unsupported=="dtype":
            engine.main_store.dtype=np.dtype("float32")
            reason="checkpoint_conversion_is_not_implicit"
        elif unsupported=="budget":
            engine.budget=0
            reason="insufficient_budget_for_state_and_one_output"
        elif unsupported=="inspection":
            engine.inspection=True
            reason="local_inspection_uses_demand_execution"
        else:
            engine.operations[-1]["parameters"]["analysis"]="curve"
            reason="requires_fixed_lag_sequential_T_outputs"
        assert cpu_engine_blocks(engine,3) is None
        assert not engine.source.reader.reads
        assert engine.execution_metrics["cpu_output_blocks"]["3"]["reason"]==reason
    finally:
        engine.close()


def test_sparse_windows_do_not_prepare_the_unused_whole_signal(tmp_path):
    from test_spatial_execution import node,alignment,make
    values = np.ones((173,12,20),dtype=np.float64)
    steps = [alignment(),node("point_extract"),node("correlation_map",window=7,stride=31)]
    engine = make(tmp_path,values,steps,"cpu",True)
    try:
        assert cpu_engine_blocks(engine,3) is None
        assert engine.execution_metrics["cpu_output_blocks"]["3"]["reason"] == "sparse_windows_keep_demand_loading"
        assert not engine.source.reader.reads
    finally:
        engine.close()
