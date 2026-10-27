"""Execution reuse and resource bounds; these are not GPU speed measurements."""
from copy import deepcopy

import numpy as np
import pytest

from eip.module_contracts import Contract
from eip.module_registry import new_node
from eip.sequence_engine import SequenceEngine
from test_gpu_pipeline_0_8_4 import Reader, device, make, node


def resources(**changes):
    return dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=128,backend="cpu",**changes)


def test_expanding_windows_reuse_baseline_without_periodic_full_reread(tmp_path):
    values = np.random.default_rng(903).uniform(1,8,(701,2,2,3))
    scan = node("correlation_window",window_min=2,window_max=302,window_step=1,center=351,lag=0)
    reader = Reader(values,Contract("PYX",(2,2,3),701))
    engine = SequenceEngine(reader,[scan],workdir=tmp_path,resources=resources())
    try:
        for j,w in enumerate(range(2,303)):
            actual = engine.get(1,j)
            if j in (0,254,255,256,300):
                start = 350-(w-1)//2
                left,right = values[start:start+w,0],values[start:start+w,1]
                expected = (left*right).mean(axis=0)/(left.mean(axis=0)*right.mean(axis=0))
                np.testing.assert_allclose(actual,expected,rtol=2e-13)
        # One read for the baseline, then exactly the 302 contributing pairs.
        # Zero lag obtains both channels from a single upstream sample.
        assert len(reader.reads)==303
        assert reader.reads.count(350)==2
    finally:
        engine.close()


def test_recent_frame_reference_does_not_decode_pixels_again(tmp_path):
    values = np.ones((301,2,3))
    reader = Reader(values)
    engine = SequenceEngine(reader,[],workdir=tmp_path,resources=resources())
    try:
        for i in range(301):
            engine.get(0,i)
            assert engine.ref(0,i)["time"]==pytest.approx(i*.01)
        assert len(reader.reads)==301
        assert len(engine.source.ref_cache)<=256
    finally:
        engine.close()


def test_source_dtype_and_validated_slack_increase_real_transfer_batch(tmp_path,device):
    values = np.ones((40,256,256),dtype=np.uint16)
    engine = make(tmp_path,values,[node("gain_offset",gain=2)],True)
    try:
        # Former policy estimated every uint16 pixel as float64 and allowed
        # only one upload at these dimensions / budgets.
        assert engine.device.batch_size(0)>1
        assert engine.device.cache_budget<=engine.device.budget//4
        persistent = sum(p.get("device_retained_bytes",0) for p in engine.resource_plan)
        working = max(p.get("device_working_bytes",0) for p in engine.resource_plan)
        assert persistent+working+engine.device.cache_budget<=engine.device.budget*.75
        for i in range(40):
            np.testing.assert_array_equal(engine.get(1,i),values[i]*2)
        metrics = engine.device.record()
        assert metrics["uploaded_samples"]==40
        assert metrics["upload_batches"]<40
        assert metrics["cache_peak_bytes"]<=metrics["cache_budget_bytes"]
    finally:
        engine.close()


def test_reverse_sample_requests_reuse_aligned_upload_chunks(tmp_path,device):
    values = np.arange(128*4*8,dtype=float).reshape(128,4,8)
    engine = make(tmp_path,values,[node("gain_offset",gain=2)],True)
    try:
        for i in range(70,49,-1):
            np.testing.assert_array_equal(engine.get(1,i),values[i]*2)
        assert engine.device.metrics["upload_batches"]==2
        assert len(engine.source.reader.reads)==64
    finally:
        engine.close()


def test_adaptive_cache_can_shrink_below_initial_size(tmp_path,device):
    values = np.ones((2,2,4))
    engine = make(tmp_path,values,[node("gain_offset")],True)
    try:
        budget = engine.device.budget
        engine.resource_plan = [dict(device_retained_bytes=budget//2,device_working_bytes=budget//4-1024)]
        engine.device.configure_cache()
        assert engine.device.cache_budget==1024
        assert engine.device.cache.budget+engine.device.upload_cache.budget==1024
    finally:
        engine.close()


def test_retained_downstream_statistics_must_fit_while_large_upstream_runs(tmp_path,device):
    # Every node used to fit its own prefix check, but after all side statistics
    # existed the next large upstream frame needed 50 MiB in a 48 MiB worker.
    # Removing cache cannot make that execution fit.
    values = np.ones((1,1024,1024),dtype=np.float64)
    nodes = [node("curve"),node("roi_crop",width=256,height=256)]
    nodes += [node("global_statistics",scope="samples",output="side") for _ in range(26)]
    with pytest.raises(ValueError,match="全流程保留状态"):
        make(tmp_path,values,nodes,True,resources=dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=64,backend="cpu"))


@pytest.mark.parametrize("lag",[-3,0,2])
def test_small_gpu_signals_use_bounded_batches_with_unchanged_pair_order(tmp_path,device,monkeypatch,lag):
    from eip.modes.split_correlation.correlation import RollingMoments
    values = np.random.default_rng(990).uniform(1,8,(201,2,2,3))
    values[80,0,0,1] = np.nan
    scan = node("correlation_window",window_list=[3,75,101],center=101,lag=lag)
    calls = []
    original = RollingMoments.add_many

    def tracked(self,left,right,sign=1):
        calls.append((left.shape[0],sign))
        return original(self,left,right,sign)

    monkeypatch.setattr(RollingMoments,"add_many",tracked)
    engines = []
    try:
        for gpu in (False,True):
            nodes = [deepcopy(scan)]
            nodes[0]["execution"] = {"backend":"gpu" if gpu else "cpu"}
            engines.append(SequenceEngine(Reader(values,Contract("PYX",(2,2,3),201)),nodes,
                                          workdir=tmp_path/str(gpu),resources=resources()))
        for j in (0,1,2,1,0,2):
            # Force random-access recomputation rather than a valid cached
            # result, so this also exercises accumulator sample removal.
            engines[1].device.cache.clear()
            np.testing.assert_allclose(engines[1].get(1,j),engines[0].get(1,j),rtol=2e-13,equal_nan=True)
        assert calls and all(1<count<=64 for count,_ in calls)
        assert any(sign==-1 for _,sign in calls)
    finally:
        for engine in engines:
            engine.close()
