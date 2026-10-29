"""Integration of owned compact main data with actual window calculations."""
import numpy as np
import pytest
from dataclasses import replace

from eip.sequence_engine import SequenceEngine
from test_gpu_pipeline_0_8_4 import Reader, node, device


def pipeline(*, side=False):
    steps = [node("time_crop"), node("calibration",curve="linear",input_max=4095),
             node("alignment"),node("roi_crop",x=1,y=1,width=2,height=2),
             node("time_statistics",mode="mean",validity="independent"),
             node("correlation_map",window=101,stride=1,map_lag=0)]
    if side:
        steps.insert(3,node("data_tap"))
    return steps


def resource(backend="cpu"):
    return dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=128,backend=backend)


def test_reduced_sequence_is_evaluated_once_across_accumulator_rebuilds(tmp_path):
    values = np.random.default_rng(2810).integers(100,2000,(501,4,8),dtype=np.uint16)
    reader = Reader(values)
    steps = pipeline()
    engine = SequenceEngine(reader,steps,workdir=tmp_path,resources=resource())
    try:
        assert engine.main_store.stage==5 and engine.main_store.mode=="ram"
        left = values[:,1:3,1:3].astype(np.float64).mean(axis=(1,2))/4095
        right = values[:,1:3,5:7].astype(np.float64).mean(axis=(1,2))/4095
        for index in range(engine.output.length):
            actual = engine.get(6,index)
            if index in (0,255,256,400):
                l,r = left[index:index+101],right[index:index+101]
                expected = np.mean(l*r)/(l.mean()*r.mean())
                np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
            engine.ref(6,index)
        assert engine.main_store.record()["built_samples"]==501
        assert len(reader.reads)==501
        assert engine.main_store.values.nbytes==501*2*8
        assert engine.cache.size==0
        for stage in range(6):
            assert engine.execution_metrics["stage_evaluations"][str(stage)]==501
        # Random access including a numerical rebase uses the same owned data.
        before = len(reader.reads)
        for index in (256,0,399,100):
            engine.get(6,index)
        assert len(reader.reads)==before
        np.testing.assert_array_equal(reader.values,values)
    finally:
        engine.close()


@pytest.mark.parametrize("disk",[False,True])
def test_real_cuda_reuse_and_disk_boundary_preserve_explicit_cpu_node(tmp_path,disk):
    from eip.modes.split_correlation.backend import devices
    if not devices()["gpu"]:
        pytest.skip("Real CUDA is required")
    values=np.random.default_rng(823).uniform(100,2000,(137,4,8))
    steps=pipeline()
    steps[2].setdefault("execution",{})["backend"]="cpu"
    engine=SequenceEngine(Reader(values),steps,workdir=tmp_path,resources=resource("gpu"))
    try:
        assert engine.main_store.mode=="device"
        if disk:
            # Force the tested spill carrier without allocating a huge fixture.
            engine.main_store.mode="disk"
            engine.main_store.metrics["storage"]="disk"
        expected_left=values[:,1:3,1:3].mean(axis=(1,2))/4095
        expected_right=values[:,1:3,5:7].mean(axis=(1,2))/4095
        for index in (0,1,20,36,0):
            actual=engine.get(len(steps),index)
            l,r=expected_left[index:index+101],expected_right[index:index+101]
            np.testing.assert_allclose(actual,np.mean(l*r)/(l.mean()*r.mean()),rtol=2e-13)
        assert engine.device_records[steps[2]["id"]]["actual"]=="cpu"
        assert engine.device_records[steps[-1]["id"]]["actual"]=="gpu"
        assert engine.main_store.record()["built_samples"]==137
        if disk:
            assert engine.main_store.record()["disk_write_bytes"]==137*2*8
            assert engine.main_store.path.exists()
        else:
            import cupy as cp
            assert isinstance(engine.main_store.values,cp.ndarray)
        stored_path=engine.main_store.path
    finally:
        engine.close()
    assert not stored_path.exists()


def test_upstream_side_outputs_keep_compatible_lifetime(tmp_path):
    steps=pipeline(side=True)
    values=np.ones((137,4,8),dtype=np.uint16)*100
    engine=SequenceEngine(Reader(values),steps,workdir=tmp_path,resources=resource())
    try:
        assert engine.main_store is None
        engine.get(len(steps),0)
        side=engine.side_outputs()[0]
        actual=engine.get_side_output(side["id"],136)
        np.testing.assert_allclose(actual,np.stack((values[136,:,:4],values[136,:,4:]))/4095)
    finally:
        engine.close()


def test_inspection_only_builds_requested_compact_sample(tmp_path):
    steps=pipeline()
    values=np.ones((137,4,8),dtype=np.uint16)*100
    engine=SequenceEngine(Reader(values),steps,workdir=tmp_path,resources=resource(),inspection=True,single=True)
    try:
        engine.get(5,75)
        assert engine.main_store.record()["built_samples"]==1
        assert engine.main_store.block_size==1
    finally:
        engine.close()


def test_large_inspection_does_not_reserve_whole_disk_carrier(tmp_path):
    reader=Reader(np.ones((1,4,8),dtype=np.uint16))
    reader.meta=replace(reader.meta,frame_count=800000)
    steps=[node("time_crop"),node("alignment"),node("roi_crop",width=2,height=3),
           node("correlation_map",window=101)]
    engine=SequenceEngine(reader,steps,workdir=tmp_path,resources=resource(),inspection=True,single=True)
    try:
        assert engine.main_store is None
        np.testing.assert_array_equal(engine.get(3,0),np.ones((2,3,2)))
        assert not list(tmp_path.glob("stage-*.bin"))
    finally:
        engine.close()


def test_main_device_store_includes_explicit_instance_working_set(tmp_path,device):
    from test_gpu_pipeline_0_8_4 import make
    engine=make(tmp_path,np.ones((137,4,8)),pipeline(),True)
    try:
        engine.main_store.close()
        engine.main_store=None
        engine.contracts[5]=replace(engine.contracts[5],length=786432)
        engine.nodes[4]["execution"]["memory_mb"]=16
        engine.resource_plan[4].update(working_bytes=6*1024**2,device_working_bytes=6*1024**2)
        engine._configure_main_store()
        assert engine.main_store.capacity_bytes==12*1024**2
        assert engine.main_store.mode=="disk"
    finally:
        engine.close()


def test_optional_stage_store_does_not_require_extra_disk(tmp_path,monkeypatch):
    engine=SequenceEngine(Reader(np.ones((137,4,8))),pipeline(),workdir=tmp_path,resources=resource())
    try:
        engine.main_store.close()
        engine.main_store=None
        engine.contracts[5]=replace(engine.contracts[5],length=786432)
        engine.nodes[4].setdefault("execution",{})["memory_mb"]=16
        engine.resource_plan[4]["working_bytes"]=6*1024**2
        from types import SimpleNamespace
        monkeypatch.setattr("eip.sequence_engine.shutil.disk_usage",lambda p:SimpleNamespace(free=1024))
        engine._configure_main_store()
        assert engine.main_store is None
        assert engine.execution_metrics["main_storage_fallback"]=="insufficient_temporary_disk"
        assert not list(tmp_path.glob("stage-*.bin"))
    finally:
        engine.close()
