"""Full-frame CPU streaming must avoid unnecessary block copies."""
import numpy as np

from test_gpu_pipeline_0_8_4 import Reader,node
from eip.sequence_engine import SequenceEngine


def test_large_local_cpu_chain_keeps_original_streaming_and_values(tmp_path):
    values=np.random.default_rng(904).uniform(1,30,(3,512,384))
    steps=[node("gain_offset",gain=1.125,offset=.25),node("curve",curve="square",input_max=4095.)]
    engine=SequenceEngine(Reader(values),steps,workdir=tmp_path,
                          resources=dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=128,backend="cpu"))
    try:
        engine.spatial.compute=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("No full-frame batch copy"))
        actual=np.stack([v for _,v in engine.iter_outputs(range(3))])
        expected=((values*1.125+.25)/4095.)**2
        np.testing.assert_array_equal(actual,expected)
        assert engine.execution_metrics["full_frame_cpu_policy"]=="sample_streaming_avoids_redundant_block_copy"
        assert len(engine.source.reader.reads)==3
    finally:
        engine.close()


def test_roi_still_requests_only_its_physical_domain(tmp_path):
    values=np.ones((3,512,384))
    steps=[node("gain_offset",gain=2),node("roi_crop",x=3,y=4,width=8,height=9)]
    engine=SequenceEngine(Reader(values),steps,workdir=tmp_path,
                          resources=dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=128,backend="cpu"))
    try:
        actual=np.stack([v for _,v in engine.iter_outputs(range(3))])
        np.testing.assert_array_equal(actual,np.full((3,9,8),2.))
        assert "full_frame_cpu_policy" not in engine.execution_metrics
        assert engine.spatial.metrics["region_elements"]["1"]==3*9*8
    finally:
        engine.close()
