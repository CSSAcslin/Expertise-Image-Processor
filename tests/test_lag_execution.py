"""Q delivery keeps module precision/devices, exact CINE times and trial scope."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import pytest

from eip.domain import SourceSpec
from eip.module_registry import new_node
from eip.readers import open_source
from eip.sequence_engine import SequenceEngine


@pytest.mark.parametrize("device",["cpu","gpu"])
@pytest.mark.parametrize("single",[False,True])
@pytest.mark.parametrize("suffix",[False,True])
def test_complete_Q_trial_and_run_match_compatibility_with_real_times(
        tmp_path,cine_factory,device,single,suffix):
    if device=="gpu":
        from eip.modes.split_correlation.backend import devices
        available=devices()
        if not available["gpu"]:
            pytest.skip(available["reason"])
    raw=np.random.default_rng(6430).integers(1,3500,(177,12,20),dtype=np.uint16)
    ticks=[(1700000000<<32)+i*64000+i//5*256 for i in range(len(raw))]
    path,*_=cine_factory(pixels=raw,ticks=ticks,timestamps=True,packed=False,bits=12)
    nodes=[new_node(a) for a in ("time_crop","calibration","alignment","point_extract","correlation_curve")]
    for n in nodes:
        n.setdefault("execution",{})
    nodes[1]["parameters"].update(curve="linear",input_max=4095)
    nodes[2]["parameters"].update(left=dict(x=0,y=0,width=10,height=12),right=dict(x=10,y=0,width=10,height=12),
                                    midline=[10.,10.],dx=0,dy=0,mirror_x=False,mirror_y=False,interpolation="nearest")
    nodes[2]["execution"]["backend"]="cpu"
    nodes[3]["parameters"].update(x=3,y=4,output="main")
    nodes[3]["execution"]["backend"]="auto" if device=="gpu" else "cpu"
    nodes[4]["parameters"].update(analysis="curve",window=31,min_samples=25,lag_list=[-37,-5,0,19,29])
    if suffix:
        nodes[4]["execution"]["precision"]="float32"
        gain=new_node("gain_offset")
        gain["parameters"].update(gain=1.123456789,offset=-.234567891)
        gain.setdefault("execution",{}).update(backend="cpu",precision="float32")
        curve=new_node("curve")
        curve["parameters"].update(curve="square",input_max=1.7)
        curve.setdefault("execution",{}).update(backend=device,precision="float64")
        nodes.extend([gain,curve])
    def engine(folder,optimize):
        reader=open_source(SourceSpec(str(path)))
        return reader,SequenceEngine(reader,deepcopy(nodes),workdir=folder,optimize=optimize,single=single,
                                     resources=dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=128,backend=device))
    pairs=[engine(tmp_path/str(opt),opt) for opt in (False,True)]
    old,new=(pair[1] for pair in pairs)
    try:
        stage=len(nodes)
        expected=np.asarray([old.get(stage,j) for j in range(old.output.length)])
        refs=[old.ref(stage,j) for j in range(old.output.length)]
        actual=np.asarray([value for _,value in new.iter_outputs(range(new.output.length))])
        np.testing.assert_array_equal(actual,expected)
        assert actual.dtype==expected.dtype==np.dtype(new.output.dtype)
        assert new.output.record()==old.output.record()
        for j,ref in enumerate(refs):
            assert new.ref(stage,j)==ref
        row=new.execution_metrics["lag_output_blocks"]["5"]
        assert row["output_blocks"]==1 and row["output_samples"]==5
        assert row["pair_updates"]==155 and row["kernel_calls"]==int(device=="gpu")
        assert new.main_store.block_size>1
        assert new.main_store.metrics["range_requests"]>0
        assert new.reference_metrics["lag_time_bulk_requests"]>0
        assert new.reference_metrics["lag_time_source_queries"]<=len(raw)
        assert new.reference_metrics.get("tree_rebuilds",0)==0
        assert new.device_records[nodes[2]["id"]]["actual"]=="cpu"
        if suffix:
            assert new.device_records[nodes[5]["id"]]["actual"]=="cpu"
            assert new.device_records[nodes[6]["id"]]["actual"]==device
        # Random access after delivery still computes the same scientific Q.
        np.testing.assert_array_equal(new.get(stage,2),expected[2])
    finally:
        for reader,e in pairs:
            e.close()
            reader.close()


def test_compound_time_editor_does_not_use_source_index_shortcut(tmp_path):
    from test_reference_index import make
    from test_axis_modules import node
    times=[float(i*i)*.001 for i in range(81)]
    nodes=[node("time_crop",first=2,last=80),node("time_sample",step=2),node("time_bin",size=3),
           node("alignment"),node("time_statistics"),node("correlation_curve",analysis="curve",window=3,lag_list=[-1,0,1])]
    reader,e=make(tmp_path,81,times,nodes)
    try:
        assert e._reference_times(5,2,5) is None
        for j in range(e.output.length):
            ref=e.ref(len(nodes),j)
            center=e.correlation_position(5,j)[0]
            assert ref["contribution"]["operation"]=="time_bin"
            assert ref["time"]==pytest.approx(np.cumsum([times[1+6*center+2*k] for k in range(3)])[-1]/3)
            assert ref["correlation"]["known_pairs"]==3
        assert e.reference_metrics["lag_time_source_queries"]>0
    finally:
        e.close()
