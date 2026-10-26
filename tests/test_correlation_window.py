from dataclasses import asdict
import numpy as np
import pytest

from eip.module_registry import new_node, catalog, normalize_nodes
from eip.module_contracts import compile_flow, source_contract
from eip.sequence_engine import SequenceEngine
from test_axis_modules import Reader
from test_sequence_web import client,open_tiff,submit
from test_gpu_pipeline_0_8_4 import device,compare


def node(algorithm, **parameters):
    value = new_node(algorithm)
    value["parameters"].update(parameters)
    return value


def aligned():
    return node("alignment",left=dict(x=0,y=0,width=2,height=2),right=dict(x=2,y=0,width=2,height=2),midline=[2,2])


def reference(left,right):
    valid = np.isfinite(left)&np.isfinite(right)
    left,right = left[valid],right[valid]
    if len(left)<2 or left.mean()<=1e-9 or right.mean()<=1e-9:
        return np.nan
    return np.mean(left*right)/(left.mean()*right.mean())


@pytest.mark.parametrize("lag",[-3,0,2])
def test_fixed_lag_windows_match_independent_pairs(tmp_path,lag):
    rng=np.random.default_rng(179)
    values=rng.uniform(.5,12,(21,2,4))
    values[8,0,0]=np.nan;values[11,0,2]=np.inf
    scan=node("correlation_window",window_list=[7,2,4,3,7],lag=lag,center=10)
    engine=SequenceEngine(Reader(values),[aligned(),node("point_extract",x=0,y=0),scan],workdir=tmp_path)
    try:
        assert engine.output.sequence_axis=="W" and engine.output.axes=="" and engine.output.length==4
        assert engine.output.record()["axis_meanings"]["W"]=="平均窗长 / 上游样本数"
        for index,(window,start,stop) in enumerate([(2,9,11),(3,8,11),(4,8,12),(7,6,13)]):
            expected=reference(values[start:stop,0,0],values[start+lag:stop+lag,0,2])
            assert engine.get(3,index)==pytest.approx(expected,nan_ok=True,rel=1e-12)
            ref=engine.ref(3,index)["correlation"]
            assert ref["window"]==window and ref["lag_samples"]==lag and ref["center_sample"]==9
            assert (ref["left_start"],ref["left_stop"],ref["right_start"],ref["right_stop"])==(start,stop,start+lag,stop+lag)
        # Random access and decreasing windows must not reuse the wrong accumulator.
        for index,start,stop in [(3,6,13),(1,8,11),(2,8,12),(0,9,11)]:
            assert engine.get(3,index)==pytest.approx(reference(values[start:stop,0,0],values[start+lag:stop+lag,0,2]),nan_ok=True)
    finally:
        engine.close()


def test_window_units_follow_upstream_binning(tmp_path):
    values=np.arange(21*2*4,dtype=float).reshape(21,2,4)+3
    scan=node("correlation_window",window_list=[2,3,4],lag=-1,center=4)
    engine=SequenceEngine(Reader(values),[aligned(),node("point_extract",x=1,y=1),node("time_bin",size=3),scan],workdir=tmp_path)
    reduced=values.reshape(7,3,2,4).mean(axis=1)
    try:
        for index,(window,start,stop) in enumerate([(2,3,5),(3,2,5),(4,2,6)]):
            assert engine.get(4,index)==pytest.approx(reference(reduced[start:stop,1,1],reduced[start-1:stop-1,1,3]))
            ref=engine.ref(4,index)["correlation"]
            assert ref["window"]==window and ref["known_pairs"]==window
            expected_times=[engine.ref(3,j-1)["time"]-engine.ref(3,j)["time"] for j in range(start,stop)]
            assert ref["lag_seconds"]==pytest.approx(np.mean(expected_times))
    finally:
        engine.close()


def test_pixel_mean_and_intensity_sum_are_distinct(tmp_path):
    rng=np.random.default_rng(91);values=rng.uniform(1,20,(15,2,4))
    scan=node("correlation_window",window_list=[3,7],lag=1,center=7)
    pipelines=[[aligned(),scan,node("time_statistics")],[aligned(),node("time_statistics",mode="sum"),scan]]
    results=[]
    for number,nodes in enumerate(pipelines):
        engine=SequenceEngine(Reader(values),nodes,workdir=tmp_path/str(number))
        try:
            actual=np.array([engine.get(3,i) for i in range(2)])
            expected=[]
            for start,stop in [(5,8),(3,10)]:
                left,right=values[start:stop,:,:2],values[start+1:stop+1,:,2:]
                expected.append(np.mean(np.mean(left*right,axis=0)/(left.mean(axis=0)*right.mean(axis=0))) if number==0 else reference(left.sum(axis=(1,2)),right.sum(axis=(1,2))))
            np.testing.assert_allclose(actual,expected)
            results.append(actual)
        finally:
            engine.close()
    assert not np.allclose(*results)


@pytest.mark.parametrize("parameters",[dict(window_min=5,window_max=3),dict(window_list=[]),dict(window_list=[True]),dict(window_step=0),dict(window_list=[1],min_samples=2),dict(window_min=1,window_max=10000,window_step=1)])
def test_invalid_window_parameters_fail(parameters):
    with pytest.raises(ValueError):
        normalize_nodes([node("correlation_window",**parameters)],check_types=False)


def test_bounds_input_type_and_window_axis_are_explicit():
    source=source_contract(asdict(Reader(np.ones((15,2,4))).meta))
    with pytest.raises(ValueError,match="对齐双路"):
        compile_flow([node("correlation_window",window_max=7)],source)
    with pytest.raises(ValueError,match="容纳|中心"):
        compile_flow([aligned(),node("correlation_window",window_max=13,lag=4)],source)
    with pytest.raises(ValueError,match="不是时间"):
        compile_flow([aligned(),node("correlation_window",window_max=7),node("time_sample")],source)
    assert catalog()["correlation_window"]["category"]=="algorithm"


def test_blank_center_is_identical_in_trial_and_formal_context(tmp_path):
    values=np.arange(21*2*4,dtype=float).reshape(21,2,4)+2
    scan=node("correlation_window",window_list=[2,3,7],lag=-1)
    centres=[]
    for sample in (0,20,None):
        engine=SequenceEngine(Reader(values),[aligned(),node("point_extract"),scan],workdir=tmp_path/str(sample),center=sample,single=sample is not None)
        try:
            centres.append([engine.ref(3,j)["correlation"]["center_sample"] for j in range(3)])
        finally:
            engine.close()
    assert centres[0]==centres[1]==centres[2]


def test_irregular_time_differences_are_recorded_per_window(tmp_path):
    values=np.arange(17*2*4,dtype=float).reshape(17,2,4)+2
    times=np.arange(17,dtype=float)**3*.001
    scan=node("correlation_window",window_list=[2,3,7],lag=-2,center=9)
    engine=SequenceEngine(Reader(values,times),[aligned(),node("point_extract"),scan],workdir=tmp_path)
    try:
        for index,(start,stop) in enumerate([(8,10),(7,10),(5,12)]):
            ref=engine.ref(3,index)["correlation"]
            deltas=times[start-2:stop-2]-times[start:stop]
            assert ref["lag_seconds"]==pytest.approx(deltas.mean())
            assert ref["lag_seconds_min"]==pytest.approx(deltas.min())
            assert ref["lag_seconds_max"]==pytest.approx(deltas.max())
    finally:
        engine.close()


@pytest.mark.parametrize("lag",[-2,0,3])
def test_window_cpu_and_device_flow_match(tmp_path,device,lag):
    values=np.arange(21*2*4,dtype=float).reshape(21,2,4)+2
    scan=node("correlation_window",window_list=[2,3,4,7],lag=lag,center=10)
    compare(tmp_path,values,[aligned(),node("point_extract"),scan],indices=[0,1,2,3,1,2])


def test_saved_curve_csv_svg_and_comparison_keep_real_windows(client,tmp_path):
    import csv
    import io
    from pathlib import Path
    values=np.arange(21*2*4,dtype=np.uint16).reshape(21,2,4)+7
    source=open_tiff(client,tmp_path,values)
    scan=node("correlation_window",window_list=[2,3,4,7],lag=-2,center=10)
    tap=node("data_tap",name="窗长曲线",export_format="csv")
    nodes=[aligned(),node("point_extract",x=1,y=1),scan,tap]
    job=submit(client,source,nodes)
    assert job["status"]=="succeeded",job
    expected=[reference(values[start:stop,1,1],values[start-2:stop-2,1,3]) for start,stop in [(9,11),(8,11),(8,12),(6,13)]]
    identifier=job["id"]
    result=client.get(f"/api/jobs/{identifier}/download?file=array")
    np.testing.assert_allclose(np.load(io.BytesIO(result.content)),expected)
    table=list(csv.DictReader(io.StringIO(client.get(f"/api/jobs/{identifier}/download?file=csv").content.decode("utf-8-sig"))))
    assert [int(row["window_samples"]) for row in table]==[2,3,4,7]
    assert {int(row["lag_samples"]) for row in table}=={-2}
    np.testing.assert_allclose([float(row["value_1"]) for row in table],expected)
    assert "Average window W" in client.get(f"/api/jobs/{identifier}/download?file=svg").text
    parent=job["run"]["results"][0];side=parent["side_outputs"][0]
    exported=Path(job["directory"])/parent["output_directory"]/side["output_directory"]/"export.csv"
    side_rows=list(csv.DictReader(io.StringIO(exported.read_text(encoding="utf-8-sig"))))
    assert [int(row["window_samples"]) for row in side_rows]==[2,3,4,7]
    selected=dict(job_id=identifier,source=0)
    comparison=client.post("/api/curves/compare",json=dict(items=[selected],x="samples"))
    assert comparison.status_code==200,comparison.text
    assert comparison.json()["kind"]=="W" and comparison.json()["xs"]==[2,3,4,7]
    np.testing.assert_allclose(comparison.json()["curves"][0]["values"],expected)
    assert client.post("/api/curves/compare",json=dict(items=[selected],x="time")).status_code==400
    trial=submit(client,source,nodes,mode="trial")
    assert trial["status"]=="succeeded",trial
    np.testing.assert_allclose(np.load(io.BytesIO(client.get(f"/api/jobs/{trial['id']}/download?file=array").content)),expected)
    lag_scan=node("correlation_curve",window=3,lag_list=[2,3,4,7],center=10)
    other=submit(client,source,[aligned(),node("point_extract",x=1,y=1),lag_scan])
    assert other["status"]=="succeeded",other
    rejected=client.post("/api/curves/compare",json=dict(items=[selected,dict(job_id=other["id"],source=0)]))
    assert rejected.status_code==400 and "横轴" in rejected.text


def test_cuda_window_scan_when_available(tmp_path):
    from eip.modes.split_correlation.backend import devices
    if not devices()["gpu"]:
        pytest.skip("需要可用 CUDA；主机适配器不替代真机验证")
    values=np.arange(21*2*4,dtype=float).reshape(21,2,4)+2
    compare(tmp_path,values,[aligned(),node("point_extract"),node("correlation_window",window_list=[2,3,4,7],lag=-2,center=10)])
