"""Corresponding paired traces and saved one-dimensional comparison only."""
import csv
from dataclasses import replace
import io
import json
import zipfile
import numpy as np
import pytest

from eip.module_contracts import Contract,compile_flow
from eip.module_registry import new_node,catalog
from eip.sequence_engine import SequenceEngine
from test_axis_modules import Reader
from test_sequence_web import client,open_tiff,submit
from test_gpu_pipeline_0_8_4 import HostDevice


def trace(**p):
    n=new_node("paired_trace");n["parameters"].update(p);return n


@pytest.mark.parametrize("mode",["mean","min","max","variance","std","sum","quantile","count"])
@pytest.mark.parametrize("gpu",[False,True])
def test_corresponding_region_retains_two_temporal_values(tmp_path,monkeypatch,mode,gpu):
    monkeypatch.setattr("eip.modes.split_correlation.backend.Backend",HostDevice)
    values=np.array([[[[1.,2.],[3.,4.]],[[11.,12.],[13.,14.]]],[[[2.,np.nan],[4.,6.]],[[12.,22.],[14.,16.]]]])
    reader=Reader(values);reader.meta=replace(reader.meta,contract=Contract("PYX",(2,2,2),2).record())
    node=trace(selection="region",width=2,height=2,mode=mode,quantile=.25,output="main")
    node["execution"]={"backend":"gpu" if gpu else "cpu"}
    e=SequenceEngine(reader,[node],workdir=tmp_path)
    try:
        assert e.output.record()["full_axes"]=="TP" and e.output.length==2
        for t in range(2):
            mask=np.isfinite(values[t]).all(axis=0)
            expected=[]
            for side in values[t]:
                finite=side[mask]
                expected.append(len(finite) if mode=="count" else np.quantile(finite,.25) if mode=="quantile" else getattr(np,"var" if mode=="variance" else mode)(finite))
            np.testing.assert_allclose(e.get(1,t),expected)
        assert e.ref(1,1)["index"]==1
    finally:e.close()


def test_point_independent_validity_and_multiple_side_nodes(tmp_path):
    values=np.arange(3*2*2*2,dtype=float).reshape(3,2,2,2)
    values[1,1,0,0]=np.nan
    reader=Reader(values);reader.meta=replace(reader.meta,contract=Contract("PYX",(2,2,2),3).record())
    first,second=trace(x=0,y=0),trace(x=1,y=1,validity="independent")
    e=SequenceEngine(reader,[first,second],workdir=tmp_path)
    try:
        assert e.output.axes=="PYX" and len(e.side_outputs())==2
        np.testing.assert_array_equal(e.get(2,0),values[0])
        assert np.isnan(e.get_side_output(first["id"]+":data",1)).all()
        np.testing.assert_array_equal(e.get_side_output(second["id"]+":data",0),[3,7])
    finally:e.close()


def test_trace_requires_paired_intensity_and_valid_selection():
    for c in [Contract("YX",(3,4),5),Contract("PYX",(2,3,4),5,semantic="g2"),Contract("PYX",(2,3,4),5,sequence_axis="Q")]:
        with pytest.raises(ValueError,match="配准"):
            compile_flow([trace()],c)
    with pytest.raises(ValueError,match="ROI"):
        compile_flow([trace(x=4)],Contract("PYX",(2,3,4),5))
    assert catalog()["paired_trace"]["category"]=="algorithm"
    assert catalog()["paired_trace"]["formula"]["latex"]
    alignment=new_node("alignment",shape=(3,8));alignment["execution"]={"memory_mb":None,"backend":"inherit"}
    assert compile_flow([alignment,trace()],Contract("YX",(3,8),5))["errors"]==[]


def saved(client,tmp_path,name,values,*,times=None,axis="T",axes="",mappings=None,meaning="fixture",semantic="intensity"):
    values=np.asarray(values,dtype=float);c=Contract(axes,values.shape[1:],len(values),sequence_axis=axis,meaning=meaning,semantic=semantic).record()
    folder=tmp_path/name;part=folder/"source_000";part.mkdir(parents=True)
    np.save(part/"values.npy",values)
    offsets=[]
    with (part/"references.jsonl").open("wb") as file:
        for i in range(len(values)):
            offsets.append(file.tell());r=dict(index=i,sample_index=i,time=times[i] if times is not None else i*.1)
            if axis=="Q":r["correlation"]={"lag_samples":i-1,"lag_seconds":r["time"]}
            file.write((json.dumps(r)+"\n").encode())
    np.save(part/"reference_offsets.npy",np.array(offsets,dtype=np.uint64))
    (part/"dataset.json").write_text(json.dumps(dict(schema_version=1,status="succeeded",contract=c)),encoding="utf-8")
    item=dict(status="succeeded",contract=c,output_directory="source_000",name=name,mappings=mappings or {},side_outputs=[])
    workbench=client.app.state.workbench
    workbench.jobs[name]=dict(id=name,name=name,engine="sequence",status="succeeded",mode="run",directory=str(folder),run=dict(app_version="fixture",results=[item],spec=dict(sources=[])))
    return dict(job_id=name,source=0,side_output=None,column=0)


def test_compare_full_coordinates_even_when_display_is_sampled(client,tmp_path):
    a=saved(client,tmp_path,"one",np.arange(4101))
    times=np.arange(4101)*.1;times[1]+=.01  # index 1 is not on the display sampling grid
    b=saved(client,tmp_path,"two",np.arange(4101)*2,times=times)
    response=client.post("/api/curves/compare",json=dict(items=[a,b],x="time"))
    assert response.status_code==400 and "坐标不同" in response.json()["detail"]
    response=client.post("/api/curves/compare",json=dict(items=[a,b],x="samples"))
    data=response.json();assert response.status_code==200 and data["downsampled"]
    assert len(data["xs"])<=2001 and data["warnings"]
    assert data["curves"][1]["values"][-1]==8200
    b["column"]=3
    assert client.post("/api/curves/compare",json=dict(items=[a,b])).status_code==400


def test_compare_rejects_dimensions_axis_and_duplicate_results(client,tmp_path):
    a=saved(client,tmp_path,"one",[1,2,3]);b=saved(client,tmp_path,"short",[1,2]);q=saved(client,tmp_path,"lag",[1,2,3],axis="Q")
    for entries in [[a,a],[a,b],[a,q],[],[a]*33]:
        assert client.post("/api/curves/compare",json=dict(items=entries)).status_code==400
    assert len(client.get("/api/curves/candidates").json())==3


def test_component_comparison_checks_distances(client,tmp_path):
    a=saved(client,tmp_path,"line1",[[2,4,8]],axis="N",axes="R",mappings={"line":dict(axis="R",coordinates=[0,2,4])})
    b=saved(client,tmp_path,"line2",[[3,5,9]],axis="N",axes="R",mappings={"line":dict(axis="R",coordinates=[0,3,6])})
    assert client.post("/api/curves/compare",json=dict(items=[a,b],mode="component")).status_code==400
    multi=saved(client,tmp_path,"multi",np.ones((1,2,3)),axis="N",axes="PR")
    assert client.post("/api/curves/compare",json=dict(items=[multi],mode="component")).status_code==400


def test_histogram_uses_own_module_edges_and_checks_comparison_bins(client,tmp_path):
    maps={"first":dict(axis="B",edges=[0,1,2]),"later":dict(axis="B",edges=[0,2,4])}
    a=saved(client,tmp_path,"hist",[[2,3]],axes="B",semantic="distribution",mappings=maps)
    client.app.state.workbench.jobs["hist"]["run"]["results"][0]["node_id"]="first"
    data=client.post("/api/curves/compare",json=dict(items=[a],mode="component")).json()
    assert data["xs"]==[.5,1.5] and data["edges"]==[0,1,2]
    b=saved(client,tmp_path,"other",[[2,3]],axes="B",semantic="distribution",mappings={"other":maps["later"]})
    assert client.post("/api/curves/compare",json=dict(items=[a,b],mode="component")).status_code==400


def test_unknown_time_and_missing_values_are_explicit(client,tmp_path):
    a=saved(client,tmp_path,"unknown",[1,np.nan,3],times=[None,None,None])
    data=client.post("/api/curves/compare",json=dict(items=[a])).json()
    assert data["curves"][0]["values"]==[1,None,3] and data["warnings"]
    assert client.post("/api/curves/compare",json=dict(items=[a],x="time")).status_code==400
    q1=saved(client,tmp_path,"lag1",[1,2,3],axis="Q",times=[-.1,0,.1])
    q2=saved(client,tmp_path,"lag2",[2,3,4],axis="Q",times=[-.2,0,.2])
    data=client.post("/api/curves/compare",json=dict(items=[q1,q2])).json()
    assert data["warnings"]==["按有符号延迟样本对齐；各组实际配对时差不同"]


def test_comparison_package_exports_all_values_and_recipe(client,tmp_path):
    a=saved(client,tmp_path,"pair",np.arange(8202).reshape(4101,2),axes="P")
    b=dict(a,column=1)
    value=dict(items=[dict(a,name="左值",visible=True),dict(b,name="右值",visible=False)],view=dict(focus="left"))
    r=client.post("/api/curves/export",json=value);assert r.status_code==200,r.text
    response=client.get(r.json()["url"]);assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        recipe=json.loads(archive.read("comparison.json"));assert recipe["full_values"] and len(recipe["items"])==2
        records=list(csv.reader(io.StringIO(archive.read("values.csv").decode("utf-8-sig"))))
        assert len(records)==4102 and records[0]==["coordinate","左值","右值"]
        assert records[-1]==["4101","8200.0","8201.0"]
    assert client.get(r.json()["url"]).status_code==400
    assert not list((tmp_path/".eip-cache/downloads").glob("*.zip"))


def test_real_job_inspection_and_saved_paired_side_result(client,tmp_path):
    values=np.arange(5*3*8,dtype=np.uint16).reshape(5,3,8)
    source=open_tiff(client,tmp_path,values)
    alignment=new_node("alignment",shape=(3,8));point=trace(x=1,y=1)
    roi=trace(selection="region",x=0,y=0,width=2,height=2,mode="variance")
    response=client.post("/api/modules/inspect",json=dict(source_id=source["id"],steps=[alignment,point],side_output=point["id"]+":data",sample_index=2))
    assert response.status_code==200,response.text
    assert response.json()["values"]==[[57.,61.]]
    assert client.get("/api/jobs").json()==[]
    job=submit(client,source,[alignment,point,roi]);assert job["status"]=="succeeded",job
    entries=[dict(job_id=job["id"],source=0,side_output=n["id"]+":data",column=p) for n in (point,roi) for p in (0,1)]
    response=client.post("/api/curves/compare",json=dict(items=entries,x="time"))
    assert response.status_code==200,response.text
    data=response.json()
    assert len(data["curves"])==4 and data["xs"]==[0,.01,.02,.03,.04]
    assert data["curves"][0]["values"]==[9,33,57,81,105]
    assert data["curves"][2]["values"]==[16.25]*5
    assert len(data["warnings"])==1  # different scientific quantities share an unnormalised y axis


def test_cuda_paired_trace_when_available(tmp_path):
    from eip.modes.split_correlation.backend import devices
    from test_gpu_pipeline_0_8_4 import compare
    if not devices()["gpu"]:
        pytest.skip("需要可用 CUDA；主机适配器不替代真机验证")
    values=np.arange(21*4*8,dtype=float).reshape(21,4,8)+1
    values[2,1,6]=np.nan
    alignment=new_node("alignment",shape=(4,8))
    compare(tmp_path,values,[alignment,trace(x=1,y=1,output="main")])
    compare(tmp_path,values,[alignment,trace(selection="region",width=3,height=3,mode="variance",output="main")])
