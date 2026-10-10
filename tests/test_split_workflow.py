from copy import deepcopy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from eip.modes.split_correlation.config import CorrelationConfig, PhotometryConfig
from eip.modes.split_correlation.photometry import Photometry
from eip.modes.split_correlation.workflow import build_geometry, intensity_curve, transform_array
from eip.modes.split_correlation.geometry import Geometry
from test_split_correlation import config
from test_split_web import client, opened, wait_job
from eip.web.records import saved_split_summary


def workflow_config(**kwargs):
    value=config(**kwargs).to_dict()
    value["workflow"]=[
        {"id":"cal","algorithm":"calibration","parameters":deepcopy(value["photometry"])},
        {"id":"align","algorithm":"alignment","parameters":deepcopy(value["geometry"])},
        {"id":"stats","algorithm":"statistics","parameters":{"normalization":value["photometry"]["normalization"]}},
        {"id":"g2","algorithm":"correlation","parameters":{k:value[k] for k in ("window","min_mean","min_samples")}},
    ]
    return value


def operator(identifier,algorithm="affine",**parameters):
    return {"id":identifier,"algorithm":algorithm,"parameters":parameters}


def test_presets_are_fixed_without_clipping_and_keep_gain_invariance():
    p=Photometry(PhotometryConfig(curve="linear",input_max=100))
    np.testing.assert_allclose(p.apply(np.array([-10,50,200]),"left"),[-.1,.5,2])
    np.testing.assert_allclose(intensity_curve(np.array([0.,25.,100.]),"sqrt",100),[0,.5,1])
    assert np.isnan(intensity_curve(np.array([-1.]),"sqrt",100)[0])
    assert PhotometryConfig.from_dict({"curve":"gamma","gamma":2}).gamma==2
    with pytest.raises(ValueError):
        PhotometryConfig.from_dict({"curve":"linear","lut_path":"calibration.npy"})


def test_module_instances_run_in_order_with_independent_parameters():
    value=workflow_config()
    value["workflow"][2:2]=[operator("mul",gain=2,offset=0),operator("add",gain=1,offset=3)]
    c=CorrelationConfig.from_dict(value)
    raw=np.full((4,10),10.)
    a=build_geometry(c,raw.shape).apply(raw,Photometry(c.photometry))[0]
    np.testing.assert_allclose(a,23.)
    value["workflow"][2:4]=list(reversed(value["workflow"][2:4]))
    c=CorrelationConfig.from_dict(value)
    b=build_geometry(c,raw.shape).apply(raw,Photometry(c.photometry))[0]
    np.testing.assert_allclose(b,26.)
    assert not np.array_equal(a,b)


def test_adjacent_calibration_optimization_keeps_lut_before_interpolation(tmp_path):
    lut=tmp_path/"response.npy"
    np.save(lut,np.arange(128,dtype=float)**2)
    value=workflow_config(photometry={"lut_path":str(lut),"black_left":3,"black_right":7})
    value["geometry"]["dx"]=.5
    value["geometry"]["interpolation"]="bilinear"
    value["workflow"][1]["parameters"]=deepcopy(value["geometry"])
    c=CorrelationConfig.from_dict(value)
    raw=np.random.default_rng(2026).integers(10,100,(4,10),dtype=np.uint16)
    expected=Geometry(c.geometry,raw.shape).apply(raw,Photometry(c.photometry))
    optimized=build_geometry(c,raw.shape).apply(raw,Photometry(c.photometry))
    for a,b in zip(expected,optimized):
        np.testing.assert_array_equal(a,b)
    # Inserting a raw-space identity node uses the general, non-fused path.
    value["workflow"].insert(1,operator("identity",gain=1,offset=0))
    c=CorrelationConfig.from_dict(value)
    unfused=build_geometry(c,raw.shape).apply(raw,Photometry(c.photometry))
    for a,b in zip(expected,unfused):
        np.testing.assert_array_equal(a,b)


def test_workflow_rejects_invalid_types_ids_and_parameters():
    value=workflow_config()
    value["workflow"][1],value["workflow"][2]=value["workflow"][2],value["workflow"][1]
    with pytest.raises(ValueError,match="数据类型"):
        CorrelationConfig.from_dict(value)
    value=workflow_config()
    value["workflow"].insert(2,operator("bad","gaussian",sigma=-1,radius=2))
    with pytest.raises(ValueError,match="sigma"):
        CorrelationConfig.from_dict(value)
    value=workflow_config()
    value["workflow"].insert(2,operator("cal",gain=2))
    with pytest.raises(ValueError,match="实例"):
        CorrelationConfig.from_dict(value)
    with pytest.raises(ValueError,match="零延迟"):
        config(lag_min=-4095,lag_max=4095,lag_step=2,lag_list=None)


def run_job(client,source,value,task="trial",name="实验 A"):
    response=client.post("/api/split/jobs",json={"source_id":source["id"],"config":value,"task":task,"name":name})
    assert response.status_code==200,response.text
    job=wait_job(client,response.json()["id"])
    assert job["status"]=="succeeded",job
    return job


def image(client,job):
    response=client.get(f"/api/split/jobs/{job['id']}/download?file=image&lag_index=0")
    assert response.status_code==200,response.text
    return tifffile.imread(io.BytesIO(response.content))


def test_workflow_preview_worker_naming_and_history(client,tmp_path):
    source,raw=opened(client,tmp_path)
    value=workflow_config(lag_list=[0],output_dtype="float64")
    value["workflow"].insert(2,operator("offset",gain=2.,offset=3.))
    cfg=CorrelationConfig.from_dict(value)
    preview=client.post("/api/split/geometry",json={"source_id":source["id"],"index":10,"geometry":value["geometry"],"workflow_config":value})
    assert preview.status_code==200,preview.text
    token=preview.json()["left"]["token"]
    pixel=client.get(f"/api/frames/{token}/pixel?x=2&y=1").json()["value"]
    assert pixel==pytest.approx(raw[10,1,2]*2+3)
    job=run_job(client,source,value)
    left,right=raw[8:13,:,:5].astype(float)*2+3,raw[8:13,:,5:].astype(float)*2+3
    expected=(left*right).mean(0)/(left.mean(0)*right.mean(0))
    np.testing.assert_allclose(image(client,job),expected,atol=1e-14)
    assert job["name"]==job["run"]["display_name"]=="实验 A"
    assert job["parameters"]["workflow"][2]["id"]=="offset"
    assert "q=+0" in job["summary"]
    renamed=client.post(f"/api/jobs/{job['id']}/rename",json={"name":"样本 A · W5 · τ0"})
    assert renamed.status_code==200,renamed.text
    manifest=json.loads((Path(job["directory"])/"run.json").read_text(encoding="utf-8"))
    assert manifest["display_name"]=="样本 A · W5 · τ0"
    for bad in ("",None,"x\ny","x"*161):
        assert client.post(f"/api/jobs/{job['id']}/rename",json={"name":bad}).status_code==400
    assert client.post("/api/split/restore",json={"path":str(Path(job["directory"])/"run.json")}).status_code==200
    current=client.get(f"/api/jobs/{job['id']}").json()
    assert current["name"]=="样本 A · W5 · τ0"


def test_postprocess_and_cache_follow_actual_module_order(client,tmp_path):
    source,_=opened(client,tmp_path)
    plain=workflow_config(lag_list=[0],output_dtype="float64")
    a=run_job(client,source,plain)
    post=deepcopy(plain)
    post["workflow"].append(operator("post",gain=2.,offset=3.))
    b=run_job(client,source,post,name="系数后处理")
    np.testing.assert_allclose(image(client,b),image(client,a)*2+3,atol=1e-14)
    assert a["run"]["statistics_cache"]==b["run"]["statistics_cache"]
    pre=deepcopy(plain)
    pre["workflow"].insert(2,operator("pre",gain=2.,offset=3.))
    c=run_job(client,source,pre)
    assert c["run"]["statistics_cache"]!=a["run"]["statistics_cache"]
    assert not np.allclose(image(client,b),image(client,c))
    assert b["run"]["result_semantics"]=="processed_g2"
    assert b["run"]["coefficient_reference"]==5


def test_curve_rejects_spatial_postprocessing_and_stays_independent(client,tmp_path):
    source,_=opened(client,tmp_path)
    value=workflow_config(points=[[2,1]])
    value["workflow"].append(operator("blur","gaussian",sigma=1,radius=2))
    response=client.post("/api/split/preflight",json={"source_id":source["id"],"config":value,"task":"curve"})
    assert response.status_code==400 and "空间平滑" in response.text
    value["workflow"].pop()
    job=run_job(client,source,value,task="curve",name="负正延迟曲线")
    assert job["run"]["lags"]==[-2,-1,0,1,2,3]
    assert not (Path(job["directory"])/"g2.npy").exists()
    assert client.get("/api/split/modules").json()["affine"]["repeatable"]


def test_historical_jobs_without_worker_payload_can_be_named(client,tmp_path):
    source,_=opened(client,tmp_path)
    job=run_job(client,source,workflow_config(lag_list=[0]))
    client.app.state.workbench.jobs[job["id"]].pop("payload")
    response=client.post(f"/api/jobs/{job['id']}/rename",json={"name":"历史运行名称"})
    assert response.status_code==200,response.text
    current=client.get(f"/api/jobs/{job['id']}").json()
    assert current["name"]=="历史运行名称"
    assert current["parameters"]["window"]==5
    assert current["source_specs"][0]["path"]==source["source"]["path"]
    legacy={**current["run"]["config"],"lag_list":None,"lag_min":-4095,"lag_max":4095,"lag_step":2}
    assert "历史参数" in saved_split_summary(legacy,"trial")


@pytest.mark.parametrize("estimator",["pixel_mean","intensity_sum"])
def test_postprocessing_of_point_and_roi_curves_is_explicit(client,tmp_path,estimator):
    source,_=opened(client,tmp_path)
    value=workflow_config(points=[[2,1]],roi={"x":1,"y":0,"width":3,"height":3},roi_estimator=estimator)
    a=run_job(client,source,value,task="curve")
    original=client.get(f"/api/split/jobs/{a['id']}").json()["curve_result"]
    value["workflow"].append(operator("post",gain=2,offset=3))
    b=run_job(client,source,value,task="curve")
    processed=client.get(f"/api/split/jobs/{b['id']}").json()["curve_result"]
    assert processed["result_semantics"]=="processed_g2"
    assert processed["coefficient_reference"]==5
    for before,after in zip(original["curves"],processed["curves"]):
        np.testing.assert_allclose(after["g2"],np.asarray(before["g2"])*2+3,atol=1e-14)


def test_gaussian_matches_explicit_two_dimensional_kernel_with_missing_samples():
    a=np.arange(1,10,dtype=float).reshape(3,3)
    a[1,1]=np.nan
    p={"sigma":.7,"radius":1,"target":"both"}
    actual=transform_array(a,"gaussian",p)
    expected=np.full_like(a,np.nan)
    for y in range(3):
        for x in range(3):
            if not np.isfinite(a[y,x]):continue
            total=weight=0.
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    sample=a[np.clip(y+dy,0,2),np.clip(x+dx,0,2)]
                    if np.isfinite(sample):
                        w=np.exp(-(dx*dx+dy*dy)/(2*.7**2))
                        total+=w*sample;weight+=w
            expected[y,x]=total/weight
    np.testing.assert_allclose(actual,expected,atol=1e-14,equal_nan=True)
