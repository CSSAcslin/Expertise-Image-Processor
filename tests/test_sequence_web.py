from pathlib import Path
import io
import time
import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from eip.module_registry import new_node
from eip.web.app import create_app


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        client.headers["X-EIP-Token"] = client.get("/api/state").json()["session_token"]
        yield client


def wait(client,id):
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        value = client.get(f"/api/jobs/{id}").json()
        if value["status"] in {"succeeded","completed_with_errors","failed","cancelled"}:
            return value
        time.sleep(.05)
    pytest.fail("worker timeout")


def open_tiff(client,tmp_path,values):
    path = tmp_path/"source.tif"
    tifffile.imwrite(path,values,photometric="minisblack",metadata={"axes":"TYX"})
    response = client.post("/api/sources/open",json=dict(path=str(path),interval=.01))
    assert response.status_code==200,response.text
    return response.json()["source"]


def submit(client,source,nodes,**kw):
    payload=dict(contract_version=3,mode="run",source_ids=[source["id"]],steps=nodes)
    payload.update(kw)
    response = client.post("/api/jobs",json=payload)
    assert response.status_code==200,response.text
    return wait(client,response.json()["id"])


def test_matrix_export_and_shape_error(client,tmp_path):
    a = np.arange(5*4*6,dtype=np.uint16).reshape(5,4,6)
    source = open_tiff(client,tmp_path,a)
    nodes = [new_node("time_crop"),new_node("spatial_roi")]
    nodes[0]["parameters"].update(first=2,last=5)
    nodes[1]["parameters"].update(x=1,y=1,width=3,height=2)
    nodes += [new_node("time_sample")]
    job = submit(client,source,nodes)
    assert job["status"]=="succeeded",job
    output = np.load(io.BytesIO(client.get(f"/api/jobs/{job['id']}/download?file=array").content),allow_pickle=False)
    np.testing.assert_array_equal(output,a[[1,3],1:3,1:4])
    frame = client.post(f"/api/jobs/{job['id']}/sequence",json=dict(index=1)).json()
    assert frame["reference"]["index"]==3
    assert frame["frame"]["shape"]==[2,3]
    nodes[1]["parameters"]["mode"]="mean"
    nodes.append(new_node("gaussian"))
    plan = client.post("/api/modules/plan",json=dict(source_id=source["id"],steps=nodes)).json()
    assert "Y/X" in plan["errors"][0]
    assert client.post("/api/jobs",json=dict(contract_version=3,source_ids=[source["id"]],steps=nodes)).status_code==400
    # Editing an incomplete flow must not poison source preview.
    assert client.post(f"/api/sources/{source['id']}/frame",json=dict(action="index",index=4)).status_code==200


def test_signal_export_promote_and_reprocess(client,tmp_path):
    a = np.arange(5*2*4,dtype=np.uint16).reshape(5,2,4)
    source = open_tiff(client,tmp_path,a)
    nodes = [new_node("spatial_roi")]
    nodes[0]["parameters"]["mode"]="mean"
    job = submit(client,source,nodes)
    assert job["status"]=="succeeded",job
    data = client.post(f"/api/jobs/{job['id']}/sequence",json={}).json()
    assert data["kind"]=="signals" and data["contract"]["shape"]==[]
    np.testing.assert_allclose(np.asarray(data["values"])[:,0],a.mean(axis=(1,2)))
    for file in ("png","svg","csv","references","array"):
        assert client.get(f"/api/jobs/{job['id']}/download?file={file}").status_code==200
    result = next(s for s in client.get("/api/state").json()["sources"] if s.get("run_id")==job["id"])
    response = client.post(f"/api/sources/{result['id']}/promote",json={})
    assert response.status_code==200,response.text
    derived = response.json()["source"]
    assert derived["meta"]["contract"]["axes"]==""
    assert len(derived["lineage"])==1
    affine = new_node("affine")
    affine["parameters"]["gain"] = 2
    follow = submit(client,derived,[affine])
    assert follow["status"]=="succeeded",follow
    follow_data = client.post(f"/api/jobs/{follow['id']}/sequence",json={}).json()
    np.testing.assert_allclose(np.asarray(follow_data["values"])[:,0],a.mean(axis=(1,2))*2)


def test_scalar_signed_curve_from_edited_sequence(client,tmp_path):
    a = np.zeros((13,1,2),dtype=np.float64)
    a[:,0,0] = np.arange(13)+1
    a[:,0,1] = np.arange(13)**2+1
    source = open_tiff(client,tmp_path,a)
    sample,align,roi,corr = [new_node(k) for k in ("time_sample","alignment","spatial_roi","correlation")]
    align["parameters"].update(left=dict(x=0,y=0,width=1,height=1),right=dict(x=1,y=0,width=1,height=1),midline=[1,1])
    roi["parameters"]["mode"] = "mean"
    corr["parameters"].update(analysis="curve",center=4,window=3,lag_list=[-1,0,1])
    job = submit(client,source,[sample,align,roi,corr])
    assert job["status"]=="succeeded",job
    data = client.post(f"/api/jobs/{job['id']}/sequence",json={}).json()
    assert data["contract"]["full_axes"]=="Q"
    for i,q in enumerate([-1,0,1]):
        left,right = a[2*np.arange(2,5),0,0],a[2*(np.arange(2,5)+q),0,1]
        assert data["values"][i][0]==pytest.approx(np.mean(left*right)/(left.mean()*right.mean()))
        assert data["references"][i]["correlation"]["lag_seconds"]==pytest.approx(q*.02)


def test_frame_plugin_dtype_is_retained(client,tmp_path):
    source = open_tiff(client,tmp_path,np.ones((3,4,6),dtype=np.uint16)*12)
    manifest = Path(__file__).resolve().parents[1]/"examples/plugins/invert.json"
    spec = client.post("/api/plugins/register",json=dict(path=str(manifest))).json()
    node = dict(algorithm=spec["id"],parameters={p["key"]:p["default"] for p in spec["parameters"]},manifest=spec["manifest"],code_hash=spec["code_hash"],version=spec["version"])
    job = submit(client,source,[node])
    assert job["status"]=="succeeded",job
    arr = np.load(Path(job["directory"])/"source_000/values.npy",allow_pickle=False)
    assert arr.dtype==np.dtype("float32")
    assert job["run"]["results"][0]["contract"]["dtype"]=="float32"


def test_vfr_time_crop_and_bin_use_decoded_timestamps(client,tmp_path,vfr_video):
    path,stamps = vfr_video
    response = client.post("/api/sources/open",json=dict(path=str(path)))
    assert response.status_code==200,response.text
    crop,binning = new_node("time_crop"),new_node("time_bin")
    crop["parameters"].update(unit="seconds",start=stamps[1],end=stamps[4])
    job = submit(client,response.json()["source"],[crop,binning])
    assert job["status"]=="succeeded",job
    arr = np.load(Path(job["directory"])/"source_000/values.npy",allow_pickle=False)
    np.testing.assert_array_equal(arr,np.full((1,*arr.shape[1:]),2500.))
    data = client.post(f"/api/jobs/{job['id']}/sequence",json={}).json()
    assert data["reference"]["time"]==pytest.approx((stamps[1]+stamps[2])/2)
    assert data["reference"]["index"] is None
    assert data["reference"]["source_span"]==[1,3]


def test_typed_result_restore_and_project_roundtrip(client,tmp_path):
    a = np.arange(5*2*4,dtype=np.uint16).reshape(5,2,4)
    source = open_tiff(client,tmp_path,a)
    roi = new_node("spatial_roi")
    roi["parameters"]["mode"] = "mean"
    job = submit(client,source,[roi],name="区域均值")
    assert job["status"]=="succeeded",job
    manifest = Path(job["directory"])/"run.json"
    project = tmp_path/"typed-project.json"
    # Incomplete editing is saveable; it must not affect saved numeric results.
    invalid = [roi,new_node("gaussian")]
    response = client.post("/api/project/save",json=dict(path=str(project),view=dict(steps=invalid,web_module_contract=3)))
    assert response.status_code==200,response.text
    with TestClient(create_app(tmp_path/"restored")) as fresh:
        fresh.headers["X-EIP-Token"] = fresh.get("/api/state").json()["session_token"]
        restored = fresh.post("/api/sequence/restore",json=dict(path=str(manifest)))
        assert restored.status_code==200,restored.text
        data = fresh.post(f"/api/jobs/{job['id']}/sequence",json={}).json()
        np.testing.assert_array_equal(np.asarray(data["values"])[:,0],a.mean(axis=(1,2)))
        loaded = fresh.post("/api/project/load",json=dict(path=str(project)))
        assert loaded.status_code==200,loaded.text
        assert loaded.json()["view"]["steps"]==invalid
        sources = loaded.json()["state"]["sources"]
        result = next(s for s in sources if s.get("run_id")==job["id"])
        assert result["meta"]["contract"]["axes"]==""
        frame = fresh.post(f"/api/sources/{result['id']}/frame",json=dict(action="index",index=2))
        assert frame.status_code==200,frame.text
        assert frame.json()["signal_values"]==[a[2].mean()]


def test_stage_calibration_dependency_and_geometry_tool(client,tmp_path):
    a = np.tile(np.array([[10.,10.,20.,20.]]),(3,2,1))
    source = open_tiff(client,tmp_path,a)
    calibrate = new_node("calibration")
    calibrate["parameters"].update(black_left=1.,black_right=2.)
    align = new_node("alignment",shape=source["meta"]["shape"])
    job = submit(client,source,[calibrate],mode="trial",preview_dependencies=[align])
    assert job["status"]=="succeeded",job
    values = np.load(Path(job["directory"])/"source_000/values.npy",allow_pickle=False)
    np.testing.assert_array_equal(values,a[:1]-np.array([1.,1.,2.,2.]))
    assert job["run"]["results"][0]["modules"]["preview_dependencies"][0]["id"]==align["id"]
    response = client.post("/api/split/geometry",json=dict(contract_version=3,source_id=source["id"],module_prefix=[calibrate],module_dependencies=[align],geometry=align["parameters"],sample_index=0))
    assert response.status_code==200,response.text
    view = response.json()
    assert view["left"]["statistics"]["mean"]==9.
    assert view["right"]["statistics"]["mean"]==18.
    response = client.post("/api/jobs",json=dict(contract_version=3,mode="run",source_ids=[source["id"]],steps=[calibrate],preview_dependencies=[align]))
    assert response.status_code==400
