from pathlib import Path
import io
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest
import tifffile

from eip.web.app import create_app
from test_split_correlation import config


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        client.headers["X-EIP-Token"]=client.get("/api/state").json()["session_token"]
        yield client


def opened(client,tmp_path):
    rng=np.random.default_rng(41)
    array=rng.uniform(30,100,(24,4,10)).astype(np.float32)
    path=tmp_path/"web_split.tif"
    tifffile.imwrite(path,array,photometric="minisblack",metadata={"axes":"TYX"})
    data=client.post("/api/sources/open",json={"path":str(path),"interval":.01})
    assert data.status_code==200,data.text
    return data.json()["source"],array


def wait_job(client,identifier):
    deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        value=client.get(f"/api/jobs/{identifier}").json()
        if value["status"] in {"succeeded","failed","cancelled"}:
            return value
        time.sleep(.04)
    pytest.fail("split job timed out")


def submit(client,source,cfg,task="trial"):
    response=client.post("/api/split/jobs",json={"source_id":source["id"],"task":task,"config":cfg.to_dict(),"calibration_index":3})
    assert response.status_code==200,response.text
    job=wait_job(client,response.json()["id"])
    assert job["status"]=="succeeded",job
    return job


def test_geometry_preflight_signed_lag_slice_and_security(client,tmp_path):
    source,array=opened(client,tmp_path)
    geometry=client.post("/api/split/geometry",json={"source_id":source["id"],"index":10})
    assert geometry.status_code==200,geometry.text
    assert geometry.json()["mapping"]["shape"]==[4,5]
    assert geometry.json()["left"]["origin"]==[0,0]
    assert client.post("/api/split/geometry",headers={"X-EIP-Token":"wrong"},json={}).status_code==403
    cfg=config()
    estimate=client.post("/api/split/preflight",json={"source_id":source["id"],"config":cfg.to_dict(),"task":"run"})
    assert estimate.status_code==200,estimate.text
    assert estimate.json()["lags"]==[-2,-1,0,1,2,3]
    assert estimate.json()["background_scope"]=="all_source_frames"
    job=submit(client,source,cfg)
    assert job["run"]["provenance"]["calibration_reference"]["index"]==3
    assert job["run"]["provenance"]["calibration_reference"]["time"]==.03
    identifier=job["id"]
    selected=client.post(f"/api/split/jobs/{identifier}/slice",json={"time_index":0,"lag_index":0})
    assert selected.status_code==200,selected.text
    ref=selected.json()["window_ref"]
    assert ref["lag_frames"]==-2 and ref["right_start"]==6
    image=client.get(f"/api/split/jobs/{identifier}/download?file=image&lag_index=0")
    assert image.status_code==200
    pixels=tifffile.imread(io.BytesIO(image.content))
    left,right=array[8:13,:,:5],array[6:11,:,5:]
    expected=(left.astype(float)*right).mean(axis=0)/(left.mean(axis=0,dtype=float)*right.mean(axis=0,dtype=float))
    np.testing.assert_allclose(pixels,expected,rtol=7e-8)
    assert client.get(f"/api/split/jobs/{identifier}/download?file=../run.json").status_code==400
    assert client.post(f"/api/split/jobs/{identifier}/slice",json={"lag_index":-1}).status_code==400


def test_independent_curves_statistics_and_exports(client,tmp_path):
    source,_=opened(client,tmp_path)
    cfg=config(points=[[2,1]],roi={"x":1,"y":0,"width":3,"height":3})
    job=submit(client,source,cfg,task="curve")
    result=client.get(f"/api/split/jobs/{job['id']}").json()
    assert len(result["curve_result"]["curves"])==2
    assert result["run"]["numeric_semantics"]=="sensor_codes"
    assert not (Path(job["directory"])/"g2.npy").exists()
    for kind in ("csv","png","svg","curve_json","manifest"):
        export=client.get(f"/api/split/jobs/{job['id']}/download?file={kind}")
        assert export.status_code==200,export.text
    stats=client.post(f"/api/split/jobs/{job['id']}/statistics",json={"view":"fluctuation_left","index":10})
    assert stats.status_code==200,stats.text
    assert stats.json()["frame"]["dtype"]=="float64"
    assert client.post(f"/api/split/jobs/{job['id']}/promote",json={"lag_index":0}).status_code==400


def test_lag_sequence_chaining_project_and_reopen(client,tmp_path):
    source,_=opened(client,tmp_path)
    job=submit(client,source,config(stride=2),task="run")
    identifier=job["id"]
    promoted=client.post(f"/api/split/jobs/{identifier}/promote",json={"lag_index":0})
    assert promoted.status_code==200,promoted.text
    entry=promoted.json()["source"]
    assert entry["source"]["kind"]=="correlation"
    assert entry["lineage"][-1]["lag_frames"]==-2
    assert entry["meta"]["frame_count"]>1
    frame=client.post(f"/api/sources/{entry['id']}/frame",json={"action":"index","index":1})
    assert frame.status_code==200,frame.text
    following=client.post("/api/jobs",json={"mode":"run","source_ids":[entry["id"]],"steps":[],"selection":{"mode":"all"}})
    assert following.status_code==200,following.text
    exported=wait_job(client,following.json()["id"])
    assert exported["status"]=="succeeded",exported
    manifest=client.get(f"/api/jobs/{exported['id']}/download").json()
    assert manifest["spec"]["provenance"][0]["lineage"][-1]["window"]==5
    project=tmp_path/"mode_project.json"
    saved=client.post("/api/project/save",json={"path":str(project),"view":{"steps":[],"analysis_mode":"split_correlation","split_mode":{"records":{source["id"]:{"config":config().to_dict()}}}}})
    assert saved.status_code==200,saved.text
    data=__import__("json").loads(project.read_text(encoding="utf-8"))
    assert str(Path(job["directory"])/"run.json") in data["split_result_paths"]
    deadline=time.monotonic()+4
    while True:
        restored=client.post("/api/project/load",json={"path":str(project)})
        if restored.status_code==200 or time.monotonic()>deadline:
            break
        time.sleep(.1)
    assert restored.status_code==200,restored.text
    assert restored.json()["view"]["analysis_mode"]=="split_correlation"
    assert client.post("/api/split/restore",json={"path":str(Path(job["directory"])/"run.json")}).status_code==200


def test_frame_index_and_geometry_changes_do_not_fake_results(client,tmp_path):
    source,_=opened(client,tmp_path)
    value=config().to_dict()
    value["lag_min"]=-10_000_000
    assert client.post("/api/split/preflight",json={"source_id":source["id"],"config":value}).status_code==400
    value=config(center=0).to_dict()
    assert client.post("/api/split/jobs",json={"source_id":source["id"],"task":"curve","config":value}).status_code==400


def test_saved_lut_keeps_historical_fluctuation_reproducible(client,tmp_path):
    path,lut=tmp_path/"codes.tif",tmp_path/"linear.npy"
    array=np.random.default_rng(71).integers(15,100,(24,4,10),dtype=np.uint16)
    tifffile.imwrite(path,array,photometric="minisblack",metadata={"axes":"TYX"})
    np.save(lut,np.arange(128,dtype=float)**2)
    source=client.post("/api/sources/open",json={"path":str(path)}).json()["source"]
    job=submit(client,source,config(photometry={"lut_path":str(lut)}),task="statistics")
    def pixel():
        response=client.post(f"/api/split/jobs/{job['id']}/statistics",json={"view":"fluctuation_left","index":10})
        assert response.status_code==200,response.text
        token=response.json()["frame"]["token"]
        return client.get(f"/api/frames/{token}/pixel?x=2&y=1").json()["value"]
    original=pixel()
    np.save(lut,np.arange(128,dtype=float)**3)
    assert pixel()==original
    snapshot=Path(job["directory"])/"linearization_lut.npy"
    np.save(snapshot,np.arange(128,dtype=float))
    assert client.post(f"/api/split/jobs/{job['id']}/statistics",json={"view":"fluctuation_left","index":10}).status_code==400
