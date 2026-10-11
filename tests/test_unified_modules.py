"""Cross-layout execution, removable stages, and preview isolation regressions."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import tifffile

from eip.module_engine import FrameModules
from eip.module_registry import new_node, normalize_nodes
from eip.modes.split_correlation.config import CorrelationConfig
from test_split_web import client, opened, wait_job
from test_split_workflow import workflow_config, run_job, image


def pixel(client, frame, x=2, y=1):
    r=client.get(f"/api/frames/{frame['token']}/pixel?x={x}&y={y}")
    assert r.status_code==200,r.text
    return r.json()["value"]


def general_job(client, source, nodes, mode="trial", **kwargs):
    r=client.post("/api/jobs",json=dict(source_ids=[source["id"]],steps=nodes,mode=mode,selection=dict(mode="current" if mode=="trial" else "all",current=dict(index=10,time=.1)),**kwargs))
    assert r.status_code==200,r.text
    job=wait_job(client,r.json()["id"])
    assert job["status"]=="succeeded",job
    return job


def test_catalog_is_shared_and_partial_preview_ignores_temporal_parameters(client,tmp_path):
    source,raw=opened(client,tmp_path)
    catalog=client.get("/api/modules").json()
    assert catalog==client.get("/api/split/modules").json()
    assert {"intensity","box_blur","threshold","affine","gaussian","curve","calibration","alignment","statistics","correlation","select_channel"}<=catalog.keys()
    cfg=workflow_config()
    cfg["workflow"][-1]["parameters"]["window"]=4
    cfg["lag_list"]=[99999]
    r=client.post("/api/split/geometry",json=dict(source_id=source["id"],index=10,geometry=cfg["geometry"],workflow_config=cfg))
    assert r.status_code==200,r.text
    assert pixel(client,r.json()["left"])==raw[10,1,2]
    assert client.post("/api/split/preflight",json=dict(source_id=source["id"],config=cfg,task="trial")).status_code==400
    assert client.post("/api/split/geometry",json=dict(source_id=source["id"],index=10,workflow_config=None)).status_code==200


def test_removing_statistics_and_calibration_skips_full_source_scan(client,tmp_path):
    source,raw=opened(client,tmp_path)
    cfg=workflow_config(lag_list=[0],output_dtype="float64")
    cfg.update(workflow_version=2)
    cfg["workflow"]=[n for n in cfg["workflow"] if n["algorithm"] not in {"statistics","calibration"}]
    job=run_job(client,source,cfg)
    run=job["run"]
    assert run["statistics"]["scope"]=="none" and run["statistics"]["frames"]==0
    assert run["estimate"]["full_source_frames"]==0
    assert run["transformed_frame_reads"]==5
    assert not (Path(job["directory"])/"mean.npy").exists()
    left,right=raw[8:13,:,:5].astype(float),raw[8:13,:,5:].astype(float)
    np.testing.assert_allclose(image(client,job),(left*right).mean(0)/(left.mean(0)*right.mean(0)),atol=1e-14)
    r=client.post(f"/api/split/jobs/{job['id']}/slice",json={}).json()["window_ref"]
    assert r["center_index"]==10 and r["center_time"]==.1
    assert client.post(f"/api/split/jobs/{job['id']}/statistics",json={}).status_code==400


def test_empty_modern_workflow_is_passthrough_and_old_empty_is_legacy(client,tmp_path):
    source,raw=opened(client,tmp_path)
    cfg=workflow_config()
    cfg.update(workflow=[],workflow_version=2)
    job=run_job(client,source,cfg)
    assert job["mode"]=="trial"
    assert pixel(client,job["result"])==raw[10,1,2]
    assert job["parameters"]["steps"]==[]
    old=deepcopy(cfg)
    old.pop("workflow_version")
    assert CorrelationConfig.from_dict(old).workflow is None


def test_general_layout_can_execute_split_geometry_and_single_channel(client,tmp_path):
    source,raw=opened(client,tmp_path)
    nodes=[new_node("alignment",(4,10)),new_node("select_channel"),new_node("affine")]
    nodes[1]["parameters"]["channel"]="right"
    nodes[2]["parameters"].update(gain=2,offset=3)
    trial=general_job(client,source,nodes)
    assert trial["result"]["base_shape"]==[4,5]
    assert pixel(client,trial["result"])==pytest.approx(float(raw[10,1,7])*2+3)
    saved=general_job(client,source,nodes,"run",index_selection=[8,13])
    record=saved["run"]["results"][0]
    assert record["completed_frames"]==5
    assert record["base_shape"]==[4,5]
    result=tifffile.imread(Path(saved["directory"])/"source_001/frame_00000000.tif")
    np.testing.assert_allclose(result,raw[8,:,5:]*2+3,rtol=1e-7)
    assert record["modules"]["mappings"]
    result_source=next(s for s in client.get("/api/state").json()["sources"] if s.get("run_id")==saved["id"])
    promoted=client.post(f"/api/sources/{result_source['id']}/promote",json={})
    assert promoted.status_code==200,promoted.text
    mapping=promoted.json()["source"]["lineage"][-1]
    assert mapping["coordinate_space"]=="module_output_grid" and mapping["mappings"]


def test_general_layout_routes_temporal_modules_by_contract(client,tmp_path):
    source,_=opened(client,tmp_path)
    cfg=workflow_config(lag_list=[0])
    cfg["workflow"][-1]["parameters"].update(map_lag=-2,analysis="map")
    job=general_job(client,source,cfg["workflow"])
    assert job["mode"]=="split" and job["task"]=="trial"
    assert job["run"]["lags"]==[-2]
    assert [n["id"] for n in job["run"]["config"]["workflow"]]==["cal","align","stats","g2"]


def test_repeated_calibrations_and_common_frame_algorithms_have_real_order():
    raw=np.full((4,10),10,dtype=np.uint16)
    a,b=new_node("calibration"),new_node("calibration")
    a["parameters"].update(curve="linear",input_max=2)
    b["parameters"].update(black_left=1,black_right=1)
    align=new_node("alignment",raw.shape)
    gain=new_node("intensity")
    gain["parameters"].update(gain=3,offset=0)
    gain["target"]="right"
    engine=FrameModules([a,b,align,gain],raw.shape)
    try:
        left,right=engine.execute(raw)
        np.testing.assert_array_equal(left,4)
        np.testing.assert_array_equal(right,12)
        assert len(engine.record["calibrations"])==2
    finally:
        engine.close()
    engine=FrameModules([b,a,align],raw.shape)
    try:
        np.testing.assert_array_equal(engine.execute(raw)[0],4.5)
    finally:
        engine.close()


def test_incompatible_module_scope_and_label_inputs_are_rejected():
    a=new_node("alignment",(4,10))
    g=new_node("correlation",frames=24)
    g["target"]="left"
    with pytest.raises(ValueError,match="完整输入"):
        normalize_nodes([a,g])
    g["target"]="both"
    with pytest.raises(ValueError,match="标签"):
        normalize_nodes([a,new_node("threshold"),g])


def test_plugin_is_available_and_executed_in_both_layouts(client,tmp_path):
    source,raw=opened(client,tmp_path)
    manifest=Path(__file__).parents[1]/"examples/plugins/invert.json"
    r=client.post("/api/plugins/register",json={"path":str(manifest)})
    assert r.status_code==200,r.text
    spec=client.get("/api/modules").json()["example.invert"]
    node={k:spec[k] for k in ("version","manifest","code_hash")}
    node.update(id="invert-instance",algorithm=spec["id"],parameters={"reference":200.},enabled=True,target="both")
    trial=general_job(client,source,[node])
    assert pixel(client,trial["result"])==pytest.approx(200-float(raw[10,1,2]),abs=1e-5)
    cfg=workflow_config()
    cfg["workflow"].insert(0,node)
    r=client.post("/api/split/geometry",json=dict(source_id=source["id"],index=10,geometry=cfg["geometry"],workflow_config=cfg))
    assert r.status_code==200,r.text
    assert pixel(client,r.json()["left"])==pytest.approx(pixel(client,trial["result"]),abs=1e-5)
    job=run_job(client,source,cfg)
    assert job["run"]["geometry"]["workflow"][0]["code_hash"]==spec["code_hash"]


def test_lut_snapshot_reconstructs_statistics_after_original_is_removed(client,tmp_path):
    raw=np.arange(24*4*10,dtype=np.uint16).reshape(24,4,10)%100+10
    path=tmp_path/"codes.tif"
    tifffile.imwrite(path,raw,photometric="minisblack",metadata={"axes":"TYX"})
    source=client.post("/api/sources/open",json={"path":str(path),"interval":.01}).json()["source"]
    lut=tmp_path/"lut.npy"
    np.save(lut,np.arange(128,dtype=float)**2)
    cfg=workflow_config(lag_list=[0],photometry={"lut_path":str(lut),"normalization":"none"})
    job=run_job(client,source,cfg)
    assert job["run"]["module_lut_snapshots"]["cal"]["sha256"]
    lut.unlink()
    r=client.post(f"/api/split/jobs/{job['id']}/statistics",json={"view":"fluctuation_left","index":10})
    assert r.status_code==200,r.text
    expected=float(raw[10,1,2])**2-np.mean(raw[:,1,2].astype(float)**2)
    assert pixel(client,r.json()["frame"])==pytest.approx(expected)


def test_label_postprocessing_preserves_invalid_correlation_samples(client,tmp_path):
    source,raw=opened(client,tmp_path)
    raw[:,:,0]=0
    path=tmp_path/"invalid.tif"
    tifffile.imwrite(path,raw,photometric="minisblack",metadata={"axes":"TYX"})
    source=client.post("/api/sources/open",json={"path":str(path),"interval":.01}).json()["source"]
    cfg=workflow_config(lag_list=[0])
    cfg["workflow"].append(new_node("threshold"))
    job=run_job(client,source,cfg)
    values=image(client,job)
    assert np.isnan(values[:,0]).all()
    assert np.isfinite(values[:,1:]).all()
    assert job["run"]["value_kind"]=="label"
    assert job["run"]["result_semantics"]=="processed_g2"
    r=client.post(f"/api/split/jobs/{job['id']}/slice",json={})
    assert r.status_code==200 and r.json()["frame"]["kind"]=="label"
    r=client.post(f"/api/split/jobs/{job['id']}/promote",json={"lag_index":0})
    assert r.status_code==200,r.text
    assert r.json()["source"]["result_kind"]=="label"
