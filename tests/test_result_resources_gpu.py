"""Lifecycle, bounded windows, resource snapshots and optional CUDA parity."""
from pathlib import Path
import io
import json
import time
import zipfile
import asyncio

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from eip.algorithms import BuiltinAlgorithm,RunContext
from eip.domain import SourceSpec
from eip.gpu_ops import element_operation
from eip.module_registry import new_node,normalize_nodes
from eip.readers import open_source
from eip.resources import ResourceSettings,process_memory,system_memory
from eip.sequence_engine import SequenceEngine
from eip.web.app import create_app,DownloadResponse


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path)) as session:
        session.headers["X-EIP-Token"] = session.get("/api/state").json()["session_token"]
        yield session


def source(client,path):
    values = (np.arange(9*4*6).reshape(9,4,6)+10).astype(np.uint16)
    tifffile.imwrite(path,values,photometric="minisblack",metadata={"axes":"TYX"})
    response = client.post("/api/sources/open",json=dict(path=str(path),interval=.01))
    assert response.status_code==200,response.text
    return response.json()["source"],values


def run(client,entry,steps,**options):
    response = client.post("/api/jobs",json=dict(contract_version=3,source_ids=[entry["id"]],steps=steps,**options))
    assert response.status_code==200,response.text
    identifier = response.json()["id"]
    for _ in range(600):
        job = client.get(f"/api/jobs/{identifier}").json()
        if job["status"] in {"succeeded","completed_with_errors","failed","cancelled"} and not client.app.state.workbench.active:
            return job
        time.sleep(.02)
    pytest.fail("worker timeout")


def test_saved_output_reuse_package_dependencies_and_delete(client,tmp_path):
    entry,values = source(client,tmp_path/"raw.tif")
    node = new_node("gain_offset");node["parameters"]["gain"] = 2
    job = run(client,entry,[node])
    assert job["status"]=="succeeded",job
    identifier=job["id"]
    response = client.post(f"/api/results/{identifier}/package",json={})
    assert response.status_code==200,response.text
    ticket=response.json()["url"]
    response=client.get(ticket,headers={"X-EIP-Token":""})
    assert response.status_code==200,response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        prefix=f"run_{identifier}/"
        record=json.loads(archive.read(prefix+"run.json"))
        assert record["spec"]["resources"]["memory_mb"]>0
        np.testing.assert_array_equal(np.load(io.BytesIO(archive.read(prefix+"source_000/values.npy"))),values*2)
        assert prefix+"source_000/references.jsonl" in archive.namelist()
    assert client.get(ticket).status_code==400
    result=next(s for s in client.get("/api/state").json()["sources"] if s.get("run_id")==identifier)
    promoted=client.post(f"/api/sources/{result['id']}/promote",json={})
    assert promoted.status_code==200,promoted.text
    derived=promoted.json()["source"]
    assert derived["lineage"][-1]["contract"]["axes"]=="YX"
    follow=run(client,derived,[new_node("gain_offset")])
    assert follow["status"]=="succeeded",follow
    preview=client.get(f"/api/results/{identifier}/delete-preview").json()
    assert preview["dependencies"] and not preview["can_delete_files"]
    response=client.post(f"/api/results/{identifier}/delete",json=dict(mode="files",confirmation=preview["confirmation"]))
    assert response.status_code==400 and "引用" in response.text
    preview=client.get(f"/api/results/{identifier}/delete-preview").json()
    response=client.post(f"/api/results/{identifier}/delete",json=dict(mode="reference",confirmation=preview["confirmation"]))
    assert response.status_code==200,response.text
    assert Path(job["directory"]).is_dir()
    assert any(s["id"]==derived["id"] for s in client.get("/api/state").json()["sources"])
    np.testing.assert_array_equal(tifffile.imread(tmp_path/"raw.tif"),values)


def test_delete_files_and_guard_invalid_target(client,tmp_path):
    entry,values=source(client,tmp_path/"raw.tif")
    job=run(client,entry,[])
    assert job["status"]=="succeeded",job
    assert client.post(f"/api/results/{job['id']}/delete",json=dict(mode="files")).status_code==400
    preview=client.get(f"/api/results/{job['id']}/delete-preview").json()
    assert preview["can_delete_files"] and preview["size_bytes"]>values.nbytes
    assert client.post(f"/api/results/{job['id']}/delete",json=dict(mode="files",confirmation=preview["confirmation"])).status_code==200
    assert not Path(job["directory"]).exists() and (tmp_path/"raw.tif").is_file()
    other=run(client,entry,[])
    workbench=client.app.state.workbench
    workbench.jobs[other["id"]]["directory"]=str(tmp_path)
    preview=client.get(f"/api/results/{other['id']}/delete-preview").json()
    assert not preview["can_delete_files"] and "不一致" in preview["error"]
    response=client.post(f"/api/results/{other['id']}/delete",json=dict(mode="files",confirmation=preview["confirmation"]))
    assert response.status_code==400 and (tmp_path/"raw.tif").is_file()


def test_direct_download_is_single_use_protects_files_and_validates_format(client,tmp_path):
    entry,values=source(client,tmp_path/"raw.tif")
    job=run(client,entry,[])
    url=f"/api/jobs/{job['id']}/download-link"
    assert client.post(url,json=dict(file="svg")).status_code==404
    response=client.post(url,json=dict(file="array"))
    assert response.status_code==200,response.text
    preview=client.get(f"/api/results/{job['id']}/delete-preview").json()
    assert not preview["can_delete_files"] and preview["dependencies"][0]["kind"]=="download"
    ticket=response.json()["url"]
    response=client.get(ticket,headers={"X-EIP-Token":""})
    np.testing.assert_array_equal(np.load(io.BytesIO(response.content)),values)
    assert client.get(ticket).status_code==400
    preview=client.get(f"/api/results/{job['id']}/delete-preview").json()
    assert preview["can_delete_files"]
    assert (Path(job["directory"])/"source_000/values.npy").is_file()


@pytest.mark.parametrize("spec_version,abort",[
    ("2.0","send_failure"),("2.4","send_failure"),("2.0","disconnect"),
])
def test_aborted_download_releases_lease(client,tmp_path,spec_version,abort):
    entry,_=source(client,tmp_path/"raw.tif")
    job=run(client,entry,[])
    link=client.post(f"/api/jobs/{job['id']}/download-link",json=dict(file="array")).json()
    ticket=link["url"].rsplit("/",1)[1]
    manager=client.app.state.workbench.results
    saved=manager.take_download(ticket)
    response=DownloadResponse(saved["path"],saved["name"],lambda:manager.finish_download(ticket,saved))
    async def receive():
        if abort=="disconnect":
            return {"type":"http.disconnect"}
        # Keep the connection alive until send fails; two simultaneous abort
        # signals race in frameworks that listen for disconnection.
        await asyncio.Event().wait()
    async def send(message):
        if abort=="send_failure" and message["type"]=="http.response.body":
            raise RuntimeError("client aborted")
    scope=dict(type="http",method="GET",headers=[],extensions={},
               asgi=dict(version="3.0",spec_version=spec_version))
    if abort=="send_failure":
        with pytest.raises(RuntimeError,match="client aborted"):
            asyncio.run(response(scope,receive,send))
    else:
        asyncio.run(response(scope,receive,send))
    assert not manager.active_files
    assert saved["path"].is_file()
    assert client.get(f"/api/results/{job['id']}/delete-preview").json()["can_delete_files"]


def test_resources_snapshot_cache_release_and_restart(tmp_path):
    policy=dict(memory_mb=192,preview_mb=32,cache_mb=16,gpu_mb=64,backend="cpu")
    with TestClient(create_app(tmp_path)) as client:
        client.headers["X-EIP-Token"]=client.get("/api/state").json()["session_token"]
        response=client.post("/api/resources",json=policy)
        assert response.status_code==200,response.text
        entry,_=source(client,tmp_path/"raw.tif")
        job=run(client,entry,[new_node("gain_offset")])
        assert job["run"]["spec"]["resources"]==policy
        assert job["run"]["results"][0]["modules"]["devices"]
        status=client.get("/api/resources").json()
        assert status["frame_cache_count"]>0
        assert client.post("/api/resources/clear",json={}).json()["frame_cache_count"]==0
    with TestClient(create_app(tmp_path)) as client:
        state=client.get("/api/state").json()
        assert state["resources"]==policy
        client.headers["X-EIP-Token"]=state["session_token"]
        assert any(j["id"]==job["id"] for j in client.get("/api/jobs").json())
        restored=client.post(f"/api/jobs/{job['id']}/sequence/promote",json={})
        assert restored.status_code==200,restored.text


@pytest.mark.parametrize("changes",[{"memory_mb":True},{"memory_mb":127},{"cache_mb":9999},{"preview_mb":2048},{"backend":"fake"},{"gpu_mb":0}])
def test_invalid_resources(changes):
    with pytest.raises(ValueError):
        ResourceSettings.from_dict(dict(ResourceSettings().record(),**changes))


def test_exact_window_quantile_uses_disk_and_low_budget(tmp_path,monkeypatch):
    rng=np.random.default_rng(9)
    values=rng.normal(size=(101,128,128));values[3,2,1]=np.nan;values[:,4,3]=np.inf
    path=tmp_path/"sequence.tif";tifffile.imwrite(path,values,photometric="minisblack",metadata={"axes":"TYX"})
    window=new_node("time_smooth");window["parameters"].update(window=101,mode="quantile",quantile=.73)
    window["execution"]={"memory_mb":16}
    reader=open_source(SourceSpec(str(path)))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            expected=np.nanquantile(np.where(np.isfinite(values),values,np.nan),.73,axis=0)
    def no_stack(*args,**kwargs):
        raise AssertionError("whole-window stacking is forbidden")
    monkeypatch.setattr(np,"stack",no_stack)
    engine=SequenceEngine(reader,[window],workdir=tmp_path/"scratch",resources=dict(memory_mb=128,preview_mb=16,cache_mb=0,gpu_mb=32,backend="cpu"))
    try:
        np.testing.assert_allclose(engine.get(1,0),expected,equal_nan=True)
        assert engine.plan["resource_plan"][0]["window_storage"]=="disk"
        assert engine.cache.size==0
        assert not list((tmp_path/"scratch").glob("window-quantile-*"))
    finally:
        engine.close();reader.close()


@pytest.mark.parametrize("algorithm,parameters",[
    ("intensity",dict(gain=2.,offset=-3.)),("affine",dict(gain=.7,offset=4.)),
    ("box_blur",dict(radius=2)),("gaussian",dict(radius=2,sigma=.8)),
    ("threshold",dict(threshold=50.)),("curve",dict(curve="gamma",input_max=100.,gamma=.7))])
def test_device_operations_match_cpu_reference(algorithm,parameters):
    pixels=np.arange(7*9,dtype=np.uint16).reshape(7,9)
    cpu=BuiltinAlgorithm(algorithm);cpu.prepare(parameters,RunContext())
    expected=cpu.process(pixels,RunContext())
    np.testing.assert_allclose(element_operation(algorithm,pixels,parameters,np),expected,rtol=1e-12,atol=1e-12)


def test_gpu_contract_and_auto_fallback(tmp_path,monkeypatch):
    import eip.modes.split_correlation.backend as backend
    monkeypatch.setattr(backend,"devices",lambda:dict(gpu=False,reason="测试无 CUDA"))
    values=np.arange(24,dtype=np.uint16).reshape(4,6)
    path=tmp_path/"frame.tif";tifffile.imwrite(path,values)
    node=new_node("gain_offset");node["execution"]={"backend":"auto"}
    normalize_nodes([dict(node,execution={"backend":"gpu"})])
    reader=open_source(SourceSpec(str(path)))
    engine=SequenceEngine(reader,[node],workdir=tmp_path)
    try:
        assert engine.plan["devices"][node["id"]]["actual"]=="cpu"
        assert engine.plan["devices"][node["id"]]["reason"]=="测试无 CUDA"
        np.testing.assert_array_equal(engine.get(1,0),values)
    finally:
        engine.close();reader.close()
    with pytest.raises(ValueError,match="测试无 CUDA"):
        backend.Backend("gpu",32*1024**2)


@pytest.mark.parametrize("precision",["native","float32","float64"])
def test_device_adapter_preserves_pair_target_and_precision(tmp_path,monkeypatch,precision):
    """Exercise GPU dispatch with a host-array adapter; this is not a CUDA test."""
    import eip.modes.split_correlation.backend as backend
    from eip.gpu_ops import GPU_MODULES
    values=np.arange(3*8*12,dtype=np.uint16).reshape(3,8,12)+10
    path=tmp_path/"pairs.tif";tifffile.imwrite(path,values,photometric="minisblack",metadata={"axes":"TYX"})
    align,gain,smooth,select,curve=[new_node(a) for a in ("alignment","gain_offset","spatial_smooth","select_channel","curve")]
    align["parameters"].update(left=dict(x=0,y=0,width=6,height=8),right=dict(x=6,y=0,width=6,height=8),midline=[6,6])
    gain["parameters"].update(gain=1.2,offset=-1);gain.update(target="left",execution=dict(precision=precision,backend="cpu"))
    smooth["parameters"].update(method="gaussian",sigma=.7,radius=2);smooth.update(target="right",execution=dict(precision=precision,backend="cpu"))
    curve["parameters"].update(curve="sqrt",input_max=1000)
    nodes=[align,gain,smooth,select,curve]
    reader=open_source(SourceSpec(str(path)))
    cpu=SequenceEngine(reader,nodes,workdir=tmp_path)
    try:expected=cpu.get(len(nodes),1).copy()
    finally:cpu.close();reader.close()
    class HostDevice:
        def __init__(self,requested,budget):self.name,self.reason,self.xp="gpu",None,np
        def host(self,value):return np.asarray(value)
        def close(self):pass
    monkeypatch.setattr(backend,"Backend",HostDevice)
    for n in nodes:
        if n["algorithm"] in GPU_MODULES:n.setdefault("execution",{})["backend"]="gpu"
    reader=open_source(SourceSpec(str(path)))
    engine=SequenceEngine(reader,nodes,workdir=tmp_path)
    try:
        actual=engine.get(len(nodes),1)
        assert actual.dtype==expected.dtype
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        assert engine.plan["devices"][gain["id"]]["actual"]=="gpu"
    finally:engine.close();reader.close()


def test_cuda_numerical_parity_when_available():
    from eip.modes.split_correlation.backend import devices,Backend
    if not devices()["gpu"]:
        pytest.skip("可选 CUDA 硬件不可用")
    backend=Backend("gpu",64*1024**2)
    try:
        rng=np.random.default_rng(20)
        image=rng.normal(50,10,(32,48));image[4,6]=np.nan
        for a,p in (("affine",dict(gain=1.5,offset=-2)),("gaussian",dict(radius=3,sigma=1.2)),
                    ("curve",dict(curve="log",input_max=100,gamma=1)),("threshold",dict(threshold=48))):
            expected=element_operation(a,image,p,np)
            actual=backend.host(element_operation(a,image,p,backend.xp))
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10,equal_nan=True)
        from eip.modes.split_correlation.correlation import RollingMoments,direct_g2
        from types import SimpleNamespace
        pairs=rng.uniform(2,5,(7,2,3,4));pairs[2,1,0,0]=np.nan
        for lag in (-1,0,1):
            moments=RollingMoments(pairs[3],SimpleNamespace(min_mean=1e-9,min_samples=2),backend)
            for i in range(2,5):moments.add(pairs[i,0],pairs[i+lag,1])
            expected,count=direct_g2(pairs[2:5,0],pairs[2+lag:5+lag,1])
            actual,n=moments.result()
            np.testing.assert_allclose(actual,expected,rtol=1e-10,equal_nan=True);np.testing.assert_array_equal(n,count)
    finally:
        backend.close()
