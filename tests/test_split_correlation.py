from dataclasses import asdict
import json
from pathlib import Path
import threading

import numpy as np
import pytest
import tifffile

from eip.domain import SourceSpec
from eip.modes.split_correlation.backend import Backend
from eip.modes.split_correlation.config import CorrelationConfig,GeometryConfig,PhotometryConfig
from eip.modes.split_correlation.correlation import RollingMoments,direct_g2
from eip.modes.split_correlation.geometry import Geometry,phase_registration
from eip.modes.split_correlation.photometry import Photometry
from eip.modes.split_correlation.runner import preflight,split_job_entry
from eip.modes.split_correlation.storage import CorrelationReader,read_slice
from eip.modes.split_correlation.storage import window_ref
from eip.modes.split_correlation.storage import ResultStore
from eip.readers import open_source


def config(h=4,w=5,**extra):
    return CorrelationConfig.from_dict({"geometry":{"left":{"x":0,"y":0,"width":w,"height":h},"right":{"x":w,"y":0,"width":w,"height":h},"midline":[w,w]},"window":5,"stride":3,"lag_min":-2,"lag_max":3,"center":10,**extra})


def series(tmp_path,n=24,h=4,w=5):
    rng = np.random.default_rng(21)
    left = rng.uniform(20,100,(n,h,w))+np.arange(n)[:,None,None]*1.4
    right = rng.uniform(30,80,(n,h,w))+np.arange(n)[:,None,None]*.3
    array = np.concatenate((left,right),axis=2).astype(np.float32)
    path = tmp_path/"split.tif"
    tifffile.imwrite(path,array,photometric="minisblack",metadata={"axes":"TYX"})
    return path,left.astype(np.float32),right.astype(np.float32)


class Events:
    def __init__(self):
        self.values = []
    def put(self,event):
        self.values.append(event)


def execute(tmp_path,path,cfg,task="run",cancel=None,identifier="run1"):
    source = SourceSpec(str(path),interval=.01)
    reader = open_source(source)
    revision = reader.meta.revision
    reader.close()
    payload = {"id":identifier,"source":asdict(source),"revision":revision,"config":cfg.to_dict(),"task":task,"output_dir":str(tmp_path/"out"),"cache_root":str(tmp_path/"cache"),"lut_sha256":None}
    events = Events()
    split_job_entry(payload,events,cancel or threading.Event())
    return events.values[-1]


def test_signed_boundary_plan_and_fixed_center():
    cfg = config(lag_min=-3,lag_max=5,stride=4,center=5)
    plan = cfg.plan(30)
    assert list(plan["centers"]) == [5,9,13,17,21]
    assert plan["trim_head"] == 5 and plan["trim_tail"] == 7
    assert cfg.plan(30,single=True)["centers"][0] == 5
    with pytest.raises(ValueError,match="中心帧"):
        config(center=2).plan(30,single=True)
    with pytest.raises(ValueError,match="不足"):
        config().plan(8)
    assert 0 in config(lag_min=-5,lag_max=4,lag_step=3).lags()
    assert config(lag_list=[3,-2,0,-2]).lags() == (-2,0,3)


@pytest.mark.parametrize("extra",[{"window":4},{"stride":0},{"lag_step":0},{"lag_list":[1.5]},{"backend":"pretend"},{"min_mean":0},{"memory_mb":63},{"points":[[-1,0]]}])
def test_invalid_config(extra):
    with pytest.raises((ValueError,TypeError)):
        config(**extra)


def test_large_window_actual_signed_timestamps_and_missing_times():
    n,center,window = 20005,10000,17001
    times = np.cumsum(np.linspace(.000009,.000011,n))
    timestamps = np.column_stack([times,times-times[center]])
    numbers = np.arange(n,dtype=np.int64)-center
    for q in (-17,0,23):
        ref = window_ref(center,q,window,timestamps,numbers)
        expected = times[center-window//2+q:center+window//2+q+1]-times[center-window//2:center+window//2+1]
        assert abs(ref["lag_seconds"]-expected.mean())<1e-16
        assert ref["lag_seconds_min"]==expected.min() and ref["lag_seconds_max"]==expected.max()
        assert ref["source_number"]==0
    timestamps[center,0]=np.nan
    assert window_ref(center,-17,window,timestamps,numbers)["lag_seconds"] is None


def test_combined_working_array_and_cache_budget():
    cfg=config(h=360,w=400,memory_mb=64)
    estimate=preflight({"kind":"sequence","shape":[360,800],"frame_count":24},cfg,"run")
    assigned=sum(estimate[key] for key in ("working_array_bytes","frame_cache_budget_bytes","moment_budget_bytes","reader_cache_budget_bytes"))
    assert assigned<=estimate["memory_budget_bytes"]
    assert estimate["frame_cache_budget_bytes"]<64*1024**2*.32


def test_auto_falls_back_if_detected_gpu_cannot_start(monkeypatch):
    import sys
    from types import SimpleNamespace
    import eip.modes.split_correlation.backend as module
    monkeypatch.setattr(module,"devices",lambda:{"gpu":True,"reason":None})
    def fail():raise RuntimeError("test driver unavailable")
    monkeypatch.setitem(sys.modules,"cupy",SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=fail))))
    b=Backend("auto",64*1024**2)
    assert b.name=="cpu" and "test driver unavailable" in b.reason
    with pytest.raises(ValueError,match="test driver unavailable"):
        Backend("gpu",64*1024**2)


def test_float64_storage_preserves_small_g2_deviation(tmp_path):
    values=np.asarray([[1+1e-10,1-2e-10]])
    store=ResultStore(tmp_path,(1,1,1,2),dtype="float64")
    store.write(0,0,values,np.full_like(values,31,dtype=np.uint32))
    store.close()
    saved=np.load(tmp_path/"g2.npy",allow_pickle=False)
    np.testing.assert_array_equal(saved[0,0],values)
    assert np.all(saved!=1)


def test_float64_run_slice_and_derived_reader(tmp_path):
    path,_,_=series(tmp_path)
    cfg=config(output_dtype="float64")
    event=execute(tmp_path,path,cfg)
    assert event["type"]=="split_completed",event
    pixels,_,_,run=read_slice(event["directory"],0,0)
    assert pixels.dtype==np.float64 and run["output_dtype"]=="float64"
    assert run["estimate"]["output_bytes"]==np.prod(run["array_shape"])*12
    selected=Path(event["directory"])/"lag.json"
    selected.write_text(json.dumps({"schema_version":1,"type":"eip_correlation_lag","directory":event["directory"],"lag_index":0}),encoding="utf-8")
    reader=CorrelationReader(SourceSpec(str(selected)))
    try:assert reader.meta.dtype=="float64" and reader.read_frame(0).pixels.dtype==np.float64
    finally:reader.close()


def test_local_g2_with_global_baseline_drift_and_joint_missing():
    left = np.array([21,22,np.nan,25,31,35],dtype=float)[:,None]
    right = np.array([19,24,28,25,np.nan,20],dtype=float)[:,None]
    cfg = config(window=7)
    baseline = np.array([[12.],[31.]])
    b = Backend("cpu",64*1024**2)
    m = RollingMoments(baseline,cfg,b)
    for l,r in zip(left,right):
        m.add(l,r)
    values,count = m.result()
    expected,n = direct_g2(left,right)
    np.testing.assert_allclose(values,expected,rtol=1e-13)
    np.testing.assert_array_equal(count,n)
    scaled,_ = direct_g2(left*3,right*.25)
    np.testing.assert_allclose(values,scaled,rtol=1e-13)
    invalid,_ = direct_g2(np.zeros((3,1)),np.ones((3,1)))
    assert np.isnan(invalid[0])


def test_sensor_lut_precedes_subpixel_resampling(tmp_path):
    lut = tmp_path/"lut.npy"
    np.save(lut,np.arange(16,dtype=float)**2)
    cfg = config(h=2,w=3)
    geometry = Geometry(GeometryConfig.from_dict({**asdict(cfg.geometry),"dx":.5,"interpolation":"bilinear"}),(2,6))
    raw = np.tile(np.array([1,2,3,2,4,6],dtype=np.uint16),(2,1))
    l,r = geometry.apply(raw,Photometry(PhotometryConfig(lut_path=str(lut))))
    assert geometry.shape == (2,2)
    np.testing.assert_allclose(r[0],[10,26])
    np.testing.assert_allclose(l[0],[1,4])
    assert r[0,0] != ((2+4)/2)**2


def test_nearest_half_pixel_uses_consistent_neighbor():
    cfg=config(h=2,w=4)
    geometry=Geometry(GeometryConfig.from_dict({**asdict(cfg.geometry),"dx":.5}),(2,8))
    raw=np.tile(np.asarray([1,2,3,4,10,20,30,40],dtype=np.uint16),(2,1))
    _,right=geometry.apply(raw,Photometry(cfg.photometry))
    np.testing.assert_array_equal(right[0],[20,30,40])


def test_static_phase_mapping_offset_sign():
    rng = np.random.default_rng(4)
    left = rng.normal(size=(48,48))
    right = np.roll(np.roll(left,2,axis=1),-1,axis=0)*1.8+100
    proposed = phase_registration(left,right)
    assert abs(proposed["dx"]-2)<.15 and abs(proposed["dy"]+1)<.15
    with pytest.raises(ValueError,match="结构"):
        phase_registration(np.ones((16,16)),np.ones((16,16)))


def test_streaming_maps_and_reusable_global_statistics(tmp_path):
    path,left,right = series(tmp_path)
    cfg = config(stride=2)
    event = execute(tmp_path,path,cfg)
    assert event["type"] == "split_completed",event
    run = event["run"]
    directory = Path(event["directory"])
    plan = cfg.plan(len(left))
    for ti in (0,2,len(plan["centers"])-1):
        center = plan["centers"][ti]
        for qi,q in enumerate(plan["lags"]):
            g,n,ref,_ = read_slice(directory,ti,qi)
            expected,count = direct_g2(left[center-2:center+3],right[center-2+q:center+3+q])
            np.testing.assert_allclose(g,expected,rtol=8e-8)
            np.testing.assert_array_equal(n,count)
            assert ref["lag_frames"] == q
            assert ref["right_start"] == center-2+q
            assert ref["lag_seconds"] == pytest.approx(q*.01)
    np.testing.assert_allclose(np.load(directory/"mean.npy"),np.stack((left.mean(axis=0,dtype=np.float64),right.mean(axis=0,dtype=np.float64))),rtol=1e-14)
    assert run["peak_frame_cache_bytes"] <= cfg.memory_mb*1024**2*.32
    original_cache_time = Path(run["statistics_cache"])/"statistics.json"
    before = original_cache_time.stat().st_mtime_ns
    trial = execute(tmp_path,path,config(center=10),task="trial",identifier="trial2")
    assert trial["type"] == "split_completed",trial
    assert original_cache_time.stat().st_mtime_ns == before
    assert "windows.csv" in [p.name for p in directory.iterdir()]


@pytest.mark.parametrize("estimator",["pixel_mean","intensity_sum"])
def test_fixed_center_point_and_both_roi_curves(tmp_path,estimator):
    path,left,right = series(tmp_path)
    cfg = config(points=[[1,1]],roi={"x":0,"y":0,"width":3,"height":3},roi_estimator=estimator)
    event = execute(tmp_path,path,cfg,task="curve")
    assert event["type"] == "split_completed",event
    directory = Path(event["directory"])
    value = json.loads((directory/"curves.json").read_text(encoding="utf-8"))
    curves = value["curves"]
    for qi,q in enumerate(cfg.lags()):
        l,r = left[8:13,:3,:3],right[8+q:13+q,:3,:3]
        g,count = direct_g2(l,r)
        assert curves[0]["g2"][qi] == pytest.approx(g[1,1],rel=1e-12)
        if estimator == "pixel_mean":
            expected = g.mean()
        else:
            expected,_ = direct_g2(l.sum(axis=(1,2),dtype=np.float64),r.sum(axis=(1,2),dtype=np.float64))
        assert curves[1]["g2"][qi] == pytest.approx(expected,rel=1e-12)
    assert not (directory/"g2.npy").exists()
    for name in ("curves.csv","curves.png","curves.svg"):
        assert (directory/name).stat().st_size > 100


def test_partial_cancellation_and_derived_reader(tmp_path):
    path,left,right = series(tmp_path)
    stop = threading.Event()
    stop.set()
    cancelled = execute(tmp_path,path,config(),cancel=stop)
    assert cancelled["type"] == "cancelled"
    assert not (tmp_path/"out"/"split_run1").exists()
    event = execute(tmp_path,path,config())
    assert event["type"] == "split_completed",event
    directory = Path(event["directory"])
    selected = directory/"lag.json"
    selected.write_text(json.dumps({"schema_version":1,"type":"eip_correlation_lag","directory":str(directory),"lag_index":0}),encoding="utf-8")
    reader = CorrelationReader(SourceSpec(str(selected)))
    try:
        frame = reader.read_frame(2)
        expected,_,ref,_ = read_slice(directory,2,0)
        np.testing.assert_array_equal(frame.pixels,expected)
        assert frame.ref.time == ref["center_time"]
        assert reader.meta.frame_indexed and reader.meta.shape == (4,5)
    finally:
        reader.close()
