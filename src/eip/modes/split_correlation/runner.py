from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import time
import uuid

import numpy as np

from eip import __version__
from eip.algorithms import sha256_file
from eip.domain import ROI, SourceSpec
from eip.modes.split_correlation import MODE_VERSION
from eip.modes.split_correlation.backend import Backend
from eip.modes.split_correlation.config import CorrelationConfig
from eip.modes.split_correlation.correlation import PairCache, RollingMoments, correlate_stream
from eip.modes.split_correlation.geometry import Geometry
from eip.modes.split_correlation.workflow import build_geometry, output_semantics
from eip.modes.split_correlation.photometry import GlobalStatistics, Photometry
from eip.modes.split_correlation.storage import ResultStore, clean_number, close_mapping, window_ref, write_curves
from eip.persistence import write_json
from eip.readers import open_source


class SplitCancelled(Exception):
    pass


def validate_source(meta):
    if len(meta["shape"]) != 2:
        raise ValueError("分幅关联需二维灰度序列；彩色数据请先明确灰度通道")
    if not meta.get("frame_count") or (meta["kind"] == "video" and not meta.get("frame_indexed")):
        raise ValueError("本模式需精确帧索引；支持 CINE 和图像 / TIFF 序列，常规视频请先导出逐帧序列")


def target_region(config,geometry,task):
    if config.roi:
        config.roi.validate(geometry.shape)
        region = config.roi
    elif task == "curve" and config.points:
        xs,ys = zip(*config.points)
        region = ROI(min(xs),min(ys),max(xs)-min(xs)+1,max(ys)-min(ys)+1)
    else:
        region = ROI(0,0,geometry.shape[1],geometry.shape[0])
    region.validate(geometry.shape)
    for x,y in config.points:
        if not region.x<=x<region.x+region.width or not region.y<=y<region.y+region.height:
            raise ValueError("点坐标需在共同有效图内且位于当前分析 ROI 中；可清除 ROI 后选择点")
    if task == "curve" and not config.points and config.roi is None:
        raise ValueError("请为曲线任务选择至少一个点或 ROI")
    return region


def preflight(meta,config,task):
    if task not in {"trial","curve","run","statistics"}:
        raise ValueError("未知分幅任务类型")
    validate_source(meta)
    geometry = build_geometry(config,tuple(meta["shape"]))
    region = target_region(config,geometry,task)
    photo = Photometry(config.photometry)
    if task == "curve" and config.workflow:
        after = False
        for node in config.workflow:
            if node.get("enabled", True):
                if after and node["algorithm"] == "gaussian":
                    raise ValueError("τ 曲线任务的空间平滑需放在关联模块之前；系数图允许关联后的空间平滑")
                after |= node["algorithm"] == "correlation"
    plan = config.plan(meta["frame_count"],single=task in {"trial","curve"}) if task != "statistics" else None
    pixels = region.width*region.height
    working = int(np.prod(geometry.shape))*160+int(np.prod(meta["shape"]))*16+pixels*64
    if working > config.memory_mb*1024**2*.6:
        raise ValueError("有效区域的工作数组超出内存预算，请缩小左右幅有效区或增大预算")
    remaining = max(0,config.memory_mb*1024**2*.9-working)
    frame_budget = int(min(config.memory_mb*1024**2*.32,remaining*.6))
    moment_budget = int(min(config.memory_mb*1024**2*.28,remaining*.4))
    output_bytes = 0 if task in {"curve","statistics"} else plan["window_count"]*plan["lag_count"]*pixels*(np.dtype(config.output_dtype).itemsize+4)
    estimate = {"task":task,"reference_shape":list(geometry.shape),"region":asdict(region),"output_shape":None if task in {"curve","statistics"} else [plan["window_count"],plan["lag_count"],region.height,region.width],"output_bytes":output_bytes,"working_array_bytes":working,"memory_budget_bytes":config.memory_mb*1024**2,"full_source_frames":meta["frame_count"],"background_scope":"all_source_frames","numeric_semantics":photo.record["numeric_semantics"],"lut_sha256":photo.record["lut_sha256"],"geometry":geometry.record,"lags":list(config.lags()),"notes":["固定 RMS 是每路一个全源时间涨落尺度，仅作乘性缩放；标准 g² 不因此改变", "保存码值的物理线性关系需单独确认；不自动套用 CINE 显示伽马", "负延迟取较早右路帧，正延迟取较晚右路帧"]}
    estimate.update(frame_cache_budget_bytes=frame_budget,moment_budget_bytes=moment_budget,reader_cache_budget_bytes=config.memory_mb*1024**2//10)
    if plan:
        estimate["window_count"] = plan["window_count"]
        estimate["first_center"] = plan["first_center"]
        estimate["last_center"] = plan["last_center"]
        estimate["trim_head"] = plan["trim_head"]
        estimate["trim_tail"] = plan["trim_tail"]
        updates = config.window+(plan["window_count"]-1)*min(config.window,config.stride*2)
        estimate["pair_update_upper_estimate"] = updates*plan["lag_count"]
    return estimate


def statistics_key(meta,geometry,photometry):
    value = {"source_revision":meta["revision"],"geometry":geometry.record,"photometry":photometry.record,"mode_version":MODE_VERSION}
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_revision(source,revision):
    reader = open_source(source)
    try:
        if reader.meta.revision != revision:
            raise ValueError("运行期间来源发生变化；结果未发布，请重新导入")
    finally:
        reader.close()


def fit_statistics(reader,source,geometry,photometry,cache_root,check,progress):
    key = statistics_key(asdict(reader.meta),geometry,photometry)
    destination = Path(cache_root)/key
    if (destination/"statistics.json").is_file():
        record = json.loads((destination/"statistics.json").read_text(encoding="utf-8"))
        if record.get("key") == key and record.get("complete") and record.get("frames") == reader.meta.frame_count:
            with np.load(destination/"moments.npz",allow_pickle=False) as archive:
                stats = {name:archive[name].copy() for name in ("mean","count","variance","scales")}
            if stats["mean"].shape != (2,*geometry.shape):
                raise ValueError("全源统计缓存损坏，请移除对应缓存后重算")
            progress("复用完整全源统计",reader.meta.frame_count,reader.meta.frame_count)
            return stats,destination,record
    staging = Path(cache_root)/(key+"."+uuid.uuid4().hex[:8]+".partial")
    staging.mkdir(parents=True,exist_ok=False)
    moments = GlobalStatistics(geometry.shape)
    timestamps = np.lib.format.open_memmap(staging/"timestamps.npy",mode="w+",dtype=np.float64,shape=(reader.meta.frame_count,2))
    numbers = np.lib.format.open_memmap(staging/"source_numbers.npy",mode="w+",dtype=np.int64,shape=(reader.meta.frame_count,))
    started = time.monotonic()
    try:
        for i in range(reader.meta.frame_count):
            check()
            frame = reader.read_frame(i)
            moments.add(geometry.apply(frame.pixels,photometry))
            timestamps[i] = [np.nan if frame.ref.time is None else frame.ref.time,np.nan if frame.ref.trigger_time is None else frame.ref.trigger_time]
            numbers[i] = np.iinfo(np.int64).min if frame.ref.source_number is None else frame.ref.source_number
            progress("全源均值 / 固定尺度",i+1,reader.meta.frame_count)
        verify_revision(source,reader.meta.revision)
        stats = moments.finish(photometry.config.normalization)
        np.savez(staging/"moments.npz",**{key:stats[key] for key in ("mean","count","variance","scales")})
        record = {"key":key,"complete":True,"frames":reader.meta.frame_count,"scope":"all_source_frames","scales":stats["scales"].tolist(),"fit_seconds":time.monotonic()-started,"geometry":geometry.record,"photometry":photometry.record,"source_revision":reader.meta.revision,"mode_version":MODE_VERSION}
        write_json(staging/"statistics.json",record)
    finally:
        close_mapping(timestamps)
        close_mapping(numbers)
    if destination.exists():
        # A previous complete cache is never replaced by a half-computed one.
        raise ValueError("统计缓存目录已存在但不完整，请检查该目录后重算")
    staging.rename(destination)
    return stats,destination,record


def _new_curve(identifier,name,estimator,**extra):
    return {"id":identifier,"name":name,"estimator":estimator,"g2":[],"counts_min":[],"counts_max":[],"pixel_count":extra.pop("pixel_count",1),**extra}


def _append(curve,value,low,high):
    curve["g2"].append(clean_number(value))
    curve["counts_min"].append(int(low))
    curve["counts_max"].append(int(high))


def curve_analysis(cache,baseline,config,plan,backend,check,progress):
    region = cache.roi
    curves = [_new_curve(f"point_{i+1}",f"点 ({x}, {y})","pixel",point=[x,y]) for i,(x,y) in enumerate(config.points)]
    common = np.ones(baseline.shape[1:],dtype=bool)
    point_values = {c["id"]:{} for c in curves}
    def report(done,total):
        progress("固定中心：点 / ROI 延迟扫描",done,total)
    for _,qi,_,_,values,n in correlate_stream(cache,baseline,config,plan,backend,check,report,inner_progress=progress):
        common &= np.isfinite(values)
        for c in curves:
            x,y = c["point"]
            point_values[c["id"]][qi] = (values[y-region.y,x-region.x],n[y-region.y,x-region.x])
    for c in curves:
        for qi in range(len(plan["lags"])):
            value,count = point_values[c["id"]][qi]
            _append(c,value,count,count)
    if config.roi and config.roi_estimator == "pixel_mean":
        area = _new_curve("roi","ROI 逐像素平均","pixel_mean",pixel_count=int(common.sum()),roi=asdict(config.roi),original_pixel_count=region.width*region.height)
        by_lag = {}
        # A second bounded pass uses the same spatial mask at every signed lag.
        for _,qi,_,_,values,n in correlate_stream(cache,baseline,config,plan,backend,check,lambda d,t:progress("ROI 共同有效像素聚合",d,t),inner_progress=progress):
            by_lag[qi] = (float(np.mean(values[common])) if common.any() else None,int(n[common].min()) if common.any() else 0,int(n[common].max()) if common.any() else 0)
        for qi in range(len(plan["lags"])):
            _append(area,*by_lag[qi])
        curves.append(area)
    elif config.roi:
        center,half = plan["centers"][0],config.window//2
        mask = np.isfinite(baseline).all(axis=0)
        for i in range(center-half,center+half+1):
            mask &= np.isfinite(cache.get(i)[0][0])
        intervals = sorted((center-half+q,center+half+q+1) for q in plan["lags"])
        merged = []
        for a,b in intervals:
            if merged and a<=merged[-1][1]:
                merged[-1][1] = max(b,merged[-1][1])
            else:
                merged.append([a,b])
        for a,b in merged:
            for i in range(a,b):
                mask &= np.isfinite(cache.get(i)[0][1])
        area = _new_curve("roi","ROI 总强度","intensity_sum",pixel_count=int(mask.sum()),roi=asdict(config.roi),original_pixel_count=region.width*region.height)
        for qi,q in enumerate(plan["lags"]):
            if mask.any():
                regional_baseline = np.asarray([baseline[0][mask].sum(),baseline[1][mask].sum()])
                m = RollingMoments(regional_baseline,config,backend)
                for i in range(center-half,center+half+1):
                    left = cache.get(i)[0][0][mask].sum()
                    right = cache.get(i+q)[0][1][mask].sum()
                    m.add(left,right)
                value,n = m.result()
                if hasattr(cache.geometry, "postprocess"):
                    value = cache.geometry.postprocess(value)
                _append(area,value,n,n)
            else:
                _append(area,None,0,0)
            progress("ROI 总强度延迟扫描",qi+1,len(plan["lags"]))
        curves.append(area)
    return curves


def split_job_entry(payload,events,cancel):
    reader,store,backend,timestamps,numbers = None,None,None,None,None
    destination = Path(payload["output_dir"])/f"split_{payload['id']}"
    staging = destination.with_name(destination.name+".partial")
    last_progress = [0.0]
    def check():
        if cancel.is_set():
            raise SplitCancelled()
    def emit(event):
        events.put(event)
    def progress(name,done,total):
        now = time.monotonic()
        if done == total or now-last_progress[0]>.15:
            emit({"type":"progress","name":name,"completed":done,"total":total,"source":0})
            last_progress[0] = now
    run = {"schema_version":1,"mode":"split_correlation","mode_version":MODE_VERSION,"app_version":__version__,"run_id":payload["id"],"task":payload["task"],"status":"running","created_at":datetime.now(timezone.utc).isoformat(),"results":[],"source":payload["source"],"source_revision":payload["revision"],"provenance":payload.get("provenance",{}),"config":payload["config"]}
    run["display_name"] = payload.get("display_name",payload["id"])
    try:
        check()
        config = CorrelationConfig.from_dict(payload["config"])
        run.update(output_semantics(config.workflow))
        source = SourceSpec(**payload["source"])
        reader = open_source(source)
        if reader.meta.revision != payload["revision"]:
            raise ValueError("来源已改变，请重新导入")
        if hasattr(reader,"_budget"):
            reader._budget = max(1,config.memory_mb*1024**2//10)
        estimate = preflight(asdict(reader.meta),config,payload["task"])
        geometry,photo = build_geometry(config,reader.meta.shape),Photometry(config.photometry)
        if photo.record["lut_sha256"] != payload.get("lut_sha256"):
            raise ValueError("线性化 LUT 在排队期间发生变化，请重新提交")
        region = target_region(config,geometry,payload["task"])
        parent = destination.parent
        parent.mkdir(parents=True,exist_ok=True)
        required = estimate["output_bytes"]+reader.meta.frame_count*24+int(np.prod(geometry.shape))*64+16*1024**2
        if shutil.disk_usage(parent).free < required:
            raise ValueError(f"输出空间不足，估算至少需 {required/1024**3:.2f} GiB")
        staging.mkdir(exist_ok=False)
        run.update(estimate=estimate,geometry=geometry.record,photometry=photo.record,analysis_region=asdict(region),lags=list(config.lags()),numeric_semantics=photo.record["numeric_semantics"],output_dtype=config.output_dtype,accumulator_dtype="float64",data_spec=asdict(reader.meta))
        if photo.lut is not None:
            np.save(staging/"linearization_lut.npy",photo.lut)
            run.update(lut_snapshot="linearization_lut.npy",lut_snapshot_sha256=sha256_file(staging/"linearization_lut.npy"))
        write_json(staging/"run.json",run)
        started = time.monotonic()
        stats,cache_path,fit_record = fit_statistics(reader,source,geometry,photo,payload["cache_root"],check,progress)
        run.update(statistics_cache=str(cache_path),statistics=fit_record)
        for name in ("timestamps.npy","source_numbers.npy"):
            shutil.copyfile(cache_path/name,staging/name)
        np.save(staging/"mean.npy",stats["mean"])
        np.save(staging/"variance.npy",stats["variance"])
        np.save(staging/"mean_counts.npy",stats["count"])
        if payload["task"] != "statistics":
            backend = Backend(config.backend,config.memory_mb*1024**2)
            run.update(backend=backend.name,backend_fallback_reason=backend.reason)
            q = region
            baseline = stats["mean"][:,q.y:q.y+q.height,q.x:q.x+q.width]/stats["scales"][:,None,None]
            cache = PairCache(reader,geometry,photo,stats["scales"],region,estimate["frame_cache_budget_bytes"],check,moment_budget=estimate["moment_budget_bytes"])
            plan = config.plan(reader.meta.frame_count,single=payload["task"] in {"trial","curve"})
            run["centers"] = {"first":plan["first_center"],"last":plan["last_center"],"count":plan["window_count"],"stride":config.stride if payload["task"]=="run" else 1}
            timestamps = np.load(staging/"timestamps.npy",mmap_mode="r",allow_pickle=False)
            numbers = np.load(staging/"source_numbers.npy",mmap_mode="r",allow_pickle=False)
            if payload["task"] == "curve":
                curves = curve_analysis(cache,baseline,config,plan,backend,check,progress)
                refs = [window_ref(plan["centers"][0],lag,config.window,timestamps,numbers) for lag in plan["lags"]]
                write_curves(staging,curves,refs,config.to_dict())
                run["curve_count"] = len(curves)
            else:
                shape = tuple(estimate["output_shape"])
                run["array_shape"] = list(shape)
                run["axes"] = "TQYX"
                store = ResultStore(staging,shape,dtype=config.output_dtype)
                csv_path = staging/"windows.csv"
                import csv
                columns = list(window_ref(plan["centers"][0],plan["lags"][0],config.window,timestamps,numbers))
                with csv_path.open("w",encoding="utf-8-sig",newline="") as file:
                    writer = csv.DictWriter(file,fieldnames=["output_time_index","lag_index",*columns])
                    writer.writeheader()
                    for ti,qi,center,lag,values,n in correlate_stream(cache,baseline,config,plan,backend,check,lambda d,t:progress("逐像素正负延迟关联",d,t),inner_progress=progress):
                        store.write(ti,qi,values,n)
                        writer.writerow({"output_time_index":ti,"lag_index":qi,**window_ref(center,lag,config.window,timestamps,numbers)})
                store.close()
                store = None
                run["curve_count"] = 0
            run.update(transformed_frame_reads=cache.reads,peak_frame_cache_bytes=cache.peak_bytes,compute_seconds=time.monotonic()-started)
        else:
            run["backend"] = "cpu"
        check()
        verify_revision(source,payload["revision"])
        close_mapping(timestamps)
        close_mapping(numbers)
        timestamps = numbers = None
        run.update(status="succeeded",completed_at=datetime.now(timezone.utc).isoformat())
        write_json(staging/"run.json",run)
        staging.rename(destination)
        emit({"type":"split_completed","run":run,"directory":str(destination)})
    except SplitCancelled:
        run["status"] = "cancelled"
        if staging.is_dir():
            write_json(staging/"run.json",run)
        emit({"type":"cancelled","directory":str(staging) if staging.exists() else None})
    except Exception as error:
        run.update(status="failed",error=str(error))
        if staging.is_dir():
            write_json(staging/"run.json",run)
        emit({"type":"failed","message":str(error),"directory":str(staging) if staging.exists() else None})
    finally:
        if store:
            store.close()
        close_mapping(timestamps)
        close_mapping(numbers)
        if reader:
            reader.close()
        if backend:
            backend.close()
