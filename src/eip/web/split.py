from __future__ import annotations

from dataclasses import asdict,replace
from copy import deepcopy
from pathlib import Path
import json
import time
import uuid

import numpy as np

from eip.algorithms import sha256_file
from eip.domain import SourceSpec
from eip.modes.split_correlation.backend import devices
from eip.modes.split_correlation.config import CorrelationConfig,GeometryConfig,PhotometryConfig,finite,integer
from eip.modes.split_correlation.geometry import Geometry,REGISTRATION_METHODS,propose_rectangles
from eip.modes.split_correlation.photometry import Photometry
from eip.modes.split_correlation.runner import preflight,validate_source
from eip.modes.split_correlation.storage import close_mapping,read_slice
from eip.persistence import write_json
from eip.web.records import display_name, automatic_name, split_summary, saved_split_summary
from eip.modes.split_correlation.workflow import MODULES, build_geometry
from eip.module_registry import normalize_nodes, temporal


class SplitService:
    def __init__(self,workbench):
        self.workbench = workbench
        self.devices = devices()

    def _source(self,value):
        entry = self.workbench.get_source(value["source_id"])
        if entry["role"] != "input":
            raise ValueError("请明确选择输入来源")
        validate_source(entry["meta"])
        return entry

    def _frame(self,entry,index):
        integer(index,"帧索引",0)
        event = self.workbench.preview.request(entry["source"],"index",index=index)
        if event.get("meta",{}).get("revision") != entry["meta"]["revision"]:
            raise ValueError("来源已改变，请重新导入")
        return event

    def _snapshot(self,pixels,ref,*,kind="image",origin=(0,0),base_shape=None,display=None):
        token = self.workbench.frames.add(pixels,ref,origin,base_shape or pixels.shape[:2],kind)
        return self.workbench.frames.render(token,display)

    def geometry(self,value):
        axis = value.get("contract_version")==3
        entry = self.workbench.get_source(value["source_id"]) if axis else self._source(value)
        if entry["role"]!="input":
            raise ValueError("请选择输入来源")
        if axis:
            event = self.workbench.module_sample(dict(source_id=entry["id"],steps=value.get("module_prefix",[]),dependencies=value.get("module_dependencies",[]),sample_index=value.get("sample_index",0)))
            if event["kind"]!="image" or event["contract"]["axes"]!="YX":
                raise ValueError("配准工具需要单幅灰度矩阵 YX 输入")
            event["frame"] = event["reference"]
        else:
            event = self._frame(entry,value.get("index",0))
        if value.get("geometry"):
            config = GeometryConfig.from_dict(value["geometry"])
        else:
            line = value.get("midline",[event["pixels"].shape[1]/2]*2)
            if len(line)!=2 or not all(0<finite(x,"中线坐标")<event["pixels"].shape[1] for x in line):
                raise ValueError("中线需有两个位于图内的 X 坐标")
            threshold = finite(value.get("threshold",0),"黑边识别阈值")
            config = GeometryConfig.from_dict(propose_rectangles(event["pixels"],line,threshold))
        photo = Photometry(PhotometryConfig.from_dict(dict(curve="identity",normalization="none"))) if axis else Photometry(PhotometryConfig.from_dict(value.get("photometry")))
        mapping = Geometry(config,event["pixels"].shape)
        requested = (value.get("workflow_config") or {}).get("workflow")
        if requested is not None:
            prefix = []
            for node in requested:
                if node.get("enabled",True) and node["algorithm"] in {"statistics","correlation"}:
                    break
                prefix.append(node)
            nodes, state, kind = normalize_nodes(prefix)
            self.workbench.validate_steps(list(nodes))
            # Full correlation parameters are irrelevant to single-frame preview.
            transformed = self.workbench.preview.request(entry["source"],"index",index=value.get("index",0),modules=nodes)
            if state != "pair":
                return {"geometry":asdict(config),"mapping":None,"raw":self._snapshot(event["pixels"],event["frame"]),"processed":self._snapshot(transformed["processed"],event["frame"],kind=kind),"module_record":transformed["module_record"]}
            left,right = transformed["processed"]
            record = transformed["module_record"]
            mapping_record = dict(record["mappings"][-1],workflow=nodes)
        else:
            left,right = mapping.apply(event["pixels"],photo)
            mapping_record = mapping.record
        common = np.concatenate((left[np.isfinite(left)],right[np.isfinite(right)]))
        low,high = (float(np.percentile(common,1)),float(np.percentile(common,99))) if common.size else (0,1)
        high = max(low+1e-9,high)
        display = {"low":low,"high":high,"gamma":1,"colormap":"gray"}
        overlay = np.stack((left,right,(left+right)*.5),axis=2)
        # Preview overlays have their own display coordinates; raw numbers stay separate.
        result = {"geometry":asdict(config),"mapping":mapping_record,"numeric_semantics":photo.record["numeric_semantics"],"raw":self._snapshot(event["pixels"],event["frame"]),"left":self._snapshot(left,event["frame"],display=display),"right":self._snapshot(right,event["frame"],display=display),"overlay":self._snapshot(overlay,event["frame"],display=display)}
        if value.get("calibrate"):
            l,r = config.left,config.right
            a = photo.apply(event["pixels"][l.y:l.y+l.height,l.x:l.x+l.width],"left")
            b = photo.apply(event["pixels"][r.y:r.y+r.height,r.x:r.x+r.width],"right")
            if config.mirror_x:
                b = b[:,::-1]
            if config.mirror_y:
                b = b[::-1]
            result["proposal"] = REGISTRATION_METHODS["phase"](a,b)
        return result

    def submit_modules(self,value,entry):
        """Route by module data requirements, never by the selected UI mode."""
        from copy import deepcopy
        nodes, state, _ = normalize_nodes(value["steps"])
        correlation = next((n for n in nodes if n["enabled"] and n["algorithm"]=="correlation"),None)
        alignment = next((n for n in nodes if n["enabled"] and n["algorithm"]=="alignment"),None)
        if alignment is None:
            raise ValueError("时序关联需要分幅模块提供对齐双路")
        selection = value["selection"]
        config = deepcopy(value.get("sequence_context") or {})
        n = entry["meta"]["frame_count"]
        config.update(geometry=alignment["parameters"],workflow=list(nodes),workflow_version=2)
        config.setdefault("center",(selection.get("current") or {}).get("index"))
        config.setdefault("start",0)
        config.setdefault("end",n)
        config.setdefault("roi",selection.get("roi"))
        config.setdefault("points",[])
        if selection["mode"]=="range" and "sequence_context" not in value:
            if entry["meta"]["kind"]=="video":
                raise ValueError("时序流程请使用全部数据或在分幅布局中指定精确帧范围")
            config.update(start=int(selection["start"]),end=int(selection["end"]))
        if correlation:
            p = correlation["parameters"]
            task = "curve" if p["analysis"]=="curve" else value["mode"]
            config.update(lag_min=p["lag_min"],lag_max=p["lag_max"],lag_step=p["lag_step"],lag_list=p["lag_list"] if task=="curve" else [p["map_lag"]])
        else:
            task = "statistics"
            config.update(window=1,min_samples=1,lag_list=[0])
        return self.submit(dict(source_id=entry["id"],config=config,task=task,name=value.get("name"),output_dir=value.get("output_dir"),config_revision=value.get("config_revision")))

    def preflight(self,value):
        entry = self._source(value)
        if entry.get("result_kind")=="label":
            raise ValueError("标签结果不能隐式当作光强计算关联")
        config = self.resource_config(value["config"])
        result = preflight(entry["meta"],config,value.get("task","trial"))
        result["devices"] = self.devices
        return result

    def resource_config(self,value):
        policy = self.workbench.resources
        settings = dict(value)
        settings["memory_mb"] = min(settings.get("memory_mb") or policy.job_mb,policy.job_mb,4096)
        settings.setdefault("backend",policy.backend)
        return CorrelationConfig.from_dict(settings)

    def submit(self,value):
        entry = self._source(value)
        if value["config"].get("workflow_version",1)>=2 and not temporal(value["config"].get("workflow",())):
            config = value["config"]
            task = value.get("task","trial")
            if task not in {"trial","run"}:
                raise ValueError("当前流程没有对应时序模块；可按单帧试算或运行图像流程")
            frame = self._frame(entry,config.get("center",0))
            return self.workbench.submit(dict(mode=task,source_ids=[entry["id"]],steps=config.get("workflow",[]),selection=dict(mode="current" if task=="trial" else "all",current=frame["frame"],roi=config.get("roi")),index_selection=[config.get("start",0),config.get("end") or entry["meta"]["frame_count"]] if task=="run" else None,name=value.get("name"),output_dir=value.get("output_dir"),config_revision=value.get("config_revision")))
        config = self.resource_config(value["config"])
        if config.workflow is not None:
            self.workbench.validate_steps(list(config.workflow))
            normalize_nodes(config.workflow,input_kind=entry.get("result_kind","image"))
        elif entry.get("result_kind")=="label":
            raise ValueError("标签结果不能隐式当作光强计算关联")
        task = value.get("task","trial")
        estimate = preflight(entry["meta"],config,task)
        if config.backend == "gpu" and not self.devices["gpu"]:
            raise ValueError(self.devices["reason"])
        identifier = uuid.uuid4().hex[:12]
        name = display_name(value.get("name"), automatic_name(entry["meta"]["name"], {"statistics":"全源统计", "trial":"空间图试算", "run":"空间系数图", "curve":"τ 扫描曲线"}[task]))
        payload = {"id":identifier,"source":entry["source"],"revision":entry["meta"]["revision"],"config":config.to_dict(),"task":task,"output_dir":str(Path(value.get("output_dir") or self.workbench.workspace/"outputs").resolve()),"cache_root":str(self.workbench.workspace/".eip-cache"/"split"),"lut_sha256":estimate["lut_sha256"],"provenance":{"source_id":entry["id"],"lineage":entry.get("lineage",[])}}
        payload["display_name"] = name
        payload["module_fingerprints"] = estimate["module_fingerprints"]
        if value.get("calibration_index") is not None:
            payload["provenance"]["calibration_reference"] = self._frame(entry,value["calibration_index"])["frame"]
        payload = json.loads(json.dumps(payload,allow_nan=False))
        job = {"id":identifier,"mode":"split","task":task,"name":entry["meta"]["name"],"source_ids":[entry["id"]],"source_count":1,"status":"queued","completed":0,"total":None,"logs":[],"payload":payload,"base_shape":entry["meta"]["shape"],"created":time.time(),"config_revision":value.get("config_revision")}
        job.update(name=name, source_name=entry["meta"]["name"], summary=split_summary(config,task))
        self.workbench.enqueue(identifier,job)
        return {"id":identifier,"estimate":estimate}

    def completed(self,identifier):
        job = self.workbench.job(identifier)
        if job["mode"] != "split" or job["status"] != "succeeded":
            raise ValueError("请选择已完成的分幅任务")
        directory = Path(job["directory"])
        run = json.loads((directory/"run.json").read_text(encoding="utf-8"))
        if run.get("mode") != "split_correlation" or run.get("status") != "succeeded":
            raise ValueError("该分幅输出未完成")
        return job,directory,run

    def result(self,identifier):
        job,directory,run = self.completed(identifier)
        run.setdefault("display_name",job["name"])
        curves = json.loads((directory/"curves.json").read_text(encoding="utf-8")) if (directory/"curves.json").is_file() else None
        return {"id":identifier,"directory":str(directory),"run":run,"curve_result":curves}

    def slice(self,identifier,value):
        _,directory,run = self.completed(identifier)
        if run["task"] not in {"trial","run"}:
            raise ValueError("该任务只保存曲线或全源统计，没有系数图")
        pixels,count,ref,run = read_slice(directory,value.get("time_index",0),value.get("lag_index",0))
        display = value.get("display") or {"low":.8,"high":1.2,"gamma":1,"colormap":"heat"}
        q = run["analysis_region"]
        snapshot = self._snapshot(pixels,{"index":ref["center_index"],"time":ref["center_time"],"source_number":ref["source_number"],"trigger_time":ref["trigger_time"]},kind="label" if run.get("value_kind")=="label" else "coefficient",origin=(q["x"],q["y"]),base_shape=run["geometry"]["shape"],display=display)
        return {"frame":snapshot,"window_ref":ref,"valid_pixels":int(np.isfinite(pixels).sum()),"count_min":int(count.min()),"count_max":int(count.max())}

    def statistics_view(self,identifier,value):
        _,directory,run = self.completed(identifier)
        if not run.get("statistics",{}).get("frames"):
            raise ValueError("该流程没有全源统计模块；结果未计算背景均值或方差")
        mode = value.get("view","mean_left")
        if mode not in {"mean_left","mean_right","variance_left","variance_right","fluctuation_left","fluctuation_right"}:
            raise ValueError("未知预处理检查图")
        channel = 0 if mode.endswith("left") else 1
        if mode.startswith("fluctuation"):
            view_config = {**run["config"],"lag_list":[0],"lag_min":0,"lag_max":0,"lag_step":1}
            config = CorrelationConfig.from_dict(view_config)
            entry = {"source":run["source"],"meta":run["data_spec"]}
            if config.workflow is not None:
                prefix = []
                for node in deepcopy(config.workflow):
                    if node["enabled"] and node["algorithm"] in {"statistics","correlation"}:
                        break
                    saved = run.get("module_lut_snapshots",{}).get(node["id"])
                    if not saved and node["algorithm"]=="calibration" and node["parameters"].get("lut_path") and run.get("lut_snapshot") and not run.get("module_lut_snapshots"):
                        saved = {"file":run["lut_snapshot"],"sha256":run["lut_snapshot_sha256"]}
                    if saved:
                        path = directory/saved["file"]
                        if sha256_file(path)!=saved["sha256"]:
                            raise ValueError("模块 LUT 快照已变化，无法重建历史涨落")
                        node["parameters"]["lut_path"] = str(path)
                    prefix.append(node)
            else:
                if run.get("lut_snapshot"):
                    photo = Photometry(replace(config.photometry,lut_path=str(directory/run["lut_snapshot"])))
                    expected = run["lut_snapshot_sha256"]
                else:
                    photo = Photometry(config.photometry)
                    expected = run["photometry"]["lut_sha256"]
                if photo.record["lut_sha256"] != expected:
                    raise ValueError("线性化 LUT 已改变，无法用旧统计重建涨落；请重新计算")
                parameters = asdict(photo.config)
                parameters.pop("normalization",None)
                prefix = [{"algorithm":"calibration","parameters":parameters},{"algorithm":"alignment","parameters":asdict(config.geometry)}]
            event = self.workbench.preview.request(entry["source"],"index",index=value.get("index",0),modules=prefix)
            if event["meta"]["revision"]!=entry["meta"]["revision"]:
                raise ValueError("来源已改变，无法重建历史涨落")
            pair = event["processed"]
            mean = np.load(directory/"mean.npy",allow_pickle=False)[channel]
            pixels = (pair[channel]-mean)/run["statistics"]["scales"][channel]
            finite_pixels = pixels[np.isfinite(pixels)]
            bound = max(1e-9,float(np.percentile(np.abs(finite_pixels),99))) if finite_pixels.size else 1
            display = {"low":-bound,"high":bound,"gamma":1,"colormap":"heat"}
            ref = event["frame"]
        else:
            pixels = np.load(directory/("variance.npy" if mode.startswith("variance") else "mean.npy"),allow_pickle=False)[channel]
            display,ref = None,{"index":None,"time":None}
        return {"frame":self._snapshot(pixels,ref,display=display),"view":mode,"scope":"all_source_frames","frames":run["statistics"]["frames"],"scales":run["statistics"]["scales"],"geometry":run["geometry"],"config":run["config"],"source_id":run.get("provenance",{}).get("source_id")}

    def promote(self,identifier,lag_index):
        job,directory,run = self.completed(identifier)
        integer(lag_index,"延迟索引",0)
        if run["task"] not in {"trial","run"} or lag_index>=len(run["lags"]):
            raise ValueError("请选择完成的系数图及有效延迟")
        path = directory/f"lag_{lag_index:04d}.eip.json"
        write_json(path,{"schema_version":1,"type":"eip_correlation_lag","directory":str(directory),"lag_index":lag_index})
        previous = run.get("provenance",{}).get("lineage",[])
        if len(previous)>=32:
            raise ValueError("来源链最多 32 个阶段")
        origin = run["geometry"]["origin_in_sensor"]
        region = run["analysis_region"]
        lineage = [*previous,{"run_id":identifier,"manifest":str(directory/"run.json"),"manifest_sha256":sha256_file(directory/"run.json"),"parent_source_id":run.get("provenance",{}).get("source_id"),"parent_source":run["source"],"crop_origin":[origin[0]+region["x"],origin[1]+region["y"]],"parent_shape":run["data_spec"]["shape"],"lag_frames":run["lags"][lag_index],"window":run["config"]["window"],"reference_geometry":run["geometry"],"analysis_region":region,"timestamp_source":"correlation_window_ref"}]
        lineage[-1].update(result_semantics=run.get("result_semantics","g2"),value_kind=run.get("value_kind","image"))
        title = "派生关联后处理" if run.get("result_semantics")=="processed_g2" else "派生 g²"
        return self.workbench.open(asdict(SourceSpec(str(path),"correlation")),role="input",lineage=lineage,display_name=f"{title} · τ={run['lags'][lag_index]:+d}",result_kind="label" if run.get("value_kind")=="label" else "coefficient")

    def restore(self,path):
        path = Path(path).resolve()
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("schema_version")!=1 or run.get("mode")!="split_correlation" or run.get("status")!="succeeded" or path.parent.name.endswith(".partial"):
            raise ValueError("请选择完成的分幅 run.json")
        identifier = run["run_id"]
        if not isinstance(identifier,str) or len(identifier)>64:
            raise ValueError("运行标识无效")
        with self.workbench.lock:
            existing = self.workbench.jobs.get(identifier)
            if existing:
                if existing.get("directory") != str(path.parent):
                    raise ValueError("运行标识与已有输出冲突")
                return {"id":identifier}
        job = {"id":identifier,"mode":"split","task":run["task"],"name":Path(run["source"]["path"]).name,"source_ids":[run.get("provenance",{}).get("source_id")],"source_count":1,"status":"succeeded","completed":1,"total":1,"logs":[],"directory":str(path.parent),"run":run,"created":path.stat().st_mtime,"payload":{}}
        kind = {"statistics":"全源统计","curve":"τ 扫描曲线","trial":"空间图试算","run":"空间系数图"}[run["task"]]
        job.update(name=run.get("display_name") or automatic_name(job["name"],kind,job["created"]),source_name=Path(run["source"]["path"]).name,summary=saved_split_summary(run["config"],run["task"]))
        self.workbench.enqueue(identifier,job,queue_job=False)
        self.workbench.results.remember()
        return {"id":identifier}
