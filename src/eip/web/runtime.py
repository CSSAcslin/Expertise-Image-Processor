from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict
from pathlib import Path
import base64
import csv
import io
import json
import math
import multiprocessing
import queue
import threading
import time
import uuid

import numpy as np
from PIL import Image

from eip.algorithms import BUILTINS, load_manifest, validate_parameters
from eip.algorithms import sha256_file
from eip.display import DisplaySettings, display_range, labels_to_display, to_display
from eip.domain import ROI, SourceSpec, StepSpec, selection_from_dict
from eip.persistence import read_project, write_json
from eip.preview_worker import preview_entry
from eip.processing import job_entry, statistics
from eip.readers import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from eip.modes.split_correlation.runner import split_job_entry
from eip.web.split import SplitService
from eip.web.records import display_name, automatic_name, saved_split_summary
from eip.module_registry import catalog, normalize_nodes, temporal, output_shape
from eip.module_contracts import compile_flow, source_contract
from eip.sequence_processing import sequence_job_entry


class PreviewBroker:
    def __init__(self):
        self.context = multiprocessing.get_context("spawn")
        self.lock = threading.Lock()
        self.capacity = threading.BoundedSemaphore(2)
        self._start()

    def _start(self):
        self.commands = self.context.Queue(maxsize=2)
        self.results = self.context.Queue(maxsize=2)
        self.process = self.context.Process(target=preview_entry, args=(self.commands, self.results), name="EIP-web-preview")
        self.process.start()

    def request(self, source: dict, action="open", **values):
        if not self.capacity.acquire(blocking=False):
            raise ValueError("正在定位，请等待当前请求完成")
        try:
            with self.lock:
                if not self.process.is_alive():
                    self._dispose()
                    self._start()
                identifier = uuid.uuid4().hex
                self.commands.put({"id": identifier, "source": source, "action": action, **values}, timeout=2)
                try:
                    result = self.results.get(timeout=60)
                except queue.Empty as error:
                    self._dispose()
                    self._start()
                    raise ValueError("读取超过一分钟；预览进程已重置") from error
                if result["type"] == "failed":
                    raise ValueError(result["message"])
                return result
        finally:
            self.capacity.release()

    def _dispose(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(0.5)
        for channel in [self.commands, self.results]:
            channel.cancel_join_thread()
            channel.close()

    def close(self):
        with self.lock:
            self._dispose()


class FrameStore:
    def __init__(self):
        self.frames = OrderedDict()
        self.size = 0
        self.lock = threading.RLock()

    def add(self, pixels, ref, origin, base_shape, kind):
        token = uuid.uuid4().hex
        with self.lock:
            while self.frames and (self.size + pixels.nbytes > 128 * 1024 * 1024 or len(self.frames) >= 6):
                _, old = self.frames.popitem(last=False)
                self.size -= old["pixels"].nbytes
            self.frames[token] = {"pixels": pixels, "ref": ref, "origin": origin, "base_shape": base_shape, "kind": kind}
            self.size += pixels.nbytes
        return token

    def get(self, token):
        with self.lock:
            if token not in self.frames:
                raise ValueError("该帧缓存已过期，请重新定位或试算")
            return self.frames[token]

    def annotate(self,token,**metadata):
        with self.lock:
            self.get(token)["metadata"] = metadata

    def render(self, token, settings=None, maximum=1800):
        record = self.get(token)
        pixels = record["pixels"]
        display = DisplaySettings(**(settings or {}))
        low, high = display_range(pixels) if display.low is None or display.high is None else (display.low, display.high)
        display = DisplaySettings(low, high, display.gamma, display.colormap)
        rgb = labels_to_display(pixels) if record["kind"] == "label" else to_display(pixels, display)
        if record["kind"] == "coefficient" and pixels.ndim == 2:
            rgb[~np.isfinite(pixels)] = [128,64,148]
        image = Image.fromarray(rgb)
        image.thumbnail((maximum, maximum), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=2)
        finite = pixels[np.isfinite(pixels)]
        samples = finite[::max(1, finite.size // 100000)]
        counts, edges = np.histogram(samples, bins=64) if samples.size else (np.zeros(64, dtype=int), np.linspace(0, 1, 65))
        return {"token": token, "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), "shape": list(pixels.shape), "base_shape": list(record["base_shape"]), "origin": list(record["origin"]), "ref": record["ref"], "dtype": str(pixels.dtype), "kind": record["kind"], "display": asdict(display), "histogram": counts.tolist(), "histogram_range": [float(edges[0]), float(edges[-1])], "histogram_samples": int(samples.size), "statistics": statistics(pixels), "display_shape": [image.height, image.width], "downsampled": image.width != pixels.shape[1] or image.height != pixels.shape[0],**record.get("metadata",{})}

    def pixel(self, token, x, y):
        record = self.get(token)
        array = record["pixels"]
        x -= record["origin"][0]
        y -= record["origin"][1]
        if not (0 <= x < array.shape[1] and 0 <= y < array.shape[0]):
            raise ValueError("该位置没有像素")
        value = array[y, x]
        def clean(item):
            item = item.item() if isinstance(item, np.generic) else item
            if isinstance(item, int) and not isinstance(item, bool) and abs(item) > 2 ** 53 - 1:
                return str(item)
            return str(item) if isinstance(item, float) and not math.isfinite(item) else item
        value = [clean(v) for v in value] if isinstance(value, np.ndarray) else clean(value)
        return {"x": x + record["origin"][0], "y": y + record["origin"][1], "value": value, "dtype": str(array.dtype)}

    def region(self, token, x, y, width, height):
        record = self.get(token)
        roi = ROI(x - record["origin"][0], y - record["origin"][1], width, height)
        roi.validate(record["pixels"].shape)
        pixels = record["pixels"][roi.y:roi.y + height, roi.x:roi.x + width]
        return {"roi": {"x": x, "y": y, "width": width, "height": height}, "dtype": str(pixels.dtype), "pixel_count": width * height, "statistics": statistics(pixels)}


class Workbench:
    def __init__(self, workspace: Path, restored: dict | None = None):
        self.workspace = workspace.resolve()
        self.frames = FrameStore()
        self.lock = threading.RLock()
        self.sources = OrderedDict()
        self.algorithms = {spec.id: spec for spec in BUILTINS}
        self.jobs = OrderedDict()
        self.pending = deque()
        self.active = None
        self.project_path = None
        self.context = multiprocessing.get_context("spawn")
        self.closed = threading.Event()
        if restored:
            self.restore_session(restored)
        self.preview = PreviewBroker()
        self.split = SplitService(self)
        self.monitor = threading.Thread(target=self._monitor, name="EIP-job-monitor", daemon=True)
        self.monitor.start()

    def restore_session(self, value):
        """Restore a local maintenance snapshot without restarting completed jobs."""
        if value.get("schema_version") != 1 or Path(value["state"]["workspace"]).resolve() != self.workspace:
            raise ValueError("会话快照不属于当前工作目录")
        entries, jobs = value["state"]["sources"], value["jobs"]
        if len(entries) > 64 or len(jobs) > 64:
            raise ValueError("会话快照超出来源或任务记录上限")
        if any(job["status"] not in {"succeeded", "failed", "cancelled", "completed_with_errors"} for job in jobs):
            raise ValueError("请等待任务完成后再恢复会话")
        if len({entry["id"] for entry in entries}) != len(entries):
            raise ValueError("会话快照存在重复来源")
        for spec in value["state"]["algorithms"]:
            if spec.get("manifest"):
                self.register(spec["manifest"])
        self.sources = OrderedDict((entry["id"], entry) for entry in entries)
        self.jobs = OrderedDict((job["id"], job) for job in reversed(jobs))
        # Old maintenance snapshots predate editable result names / summaries.
        for job in self.jobs.values():
            run = job.get("run",{})
            source = run.get("source") or next(iter(run.get("spec",{}).get("sources",[])),{})
            source_name = Path(source["path"]).name if source.get("path") else job["name"]
            job.setdefault("source_name",source_name)
            if not run.get("display_name") and job["name"] == source_name:
                kind = {"statistics":"全源统计","curve":"τ 扫描曲线","trial":"空间图试算","run":"空间系数图"}.get(job.get("task"),"试算" if job["mode"]=="trial" else "处理结果")
                job["name"] = automatic_name(source_name,kind,job["created"])
            if run:
                run.setdefault("display_name",job["name"])
            if job["mode"] == "split" and run.get("config"):
                job.setdefault("summary",saved_split_summary(run["config"],job["task"]))
            elif run.get("spec"):
                spec = run["spec"]
                job.setdefault("summary",f"{len(spec.get('steps',[]))} 个步骤 · {spec.get('selection',{}).get('mode','全部')}")
        for entry in self.sources.values():
            if entry.get("run_id") in self.jobs and not entry.get("display_name"):
                entry["display_name"] = self.jobs[entry["run_id"]]["name"]
        self.project_path = value["state"].get("project_path")

    def state(self):
        with self.lock:
            return {"workspace": str(self.workspace), "output_dir": str(self.workspace / "outputs"), "project_path": self.project_path, "sources": list(self.sources.values()), "algorithms": [asdict(spec) for spec in self.algorithms.values()], "modules":catalog(self.algorithms.values()), "modes": ["general","split_correlation"], "split_devices": self.split.devices}

    def open(self, value: dict, role="input", **extra):
        source = SourceSpec(**value)
        source = SourceSpec(str(Path(source.path).resolve()), source.kind, source.interval)
        if role == "result" and not extra.get("display_name"):
            record_path = Path(source.path).parent / "run.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                item = next((item for item in record.get("results", []) if item["output_directory"] == Path(source.path).name), None)
                if item:
                    extra["display_name"] = f"{Path(item['source']['path']).name} · {record['run_id'][:6]}"
        event = self.preview.request(asdict(source))
        if event["type"] != "frame":
            raise ValueError("数据源没有可显示帧")
        token = self.frames.add(event["pixels"], event["frame"], extra.get("origin", [0, 0]), extra.get("base_shape", event["pixels"].shape[:2]), extra.get("result_kind", "image"))
        snapshot = self.frames.render(token)
        if event.get("contract"):
            snapshot.update(contract=event["contract"],signal_values=event.get("signal_values"))
            self.frames.annotate(token,contract=event["contract"],signal_values=event.get("signal_values"))
        thumbnail = self.frames.render(token, maximum=160)["image"]
        identifier = extra.pop("identifier", f"result_{source.key}" if role == "result" else source.key)
        entry = {"id": identifier, "source": asdict(source), "meta": event["meta"], "role": role, "thumbnail": thumbnail, **extra}
        with self.lock:
            if identifier not in self.sources and len(self.sources) >= 64:
                raise ValueError("当前工作区最多保留 64 个来源，请新建工作区")
            self.sources[identifier] = entry
        return {"source": entry, "frame": snapshot}

    def promote(self, identifier):
        entry = self.get_source(identifier)
        if entry["role"] != "result":
            raise ValueError("请选择正式保存的处理结果")
        output_path = Path(entry["source"]["path"])
        manifest = (output_path.parent.parent if entry["source"]["kind"]=="dataset" else output_path.parent) / "run.json"
        run = json.loads(manifest.read_text(encoding="utf-8"))
        part_name = output_path.parent.name if entry["source"]["kind"]=="dataset" else output_path.name
        item = next((item for item in run["results"] if item["output_directory"] == part_name), None)
        if not item or item["status"] != "succeeded" or run["status"] not in {"succeeded", "completed_with_errors"}:
            raise ValueError("未完成的输出不能作为下一阶段输入")
        parent = self.sources.get(entry.get("input_id"))
        # The run snapshot remains authoritative even when a source reference was removed.
        previous = next((value.get("lineage", []) for value in run["spec"].get("provenance", []) if value["source_id"] == entry.get("input_id")), parent.get("lineage", []) if parent else [])
        if len(previous) >= 32:
            raise ValueError("当前来源链最多支持 32 个阶段")
        lineage = [*previous, {"run_id": run["run_id"], "manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "parent_source_id": entry.get("input_id"), "parent_source": item["source"], "crop_origin": item["origin"], "parent_shape": item["data_spec"]["shape"], "output_directory": entry["source"]["path"], "timestamp_source": "run_record"}]
        mappings = item.get("mappings") or item.get("modules",{}).get("mappings",[])
        if mappings:
            lineage[-1].update(coordinate_space="module_output_grid",mappings=mappings,modules=item["modules"].get("nodes",item["modules"].get("modules",[])))
        return self.open(entry["source"], role="input", identifier=f"input_{SourceSpec(**entry['source']).key}", display_name=f"派生 · {entry.get('display_name', entry['meta']['name'])}", result_kind=entry.get("result_kind", "image"), lineage=lineage, parent_result_id=identifier)

    def remove(self, identifier):
        with self.lock:
            self.get_source(identifier)
            if any(identifier in job["source_ids"] and job["status"] in {"queued", "running", "cancelling"} for job in self.jobs.values()):
                raise ValueError("该来源仍有待处理任务，请等待或取消任务")
            if any(entry.get("input_id") == identifier or entry.get("parent_result_id") == identifier for entry in self.sources.values()):
                raise ValueError("该来源仍关联处理结果，请先移除对应结果引用")
            del self.sources[identifier]

    def get_source(self, identifier):
        with self.lock:
            if identifier not in self.sources:
                raise ValueError("数据源不存在，请重新打开")
            return self.sources[identifier]

    def frame(self, identifier, value):
        source = self.get_source(identifier)
        action = value.get("action", "open")
        if action == "match" and source["role"] == "result":
            reference = value["reference"]
            index = None
            mapping_path = Path(source["source"]["path"])
            if source["source"]["kind"]=="dataset":
                mapping_path = mapping_path.parent
            with (mapping_path / "measurements.csv").open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    if reference.get("index") is not None and row["source_frame"]:
                        matches = int(row["source_frame"]) == reference["index"]
                    elif reference.get("pts") is not None and row.get("pts"):
                        matches = int(row["pts"]) == reference["pts"]
                    else:
                        matches = bool(row["time_seconds"] and reference.get("time") is not None and abs(float(row["time_seconds"]) - reference["time"]) < 1e-9)
                    if matches:
                        index = int(row["output_index"])
                        break
            if index is None:
                return {"unavailable": True}
            value = {**value, "index": index}
            action = "index"
        if action not in {"open", "time", "next", "previous", "index"}:
            raise ValueError("未知定位方式")
        if action == "index" and (type(value.get("index")) is not int or value["index"] < 0):
            raise ValueError("帧索引必须为非负整数")
        if action == "index" and source["meta"]["kind"] == "video" and not source["meta"].get("frame_indexed"):
            raise ValueError("该视频没有精确帧索引，请使用时间定位")
        args = {key: value[key] for key in ["time", "index"] if key in value}
        event = self.preview.request(source["source"], action, **args)
        if event["type"] == "eof":
            return {"eof": True}
        if event["meta"]["revision"] != source["meta"]["revision"]:
            raise ValueError("来源已变化，请重新打开")
        token = self.frames.add(event["pixels"], event["frame"], source.get("origin", [0, 0]), source.get("base_shape", event["pixels"].shape[:2]), source.get("result_kind", "image"))
        snapshot = self.frames.render(token, value.get("display"))
        if event.get("contract"):
            snapshot.update(contract=event["contract"],signal_values=event.get("signal_values"))
            self.frames.annotate(token,contract=event["contract"],signal_values=event.get("signal_values"))
        return snapshot

    def register(self, path):
        spec = load_manifest(path)
        if spec.id in catalog():
            raise ValueError("外部算法不能覆盖内置算法")
        with self.lock:
            self.algorithms[spec.id] = spec
        return asdict(spec)

    def validate_steps(self, values, *, check_types=True):
        nodes, state, kind = normalize_nodes(values,check_types=check_types)
        registered = catalog(self.algorithms.values())
        for node in nodes:
            if node["algorithm"] not in registered or node.get("code_hash") != registered[node["algorithm"]]["code_hash"]:
                raise ValueError("请先注册流程中的算法版本")
        return [StepSpec(**item) for item in nodes]

    def submit(self, value):
        if value.get("contract_version")==3 or any(n.get("algorithm") in {"time_crop","spatial_roi","time_sample","time_smooth","time_bin"} for n in value.get("steps",[])):
            return self.submit_sequence(value)
        mode = value.get("mode", "trial")
        if mode not in {"trial", "run"}:
            raise ValueError("未知任务类型")
        identifiers = value.get("source_ids", [])
        if not identifiers or len(set(identifiers)) != len(identifiers) or len(identifiers) > 32:
            raise ValueError("请选择 1–32 个不重复的输入来源")
        entries = [self.get_source(identifier) for identifier in identifiers]
        if any(entry["role"] != "input" for entry in entries):
            raise ValueError("请明确选择输入数据，结果浏览不会替换输入")
        steps = self.validate_steps(value["steps"])
        for entry in entries:
            normalize_nodes(value["steps"],input_kind=entry.get("result_kind","image"))
        if temporal(value["steps"]):
            if len(entries)!=1:
                raise ValueError("时序关联流程目前需要单个来源")
            return self.split.submit_modules(value,entries[0])
        selection = selection_from_dict(value["selection"])
        if mode == "trial" and (len(entries) != 1 or selection.mode != "current"):
            raise ValueError("精确试算需要单个来源的当前帧")
        if len(entries) > 1 and selection.mode == "current":
            raise ValueError("批量处理请使用全部数据或指定范围")
        if selection.mode == "range" and len({entry["meta"]["kind"] == "video" for entry in entries}) > 1:
            raise ValueError("帧范围与时间范围不能混用")
        if selection.mode == "range" and any(entry["meta"]["kind"] == "video" and entry["meta"]["timestamp_source"] == "unavailable" for entry in entries):
            raise ValueError("视频缺少逐帧时间记录，请选择当前帧或全部数据处理")
        if selection.mode == "range" and entries[0]["meta"]["kind"] != "video":
            if int(selection.start) != selection.start or int(selection.end) != selection.end:
                raise ValueError("帧范围必须使用整数，不能将小数默认为帧号")
        if selection.roi:
            for entry in entries:
                selection.roi.validate(output_shape(value["steps"],entry["meta"]["shape"]))
        _, output_type, _ = normalize_nodes(value["steps"])
        if output_type == "pair":
            raise ValueError("流程输出为双路数据；请添加「选择幅面」或二阶关联模块")
        identifier = uuid.uuid4().hex[:12]
        name = display_name(value.get("name"), automatic_name(entries[0]["meta"]["name"], "试算" if mode == "trial" else "处理结果"))
        payload = {"id": identifier, "mode": mode, "sources": [entry["source"] for entry in entries], "revisions": [entry["meta"]["revision"] for entry in entries], "steps": value["steps"], "selection": value["selection"], "output_dir": str(Path(value.get("output_dir") or self.workspace / "outputs").resolve()), "seed": 0, "config_revision": value.get("config_revision"), "display": value.get("result_display", value.get("display", {})), "provenance": [{"source_id": entry["id"], "lineage": entry.get("lineage", [])} for entry in entries]}
        payload["steps"] = [asdict(step) for step in steps]
        payload = json.loads(json.dumps(payload, allow_nan=False))
        payload["input_kinds"] = [entry.get("result_kind", "image") for entry in entries]
        if value.get("index_selection") is not None:
            bounds = value["index_selection"]
            if not isinstance(bounds,list) or len(bounds)!=2 or any(type(i) is not int for i in bounds) or not 0<=bounds[0]<bounds[1] or any(bounds[1]>entry["meta"]["frame_count"] or (entry["meta"]["kind"]=="video" and not entry["meta"].get("frame_indexed")) for entry in entries):
                raise ValueError("帧范围需为可精确定位来源内的整数半开区间")
            payload["index_selection"] = bounds
        job = {"id": identifier, "mode": mode, "name": entries[0]["meta"]["name"], "source_ids": identifiers, "source_count": len(entries), "status": "queued", "completed": 0, "total": None, "logs": [], "payload": payload, "base_shape": entries[0]["meta"]["shape"][:2], "created": time.time()}
        payload["display_name"] = name
        job["base_shape"] = output_shape(value["steps"],entries[0]["meta"]["shape"])
        job.update(name=name,source_name=entries[0]["meta"]["name"],summary=f"{sum(bool(s.get('enabled',True)) for s in value['steps'])} 个步骤 · {selection.mode}" )
        self.enqueue(identifier,job)
        return {"id": identifier}

    def module_plan(self,value):
        entry = self.get_source(value["source_id"])
        contract = source_contract(entry["meta"],entry.get("result_kind","image"))
        return compile_flow(value.get("steps",[]),contract,center=value.get("center"),single=value.get("single",False),strict=False)

    def submit_sequence(self,value):
        mode = value.get("mode","run")
        if mode not in {"run","trial"}:
            raise ValueError("未知任务类型")
        identifiers = value.get("source_ids",[])
        if not identifiers or len(identifiers)>32 or len(set(identifiers))!=len(identifiers):
            raise ValueError("请选择 1–32 个不重复的输入来源")
        entries = [self.get_source(i) for i in identifiers]
        if any(e["role"]!="input" for e in entries):
            raise ValueError("请明确选择输入来源")
        if value.get("selection",{}).get("roi") or value.get("selection",{}).get("mode") in {"range","current"}:
            raise ValueError("新流程的范围 / ROI 只能由时间裁剪与空间 ROI 模块设置")
        nodes,_,_ = normalize_nodes(value.get("steps",[]),check_types=False)
        dependencies,_,_ = normalize_nodes(value.get("preview_dependencies",[]),check_types=False)
        if dependencies and (mode!="trial" or dependencies[-1]["algorithm"]!="alignment" or len(nodes)+len(dependencies)>32):
            raise ValueError("阶段检查的几何依赖只能用于试算，并止于后续配准模块")
        registered = catalog(self.algorithms.values())
        if any(n["algorithm"] not in registered or n.get("code_hash")!=registered[n["algorithm"]]["code_hash"] for n in [*nodes,*dependencies]):
            raise ValueError("请先注册流程中的算法版本")
        center = value.get("center")
        sample = value.get("sample_index",0)
        if center is not None and (type(center) is not int or center<0) or type(sample) is not int or sample<0:
            raise ValueError("试算样本与关联中心必须为非负整数位置")
        plans = [compile_flow(nodes,source_contract(e["meta"],e.get("result_kind","image")),center=center,single=mode=="trial",following_nodes=dependencies) for e in entries]
        identifier = uuid.uuid4().hex[:12]
        name = display_name(value.get("name"),automatic_name(entries[0]["meta"]["name"],"模块试算" if mode=="trial" else "模块处理"))
        payload = dict(id=identifier,mode=mode,contract_version=3,steps=list(nodes),sources=[e["source"] for e in entries],revisions=[e["meta"]["revision"] for e in entries],center=center,sample_index=sample,output_dir=str(Path(value.get("output_dir") or self.workspace/"outputs").resolve()),display_name=name,seed=0,config_revision=value.get("config_revision"),provenance=[dict(source_id=e["id"],lineage=e.get("lineage",[])) for e in entries])
        payload = json.loads(json.dumps(payload,allow_nan=False))
        payload["input_kinds"] = [e.get("result_kind","image") for e in entries]
        payload["preview_dependencies"] = list(dependencies)
        job = dict(id=identifier,mode=mode,engine="sequence",contract_version=3,name=name,source_name=entries[0]["meta"]["name"],source_ids=identifiers,source_count=len(entries),status="queued",completed=0,total=None,logs=[],payload=payload,created=time.time(),summary=f"{len(nodes)} 个模块 · {plans[0]['output']['label']}",base_shape=entries[0]["meta"]["shape"][:2])
        self.enqueue(identifier,job)
        return dict(id=identifier,plan=plans[0])

    def sequence_result(self,identifier,value=None):
        job = self.job(identifier)
        if job.get("engine")!="sequence" or job["status"] not in {"succeeded","completed_with_errors"}:
            raise ValueError("请选择已完成的模块流程结果")
        items = job["run"]["results"]
        if value is None:
            return dict(id=identifier,name=job["name"],mode=job["mode"],results=items)
        position = value.get("source",0)
        if type(position) is not int or not 0<=position<len(items) or items[position]["status"]!="succeeded":
            raise ValueError("该来源没有完整结果")
        item = items[position]
        from eip.sequence_storage import DatasetReader
        reader = DatasetReader(SourceSpec(str(Path(job["directory"])/item["output_directory"]/"dataset.json"),"dataset"))
        try:
            c = item["contract"]
            if "Y" not in c["axes"]:
                indices = sorted(set([*range(0,c["length"],max(1,math.ceil(c["length"]/2000))),c["length"]-1]))
                refs = [reader.sample_ref(i) for i in indices]
                values = reader.values[indices].reshape(len(indices),-1)
                return dict(kind="signals",contract=c,indices=indices,references=refs,values=[[float(v) if np.isfinite(v) else None for v in row] for row in values],downsampled=len(indices)<c["length"],total=c["length"])
            index,channel = value.get("index",0),value.get("channel",0)
            if type(index) is not int or not 0<=index<c["length"] or type(channel) is not int or channel not in {0,1}:
                raise ValueError("结果样本或幅面超出范围")
            pixels = reader.read_sample(index)
            if c["axes"].startswith("P"):
                pixels = pixels[channel]
            ref = reader.sample_ref(index)
            token = self.frames.add(pixels,ref,[0,0],pixels.shape[:2],item["result_kind"])
            return dict(kind="image",contract=c,frame=self.frames.render(token,value.get("display")),reference=ref)
        finally:
            reader.close()

    def restore_sequence(self,path):
        from datetime import datetime
        path = Path(path).resolve()
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("contract_version")!=3 or run.get("status") not in {"succeeded","completed_with_errors"}:
            raise ValueError("请选择已完成的模块流程 run.json")
        identifier = run["run_id"]
        payload = run["spec"]
        with self.lock:
            if identifier in self.jobs:
                return dict(id=identifier,engine="sequence")
        good = [r for r in run["results"] if r["status"]=="succeeded"]
        if not good:
            raise ValueError("此记录没有完整输出")
        from eip.sequence_storage import DatasetReader
        for item in good:
            if Path(item["output_directory"]).name!=item["output_directory"]:
                raise ValueError("运行记录的结果目录无效")
            reader = DatasetReader(SourceSpec(str(path.parent/item["output_directory"]/"dataset.json"),"dataset"))
            reader.close()
        ids = [p["source_id"] for p in payload.get("provenance",[])]
        job = dict(id=identifier,mode=payload["mode"],engine="sequence",contract_version=3,name=run.get("display_name",identifier),source_name=good[0]["data_spec"]["name"],source_ids=ids,source_count=len(payload["sources"]),status=run["status"],completed=sum(r["completed_frames"] for r in good),total=None,logs=[],payload=payload,run=run,directory=str(path.parent),result_available=True,created=datetime.fromisoformat(run["created_at"]).timestamp(),summary=f"{len(payload['steps'])} 个模块 · {good[0]['contract']['label']}")
        self.enqueue(identifier,job,queue_job=False)
        if payload["mode"]=="run":
            for position,item in enumerate(run["results"]):
                if item["status"]=="succeeded" and not any(s.get("run_id")==identifier and Path(s["source"]["path"]).parent.name==item["output_directory"] for s in self.sources.values()):
                    self.open(asdict(SourceSpec(str(path.parent/item["output_directory"]/"dataset.json"),"dataset")),role="result",input_id=ids[position] if position<len(ids) else None,result_kind=item["result_kind"],run_id=identifier,display_name=job["name"])
        return dict(id=identifier,engine="sequence")

    def enqueue(self,identifier,job,*,queue_job=True):
        with self.lock:
            if len(self.pending) >= 32:
                raise ValueError("任务队列已满")
            while len(self.jobs) >= 64:
                oldest = next((key for key, job in self.jobs.items() if job["status"] in {"succeeded", "failed", "cancelled", "completed_with_errors"}), None)
                if oldest is None:
                    raise ValueError("任务记录已满")
                self.jobs.pop(oldest)
            self.jobs[identifier] = job
            if queue_job:
                self.pending.append(identifier)

    def job_list(self):
        with self.lock:
            return [{key: value for key, value in job.items() if key not in {"payload", "result", "run"}} for job in reversed(self.jobs.values())]

    def job(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise ValueError("任务记录不存在")
            return dict(self.jobs[identifier])

    def rename_job(self, identifier, name):
        name = display_name(name)
        with self.lock:
            if identifier not in self.jobs:
                raise ValueError("任务记录不存在")
            job = self.jobs[identifier]
            job["name"] = name
            job.setdefault("payload",{})["display_name"] = name
            if job.get("run"):
                job["run"]["display_name"] = name
            if job.get("directory"):
                path = Path(job["directory"])/"run.json"
                if path.is_file():
                    run = json.loads(path.read_text(encoding="utf-8"))
                    run["display_name"] = name
                    write_json(path,run)
            for entry in self.sources.values():
                if entry.get("run_id") == identifier:
                    entry["display_name"] = name
            return {"id":identifier,"name":name}

    def cancel(self, identifier):
        with self.lock:
            job = self.jobs[identifier]
            if identifier in self.pending:
                self.pending.remove(identifier)
                job["status"] = "cancelled"
            elif self.active and self.active["id"] == identifier:
                self.active["cancel"].set()
                job["status"] = "cancelling"

    def _monitor(self):
        while not self.closed.wait(0.02):
            with self.lock:
                if not self.active and self.pending:
                    identifier = self.pending.popleft()
                    results = self.context.Queue(maxsize=64)
                    cancel = self.context.Event()
                    target = sequence_job_entry if self.jobs[identifier].get("engine")=="sequence" else split_job_entry if self.jobs[identifier]["mode"] == "split" else job_entry
                    process = self.context.Process(target=target, args=(self.jobs[identifier]["payload"], results, cancel), name="EIP-web-compute")
                    try:
                        process.start()
                    except Exception as error:
                        self.jobs[identifier].update(status="failed", error=str(error))
                        results.close()
                        continue
                    self.active = {"id": identifier, "process": process, "results": results, "cancel": cancel, "terminal": False, "dead_since": None}
                    self.jobs[identifier]["status"] = "running"
                active = self.active
            if active is None:
                continue
            for _ in range(128):
                try:
                    event = active["results"].get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle(active, event)
                except Exception as error:
                    with self.lock:
                        self.jobs[active["id"]]["status"] = "failed"
                        self.jobs[active["id"]]["error"] = str(error)
                        active["terminal"] = True
            if not active["process"].is_alive():
                active["dead_since"] = active["dead_since"] or time.monotonic()
                if time.monotonic() - active["dead_since"] < 0.15:
                    continue
                with self.lock:
                    if not active["terminal"]:
                        self.jobs[active["id"]].update(status="failed", error="计算进程意外退出，临时输出未发布")
                    active["process"].join()
                    active["results"].cancel_join_thread()
                    active["results"].close()
                    self.active = None

    def _handle(self, active, event):
        with self.lock:
            job = self.jobs[active["id"]]
            kind = event["type"]
            if kind == "progress":
                job.update(completed=event["completed"], total=event["total"], source_position=event["source"], progress_name=event["name"])
            elif kind == "log":
                job["logs"] = (job["logs"] + [event["message"]])[-100:]
            elif kind in {"cancelled", "failed"}:
                job.update(status=kind, error=event.get("message"), directory=event.get("directory"))
                active["terminal"] = True
            elif kind == "split_completed":
                event["run"]["display_name"] = job["name"]
                write_json(Path(event["directory"])/"run.json",event["run"])
                job.update(status="succeeded",run=event["run"],directory=event["directory"],result_available=True)
                active["terminal"] = True
            elif kind == "trial":
                token = self.frames.add(event["pixels"], event["frame"], event["origin"], event.get("base_shape",job["base_shape"]), event["kind"])
                job["result"] = self.frames.render(token, job["payload"].get("display"))
                job.update(status="succeeded", completed=1, total=1, result_available=True)
                active["terminal"] = True
            elif kind == "sequence_completed":
                event["run"]["display_name"] = job["name"]
                write_json(Path(event["directory"])/"run.json",event["run"])
                job.update(status=event["run"]["status"],run=event["run"],directory=event["directory"],result_available=any(i["status"]=="succeeded" for i in event["run"]["results"]))
                active["terminal"] = True
                if job["mode"]=="run":
                    for position,item in enumerate(event["run"]["results"]):
                        if item["status"]!="succeeded":
                            continue
                        try:
                            path = Path(event["directory"])/item["output_directory"]/"dataset.json"
                            self.open(asdict(SourceSpec(str(path),"dataset")),role="result",input_id=job["source_ids"][position],result_kind=item["result_kind"],run_id=job["id"],display_name=job["name"])
                        except Exception as error:
                            job["logs"].append(f"结果已保存，自动预览失败：{error}")
            elif kind == "completed":
                event["run"]["display_name"] = job["name"]
                write_json(Path(event["directory"])/"run.json",event["run"])
                job.update(status=event["run"]["status"], run=event["run"], directory=event["directory"], result_available=False)
                active["terminal"] = True
                for position, item in enumerate(event["run"]["results"]):
                    if item["status"] != "succeeded":
                        continue
                    path = Path(event["directory"]) / item["output_directory"]
                    # Result readers retain CSV timestamp mappings.
                    try:
                        self.open(asdict(SourceSpec(str(path), "sequence")), role="result", input_id=job["source_ids"][position], origin=item["origin"], base_shape=item.get("base_shape",item["data_spec"]["shape"][:2]), result_kind=item["result_kind"], run_id=job["id"],display_name=job["name"])
                    except Exception as error:
                        job["logs"].append(f"输出已保存，但自动预览失败：{error}")

    def browse(self, path=None, mode="data"):
        path = Path(path or self.workspace).resolve()
        if not path.is_dir():
            raise ValueError("目录不存在或无访问权限")
        extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS if mode == "data" else set() if mode == "directory" else {".json"}
        entries = []
        for item in path.iterdir():
            try:
                if item.name.startswith("."):
                    continue
                directory = item.is_dir()
                if directory or item.suffix.lower() in extensions:
                    entries.append({"name": item.name, "path": str(item), "directory": directory, "size": None if directory else item.stat().st_size})
            except OSError:
                continue
        entries.sort(key=lambda item: (not item["directory"], item["name"].lower()))
        return {"path": str(path), "parent": str(path.parent), "entries": entries[:1000], "truncated": len(entries) > 1000, "drives": [f"{letter}:/" for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{letter}:/").exists()]}

    def inspect_path(self, path):
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise ValueError("路径不存在，请检查文件或目录名称")
        if not resolved.is_file() and not resolved.is_dir():
            raise ValueError("路径不是可读取的文件或目录")
        return {"path": str(resolved), "directory": resolved.is_dir(), "suffix": resolved.suffix.lower(), "size": resolved.stat().st_size if resolved.is_file() else None, "supported_data": resolved.is_dir() or resolved.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}

    def save_project(self, path, view):
        self.validate_steps(view.get("steps", []),check_types=view.get("web_module_contract")!=3)
        value = {**view, "schema_version": 1, "sources": [entry["source"] for entry in self.sources.values() if entry["role"] == "input"], "results": [{"source": entry["source"], "id": entry["id"], "run_id": entry.get("run_id"), "display_name": entry.get("display_name"), "result": True, "input_key": entry.get("input_id"), "origin": entry.get("origin", [0, 0]), "base_shape": entry.get("base_shape"), "result_kind": entry.get("result_kind", "image")} for entry in self.sources.values() if entry["role"] == "result"], "split_result_paths": [str(Path(job["directory"])/"run.json") for job in self.jobs.values() if job["mode"]=="split" and job["status"]=="succeeded"]}
        value["source_entries"] = [{key: entry[key] for key in ["source", "id", "lineage", "parent_result_id", "display_name", "result_kind"] if key in entry} for entry in self.sources.values() if entry["role"] == "input"]
        value["sequence_result_paths"] = [str(Path(j["directory"])/"run.json") for j in self.jobs.values() if j.get("engine")=="sequence" and j["status"] in {"succeeded","completed_with_errors"}]
        target = Path(path).resolve()
        if target.suffix != ".json":
            raise ValueError("项目文件请使用 .json 后缀")
        if any(target == Path(entry["source"]["path"]) for entry in self.sources.values()):
            raise ValueError("项目保存不能覆盖来源")
        if any(spec.manifest and target == Path(spec.manifest) for spec in self.algorithms.values()):
            raise ValueError("项目保存不能覆盖算法描述")
        write_json(target, value)
        self.project_path = str(target)
        return {"path": str(target)}

    def load_project(self, path):
        with self.lock:
            if self.active or self.pending:
                raise ValueError("请等待或取消任务后再切换项目")
        value = read_project(Path(path))
        if len(value["sources"]) > 32:
            raise ValueError("项目加载最多支持 32 个输入来源")
        for step in value["steps"]:
            if step.get("manifest"):
                self.register(step["manifest"])
        self.validate_steps(value["steps"],check_types=value.get("web_module_contract")!=3)
        if any(not Path(source["path"]).exists() for source in value["sources"]):
            raise ValueError("项目引用的部分来源缺失，请恢复来源路径")
        previous = self.sources.copy()
        self.sources.clear()
        try:
            for position, source in enumerate(value["sources"]):
                details = value.get("source_entries", [])
                detail = details[position] if position < len(details) else {}
                if detail and detail["source"] != source:
                    raise ValueError("项目来源与来源详情顺序不一致")
                extra = {key: detail[key] for key in ["lineage", "parent_result_id", "display_name", "result_kind"] if key in detail}
                if detail.get("id"):
                    extra["identifier"] = detail["id"]
                self.open(source, **extra)
            for result in value.get("results", []):
                if Path(result["source"]["path"]).exists():
                    original = self.sources.get(result.get("input_key"))
                    base_shape = result.get("base_shape") or (original["meta"]["shape"][:2] if original else None)
                    extra = {key: result[key] for key in ["run_id", "display_name"] if result.get(key)}
                    if result.get("id"):
                        extra["identifier"] = result["id"]
                    self.open(result["source"], role="result", input_id=result.get("input_key"), origin=result.get("origin", [0, 0]), **({"base_shape": base_shape} if base_shape else {}), result_kind=result.get("result_kind", "image"), **extra)
        except Exception:
            self.sources = previous
            raise
        self.project_path = str(Path(path).resolve())
        for result_path in value.get("split_result_paths",[]):
            if Path(result_path).is_file():
                self.split.restore(result_path)
        for result_path in value.get("sequence_result_paths",[]):
            if Path(result_path).is_file():
                self.restore_sequence(result_path)
        return {"state": self.state(), "view": value}

    def close(self):
        self.closed.set()
        self.monitor.join(2)
        with self.lock:
            if self.active:
                self.active["cancel"].set()
                self.active["process"].join(0.5)
                if self.active["process"].is_alive():
                    self.active["process"].terminate()
                    self.active["process"].join(0.5)
                self.active["results"].cancel_join_thread()
                self.active["results"].close()
        self.preview.close()
