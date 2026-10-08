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
from eip.display import DisplaySettings, display_range, labels_to_display, to_display
from eip.domain import SourceSpec, StepSpec, selection_from_dict
from eip.persistence import read_project, write_json
from eip.preview_worker import preview_entry
from eip.processing import job_entry, statistics
from eip.readers import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


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

    def render(self, token, settings=None, maximum=1800):
        record = self.get(token)
        pixels = record["pixels"]
        display = DisplaySettings(**(settings or {}))
        low, high = display_range(pixels) if display.low is None or display.high is None else (display.low, display.high)
        display = DisplaySettings(low, high, display.gamma, display.colormap)
        rgb = labels_to_display(pixels) if record["kind"] == "label" else to_display(pixels, display)
        image = Image.fromarray(rgb)
        image.thumbnail((maximum, maximum), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=2)
        finite = pixels[np.isfinite(pixels)]
        samples = finite[::max(1, finite.size // 100000)]
        histogram = np.histogram(samples, bins=64)[0].tolist() if samples.size else [0] * 64
        return {"token": token, "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), "shape": list(pixels.shape), "base_shape": list(record["base_shape"]), "origin": list(record["origin"]), "ref": record["ref"], "dtype": str(pixels.dtype), "kind": record["kind"], "display": asdict(display), "histogram": histogram, "statistics": statistics(pixels), "display_shape": [image.height, image.width], "downsampled": image.width != pixels.shape[1] or image.height != pixels.shape[0]}

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
            return str(item) if isinstance(item, float) and not math.isfinite(item) else item
        value = [clean(v) for v in value] if isinstance(value, np.ndarray) else clean(value)
        return {"x": x + record["origin"][0], "y": y + record["origin"][1], "value": value, "dtype": str(array.dtype)}


class Workbench:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.preview = PreviewBroker()
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
        self.monitor = threading.Thread(target=self._monitor, name="EIP-job-monitor", daemon=True)
        self.monitor.start()

    def state(self):
        with self.lock:
            return {"workspace": str(self.workspace), "output_dir": str(self.workspace / "outputs"), "project_path": self.project_path, "sources": list(self.sources.values()), "algorithms": [asdict(spec) for spec in self.algorithms.values()]}

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
        thumbnail = self.frames.render(token, maximum=160)["image"]
        entry = {"id": source.key, "source": asdict(source), "meta": event["meta"], "role": role, "thumbnail": thumbnail, **extra}
        with self.lock:
            if source.key not in self.sources and len(self.sources) >= 64:
                raise ValueError("当前工作区最多保留 64 个来源，请新建工作区")
            self.sources[source.key] = entry
        return {"source": entry, "frame": snapshot}

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
            with (Path(source["source"]["path"]) / "measurements.csv").open(encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    matches = (row["time_seconds"] and abs(float(row["time_seconds"]) - reference["time"]) < 1e-6) if reference.get("time") is not None else (row["source_frame"] and int(row["source_frame"]) == reference.get("index"))
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
        args = {key: value[key] for key in ["time", "index"] if key in value}
        event = self.preview.request(source["source"], action, **args)
        if event["type"] == "eof":
            return {"eof": True}
        if event["meta"]["revision"] != source["meta"]["revision"]:
            raise ValueError("来源已变化，请重新打开")
        token = self.frames.add(event["pixels"], event["frame"], source.get("origin", [0, 0]), source.get("base_shape", event["pixels"].shape[:2]), source.get("result_kind", "image"))
        return self.frames.render(token, value.get("display"))

    def register(self, path):
        spec = load_manifest(path)
        if spec.id in {item.id for item in BUILTINS}:
            raise ValueError("外部算法不能覆盖内置算法")
        with self.lock:
            self.algorithms[spec.id] = spec
        return asdict(spec)

    def validate_steps(self, values):
        if len(values) > 32:
            raise ValueError("处理链最多 32 步")
        steps = [StepSpec(**item) for item in values]
        for step in steps:
            spec = self.algorithms.get(step.algorithm)
            if spec is None:
                raise ValueError("请先注册处理链中的算法")
            if spec.manifest != step.manifest or spec.code_hash != step.code_hash or spec.version != step.version:
                raise ValueError("算法版本已变化，请重新添加该步骤")
            validate_parameters(spec, step.parameters)
        return steps

    def submit(self, value):
        mode = value.get("mode", "trial")
        if mode not in {"trial", "run"}:
            raise ValueError("未知任务类型")
        identifiers = value.get("source_ids", [])
        if not identifiers or len(set(identifiers)) != len(identifiers) or len(identifiers) > 32:
            raise ValueError("请选择 1–32 个不重复的输入来源")
        entries = [self.get_source(identifier) for identifier in identifiers]
        if any(entry["role"] != "input" for entry in entries):
            raise ValueError("请明确选择输入数据，结果浏览不会替换输入")
        selection = selection_from_dict(value["selection"])
        if mode == "trial" and (len(entries) != 1 or selection.mode != "current"):
            raise ValueError("精确试算需要单个来源的当前帧")
        if len(entries) > 1 and selection.mode == "current":
            raise ValueError("批量处理请使用全部数据或指定范围")
        if selection.mode == "range" and len({entry["meta"]["kind"] == "video" for entry in entries}) > 1:
            raise ValueError("帧范围与时间范围不能混用")
        if selection.mode == "range" and entries[0]["meta"]["kind"] != "video":
            if int(selection.start) != selection.start or int(selection.end) != selection.end:
                raise ValueError("帧范围必须使用整数，不能将小数默认为帧号")
        if selection.roi:
            for entry in entries:
                selection.roi.validate(tuple(entry["meta"]["shape"]))
        self.validate_steps(value["steps"])
        identifier = uuid.uuid4().hex[:12]
        payload = {"id": identifier, "mode": mode, "sources": [entry["source"] for entry in entries], "revisions": [entry["meta"]["revision"] for entry in entries], "steps": value["steps"], "selection": value["selection"], "output_dir": str(Path(value.get("output_dir") or self.workspace / "outputs").resolve()), "seed": 0, "config_revision": value.get("config_revision"), "display": value.get("display", {})}
        payload = json.loads(json.dumps(payload, allow_nan=False))
        with self.lock:
            if len(self.pending) >= 32:
                raise ValueError("任务队列已满")
            while len(self.jobs) >= 64:
                oldest = next((key for key, job in self.jobs.items() if job["status"] in {"succeeded", "failed", "cancelled", "completed_with_errors"}), None)
                if oldest is None:
                    raise ValueError("任务记录已满")
                self.jobs.pop(oldest)
            self.jobs[identifier] = {"id": identifier, "mode": mode, "name": entries[0]["meta"]["name"], "source_ids": identifiers, "source_count": len(entries), "status": "queued", "completed": 0, "total": None, "logs": [], "payload": payload, "base_shape": entries[0]["meta"]["shape"][:2], "created": time.time()}
            self.pending.append(identifier)
        return {"id": identifier}

    def job_list(self):
        with self.lock:
            return [{key: value for key, value in job.items() if key not in {"payload", "result", "run"}} for job in reversed(self.jobs.values())]

    def job(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise ValueError("任务记录不存在")
            return dict(self.jobs[identifier])

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
                    process = self.context.Process(target=job_entry, args=(self.jobs[identifier]["payload"], results, cancel), name="EIP-web-compute")
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
            elif kind == "trial":
                token = self.frames.add(event["pixels"], event["frame"], event["origin"], job["base_shape"], event["kind"])
                job["result"] = self.frames.render(token, job["payload"].get("display"))
                job.update(status="succeeded", completed=1, total=1, result_available=True)
                active["terminal"] = True
            elif kind == "completed":
                job.update(status=event["run"]["status"], run=event["run"], directory=event["directory"], result_available=False)
                active["terminal"] = True
                for item in event["run"]["results"]:
                    if item["status"] != "succeeded":
                        continue
                    path = Path(event["directory"]) / item["output_directory"]
                    # Result readers retain CSV timestamp mappings.
                    try:
                        self.open(asdict(SourceSpec(str(path), "sequence")), role="result", input_id=SourceSpec(**item["source"]).key, origin=item["origin"], base_shape=item["data_spec"]["shape"][:2], result_kind=item["result_kind"], run_id=job["id"])
                    except Exception as error:
                        job["logs"].append(f"输出已保存，但自动预览失败：{error}")

    def browse(self, path=None, mode="data"):
        path = Path(path or self.workspace).resolve()
        if not path.is_dir():
            raise ValueError("目录不存在或无访问权限")
        extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS if mode == "data" else {".json"}
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

    def save_project(self, path, view):
        self.validate_steps(view.get("steps", []))
        value = {**view, "schema_version": 1, "sources": [entry["source"] for entry in self.sources.values() if entry["role"] == "input"], "results": [{"source": entry["source"], "result": True, "input_key": entry.get("input_id"), "origin": entry.get("origin", [0, 0]), "base_shape": entry.get("base_shape"), "result_kind": entry.get("result_kind", "image")} for entry in self.sources.values() if entry["role"] == "result"]}
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
        self.validate_steps(value["steps"])
        if any(not Path(source["path"]).exists() for source in value["sources"]):
            raise ValueError("项目引用的部分来源缺失，请恢复来源路径")
        previous = self.sources.copy()
        self.sources.clear()
        try:
            for source in value["sources"]:
                self.open(source)
            for result in value.get("results", []):
                if Path(result["source"]["path"]).exists():
                    original = self.sources.get(result.get("input_key"))
                    base_shape = result.get("base_shape") or (original["meta"]["shape"][:2] if original else None)
                    self.open(result["source"], role="result", input_id=result.get("input_key"), origin=result.get("origin", [0, 0]), **({"base_shape": base_shape} if base_shape else {}), result_kind=result.get("result_kind", "image"))
        except Exception:
            self.sources = previous
            raise
        self.project_path = str(Path(path).resolve())
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
