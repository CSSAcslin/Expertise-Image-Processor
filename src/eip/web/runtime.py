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
        counts, edges = np.histogram(samples, bins=64) if samples.size else (np.zeros(64, dtype=int), np.linspace(0, 1, 65))
        return {"token": token, "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), "shape": list(pixels.shape), "base_shape": list(record["base_shape"]), "origin": list(record["origin"]), "ref": record["ref"], "dtype": str(pixels.dtype), "kind": record["kind"], "display": asdict(display), "histogram": counts.tolist(), "histogram_range": [float(edges[0]), float(edges[-1])], "histogram_samples": int(samples.size), "statistics": statistics(pixels), "display_shape": [image.height, image.width], "downsampled": image.width != pixels.shape[1] or image.height != pixels.shape[0]}

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
        self.project_path = value["state"].get("project_path")

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
        manifest = Path(entry["source"]["path"]).parent / "run.json"
        run = json.loads(manifest.read_text(encoding="utf-8"))
        item = next((item for item in run["results"] if item["output_directory"] == Path(entry["source"]["path"]).name), None)
        if not item or item["status"] != "succeeded" or run["status"] not in {"succeeded", "completed_with_errors"}:
            raise ValueError("未完成的输出不能作为下一阶段输入")
        parent = self.sources.get(entry.get("input_id"))
        # The run snapshot remains authoritative even when a source reference was removed.
        previous = next((value.get("lineage", []) for value in run["spec"].get("provenance", []) if value["source_id"] == entry.get("input_id")), parent.get("lineage", []) if parent else [])
        if len(previous) >= 32:
            raise ValueError("当前来源链最多支持 32 个阶段")
        lineage = [*previous, {"run_id": run["run_id"], "manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "parent_source_id": entry.get("input_id"), "parent_source": item["source"], "crop_origin": item["origin"], "parent_shape": item["data_spec"]["shape"], "output_directory": entry["source"]["path"], "timestamp_source": "run_record"}]
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
            with (Path(source["source"]["path"]) / "measurements.csv").open(encoding="utf-8-sig", newline="") as file:
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
        if selection.mode == "range" and any(entry["meta"]["kind"] == "video" and entry["meta"]["timestamp_source"] == "unavailable" for entry in entries):
            raise ValueError("视频缺少逐帧时间记录，请选择当前帧或全部数据处理")
        if selection.mode == "range" and entries[0]["meta"]["kind"] != "video":
            if int(selection.start) != selection.start or int(selection.end) != selection.end:
                raise ValueError("帧范围必须使用整数，不能将小数默认为帧号")
        if selection.roi:
            for entry in entries:
                selection.roi.validate(tuple(entry["meta"]["shape"]))
        self.validate_steps(value["steps"])
        identifier = uuid.uuid4().hex[:12]
        payload = {"id": identifier, "mode": mode, "sources": [entry["source"] for entry in entries], "revisions": [entry["meta"]["revision"] for entry in entries], "steps": value["steps"], "selection": value["selection"], "output_dir": str(Path(value.get("output_dir") or self.workspace / "outputs").resolve()), "seed": 0, "config_revision": value.get("config_revision"), "display": value.get("result_display", value.get("display", {})), "provenance": [{"source_id": entry["id"], "lineage": entry.get("lineage", [])} for entry in entries]}
        payload = json.loads(json.dumps(payload, allow_nan=False))
        payload["input_kinds"] = [entry.get("result_kind", "image") for entry in entries]
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
                for position, item in enumerate(event["run"]["results"]):
                    if item["status"] != "succeeded":
                        continue
                    path = Path(event["directory"]) / item["output_directory"]
                    # Result readers retain CSV timestamp mappings.
                    try:
                        self.open(asdict(SourceSpec(str(path), "sequence")), role="result", input_id=job["source_ids"][position], origin=item["origin"], base_shape=item["data_spec"]["shape"][:2], result_kind=item["result_kind"], run_id=job["id"])
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
        self.validate_steps(view.get("steps", []))
        value = {**view, "schema_version": 1, "sources": [entry["source"] for entry in self.sources.values() if entry["role"] == "input"], "results": [{"source": entry["source"], "id": entry["id"], "run_id": entry.get("run_id"), "display_name": entry.get("display_name"), "result": True, "input_key": entry.get("input_id"), "origin": entry.get("origin", [0, 0]), "base_shape": entry.get("base_shape"), "result_kind": entry.get("result_kind", "image")} for entry in self.sources.values() if entry["role"] == "result"]}
        value["source_entries"] = [{key: entry[key] for key in ["source", "id", "lineage", "parent_result_id", "display_name", "result_kind"] if key in entry} for entry in self.sources.values() if entry["role"] == "input"]
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
