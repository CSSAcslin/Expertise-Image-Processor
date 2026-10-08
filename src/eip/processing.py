from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import platform
import traceback

import numpy as np
import tifffile

from eip import __version__
from eip.algorithms import Cancelled, Pipeline, RunContext, sha256_file
from eip.domain import StepSpec, selection_from_dict, source_from_dict
from eip.persistence import write_json
from eip.readers import open_source


def statistics(pixels: np.ndarray) -> dict:
    finite = pixels[np.isfinite(pixels)]
    return {"min": float(finite.min()) if finite.size else None, "max": float(finite.max()) if finite.size else None, "mean": float(finite.mean()) if finite.size else None, "finite_count": int(finite.size), "invalid_count": int(pixels.size - finite.size)}


def process_job(payload: dict, emit, cancelled):
    selection = selection_from_dict(payload["selection"])
    steps = [StepSpec(**value) for value in payload["steps"]]
    context = RunContext(cancelled, lambda text: emit({"type": "log", "message": text}), payload.get("seed", 0))
    pipeline = Pipeline(steps, context)
    if payload["mode"] == "trial":
        reader = None
        try:
            reader = open_source(source_from_dict(payload["sources"][0]))
            if reader.meta.revision != payload["revisions"][0]:
                raise ValueError("输入文件已变化，请重新打开数据")
            context.check_cancelled()
            frame = next(reader.iter_frames(selection))
            output = pipeline.execute(frame.pixels)
            if selection.roi:
                output = selection.roi.crop(output)
            emit({"type": "trial", "pixels": output, "frame": asdict(frame.ref), "origin": [selection.roi.x, selection.roi.y] if selection.roi else [0, 0], "kind": pipeline.output_kind, "statistics": statistics(output), "revision": payload.get("config_revision")})
        finally:
            if reader:
                reader.close()
            pipeline.close()
        return

    root = Path(payload["output_dir"]).resolve()
    run_id = payload["id"]
    staging = root / f"{run_id}.partial"
    destination = root / run_id
    if staging.exists() or destination.exists():
        pipeline.close()
        raise ValueError("运行目录已存在；请使用新的任务标识")
    try:
        staging.mkdir(parents=True)
    except Exception:
        pipeline.close()
        raise
    run = {"schema_version": 1, "run_id": run_id, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "app_version": __version__, "python": platform.python_version(), "numpy": np.__version__, "tifffile": tifffile.__version__, "execution_backend": "local_python_process", "builtin_device": "CPU", "external_plugin_device": "not_reported" if any(step.manifest for step in steps if step.enabled) else None, "seed": context.seed, "fingerprint_kind": "path_size_mtime_ns", "content_sha256": None, "builtin_code_sha256": sha256_file(Path(__file__).with_name("algorithms.py")), "spec": payload, "results": [], "warnings": ["来源记录为文件属性指纹，不是完整内容哈希", "ROI 采用全帧处理后裁剪输出；非有限值统计只计入有限像素", "外部插件的额外依赖、设备及随机行为由插件声明；当前运行记录不推断这些信息"]}
    write_json(staging / "run.json", run)
    try:
        for position, source_value in enumerate(payload["sources"]):
            context.check_cancelled()
            reader = None
            item = {"source": source_value, "status": "running", "completed_frames": 0, "output_directory": f"source_{position + 1:03d}", "artifacts": [], "origin": [selection.roi.x, selection.roi.y] if selection.roi else [0, 0]}
            run["results"].append(item)
            out_dir = staging / item["output_directory"]
            out_dir.mkdir()
            try:
                reader = open_source(source_from_dict(source_value))
                if reader.meta.revision != payload["revisions"][position]:
                    raise ValueError("来源发生变化，请重新打开后运行")
                item["data_spec"] = asdict(reader.meta)
                if selection.roi:
                    selection.roi.validate(reader.meta.shape)
                total = 1 if selection.mode == "current" else (int(selection.end - selection.start) if selection.mode == "range" and reader.meta.kind != "video" else reader.meta.frame_count if selection.mode == "all" else None)
                rows_path = out_dir / "measurements.csv"
                with rows_path.open("w", encoding="utf-8-sig", newline="") as file:
                    fields = ["output_index", "source_frame", "time_seconds", "pts", "dtype", "min", "max", "mean", "finite_count", "invalid_count"]
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader()
                    for ordinal, frame in enumerate(reader.iter_frames(selection)):
                        context.check_cancelled()
                        output = pipeline.execute(frame.pixels)
                        if selection.roi:
                            output = selection.roi.crop(output)
                        context.check_cancelled()
                        filename = f"frame_{ordinal:08d}.tif"
                        tifffile.imwrite(out_dir / filename, output.astype(np.uint8) if output.dtype.kind == "b" else output, photometric="rgb" if output.ndim == 3 else "minisblack", metadata={"axes": "YX" if output.ndim == 2 else "YXS", "run_id": run_id, "source_frame": frame.ref.index, "time_seconds": frame.ref.time, "origin_xy": item["origin"], "result_kind": pipeline.output_kind})
                        writer.writerow({"output_index": ordinal, "source_frame": frame.ref.index, "time_seconds": frame.ref.time, "pts": frame.ref.pts, "dtype": str(output.dtype), **statistics(output)})
                        item["completed_frames"] = ordinal + 1
                        if not item["artifacts"]:
                            item["artifacts"].append(filename)
                        emit({"type": "progress", "source": position, "completed": ordinal + 1, "total": total, "name": reader.meta.name})
                if item["completed_frames"] == 0:
                    raise ValueError("选定范围没有数据")
                # Recheck file identity before declaring the source completed.
                verify = open_source(source_from_dict(source_value))
                try:
                    if verify.meta.revision != reader.meta.revision:
                        raise ValueError("运行期间来源发生变化，本次结果未完成")
                finally:
                    verify.close()
                item["status"] = "succeeded"
                item["result_kind"] = pipeline.output_kind
            except Cancelled:
                item["status"] = "cancelled"
                raise
            except Exception as error:
                item["status"] = "failed"
                item["error"] = str(error)
                emit({"type": "log", "message": f"{source_value['path']}：{error}"})
            finally:
                if reader:
                    reader.close()
                write_json(staging / "run.json", run)
        context.check_cancelled()
        successes = sum(item["status"] == "succeeded" for item in run["results"])
        run["status"] = "succeeded" if successes == len(run["results"]) else "completed_with_errors" if successes else "failed"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(staging / "run.json", run)
        if successes:
            staging.rename(destination)
        emit({"type": "completed", "run": run, "directory": str(destination if successes else staging)})
    except Cancelled:
        run["status"] = "cancelled"
        write_json(staging / "run.json", run)
        emit({"type": "cancelled", "directory": str(staging)})
    except Exception as error:
        run["status"] = "failed"
        run["error"] = str(error)
        write_json(staging / "run.json", run)
        raise
    finally:
        pipeline.close()


def job_entry(payload: dict, output_queue, cancel_event):
    try:
        process_job(payload, output_queue.put, cancel_event.is_set)
    except Cancelled:
        output_queue.put({"type": "cancelled"})
    except Exception as error:
        output_queue.put({"type": "failed", "message": str(error), "details": traceback.format_exc()})
    finally:
        output_queue.put({"type": "worker_done"})
