"""Replay a saved formal run in fresh workers and validate its complete output.

Examples (use the interpreter used to launch EIP):
    python tools/benchmark_execution_time.py --baseline outputs/run_ID/run.json
    python tools/benchmark_execution_time.py --baseline saved-run.snapshot.json --repeats 3 --backend cpu
    python tools/benchmark_execution_time.py --baseline outputs/run_ID/run.json --all-gpu

The original source, run directory, and application settings are not modified.
Only a new unique directory below --output-dir is written. Original source
revisions, ranges, module order, parameters, precision, and explicit execution
overrides are retained. --backend changes the global default only; --all-gpu is
an explicit request to change every enabled module's device override.

Job-call time includes preparation, I/O, computation and normal exports. Worker
wall time also includes interpreter startup. Neither is pure GPU kernel time.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--source-root", type=Path, default=ROOT / "src")
_early_parser.add_argument("--worker", type=Path)
_early, _ = _early_parser.parse_known_args()
SOURCE_ROOT = _early.source_root.resolve()
sys.path.insert(0, str(SOURCE_ROOT))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def worker(payload_path):
    from eip import __version__
    from eip.sequence_processing import process_sequence_job

    payload_path = Path(payload_path)
    payload = read_json(payload_path)
    result = None

    def emit(event):
        nonlocal result
        if event["type"] in {"sequence_completed", "failed", "cancelled"}:
            result = event

    started = time.perf_counter()
    try:
        process_sequence_job(payload, emit, lambda: False)
    except Exception as error:
        write_json(payload_path.with_suffix(".summary.json"), dict(
            status="failed", error=f"{type(error).__name__}: {error}",
            job_call_seconds=time.perf_counter() - started, event=result))
        raise
    elapsed = time.perf_counter() - started
    passed = bool(result and result.get("run", {}).get("status") == "succeeded")
    write_json(payload_path.with_suffix(".summary.json"), dict(
        status="succeeded" if passed else "failed", app_version=__version__,
        source_root=str(SOURCE_ROOT), job_call_seconds=elapsed, event=result))
    return 0 if passed else 1


def baseline_directory(path, run):
    """An archived run.json may sit beside the original run rather than in it."""
    expected = Path(run["results"][0]["output_directory"]) / "values.npy"
    candidates = [Path(path).parent]
    if run.get("spec", {}).get("output_dir") and run.get("run_id"):
        candidates.append(Path(run["spec"]["output_dir"]) / f"run_{run['run_id']}")
    for candidate in candidates:
        if (candidate / expected).is_file():
            return candidate.resolve()
    raise ValueError("Cannot locate saved numerical outputs; pass the original run.json or keep its saved output directory accessible")


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def compare_values(expected_path, actual_path, *, rtol, atol, validation_bytes):
    import numpy as np

    expected = np.load(expected_path, mmap_mode="r", allow_pickle=False)
    actual = None
    try:
        actual = np.load(actual_path, mmap_mode="r", allow_pickle=False)
        result = dict(shape=list(expected.shape), dtype=str(expected.dtype),
                      shape_equal=actual.shape == expected.shape,
                      dtype_equal=actual.dtype == expected.dtype,
                      finite_masks_equal=True, nonfinite_values_equal=True,
                      exact_values_equal=True, max_absolute_error=0.0,
                      max_relative_error=0.0, checked_values=0, rtol=rtol, atol=atol)
        if not result["shape_equal"] or not result["dtype_equal"]:
            return dict(result, passed=False, actual_shape=list(actual.shape), actual_dtype=str(actual.dtype))
        if expected.dtype.kind not in "buif":
            raise ValueError(f"Unsupported saved numerical dtype: {expected.dtype}")
        integer = expected.dtype.kind in "bui"
        maximum_absolute = maximum_relative = 0.0
        passed = True
        chunk = max(1, int(validation_bytes) // max(1, expected.dtype.itemsize * 8))
        # nditer handles C/F storage order without flattening an entire array.
        with np.nditer([expected, actual], flags=["external_loop", "buffered", "zerosize_ok"],
                       op_flags=[["readonly"], ["readonly"]], order="C", buffersize=chunk) as iterator:
            for left, right in iterator:
                result["checked_values"] += int(left.size)
                if integer:
                    same = bool(np.array_equal(left, right))
                    result["exact_values_equal"] &= same
                    passed &= same
                    continue
                finite_left, finite_right = np.isfinite(left), np.isfinite(right)
                masks_equal = bool(np.array_equal(finite_left, finite_right))
                invalid_equal = masks_equal and bool(np.array_equal(left[~finite_left], right[~finite_right], equal_nan=True))
                result["finite_masks_equal"] &= masks_equal
                result["nonfinite_values_equal"] &= invalid_equal
                result["exact_values_equal"] &= bool(np.array_equal(left, right, equal_nan=True))
                passed &= masks_equal and invalid_equal
                joint = finite_left & finite_right
                if bool(np.any(joint)):
                    # Use float64 validation arithmetic even for float32 arrays.
                    x, y = left[joint].astype(np.float64), right[joint].astype(np.float64)
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        difference = np.abs(y - x)
                        maximum_absolute = max(maximum_absolute, float(np.max(difference)))
                        relative = difference / np.maximum(np.abs(x), np.finfo(np.float64).tiny)
                        maximum_relative = max(maximum_relative, float(np.max(relative)))
                    passed &= bool(np.allclose(y, x, rtol=rtol, atol=atol))
        result["max_absolute_error"] = finite_number(maximum_absolute) if not integer or passed else None
        result["max_relative_error"] = finite_number(maximum_relative) if not integer or passed else None
        result["passed"] = bool(passed)
        return result
    finally:
        for value in (expected, actual):
            if value is not None and getattr(value, "_mmap", None) is not None:
                value._mmap.close()


def reference_difference(expected, actual, path="", *, atol):
    """Only the accumulated mean lag may differ within absolute tolerance.

    Frame identifiers, coordinate/time maps, ranges, contributor counts and
    extrema must be exact. This cannot hide an index or scientific-scope change.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            return path or "record keys"
        for key in expected:
            difference = reference_difference(expected[key], actual[key], f"{path}.{key}" if path else key, atol=atol)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return path
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = reference_difference(left, right, f"{path}[{index}]", atol=atol)
            if difference is not None:
                return difference
        return None
    if path.endswith(".lag_seconds") and expected is not None and actual is not None:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return None if math.isclose(expected, actual, rel_tol=0, abs_tol=atol) else path
    # Prevent bool/int coercion from hiding malformed scientific records.
    return None if type(expected) is type(actual) and expected == actual else path


def compare_references(expected_path, actual_path, *, atol):
    checked, byte_equal = 0, True
    mismatches = []
    limit = 32 * 1024**2
    with Path(expected_path).open("rb") as expected, Path(actual_path).open("rb") as actual:
        while True:
            left, right = expected.readline(limit + 1), actual.readline(limit + 1)
            if len(left) > limit or len(right) > limit:
                raise ValueError("A source reference record exceeds the bounded 32 MiB validation limit")
            if not left and not right:
                break
            checked += 1
            byte_equal &= left == right
            if not left or not right:
                difference = "record count"
            else:
                difference = reference_difference(json.loads(left), json.loads(right), atol=atol)
            if difference is not None and len(mismatches) < 10:
                mismatches.append(dict(output_index=checked - 1, field=difference))
    return dict(passed=not mismatches, checked_records=checked,
                byte_equal=bool(byte_equal), lag_seconds_atol=atol,
                first_mismatches=mismatches)


def compare_output(expected_folder, actual_folder, expected, actual, args):
    if expected.get("contract") != actual.get("contract"):
        return dict(passed=False, error="Output contracts differ")
    values = compare_values(expected_folder / "values.npy", actual_folder / "values.npy",
                            rtol=args.rtol, atol=args.atol, validation_bytes=args.validation_mb * 1024**2)
    references = compare_references(expected_folder / "references.jsonl", actual_folder / "references.jsonl",
                                    atol=args.reference_atol)
    return dict(passed=values["passed"] and references["passed"], values=values, references=references)


def compare_run(baseline, baseline_root, result_run, result_root, args):
    expected_results, actual_results = baseline["results"], result_run["results"]
    if len(expected_results) != len(actual_results):
        return dict(passed=False, error="Source count differs")
    results = []
    for position, (left, right) in enumerate(zip(expected_results, actual_results)):
        if left.get("source") != right.get("source") or right.get("status") != "succeeded":
            results.append(dict(source_position=position, passed=False, error="Source identity or completion differs"))
            continue
        left_folder = baseline_root / left["output_directory"]
        right_folder = result_root / right["output_directory"]
        result = compare_output(left_folder, right_folder, left, right, args)
        result["source_position"] = position
        result["performance"] = right.get("performance")
        result["modules"] = right.get("modules")
        result["side_outputs"] = []
        sides_left, sides_right = left.get("side_outputs", []), right.get("side_outputs", [])
        if len(sides_left) != len(sides_right):
            result.update(passed=False, side_output_error="Side output count differs")
        for side_left, side_right in zip(sides_left, sides_right):
            identity = ("node_id", "port", "status", "name", "destination", "export_format")
            if any(side_left.get(k) != side_right.get(k) for k in identity):
                side = dict(passed=False, error="Side output identity or status differs")
            else:
                side = compare_output(left_folder / side_left["output_directory"],
                                      right_folder / side_right["output_directory"], side_left, side_right, args)
            result["side_outputs"].append(side)
            result["passed"] &= side["passed"]
        results.append(result)
    return dict(passed=all(result["passed"] for result in results), sources=results)


def make_payload(spec, directory, repeat, args):
    payload = deepcopy(spec)
    payload["id"] = f"replay-{repeat:02d}-{uuid.uuid4().hex[:10]}"
    payload["display_name"] = f"Execution replay {repeat}"
    payload["output_dir"] = str(directory)
    if args.backend != "original":
        payload.setdefault("resources", {})["backend"] = "cpu" if args.backend == "cpu" else "gpu"
    if args.all_gpu:
        payload.setdefault("resources", {})["backend"] = "gpu"
        for node in payload["steps"]:
            if node.get("enabled", True):
                node.setdefault("execution", {})["backend"] = "gpu"
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="Saved successful contract-v3 formal run.json or snapshot")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".test-output", help="Parent of a new unique replay directory")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--backend", choices=["original", "cpu", "gpu-inherit"], default="original")
    parser.add_argument("--all-gpu", action="store_true", help="Explicitly replace every enabled module device override with GPU")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-worker seconds, including imports and exports")
    parser.add_argument("--rtol", type=float, default=2e-13)
    parser.add_argument("--atol", type=float, default=2e-13)
    parser.add_argument("--reference-atol", type=float, default=1e-12, help="Absolute tolerance only for accumulated lag_seconds")
    parser.add_argument("--validation-mb", type=int, default=16, help="Budget for temporary numerical validation arrays")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or args.timeout < 1 or not 1 <= args.validation_mb <= 256:
        parser.error("Use 1..10 repeats, positive timeout, and 1..256 validation MiB")
    if any(not math.isfinite(v) or v < 0 for v in (args.rtol, args.atol, args.reference_atol)):
        parser.error("Tolerances must be finite and nonnegative")
    if args.all_gpu and args.backend == "cpu":
        parser.error("--all-gpu cannot be combined with --backend cpu")
    baseline_path = args.baseline.resolve()
    baseline = read_json(baseline_path)
    if (baseline.get("contract_version") != 3 or baseline.get("status") != "succeeded"
            or baseline.get("spec", {}).get("mode") != "run" or not baseline.get("results")
            or any(r.get("status") != "succeeded" for r in baseline["results"])):
        parser.error("Baseline must be a fully successful contract-v3 formal run")
    baseline_root = baseline_directory(baseline_path, baseline)
    output_parent = args.output_dir.resolve()
    if output_parent.is_relative_to(baseline_root):
        parser.error("Replay output must be outside the original run directory")
    for source in baseline["spec"]["sources"]:
        source_path = Path(source["path"]).resolve()
        if source_path.is_dir() and output_parent.is_relative_to(source_path):
            parser.error("Replay output must be outside every source directory")
    folder = output_parent / f"execution-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    versions = {}
    for name in ("numpy", "tifffile", "av", "cupy-cuda12x", "cupy-cuda13x"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = dict(baseline=str(baseline_path), baseline_directory=str(baseline_root),
                  baseline_run_id=baseline["run_id"], source_root=str(SOURCE_ROOT),
                  created_at=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                  python_version=platform.python_version(), platform=platform.platform(), versions=versions,
                  backend_policy=args.backend, all_gpu=args.all_gpu,
                  scope="Complete formal replay in a fresh worker per repeat; validation is outside measured worker wall time. No pure CUDA kernel time is inferred.",
                  baseline_performance=[r.get("performance") for r in baseline["results"]], repeats=[])
    write_json(folder / "baseline.snapshot.json", baseline)
    for repeat in range(1, args.repeats + 1):
        payload = make_payload(baseline["spec"], folder, repeat, args)
        payload_path = folder / f"repeat-{repeat:02d}.json"
        write_json(payload_path, payload)
        command = [sys.executable, str(Path(__file__).resolve()), "--source-root", str(SOURCE_ROOT), "--worker", str(payload_path)]
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=args.timeout,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            wall = time.perf_counter() - started
            (folder / f"repeat-{repeat:02d}.stdout.log").write_text(completed.stdout, encoding="utf-8")
            (folder / f"repeat-{repeat:02d}.stderr.log").write_text(completed.stderr, encoding="utf-8")
            summary_path = payload_path.with_suffix(".summary.json")
            summary = read_json(summary_path) if summary_path.is_file() else {}
            result = dict(repeat=repeat, process_wall_seconds=wall,
                          job_call_seconds=summary.get("job_call_seconds"), status="failed")
            if completed.returncode or summary.get("status") != "succeeded":
                result["error"] = summary.get("error") or (completed.stderr or completed.stdout)[-4000:]
            else:
                event = summary["event"]
                candidate = event["run"]
                result["directory"] = event["directory"]
                result["submitted_spec_preserved"] = candidate.get("spec") == payload
                result["validation"] = compare_run(baseline, baseline_root, candidate, Path(event["directory"]), args)
                device_records = [d for item in candidate["results"]
                                  for d in item.get("modules", {}).get("devices", {}).values()]
                all_gpu_used = bool(device_records) and all(d.get("actual") == "gpu" for d in device_records)
                result["explicit_all_gpu_satisfied"] = all_gpu_used if args.all_gpu else None
                result["status"] = "succeeded" if (result["submitted_spec_preserved"] and result["validation"]["passed"]
                                                     and (not args.all_gpu or all_gpu_used)) else "failed"
        except subprocess.TimeoutExpired:
            result = dict(repeat=repeat, status="failed", process_wall_seconds=time.perf_counter() - started,
                          error=f"Worker exceeded {args.timeout} seconds")
        except Exception as error:
            result = dict(repeat=repeat, status="failed", process_wall_seconds=time.perf_counter() - started,
                          error=f"{type(error).__name__}: {error}")
        report["repeats"].append(result)
        write_json(folder / "report.json", report)
        print(f"Repeat {repeat}: {result['status']}; process {result['process_wall_seconds']:.3f} s", flush=True)
    passed = [r for r in report["repeats"] if r["status"] == "succeeded"]
    report["status"] = "succeeded" if len(passed) == args.repeats else "failed"
    if passed:
        report["medians"] = dict(repeats=len(passed), job_call_seconds=median(r["job_call_seconds"] for r in passed),
                                  process_wall_seconds=median(r["process_wall_seconds"] for r in passed))
    write_json(folder / "report.json", report)
    print(f"Report: {folder / 'report.json'}")
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if _early.worker:
        raise SystemExit(worker(_early.worker))
    raise SystemExit(main())
