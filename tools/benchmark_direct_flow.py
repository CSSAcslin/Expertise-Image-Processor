"""Compare this CINE workflow's dedicated code with complete workbench runs.

Each trial has a fresh interpreter. Variants run serially in rotating order;
no UI settings or original outputs are changed. Validation runs after timing.
The dedicated literal path retains full-frame stages; the specialized path
pushes a proven equivalent ROI upstream. Their differences are not all module
dispatch overhead. CUDA event samples are never extrapolated to kernel totals.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

VARIANTS = {
    "app_cpu": dict(kind="app", policy="cpu"),
    "literal_cpu": dict(kind="direct", policy="cpu", implementation="literal"),
    "specialized_cpu": dict(kind="direct", policy="cpu", implementation="specialized"),
    "app_gpu_mixed": dict(kind="app", policy="mixed"),
    "literal_gpu_mixed": dict(kind="direct", policy="mixed", implementation="literal"),
    "app_gpu": dict(kind="app", policy="gpu"),
    "literal_gpu": dict(kind="direct", policy="gpu", implementation="literal"),
    "specialized_gpu": dict(kind="direct", policy="gpu", implementation="specialized"),
    "app_compat_cpu": dict(kind="app", policy="cpu", optimize=False),
    "app_compat_gpu_mixed": dict(kind="app", policy="mixed", optimize=False),
    "app_compat_gpu": dict(kind="app", policy="gpu", optimize=False),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_hashes():
    paths = list((ROOT / "src" / "eip").rglob("*.py"))
    paths += [ROOT / "tools" / name for name in ("direct_cine_flow.py", "direct_cine_exports.py", "benchmark_direct_flow.py", "benchmark_execution_time.py")]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def payload_for(baseline, folder, policy):
    payload = deepcopy(baseline["spec"])
    payload.update(id=f"direct-comparison-{uuid.uuid4().hex[:10]}",
                   display_name="Direct workflow comparison", output_dir=str(folder))
    payload.setdefault("resources", {})["backend"] = "cpu" if policy == "cpu" else "gpu"
    for node in payload["steps"]:
        if policy == "mixed" and node.get("enabled", True):
            requested = node.get("execution", {}).get("backend", "inherit")
            if requested not in ({"cpu"} if node["algorithm"] == "alignment" else {"inherit", "gpu"}):
                raise ValueError("Mixed policy requires explicit CPU alignment and GPU/inherited other stages")
        if policy in {"cpu", "gpu"} and node.get("enabled", True):
            node.setdefault("execution", {})["backend"] = policy
    return payload


def app_worker(config_path):
    config = read(config_path)
    import eip.sequence_processing as processing
    # Worker-local diagnostic switch. It never changes saved workbench settings
    # or introduces a second user-editable algorithm configuration.
    if config.get("optimize") is False:
        from functools import partial
        processing.SequenceEngine = partial(processing.SequenceEngine, optimize=False)
    payload = read(config["payload_path"])
    event = None

    def emit(value):
        nonlocal event
        if value["type"] in {"sequence_completed", "failed", "cancelled"}:
            event = value

    profiler = None
    if config.get("profile"):
        import cProfile
        profiler = cProfile.Profile()
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        processing.process_sequence_job(payload, emit, lambda: False)
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(Path(config["summary_path"]).with_suffix(".pstats")))
    elapsed = time.perf_counter() - started
    passed = bool(event and event.get("run", {}).get("status") == "succeeded")
    summary = dict(status="succeeded" if passed else "failed", job_call_seconds=elapsed, event=event)
    write(config["summary_path"], summary)
    return 0 if passed else 1


def compare_csv(left_path, right_path, *, rtol, atol):
    import csv
    from itertools import zip_longest
    import math

    mismatches, count = [], 0
    with left_path.open(encoding="utf-8-sig", newline="") as left, right_path.open(encoding="utf-8-sig", newline="") as right:
        x, y = csv.DictReader(left), csv.DictReader(right)
        if x.fieldnames != y.fieldnames:
            return dict(passed=False, error="CSV fields differ")
        numeric = {"min", "max", "mean", "value_1", "lag_seconds", "lag_seconds_min", "lag_seconds_max"}
        for row_x, row_y in zip_longest(x, y):
            count += 1
            if row_x is None or row_y is None:
                mismatches.append(dict(row=count - 1, field="row count"))
                break
            for key in x.fieldnames:
                a, b = row_x[key], row_y[key]
                if a == b:
                    continue
                equal = False
                if key in numeric and a and b:
                    try:
                        a, b = float(a), float(b)
                        equal = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
                    except ValueError:
                        pass
                if not equal and len(mismatches) < 10:
                    mismatches.append(dict(row=count - 1, field=key))
    return dict(passed=not mismatches, checked_rows=count, first_mismatches=mismatches)


def validate(baseline, baseline_root, folder, summary, kind):
    from benchmark_execution_time import compare_run, compare_values, compare_references
    tolerance = SimpleNamespace(rtol=2e-13, atol=2e-13, reference_atol=1e-12, validation_mb=16)
    expected = baseline_root / baseline["results"][0]["output_directory"]
    if kind == "app":
        run, destination = summary["event"]["run"], Path(summary["event"]["directory"])
        result = compare_run(baseline, baseline_root, run, destination, tolerance)
        actual = destination / run["results"][0]["output_directory"]
        result["actual_devices"] = run["results"][0]["modules"]["devices"]
    else:
        actual = folder
        values = compare_values(expected / "values.npy", actual / "values.npy", rtol=tolerance.rtol, atol=tolerance.atol, validation_bytes=16*1024**2)
        refs = compare_references(expected / "references.jsonl", actual / "references.jsonl", atol=tolerance.reference_atol)
        dataset = read(actual / "dataset.json")
        contract_equal = dataset.get("contract") == baseline["results"][0]["contract"]
        source_equal = dataset.get("source") == baseline["results"][0]["source"]
        result = dict(passed=values["passed"] and refs["passed"] and contract_equal and source_equal,
                      values=values, references=refs, contract_equal=contract_equal, source_equal=source_equal)
    result["csv"] = compare_csv(expected / "measurements.csv", actual / "measurements.csv", rtol=tolerance.rtol, atol=tolerance.atol)
    result["plots_present"] = all((actual / name).is_file() for name in ("curve.png", "curve.svg"))
    result["passed"] &= result["csv"]["passed"] and result["plots_present"]
    return result


def measured_scopes(summary, kind):
    if kind == "direct":
        phases = summary["phases_seconds"]
        return dict(job_call_seconds=summary["job_call_seconds"],
                    read_compute_seconds=summary["read_compute_seconds"],
                    output_and_provenance_seconds=phases["output"], setup_seconds=phases["setup"])
    performance = summary["event"]["run"]["results"][0]["performance"]
    host = performance["host_seconds"]
    core = sum(host.get(key, 0) for key in ("input_read", "index_write", "cpu_compute", "gpu_submit", "upload", "download"))
    exports = sum(host.get(key, 0) for key in ("references", "output_statistics", "output_render", "output_write"))
    return dict(job_call_seconds=summary["job_call_seconds"], read_compute_seconds=core,
                output_and_provenance_seconds=exports, setup_seconds=host.get("setup", 0))


def run_trial(name, variant, baseline, baseline_path, baseline_root, folder, args, *, profile=False):
    folder.mkdir(parents=True)
    config = dict(variant, summary_path=str(folder / "summary.json"), profile=profile)
    if variant["kind"] == "app":
        payload = payload_for(baseline, folder, variant["policy"])
        config["payload_path"] = str(folder / "payload.json")
        write(config["payload_path"], payload)
        config_path = folder / "worker.json"
        write(config_path, config)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-config", str(config_path)]
    else:
        direct_spec_path = folder.parent / f"{name}.payload.json"
        write(direct_spec_path, payload_for(baseline, folder, variant["policy"]))
        command = [sys.executable, str(ROOT / "tools" / "direct_cine_flow.py"),
                   "--spec", str(direct_spec_path), "--backend", "cpu" if variant["policy"] == "cpu" else "gpu",
                   "--implementation", variant["implementation"], "--alignment-backend", "cpu" if variant["policy"] in {"cpu", "mixed"} else "inherit",
                   "--output-dir", str(folder), "--report", config["summary_path"],
                   "--batch-size", str(args.batch_size), "--exports", "full"]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=args.timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    elapsed = time.perf_counter() - started
    (folder / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (folder / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    summary_path = Path(config["summary_path"])
    summary = read(summary_path) if summary_path.is_file() else {}
    result = dict(variant=name, process_wall_seconds=elapsed, summary=summary,
                  status="failed", folder=str(folder), profile=profile)
    if completed.returncode or summary.get("status") not in {"succeeded", "passed"}:
        result["error"] = (summary.get("error") or completed.stderr or completed.stdout)[-5000:]
        return result
    result["validation"] = validate(baseline, baseline_root, folder, summary, variant["kind"])
    if variant["kind"] == "app":
        devices = result["validation"]["actual_devices"]
        correct = all(devices.get(node["id"], {}).get("actual") ==
                      ("cpu" if variant["policy"] == "cpu" or variant["policy"] == "mixed" and node["algorithm"] == "alignment" else "gpu")
                      for node in baseline["spec"]["steps"] if node.get("enabled", True))
    else:
        correct = (summary.get("actual_backend") == ("cpu" if variant["policy"] == "cpu" else "gpu")
                   and summary.get("alignment_backend") == ("cpu" if variant["policy"] in {"cpu", "mixed"} else "gpu"))
    result["validation"]["device_policy_satisfied"] = correct
    result["validation"]["passed"] &= correct
    result["status"] = "succeeded" if result["validation"]["passed"] else "failed"
    result["scopes"] = measured_scopes(summary, variant["kind"])
    if profile:
        import pstats
        stats = pstats.Stats(str(summary_path.with_suffix(".pstats")))
        rows = [dict(file=key[0], line=key[1], function=key[2], primitive_calls=row[0], calls=row[1],
                     own_seconds=row[2], cumulative_seconds=row[3]) for key, row in stats.stats.items()]
        result["profile_top_cumulative"] = sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)[:60]
        result["profile_top_own"] = sorted(rows, key=lambda row: row["own_seconds"], reverse=True)[:60]
    return result


def main():
    from benchmark_execution_time import baseline_directory
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".test-output" / "direct-comparison")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)[:8])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--profile-apps", action="store_true", help="Separate CPU and full-GPU cProfile diagnostics; excluded from medians")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5 or not 1 <= args.batch_size <= 256 or args.timeout <= 0:
        parser.error("Use 1..5 repeats, batch size 1..256 and positive timeout")
    baseline_path = args.baseline.resolve()
    baseline = read(baseline_path)
    if baseline.get("status") != "succeeded" or len(baseline.get("results", [])) != 1:
        parser.error("Use one successful saved workflow source")
    baseline_root = baseline_directory(baseline_path, baseline)
    parent = args.output_dir.resolve()
    if parent.is_relative_to(baseline_root):
        parser.error("Write outside original outputs")
    for source in baseline["spec"]["sources"]:
        source_path = Path(source["path"]).resolve()
        if source_path.is_dir() and parent.is_relative_to(source_path):
            parser.error("Write outside source directories")
    folder = parent / f"comparison-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    versions = {}
    for name in ("numpy", "av", "cupy-cuda13x", "tifffile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), baseline=str(baseline_path),
                  python=sys.executable, python_version=platform.python_version(), platform=platform.platform(),
                  versions=versions, variants={name: VARIANTS[name] for name in args.variants}, repeats=args.repeats,
                  batch_size=args.batch_size, trials=[], diagnostics=[],
                  semantics="Serial rotating fresh workers. Full exports validated outside timing. OS file cache is not forcibly cleared; first trials retained. Specialized ROI pushdown and batching are execution-plan changes, not just module dispatch removal. GPU completion is awaited. Profiles are separate and excluded from timings.")
    report["code_sha256_before"] = code_hashes()
    report["source_stat"] = [dict(bytes=Path(source["path"]).stat().st_size, mtime_ns=Path(source["path"]).stat().st_mtime_ns,
                                  revision=baseline["spec"]["revisions"][i]) for i, source in enumerate(baseline["spec"]["sources"])]
    write(folder / "baseline.snapshot.json", baseline)
    write(folder / "report.json", report)
    print(f"Report: {folder / 'report.json'}", flush=True)
    failed = False
    for repeat in range(1, args.repeats + 1):
        shift = ((repeat - 1) * 3) % len(args.variants)
        order = args.variants[shift:] + args.variants[:shift]
        for name in order:
            print(f"Round {repeat}/{args.repeats}: {name} starting", flush=True)
            try:
                result = run_trial(name, VARIANTS[name], baseline, baseline_path, baseline_root,
                                   folder / f"round-{repeat:02d}" / name, args)
            except Exception as error:
                result = dict(variant=name, status="failed", error=f"{type(error).__name__}: {error}")
            result["repeat"] = repeat
            report["trials"].append(result)
            failed |= result["status"] != "succeeded"
            write(folder / "report.json", report)
            print(f"Round {repeat}: {name} {result['status']}; process {result.get('process_wall_seconds', 0):.3f} s", flush=True)
            if failed:
                print(f"Validation failed; stop without presenting speed claims. {result.get('error', result.get('validation'))}", flush=True)
                break
        if failed:
            break
    if not failed and args.profile_apps:
        for name in ("app_cpu", "app_gpu"):
            print(f"Separate diagnostic profile: {name}", flush=True)
            try:
                result = run_trial(name, VARIANTS[name], baseline, baseline_path, baseline_root, folder / "profiles" / name, args, profile=True)
            except Exception as error:
                result = dict(variant=name, status="failed", error=f"{type(error).__name__}: {error}", profile=True)
            report["diagnostics"].append(result)
            write(folder / "report.json", report)
        report["diagnostics_status"] = "succeeded" if all(row["status"] == "succeeded" for row in report["diagnostics"]) else "failed"
    report["status"] = "failed" if failed else "succeeded"
    report["code_sha256_after"] = code_hashes()
    report["code_unchanged"] = report["code_sha256_before"] == report["code_sha256_after"]
    if not report["code_unchanged"]:
        report["status"] = "failed"
        failed = True
    report["medians"] = {}
    for name in args.variants:
        trials = [trial for trial in report["trials"] if trial["variant"] == name and trial["status"] == "succeeded"]
        if trials:
            report["medians"][name] = dict(repeats=len(trials), process_wall_seconds=median(t["process_wall_seconds"] for t in trials),
                                          **{key: median(t["scopes"][key] for t in trials) for key in trials[0]["scopes"]})
    write(folder / "report.json", report)
    print(json.dumps(dict(status=report["status"], report=str(folder / "report.json"), medians=report["medians"])), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--worker-config", type=Path)
    worker_args, _ = early.parse_known_args()
    if worker_args.worker_config:
        raise SystemExit(app_worker(worker_args.worker_config))
    raise SystemExit(main())
