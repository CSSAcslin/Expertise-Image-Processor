"""Serial dedicated Q-scan experiments against saved, independently validated outputs.

The saved full 3001-Q application run is retained as the full-workload baseline.
Fresh application workers also evaluate 31 selected Qs without changing W or
the fixed centre. This short diagnostic is never extrapolated to a full time.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from statistics import median
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src" / "eip").rglob("*.py"))}


def diagnostic_payload(baseline, folder, policy, indices):
    """Keep W, signed lags and the actual fixed centre from the saved run."""
    payload = deepcopy(baseline["spec"])
    active = [n for n in payload["steps"] if n.get("enabled", True)]
    node = active[-1]
    if node["algorithm"] != "correlation_curve":
        raise ValueError("Requires the saved point/Q-curve workflow")
    mapping = baseline["results"][0]["mappings"][node["id"]]
    lags = mapping["lags"]
    center = mapping["centers"]["start"]
    node["parameters"].update(center=center + 1, lag_list=[lags[j] for j in indices])
    payload.update(id=f"q-diagnostic-{uuid.uuid4().hex[:10]}",
                   display_name="Independent Q diagnostic", output_dir=str(folder))
    payload["resources"]["backend"] = "cpu" if policy == "cpu" else "gpu"
    if policy == "cpu":
        for n in active:
            n.setdefault("execution", {})["backend"] = "cpu"
    elif policy != "mixed":
        raise ValueError("Use cpu or the saved GPU policy with explicit CPU alignment")
    return payload


def full_application_payload(baseline,folder,policy):
    payload = deepcopy(baseline["spec"])
    payload.update(id=f"q-full-{uuid.uuid4().hex[:10]}",display_name="Complete Q performance validation",output_dir=str(folder))
    payload["resources"]["backend"] = "cpu" if policy=="cpu" else "gpu"
    if policy=="cpu":
        for node in payload["steps"]:
            if node.get("enabled",True):
                node.setdefault("execution",{})["backend"] = "cpu"
    elif policy!="mixed":
        raise ValueError("Use CPU or the saved mixed device policy")
    return payload


def subset_expected(baseline, baseline_root, folder, indices):
    """Create a small validation fixture without touching the saved source run."""
    import csv
    import numpy as np
    original = baseline_root / baseline["results"][0]["output_directory"]
    folder.mkdir()
    np.save(folder / "values.npy", np.load(original / "values.npy")[indices], allow_pickle=False)
    refs = [json.loads(line) for line in (original / "references.jsonl").read_text(encoding="utf-8").splitlines()]
    with (folder / "references.jsonl").open("w", encoding="utf-8") as f:
        for ordinal, index in enumerate(indices):
            ref = refs[index]
            ref["sample_index"] = ordinal
            f.write(json.dumps(ref, ensure_ascii=False, allow_nan=False) + "\n")
    with (original / "measurements.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    with (folder / "measurements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ordinal, index in enumerate(indices):
            row = rows[index]
            row["output_index"] = ordinal
            writer.writerow(row)
    return folder


def validate_folder(expected, actual):
    from benchmark_execution_time import compare_values, compare_references
    from benchmark_direct_flow import compare_csv
    values = compare_values(expected / "values.npy", actual / "values.npy", rtol=2e-13, atol=2e-13, validation_bytes=16*1024**2)
    refs = compare_references(expected / "references.jsonl", actual / "references.jsonl", atol=1e-12)
    csv = compare_csv(expected / "measurements.csv", actual / "measurements.csv", rtol=2e-13, atol=2e-13)
    plots = all((actual / n).is_file() for n in ("curve.png", "curve.svg"))
    return dict(passed=values["passed"] and refs["passed"] and csv["passed"] and plots,
                values=values, references=refs, csv=csv, plots_present=plots)


def validate_contract_devices(baseline, config, summary, actual):
    dataset = read(actual / "dataset.json")
    contract = deepcopy(baseline["results"][0]["contract"])
    if config["kind"] == "app":
        nodes = [n for n in read(config["payload_path"])["steps"] if n.get("enabled", True)]
        if not config.get("full_scan"):
            length = len(nodes[-1]["parameters"]["lag_list"])
            contract.update(length=length, full_shape=[length], label=f"Q:{length}")
        devices = summary["event"]["run"]["results"][0]["modules"]["devices"]
        correct = all(devices.get(n["id"], {}).get("actual") ==
                      ("cpu" if config["policy"] == "cpu" or n.get("execution", {}).get("backend") == "cpu" else "gpu") for n in nodes)
    else:
        correct = summary.get("actual_backend", summary.get("backend")) == config["policy"]
    return dict(contract_equal=dataset.get("contract") == contract,
                source_equal=dataset.get("source") == baseline["spec"]["sources"][0], device_policy_satisfied=correct)


def worker(config_path):
    config = read(config_path)
    if config["kind"] == "app":
        from benchmark_direct_flow import app_worker
        return app_worker(config_path)
    from direct_lag_scan import run_direct
    summary = run_direct(read(config["payload_path"]), output_dir=config["output_directory"],
                         backend=config["policy"], exports="full")
    write(config["summary_path"], summary)
    return 0 if summary.get("status") == "succeeded" else 1


def trial(config, folder, expected, timeout, baseline):
    config_path = folder / "worker.json"
    write(config_path, config)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    wall = time.perf_counter() - started
    (folder / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (folder / "stderr.log").write_text(result.stderr, encoding="utf-8")
    summary = read(config["summary_path"]) if Path(config["summary_path"]).is_file() else {}
    row = dict(status="failed", process_wall_seconds=wall, folder=str(folder), summary=summary)
    if result.returncode or summary.get("status") != "succeeded":
        row["error"] = (summary.get("error") or result.stderr or result.stdout)[-4000:]
        return row
    if config["kind"] == "app":
        run = summary["event"]["run"]
        actual = Path(summary["event"]["directory"]) / run["results"][0]["output_directory"]
        row["performance"] = run["results"][0]["performance"]
        row["actual_devices"] = run["results"][0]["modules"]["devices"]
        row["scopes"] = dict(job_call_seconds=summary["job_call_seconds"],
                             **row["performance"]["host_seconds"])
    else:
        actual = Path(config["output_directory"])
        row["scopes"] = dict(job_call_seconds=summary.get("job_call_seconds", summary.get("elapsed_seconds")),
                             **summary.get("phases_seconds", {}))
    row["validation"] = validate_folder(expected, actual)
    row["validation"].update(validate_contract_devices(baseline, config, summary, actual))
    row["validation"]["passed"] &= all(row["validation"][k] for k in ("contract_equal", "source_equal", "device_policy_satisfied"))
    row["status"] = "succeeded" if row["validation"]["passed"] else "failed"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/run_778f942a3723/run.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".test-output/qlag")
    parser.add_argument("--phase", choices=("app", "app-full", "direct"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("Use 1..5 repeats")
    baseline = read(args.baseline)
    baseline_root = args.baseline.resolve().parent
    folder = args.output_dir.resolve() / f"{args.phase}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    if folder.is_relative_to(baseline_root):
        parser.error("Write outside original results")
    folder.mkdir(parents=True)
    node = [n for n in baseline["spec"]["steps"] if n.get("enabled", True)][-1]
    mapping = baseline["results"][0]["mappings"][node["id"]]
    indices = list(range(0, len(mapping["lags"]), 100))
    if indices[-1] != len(mapping["lags"]) - 1:
        indices.append(len(mapping["lags"]) - 1)
    expected = subset_expected(baseline, baseline_root, folder / "expected", indices) if args.phase == "app" else baseline_root / baseline["results"][0]["output_directory"]
    report = dict(status="running", created_at=datetime.now(timezone.utc).isoformat(), baseline=str(args.baseline),
                  phase=args.phase, repeats=args.repeats, python_version=platform.python_version(),
                  platform=platform.platform(), versions={n: importlib.metadata.version(n) for n in ("numpy", "cupy-cuda13x", "av")},
                  diagnostic_original_indices=indices if args.phase == "app" else None,
                  source_hashes_before=source_hashes(), trials=[],
                  semantics="Fresh serial workers, alternating CPU/GPU order. Complete exports included. Validation outside timing. File cache not cleared. app uses 31 Q diagnostics; app-full preserves all saved Qs and original automatic centre. Never extrapolate diagnostics. Full saved GPU baseline is historical.")
    write(folder / "report.json", report)
    print(f"Report: {folder / 'report.json'}", flush=True)
    for repeat in range(1, args.repeats + 1):
        policies = ["cpu", "mixed"] if args.phase.startswith("app") else ["cpu", "gpu"]
        if repeat % 2 == 0:
            policies.reverse()
        for policy in policies:
            dest = folder / f"r{repeat}-{policy}"
            dest.mkdir()
            if args.phase=="app":
                payload = diagnostic_payload(baseline,dest,policy,indices)
            elif args.phase=="app-full":
                payload = full_application_payload(baseline,dest,policy)
            else:
                payload = deepcopy(baseline["spec"])
            payload_path = dest / "payload.json"
            write(payload_path, payload)
            config = dict(kind="app" if args.phase.startswith("app") else "direct", policy=policy,
                          full_scan=args.phase=="app-full",payload_path=str(payload_path),
                          summary_path=str(dest / "summary.json"), output_directory=str(dest / "direct"))
            print(f"Round {repeat}: {policy} starting", flush=True)
            row = trial(config, dest, expected, args.timeout, baseline)
            row.update(repeat=repeat, policy=policy)
            report["trials"].append(row)
            write(folder / "report.json", report)
            print(json.dumps(dict(repeat=repeat, policy=policy, status=row["status"], scopes=row.get("scopes"), error=row.get("error"), validation=row.get("validation"))), flush=True)
            if row["status"] != "succeeded":
                report["status"] = "failed"
                write(folder / "report.json", report)
                return 1
    report["source_hashes_after"] = source_hashes()
    report["source_unchanged"] = report["source_hashes_before"] == report["source_hashes_after"]
    report["status"] = "succeeded" if report["source_unchanged"] else "failed"
    report["medians"] = {}
    for policy in sorted({r["policy"] for r in report["trials"]}):
        rows = [r for r in report["trials"] if r["policy"] == policy]
        report["medians"][policy] = dict(repeats=len(rows), process_wall_seconds=median(r["process_wall_seconds"] for r in rows),
                                       **{k: median(r["scopes"][k] for r in rows) for k in rows[0]["scopes"]})
    write(folder / "report.json", report)
    print(json.dumps(dict(status=report["status"], report=str(folder / "report.json"), medians=report["medians"])), flush=True)
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--worker", type=Path)
    known, _ = early.parse_known_args()
    raise SystemExit(worker(known.worker) if known.worker else main())
