"""Measure general module chains with optimization disabled/enabled.

Synthetic float64 sources live in worker memory. Timings cover engine setup,
all logical outputs and references, and engine cleanup. They exclude synthetic
source generation, artifact saving and validation, and are not full application
or disk/export benchmarks. Every timed trial runs in a fresh serial process.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CASES = ("pointwise_roi", "pointwise_point", "gaussian_roi", "time_edits_roi",
         "line_segment", "full_frame_control")
RTOL = ATOL = 2e-13


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def code_hashes():
    paths = sorted((ROOT / "src" / "eip").rglob("*.py")) + [Path(__file__).resolve()]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def case_nodes(case, height, width, backend):
    """Deterministic IDs keep contribution references identical across workers."""
    from eip.module_registry import new_node

    if case not in CASES:
        raise ValueError(f"Unknown benchmark case: {case}")
    result = []

    def add(algorithm, **parameters):
        node = new_node(algorithm, shape=(height, width), frames=48)
        node["id"] = f"{case}-{len(result):02d}-{algorithm}"
        node["parameters"].update(parameters)
        node["execution"] = dict(backend=backend, precision="float64")
        result.append(node)

    x, y = width // 3, height // 3
    w, h = max(2, width // 6), max(2, height // 6)
    add("gain_offset", gain=1.125, offset=-.25)
    if case == "gaussian_roi":
        add("spatial_smooth", method="gaussian", radius=3, sigma=1.1)
    else:
        add("curve", curve="square", input_max=4095.)
    if case == "pointwise_point":
        add("point_extract", x=x, y=y, output="main")
    elif case == "line_segment":
        add("line_extract", mode="segment", x0=width*.2, y0=height*.3,
            x1=width*.85, y1=height*.8, samples=129, interpolation="bilinear", output="main")
    elif case == "time_edits_roi":
        add("time_sample", step=2, offset=1)
        add("time_smooth", window=5, mode="mean")
        add("time_bin", size=3, mode="mean")
        add("roi_crop", x=x, y=y, width=w, height=h)
    elif case != "full_frame_control":
        add("roi_crop", x=x, y=y, width=w, height=h)
    return result


def worker(config_path):
    import numpy as np
    from eip import __version__
    from eip.domain import DataSpec, FrameData, FrameRef, SourceSpec
    from eip.performance import RunProfiler
    from eip.sequence_engine import SequenceEngine

    config = read_json(config_path)
    folder = Path(config_path).parent
    shape = tuple(config["shape"])
    started_generation = time.perf_counter()
    values = np.random.default_rng(config["seed"]).uniform(10., 3000., shape)
    values[::11, shape[1]//3, shape[2]//3] = np.nan
    values[::17, shape[1]//2, shape[2]//2] = np.inf
    source_generation_seconds = time.perf_counter() - started_generation
    before = hashlib.sha256(values.tobytes()).hexdigest()

    class MemoryReader:
        def __init__(self):
            self.meta = DataSpec("execution-planner-synthetic", "sequence", shape[1:], "float64", "YX",
                                 shape[0], None, None, "integer_fixture_ticks", "RGB", before)
            self.source = SourceSpec("execution-planner-synthetic", "sequence")
            self.reads = 0

        def read_frame(self, index):
            self.reads += 1
            return FrameData(values[index], FrameRef(index, index/1000., pts=index,
                                                    source_number=100+index, trigger_time=(index-3)/1000.))

    reader = MemoryReader()
    nodes = case_nodes(config["case"], shape[1], shape[2], config["backend"])
    engine = None
    profiler = RunProfiler()
    started = time.perf_counter()
    try:
        with profiler.measure("setup"):
            engine = SequenceEngine(reader, nodes, workdir=folder/"scratch",
                                    resources=config["resources"], optimize=config["optimize"], profiler=profiler)
        output = np.empty((engine.output.length, *engine.output.shape), dtype=engine.output.dtype)
        references = []
        for index, value in engine.iter_outputs(range(engine.output.length)):
            output[index] = value
            with profiler.measure("references"):
                references.append(engine.ref(len(nodes), index))
            profiler.add_output(value.nbytes)
        # Host outputs have already completed all needed GPU work. Device
        # event completion and normal cleanup remain inside the engine timing.
        profiler.finish_gpu(raise_errors=True)
        devices = deepcopy(engine.device_records)
        if any(record.get("actual") != config["backend"] for record in devices.values()):
            raise RuntimeError(f"Backend mismatch: {devices}")
        contract = engine.output.record()
        plan = engine._actual_plan()
    finally:
        if engine is not None:
            engine.close()
    performance = profiler.finish()
    elapsed = time.perf_counter() - started
    if hashlib.sha256(values.tobytes()).hexdigest() != before:
        raise AssertionError("The execution modified the original synthetic source")
    # Trial artifacts and comparison are deliberately outside elapsed_seconds.
    np.save(folder/"values.npy", output, allow_pickle=False)
    write_json(folder/"references.json", references)
    summary = dict(status="succeeded", app_version=__version__, case=config["case"],
                   backend=config["backend"], optimize=config["optimize"],
                   elapsed_seconds=elapsed, source_generation_seconds=source_generation_seconds,
                   source_sha256=before, source_reads=reader.reads, nodes=nodes, contract=contract,
                   devices=devices, performance=performance,
                   execution=plan.get("execution"), device_execution=plan.get("device_execution"),
                   input_shape=list(shape), output_shape=list(output.shape), dtype=str(output.dtype))
    write_json(folder/"summary.json", summary)
    return 0


def compare_trial(reference, actual):
    import numpy as np

    reference, actual = Path(reference), Path(actual)
    left, right = read_json(reference/"summary.json"), read_json(actual/"summary.json")
    for key in ("source_sha256", "contract", "output_shape", "dtype"):
        if left[key] != right[key]:
            raise AssertionError(f"{key} mismatch")
    expected = np.load(reference/"values.npy", mmap_mode="r", allow_pickle=False)
    observed = np.load(actual/"values.npy", mmap_mode="r", allow_pickle=False)
    if expected.shape != observed.shape or expected.dtype != observed.dtype:
        raise AssertionError("Numerical shape or dtype mismatch")
    max_error = 0.
    finite_values = 0
    # Bound validation scratch independently of output image size.
    a, b = expected.reshape(-1), observed.reshape(-1)
    for start in range(0, a.size, 131072):
        av, bv = a[start:start+131072], b[start:start+131072]
        np.testing.assert_array_equal(np.isfinite(av), np.isfinite(bv))
        np.testing.assert_allclose(bv, av, rtol=RTOL, atol=ATOL, equal_nan=True)
        finite = np.isfinite(av)
        finite_values += int(finite.sum())
        if finite.any():
            max_error = max(max_error, float(np.max(np.abs(av[finite]-bv[finite]))))
    del a, b, expected, observed
    if read_json(reference/"references.json") != read_json(actual/"references.json"):
        raise AssertionError("Output references differ")
    return dict(status="passed", checked_values=int(np.prod(left["output_shape"])),
                finite_values=finite_values, max_abs_error=max_error,
                references_equal=True, contract_equal=True)


def run_trial(folder, config, timeout):
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/"config.json", config)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(folder/"config.json")],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    wall = time.perf_counter()-started
    (folder/"worker.log").write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Worker failed ({result.returncode}); see {folder/'worker.log'}")
    summary = read_json(folder/"summary.json")
    summary["process_wall_seconds"] = wall
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--gpu-mb", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=ROOT/".test-output"/"execution-planner")
    parser.add_argument("--keep-values", action="store_true", help="Retain every timed trial's arrays after validation")
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if args.samples < 24 or args.height < 12 or args.width < 16:
        parser.error("Use at least 24 samples and 12×16 pixels for these fixed windows")
    if not 1 <= args.repeats <= 50 or args.timeout < 1:
        parser.error("repeats must be 1–50 and timeout must be positive")
    if len(set(args.cases)) != len(args.cases):
        parser.error("cases must not contain duplicates")
    from eip.resources import ResourceSettings
    resources = ResourceSettings.from_dict(dict(memory_mb=args.memory_mb, preview_mb=64,
                                               cache_mb=0, gpu_mb=args.gpu_mb, backend="cpu")).record()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = args.output_dir.resolve()/f"planner-{stamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    hashes = code_hashes()
    report = dict(schema_version=1, status="running", created_at=datetime.now(timezone.utc).isoformat(),
                  scope="synthetic_in_memory_engine_execution_with_references_and_cleanup",
                  process_wall_scope="process start, imports, source generation, engine execution, validation hash and artifact saving",
                  excludes=["source generation", "disk decoding", "application exports", "validation", "artifact saving"],
                  fresh_processes=True, serial=True, seed=args.seed,
                  input_shape=[args.samples, args.height, args.width], resources=resources,
                  repeats=args.repeats, python=platform.python_version(), platform=platform.platform(),
                  code_sha256_before=hashes, references={}, trials=[], medians={})
    write_json(folder/"report.json", report)
    common = dict(seed=args.seed, shape=report["input_shape"], resources=resources)
    backends = ["cpu", "gpu"] if args.backend == "both" else [args.backend]
    try:
        # Independent untimed CPU reference is available even for --backend gpu.
        for case in args.cases:
            reference_folder = folder/"references"/case
            summary = run_trial(reference_folder, dict(common, case=case, backend="cpu", optimize=False), args.timeout)
            report["references"][case] = dict(folder=str(reference_folder.relative_to(folder)),
                                              source_sha256=summary["source_sha256"], contract=summary["contract"])
        groups = [(case, backend, optimize) for case in args.cases for backend in backends for optimize in (False, True)]
        for round_index in range(args.repeats):
            # Reverse the first pass, then rotate; neither optimization mode
            # always receives the same initial process/cache conditions.
            ordered = groups if round_index % 2 == 0 else list(reversed(groups))
            shift = (round_index*3) % len(ordered)
            ordered = ordered[shift:]+ordered[:shift]
            for case, backend, optimize in ordered:
                name = f"{case}-{backend}-{'optimized' if optimize else 'baseline'}"
                trial_folder = folder/"trials"/f"{round_index+1:02d}-{name}"
                summary = run_trial(trial_folder, dict(common, case=case, backend=backend, optimize=optimize), args.timeout)
                validation = compare_trial(folder/"references"/case, trial_folder)
                report["trials"].append(dict(round=round_index+1, group=name,
                                             folder=str(trial_folder.relative_to(folder)),
                                             validation=validation, **summary))
                if not args.keep_values:
                    # Only a newly created trial artifact is removed, after its
                    # complete validation. CPU reference arrays remain available.
                    (trial_folder/"values.npy").unlink()
                write_json(folder/"report.json", report)
                print(f"{name} round={round_index+1}: {summary['elapsed_seconds']:.6f}s; validation=passed", flush=True)
        for group in sorted({trial["group"] for trial in report["trials"]}):
            values = [trial for trial in report["trials"] if trial["group"] == group]
            report["medians"][group] = dict(repeats=len(values),
                                           elapsed_seconds=median(value["elapsed_seconds"] for value in values),
                                           process_wall_seconds=median(value["process_wall_seconds"] for value in values))
        report["code_sha256_after"] = code_hashes()
        report["code_unchanged"] = report["code_sha256_after"] == hashes
        report["status"] = "succeeded" if report["code_unchanged"] else "failed"
        if not report["code_unchanged"]:
            report["error"] = "Source changed while benchmarking; rerun against stable code"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", code_sha256_after=code_hashes())
        report["code_unchanged"] = report["code_sha256_after"] == hashes
        write_json(folder/"report.json", report)
        print(f"Report: {folder/'report.json'}", flush=True)
        raise
    write_json(folder/"report.json", report)
    print(f"Report: {folder/'report.json'}", flush=True)
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
