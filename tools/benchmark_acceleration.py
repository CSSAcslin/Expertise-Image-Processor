"""Synthetic TIFF and formal g²(W) benchmark; real CUDA is optional.

Run with the same Python used to launch EIP, for example:
    python tools/benchmark_acceleration.py --backend both --output-dir .test-output

Each formal case uses a fresh Python worker, exports ordinary EIP run records,
and checks three windows against an independent direct formula. No source files
outside the newly created benchmark directory are read or modified. Throughput
for W scans is labelled outputs/s, never source frames/s.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr,"reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
# Resolve the selected source tree before importing EIP. This also lets the
# same benchmark compare an explicitly prepared older source tree; it never
# checks out, downloads, or edits another version itself.
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--source-root", type=Path, default=ROOT/"src")
_early_parser.add_argument("--worker", type=Path)
_early, _ = _early_parser.parse_known_args()
SOURCE_ROOT = _early.source_root.resolve()
sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import tifffile

from eip.domain import SourceSpec
from eip.module_registry import new_node
from eip.readers import open_source, read_image
from eip.resources import configure_reader


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def synthetic_frame(index, height, width):
    rng = np.random.default_rng(179+index)
    phase = int(200*np.sin(index*.17))
    return rng.integers(0, 1024, (height, width*2), dtype=np.uint16)+np.uint16(2400+phase)


def node(algorithm, **parameters):
    result = new_node(algorithm)
    result["parameters"].update(parameters)
    return result


def modules(case, height, width, windows, center, lag):
    nodes = [node("alignment", left=dict(x=0,y=0,width=width,height=height),
                  right=dict(x=width,y=0,width=width,height=height), midline=[width,width])]
    if case == "point":
        nodes.append(node("point_extract", x=width//2, y=height//2))
    nodes.append(node("correlation_window", window_list=windows, center=center+1, lag=lag))
    if case == "pixel-mean":
        nodes.append(node("time_statistics", mode="mean"))
    return nodes


def worker(payload_path):
    # Import here so the measured process duration also exposes worker imports.
    from eip.sequence_processing import process_sequence_job
    payload_path = Path(payload_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    result = None

    def emit(event):
        nonlocal result
        if event["type"] in {"sequence_completed", "failed", "cancelled"}:
            result = event

    started = time.perf_counter()
    process_sequence_job(payload, emit, lambda: False)
    elapsed = time.perf_counter()-started
    from eip import __version__
    summary = dict(job_call_seconds=elapsed, app_version=__version__, source_root=str(SOURCE_ROOT), event=result)
    write_json(payload_path.with_name(payload_path.stem+"-summary.json"), summary)
    return 0 if result and result.get("run",{}).get("status") == "succeeded" else 1


def reader_benchmark(source, count):
    # Both methods use an OS-warmed file. These are small synthetic, cached-file
    # timings, not claims about the user's physical disk or original dataset.
    checksum = []
    started = time.perf_counter()
    for index in range(count):
        value = read_image(Path(source.path), index)
        checksum.append(int(value[0,0]))
    reopened_seconds = time.perf_counter()-started
    reader = open_source(source)
    configure_reader(reader, 0)
    try:
        started = time.perf_counter()
        actual = [int(reader.read_frame(index).pixels[0,0]) for index in range(count)]
        retained_seconds = time.perf_counter()-started
    finally:
        reader.close()
    if actual != checksum:
        raise AssertionError("TIFF reader pixel mismatch")
    return dict(frames=count, per_frame_open_seconds=reopened_seconds,
                retained_handle_seconds=retained_seconds,
                retained_frames_per_second=count/retained_seconds,
                speedup=reopened_seconds/retained_seconds,
                scope="OS-warmed synthetic TIFF; pixel cache disabled; metadata open excluded")


def formula_error(path, values, windows, center, lag, case, height, width):
    reader = open_source(SourceSpec(str(path)))
    configure_reader(reader, 0)
    maximum, masks_equal = 0., True
    checked = sorted({0,len(windows)//2,len(windows)-1})
    try:
        for output in checked:
            window = windows[output]
            start = center-(window-1)//2
            left_sum = np.zeros((height,width), dtype=np.float64)
            right_sum = left_sum.copy()
            product_sum = left_sum.copy()
            for index in range(start,start+window):
                left = reader.read_frame(index).pixels[:,:width].astype(np.float64)
                right = reader.read_frame(index+lag).pixels[:,width:].astype(np.float64)
                left_sum += left
                right_sum += right
                product_sum += left*right
            expected = product_sum*window/(left_sum*right_sum)
            if case == "point":
                expected = expected[height//2,width//2]
            elif case == "pixel-mean":
                expected = expected.mean()
            actual = np.asarray(values[output])
            masks_equal = masks_equal and bool(np.array_equal(np.isfinite(actual),np.isfinite(expected)))
            error = np.abs(actual-expected)
            if np.any(np.isfinite(error)):
                maximum = max(maximum,float(np.nanmax(error)))
    finally:
        reader.close()
    return dict(checked_windows=[windows[i] for i in checked], max_absolute_error=maximum,
                finite_masks_equal=masks_equal, tolerance=2e-11,
                passed=masks_equal and maximum<=2e-11,
                reference="direct float64 mean(L*R)/(mean(L)*mean(R)); positive uint16 synthetic input")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT, help="Explicit EIP src directory to import, including in fresh workers")
    parser.add_argument("--output-dir", type=Path, help="Parent for a new unique benchmark directory; default system temporary directory")
    parser.add_argument("--backend", choices=["cpu","gpu","both"], default="both")
    parser.add_argument("--frames", type=int, default=129)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128, help="Width of each aligned half")
    parser.add_argument("--window-max", type=int, default=65)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--gpu-mb", type=int, default=512)
    parser.add_argument("--cases", nargs="+", choices=["image","point","pixel-mean"], default=["image","point","pixel-mean"])
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if min(args.height,args.width,args.repeats,args.timeout)<1 or args.frames<9 or not 2<=args.window_max<=args.frames-2*abs(args.lag):
        parser.error("Use positive sizes/repeats/timeout, at least 9 frames, and a window that fits the signed lag")
    if args.repeats>10 or args.window_max>8193:
        parser.error("At most 10 repeats and 4096 W outputs per case")
    windows = list(range(2,args.window_max+1,2))
    input_bytes = args.frames*args.height*args.width*4
    output_bytes = len(windows)*args.height*args.width*8
    if max(input_bytes,output_bytes)>256*1024**2:
        parser.error("Synthetic source and each image output are limited to 256 MiB; reduce sizes")
    copies = args.repeats*(2 if args.backend=="both" else 1)
    projected = input_bytes+copies*sum(output_bytes if case=="image" else len(windows)*8 for case in set(args.cases))
    if projected>1024**3:
        parser.error("Combined synthetic arrays are limited to 1 GiB; reduce sizes, cases or repeats")
    if args.output_dir:
        folder = args.output_dir.resolve()/f"acceleration-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        folder.mkdir(parents=True)
    else:
        folder = Path(tempfile.mkdtemp(prefix="eip-acceleration-"))
    path = folder/"synthetic.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write((synthetic_frame(index,args.height,args.width) for index in range(args.frames)),
                     shape=(args.frames,args.height,args.width*2), dtype=np.uint16,
                     photometric="minisblack", metadata=dict(axes="TYX"))
    source = SourceSpec(str(path), interval=.001)
    reader = open_source(source)
    revision = reader.meta.revision
    reader.close()
    from eip import __version__
    from eip.correlation_windows import window_scan_layout
    from eip.modes.split_correlation.backend import devices
    from eip.resources import ResourceSettings
    availability = devices() if args.backend in {"gpu","both"} else dict(gpu=None,reason="not probed")
    if availability["gpu"]:
        import cupy as cp
        availability.update(cupy_version=cp.__version__,cuda_runtime_version=cp.cuda.runtime.runtimeGetVersion(),
                            cuda_driver_version=cp.cuda.runtime.driverGetVersion())
    report = dict(app_version=__version__,source_root=str(SOURCE_ROOT), created_at=datetime.now(timezone.utc).isoformat(),
                  python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),
                  numpy=np.__version__,tifffile=tifffile.__version__,gpu=availability,
                  source=dict(path=str(path),frames=args.frames,shape=[args.height,args.width*2],dtype="uint16"),
                  scope="Synthetic formal EIP jobs, fresh process per case; no measured user GPU speed is inferred",
                  reader=reader_benchmark(source,args.frames),cases=[])
    params = dict(lag=args.lag,window_list=windows,center=None,window_min=2,window_max=args.window_max,
                  window_step=2,min_samples=2,min_mean=1e-9)
    _, center = window_scan_layout(args.frames,params)
    backends = ["cpu","gpu"] if args.backend=="both" else [args.backend]
    for backend in backends:
        if backend=="gpu" and not availability["gpu"]:
            report["cases"].append(dict(backend="gpu",status="skipped",reason=availability["reason"]))
            continue
        settings = ResourceSettings.from_dict(dict(memory_mb=args.memory_mb,preview_mb=16,
            cache_mb=min(128,(args.memory_mb-16)//3),gpu_mb=args.gpu_mb,backend=backend))
        for case in dict.fromkeys(args.cases):
            for repeat in range(args.repeats):
                identifier=f"{backend}-{case}-{repeat+1}"
                payload=dict(id=identifier,display_name=identifier,mode="run",output_dir=str(folder),
                    sources=[asdict(source)],revisions=[revision],steps=modules(case,args.height,args.width,windows,center,args.lag),
                    resources=settings.record(),seed=0,contract_version=3)
                payload_path=folder/f"{identifier}-payload.json"
                write_json(payload_path,payload)
                started=time.perf_counter()
                try:
                    run=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--source-root",str(SOURCE_ROOT),"--worker",str(payload_path)],
                        cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=args.timeout,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                except subprocess.TimeoutExpired:
                    result=dict(case=case,backend=backend,repeat=repeat+1,status="failed",
                                error=f"Worker exceeded {args.timeout} seconds",resources=settings.record(),windows=windows)
                    report["cases"].append(result)
                    write_json(folder/"report.json",report)
                    print(f"{identifier}: failed; timed out",flush=True)
                    continue
                wall=time.perf_counter()-started
                result=dict(case=case,backend=backend,repeat=repeat+1,process_wall_seconds=wall,
                            resources=settings.record(),windows=windows)
                summary_path=payload_path.with_name(payload_path.stem+"-summary.json")
                if run.returncode or not summary_path.is_file():
                    result.update(status="failed",error=(run.stderr or run.stdout)[-4000:])
                    if summary_path.is_file():
                        result["worker"]=json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    summary=json.loads(summary_path.read_text(encoding="utf-8"))
                    event=summary["event"]
                    item=event["run"]["results"][0]
                    values=np.load(Path(event["directory"])/item["output_directory"]/"values.npy",mmap_mode="r")
                    error=formula_error(path,values,windows,center,args.lag,case,args.height,args.width)
                    del values
                    result.update(status="succeeded" if error["passed"] else "failed",job_call_seconds=summary["job_call_seconds"],
                        output_samples=len(windows),outputs_per_second=len(windows)/summary["job_call_seconds"],
                        run_directory=event["directory"],devices=item.get("modules",{}).get("devices"),
                        device_execution=item.get("modules",{}).get("device_execution"),
                        performance=item.get("performance"),validation=error)
                report["cases"].append(result)
                write_json(folder/"report.json",report)
                print(f"{identifier}: {result['status']}; process {wall:.3f} s",flush=True)
    report["medians"] = []
    for backend in backends:
        for case in dict.fromkeys(args.cases):
            successful = [item for item in report["cases"] if item.get("backend")==backend
                          and item.get("case")==case and item["status"]=="succeeded"]
            if successful:
                report["medians"].append(dict(backend=backend,case=case,repeats=len(successful),
                    job_call_seconds=median(item["job_call_seconds"] for item in successful),
                    process_wall_seconds=median(item["process_wall_seconds"] for item in successful),
                    outputs_per_second=median(item["outputs_per_second"] for item in successful)))
    write_json(folder/"report.json",report)
    print(f"Report: {folder/'report.json'}")
    print("W-scan throughput is outputs/s; each output represents a different window size.")
    return 1 if any(item["status"]=="failed" for item in report["cases"]) else 0


if __name__ == "__main__":
    if _early.worker:
        raise SystemExit(worker(_early.worker))
    raise SystemExit(main())
