"""Dedicated fixed-centre paired-point delay scan, independent of the executor.

Only the source decoder and the previous experiment's independent static
coordinate derivation are shared. Numerical g2 uses no production estimator,
module graph, stage store, provenance tree, or GPU production kernel. The
measured flow is full time range -> fixed linear calibration -> fixed integer
nearest alignment -> paired point -> fixed-centre scalar g2(Q).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_spec(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return value.get("spec", value)


def validate_spec(spec):
    """Reject unsupported science instead of silently simplifying the flow."""
    spec = spec.get("spec", spec)
    nodes = [n for n in spec["steps"] if n.get("enabled", True)]
    if ([n["algorithm"] for n in nodes] !=
            ["time_crop", "calibration", "alignment", "point_extract", "correlation_curve"]
            or len(spec["sources"]) != 1):
        raise ValueError("Dedicated delay experiment requires the measured paired-point curve flow")
    if spec.get("preview_dependencies") or spec.get("input_kinds", ["image"]) != ["image"]:
        raise ValueError("Dependencies and non-image inputs are outside this experiment")
    if any(n.get("manifest") or n.get("code_hash") or n.get("target", "both") != "both"
           or n.get("execution", {}).get("precision", "native") != "native" for n in nodes):
        raise ValueError("Custom algorithms, target restrictions and precision overrides are unsupported")
    t, c, a, q, p = [n["parameters"] for n in nodes]
    if t.get("unit", "samples") != "samples" or t.get("first", 1) != 1 or t.get("last") is not None:
        raise ValueError("The measured crop must retain the complete sample sequence")
    if (c.get("curve") != "linear" or c.get("lut_path") is not None
            or c.get("black_left") != 0 or c.get("black_right") != 0
            or c.get("saturation_code") is not None or c.get("gamma", 1) != 1
            or c.get("input_semantics", "sensor_codes") != "sensor_codes"
            or not np.isfinite(c["input_max"]) or c["input_max"] <= 0):
        raise ValueError("Only fixed linear /input_max calibration without clipping or offsets is supported")
    if (a.get("interpolation") != "nearest" or a.get("mirror_x") or a.get("mirror_y")
            or int(a["dx"]) != a["dx"] or int(a["dy"]) != a["dy"]):
        raise ValueError("Only integer nearest alignment without mirrors is supported")
    if q.get("output") != "main" or any(type(q[k]) is not int or q[k] < 0 for k in ("x", "y")):
        raise ValueError("Only main-route nonnegative integer point extraction is supported")
    if (p.get("analysis") != "curve" or type(p["window"]) is not int
            or p["window"] < 3 or p["window"] % 2 != 1
            or type(p["min_samples"]) is not int or not 1 <= p["min_samples"] <= p["window"]
            or not np.isfinite(p["min_mean"]) or p["min_mean"] <= 0):
        raise ValueError("Only fixed-centre scalar g2 curves with odd W and valid finite thresholds are supported")
    if p.get("center") is not None and (type(p["center"]) is not int or p["center"] < 1):
        raise ValueError("A configured centre is a positive one-based upstream sample")
    return dict(calibration=c, alignment=a, point=q, correlation=p,
                node_ids=[n["id"] for n in nodes])


def delay_values(parameters):
    p = parameters
    if p.get("lag_list") is None:
        if (any(type(p[k]) is not int for k in ("lag_min", "lag_max", "lag_step"))
                or p["lag_step"] < 1 or p["lag_min"] > p["lag_max"]):
            raise ValueError("Invalid signed delay interval")
        if (p["lag_max"]-p["lag_min"])//p["lag_step"]+1 > 4096:
            raise ValueError("The delay list exceeds 4096 entries")
        lags = list(range(p["lag_min"], p["lag_max"]+1, p["lag_step"]))
        if p["lag_min"] <= 0 <= p["lag_max"]:
            lags.append(0)
    else:
        lags = list(p["lag_list"])
    if not lags or len(lags) > 4097 or any(type(q) is not int or abs(q) > 8_000_000 for q in lags):
        raise ValueError("Invalid signed delay list")
    lags = sorted(set(lags))
    if len(lags) > 4096:
        raise ValueError("The delay list exceeds 4096 entries")
    return lags


def _merge_intervals(intervals):
    merged = []
    for first, stop in sorted(intervals):
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(stop, merged[-1][1])
        else:
            merged.append([int(first), int(stop)])
    return merged


def scan_layout(length, parameters, lags=None, *, center=None, read_block=32):
    """Independent Q plan; round the exact needed union to bounded read blocks.

    `center` is zero-based; the saved parameter is one-based. Overrides of Q
    retain the caller's order; by default the configured Q list is sorted as in
    the workbench. The centre is derived from the configured full Q range even
    if only a measured subset is requested, unless explicitly supplied.
    """
    if type(length) is not int or length < 1 or type(read_block) is not int or not 1 <= read_block <= 256:
        raise ValueError("Positive source length and read block 1..256 are required")
    full_lags = delay_values(parameters)
    selected = full_lags if lags is None else list(lags)
    if not selected or len(selected) > 4096 or any(type(q) is not int or abs(q)>8_000_000 for q in selected):
        raise ValueError("Explicit Q entries must be 1..4096 integer signed delays")
    w = parameters["window"]
    if type(w) is not int or w < 3 or w % 2 != 1:
        raise ValueError("The dedicated experiment supports positive odd windows")
    half = w//2
    first = half+max(0, -min(full_lags))
    last = length-1-half-max(0, max(full_lags))
    if last < first:
        raise ValueError("Source is too short for W and the configured signed delays")
    if center is None:
        center = parameters["center"]-1 if parameters.get("center") is not None else (first+last)//2
    if type(center) is not int or not first <= center <= last:
        raise ValueError("The fixed centre is outside the complete valid Q domain")
    start, stop = center-half, center+half+1
    needed = _merge_intervals([(center, center+1), (start, stop), *[(start+q, stop+q) for q in selected]])
    if needed[0][0] < 0 or needed[-1][1] > length:
        raise ValueError("An explicit delay accesses samples outside the source")
    physical = _merge_intervals([(a//read_block*read_block, min(length, (b+read_block-1)//read_block*read_block)) for a,b in needed])
    return dict(center=center, start=start, stop=stop, window=w, lags=selected,
                needed_intervals=needed, read_intervals=physical, read_block=read_block,
                exact_needed_samples=sum(b-a for a,b in needed),
                read_samples=sum(b-a for a,b in physical))


def prepare_signals(spec, *, lags=None, center=None, read_block=32):
    """Read CINE frames once; retain only the two consumed sensor intensities.

    Returns `(signals, times, metadata, reader)`. The caller owns the live reader
    for exact source metadata / exports and must close it. Signals are full-T
    float64 with NaN in unread slots; actual read indices are saved separately.
    CPU coordinate gathering and /input_max are equivalent for this fixed,
    nearest, equal-zero-dark calibration. No full frame is uploaded to the GPU.
    """
    tick = time.perf_counter()
    cfg = validate_spec(spec)
    from eip.domain import SourceSpec
    from eip.readers import open_source
    from direct_cine_flow import point_layout
    reader = open_source(SourceSpec(**spec["sources"][0]))
    try:
        if spec.get("revisions") and reader.meta.revision != spec["revisions"][0]:
            raise ValueError("The source revision differs from the saved experiment")
        n = reader.meta.frame_count
        if type(n) is not int:
            raise ValueError("This experiment requires exact indexed source length")
        if n*(2*8+8+8) > 256*1024**2:
            raise ValueError("Dedicated signal/index storage exceeds 256 MiB")
        layout = point_layout(reader.meta.shape, cfg["alignment"], cfg["point"])
        plan = scan_layout(n, cfg["correlation"], lags, center=center, read_block=read_block)
        indices = np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in plan["read_intervals"]])
        signals = np.full((n,2), np.nan, np.float64)
        codes = np.empty((len(indices),2), np.float64)
        times = np.asarray(reader.times, dtype=np.float64).copy() if getattr(reader,"times",None) is not None else np.full(n,np.nan)
        setup = time.perf_counter()-tick
        read, gather = 0., 0.
        for offset, index in enumerate(indices):
            tick = time.perf_counter()
            frame = reader.read_frame(int(index))
            read += time.perf_counter()-tick
            tick = time.perf_counter()
            codes[offset,0] = frame.pixels[layout["point_left"]]
            codes[offset,1] = frame.pixels[layout["point_right"]]
            gather += time.perf_counter()-tick
        tick = time.perf_counter()
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            codes /= cfg["calibration"]["input_max"]
        codes[~np.isfinite(codes)] = np.nan
        if not layout["point_valid"]:
            codes.fill(np.nan)
        signals[indices] = codes
        gather += time.perf_counter()-tick
        meta = dict(plan=plan, samples=n, shape=list(reader.meta.shape), source_revision=reader.meta.revision,
                    source_name=reader.meta.name, read_indices=indices,
                    signal_bytes=signals.nbytes, times_bytes=times.nbytes,
                    source_dtype=reader.meta.dtype, calibration_backend="cpu", alignment_backend="cpu",
                    selected_sensor_coordinates=dict(left=list(layout["point_left"]),right=list(layout["point_right"])),
                    point_valid=layout["point_valid"], calibrated_sensor_values=int(2*len(indices)),
                    phases_seconds=dict(source_setup=setup, read=read, preprocess=gather),
                    cfg=cfg)
        return signals,times,meta,reader
    except BaseException:
        reader.close()
        raise


def _inputs(signals, parameters, lags, center):
    if (signals.ndim != 2 or signals.shape[1] != 2 or str(signals.dtype) != "float64"
            or not signals.flags.c_contiguous):
        raise ValueError("The experiment requires contiguous float64 signals (T,2)")
    plan = scan_layout(len(signals), parameters, lags, center=center)
    if (not np.isfinite(parameters["min_mean"]) or parameters["min_mean"] <= 0
            or type(parameters["min_samples"]) is not int or parameters["min_samples"] < 1):
        raise ValueError("Invalid finite g2 thresholds")
    return plan


def compute_cpu(signals, parameters, *, lags=None, center=None):
    """Independent ordered float64 estimator; Q iterations use NumPy cumsum.

    cumsum's last value preserves increasing-sample accumulation, avoiding both
    Python-per-pair overhead and the reassociation of NumPy's parallel/pairwise
    sum. Temporary storage is O(W); it never constructs a Q x W pair matrix.
    """
    plan = _inputs(signals, parameters, lags, center)
    base = np.nan_to_num(signals[plan["center"]], nan=0., posinf=0., neginf=0.)
    left = signals[plan["start"]:plan["stop"],0]
    result = np.empty(len(plan["lags"]), np.float64)
    counts = np.empty(len(result), np.uint32)
    with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
        for out, lag in enumerate(plan["lags"]):
            right = signals[plan["start"]+lag:plan["stop"]+lag,1]
            valid = np.isfinite(left)&np.isfinite(right)
            dl = np.where(valid, left-base[0], 0.)
            dr = np.where(valid, right-base[1], 0.)
            sl = np.cumsum(dl, dtype=np.float64)[-1]
            sr = np.cumsum(dr, dtype=np.float64)[-1]
            slr = np.cumsum(dl*dr, dtype=np.float64)[-1]
            count = int(np.count_nonzero(valid))
            safe = max(count, 1)
            ml, mr = sl/safe, sr/safe
            il, ir = base[0]+ml, base[1]+mr
            value = 1.+(slr/safe-ml*mr)/(il*ir)
            counts[out] = count
            result[out] = value if (count >= parameters["min_samples"] and il > parameters["min_mean"]
                                    and ir > parameters["min_mean"] and np.isfinite(value)) else np.nan
    return result, counts


_GPU_SOURCE = r'''
extern "C" __global__ void dedicated_lag_scan_ordered_float64(
    const double* signals, const long long* lags, double* output, unsigned int* counts,
    const long long q_count, const long long center, const long long first,
    const long long stop, const double min_mean, const long long min_samples) {
    const long long q = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (q >= q_count) return;
    const long long lag = lags[q];
    double bl = signals[center*2], br = signals[center*2+1];
    if (!isfinite(bl)) bl = 0.0;
    if (!isfinite(br)) br = 0.0;
    double sl = 0.0, sr = 0.0, slr = 0.0, n = 0.0;
    for (long long j=first; j<stop; ++j) {
        const double left = signals[j*2], right = signals[(j+lag)*2+1];
        const bool valid = isfinite(left) && isfinite(right);
        const double dl = valid ? left-bl : 0.0;
        const double dr = valid ? right-br : 0.0;
        sl += dl;
        sr += dr;
        slr += dl*dr;
        n += valid ? 1.0 : 0.0;
    }
    const double safe = n>1.0?n:1.0;
    const double ml=sl/safe, mr=sr/safe;
    const double il=bl+ml, ir=br+mr;
    const double value=1.0+(slr/safe-ml*mr)/(il*ir);
    output[q] = (n>=min_samples && il>min_mean && ir>min_mean && isfinite(value)) ? value : nan("");
    counts[q] = (unsigned int)n;
}
'''
_GPU_KERNELS = {}


def gpu_kernel(cp):
    """Caller may compile before timing; first compilation is otherwise charged."""
    key = (id(cp), int(cp.cuda.Device().id))
    if key not in _GPU_KERNELS:
        _GPU_KERNELS[key] = cp.RawKernel(_GPU_SOURCE, "dedicated_lag_scan_ordered_float64", options=("--fmad=false",))
    return _GPU_KERNELS[key]


def compute_gpu(signals, parameters, *, lags=None, center=None, cp=None):
    """One independent Q per CUDA thread; original W-order float64 accumulation.

    Input and returned `(values, counts)` reside on the GPU. The caller accounts
    for upload/download and synchronization. No time parallel reduction or FMA
    is used. Only O(Q) output/delay arrays are allocated alongside the signal.
    """
    if cp is None:
        import cupy as cp
    plan = _inputs(signals, parameters, lags, center)
    delays = cp.asarray(np.asarray(plan["lags"],dtype=np.int64))
    values = cp.empty(len(delays),cp.float64)
    counts = cp.empty(len(delays),cp.uint32)
    gpu_kernel(cp)(((len(delays)+127)//128,), (128,),
        (signals, delays, values, counts, np.int64(len(delays)), np.int64(plan["center"]),
         np.int64(plan["start"]), np.int64(plan["stop"]), np.float64(parameters["min_mean"]),
         np.int64(parameters["min_samples"])))
    return values,counts


def run_direct(spec, output_dir, backend="cpu", exports="full", *, lags=None, center=None,
               read_block=32, warm_repeats=3):
    """Complete fresh-process experiment; first setup/compile included in total.

    Warm repeated pure numerical calls are recorded separately *after* the
    first full-flow timer, and are not substituted for cold end-to-end latency.
    Values/counts/signals/read_indices/times are retained for independent checks.
    """
    started = time.perf_counter()
    if backend not in {"cpu", "gpu"} or exports not in {"full", "none"} or not 0 <= warm_repeats <= 20:
        raise ValueError("Unknown backend/exports or invalid warm repetition count")
    spec = spec.get("spec",spec)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("The experiment output directory must be empty")
    reader = None
    cp = None
    phases = dict(source_setup=0.,read=0.,preprocess=0.,device_setup=0.,compile=0.,upload=0.,correlation=0.,download=0.,output=0.,cleanup=0.)
    try:
        signals,times,meta,reader = prepare_signals(spec,lags=lags,center=center,read_block=read_block)
        phases.update(meta.pop("phases_seconds"))
        plan = meta["plan"]
        parameters = meta["cfg"]["correlation"]
        transfers = dict(upload_calls=0,upload_bytes=0,download_calls=0,download_bytes=0)
        kernel_events_ms = None
        if backend == "gpu":
            tick = time.perf_counter()
            import cupy as cp
            cp.cuda.Device().use()
            cp.get_default_memory_pool().set_limit(size=128*1024**2)
            device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            if isinstance(device_name,bytes):
                device_name = device_name.decode()
            cp.cuda.get_current_stream().synchronize()
            phases["device_setup"] = time.perf_counter()-tick
            tick = time.perf_counter()
            gpu_kernel(cp).compile()
            phases["compile"] = time.perf_counter()-tick
            tick = time.perf_counter()
            device_signals = cp.asarray(signals)
            cp.cuda.get_current_stream().synchronize()
            phases["upload"] = time.perf_counter()-tick
            transfers.update(upload_calls=2,upload_bytes=signals.nbytes+len(plan["lags"])*8,
                             download_calls=2,download_bytes=len(plan["lags"])*12)
            before,after = cp.cuda.Event(),cp.cuda.Event()
            tick = time.perf_counter()
            before.record()
            device_values,device_counts = compute_gpu(device_signals,parameters,lags=plan["lags"],center=plan["center"],cp=cp)
            after.record()
            after.synchronize()
            phases["correlation"] = time.perf_counter()-tick
            kernel_events_ms = float(cp.cuda.get_elapsed_time(before,after))
            tick = time.perf_counter()
            values,counts = cp.asnumpy(device_values),cp.asnumpy(device_counts)
            phases["download"] = time.perf_counter()-tick
        else:
            device_name = "CPU / NumPy ordered float64 cumsum"
            tick = time.perf_counter()
            values,counts = compute_cpu(signals,parameters,lags=plan["lags"],center=plan["center"])
            phases["correlation"] = time.perf_counter()-tick
        tick = time.perf_counter()
        for name,value in (("signals",signals),("times",times),("read_indices",meta["read_indices"]),("values",values),("counts",counts)):
            np.save(output_dir/f"{name}.npy",value,allow_pickle=False)
        if exports == "full":
            from direct_lag_references import write_exports
            artifacts = write_exports(output_dir,spec,reader,values,center=plan["center"],lags=plan["lags"])
        else:
            artifacts = dict(scope="Numeric arrays only; excludes source provenance, CSV and plots")
        phases["output"] = time.perf_counter()-tick
        tick = time.perf_counter()
        reader.close()
        reader = None
        phases["cleanup"] = time.perf_counter()-tick
        elapsed = time.perf_counter()-started
        warm = []
        for _ in range(warm_repeats):
            tick = time.perf_counter()
            if cp is None:
                repeated,repeated_counts = compute_cpu(signals,parameters,lags=plan["lags"],center=plan["center"])
            else:
                dv,dc = compute_gpu(device_signals,parameters,lags=plan["lags"],center=plan["center"],cp=cp)
                cp.cuda.get_current_stream().synchronize()
            warm.append(time.perf_counter()-tick)
            if cp is None:
                np.testing.assert_array_equal(repeated,values)
                np.testing.assert_array_equal(repeated_counts,counts)
        kernel_only = []
        if cp is not None and warm_repeats:
            # All inputs, outputs and launch arguments are prepared outside the
            # event interval. A preceding complete launch is explicitly warm.
            # These are actual CUDA events around one kernel, not host submit
            # latency, and do not include allocation, upload or download.
            kernel = gpu_kernel(cp)
            device_lags = cp.asarray(np.asarray(plan["lags"],dtype=np.int64))
            warm_values,warm_counts = cp.empty_like(device_values),cp.empty_like(device_counts)
            arguments = (device_signals,device_lags,warm_values,warm_counts,np.int64(len(plan["lags"])),
                         np.int64(plan["center"]),np.int64(plan["start"]),np.int64(plan["stop"]),
                         np.float64(parameters["min_mean"]),np.int64(parameters["min_samples"]))
            grid,block = ((len(plan["lags"])+127)//128,),(128,)
            kernel(grid,block,arguments)
            cp.cuda.get_current_stream().synchronize()
            for _ in range(warm_repeats):
                before,after = cp.cuda.Event(),cp.cuda.Event()
                before.record()
                kernel(grid,block,arguments)
                after.record()
                after.synchronize()
                kernel_only.append(float(cp.cuda.get_elapsed_time(before,after))/1000.)
            np.testing.assert_array_equal(cp.asnumpy(warm_values),values)
            np.testing.assert_array_equal(cp.asnumpy(warm_counts),counts)
        indices = meta.pop("read_indices")
        finite = signals[indices]
        stats = []
        for channel in range(2):
            column = finite[:,channel]
            good = column[np.isfinite(column)]
            stats.append(dict(channel=channel,finite_count=int(good.size),invalid_count=int(len(column)-good.size),
                              minimum=float(good.min()) if good.size else None,maximum=float(good.max()) if good.size else None))
        result = dict(status="succeeded",backend=backend,actual_backend=backend,device_name=device_name,
            elapsed_seconds=elapsed,job_call_seconds=elapsed,phases_seconds=phases,
            numerical_warm_seconds=warm,numerical_warm_median_seconds=float(np.median(warm)) if warm else None,
            warm_kernel_seconds=kernel_only,warm_kernel_median_seconds=float(np.median(kernel_only)) if kernel_only else None,
            warm_kernel_scope="CUDA events around one already-compiled kernel launch; preallocated signal/delays/outputs; preceding untimed warm launch; excludes allocations, transfers and end-to-end work" if cp is not None else None,
            gpu_event_registered_call_ms=kernel_events_ms,gpu_event_scope="Complete dedicated numeric call including delay upload/allocation and host launch gaps; not pure kernel-only time" if cp is not None else None,
            first_compile_seconds=phases["compile"],signal_statistics=stats,
            finite_output_samples=int(np.isfinite(values).sum()),output_samples=len(values),
            pair_updates=len(values)*plan["window"],correlation_kernel_calls=1 if cp is not None else 0,
            transfers=transfers,artifacts=artifacts,exports=exports,output_dir=str(output_dir),**meta,
            accumulation="float64 fluctuations, increasing j accumulation, joint finite pairs, fixed baseline at the actual centre; CUDA --fmad=false",
            scope="Fresh complete direct flow, same indexed decoder and independent geometry helper; coordinate gather and fixed /input_max on CPU; exactly bounded physical source reads; GPU context/compile included; full requested exports. Warm numerical repeats happen after this end-to-end interval.")
        (output_dir/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        return result
    finally:
        if reader is not None:
            reader.close()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--backend",choices=("cpu","gpu"),required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--exports",choices=("full","none"),default="full")
    parser.add_argument("--read-block",type=int,default=32)
    parser.add_argument("--warm-repeats",type=int,default=3)
    args = parser.parse_args()
    result = run_direct(load_spec(args.spec),args.output_dir,backend=args.backend,exports=args.exports,
                        read_block=args.read_block,warm_repeats=args.warm_repeats)
    print(json.dumps(dict(status=result["status"],backend=result["backend"],elapsed_seconds=result["elapsed_seconds"],
                          numerical_warm_median_seconds=result["numerical_warm_median_seconds"],output_dir=result["output_dir"])))


if __name__ == "__main__":
    main()
