"""Dedicated CINE experiments, independent of the module executor.

This deliberately supports only the measured flow: all samples, fixed linear
sensor calibration, integer nearest alignment, then either a rectangular ROI's
independent finite spatial means or direct paired point extraction, followed by
zero-lag stride-one scalar g2. It is not a new backend
or a general pipeline. The existing reader supplies only decoding and metadata.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
    steps = [node for node in spec["steps"] if node.get("enabled", True)]
    roi_flow = ["time_crop", "calibration", "alignment", "roi_crop", "time_statistics", "correlation_map"]
    point_flow = ["time_crop", "calibration", "alignment", "point_extract", "correlation_map"]
    algorithms = [node["algorithm"] for node in steps]
    if algorithms not in (roi_flow, point_flow) or len(spec["sources"]) != 1:
        raise ValueError("Dedicated experiment requires the measured ROI-mean or point single-source flow")
    if spec.get("preview_dependencies") or spec.get("input_kinds", ["image"]) != ["image"]:
        raise ValueError("Dependencies and non-image inputs are outside this experiment")
    if any(node.get("execution", {}).get("precision", "native") != "native" for node in steps):
        raise ValueError("Precision overrides are outside this experiment")
    if any(node.get("target", "both") != "both" or node.get("manifest") or node.get("code_hash") for node in steps):
        raise ValueError("Custom algorithms and single-channel targets are outside this experiment")
    if algorithms == roi_flow:
        t, c, a, r, s, g = [node["parameters"] for node in steps]
        point = None
        sampling = dict(kind="regional_mean", roi=r, validity="independent", semantic="regional_mean_signal_g2")
    else:
        t, c, a, point, g = [node["parameters"] for node in steps]
        r, s = None, None
        sampling = dict(kind="point", point=point, semantic="pixel_g2")
    if t.get("unit") != "samples" or t.get("first") != 1 or t.get("last") is not None:
        raise ValueError("The measured time crop must retain the complete sample sequence")
    if (c.get("curve") != "linear" or c.get("lut_path") is not None
            or c.get("black_left") != 0 or c.get("black_right") != 0
            or c.get("saturation_code") is not None or c.get("gamma", 1) != 1
            or c.get("input_semantics", "sensor_codes") != "sensor_codes"
            or not np.isfinite(c["input_max"]) or c["input_max"] <= 0):
        raise ValueError("Only fixed linear / input_max calibration without clipping or offsets is supported")
    if (a.get("interpolation") != "nearest" or a.get("mirror_x") or a.get("mirror_y")
            or int(a["dx"]) != a["dx"] or int(a["dy"]) != a["dy"]):
        raise ValueError("Only integer nearest alignment without mirrors is supported")
    if s is not None and (s.get("mode") != "mean" or s.get("validity") != "independent" or s.get("output") != "main"):
        raise ValueError("Only independent finite spatial means on the main route are supported")
    if point is not None and (point.get("output") != "main" or any(type(point[k]) is not int or point[k] < 0 for k in ("x", "y"))):
        raise ValueError("Only main-route, nonnegative integer paired point extraction is supported")
    if (g.get("analysis") != "map" or g.get("map_lag") != 0 or g.get("stride") != 1
            or g.get("center") is not None or type(g["window"]) is not int
            or g["window"] < 3 or g["window"] % 2 != 1):
        raise ValueError("Only zero-lag, stride-one, moving-center g2 is supported")
    return dict(calibration=c, alignment=a, roi=r, point=point, sampling=sampling, correlation=g,
                node_ids=[node["id"] for node in steps])


def alignment_layout(shape, parameters, roi):
    """Independently derive the common valid sensor geometry and ROI indices."""
    if len(shape) != 2:
        raise ValueError("Dedicated flow requires 2D grayscale frames")
    height, width = shape
    left, right = parameters["left"], parameters["right"]
    for rectangle in (left, right):
        x, y, w, h = [rectangle[key] for key in ("x", "y", "width", "height")]
        if min(x, y) < 0 or min(w, h) < 1 or x+w > width or y+h > height:
            raise ValueError("Alignment rectangle is outside the input")
    yy, xx = np.indices((left["height"], left["width"]), dtype=np.float64)
    xr, yr = xx+parameters["dx"], yy+parameters["dy"]
    xr, yr = np.floor(xr+.5), np.floor(yr+.5)
    line_l = np.interp(yy+left["y"], [0, max(1, height-1)], parameters["midline"])
    line_r = np.interp(yr+right["y"], [0, max(1, height-1)], parameters["midline"])
    valid = ((xx+left["x"] < line_l) & (xr+right["x"] >= line_r)
             & (xr >= 0) & (yr >= 0) & (xr <= right["width"]-1) & (yr <= right["height"]-1))
    vy, vx = np.nonzero(valid)
    if not vx.size:
        raise ValueError("No common valid alignment pixels")
    x0, y0, x1, y1 = int(vx.min()), int(vy.min()), int(vx.max())+1, int(vy.max())+1
    x, y, w, h = [roi[key] for key in ("x", "y", "width", "height")]
    if min(x, y) < 0 or min(w, h) < 1 or x+w > x1-x0 or y+h > y1-y0:
        raise ValueError("ROI is outside the aligned common bounding box")
    mask = valid[y0:y1, x0:x1]
    xr, yr = xr[y0:y1, x0:x1], yr[y0:y1, x0:x1]
    return dict(left=left, right=right, crop=(x0, y0, x1, y1), mask=mask,
                xr=np.clip(xr.astype(np.int64), 0, right["width"]-1),
                yr=np.clip(yr.astype(np.int64), 0, right["height"]-1),
                roi=(x, y, w, h), roi_mask=mask[y:y+h, x:x+w],
                origin=[left["x"]+x0+x, left["y"]+y0+y],
                specialized_left=(slice(left["y"]+y0+y, left["y"]+y0+y+h), slice(left["x"]+x0+x, left["x"]+x0+x+w)))


def calibrated(codes, input_max, xp):
    valid = xp.isfinite(codes)
    value = codes.astype(xp.float64) / input_max
    value = xp.where(xp.isfinite(value), value, xp.nan)
    value -= 0
    return xp.where(valid, value, xp.nan)


def point_layout(shape, parameters, point):
    """Reuse only coordinate bounding-box derivation; no spatial reduction."""
    layout = alignment_layout(shape, parameters, dict(x=point["x"], y=point["y"], width=1, height=1))
    x, y = point["x"], point["y"]
    x0, y0, _, _ = layout["crop"]
    left, right = layout["left"], layout["right"]
    layout.update(point=(x, y), point_valid=bool(layout["mask"][y, x]),
                  point_left=(left["y"]+y0+y, left["x"]+x0+x),
                  point_right=(right["y"]+int(layout["yr"][y,x]), right["x"]+int(layout["xr"][y,x])),
                  contract_origin=[left["x"]+x0, left["y"]+y0])
    return layout


def aligned(value, layout, xp):
    l, r = layout["left"], layout["right"]
    x0, y0, x1, y1 = layout["crop"]
    left = value[l["y"]:l["y"]+l["height"], l["x"]:l["x"]+l["width"]][y0:y1, x0:x1].copy()
    right = value[r["y"]:r["y"]+r["height"], r["x"]:r["x"]+r["width"]]
    result = xp.zeros(layout["mask"].shape, dtype=xp.float64)
    result += right[layout["yr"], layout["xr"]]
    return xp.stack((xp.where(layout["mask"], left, xp.nan), xp.where(layout["mask"], result, xp.nan)))


def mean_finite(value, xp, axes):
    finite = xp.isfinite(value)
    count = finite.sum(axis=axes, dtype=xp.uint64)
    total = xp.where(finite, value, 0).sum(axis=axes)
    return xp.where(count, total/xp.maximum(count, 1), xp.nan)


def cropped_codes(raw, layout, out=None):
    """Pull only consumed sensor samples; calibration still precedes averaging."""
    x, y, w, h = layout["roi"]
    r = layout["right"]
    right = raw[r["y"]:r["y"]+r["height"], r["x"]:r["x"]+r["width"]]
    left = raw[layout["specialized_left"]]
    sampled = right[layout["yr"][y:y+h, x:x+w], layout["xr"][y:y+h, x:x+w]]
    if out is None:
        return np.stack((left, sampled))
    out[0], out[1] = left, sampled
    return out


def cpu_correlation(signals, parameters):
    """Preserve fluctuation summation order and 256-output rebuilds exactly."""
    w = parameters["window"]
    out = np.empty(len(signals)-w+1, np.float64)
    sums = np.zeros(4, np.float64)
    base = None
    def add(index, sign):
        left, right = signals[index]
        valid = np.isfinite(left) and np.isfinite(right)
        dl = left-base[0] if valid else np.float64(0)
        dr = right-base[1] if valid else np.float64(0)
        sums[0] += sign*dl
        sums[1] += sign*dr
        sums[2] += sign*(dl*dr)
        sums[3] += sign*int(valid)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for k in range(len(out)):
            if k % 256 == 0:
                base = np.nan_to_num(signals[k+(w-1)//2], nan=0., posinf=0., neginf=0.)
                sums.fill(0)
                for j in range(k, k+w):
                    add(j, 1)
            else:
                add(k-1, -1)
                add(k+w-1, 1)
            safe = max(sums[3], 1)
            ml, mr = sums[0]/safe, sums[1]/safe
            il, ir = base[0]+ml, base[1]+mr
            covariance = sums[2]/safe-ml*mr
            result = 1+covariance/(il*ir)
            out[k] = result if sums[3] >= parameters["min_samples"] and il > parameters["min_mean"] and ir > parameters["min_mean"] and np.isfinite(result) else np.nan
    return out


_UPDATE = r'''
const bool valid = isfinite(l) && isfinite(r);
const double dl = valid ? l - bl : 0.0;
const double dr = valid ? r - br : 0.0;
sl += sign * dl;
sr += sign * dr;
slr += sign * (dl * dr);
n += sign * (valid ? 1.0 : 0.0);
'''


def gpu_literal_correlation(signals, p, cp):
    # Dedicated kernels; no import or reuse of production algorithm kernels.
    update = cp.ElementwiseKernel("float64 l, float64 r, float64 bl, float64 br, int64 sign",
        "float64 sl, float64 sr, float64 slr, float64 n", _UPDATE,
        "direct_flow_scalar_g2_update", options=("--fmad=false",))
    update_many = cp.ElementwiseKernel("raw float64 lb, raw float64 rb, float64 bl, float64 br, int64 sign, int64 batch",
        "float64 sl, float64 sr, float64 slr, float64 n", r'''
        for (long long j=0; j<batch; ++j) {
            const double l=lb[j], r=rb[j];
            const bool valid=isfinite(l)&&isfinite(r);
            const double dl=valid?l-bl:0.0, dr=valid?r-br:0.0;
            sl += sign*dl; sr += sign*dr; slr += sign*(dl*dr); n += sign*(valid?1.0:0.0);
        }
        ''', "direct_flow_scalar_g2_update_many", options=("--fmad=false",))
    result = cp.ElementwiseKernel("float64 sl, float64 sr, float64 slr, float64 n, float64 bl, float64 br, float64 minimum, int64 count",
        "float64 output", r'''
        const double safe = n > 1.0 ? n : 1.0;
        const double ml = sl / safe, mr = sr / safe;
        const double il = bl + ml, ir = br + mr;
        const double covariance = slr / safe - ml * mr;
        const double g = 1.0 + covariance / (il * ir);
        output = (n >= count && il > minimum && ir > minimum && isfinite(g)) ? g : nan("");
        ''', "direct_flow_scalar_g2_result", options=("--fmad=false",))
    n, w = len(signals), p["window"]
    outputs = cp.empty(n-w+1, cp.float64)
    sums = cp.zeros(4, cp.float64)
    views = tuple(sums[i:i+1] for i in range(4))
    base = None
    for k in range(n-w+1):
        if k % 256 == 0:
            base = cp.nan_to_num(signals[k+(w-1)//2], nan=0., posinf=0., neginf=0.)
            sums.fill(0)
            for j in range(k, k+w, 64):
                end = min(j+64, k+w)
                if end-j == 1:
                    update(signals[j, 0:1], signals[j, 1:2], base[0:1], base[1:2], 1, *views)
                else:
                    update_many(signals[j:end, 0], signals[j:end, 1], base[0:1], base[1:2], 1, end-j, *views)
        else:
            for j, sign in ((k-1, -1), (k+w-1, 1)):
                update(signals[j, 0:1], signals[j, 1:2], base[0:1], base[1:2], sign, *views)
        result(*views, base[0:1], base[1:2], p["min_mean"], p["min_samples"], outputs[k:k+1])
    return outputs


_SERIAL = r'''
extern "C" __global__ void direct_flow_g2_serial(const double* signal, double* output,
    const long long length, const long long window, const double minimum, const long long count) {
    if (blockIdx.x || threadIdx.x) return;
    double sl = 0.0, sr = 0.0, slr = 0.0, n = 0.0, bl = 0.0, br = 0.0;
    for (long long k = 0; k < length-window+1; ++k) {
        if (k % 256 == 0) {
            bl = signal[(k+(window-1)/2)*2]; br = signal[(k+(window-1)/2)*2+1];
            if (!isfinite(bl)) bl = 0.0;
            if (!isfinite(br)) br = 0.0;
            sl = sr = slr = n = 0.0;
            for (long long j=k; j<k+window; ++j) {
                const double l=signal[j*2], r=signal[j*2+1];
                const bool v=isfinite(l)&&isfinite(r);
                const double dl=v?l-bl:0.0, dr=v?r-br:0.0;
                sl += dl; sr += dr; slr += dl*dr; n += v?1.0:0.0;
            }
        } else {
            for (int side=0; side<2; ++side) {
                const long long j=side?k+window-1:k-1;
                const double sign=side?1.0:-1.0;
                const double l=signal[j*2], r=signal[j*2+1];
                const bool v=isfinite(l)&&isfinite(r);
                const double dl=v?l-bl:0.0, dr=v?r-br:0.0;
                sl += sign*dl; sr += sign*dr; slr += sign*(dl*dr); n += sign*(v?1.0:0.0);
            }
        }
        const double safe=n>1.0?n:1.0, ml=sl/safe, mr=sr/safe;
        const double il=bl+ml, ir=br+mr, covariance=slr/safe-ml*mr;
        const double g=1.0+covariance/(il*ir);
        output[k]=(n>=count&&il>minimum&&ir>minimum&&isfinite(g))?g:nan("");
    }
}
'''


def gpu_specialized_correlation(signals, p, cp):
    kernel = cp.RawKernel(_SERIAL, "direct_flow_g2_serial", options=("--fmad=false",))
    outputs = cp.empty(len(signals)-p["window"]+1, cp.float64)
    kernel((1,), (1,), (signals, outputs, np.int64(len(signals)), np.int64(p["window"]),
                          np.float64(p["min_mean"]), np.int64(p["min_samples"])))
    return outputs


def run_direct(spec, *, backend="cpu", implementation="literal", alignment_backend="inherit",
               batch_size=32, output_dir, exports="full"):
    started = time.perf_counter()
    phases = dict(setup=0., read=0., preprocess=0., correlation=0., output=0., cleanup=0.)
    transfers = dict(upload_calls=0, upload_bytes=0, download_calls=0, download_bytes=0)
    cp = None
    @contextmanager
    def phase(name):
        tick = time.perf_counter()
        try:
            yield
        finally:
            phases[name] += time.perf_counter()-tick
    def upload(value):
        transfers["upload_calls"] += 1; transfers["upload_bytes"] += value.nbytes
        return cp.asarray(value)
    def download(value):
        transfers["download_calls"] += 1; transfers["download_bytes"] += value.nbytes
        return cp.asnumpy(value)
    reader = None
    try:
        with phase("setup"):
            if backend not in {"cpu", "gpu"} or implementation not in {"literal", "specialized"}:
                raise ValueError("Unknown backend or implementation")
            if not 1 <= batch_size <= 256 or alignment_backend not in {"inherit", "cpu"}:
                raise ValueError("Batch must be 1..256 and alignment backend inherit/cpu")
            if implementation == "specialized" and backend == "gpu" and alignment_backend == "cpu":
                raise ValueError("Specialized GPU experiment is GPU throughout; use literal for mixed devices")
            cfg = validate_spec(spec)
            from eip.domain import SourceSpec
            from eip.readers import open_source
            reader = open_source(SourceSpec(**spec["sources"][0]))
            if spec.get("revisions") and reader.meta.revision != spec["revisions"][0]:
                raise ValueError("Input source revision differs from the saved experiment")
            is_point = cfg["sampling"]["kind"] == "point"
            layout = point_layout(reader.meta.shape, cfg["alignment"], cfg["point"]) if is_point else alignment_layout(reader.meta.shape, cfg["alignment"], cfg["roi"])
            n = reader.meta.frame_count
            if n is None or n < cfg["correlation"]["window"]:
                raise ValueError("Input is too short for the measured window")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if any(output_dir.iterdir()):
                raise ValueError("Dedicated output directory must be empty; previous outputs are never overwritten")
            if backend == "gpu":
                import cupy as cp
                device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
                device_name = device_name.decode() if isinstance(device_name, bytes) else device_name
                cp.get_default_memory_pool().set_limit(size=512*1024**2)
                xp = cp
                device_layout = dict(layout)
                constant_keys = (() if is_point else ("roi_mask",)) if implementation == "specialized" else ("mask", "xr", "yr", "roi_mask")
                for key in constant_keys:
                    device_layout[key] = upload(layout[key])
            else:
                xp, device_layout, device_name = np, layout, "CPU / NumPy"
            signals = np.empty((n, 2), np.float64)
            device_signals = cp.empty((n, 2), cp.float64) if cp is not None and implementation == "specialized" else None
            x, y, w, h = layout["roi"]
            code_shape = (min(n, batch_size if cp is not None else 1), 2) if is_point else (min(n, batch_size if cp is not None else 1), 2, h, w)
            codes_buffer = np.empty(code_shape, dtype=reader.meta.dtype) if implementation == "specialized" else None
            input_max = cfg["calibration"]["input_max"]
        for first in range(0, n, batch_size if backend == "gpu" else 1):
            stop = min(n, first+(batch_size if backend == "gpu" else 1))
            if implementation == "specialized":
                # Coordinates are a compile-time specialization of this fixed
                # nearest/linear flow; no intensities are averaged before /max.
                for offset,j in enumerate(range(first, stop)):
                    with phase("read"):
                        raw = reader.read_frame(j).pixels
                    with phase("preprocess"):
                        if is_point:
                            codes_buffer[offset,0] = raw[layout["point_left"]]
                            codes_buffer[offset,1] = raw[layout["point_right"]]
                        else:
                            cropped_codes(raw, layout, codes_buffer[offset])
                with phase("preprocess"):
                    codes = codes_buffer[:stop-first]
                    values = calibrated(upload(codes) if cp is not None else codes, input_max, xp)
                    if is_point:
                        # Two calibrated pixel values, not a 1x1 image statistic.
                        means = values if layout["point_valid"] else xp.full_like(values, xp.nan)
                    else:
                        values = xp.where(device_layout["roi_mask"][None, None, ...], values, xp.nan)
                        means = mean_finite(values, xp, (2, 3))
                    if cp is not None:
                        device_signals[first:stop] = means
                    else:
                        signals[first:stop] = means
            else:
                with phase("read"):
                    raw = [reader.read_frame(j).pixels for j in range(first, stop)]
                with phase("preprocess"):
                    block = upload(np.stack(raw)) if cp is not None else raw
                    for offset, j in enumerate(range(first, stop)):
                        value = calibrated(block[offset], input_max, xp)
                        if cp is not None and alignment_backend == "cpu":
                            pair = upload(aligned(download(value), layout, np))
                        else:
                            pair = aligned(value, device_layout, xp)
                        x, y, w, h = layout["roi"]
                        if is_point:
                            mean = pair[:, y, x]
                        else:
                            roi = pair[:, y:y+h, x:x+w].copy()
                            mean = mean_finite(roi, xp, (1, 2))
                        signals[j] = download(mean) if cp is not None else mean
        if cp is not None:
            with phase("preprocess"):
                cp.cuda.get_current_stream().synchronize()
        with phase("correlation"):
            if cp is None:
                values = cpu_correlation(signals, cfg["correlation"])
            else:
                if device_signals is None:
                    device_signals = upload(signals)
                function = gpu_literal_correlation if implementation == "literal" else gpu_specialized_correlation
                values = download(function(device_signals, cfg["correlation"], cp))
        with phase("output"):
            if cp is not None and implementation == "specialized":
                signals = download(device_signals)
            np.save(output_dir/"signals.npy", signals, allow_pickle=False)
            np.save(output_dir/"values.npy", values, allow_pickle=False)
            if exports == "full":
                from direct_cine_exports import write_exports, output_contract
                contract_sampling = dict(cfg["point"],kind="point") if is_point else None
                contract = output_contract(len(values), cfg["alignment"], cfg["roi"], sampling=contract_sampling)
                contract["origin"] = layout["contract_origin"] if is_point else layout["origin"]
                artifacts = write_exports(output_dir, spec, reader, values, contract=contract)
            elif exports == "none":
                artifacts = {"scope": "Numeric arrays only; excludes provenance, CSV and figures"}
            else:
                raise ValueError("Unknown exports mode")
            outputs = len(values)
            rebuilds = (outputs+255)//256
            rolling = outputs-rebuilds
            window = cfg["correlation"]["window"]
            singles = 2*rolling+rebuilds*int(window % 64 == 1)
            many = rebuilds*(window//64+int(window % 64 > 1))
            gpu_calls = 0 if cp is None else singles+many+outputs if implementation == "literal" else 1
            actual_batch = min(n, batch_size if cp is not None else 1)
            signal_statistics = []
            for channel in range(2):
                column = signals[:,channel]
                finite = column[np.isfinite(column)]
                signal_statistics.append(dict(channel="left" if channel == 0 else "right",
                    finite_count=int(finite.size), invalid_count=int(n-finite.size),
                    zero_count=int(np.count_nonzero(finite == 0)),
                    minimum=float(finite.min()) if finite.size else None,
                    maximum=float(finite.max()) if finite.size else None,
                    mean=float(finite.mean()) if finite.size else None))
            result = dict(status="succeeded", backend=backend, actual_backend=backend, implementation=implementation,
                          alignment_backend="cpu" if backend == "cpu" or alignment_backend == "cpu" else "gpu",
                          device_name=device_name, batch_size=batch_size, gpu_pool_limit_bytes=512*1024**2 if cp is not None else None,
                          samples=n, output_samples=len(values), parameters=cfg, transfers=transfers, artifacts=artifacts,
                          signal_statistics=signal_statistics,
                          joint_finite_signal_samples=int(np.count_nonzero(np.isfinite(signals).all(axis=1))),
                          finite_output_samples=int(np.count_nonzero(np.isfinite(values))),
                          correlation_kernel_calls=gpu_calls,
                          correlation_calls=dict(rebuilds=rebuilds, scalar_updates=singles if cp is not None and implementation == "literal" else 0,
                                                 batched_updates=many if cp is not None and implementation == "literal" else 0,
                                                 result_calls=outputs if cp is not None and implementation == "literal" else 0,
                                                 pair_updates=rebuilds*window+2*rolling,
                                                 specialized_serial_calls=int(cp is not None and implementation == "specialized")),
                          preprocessing_work=dict(batches=(n+actual_batch-1)//actual_batch,
                              calibrated_pixels=n*(2*w*h if implementation == "specialized" else int(np.prod(reader.meta.shape))),
                              consumed_roi_pixels=None if is_point else n*2*w*h,
                              consumed_point_samples=n*2 if is_point else None,
                              consumed_sensor_values=n*2*w*h, signals_bytes=signals.nbytes,
                              spatial_reduction="none (direct paired point extraction)" if is_point else "independent finite spatial mean",
                              sampling_sensor_coordinates=dict(left=list(layout["point_left"]),right=list(layout["point_right"])) if is_point else None,
                              input_transfer_block_bytes=actual_batch*(2*w*h if implementation == "specialized" else int(np.prod(reader.meta.shape)))*np.dtype(reader.meta.dtype).itemsize,
                              retained_raw_frames=1 if implementation == "specialized" else actual_batch,
                              signal_residency="device through correlation" if cp is not None and implementation == "specialized" else "host between preprocessing and correlation"),
                          output_dir=str(output_dir), exports=exports, source_revision=reader.meta.revision,
                          semantic=cfg["sampling"]["semantic"], accumulation="float64 fluctuations; joint finite pair counts; baseline rebuilt every 256 outputs; no FMA",
                          scope="Dedicated numerical flow and requested exports; shared source decoder only. Setup includes reader opening, CUDA import/context and first allocation. Host phase times include stage-boundary synchronization waits; no forced per-frame device barrier. Specialized static sensor gather is charged to preprocessing; point extraction performs no spatial statistic. GPU signals remain resident through correlation.")
    finally:
        with phase("cleanup"):
            if reader is not None:
                reader.close()
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
    result.update(elapsed_seconds=time.perf_counter()-started, phases_seconds=phases,
                  read_compute_seconds=phases["read"]+phases["preprocess"]+phases["correlation"])
    result["job_call_seconds"] = result["elapsed_seconds"]
    (output_dir/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--implementation", choices=("literal", "specialized"), required=True)
    parser.add_argument("--alignment-backend", choices=("cpu", "inherit"), default="inherit")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--exports", choices=("full", "none"), default="full")
    args = parser.parse_args()
    if args.report:
        report_path, output_path = args.report.resolve(), args.output_dir.resolve()
        if report_path == output_path or not report_path.is_relative_to(output_path) or report_path.exists():
            parser.error("--report must be a new file inside this trial's output directory")
    result = run_direct(load_spec(args.spec), backend=args.backend, implementation=args.implementation,
                        alignment_backend=args.alignment_backend, batch_size=args.batch_size,
                        output_dir=args.output_dir, exports=args.exports)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if args.report.resolve() != args.output_dir.resolve()/"report.json":
            with args.report.open("x", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status="succeeded", backend=result["backend"], implementation=result["implementation"],
                          elapsed_seconds=result["elapsed_seconds"], output_dir=result["output_dir"])))


if __name__ == "__main__":
    main()
