"""Independent real-time provenance and exports for the measured scalar Q scan.

This experiment deliberately accepts only the measured, time-preserving
time_crop -> calibration -> alignment -> point_extract -> correlation_curve
flow. It does not execute modules or use the processor's source-map tree. The
reader supplies exact indexed CINE metadata; nominal FPS is never a substitute.
Each lag uses a bounded NumPy difference vector, rather than Python work and a
tree update for every pair. Intensity validity has no effect on time metadata.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import time

import numpy as np


ALGORITHMS = ("time_crop", "calibration", "alignment", "point_extract", "correlation_curve")
FIELDS = ["output_index", "source_frame", "time_seconds", "pts", "source_frame_number",
          "trigger_time_seconds", "lag_samples", "lag_seconds", "lag_seconds_min",
          "lag_seconds_max", "known_pairs", "dtype", "min", "max", "mean",
          "finite_count", "invalid_count", "value_1"]


def _integer(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def scan_layout(length, parameters, *, center=None, lags=None):
    """Derive a fixed-center Q scan independently; center is zero based.

    A passed lag order is retained, including duplicates, for independent
    numerical tests. The normal configured lag list follows the workbench's
    sorted unique order and includes zero for a range spanning zero.
    """
    length = _integer(length, "source length")
    window = _integer(parameters["window"], "window")
    if length < 1 or window < 1 or window % 2 != 1:
        raise ValueError("The measured Q scan requires a positive odd W and known source length")
    if lags is None:
        if parameters.get("lag_list") is not None:
            lags = sorted(set(_integer(q, "lag") for q in parameters["lag_list"]))
        else:
            first, last = (_integer(parameters[key], key) for key in ("lag_min", "lag_max"))
            step = _integer(parameters["lag_step"], "lag step")
            if step < 1 or first > last:
                raise ValueError("Invalid lag range")
            if (last-first)//step+1 > 4096:
                raise ValueError("A Q scan supports at most 4096 lags")
            lags = list(range(first, last+1, step))
            if first <= 0 <= last and 0 not in lags:
                lags = sorted([*lags, 0])
    else:
        lags = [_integer(q, "lag") for q in lags]
    if not lags or len(lags) > 4096 or max(abs(q) for q in lags) > 8_000_000:
        raise ValueError("Empty or oversized lag list")
    first = window//2 + max(0, -min(lags))
    last = length-1-window//2-max(0, max(lags))
    if last < first:
        raise ValueError("The complete signed-lag windows do not fit the source")
    if center is None:
        configured = parameters.get("center")
        center = _integer(configured, "center")-1 if configured is not None else (first+last)//2
    center = _integer(center, "center")
    if not first <= center <= last:
        raise ValueError("Center cannot accommodate every complete signed-lag window")
    start = center-window//2
    return dict(center=center, start=start, stop=start+window, window=window, lags=lags)


def _configuration(spec):
    spec = spec.get("spec", spec)
    nodes = [node for node in spec["steps"] if node.get("enabled", True)]
    if tuple(node["algorithm"] for node in nodes) != ALGORITHMS:
        raise ValueError("Direct Q exports require the measured time-preserving paired point flow")
    crop, _, _, point, correlation = nodes
    p = crop["parameters"]
    if (p.get("unit", "samples") != "samples" or p.get("first", 1) != 1
            or p.get("last") is not None):
        raise ValueError("Direct Q exports require the complete source sample range")
    if (point["parameters"].get("output", "main") != "main"
            or correlation["parameters"].get("analysis") != "curve"):
        raise ValueError("Direct Q exports require a main-path point and Q curve")
    if any(node.get("manifest") or node.get("code_hash") for node in nodes):
        raise ValueError("Custom algorithm provenance is outside this fixed experiment")
    if len(spec.get("sources", [])) != 1:
        raise ValueError("Direct Q exports require exactly one source")
    return spec, correlation["parameters"], correlation["id"]


def source_reference(reader, index):
    """Read a CINE's indexed TIME64 values, without decoding image pixels."""
    if not hasattr(reader, "times") or not hasattr(reader, "first_number"):
        raise ValueError("Direct Q exports require indexed CINE metadata")
    ticks = int(reader.ticks[index]) if reader.ticks is not None else None
    value = reader.times[index] if reader.times is not None else None
    return dict(index=index, time=None if value is None else float(value),
                pts=ticks-int(reader.ticks[0]) if ticks is not None else None,
                source_number=int(reader.first_number)+index,
                trigger_time=(ticks-int(reader.trigger_ticks))/4294967296 if ticks is not None else None)


def summarize_pair_times(times, start, stop, lags, *, check=None):
    """Return true paired-time summaries with O(W) transient storage.

    ``None`` is unknown. As in the processor, a nonfinite float still counts as
    a present timestamp: such unusual input follows an explicit sequential
    fallback to retain its order-sensitive min/max behavior. Normal CINE
    TIME64 differences are finite. Sum reduction order can differ by float64
    rounding from the processor's balanced tree; no intensity sum uses it.
    """
    start, stop = _integer(start, "start"), _integer(stop, "stop")
    lags = [_integer(q, "lag") for q in lags]
    window = stop-start
    if start < 0 or window < 1:
        raise ValueError("Invalid left window")
    metrics = dict(algorithm="NumPy per-lag exact-time difference and reduction",
                   pair_elements=window*len(lags), lag_queries=len(lags),
                   pixel_reads=0, max_scratch_bytes=0, timestamp_array_bytes=0,
                   nonfinite_fallback_windows=0, time_source="indexed metadata; no nominal FPS")
    if times is None:
        return [dict(lag_seconds=None, lag_seconds_min=None, lag_seconds_max=None,
                     known_pairs=0) for _ in lags], metrics
    raw = np.asarray(times)
    if raw.ndim != 1:
        raise ValueError("Timestamps must be a one-dimensional indexed sequence")
    if min([start, *[start+q for q in lags]]) < 0 or max([stop, *[stop+q for q in lags]]) > len(raw):
        raise ValueError("A paired-time window is outside the source")
    if raw.dtype == object:
        present = np.fromiter((value is not None for value in raw), dtype=bool, count=len(raw))
        numbers = np.fromiter((0. if value is None else float(value) for value in raw),
                              dtype=np.float64, count=len(raw))
    else:
        numbers = np.asarray(raw, dtype=np.float64)
        present = np.ones(len(numbers), dtype=bool)
    metrics["timestamp_array_bytes"] = numbers.nbytes+present.nbytes
    left, left_present = numbers[start:stop], present[start:stop]
    records = []
    for lag in lags:
        if check is not None:
            check()
        known = left_present & present[start+lag:stop+lag]
        # lag==0 must still subtract: inf-inf and nan-nan stay nonfinite.
        with np.errstate(invalid="ignore", over="ignore"):
            differences = numbers[start+lag:stop+lag]-left
        count = int(np.count_nonzero(known))
        values = differences[known]
        metrics["max_scratch_bytes"] = max(metrics["max_scratch_bytes"],
                                            known.nbytes+differences.nbytes+values.nbytes)
        if not count:
            total, minimum, maximum = 0., None, None
        elif np.all(np.isfinite(values)):
            total = float(np.sum(values, dtype=np.float64))
            minimum, maximum = float(np.min(values)), float(np.max(values))
        else:
            metrics["nonfinite_fallback_windows"] += 1
            total, minimum, maximum = 0., None, None
            for value in values:
                value = float(value)
                total += value
                minimum = value if minimum is None else min(minimum, value)
                maximum = value if maximum is None else max(maximum, value)
        records.append(dict(lag_seconds=total/window if count == window else None,
                            lag_seconds_min=minimum, lag_seconds_max=maximum,
                            known_pairs=count))
    return records, metrics


def output_contract(length, supplied=None):
    """Scalar g² Q contract: Q denotes lag, never acquisition time."""
    contract = dict(axes="", shape=[], length=length, sequence_axis="Q", dtype="float64",
                    semantic="g2", statistics=False, origin=[0, 0], spatial_operation=None,
                    meaning="pixel_g2", color_order="RGB", full_axes="Q", full_shape=[length],
                    label=f"Q:{length}", axis_meanings={"Q": "有符号延迟"})
    if supplied is not None:
        if supplied != contract:
            raise ValueError("The supplied contract is not the measured scalar float64 g² Q result")
        return deepcopy(supplied)
    return contract


def write_exports(folder, spec, reader, values, *, center=None, lags=None,
                  contract=None, make_plots=True):
    """Rebuild complete Q references, CSV and plot from new values and source.

    ``values.npy`` must already hold the independently computed values. Input
    run records supply configuration only; no historical result or reference is
    copied. The static plot renderer is a shared presentation utility.
    """
    started = time.perf_counter()
    folder = Path(folder)
    spec, parameters, module_id = _configuration(spec)
    layout = scan_layout(int(reader.meta.frame_count), parameters, center=center, lags=lags)
    center, start, stop, window, lags = (layout[key] for key in ("center", "start", "stop", "window", "lags"))
    values = np.asarray(values)
    if values.shape != (len(lags),) or values.dtype != np.dtype("float64"):
        raise ValueError("Direct Q values have the wrong shape / dtype")
    if not (folder/"values.npy").is_file():
        raise ValueError("Save newly computed values.npy before exporting its provenance")
    contract = output_contract(len(lags), supplied=contract)
    summaries, metrics = summarize_pair_times(reader.times, start, stop, lags)
    summary_seconds = time.perf_counter()-started
    records_start = time.perf_counter()
    base = source_reference(reader, center)
    offsets = np.lib.format.open_memmap(folder/"reference_offsets.npy", mode="w+",
                                        dtype="uint64", shape=(len(lags),))
    try:
        with (folder/"references.jsonl").open("wb") as references, (folder/"measurements.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for ordinal, (lag, value, summary) in enumerate(zip(lags, values, summaries, strict=True)):
                correlation = dict(module_id=module_id, center_sample=center, lag_samples=lag,
                                   window=window, left_start=start, left_stop=stop,
                                   right_start=start+lag, right_stop=stop+lag, **summary)
                reference = dict(base, sample_index=ordinal, source_sample=center,
                                 source_span=[center, center+1], correlation=correlation)
                offsets[ordinal] = references.tell()
                references.write((json.dumps(reference, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8"))
                finite = bool(np.isfinite(value))
                scalar = float(value)
                statistic = scalar if finite else None
                writer.writerow(dict(output_index=ordinal, source_frame=base["index"], time_seconds=base["time"],
                                     pts=base["pts"], source_frame_number=base["source_number"],
                                     trigger_time_seconds=base["trigger_time"], dtype="float64", lag_samples=lag,
                                     **summary, min=statistic, max=statistic, mean=statistic,
                                     finite_count=int(finite), invalid_count=int(not finite), value_1=scalar))
        offsets.flush()
    finally:
        offsets._mmap.close()
    dataset = dict(schema_version=1, status="succeeded", contract=contract,
                   run_id=spec["id"], source=spec["sources"][0], temporary=False)
    (folder/"dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    artifacts = ["values.npy", "reference_offsets.npy", "references.jsonl", "measurements.csv", "dataset.json"]
    records_seconds = time.perf_counter()-records_start
    plot_start = time.perf_counter()
    if make_plots:
        from eip.sequence_storage import plot_signals
        plot_signals(folder, contract)
        artifacts += ["curve.png", "curve.svg"]
    return dict(contract=contract, artifacts=artifacts, output_samples=len(lags),
                reference_records=len(lags), csv_records=len(lags), metadata_pixel_reads=0,
                time_summary_seconds=summary_seconds, records_export_seconds=records_seconds,
                plot_seconds=time.perf_counter()-plot_start, reference_metrics=metrics,
                center=center, left_window=[start, stop])
