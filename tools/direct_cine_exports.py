"""Direct scalar-g² exports for the fixed CINE benchmark, without flow machinery.

Only source metadata and the caller's newly computed values are consumed.  No
saved references or values are copied from a previous workbench run.  The common
static plot renderer is reused as an export utility, not a numerical algorithm.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np


ALGORITHMS = ("time_crop", "calibration", "alignment", "roi_crop",
              "time_statistics", "correlation_map")
POINT_ALGORITHMS = ("time_crop", "calibration", "alignment", "point_extract", "correlation_map")

FIELDS = ["output_index", "source_frame", "time_seconds", "pts", "source_frame_number",
          "trigger_time_seconds", "lag_samples", "lag_seconds", "lag_seconds_min",
          "lag_seconds_max", "known_pairs", "dtype", "min", "max", "mean",
          "finite_count", "invalid_count", "value_1"]


def _configuration(spec):
    spec = spec.get("spec", spec)
    nodes = [node for node in spec["steps"] if node.get("enabled", True)]
    algorithms = tuple(node["algorithm"] for node in nodes)
    if algorithms == ALGORITHMS:
        crop, _, alignment, roi, statistics, correlation = nodes
        p = statistics["parameters"]
        if (p.get("mode") != "mean" or p.get("output", "main") != "main"
                or p.get("validity", "independent") != "independent"):
            raise ValueError("Direct exports require a main-path spatial mean signal")
        selection = dict(roi["parameters"],kind="region")
    elif algorithms == POINT_ALGORITHMS:
        crop, _, alignment, point, correlation = nodes
        p = point["parameters"]
        if p.get("output", "main") != "main":
            raise ValueError("Direct point exports require main-path point extraction")
        selection = dict(p,kind="point")
    else:
        raise ValueError("Direct exports support only the fixed ROI-mean or point scalar-g² flows")
    p = crop["parameters"]
    if (p.get("unit", "samples") != "samples" or p.get("first", 1) != 1
            or p.get("last") is not None):
        raise ValueError("Direct exports require the complete input sample range")
    p = correlation["parameters"]
    window = p["window"]
    if (type(window) is not int or window < 1 or window % 2 != 1
            or p.get("stride", 1) != 1 or p.get("analysis", "map") != "map"
            or p.get("map_lag", 0) != 0):
        raise ValueError("Direct exports require odd W, stride 1 and fixed zero lag")
    return spec, alignment["parameters"], selection, correlation["id"], window


def source_reference(reader, index):
    """Read exact indexed metadata without reading or decoding image pixels."""
    if hasattr(reader, "times"):
        # CINE TIME64 ticks are fixed-point timestamps, not averaged frame rate.
        ticks = int(reader.ticks[index]) if reader.ticks is not None else None
        return dict(index=index,
                    time=float(reader.times[index]) if reader.times is not None else None,
                    pts=ticks-int(reader.ticks[0]) if ticks is not None else None,
                    source_number=reader.first_number+index,
                    trigger_time=(ticks-reader.trigger_ticks)/4294967296 if ticks is not None else None)
    if hasattr(reader, "timestamps"):
        timestamps = reader.timestamps
        interval = reader.source.interval
        numbers = reader.source_numbers
        trigger_times = reader.trigger_times
        return dict(index=index,
                    time=timestamps[index] if timestamps is not None else index*interval if interval else None,
                    pts=None, source_number=numbers[index] if numbers is not None and len(numbers) else None,
                    trigger_time=trigger_times[index] if trigger_times is not None and len(trigger_times) else None)
    raise ValueError("Direct exports require indexed CINE or sequence metadata; pixel decoding is not a metadata fallback")


def output_contract(length, alignment, roi=None, supplied=None, *, sampling=None):
    selection = sampling if sampling is not None else roi
    if selection is None:
        raise ValueError("An explicit region or point sampling selection is required")
    kind = selection.get("kind", "region")
    if kind not in {"region", "point"}:
        raise ValueError("Unknown direct-export sampling kind")
    spatial_operation = "mean" if kind == "region" else None
    meaning = "regional_mean_signal_g2" if kind == "region" else "pixel_g2"
    if supplied is not None:
        contract = deepcopy(supplied)
        if (contract.get("axes") != "" or contract.get("shape") != []
                or contract.get("sequence_axis") != "T" or contract.get("dtype") != "float64"
                or contract.get("semantic") != "g2" or contract.get("length") != length
                or contract.get("meaning") != meaning or contract.get("spatial_operation") != spatial_operation):
            raise ValueError("The supplied direct-output contract is not scalar float64 g² on T")
        return contract
    if (alignment.get("dx", 0) < 0 or alignment.get("dy", 0) < 0
            or alignment.get("mirror_x", False) or alignment.get("mirror_y", False)):
        raise ValueError("Provide the independently computed output contract for shifted origins or mirrored geometry")
    line = alignment.get("midline")
    if (not line or alignment["left"]["x"] >= min(line)
            or alignment["right"]["x"]+alignment.get("dx", 0) < max(line)):
        raise ValueError("Provide the independently computed output contract when the common geometry may have a nonzero origin")
    # The workbench point contract retains its upstream image origin. The
    # selected point coordinates remain in the spec; do not silently redefine
    # that contract while benchmarking it. ROI cropping actually shifts origin.
    origin = [alignment["left"]["x"]+(selection["x"] if kind == "region" else 0),
              alignment["left"]["y"]+(selection["y"] if kind == "region" else 0)]
    return dict(axes="", shape=[], length=length, sequence_axis="T", dtype="float64",
                semantic="g2", statistics=False, origin=origin, spatial_operation=spatial_operation,
                meaning=meaning, color_order="RGB", full_axes="T",
                full_shape=[length], label=f"T:{length}", axis_meanings={"T": "时间样本"})


def write_exports(folder, spec, reader, values, *, contract=None, make_plots=True):
    """Write source references, scalar CSV, typed manifest and optional graphics.

    ``values.npy`` must already contain the caller's newly computed ``values``.
    All metadata is rebuilt from this reader and the supplied flow, not from a
    baseline result.  The returned contract/artifact list can be retained in the
    standalone benchmark report.  Time this call as normal output work.
    """
    folder = Path(folder)
    spec, alignment, selection, module_id, window = _configuration(spec)
    values = np.asarray(values)
    length = int(reader.meta.frame_count)-window+1
    if length < 1 or values.shape != (length,) or values.dtype != np.dtype("float64"):
        raise ValueError("Direct scalar output shape / dtype differs from the source and W")
    if not (folder/"values.npy").is_file():
        raise ValueError("Save the newly computed values.npy before exporting its metadata")
    contract = output_contract(length, alignment, selection, supplied=contract)
    # Zero-lag pair time differences are exactly zero when both input times
    # exist.  Prefix counts handle unknown times without a W-long rescan.
    known = np.empty(int(reader.meta.frame_count)+1, dtype=np.int64)
    known[0] = 0
    for i in range(int(reader.meta.frame_count)):
        time = source_reference(reader, i)["time"]
        known[i+1] = known[i]+int(time is not None and np.isfinite(time))
    offsets = np.lib.format.open_memmap(folder/"reference_offsets.npy", mode="w+",
                                        dtype="uint64", shape=(length,))
    try:
        with (folder/"references.jsonl").open("wb") as references, (folder/"measurements.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for out, value in enumerate(values):
                center = out+window//2
                base = source_reference(reader, center)
                count = int(known[out+window]-known[out])
                correlation = dict(module_id=module_id, center_sample=center, lag_samples=0,
                                   window=window, left_start=out, left_stop=out+window,
                                   right_start=out, right_stop=out+window,
                                   lag_seconds=0.0 if count == window else None,
                                   lag_seconds_min=0.0 if count else None,
                                   lag_seconds_max=0.0 if count else None, known_pairs=count)
                reference = dict(base, sample_index=out, source_sample=center,
                                 source_span=[center, center+1], correlation=correlation)
                offsets[out] = references.tell()
                references.write((json.dumps(reference, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8"))
                finite = bool(np.isfinite(value))
                scalar = float(value)
                statistic = scalar if finite else None
                row = dict(output_index=out, source_frame=base["index"], time_seconds=base["time"],
                           pts=base["pts"], source_frame_number=base["source_number"],
                           trigger_time_seconds=base["trigger_time"], dtype="float64",
                           **{key: correlation[key] for key in ("lag_samples", "lag_seconds", "lag_seconds_min", "lag_seconds_max", "known_pairs")},
                           min=statistic, max=statistic, mean=statistic, finite_count=int(finite),
                           invalid_count=int(not finite), value_1=scalar)
                writer.writerow(row)
        offsets.flush()
    finally:
        offsets._mmap.close()
    dataset = dict(schema_version=1, status="succeeded", contract=contract,
                   run_id=spec["id"], source=spec["sources"][0], temporary=False)
    (folder/"dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    artifacts = ["values.npy", "reference_offsets.npy", "references.jsonl", "measurements.csv", "dataset.json"]
    if make_plots:
        from eip.sequence_storage import plot_signals
        plot_signals(folder, contract)
        artifacts += ["curve.png", "curve.svg"]
    return dict(contract=contract, sampling=selection, artifacts=artifacts, output_samples=length,
                reference_records=length, csv_records=length,
                metadata_pixel_reads=0, source_time_scope="exact indexed metadata")
