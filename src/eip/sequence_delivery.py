"""Stream typed side outputs without changing the pipeline's main output.

Every delivery has an independently completed manifest. A partial carrier never
becomes a preview source, and trial carriers cannot be promoted to formal input.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json
import shutil
import zipfile

import numpy as np
import tifffile

from eip.algorithms import Cancelled
from eip.persistence import write_json


def result_kind(contract):
    return "label" if contract.semantic == "label" else "coefficient" if contract.semantic in {"g2", "processed_g2"} else "image"


def persist_side_outputs(engine, directory, payload, item, context, emit):
    """Persist each declared side port with bounded memory and full provenance."""
    descriptors = engine.side_outputs()
    item["side_outputs"] = []
    for ordinal, descriptor in enumerate(descriptors):
        contract = descriptor["contract"]
        side = {key: value for key, value in descriptor.items() if key != "contract"}
        # Directory names depend only on an ordinal, never on a user alias.
        folder = Path(directory) / f"side_{ordinal:03d}"
        folder.mkdir()
        side.update(output_directory=folder.name, status="running", completed_frames=0,
                    temporary=payload["mode"] == "trial", artifacts=[], contract=contract.record(),
                    result_kind=result_kind(contract), origin=list(contract.origin),
                    base_shape=[contract.shape[contract.axes.index(a)] for a in "YX"] if contract.spatial else [],
                    upstream_modules=engine.nodes[:descriptor["stage"]],
                    module=next((node for node in engine.nodes if node["id"]==descriptor["node_id"]),None),
                    mappings=engine.mappings, source_revision=engine.source.reader.meta.revision,
                    source=item["source"], run_id=payload["id"],run_spec=payload)
        item["side_outputs"].append(side)
        manifest = folder / "delivery.json"
        write_json(manifest, side)
        values = offsets = None
        try:
            output_format = descriptor.get("export_format", "dataset")
            if output_format not in {"dataset", "npy", "csv", "tiff"}:
                raise ValueError("未知截流导出形式")
            if output_format == "csv" and (contract.spatial or int(np.prod(contract.shape)) > 4096):
                raise ValueError("CSV 截流仅支持每样本不超过 4096 个值的标量或向量；图像请选择 NPY / 数据集 / TIFF")
            if output_format == "tiff" and not contract.spatial:
                raise ValueError("TIFF 截流需要 Y/X 空间轴；信号请选择 NPY / 数据集 / CSV")
            sample = payload.get("sample_index", 0)
            indices = range(sample, sample + 1) if payload["mode"] == "trial" and contract.sequence_axis == "T" and contract.length > 1 else range(contract.length)
            if not indices or indices[-1] >= contract.length:
                raise ValueError("试算截流样本超出此模块直接上游的样本范围")
            first = np.asarray(engine.get_side_output(descriptor["id"], indices[0]))
            actual = replace(contract, dtype=str(first.dtype), length=len(indices))
            side["contract"] = actual.record()
            side["sample_indices"] = dict(start=indices.start, stop=indices.stop, step=indices.step)
            size = len(indices) * max(1, int(np.prod(contract.shape))) * first.dtype.itemsize
            if size + 32 * 1024**2 > shutil.disk_usage(folder).free:
                raise ValueError("输出盘剩余空间不足以保存截流数据")
            values = np.lib.format.open_memmap(folder / "values.npy", mode="w+", dtype=first.dtype, shape=(len(indices), *contract.shape))
            offsets = np.lib.format.open_memmap(folder / "reference_offsets.npy", mode="w+", dtype="uint64", shape=(len(indices),))
            # The measurements table is also used for matching input/result frames.
            with (folder / "references.jsonl").open("wb") as references, (folder / "measurements.csv").open("w", encoding="utf-8-sig", newline="") as table:
                writer = csv.writer(table)
                writer.writerow(["output_index", "source_frame", "time_seconds", "pts"])
                for output_index, source_index in enumerate(indices):
                    context.check_cancelled()
                    value = first if output_index == 0 else np.asarray(engine.get_side_output(descriptor["id"], source_index))
                    if value.shape != contract.shape or value.dtype != first.dtype:
                        raise ValueError("截流输出在序列中改变了形状或数值类型")
                    reference = engine.side_ref(descriptor["id"], source_index)
                    values[output_index] = value
                    offsets[output_index] = references.tell()
                    references.write((json.dumps(reference, ensure_ascii=False, allow_nan=False) + "\n").encode())
                    writer.writerow([output_index, reference.get("index"), reference.get("time"), reference.get("pts")])
                    side["completed_frames"] = output_index + 1
                    if output_index % 128 == 0:
                        emit(dict(type="log", message=f"截流 {side['name']}：{output_index + 1}/{len(indices)}"))
            values.flush()
            offsets.flush()
            del values, offsets
            values = offsets = None
            side["artifacts"] = ["values.npy", "references.jsonl", "reference_offsets.npy", "measurements.csv", "delivery.json", "dataset.json"]
            # A typed carrier exists even for preview-only delivery; it is not
            # promoted until the complete source/run has been published.
            write_json(folder / "dataset.json", dict(schema_version=1, status="succeeded", contract=actual.record(), run_id=payload["id"], source=item["source"], side_output=descriptor["id"], temporary=side["temporary"], upstream_modules=side["upstream_modules"]))
            _export_side(folder, side, context)
            side["status"] = "succeeded"
            write_json(manifest, side)
            if output_format == "dataset" and descriptor.get("destination", "preview") in {"file", "both"}:
                # ZIP keeps numeric axes, references and metadata together.
                with zipfile.ZipFile(folder / "export.zip", "x", compression=zipfile.ZIP_STORED) as archive:
                    for filename in side["artifacts"]:
                        context.check_cancelled()
                        archive.write(folder / filename, arcname=filename)
                side["artifacts"].append("export.zip")
                side["export_file"] = "export.zip"
                write_json(manifest, side)
        except Cancelled:
            side["status"] = "cancelled"
            write_json(manifest, side)
            (folder / "dataset.json").unlink(missing_ok=True)
            raise
        except Exception as error:
            side.update(status="failed", error=str(error))
            write_json(manifest, side)
            (folder / "dataset.json").unlink(missing_ok=True)
            raise
        finally:
            if values is not None:
                values.flush()
            if offsets is not None:
                offsets.flush()


def _export_side(folder, side, context):
    output_format = side.get("export_format", "dataset")
    if side.get("destination", "preview") not in {"file", "both"}:
        return
    if output_format == "npy":
        side["export_file"] = "values.npy"
        return
    if output_format == "dataset":
        return
    contract = side["contract"]
    values = np.load(folder / "values.npy", mmap_mode="r", allow_pickle=False)
    if output_format == "csv":
        shape, axes = contract["shape"], contract["axes"]
        names = ["_".join(f"{axis}{position}" for axis, position in zip(axes, index)) or "value" for index in np.ndindex(tuple(shape))]
        with (folder / "export.csv").open("x", encoding="utf-8-sig", newline="") as table, (folder / "references.jsonl").open(encoding="utf-8") as references:
            writer = csv.writer(table)
            writer.writerow(["output_index", "time_seconds", "lag_samples", *names])
            for index, (value, line) in enumerate(zip(values, references)):
                context.check_cancelled()
                reference = json.loads(line)
                writer.writerow([index, reference.get("time"), reference.get("correlation", {}).get("lag_samples"), *value.reshape(-1).tolist()])
        filename = "export.csv"
    else:
        # Each sample retains P and C dimensions. No selected-channel shortcut.
        axes = contract["axes"].replace("C", "S")
        conversion = contract.get("color_order","RGB")=="BGR" and "C" in contract["axes"]
        export_contract = dict(contract,color_order="RGB") if conversion else contract
        side["export_contract"] = export_contract
        if conversion:
            side["color_conversion"] = "BGR to RGB for TIFF photometric RGB; NPY preserves BGR"
        with tifffile.TiffWriter(folder / "export.tif", bigtiff=values.nbytes > 3_500_000_000) as writer:
            for index, value in enumerate(values):
                context.check_cancelled()
                shown = value[...,::-1] if conversion else value
                writer.write(shown, photometric="rgb" if "C" in contract["axes"] else "minisblack", metadata=dict(axes=axes, output_index=index, contract=export_contract,source_contract=contract,color_conversion=side.get("color_conversion"), side_output=side["id"], run_id=side["run_id"]))
        filename = "export.tif"
    side["export_file"] = filename
    side["artifacts"].append(filename)
