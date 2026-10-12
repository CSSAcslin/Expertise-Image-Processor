"""Worker for contract v3. Trials and saved runs use SequenceEngine unchanged."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import shutil
import tempfile
import traceback
import numpy as np
import tifffile

from eip import __version__
from eip.algorithms import Cancelled, RunContext
from eip.persistence import write_json
from eip.domain import source_from_dict
from eip.readers import open_source
from eip.processing import statistics
from eip.sequence_engine import SequenceEngine
from eip.sequence_storage import plot_signals
from eip.sequence_delivery import persist_side_outputs


def process_sequence_job(payload,emit,cancelled):
    destination = Path(payload["output_dir"])/f"run_{payload['id']}"
    staging = destination.with_name(destination.name+".partial")
    staging.mkdir(parents=True,exist_ok=False)
    context = RunContext(cancelled,lambda m:emit(dict(type="log",message=m)),payload.get("seed",0))
    run = dict(schema_version=3,contract_version=3,app_version=__version__,run_id=payload["id"],display_name=payload["display_name"],created_at=datetime.now(timezone.utc).isoformat(),status="running",spec=payload,results=[])
    try:
        for position,source_value in enumerate(payload["sources"]):
            reader,engine = None,None
            part = staging/f"source_{position:03d}"
            part.mkdir()
            item = dict(source=source_value,output_directory=part.name,status="running",completed_frames=0,artifacts=[],origin=[0,0])
            run["results"].append(item)
            try:
                reader = open_source(source_from_dict(source_value))
                if reader.meta.revision!=payload["revisions"][position]:
                    raise ValueError("来源自导入后已变化，请重新打开")
                item["data_spec"] = asdict(reader.meta)
                with tempfile.TemporaryDirectory(prefix="index-",dir=staging) as workdir:
                    engine = SequenceEngine(reader,payload["steps"],context,workdir=workdir,center=payload.get("center"),single=payload["mode"]=="trial",input_kind=payload.get("input_kinds",["image"]*len(payload["sources"]))[position],dependencies=payload.get("preview_dependencies",[]))
                    c = engine.output
                    # A trial of a non-correlation chain evaluates one edited
                    # output sample. A τ scan remains a complete Q curve.
                    single = payload["mode"]=="trial" and c.sequence_axis=="T"
                    ordinal = payload.get("sample_index",0)
                    if single and not 0<=ordinal<c.length:
                        raise ValueError("试算样本号超出编辑后的输出范围")
                    indices = range(ordinal,ordinal+1) if single else range(c.length)
                    # Existing frame plugins declare axes, but may return a new
                    # dtype. Learn it from actual numerical output, never cast it
                    # back to the source dtype without an explicit precision node.
                    if c.dtype=="dynamic":
                        c = replace(c,dtype=str(engine.get(len(engine.nodes),indices[0]).dtype))
                        engine.contracts[-1] = c
                        engine.plan = engine._actual_plan()
                    output_c = replace(c,length=len(indices))
                    # Check the disk before allocating a potentially large memmap.
                    size = len(indices)*max(1,int(np.prod(c.shape)))*np.dtype(c.dtype).itemsize
                    if size+32*1024**2>shutil.disk_usage(staging).free:
                        raise ValueError("输出盘剩余空间不足以保存当前形状")
                    values = np.lib.format.open_memmap(part/"values.npy",mode="w+",dtype=c.dtype,shape=(len(indices),*c.shape))
                    offsets = np.lib.format.open_memmap(part/"reference_offsets.npy",mode="w+",dtype="uint64",shape=(len(indices),))
                    signal = not c.spatial
                    columns = int(np.prod(c.shape)) if c.shape else 1
                    fields = ["output_index","source_frame","time_seconds","pts","source_frame_number","trigger_time_seconds","lag_samples","lag_seconds","lag_seconds_min","lag_seconds_max","known_pairs","dtype","min","max","mean","finite_count","invalid_count"]
                    if signal:
                        fields += [f"value_{j+1}" for j in range(columns)]
                    with (part/"references.jsonl").open("wb") as refs,(part/"measurements.csv").open("w",encoding="utf-8-sig",newline="") as file:
                        writer = csv.DictWriter(file,fieldnames=fields)
                        writer.writeheader()
                        for out,j in enumerate(indices):
                            context.check_cancelled()
                            arr = engine.get(len(engine.nodes),j)
                            ref = engine.ref(len(engine.nodes),j)
                            values[out] = arr
                            offsets[out] = refs.tell()
                            refs.write((json.dumps(ref,ensure_ascii=False,allow_nan=False)+"\n").encode())
                            corr = ref.get("correlation",{})
                            row = dict(output_index=out,source_frame=ref.get("index"),time_seconds=ref.get("time"),pts=ref.get("pts"),source_frame_number=ref.get("source_number"),trigger_time_seconds=ref.get("trigger_time"),dtype=str(arr.dtype),**{key:corr.get(key) for key in ("lag_samples","lag_seconds","lag_seconds_min","lag_seconds_max","known_pairs")},**statistics(arr))
                            if signal:
                                row.update({f"value_{k+1}":float(v) for k,v in enumerate(arr.reshape(-1))})
                            writer.writerow(row)
                            if out==0 and c.spatial:
                                axes = c.axes.removeprefix("P")
                                first = arr[0] if c.paired else arr
                                bgr = axes=="YXC" and c.color_order=="BGR"
                                shown = first[...,::-1] if bgr else first
                                image_contract = replace(output_c,color_order="RGB").record() if bgr else output_c.record()
                                tifffile.imwrite(part/"first.tif",shown,photometric="rgb" if axes=="YXC" else "minisblack",metadata=dict(axes="YXS" if axes=="YXC" else "YX",contract=image_contract,source_contract=output_c.record(),color_conversion="BGR to RGB" if bgr else None,reference=ref))
                                item["artifacts"].append("first.tif")
                            item["completed_frames"] = out+1
                            emit(dict(type="progress",source=position,completed=out+1,total=len(indices),name=reader.meta.name))
                    values.flush()
                    offsets.flush()
                    del values,offsets
                    for i,stats in engine.statistics.items():
                        folder = part/f"statistics_{engine.nodes[i]['id']}"
                        folder.mkdir()
                        for key in ("mean","variance","count","scales"):
                            np.save(folder/f"{key}.npy",stats[key],allow_pickle=False)
                    persist_side_outputs(engine,part,payload,item,context,emit)
                    verify = open_source(source_from_dict(source_value))
                    try:
                        if verify.meta.revision!=reader.meta.revision:
                            raise ValueError("运行期间来源发生变化，本次结果未发布")
                    finally:
                        verify.close()
                    calibrations = {}
                    for i,instance in engine.instances.items():
                        if engine.nodes[i]["algorithm"]=="calibration":
                            calibrations[engine.nodes[i]["id"]] = instance.record
                            if instance.lut is not None:
                                np.save(part/f"lut_{engine.nodes[i]['id']}.npy",instance.lut,allow_pickle=False)
                    item.update(status="succeeded",contract=output_c.record(),base_shape=[c.shape[c.axes.index(a)] for a in "YX"] if c.spatial else [],origin=list(c.origin),result_kind="label" if c.semantic=="label" else "coefficient" if c.semantic in {"g2","processed_g2"} else "image",modules=engine.plan,mappings=engine.mappings,calibrations=calibrations)
                    # This manifest is written only after all numeric samples and
                    # references succeeded. Partial datasets cannot be promoted.
                    write_json(part/"dataset.json",dict(schema_version=1,status="succeeded",contract=output_c.record(),run_id=payload["id"],source=source_value,temporary=payload["mode"]=="trial"))
                    if signal:
                        histogram_node = next((n["id"] for n in reversed(engine.nodes)
                                               if n["algorithm"]=="distribution" and n["parameters"]["output"]=="main"),None)
                        plot_signals(part,output_c.record(),engine.mappings,histogram_node)
                        item["artifacts"].extend(["curve.png","curve.svg"])
            except Cancelled:
                item["status"] = "cancelled"
                raise
            except Exception as error:
                item.update(status="failed",error=str(error))
                emit(dict(type="log",message=str(error)))
            finally:
                if item["status"] != "succeeded":
                    for side in item.get("side_outputs",[]):
                        if side["status"] == "succeeded":
                            side.update(status="incomplete",error="所属来源未完整完成，截流输出未发布")
                        folder = part/side["output_directory"]
                        (folder/"dataset.json").unlink(missing_ok=True)
                        (folder/"export.zip").unlink(missing_ok=True)
                        write_json(folder/"delivery.json",side)
                if engine:
                    engine.close()
                if reader:
                    reader.close()
                write_json(staging/"run.json",run)
        successes = sum(item["status"]=="succeeded" for item in run["results"])
        run["status"] = "succeeded" if successes==len(run["results"]) else "completed_with_errors" if successes else "failed"
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(staging/"run.json",run)
        if successes:
            staging.rename(destination)
        emit(dict(type="sequence_completed",run=run,directory=str(destination if successes else staging)))
    except Cancelled:
        run["status"] = "cancelled"
        write_json(staging/"run.json",run)
        emit(dict(type="cancelled",directory=str(staging)))
    except Exception as error:
        run.update(status="failed",error=str(error))
        write_json(staging/"run.json",run)
        raise


def sequence_job_entry(payload,queue,cancel_event):
    try:
        process_sequence_job(payload,queue.put,cancel_event.is_set)
    except Cancelled:
        queue.put(dict(type="cancelled"))
    except Exception as error:
        queue.put(dict(type="failed",message=str(error),details=traceback.format_exc()))
    finally:
        queue.put(dict(type="worker_done"))
