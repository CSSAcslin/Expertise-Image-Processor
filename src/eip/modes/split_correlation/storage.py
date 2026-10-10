from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import json
import math

import numpy as np

from eip.domain import DataSpec, FrameData, FrameRef, Selection, file_revision
from eip.persistence import write_json


def close_mapping(array):
    if isinstance(array,np.memmap):
        if array.flags.writeable:
            array.flush()
        array._mmap.close()


def clean_number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def load_mappings(paths):
    arrays = []
    try:
        for path in paths:
            arrays.append(np.load(path,mmap_mode="r",allow_pickle=False))
        return arrays
    except Exception:
        for array in arrays:
            close_mapping(array)
        raise


def window_ref(center,lag,window,timestamps,source_numbers):
    h = window//2
    total,count,low,high = 0.0,0,math.inf,-math.inf
    # Large averaging windows must not allocate W-sized temporary arrays.
    for a in range(center-h,center+h+1,8192):
        b = min(a+8192,center+h+1)
        differences = timestamps[a+lag:b+lag,0]-timestamps[a:b,0]
        finite = differences[np.isfinite(differences)]
        if finite.size:
            total += float(finite.sum())
            count += int(finite.size)
            low,high = min(low,float(finite.min())),max(high,float(finite.max()))
    number = int(source_numbers[center])
    return {"center_index": int(center),"center_time": clean_number(timestamps[center,0]),"source_number": None if number == np.iinfo(np.int64).min else number,"trigger_time": clean_number(timestamps[center,1]),"left_start": int(center-h),"left_end": int(center+h+1),"right_start": int(center-h+lag),"right_end": int(center+h+lag+1),"lag_frames": int(lag),"lag_seconds": total/window if count==window else None,"lag_seconds_min": low if count==window else None,"lag_seconds_max": high if count==window else None}


class ResultStore:
    """Disk-backed TQYX arrays; slices are copied before closing their mappings."""

    def __init__(self,path,shape,*,dtype="float32"):
        self.path = Path(path)
        self.g2 = np.lib.format.open_memmap(self.path/"g2.npy",mode="w+",dtype=dtype,shape=shape)
        try:
            self.count = np.lib.format.open_memmap(self.path/"counts.npy",mode="w+",dtype=np.uint32,shape=shape)
        except Exception:
            close_mapping(self.g2)
            raise

    def write(self,t,q,values,count):
        valid = np.isfinite(values)&(np.abs(values)<=np.finfo(self.g2.dtype).max)
        self.g2[t,q] = np.where(valid,values,np.nan)
        self.count[t,q] = count

    def close(self):
        close_mapping(self.g2)
        close_mapping(self.count)


def read_slice(directory,center_index,lag_index):
    path = Path(directory)
    run = json.loads((path/"run.json").read_text(encoding="utf-8"))
    if run.get("mode") != "split_correlation" or run.get("status") != "succeeded" or run.get("task") == "curve":
        raise ValueError("该任务没有完成的系数图")
    shape = tuple(run["array_shape"])
    if type(center_index) is not int or type(lag_index) is not int or not 0<=center_index<shape[0] or not 0<=lag_index<shape[1]:
        raise ValueError("结果时间 / 延迟索引超出范围")
    array,counts = load_mappings([path/"g2.npy",path/"counts.npy"])
    try:
        if array.shape != shape or counts.shape != shape or run["output_dtype"] not in {"float32","float64"} or array.dtype != np.dtype(run["output_dtype"]) or counts.dtype != np.uint32:
            raise ValueError("结果数组与运行记录不匹配")
        pixels,n = array[center_index,lag_index].copy(),counts[center_index,lag_index].copy()
    finally:
        close_mapping(array)
        close_mapping(counts)
    timestamps,numbers = load_mappings([path/"timestamps.npy",path/"source_numbers.npy"])
    try:
        center = run["centers"]["first"]+center_index*run["centers"]["stride"]
        ref = window_ref(center,run["lags"][lag_index],run["config"]["window"],timestamps,numbers)
    finally:
        close_mapping(timestamps)
        close_mapping(numbers)
    return pixels,n,ref,run


def write_curves(directory,curves,refs,config):
    from .workflow import output_semantics
    value = {"schema_version": 1,"config": config,"window_refs": refs,"curves": curves}
    value.update(output_semantics(config.get("workflow", [])))
    write_json(Path(directory)/"curves.json",value)
    columns = ["curve_id","name","estimator","point_x","point_y","center_index","center_time","source_number","left_start","left_end","right_start","right_end","lag_frames","lag_seconds","lag_seconds_min","lag_seconds_max","g2","valid_samples_min","valid_samples_max","pixel_count","result_semantics"]
    with (Path(directory)/"curves.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=columns)
        writer.writeheader()
        for curve in curves:
            for index,ref in enumerate(refs):
                writer.writerow({"curve_id":curve["id"],"name":curve["name"],"estimator":curve["estimator"],"point_x":curve.get("point",[None,None])[0],"point_y":curve.get("point",[None,None])[1],**{key:ref[key] for key in columns if key in ref},"g2":curve["g2"][index],"valid_samples_min":curve["counts_min"][index],"valid_samples_max":curve["counts_max"][index],"pixel_count":curve["pixel_count"],"result_semantics":value["result_semantics"]})
    # Scientific exports use a plotting library; neither browser smoothing nor display LUTs.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig,ax = plt.subplots(figsize=(8,4.6),layout="constrained")
    lags = [ref["lag_frames"] for ref in refs]
    for curve in curves:
        values = [np.nan if v is None else v for v in curve["g2"]]
        ax.plot(lags,values,label=curve["id"]+" · "+curve["estimator"],linewidth=1.5)
    ax.axvline(0,color="#8794a7",linestyle="--",linewidth=.8)
    if value["coefficient_reference"] is not None:
        ax.axhline(value["coefficient_reference"],color="#8794a7",linestyle="--",linewidth=.8)
    ax.set(xlabel="Signed frame lag q (right i+q)",ylabel=r"$g^{(2)}$",title=f"Center frame {refs[0]['center_index']+1} · window {config['window']}")
    if value["result_semantics"] == "processed_g2":
        ax.set_ylabel("Processed g2")
    ax.grid(alpha=.15)
    if curves:
        ax.legend(fontsize=8)
    fig.savefig(Path(directory)/"curves.png",dpi=180)
    fig.savefig(Path(directory)/"curves.svg")
    plt.close(fig)
    return value


class CorrelationReader:
    """A selected lag exposed as an ordinary YX sequence, retaining WindowRef lineage."""

    def __init__(self,source):
        self.source,self.path = source,Path(source.path)
        self.selection = json.loads(self.path.read_text(encoding="utf-8"))
        if self.selection.get("type") != "eip_correlation_lag" or self.selection.get("schema_version") != 1:
            raise ValueError("关联序列描述无效")
        self.directory = Path(self.selection["directory"])
        self.run = json.loads((self.directory/"run.json").read_text(encoding="utf-8"))
        if self.run.get("mode") != "split_correlation" or self.run.get("status") != "succeeded" or self.run.get("task") == "curve":
            raise ValueError("只有完成的系数图可作为下一阶段输入")
        self.lag_index = self.selection["lag_index"]
        shape = tuple(self.run["array_shape"])
        if type(self.lag_index) is not int or not 0<=self.lag_index<shape[1]:
            raise ValueError("延迟索引不在结果范围内")
        self.array,self.timestamps,self.numbers = load_mappings([self.directory/"g2.npy",self.directory/"timestamps.npy",self.directory/"source_numbers.npy"])
        if self.array.shape != shape or self.run["output_dtype"] not in {"float32","float64"} or self.array.dtype != np.dtype(self.run["output_dtype"]):
            self.close()
            raise ValueError("关联数据数组与记录不匹配")
        self.count = shape[0]
        self.centers = self.run["centers"]
        start = clean_number(self.timestamps[self.centers["first"],0])
        last_center = self.centers["first"]+(self.count-1)*self.centers["stride"]
        end = clean_number(self.timestamps[last_center,0])
        self.meta = DataSpec(f"g² · τ={self.run['lags'][self.lag_index]:+d} · {Path(self.run['source']['path']).name}","sequence",shape[2:],self.run["output_dtype"],"YX",self.count,end-start if start is not None and end is not None else None,None,"correlation_window_ref","gray",file_revision([self.path,self.directory/"run.json",self.directory/"g2.npy",self.directory/"timestamps.npy",self.directory/"source_numbers.npy"]),("选定延迟的 g² 时间序列；窗口 / 配对 /空间映射见关联运行记录",),frame_indexed=True)

    def read_frame(self,index):
        if type(index) is not int or not 0<=index<self.count:
            raise ValueError("关联序列帧索引超出范围")
        center = self.centers["first"]+index*self.centers["stride"]
        ref = window_ref(center,self.run["lags"][self.lag_index],self.run["config"]["window"],self.timestamps,self.numbers)
        pixels = self.array[index,self.lag_index].copy()
        pixels.setflags(write=False)
        return FrameData(pixels,FrameRef(index,ref["center_time"],source_number=ref["source_number"],trigger_time=ref["trigger_time"]))

    def iter_frames(self,selection: Selection):
        if selection.mode == "current":
            yield self.read_frame(selection.current.index)
        else:
            a,b = (0,self.count) if selection.mode=="all" else (int(selection.start),min(self.count,int(selection.end)))
            for index in range(a,b):
                yield self.read_frame(index)

    def close(self):
        for name in ("array","timestamps","numbers"):
            if hasattr(self,name):
                close_mapping(getattr(self,name))
