"""Saved typed pipeline output, real sample references and signal rendering."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

from eip.domain import DataSpec, FrameData, FrameRef, Selection, file_revision


def signal_preview(values):
    """A labelled display graphic; it never changes the numerical data contract."""
    values = np.asarray(values).reshape(-1)
    image = Image.new("RGB",(640,320),(18,25,35))
    draw = ImageDraw.Draw(image)
    draw.text((25,20),"Signal sample (numerical shape retained)",fill=(190,205,220))
    valid = values[np.isfinite(values)]
    low,high = (float(valid.min()),float(valid.max())) if valid.size else (0.,1.)
    if low==high:
        low,high = low-1,high+1
    selected = np.unique(np.linspace(0,max(0,len(values)-1),min(len(values),2048),dtype=int))
    points = [(int(65+i*510/max(1,len(values)-1)),int(260-180*(float(values[i])-low)/(high-low))) for i in selected if np.isfinite(values[i])]
    if len(points)>1:
        draw.line(points,fill=(96,210,209),width=3)
    for x,y in points:
        draw.ellipse((x-4,y-4,x+4,y+4),fill=(96,210,209))
    for i,v in enumerate(values[:8]):
        draw.text((30,280-i*16),f"{i+1}: {float(v):.8g}",fill=(220,230,240))
    return np.asarray(image)


class DatasetReader:
    def __init__(self,source):
        self.source = source
        self.path = Path(source.path)
        self.directory = self.path.parent
        manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if manifest.get("schema_version")!=1 or manifest.get("status")!="succeeded":
            raise ValueError("仅完整保存的类型化数据可作为来源")
        self.contract = manifest["contract"]
        self.values = np.load(self.directory/"values.npy",mmap_mode="r",allow_pickle=False)
        self.offsets = np.load(self.directory/"reference_offsets.npy",mmap_mode="r",allow_pickle=False)
        self.count = self.contract["length"]
        if tuple(self.values.shape)!=(self.count,*self.contract["shape"]) or len(self.offsets)!=self.count or str(self.values.dtype)!=self.contract["dtype"]:
            raise ValueError("数据文件与形状 / 类型契约不匹配")
        self.reference_file = (self.directory/"references.jsonl").open("rb")
        first = self.read_frame(0).pixels
        files = [self.path,self.directory/"values.npy",self.directory/"reference_offsets.npy",self.directory/"references.jsonl"]
        self.meta = DataSpec(self.directory.name,"dataset",first.shape,str(first.dtype),"YX" if first.ndim==2 else "YXS",self.count,None,None,"module_record","gray" if first.ndim==2 else "RGB/RGBA",file_revision(files),("显示图与数值契约分开；来源坐标和时间见模块运行记录",),True,contract=self.contract)

    def read_sample(self,index):
        if not 0<=index<self.count:
            raise IndexError("样本号超出范围")
        return self.values[index]

    def sample_ref(self,index):
        if not 0<=index<self.count:
            raise IndexError("样本号超出范围")
        self.reference_file.seek(int(self.offsets[index]))
        return json.loads(self.reference_file.readline())

    def read_frame(self,index):
        value = self.read_sample(index)
        axes = self.contract["axes"]
        if axes.startswith("P"):
            value,axes = value[0],axes[1:]
        if not ("Y" in axes and "X" in axes):
            value = signal_preview(value)
        ref = self.sample_ref(index)
        # The source ordinal is exact for the saved sequence, while the original
        # acquisition index stays in sample_ref and its contribution record.
        return FrameData(value,FrameRef(index,ref.get("time"),ref.get("pts"),ref.get("source_number"),ref.get("trigger_time")))

    def read_at_time(self,time):
        if self.contract["sequence_axis"]!="T":
            axis = self.contract["sequence_axis"]
            label = "Q（延迟）" if axis=="Q" else f"{axis}（记录）"
            raise ValueError(f"此结果的样本轴是 {label}，不能按采集时间定位")
        best = None
        for i in range(self.count):
            t = self.sample_ref(i).get("time")
            if t is not None and (best is None or abs(t-time)<best[0]):
                best = abs(t-time),i
        if best is None:
            raise ValueError("此结果没有可用的真实时间")
        return self.read_frame(best[1])

    def iter_frames(self,selection):
        for i in range(self.count):
            yield self.read_frame(i)

    def close(self):
        if getattr(self,"reference_file",None):
            self.reference_file.close()
        self.values = self.offsets = None


def plot_signals(directory,contract,mappings=None,node_id=None):
    """Bounded static graphic; NPY and CSV always retain every numerical sample."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    directory = Path(directory)
    values = np.load(directory/"values.npy",mmap_mode="r",allow_pickle=False)
    count = max(1,int(np.prod(contract["shape"])))
    if contract["semantic"]=="distribution" and contract["axes"]=="B":
        candidates = [(key,value) for key,value in (mappings or {}).items()
                      if value.get("axis")=="B" and len(value.get("edges",[]))==count+1]
        mapping = next((value for key,value in candidates if key==node_id),candidates[-1][1] if candidates else {})
        edges = np.asarray(mapping.get("edges",np.arange(count+1)))
        fig,ax = plt.subplots(figsize=(9,4.8),layout="constrained")
        ax.bar(edges[:-1],values[0].reshape(-1),width=np.diff(edges),align="edge",color="#278b78")
        ax.set_xlim(edges[0],edges[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Intensity" if "edges" in mapping else "Bin index (intensity boundaries unavailable)")
        ax.set_ylabel("Frequency")
        ax.yaxis.get_major_locator().set_params(integer=True)
        with (directory/"references.jsonl").open(encoding="utf-8") as file:
            reference = json.loads(file.readline())
        sample = reference.get("sample_index",0)+1
        ax.set_title(f"Histogram: sample {sample}; {len(values)} stored samples in NPY / CSV",fontsize=10)
        ax.grid(axis="y",alpha=.2)
        for ext in ("png","svg"):
            fig.savefig(directory/f"curve.{ext}",dpi=160)
        plt.close(fig)
        return
    selected_columns = np.unique(np.linspace(0,count-1,min(count,64),dtype=int))
    budget = min(20000,max(1,200000//len(selected_columns)))
    indices = np.unique(np.r_[np.arange(0,len(values),max(1,int(np.ceil(len(values)/budget)))),len(values)-1])
    chosen = set(indices.tolist())
    xs,actual = [],True
    with (directory/"references.jsonl").open(encoding="utf-8") as file:
        for i,line in enumerate(file):
            if i not in chosen:
                continue
            ref = json.loads(line)
            axis = contract["sequence_axis"]
            x = ref.get("correlation",{}).get("lag_samples") if axis=="Q" else ref.get("time") if axis=="T" else i
            if x is None:
                actual = False
            xs.append(x)
    if not actual:
        xs = (indices+1).tolist()
    fig,ax = plt.subplots(figsize=(9,4.8),layout="constrained")
    columns = np.array([values[i].reshape(-1)[selected_columns] for i in indices])
    if contract["sequence_axis"]=="N" and len(values)==1 and count>1:
        positions = np.unique(np.linspace(0,count-1,min(count,20000),dtype=int))
        ax.plot(positions,values[0].reshape(-1)[positions],linewidth=1.2)
        ax.set_xlabel(f"Component index (axes {contract['axes']})")
    else:
        for i,component in enumerate(selected_columns):
            ax.plot(xs,columns[:,i],label=f"component {component}",linewidth=1.2)
        ax.set_xlabel("Signed lag (upstream samples)" if contract["sequence_axis"]=="Q" else "Actual time (s)" if contract["sequence_axis"]=="T" and actual else "Edited sample" if contract["sequence_axis"]=="T" else f"Record ({contract['sequence_axis']})")
    ax.set_ylabel(contract["semantic"])
    ax.grid(alpha=.2)
    if len(indices)<len(values) or len(selected_columns)<count:
        ax.set_title(f"Plot: {len(indices)} / {len(values)} samples, {len(selected_columns)} / {count} components; full data in NPY",fontsize=10)
    if 1<columns.shape[1]<=16 and contract["sequence_axis"]!="N":
        ax.legend()
    for ext in ("png","svg"):
        fig.savefig(directory/f"curve.{ext}",dpi=160)
    plt.close(fig)
