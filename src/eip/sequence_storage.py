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
    points = [(int(65+i*510/max(1,len(values)-1)),int(260-180*(float(v)-low)/(high-low))) for i,v in enumerate(values) if np.isfinite(v)]
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
        if "Y" not in axes:
            value = signal_preview(value)
        ref = self.sample_ref(index)
        # The source ordinal is exact for the saved sequence, while the original
        # acquisition index stays in sample_ref and its contribution record.
        return FrameData(value,FrameRef(index,ref.get("time"),ref.get("pts"),ref.get("source_number"),ref.get("trigger_time")))

    def read_at_time(self,time):
        if self.contract["sequence_axis"]!="T":
            raise ValueError("此结果的样本轴是 τ，不能按采集时间定位")
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


def plot_signals(directory,contract):
    """Bounded static graphic; NPY and CSV always retain every numerical sample."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    directory = Path(directory)
    values = np.load(directory/"values.npy",mmap_mode="r",allow_pickle=False)
    indices = np.unique(np.r_[np.arange(0,len(values),max(1,int(np.ceil(len(values)/20000)))),len(values)-1])
    chosen = set(indices.tolist())
    xs,actual = [],True
    with (directory/"references.jsonl").open(encoding="utf-8") as file:
        for i,line in enumerate(file):
            if i not in chosen:
                continue
            ref = json.loads(line)
            x = ref.get("correlation",{}).get("lag_samples") if contract["sequence_axis"]=="Q" else ref.get("time")
            if x is None:
                actual = False
            xs.append(x)
    if not actual:
        xs = (indices+1).tolist()
    fig,ax = plt.subplots(figsize=(9,4.8),layout="constrained")
    columns = values[indices].reshape(len(indices),-1)
    for i in range(columns.shape[1]):
        ax.plot(xs,columns[:,i],label=f"channel {i+1}",linewidth=1.2)
    ax.set_xlabel("Signed lag (upstream samples)" if contract["sequence_axis"]=="Q" else "Actual time (s)" if actual else "Edited sample")
    ax.set_ylabel(contract["semantic"])
    ax.grid(alpha=.2)
    if len(indices)<len(values):
        ax.set_title(f"Plot: {len(indices)} / {len(values)} samples; full data in CSV / NPY",fontsize=10)
    if columns.shape[1]>1:
        ax.legend()
    for ext in ("png","svg"):
        fig.savefig(directory/f"curve.{ext}",dpi=160)
    plt.close(fig)
