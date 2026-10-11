"""Lazy, bounded, axis-aware execution, shared by both algorithm presets.

No display adjustment enters this engine. A sample may be an image, an aligned
pair, a scalar, or a colour vector. It is never disguised as a 1 x 1 image.
"""
from __future__ import annotations

from collections import OrderedDict
from bisect import bisect_right
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import json
import hashlib
import shutil
import struct
import warnings
import numpy as np

from eip.algorithms import BuiltinAlgorithm, Pipeline, RunContext
from eip.domain import Selection, StepSpec
from eip.module_contracts import compile_flow, source_contract, roi_bounds, correlation_layout
from eip.module_engine import IdentityPhotometry
from eip.module_ops import intensity_curve
from eip.module_registry import normalize_nodes


def reduce_finite(a, axes, mode, q=.5):
    a = np.asarray(a,dtype=np.float64)
    valid = np.isfinite(a)
    count = valid.sum(axis=axes)
    if mode in {"mean","sum"}:
        total = np.where(valid,a,0).sum(axis=axes)
        value = total/np.maximum(count,1) if mode=="mean" else total
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            value = np.nanquantile(np.where(valid,a,np.nan),q,axis=axes,method="linear")
    return np.where(count,value,np.nan),count


class Cache:
    def __init__(self,budget):
        self.budget,self.size = budget,0
        self.entries = OrderedDict()

    def get(self,key):
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]

    def put(self,key,value):
        value = np.asarray(value)
        value.setflags(write=False)
        size = value.nbytes
        while self.entries and self.size+size>self.budget:
            _,old = self.entries.popitem(last=False)
            self.size -= old.nbytes
        if size<=self.budget:
            self.entries[key] = value
            self.size += size
        return value


class IndexRanges:
    """Compact ordered selection, including nonmonotonic timestamp selections."""
    def __init__(self):
        self.ranges,self.starts,self.length = [],[],0

    def add(self,index):
        if self.ranges and self.ranges[-1][1]==index:
            self.ranges[-1][1] += 1
        else:
            if len(self.ranges)>=4096:
                raise ValueError("时间选择超过 4096 个不连续区间，请检查时间戳或按样本裁剪")
            self.ranges.append([index,index+1])
            self.starts.append(self.length)
        self.length += 1

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        if not 0<=index<self.length:
            raise IndexError(index)
        segment = bisect_right(self.starts,index)-1
        return self.ranges[segment][0]+index-self.starts[segment]


class Source:
    def __init__(self,reader,context,workdir,input_kind="image"):
        self.reader,self.context = reader,context
        self.contract = source_contract(asdict(reader.meta),input_kind)
        self.spooled = None
        if not hasattr(reader,"read_frame"):
            # Decode ordinary video once in the compute worker. The exact count
            # and PTS come from decoding, never from average FPS. RAM stays bounded.
            folder = Path(workdir)/"decoded"
            folder.mkdir(parents=True,exist_ok=False)
            self.spooled = folder
            self.ref_cache = OrderedDict()
            count = 0
            with (folder/"references.jsonl").open("wb") as refs,(folder/"offsets.bin").open("wb") as offsets:
                for i,frame in enumerate(reader.iter_frames(Selection())):
                    context.check_cancelled()
                    if shutil.disk_usage(folder).free < frame.pixels.nbytes*2+16*1024**2:
                        raise ValueError("时间编辑需要临时视频索引，输出盘剩余空间不足")
                    np.save(folder/f"{i}.npy",frame.pixels,allow_pickle=False)
                    offsets.write(struct.pack("<Q",refs.tell()))
                    refs.write((json.dumps(asdict(frame.ref),allow_nan=False)+"\n").encode())
                    count += 1
                    if i%100==0:
                        context.log(f"正在建立真实视频样本索引：{i+1} 帧")
            if not count:
                raise ValueError("来源没有可读取样本")
            self.contract = replace(self.contract,length=count)

    def get(self,i):
        self.context.check_cancelled()
        if self.spooled:
            return np.load(self.spooled/f"{i}.npy",allow_pickle=False)
        return self.reader.read_sample(i) if hasattr(self.reader,"read_sample") else self.reader.read_frame(i).pixels

    def ref(self,i):
        r = self.reader
        if self.spooled:
            if i not in self.ref_cache:
                with (self.spooled/"offsets.bin").open("rb") as index,(self.spooled/"references.jsonl").open("rb") as refs:
                    index.seek(i*8)
                    refs.seek(struct.unpack("<Q",index.read(8))[0])
                    self.ref_cache[i] = json.loads(refs.readline())
                if len(self.ref_cache)>256:
                    self.ref_cache.popitem(last=False)
            self.ref_cache.move_to_end(i)
            return dict(self.ref_cache[i])
        if hasattr(r,"sample_ref"):
            return r.sample_ref(i)
        if hasattr(r,"times"):
            from eip.cine_reader import TICKS_PER_SECOND
            ticks = int(r.ticks[i]) if r.ticks is not None else None
            return dict(index=i,time=float(r.times[i]) if r.times is not None else None,pts=ticks-int(r.ticks[0]) if ticks is not None else None,source_number=r.first_number+i,trigger_time=(ticks-r.trigger_ticks)/TICKS_PER_SECOND if ticks is not None else None)
        if hasattr(r,"timestamps"):
            return dict(index=i,time=r.timestamps[i] if r.timestamps is not None else i*r.source.interval if r.source.interval else None,pts=None,source_number=r.source_numbers[i] if r.source_numbers else None,trigger_time=r.trigger_times[i] if r.trigger_times else None)
        return asdict(r.read_frame(i).ref)


class SequenceEngine:
    def __init__(self,reader,nodes,context=None,*,workdir,center=None,single=False,input_kind="image",dependencies=()):
        self.context = context or RunContext()
        self.source = Source(reader,self.context,workdir,input_kind)
        self.center,self.single = center,single
        self.dependencies = normalize_nodes(dependencies,check_types=False)[0]
        self.plan = compile_flow(nodes,self.source.contract,center=center,single=single,following_nodes=self.dependencies)
        self.nodes = [n for n in self.plan["nodes"] if n["enabled"]]
        self.contracts = [self.source.contract]
        self.operations,self.instances,self.statistics = [],{},{}
        self.budget = max([n.get("execution",{}).get("memory_mb",256) for n in self.nodes] or [256])*1024**2
        self.cache = Cache(min([n.get("execution",{}).get("memory_mb",256) for n in self.nodes] or [256])*1024**2//3)
        self.mappings,self.backends = {},{}
        self.last_correlation = {}
        retained = 0
        try:
            for i,n in enumerate(self.nodes):
                a,p = n["algorithm"],n["parameters"]
                c = self.contracts[-1]
                op = dict(algorithm=a,parameters=p,node=n)
                if a=="time_crop" and p["unit"]=="seconds":
                    indices = IndexRanges()
                    for j in range(c.length):
                        self.context.check_cancelled()
                        t = self.ref(i,j).get("time")
                        if t is None:
                            raise ValueError("来源缺少真实时间，时间裁剪请改用样本号")
                        if t>=p["start"] and (p["end"] is None or t<p["end"]):
                            indices.add(j)
                    if not indices:
                        raise ValueError("时间裁剪后没有样本")
                    op["indices"] = indices
                single_here = single and not any(m["algorithm"] in {"time_sample","time_smooth","time_bin"} for m in self.nodes[i+1:])
                stage = compile_flow([n],c,center=center,single=single_here,following_nodes=[*self.nodes[i+1:],*self.dependencies])["output"]
                after = type(c)(**{k:tuple(stage[k]) if k in {"shape","origin"} else stage[k] for k in type(c).__dataclass_fields__})
                if "indices" in op:
                    after = replace(after,length=len(op["indices"]))
                    self.mappings[n["id"]] = dict(upstream_ranges=op["indices"].ranges,range_convention="zero_based_half_open")
                if a=="alignment":
                    from eip.modes.split_correlation.config import GeometryConfig
                    from eip.modes.split_correlation.geometry import Geometry
                    self.instances[i] = Geometry(GeometryConfig.from_dict(p),c.shape)
                    self.mappings[n["id"]] = self.instances[i].record
                elif a=="calibration":
                    from eip.modes.split_correlation.config import PhotometryConfig
                    from eip.modes.split_correlation.photometry import Photometry
                    self.instances[i] = Photometry(PhotometryConfig.from_dict(dict(p,normalization="none")))
                elif a=="correlation":
                    from eip.modes.split_correlation.backend import Backend
                    lags,centers,_,_ = correlation_layout(c,p,center,single_here)
                    op.update(lags=lags,centers=centers)
                    self.backends[i] = Backend(n.get("execution",{}).get("backend","cpu"),n.get("execution",{}).get("memory_mb",256)*1024**2)
                    self.mappings[n["id"]] = dict(lag_units="upstream_samples",window_units="upstream_samples",lags=lags,centers=dict(start=centers.start,stop=centers.stop,step=centers.step))
                elif n.get("manifest"):
                    self.instances[i] = Pipeline([StepSpec(**n)],self.context)
                elif a in {"intensity","affine","box_blur","threshold","gaussian","curve"}:
                    self.instances[i] = BuiltinAlgorithm(a)
                    self.instances[i].prepare(p,self.context)
                self.operations.append(op)
                self.contracts.append(after)
                # Temporary arrays as well as caches must fit the chosen budget.
                elements = int(np.prod(c.shape,dtype=np.int64))
                multiplier = 1 if a in {"time_crop","time_sample","select_channel"} else 3 if a in {"affine","intensity","curve","threshold","spatial_roi"} else 12 if a=="correlation" else p.get("window",1)+4 if a=="time_smooth" and p["mode"]=="quantile" else 8
                working = elements*8*multiplier
                if working>n.get("execution",{}).get("memory_mb",256)*1024**2*2/3:
                    raise ValueError(f"{a} 工作数组超过该流程内存预算；请先裁剪 ROI、减小窗口或增加执行内存")
                if a=="statistics":
                    retained += elements*20
                if retained+working+self.cache.budget>self.budget:
                    raise ValueError("重复统计保留数组与工作数组合计超过流程预算；请减少统计节点、先裁剪或提高内存预算")
            self.plan = self._actual_plan()
        except Exception:
            self.close()
            raise

    def _actual_plan(self):
        # Include disabled cards in provenance but only active operations in shape chain.
        stages,active = [],0
        upstream = []
        for n in self.plan["nodes"]:
            before = self.contracts[active]
            if n["enabled"]:
                active += 1
                upstream.append(n)
            fingerprint = hashlib.sha256(json.dumps(dict(source_revision=self.source.reader.meta.revision,contract_version=3,upstream=upstream,preview_dependencies=self.dependencies),sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
            stages.append(dict(id=n["id"],algorithm=n["algorithm"],enabled=n["enabled"],input=before.record(),output=self.contracts[active].record(),upstream_fingerprint=fingerprint))
        return dict(self.plan,stages=stages,output=self.contracts[-1].record(),preview_dependencies=list(self.dependencies))

    @property
    def output(self):
        return self.contracts[-1]

    def get(self,stage,index):
        self.context.check_cancelled()
        c = self.contracts[stage]
        if not 0<=index<c.length:
            raise IndexError("编辑后的样本号超出范围")
        key = stage,index
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if stage==0:
            value = self.source.get(index)
        else:
            i = stage-1
            op = self.operations[i]
            a,p,n = op["algorithm"],op["parameters"],op["node"]
            before = self.contracts[i]
            if a=="time_crop":
                value = self.get(i,op["indices"][index] if "indices" in op else index+p["first"]-1)
            elif a=="time_sample":
                value = self.get(i,p["offset"]+index*p["step"])
            elif a in {"time_smooth","time_bin"}:
                size = p["window"] if a=="time_smooth" else p["size"]
                first = index if a=="time_smooth" else index*size
                if p["mode"]=="quantile":
                    value,_ = reduce_finite(np.stack([self.get(i,j) for j in range(first,first+size)]),0,"quantile",p["quantile"])
                else:
                    total,count = np.zeros(before.shape),np.zeros(before.shape,dtype=np.uint32)
                    for j in range(first,first+size):
                        arr = self.get(i,j)
                        valid = np.isfinite(arr)
                        total += np.where(valid,arr,0)
                        count += valid
                    value = np.where(count,total/np.maximum(count,1) if p["mode"]=="mean" else total,np.nan)
            elif a=="statistics":
                stats = self.fit(i)
                value = self.get(i,index)/stats["scales"]
            elif a=="correlation":
                value = self.correlate(i,index)[0]
            else:
                value = self.get(i,index)
                if a=="spatial_roi":
                    x,y,w,h = roi_bounds(before,p)
                    slices = [slice(None)]*len(before.shape)
                    slices[before.axes.index("Y")],slices[before.axes.index("X")] = slice(y,y+h),slice(x,x+w)
                    value = value[tuple(slices)].copy()
                    if p["mode"]!="pixels":
                        if before.paired and p["validity"]=="joint":
                            value = np.where(np.isfinite(value).all(axis=0)[None,...],value,np.nan)
                        value,_ = reduce_finite(value,tuple(before.axes.index(axis) for axis in "YX"),p["mode"],p["quantile"])
                elif a=="alignment":
                    value = np.stack(self.instances[i].apply(value,IdentityPhotometry()))
                elif a=="select_channel":
                    value = value[0 if p["channel"]=="left" else 1]
                elif a=="calibration":
                    photo = self.instances[i]
                    if before.paired:
                        value = np.stack([photo.apply(arr,side) for arr,side in zip(value,("left","right"))])
                    elif p["black_left"]!=p["black_right"]:
                        # Split offsets require an explicit downstream geometry in
                        # these exact input coordinates; no hidden middle split.
                        future = [*self.nodes[i+1:],*[n for n in self.dependencies if n["enabled"]]]
                        following = next((o for o in future if o["algorithm"]=="alignment"),None)
                        if following is None or any(o["algorithm"]=="spatial_roi" for o in future[:future.index(following)]):
                            raise ValueError("不同左右暗场需紧邻后续配准，或把校准移到配准后")
                        line = np.interp(np.arange(value.shape[0])[:,None],[0,max(1,value.shape[0]-1)],following["parameters"]["midline"])
                        value = np.where(np.arange(value.shape[1])[None,:]<line,photo.apply(value,"left"),photo.apply(value,"right"))
                    else:
                        value = photo.apply(value,"left")
                else:
                    def apply(arr):
                        # Scalar/vector element operations use their true rank.
                        if a=="threshold" and not before.spatial and "C" in before.axes:
                            return np.asarray(np.mean(arr[..., :3],axis=-1)>=p["threshold"],dtype=np.uint8)
                        return self.instances[i].execute(arr) if n.get("manifest") else self.instances[i].process(arr,self.context)
                    value = np.stack([apply(arr) if n["target"] in {"both",side} else arr for arr,side in zip(value,("left","right"))]) if before.paired else apply(value)
            precision = n.get("execution",{}).get("precision","native")
            if precision!="native":
                value = np.asarray(value,dtype=precision)
        value = np.asarray(value)
        if value.shape!=c.shape:
            raise ValueError(f"算法实际输出 {value.shape} 与声明形状 {c.shape} 不符")
        if stage and self.nodes[stage-1].get("manifest"):
            if c.dtype=="dynamic":
                self.contracts[stage] = replace(c,dtype=str(value.dtype))
            elif value.dtype!=np.dtype(c.dtype):
                raise ValueError(f"插件输出 dtype 从 {c.dtype} 变为 {value.dtype}；请保持一致或显式设置数值精度")
        return self.cache.put(key,value)

    def fit(self,i):
        if i in self.statistics:
            return self.statistics[i]
        c,p = self.contracts[i],self.operations[i]["parameters"]
        mean,m2,count = np.zeros(c.shape),np.zeros(c.shape),np.zeros(c.shape,dtype=np.uint32)
        for j in range(c.length):
            arr = self.get(i,j)
            finite = np.isfinite(arr)
            count += finite
            delta = np.where(finite,arr-mean,0)
            mean += delta/np.maximum(count,1)
            m2 += delta*np.where(finite,arr-mean,0)
            if j%100==0:
                self.context.log(f"统计当前上游序列：{j+1}/{c.length} 样本")
        variance = np.where(count,m2/np.maximum(count,1),np.nan)
        mean = np.where(count,mean,np.nan)
        scales = np.ones((2,)+tuple(1 for _ in c.shape[1:])) if c.paired else np.asarray(1.)
        if p["normalization"]=="fixed_rms":
            axes = tuple(range(1,len(c.shape))) if c.paired else tuple(range(len(c.shape)))
            scales = np.sqrt(m2.sum(axis=axes,keepdims=True)/np.maximum(count.sum(axis=axes,keepdims=True),1))
            scales = np.where(np.isfinite(scales)&(scales>1e-12),scales,1.)
        self.statistics[i] = dict(mean=mean,variance=variance,count=count,scales=scales,frames=c.length)
        return self.statistics[i]

    def correlate(self,i,index):
        from eip.modes.split_correlation.correlation import RollingMoments
        op = self.operations[i]
        p = op["parameters"]
        center,lag = (op["centers"][0],op["lags"][index]) if p["analysis"]=="curve" else (op["centers"][index],op["lags"][0])
        # Use a nearby positive-intensity baseline for stable fluctuations when
        # no statistics node is present. It is a numerical identity, not a filter.
        baseline = np.nan_to_num(self.get(i,center),nan=0.,posinf=0.,neginf=0.)
        previous = self.last_correlation.get(i)
        can_roll = previous and previous["lag"]==lag and 0<center-previous["center"]<p["window"] and index%256!=0
        if can_roll:
            moments = previous["moments"]
            old = previous["center"]
            for j in range(old-p["window"]//2,center-p["window"]//2):
                moments.add(self.get(i,j)[0],self.get(i,j+lag)[1],-1)
            indices = range(old+p["window"]//2+1,center+p["window"]//2+1)
        else:
            moments = RollingMoments(baseline,SimpleNamespace(**p),self.backends[i])
            indices = range(center-p["window"]//2,center+p["window"]//2+1)
        for j in indices:
            moments.add(self.get(i,j)[0],self.get(i,j+lag)[1])
        self.last_correlation[i] = dict(lag=lag,center=center,moments=moments)
        return moments.result()

    def ref(self,stage,index):
        self.context.check_cancelled()
        if stage==0:
            original = self.source.ref(index)
            return dict(original,sample_index=index,source_sample=index,source_span=original.get("source_span",[index,index+1]))
        i = stage-1
        op = self.operations[i]
        a,p,n = op["algorithm"],op["parameters"],op["node"]
        if a=="time_crop":
            r = self.ref(i,op["indices"][index] if "indices" in op else index+p["first"]-1)
        elif a=="time_sample":
            r = self.ref(i,p["offset"]+index*p["step"])
        elif a in {"time_smooth","time_bin"}:
            size = p["window"] if a=="time_smooth" else p["size"]
            start = index if a=="time_smooth" else index*size
            if a=="time_smooth":
                r = self.ref(i,start+size//2)
            else:
                r = dict(index=None,pts=None,source_number=None,time=None,trigger_time=None)
                totals,known = dict(time=0.,trigger_time=0.),dict(time=0,trigger_time=0)
                for j in range(start,start+size):
                    ref = self.ref(i,j)
                    for key in totals:
                        if ref.get(key) is not None:
                            totals[key] += ref[key]
                            known[key] += 1
                for key in totals:
                    r[key] = totals[key]/size if known[key]==size else None
            first,last = self.ref(i,start),self.ref(i,start+size-1)
            spans = [v.get("source_span",[v.get("source_sample",0),v.get("source_sample",0)+1]) for v in (first,last)]
            r = dict(r,source_span=[min(v[0] for v in spans),max(v[1] for v in spans)],contribution=dict(module_id=n["id"],upstream_start=start,upstream_stop=start+size,operation=a,source_span_is_envelope=True))
        elif a=="correlation":
            center,lag = (op["centers"][0],op["lags"][index]) if p["analysis"]=="curve" else (op["centers"][index],op["lags"][0])
            r = self.ref(i,center)
            half = p["window"]//2
            total,minimum,maximum,known = 0.,None,None,0
            for j in range(center-half,center+half+1):
                left,right = self.ref(i,j).get("time"),self.ref(i,j+lag).get("time")
                if left is not None and right is not None:
                    delta = right-left
                    total += delta
                    known += 1
                    minimum = delta if minimum is None else min(minimum,delta)
                    maximum = delta if maximum is None else max(maximum,delta)
            r = dict(r,correlation=dict(module_id=n["id"],center_sample=center,lag_samples=lag,window=p["window"],left_start=center-half,left_stop=center+half+1,right_start=center-half+lag,right_stop=center+half+1+lag,lag_seconds=total/p["window"] if known==p["window"] else None,lag_seconds_min=minimum,lag_seconds_max=maximum,known_pairs=known))
        else:
            r = self.ref(i,index)
        return dict(r,sample_index=index)

    def close(self):
        for instance in self.instances.values():
            if hasattr(instance,"close"):
                instance.close()
        for backend in self.backends.values():
            backend.close()
        self.instances.clear()
        self.backends.clear()
