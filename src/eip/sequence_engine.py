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
from contextlib import nullcontext
import numpy as np

from eip.algorithms import BuiltinAlgorithm, Pipeline, RunContext, preserve_dtype
from eip.domain import Selection, StepSpec
from eip.module_contracts import compile_flow, source_contract, roi_bounds, correlation_layout, Contract
from eip.module_engine import IdentityPhotometry
from eip.module_ops import intensity_curve
from eip.module_registry import normalize_nodes, module_spec, CORRELATIONS, MODIFIERS, is_reference, needs_full_sequence
from eip.module_statistics import reduce_finite, retained_axes, joint_values, global_statistic
from eip.resources import ResourceSettings, configure_reader, MIB
from eip.gpu_ops import GPU_MODULES, element_operation


class Cache:
    def __init__(self,budget):
        self.budget,self.size,self.peak = max(0,int(budget)),0,0
        self.entries = OrderedDict()

    def get(self,key):
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]

    def put(self,key,value):
        value = np.asarray(value)
        size = value.nbytes
        old = self.entries.pop(key,None)
        if old is not None:
            self.size -= old.nbytes
        if size and size<=self.budget:
            # A small ROI/point may own an entire frame or upload-sized block.
            # Account the actual cache allocation by detaching that small view.
            owner = value
            while isinstance(getattr(owner,"base",None),np.ndarray):
                owner = owner.base
            retained = owner.nbytes
            backing = getattr(owner,"base",None)
            while isinstance(backing,memoryview):
                backing = backing.obj
            if backing is not None:
                try:
                    retained = max(retained,memoryview(backing).nbytes)
                except TypeError:
                    pass
            if retained>size:
                value = value.copy()
            while self.entries and self.size+size>self.budget:
                self.size -= self.entries.popitem(last=False)[1].nbytes
            self.entries[key] = value
            self.size += size
            self.peak = max(self.peak,self.size)
        value.setflags(write=False)
        return value

    def clear(self):
        self.entries.clear()
        self.size = 0

    def resize(self,budget):
        self.budget = max(0,int(budget))
        while self.entries and self.size>self.budget:
            self.size -= self.entries.popitem(last=False)[1].nbytes

    def discard_stages(self,stages):
        """Release cache references after their last numeric consumer finished."""
        stages = set(stages)
        removed = 0
        for key in list(self.entries):
            if isinstance(key,tuple) and key and key[0] in stages:
                removed += self.entries.pop(key).nbytes
        self.size -= removed
        return removed


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
    def __init__(self,reader,context,workdir,input_kind="image",profiler=None):
        self.reader,self.context = reader,context
        self.profiler = profiler
        self.contract = source_contract(asdict(reader.meta),input_kind)
        self.spooled = None
        self.ref_cache = OrderedDict()
        if not hasattr(reader,"read_frame"):
            # Decode ordinary video once in the compute worker. The exact count
            # and PTS come from decoding, never from average FPS. RAM stays bounded.
            folder = Path(workdir)/"decoded"
            folder.mkdir(parents=True,exist_ok=False)
            self.spooled = folder
            count = 0
            with (folder/"references.jsonl").open("wb") as refs,(folder/"offsets.bin").open("wb") as offsets:
                iterator = iter(reader.iter_frames(Selection()))
                while True:
                    context.check_cancelled()
                    with profiler.measure("input_read") if profiler else nullcontext():
                        try:
                            frame = next(iterator)
                        except StopIteration:
                            break
                    i = count
                    if profiler:
                        profiler.add_input(frame.pixels.nbytes)
                    if shutil.disk_usage(folder).free < frame.pixels.nbytes*2+16*1024**2:
                        raise ValueError("时间编辑需要临时视频索引，输出盘剩余空间不足")
                    with profiler.measure("index_write") if profiler else nullcontext():
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
        if self.profiler is None:
            return self._get(i)
        with self.profiler.measure("input_read"):
            value = self._get(i)
        self.profiler.add_input(value.nbytes)
        return value

    def _get(self,i):
        self.context.check_cancelled()
        if self.spooled:
            return np.load(self.spooled/f"{i}.npy",allow_pickle=False)
        if hasattr(self.reader,"read_sample"):
            return self.reader.read_sample(i)
        frame = self.reader.read_frame(i)
        self.ref_cache[i] = asdict(frame.ref)
        self.ref_cache.move_to_end(i)
        if len(self.ref_cache)>256:
            self.ref_cache.popitem(last=False)
        return frame.pixels

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
        if i in self.ref_cache:
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
        with self.profiler.measure("input_read") if self.profiler else nullcontext():
            frame = r.read_frame(i)
        if self.profiler:
            self.profiler.add_input(frame.pixels.nbytes)
        self.ref_cache[i] = asdict(frame.ref)
        if len(self.ref_cache)>256:
            self.ref_cache.popitem(last=False)
        return dict(self.ref_cache[i])


class SequenceEngine:
    def __init__(self,reader,nodes,context=None,*,workdir,center=None,single=False,input_kind="image",dependencies=(),range_cache=None,resources=None,inspection=False,profiler=None,optimize=True):
        self.context = context or RunContext()
        self.profiler = profiler
        self.optimize = bool(optimize)
        self.spatial = None
        self._output_runners = []
        self.workdir = Path(workdir)
        self.global_statistics = {}
        self.source = Source(reader,self.context,workdir,input_kind,profiler=profiler)
        self.center,self.single = center,single
        self.dependencies = normalize_nodes(dependencies,check_types=False)[0]
        self.resources = ResourceSettings.from_dict(resources) if resources is not None else None
        allocation = (self.resources.inspection_mb if inspection else self.resources.job_mb) if self.resources else None
        self.plan = compile_flow(nodes,self.source.contract,center=center,single=single,following_nodes=self.dependencies,memory_mb=allocation)
        self.nodes = [n for n in self.plan["nodes"] if n["enabled"]]
        self.contracts = [self.source.contract]
        self.operations,self.instances,self.statistics = [],{},{}
        self.main_store = None
        self._store_building = False
        self._transient_target = None
        self._single = single
        self.execution_metrics = dict(stage_evaluations={},main_storage=None)
        legacy_mb = [n.get("execution",{}).get("memory_mb") or 256 for n in self.nodes] or [256]
        self.inspection = inspection
        self.budget = ((self.resources.inspection_mb if inspection else self.resources.job_mb) if self.resources else max(legacy_mb))*MIB
        cache_budget = min(self.resources.cache_mb*MIB,self.budget//3) if self.resources else min(legacy_mb)*MIB//3
        self.reader_cache_budget = cache_budget//4
        configure_reader(reader,self.reader_cache_budget)
        self.cache = Cache(cache_budget-self.reader_cache_budget)
        # Small reference state replaces part of the ordinary cache when it is
        # present; with cache disabled it is still bounded task working state.
        self.reference_budget = min(8*MIB, self.budget//64)
        self.cache.budget = max(0, self.cache.budget-self.reference_budget)
        self.reference_metrics = dict(queries=0, source_queries=0, passthrough_hops_skipped=0,
                                      index_transforms=0, budget_bytes=self.reference_budget)
        self.reference_states = {}
        self.mappings,self.backends = {},{}
        self.device_backends,self.device_records = {},{}
        self.resource_plan = []
        self.temporal_states = {}
        from eip.gpu_pipeline import DevicePipeline
        self.device = DevicePipeline(self)
        self.last_correlation = {}
        self.window_references = {}
        self.resolved_bindings = {}
        self._sides = []
        self.range_cache = range_cache
        retained = 0
        try:
            for i,n in enumerate(self.nodes):
                a = n["algorithm"]
                p = self._resolve_parameters(n,i)
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
                single_here = single and not needs_full_sequence(self.nodes[i+1:])
                stage_plan = compile_flow([n],c,center=center,single=single_here,following_nodes=[*self.nodes[i+1:],*self.dependencies],validate_bindings=False,memory_mb=allocation)
                stage = stage_plan["output"]
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
                elif a in CORRELATIONS:
                    if a=="correlation_window":
                        from eip.correlation_windows import window_scan_layout
                        windows,anchor = window_scan_layout(c.length,p)
                        lags,centers = [p["lag"]],range(anchor,anchor+1)
                        op["windows"] = windows
                    else:
                        lags,centers,_,_ = correlation_layout(c,p,center,single_here)
                    op.update(lags=lags,centers=centers)
                    self.mappings[n["id"]] = dict(lag_units="upstream_samples",window_units="upstream_samples",lags=lags,centers=dict(start=centers.start,stop=centers.stop,step=centers.step))
                    if a=="correlation_window":
                        self.mappings[n["id"]].update(axis="W",coordinates=windows,unit="upstream_samples",even_window="one_extra_sample_on_right")
                elif a=="line_extract" and p["mode"]=="segment":
                    length=float(np.hypot(p["x1"]-p["x0"],p["y1"]-p["y0"]))
                    self.mappings[n["id"]]=dict(axis="R",unit="pixel",coordinates=np.linspace(0,length,p["samples"]).tolist(),endpoints=[[p["x0"],p["y0"]],[p["x1"],p["y1"]]],interpolation=p["interpolation"])
                elif a=="line_extract":
                    self.mappings[n["id"]]=dict(axis="R",unit="pixel",coordinates=list(range(c.shape[c.axes.index(p["axis"])])),source_axis=p["axis"],fixed_index=p["index"],origin=list(c.origin))
                elif a=="distribution":
                    self.mappings[n["id"]]=dict(axis="B",count_unit="finite_values")
                    if p["minimum"] is not None and p["maximum"] is not None:
                        self.mappings[n["id"]]["edges"]=np.linspace(p["minimum"],p["maximum"],p["bins"]+1).tolist()
                elif n.get("manifest"):
                    self.instances[i] = Pipeline([StepSpec(**dict(n,parameters=p))],self.context)
                elif a in {"intensity","affine","box_blur","threshold","gaussian","curve"}:
                    self.instances[i] = BuiltinAlgorithm(a)
                    self.instances[i].prepare(p,self.context)
                elif a=="spatial_smooth":
                    self.instances[i] = BuiltinAlgorithm("box_blur" if p["method"]=="mean" else "gaussian")
                    self.instances[i].prepare(p,self.context)
                self._configure_backend(i,n)
                self.operations.append(op)
                self.contracts.append(after)
                for side in stage_plan["stages"][0].get("side_outputs",[]):
                    record=side["contract"]
                    side_c=Contract(**{k:tuple(record[k]) if k in {"shape","origin"} else record[k] for k in Contract.__dataclass_fields__})
                    port=side["port"]
                    port_name={"mean":"均值","variance":"方差","count":"有效样本数","scale":"固定尺度","value":"变量值"}.get(port)
                    title=module_spec(n)[0]["name"]+(f" · {port_name}" if port_name else "")
                    if a=="paired_trace":
                        if p["selection"]=="point":
                            title+=f" · 点 ({p['x']}, {p['y']})"
                        else:
                            x,y,w,h=roi_bounds(c,p)
                            label={"mean":"均值","max":"最大值","min":"最小值","variance":"方差","std":"标准差","sum":"总和","count":"计数","quantile":f"分位值 q={p['quantile']}"}[p["mode"]]
                            title+=f" · {label} ({x}, {y}; {w}×{h})"
                    self._sides.append(dict(id=f"{n['id']}:{port}",node_id=n["id"],port=port,name=p.get("name") or n.get("name") or title,contract=side_c,kind=a,stage=i,destination=p.get("destination","result"),export_format=p.get("export_format","dataset")))
                # Temporary arrays as well as caches must fit the chosen budget.
                elements = int(np.prod(c.shape,dtype=np.int64))
                multiplier = 12 if a=="line_extract" and p["mode"]=="segment" else 8 if a in {"global_statistics","time_statistics","region_reduce","paired_trace"} else 1 if a in {"time_crop","time_sample","select_channel","data_tap","parameter_binding"} else 3 if a in {"affine","intensity","curve","threshold","spatial_roi","roi_crop",*MODIFIERS} else 12 if a in CORRELATIONS else 8
                side_elements=max([int(np.prod(s["contract"].shape)) for s in self._sides if s["node_id"]==n["id"]] or [0])
                working=max(elements,int(np.prod(after.shape)),side_elements)*8*multiplier
                limit = min(self.budget,(n.get("execution",{}).get("memory_mb") or self.budget/MIB)*MIB)
                if working>limit*2/3:
                    remedy = "此实例另设内存上限，可清空上限或点击「流程继承统一资源」" if n.get("execution",{}).get("memory_mb") and limit<self.budget else "请先裁剪 ROI 或在运行设置增加统一预算"
                    raise ValueError(f"{n.get('name') or a} 工作数组估计 {working/MIB:.1f} MiB，超过内存预算 {limit/MIB:.0f} MiB；{remedy}")
                if a in {"statistics","sequence_statistics"}:
                    retained += elements*20
                elif a=="global_statistics":
                    retained += max(1,int(np.prod(after.shape if p["output"]=="main" else self._sides[-1]["contract"].shape)))*16
                elif a in CORRELATIONS:
                    retained += elements*24
                # Outer temporal accumulators survive while upstream is computed.
                nested = sum(int(np.prod(self.contracts[j].shape))*24 for j,old in enumerate(self.nodes[:i]) if old["algorithm"] in {"time_smooth","time_bin"})
                if retained+working+nested+self.cache.budget+self.reader_cache_budget+self.reference_budget>self.budget:
                    raise ValueError("重复统计保留数组与工作数组合计超过流程预算；请减少统计节点、先裁剪或提高内存预算")
                self.resource_plan.append(dict(node_id=n["id"],working_bytes=working,retained_bytes=retained,nested_bytes=nested,limit_bytes=int(limit),window_storage="disk" if a=="time_smooth" and p["mode"]=="quantile" else None))
                if a in {"time_smooth","time_bin"}:
                    self.resource_plan[-1]["temporal_strategy"]="exact_ring" if p["mode"]=="quantile" else "compensated_rolling_reduction"
                if i in self.backends and self.backends[i].name=="gpu":
                    gpu_retained=sum(r.get("device_retained_bytes",0) for r in self.resource_plan[:-1])
                    persistent = elements*(128 if a=="alignment" else 32 if a in {"time_smooth","time_bin"} else 24 if a in CORRELATIONS else 16 if a=="global_statistics" else 0)
                    if a=="global_statistics":
                        from eip.module_contracts import statistic_contract
                        persistent=max(1,int(np.prod(statistic_contract(c,a,p).shape)))*16
                    if a=="time_smooth" and p["mode"]=="quantile":
                        persistent=min(elements*p["window"]*8,self.device.scratch(i)//8)
                    if working+gpu_retained+persistent+self.device.cache_budget>self.device.budget*.75:
                        raise ValueError(f"{n.get('name') or a} 工作数组、状态和缓存超过显存预算；请裁剪输入、提高显存上限或对此实例选择 CPU")
                    self.resource_plan[-1].update(device_retained_bytes=persistent,device_working_bytes=working)
            self.device.configure_cache()
            if self.optimize:
                from eip.spatial_execution import SpatialExecution
                self.spatial = SpatialExecution(self)
                # Reserve physical raw-block and regional window state from
                # the existing cache allocation, rather than adding a cache.
                reserve = self.spatial.raw_limit+self.spatial.temporal_limit
                self.cache.resize(max(0, self.cache.budget-reserve))
            self._configure_main_store()
            self.plan = self._actual_plan()
        except Exception:
            self.close()
            raise

    def _configure_backend(self,i,node):
        from eip.modes.split_correlation.backend import Backend
        requested = node.get("execution",{}).get("backend","inherit")
        requested = (self.resources.backend if self.resources else "cpu") if requested=="inherit" else requested
        a = node["algorithm"]
        if a not in GPU_MODULES:
            self.device_records[node["id"]] = dict(requested=requested,actual="cpu",reason="此模块使用 CPU")
            return
        if requested not in self.device_backends:
            budget = self.resources.gpu_mb*MIB*(.25 if self.inspection else .75) if self.resources else self.budget
            self.device_backends[requested] = Backend(requested,budget)
        backend = self.device_backends[requested]
        if self.profiler and backend.name == "gpu":
            backend.gpu_timer = self.profiler.gpu_timer(backend.xp)
        self.backends[i] = backend
        self.device_records[node["id"]] = dict(requested=requested,actual=backend.name,reason=backend.reason)
        if backend.reason and not any(r.get("reason")==backend.reason for k,r in self.device_records.items() if k!=node["id"]):
            self.context.log(f"自动设备选择：使用 CPU；{backend.reason}")

    def _configure_main_store(self):
        from eip.stage_storage import compact_checkpoint, StageStore
        stage = compact_checkpoint(self)
        if stage is None:
            return
        contract = self.contracts[stage]
        size = contract.length*max(1,int(np.prod(contract.shape)))*np.dtype(contract.dtype).itemsize
        metadata = contract.length
        working = (max((r["working_bytes"] for r in self.resource_plan),default=0)
                   +max((r["retained_bytes"] for r in self.resource_plan),default=0)
                   +max((r["nested_bytes"] for r in self.resource_plan),default=0))
        transfers = self.budget//8 if any(b.name=="gpu" for b in self.backends.values()) else 0
        free = max(0,self.budget-working-transfers-self.cache.budget-self.reader_cache_budget-self.reference_budget-4*MIB)
        # A readiness byte is resident even when the numerical carrier is on
        # disk. Do not replace a bounded cache with an unbounded Python index.
        if metadata>min(self.budget//64,8*MIB) or metadata>free:
            return
        backend = self.device.backend(stage)
        plans = [r for r in self.resource_plan if "device_working_bytes" in r]
        retained = sum(r.get("device_retained_bytes",0) for r in plans)
        device_working = max((r["device_working_bytes"] for r in plans),default=0)
        device_free = max(0,int(self.device.budget*.75)-retained-device_working)
        limit = self.nodes[stage-1].get("execution",{}).get("memory_mb")
        if limit:
            free = min(free,max(0,int(limit*MIB)-self.resource_plan[stage-1]["working_bytes"]))
            own = self.resource_plan[stage-1]
            device_free = min(device_free,max(0,int(limit*MIB)-own.get("device_working_bytes",0)-own.get("device_retained_bytes",0)))
        if metadata>free:
            return
        if backend is not None and size<=device_free:
            mode = "device"
            self.device.configure_cache(extra_retained=size)
        elif size+metadata<=free:
            mode = "ram"
        else:
            mode = "disk"
        if mode=="disk" and (self.inspection or self._single):
            # A local check must not reserve a whole multi-GiB scratch carrier
            # just to inspect one frame/window. The lazy path is already bounded.
            return
        if mode=="disk":
            probe = self.workdir
            while not probe.exists():
                probe = probe.parent
            if shutil.disk_usage(probe).free<size+32*MIB:
                # This store is an execution optimization, not a new scientific
                # dependency. Keep the valid bounded reader path when it cannot
                # be spooled, rather than making a small final output fail.
                self.execution_metrics["main_storage_fallback"] = "insufficient_temporary_disk"
                return
        local_sample = self.inspection or self._single and self.output.sequence_axis=="T"
        block_size = 1 if local_sample else self.spatial.batch_size(stage) if self.spatial is not None else 32
        self.main_store = StageStore(stage,contract,mode=mode,workdir=self.workdir,
                                    backend=backend,block_size=block_size)
        self.execution_metrics["main_storage"] = self.main_store.record()

    def _store_callbacks(self):
        store = self.main_store
        def compute(sample):
            old_building,old_target = self._store_building,self._transient_target
            self._store_building,self._transient_target = True,store.stage
            try:
                backend = self.device.backend(store.stage)
                if backend is not None:
                    value = self.device.get(store.stage,sample,backend)
                    return value if store.mode=="device" else self.device.host(value,backend)
                return self.get(store.stage,sample)
            finally:
                self._store_building,self._transient_target = old_building,old_target
        def release(stop):
            removed = self.cache.discard_stages(range(store.stage+1))
            # Source data remain independently readable. Only completed prefix
            # cache entries are released, not files, transforms or side records.
            if hasattr(self.source.reader,"_cache"):
                self.source.reader._cache.clear()
                self.source.reader._cache_size = 0
            removed += self.device.release_consumed_prefix(store.stage,stop)
            return removed
        def compute_block(start, stop):
            old_building,old_target = self._store_building,self._transient_target
            self._store_building,self._transient_target = True,store.stage
            try:
                backend = store.backend if store.mode=="device" else None
                return self.spatial.compute(store.stage,range(start,stop),destination=backend)
            finally:
                self._store_building,self._transient_target = old_building,old_target
        return compute,release,compute_block if self.spatial is not None else None

    def _stored_main(self,index,device_backend=None):
        store = self.main_store
        compute,release,compute_block = self._store_callbacks()
        value = store.get(index,compute,self.context.check_cancelled,release,
                          compute_block=compute_block)
        if store.mode=="device":
            return value if device_backend is not None else self.device.host(value,store.backend)
        if device_backend is not None:
            # Respect a real CPU/GPU boundary using the existing aligned,
            # budgeted upload cache rather than a per-access hidden conversion.
            return self.device.upload(store.stage,index,device_backend)
        return value

    def _stored_range(self,start,stop,device_backend=None,*,max_bytes=None):
        """Read an owned checkpoint interval without per-sample array objects.

        Device changes remain explicit: the numerical adapter must request the
        checkpoint's actual carrier. Cross-device module boundaries were already
        applied while filling the checkpoint through the ordinary executor.
        """
        store = self.main_store
        if store is None:
            raise ValueError("连续信号读取需要已规划的阶段存储")
        if ((store.mode=="device") != (device_backend is not None)
                or store.mode=="device" and device_backend is not store.backend):
            raise ValueError("连续信号读取不能隐式改变阶段存储设备")
        compute,release,compute_block = self._store_callbacks()
        return store.get_range(start,stop,compute,self.context.check_cancelled,release,
                               compute_block=compute_block,max_bytes=max_bytes)

    def _gpu_apply(self,i,value):
        op,backend,c = self.operations[i],self.backends[i],self.contracts[i]
        a,p,n = op["algorithm"],op["parameters"],op["node"]
        if value.dtype.kind in "ui" and value.dtype.itemsize>4 and a!="fixed_normalize":
            raise ValueError("内置数值算法暂不处理 64 位整数；请显式提供精度策略")
        def transform(arr):
            # A nonspatial colour vector has the same threshold reduction as CPU.
            if a=="threshold" and not c.spatial and "C" in c.axes:
                xp=backend.xp
                return (xp.mean(xp.asarray(arr,dtype=xp.float64)[...,:3],axis=-1)>=p["threshold"]).astype(xp.uint8)
            result = element_operation(a,arr,p,backend.xp,color_order=c.color_order,
                                       preserve_input=a in {"gain_offset","spatial_smooth"} and n.get("execution",{}).get("precision","native")=="native")
            return result
        if c.paired and a not in {"fixed_normalize","grayscale"}:
            return backend.xp.stack([transform(arr) if n["target"] in {"both",side} else arr for arr,side in zip(value,("left","right"))])
        return transform(value)

    def _actual_plan(self):
        # Include disabled cards in provenance but only active operations in shape chain.
        stages,active = [],0
        upstream = []
        for n in self.plan["nodes"]:
            before = self.contracts[active]
            if n["enabled"]:
                active += 1
                upstream.append({key:value for key,value in n.items() if key!="name"})
            fingerprint = hashlib.sha256(json.dumps(dict(source_revision=self.source.reader.meta.revision,contract_version=3,upstream=upstream,preview_dependencies=self.dependencies),sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
            stages.append(dict(id=n["id"],algorithm=n["algorithm"],enabled=n["enabled"],input=before.record(),output=self.contracts[active].record(),upstream_fingerprint=fingerprint,side_outputs=[dict(port=s["port"],contract=s["contract"].record()) for s in self._sides if s["node_id"]==n["id"]]))
        from eip.execution_plan import compile_execution_plan
        physical = compile_execution_plan(self.plan["nodes"],self.contracts,consumers=self._sides,devices=self.device_records)
        execution = dict(self.execution_metrics,main_storage=self.main_store.record() if self.main_store else None,
                         optimization_enabled=self.optimize,physical_plan=physical,
                         spatial_execution=self.spatial.record() if self.spatial is not None else None)
        return dict(self.plan,stages=stages,output=self.contracts[-1].record(),preview_dependencies=list(self.dependencies),resolved_bindings=self.resolved_bindings,devices=self.device_records,resources=self.resources.record() if self.resources else None,resource_role="inspection" if self.inspection else "job",worker_budget_bytes=self.budget,resource_plan=self.resource_plan,device_execution=self.device.record(),execution=execution,reference_execution=dict(self.reference_metrics))

    def _resolve_parameters(self,node,position):
        from copy import deepcopy
        p=deepcopy(node["parameters"])
        for key,value in list(p.items()):
            if not is_reference(value):
                continue
            ref=value["ref"]
            previous=next((i for i,n in enumerate(self.nodes[:position]) if n["id"]==ref["node_id"]),None)
            if previous is None:
                raise ValueError("参数引用必须指向已启用的前序模块")
            op=self.operations[previous]
            port=ref["port"]
            if port.startswith("parameter:"):
                resolved=op["parameters"][port.partition(":")[2]]
            elif op["algorithm"]=="parameter_binding" and port=="value":
                resolved=op["parameters"]["value"]
            elif op["algorithm"]=="sequence_statistics" and port in {"mean","variance","count","scale"}:
                resolved=self.fit(previous)["scales" if port=="scale" else port]
            elif op["algorithm"]=="global_statistics" and port in {"value","count"}:
                resolved=self.global_fit(previous)[0 if port=="value" else 1]
            else:
                raise ValueError("参数端口不存在")
            resolved=np.asarray(resolved)
            if node["algorithm"]=="fixed_normalize" and key=="scale" and resolved.shape==(2,) and self.contracts[position].paired:
                resolved=resolved.reshape((2,)+tuple(1 for _ in self.contracts[position].shape[1:]))
            if resolved.size!=1 and not (node["algorithm"]=="fixed_normalize" and key=="scale"):
                raise ValueError("该参数只接受数值标量")
            p[key]=resolved.item() if resolved.size==1 else resolved
            self.resolved_bindings.setdefault(node["id"],{})[key]=dict(reference=ref,value=resolved.tolist())
        check=dict(p)
        if node["algorithm"]=="fixed_normalize" and isinstance(check.get("scale"),np.ndarray):
            scale=check["scale"]
            if not np.isfinite(scale).all() or (scale<=0).any():
                raise ValueError("引用的固定尺度必须为有限正数")
            check["scale"]=1.
        normalize_nodes([dict(node,parameters=check)],check_types=False)
        return p

    def side_outputs(self):
        return [dict(s) for s in self._sides]

    def get_side_output(self,identifier,index):
        side=next((s for s in self._sides if s["id"]==identifier),None)
        if side is None:
            raise KeyError(identifier)
        if not 0<=index<side["contract"].length:
            raise IndexError("侧输出样本超出范围")
        i=side["stage"]; op=self.operations[i]; a,p=op["algorithm"],op["parameters"]
        if a=="sequence_statistics":
            value=self.fit(i)["scales" if side["port"]=="scale" else side["port"]]
            return np.asarray(value).reshape(side["contract"].shape)
        if a=="parameter_binding":
            return np.asarray(p["value"],dtype=np.float64)
        if a=="data_tap":
            return self.get(i,index)
        if i in self.backends and self.backends[i].name=="gpu":
            return self.device.host(self.device.modifier(i,index,self.backends[i]),self.backends[i])
        return self._modifier_value(i,index)

    def side_ref(self,identifier,index):
        side=next(s for s in self._sides if s["id"]==identifier)
        i=side["stage"]; op=self.operations[i]
        if op["algorithm"]=="global_statistics":
            return dict(self.global_ref(i),sample_index=index,side_output=identifier)
        if op["algorithm"] in {"sequence_statistics","parameter_binding"}:
            return dict(index=None,time=None,sample_index=index,side_output=identifier,upstream_samples=self.contracts[i].length)
        p=op["parameters"]
        selected=p["index"] if op["algorithm"]=="axis_slice" and p["axis"]==self.contracts[i].sequence_axis else index
        return dict(self.ref(i,selected),sample_index=index,side_output=identifier,selection=p)

    def _histogram_edges(self,i,*,xp=np,get_sample=None,host=np.asarray,batch_size=1):
        op = self.operations[i]
        p, mapping = op["parameters"], self.mappings[op["node"]["id"]]
        if "edges" in mapping:
            return mapping["edges"]
        contract = self.contracts[i]
        fingerprint = hashlib.sha256(json.dumps(dict(
            source_revision=self.source.reader.meta.revision,upstream=self.nodes[:i],
            contract=contract.record(),dependencies=self.dependencies,center=self.center,single=self.single,
            seed=self.context.seed),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        detected = self.range_cache.get(fingerprint) if self.range_cache is not None else None
        if detected is None:
            minimum,maximum = xp.asarray(xp.inf),xp.asarray(-xp.inf)
            self.context.log(f"分布统计：正在自动识别边界（{contract.length} 个上游样本）")
            interval = max(1,contract.length//20)
            for sample in range(0,contract.length,batch_size):
                self.context.check_cancelled()
                stop=min(contract.length,sample+batch_size)
                get=get_sample or (lambda j:self.get(i,j))
                value=xp.stack([get(j) for j in range(sample,stop)]) if stop-sample>1 else get(sample)
                finite = xp.isfinite(value)
                minimum=xp.minimum(minimum,xp.min(xp.where(finite,value,xp.inf)))
                maximum=xp.maximum(maximum,xp.max(xp.where(finite,value,-xp.inf)))
                if sample//interval!=stop//interval or stop==contract.length:
                    self.context.log(f"分布边界识别：{stop}/{contract.length}")
            minimum,maximum=float(host(minimum)),float(host(maximum))
            if not np.isfinite([minimum,maximum]).all():
                raise ValueError("分布自动边界无法识别：上游范围没有有限值，请手动设置上下界")
            detected = dict(minimum=minimum,maximum=maximum,samples=contract.length)
            if self.range_cache is not None:
                self.range_cache[fingerprint] = detected
                while len(self.range_cache)>16:
                    self.range_cache.popitem(last=False)
        elif self.range_cache is not None:
            self.range_cache.move_to_end(fingerprint)
        lower = detected["minimum"] if p["minimum"] is None else p["minimum"]
        upper = detected["maximum"] if p["maximum"] is None else p["maximum"]
        if lower==upper:
            padding = max(.5,abs(lower)*1e-6)
            if p["minimum"] is None:
                lower -= padding
            if p["maximum"] is None:
                upper += padding
        if not np.isfinite([lower,upper]).all() or lower>=upper:
            raise ValueError("分布自动边界与手动边界不兼容：上界须大于下界")
        mapping.update(edges=np.linspace(lower,upper,p["bins"]+1).tolist(),
                       bounds_mode=dict(minimum="auto" if p["minimum"] is None else "manual",
                                        maximum="auto" if p["maximum"] is None else "manual"),
                       detected=detected,range_scope="complete_upstream_sequence")
        return mapping["edges"]

    def _modifier_value(self,i,index,*,xp=np,get_sample=None):
        if get_sample is None and self.spatial is not None and self.spatial.can_select(i,side=True):
            return self.spatial.modifier(i,index)
        op=self.operations[i]; a,p=op["algorithm"],op["parameters"]; c=self.contracts[i]
        get=get_sample or (lambda j:self.get(i,j))
        if a=="axis_slice" and p["axis"]==c.sequence_axis:
            return get(p["index"])
        if a=="global_statistics":
            return self.global_fit(i)[0]
        value=get(index)
        if a=="paired_trace":
            x,y,w,h=roi_bounds(c,dict(p,width=1,height=1) if p["selection"]=="point" else p)
            values=joint_values(value[:,y:y+h,x:x+w],True,p["validity"],xp)
            if p["selection"]=="point":
                return xp.asarray(values[:,0,0],dtype=xp.float64)
            return reduce_finite(values,(1,2),p["mode"],p["quantile"],xp)[0]
        if a=="time_statistics":
            value=joint_values(value,c.paired,p["validity"],xp)
            axes=tuple(k for k,axis in enumerate(c.axes) if axis not in retained_axes(c.axes,"space_time"))
            return reduce_finite(value,axes,p["mode"],p["quantile"],xp)[0]
        if a=="region_reduce":
            if c.paired and p["validity"]=="joint":
                value=xp.where(xp.isfinite(value).all(axis=0)[None,...],value,xp.nan)
            return reduce_finite(value,tuple(c.axes.index(axis) for axis in "YX"),p["mode"],p["quantile"],xp)[0]
        if a=="point_extract":
            slices=[slice(None)]*value.ndim
            slices[c.axes.index("Y")],slices[c.axes.index("X")]=p["y"],p["x"]
            return value[tuple(slices)]
        if a=="line_extract":
            if p["mode"]=="axis":
                return xp.take(value,p["index"],axis=c.axes.index("Y" if p["axis"]=="X" else "X"))
            ys=xp.linspace(p["y0"],p["y1"],p["samples"]); xs=xp.linspace(p["x0"],p["x1"],p["samples"])
            matrix=xp.moveaxis(value,(c.axes.index("Y"),c.axes.index("X")),(-2,-1))
            if p["interpolation"]=="nearest":
                line=matrix[...,xp.floor(ys+.5).astype(int),xp.floor(xs+.5).astype(int)]
            else:
                y0=xp.floor(ys).astype(int); x0=xp.floor(xs).astype(int); y1=xp.minimum(y0+1,matrix.shape[-2]-1); x1=xp.minimum(x0+1,matrix.shape[-1]-1)
                dy,dx=ys-y0,xs-x0
                line=xp.zeros((*matrix.shape[:-2],p["samples"]),dtype=xp.float64)
                valid=xp.ones_like(line,dtype=bool)
                for yy,xx,weight in ((y0,x0,(1-dy)*(1-dx)),(y0,x1,(1-dy)*dx),(y1,x0,dy*(1-dx)),(y1,x1,dy*dx)):
                    sample=matrix[...,yy,xx]
                    valid &= xp.isfinite(sample)|(weight==0)
                    line += xp.where(weight==0,0,sample*weight)
                line=xp.where(valid,line,xp.nan)
            return xp.moveaxis(line,-1,c.axes.index("Y"))
        if a=="axis_slice":
            return xp.take(value,p["index"],axis=c.axes.index(p["axis"]))
        if a=="distribution":
            edges = self._histogram_edges(i,xp=xp,get_sample=get_sample,host=self.backends[i].host if get_sample else np.asarray)
            return xp.histogram(value[xp.isfinite(value)],bins=p["bins"],range=(edges[0],edges[-1]))[0].astype(xp.int64)
        raise ValueError("未知修饰操作")

    @property
    def output(self):
        return self.contracts[-1]

    def iter_outputs(self, indices):
        """Yield logical host samples from bounded physical output batches.

        Random inspection continues through get(). Full delivery can group work
        without changing output order, time axes or per-node provenance.
        """
        stage = len(self.nodes)
        if self.spatial is not None and self.spatial.prefer_sample_streaming():
            self.execution_metrics["full_frame_cpu_policy"] = "sample_streaming_avoids_redundant_block_copy"
            for j in indices:
                yield j,self.get(stage,j)
            return
        if self.spatial is None or not isinstance(indices, range) or indices.step != 1:
            for j in indices:
                yield j, self.get(stage, j)
            return
        runner = None
        correlation_stage = stage
        while correlation_stage and self.spatial.can_forward(correlation_stage-1):
            correlation_stage -= 1
        if stage and indices.start == 0 and len(indices) == self.output.length:
            if correlation_stage:
                from eip.lag_scan_blocks import lag_engine_blocks
                runner = lag_engine_blocks(self,correlation_stage)
            if runner is None and correlation_stage:
                from eip.gpu_sequence_blocks import engine_blocks
                runner = engine_blocks(self,correlation_stage)
                if runner is None:
                    from eip.cpu_sequence_blocks import cpu_engine_blocks
                    runner = cpu_engine_blocks(self,correlation_stage)
        if runner is not None:
            self._output_runners.append(runner)
            backend = self.device.backend(correlation_stage)
            precision = self.nodes[correlation_stage-1].get("execution", {}).get("precision", "native")
            try:
                for start in range(0, len(indices), runner.max_output_block):
                    self.context.check_cancelled()
                    stop = min(len(indices), start+runner.max_output_block)
                    with self.profiler.measure("cpu_compute" if backend is None else "gpu_submit") if self.profiler else nullcontext():
                        values, _ = runner.compute(start, stop)
                        if precision != "native":
                            values = values.astype(precision)
                    counts = self.execution_metrics["stage_evaluations"]
                    counts[str(correlation_stage)] = counts.get(str(correlation_stage),0)+stop-start
                    values = self.spatial.forward(correlation_stage,stage,values,backend)
                    for j, value in zip(range(start, stop), values):
                        yield j, value
            finally:
                runner.close()
                self._output_runners.remove(runner)
            return
        # Ineligible Q/W schedules retain their validated demand states.
        # Do not stack multiple device results merely to download a host block.
        if any(n["algorithm"] in CORRELATIONS for n in self.nodes):
            for j in indices:
                yield j, self.get(stage, j)
            return
        size = self.spatial.batch_size(stage)
        for first in range(indices.start, indices.stop, size):
            stop = min(indices.stop, first+size)
            values = self.spatial.compute(stage, range(first, stop))
            for j, value in zip(range(first, stop), values):
                yield j, value

    def get(self,stage,index):
        if self.profiler is None:
            return self._get(stage,index)
        category = "gpu_submit" if self.device.backend(stage) is not None else "cpu_compute"
        with self.profiler.measure(category):
            return self._get(stage,index)

    def _get(self,stage,index):
        self.context.check_cancelled()
        c = self.contracts[stage]
        if not 0<=index<c.length:
            raise IndexError("编辑后的样本号超出范围")
        if self.main_store is not None and stage==self.main_store.stage and not self._store_building:
            return self._stored_main(index)
        transient = self._transient_target is not None and stage<=self._transient_target
        key = stage,index
        hit = None if transient else self.cache.get(key)
        if hit is not None:
            return hit
        if stage and self.spatial is not None and self.device.backend(stage) is None and self.spatial.can_select(stage-1):
            value = self.spatial.evaluate(stage-1,index)
            return value if transient else self.cache.put(key,value)
        backend=self.device.backend(stage)
        if backend is not None:
            value=np.asarray(self.device.host(self.device.get(stage,index,backend),backend))
            return value if transient else self.cache.put(key,value)
        evaluations = self.execution_metrics["stage_evaluations"]
        evaluations[str(stage)] = evaluations.get(str(stage),0)+1
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
                    from eip.window_statistics import WindowQuantile
                    if i not in self.temporal_states:
                        self.temporal_states[i]=WindowQuantile(before.shape,size,self.resource_plan[i]["limit_bytes"]//8,self.workdir)
                    value = self.temporal_states[i].get(lambda j:self.get(i,j),first,p["quantile"],self.context.check_cancelled)
                else:
                    from eip.window_statistics import RollingReduction
                    if i not in self.temporal_states:
                        self.temporal_states[i]=RollingReduction(before.shape)
                    value=self.temporal_states[i].get(lambda j:self.get(i,j),first,size,p["mode"],self.context.check_cancelled)
            elif a=="statistics":
                stats = self.fit(i)
                value = self.get(i,index)/stats["scales"]
            elif a in {"data_tap","sequence_statistics","parameter_binding"}:
                value=self.get(i,index)
            elif a=="fixed_normalize":
                scale=np.asarray(p["scale"],dtype=np.float64)
                if not np.isfinite(scale).all() or (scale<=0).any():
                    raise ValueError("固定尺度必须为有限正数")
                value=np.asarray(self.get(i,index),dtype=np.float64)/scale
            elif a in MODIFIERS:
                value=self.get(i,index) if p["output"]=="side" else self._modifier_value(i,index)
            elif a in CORRELATIONS:
                value = self.correlate(i,index)[0]
            else:
                value = self.get(i,index)
                if a in {"spatial_roi","roi_crop"}:
                    x,y,w,h = roi_bounds(before,p)
                    slices = [slice(None)]*len(before.shape)
                    slices[before.axes.index("Y")],slices[before.axes.index("X")] = slice(y,y+h),slice(x,x+w)
                    value = value[tuple(slices)].copy()
                    if p.get("mode","pixels")!="pixels":
                        if before.paired and p["validity"]=="joint":
                            value = np.where(np.isfinite(value).all(axis=0)[None,...],value,np.nan)
                        value,_ = reduce_finite(value,tuple(before.axes.index(axis) for axis in "YX"),p["mode"],p["quantile"])
                elif a=="alignment":
                    value = np.stack(self.instances[i].apply(value,IdentityPhotometry()))
                elif a=="select_channel":
                    value = value[0 if p["channel"]=="left" else 1]
                elif a=="grayscale":
                    rgb=value[...,:3].astype(np.float64)
                    if before.color_order=="BGR":
                        rgb=rgb[...,::-1]
                    value=rgb.mean(axis=-1) if p["method"]=="mean" else np.sum(rgb*np.array([.2126,.7152,.0722]),axis=-1)
                elif a=="color_convert":
                    if "C" not in before.axes:
                        value=np.repeat(value[...,None],3,axis=-1)
                    elif before.color_order!=p["mode"].upper():
                        value=value[...,::-1].copy()
                elif a=="calibration":
                    photo = self.instances[i]
                    if before.paired:
                        value = np.stack([photo.apply(arr,side) for arr,side in zip(value,("left","right"))])
                    elif p["black_left"]!=p["black_right"]:
                        # Split offsets require an explicit downstream geometry in
                        # these exact input coordinates; no hidden middle split.
                        future = [*self.nodes[i+1:],*[n for n in self.dependencies if n["enabled"]]]
                        following = next((o for o in future if o["algorithm"]=="alignment"),None)
                        if following is None or any(o["algorithm"] in {"spatial_roi","roi_crop"} for o in future[:future.index(following)]):
                            raise ValueError("不同左右暗场需紧邻后续配准，或把校准移到配准后")
                        line = np.interp(np.arange(value.shape[0])[:,None],[0,max(1,value.shape[0]-1)],following["parameters"]["midline"])
                        value = np.where(np.arange(value.shape[1])[None,:]<line,photo.apply(value,"left"),photo.apply(value,"right"))
                    else:
                        value = photo.apply(value,"left")
                elif a in {"spatial_smooth","gain_offset"}:
                    if value.dtype.kind in "ui" and value.dtype.itemsize>4:
                        raise ValueError("内置数值算法暂不处理 64 位整数；请显式提供精度策略")
                    def transform(arr):
                        arr64=np.asarray(arr,dtype=np.float64)
                        if a=="gain_offset":
                            with np.errstate(over="ignore",invalid="ignore"):
                                result=arr64*p["gain"]+p["offset"]
                            result=np.where(np.isfinite(result),result,np.nan)
                        else:
                            result=self.instances[i].process(arr64,self.context)
                        return preserve_dtype(result,arr.dtype) if n.get("execution",{}).get("precision","native")=="native" else result
                    value=np.stack([transform(arr) if n["target"] in {"both",side} else arr for arr,side in zip(value,("left","right"))]) if before.paired else transform(value)
                else:
                    def apply(arr):
                        # Scalar/vector element operations use their true rank.
                        if a=="threshold" and not before.spatial and "C" in before.axes:
                            return np.asarray(np.mean(arr[..., :3],axis=-1)>=p["threshold"],dtype=np.uint8)
                        return self.instances[i].execute(arr) if n.get("manifest") else self.instances[i].process(arr,self.context)
                    value = np.stack([apply(arr) if n["target"] in {"both",side} else arr for arr,side in zip(value,("left","right"))]) if before.paired else apply(value)
            precision = n.get("execution",{}).get("precision","native")
            if precision!="native" and a not in {"data_tap","sequence_statistics","parameter_binding"} and not (a in MODIFIERS and p["output"]=="side"):
                value = np.asarray(value,dtype=precision)
        value = np.asarray(value)
        if value.shape!=c.shape:
            raise ValueError(f"算法实际输出 {value.shape} 与声明形状 {c.shape} 不符")
        if stage and self.nodes[stage-1].get("manifest"):
            if c.dtype=="dynamic":
                self.contracts[stage] = replace(c,dtype=str(value.dtype))
            elif value.dtype!=np.dtype(c.dtype):
                raise ValueError(f"插件输出 dtype 从 {c.dtype} 变为 {value.dtype}；请保持一致或显式设置数值精度")
        return value if transient else self.cache.put(key,value)

    def global_fit(self,i):
        if i not in self.global_statistics:
            op=self.operations[i]
            backend=self.backends.get(i)
            if backend is not None and backend.name=="gpu":
                values=self.device.global_fit(i,backend)
                value,count=(self.device.host(a,backend) for a in values)
            else:
                value,count=global_statistic(self.contracts[i],op["parameters"],lambda j:self.get(i,j),self.context,self.workdir)
            precision=op["node"].get("execution",{}).get("precision","native")
            if precision!="native":
                value=np.asarray(value,dtype=precision)
            self.global_statistics[i]=value,count
        return self.global_statistics[i]

    def global_ref(self,i):
        c=self.contracts[i]; n=self.operations[i]["node"]; p=self.operations[i]["parameters"]
        return dict(index=None,time=None,pts=None,source_number=None,trigger_time=None,
                    contribution=dict(module_id=n["id"],upstream_start=0,upstream_stop=c.length,operation="global_statistics"),
                    statistic=dict(scope=p["scope"],mode=p["mode"],quantile=p["quantile"] if p["mode"]=="quantile" else None,
                                   input_axes=c.record()["full_axes"],retained_axes=retained_axes(c.axes,p["scope"]),validity=p["validity"]))

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

    def correlation_position(self,i,index):
        from eip.correlation_windows import window_bounds
        op = self.operations[i];p = op["parameters"]
        if op["algorithm"]=="correlation_window":
            center,lag,window = op["centers"][0],op["lags"][0],op["windows"][index]
        else:
            center,lag = (op["centers"][0],op["lags"][index]) if p["analysis"]=="curve" else (op["centers"][index],op["lags"][0])
            window = p["window"]
        start,stop = window_bounds(center,window)
        return center,lag,window,start,stop

    def correlate(self,i,index,*,device=False):
        from eip.modes.split_correlation.correlation import RollingMoments
        op = self.operations[i]
        p = op["parameters"]
        center,lag,window,start,stop = self.correlation_position(i,index)
        backend=self.backends[i]
        xp=backend.xp
        get=(lambda j:self.device.get(i,j,backend)) if backend.name=="gpu" else (lambda j:self.get(i,j))
        previous = self.last_correlation.get(i)
        # The fixed centre uses exactly the same baseline for every W / lag.
        # Retain the accumulator's baseline rather than rereading or realigning
        # the centre image for every output, even when the sample cache is off.
        baseline = previous["moments"].baseline if previous and previous["baseline_center"]==center else None
        expanding = (op["algorithm"]=="correlation_window" and previous
                     and start<=previous["start"] and stop>=previous["stop"])
        # Purely expanding W never subtracts old samples. Its accumulation order
        # is retained rather than periodically rereading the growing window.
        can_roll = previous and previous["lag"]==lag and start<previous["stop"] and previous["start"]<stop and (index%256!=0 or expanding)
        if can_roll:
            moments = previous["moments"]
            baseline_center = previous["baseline_center"]
            old_start,old_stop = previous["start"],previous["stop"]
            for bounds in ((old_start,min(start,old_stop)),(max(stop,old_start),old_stop)):
                self._add_correlation_pairs(moments,get,bounds,lag,-1,i,backend)
            additions = ((start,min(old_start,stop)),(max(old_stop,start),stop))
        else:
            if baseline is None:
                # This is a numerical identity, not a photometric filter.
                baseline = xp.nan_to_num(get(center),nan=0.,posinf=0.,neginf=0.)
            moments = RollingMoments(baseline,SimpleNamespace(**p),self.backends[i])
            baseline_center = center
            additions = ((start,stop),)
        for bounds in additions:
            self._add_correlation_pairs(moments,get,bounds,lag,1,i,backend)
        self.last_correlation[i] = dict(lag=lag,center=center,baseline_center=baseline_center,start=start,stop=stop,moments=moments)
        self.mappings[op["node"]["id"]].update(correlation_implementation=moments.implementation)
        return moments.result(device=device,count=False)

    def _add_correlation_pairs(self,moments,get,bounds,lag,sign,i,backend):
        first,stop = bounds
        pixels = max(1,int(np.prod(moments.baseline.shape[1:])))
        batch = 1
        if backend.name=="gpu" and pixels<=4096:
            # Lists, paired allocations retained by their views and the float64
            # stacks coexist. Reserve eight arrays per pair and bound pressure;
            # large maps use the
            # single-pair fused path without a second image-sized stack.
            limit = min(self.budget//32,self.device.budget//64,self.resource_plan[i]["limit_bytes"]//32)
            batch = max(1,min(64,limit//(pixels*8*8)))
        for begin in range(first,stop,batch):
            self.context.check_cancelled()
            end = min(stop,begin+batch)
            if backend.name=="gpu":
                metrics = self.device.metrics
                metrics["correlation_update_calls"] += 1
                metrics["correlation_pairs"] += end-begin
                metrics["largest_correlation_batch"] = max(metrics["largest_correlation_batch"],end-begin)
            if end-begin==1:
                pair = get(begin)
                moments.add(pair[0],pair[1] if lag==0 else get(begin+lag)[1],sign)
            else:
                left,right = [],[]
                xp = backend.xp
                def detached_pair(j):
                    value = get(j)
                    # An uncached point/ROI view may own an entire image or
                    # upload block. Retain only this small pair in the batch.
                    return value.astype(xp.float64,copy=True) if getattr(value,"base",None) is not None else value
                for j in range(begin,end):
                    pair = detached_pair(j)
                    left.append(pair[0]);right.append(pair[1] if lag==0 else detached_pair(j+lag)[1])
                moments.add_many(xp.stack(left).astype(xp.float64,copy=False),
                                 xp.stack(right).astype(xp.float64,copy=False),sign)

    def correlation_reference(self,i,index):
        from eip.reference_index import WindowTimeIndex, LagTimeIndex
        op = self.operations[i]
        center,lag,window,start,stop = self.correlation_position(i,index)
        r = self.ref(i,center)
        if i not in self.reference_states:
            maximum = max(op["windows"]) if op["algorithm"]=="correlation_window" else window
            count = sum(n["algorithm"] in CORRELATIONS for n in self.nodes)
            budget = self.reference_budget//max(1,count)
            if self.optimize and op["algorithm"]!="correlation_window" and op["parameters"].get("analysis")=="curve":
                self.reference_states[i] = LagTimeIndex(maximum,self.contracts[i].length,budget,
                    self.reference_metrics,get_times=lambda first,last:self._reference_times(i,first,last))
            else:
                self.reference_states[i] = WindowTimeIndex(maximum,budget,self.reference_metrics)
        total,minimum,maximum,known = self.reference_states[i].summary(
            start,stop,lag,lambda j:self.ref(i,j).get("time"),self.context.check_cancelled)
        return dict(r,correlation=dict(module_id=op["node"]["id"],center_sample=center,lag_samples=lag,window=window,left_start=start,left_stop=stop,right_start=start+lag,right_stop=stop+lag,lag_seconds=total/window if known==window else None,lag_seconds_min=minimum,lag_seconds_max=maximum,known_pairs=known))

    def _reference_times(self,stage,first,stop):
        """Exact indexed times for simple upstream index maps, or no shortcut.

        Compound temporal provenance and custom sample metadata still use ref().
        Only the requested bounded interval is allocated, never an unbudgeted
        full-source time table. Presence is independent of numerical finiteness.
        """
        reader = self.source.reader
        if self.source.spooled or hasattr(reader,"sample_ref"):
            return None
        if hasattr(reader,"times"):
            source_times = reader.times
            if source_times is None:
                return np.zeros(stop-first,dtype=np.float64),np.zeros(stop-first,dtype=bool)
        elif hasattr(reader,"timestamps") and reader.timestamps is not None:
            source_times = reader.timestamps
        else:
            return None
        positions = np.arange(first,stop,dtype=np.int64)
        for i in range(stage-1,-1,-1):
            op = self.operations[i]
            a,p = op["algorithm"],op["parameters"]
            if a=="time_crop":
                if "indices" in op:
                    selected = op["indices"]
                    if not isinstance(selected,range):
                        return None
                    positions = selected.start+positions*selected.step
                else:
                    positions += p["first"]-1
            elif a=="time_sample":
                positions = p["offset"]+positions*p["step"]
            elif (a in {"time_smooth","time_bin",*CORRELATIONS}
                  or a=="axis_slice" and p["output"]=="main" and p["axis"]==self.contracts[i].sequence_axis
                  or a=="global_statistics" and p["output"]=="main"):
                return None
        if isinstance(source_times,np.ndarray) and source_times.dtype.kind in "fiu":
            return np.asarray(source_times[positions],dtype=np.float64),np.ones(len(positions),dtype=bool)
        values = [source_times[int(j)] for j in positions]
        present = np.fromiter((v is not None for v in values),dtype=bool,count=len(values))
        numbers = np.fromiter((0. if v is None else float(v) for v in values),dtype=np.float64,count=len(values))
        return numbers,present

    def ref(self,stage,index):
        self.reference_metrics["queries"] += 1
        if self.profiler is None:
            return self._ref(stage,index)
        with self.profiler.measure("references"):
            return self._ref(stage,index)

    def _ref(self,stage,index):
        self.context.check_cancelled()
        requested_index = index
        # Collapse identity time paths and simple index transforms once per
        # query. Spatial/numerical modules retain their separate provenance in
        # the plan but need no recursive dictionaries for the same frame time.
        while stage:
            i = stage-1
            op = self.operations[i]
            a,p = op["algorithm"],op["parameters"]
            if a=="time_crop":
                index = op["indices"][index] if "indices" in op else index+p["first"]-1
                self.reference_metrics["index_transforms"] += 1
            elif a=="time_sample":
                index = p["offset"]+index*p["step"]
                self.reference_metrics["index_transforms"] += 1
            elif (a in {"time_smooth","time_bin",*CORRELATIONS}
                  or a=="axis_slice" and p["output"]=="main" and p["axis"]==self.contracts[i].sequence_axis
                  or a=="global_statistics" and p["output"]=="main"):
                break
            else:
                self.reference_metrics["passthrough_hops_skipped"] += 1
            stage -= 1
        if stage==0:
            self.reference_metrics["source_queries"] += 1
            original = self.source.ref(index)
            return dict(original,sample_index=requested_index,source_sample=index,source_span=original.get("source_span",[index,index+1]))
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
        elif a in CORRELATIONS:
            r = self.correlation_reference(i,index)
        elif a=="axis_slice" and op["parameters"]["output"]=="main" and op["parameters"]["axis"]==self.contracts[i].sequence_axis:
            r=dict(self.ref(i,op["parameters"]["index"]),selection=dict(axis=op["parameters"]["axis"],index=op["parameters"]["index"]))
        elif a=="global_statistics" and p["output"]=="main":
            r=self.global_ref(i)
        else:
            r = self.ref(i,index)
        return dict(r,sample_index=requested_index)

    def close(self):
        if self.profiler:
            # Finalize before releasing backend state. On a failed/cancelled
            # path the original computation exception remains authoritative.
            self.profiler.finish_gpu(raise_errors=False)
        if getattr(self,"spatial",None) is not None:
            self.spatial.close()
        for runner in getattr(self,"_output_runners",()):
            runner.close()
        if getattr(self,"main_store",None) is not None:
            self.main_store.close()
            self.main_store = None
        if hasattr(self,"device"):
            self.device.close()
        for state in self.temporal_states.values():
            if hasattr(state,"close"):
                state.close()
        self.temporal_states.clear()
        self.last_correlation.clear()
        self.window_references.clear()
        for state in self.reference_states.values():
            state.close()
        self.reference_states.clear()
        for instance in self.instances.values():
            if hasattr(instance,"close"):
                instance.close()
        for backend in self.device_backends.values():
            backend.close()
        self.instances.clear()
        self.backends.clear()
        self.device_backends.clear()
        self.global_statistics.clear()
        self.statistics.clear()
        self.cache.entries.clear()
        self.cache.size = 0
