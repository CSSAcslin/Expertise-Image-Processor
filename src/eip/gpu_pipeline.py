"""Budgeted device execution between explicit host preview / storage boundaries."""
from collections import OrderedDict
from contextlib import nullcontext
import numpy as np

from eip.exact_quantile import finite_quantile
from eip.module_ops import intensity_curve
from eip.module_registry import CORRELATIONS, MODIFIERS
from eip.module_statistics import global_statistic
from eip.window_statistics import RollingReduction, WindowQuantile


class DeviceCache:
    def __init__(self, budget):
        self.budget, self.size, self.peak = int(budget), 0, 0
        self.entries = OrderedDict()

    def get(self, key):
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]

    def put(self, key, value):
        if value.nbytes > self.budget:
            return value
        # A cached slice must not retain an unaccounted much larger allocation.
        if getattr(value, "base", None) is not None:
            value = value.copy()
        if key in self.entries:
            self.size -= self.entries.pop(key).nbytes
        while self.entries and self.size+value.nbytes > self.budget:
            self.size -= self.entries.popitem(last=False)[1].nbytes
        self.entries[key] = value
        self.size += value.nbytes
        self.peak = max(self.peak, self.size)
        return value

    def clear(self):
        self.entries.clear()
        self.size = 0

    def resize(self, budget):
        self.budget = max(0,int(budget))
        while self.entries and self.size>self.budget:
            self.size -= self.entries.popitem(last=False)[1].nbytes


class DevicePipeline:
    def __init__(self, engine):
        self.engine = engine
        self.budget = int(engine.resources.gpu_mb*1024**2*(.25 if engine.inspection else .75)) if engine.resources else engine.budget
        self.cache = DeviceCache(self.budget//16)
        self.upload_cache = DeviceCache(self.budget//16)
        self.cache_budget = self.cache.budget+self.upload_cache.budget
        self.constants, self.windows, self.global_values = {}, {}, {}
        self.metrics = dict(upload_batches=0, uploaded_samples=0, downloads=0,
                            device_cache_hits=0, largest_upload_batch=0,
                            correlation_update_calls=0,correlation_pairs=0,largest_correlation_batch=0)

    def configure_cache(self):
        """Use validated slack, leaving all persistent states and scratch room.

        This changes capacity only; arrays are still allocated on demand. Using
        the entire flow's retained state also protects nested downstream nodes.
        """
        plans = [r for r in self.engine.resource_plan if "device_working_bytes" in r]
        if not plans:
            return
        retained = sum(r.get("device_retained_bytes",0) for r in plans)
        working = max(r["device_working_bytes"] for r in plans)
        available = max(0,int(self.budget*.75)-retained-working)
        total = min(self.budget//4,available)
        self.cache.resize(total//2)
        self.upload_cache.resize(total-total//2)
        self.cache_budget = total

    def backend(self, stage):
        backend = self.engine.backends.get(stage-1)
        return backend if backend is not None and backend.name == "gpu" else None

    def host(self, value, backend):
        self.metrics["downloads"] += 1
        profiler = self.engine.profiler
        with profiler.measure("download") if profiler else nullcontext():
            return backend.host(value)

    def scratch(self, i):
        limit = self.engine.resource_plan[i]["limit_bytes"] if i < len(self.engine.resource_plan) else self.engine.budget
        return max(16384, min(limit//8, self.budget//8))

    def batch_size(self, stage, *, working=False):
        if self.engine.inspection:
            return 1
        c = self.engine.contracts[stage]
        item = 8 if c.dtype == "dynamic" else np.dtype(c.dtype).itemsize
        # Transfers use the actual source dtype; working reductions need float64.
        size = max(1, int(np.prod(c.shape)))*(max(8,item) if working else item)
        allowance = min(self.upload_cache.budget//2, self.engine.budget//16)
        if working:
            allowance = min(allowance, self.budget//64)
        if any(n["algorithm"] == "time_sample" and n["parameters"]["step"] > 1 for n in self.engine.nodes):
            return 1
        return max(1, min(32, allowance//max(1, size)))

    def upload(self, stage, index, backend):
        for key in reversed(self.upload_cache.entries):
            if key[0] == "upload" and key[1] == stage and key[2] <= index < key[3]:
                self.metrics["device_cache_hits"] += 1
                block = self.upload_cache.get(key)
                return block[index-key[2]]
        e = self.engine
        batch = self.batch_size(stage)
        # Fixed chunks are reusable when W grows towards decreasing indices.
        # Anchoring every upload at the requested index used to reread nearly
        # the same forward block for each newly added left-hand sample.
        start = index//batch*batch
        stop = min(e.contracts[stage].length,start+batch)
        # This is a real CPU boundary (source decode or an explicitly CPU node).
        first = e.get(stage,start)
        block = np.empty((stop-start,*first.shape),dtype=first.dtype)
        block[0] = first
        for j in range(start+1,stop):
            e.context.check_cancelled()
            block[j-start] = e.get(stage,j)
        profiler = e.profiler
        with profiler.measure("upload") if profiler else nullcontext():
            value = backend.xp.asarray(block)
        self.metrics["upload_batches"] += 1
        self.metrics["uploaded_samples"] += stop-start
        self.metrics["largest_upload_batch"] = max(self.metrics["largest_upload_batch"],stop-start)
        value = self.upload_cache.put(("upload",stage,start,stop),value)
        return value[index-start]

    def get(self, stage, index, backend):
        profiler = self.engine.profiler
        if profiler is None:
            return self._get(stage,index,backend)
        with profiler.measure("gpu_submit"):
            return self._get(stage,index,backend)

    def _get(self, stage, index, backend):
        e = self.engine
        e.context.check_cancelled()
        if not 0 <= index < e.contracts[stage].length:
            raise IndexError("编辑后的样本号超出范围")
        if not self.backend(stage):
            return self.upload(stage, index, backend)
        key = ("result", stage, index)
        value = self.cache.get(key)
        if value is not None:
            self.metrics["device_cache_hits"] += 1
            return value
        i = stage-1
        op = e.operations[i]
        a, p = op["algorithm"], op["parameters"]
        xp = backend.xp
        get = lambda j: self.get(i, j, backend)
        if a == "time_crop":
            value = get(op["indices"][index] if "indices" in op else index+p["first"]-1)
        elif a == "time_sample":
            value = get(p["offset"]+index*p["step"])
        elif a in {"time_smooth", "time_bin"}:
            value = self.temporal(i, index, backend)
        elif a in {"data_tap", "parameter_binding"} or a in MODIFIERS and p["output"] == "side":
            return get(index)
        elif a in CORRELATIONS:
            value = e.correlate(i, index, device=True)[0]
        elif a in MODIFIERS:
            value = self.modifier(i, index, backend)
        elif a in {"roi_crop", "select_channel", "color_convert", "calibration", "alignment"}:
            value = self.structural(i, get(index), backend)
        else:
            value = e._gpu_apply(i, get(index))
        precision = op["node"].get("execution", {}).get("precision", "native")
        value = xp.asarray(value, dtype=precision if precision != "native" else None)
        if value.shape != e.contracts[stage].shape:
            raise ValueError("GPU 模块实际输出与声明形状不符")
        return self.cache.put(key, value)

    def temporal(self, i, index, backend):
        e, xp = self.engine, backend.xp
        op = e.operations[i]; p = op["parameters"]
        size = p["window"] if op["algorithm"] == "time_smooth" else p["size"]
        first = index if op["algorithm"] == "time_smooth" else index*size
        get = lambda j: self.get(i, j, backend)
        if p["mode"] != "quantile":
            if i not in self.windows:
                self.windows[i] = RollingReduction(e.contracts[i].shape, xp)
            state = self.windows[i]
            return state.get(get, first, size, p["mode"], e.context.check_cancelled)
        scratch = self.scratch(i)
        required = max(1, int(np.prod(e.contracts[i].shape)))*size*8
        if required*8 <= scratch:
            if i not in self.windows:
                self.windows[i] = dict(first=None, values=xp.empty((size, *e.contracts[i].shape), dtype=xp.float64))
            state = self.windows[i]
            begin = state["first"]+size if state["first"] is not None and 0 <= first-state["first"] < size else first
            state["first"] = None
            for j in range(begin, first+size):
                e.context.check_cancelled()
                state["values"][j % size] = get(j)
            state["first"] = first
            e.mappings.setdefault(op["node"]["id"], {}).update(window_storage="device_ring", exact=True)
            return finite_quantile(state["values"], p["quantile"], (0,), xp)
        if i not in self.windows:
            self.windows[i] = WindowQuantile(e.contracts[i].shape, size, scratch, e.workdir, xp, lambda a:self.host(a, backend))
        e.mappings.setdefault(op["node"]["id"], {}).update(window_storage="disk_ring_gpu_blocks", exact=True)
        return self.windows[i].get(get, first, p["quantile"], e.context.check_cancelled)

    def global_fit(self, i, backend):
        if i not in self.global_values:
            e = self.engine; op = e.operations[i]
            self.global_values[i] = global_statistic(e.contracts[i], op["parameters"], lambda j:self.get(i, j, backend),
                e.context, e.workdir, xp=backend.xp, host=lambda a:self.host(a, backend),
                batch_size=self.batch_size(i, working=True), device_budget=self.scratch(i))
            e.mappings.setdefault(op["node"]["id"], {}).update(statistics_device="gpu", exact=True,
                quantile_selection="bounded_rank_selection" if op["parameters"]["mode"] == "quantile" else None)
        return self.global_values[i]

    def modifier(self, i, index, backend):
        if self.engine.operations[i]["algorithm"] == "global_statistics":
            return self.global_fit(i, backend)[0]
        if self.engine.operations[i]["algorithm"] == "distribution":
            return self.histogram(i,index,backend)
        return self.engine._modifier_value(i, index, xp=backend.xp, get_sample=lambda j:self.get(i, j, backend))

    def histogram(self,i,index,backend):
        for key in reversed(self.cache.entries):
            if key[0]=="histogram" and key[1]==i and key[2]<=index<key[3]:
                block=self.cache.get(key)
                return block[index-key[2]]
        e,xp=self.engine,backend.xp
        op=e.operations[i];p=op["parameters"]
        batch=max(1,min(self.batch_size(i,working=True),self.cache.budget//max(1,p["bins"]*8*2)))
        get=lambda j:self.get(i,j,backend)
        edges=e._histogram_edges(i,xp=xp,get_sample=get,host=lambda a:self.host(a,backend),batch_size=batch)
        stop=min(e.contracts[i].length,index+batch)
        matrix=xp.stack([get(j) for j in range(index,stop)]).reshape(stop-index,-1)
        columns=xp.searchsorted(xp.asarray(edges),matrix,side="right")-1
        columns=xp.where(matrix==edges[-1],p["bins"]-1,columns)
        valid=xp.isfinite(matrix)&(columns>=0)&(columns<p["bins"])
        ids=columns+xp.arange(stop-index,dtype=xp.int64)[:,None]*p["bins"]
        result=xp.bincount(ids[valid].astype(xp.int64),minlength=(stop-index)*p["bins"]).reshape(stop-index,p["bins"]).astype(xp.int64)
        e.mappings[op["node"]["id"]].update(histogram_batch_size=stop-index,implementation="device_bincount")
        result=self.cache.put(("histogram",i,index,stop),result)
        return result[0]

    def structural(self, i, value, backend):
        e, xp = self.engine, backend.xp
        op = e.operations[i]; a, p = op["algorithm"], op["parameters"]; c = e.contracts[i]
        if a == "roi_crop":
            from eip.module_contracts import roi_bounds
            x, y, w, h = roi_bounds(c, p)
            slices = [slice(None)]*value.ndim
            slices[c.axes.index("Y")], slices[c.axes.index("X")] = slice(y, y+h), slice(x, x+w)
            return value[tuple(slices)]
        if a == "select_channel":
            return value[0 if p["channel"] == "left" else 1]
        if a == "color_convert":
            return xp.repeat(value[..., None], 3, axis=-1) if "C" not in c.axes else value[..., ::-1] if c.color_order != p["mode"].upper() else value
        if a == "calibration":
            photo = e.instances[i]
            def apply(codes, side):
                valid = xp.isfinite(codes)
                if photo.config.saturation_code is not None:
                    valid &= codes < photo.config.saturation_code
                if photo.lut is not None:
                    if codes.dtype.kind not in "uib" or bool(xp.any((codes[valid] < 0) | (codes[valid] >= photo.lut.size))):
                        raise ValueError("LUT 仅用于覆盖全部有效整数码值的输入")
                    lut = self.constants.setdefault((i, "lut"), xp.asarray(photo.lut)) if (i, "lut") not in self.constants else self.constants[i, "lut"]
                    result = lut[xp.where(valid, codes, 0).astype(xp.int64)]
                else:
                    result = intensity_curve(codes, p["curve"], p["input_max"], p["gamma"], xp=xp)
                result = result-(p["black_left"] if side == "left" else p["black_right"])
                return xp.where(valid, result, xp.nan)
            if c.paired:
                return xp.stack([apply(arr, side) for arr, side in zip(value, ("left", "right"))])
            if p["black_left"] != p["black_right"]:
                future = [*e.nodes[i+1:], *[n for n in e.dependencies if n["enabled"]]]
                following = next(n for n in future if n["algorithm"] == "alignment")
                key = (i, "midline")
                if key not in self.constants:
                    line = np.interp(np.arange(value.shape[0])[:, None], [0, max(1,value.shape[0]-1)], following["parameters"]["midline"])
                    self.constants[key] = xp.asarray(np.arange(value.shape[1])[None, :] < line)
                return xp.where(self.constants[key], apply(value, "left"), apply(value, "right"))
            return apply(value, "left")
        if a == "alignment":
            geometry = e.instances[i]
            if i not in self.constants:
                xr, yr = xp.asarray(geometry.xr), xp.asarray(geometry.yr)
                r = geometry.config.right
                if p["interpolation"] == "nearest":
                    samples = [(xp.clip(yr.astype(xp.int64),0,r.height-1), xp.clip(xr.astype(xp.int64),0,r.width-1), None)]
                else:
                    bx, by = xp.floor(xr).astype(xp.int64), xp.floor(yr).astype(xp.int64)
                    fx, fy = xr-bx, yr-by
                    samples = [(xp.clip(by+oy,0,r.height-1), xp.clip(bx+ox,0,r.width-1), weight)
                        for ox,oy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy))]
                self.constants[i] = xp.asarray(geometry.mask), samples
            mask, samples = self.constants[i]
            l, r = geometry.config.left, geometry.config.right
            x0,y0,x1,y1 = geometry.crop
            left = value[l.y:l.y+l.height, l.x:l.x+l.width][y0:y1, x0:x1]
            right = value[r.y:r.y+r.height, r.x:r.x+r.width]
            result = xp.zeros(geometry.shape, dtype=xp.float64)
            for yy,xx,weight in samples:
                picked = right[yy,xx]
                result += picked if weight is None else xp.where(weight>0,picked*weight,0)
            return xp.stack((xp.where(mask,left,xp.nan), xp.where(mask,result,xp.nan)))
        raise ValueError("未知 GPU 结构模块")

    def record(self):
        active=any(backend.name=="gpu" for backend in self.engine.backends.values())
        return dict(self.metrics, cache_budget_bytes=self.cache_budget, cache_peak_bytes=self.cache.peak+self.upload_cache.peak,
                    source_batch_limit=32, intermediate_residency=active,
                    cache_policy="validated_flow_slack_v2",upload_dtype_policy="source_dtype",upload_alignment="sample_chunks",
                    transfer_scope="array boundaries; excludes scalar synchronization and library internals")

    def close(self):
        self.cache.clear()
        self.upload_cache.clear()
        self.constants.clear()
        self.global_values.clear()
        for state in self.windows.values():
            if hasattr(state, "close"):
                state.close()
        self.windows.clear()
