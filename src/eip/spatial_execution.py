"""Bounded range execution derived from logical module contracts.

Regions use (x, y, width, height) in the requested stage's coordinates.  A
physical batch axis never changes a public contract.  Unsupported operations
are barriers: obtain their real complete output before selecting a region.
"""
from collections import OrderedDict
from contextlib import nullcontext
import numpy as np

from eip.gpu_ops import element_operation
from eip.module_contracts import roi_bounds
from eip.module_registry import MODIFIERS
from eip.module_statistics import reduce_finite, retained_axes


SELECTORS = frozenset({"roi_crop", "spatial_roi", "point_extract", "line_extract", "axis_slice", "paired_trace"})
POINTWISE = frozenset({"intensity", "affine", "curve", "gain_offset", "threshold", "fixed_normalize", "grayscale", "color_convert"})


def full_bounds(c):
    return (0, 0, c.shape[c.axes.index("X")], c.shape[c.axes.index("Y")]) if c.spatial else None


def select_region(value, c, bounds, *, batched=True):
    if bounds is None:
        return value
    x, y, w, h = bounds
    slices = [slice(None)] * value.ndim
    offset = int(batched)
    slices[offset+c.axes.index("Y")] = slice(y, y+h)
    slices[offset+c.axes.index("X")] = slice(x, x+w)
    return value[tuple(slices)]


class SpatialExecution:
    def __init__(self, engine):
        self.e = engine
        from eip.execution_plan import capability_for
        self.capabilities = [capability_for(dict(op["node"],parameters=op["parameters"]),engine.contracts[i]) for i,op in enumerate(engine.operations)]
        self.active = 0
        self.evaluated = set()
        self.raw = OrderedDict()
        self.raw_bytes = 0
        self.raw_limit = max(1, engine.budget//32)
        self.temporal = OrderedDict()
        self.temporal_active = set()
        self.temporal_bytes = 0
        self.temporal_limit = max(1, engine.budget//64)
        self.metrics = dict(schema_version=1, blocks=0, samples=0, source_reads=0,
                            source_reuses=0, region_requests=0, region_elements={},
                            region_evaluations={},
                            full_barriers={}, uploaded_bytes=0, downloaded_bytes=0,
                            raw_peak_bytes=0, temporal_peak_bytes=0)
        self.metrics.update(geometry_map_uploads=0,geometry_map_upload_bytes=0,
                            compact_copies=0,ownership_transfers=0)

    def _compact(self, value):
        owner = value
        while getattr(owner,"base",None) is not None and hasattr(owner.base,"nbytes"):
            owner = owner.base
        if owner.nbytes > value.nbytes or not value.flags.c_contiguous:
            self.metrics["compact_copies"] += 1
            return value.copy()
        self.metrics["ownership_transfers"] += 1
        return value

    def batch_size(self, stage):
        if self.e.inspection or self.e._single and self.e.output.sequence_axis=="T":
            return 1
        # Decode buffers and nested physical arrays coexist. Keep a conservative
        # working share separate from the existing cache and retained states.
        peak = max([max(1, int(np.prod(c.shape)))*max(8, np.dtype(c.dtype).itemsize)
                    for c in self.e.contracts[:stage+1] if c.dtype != "dynamic"] or [1])
        allowance = self.e.budget//32
        plans = self.e.resource_plan[:stage]
        if plans:
            allowance = min(allowance, min(int(r["limit_bytes"])//8 for r in plans))
        if any(self.e.device.backend(s) is not None for s in range(1, stage+1)):
            allowance = min(allowance, self.e.device.budget//16)
        return max(1, min(32, allowance//max(1, peak*4)))

    def can_select(self, i, *, side=False):
        op = self.e.operations[i]
        a, p = op["algorithm"], op["parameters"]
        return (a in SELECTORS and self.e.contracts[i].spatial
                and (side or p.get("output", "main") == "main")
                and (a != "axis_slice" or p["axis"] in {"X", "Y"}))

    def can_forward(self, i):
        op = self.e.operations[i]
        a, p = op["algorithm"],op["parameters"]
        return (not op["node"].get("manifest") and self.capabilities[i].domain_pushdown
                and (a in POINTWISE or a == "axis_slice" and p["output"] == "main" and p["axis"] != self.e.contracts[i].sequence_axis))

    def prefer_sample_streaming(self):
        """Large CPU full frames gain nothing from an extra batch allocation.

        Preserve streaming for local shape-preserving chains. Small signals,
        spatial selections/reductions, temporal dependencies and device paths
        still use their own planned adapters.
        """
        c = self.e.contracts[0]
        if not c.spatial or c.dtype == "dynamic" or int(np.prod(c.shape))*np.dtype(c.dtype).itemsize < 1024**2:
            return False
        return bool(self.capabilities) and all(
            cap.spatial_dependency in {"pointwise","neighborhood"}
            and cap.temporal_dependency == "local" and not cap.side_effect
            and self.e.device.backend(i+1) is None
            for i,cap in enumerate(self.capabilities))

    def forward(self, first_stage, stop_stage, values, native):
        """Apply a validated per-sample suffix, preserving device boundaries."""
        for i in range(first_stage,stop_stage):
            if not self.can_forward(i):
                raise ValueError("此后处理模块不支持连续输出块")
            op,c = self.e.operations[i],self.e.contracts[i]
            destination = self.e.device.backend(i+1)
            values = self._convert(values,native,destination,len(values))
            native = destination
            xp = np if native is None else native.xp
            with self.e.profiler.measure("cpu_compute" if native is None else "gpu_submit") if self.e.profiler else nullcontext():
                if op["algorithm"] == "axis_slice":
                    values = xp.take(values,op["parameters"]["index"],axis=1+c.axes.index(op["parameters"]["axis"]))
                else:
                    values = self._element(i,values,xp,device=native is not None)
                precision = op["node"].get("execution",{}).get("precision","native")
                if precision != "native":
                    values = xp.asarray(values,dtype=precision)
            counts = self.e.execution_metrics["stage_evaluations"]
            counts[str(i+1)] = counts.get(str(i+1),0)+len(values)
        return self._convert(values,native,None,len(values))

    def evaluate(self, i, index, backend=None):
        if backend is not None and not self.e.inspection and (not self.e._single or self.e.output.sequence_axis!="T"):
            stage = i+1
            size = self.batch_size(stage)
            sample_bytes = max(1, int(np.prod(self.e.contracts[stage].shape)))*8
            size = max(1, min(size, self.e.device.upload_cache.budget//max(1, sample_bytes*2)))
            start = index//size*size
            stop = min(self.e.contracts[stage].length, start+size)
            key = ("spatial_output", stage, start, stop)
            hit = self.e.device.upload_cache.get(key)
            if hit is None:
                hit = self.compute(stage, range(start, stop), destination=backend)
                hit = self.e.device.upload_cache.put(key, hit)
            return hit[index-start]
        return self.compute(i+1, [index], destination=backend)[0]

    def modifier(self, i, index, backend=None):
        self.active += 1
        try:
            return self._selector(i, (index,), backend)[0]
        finally:
            self.active -= 1
            if not self.active:
                self.raw.clear()
                self.raw_bytes = 0
                self.evaluated.clear()

    def compute(self, stage, indices, bounds=None, destination=None):
        indices = tuple(indices)
        if not indices:
            raise ValueError("物理执行批次不能为空")
        outer = not self.active
        self.active += 1
        try:
            if outer:
                self.metrics["blocks"] += 1
                self.metrics["samples"] += len(indices)
            return self._compute(stage, indices, bounds, destination)
        finally:
            self.active -= 1
            if outer:
                self.raw.clear()
                self.raw_bytes = 0
                self.evaluated.clear()

    def _convert(self, value, native, destination, count):
        if native is destination or native is not None and destination is not None:
            return value
        profiler = self.e.profiler
        if native is not None:
            self.metrics["downloaded_bytes"] += value.nbytes
            return self.e.device.host(value, native)
        if destination is not None:
            with profiler.measure("upload") if profiler else nullcontext():
                result = destination.xp.asarray(value)
            metrics = self.e.device.metrics
            metrics["upload_batches"] += 1
            metrics["uploaded_samples"] += count
            metrics["largest_upload_batch"] = max(metrics["largest_upload_batch"], count)
            self.metrics["uploaded_bytes"] += value.nbytes
            return result
        return value

    def _source(self, index):
        if index in self.raw:
            self.metrics["source_reuses"] += 1
            self.raw.move_to_end(index)
            return self.raw[index]
        value = self.e.source.get(index)
        if tuple(value.shape) != self.e.contracts[0].shape:
            raise ValueError("来源帧尺寸发生变化")
        self.metrics["source_reads"] += 1
        counts = self.e.execution_metrics["stage_evaluations"]
        counts["0"] = counts.get("0", 0)+1
        if value.nbytes <= self.raw_limit:
            while self.raw and self.raw_bytes+value.nbytes > self.raw_limit:
                self.raw_bytes -= self.raw.popitem(last=False)[1].nbytes
            self.raw[index] = value
            self.raw_bytes += value.nbytes
            self.metrics["raw_peak_bytes"] = max(self.metrics["raw_peak_bytes"], self.raw_bytes)
        return value

    def _barrier(self, stage, indices, bounds, native, reason):
        key = f"{stage}:{reason}"
        self.metrics["full_barriers"][key] = self.metrics["full_barriers"].get(key, 0)+len(indices)
        c = self.e.contracts[stage]
        xp = np if native is None else native.xp
        values = []
        for j in indices:
            self.e.context.check_cancelled()
            cache_key = ("barrier", stage, j)
            value = self.raw.get(cache_key) if native is None else None
            if value is None:
                value = self.e.get(stage, j) if native is None else self.e.device.get(stage, j, native)
                if native is None and value.nbytes <= self.raw_limit:
                    while self.raw and self.raw_bytes+value.nbytes > self.raw_limit:
                        self.raw_bytes -= self.raw.popitem(last=False)[1].nbytes
                    self.raw[cache_key] = value
                    self.raw_bytes += value.nbytes
                    self.metrics["raw_peak_bytes"] = max(self.metrics["raw_peak_bytes"], self.raw_bytes)
            else:
                self.raw.move_to_end(cache_key)
            values.append(select_region(value, c, bounds, batched=False).copy())
        return xp.stack(values)

    def _compute(self, stage, indices, bounds, destination):
        e = self.e
        e.context.check_cancelled()
        c = e.contracts[stage]
        if any(j < 0 or j >= c.length for j in indices):
            raise IndexError("编辑后的样本号超出范围")
        if bounds is not None:
            x, y, w, h = bounds
            if not c.spatial or min(x, y) < 0 or min(w, h) < 1 or x+w > c.shape[c.axes.index("X")] or y+h > c.shape[c.axes.index("Y")]:
                raise ValueError("物理请求超出模块空间范围")
            self.metrics["region_requests"] += 1
        if stage == 0:
            # Copy straight into the compact block. A list of tiny views would
            # retain evicted frames; a list of copies adds a second block copy.
            sample = select_region(self._source(indices[0]),c,bounds,batched=False)
            value = np.empty((len(indices),*sample.shape),dtype=sample.dtype)
            value[0] = sample
            for k,j in enumerate(indices[1:],1):
                sample = select_region(self._source(j),c,bounds,batched=False)
                if sample.shape != value.shape[1:] or sample.dtype != value.dtype:
                    raise ValueError("来源帧形状或类型发生变化")
                value[k] = sample
            return self._convert(value, None, destination, len(indices))
        i = stage-1
        op = e.operations[i]
        a, p, node = op["algorithm"], op["parameters"], op["node"]
        before = e.contracts[i]
        native = e.device.backend(stage)
        xp = np if native is None else native.xp
        get = lambda js=indices, region=bounds: self._compute(i, js, region, native)
        cap = self.capabilities[i]
        # Side-effect/full-input validation is preserved by the canonical path.
        if node.get("manifest") or not cap.domain_pushdown and bounds is not None:
            value = self._barrier(stage, indices, bounds, native, cap.barrier_reason or "complete_input_dependency")
            return self._convert(value, native, destination, len(indices))
        category = "gpu_submit" if native is not None else "cpu_compute"
        with e.profiler.measure(category) if e.profiler else nullcontext():
            if a == "time_crop":
                selected = tuple(op["indices"][j] if "indices" in op else j+p["first"]-1 for j in indices)
                value = get(selected)
            elif a == "time_sample":
                value = get(tuple(p["offset"]+j*p["step"] for j in indices))
            elif a in {"roi_crop", "spatial_roi"}:
                rx, ry, rw, rh = roi_bounds(before, p)
                if p.get("mode", "pixels") == "pixels":
                    x, y, w, h = bounds or (0, 0, rw, rh)
                    value = self._compact(get(region=(rx+x, ry+y, w, h)))
                else:
                    value = self._compact(get(region=(rx, ry, rw, rh)))
                    if before.paired and p["validity"] == "joint":
                        value = xp.where(xp.isfinite(value).all(axis=1)[:, None, ...], value, xp.nan)
                    axes = tuple(1+before.axes.index(axis) for axis in "YX")
                    value = reduce_finite(value, axes, p["mode"], p["quantile"], xp)[0]
            elif a in MODIFIERS and p["output"] == "side":
                value = get()
            elif self.can_select(i):
                value = self._selector(i, indices, native)
                value = select_region(value, c, bounds)
            elif a == "alignment":
                value = self._alignment(i, indices, bounds or full_bounds(c), native)
            elif a == "select_channel":
                value = get()[:, 0 if p["channel"] == "left" else 1]
            elif a == "calibration" and not cap.full_input_validation and (before.paired or p["black_left"] == p["black_right"]):
                value = get()
                if native is None:
                    photo = e.instances[i]
                    value = np.stack([photo.apply(value[:, k], side) for k, side in enumerate(("left", "right"))], axis=1) if before.paired else photo.apply(value, "left")
                else:
                    # The shared pointwise primitive retains invalid/saturation
                    # semantics; batching never reinterprets B as a colour axis.
                    from eip.module_ops import intensity_curve
                    def photo(arr, side):
                        valid = xp.isfinite(arr)
                        if p["saturation_code"] is not None:
                            valid &= arr < p["saturation_code"]
                        result = intensity_curve(arr, p["curve"], p["input_max"], p["gamma"], xp=xp)
                        return xp.where(valid, result-p["black_left" if side == "left" else "black_right"], xp.nan)
                    value = xp.stack([photo(value[:, k], side) for k, side in enumerate(("left", "right"))], axis=1) if before.paired else photo(value, "left")
            elif a in POINTWISE and (a != "fixed_normalize" or cap.domain_pushdown):
                value = self._element(i, get(), xp, device=native is not None)
            elif a in {"gaussian", "spatial_smooth"} and (a == "gaussian" or p["method"] == "gaussian"):
                value = self._gaussian(i, indices, bounds or full_bounds(c), native)
            elif a in {"time_statistics", "region_reduce"} and p["output"] == "main":
                values = get(region=None)
                if before.paired and p["validity"] == "joint":
                    values = xp.where(xp.isfinite(values).all(axis=1)[:, None, ...], values, xp.nan)
                axes = tuple(1+k for k,axis in enumerate(before.axes) if axis not in retained_axes(before.axes,"space_time")) if a == "time_statistics" else tuple(1+before.axes.index(axis) for axis in "YX")
                value = reduce_finite(values,axes,p["mode"],p["quantile"],xp)[0]
            elif a in {"time_smooth", "time_bin"} and bounds is not None:
                value = self._temporal(i, indices, bounds, native)
            else:
                value = self._barrier(stage, indices, bounds, native, cap.barrier_reason or "canonical_execution")
                return self._convert(value, native, destination, len(indices))
            precision = node.get("execution", {}).get("precision", "native")
            if precision != "native" and a not in {"data_tap", "parameter_binding"} and not (a in MODIFIERS and p["output"] == "side"):
                value = xp.asarray(value, dtype=precision)
        expected = list(c.shape)
        if bounds is not None:
            expected[c.axes.index("Y")], expected[c.axes.index("X")] = bounds[3], bounds[2]
        if value.shape != (len(indices), *expected):
            raise ValueError(f"物理执行输出 {value.shape} 与模块契约 {tuple(expected)} 不符")
        counts = e.execution_metrics["stage_evaluations"]
        fresh = [(stage, j) for j in indices if (stage, j) not in self.evaluated]
        self.evaluated.update(fresh)
        counts[str(stage)] = counts.get(str(stage), 0)+len(fresh)
        regions = self.metrics["region_evaluations"]
        regions[str(stage)] = regions.get(str(stage), 0)+len(indices)
        elements = self.metrics["region_elements"]
        elements[str(stage)] = elements.get(str(stage), 0)+value.size
        return self._convert(value, native, destination, len(indices))

    def _element(self, i, value, xp, *, device=False):
        op, c = self.e.operations[i], self.e.contracts[i]
        a, p, n = op["algorithm"], op["parameters"], op["node"]
        if value.dtype.kind in "ui" and value.dtype.itemsize > 4 and (a not in {"fixed_normalize", "grayscale", "color_convert"} or device and a == "grayscale"):
            raise ValueError("内置数值算法暂不处理64位整数；请显式提供精度策略")
        if a == "fixed_normalize":
            scale = np.asarray(p["scale"], dtype=np.float64)
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("固定尺度必须为有限正数")
        if a == "color_convert":
            return xp.repeat(value[..., None], 3, axis=-1) if "C" not in c.axes else value[..., ::-1].copy() if c.color_order != p["mode"].upper() else value
        def apply(arr):
            if a == "threshold":
                f = xp.asarray(arr, dtype=xp.float64)
                if "C" in c.axes:
                    f = f[..., :3].mean(axis=-1)
                return (f >= p["threshold"]).astype(xp.uint8)
            return element_operation(a, arr, p, xp, color_order=c.color_order,
                                     preserve_input=a == "gain_offset" and n.get("execution", {}).get("precision", "native") == "native")
        if c.paired and a not in {"fixed_normalize", "grayscale"}:
            return xp.stack([apply(value[:, k]) if n["target"] in {"both", side} else value[:, k] for k, side in enumerate(("left", "right"))], axis=1)
        return apply(value)

    def _alignment(self, i, indices, bounds, backend):
        e = self.e
        xp = np if backend is None else backend.xp
        # Two spatial requests must never replay an unknown/stateful producer
        # for the same sample. A bounded LRU cannot guarantee this when a whole
        # frame exceeds its capacity. Execute the complete join exactly once.
        blocking = next((cap for cap in self.capabilities[:i] if not cap.domain_pushdown or cap.side_effect or cap.full_input_validation), None)
        if blocking is not None:
            return self._barrier(i+1, indices, bounds, backend, "upstream_complete_or_stateful_dependency:"+(blocking.barrier_reason or "side_effect"))
        mapping = e.instances[i].region_mapping(bounds)
        key = ("spatial_geometry",i)
        constants = e.device.constants.get(key) if backend is not None else None
        if constants is None or constants["bounds"] != bounds:
            with e.profiler.measure("upload") if backend is not None and e.profiler else nullcontext():
                constants = dict(bounds=bounds,mask=xp.asarray(mapping.mask),
                                 left=tuple(xp.asarray(a) for a in mapping.left_indices),
                                 right=tuple((xp.asarray(y),xp.asarray(x),None if weight is None else xp.asarray(weight)) for y,x,weight in mapping.right_indices))
            if backend is not None:
                e.device.constants[key] = constants
                self.metrics["geometry_map_uploads"] += 1
                self.metrics["geometry_map_upload_bytes"] += (constants["mask"].nbytes+sum(a.nbytes for a in constants["left"])
                    +sum(y.nbytes+x.nbytes+(0 if w is None else w.nbytes) for y,x,w in constants["right"]))
        left_source = self._compute(i, indices, mapping.left_bounds, backend)
        left = left_source[:, constants["left"][0], constants["left"][1]]
        # Request the union of the right interpolation neighbours only once.
        source = self._compute(i, indices, mapping.right_bounds, backend)
        right = xp.zeros((len(indices), *mapping.shape), dtype=xp.float64)
        for yy, xx, weight in constants["right"]:
            picked = source[:, yy, xx]
            if weight is None and backend is None:
                right = xp.asarray(picked,dtype=xp.float64)
            else:
                right += picked if weight is None else xp.where(weight > 0, picked*weight, 0)
        mask = constants["mask"]
        # NumPy advanced indexing can place B last in memory even when it is
        # first logically. Preserve the ordinary contiguous PYX sample layout
        # so a later finite spatial reduction retains its summation order.
        result = xp.empty((len(indices),2,*mapping.shape),dtype=xp.float64)
        result[:,0] = xp.where(mask,left,xp.nan)
        result[:,1] = xp.where(mask,right,xp.nan)
        return result

    def _selector(self, i, indices, backend):
        e = self.e
        op, c = e.operations[i], e.contracts[i]
        a, p = op["algorithm"], op["parameters"]
        xp = np if backend is None else backend.xp
        h, w = c.shape[c.axes.index("Y")], c.shape[c.axes.index("X")]
        if a == "point_extract" or a == "paired_trace" and p["selection"] == "point":
            bounds = (p["x"], p["y"], 1, 1)
        elif a == "paired_trace":
            bounds = roi_bounds(c, p)
        elif a == "line_extract":
            if p["mode"] == "axis":
                bounds = (0, p["index"], w, 1) if p["axis"] == "X" else (p["index"], 0, 1, h)
            else:
                # Keep the original linspace rounding before subtracting the
                # integer tile origin. Recreating linspace in local coordinates
                # can change nearest-neighbour choices on half-pixel ties.
                ys = xp.linspace(p["y0"], p["y1"], p["samples"])
                xs = xp.linspace(p["x0"], p["x1"], p["samples"])
                x0, y0 = int(np.floor(min(p["x0"], p["x1"]))), int(np.floor(min(p["y0"], p["y1"])))
                x1, y1 = min(w-1, int(np.floor(max(p["x0"], p["x1"])))+1), min(h-1, int(np.floor(max(p["y0"], p["y1"])))+1)
                values = self._compute(i, indices, (x0, y0, x1-x0+1, y1-y0+1), backend)
                matrix = xp.moveaxis(values, (1+c.axes.index("Y"), 1+c.axes.index("X")), (-2, -1))
                if p["interpolation"] == "nearest":
                    line = matrix[..., xp.floor(ys+.5).astype(int)-y0, xp.floor(xs+.5).astype(int)-x0]
                else:
                    iy, ix = xp.floor(ys).astype(int), xp.floor(xs).astype(int)
                    dy, dx = ys-iy, xs-ix
                    iy1, ix1 = xp.minimum(iy+1, h-1), xp.minimum(ix+1, w-1)
                    line = xp.zeros((*matrix.shape[:-2], p["samples"]), dtype=xp.float64)
                    valid = xp.ones_like(line, dtype=bool)
                    for yy, xx, weight in ((iy, ix, (1-dy)*(1-dx)), (iy, ix1, (1-dy)*dx), (iy1, ix, dy*(1-dx)), (iy1, ix1, dy*dx)):
                        sample = matrix[..., yy-y0, xx-x0]
                        valid &= xp.isfinite(sample) | (weight == 0)
                        line += xp.where(weight == 0, 0, sample*weight)
                    line = xp.where(valid, line, xp.nan)
                return xp.moveaxis(line, -1, 1+c.axes.index("Y"))
        elif a == "axis_slice" and p["axis"] in "YX":
            bounds = (0, p["index"], w, 1) if p["axis"] == "Y" else (p["index"], 0, 1, h)
        else:
            return self._barrier(i+1, indices, None, backend, "nonspatial_axis_selection")
        values = self._compute(i, indices, bounds, backend)
        if a == "paired_trace":
            if p["validity"] == "joint":
                values = xp.where(xp.isfinite(values).all(axis=1)[:,None,...],values,xp.nan)
            return xp.asarray(values[:,:,0,0],dtype=xp.float64) if p["selection"] == "point" else reduce_finite(values,(2,3),p["mode"],p["quantile"],xp)[0]
        if a == "point_extract":
            slices = [slice(None)]*values.ndim
            slices[1+c.axes.index("Y")],slices[1+c.axes.index("X")] = 0,0
            return values[tuple(slices)]
        removed = ("Y" if p["axis"] == "X" else "X") if a == "line_extract" else p["axis"]
        return xp.take(values,0,axis=1+c.axes.index(removed))

    def _gaussian(self, i, indices, bounds, backend):
        from eip.module_ops import gaussian
        from eip.gpu_ops import preserve
        op, c = self.e.operations[i], self.e.contracts[i]
        p, n, a = op["parameters"], op["node"], op["algorithm"]
        xp = np if backend is None else backend.xp
        x, y, w, h = bounds
        width, height = c.shape[c.axes.index("X")], c.shape[c.axes.index("Y")]
        r = p["radius"]
        x0, y0, x1, y1 = max(0, x-r), max(0, y-r), min(width, x+w+r), min(height, y+h+r)
        values = self._compute(i, indices, (x0, y0, x1-x0, y1-y0), backend)
        def apply(arr):
            if arr.dtype.kind in "ui" and arr.dtype.itemsize > 4:
                raise ValueError("内置算法暂不处理64位整数；请接入声明精度策略的专业插件")
            result = gaussian(arr, p["sigma"], r, xp=xp)
            return preserve(result, arr.dtype, xp) if a == "spatial_smooth" and n.get("execution", {}).get("precision", "native") == "native" else result
        result = xp.stack([xp.stack([apply(arr) if n["target"] in {"both", side} else arr for arr, side in zip(v, ("left", "right"))]) if c.paired else apply(v) for v in values])
        return select_region(result, c, (x-x0, y-y0, w, h)).copy()

    def _temporal(self, i, indices, bounds, backend):
        from eip.window_statistics import RollingReduction, WindowQuantile
        op, c = self.e.operations[i], self.e.contracts[i]
        p = op["parameters"]
        xp = np if backend is None else backend.xp
        shape = list(c.shape)
        shape[c.axes.index("Y")], shape[c.axes.index("X")] = bounds[3], bounds[2]
        required = max(1, int(np.prod(shape)))*32
        quantile = p["mode"] == "quantile"
        size = p["window"] if op["algorithm"] == "time_smooth" else p["size"]
        scratch = max(16384, self.temporal_limit//4)
        if quantile:
            if size*8*(4 if backend is None else 8) > scratch:
                return self._barrier(i+1, indices, bounds, backend, "regional_quantile_scratch_budget")
            required += scratch
        key = (i, bounds, backend is not None)
        if required > self.temporal_limit:
            return self._barrier(i+1, indices, bounds, backend, "regional_temporal_state_budget")
        if key not in self.temporal:
            while self.temporal and self.temporal_bytes+required > self.temporal_limit:
                removable = next((old for old in self.temporal if old not in self.temporal_active), None)
                if removable is None:
                    return self._barrier(i+1, indices, bounds, backend, "active_nested_temporal_state_budget")
                state, allocation = self.temporal.pop(removable)
                if hasattr(state, "close"):
                    state.close()
                self.temporal_bytes -= allocation
            state = WindowQuantile(tuple(shape), size, scratch, self.e.workdir, xp,
                                   np.asarray if backend is None else lambda v: self.e.device.host(v, backend)) if p["mode"] == "quantile" else RollingReduction(tuple(shape), xp)
            self.temporal[key] = (state, required)
            self.temporal_bytes += required
            self.metrics["temporal_peak_bytes"] = max(self.metrics["temporal_peak_bytes"], self.temporal_bytes)
        self.temporal.move_to_end(key)
        state, _ = self.temporal[key]
        results = []
        self.temporal_active.add(key)
        try:
            for j in indices:
                first = j if op["algorithm"] == "time_smooth" else j*size
                get = lambda k: self._compute(i, (k,), bounds, backend)[0]
                value = state.get(get, first, p["quantile"], self.e.context.check_cancelled) if quantile else state.get(get, first, size, p["mode"], self.e.context.check_cancelled)
                results.append(value)
        finally:
            self.temporal_active.remove(key)
        return xp.stack(results)

    def record(self):
        return dict(self.metrics, raw_limit_bytes=self.raw_limit, temporal_limit_bytes=self.temporal_limit)

    def close(self):
        self.raw.clear()
        for state, _ in self.temporal.values():
            if hasattr(state, "close"):
                state.close()
        self.temporal.clear()
