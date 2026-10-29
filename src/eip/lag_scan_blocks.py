"""Budgeted fixed-centre Q blocks over an owned compact paired signal.

Different signed delays are independent outputs. Each delay still accumulates
the same float64 fluctuations in increasing sample order; Q is never treated
as a moving T window. The adapter shares an existing checkpoint and prepares
only the union of samples actually consumed by its configured delays.
"""
from __future__ import annotations

from functools import lru_cache
import math
import operator
import time

import numpy as np

KERNEL_NAME = "eip_g2_lag_block_float64"
MAX_OUTPUT_BLOCK = 256
MAX_COMPONENTS = 4096
MAX_SERIAL_PAIR_UPDATES = 65536
MAX_LAUNCH_PAIR_UPDATES = 8_388_608
STATE_RESERVATION = 16_384

_SOURCE = r'''
extern "C" __global__ void eip_g2_lag_block_float64(
    const double* signals, const long long* lags, double* output,
    unsigned int* counts, const long long pixels, const long long first_q,
    const long long stop_q, const long long center, const long long first,
    const long long stop, const double min_mean, const long long min_samples,
    const int save_counts) {
    const long long element=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (element >= (stop_q-first_q)*pixels) return;
    const long long pixel=element%pixels;
    const long long q=first_q+element/pixels;
    const long long lag=lags[q];
    double bl=signals[center*2*pixels+pixel];
    double br=signals[center*2*pixels+pixels+pixel];
    if (!isfinite(bl)) bl=0.;
    if (!isfinite(br)) br=0.;
    double sl=0.,sr=0.,slr=0.,n=0.;
    for (long long j=first;j<stop;++j) {
        const double left=signals[j*2*pixels+pixel];
        const double right=signals[(j+lag)*2*pixels+pixels+pixel];
        const bool valid=isfinite(left)&&isfinite(right);
        const double dl=valid?left-bl:0.;
        const double dr=valid?right-br:0.;
        sl+=dl; sr+=dr; slr+=dl*dr; n+=valid?1.:0.;
    }
    const double safe=n>1.?n:1.;
    const double ml=sl/safe,mr=sr/safe;
    const double il=bl+ml,ir=br+mr;
    const double value=1.+(slr/safe-ml*mr)/(il*ir);
    output[element]=(n>=min_samples&&il>min_mean&&ir>min_mean&&isfinite(value))?value:nan("");
    if (save_counts) counts[element]=(unsigned int)n;
}
'''


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError(f"{name} is outside the int64 range")
    return value


@lru_cache(maxsize=1)
def _kernel(cp):
    return cp.RawKernel(_SOURCE, KERNEL_NAME, options=("--fmad=false",))


def required_intervals(center, window, lags):
    """Return disjoint, sorted upstream sample ranges, never their full hull."""
    first = center-(window-1)//2
    stop = first+window
    intervals = [(center, center+1), (first, stop)]
    intervals.extend((first+lag, stop+lag) for lag in lags)
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


class LagScanBlocks:
    """Independent signed-delay blocks with ordered CPU/CUDA float64 sums.

    Input must already be contiguous float64 ``(T,2,*components)`` in its
    actual execution device. No input conversion, copying or upload is hidden
    here. Arbitrary/repeated block requests are valid and have no shared sums.
    """

    def __init__(self, signals, backend=None, *, window, center, lags,
                 min_mean=1e-9, min_samples=2, max_output_block=256, check=None):
        self.xp = np if backend is None else backend.xp
        self.device = getattr(self.xp, "__name__", None) == "cupy"
        if self.device:
            if type(signals).__module__.split(".")[0] != "cupy":
                raise ValueError("CUDA lag blocks require a real CuPy signal")
        elif self.xp is not np or not isinstance(signals, np.ndarray):
            raise ValueError("CPU lag blocks require a NumPy signal")
        if signals.ndim < 2 or signals.shape[0] < 1 or signals.shape[1] != 2:
            raise ValueError("Signals must have shape (T,2,*components)")
        if np.dtype(signals.dtype) != np.dtype("float64") or not signals.flags.c_contiguous:
            raise ValueError("Signals must already be contiguous float64; conversions need budgeting")
        self.component_shape = tuple(signals.shape[2:])
        self.pixels = math.prod(self.component_shape)
        if not 1 <= self.pixels <= MAX_COMPONENTS:
            raise ValueError(f"Lag blocks support 1..{MAX_COMPONENTS} components")
        self.window = _integer(window, "window", 1)
        self.center = _integer(center, "center", 0)
        self.lags = tuple(_integer(q, "lag") for q in lags)
        self.output_count = len(self.lags)
        if not 1 <= self.output_count <= 4096:
            raise ValueError("Lag blocks require 1..4096 delays")
        self.max_output_block = _integer(max_output_block, "max_output_block", 1)
        if self.max_output_block > MAX_OUTPUT_BLOCK:
            raise ValueError(f"max_output_block must not exceed {MAX_OUTPUT_BLOCK}")
        if (self.window > MAX_SERIAL_PAIR_UPDATES
                or self.window*self.pixels > MAX_LAUNCH_PAIR_UPDATES):
            raise ValueError("Lag blocks exceed the bounded pair work limit")
        self.max_output_block = min(self.max_output_block, self.output_count,
                                    MAX_LAUNCH_PAIR_UPDATES//(self.window*self.pixels))
        self.min_samples = _integer(min_samples, "min_samples", 1)
        self.min_mean = float(min_mean)
        if not math.isfinite(self.min_mean) or self.min_mean <= 0:
            raise ValueError("min_mean must be finite and positive")
        self.first = self.center-(self.window-1)//2
        self.stop = self.first+self.window
        intervals = required_intervals(self.center, self.window, self.lags)
        if intervals[0][0] < 0 or intervals[-1][1] > len(signals):
            raise ValueError("The fixed center or signed delays exceed input samples")
        self.signals, self.backend = signals, backend
        self.check = check or (lambda: None)
        self._closed = False
        self._produced = np.zeros(self.output_count, dtype=np.bool_)
        self._timer = getattr(backend, "gpu_timer", None)
        self._kernel = _kernel(self.xp) if self.device else None
        self._device_lags = self.xp.asarray(np.asarray(self.lags, dtype=np.int64)) if self.device else None
        self.metrics = dict(implementation="cupy_lag_blocks_float64" if self.device else "numpy_ordered_lag_blocks_float64",
                            kernel=KERNEL_NAME if self.device else None,
                            components=self.pixels, input_bytes=int(signals.nbytes),
                            output_count=self.output_count, max_output_block=self.max_output_block,
                            output_blocks=0, output_samples=0, pair_updates=0,
                            component_pair_updates=0, kernel_calls=0, largest_output_block=0,
                            unique_output_samples=0, closed=False,
                            numerical_scratch_bytes=0 if self.device else self.window*self.pixels*80)

    def _launch(self, result, counts, start, stop, save_counts):
        size = (stop-start)*self.pixels
        self._kernel(((size+127)//128,), (128,),
                     (self.signals, self._device_lags, result, counts, np.int64(self.pixels),
                      np.int64(start), np.int64(stop), np.int64(self.center), np.int64(self.first),
                      np.int64(self.stop), np.float64(self.min_mean), np.int64(self.min_samples),
                      np.int32(save_counts)))

    def compute(self, start, stop, *, count=False):
        if self._closed:
            raise RuntimeError("Lag blocks are closed")
        start, stop = _integer(start, "start", 0), _integer(stop, "stop", 0)
        if not start < stop <= self.output_count:
            raise ValueError("Invalid Q block range")
        if stop-start > self.max_output_block:
            raise ValueError("Requested Q block exceeds max_output_block")
        self.check()
        result = self.xp.empty((stop-start, *self.component_shape), dtype=self.xp.float64)
        counts = self.xp.empty(result.shape, dtype=self.xp.uint32) if count else None
        if self.device:
            args = (result, counts if count else result, start, stop, bool(count))
            if self._timer is None:
                self._launch(*args)
            else:
                self._timer.launch(KERNEL_NAME, self._launch, *args)
        else:
            base = np.nan_to_num(self.signals[self.center], nan=0., posinf=0., neginf=0.)
            left = self.signals[self.first:self.stop, 0]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                for out, ordinal in enumerate(range(start, stop)):
                    self.check()
                    lag = self.lags[ordinal]
                    right = self.signals[self.first+lag:self.stop+lag, 1]
                    valid = np.isfinite(left)&np.isfinite(right)
                    dl = np.where(valid, left-base[0], 0.)
                    dr = np.where(valid, right-base[1], 0.)
                    # Last cumsum retains ascending-sample additions; sum may
                    # reassociate and ordinary chunk partial sums change order.
                    sl = np.cumsum(dl, axis=0, dtype=np.float64)[-1]
                    sr = np.cumsum(dr, axis=0, dtype=np.float64)[-1]
                    slr = np.cumsum(dl*dr, axis=0, dtype=np.float64)[-1]
                    n = np.count_nonzero(valid, axis=0)
                    safe = np.maximum(n, 1)
                    ml, mr = sl/safe, sr/safe
                    il, ir = base[0]+ml, base[1]+mr
                    value = 1.+(slr/safe-ml*mr)/(il*ir)
                    good = (n >= self.min_samples)&(il > self.min_mean)&(ir > self.min_mean)&np.isfinite(value)
                    result[out] = np.where(good, value, np.nan)
                    if count:
                        counts[out] = np.asarray(n).astype(np.uint32)
        # Cancellation leaves metrics uncommitted; no shared numerical state
        # exists to poison a retried block or random request.
        self.check()
        self.metrics["output_blocks"] += 1
        self.metrics["output_samples"] += stop-start
        self.metrics["largest_output_block"] = max(self.metrics["largest_output_block"], stop-start)
        self.metrics["pair_updates"] += (stop-start)*self.window
        self.metrics["component_pair_updates"] += (stop-start)*self.window*self.pixels
        self.metrics["kernel_calls"] += int(self.device)
        self._produced[start:stop] = True
        self.metrics["unique_output_samples"] = int(np.count_nonzero(self._produced))
        self.metrics["state"] = "complete" if self._produced.all() else "active"
        return result, counts

    def record(self):
        return dict(self.metrics, maximum_value_block_bytes=self.max_output_block*self.pixels*8,
                    maximum_count_block_bytes=self.max_output_block*self.pixels*4,
                    accumulation="float64 fluctuations; joint finite pairs; increasing W order; no FMA",
                    scope="fixed-center independent signed Qs; shared resident input; no input conversion")

    def close(self):
        self.signals = self._device_lags = None
        self._produced = None
        self._closed = True
        self.metrics["closed"] = True


def lag_engine_blocks(engine, stage):
    """Return an eligible CPU/CUDA Q runner; reject before materializing input."""
    stage = _integer(stage, "stage", 1)
    rows = engine.execution_metrics.setdefault("lag_output_blocks", {})
    row = dict(state="ineligible", output_stage=stage)
    rows[str(stage)] = row

    def unsupported(reason):
        row["reason"] = reason
        return None

    if stage > len(engine.operations):
        return unsupported("unknown_output_stage")
    i = stage-1
    op, p = engine.operations[i], engine.operations[i]["parameters"]
    if op["algorithm"] not in {"correlation", "correlation_map", "correlation_curve"}:
        return unsupported("not_a_fixed_window_correlation")
    before, after = engine.contracts[i], engine.contracts[stage]
    if getattr(engine, "inspection", False):
        return unsupported("local_inspection_uses_demand_execution")
    if p.get("analysis") != "curve" or before.sequence_axis != "T" or after.sequence_axis != "Q":
        return unsupported("requires_fixed_center_Q_outputs")
    centers, lags = op.get("centers"), op.get("lags", ())
    if centers is None or len(centers) != 1 or not lags or after.length != len(lags):
        return unsupported("requires_one_center_and_complete_Q_schedule")
    backend = engine.backends.get(i)
    gpu = backend is not None and backend.name == "gpu"
    if backend is None or gpu and getattr(backend.xp, "__name__", None) != "cupy":
        return unsupported("correlation_backend_is_not_native_CPU_or_CUDA")
    store = engine.main_store
    if store is None or store.stage != i or store.mode != ("device" if gpu else "ram"):
        return unsupported("direct_input_has_no_native_resident_checkpoint")
    if not before.paired or not before.shape or before.shape[0] != 2 or not before.length:
        return unsupported("requires_known_length_paired_signals")
    if before.dtype != "float64" or np.dtype(store.dtype) != np.dtype("float64"):
        return unsupported("checkpoint_conversion_is_not_implicit")
    pixels = math.prod(before.shape[1:])
    if not 1 <= pixels <= MAX_COMPONENTS:
        return unsupported("component_count_exceeds_small_signal_limit")
    if store.values is not None and not store.values.flags.c_contiguous:
        return unsupported("checkpoint_is_not_contiguous")
    window = p["window"]
    if window > MAX_SERIAL_PAIR_UPDATES or window*pixels > MAX_LAUNCH_PAIR_UPDATES:
        return unsupported("window_pair_work_exceeds_bounded_block_limit")
    intervals = required_intervals(centers[0], window, lags)
    if intervals[0][0] < 0 or intervals[-1][1] > before.length:
        return unsupported("Q_schedule_exceeds_upstream_samples")

    plans = engine.resource_plan
    own = plans[i]
    working = max((r["working_bytes"] for r in plans), default=0)
    retained = max((r.get("retained_bytes", 0) for r in plans), default=0)
    nested = max((r.get("nested_bytes", 0) for r in plans), default=0)
    host_available = (engine.budget-working-retained-nested-engine.cache.budget
                      -engine.reader_cache_budget-engine.reference_budget-int(store.length)
                      -(0 if gpu else int(store.capacity_bytes)))
    device_plans = [r for r in plans if "device_working_bytes" in r]
    device_available = (int(engine.device.budget*.75)
                        -sum(r.get("device_retained_bytes", 0) for r in device_plans)
                        -max((r["device_working_bytes"] for r in device_plans), default=0)
                        -(int(store.capacity_bytes) if gpu else 0))
    persistent = STATE_RESERVATION+len(lags)*16+pixels*16
    numerical_scratch = 0 if gpu else window*pixels*80
    instance_available = (int(own["limit_bytes"]*2/3)
                          -max(own["working_bytes"], own.get("device_working_bytes", 0))
                          -persistent-numerical_scratch)
    # Simultaneous float64 result, optional count, output cast and host transfer.
    output_bytes = pixels*(12+np.dtype(after.dtype).itemsize)
    host_per_output = output_bytes
    device_per_output = output_bytes if gpu else 0
    suffix_limits = []
    for j in range(stage, len(engine.operations)):
        if engine.spatial is None or not engine.spatial.can_forward(j):
            return unsupported("following_modules_do_not_support_block_forwarding")
        plan = plans[j]
        value_bytes = max(max(1, math.prod(c.shape))*max(8, np.dtype(c.dtype).itemsize)
                          for c in engine.contracts[j:j+2])
        scratch = max(plan["working_bytes"], plan.get("device_working_bytes", 0), value_bytes*4)
        host_per_output = max(host_per_output, scratch)
        suffix_limits.append(int(plan["limit_bytes"]*2/3)//max(1, scratch))
        if engine.device.backend(j+1) is not None:
            device_per_output = max(device_per_output, scratch)
    host_available -= persistent+numerical_scratch
    if gpu or device_per_output:
        device_available -= persistent
    block_size = min(MAX_OUTPUT_BLOCK, len(lags), MAX_LAUNCH_PAIR_UPDATES//(window*pixels),
                     max(0, host_available)//max(1, host_per_output),
                     max(0, instance_available)//max(1, output_bytes),
                     *(suffix_limits or [MAX_OUTPUT_BLOCK]))
    if device_per_output:
        block_size = min(block_size, max(0, device_available)//device_per_output)
    if block_size < 1:
        return unsupported("insufficient_budget_for_state_and_one_output")
    device_reservation = persistent+block_size*device_per_output if device_per_output else 0
    if device_reservation:
        engine.device.configure_cache(extra_retained=(int(store.capacity_bytes) if gpu else 0)+device_reservation)
    row.update(state="preparing", reason=None, input_stage=i, input_bytes=int(store.capacity_bytes),
               required_intervals=intervals, required_samples=sum(b-a for a,b in intervals),
               block_size=block_size, additional_reserved_bytes=persistent+numerical_scratch+block_size*host_per_output,
               host_bytes_per_output=host_per_output, device_bytes_per_output=device_per_output,
               numerical_scratch_bytes=numerical_scratch, suffix_stage_count=len(engine.operations)-stage,
               serial_pair_work_limit=MAX_SERIAL_PAIR_UPDATES, launch_pair_work_limit=MAX_LAUNCH_PAIR_UPDATES)
    started = time.perf_counter()
    engine.context.log(f"准备延迟扫描信号：{row['required_samples']} 个必需样本，按 {block_size} 个Q分块")
    try:
        # Each request stays bounded; StageStore expands only its own physical
        # preparation blocks. Gaps between disjoint lag windows remain unread.
        for first, stop in intervals:
            for begin in range(first, stop, store.block_size):
                engine.context.check_cancelled()
                engine._stored_range(begin, min(stop, begin+store.block_size),
                                     device_backend=store.backend if gpu else None)
        engine.context.check_cancelled()
        if store.values is None or any(not store.ready[a:b].all() for a,b in intervals):
            raise RuntimeError("Checkpoint did not fill the required signed-Q sample union")
        runner = LagScanBlocks(store.values, backend if gpu else None, window=window,
                               center=centers[0], lags=lags, min_mean=p["min_mean"],
                               min_samples=p["min_samples"], max_output_block=block_size,
                               check=engine.context.check_cancelled)
    except Exception as error:
        row.update(state="failed", error=f"{type(error).__name__}: {error}"[:512],
                   preparation_seconds=time.perf_counter()-started)
        raise
    runner.stage = stage
    runner.metrics.update(row, state="ready", preparation_seconds=time.perf_counter()-started)
    rows[str(stage)] = runner.metrics
    engine.mappings[op["node"]["id"]]["correlation_implementation"] = runner.metrics["implementation"]
    return runner
