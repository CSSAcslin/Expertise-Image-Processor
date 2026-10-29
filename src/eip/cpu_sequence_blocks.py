"""Bounded scalar CPU g² delivery without per-sample NumPy array dispatch.

Python float operations use IEEE float64 in the estimator's original order.
The input is a shared, already resident (T, 2) checkpoint, never a converted or
copied substitute. Spatial maps and Q/W scans retain their ordinary executor.
"""
from __future__ import annotations

import math
import operator
import time

import numpy as np


MAX_OUTPUT_BLOCK = 256
STATE_RESERVATION = 4096


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


class CpuCorrelationBlocks:
    """Increasing scalar T outputs, with global 256-output rebasing.

    Calls must start at ``next_index``. A cancelled block leaves the previous
    committed state intact. Random and reverse requests fail before work;
    ``reset`` explicitly restarts from output zero.
    """

    def __init__(self, signals, *, window, lag, first_center, stride,
                 output_count, min_mean=1e-9, min_samples=2,
                 max_output_block=256, check=None):
        if (not isinstance(signals, np.ndarray) or signals.ndim != 2
                or signals.shape[1] != 2 or signals.shape[0] < 1):
            raise ValueError("Signals must be a NumPy array of shape (T, 2)")
        if signals.dtype != np.dtype("float64") or not signals.flags.c_contiguous:
            raise ValueError("Signals must already be contiguous float64; conversions need caller budgeting")
        self.window = _integer(window, "window", 1)
        self.lag = _integer(lag, "lag")
        self.first_center = _integer(first_center, "first_center", 0)
        self.stride = _integer(stride, "stride", 1)
        self.output_count = _integer(output_count, "output_count", 1)
        self.min_samples = _integer(min_samples, "min_samples", 1)
        self.max_output_block = _integer(max_output_block, "max_output_block", 1)
        if self.max_output_block > MAX_OUTPUT_BLOCK:
            raise ValueError(f"max_output_block must not exceed {MAX_OUTPUT_BLOCK}")
        self.min_mean = float(min_mean)
        if not math.isfinite(self.min_mean) or self.min_mean <= 0:
            raise ValueError("min_mean must be finite and positive")
        first = self.first_center-(self.window-1)//2
        stop = first+(self.output_count-1)*self.stride+self.window
        if min(first, first+self.lag) < 0 or max(stop, stop+self.lag) > len(signals):
            raise ValueError("The output schedule or signed lag exceeds the input signals")
        self.xp, self.signals = np, signals
        self.component_shape, self.pixels = (), 1
        self.check = check or (lambda: None)
        self.next_index = 0
        self._state = (0., 0., 0., 0., 0., 0.)
        self._closed = False
        self.metrics = dict(implementation="python_float64_scalar_sequence_blocks", components=1,
                            input_bytes=int(signals.nbytes), state_bytes=6*8,
                            state_reservation_bytes=STATE_RESERVATION,
                            max_output_block=self.max_output_block, output_blocks=0,
                            output_samples=0, pair_updates=0, periodic_rebuilds=0,
                            largest_output_block=0, resets=0)

    def compute(self, start, stop, *, count=False):
        """Return ``(float64_values, uint32_counts_or_None)`` for one block."""
        if self._closed:
            raise RuntimeError("Sequence blocks are closed")
        start, stop = _integer(start, "start", 0), _integer(stop, "stop", 0)
        if start != self.next_index or not start < stop <= self.output_count:
            raise ValueError("Outputs must be requested consecutively from next_index")
        size = stop-start
        if size > self.max_output_block:
            raise ValueError("Requested result block exceeds max_output_block")
        self.check()
        result = np.empty(size, dtype=np.float64)
        counts = np.empty(size, dtype=np.uint32) if count else None
        signals = self.signals
        finite, check = math.isfinite, self.check
        base_l, base_r, sum_l, sum_r, sum_lr, samples = self._state
        window, lag, stride = self.window, self.lag, self.stride
        first_center, half = self.first_center, (window-1)//2
        mean_limit, sample_limit = self.min_mean, self.min_samples
        pair_updates = rebuilds = 0
        for ordinal in range(start, stop):
            check()
            center = first_center+ordinal*stride
            begin, end = center-half, center-half+window
            rebuild = ordinal % 256 == 0 or stride >= window
            if rebuild:
                base_l, base_r = float(signals[center, 0]), float(signals[center, 1])
                if not finite(base_l):
                    base_l = 0.
                if not finite(base_r):
                    base_r = 0.
                sum_l = sum_r = sum_lr = samples = 0.
                changes = ((begin, end, 1.),)
                rebuilds += 1
            else:
                # Preserve outgoing-before-incoming and ascending pair order.
                changes = ((begin-stride, begin, -1.), (end-stride, end, 1.))
            for first, last, sign in changes:
                for index in range(first, last):
                    if pair_updates % 64 == 0:
                        check()
                    left, right = float(signals[index, 0]), float(signals[index+lag, 1])
                    valid = finite(left) and finite(right)
                    dl = left-base_l if valid else 0.
                    dr = right-base_r if valid else 0.
                    sum_l += sign*dl
                    sum_r += sign*dr
                    sum_lr += sign*(dl*dr)
                    samples += sign*valid
                    pair_updates += 1
            safe = max(samples, 1.)
            ml, mr = sum_l/safe, sum_r/safe
            il, ir = base_l+ml, base_r+mr
            covariance = sum_lr/safe-ml*mr
            denominator = il*ir
            # NumPy defines a zero denominator as a nonfinite result; Python
            # division would otherwise raise instead of yielding the same NaN.
            value = 1.+covariance/denominator if denominator else math.nan
            result[ordinal-start] = value if (samples >= sample_limit and il > mean_limit
                                              and ir > mean_limit and finite(value)) else math.nan
            if count:
                counts[ordinal-start] = int(round(samples)) & 0xffffffff
        check()
        self._state = (base_l, base_r, sum_l, sum_r, sum_lr, samples)
        self.next_index = stop
        self.metrics["output_blocks"] += 1
        self.metrics["output_samples"] += size
        self.metrics["pair_updates"] += pair_updates
        self.metrics["periodic_rebuilds"] += rebuilds
        self.metrics["largest_output_block"] = max(self.metrics["largest_output_block"], size)
        self.metrics.update(next_index=stop, state="complete" if stop == self.output_count else "active")
        return result, counts

    def reset(self):
        if self._closed:
            raise RuntimeError("Sequence blocks are closed")
        self.next_index = 0
        self._state = (0., 0., 0., 0., 0., 0.)
        self.metrics["resets"] += 1
        self.metrics.update(next_index=0, state="ready")

    def record(self):
        return dict(self.metrics, next_index=self.next_index,
                    maximum_value_block_bytes=self.max_output_block*8,
                    maximum_count_block_bytes=self.max_output_block*4,
                    accumulation="float64 fluctuations; joint finite pairs; global T ordinal rebuild every 256; no FMA",
                    scope="scalar fixed-window, fixed-signed-lag, increasing T outputs; no input copy or conversion")

    def close(self):
        self.signals = None
        self._closed = True


def cpu_engine_blocks(engine, stage):
    """Prepare an already planned CPU/RAM scalar checkpoint for delivery.

    Unsupported schedules return None before materializing input. Failures in
    actual module evaluation propagate and never trigger a hidden replay.
    The caller owns precision conversion, record delivery and runner cleanup.
    """
    stage = _integer(stage, "stage", 1)
    rows = engine.execution_metrics.setdefault("cpu_output_blocks", {})
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
    if getattr(engine, "inspection", False) or getattr(engine, "_single", False):
        return unsupported("local_inspection_uses_demand_execution")
    if (p.get("analysis") != "map" or before.sequence_axis != "T"
            or after.sequence_axis != "T" or len(op.get("lags", ())) != 1):
        return unsupported("requires_fixed_lag_sequential_T_outputs")
    centers = op.get("centers")
    if (not isinstance(centers, range) or centers.step < 1 or not len(centers)
            or after.length != len(centers)):
        return unsupported("requires_a_regular_complete_center_schedule")
    backend = engine.backends.get(i)
    if backend is None or backend.name != "cpu":
        return unsupported("correlation_is_not_on_CPU")
    store = engine.main_store
    if store is None or store.stage != i or store.mode != "ram":
        return unsupported("direct_input_has_no_RAM_checkpoint")
    if (before.shape != (2,) or after.shape != () or not before.paired
            or before.length is None or before.length < 1):
        return unsupported("requires_known_length_paired_scalars")
    if before.dtype != "float64" or np.dtype(store.dtype) != np.dtype("float64"):
        return unsupported("checkpoint_conversion_is_not_implicit")
    from eip.execution_plan import dense_signal_schedule
    if not dense_signal_schedule(before.length,p["window"],op["lags"][0],centers):
        return unsupported("sparse_windows_keep_demand_loading")
    if store.values is not None and not store.values.flags.c_contiguous:
        return unsupported("checkpoint_is_not_contiguous")
    plans = engine.resource_plan
    own = plans[i]
    working = max((r["working_bytes"] for r in plans), default=0)
    retained = max((r.get("retained_bytes", 0) for r in plans), default=0)
    nested = max((r.get("nested_bytes", 0) for r in plans), default=0)
    reserved = int(store.capacity_bytes)+int(store.length)+STATE_RESERVATION
    available = (engine.budget-working-retained-nested-engine.cache.budget
                 -engine.reader_cache_budget-engine.reference_budget-reserved)
    instance_available = int(own["limit_bytes"]*2/3)-own["working_bytes"]-STATE_RESERVATION
    # Reserve the result, optional counts, and a simultaneously cast output.
    per_output = 12+np.dtype(after.dtype).itemsize
    device_per_output = 0
    suffix_limits = []
    for k in range(stage,len(engine.operations)):
        if engine.spatial is None or not engine.spatial.can_forward(k):
            break
        plan = plans[k]
        c = engine.contracts[k+1]
        elements = max(1,int(np.prod(c.shape)))
        scratch = max(plan["working_bytes"],elements*max(8,np.dtype(c.dtype).itemsize)*4)
        per_output = max(per_output,scratch)
        suffix_limits.append(int(plan["limit_bytes"]*2/3)//max(1,scratch))
        if engine.device.backend(k+1) is not None:
            device_per_output = max(device_per_output,scratch)
    block_size = min(MAX_OUTPUT_BLOCK, len(centers), max(0, available)//per_output,
                     max(0, instance_available)//per_output)
    if suffix_limits:
        block_size = min(block_size,*suffix_limits)
    if device_per_output:
        device_plans = [p for p in plans if "device_working_bytes" in p]
        device_available = int(engine.device.budget*.75)-sum(p.get("device_retained_bytes",0) for p in device_plans)-max((p["device_working_bytes"] for p in device_plans),default=0)
        block_size = min(block_size,max(0,device_available)//device_per_output)
    if block_size < 1:
        return unsupported("insufficient_budget_for_state_and_one_output")
    if device_per_output:
        engine.device.configure_cache(extra_retained=block_size*device_per_output)
    row.update(state="preparing", reason=None, input_stage=i,
               input_bytes=int(store.capacity_bytes),
               additional_reserved_bytes=STATE_RESERVATION+block_size*per_output,
               block_size=block_size)
    started = time.perf_counter()
    engine.context.log(f"准备CPU关联小信号：{store.length} 个上游样本，输出按 {block_size} 个分块")
    try:
        last_log = started
        for index in range(0, store.length, max(1, int(store.block_size))):
            engine.context.check_cancelled()
            engine._stored_main(index)
            now = time.perf_counter()
            if now-last_log >= 1.0:
                engine.context.log(f"准备CPU关联小信号：{min(index+store.block_size, store.length)}/{store.length}")
                last_log = now
        engine.context.check_cancelled()
        if store.values is None or not store.ready.all():
            raise RuntimeError("RAM checkpoint did not fill its complete input schedule")
        runner = CpuCorrelationBlocks(store.values, window=p["window"], lag=op["lags"][0],
                                      first_center=centers.start, stride=centers.step,
                                      output_count=len(centers), min_mean=p["min_mean"],
                                      min_samples=p["min_samples"], max_output_block=block_size,
                                      check=engine.context.check_cancelled)
    except Exception as error:
        row.update(state="failed", error=f"{type(error).__name__}: {error}"[:512],
                   preparation_seconds=time.perf_counter()-started)
        raise
    runner.stage = stage
    runner.metrics.update(row, state="ready", preparation_seconds=time.perf_counter()-started)
    rows[str(stage)] = runner.metrics
    return runner
