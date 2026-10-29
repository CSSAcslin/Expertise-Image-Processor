"""Bounded sequential CUDA g² output blocks for resident small paired signals.

This is a physical execution adapter for the existing fluctuation estimator.
Every component has one thread and retains the original sample/update order;
there is no time reduction, precision downgrade or implicit host transfer.
The caller owns input residency, resource accounting and the output schedule.
Q/W scans and random access deliberately stay with their ordinary executor.
"""
from __future__ import annotations

from functools import lru_cache
import operator
import time

import numpy as np


KERNEL_NAME = "eip_g2_sequence_block_float64"
MAX_COMPONENTS = 4096
MAX_OUTPUT_BLOCK = 4096
# Conservative scheduling limits, not a measured wall-time guarantee. A thread
# processes one component's time updates serially; a launch also has a bounded
# total amount of paired work across all its components.
MAX_SERIAL_PAIR_UPDATES = 65_536
MAX_LAUNCH_PAIR_UPDATES = 8_388_608

_SOURCE = r'''
extern "C" __global__ void eip_g2_sequence_block_float64(
    const double* signals, double* state, double* result,
    unsigned int* counts, const long long pixels,
    const long long first, const long long stop,
    const long long first_center, const long long stride,
    const long long window, const long long lag,
    const double min_mean, const long long min_samples,
    const int save_counts)
{
    const long long pixel = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    double base_l = 0.0, base_r = 0.0;
    double sum_l = 0.0, sum_r = 0.0, sum_lr = 0.0, samples = 0.0;
    if (first > 0) {
        base_l = state[pixel];
        base_r = state[pixels + pixel];
        sum_l = state[2*pixels + pixel];
        sum_r = state[3*pixels + pixel];
        sum_lr = state[4*pixels + pixel];
        samples = state[5*pixels + pixel];
    }
    const long long half = (window - 1) / 2;
    for (long long ordinal = first; ordinal < stop; ++ordinal) {
        const long long center = first_center + ordinal * stride;
        const long long start = center - half;
        const long long end = start + window;
        const bool rebuild = ordinal == 0 || ordinal % 256 == 0 || stride >= window;
        if (rebuild) {
            base_l = signals[(center*2)*pixels + pixel];
            base_r = signals[(center*2+1)*pixels + pixel];
            base_l = isfinite(base_l) ? base_l : 0.0;
            base_r = isfinite(base_r) ? base_r : 0.0;
            sum_l = sum_r = sum_lr = samples = 0.0;
        }
        // Match ordinary correlate(): outgoing samples first, then incoming.
        // On rebuild there is only the ascending full-window addition.
        const int operations = rebuild ? 1 : 2;
        for (int operation = 0; operation < operations; ++operation) {
            const long long begin = rebuild ? start :
                operation == 0 ? start-stride : end-stride;
            const long long finish = rebuild ? end :
                operation == 0 ? start : end;
            const double sign = !rebuild && operation == 0 ? -1.0 : 1.0;
            for (long long sample = begin; sample < finish; ++sample) {
                const double left = signals[(sample*2)*pixels + pixel];
                const double right = signals[((sample+lag)*2+1)*pixels + pixel];
                const bool valid = isfinite(left) && isfinite(right)
                    && isfinite(base_l) && isfinite(base_r);
                const double dl = valid ? left-base_l : 0.0;
                const double dr = valid ? right-base_r : 0.0;
                sum_l += sign * dl;
                sum_r += sign * dr;
                sum_lr += sign * (dl * dr);
                samples += sign * (valid ? 1.0 : 0.0);
            }
        }
        const double safe = samples > 1.0 ? samples : 1.0;
        const double ml = sum_l / safe;
        const double mr = sum_r / safe;
        const double il = base_l + ml;
        const double ir = base_r + mr;
        const double covariance = sum_lr / safe - ml * mr;
        const double value = 1.0 + covariance / (il * ir);
        const long long output = (ordinal-first)*pixels + pixel;
        result[output] = samples >= min_samples && il > min_mean && ir > min_mean
            && isfinite(value) ? value : nan("");
        if (save_counts) counts[output] = (unsigned int)rint(samples);
    }
    state[pixel] = base_l;
    state[pixels + pixel] = base_r;
    state[2*pixels + pixel] = sum_l;
    state[3*pixels + pixel] = sum_r;
    state[4*pixels + pixel] = sum_lr;
    state[5*pixels + pixel] = samples;
}
'''


@lru_cache(maxsize=1)
def _kernel(cp):
    return cp.RawKernel(_SOURCE, KERNEL_NAME, options=("--fmad=false",))


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if not -(1 << 63) <= result < (1 << 63):
        raise ValueError(f"{name} is outside the int64 range")
    return result


class GpuCorrelationBlocks:
    """Produce bounded, increasing T output ranges without host synchronization.

    ``signals`` must already be a contiguous float64 CuPy array of shape
    ``(T, 2, *component_shape)``. No input copy or type conversion is hidden in
    this class. ``compute`` returns device arrays; the caller may download a
    whole result block and then deliver its individual source records.

    Each call starts at ``next_index``. ``reset`` discards rolling state and
    permits recomputation from output zero; unsupported random requests raise
    before execution rather than silently using an incompatible accumulator.
    """

    def __init__(self, signals, backend, *, window, lag, first_center, stride,
                 output_count, min_mean=1e-9, min_samples=2,
                 max_output_block=256, check=None):
        cp = backend.xp
        if (getattr(cp, "__name__", None) != "cupy"
                or type(signals).__module__.split(".")[0] != "cupy"):
            raise ValueError("Sequence blocks require a real CuPy device array")
        if signals.ndim < 2 or signals.shape[1] != 2 or signals.shape[0] < 1:
            raise ValueError("Signals must have shape (T, 2, *components)")
        if np.dtype(signals.dtype) != np.dtype("float64") or not signals.flags.c_contiguous:
            raise ValueError("Signals must already be contiguous float64; conversions need caller budgeting")
        self.component_shape = tuple(signals.shape[2:])
        self.pixels = int(np.prod(self.component_shape, dtype=np.int64)) if self.component_shape else 1
        if not 1 <= self.pixels <= MAX_COMPONENTS:
            raise ValueError(f"Sequence blocks support 1..{MAX_COMPONENTS} components")
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
        if not np.isfinite(self.min_mean) or self.min_mean <= 0:
            raise ValueError("min_mean must be finite and positive")
        first = self.first_center-(self.window-1)//2
        stop = first+(self.output_count-1)*self.stride+self.window
        # Validate the complete schedule before allocating or launching. This
        # also keeps int64 pointer arithmetic within the actual array bounds.
        if min(first, first+self.lag) < 0 or max(stop, stop+self.lag) > signals.shape[0]:
            raise ValueError("The output schedule or signed lag exceeds the input signals")
        self.xp, self.backend, self.signals = cp, backend, signals
        self.check = check or (lambda: None)
        self._timer = getattr(backend, "gpu_timer", None)
        self._kernel = _kernel(cp)
        self.state = cp.empty((6, *self.component_shape), dtype=cp.float64)
        self.next_index = 0
        self._closed = False
        self.metrics = dict(implementation="cupy_sequence_blocks_float64",
                            kernel=KERNEL_NAME, components=self.pixels,
                            input_bytes=int(signals.nbytes), state_bytes=int(self.state.nbytes),
                            max_output_block=self.max_output_block, output_blocks=0,
                            output_samples=0, pair_updates=0, periodic_rebuilds=0,
                            largest_output_block=0, resets=0)

    def _launch(self, signals, state, result, counts, first, stop, save_counts):
        self._kernel(((self.pixels+127)//128,), (128,),
                     (signals, state, result, counts, np.int64(self.pixels),
                      np.int64(first), np.int64(stop), np.int64(self.first_center),
                      np.int64(self.stride), np.int64(self.window), np.int64(self.lag),
                      np.float64(self.min_mean), np.int64(self.min_samples),
                      np.int32(save_counts)))

    def compute(self, start, stop, *, count=False):
        """Return ``(device_values, device_counts_or_None)`` for ``[start, stop)``.

        The maximum live result buffer is bounded by ``max_output_block`` and
        the component shape. Numerical CUDA errors propagate to the caller.
        No intermediate state is copied back to the host.
        """
        if self._closed:
            raise RuntimeError("Sequence blocks are closed")
        start, stop = _integer(start, "start", 0), _integer(stop, "stop", 0)
        if start != self.next_index or not start < stop <= self.output_count:
            raise ValueError("Outputs must be requested consecutively from next_index")
        size = stop-start
        if size > self.max_output_block:
            raise ValueError("Requested result block exceeds max_output_block")
        self.check()
        result = self.xp.empty((size, *self.component_shape), dtype=self.xp.float64)
        # The kernel never dereferences this pointer without save_counts. Use
        # an existing device pointer to avoid an otherwise unused allocation.
        counts = self.xp.empty(result.shape, dtype=self.xp.uint32) if count else None
        args = (self.signals, self.state, result,
                counts if count else result, np.int64(start), np.int64(stop), np.int32(bool(count)))
        if self._timer is None:
            self._launch(*args)
        else:
            self._timer.launch(KERNEL_NAME, self._launch, *args)
        self.next_index = stop
        rebuilds = ((stop-1)//256-(start-1)//256) if self.stride < self.window else size
        self.metrics["output_blocks"] += 1
        self.metrics["output_samples"] += size
        self.metrics["largest_output_block"] = max(self.metrics["largest_output_block"], size)
        self.metrics["pair_updates"] += rebuilds*self.window+(size-rebuilds)*2*self.stride
        self.metrics["periodic_rebuilds"] += rebuilds
        self.metrics["next_index"] = stop
        self.metrics["state"] = "complete" if stop == self.output_count else "active"
        return result, counts

    def reset(self):
        if self._closed:
            raise RuntimeError("Sequence blocks are closed")
        self.next_index = 0
        self.metrics["resets"] += 1
        self.metrics.update(next_index=0, state="ready")

    def record(self):
        return dict(self.metrics, next_index=self.next_index,
                    maximum_value_block_bytes=self.max_output_block*self.pixels*8,
                    maximum_count_block_bytes=self.max_output_block*self.pixels*4,
                    accumulation="float64 fluctuations; joint finite pairs; global T ordinal rebuild every 256; no FMA",
                    scope="fixed-window, fixed-signed-lag, increasing T outputs; no input conversion or host transfer")

    def close(self):
        self.state = self.signals = None
        self._closed = True


def engine_blocks(engine, stage):
    """Prepare an eligible engine checkpoint for bounded device output blocks.

    ``stage`` is the one-based output stage; its input checkpoint must already
    be planned as device-resident. Eligibility and memory checks happen before
    filling that checkpoint. Unsupported plans return ``None`` with a reason
    in ``execution_metrics['gpu_output_blocks']``. Errors while evaluating
    actual input samples propagate normally and never trigger a silent replay.

    The caller owns the returned accumulator and must close it after delivery.
    It also owns node-precision conversion and the actual host transfer. The
    complete compact input is shared directly with ``StageStore``.
    """
    stage = _integer(stage, "stage", 1)
    rows = engine.execution_metrics.setdefault("gpu_output_blocks", {})
    row = dict(state="ineligible", output_stage=stage)
    rows[str(stage)] = row

    def unsupported(reason):
        row["reason"] = reason
        return None

    if stage > len(engine.operations):
        return unsupported("unknown_output_stage")
    i = stage-1
    op = engine.operations[i]
    p = op["parameters"]
    if op["algorithm"] not in {"correlation", "correlation_map", "correlation_curve"}:
        return unsupported("not_a_fixed_window_correlation")
    before, after = engine.contracts[i], engine.contracts[stage]
    if (getattr(engine, "inspection", False) or getattr(engine, "_single", False)):
        return unsupported("local_inspection_uses_demand_execution")
    if (p.get("analysis") != "map" or before.sequence_axis != "T"
            or after.sequence_axis != "T" or len(op.get("lags", ())) != 1):
        return unsupported("requires_fixed_lag_sequential_T_outputs")
    centers = op.get("centers")
    if (not isinstance(centers, range) or centers.step < 1 or not len(centers)
            or after.length != len(centers)):
        return unsupported("requires_a_regular_complete_center_schedule")
    backend = engine.backends.get(i)
    if (backend is None or backend.name != "gpu"
            or getattr(backend.xp, "__name__", None) != "cupy"):
        return unsupported("correlation_is_not_on_real_CUDA")
    store = engine.main_store
    if store is None or store.stage != i or store.mode != "device":
        return unsupported("direct_input_has_no_device_checkpoint")
    if (not before.paired or not before.shape or before.shape[0] != 2
            or before.length is None or before.length < 1):
        return unsupported("requires_known_length_paired_signals")
    if (before.dtype != "float64" or np.dtype(store.dtype) != np.dtype("float64")):
        return unsupported("checkpoint_conversion_is_not_implicit")
    from eip.execution_plan import dense_signal_schedule
    if not dense_signal_schedule(before.length,p["window"],op["lags"][0],centers):
        return unsupported("sparse_windows_keep_demand_loading")
    component_shape = tuple(before.shape[1:])
    pixels = int(np.prod(component_shape, dtype=np.int64)) if component_shape else 1
    if not 1 <= pixels <= MAX_COMPONENTS:
        return unsupported("component_count_exceeds_small_signal_limit")
    if store.values is not None and not store.values.flags.c_contiguous:
        return unsupported("checkpoint_is_not_contiguous")

    work_limit = min(MAX_SERIAL_PAIR_UPDATES, MAX_LAUNCH_PAIR_UPDATES//pixels)
    window, stride = p["window"], centers.step
    if window > work_limit:
        return unsupported("rebuild_pair_work_exceeds_bounded_launch_limit")
    if stride >= window:
        work_block_size = work_limit//window
    else:
        # A block may include one periodic rebuild or no rebuild at all. When
        # 2*stride > window the all-rolling block costs more, so reserve the
        # larger first-output cost instead of assuming every block rebuilds.
        first_work = max(window, 2*stride)
        if first_work > work_limit:
            return unsupported("rolling_pair_work_exceeds_bounded_launch_limit")
        work_block_size = 1+(work_limit-first_work)//(2*stride)

    # Reserve explicit state and bounded output on top of the full input store.
    # Counting the ordinary estimator's retained reservation as well is
    # conservative and keeps this adapter independent of allocation internals.
    plans = engine.resource_plan
    device_plans = [r for r in plans if "device_working_bytes" in r]
    if not device_plans:
        return unsupported("device_resource_plan_is_missing")
    device_retained = sum(r.get("device_retained_bytes", 0) for r in device_plans)
    device_working = max(r["device_working_bytes"] for r in device_plans)
    device_available = (int(engine.device.budget*.75)-device_retained
                        -device_working-int(store.capacity_bytes))
    state_bytes = 6*pixels*8
    output_item = np.dtype(after.dtype).itemsize
    # An explicit output cast can coexist with the float64 result buffer.
    correlation_device_per_output = pixels*(8+(output_item if after.dtype != "float64" else 0))
    device_per_output = correlation_device_per_output
    host_working = max((r["working_bytes"] for r in plans), default=0)
    host_retained = max((r.get("retained_bytes", 0) for r in plans), default=0)
    host_nested = max((r.get("nested_bytes", 0) for r in plans), default=0)
    host_available = (engine.budget-host_working-host_retained-host_nested
                      -engine.cache.budget-engine.reader_cache_budget
                      -engine.reference_budget-int(store.length))
    # Allow both downloaded values and a separately cast host block.
    host_per_output = pixels*(8+output_item)
    # The caller may forward each result block through a pointwise suffix.
    # A logical one-sample resource estimate must be multiplied by the physical
    # batch length: RGB expansion, float64 intermediates and CPU/GPU boundaries
    # can require much more than the original scalar g² result buffer.
    suffix_indices = list(range(stage, len(engine.operations)))
    spatial = getattr(engine, "spatial", None)
    suffix_contiguous = all(spatial is not None and spatial.can_forward(j)
                            for j in suffix_indices)
    suffix_device_work = suffix_host_work = 0
    largest_value = pixels*max(8, output_item)
    suffix_instance_block = 256
    for j in suffix_indices:
        plan = plans[j]
        contracts = engine.contracts[j:j+2]
        value_bytes = max(max(1, int(np.prod(c.shape)))
                          *max(8, np.dtype(c.dtype).itemsize if c.dtype != "dynamic" else 8)
                          for c in contracts)
        largest_value = max(largest_value, value_bytes)
        working = max(int(plan["working_bytes"]), value_bytes*2)
        destination = engine.device.backend(j+1)
        if destination is None:
            suffix_host_work = max(suffix_host_work, working)
        else:
            suffix_device_work = max(suffix_device_work,
                                     int(plan.get("device_working_bytes", working)), working)
        # This is conservative for a non-forwardable remainder; ordinary
        # callers only use the adapter when their complete suffix is supported.
        suffix_instance_block = min(suffix_instance_block,
                                    int(int(plan["limit_bytes"])*2/3)//max(1, working))
    if suffix_indices:
        device_per_output += suffix_device_work+largest_value*2
        host_per_output = max(host_per_output, suffix_host_work+largest_value*2)
    instance_limit = int(plans[i]["limit_bytes"])
    own = plans[i]
    instance_working = max(own["working_bytes"], own.get("device_working_bytes", 0))
    instance_available = int(instance_limit*2/3)-instance_working-state_bytes
    block_size = min(256, len(centers), work_block_size, suffix_instance_block,
                     max(0, device_available-state_bytes)//max(1, device_per_output),
                     max(0, host_available)//max(1, host_per_output),
                     max(0, instance_available)//max(1, correlation_device_per_output))
    if block_size < 1:
        return unsupported("insufficient_budget_for_state_and_one_output")
    reserved = state_bytes+block_size*device_per_output
    engine.device.configure_cache(extra_retained=int(store.capacity_bytes)+reserved)
    row.update(state="preparing", reason=None, input_stage=i,
               input_bytes=int(store.capacity_bytes), additional_reserved_bytes=reserved,
               block_size=block_size,
               suffix_stage_count=len(suffix_indices),
               suffix_budget_policy="validated_forward_chain" if suffix_contiguous else "conservative_remaining_flow",
               suffix_device_working_bytes_per_sample=suffix_device_work,
               suffix_host_working_bytes_per_sample=suffix_host_work,
               device_bytes_per_output=device_per_output,
               host_bytes_per_output=host_per_output,
               host_block_reserved_bytes=block_size*host_per_output,
               serial_pair_work_limit=MAX_SERIAL_PAIR_UPDATES,
               launch_pair_work_limit=MAX_LAUNCH_PAIR_UPDATES,
               estimated_pair_updates_per_component=(block_size*window if stride >= window
                    else max(window, 2*stride)+(block_size-1)*2*stride))
    started = time.perf_counter()
    engine.context.log(f"准备GPU关联小信号：{store.length} 个上游样本，输出按 {block_size} 个分块")
    try:
        stride = max(1, int(store.block_size))
        last_log = started
        for index in range(0, store.length, stride):
            engine.context.check_cancelled()
            engine._stored_main(index, backend)
            now = time.perf_counter()
            if now-last_log >= 1.0:
                engine.context.log(f"准备GPU关联小信号：{min(index+stride, store.length)}/{store.length}")
                last_log = now
        engine.context.check_cancelled()
        if store.values is None or not store.ready.all():
            raise RuntimeError("Device checkpoint did not fill its complete input schedule")
        runner = GpuCorrelationBlocks(store.values, backend, window=p["window"],
                                      lag=op["lags"][0], first_center=centers.start,
                                      stride=centers.step, output_count=len(centers),
                                      min_mean=p["min_mean"], min_samples=p["min_samples"],
                                      max_output_block=block_size,
                                      check=engine.context.check_cancelled)
    except Exception as error:
        row.update(state="failed", error=f"{type(error).__name__}: {error}"[:512],
                   preparation_seconds=time.perf_counter()-started)
        raise
    runner.stage = stage
    runner.metrics.update(row, state="ready", preparation_seconds=time.perf_counter()-started)
    rows[str(stage)] = runner.metrics
    return runner
