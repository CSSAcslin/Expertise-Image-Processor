"""Bounded CUDA-event timings of explicitly identified fused g² launches.

An event interval measures a stream region, including launch submission gaps.
It is not a CUPTI kernel activity duration, nor the sum of all GPU work. In
particular these regions exclude conversions, transfers and other CuPy kernels.
"""
from __future__ import annotations

from collections import deque


KERNEL_NAMES = (
    "eip_moments_add_float64",
    "eip_moments_add_many_float64",
    "eip_moments_result_float64",
    "eip_moments_result_only_float64",
    "eip_g2_sequence_block_float64",
    "eip_g2_lag_block_float64",
)


class GpuEventTimer:
    """Never synchronize in ``launch`` or retain an unbounded event history.

    First calls of each argument layout execute normally without timing: CuPy
    can compile on that call, leaving the device idle inside an event interval.
    The layout registry and pending event pairs both have hard bounds. If the
    pool is busy we omit that measurement and report the omission, rather than
    blocking computation or extrapolating an unmeasured duration.
    """

    def __init__(self, xp, *, max_pending=128, max_signatures=256,
                 initial_calls=8, sample_every=32):
        if min(max_pending,max_signatures,initial_calls,sample_every) < 1:
            raise ValueError("GPU timing bounds must be positive")
        self.xp = xp
        self.max_pending, self.max_signatures = int(max_pending), int(max_signatures)
        self.initial_calls, self.sample_every = int(initial_calls), int(sample_every)
        self._pending, self._free, self._signatures = deque(), [], set()
        self._allocated = self._peak = 0
        self._closed = self._launching = False
        self._error = None
        self._rows = {name: dict(calls=0, timed_calls=0, seconds=0.0,
                                warmup_excluded_calls=0, capacity_excluded_calls=0,
                                signature_excluded_calls=0, nested_excluded_calls=0,
                                monitoring_excluded_calls=0,
                                sampling_excluded_calls=0,
                                failed_calls=0) for name in KERNEL_NAMES}

    @staticmethod
    def _layout(args):
        # No pointers or sample indices: identical layouts reuse the warm flag.
        return tuple((v.dtype.str, tuple(v.shape), tuple(v.strides))
                     if hasattr(v, "dtype") and hasattr(v, "strides")
                     else type(v) for v in args)

    def _remember_error(self, error):
        if self._error is None:
            # Tracebacks can retain input arrays / an entire engine. Persist a
            # bounded message, not the exception and its captured working set.
            self._error = f"{type(error).__name__}: {error}"[:512]

    def _drain(self):
        while self._pending:
            start, stop, name, stream_id = self._pending[0]
            if not stop.done:
                break
            seconds = float(self.xp.cuda.get_elapsed_time(start, stop)) / 1000
            if not 0 <= seconds < float("inf"):
                raise ValueError("CUDA event returned an invalid duration")
            row = self._rows[name]
            row["seconds"] += seconds
            row["timed_calls"] += 1
            self._pending.popleft()
            self._free.append((start, stop))

    def launch(self, name, kernel, *args):
        if name not in self._rows:
            raise ValueError("Only registered fused g² launches can be timed")
        if self._closed:
            raise RuntimeError("GPU timing was already finalized")
        row = self._rows[name]
        row["calls"] += 1
        if self._launching:
            # The production sites call one ElementwiseKernel, never callbacks.
            # Keep this guard so future wrappers cannot double-count nesting.
            row["nested_excluded_calls"] += 1
            return self._call(row,kernel,args)
        self._launching = True
        try:
            if self._error is not None:
                row["monitoring_excluded_calls"] += 1
                return self._call(row,kernel,args)
            if (row["calls"] > self.initial_calls
                    and (row["calls"]-self.initial_calls) % self.sample_every):
                # Check cadence before inspecting layouts or touching CUDA.
                # Per-launch events can dominate short fused-kernel workloads.
                row["sampling_excluded_calls"] += 1
                return self._call(row,kernel,args)
            signature = (name, self._layout(args))
            if signature not in self._signatures:
                if len(self._signatures) < self.max_signatures:
                    row["warmup_excluded_calls"] += 1
                    result = self._call(row,kernel,args)
                    self._signatures.add(signature)
                    return result
                row["signature_excluded_calls"] += 1
                return self._call(row,kernel,args)
            # Query only at a small cadence or when a new event pair is needed.
            # No per-launch synchronization, even when the pool is exhausted.
            try:
                if self._pending and (len(self._pending) >= 32 or not self._free):
                    self._drain()
                if not self._free and self._allocated < self.max_pending:
                    self._free.append((self.xp.cuda.Event(), self.xp.cuda.Event()))
                    self._allocated += 1
                pair = self._free.pop() if self._free else None
                if pair is not None:
                    start, stop = pair
                    stream = self.xp.cuda.get_current_stream()
                    # Arguments and kernel lookup have already been evaluated.
                    start.record(stream)
            except Exception as error:
                self._remember_error(error)
                row["monitoring_excluded_calls"] += 1
                return self._call(row,kernel,args)
            if pair is None:
                row["capacity_excluded_calls"] += 1
                return self._call(row,kernel,args)
            result = self._call(row,kernel,args)
            try:
                stop.record(stream)
            except Exception as error:
                self._remember_error(error)
                row["monitoring_excluded_calls"] += 1
                return result
            self._pending.append((start, stop, name, stream.ptr))
            self._peak = max(self._peak, len(self._pending))
            return result
        finally:
            self._launching = False

    def _call(self, row, kernel, args):
        try:
            return kernel(*args)
        except Exception as error:
            # Actual numerical kernel errors still propagate unchanged.
            row["failed_calls"] += 1
            self._remember_error(error)
            raise

    def finish(self, *, raise_errors=False):
        if not self._closed:
            try:
                # The last event of each stream covers its earlier intervals.
                # Wait only at finalization; at most max_pending events exist.
                last = {item[3]: item[1] for item in self._pending}
                for event in last.values():
                    event.synchronize()
                self._drain()
            except Exception as error:
                self._remember_error(error)
            finally:
                self._closed = True
        if self._error is not None and raise_errors:
            raise RuntimeError(f"GPU 设备计时未完整完成：{self._error}")

    def snapshot(self):
        if not self._closed and self._error is None:
            try:
                self._drain()
            except Exception as error:
                # Progress / failure snapshots must preserve the original
                # computation exception. The monitoring error is saved below.
                self._remember_error(error)
        keys = ("calls", "timed_calls", "seconds", "warmup_excluded_calls",
                "capacity_excluded_calls", "signature_excluded_calls",
                "nested_excluded_calls", "monitoring_excluded_calls",
                "sampling_excluded_calls", "failed_calls")
        totals = {key: sum(row[key] for row in self._rows.values()) for key in keys}
        seconds = totals.pop("seconds")
        return dict(method="cuda_events", scope="g2_fused_kernel_regions",
                    status="error" if self._error else "finalized" if self._closed else "collecting",
                    device_seconds=seconds if totals["timed_calls"] else None,
                    **totals, pending_calls=len(self._pending),
                    event_pair_limit=self.max_pending, peak_pending_pairs=self._peak,
                    allocated_event_pairs=self._allocated,
                    signature_limit=self.max_signatures, warmed_signatures=len(self._signatures),
                    initial_calls_per_kernel=self.initial_calls, sample_every=self.sample_every,
                    kernels={name: dict(row) for name, row in self._rows.items()},
                    error=self._error,
                    semantics=("Sum of sampled CUDA-event stream intervals around the registered fused g2 kernel calls, including bounded sequential output blocks. Includes event/launch gaps; not pure kernel activity time or full-pipeline GPU time. First call of each dtype/shape/stride layout is excluded to avoid JIT compilation gaps. "
                               f"Each kernel's first {self.initial_calls} calls are candidates; thereafter every {self.sample_every} calls has one candidate, reducing measurement overhead. "
                               "Transfers, conversions, other CuPy kernels and excluded calls are not measured. No extrapolation. Multiple stream intervals may overlap; do not add to exclusive host seconds."))
