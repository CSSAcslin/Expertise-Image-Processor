"""Bounded, exclusive host-side timings for a single source execution.

Nested measurements pause their parent category. In particular a GPU submission
is host elapsed time (including any waits), never a claimed CUDA kernel timing.
"""
from __future__ import annotations

from contextlib import contextmanager
import time


LABELS = {
    "setup": "准备与校验",
    "input_read": "输入读取 / 解码",
    "index_write": "临时视频索引写入",
    "cpu_compute": "CPU 模块计算",
    "gpu_submit": "GPU 调度（主机，含同步等待）",
    "upload": "上传（主机，含等待）",
    "download": "回传（主机，含等待）",
    "references": "来源映射",
    "output_statistics": "输出摘要统计",
    "output_render": "曲线图生成",
    "output_write": "输出写入",
    "other": "其他 / 收尾",
}

PROGRESS_UNITS = {"T": "样本", "W": "窗长", "Q": "延迟点", "N": "记录"}


def progress_metadata(results):
    """Infer common output units without inventing an axis for old records."""
    axes = {item.get("progress_axis") or (item.get("contract") or {}).get("sequence_axis")
            for item in results}
    axis = next(iter(axes)) if len(axes)==1 else None
    return dict(progress_axis=axis,progress_unit=PROGRESS_UNITS.get(axis,"输出"))


class RunProfiler:
    """Per-source wall time, exclusive categories and actual sample counters.

    Input counts include repeated reads, including reads served by a reader's
    cache. Byte counts describe decoded arrays, not physical disk traffic. No
    per-sample history or GPU events are retained.
    """
    def __init__(self, *, clock=None):
        self._clock = clock or time.perf_counter
        self._started = self._last = self._clock()
        self._stopped = None
        self._stack = []
        self._seconds = dict.fromkeys(LABELS, 0.0)
        self.input_read_samples = self.input_read_bytes = 0
        self.output_samples = self.output_bytes = 0

    def _charge(self):
        if self._stopped is not None:
            return self._stopped
        now = self._clock()
        category = self._stack[-1] if self._stack else "other"
        self._seconds[category] += max(0.0, now-self._last)
        self._last = now
        return now

    @contextmanager
    def measure(self, category):
        if category not in LABELS:
            raise ValueError(f"Unknown performance category: {category}")
        if self._stack and self._stack[-1]==category:
            yield
            return
        self._charge()
        self._stack.append(category)
        try:
            yield
        finally:
            self._charge()
            self._stack.pop()

    def add_input(self, nbytes, count=1):
        self.input_read_samples += max(0, int(count))
        self.input_read_bytes += max(0, int(nbytes))

    def add_output(self, nbytes, count=1):
        self.output_samples += max(0, int(count))
        self.output_bytes += max(0, int(nbytes))

    def finish(self):
        self._stopped = self._charge()
        return self.snapshot()

    def snapshot(self):
        now = self._charge()
        elapsed = max(0.0, now-self._started)
        return dict(
            schema_version=1,
            elapsed_seconds=elapsed,
            host_seconds=dict(self._seconds),
            labels=dict(LABELS),
            input_read_samples=self.input_read_samples,
            input_read_bytes=self.input_read_bytes,
            output_samples=self.output_samples,
            output_bytes=self.output_bytes,
            input_reads_per_second=self.input_read_samples/elapsed if elapsed else None,
            output_per_second=self.output_samples/elapsed if elapsed else None,
            gpu_execution_seconds=None,
            semantics=dict(
                host_seconds="Exclusive host wall time; nested categories are subtracted from their parent. Includes I/O and synchronization waits.",
                gpu_execution_seconds="Not collected. GPU host submission and transfer times are not CUDA kernel execution times.",
                input_read_samples="Source sample reads, including video decoding during index creation, temporary-index rereads, repeated reads and reader-cache hits; not unique source frames.",
                input_read_bytes="Decoded source array bytes returned by reads; not physical disk bytes.",
                throughput="Output samples and input sample reads divided by source execution wall time, including setup and output.",
                scope="One source in the compute worker; excludes worker process startup, queueing and final run publication.",
            ),
        )
