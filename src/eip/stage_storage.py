"""Owned, demand-filled main-path storage with bounded RAM/device/disk use.

The store contains one actual module output, not a second algorithm pipeline.
Only built-in, frame-local prefixes are selected by SequenceEngine. Filling a
slot never changes its dtype, axes, precision or numerical implementation.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np


class StageStore:
    def __init__(self, stage, contract, *, mode, workdir, backend=None, block_size=1):
        self.stage, self.contract, self.mode = stage, contract, mode
        self.backend = backend
        self.length, self.shape = contract.length, tuple(contract.shape)
        self.dtype = np.dtype(contract.dtype)
        self.sample_bytes = max(1, int(np.prod(self.shape))) * self.dtype.itemsize
        self.capacity_bytes = self.length * self.sample_bytes
        self.block_size = max(1, min(32, int(block_size)))
        self.ready = np.zeros(self.length, dtype=np.bool_)
        self.values = self.file = None
        self.path = Path(workdir) / f"stage-{stage}.bin"
        self.metrics = dict(stage=stage, storage=mode, full_axes=contract.sequence_axis+contract.axes,
                            shape=[self.length, *self.shape], dtype=str(self.dtype),
                            capacity_bytes=self.capacity_bytes, metadata_bytes=self.ready.nbytes,
                            block_size=self.block_size, built_samples=0, hits=0,
                            range_requests=0, range_hits=0, range_samples=0,
                            input_cache_released_bytes=0, disk_read_bytes=0, disk_write_bytes=0,
                            ownership="exclusive_compact_main_output", state="unallocated")

    def _allocate(self):
        if self.metrics["state"] != "unallocated":
            return
        if self.mode == "disk":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(self.path.parent).free < self.capacity_bytes + 32*1024**2:
                raise ValueError("主路阶段存储的临时磁盘空间不足，请缩小范围或增加工作内存")
            self.file = self.path.open("x+b", buffering=0)
            self.file.truncate(self.capacity_bytes)
        else:
            xp = self.backend.xp if self.mode == "device" else np
            self.values = xp.empty((self.length, *self.shape), dtype=self.dtype)
        self.metrics["state"] = "filling"

    def get(self, index, compute, check, released=None, compute_block=None):
        check()
        if not 0 <= index < self.length:
            raise IndexError("阶段存储样本超出范围")
        self._allocate()
        if self.ready[index]:
            self.metrics["hits"] += 1
        else:
            start = index // self.block_size * self.block_size
            stop = min(self.length, start+self.block_size)
            # A single physical batch preserves all logical sample boundaries.
            # Existing callers retain incremental single-slot cancellation.
            batch = None
            if compute_block is not None and not self.ready[start:stop].any():
                batch = compute_block(start, stop)
                check()
                if tuple(batch.shape) != (stop-start, *self.shape) or np.dtype(batch.dtype) != self.dtype:
                    raise ValueError("主路阶段批量输出形状或类型与已验证契约不符")
                if self.mode == "disk":
                    self.file.seek(start*self.sample_bytes)
                    data = np.asarray(batch).tobytes(order="C")
                    if self.file.write(data) != len(data):
                        raise OSError("主路阶段批量存储写入不完整")
                    self.metrics["disk_write_bytes"] += len(data)
                else:
                    self.values[start:stop] = batch
                check()
                self.ready[start:stop] = True
                self.metrics["built_samples"] += stop-start
                self.metrics["block_commits"] = self.metrics.get("block_commits",0)+1
            for sample in range(start, stop) if batch is None else ():
                check()
                if self.ready[sample]:
                    continue
                value = compute(sample) if batch is None else batch[sample-start]
                check()
                if tuple(value.shape) != self.shape or np.dtype(value.dtype) != self.dtype:
                    raise ValueError("主路阶段输出形状或类型与已验证契约不符")
                if self.mode == "disk":
                    self.file.seek(sample*self.sample_bytes)
                    data = np.asarray(value).tobytes(order="C")
                    if self.file.write(data) != self.sample_bytes:
                        raise OSError("主路阶段存储写入不完整")
                    self.metrics["disk_write_bytes"] += len(data)
                else:
                    # Assignment detaches ROI/point views from much larger
                    # parents. Store ownership is independent of reader caches.
                    self.values[sample] = value
                self.ready[sample] = True
                self.metrics["built_samples"] += 1
            if released is not None:
                self.metrics["input_cache_released_bytes"] += released(stop)
            if self.metrics["built_samples"] == self.length:
                self.metrics["state"] = "complete"
        if self.mode == "disk":
            self.file.seek(index*self.sample_bytes)
            data = self.file.read(self.sample_bytes)
            if len(data) != self.sample_bytes:
                raise OSError("主路阶段存储读取不完整")
            self.metrics["disk_read_bytes"] += len(data)
            return np.frombuffer(data, dtype=self.dtype).reshape(self.shape)
        value = self.values[index]
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        return value

    def get_range(self, start, stop, compute, check, released=None,
                  compute_block=None, *, max_bytes=None):
        """Return an owned-stage interval without per-sample access or copies.

        Resident RAM/device stores return a contiguous borrowed view. A NumPy
        view is read-only without changing the writable stage allocation; all
        device consumers must likewise treat it as read-only. The view remains
        valid while its consumer holds it, including after ``close``.

        Disk reads allocate at most ``max_bytes``. Without an explicit limit a
        disk request is bounded to one physical block; larger ranges must be
        budgeted by their caller. Invalid or over-budget requests never fill or
        allocate a stage. Empty intervals do not allocate the stage either.

        Existing readiness, cancellation and upstream-release semantics are
        shared with ``get``: at most one call fills each missing physical block,
        selecting a missing requested slot even if a cancelled earlier attempt
        left other slots in that block ready.
        """
        check()
        if (any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
                for v in (start, stop)) or not 0 <= start <= stop <= self.length):
            raise IndexError("阶段存储范围超出边界或不是整数")
        start, stop = int(start), int(stop)
        if self.ready is None:
            raise RuntimeError("阶段存储已关闭")
        if max_bytes is not None and (not isinstance(max_bytes, (int, np.integer))
                                      or isinstance(max_bytes, (bool, np.bool_)) or max_bytes < 0):
            raise ValueError("阶段范围暂存预算必须是非负整数")
        required = (stop-start)*self.sample_bytes
        if self.mode == "disk":
            limit = self.block_size*self.sample_bytes if max_bytes is None else int(max_bytes)
            if required > limit:
                raise ValueError("阶段范围读取超过暂存预算，请缩小数据块")
        self.metrics["range_requests"] += 1
        if start == stop:
            if self.values is not None:
                value = self.values[start:stop]
            else:
                xp = self.backend.xp if self.mode == "device" else np
                value = xp.empty((0, *self.shape), dtype=self.dtype)
        else:
            if self.ready[start:stop].all():
                self.metrics["range_hits"] += 1
            else:
                first_block = start//self.block_size*self.block_size
                for block in range(first_block, stop, self.block_size):
                    check()
                    first, end = max(start, block), min(stop, block+self.block_size)
                    missing = np.flatnonzero(~self.ready[first:end])
                    if len(missing):
                        self.get(first+int(missing[0]), compute, check, released,
                                 compute_block=compute_block)
            check()
            if self.mode == "disk":
                self.file.seek(start*self.sample_bytes)
                data = self.file.read(required)
                if len(data) != required:
                    raise OSError("主路阶段范围读取不完整")
                self.metrics["disk_read_bytes"] += len(data)
                value = np.frombuffer(data, dtype=self.dtype).reshape((stop-start, *self.shape))
            else:
                value = self.values[start:stop]
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
        self.metrics["range_samples"] += stop-start
        return value

    def record(self):
        return dict(self.metrics)

    def close(self):
        self.values = None
        self.ready = None
        if self.file is not None:
            self.file.close()
            self.file = None
            self.path.unlink(missing_ok=True)


# These nodes do not require an upstream temporal neighbourhood or full-source
# numerical fitting. The plan still validates main/side ports and parameters.
FRAME_LOCAL = frozenset({
    "time_crop", "time_sample", "calibration", "alignment", "select_channel",
    "roi_crop", "spatial_roi", "point_extract", "line_extract", "time_statistics",
    "grayscale", "color_convert", "fixed_normalize", "gain_offset", "spatial_smooth",
    "curve", "threshold", "intensity", "affine", "box_blur", "gaussian",
    "parameter_binding", "data_tap",
})


def compact_checkpoint(engine):
    """Select a compressed direct input to a repeated temporal consumer.

Only the first eligible checkpoint is retained. Upstream side data, arbitrary
plugins and full-source statistics keep the compatible dependency-aware path.
"""
    from eip.module_registry import CORRELATIONS, is_reference
    candidates = []
    for index, operation in enumerate(engine.operations):
        if operation["algorithm"] not in {*CORRELATIONS, "time_smooth", "time_bin", "global_statistics"} or index == 0:
            continue
        c = engine.contracts[index]
        if c.length is None or c.length < 1 or c.dtype == "dynamic":
            continue
        prefix = engine.nodes[:index]
        if any(n.get("manifest") or n["algorithm"] not in FRAME_LOCAL
               or any(is_reference(v) for v in n["parameters"].values()) for n in prefix):
            continue
        if any(s["stage"] < index and s["kind"] != "parameter_binding" for s in engine._sides):
            continue
        before = max(max(1, int(np.prod(v.shape)))*np.dtype(v.dtype).itemsize
                     for v in engine.contracts[:index] if v.dtype != "dynamic")
        after = max(1, int(np.prod(c.shape)))*np.dtype(c.dtype).itemsize
        if after*2 <= before:
            candidates.append(index)
    return candidates[0] if candidates else None
