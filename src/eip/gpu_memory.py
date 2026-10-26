"""Bounded, read-only NVIDIA memory sampling without creating a CUDA context."""
from __future__ import annotations

import csv
import io
import math
import os
import shutil
import subprocess
import threading
import time

from eip.resources import MIB


def memory_bytes(value):
    try:
        number = float(value.strip())
        return int(number*MIB) if math.isfinite(number) and number >= 0 else None
    except (ValueError, OverflowError):
        return None


class GpuMemoryMonitor:
    """Cache driver samples across panels; apply the current EIP PID set per read."""
    def __init__(self, interval=2.0, timeout=2.0):
        self.interval, self.timeout = interval, timeout
        self.lock = threading.Lock()
        self.snapshot = None
        self.next_sample = 0

    def _query(self, executable, query):
        result = subprocess.run(
            [executable, query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=True, timeout=self.timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        if len(result.stdout) > 1024*1024:
            raise ValueError("驱动返回的监测数据过大")
        return [row for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True) if any(value.strip() for value in row)]

    def _capture(self):
        result = dict(available=False, cards=[], processes={}, process_error=None, reason=None)
        executable = shutil.which("nvidia-smi")
        if not executable:
            result["reason"] = "未找到 nvidia-smi，无法读取显存"
            return result
        try:
            rows = self._query(executable, "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_model.current")
            for row in rows[:64]:
                if len(row) != 7:
                    raise ValueError("无法识别驱动返回的显存数据")
                index, uuid, name, total, used, free, model = (v.strip() for v in row)
                result["cards"].append(dict(index=int(index), uuid=uuid, name=name,
                    total_bytes=memory_bytes(total), used_bytes=memory_bytes(used),
                    free_bytes=memory_bytes(free), driver_model=model))
            if not result["cards"]:
                result["reason"] = "未检测到 NVIDIA 显卡"
                return result
            result["available"] = True
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            result["cards"] = []
            result["reason"] = "显存读取超时" if isinstance(error, subprocess.TimeoutExpired) else "无法从 NVIDIA 驱动读取显存"
            return result
        try:
            rows = self._query(executable, "--query-compute-apps=gpu_uuid,pid,used_memory")
            for row in rows:
                if len(row) != 3:
                    raise ValueError("无法识别进程显存数据")
                uuid, pid, used = (v.strip() for v in row)
                key = (uuid, int(pid))
                size = memory_bytes(used)
                # Do not count a duplicate PID twice or conceal an unknown row.
                if key not in result["processes"]:
                    result["processes"][key] = size
                elif result["processes"][key] is None or size is None:
                    result["processes"][key] = None
                else:
                    result["processes"][key] = max(size, result["processes"][key])
        except (OSError, ValueError, subprocess.SubprocessError):
            result["processes"] = {}
            result["process_error"] = "驱动未提供进程显存"
        return result

    def sample(self, pids):
        pids = set(pids)
        with self.lock:
            if self.snapshot is None or time.monotonic() >= self.next_sample:
                self.snapshot = self._capture()
                self.snapshot["sampled_at"] = time.time()
                self.next_sample = time.monotonic()+self.interval
            raw = self.snapshot
            cards = []
            for card in raw["cards"]:
                sizes = [size for (uuid, pid), size in raw["processes"].items() if uuid == card["uuid"] and pid in pids]
                reason = raw["process_error"]
                if any(size is None for size in sizes) or not sizes and card["driver_model"].upper() == "WDDM":
                    reason = "Windows 驱动未提供进程显存" if card["driver_model"].upper() == "WDDM" else "驱动未提供完整进程显存"
                cards.append(dict(card, application_bytes=None if reason else sum(sizes), application_reason=reason))
            return dict(available=raw["available"], cards=cards, reason=raw["reason"], sampled_at=raw["sampled_at"], interval_seconds=self.interval)
