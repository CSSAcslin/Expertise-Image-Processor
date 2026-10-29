"""Read-only host telemetry; two-second snapshots never create a CUDA context.

CPU percentages use deltas of accumulated processor time, summed across all
logical processors. Application CPU uses that same denominator, so 100% means
all processors, rather than one fully occupied core. Only the current and
previous sample of the small EIP process set are retained.
"""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import threading
import time

from eip.resources import process_memory, system_memory


def system_cpu_times():
    """Return (all processor seconds, idle seconds), or None if unsupported."""
    if os.name == "nt":
        from ctypes import wintypes
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        get = ctypes.windll.kernel32.GetSystemTimes
        get.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
        get.restype = wintypes.BOOL
        if get(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            seconds = lambda value: ((value.dwHighDateTime << 32) | value.dwLowDateTime) / 10_000_000
            # GetSystemTimes includes idle time in kernel time.
            return seconds(kernel) + seconds(user), seconds(idle)
    else:
        try:
            row = Path("/proc/stat").read_text().splitlines()[0].split()
            if row[0] != "cpu" or len(row) < 5:
                return None
            ticks = [int(value) for value in row[1:9]]
            rate = os.sysconf("SC_CLK_TCK")
            return sum(ticks) / rate, (ticks[3] + (ticks[4] if len(ticks) > 4 else 0)) / rate
        except (OSError, ValueError, IndexError):
            pass
    return None


def process_cpu_times(pid):
    """Return (process creation identity, user + kernel seconds), or None."""
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        try:
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            get = kernel.GetProcessTimes
            get.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
            get.restype = wintypes.BOOL
            if get(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
                number = lambda value: (value.dwHighDateTime << 32) | value.dwLowDateTime
                return number(created), (number(system) + number(user)) / 10_000_000
        finally:
            kernel.CloseHandle(handle)
    else:
        try:
            # The command in parentheses may contain spaces and closing braces.
            row = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
            return int(row[19]), (int(row[11]) + int(row[12])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError):
            pass
    return None


def _percent(value, total):
    if total <= 0 or value < 0 or not math.isfinite(value) or not math.isfinite(total):
        return None
    return min(100.0, max(0.0, 100.0 * value / total))


def _read(reader, *args):
    try:
        return reader(*args)
    except (OSError, ValueError, TypeError, AttributeError):
        return None


class HostTelemetryMonitor:
    """Cache host readings across clients and tolerate worker replacement."""
    def __init__(self, interval=2.0):
        self.interval = interval
        self.lock = threading.Lock()
        self.snapshot = None
        self.next_sample = 0.0
        self.previous = None

    def _capture(self, pids):
        current = dict(cpu=_read(system_cpu_times), processes={}, sampled_at=time.time())
        memory = _read(system_memory) or {}
        total, available = memory.get("total_bytes"), memory.get("available_bytes")
        memory = dict(total_bytes=total, available_bytes=available,
                      used_bytes=max(0, total - available) if total is not None and available is not None else None)
        for pid in pids:
            current["processes"][pid] = dict(cpu=_read(process_cpu_times, pid), resident_bytes=_read(process_memory, pid))
        previous = self.previous
        system_percent = None
        denominator = None
        if previous and current["cpu"] is not None and previous["cpu"] is not None:
            denominator = current["cpu"][0] - previous["cpu"][0]
            idle = current["cpu"][1] - previous["cpu"][1]
            if idle >= 0:
                system_percent = _percent(denominator - idle, denominator)
        for pid, process in current["processes"].items():
            before = previous["processes"].get(pid, {}).get("cpu") if previous else None
            after = process["cpu"]
            process["percent"] = (_percent(after[1] - before[1], denominator)
                                  if denominator is not None and before is not None and after is not None and before[0] == after[0]
                                  else None)
        self.previous = current
        return dict(sampled_at=current["sampled_at"], memory=memory, processes=current["processes"],
                    system_percent=system_percent, first_sample=previous is None)

    def sample(self, pids):
        pids = set(int(pid) for pid in pids if pid)
        with self.lock:
            if self.snapshot is None or time.monotonic() >= self.next_sample:
                self.snapshot = self._capture(pids)
                self.next_sample = time.monotonic() + self.interval
            raw = self.snapshot
            processes = [dict(pid=pid,
                              resident_bytes=raw["processes"].get(pid, {}).get("resident_bytes"),
                              cpu_percent=raw["processes"].get(pid, {}).get("percent"))
                         for pid in sorted(pids)]
            percentages = [row["cpu_percent"] for row in processes]
            sizes = [row.get("resident_bytes") for row in processes]
            changed = any(pid not in raw["processes"] for pid in pids)
            cpu_reason = ("进程已切换，等待下一次采样" if changed else
                          "等待第二次采样" if raw["first_sample"] else
                          "部分进程 CPU 时间不可用" if any(value is None for value in percentages) else None)
            memory_reason = ("进程已切换，等待下一次采样" if changed else
                             "部分进程驻留内存不可用" if any(value is None for value in sizes) else None)
            return dict(sampled_at=raw["sampled_at"], interval_seconds=self.interval,
                        processes=processes,
                        cpu=dict(system_percent=raw["system_percent"],
                                 application_percent=min(100.0, sum(percentages)) if all(value is not None for value in percentages) else None,
                                 logical_processors=os.cpu_count(), reason=cpu_reason,
                                 system_reason=None if raw["system_percent"] is not None else "等待第二次采样" if raw["first_sample"] else "系统 CPU 时间不可用"),
                        memory=dict(raw["memory"], application_bytes=sum(sizes) if all(value is not None for value in sizes) else None,
                                    application_reason=memory_reason))
