"""Workspace resource policy, copied into every compute request."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import ctypes
import json
import os

MIB = 1024**2


def system_memory():
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(key, ctypes.c_ulonglong) for key in
                          ("total", "available", "page_total", "page_available", "virtual", "virtual_available", "extended")]]
        status = Status()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return dict(total_bytes=status.total, available_bytes=status.available)
    elif Path("/proc/meminfo").exists():
        values = {line.split(":")[0]: int(line.split()[1])*1024
                  for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2}
        return dict(total_bytes=values.get("MemTotal"), available_bytes=values.get("MemAvailable"))
    return dict(total_bytes=None, available_bytes=None)


def process_memory(pid):
    """Resident bytes; unavailable is distinct from zero."""
    if not pid:
        return None
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in
                          ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "page", "peak_page")]]
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)
        if handle:
            try:
                counters = Counters()
                counters.size = ctypes.sizeof(counters)
                get = ctypes.windll.psapi.GetProcessMemoryInfo
                get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
                if get(handle, ctypes.byref(counters), counters.size):
                    return counters.rss
            finally:
                kernel.CloseHandle(handle)
    else:
        try:
            return int(Path(f"/proc/{pid}/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            pass
    return None


@dataclass(frozen=True)
class ResourceSettings:
    memory_mb: int = 2048
    preview_mb: int = 128
    cache_mb: int = 128
    gpu_mb: int = 512
    backend: str = "cpu"

    @classmethod
    def from_dict(cls, value=None):
        value = {} if value is None else value
        if not isinstance(value, dict) or set(value)-set(cls.__dataclass_fields__):
            raise ValueError("资源设置包含未知字段")
        settings = cls(**value)
        for key, low, high in (("memory_mb",128,131072),("preview_mb",16,4096),
                               ("cache_mb",0,32768),("gpu_mb",16,131072)):
            number = getattr(settings,key)
            if type(number) is not int or not low<=number<=high:
                raise ValueError(f"{key} 需为 {low}–{high} MiB 的整数")
        if settings.backend not in {"cpu","gpu","auto"}:
            raise ValueError("默认计算设备需为 CPU、GPU 或自动")
        if settings.preview_mb > settings.memory_mb//2:
            raise ValueError("预览缓存不能超过总内存预算的一半")
        if settings.cache_mb > settings.processing_mb//3:
            raise ValueError("计算缓存不能超过可用处理预算的三分之一")
        return settings

    @property
    def processing_mb(self):
        return self.memory_mb-self.preview_mb

    @property
    def inspection_mb(self):
        return max(16,min(256,self.processing_mb//4))

    @property
    def job_mb(self):
        return self.processing_mb-self.inspection_mb

    def record(self):
        return asdict(self)


def load_resources(path):
    if path.is_file():
        return ResourceSettings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    total = system_memory()["total_bytes"]
    memory = max(256,min(2048,int(total/MIB/4))) if total else 2048
    return ResourceSettings(memory_mb=memory,preview_mb=min(128,memory//4),cache_mb=min(128,memory//8))


def configure_reader(reader, budget_bytes):
    if hasattr(reader,"_cache"):
        reader._cache.clear()
        reader._cache_size = 0
        reader._budget = max(0,int(budget_bytes))
