from __future__ import annotations

import importlib.util

import numpy as np


def devices():
    # Optional packages may be installed after the workbench was started.
    importlib.invalidate_caches()
    result = {"cpu": True,"gpu": False,"gpu_name": None,"reason": "未安装可选 CuPy / CUDA 运行环境"}
    if importlib.util.find_spec("cupy") is not None:
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                result["reason"] = "没有可用 CUDA 设备"
            else:
                info = cp.cuda.runtime.getDeviceProperties(0)
                name = info["name"]
                result.update(gpu=True,gpu_name=name.decode() if isinstance(name,bytes) else str(name),reason=None)
        except Exception as error:
            result["reason"] = str(error)
    return result


class Backend:
    def __init__(self, requested, budget_bytes):
        if requested not in {"cpu","auto","gpu"}:
            raise ValueError("未知计算设备")
        self.requested = requested
        self.xp = np
        self.name = "cpu"
        self.reason = None
        if requested in {"auto","gpu"}:
            available = devices()
            if available["gpu"]:
                try:
                    import cupy as cp
                    free,_ = cp.cuda.runtime.memGetInfo()
                    if free < budget_bytes:
                        raise ValueError("可用显存低于本次资源预算，请降低内存预算或选择 CPU")
                    # Auto falls back only before processing; compute failures stay visible.
                    check = cp.sum(cp.asarray([1.0,2.0],dtype=cp.float64)).item()
                    if check != 3.0:
                        raise ValueError("GPU 数值自检失败")
                    cp.get_default_memory_pool().set_limit(size=int(budget_bytes))
                    self.xp,self.name = cp,"gpu"
                except Exception as error:
                    available.update(gpu=False,reason=f"GPU 启动失败：{error}")
            if not available["gpu"]:
                if requested == "gpu":
                    raise ValueError(available["reason"])
                self.reason = available["reason"]

    def host(self, value):
        return np.asarray(value) if self.name == "cpu" else self.xp.asnumpy(value)

    def close(self):
        if self.name == "gpu":
            self.xp.get_default_memory_pool().free_all_blocks()
            self.xp.get_default_pinned_memory_pool().free_all_blocks()
