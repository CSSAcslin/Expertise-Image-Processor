"""Exact window quantiles without a window-sized RAM stack."""
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
from eip.exact_quantile import finite_quantile


class RollingReduction:
    """Overlapping windows update outgoing/incoming samples, with periodic rebase."""
    def __init__(self, shape, xp=np):
        self.xp, self.shape = xp, shape
        self.first = None
        self.updates = 0
        self.total = xp.zeros(shape, dtype=xp.float64)
        self.compensation = xp.zeros(shape, dtype=xp.float64)
        self.count = xp.zeros(shape, dtype=xp.int64)

    def get(self, get_sample, first, size, mode, check):
        xp = self.xp
        reuse = self.first is not None and 0 <= first-self.first < size and self.updates < max(64,size)
        if reuse:
            ranges = ((range(self.first, first), -1), (range(self.first+size, first+size), 1))
            self.updates += 1
        else:
            self.total.fill(0); self.compensation.fill(0); self.count.fill(0)
            ranges = ((range(first, first+size), 1),)
            self.updates = 0
        self.first = None
        for indices, sign in ranges:
            for j in indices:
                check()
                value = xp.asarray(get_sample(j),dtype=xp.float64)
                valid = xp.isfinite(value)
                delta = sign*xp.where(valid, value, 0)-self.compensation
                updated = self.total+delta
                self.compensation = (updated-self.total)-delta
                self.total = updated
                self.count += sign*valid.astype(xp.int64)
        self.first = first
        return xp.where(self.count, self.total/xp.maximum(self.count, 1) if mode == "mean" else self.total, xp.nan)


class WindowQuantile:
    """Reusable disk ring; only entering samples are written for overlapping windows."""
    def __init__(self, shape, size, budget, workdir, xp=np, host=np.asarray):
        self.shape, self.size, self.budget = shape, size, budget
        self.xp, self.host, self.first = xp, host, None
        self.elements = max(1, int(np.prod(shape)))
        directory = Path(workdir)
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(directory).free < self.elements*size*8+32*1024**2:
            raise ValueError("窗口精确分位值的临时磁盘空间不足，请缩小窗口或更换输出目录")
        self.factor = 4 if xp is np else 8
        if size*8*self.factor > budget:
            raise ValueError("单个分位值序列超过工作预算，请缩小窗口或提高统一内存预算")
        self.folder = tempfile.TemporaryDirectory(prefix="window-ring-", dir=directory)
        try:
            self.spool = np.lib.format.open_memmap(Path(self.folder.name)/"window.npy", mode="w+", dtype="float64", shape=(size, self.elements))
        except Exception:
            self.folder.cleanup()
            raise

    def get(self, get_sample, first, q, check):
        begin = self.first+self.size if self.first is not None and 0 <= first-self.first < self.size else first
        self.first = None
        for j in range(begin, first+self.size):
            check()
            value = np.asarray(self.host(get_sample(j))).reshape(-1)
            self.spool[j % self.size] = np.where(np.isfinite(value), value, np.nan)
        self.first = first
        block = max(1, min(self.elements, self.budget//(self.size*8*self.factor+64)))
        result = self.xp.empty(self.elements, dtype=self.xp.float64)
        for start in range(0, self.elements, block):
            check()
            chunk = np.array(self.spool[:, start:start+block], copy=True)
            if self.xp is np:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result[start:start+block] = np.nanquantile(chunk, q, axis=0, overwrite_input=True, method="linear")
            else:
                result[start:start+block] = finite_quantile(self.xp.asarray(chunk), q, (0,), self.xp)
        return result.reshape(self.shape)

    def close(self):
        self.spool._mmap.close()
        self.folder.cleanup()


def window_quantile(get_sample, shape, first, size, quantile, budget, workdir, check):
    elements = int(np.prod(shape))
    required = elements*size*8
    directory = Path(workdir)
    directory.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(directory).free < required+32*1024**2:
        raise ValueError("窗口精确分位值的临时磁盘空间不足，请缩小窗口或更换输出目录")
    if size*8*4 > budget:
        raise ValueError("单个分位值序列超过工作预算，请缩小窗口或提高统一内存预算")
    with tempfile.TemporaryDirectory(prefix="window-quantile-",dir=directory) as folder:
        spool = np.lib.format.open_memmap(Path(folder)/"window.npy",mode="w+",dtype="float64",shape=(size,elements))
        try:
            for offset in range(size):
                check()
                arr = np.asarray(get_sample(first+offset)).reshape(-1)
                spool[offset] = np.where(np.isfinite(arr),arr,np.nan)
            block = max(1,min(elements,budget//(size*8*4+64)))
            result = np.empty(elements,dtype=np.float64)
            for start in range(0,elements,block):
                check()
                chunk = np.array(spool[:,start:start+block],copy=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore",RuntimeWarning)
                    result[start:start+block] = np.nanquantile(chunk,quantile,axis=0,overwrite_input=True,method="linear")
            return result.reshape(shape)
        finally:
            spool._mmap.close()
