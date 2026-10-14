"""Exact window quantiles without a window-sized RAM stack."""
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np


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
