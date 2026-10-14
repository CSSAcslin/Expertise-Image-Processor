"""Numerical image operations. Shared by every workbench mode."""
from __future__ import annotations

import numpy as np


def intensity_curve(values, curve="linear", input_max=1023., gamma=1., *, xp=np):
    x = xp.asarray(values, dtype=xp.float64) / input_max
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if curve == "identity":
            out = xp.asarray(values, dtype=xp.float64).copy()
        elif curve == "linear":
            out = x
        elif curve == "sqrt":
            out = xp.sqrt(x)
        elif curve == "square":
            out = x*x
        elif curve == "gamma":
            out = xp.where(x >= 0, x**gamma, xp.nan)
        elif curve == "log":
            out = xp.log1p(x*9)/np.log(10)
        else:
            raise ValueError("未知强度预设曲线")
    return xp.where(xp.isfinite(out), out, xp.nan)


def gaussian(values, sigma, radius, *, xp=np):
    a = xp.asarray(values, dtype=xp.float64)
    coordinates = xp.arange(-radius, radius+1, dtype=float)
    kernel = xp.exp(-.5*(coordinates/sigma)**2)
    kernel /= kernel.sum()
    finite = xp.isfinite(a)
    out, weights = xp.where(finite, a, 0), finite.astype(float)
    for axis in (0, 1):
        padding = [(0, 0)]*a.ndim
        padding[axis] = (radius, radius)
        def convolve(arr):
            if xp is not np:
                from cupyx.scipy.ndimage import convolve1d
                return convolve1d(arr,kernel,axis=axis,mode="nearest")
            windows = np.lib.stride_tricks.sliding_window_view(np.pad(arr, padding, mode="edge"), 2*radius+1, axis=axis)
            return np.einsum("...k,k->...", windows, kernel, optimize=False)
        out, weights = convolve(out), convolve(weights)
    out = xp.where(weights>0,out/xp.where(weights>0,weights,1),xp.nan)
    out[~finite] = xp.nan
    return out
