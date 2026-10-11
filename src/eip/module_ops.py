"""Numerical image operations. Shared by every workbench mode."""
from __future__ import annotations

import numpy as np


def intensity_curve(values, curve="linear", input_max=1023., gamma=1.):
    x = np.asarray(values, dtype=np.float64) / input_max
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if curve == "identity":
            out = np.asarray(values, dtype=np.float64).copy()
        elif curve == "linear":
            out = x
        elif curve == "sqrt":
            out = np.sqrt(x)
        elif curve == "square":
            out = x*x
        elif curve == "gamma":
            out = np.where(x >= 0, x**gamma, np.nan)
        elif curve == "log":
            out = np.log1p(x*9)/np.log(10)
        else:
            raise ValueError("未知强度预设曲线")
    return np.where(np.isfinite(out), out, np.nan)


def gaussian(values, sigma, radius):
    a = np.asarray(values, dtype=np.float64)
    coordinates = np.arange(-radius, radius+1, dtype=float)
    kernel = np.exp(-.5*(coordinates/sigma)**2)
    kernel /= kernel.sum()
    finite = np.isfinite(a)
    out, weights = np.where(finite, a, 0), finite.astype(float)
    for axis in (0, 1):
        padding = [(0, 0)]*a.ndim
        padding[axis] = (radius, radius)
        def convolve(arr):
            windows = np.lib.stride_tricks.sliding_window_view(np.pad(arr, padding, mode="edge"), 2*radius+1, axis=axis)
            return np.einsum("...k,k->...", windows, kernel, optimize=False)
        out, weights = convolve(out), convolve(weights)
    out = np.divide(out, weights, out=np.full_like(out, np.nan), where=weights > 0)
    out[~finite] = np.nan
    return out
