"""GPU-capable operations registered on the same module instances as CPU."""
import numpy as np

from eip.module_ops import intensity_curve, gaussian

GPU_MODULES = frozenset({"intensity","affine","gain_offset","box_blur","gaussian",
                         "spatial_smooth","curve","threshold","fixed_normalize","grayscale",
                         "correlation","correlation_map","correlation_curve"})
GPU_MODULES = GPU_MODULES | frozenset({"time_crop","time_sample","time_smooth","time_bin",
    "time_statistics","global_statistics","region_reduce","distribution","alignment","calibration",
    "roi_crop","select_channel","point_extract","line_extract","axis_slice","color_convert",
    "data_tap","parameter_binding"})


def preserve(value, dtype, xp):
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        value = xp.clip(xp.rint(value),info.min,info.max)
    elif dtype.kind == "b":
        value = value != 0
    return value.astype(dtype)


def element_operation(algorithm, pixels, parameters, xp, *, color_order="RGB", preserve_input=False):
    """Use float64 intermediates and the CPU definition of rounding/invalids."""
    p = parameters
    value = xp.asarray(pixels,dtype=xp.float64)
    if algorithm in {"intensity","affine","gain_offset"}:
        result = value*p["gain"]+p["offset"]
        if algorithm != "intensity":
            result = xp.where(xp.isfinite(result),result,xp.nan)
        return preserve(result,pixels.dtype,xp) if algorithm=="intensity" or preserve_input else result
    if algorithm == "fixed_normalize":
        return value/xp.asarray(p["scale"],dtype=xp.float64)
    if algorithm == "grayscale":
        rgb = value[...,:3]
        if color_order=="BGR":
            rgb = rgb[...,::-1]
        return rgb.mean(axis=-1) if p["method"]=="mean" else xp.sum(rgb*xp.asarray([.2126,.7152,.0722]),axis=-1)
    if algorithm == "threshold":
        if value.ndim==3:
            value = value[...,:3].mean(axis=-1)
        return (value>=p["threshold"]).astype(xp.uint8)
    if algorithm == "curve":
        return intensity_curve(value,p["curve"],p["input_max"],p["gamma"],xp=xp)
    gaussian_mode = algorithm=="gaussian" or algorithm=="spatial_smooth" and p["method"]=="gaussian"
    if gaussian_mode:
        result = gaussian(value,p["sigma"],p["radius"],xp=xp)
    else:
        if not bool(xp.isfinite(value).all()):
            raise ValueError("均值滤波要求有限像素；请先显式处理 NaN/Inf")
        radius = p["radius"]
        pad = [(radius,radius),(radius,radius)]+([(0,0)] if value.ndim==3 else [])
        padded = xp.pad(value,pad,mode="reflect")
        integral = xp.pad(padded,[(1,0),(1,0)]+([(0,0)] if value.ndim==3 else [])).cumsum(axis=0).cumsum(axis=1)
        width = 2*radius+1
        result = (integral[width:,width:]-integral[:-width,width:]-integral[width:,:-width]+integral[:-width,:-width])/width**2
    return preserve(result,pixels.dtype,xp) if algorithm=="box_blur" or preserve_input else result
