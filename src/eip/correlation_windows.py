"""Fixed-centre g² window scans, counted in direct upstream samples."""
from eip.modes.split_correlation.config import finite, integer


def window_bounds(center, window):
    # Even windows retain the selected integer anchor, with one extra right sample.
    start = center-(window-1)//2
    return start,start+window


def window_sizes(parameters):
    p = parameters
    integer(p["lag"],"固定 τ")
    if abs(p["lag"])>8_000_000:
        raise ValueError("固定 τ 超出支持范围")
    integer(p["min_samples"],"最少有效样本",1)
    if finite(p["min_mean"],"最小强度均值")<=0:
        raise ValueError("最小强度均值需为正数")
    if p["center"] is not None:
        integer(p["center"],"固定中心样本",1)
    for key in ("window_min","window_max","window_step"):
        integer(p[key],key,1)
    if p["window_list"] is not None:
        if not isinstance(p["window_list"],(list,tuple)) or not 1<=len(p["window_list"])<=4096:
            raise ValueError("窗长列表需包含 1～4096 个正整数")
        for window in p["window_list"]:
            integer(window,"平均窗长",1)
        windows = sorted(set(p["window_list"]))
    else:
        if p["window_min"]>p["window_max"]:
            raise ValueError("窗长下限不能大于上限")
        if (p["window_max"]-p["window_min"])//p["window_step"]+1>4096:
            raise ValueError("窗长扫描最多 4096 项")
        windows = list(range(p["window_min"],p["window_max"]+1,p["window_step"]))
    if windows[-1]>8_000_000 or windows[0]<p["min_samples"]:
        raise ValueError("平均窗长超出支持范围或小于最少有效样本数")
    return windows


def window_scan_layout(length, parameters):
    windows = window_sizes(parameters)
    largest,lag = windows[-1],parameters["lag"]
    first = (largest-1)//2+max(0,-lag)
    last = None if length is None else length-1-largest//2-max(0,lag)
    if last is not None and last<first:
        raise ValueError("当前输入样本不足以容纳最大平均窗口及固定 τ，请减小窗长或延迟")
    anchor = parameters["center"]
    anchor = anchor-1 if anchor is not None else first if last is None else (first+last)//2
    if anchor<first or last is not None and anchor>last:
        raise ValueError(f"窗长扫描中心需在编辑后样本 {first+1}～{'?' if last is None else last+1} 内，以容纳全部窗口")
    return windows,anchor
