"""Typed, ordered scientific modules shared by preview, fitting and execution."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .geometry import Geometry


MODULES = {
    "calibration": {"name": "强度校准 / LUT", "inputs": ["sensor"], "output": "sensor", "repeatable": False},
    "alignment": {"name": "双幅划分与固定配准", "inputs": ["sensor"], "output": "pair", "repeatable": False},
    "affine": {"name": "线性缩放与偏置", "inputs": ["sensor", "pair", "map"], "output": "same", "repeatable": True,
               "defaults": {"gain": 1.0, "offset": 0.0, "target": "both"}},
    "gaussian": {"name": "空间高斯平滑", "inputs": ["sensor", "pair", "map"], "output": "same", "repeatable": True,
                 "defaults": {"sigma": 1.0, "radius": 2, "target": "both"}},
    "curve": {"name": "强度曲线", "inputs": ["sensor", "pair", "map"], "output": "same", "repeatable": True,
              "defaults": {"curve": "linear", "input_max": 1023.0, "gamma": 1.0, "target": "both"}},
    "statistics": {"name": "全源均值与固定尺度", "inputs": ["pair"], "output": "moments", "repeatable": False},
    "correlation": {"name": "标准二阶互关联 g²", "inputs": ["moments"], "output": "map", "repeatable": False},
}


def numeric(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"模块参数 {name} 需为有限数值")
    return float(value)


def intensity_curve(values, curve="linear", input_max=1023., gamma=1.):
    """Explicit fixed curves; no implicit clipping or per-frame auto stretch."""
    x = np.asarray(values, dtype=np.float64) / input_max
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if curve == "identity":
            out = np.asarray(values, dtype=np.float64).copy()
        elif curve == "linear":
            out = x
        elif curve == "sqrt":
            out = np.sqrt(x)
        elif curve == "square":
            out = x * x
        elif curve == "gamma":
            out = np.where(x >= 0, x ** gamma, np.nan)
        elif curve == "log":
            out = np.log1p(x * 9) / np.log(10)
        else:
            raise ValueError("未知强度预设曲线")
    return np.where(np.isfinite(out), out, np.nan)


def validate_transform(algorithm, parameters):
    p = deepcopy(MODULES[algorithm]["defaults"])
    if set(parameters) - set(p):
        raise ValueError(f"{MODULES[algorithm]['name']} 包含未知参数")
    p.update(parameters)
    if p["target"] not in {"both", "left", "right"}:
        raise ValueError("模块作用幅面需为双路、左路或右路")
    if algorithm == "affine":
        numeric(p["gain"], "gain")
        numeric(p["offset"], "offset")
    elif algorithm == "gaussian":
        if numeric(p["sigma"], "sigma") <= 0 or type(p["radius"]) is not int or not 1 <= p["radius"] <= 32:
            raise ValueError("高斯 sigma 需大于 0，半径需为 1～32 的整数")
    elif algorithm == "curve":
        if p["curve"] not in {"identity", "linear", "sqrt", "square", "gamma", "log"}:
            raise ValueError("未知强度预设曲线")
        if numeric(p["input_max"], "input_max") <= 0 or numeric(p["gamma"], "gamma") <= 0:
            raise ValueError("曲线输入尺度和 gamma 需大于 0")
    return p


def resolve_workflow(value):
    """Resolve node-owned parameters into legacy fields for old readers / exports."""
    nodes = value.get("workflow")
    if nodes is None:
        value["workflow"] = ()
        return
    if not isinstance(nodes, (list, tuple)) or not 4 <= len(nodes) <= 32:
        raise ValueError("模块流程需包含 4～32 个模块")
    value.setdefault("photometry", {})
    state, seen, identifiers, resolved = "sensor", set(), set(), []
    for raw in nodes:
        if not isinstance(raw, dict) or set(raw) - {"id", "algorithm", "parameters", "enabled", "name"}:
            raise ValueError("模块记录格式无效")
        node = deepcopy(raw)
        algorithm, identifier = node.get("algorithm"), node.get("id")
        if algorithm not in MODULES or not isinstance(identifier, str) or not 1 <= len(identifier) <= 80 or identifier in identifiers:
            raise ValueError("模块类型或实例标识无效；重复模块需使用独立实例")
        identifiers.add(identifier)
        spec = MODULES[algorithm]
        if type(node.get("enabled", True)) is not bool:
            raise ValueError("模块启用状态需为布尔值")
        if not isinstance(node.get("parameters", {}), dict):
            raise ValueError("模块参数需为对象")
        node.setdefault("enabled", True)
        node.setdefault("parameters", {})
        if not node["enabled"]:
            if not spec["repeatable"]:
                raise ValueError(f"必要模块 {spec['name']} 不能禁用")
            resolved.append(node)
            continue
        if state not in spec["inputs"]:
            kind = {"sensor":"传感器图像","pair":"已对齐双路","moments":"双路强度与全源统计","map":"系数图"}[state]
            raise ValueError(f"模块「{spec['name']}」无法接收当前数据类型：{kind}，请调整顺序")
        if not spec["repeatable"] and algorithm in seen:
            raise ValueError(f"结构模块「{spec['name']}」只允许一个实例")
        seen.add(algorithm)
        if spec["repeatable"]:
            node["parameters"] = validate_transform(algorithm, node["parameters"])
            if state != "pair" and node["parameters"]["target"] != "both":
                raise ValueError("左 / 右路作用范围只能放在分幅之后、全源统计之前")
        elif algorithm == "calibration":
            value["photometry"] = deepcopy(node["parameters"])
        elif algorithm == "alignment":
            value["geometry"] = deepcopy(node["parameters"])
        elif algorithm == "statistics":
            if set(node["parameters"]) != {"normalization"}:
                raise ValueError("全源统计模块需指定 normalization")
            value["photometry"]["normalization"] = node["parameters"]["normalization"]
        elif algorithm == "correlation":
            if set(node["parameters"]) != {"window", "min_mean", "min_samples"}:
                raise ValueError("关联模块需指定 window、min_mean、min_samples")
            value.update(node["parameters"])
        if spec["output"] != "same":
            state = spec["output"]
        resolved.append(node)
    if not {"calibration", "alignment", "statistics", "correlation"}.issubset(seen) or state != "map":
        raise ValueError("流程必须包含强度校准、分幅配准、全源统计、二阶关联")
    # Calibration must precede alignment; all scientific intensity samples pass it.
    names = [n["algorithm"] for n in resolved if n["enabled"]]
    if names.index("calibration") > names.index("alignment"):
        raise ValueError("强度校准需位于空间采样之前")
    value["workflow"] = tuple(resolved)


def transform_array(values, algorithm, p):
    a = np.asarray(values, dtype=np.float64)
    if algorithm == "affine":
        with np.errstate(over="ignore", invalid="ignore"):
            out = a * p["gain"] + p["offset"]
    elif algorithm == "curve":
        return intensity_curve(a, p["curve"], p["input_max"], p["gamma"])
    elif algorithm == "gaussian":
        if a.ndim != 2:
            raise ValueError("空间高斯模块只接受二维图像；τ 曲线不能使用关联后的空间平滑")
        r = p["radius"]
        coordinates = np.arange(-r, r + 1, dtype=float)
        kernel = np.exp(-coordinates**2 / (2 * p["sigma"]**2))
        kernel /= kernel.sum()
        finite = np.isfinite(a)
        out, weights = np.where(finite, a, 0), finite.astype(float)
        for axis in (0, 1):
            padding = [(0, 0), (0, 0)]
            padding[axis] = (r, r)
            def convolve(arr):
                padded = np.pad(arr, padding, mode="edge")
                windows = np.lib.stride_tricks.sliding_window_view(padded, 2*r+1, axis=axis)
                return np.einsum("...k,k->...", windows, kernel, optimize=False)
            out, weights = convolve(out), convolve(weights)
        out = np.divide(out, weights, out=np.full_like(out, np.nan), where=weights > 0)
        out[~finite] = np.nan
    else:
        raise ValueError("未知图像变换模块")
    return np.where(np.isfinite(out), out, np.nan)


class IdentityPhotometry:
    def apply(self, pixels, side):
        return np.asarray(pixels, dtype=np.float64)


class WorkflowGeometry(Geometry):
    def __init__(self, config, shape):
        super().__init__(config.geometry, shape)
        self.nodes = [n for n in config.workflow if n.get("enabled", True)]
        self.record["workflow"] = deepcopy(self.nodes[:next(i for i, n in enumerate(self.nodes) if n["algorithm"] == "statistics")])
        self.record["version"] = "1.1"
        yy = np.arange(shape[0])[:, None]
        line = np.interp(yy, [0, max(1, shape[0]-1)], self.config.midline)
        self.sensor_left_mask = np.arange(shape[1])[None, :] < line

    def apply(self, pixels, photometry):
        data = pixels
        fused_alignment = False
        for index, node in enumerate(self.nodes):
            algorithm, p = node["algorithm"], node["parameters"]
            if algorithm == "statistics":
                break
            if algorithm == "calibration":
                if index+1<len(self.nodes) and self.nodes[index+1]["algorithm"] == "alignment":
                    # Adjacent calibration / alignment only need sampled panels;
                    # avoid allocating and calibrating the entire sensor twice.
                    data = super().apply(data, photometry)
                    fused_alignment = True
                else:
                    data = np.where(self.sensor_left_mask, photometry.apply(data, "left"), photometry.apply(data, "right"))
            elif algorithm == "alignment":
                if not fused_alignment:
                    data = super().apply(data, IdentityPhotometry())
            elif isinstance(data, tuple):
                data = tuple(transform_array(a, algorithm, p) if p["target"] in {"both", side} else a for a, side in zip(data, ("left", "right")))
            else:
                data = transform_array(data, algorithm, p)
        return data

    def postprocess(self, values):
        after = False
        for node in self.nodes:
            if after:
                values = transform_array(values, node["algorithm"], node["parameters"])
            if node["algorithm"] == "correlation":
                after = True
        return values


def build_geometry(config, shape):
    return WorkflowGeometry(config, shape) if config.workflow else Geometry(config.geometry, shape)


def output_semantics(nodes):
    after, post = False, []
    for node in nodes:
        if not node.get("enabled", True):
            continue
        if after:
            post.append(node)
        after |= node["algorithm"] == "correlation"
    reference = np.ones((1, 1), dtype=float)
    for node in post:
        reference = transform_array(reference, node["algorithm"], node["parameters"])
    number = float(reference[0, 0])
    return {"result_semantics": "processed_g2" if post else "g2", "coefficient_reference": number if math.isfinite(number) else None, "postprocessing": deepcopy(post)}
