from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import hashlib
import importlib.util
import json
import math
import sys

import numpy as np

from eip.domain import StepSpec, validate_pixels
from eip.module_ops import gaussian, intensity_curve


class Cancelled(Exception):
    pass


@dataclass
class RunContext:
    cancelled: Callable[[], bool] = lambda: False
    log: Callable[[str], None] = lambda text: None
    seed: int = 0

    def check_cancelled(self):
        if self.cancelled():
            raise Cancelled("任务已取消")


@dataclass(frozen=True)
class Parameter:
    key: str
    label: str
    kind: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    description: str
    parameters: tuple[Parameter, ...] = ()
    version: str = "1.0"
    manifest: str | None = None
    code_hash: str | None = None
    execution_scope: str = "frame"
    output_kind: str = "image"
    category: str = "algorithm"
    icon: str = "plugin"
    documentation: dict = field(default_factory=dict)

    def defaults(self) -> dict:
        return {parameter.key: parameter.default for parameter in self.parameters}


BUILTINS = [
    AlgorithmSpec("intensity", "强度线性变换", "输出 = 输入 × 增益 + 偏置；整数结果裁剪到原类型范围。", (Parameter("gain", "增益", "float", 1.0, -1000, 1000), Parameter("offset", "偏置", "float", 0.0, -1e9, 1e9))),
    AlgorithmSpec("box_blur", "均值滤波", "对当前输入的空间矩阵使用反射边界；保留原类型。ROI 的前后顺序决定滤波上下文。", (Parameter("radius", "半径", "int", 1, 1, 32, "px"),)),
    AlgorithmSpec("threshold", "阈值分割", "多颜色样本先取算术平均；输出 0/1 的 uint8 标签图。", (Parameter("threshold", "阈值", "float", 128.0, -1e12, 1e12),), output_kind="label"),
    AlgorithmSpec("affine", "线性缩放与偏置", "y = gain × x + offset；浮点输出，不裁剪或量化。", (Parameter("gain", "倍率", "float", 1., -1e12, 1e12), Parameter("offset", "偏置", "float", 0., -1e12, 1e12))),
    AlgorithmSpec("gaussian", "空间高斯平滑", "可分离高斯核，边界复制、有效权重归一；原无效位置保持无效。", (Parameter("sigma", "σ", "float", 1., 1e-12, 1e12, "px"), Parameter("radius", "半径", "int", 2, 1, 32, "px"))),
    AlgorithmSpec("curve", "强度曲线", "整段固定尺度；非线性预设会改变强度涨落及关联，不隐式截断。", (Parameter("curve", "曲线", "str", "linear", choices=("identity", "linear", "sqrt", "square", "gamma", "log")), Parameter("input_max", "固定输入尺度", "float", 1023., 1e-12, 1e15), Parameter("gamma", "指数 γ", "float", 1., 1e-12, 1e6))),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: str) -> AlgorithmSpec:
    file = Path(path).resolve()
    value = json.loads(file.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("插件描述版本不支持")
    module, symbol = value["entry_point"].rsplit(":", 1)
    module_path = (file.parent / module).resolve()
    if module_path.suffix != ".py" or not module_path.is_file() or not symbol.isidentifier():
        raise ValueError("插件入口必须是存在的 Python 文件及类名")
    parameters = tuple(Parameter(**item) for item in value.get("parameters", []))
    if len({p.key for p in parameters}) != len(parameters) or len(parameters) > 32:
        raise ValueError("插件参数重复或超过 32 项")
    digest = hashlib.sha256(file.read_bytes() + module_path.read_bytes()).hexdigest()
    if value.get("category","algorithm") not in {"intermediate","algorithm","modifier"} or not isinstance(value.get("documentation",{}),dict) or not isinstance(value.get("icon","plugin"),str):
        raise ValueError("插件分类或文档描述无效")
    spec = AlgorithmSpec(value["id"], value["name"], value.get("description", ""), parameters, value.get("version", "1.0"), str(file), digest, value.get("execution_scope", "frame"), value.get("output_kind", "image"),value.get("category","algorithm"),value.get("icon","plugin"),value.get("documentation",{}))
    if spec.execution_scope != "frame":
        raise ValueError("当前执行器支持逐帧插件；时序/整体算法需后续执行器适配")
    if spec.output_kind not in {"image", "label"}:
        raise ValueError("当前处理链支持图像和标签输出")
    validate_parameters(spec, spec.defaults())
    return spec


def validate_parameters(spec: AlgorithmSpec, values: dict) -> dict:
    unknown = set(values) - {p.key for p in spec.parameters}
    if unknown:
        raise ValueError(f"未知参数：{', '.join(sorted(unknown))}")
    result = {}
    for parameter in spec.parameters:
        value = values.get(parameter.key, parameter.default)
        if parameter.kind == "bool":
            if type(value) is not bool:
                raise ValueError(f"{parameter.label} 必须为布尔值")
        elif parameter.kind in {"int", "float"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{parameter.label} 必须为有限数值")
            if parameter.kind == "int" and int(value) != value:
                raise ValueError(f"{parameter.label} 必须是整数")
            if parameter.minimum is not None and value < parameter.minimum or parameter.maximum is not None and value > parameter.maximum:
                raise ValueError(f"{parameter.label}（{parameter.key}）超出允许范围")
            value = int(value) if parameter.kind == "int" else float(value)
        elif parameter.kind == "str":
            if not isinstance(value, str):
                raise ValueError(f"{parameter.label} 必须是文本")
        else:
            raise ValueError(f"不支持参数类型 {parameter.kind}")
        if parameter.choices and value not in parameter.choices:
            raise ValueError(f"{parameter.label} 不在允许选项中")
        result[parameter.key] = value
    return result


def preserve_dtype(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        value = np.clip(np.rint(value), info.min, info.max)
    elif dtype.kind == "b":
        value = value != 0
    return value.astype(dtype)


class BuiltinAlgorithm:
    def __init__(self, identifier: str):
        self.identifier = identifier

    def prepare(self, parameters: dict, context: RunContext):
        self.parameters = parameters

    def process(self, pixels: np.ndarray, context: RunContext) -> np.ndarray:
        context.check_cancelled()
        parameters = self.parameters
        if pixels.dtype.kind in "ui" and pixels.dtype.itemsize > 4:
            raise ValueError("内置算法暂不处理 64 位整数；请接入声明精度策略的专业插件")
        if self.identifier == "intensity":
            return preserve_dtype(pixels.astype(np.float64) * parameters["gain"] + parameters["offset"], pixels.dtype)
        if self.identifier == "affine":
            with np.errstate(over="ignore", invalid="ignore"):
                result = pixels.astype(np.float64)*parameters["gain"] + parameters["offset"]
            return np.where(np.isfinite(result), result, np.nan)
        if self.identifier == "gaussian":
            return gaussian(pixels, parameters["sigma"], parameters["radius"])
        if self.identifier == "curve":
            return intensity_curve(pixels, parameters["curve"], parameters["input_max"], parameters["gamma"])
        if self.identifier == "threshold":
            value = pixels.astype(np.float64)
            if value.ndim == 3:
                value = value[..., :3].mean(axis=2)
            return (value >= parameters["threshold"]).astype(np.uint8)
        radius = parameters["radius"]
        if not np.isfinite(pixels).all():
            raise ValueError("均值滤波要求有限像素；请先显式处理 NaN/Inf")
        pad = [(radius, radius), (radius, radius)] + ([(0, 0)] if pixels.ndim == 3 else [])
        value = np.pad(pixels.astype(np.float64), pad, mode="reflect")
        integral = np.pad(value, [(1, 0), (1, 0)] + ([(0, 0)] if pixels.ndim == 3 else []), mode="constant").cumsum(axis=0).cumsum(axis=1)
        width = 2 * radius + 1
        result = (integral[width:, width:] - integral[:-width, width:] - integral[width:, :-width] + integral[:-width, :-width]) / width ** 2
        return preserve_dtype(result, pixels.dtype)

    def close(self):
        pass


class Pipeline:
    """Numerical pipeline shared by exact trial and formal export. No GUI imports."""
    def __init__(self, steps: list[StepSpec], context: RunContext):
        self.context = context
        self.instances = []
        self.module_names = []
        self.module_paths = []
        self.output_kind = "image"
        try:
            for step in steps:
                if not step.enabled:
                    continue
                context.check_cancelled()
                if step.manifest:
                    spec = load_manifest(step.manifest)
                    if spec.id != step.algorithm or step.code_hash != spec.code_hash or step.version != spec.version:
                        raise ValueError("插件内容已变化，请重新注册插件后运行")
                    value = json.loads(Path(step.manifest).read_text(encoding="utf-8"))
                    module_path, symbol = value["entry_point"].rsplit(":", 1)
                    module_path = (Path(step.manifest).parent / module_path).resolve()
                    module_spec = importlib.util.spec_from_file_location(f"eip_plugin_{spec.code_hash[:12]}", module_path)
                    module = importlib.util.module_from_spec(module_spec)
                    sys.modules[module_spec.name] = module
                    self.module_names.append(module_spec.name)
                    if str(module_path.parent) not in sys.path:
                        sys.path.insert(0, str(module_path.parent))
                        self.module_paths.append(str(module_path.parent))
                    module_spec.loader.exec_module(module)
                    instance = getattr(module, symbol)()
                else:
                    spec = next((s for s in BUILTINS if s.id == step.algorithm), None)
                    if spec is None:
                        raise ValueError(f"未知算法 {step.algorithm}")
                    instance = BuiltinAlgorithm(spec.id)
                parameters = validate_parameters(spec, step.parameters)
                self.instances.append(instance)
                instance.prepare(parameters, context)
                self.output_kind = spec.output_kind
        except Exception:
            self.close()
            raise

    def execute(self, pixels: np.ndarray) -> np.ndarray:
        validate_pixels(pixels)
        data = pixels.copy()
        for instance in self.instances:
            self.context.check_cancelled()
            data.setflags(write=False)
            output = instance.process(data, self.context)
            validate_pixels(output)
            if output.shape[:2] != pixels.shape[:2]:
                raise ValueError("当前插件链要求保留空间尺寸；几何变换需要坐标适配器")
            data = np.array(output, copy=True)
        if self.output_kind == "label" and (data.dtype.kind not in "bui" or data.ndim != 2):
            raise ValueError("标签输出必须为二维布尔或非负整数图")
        return data

    def close(self):
        for instance in reversed(self.instances):
            try:
                instance.close()
            except Exception as error:
                self.context.log(f"释放算法资源时发生错误：{error}")
        self.instances.clear()
        for name in self.module_names:
            sys.modules.pop(name, None)
        self.module_names.clear()
        for path in self.module_paths:
            if path in sys.path:
                sys.path.remove(path)
        self.module_paths.clear()
