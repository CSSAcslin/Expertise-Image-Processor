"""One module catalogue and instance contract, independent of workbench modes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import uuid

from eip.algorithms import BUILTINS, load_manifest, validate_parameters


def field(key, label, kind, default, **options):
    return dict(key=key, label=label, kind=kind, default=default, **options)


STRUCTURAL = {
    "calibration": dict(name="强度校准 / LUT", inputs=["sensor"], output="sensor", description="传感器采样前的固定强度曲线、LUT、暗场及坏点排除；可删除或重复。", parameters=[
        field("curve", "强度曲线", "str", "identity", choices=["identity", "linear", "sqrt", "square", "gamma", "log"]),
        field("input_max", "固定输入尺度", "float", 1023., minimum=1e-12), field("gamma", "指数 γ", "float", 1., minimum=1e-12),
        field("lut_path", "自定义 LUT 路径", "str", None, nullable=True), field("black_left", "左路 / 单幅暗场偏置", "float", 0.), field("black_right", "右路暗场偏置", "float", 0.),
        field("saturation_code", "饱和码值阈值", "float", None, nullable=True), field("input_semantics", "输入含义", "str", "sensor_codes", choices=["sensor_codes", "linear_intensity"])]),
    "alignment": dict(name="双幅划分与固定配准", inputs=["sensor"], output="pair", description="划分有效区并用固定几何映射得到对齐双路。", parameters=[
        *[field(f"{side}.{key}", f"{label} {title}", "int", default, minimum=0 if key in {"x", "y"} else 1) for side, label in (("left", "左幅"), ("right", "右幅")) for key, title, default in (("x", "X", 0), ("y", "Y", 0), ("width", "宽", 1), ("height", "高", 1))],
        field("midline", "中线顶部 / 底部 X", "array", [1., 1.]), field("dx", "右路采样偏移 X", "float", 0.), field("dy", "右路采样偏移 Y", "float", 0.),
        field("mirror_x", "右路水平镜像", "bool", False), field("mirror_y", "右路垂直镜像", "bool", False), field("interpolation", "空间采样", "str", "nearest", choices=["nearest", "bilinear"])]),
    "select_channel": dict(name="选择幅面", inputs=["pair"], output="sensor", description="将对齐双路中的左路或右路作为后续单幅输入；保留几何来源。", parameters=[field("channel", "输出幅面", "str", "left", choices=["left", "right"])]),
    "statistics": dict(name="全源均值与固定尺度", inputs=["pair"], output="moments", description="全序列均值 / 方差与每路固定尺度；删除后关联直接使用强度，且不扫描全源背景。", parameters=[field("normalization", "固定尺度", "str", "none", choices=["none", "fixed_rms"])]),
    "correlation": dict(name="标准二阶互关联 g²", inputs=["pair", "moments"], output="map", description="同一有效样本集合上的二阶互关联；支持固定 τ 图或双向 τ 曲线。", parameters=[
        field("window", "平均窗口 W（奇数）", "int", 101, minimum=1), field("min_mean", "最小强度均值", "float", 1e-9, minimum=1e-15), field("min_samples", "最少有效样本", "int", 2, minimum=1),
        field("stride", "输出时间步长", "int", 100, minimum=1), field("analysis", "关联输出", "str", "map", choices=["map", "curve"]), field("map_lag", "固定 τ（帧）", "int", 0),
        field("lag_min", "τ 下限（帧）", "int", -10), field("lag_max", "τ 上限（帧）", "int", 10), field("lag_step", "延迟步长", "int", 1, minimum=1), field("lag_list", "延迟列表（留空使用范围）", "array", None, nullable=True),
        field("roi_estimator", "ROI 曲线估计", "str", "pixel_mean", choices=["pixel_mean", "intensity_sum"])])}


def catalog(plugins=()):
    result = {}
    for spec in (*BUILTINS, *[s for s in plugins if s.manifest]):
        result[spec.id] = dict(asdict(spec), inputs=["sensor", "pair", "map"], output="same", repeatable=True)
    for key, spec in STRUCTURAL.items():
        result[key] = dict(deepcopy(spec), id=key, version="1.0", manifest=None, code_hash=None, execution_scope="sequence" if key in {"statistics", "correlation"} else "frame", output_kind="image", repeatable=True)
    for spec in result.values():
        spec["defaults"] = defaults(spec)
    return result


def set_path(value, path, item):
    parts = path.split(".")
    for part in parts[:-1]:
        value = value.setdefault(part, {})
    value[parts[-1]] = deepcopy(item)


def defaults(spec):
    result = {}
    for p in spec["parameters"]:
        set_path(result, p["key"], p["default"])
    return result


def new_node(algorithm, shape=None, frames=None, plugins=()):
    spec = catalog(plugins)[algorithm]
    p = deepcopy(spec["defaults"])
    if algorithm == "alignment" and shape:
        h, w = shape[:2]
        mid = max(1, w//2)
        p.update(left=dict(x=0, y=0, width=mid, height=h), right=dict(x=mid, y=0, width=max(1, w-mid), height=h), midline=[mid, mid])
    if algorithm == "correlation" and frames:
        lag = min(10, max(0, (frames-3)//4))
        window = min(101, max(1, frames-2*lag))
        if window%2 == 0:
            window -= 1
        p.update(window=window, min_samples=min(2, window), lag_min=-lag, lag_max=lag)
    return dict(id=uuid.uuid4().hex, algorithm=algorithm, parameters=p, enabled=True, target="both", version=spec["version"], manifest=spec["manifest"], code_hash=spec["code_hash"])


def module_spec(node):
    if node.get("manifest"):
        spec = load_manifest(node["manifest"])
        if spec.id != node["algorithm"] or spec.code_hash != node.get("code_hash") or spec.version != node.get("version", "1.0"):
            raise ValueError("插件版本或内容已变化，请重新注册并添加模块")
        if spec.id in catalog():
            raise ValueError("插件不能覆盖内置模块")
        return catalog([spec])[spec.id], spec
    spec = catalog().get(node["algorithm"])
    if spec is None:
        raise ValueError(f"未知算法模块：{node['algorithm']}")
    if node.get("code_hash") is not None or node.get("version", "1.0") != spec["version"]:
        raise ValueError(f"{spec['name']} 的版本已变化，请重新添加模块")
    return spec, next((s for s in BUILTINS if s.id == node["algorithm"]), None)


def normalize_nodes(nodes, *, check_types=True, input_type="sensor", input_kind="image"):
    if not isinstance(nodes, (list, tuple)) or len(nodes)>32:
        raise ValueError("模块流程需为列表，最多 32 个模块")
    from eip.modes.split_correlation.config import GeometryConfig, PhotometryConfig, finite, integer
    result, identifiers, state, kind = [], set(), input_type, input_kind
    labelled = input_kind=="label"
    for position, raw in enumerate(nodes):
        if not isinstance(raw, dict) or set(raw)-{"id", "algorithm", "parameters", "enabled", "name", "version", "manifest", "code_hash", "target"}:
            raise ValueError("模块记录格式无效")
        node = deepcopy(raw)
        identifier = node.get("id") or f"legacy-{position}"
        node["id"] = identifier
        if not isinstance(identifier, str) or not 1<=len(identifier)<=80 or identifier in identifiers:
            raise ValueError("模块实例标识无效或重复")
        identifiers.add(identifier)
        if type(node.setdefault("enabled", True)) is not bool or not isinstance(node.setdefault("parameters", {}), dict):
            raise ValueError("模块启用状态或参数格式无效")
        p = node["parameters"]
        node.setdefault("target", p.pop("target", "both"))
        if node["target"] not in {"both", "left", "right"}:
            raise ValueError("模块作用范围必须为双路、左路或右路")
        spec, frame_spec = module_spec(node)
        node.setdefault("version", spec["version"])
        node.setdefault("manifest", spec["manifest"])
        node.setdefault("code_hash", spec["code_hash"])
        algorithm = node["algorithm"]
        if algorithm in STRUCTURAL and node["target"] != "both":
            raise ValueError("双路结构与时序模块作用于完整输入；左 / 右路范围仅适用于逐帧算法")
        if frame_spec:
            node["parameters"] = validate_parameters(frame_spec, p)
        elif algorithm == "alignment":
            node["parameters"] = asdict(GeometryConfig.from_dict(p))
        elif algorithm == "calibration":
            node["parameters"] = asdict(PhotometryConfig.from_dict(p))
            node["parameters"].pop("normalization", None)
        elif algorithm == "statistics":
            if set(p)-{"normalization"} or p.get("normalization", "none") not in {"none", "fixed_rms"}:
                raise ValueError("全源统计模块的固定尺度选项无效")
            node["parameters"] = {"normalization":p.get("normalization", "none")}
        elif algorithm == "select_channel":
            if set(p)-{"channel"} or p.get("channel", "left") not in {"left", "right"}:
                raise ValueError("输出幅面必须为左路或右路")
            node["parameters"] = {"channel":p.get("channel", "left")}
        else:
            allowed = set(spec["defaults"])
            if set(p)-allowed:
                raise ValueError("二阶关联模块包含未知参数")
            merged = dict(spec["defaults"], **p)
            for key in ("window", "min_samples", "stride", "lag_step"):
                integer(merged[key], key, 1)
            if merged["window"]%2 != 1 or merged["min_samples"]>merged["window"]:
                raise ValueError("关联窗口须为奇数，且不小于最少有效样本数")
            if finite(merged["min_mean"], "最小强度均值")<=0:
                raise ValueError("最小强度均值需为正数")
            for key in ("map_lag", "lag_min", "lag_max"):
                integer(merged[key], key)
            if merged["analysis"] not in {"map", "curve"} or merged["roi_estimator"] not in {"pixel_mean", "intensity_sum"}:
                raise ValueError("关联输出或 ROI 估计方式无效")
            if merged["lag_list"] is not None:
                if not isinstance(merged["lag_list"], (list, tuple)) or len(merged["lag_list"])>4096:
                    raise ValueError("延迟列表需为最多 4096 项的列表")
                for q in merged["lag_list"]:
                    integer(q, "延迟")
            node["parameters"] = merged
        if node["enabled"] and check_types:
            if state not in spec["inputs"]:
                raise ValueError(f"第 {position+1} 个模块「{spec['name']}」无法接收数据类型 {state}；请调整顺序或补充需要的输入模块")
            if node["target"]!="both" and state!="pair":
                raise ValueError("左 / 右路作用范围仅适用于已对齐双路")
            if algorithm in {"statistics", "correlation"} and labelled:
                raise ValueError("标签图不能隐式当作光强计算 g²；请调整阈值模块的位置")
            if spec["output"]!="same":
                state = spec["output"]
            if frame_spec:
                kind = frame_spec.output_kind
                labelled |= kind=="label"
        result.append(node)
    return tuple(result), state, kind


def temporal(nodes):
    return any(n.get("enabled", True) and n["algorithm"] in {"statistics", "correlation"} for n in nodes)


def output_shape(nodes, shape):
    from eip.modes.split_correlation.config import GeometryConfig
    from eip.modes.split_correlation.geometry import Geometry
    for node in nodes:
        if node.get("enabled",True) and node["algorithm"]=="alignment":
            shape = Geometry(GeometryConfig.from_dict(node["parameters"]),tuple(shape)).shape
    return tuple(shape)
