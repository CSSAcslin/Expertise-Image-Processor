"""One module catalogue and instance contract, independent of workbench modes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import uuid

from eip.algorithms import BUILTINS, load_manifest, validate_parameters


def field(key, label, kind, default, **options):
    return dict(key=key, label=label, kind=kind, default=default, **options)


STRUCTURAL = {
    "calibration": dict(name="强度校准 / LUT", inputs=["sensor"], output="sensor", description="固定强度曲线、LUT、暗场与无效值标记；输入含义随上游模块确定。", parameters=[
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
        field("window", "平均窗口 W（输入样本，奇数）", "int", 101, minimum=1), field("min_mean", "最小强度均值", "float", 1e-9, minimum=1e-15), field("min_samples", "最少有效样本", "int", 2, minimum=1),
        field("stride", "输出步长（输入样本）", "int", 100, minimum=1), field("analysis", "关联输出", "str", "map", choices=["map", "curve"]), field("map_lag", "固定 τ（输入样本）", "int", 0),
        field("lag_min", "τ 下限（输入样本）", "int", -10), field("lag_max", "τ 上限（输入样本）", "int", 10), field("lag_step", "延迟步长（输入样本）", "int", 1, minimum=1), field("lag_list", "延迟列表（留空使用范围）", "array", None, nullable=True),
        field("roi_estimator", "ROI 曲线估计", "str", "pixel_mean", choices=["pixel_mean", "intensity_sum"])])}

AXIS_MODULES = {
    "time_crop": dict(name="时间裁剪", description="流程起点：按样本或真实时间保留数据；后续模块不再重复设置运行范围。", parameters=[
        field("unit", "范围单位", "str", "samples", choices=["samples", "seconds"]),
        field("first", "首样本（从 1 起）", "int", 1, minimum=1), field("last", "尾样本（包含，留空至末尾）", "int", None, minimum=1, nullable=True),
        field("start", "起始时间（秒）", "float", 0., minimum=0), field("end", "结束时间（秒，不含）", "float", None, minimum=0, nullable=True)]),
    "spatial_roi": dict(name="空间 ROI / 区域归约", description="相对本步骤输入裁剪；保留像素，或对空间轴取均值、总和、分位值。归约后仅保留时间及显式通道。", parameters=[
        field("x", "X", "int", 0, minimum=0), field("y", "Y", "int", 0, minimum=0),
        field("width", "宽（留空至边界）", "int", None, minimum=1, nullable=True), field("height", "高（留空至边界）", "int", None, minimum=1, nullable=True),
        field("mode", "区域操作", "str", "pixels", choices=["pixels", "mean", "sum", "quantile"]), field("quantile", "分位 q", "float", .5, minimum=0, maximum=1),
        field("validity", "双路归约有效像素", "str", "joint", choices=["joint", "independent"])]),
    "time_sample": dict(name="时间取间隔", description="按当前序列取间隔，保留实际样本时间与来源。", parameters=[field("step", "样本间隔", "int", 2, minimum=1), field("offset", "起始偏移（从 0 起）", "int", 0, minimum=0)]),
    "time_smooth": dict(name="时间加窗平滑", description="逐元素滑动均值或分位值；只输出完整窗口，时间取中心样本。", parameters=[field("window", "窗口 W（奇数）", "int", 3, minimum=1), field("mode", "窗口操作", "str", "mean", choices=["mean", "quantile"]), field("quantile", "分位 q", "float", .5, minimum=0, maximum=1)]),
    "time_bin": dict(name="时间合帧", description="连续样本逐元素取均值或总和；不完整尾组舍弃，保留贡献样本记录。", parameters=[field("size", "每组样本数", "int", 2, minimum=1), field("mode", "合帧操作", "str", "mean", choices=["mean", "sum"])])}
for _value in AXIS_MODULES.values():
    _value.update(inputs=["sensor", "pair", "moments", "map"], output="same")
STRUCTURAL["correlation"]["parameters"].append(field("center", "固定中心样本（编辑后，从 1 起；留空居中）", "int", None, minimum=1, nullable=True))
STRUCTURAL["statistics"].update(name="上游序列统计与固定尺度",inputs=["sensor","pair"],description="统计本节点直接收到的完整时间序列；此前的裁剪、抽样、合帧及空间归约都参与计算。")
STRUCTURAL["calibration"]["inputs"] = ["sensor","pair"]

WORKBENCH = {
    "spatial_smooth": dict(description="选择均值或高斯空间平滑；输出类型由实例精度决定。", parameters=[
        field("method","平滑类型","str","mean",choices=["mean","gaussian"]),
        field("radius","空间半径 / px","int",1,minimum=1,maximum=32), field("sigma","高斯 σ / px","float",1.,minimum=1e-12,maximum=1e12)]),
    "gain_offset": dict(description="增益乘法与偏置加法；算法默认保留输入类型，浮点精度不先量化。", parameters=[
        field("gain","增益","float",1.,minimum=-1e12,maximum=1e12),field("offset","偏置","float",0.,minimum=-1e12,maximum=1e12)]),
    "time_statistics": dict(description="逐样本统计空间 / 分量，保留 T 或 Q 及显式 P/C；可替换主路或输出旁路。", parameters=[
        field("mode","统计量","str","mean",choices=["mean","sum","min","max","variance","std","quantile","count"]),
        field("quantile","分位 q","float",.5,minimum=0,maximum=1),
        field("validity","双路有效值","str","joint",choices=["joint","independent"]),field("output","数据流作用","str","main",choices=["main","side"])]),
    "global_statistics": dict(description="压缩样本轴得到一条静态统计记录；默认统计全部数据，可仅压缩样本轴或保留 P/C。", parameters=[
        field("scope","统计范围","str","all",choices=["all","samples","space_time"]),
        field("mode","统计量","str","mean",choices=["mean","sum","min","max","variance","std","quantile","count"]),
        field("quantile","分位 q","float",.5,minimum=0,maximum=1),
        field("validity","双路有效值","str","independent",choices=["joint","independent"]),field("output","数据流作用","str","main",choices=["main","side"])]),
    "roi_crop": dict(description="裁剪直接输入的矩形空间区域。", parameters=[deepcopy(p) for p in AXIS_MODULES["spatial_roi"]["parameters"][:4]]),
    "region_reduce": dict(description="对当前完整空间区域求均值、总和或分位值。", parameters=[
        field("mode","区域运算","str","mean",choices=["mean","sum","quantile"]), field("quantile","分位 q","float",.5,minimum=0,maximum=1),
        field("validity","双路有效像素","str","joint",choices=["joint","independent"]), field("output","数据流作用","str","main",choices=["main","side"])]),
    "sequence_statistics": dict(description="扫描直接上游完整 T 序列，主流透传并产生统计端口。", parameters=[field("normalization","尺度统计","str","fixed_rms",choices=["none","fixed_rms"])]),
    "fixed_normalize": dict(description="使用明确固定尺度归一，可引用前序统计的 scale 端口。",parameters=[field("scale","固定尺度","float",1.,minimum=1e-15)]),
    "parameter_binding": dict(description="主流透传，发布命名数值变量，可取常量或前序参数 / 统计端口。",parameters=[field("name","变量名","str","value"),field("value","变量值","float",1.),field("unit","单位","str","")]),
    "grayscale": dict(description="将明确 RGB 颜色分量转换为灰度。",parameters=[field("method","灰度定义","str","luma",choices=["luma","mean"])]),
    "color_convert": dict(description="灰度扩展为 RGB，或明确调整 RGB/BGR 排列。",parameters=[field("mode","输出排列","str","rgb",choices=["rgb","bgr"])]),
    "point_extract": dict(description="获取直接输入指定像素随 T/Q 的信号。",parameters=[field("x","X","int",0,minimum=0),field("y","Y","int",0,minimum=0),field("output","数据流作用","str","main",choices=["main","side"])]),
    "line_extract": dict(description="提取横纵便捷线，或按端点及插值采样任意直线，R 为距离位置轴。",parameters=[field("mode","直线定义","str","axis",choices=["axis","segment"]),field("axis","保留方向","str","X",choices=["X","Y"]),field("index","固定另一轴位置（从0起）","int",0,minimum=0),*[field(k,label,"float",0.,minimum=0) for k,label in (("x0","起点 X"),("y0","起点 Y"),("x1","终点 X"),("y1","终点 Y"))],field("samples","沿线样本数","int",64,minimum=2,maximum=100000),field("interpolation","沿线插值","str","nearest",choices=["nearest","bilinear"]),field("output","数据流作用","str","side",choices=["main","side"])]),
    "axis_slice": dict(description="按明确轴选一个截面，不重新解释 T/Q。",parameters=[field("axis","截面轴","str","T",choices=["T","Q","P","C","X","Y","R","B","N"]),field("index","位置（从0起）","int",0,minimum=0),field("output","数据流作用","str","side",choices=["main","side"])]),
    "distribution": dict(description="对每个上游样本产生固定箱边界的频数；自动边界使用上游完整范围的有限值最小 / 最大值。",parameters=[field("bins","箱数","int",64,minimum=2,maximum=4096),field("minimum","下界","float",0.,nullable=True),field("maximum","上界","float",65535.,nullable=True),field("output","数据流作用","str","side",choices=["main","side"])]),
    "data_tap": dict(description="获取直接上游真实数据用于预览或文件导出，主流原样通过。",parameters=[field("name","输出名称","str","截流输出"),field("destination","交付方式","str","both",choices=["preview","file","both"]),field("export_format","导出格式","str","dataset",choices=["dataset","npy","csv","tiff"])])}
for _key,_value in WORKBENCH.items():
    _value.update(name=_key,inputs=["sensor","pair","moments","map"],output="same")
for _key,_analysis in (("correlation_map","map"),("correlation_curve","curve")):
    WORKBENCH[_key] = deepcopy(STRUCTURAL["correlation"])
    for _p in WORKBENCH[_key]["parameters"]:
        if _p["key"]=="analysis":
            _p["default"]=_analysis

CORRELATIONS = {"correlation","correlation_map","correlation_curve"}
MODIFIERS = {"region_reduce","time_statistics","global_statistics","point_extract","line_extract","axis_slice","distribution"}


def needs_full_sequence(nodes):
    return any(n["enabled"] and (n["algorithm"] in {"time_sample","time_smooth","time_bin","global_statistics"}
               or n["algorithm"]=="distribution" and any(n["parameters"].get(k) is None for k in ("minimum","maximum")))
               for n in nodes)
REFERENCABLE = {"intensity":{"gain","offset"},"affine":{"gain","offset"},"gain_offset":{"gain","offset"},"threshold":{"threshold"},"curve":{"input_max","gamma"},"gaussian":{"sigma"},"spatial_smooth":{"sigma"},"fixed_normalize":{"scale"},"parameter_binding":{"value"}}


def is_reference(value):
    return isinstance(value,dict) and set(value)=={"ref"}


def catalog(plugins=()):
    result = {}
    for spec in (*BUILTINS, *[s for s in plugins if s.manifest]):
        result[spec.id] = dict(asdict(spec), inputs=["sensor", "pair", "map"], output="same", repeatable=True)
    for key, spec in {**STRUCTURAL, **AXIS_MODULES, **WORKBENCH}.items():
        result[key] = dict(deepcopy(spec), id=key, version="1.0", manifest=None, code_hash=None, execution_scope="sequence" if key in {"statistics", "sequence_statistics", "global_statistics", *CORRELATIONS, "time_crop", "time_sample", "time_smooth", "time_bin", "data_tap", "axis_slice"} else "frame", output_kind="image", repeatable=True)
    for spec in result.values():
        from eip.gpu_ops import GPU_MODULES
        spec["backends"] = ["cpu","gpu"] if spec["id"] in GPU_MODULES else ["cpu"]
        spec["defaults"] = defaults(spec)
        spec["bindable_parameters"] = sorted({p["key"] for p in spec["parameters"] if p["kind"] in {"int","float"}} if spec.get("manifest") else REFERENCABLE.get(spec["id"],set()))
        from eip.module_documentation import enrich
        enrich(spec)
        spec["ports"] = ([dict(id=port,kind="statistic",dtype="uint32" if port=="count" else "float64") for port in ("mean","variance","count","scale")] if spec["id"]=="sequence_statistics" else [dict(id="value",kind="statistic",dtype="selected"),dict(id="count",kind="statistic",dtype="uint64")] if spec["id"]=="global_statistics" else [dict(id="value",kind="scalar",dtype="float64")] if spec["id"]=="parameter_binding" else [dict(id="data",kind="upstream" if spec["id"]=="data_tap" else "output")])
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
    if algorithm in CORRELATIONS and frames:
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
        if not isinstance(raw, dict) or set(raw)-{"id", "algorithm", "parameters", "enabled", "name", "version", "manifest", "code_hash", "target", "execution"}:
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
        bindings = {key:deepcopy(value) for key,value in p.items() if is_reference(value)}
        for key,binding in bindings.items():
            ref = binding["ref"]
            allowed_bindings={item["key"] for item in spec["parameters"] if item["kind"] in {"int","float"}} if spec.get("manifest") else REFERENCABLE.get(algorithm,set())
            if key not in allowed_bindings or not isinstance(ref,dict) or set(ref)!={"node_id","port"} or not all(isinstance(ref[k],str) and ref[k] for k in ref):
                raise ValueError("参数引用需为可绑定数值参数及前序实例的明确端口")
            p[key] = deepcopy(spec["defaults"][key])
        execution = node.get("execution", {})
        if not isinstance(execution, dict) or set(execution)-{"backend", "memory_mb", "precision"}:
            raise ValueError("模块执行设置无效")
        if execution.get("backend", "inherit") not in {"inherit", "cpu", "auto", "gpu"} or execution.get("precision", "native") not in {"native", "float32", "float64"}:
            raise ValueError("模块设备或精度设置无效")
        if execution.get("memory_mb") is not None:
            integer(execution["memory_mb"], "模块内存上限", 16)
            if execution["memory_mb"]>131072:
                raise ValueError("模块内存上限不能超过 131072 MiB")
        from eip.gpu_ops import GPU_MODULES
        if execution.get("backend")=="gpu" and algorithm not in GPU_MODULES:
            raise ValueError("此模块当前支持 CPU，请选择继承或 CPU")
        node["execution"] = deepcopy(execution)
        if algorithm in {*STRUCTURAL,*AXIS_MODULES,*WORKBENCH}-{"spatial_smooth","gain_offset"} and node["target"] != "both":
            raise ValueError("双路结构与时序模块作用于完整输入；左 / 右路范围仅适用于逐帧算法")
        if algorithm in AXIS_MODULES or algorithm in WORKBENCH and algorithm not in CORRELATIONS:
            if set(p)-set(spec["defaults"]):
                raise ValueError("空间 / 时间编辑模块包含未知参数")
            merged = deepcopy(spec["defaults"])
            merged.update(p)
            for parameter in spec["parameters"]:
                key, item = parameter["key"], merged[parameter["key"]]
                if parameter.get("nullable") and item is None:
                    continue
                if parameter["kind"]=="int":
                    integer(item, parameter["label"], parameter.get("minimum"))
                    if parameter.get("maximum") is not None and item>parameter["maximum"]:
                        raise ValueError(f"{parameter['label']} 超出允许范围")
                elif parameter["kind"]=="float":
                    v = finite(item, parameter["label"])
                    if v<parameter.get("minimum", -float("inf")) or v>parameter.get("maximum", float("inf")):
                        raise ValueError(f"{parameter['label']} 超出允许范围")
                elif parameter["kind"]=="str" and not isinstance(item,str):
                    raise ValueError(f"{parameter['label']} 必须为文字")
                if parameter.get("choices") and item not in parameter["choices"]:
                    raise ValueError(f"{parameter['label']} 选项无效")
            if algorithm=="time_crop" and (merged["last"] is not None and merged["last"]<merged["first"] or merged["end"] is not None and merged["end"]<=merged["start"]):
                raise ValueError("时间裁剪终点必须在起点之后")
            if algorithm=="time_smooth" and merged["window"]%2!=1:
                raise ValueError("时间平滑窗口必须为奇数")
            if algorithm=="distribution" and merged["minimum"] is not None and merged["maximum"] is not None and merged["maximum"]<=merged["minimum"]:
                raise ValueError("分布上界须大于下界")
            if algorithm=="data_tap" and (not isinstance(merged["name"],str) or not 1<=len(merged["name"].strip())<=120):
                raise ValueError("截流输出名称需为1至120个字符")
            node["parameters"] = merged
        elif frame_spec:
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
            if merged["center"] is not None:
                integer(merged["center"],"关联中心样本",1)
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
            if algorithm in {"correlation_map","correlation_curve"} and merged["analysis"]!=("map" if algorithm=="correlation_map" else "curve"):
                raise ValueError("关联输出与此模块入口不一致")
            if merged["lag_list"] is not None:
                if not isinstance(merged["lag_list"], (list, tuple)) or len(merged["lag_list"])>4096:
                    raise ValueError("延迟列表需为最多 4096 项的列表")
                for q in merged["lag_list"]:
                    integer(q, "延迟")
            node["parameters"] = merged
        node["parameters"].update(bindings)
        if (algorithm in {"data_tap","sequence_statistics","parameter_binding"} or algorithm in MODIFIERS and node["parameters"]["output"]=="side") and execution.get("precision","native")!="native":
            raise ValueError("透传 / 旁路模块不修改主流数值精度；请在实际变换模块设置精度")
        if node["enabled"] and check_types:
            if state not in spec["inputs"]:
                raise ValueError(f"第 {position+1} 个模块「{spec['name']}」无法接收数据类型 {state}；请调整顺序或补充需要的输入模块")
            if node["target"]!="both" and state!="pair":
                raise ValueError("左 / 右路作用范围仅适用于已对齐双路")
            if algorithm in {"statistics", "sequence_statistics", *CORRELATIONS} and labelled:
                raise ValueError("标签图不能隐式当作光强计算 g²；请调整阈值模块的位置")
            if spec["output"]!="same":
                state = spec["output"]
            if frame_spec:
                kind = frame_spec.output_kind
                labelled |= kind=="label"
        result.append(node)
    return tuple(result), state, kind


def temporal(nodes):
    return any(n.get("enabled", True) and n["algorithm"] in {"statistics", "sequence_statistics", "spatial_smooth", "gain_offset", "time_statistics", "global_statistics", *CORRELATIONS, "time_crop", "time_sample", "time_smooth", "time_bin", "data_tap", "axis_slice"} for n in nodes)


def migrate_nodes(nodes):
    """Create an editable canonical copy; never reinterpret completed datasets."""
    result = []
    for raw in nodes:
        n = deepcopy(raw)
        a,p = n["algorithm"],n.setdefault("parameters",{})
        if a=="spatial_roi":
            mode = p.get("mode","pixels")
            crop = deepcopy(n)
            crop.update(algorithm="roi_crop",parameters={k:p.get(k,catalog()["roi_crop"]["defaults"][k]) for k in ("x","y","width","height")})
            if mode=="pixels":
                result.append(crop)
            else:
                crop["id"] = f"{n.get('id') or uuid.uuid4().hex}-crop"
                result.append(crop)
                n.update(algorithm="time_statistics",parameters=dict(mode=mode,quantile=p.get("quantile",.5),validity=p.get("validity","joint"),output="main"))
                result.append(n)
        elif a=="region_reduce":
            n["algorithm"]="time_statistics"
            result.append(n)
        elif a in {"affine","gaussian"} or a in {"intensity","box_blur"} and n.get("execution",{}).get("precision","native")=="native":
            if a in {"affine","gaussian"}:
                execution=n.setdefault("execution",{})
                if execution.get("precision","native")=="native":
                    execution["precision"]="float64"
            n["algorithm"]="gain_offset" if a in {"intensity","affine"} else "spatial_smooth"
            if a in {"box_blur","gaussian"}:
                n["parameters"]=dict(method="mean" if a=="box_blur" else "gaussian",radius=p["radius"],sigma=p.get("sigma",1.))
            result.append(n)
        elif a=="statistics":
            stats = deepcopy(n)
            stats.update(id=f"{n.get('id') or uuid.uuid4().hex}-stats",algorithm="sequence_statistics",parameters=dict(normalization=p.get("normalization","none")))
            stats.setdefault("execution",{})["precision"]="native"
            result.append(stats)
            n.update(algorithm="fixed_normalize",parameters=dict(scale={"ref":{"node_id":stats["id"],"port":"scale"}}))
            result.append(n)
        elif a=="correlation":
            n["algorithm"]="correlation_curve" if p.get("analysis")=="curve" else "correlation_map"
            result.append(n)
        else:
            result.append(n)
    return result


def migrate_workbench_nodes(nodes):
    migrated=migrate_nodes(nodes)
    report=[]
    if any(n.get("algorithm")=="spatial_roi" for n in nodes):
        report.append("旧 ROI 已拆为裁剪 / 区域归约，原坐标、有效像素规则与顺序保留。")
    if any(n.get("algorithm")=="statistics" for n in nodes):
        report.append("旧统计已拆为序列统计 / 显式固定尺度归一，保留原 float64 与固定尺度。")
    if any(n.get("algorithm")=="correlation" for n in nodes):
        report.append("关联输出已转换为固定延迟图或延迟扫描的独立模块入口。")
    if any(n.get("algorithm") in {"affine","intensity","box_blur","gaussian","region_reduce"} for n in nodes):
        report.append("平滑、增益和区域统计已迁入合并入口；原数值精度保留。旧定类型节点若另设浮点覆盖则保留兼容计算，避免改变原来的先量化行为。")
    if any(n.get("algorithm")=="sequence_statistics" for n in nodes):
        report.append("旧序列统计保留为兼容节点及原参数端口；新模块库使用时序统计 / 全域统计，g² 模板不再添加序列统计。")
    return migrated,report


def output_shape(nodes, shape):
    from eip.modes.split_correlation.config import GeometryConfig
    from eip.modes.split_correlation.geometry import Geometry
    for node in nodes:
        if node.get("enabled",True) and node["algorithm"]=="alignment":
            shape = Geometry(GeometryConfig.from_dict(node["parameters"]),tuple(shape)).shape
    return tuple(shape)
