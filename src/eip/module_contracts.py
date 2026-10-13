"""Compile actual axes, sizes and scientific meaning for an editable module chain.

This compiler does no reading and executes no algorithms. Unknown sequence lengths
stay unknown until the reader has counted them; display FPS is never an index.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import numpy as np

from eip.module_registry import normalize_nodes, module_spec, CORRELATIONS, MODIFIERS, is_reference, needs_full_sequence
from eip.module_statistics import retained_axes

CONTRACT_VERSION = 3


@dataclass(frozen=True)
class Contract:
    axes: str
    shape: tuple[int, ...]
    length: int | None
    sequence_axis: str = "T"
    dtype: str = "float64"
    semantic: str = "intensity"
    statistics: bool = False
    origin: tuple[int, int] = (0, 0)
    spatial_operation: str | None = None
    meaning: str = "intensity"
    color_order: str = "RGB"

    @property
    def spatial(self):
        return "Y" in self.axes and "X" in self.axes

    @property
    def paired(self):
        return self.axes.startswith("P")

    def record(self):
        value = asdict(self)
        value["full_axes"] = self.sequence_axis+self.axes
        value["full_shape"] = [self.length, *self.shape]
        value["label"] = " × ".join(f"{a}:{'?' if n is None else n}" for a,n in zip(value["full_axes"],value["full_shape"]))
        meanings={"T":"时间样本","Q":"有符号延迟","N":"静态记录","R":"沿线距离 / 像素","B":"分布箱","P":"对齐双路","Y":"空间Y","X":"空间X","C":self.color_order+"颜色分量"}
        value["axis_meanings"]={a:meanings.get(a,a) for a in value["full_axes"]}
        return value


def source_contract(meta, kind="image"):
    # Typed saved results carry their own contract, including Q and scalar axes.
    if meta.get("contract"):
        v = meta["contract"]
        return Contract(**{k:tuple(v[k]) if k in {"shape","origin"} else v[k] for k in Contract.__dataclass_fields__ if k in v})
    shape = tuple(meta["shape"])
    semantic = "label" if kind=="label" else "g2" if kind=="coefficient" else "intensity"
    return Contract("YX" if len(shape)==2 else "YXC",shape,meta.get("frame_count") if meta["kind"]!="video" or meta.get("frame_indexed") else None,dtype=meta["dtype"],semantic=semantic,meaning="pixel_g2" if semantic=="g2" else semantic,color_order=meta.get("color") if meta.get("color") in {"RGB","BGR"} else "RGB")


def roi_bounds(contract, p):
    if not contract.spatial:
        raise ValueError("空间操作需要 Y/X 轴；上游已归约为空间标量或向量")
    h,w = (contract.shape[contract.axes.index(a)] for a in "YX")
    x,y = p["x"],p["y"]
    width,height = p["width"] or w-x,p["height"] or h-y
    if min(width,height)<1 or x+width>w or y+height>h:
        raise ValueError(f"ROI 超出此步骤输入的 {w} × {h} 空间范围")
    return x,y,width,height


def statistic_contract(c,a,p):
    axes=retained_axes(c.axes,p.get("scope","space_time"))
    shape=tuple(size for axis,size in zip(c.axes,c.shape) if axis in axes)
    global_=a=="global_statistics"
    semantic="statistic" if global_ or p["mode"] in {"variance","count"} else "processed_g2" if c.semantic in {"g2","processed_g2"} else "processed_label" if "label" in c.semantic else c.semantic
    meaning=f"global_{p['scope']}_{p['mode']}_of_{c.meaning}" if global_ else f"pixel_g2_spatial_{p['mode']}" if c.semantic in {"g2","processed_g2"} else f"regional_{p['mode']}_label" if "label" in c.semantic else f"regional_{p['mode']}_signal"
    return replace(c,axes=axes,shape=shape,dtype="uint64" if p["mode"]=="count" else "float64",statistics=False,
                   sequence_axis="N" if global_ else c.sequence_axis,length=1 if global_ else c.length,
                   spatial_operation=None if global_ else p["mode"],meaning=meaning,semantic=semantic,
                   origin=c.origin if not global_ or "Y" in axes and "X" in axes else (0,0))


def modifier_contract(c,a,p):
    """Contract of the selected output, before choosing main or side routing."""
    if a in {"region_reduce","point_extract","line_extract"} and not c.spatial:
        raise ValueError("此提取需要 Y/X 空间轴；上游已没有空间矩阵")
    if a=="region_reduce":
        meaning=f"pixel_g2_spatial_{p['mode']}" if c.semantic in {"g2","processed_g2"} else f"regional_{p['mode']}_label" if "label" in c.semantic else f"regional_{p['mode']}_signal"
        return replace(c,axes="".join(a for a in c.axes if a not in "YX"),shape=tuple(s for a,s in zip(c.axes,c.shape) if a not in "YX"),dtype="float64",statistics=False,spatial_operation=p["mode"],meaning=meaning,semantic="processed_g2" if c.semantic in {"g2","processed_g2"} else "processed_label" if "label" in c.semantic else c.semantic)
    if a in {"time_statistics","global_statistics"}:
        return statistic_contract(c,a,p)
    if a=="point_extract":
        roi_bounds(c,dict(x=p["x"],y=p["y"],width=1,height=1))
        return replace(c,axes="".join(a for a in c.axes if a not in "YX"),shape=tuple(s for a,s in zip(c.axes,c.shape) if a not in "YX"),statistics=False,meaning="point_"+c.meaning)
    if a=="line_extract":
        if p["mode"]=="segment":
            h,w=(c.shape[c.axes.index(a)] for a in "YX")
            if max(p["x0"],p["x1"])>w-1 or max(p["y0"],p["y1"])>h-1:
                raise ValueError("线段端点超出此步骤输入像素中心范围")
            axes=c.axes.replace("YX","R")
            shape=list(c.shape); k=c.axes.index("Y"); shape[k:k+2]=[p["samples"]]
            return replace(c,axes=axes,shape=tuple(shape),dtype="float64" if p["interpolation"]=="bilinear" else c.dtype,statistics=False,meaning="line_"+c.meaning)
        removed="Y" if p["axis"]=="X" else "X"
        k=c.axes.index(removed)
        if p["index"]>=c.shape[k]:
            raise ValueError("直线固定位置超出此步骤输入")
        return replace(c,axes=(c.axes[:k]+c.axes[k+1:]).replace(p["axis"],"R"),shape=c.shape[:k]+c.shape[k+1:],statistics=False,meaning="line_"+c.meaning)
    if a=="axis_slice":
        if p["axis"]==c.sequence_axis:
            if c.length is not None and p["index"]>=c.length:
                raise ValueError("截面样本超出此步骤输入")
            return replace(c,sequence_axis="N",length=1,statistics=False,meaning=f"{c.sequence_axis}_slice_of_{c.meaning}")
        if p["axis"] not in c.axes:
            raise ValueError("输入不包含所选截面轴")
        k=c.axes.index(p["axis"])
        if p["index"]>=c.shape[k]:
            raise ValueError("截面位置超出此步骤输入")
        return replace(c,axes=c.axes[:k]+c.axes[k+1:],shape=c.shape[:k]+c.shape[k+1:],statistics=False,meaning=f"{p['axis']}_slice_of_{c.meaning}")
    if a=="distribution":
        return replace(c,axes="B",shape=(p["bins"],),dtype="int64",semantic="distribution",statistics=False,origin=(0,0),spatial_operation=None,meaning="finite_value_histogram")
    raise ValueError("未知修饰输出")


def binding_contract(ref, providers):
    provider = providers.get(ref["node_id"])
    if provider is None or not provider["node"]["enabled"]:
        raise ValueError("参数引用必须指向已启用的前序模块；请修复顺序或引用")
    node,c=provider["node"],provider["input"]
    port=ref["port"]
    if port.startswith("parameter:"):
        key=port.partition(":")[2]
        value=node["parameters"].get(key)
        if is_reference(value):
            return binding_contract(value["ref"],providers)
        if value is None or isinstance(value,(dict,list,str,bool)) or not isinstance(value,(int,float)):
            raise ValueError("被引用参数不是确定的数值标量")
        return (),value
    if node["algorithm"]=="parameter_binding" and port=="value":
        value=node["parameters"]["value"]
        return binding_contract(value["ref"],providers) if is_reference(value) else ((),value)
    if node["algorithm"]=="global_statistics" and port in {"value","count"}:
        return statistic_contract(c,"global_statistics",node["parameters"]).shape,None
    if node["algorithm"]!="sequence_statistics" or port not in {"mean","variance","count","scale"}:
        raise ValueError("前序模块没有声明此参数端口")
    return ((2,)+tuple(1 for _ in c.shape[1:]) if c.paired else ()) if port=="scale" else c.shape,None


def correlation_layout(c, p, center=None, single=False):
    w = p["window"]
    if p["analysis"]=="map":
        lags = [p["map_lag"]]
    elif p["lag_list"] is not None:
        lags = sorted(set(p["lag_list"]))
    else:
        if p["lag_min"]>p["lag_max"]:
            raise ValueError("τ 下限不能大于上限")
        if (p["lag_max"]-p["lag_min"])//p["lag_step"]+1>4096:
            raise ValueError("τ 列表最多 4096 项")
        lags = list(range(p["lag_min"],p["lag_max"]+1,p["lag_step"]))
        if p["lag_min"]<=0<=p["lag_max"] and 0 not in lags:
            lags = sorted([*lags,0])
    if not lags or len(lags)>4096 or max(abs(q) for q in lags)>8_000_000:
        raise ValueError("延迟列表为空或超出支持范围")
    first = w//2+max(0,-min(lags))
    last = None if c.length is None else c.length-1-w//2-max(0,max(lags))
    if last is not None and last<first:
        raise ValueError("当前输入样本不足以形成完整关联窗口和全部正负延迟")
    if single or p["analysis"]=="curve":
        # center is a zero-based sample position in the direct upstream sequence.
        center = p["center"]-1 if p.get("center") is not None else center if center is not None else first if last is None else (first+last)//2
        if center<first or last is not None and center>last:
            raise ValueError(f"关联中心需在编辑后样本 {first+1}～{'?' if last is None else last+1} 内")
        centers = range(center,center+1)
    else:
        centers = None if last is None else range(first,last+1,p["stride"])
    return lags,centers,first,last


def compile_flow(nodes, source, *, center=None, single=False, strict=True, following_nodes=(), validate_bindings=True):
    canonical,_,_ = normalize_nodes(nodes,check_types=False)
    c = source if isinstance(source,Contract) else source_contract(source)
    stages,errors = [],[]
    enabled = 0
    providers = {}
    for i,n in enumerate(canonical):
        before = c
        a,p = n["algorithm"],n["parameters"]
        spec,frame = module_spec(n)
        warning = None
        sides=[]
        try:
            if n["enabled"]:
                if validate_bindings:
                    for key,value in p.items():
                        if is_reference(value):
                            shape,literal= binding_contract(value["ref"],providers)
                            if key!="scale" and shape:
                                raise ValueError("此参数只接受标量，不能引用矩阵 / 向量端口")
                            if key=="scale" and shape and shape not in ({(2,),(2,)+tuple(1 for _ in c.shape[1:])} if c.paired else {()}):
                                raise ValueError("尺度端口形状与当前输入不兼容")
                            if literal is not None:
                                parameter=next(f for f in spec["parameters"] if f["key"]==key)
                                lower=parameter.get("minimum"); upper=parameter.get("maximum")
                                if not math.isfinite(literal) or lower is not None and literal<lower or upper is not None and literal>upper:
                                    raise ValueError(f"引用值不满足参数 {parameter['label']} 的有限值 / 范围要求")
                if n["target"]!="both" and not c.paired:
                    raise ValueError("左 / 右路执行目标需要对齐双路输入")
                if a=="time_crop":
                    if enabled or c.sequence_axis!="T":
                        raise ValueError("时间裁剪只能放在流程起点，且仅保留一个")
                    if p["unit"]=="samples":
                        end = p["last"] if p["last"] is not None else c.length
                        if c.length is not None and (p["first"]>c.length or end>c.length):
                            raise ValueError("时间裁剪超出来源样本数")
                        c = replace(c,length=None if end is None else end-p["first"]+1)
                    else:
                        c = replace(c,length=None)
                elif a in {"time_sample","time_smooth","time_bin"}:
                    if c.sequence_axis!="T":
                        raise ValueError("当前结果的横轴是 τ（Q），不是时间；不能当作时间编辑")
                    if a=="time_sample":
                        size = None if c.length is None else max(0,math.ceil((c.length-p["offset"])/p["step"]))
                    elif a=="time_smooth":
                        size = None if c.length is None else c.length-p["window"]+1
                        c = replace(c,dtype="float64")
                    else:
                        size = None if c.length is None else c.length//p["size"]
                        c = replace(c,dtype="float64")
                    if size is not None and size<1:
                        raise ValueError("时间编辑后没有完整输出样本")
                    if a!="time_sample" and c.semantic in {"label","processed_label","g2","processed_g2"}:
                        c = replace(c,semantic="processed_label" if "label" in c.semantic else "processed_g2",meaning=f"{a}_{p['mode']}_of_{c.meaning}")
                    c = replace(c,length=size,statistics=False)
                elif a in {"spatial_roi","roi_crop"}:
                    x,y,w,h = roi_bounds(c,p)
                    if p.get("mode","pixels")=="pixels":
                        shape = list(c.shape)
                        shape[c.axes.index("Y")],shape[c.axes.index("X")] = h,w
                        c = replace(c,shape=tuple(shape),origin=(c.origin[0]+x,c.origin[1]+y),statistics=False)
                    else:
                        meaning=f"pixel_g2_spatial_{p['mode']}" if c.semantic in {"g2","processed_g2"} else f"regional_{p['mode']}_label" if "label" in c.semantic else f"regional_{p['mode']}_signal"
                        semantic="processed_g2" if c.semantic in {"g2","processed_g2"} else "processed_label" if "label" in c.semantic else c.semantic
                        c = replace(c,axes="".join(a for a in c.axes if a not in "YX"),shape=tuple(s for a,s in zip(c.axes,c.shape) if a not in "YX"),dtype="float64",statistics=False,spatial_operation=p["mode"],meaning=meaning,semantic=semantic)
                elif a=="alignment":
                    if c.axes!="YX" or c.semantic not in {"intensity","sensor_codes"}:
                        raise ValueError("双幅配准需要单幅灰度强度矩阵 YX")
                    from eip.modes.split_correlation.config import GeometryConfig
                    from eip.modes.split_correlation.geometry import geometry_bounds
                    if int(np.prod(c.shape))*64>n.get("execution",{}).get("memory_mb",256)*1024**2*2/3:
                        raise ValueError("配准工作数组超过此模块内存预算；请先裁剪或增加执行内存")
                    x0,y0,x1,y1 = geometry_bounds(GeometryConfig.from_dict(p),c.shape)
                    c = replace(c,axes="PYX",shape=(2,y1-y0,x1-x0),dtype="float64",statistics=False,origin=(0,0))
                elif a=="select_channel":
                    if not c.paired:
                        raise ValueError("选择幅面需要双路 P 轴")
                    c = replace(c,axes=c.axes[1:],shape=c.shape[1:],statistics=False)
                elif a=="sequence_statistics":
                    if c.sequence_axis!="T" or c.semantic not in {"intensity","sensor_codes"}:
                        raise ValueError("序列统计需要光强时间序列")
                    for port in ("mean","variance","count","scale"):
                        side=replace(c,sequence_axis="N",length=1,dtype="uint32" if port=="count" else "float64",semantic="statistic",meaning=port,statistics=False)
                        if port=="scale":
                            side=replace(side,axes="P" if c.paired else "",shape=(2,) if c.paired else ())
                        sides.append(dict(port=port,contract=side.record()))
                elif a=="fixed_normalize":
                    if "label" in c.semantic:
                        raise ValueError("标签不能通过强度尺度归一")
                    c=replace(c,dtype="float64",statistics=False,semantic="processed_g2" if c.semantic in {"g2","processed_g2"} else c.semantic,meaning="scaled_"+c.meaning if c.semantic in {"g2","processed_g2"} else c.meaning)
                elif a=="parameter_binding":
                    sides.append(dict(port="value",contract=replace(c,axes="",shape=(),sequence_axis="N",length=1,dtype="float64",semantic="parameter",meaning=p["name"],statistics=False).record()))
                elif a=="data_tap":
                    if p["export_format"]=="csv" and (c.spatial or int(np.prod(c.shape))>4096):
                        raise ValueError("CSV 截流需要标量 / 向量且每样本不超过4096项；矩阵请用数据集 / NPY / TIFF")
                    if p["export_format"]=="tiff" and not c.spatial:
                        raise ValueError("TIFF 截流需要 Y/X 图像空间轴")
                    sides.append(dict(port="data",contract=c.record()))
                elif a in MODIFIERS:
                    side=modifier_contract(c,a,p)
                    if p["output"]=="side":
                        sides.append(dict(port="data",contract=side.record()))
                    else:
                        c=side
                elif a in {"grayscale","color_convert"}:
                    if not c.spatial:
                        raise ValueError("颜色转换需要空间图像")
                    if a=="grayscale":
                        if "C" not in c.axes:
                            raise ValueError("灰度转换需要明确 RGB / BGR 颜色轴")
                        c=replace(c,axes=c.axes.replace("C",""),shape=c.shape[:-1],dtype="float64",statistics=False,color_order="gray")
                    else:
                        if "C" in c.axes and c.shape[-1]!=3:
                            raise ValueError("颜色排列当前需要 RGB 三分量；RGBA 请先灰度转换")
                        c=replace(c,axes=c.axes if "C" in c.axes else c.axes+"C",shape=c.shape if "C" in c.axes else (*c.shape,3),color_order=p["mode"].upper(),statistics=False)
                elif a=="statistics":
                    if c.sequence_axis!="T" or c.semantic not in {"intensity","sensor_codes"}:
                        raise ValueError("统计模块需要光强时间序列，不能接标签或关联结果")
                    c = replace(c,statistics=True,dtype="float64")
                elif a in CORRELATIONS:
                    if not c.paired or "C" in c.axes or c.semantic not in {"intensity","sensor_codes"} or c.sequence_axis!="T":
                        raise ValueError("标准 g² 需要对齐双路强度矩阵 PYX 或双路标量 P")
                    if p.get("roi_estimator")!="pixel_mean":
                        raise ValueError("区域总强度请用关联之前的 ROI 求和模块表达；关联不再隐含空间归约")
                    downstream_sequence = needs_full_sequence(canonical[i+1:])
                    lags,centers,_,_ = correlation_layout(c,p,center,single and not downstream_sequence)
                    c = replace(c,axes=c.axes[1:],shape=c.shape[1:],length=len(lags) if p["analysis"]=="curve" else None if centers is None else len(centers),sequence_axis="Q" if p["analysis"]=="curve" else "T",semantic="g2",dtype="float64",statistics=False,meaning=f"regional_{c.spatial_operation}_signal_g2" if c.spatial_operation else "pixel_g2")
                    if p["analysis"]=="curve" and c.spatial:
                        warning="当前 τ 扫描仍输出逐像素矩阵；需要单条曲线时，请在关联前归约区域信号，或在关联后对像素 g² 添加空间归约。"
                else:
                    if a in {"gaussian","box_blur","spatial_smooth"} or n.get("manifest"):
                        if not c.spatial:
                            raise ValueError("此空间算法需要 Y/X 轴；不能接收区域归约后的时间标量 / 向量")
                        if n.get("manifest") and c.axes.removeprefix("P") not in {"YX","YXC"}:
                            raise ValueError("此插件只声明了逐帧 YX / YXC 契约")
                    if a=="calibration" and c.semantic not in {"intensity","sensor_codes"}:
                        raise ValueError("校准模块需要光强输入")
                    if a=="calibration":
                        if p.get("lut_path") and c.dtype!="dynamic" and np.dtype(c.dtype).kind not in "uib":
                            raise ValueError("LUT 需要整数码值；请把 LUT 校准移到插值、归一化或时间平均之前")
                        if not c.paired and p["black_left"]!=p["black_right"]:
                            future=[*canonical[i+1:],*following_nodes]
                            following=next((m for m in future if m["enabled"] and m["algorithm"]=="alignment"),None)
                            if c.axes!="YX" or following is None or any(m["enabled"] and m["algorithm"] in {"spatial_roi","roi_crop"} for m in future[:future.index(following)]):
                                raise ValueError("不同左右暗场需要后续明确配准中线；请调整校准位置，或配准后再校准")
                    if a=="threshold" or n.get("manifest") and spec.get("output_kind")=="label":
                        if n["target"]!="both" and "C" in c.axes:
                            raise ValueError("双路彩色阈值会改变通道数，请同时处理两路")
                        c = replace(c,axes=c.axes.replace("C",""),shape=tuple(s for axis,s in zip(c.axes,c.shape) if axis!="C"),semantic="label",meaning="label",dtype="dynamic" if n.get("manifest") else "uint8",statistics=False)
                    else:
                        c = replace(c,dtype="dynamic" if n.get("manifest") else "float64" if a in {"affine","curve","calibration","gaussian"} else c.dtype,statistics=False,semantic="processed_g2" if c.semantic in {"g2","processed_g2"} else "processed_label" if c.semantic in {"label","processed_label"} else c.semantic,meaning="processed_"+c.meaning if c.semantic in {"g2","processed_g2","label","processed_label"} else c.meaning)
                if before.statistics and not c.statistics and a not in CORRELATIONS:
                    warning = "上游统计随数据编辑失效；后续关联直接使用当前强度，需背景统计时请在此后重新添加统计模块。"
                precision = n.get("execution",{}).get("precision","native")
                if c.semantic=="label" and precision!="native":
                    raise ValueError("标签模块使用整数标签类型，不能通过执行设置转为浮点")
                if precision!="native" and a not in {"data_tap","sequence_statistics","parameter_binding"} and not (a in MODIFIERS and p["output"]=="side"):
                    c = replace(c,dtype=precision)
                enabled += 1
            providers[n["id"]]=dict(node=n,input=before,output=c)
            stages.append(dict(id=n["id"],algorithm=a,enabled=n["enabled"],input=before.record(),output=c.record(),side_outputs=sides,warning=warning,error=None))
        except (ValueError,TypeError,IndexError) as error:
            message = f"第 {i+1} 个模块「{spec['name']}」：{error}"
            if strict:
                raise ValueError(message) from error
            errors.append(message)
            stages.append(dict(id=n["id"],algorithm=a,enabled=n["enabled"],input=before.record(),output=None,error=message,warning=None))
            break
    return dict(contract_version=CONTRACT_VERSION,nodes=list(canonical),stages=stages,output=c.record(),errors=errors)
