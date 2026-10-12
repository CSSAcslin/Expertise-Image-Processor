"""Scientific documentation travels with the executable module catalogue."""
from copy import deepcopy


ENTRIES = {
    "intensity": ("intermediate", "sliders", "定类型增益 / 偏置", "B=cast_clip_round(gA+b)", "整数先舍入再限幅；输出保持原类型。"),
    "affine": ("intermediate", "sliders", "浮点增益 / 偏置", "B=gA+b", "float64 输出，不量化、不限幅；溢出为 NaN。"),
    "box_blur": ("intermediate", "blur", "空间均值平滑", "B[y,x]=sum(A[y+dy,x+dx])/(2r+1)^2", "空间反射边界，要求输入有限；保持原类型。"),
    "gaussian": ("intermediate", "blur", "空间高斯平滑", "K[k]∝exp(-k²/(2σ²)); B=(K*(A·valid))/(K*valid)", "边界复制；原无效位置仍无效。"),
    "curve": ("intermediate", "curve", "固定尺度强度映射", "x=A/M; B=f(x)", "整段固定尺度；非法定义域及非有限值为 NaN。identity 返回原值。"),
    "threshold": ("algorithm", "threshold", "阈值分割", "B=uint8(A≥θ)", "输出为标签；不能将标签当光强进入关联。"),
    "calibration": ("algorithm", "calibrate", "传感器强度标定", "I=F(A)-black", "固定 LUT / 曲线；饱和及非有限输入为 NaN。显示 LUT 不参与。"),
    "alignment": ("algorithm", "align", "双幅配准", "L=crop(A,left); R=sample(crop(A,right),mirror(x+dx,y+dy))", "只保留公共有效区，插值方式明确；输入为灰度强度。"),
    "correlation": ("algorithm", "correlation", "二阶互关联 g²（兼容）", "g²_q(c)=⟨L_iR_(i+q)⟩/(⟨L_i⟩⟨R_(i+q)⟩)", "同一有效配对集合；完整 W 窗口；正负延迟独立，不能假设对称。"),
    "correlation_map": ("algorithm", "heatmap", "固定延迟 g² 系数图", "g²_q(c)=⟨L_iR_(i+q)⟩/(⟨L_i⟩⟨R_(i+q)⟩)", "固定 q，沿中心样本 T 输出；W、q、步长按直接上游样本计数。"),
    "correlation_curve": ("algorithm", "chart", "延迟扫描 g²", "g²_q(c)=⟨L_iR_(i+q)⟩/(⟨L_i⟩⟨R_(i+q)⟩)", "固定中心 c，沿有符号 Q 输出；Q 不是时间。"),
    "time_crop": ("intermediate", "range", "时间范围", "B=A[first-1:last] 或 start≤t<end", "只放流程起点；时间使用真实样本时间。"),
    "spatial_roi": ("intermediate", "crop", "空间 ROI / 归约（兼容）", "B=A[y:y+h,x:x+w]", "历史复合节点；新流程分别使用 ROI 裁剪与区域归约。"),
    "roi_crop": ("intermediate", "crop", "ROI 裁剪", "B=A[y:y+h,x:x+w]", "相对此模块直接输入坐标；保留空间轴和来源原点。"),
    "region_reduce": ("modifier", "reduce", "区域归约", "mean=Σvalid(A)/n; sum=Σvalid(A); quantile=Q_q(A)", "归约 Y/X；全部无效为 NaN；双路共同 / 独立支持显式选择。"),
    "statistics": ("modifier", "statistics", "统计与归一（兼容）", "μ=ΣI/n; variance=M2/n; B=I/s", "历史节点包含主流归一与 float64 转换；不作为新模块展示。"),
    "sequence_statistics": ("modifier", "statistics", "序列统计", "μ=Σvalid(I)/n; variance=M2/n; s=√(ΣM2/Σn)", "主流原样通过；mean / variance / count / scale 为独立侧输出。"),
    "fixed_normalize": ("intermediate", "normalize", "固定尺度归一", "B=A/s", "显式固定尺度，必须有限且大于 0；输出 float64。"),
    "select_channel": ("modifier", "channels", "双路选幅", "B=L 或 R", "仅移除 P 轴，主流明确改变；不能接单路。"),
    "time_sample": ("intermediate", "sample", "间隔采样", "B[j]=A[offset+j·step]", "不插值、不合帧；保留实际时间；只接受 T。"),
    "time_smooth": ("intermediate", "window", "时间滑窗", "B[j]=mean_valid(A[j:j+W]) 或 quantile", "完整滑窗、不补帧；输出时间为中心样本；只接受 T。"),
    "time_bin": ("intermediate", "bin", "时间合帧", "B[j]=mean_valid 或 sum_valid(A[jB:(j+1)B])", "不重叠分组，丢弃不满尾组；不伪造采集帧；只接受 T。"),
    "grayscale": ("intermediate", "palette", "灰度转换", "Y=.2126R+.7152G+.0722B 或 (R+G+B)/3", "明确转换 RGB；忽略 alpha，float64 输出；灰度输入拒绝重复转换。"),
    "color_convert": ("intermediate", "palette", "颜色排列 / 扩展", "RGB←repeat(gray,3); BGR=RGB[...,::-1]", "显式声明输出颜色顺序；不暗中改变强度尺度。"),
    "point_extract": ("modifier", "point", "点信号提取", "B=A[...,y,x,...]", "去除 Y/X，保留 T 或 Q 及显式通道；坐标为直接上游。"),
    "line_extract": ("modifier", "line", "直线截面提取", "p_j=p₀+j(p₁−p₀)/(N−1); B_j=sample(A,p_j)", "横纵便捷线或任意端点线段；R 为像素距离，最近邻或明确双线性。"),
    "axis_slice": ("modifier", "slice", "按轴截面", "B=take(A,index,axis)", "按明确轴与从 0 起位置提取；T/Q 单样本截面改为 N 记录轴。"),
    "distribution": ("modifier", "histogram", "分布统计", "H[b]=count(e_b≤finite(A)<e_(b+1))", "上下界可独立手动设置或自动识别上游完整范围的有限值极值；整段固定等宽箱，末箱含右边界。常量数据自动边界向外扩展；无有限值时自动识别报错。"),
    "data_tap": ("modifier", "export", "数据截流导出", "main=A; captured=A", "获取直接上游真实数据；主流透传，不更改轴、dtype 或来源。"),
    "parameter_binding": ("modifier", "variable", "参数传递", "main=A; value=v", "只发布数值标量；允许常量或已启用前序模块的明确端口，禁止循环与任意代码。"),
}

LATEX = {
    "intensity":r"B=\operatorname{cast}_{d}\!\left(\operatorname{clip}_{d}(\operatorname{round}(gA+b))\right)",
    "affine":r"B=gA+b", "box_blur":r"B_{yx}=\frac{\sum_{i=-r}^{r}\sum_{j=-r}^{r} A_{y+i,x+j}}{(2r+1)^2}",
    "gaussian":r"K_k\propto\exp\!\left(-\frac{k^2}{2\sigma^2}\right),\quad B=\frac{K*(A\,V)}{K*V}",
    "curve":r"x=\frac{A}{M},\quad B\in\{A,x,\sqrt{x},x^2,x^{\gamma},\frac{\ln(1+9x)}{\ln 10}\}",
    "threshold":r"B=\mathbf{1}_{A\geq\theta}", "calibration":r"I=F(A)-b_{\mathrm{side}}",
    "alignment":r"L=\operatorname{crop}(A,L_0),\quad R(x,y)=\operatorname{sample}(\operatorname{crop}(A,R_0),\operatorname{mirror}(x+\Delta x,y+\Delta y))",
    "correlation":r"g_q^{(2)}(c)=\frac{\langle L_iR_{i+q}\rangle_{i\in V_c}}{\langle L_i\rangle_{i\in V_c}\langle R_{i+q}\rangle_{i\in V_c}}",
    "time_crop":r"B=A_{[f-1:l)},\quad\text{or}\quad t_0\leq t<t_1",
    "roi_crop":r"B=A_{y:y+h,\,x:x+w}",
    "region_reduce":r"\mu=\frac{\sum_{j\in V} A_j}{|V|},\quad S=\sum_{j\in V}A_j,\quad u=(n-1)q,\quad Q_q=(1-\{u\})A_{(\lfloor u\rfloor)}+\{u\}A_{(\lceil u\rceil)}",
    "sequence_statistics":r"\mu=\frac{\sum_{i\in V}I_i}{n},\quad M_2=\sum_{i\in V}(I_i-\mu)^2,\quad \sigma^2=\frac{M_2}{n},\quad s=\sqrt{\frac{\sum M_2}{\sum n}}",
    "fixed_normalize":r"B=\frac{A}{s},\quad s>0", "select_channel":r"B=A_{P=p}",
    "time_sample":r"B_j=A_{o+j\,s}", "time_smooth":r"B_j=\operatorname{mean}_{\mathrm{finite}}(A_{j:j+W})\quad\text{or}\quad Q_q(A_{j:j+W})",
    "time_bin":r"B_j=\operatorname{mean}_{\mathrm{finite}}(A_{jB:(j+1)B})\quad\text{or}\quad\operatorname{sum}_{\mathrm{finite}}(A_{jB:(j+1)B})",
    "grayscale":r"Y=0.2126R+0.7152G+0.0722B\quad\text{or}\quad\frac{R+G+B}{3}",
    "color_convert":r"\mathrm{RGB}=(A,A,A)\quad\text{or}\quad\mathrm{BGR}=\mathrm{RGB}_{[2,1,0]}",
    "point_extract":r"B=A_{\ldots,y,x,\ldots}", "line_extract":r"p_j=p_0+\frac{j}{N-1}(p_1-p_0),\quad B_j=\operatorname{sample}(A,p_j),\quad R_j=\frac{j}{N-1}\|p_1-p_0\|_2",
    "axis_slice":r"B=\operatorname{take}(A,k,a)", "distribution":r"H_b=\sum_{j\in V}\mathbf{1}_{e_b\leq A_j<e_{b+1}}",
    "data_tap":r"B=A,\quad O_{\mathrm{captured}}=A", "parameter_binding":r"B=A,\quad O_{\mathrm{value}}=v",
}
LATEX["correlation_map"]=LATEX["correlation_curve"]=LATEX["correlation"]
LATEX["spatial_roi"]=LATEX["roi_crop"]
LATEX["statistics"]=LATEX["sequence_statistics"]+r",\quad B=I/s"

VALIDITY = {
    "box_blur":"输入必须全部有限；遇 NaN / Inf 直接报错。",
    "threshold":"NaN 比较为 false、输出0；±Inf按阈值比较；输出 uint8 标签。",
    "data_tap":"原始值、dtype、NaN / Inf 原样保留。",
    "parameter_binding":"变量值必须为有限数值标量；主流完全透传。",
    "sequence_statistics":"逐元素统计有限输入；计数0时均值/方差NaN；固定RMS非有限或≤1e−12时尺度1；主流完全透传。",
    "distribution":"只计有限且落在固定范围内的值；空集合输出全零计数。",
    "roi_crop":"仅裁剪，保留原始值及其有效性。", "point_extract":"仅提取，保留原始值及其有效性。", "axis_slice":"仅提取，保留原始值及其有效性。",
    "line_extract":"最近邻保留采样像素原值；双线性若非零权重支持含非有限值则该输出NaN。",
    "select_channel":"仅选择幅面，保留原始值及其有效性。", "time_sample":"仅选择样本，保留原始值及其有效性。", "time_crop":"仅选择范围，保留原始值及其有效性。",
    "correlation":"左右同一有限配对集合；有效数不足或任一路均值不大于 min_mean 时NaN。",
    "fixed_normalize":"尺度必须有限且>0；源数据非有限值随除法传播。",
    "intensity":"整数输出按 NumPy rint（半整数到偶数）舍入并限幅；浮点保持原dtype。",
    "affine":"不量化、不限幅；计算后非有限结果置NaN。",
    "curve":"负数开方/非整数幂或对数非法定义域置NaN；其余非有限结果置NaN。",
    "gaussian":"有限权重归一，原无效位置仍NaN；零有效权重为NaN。",
    "calibration":"饱和或非有限输入置NaN；LUT须单调非减且覆盖有效整数码值。",
    "alignment":"最近邻取原采样值；双线性按明确支持插值；只保留几何公共有效区。",
    "statistics":"有限输入参与Welford；无样本均值/方差NaN；退化尺度取1；主流仍执行I/s。",
    "spatial_roi":"pixels模式只裁剪；归约模式仅有限值参与，全无效NaN；新流程拆为裁剪与归约。",
    "grayscale":"任一参与颜色分量非有限时结果非有限；忽略alpha。", "color_convert":"重排 / 复制，不改变原值有效性。",
}
VALIDITY["correlation_map"]=VALIDITY["correlation_curve"]=VALIDITY["correlation"]
SYMBOLS={"gain":"g","offset":"o / b","sigma":"σ","radius":"r","threshold":"θ","input_max":"M","gamma":"γ","quantile":"q","window":"W","map_lag":"q / τ","stride":"S","scale":"s","bins":"N","minimum":"e₀","maximum":"eₙ","samples":"N","step":"s","size":"B","center":"c"}

IO = {
    "calibration":("强度 / 传感器码值 T(P)YX(C) 或 T(P)(C)","同轴同形状 float64 强度；不接受标签 / g²"),
    "alignment":("灰度强度 TYX","对齐双路 TPYX，float64，公共有效区"),
    "correlation":("T(P=2)YX 或 TP 的双路强度；不接受C/标签/Q","固定τ：T′YX 或 T′；固定中心扫描：QYX 或 Q，标准g²"),
    "roi_crop":("包含 Y/X 的 T/Q(P)YX(C)","保留像素的 T/Q(P)hw(C)，原dtype与坐标来源"),
    "region_reduce":("包含 Y/X 的矩阵","归约空间后的 T/Q(P)(C)，float64；或同类型侧输出且主流透传"),
    "sequence_statistics":("光强时间序列 T(P)YX(C) 或 T(P)(C)","主流保持轴/形状/dtype；N=1统计记录 mean/variance/count/scale"),
    "fixed_normalize":("数值强度或系数；不接受标签","同轴同形状float64；系数标为派生g²"),
    "point_extract":("含Y/X的T/Q空间矩阵","去除Y/X的T/Q(P)(C)点信号；可旁路"),
    "line_extract":("含Y/X的T/Q空间矩阵","T/Q(P)R(C)线信号，R为像素距离；可旁路"),
    "axis_slice":("包含指定轴的输入","去除被选轴；若选择T/Q，输出N=1静态记录；可旁路"),
    "distribution":("任意有限或含无效值的当前数值样本","T/Q/N×B固定箱计数int64及箱边界；可旁路"),
    "data_tap":("任意已声明轴、dtype和语义的直接上游","主流透传；完整真实上游数据侧输出"),
    "parameter_binding":("任意主流 + 常量/前序数值参数端口","主流透传；N=1命名标量value端口"),
    "grayscale":("空间RGB/RGBA或明确BGR颜色数据","去除C轴的float64灰度矩阵"),
    "color_convert":("灰度或RGB/BGR三分量空间图像","同尺度RGB/BGR，显式color_order；不接受RGBA重排"),
    "threshold":("矩阵/标量/颜色向量数值输入","同空间尺寸uint8 0/1标签；颜色输入去除C"),
}
for _key in ("correlation_map","correlation_curve"):
    IO[_key]=IO["correlation"]
for _key in ("box_blur","gaussian"):
    IO[_key]=("含Y/X的空间矩阵；双路分别处理","同形状空间平滑输出；不能接标量或线信号")
for _key in ("time_crop","time_sample","time_smooth","time_bin"):
    IO[_key]=("当前时间T×…；不接受Q延迟轴或N静态记录","编辑后的T′×…，保留真实时间与贡献来源")


def enrich(spec):
    if spec["id"] not in ENTRIES:
        document=deepcopy(spec.get("documentation") or {})
        formula=document.get("formula") or {}
        if not isinstance(formula,dict):
            formula={"expression":str(formula)}
        spec.update(category=spec.get("category","algorithm"),icon=spec.get("icon","plugin"),legacy=False,effect="replace",documentation=document)
        spec["formula"]={"latex":"","expression":"未提供公式","symbols":[],"domain":"以已声明插件契约为准","validity":"以插件实现及输出校验为准","boundary":"未提供边界说明",**formula}
        return spec
    category, icon, name, expression, boundary = ENTRIES.get(spec["id"], ("algorithm", "plugin", spec["name"], "用户模块执行契约", "以插件声明及实际输出校验为准。"))
    spec.update(category=category, icon=icon, name=name, legacy=spec["id"] in {"spatial_roi", "statistics", "correlation"})
    symbols = [{"symbol":"A / I", "meaning":"本模块直接上游数值", "unit":"上游单位"}, {"symbol":"B", "meaning":"本模块输出", "unit":"由输出契约确定"}]
    if spec["id"].startswith("correlation"):
        symbols = [
            {"symbol":"Lᵢ / Rᵢ", "meaning":"直接上游左 / 右路在第 i 个样本的强度；空间矩阵逐像素计算，归约后的区域信号按该信号计算", "unit":"上游强度"},
            {"symbol":"g²", "meaning":"标准二阶互关联；分子与分母使用同一有效配对集合", "unit":"无量纲"},
            {"symbol":"W", "meaning":"以 c 为中心的完整平均窗口，奇数", "unit":"上游样本"},
            {"symbol":"q / τ", "meaning":"右路索引 = 左路索引 + q；可为负、零或正值", "unit":"上游样本"},
            {"symbol":"c", "meaning":"窗口中心，按编辑后的直接上游计数", "unit":"上游样本"},
            {"symbol":"V_c", "meaning":"窗口中 Lᵢ 与 Rᵢ₊q 同时有限的配对集合；每个像素可不同", "unit":"样本集合"},
            {"symbol":"⟨·⟩", "meaning":"在 V_c 内的算术平均；非对整段或不同集合分别平均", "unit":"由被平均量确定"},
        ]
    if spec["id"] in {"sequence_statistics","statistics"}:
        symbols += [
            {"symbol":"μ", "meaning":"每个位置沿完整直接上游序列的有限值均值", "unit":"上游强度"},
            {"symbol":"M₂", "meaning":"有限值对均值的离差平方和", "unit":"强度平方"},
            {"symbol":"σ²", "meaning":"总体方差 M₂ / n，非样本无偏方差", "unit":"强度平方"},
            {"symbol":"s", "meaning":"固定涨落 RMS：各幅面内所有位置的 M₂ 总和除以有效计数总和后开方", "unit":"上游强度"},
        ]
    if spec["id"]=="distribution":
        symbols += [{"symbol":"H_b", "meaning":"落入第 b 个固定箱的有限值数量；末箱含右边界", "unit":"计数"},
                    {"symbol":"e_b", "meaning":"由手动值或上游完整范围自动识别的有限值极值及箱数确定，整段共用固定边界", "unit":"上游单位"},
                    {"symbol":"V", "meaning":"当前样本中的有限值集合，超范围值不计入任何箱", "unit":"值集合"}]
    symbols += [{"symbol":SYMBOLS.get(p["key"],p["key"]),"meaning":p["label"],"unit":p.get("unit","")} for p in spec.get("parameters",[]) if p["kind"] in {"int","float"}]
    if spec["id"] in {"region_reduce","sequence_statistics","statistics","gaussian"}:
        symbols += [{"symbol":"V / valid","meaning":"本公式声明的有限值支持集合","unit":"无"},{"symbol":"n","meaning":"有效支持数量","unit":"样本或像素"}]
    # Formula definitions take precedence over less specific parameter labels.
    symbols = list({entry["symbol"]:entry for entry in reversed(symbols)}.values())[::-1]
    spec["formula"] = dict(latex=LATEX[spec["id"]], expression=expression, symbols=symbols, domain=spec.get("description", ""), validity=VALIDITY.get(spec["id"],"有限值参与统计 / 平滑；全部无效输出 NaN，具体集合和边界见公式及算法说明。"), boundary=boundary)
    input_,output_=IO.get(spec["id"],("数值矩阵 / 双路 / 标量 / 向量；64位整数不作隐式浮点计算","同轴同形状；实际dtype和语义见公式边界及实例契约"))
    spec["documentation"] = dict(summary=spec.get("description", ""), formula=deepcopy(spec["formula"]), input=input_, output=output_)
    spec["effect"] = "pass" if spec["id"] in {"data_tap", "sequence_statistics", "parameter_binding"} else "selectable" if spec["id"] in {"region_reduce","point_extract","line_extract","axis_slice","distribution"} else "replace"
    return spec
