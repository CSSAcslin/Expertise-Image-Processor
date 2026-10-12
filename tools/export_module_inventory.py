"""Export the current module catalogue, documentation and scientific contracts."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from eip import __version__
from eip.module_registry import catalog


DETAILS={
    'intensity':('B = cast_clip_round(A × gain + offset)','矩阵、双路、标量 / 向量 → 同形状、同 dtype。','整数先 round 再裁剪到原 dtype 范围；bool 以非零为真。内置数值算法拒绝 64 位整数以避免隐式精度损失。'),
    'affine':('B = gain × A + offset','矩阵、双路、标量 / 彩色向量 → 同形状 float64。','不量化，不截断；非有限结果置 NaN。标量保持真实的零维形状，不伪装成图像。'),
    'box_blur':('B[y,x] = Σ A[y+dy,x+dx] / (2r+1)²','YX / YXC；双路分别处理 → 同形状、同 dtype。','二维空间反射边界，积分图求均值；要求全部输入有限。只作用于空间轴；ROI 前后的排列决定真实滤波上下文。'),
    'threshold':('B = uint8(A ≥ threshold)；彩色 A = mean(R,G,B)','矩阵 / 标量 → 同形状；含 C 的彩色输入去除 C → 0/1 uint8 标签。','NaN 比较得到 0；标签及其派生值不能隐式作为光强进入统计或关联。'),
    'gaussian':('K[k] ∝ exp(−k²/(2σ²))；B = (Kx*Ky*(A·valid))/(Kx*Ky*valid)','YX / YXC；双路分别处理 → 同形状 float64。','空间轴 0/1，边界复制；原无效位置仍为 NaN，不把时间轴作为空间。'),
    'curve':('x=A/input_max；identity:A，linear:x，sqrt:√x，square:x²，gamma:x^γ，log:ln(1+9x)/ln(10)','矩阵、双路、标量 / 向量、系数 → 同形状 float64。','固定输入尺度；负数非法定义域或非有限结果为 NaN；identity 不除以 input_max。'),
    'calibration':('I = F(A) − black_side；F 为固定曲线或 LUT[A]','光强矩阵 / 双路 / 标量 → 同形状 float64。','饱和条件 A ≥ saturation_code 或输入非有限时置 NaN。LUT 仅接受覆盖有效整数码值的单调非减有限数组（2–65536 项，≤8 MiB）。单幅不同左右暗场需后续明确配准中线；有空间编辑时请校准后再编辑，或配准后按两路校准。LUT 与非 identity 曲线不能同时启用，内容保存快照。'),
    'alignment':('L = crop(left)；R[x,y] = sample(crop(right), mirror(x+dx,y+dy))','单幅灰度 TYX → 对齐双路 TPYX，P=2。','依据中线、镜像和采样坐标取公共有效区；最近邻 floor(v+0.5)，或显式双线性；不支持彩色或标量。自动建议：去均值、Hann 窗、归一互功率谱 FFT 相位相关、峰值二次细化；建议须检查后接受。坐标相对此模块的直接输入。'),
    'select_channel':('B = L 或 R','TPYX → TYX；TP → T；保留 C 时 TPC → TC。','只移除双路 P 轴；输入为单路时拒绝。'),
    'statistics':('μ=Σ_valid I/n；M2 由 Welford 累积；variance=M2/n；s=sqrt(Σ_space M2/Σ_space n)；输出 I/s','光强时间序列 T(P)YX(C) 或 T(P)(C) → 同形状浮点序列，并保存均值、方差、计数、尺度。','扫描本模块直接输入的全部样本，之前的时间裁剪 / 抽样 / 合帧和 ROI 都参与。none 时 s=1；fixed_rms 且 RMS≤1e−12 时 s=1；全无效均值为 NaN。标准 g² 不将去均值后的残差当作分母；后续编辑会使统计元数据失效。'),
    'correlation':('g²_q(c)=mean_valid(L[i]R[i+q])/(mean_valid(L[i])mean_valid(R[i+q]))；i=c−W//2…c+W//2','TPYX → 固定 τ 的 T′YX 或固定中心的 QYX；TP → T′ 或 Q。','W、τ、输出步长均按直接上游的编辑后样本计数。负 q 反向配对，不能假设对称。左右采用同一有效时间样本集合，float64 累加；均值须大于 min_mean、计数至少 min_samples，否则 NaN。窗口仅保留完整边界；保存逐对真实时差的均值 / 最小 / 最大。ROI 不再藏在关联中：之前归约得到区域信号 g²，之后归约得到像素 g² 的空间统计。roi_estimator 仅保留旧记录兼容，新流程固定 pixel_mean（实际空间操作由 ROI 模块决定）。'),
    'time_crop':('样本：B=A[first−1:last]；时间：保留 start≤t<end 的实际样本','T×… → Tc×…，不改变各样本形状。','只在起点设置一次；样本尾号包含、真实时间终点不包含。普通视频按实际解码时间建立临时索引，不以平均帧率推算。'),
    'spatial_roi':('区域像素 B=A[y:y+h,x:x+w]；mean=Σ_valid/n；sum=Σ_valid；quantile 在 u=(n−1)q 两侧线性插值','T(P)YX(C) → T(P)hw(C)，或归约为 T(P)(C)；QYX → Qhw 或 Q。','空间坐标相对此步直接输入；范围与原点随输出传递。全无效为 NaN。双路归约默认取当前样本共同有效的像素，亦可明确选择各路独立集合；掩码可能随样本改变，不能暗称固定公共支持。标量后再接空间操作会报错。'),
    'time_sample':('B[j]=A[offset+j·step]','T×… → ceil((T−offset)/step)×…。','保留真实时间和来源；输出不能为空；Q 延迟轴不能当作时间轴抽样。'),
    'time_smooth':('B[j]=mean_valid(A[j:j+W])；或逐元素 quantile','T×… → (T−W+1)×… float64。','W 为正奇数；完整窗口、不补帧；时间对应中心样本，贡献范围保留；空间 / 通道轴不归约，分位值窗口需要满足内存预算。放在 g² 后只编辑结果，不改变原关联 W/τ。'),
    'time_bin':('B[j]=mean_valid 或 sum_valid(A[jB:(j+1)B])','T×… → floor(T/B)×… float64。','不满尾组舍弃；组内真实时间取均值，任一未知则未知；合成样本的采集帧号为空，保留贡献范围。全无效为 NaN。'),
}


def literal(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('|','\\|')


def main():
    specs=catalog()
    current={key:spec for key,spec in specs.items() if not spec.get('legacy')}
    categories={'intermediate':'中间模块 · 数据编辑','algorithm':'算法模块 · 科学计算','modifier':'修饰模块 · 提取与传递'}
    counts={label:sum(spec.get('category')==key for spec in current.values()) for key,label in categories.items()}
    lines=['# 当前已实现模块清单（代码核对）','',f'目录取自版本 `{__version__}`，新流程契约版本为 `3`。模块库提供 **{len(current)} 个模块**，另保留 {len(specs)-len(current)} 个旧记录兼容定义；插件另行注册。分类、公式与参数取自统一代码目录。','',
           '完整形状与科学规则见 [MODULE_CONTRACTS_0_6_0.md](MODULE_CONTRACTS_0_6_0.md)；重构前行为保留在 [MODULE_INVENTORY_0_5_2.md](MODULE_INVENTORY_0_5_2.md)。旧结果按旧记录读取，不重新解释。','',
           '## 轴、参数与执行归属','',
           '- T 为当前时间样本，Q 为有符号延迟，P 为左右双路，Y/X 为空间，C 为明确的 RGB/RGBA 分量；R 为剖面位置。分布使用单独的箱轴。标量与向量保留真实维度。','- 实例：algorithm、id、version、parameters、enabled、target、execution、插件路径与内容哈希。每个实例独立；停用原样通过。','- execution：backend=cpu/auto/gpu（GPU 当前仅关联）；memory_mb=16…4096，默认256；precision=native/float32/float64，默认算法类型。每步工作数组按自身预算校验，重复统计保留数组与缓存合计受流程预算约束。标签禁止浮点精度转换。','- 运行范围只在起点 time_crop；ROI 由 roi_crop 裁剪，区域运算由 region_reduce 完成。输出目录、名称、试算与运行按钮属于工作台。','- 每步输入输出轴、形状、样本数、数据含义、坐标原点及上游指纹随记录保存；未知时间 / 采集帧保持未知。','- 通用与分幅共用 SequenceEngine 与结果类型，仅预置不同流程。编辑中的不完整连接不影响原始数据预览。','- 默认配准按当前来源宽高初始化两半；关联按来源帧数给初始 W/τ。之后的裁剪会改变样本数，需据连接提示调整关联参数，不会静默改小。','']
    lines += ['## 三类模块','', '| 类别 | 数量 |','|---|---|',*[f'| {label} | {count} |' for label,count in counts.items()], '']
    for key,spec in specs.items():
        documentation=spec.get('documentation',{})
        definition=spec.get('formula',{})
        if key in DETAILS:
            formula,shape,limits=DETAILS[key]
        else:
            formula=definition.get('expression')
            shape=f"{documentation.get('input','以实例编译后的输入契约为准')} → {documentation.get('output','以实例编译后的输出契约为准')}"
            limits=definition.get('boundary','')
        if not formula or not limits:
            raise RuntimeError(f'{key} 缺少公式或边界说明')
        lines += [f'## {spec["name"]} `{key}`','',f'- 类别：{categories.get(spec.get("category"),"旧模块")}；{"仅用于旧记录兼容；新界面不提供。" if spec.get("legacy") else "可从模块库添加。"}',f'- 版本：`{spec["version"]}`；执行范围：`{spec["execution_scope"]}`。',f'- 公式 / 算法：{formula}',f'- 实际输入输出：{shape}',f'- 边界：{limits}','', '| 参数 | 类型 | 静态默认值 | 范围 / 选项 |','|---|---|---|---|']
        for p in spec['parameters']:
            bounds=[]
            if p.get('minimum') is not None:bounds.append('≥ '+str(p['minimum']))
            if p.get('maximum') is not None:bounds.append('≤ '+str(p['maximum']))
            if p.get('choices'):bounds.append(' / '.join(p['choices']))
            if p.get('nullable'):bounds.append('可留空')
            lines.append(f'| {p["label"]} `{p["key"]}` | {p["kind"]} | `{literal(p["default"])}` | {"；".join(bounds) or "需满足算法有限值与结构校验"} |')
        lines.append('')
    lines += ['## 自定义算法与机器学习边界','',
              '已注册插件沿用逐帧 prepare/process/close 契约，输入输出须为明确的 YX / YXC 图像，空间尺寸保持不变；标签输出可去除颜色轴。实际输出 dtype 在运行确定并保存，不强制转回来源类型。未声明的标量、时间窗、形状变换明确拒绝。机器学习可通过此契约接入，尚无内置训练 / ONNX 推理模块。','',
              '## 执行与存储位置','',
              '- module_registry.py：公共目录、独立实例参数及执行设置；module_contracts.py：真实轴、形状、含义和连接编译。','- sequence_engine.py：按需读取、空间编辑、时间编辑、统计、逐帧扩展与标准 g²；两个预设和试算 / 正式运行复用。','- sequence_processing.py / sequence_storage.py：完整 values.npy、数值 CSV、逐样本 references.jsonl、索引、统计与 LUT 快照、运行记录；矩阵附首幅 TIFF，信号附 PNG/SVG。','- 曲线网页最多显示约2000个样本，静态图最多约20000个；数值 CSV/NPY 保留全部数据，图中注明抽样。普通视频的精确时间编辑在计算进程建立临时解码索引，任务结束后清理。','- 旧 module_engine / split runner 保留为历史兼容执行与结果读取；新 Web 工作流显式使用契约3，不能混用旧任务级 ROI/范围。','']
    path=ROOT/'docs'/'MODULE_INVENTORY_CURRENT.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(f'已核对并导出 {len(specs)} 个内置模块：{path}')


if __name__=='__main__':main()
