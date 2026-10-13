# 当前已实现模块清单（代码核对）

目录取自版本 `0.7.4`，新流程契约版本为 `3`。模块库提供 **25 个模块**，另保留 9 个旧记录兼容定义；插件另行注册。分类、公式与参数取自统一代码目录。

完整形状与科学规则见 [MODULE_CONTRACTS_0_6_0.md](MODULE_CONTRACTS_0_6_0.md)；重构前行为保留在 [MODULE_INVENTORY_0_5_2.md](MODULE_INVENTORY_0_5_2.md)。旧结果按旧记录读取，不重新解释。

## 轴、参数与执行归属

- T 为当前时间样本，Q 为有符号延迟，P 为左右双路，Y/X 为空间，C 为明确的 RGB/RGBA 分量；R 为剖面位置。分布使用单独的箱轴。标量与向量保留真实维度。
- 实例：algorithm、id、version、parameters、enabled、target、execution、插件路径与内容哈希。每个实例独立；停用原样通过。
- execution：backend=cpu/auto/gpu（GPU 当前仅关联）；memory_mb=16…4096，默认256；precision=native/float32/float64，默认算法类型。每步工作数组按自身预算校验，重复统计保留数组与缓存合计受流程预算约束。标签禁止浮点精度转换。
- 运行范围只在起点 time_crop；ROI 由 roi_crop 裁剪，逐样本区域运算由 time_statistics 完成；全域统计可压缩样本轴。输出目录、名称、试算与运行按钮属于工作台。
- 每步输入输出轴、形状、样本数、数据含义、坐标原点及上游指纹随记录保存；未知时间 / 采集帧保持未知。
- 通用与分幅共用 SequenceEngine 与结果类型，仅预置不同流程。编辑中的不完整连接不影响原始数据预览。
- 默认配准按当前来源宽高初始化两半；关联按来源帧数给初始 W/τ。之后的裁剪会改变样本数，需据连接提示调整关联参数，不会静默改小。

## 三类模块

| 类别 | 数量 |
|---|---|
| 中间模块 · 数据编辑 | 11 |
| 算法模块 · 科学计算 | 5 |
| 修饰模块 · 统计与编辑 | 9 |

## 定类型增益 / 偏置（兼容） `intensity`

- 类别：中间模块 · 数据编辑；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B = cast_clip_round(A × gain + offset)
- 实际输入输出：矩阵、双路、标量 / 向量 → 同形状、同 dtype。
- 边界：整数先 round 再裁剪到原 dtype 范围；bool 以非零为真。内置数值算法拒绝 64 位整数以避免隐式精度损失。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 增益 `gain` | float | `1.0` | ≥ -1000；≤ 1000 |
| 偏置 `offset` | float | `0.0` | ≥ -1000000000.0；≤ 1000000000.0 |

## 空间均值平滑（兼容） `box_blur`

- 类别：中间模块 · 数据编辑；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B[y,x] = Σ A[y+dy,x+dx] / (2r+1)²
- 实际输入输出：YX / YXC；双路分别处理 → 同形状、同 dtype。
- 边界：二维空间反射边界，积分图求均值；要求全部输入有限。只作用于空间轴；ROI 前后的排列决定真实滤波上下文。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 半径 `radius` | int | `1` | ≥ 1；≤ 32 |

## 阈值分割 `threshold`

- 类别：算法模块 · 科学计算；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B = uint8(A ≥ threshold)；彩色 A = mean(R,G,B)
- 实际输入输出：矩阵 / 标量 → 同形状；含 C 的彩色输入去除 C → 0/1 uint8 标签。
- 边界：NaN 比较得到 0；标签及其派生值不能隐式作为光强进入统计或关联。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 阈值 `threshold` | float | `128.0` | ≥ -1000000000000.0；≤ 1000000000000.0 |

## 浮点增益 / 偏置（兼容） `affine`

- 类别：中间模块 · 数据编辑；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B = gain × A + offset
- 实际输入输出：矩阵、双路、标量 / 彩色向量 → 同形状 float64。
- 边界：不量化，不截断；非有限结果置 NaN。标量保持真实的零维形状，不伪装成图像。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 倍率 `gain` | float | `1.0` | ≥ -1000000000000.0；≤ 1000000000000.0 |
| 偏置 `offset` | float | `0.0` | ≥ -1000000000000.0；≤ 1000000000000.0 |

## 空间高斯平滑（兼容） `gaussian`

- 类别：中间模块 · 数据编辑；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：K[k] ∝ exp(−k²/(2σ²))；B = (Kx*Ky*(A·valid))/(Kx*Ky*valid)
- 实际输入输出：YX / YXC；双路分别处理 → 同形状 float64。
- 边界：空间轴 0/1，边界复制；原无效位置仍为 NaN，不把时间轴作为空间。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| σ `sigma` | float | `1.0` | ≥ 1e-12；≤ 1000000000000.0 |
| 半径 `radius` | int | `2` | ≥ 1；≤ 32 |

## 固定尺度强度映射 `curve`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：x=A/input_max；identity:A，linear:x，sqrt:√x，square:x²，gamma:x^γ，log:ln(1+9x)/ln(10)
- 实际输入输出：矩阵、双路、标量 / 向量、系数 → 同形状 float64。
- 边界：固定输入尺度；负数非法定义域或非有限结果为 NaN；identity 不除以 input_max。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 曲线 `curve` | str | `"linear"` | identity / linear / sqrt / square / gamma / log |
| 固定输入尺度 `input_max` | float | `1023.0` | ≥ 1e-12；≤ 1000000000000000.0 |
| 指数 γ `gamma` | float | `1.0` | ≥ 1e-12；≤ 1000000.0 |

## 传感器强度标定 `calibration`

- 类别：算法模块 · 科学计算；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：I = F(A) − black_side；F 为固定曲线或 LUT[A]
- 实际输入输出：光强矩阵 / 双路 / 标量 → 同形状 float64。
- 边界：饱和条件 A ≥ saturation_code 或输入非有限时置 NaN。LUT 仅接受覆盖有效整数码值的单调非减有限数组（2–65536 项，≤8 MiB）。单幅不同左右暗场需后续明确配准中线；有空间编辑时请校准后再编辑，或配准后按两路校准。LUT 与非 identity 曲线不能同时启用，内容保存快照。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 强度曲线 `curve` | str | `"identity"` | identity / linear / sqrt / square / gamma / log |
| 固定输入尺度 `input_max` | float | `1023.0` | ≥ 1e-12 |
| 指数 γ `gamma` | float | `1.0` | ≥ 1e-12 |
| 自定义 LUT 路径 `lut_path` | str | `null` | 可留空 |
| 左路 / 单幅暗场偏置 `black_left` | float | `0.0` | 需满足算法有限值与结构校验 |
| 右路暗场偏置 `black_right` | float | `0.0` | 需满足算法有限值与结构校验 |
| 饱和码值阈值 `saturation_code` | float | `null` | 可留空 |
| 输入含义 `input_semantics` | str | `"sensor_codes"` | sensor_codes / linear_intensity |

## 双幅配准 `alignment`

- 类别：算法模块 · 科学计算；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：L = crop(left)；R[x,y] = sample(crop(right), mirror(x+dx,y+dy))
- 实际输入输出：单幅灰度 TYX → 对齐双路 TPYX，P=2。
- 边界：依据中线、镜像和采样坐标取公共有效区；最近邻 floor(v+0.5)，或显式双线性；不支持彩色或标量。自动建议：去均值、Hann 窗、归一互功率谱 FFT 相位相关、峰值二次细化；建议须检查后接受。坐标相对此模块的直接输入。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 左幅 X `left.x` | int | `0` | ≥ 0 |
| 左幅 Y `left.y` | int | `0` | ≥ 0 |
| 左幅 宽 `left.width` | int | `1` | ≥ 1 |
| 左幅 高 `left.height` | int | `1` | ≥ 1 |
| 右幅 X `right.x` | int | `0` | ≥ 0 |
| 右幅 Y `right.y` | int | `0` | ≥ 0 |
| 右幅 宽 `right.width` | int | `1` | ≥ 1 |
| 右幅 高 `right.height` | int | `1` | ≥ 1 |
| 中线顶部 / 底部 X `midline` | array | `[1.0,1.0]` | 需满足算法有限值与结构校验 |
| 右路采样偏移 X `dx` | float | `0.0` | 需满足算法有限值与结构校验 |
| 右路采样偏移 Y `dy` | float | `0.0` | 需满足算法有限值与结构校验 |
| 右路水平镜像 `mirror_x` | bool | `false` | 需满足算法有限值与结构校验 |
| 右路垂直镜像 `mirror_y` | bool | `false` | 需满足算法有限值与结构校验 |
| 空间采样 `interpolation` | str | `"nearest"` | nearest / bilinear |

## 双路选幅 `select_channel`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B = L 或 R
- 实际输入输出：TPYX → TYX；TP → T；保留 C 时 TPC → TC。
- 边界：只移除双路 P 轴；输入为单路时拒绝。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 输出幅面 `channel` | str | `"left"` | left / right |

## 统计与归一（兼容） `statistics`

- 类别：修饰模块 · 统计与编辑；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：μ=Σ_valid I/n；M2 由 Welford 累积；variance=M2/n；s=sqrt(Σ_space M2/Σ_space n)；输出 I/s
- 实际输入输出：光强时间序列 T(P)YX(C) 或 T(P)(C) → 同形状浮点序列，并保存均值、方差、计数、尺度。
- 边界：扫描本模块直接输入的全部样本，之前的时间裁剪 / 抽样 / 合帧和 ROI 都参与。none 时 s=1；fixed_rms 且 RMS≤1e−12 时 s=1；全无效均值为 NaN。标准 g² 不将去均值后的残差当作分母；后续编辑会使统计元数据失效。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 固定尺度 `normalization` | str | `"none"` | none / fixed_rms |

## 二阶互关联 g²（兼容） `correlation`

- 类别：算法模块 · 科学计算；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：g²_q(c)=mean_valid(L[i]R[i+q])/(mean_valid(L[i])mean_valid(R[i+q]))；i=c−W//2…c+W//2
- 实际输入输出：TPYX → 固定 τ 的 T′YX 或固定中心的 QYX；TP → T′ 或 Q。
- 边界：W、τ、输出步长均按直接上游的编辑后样本计数。负 q 反向配对，不能假设对称。左右采用同一有效时间样本集合，float64 累加；均值须大于 min_mean、计数至少 min_samples，否则 NaN。窗口仅保留完整边界；保存逐对真实时差的均值 / 最小 / 最大。ROI 不再藏在关联中：之前归约得到区域信号 g²，之后归约得到像素 g² 的空间统计。roi_estimator 仅保留旧记录兼容，新流程固定 pixel_mean（实际空间操作由 ROI 模块决定）。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 平均窗口 W（输入样本，奇数） `window` | int | `101` | ≥ 1 |
| 最小强度均值 `min_mean` | float | `1e-09` | ≥ 1e-15 |
| 最少有效样本 `min_samples` | int | `2` | ≥ 1 |
| 输出步长（输入样本） `stride` | int | `100` | ≥ 1 |
| 关联输出 `analysis` | str | `"map"` | map / curve |
| 固定 τ（输入样本） `map_lag` | int | `0` | 需满足算法有限值与结构校验 |
| τ 下限（输入样本） `lag_min` | int | `-10` | 需满足算法有限值与结构校验 |
| τ 上限（输入样本） `lag_max` | int | `10` | 需满足算法有限值与结构校验 |
| 延迟步长（输入样本） `lag_step` | int | `1` | ≥ 1 |
| 延迟列表（留空使用范围） `lag_list` | array | `null` | 可留空 |
| ROI 曲线估计 `roi_estimator` | str | `"pixel_mean"` | pixel_mean / intensity_sum |
| 固定中心样本（编辑后，从 1 起；留空居中） `center` | int | `null` | ≥ 1；可留空 |

## 时间范围 `time_crop`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：样本：B=A[first−1:last]；时间：保留 start≤t<end 的实际样本
- 实际输入输出：T×… → Tc×…，不改变各样本形状。
- 边界：只在起点设置一次；样本尾号包含、真实时间终点不包含。普通视频按实际解码时间建立临时索引，不以平均帧率推算。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 范围单位 `unit` | str | `"samples"` | samples / seconds |
| 首样本（从 1 起） `first` | int | `1` | ≥ 1 |
| 尾样本（包含，留空至末尾） `last` | int | `null` | ≥ 1；可留空 |
| 起始时间（秒） `start` | float | `0.0` | ≥ 0 |
| 结束时间（秒，不含） `end` | float | `null` | ≥ 0；可留空 |

## 空间 ROI / 归约（兼容） `spatial_roi`

- 类别：中间模块 · 数据编辑；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：区域像素 B=A[y:y+h,x:x+w]；mean=Σ_valid/n；sum=Σ_valid；quantile 在 u=(n−1)q 两侧线性插值
- 实际输入输出：T(P)YX(C) → T(P)hw(C)，或归约为 T(P)(C)；QYX → Qhw 或 Q。
- 边界：空间坐标相对此步直接输入；范围与原点随输出传递。全无效为 NaN。双路归约默认取当前样本共同有效的像素，亦可明确选择各路独立集合；掩码可能随样本改变，不能暗称固定公共支持。标量后再接空间操作会报错。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| X `x` | int | `0` | ≥ 0 |
| Y `y` | int | `0` | ≥ 0 |
| 宽（留空至边界） `width` | int | `null` | ≥ 1；可留空 |
| 高（留空至边界） `height` | int | `null` | ≥ 1；可留空 |
| 区域操作 `mode` | str | `"pixels"` | pixels / mean / sum / quantile |
| 分位 q `quantile` | float | `0.5` | ≥ 0；≤ 1 |
| 双路归约有效像素 `validity` | str | `"joint"` | joint / independent |

## 间隔采样 `time_sample`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：B[j]=A[offset+j·step]
- 实际输入输出：T×… → ceil((T−offset)/step)×…。
- 边界：保留真实时间和来源；输出不能为空；Q 延迟轴不能当作时间轴抽样。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 样本间隔 `step` | int | `2` | ≥ 1 |
| 起始偏移（从 0 起） `offset` | int | `0` | ≥ 0 |

## 时间滑窗 `time_smooth`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：B[j]=mean_valid(A[j:j+W])；或逐元素 quantile
- 实际输入输出：T×… → (T−W+1)×… float64。
- 边界：W 为正奇数；完整窗口、不补帧；时间对应中心样本，贡献范围保留；空间 / 通道轴不归约，分位值窗口需要满足内存预算。放在 g² 后只编辑结果，不改变原关联 W/τ。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 窗口 W（奇数） `window` | int | `3` | ≥ 1 |
| 窗口操作 `mode` | str | `"mean"` | mean / quantile |
| 分位 q `quantile` | float | `0.5` | ≥ 0；≤ 1 |

## 时间合帧 `time_bin`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：B[j]=mean_valid 或 sum_valid(A[jB:(j+1)B])
- 实际输入输出：T×… → floor(T/B)×… float64。
- 边界：不满尾组舍弃；组内真实时间取均值，任一未知则未知；合成样本的采集帧号为空，保留贡献范围。全无效为 NaN。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 每组样本数 `size` | int | `2` | ≥ 1 |
| 合帧操作 `mode` | str | `"mean"` | mean / sum |

## 空间平滑 `spatial_smooth`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：mean: B=ΣA/(2r+1)²; gaussian: B=(K*(A·valid))/(K*valid)
- 实际输入输出：含Y/X的空间矩阵；双路分别处理 → 同形状空间平滑输出；不能接标量或线信号
- 边界：类型在模块内选择。均值采用反射边界，高斯采用边界复制；算法默认保留输入类型，浮点精度不先量化。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 平滑类型 `method` | str | `"mean"` | mean / gaussian |
| 空间半径 / px `radius` | int | `1` | ≥ 1；≤ 32 |
| 高斯 σ / px `sigma` | float | `1.0` | ≥ 1e-12；≤ 1000000000000.0 |

## 增益 / 偏置 `gain_offset`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B=gA+b
- 实际输入输出：矩阵 / 双路 / 标量 / 向量数值 → 同轴同形状，dtype 由实例精度决定
- 边界：算法默认保留输入类型；整数舍入限幅。选择 float32/64 时直接输出浮点，不先量化。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 增益 `gain` | float | `1.0` | ≥ -1000000000000.0；≤ 1000000000000.0 |
| 偏置 `offset` | float | `0.0` | ≥ -1000000000000.0；≤ 1000000000000.0 |

## 时序统计 `time_statistics`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：每个样本：mean / sum / min / max / variance / std / quantile / count
- 实际输入输出：T/Q/N×空间或分量；双路与颜色分量明确 → T/Q/N(P)(C)，逐样本统计值；count 为 uint64，其余 float64；可旁路
- 边界：逐样本归约空间 / 分量，保留 T 或 Q 及 P/C，不压缩时间。方差为总体方差（ddof=0）。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 统计量 `mode` | str | `"mean"` | mean / sum / min / max / variance / std / quantile / count |
| 分位 q `quantile` | float | `0.5` | ≥ 0；≤ 1 |
| 双路有效值 `validity` | str | `"joint"` | joint / independent |
| 数据流作用 `output` | str | `"main"` | main / side |

## 全域统计 `global_statistics`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：全上游：mean / sum / min / max / variance / std / quantile / count
- 实际输入输出：任意 T/Q/N 数值序列的完整直接上游 → N=1 静态统计记录；保留轴由统计范围决定；value/count 可用于后序参数绑定
- 边界：默认压缩所有轴；也可只压缩样本轴，或压缩样本与空间轴并保留 P/C。输出 N=1，不伪造时间。方差为总体方差；精确分位值使用临时磁盘缓存。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 统计范围 `scope` | str | `"all"` | all / samples / space_time |
| 统计量 `mode` | str | `"mean"` | mean / sum / min / max / variance / std / quantile / count |
| 分位 q `quantile` | float | `0.5` | ≥ 0；≤ 1 |
| 双路有效值 `validity` | str | `"independent"` | joint / independent |
| 数据流作用 `output` | str | `"main"` | main / side |

## ROI 裁剪 `roi_crop`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B=A[y:y+h,x:x+w]
- 实际输入输出：包含 Y/X 的 T/Q(P)YX(C) → 保留像素的 T/Q(P)hw(C)，原dtype与坐标来源
- 边界：相对此模块直接输入坐标；保留空间轴和来源原点。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| X `x` | int | `0` | ≥ 0 |
| Y `y` | int | `0` | ≥ 0 |
| 宽（留空至边界） `width` | int | `null` | ≥ 1；可留空 |
| 高（留空至边界） `height` | int | `null` | ≥ 1；可留空 |

## 区域归约（兼容） `region_reduce`

- 类别：修饰模块 · 统计与编辑；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：mean=Σvalid(A)/n; sum=Σvalid(A); quantile=Q_q(A)
- 实际输入输出：包含 Y/X 的矩阵 → 归约空间后的 T/Q(P)(C)，float64；或同类型侧输出且主流透传
- 边界：归约 Y/X；全部无效为 NaN；双路共同 / 独立支持显式选择。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 区域运算 `mode` | str | `"mean"` | mean / sum / quantile |
| 分位 q `quantile` | float | `0.5` | ≥ 0；≤ 1 |
| 双路有效像素 `validity` | str | `"joint"` | joint / independent |
| 数据流作用 `output` | str | `"main"` | main / side |

## 序列统计（兼容） `sequence_statistics`

- 类别：修饰模块 · 统计与编辑；仅用于旧记录兼容；新界面不提供。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：μ=Σvalid(I)/n; variance=M2/n; s=√(ΣM2/Σn)
- 实际输入输出：光强时间序列 T(P)YX(C) 或 T(P)(C) → 主流保持轴/形状/dtype；N=1统计记录 mean/variance/count/scale
- 边界：旧流程专用：主流原样通过，保留统计与固定涨落 RMS 端口。新流程使用时序统计 / 全域统计。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 尺度统计 `normalization` | str | `"fixed_rms"` | none / fixed_rms |

## 固定尺度归一 `fixed_normalize`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B=A/s
- 实际输入输出：数值强度或系数；不接受标签 → 同轴同形状float64；系数标为派生g²
- 边界：显式固定尺度，必须有限且大于 0；输出 float64。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 固定尺度 `scale` | float | `1.0` | ≥ 1e-15 |

## 参数传递 `parameter_binding`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：main=A; value=v
- 实际输入输出：任意主流 + 常量/前序数值参数端口 → 主流透传；N=1命名标量value端口
- 边界：只发布数值标量；允许常量或已启用前序模块的明确端口，禁止循环与任意代码。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 变量名 `name` | str | `"value"` | 需满足算法有限值与结构校验 |
| 变量值 `value` | float | `1.0` | 需满足算法有限值与结构校验 |
| 单位 `unit` | str | `""` | 需满足算法有限值与结构校验 |

## 灰度转换 `grayscale`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：Y=.2126R+.7152G+.0722B 或 (R+G+B)/3
- 实际输入输出：空间RGB/RGBA或明确BGR颜色数据 → 去除C轴的float64灰度矩阵
- 边界：明确转换 RGB；忽略 alpha，float64 输出；灰度输入拒绝重复转换。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 灰度定义 `method` | str | `"luma"` | luma / mean |

## 颜色排列 / 扩展 `color_convert`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：RGB←repeat(gray,3); BGR=RGB[...,::-1]
- 实际输入输出：灰度或RGB/BGR三分量空间图像 → 同尺度RGB/BGR，显式color_order；不接受RGBA重排
- 边界：显式声明输出颜色顺序；不暗中改变强度尺度。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 输出排列 `mode` | str | `"rgb"` | rgb / bgr |

## 点信号提取 `point_extract`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B=A[...,y,x,...]
- 实际输入输出：含Y/X的T/Q空间矩阵 → 去除Y/X的T/Q(P)(C)点信号；可旁路
- 边界：去除 Y/X，保留 T 或 Q 及显式通道；坐标为直接上游。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| X `x` | int | `0` | ≥ 0 |
| Y `y` | int | `0` | ≥ 0 |
| 数据流作用 `output` | str | `"main"` | main / side |

## 直线截面提取 `line_extract`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：p_j=p₀+j(p₁−p₀)/(N−1); B_j=sample(A,p_j)
- 实际输入输出：含Y/X的T/Q空间矩阵 → T/Q(P)R(C)线信号，R为像素距离；可旁路
- 边界：横纵便捷线或任意端点线段；R 为像素距离，最近邻或明确双线性。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 直线定义 `mode` | str | `"axis"` | axis / segment |
| 保留方向 `axis` | str | `"X"` | X / Y |
| 固定另一轴位置（从0起） `index` | int | `0` | ≥ 0 |
| 起点 X `x0` | float | `0.0` | ≥ 0 |
| 起点 Y `y0` | float | `0.0` | ≥ 0 |
| 终点 X `x1` | float | `0.0` | ≥ 0 |
| 终点 Y `y1` | float | `0.0` | ≥ 0 |
| 沿线样本数 `samples` | int | `64` | ≥ 2；≤ 100000 |
| 沿线插值 `interpolation` | str | `"nearest"` | nearest / bilinear |
| 数据流作用 `output` | str | `"side"` | main / side |

## 按轴截面 `axis_slice`

- 类别：中间模块 · 数据编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：B=take(A,index,axis)
- 实际输入输出：包含指定轴的输入 → 去除被选轴；若选择T/Q，输出N=1静态记录；可旁路
- 边界：按明确轴与从 0 起位置提取；T/Q 单样本截面改为 N 记录轴。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 截面轴 `axis` | str | `"T"` | T / Q / P / C / X / Y / R / B / N |
| 位置（从0起） `index` | int | `0` | ≥ 0 |
| 数据流作用 `output` | str | `"side"` | main / side |

## 分布统计 `distribution`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：H[b]=count(e_b≤finite(A)<e_(b+1))
- 实际输入输出：任意有限或含无效值的当前数值样本 → T/Q/N×B固定箱计数int64及箱边界；可旁路
- 边界：上下界可独立手动设置或自动识别上游完整范围的有限值极值；整段固定等宽箱，末箱含右边界。常量数据自动边界向外扩展；无有限值时自动识别报错。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 箱数 `bins` | int | `64` | ≥ 2；≤ 4096 |
| 下界 `minimum` | float | `0.0` | 可留空 |
| 上界 `maximum` | float | `65535.0` | 可留空 |
| 数据流作用 `output` | str | `"side"` | main / side |

## 数据截流导出 `data_tap`

- 类别：修饰模块 · 统计与编辑；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：main=A; captured=A
- 实际输入输出：任意已声明轴、dtype和语义的直接上游 → 主流透传；完整真实上游数据侧输出
- 边界：获取直接上游真实数据；主流透传，不更改轴、dtype 或来源。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 输出名称 `name` | str | `"截流输出"` | 需满足算法有限值与结构校验 |
| 交付方式 `destination` | str | `"both"` | preview / file / both |
| 导出格式 `export_format` | str | `"dataset"` | dataset / npy / csv / tiff |

## 固定延迟 g² 系数图 `correlation_map`

- 类别：算法模块 · 科学计算；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：g²_q(c)=⟨L_iR_(i+q)⟩/(⟨L_i⟩⟨R_(i+q)⟩)
- 实际输入输出：T(P=2)YX 或 TP 的双路强度；不接受C/标签/Q → 固定τ：T′YX 或 T′；固定中心扫描：QYX 或 Q，标准g²
- 边界：固定 q，沿中心样本 T 输出；W、q、步长按直接上游样本计数。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 平均窗口 W（输入样本，奇数） `window` | int | `101` | ≥ 1 |
| 最小强度均值 `min_mean` | float | `1e-09` | ≥ 1e-15 |
| 最少有效样本 `min_samples` | int | `2` | ≥ 1 |
| 输出步长（输入样本） `stride` | int | `100` | ≥ 1 |
| 关联输出 `analysis` | str | `"map"` | map / curve |
| 固定 τ（输入样本） `map_lag` | int | `0` | 需满足算法有限值与结构校验 |
| τ 下限（输入样本） `lag_min` | int | `-10` | 需满足算法有限值与结构校验 |
| τ 上限（输入样本） `lag_max` | int | `10` | 需满足算法有限值与结构校验 |
| 延迟步长（输入样本） `lag_step` | int | `1` | ≥ 1 |
| 延迟列表（留空使用范围） `lag_list` | array | `null` | 可留空 |
| ROI 曲线估计 `roi_estimator` | str | `"pixel_mean"` | pixel_mean / intensity_sum |
| 固定中心样本（编辑后，从 1 起；留空居中） `center` | int | `null` | ≥ 1；可留空 |

## 延迟扫描 g² `correlation_curve`

- 类别：算法模块 · 科学计算；可从模块库添加。
- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：g²_q(c)=⟨L_iR_(i+q)⟩/(⟨L_i⟩⟨R_(i+q)⟩)
- 实际输入输出：T(P=2)YX 或 TP 的双路强度；不接受C/标签/Q → 固定τ：T′YX 或 T′；固定中心扫描：QYX 或 Q，标准g²
- 边界：固定中心 c，沿有符号 Q 输出；Q 不是时间。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 平均窗口 W（输入样本，奇数） `window` | int | `101` | ≥ 1 |
| 最小强度均值 `min_mean` | float | `1e-09` | ≥ 1e-15 |
| 最少有效样本 `min_samples` | int | `2` | ≥ 1 |
| 输出步长（输入样本） `stride` | int | `100` | ≥ 1 |
| 关联输出 `analysis` | str | `"curve"` | map / curve |
| 固定 τ（输入样本） `map_lag` | int | `0` | 需满足算法有限值与结构校验 |
| τ 下限（输入样本） `lag_min` | int | `-10` | 需满足算法有限值与结构校验 |
| τ 上限（输入样本） `lag_max` | int | `10` | 需满足算法有限值与结构校验 |
| 延迟步长（输入样本） `lag_step` | int | `1` | ≥ 1 |
| 延迟列表（留空使用范围） `lag_list` | array | `null` | 可留空 |
| ROI 曲线估计 `roi_estimator` | str | `"pixel_mean"` | pixel_mean / intensity_sum |
| 固定中心样本（编辑后，从 1 起；留空居中） `center` | int | `null` | ≥ 1；可留空 |

## 自定义算法与机器学习边界

已注册插件沿用逐帧 prepare/process/close 契约，输入输出须为明确的 YX / YXC 图像，空间尺寸保持不变；标签输出可去除颜色轴。实际输出 dtype 在运行确定并保存，不强制转回来源类型。未声明的标量、时间窗、形状变换明确拒绝。机器学习可通过此契约接入，尚无内置训练 / ONNX 推理模块。

## 执行与存储位置

- module_registry.py：公共目录、独立实例参数及执行设置；module_contracts.py：真实轴、形状、含义和连接编译。
- sequence_engine.py：按需读取、空间编辑、时间编辑、统计、逐帧扩展与标准 g²；两个预设和试算 / 正式运行复用。
- sequence_processing.py / sequence_storage.py：完整 values.npy、数值 CSV、逐样本 references.jsonl、索引、统计与 LUT 快照、运行记录；矩阵附首幅 TIFF，信号附 PNG/SVG。
- 曲线网页最多显示约2000个样本，静态图最多约20000个；数值 CSV/NPY 保留全部数据，图中注明抽样。普通视频的精确时间编辑在计算进程建立临时解码索引，任务结束后清理。
- 旧 module_engine / split runner 保留为历史兼容执行与结果读取；新 Web 工作流显式使用契约3，不能混用旧任务级 ROI/范围。
