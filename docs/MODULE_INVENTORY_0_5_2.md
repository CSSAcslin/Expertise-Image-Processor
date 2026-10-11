# 0.5.2 旧实现模块盘点（重构前）

目录取自版本 `0.5.2`。该版本内置 **11 个模块**：6 个逐帧数值算法、5 个校准 / 结构 / 时序模块；插件不计入内置数量。本文保留重构前代码的实际行为，用于对照历史记录；当前实现见 [MODULE_INVENTORY_CURRENT.md](MODULE_INVENTORY_CURRENT.md)。

新设计及计划新增的 5 个空间 / 时间模块见 [MODULE_CONTRACTS_0_6_0.md](MODULE_CONTRACTS_0_6_0.md)。

## 当前公共参数归属

- 模块实例：algorithm、id、version、parameters、enabled、target、插件路径与内容哈希。target 只允许双路或当前图、左路、右路，单路目标只适用于配准后的逐帧算法。
- 通用任务：处理范围与输出 ROI 存于 Selection；ROI 是链末输出裁剪。
- 分幅任务：start/end/center/roi/points/memory_mb/backend/output_dtype 存于 CorrelationConfig；流程内部不能表达空间或时间裁剪节点。
- 当前目录只声明 sensor/pair/moments/map，缺少时间轴长度、空间轴消除、标量、向量和具体尺寸连接契约；output_shape 仅识别配准。
- 表中的默认值来自静态目录。添加配准时会按来源宽高初始化左右半幅；添加关联时会按来源帧数缩小 W 和 τ 范围。

## 强度线性变换 `intensity`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B = cast_clip_round(A × gain + offset)
- 实际输入输出：YX / YXC；已配准双路分别处理；关联后 YX → 同形状、同 dtype。
- 边界：整数先 round 再裁剪到原 dtype 范围；bool 以非零为真。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 增益 `gain` | float | `1.0` | ≥ -1000；≤ 1000 |
| 偏置 `offset` | float | `0.0` | ≥ -1000000000.0；≤ 1000000000.0 |

## 均值滤波 `box_blur`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B[y,x] = Σ A[y+dy,x+dx] / (2r+1)²
- 实际输入输出：YX / YXC；双路分别处理 → 同形状、同 dtype。
- 边界：二维空间反射边界，积分图求均值；要求全部输入有限。当前任务 ROI 在所有算法之后裁剪，不能改变滤波输入上下文。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 半径 `radius` | int | `1` | ≥ 1；≤ 32 |

## 阈值分割 `threshold`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B = uint8(A ≥ threshold)；彩色 A = mean(R,G,B)
- 实际输入输出：YX → YX；YXC → YX；双路分别处理 → 0/1 uint8 标签。
- 边界：NaN 比较得到 0；标签不得继续作为光强进入统计或关联。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 阈值 `threshold` | float | `128.0` | ≥ -1000000000000.0；≤ 1000000000000.0 |

## 线性缩放与偏置 `affine`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B = gain × A + offset
- 实际输入输出：同 intensity 的图像输入 → 同形状 float64。
- 边界：不量化，不截断；非有限结果置 NaN。内部对区域总强度曲线的标量通过 1×1 适配，尚无显式标量契约。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 倍率 `gain` | float | `1.0` | ≥ -1000000000000.0；≤ 1000000000000.0 |
| 偏置 `offset` | float | `0.0` | ≥ -1000000000000.0；≤ 1000000000000.0 |

## 空间高斯平滑 `gaussian`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：K[k] ∝ exp(−k²/(2σ²))；B = (Kx*Ky*(A·valid))/(Kx*Ky*valid)
- 实际输入输出：YX / YXC；双路分别处理 → 同形状 float64。
- 边界：空间轴 0/1，边界复制；原无效位置仍为 NaN，不把时间轴作为空间。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| σ `sigma` | float | `1.0` | ≥ 1e-12；≤ 1000000000000.0 |
| 半径 `radius` | int | `2` | ≥ 1；≤ 32 |

## 强度曲线 `curve`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：x=A/input_max；identity:A，linear:x，sqrt:√x，square:x²，gamma:x^γ，log:ln(1+9x)/ln(10)
- 实际输入输出：图像 / 双路 / 系数图 → 同形状 float64。
- 边界：固定输入尺度；负数非法定义域或非有限结果为 NaN；identity 不除以 input_max。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 曲线 `curve` | str | `"linear"` | identity / linear / sqrt / square / gamma / log |
| 固定输入尺度 `input_max` | float | `1023.0` | ≥ 1e-12；≤ 1000000000000000.0 |
| 指数 γ `gamma` | float | `1.0` | ≥ 1e-12；≤ 1000000.0 |

## 强度校准 / LUT `calibration`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：I = F(A) − black_side；F 为固定曲线或 LUT[A]
- 实际输入输出：当前登记仅 sensor：YX → YX float64；配准前分别校准左右区域。
- 边界：饱和条件 A ≥ saturation_code 或输入非有限时置 NaN。LUT 仅接受覆盖有效整数码值的单调非减有限数组（2–65536 项，≤8 MiB）。没有后续配准时左右暗场必须相同；LUT 与非 identity 曲线不能同时启用。

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

## 双幅划分与固定配准 `alignment`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：L = crop(left)；R[x,y] = sample(crop(right), mirror(x+dx,y+dy))
- 实际输入输出：灰度 YX → 二元组 (L[h,w],R[h,w])，两路均 float64。
- 边界：依据中线、镜像和采样坐标取公共有效区；最近邻 floor(v+0.5)，或显式双线性；不支持彩色。自动建议：去均值、Hann 窗、归一互功率谱 FFT 相位相关、峰值二次细化；建议须检查后接受。

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

## 选择幅面 `select_channel`

- 版本：`1.0`；执行范围：`frame`。
- 公式 / 算法：B = L 或 R
- 实际输入输出：pair → sensor：当前只能得到 YX 幅面。
- 边界：保留该幅面的几何来源，输入为单幅时拒绝。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 输出幅面 `channel` | str | `"left"` | left / right |

## 全源均值与固定尺度 `statistics`

- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：μ=Σ_valid I/n；M2 由 Welford 累积；variance=M2/n；s=sqrt(Σ_pixels M2/Σ_pixels n)
- 实际输入输出：已配准 pair[h,w] 的全序列 → 同双路的均值、方差、计数及两路固定尺度。
- 边界：当前时序适配器标记为 moments，不是独立的数据流对象；扫描原始来源的全部帧，任务时间范围不改变基线范围。none 时 s=1；fixed_rms 且 RMS≤1e−12 时 s=1；均值无有效样本为 NaN。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 固定尺度 `normalization` | str | `"none"` | none / fixed_rms |

## 标准二阶互关联 g² `correlation`

- 版本：`1.0`；执行范围：`sequence`。
- 公式 / 算法：g²_q(c)=mean_valid(L[i]R[i+q])/(mean_valid(L[i])mean_valid(R[i+q]))；i=c−W//2…c+W//2
- 实际输入输出：pair / moments → 固定 τ 随时间的图（存储 T×Q×Y×X）；或固定中心的点 / ROI τ 曲线。
- 边界：左右采用同一有效样本集合，float64 累加。负 q 反向配对；均值必须大于 min_mean，计数至少 min_samples，否则 NaN。真实均值通过基线加回局部残差均值保留；不能用零均值残差作分母。ROI 的 pixel_mean 为像素 g² 的区域均值；intensity_sum 为固定公共有效像素的两路总强度 g²。目前这两个操作藏在曲线估计器内。

| 参数 | 类型 | 静态默认值 | 范围 / 选项 |
|---|---|---|---|
| 平均窗口 W（奇数） `window` | int | `101` | ≥ 1 |
| 最小强度均值 `min_mean` | float | `1e-09` | ≥ 1e-15 |
| 最少有效样本 `min_samples` | int | `2` | ≥ 1 |
| 输出时间步长 `stride` | int | `100` | ≥ 1 |
| 关联输出 `analysis` | str | `"map"` | map / curve |
| 固定 τ（帧） `map_lag` | int | `0` | 需满足算法有限值与结构校验 |
| τ 下限（帧） `lag_min` | int | `-10` | 需满足算法有限值与结构校验 |
| τ 上限（帧） `lag_max` | int | `10` | 需满足算法有限值与结构校验 |
| 延迟步长 `lag_step` | int | `1` | ≥ 1 |
| 延迟列表（留空使用范围） `lag_list` | array | `null` | 可留空 |
| ROI 曲线估计 `roi_estimator` | str | `"pixel_mean"` | pixel_mean / intensity_sum |

## 自定义算法与机器学习边界

已注册插件沿用逐帧 prepare/process/close 契约，输入输出必须为明确的 YX / RGB(A) 图像，空间尺寸保持不变，输出可声明 image 或 label；不默认支持标量、向量、时间窗或改变空间尺寸。模型运行协议已预留，尚无内置 ONNX 推理、训练或模型调度模块。新扩展必须在注册描述中声明实际支持的轴和输出规则。

## 重构必须改动的执行位置

- module_registry：从语义枚举扩展为轴、形状、样本数、参数和依赖契约；不能把不兼容的操作默默适配成 1×1。
- module_engine / workflow：按真实列表逐节点执行空间和时间操作，不能只把流程切成配准前、统计、关联后几段。
- processing：将任务尾 ROI 和任务范围迁移到显式模块；试算及正式处理共用节点实现。
- runner / statistics_key：统计遍历该节点直接输入，缓存指纹包含其实际空间 / 时间上游；τ 和 W 的计数语义确定后一起迁移窗口计划。
- domain / storage / readers：同时支持矩阵序列与标量 / 向量序列，保存真实时间和来源映射，导出不能把标量假装成单像素图。
- flow / app / split：去掉各算法共享的范围 / ROI 参数附加区；画布编辑明确指向空间模块实例；显示每条连接的实际形状。
