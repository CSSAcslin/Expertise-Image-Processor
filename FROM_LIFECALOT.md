# LifeCalor 算法参考模块清单（源码核对）

参考项目：`H:\Newera\programing\transient_data_processing`。整理日期：2026-10-12。源项目名称为 **LifeCalor / Carrier Lifetime Calculator**；本文件按用户指定名称保存为 `FROM_LIFECALOT.md`。

来源版本：`core/app_metadata.py` 标记 **1.1.1**，`updatinglog.md` 包含 **1.1.2 开发中、未发布**的自定义时空卷积。读取时 Git HEAD 为 `c467e52b22899f7fc339de620f8899a2f9066e27`。根目录旧原型及部分实验脚本为未跟踪文件，因此 HEAD 不能单独代表本次阅读的全部内容。

本清单采用 [当前模块清单](docs/MODULE_INVENTORY_CURRENT.md) 的“算法 / 公式、实际输入输出、边界、参数表”结构，增加来源、实现状态及迁移说明。共归纳 **40 组参考条目**：36 组主程序或辅助函数候选、4 组离线实验参考。这是按功能拆分及合并后的清单，**不等于旧注册表的算法数量，也不表示本程序已接入这些模块**。所有 `lc_*` 标识均为参考标识，尚未注册。

本次核对 README、更新日志、主程序调用、算法内核、导入处理、计算器、卷积工作台、绘图中的数值操作及离线脚本；通过静态读取核对实际公式，没有启动旧 PyQt5 程序或执行其算法。源项目测试文件及历史验收记录仅作为后续验证入口，不代表本次重新验收。缓存、主题、任务管理、导入 / 导出格式和绘图样式属于工作台基础设施，未作为独立算法计数。

## 轴、参数与执行归属

- 为便于对照本项目，旧 `H/W` 在下文统一写为 `Y/X`；`T` 为直接上游时间样本，`C` 为颜色分量。`T′` 表示变换后另有时间映射的序列。
- `F` 为空间或时间频率、`S` 为小波尺度、`R` 为线段位置、`K` 为事件、`V` 为位移向量分量；这些是**迁移所需的候选轴语义**，不能假定当前仅支持 T/Q/P/Y/X/C 的契约已支持它们。频率 F、小波 S、事件 K 均不能冒充 T 或 Q。
- “多个字段”表示参数、质量、状态等独立具名输出，不把它们编码为 RGB 通道；“标量”保持零维，不能伪装成 1×1 图像。
- 参数表区分**函数默认值、界面初始值、硬编码值、调用方提供**。它们来自读取到的代码，不是建议直接沿用的科学默认值。旧代码未固定的库默认行为不在本清单中冒充冻结契约。
- 每个新模块仍应只拥有算法参数及 execution。时间范围放在起点时间裁剪，空间选择由显式 ROI / 掩码 / 线段节点提供。寿命前后滤波、光流前处理、阈值归约等旧组合流程应拆开执行。
- STFT、CWT、Welch 和旧光流速度换算依赖均匀采样或 FPS；迁移后要检查真实时间轴。非均匀数据应明确拒绝，或先接用户可见的时间重采样模块，不能用平均 FPS 代替真实时间。
- 背景参考、掩码、拟合状态和多数据操作数是显式依赖；需要增加具名附加结果或多输入契约的候选，不能靠隐藏读取某个历史对象接入。
- 旧计算框架提供 CPU / 可选 CUDA、精度、空间分块、内存预算、取消和原子落盘。它们可借鉴为执行设施；Qt 对象、信号槽、全局设置及历史列表不应成为新数值模块的依赖。

## 快速索引与复用关系

| 编号 | 候选模块 / 功能 | 核心输入 → 输出 | 来源状态 | 与当前程序的关系 |
|---|---|---|---|---|
| 01 | RGB 亮度转换 | YXC → YX | 导入实现 | 需显式颜色转换；不同于 RGB 算术均值 |
| 02 | 全输入归一化族 | 数值数组 → 同形状浮点 | 函数实现 | 可扩展 curve / statistics，保留固定尺度 |
| 03 | 前段中位数背景 | TYX → YX 参考 | EM 管线 | 新参考统计，与现有全输入均值区分 |
| 04 | 相对背景信号 | TYX + 背景 → TYX | EM 管线 | 新归一化语义，不能直接当标准 g² 光强 |
| 05 | 坏帧检测 | TYX → 帧掩码 / 索引 | 主程序函数 | 新质量节点 |
| 06 | 坏帧修复 | TYX + 坏帧 → TYX | 主程序函数 | 新时间修复节点 |
| 07 | ROI 权重 / 频域掩码 | 数组 + YX 掩码 → 数组 | ROI 处理 | 可扩展 spatial_roi，含义不同于裁剪 |
| 08 | 空间三次插值 | TYX → TY′X′ | ROI / 单通道内部 | 新空间重采样节点 |
| 09 | 切片、换轴、去单维 | 任意轴 → 选定轴 | 切片器 / 计算器 | 裁剪已覆盖，换轴需契约扩展 |
| 10 | ROI / 全幅统计 | TYX → T | 多条旧实现 | 扩展已有 spatial_roi，核对分母 |
| 11 | 值分布 / 直方图 | YX 或 T → bins / counts | 主程序 / 绘图 | 新统计输出类型 |
| 12 | 有宽度线段剖面 | TYX → TR | 线段 ROI | 新空间采样类型 |
| 13 | 寿命拟合候选筛选 | T → 质量标记 | 寿命热图内核 | 显式质量节点候选 |
| 14 | 单指数寿命 | T 或 TYX → 参数 / YX 字段 | CPU + CUDA 路径 | 新时序拟合模块 |
| 15 | 双指数寿命 | T 或 TYX → 多参数 / YX 字段 | CPU + CUDA 路径 | 新多输出模块；旧边界需复核 |
| 16 | 一维高斯剖面拟合 | R 或 TR → σ² 等字段 | 扩散分析 | 新拟合模块 |
| 17 | 方差演化与扩散系数 | T 上的 σ² → D、拟合线 | 结果绘图内部 | 应移入处理内核 |
| 18 | 寿命到传热系数 | YX 寿命 → YX 系数 | 主程序实现 | 经验常数须实验确认 |
| 19 | 二维卷积核族 | YX / TYX → 同形状 | 寿命前后处理 | 复用 / 扩展 gaussian、box_blur |
| 20 | 自定义时空卷积 | YX / TYX → 同形状 | 1.1.2 开发中 | 新通用卷积与核编辑工具 |
| 21 | STFT 频带振幅 | T / TYX → T′ / T′YX | 有界管线 | 新时频模块 |
| 22 | STFT 频谱诊断 | T → FT′ 复系数 | 质量分析 | 与频带振幅分开 |
| 23 | CWT 频带振幅 | T / TYX → T / TYX | 有界管线 | 新时频模块 |
| 24 | CWT 频谱诊断 | T → ST 幅值 | 质量分析 | 旧名称“功率谱”不准确 |
| 25 | 时间均值投影 | TYX → YX | 累计振幅图 | 新时间轴归约；不是积分 |
| 26 | 二维高斯定位 / 振幅 | TYX → T 上的拟合字段 | 单通道分析 | 与空间重采样拆分 |
| 27 | 阈值控制的最大值信号 | TYX → T | 简单单通道 | 与已有 ROI 归约可复用 |
| 28 | 二维傅里叶变换 | YX / TYX → 复数频域 | 主程序 + 注册核 | 新复数频域契约 |
| 29 | 二维傅里叶逆变换 | 复数频域 → 空间数组 | 主程序函数 | 保留相位、明确所选字段 |
| 30 | Welch 功率谱 | T 信号 → F / PSD | 绘图内部 | 应移入处理内核 |
| 31 | 区间基线与峰参数 | T 信号 → 峰高 / 面积 / FWHM | 绘图内部 | 应移入处理内核 |
| 32 | 阈值事件幅值与时长 | T → K 事件表 | 辅助函数 | 固定时间间隔需移除 |
| 33 | 事件边沿指数时间 | T → K 时间常数 | 明确未启用 | 原型，需修正后验证 |
| 34 | CLAHE + 轻度模糊 | YX 灰度 → YX | 光流辅助函数 | 与显示、光强处理分开 |
| 35 | Farneback 光流 / 速度 | 帧对 → YXV 等字段 | 心肌运动分析 | 新向量场输出 |
| 36 | 安全数组表达式 | 多来源数组 → 推导形状 | 计算器实现 | 需显式多输入契约 |
| 37 | 分块 FFT 目标频率幅值 | T → T′ | 离线脚本 | 实验参考，未接主程序 |
| 38 | 合成脉冲 / 调制 / 噪声 | 参数 → T | 离线脚本 | 验证数据生成候选 |
| 39 | OU 有色噪声 | 参数 → T | 离线脚本 | 验证数据生成候选 |
| 40 | Hodgkin–Huxley 仿真 | 模型参数 / 刺激 → 状态序列 | 离线脚本 | 生物模型实验，需单独校验 |

## 01. RGB 亮度转换 `lc_luminance`

- 版本 / 状态：旧项目 1.1.1 导入内核；CPU。
- 来源：[core/importing/readers.py::_luminance](<H:/Newera/programing/transient_data_processing/core/importing/readers.py:135>)。
- 公式 / 算法：`I = 0.2126 R + 0.7152 G + 0.0722 B`；先转 float32，RGBA 只取前三分量。
- 实际输入输出：`…YXC → …YX`，float32 科学标量数组；原彩色显示来源可另存。
- 边界：这些权重不等同 `mean(R,G,B)`；代码未执行颜色空间线性化。迁移时需声明原始 RGB 编码、alpha 规则和转换的科学含义。
- 迁移：候选显式颜色转换模块；不得因切换彩色显示而重新定义处理输入。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| 亮度权重 | float[3] | 固定 0.2126 / 0.7152 / 0.0722 | 旧代码不可编辑 |
| alpha 处理 | enum | 忽略 | 新模块应明确 |

## 02. 全输入归一化族 `lc_normalize`

- 版本 / 状态：旧项目 1.1.1，SIF / 视频导入辅助实现；CPU。
- 来源：[core/ImportManager.py::normalize_data](<H:/Newera/programing/transient_data_processing/core/ImportManager.py:489>)；显示反相另见 [core/DataProcessor.py::process_data](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:117>)。
- 公式 / 算法：linear：`(A−min A)/(max A−min A)`；percentile：令 `a=Q_low(A), b=Q_high(A)`，输出 `(clip(A,a,b)−a)/(b−a)`；sigmoid：`1/(1+exp(−k(A−median A)))`；log：令 `L=ln(A+eps)`，再对 L 做 min–max。
- 实际输入输出：整段输入数组 → 同形状浮点数组；min、max、median、std、分位点按传入的**整个数组**计算，未逐帧分别拉伸。
- 边界：常量输入、零方差、非有限值及 `A+eps≤0` 缺少完整保护；函数遇到 `clahe` 或未知方法无有效分支。虽然说明列出 CLAHE，其导入实现已注释。`process_data` 另有正 / 负中心信号的显示反相规则，不能当此函数的处理定义。
- 迁移：固定统计 + 曲线变换可扩展已有 statistics / curve。sigmoid、log 和裁剪会改变强度涨落；不能作为 g² 的隐含预处理。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| method | str | 函数默认 linear | linear / percentile / sigmoid / log |
| low / high | float | 10 / 100 | 百分位参数；新实现应校验 0≤low<high≤100 |
| k | float / null | null 时 10/std(A) | sigmoid 固定全输入尺度 |
| eps | float | 1e−6 | log 的输入偏置 |
| clip_limit | float | 0.03 | 原函数保留参数但没有执行分支 |

## 03. 前段中位数背景参考 `lc_background_median`

- 版本 / 状态：旧项目 1.1.1 EM 预处理的第一阶段；当前执行管线只允许 CPU。
- 来源：[core/compute/algorithms/preprocess_pipeline.py::_safe_background](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/preprocess_pipeline.py:20>)、[core/compute/algorithms/preprocess_pipeline.py::run_preprocess_pipeline](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/preprocess_pipeline.py:31>)。
- 公式 / 算法：`B[y,x] = median(A[0:N,y,x])`，N 限制到 `[1,T]`。分空间块读取这些帧，计算精确中位数。
- 实际输入输出：`TYX → YX` 原始背景参考；旧管线同时保存 B，并生成供下一阶段使用的安全背景 B_safe。
- 边界：`abs(B)<1e−10` 的位置被替换为**该空间块**非近零背景的最小绝对值；整块均近零时替换为 1。替换值可能随空间分块划分变化，不能误称全图统一阈值策略；普通 median 不自动忽略 NaN。
- 迁移：独立参考统计模块，明确“前 N 个直接上游样本”，保存 B、有效掩码及来源；近零规则应与计算分块无关。它与现有全输入均值统计的定义不同。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| background_frames | int | 界面初值 300；旧外层函数默认 360 | 管线实际夹到 1…T |
| 近零阈值 | float | 固定 1e−10 | abs(B) 判定 |
| 近零替换 | enum | 块内最小非零绝对值，否则 1 | 需重定为明确可复现规则 |

## 04. 相对背景信号 `lc_background_relative`

- 版本 / 状态：旧项目 1.1.1 EM 预处理的第二阶段；CPU。
- 来源：[core/compute/algorithms/preprocess_pipeline.py::run_preprocess_pipeline](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/preprocess_pipeline.py:31>)。
- 公式 / 算法：`S(t,y,x) = (A(t,y,x)−B_safe(y,x))/B_safe(y,x)`；注意分子也用替换后的 B_safe，而非单独保存的原 B。
- 实际输入输出：实数 `TYX + YX 背景参考 → TYX` 相对背景变化；旧兼容输出 float32，可按计算策略指定精度。
- 边界：S 为有符号相对信号，不是绝对光强；分母使用安全背景不能混同实验暗场扣除。需保留被替换位置，避免把无信号区域解释成可靠变化。
- 迁移：与 03 分开；依赖明确参考输出。用于 EM-iSCAT 信号时有意义，输入标准 g² 时必须先确认物理定义，不能暗中把零均值残差作光强。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| background | reference YX | 前段中位数及近零处理 | 来源 / 范围必须记录 |
| output_dtype | dtype | 兼容 float32 | 随精度策略；实数 |

## 05. 坏帧检测 `lc_bad_frame_detect`

- 版本 / 状态：旧项目主程序辅助函数；CPU；菜单称“坏点处理”，本函数检测的是**整帧异常**。
- 来源：[core/DataProcessor.py::detect_bad_frames_auto](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:162>)、更新日志 1.4.0 的自动处理提示。
- 公式 / 算法：`m_t=mean_YX(A_t)`；`d_t=abs(m_(t+1)−m_t)`；`s=1.4826 median(abs(d−median d))`；`z_t=abs(d_t−median d)/s`。超过 threshold 时标记变化两端的帧并合并索引。
- 实际输入输出：`TYX → 坏帧索引列表`；建议新契约输出 T 布尔质量掩码及检测分数。
- 边界：未保护 MAD=0、短序列或 NaN；真实快速信号也可能触发。检测不能自动等同删除或修复。
- 迁移：显式质量节点，保留原帧映射；用户确认掩码后再选择修复策略。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| threshold | float | 函数默认 3.0 | 稳健差分分数阈值 |
| 帧统计量 | enum | 固定空间均值 | 可扩展 ROI 统计，但不能隐藏选择 |
| MAD 尺度系数 | float | 固定 1.4826 | 需记录 |

## 06. 邻帧平均修复 `lc_bad_frame_repair`

- 版本 / 状态：旧项目主程序辅助函数；CPU。
- 来源：[core/DataProcessor.py::fix_bad_frames](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:190>)。
- 公式 / 算法：对坏帧 t，在 `[t−n,t+n]` 内剔除所有坏帧，用剩余原始帧均值替换 `A_t`；不是时间距离加权插值。
- 实际输入输出：`TYX + 坏帧索引 → TYX`；旧函数复制 data_origin 后更新原 Data 对象。
- 边界：无可用邻帧时打印提示并保留原值；整数数组回填均值会量化；全段复制和原对象更新不适合直接迁入新体系。
- 迁移：生成新结果，显式记录每个修复样本的贡献帧、插补标记及 dtype；原数据不可变。其后关联计算必须知道插补对涨落的影响。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| bad_frames | int[] | 调用方提供 | 0 基索引；需边界校验 |
| n_frames | int | 2 | 前后邻域半径 |
| 无参考策略 | enum | 保留原值 + 提示 | 新实现应显式选择 |

## 07. ROI 权重与频域掩码 `lc_mask_weight`

- 版本 / 状态：旧项目 ROI 通用处理；CPU。
- 来源：[core/DataManager.py::_roi_processed_impl](<H:/Newera/programing/transient_data_processing/core/DataManager.py:692>)、[core/MainWindow.py::draw_result](<H:/Newera/programing/transient_data_processing/core/MainWindow.py:2967>)。
- 公式 / 算法：`B = where(M, factor×A, A)`；可先按 M 的包围盒裁剪，再对盒内选中像素乘权重。**不是把选中像素设为 factor 这个常数**。
- 实际输入输出：`YX / TYX + YX 布尔掩码 → 同形状或包围盒裁剪结果`。在保留完整复数频域且 factor=0 时，可表达频率阻断掩码。
- 边界：非矩形掩码的包围盒内未选中值保持原值；旧末尾 squeeze 会删除单维；空掩码缺少完整保护。频域裁剪改变频率网格，不能等同保持尺寸的频域滤波。
- 迁移：可扩展 ROI 模块的“权重 / 掩码”操作；与空间裁剪、归约、后续插值分别配置，频域明确空间频率轴及共轭对称要求。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| mask | bool YX | 画布传入 | 可取反；须与输入坐标系匹配 |
| multiply_factor | float | 调用方提供 | ROI 内乘数 |
| is_crop | bool | false | 按掩码包围盒裁剪 |
| is_zoom / zoom_factor | bool / float | false / 1.0 | 建议移到独立空间重采样模块 |

## 08. 空间三次插值 `lc_spatial_resample`

- 版本 / 状态：旧 ROI 处理和单通道分析内部步骤；CPU。
- 来源：[core/DataManager.py::_roi_processed_impl](<H:/Newera/programing/transient_data_processing/core/DataManager.py:692>)、[core/DataProcessor.py::twoD_gaussian_fit](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1399>)、[core/DataProcessor.py::simple_single_channel](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1515>)。
- 公式 / 算法：逐帧 `scipy.ndimage.zoom(frame, factor, order=3)`，执行三次样条插值；这是数据重采样，不是显示缩放。
- 实际输入输出：`YX / TYX → Y′X′ / TY′X′`；旧实现未统一冻结边界、预滤波与量化约定。
- 边界：插值可能过冲，整数输入需核对输出量化；旧 metadata 没有完整同步新像素尺度，二维高斯中心以放大后的像素坐标计算。
- 迁移：独立模块，记录插值核、输出尺寸、原点与实际尺度映射；不得借用显示 zoom 隐式执行。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| factor | float | ROI 函数默认 1.0；单通道 zm 默认 2 | 单通道 zm≤1 跳过 |
| order | int | 固定 3 | 三次样条 |
| boundary / prefilter | enum / bool | 库默认，旧调用未固定 | 迁移时冻结 |

## 09. 多轴切片、换轴与去单维 `lc_slice_axes`

- 版本 / 状态：旧切片器及安全计算器实现；CPU。
- 来源：[core/SpatialExtractor.py::accept_extraction](<H:/Newera/programing/transient_data_processing/core/SpatialExtractor.py:461>)、[core/calculator/engine.py::slice_tuple](<H:/Newera/programing/transient_data_processing/core/calculator/engine.py:118>)、[core/calculator/engine.py::_validate_call](<H:/Newera/programing/transient_data_processing/core/calculator/engine.py:256>)。
- 公式 / 算法：`B = A[slices]` 后按指定 axes 转置，可选择 squeeze；切片器上下限为闭区间，实际构造 `slice(low, high+1)`。
- 实际输入输出：任意数值轴 → 明确选定的新尺寸及轴排列；计算器另支持整数和冒号索引。
- 边界：切片器修改时间轴有独立勾选项，可能与数据裁剪不同步；squeeze 会改变轴秩。它们不能作为不可见的便利操作。
- 迁移：空间 / 时间裁剪优先复用现有模块；显式换轴仅在同步更新坐标、时间及物理轴后扩展。不再把独立范围附着到每个算法。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| 各轴区间 / index | slice[] | 用户指定 | 闭区间切片器与半开索引需区分 |
| axes | int[] | 用户指定 | 必须为不重复完整排列 |
| squeeze | bool | 界面选项 | 仅删除已明确指定的单维 |
| 时间同步 | bool | 界面选项 | 迁移后必须与数据一致 |

## 10. ROI 与全幅统计 `lc_roi_reduce`

- 版本 / 状态：旧 Anchor、区域寿命、全细胞信号及分块均值多条实现；CPU。
- 来源：[core/DataProcessor.py::get_fast_selection](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:313>)、[core/DataProcessor.py::get_mean_time_trace](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:81>)、[core/LifetimeCalculator.py::analyze_region](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:84>)、[core/LifetimeCalculator.py::easy_process](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:786>)。
- 公式 / 算法：在明确有效集合 Ω 内计算 mean / max / min / median / Q0.75 / std / sum / var；std、var 使用总体统计约定。全幅分块均值为 `Σ_YX A/(Y×X)`。
- 实际输入输出：`TYX + YX 区域 → T`；旧 Anchor 再拼成 `[time,value]` 二列表，内核迁移应保留独立轴 / 元数据。
- 边界：Anchor 以输入 dtype 分配结果，整数均值 / 方差可能被截断，空区 NaN 不能安全写入整数。`easy_process` 先将 ROI 外置零，再除以**全幅像素数**，与区域寿命的真正 ROI 均值不同；不能以同一个“均值”名称照搬。
- 迁移：复用已有 spatial_roi 的 mean / sum / quantile，扩展 max / min / median / std / var；显式定义有限像素集合、两路共同掩码与分母，优先消除重复实现。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| method | enum | 调用方提供 | mean / max / min / median / quantile_075 / std / sum / var |
| mask | bool YX | 画布 / 区域传入 | 全幅时所有位置 |
| quantile | float | Anchor 固定 0.75 | 新模块已有可调分位值 |
| ddof | int | NumPy 默认 0 | 显式记录 |
| row_step | int | get_mean_time_trace 默认 32 | 执行分块参数，不改变统计定义 |

## 11. 值分布与直方图 `lc_histogram`

- 版本 / 状态：旧 1.0.5 后的单帧 ROI 内核及绘图区信号直方图；CPU。
- 来源：[core/DataProcessor.py::value_distribution_from_frame](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:366>)、[core/PlotGraphWidget.py::_apply_histogram](<H:/Newera/programing/transient_data_processing/core/PlotGraphWidget.py:508>)。
- 公式 / 算法：`counts,edges = histogram(values,bins,range)`；`centers=(edges[:-1]+edges[1:])/2`。输出频数，不是概率密度或累计分布。
- 实际输入输出：`YX + ROI → bin_centers/counts`，或 T 信号 → 同类统计；单帧版本还输出 count、min/max、mean/std、median、dtype 和复数取幅标记。
- 边界：单帧版本剔除 NaN/Inf，复数取 abs，空区域或没有有限值报错；绘图版直接 histogram(y)，未采用同样的无效值规则。`bins='auto'` 由 NumPy 决定，不保证只使用某一种分箱算法。
- 迁移：合并为明确统计模块，声明非时间的分箱轴、边缘区间及完整计数；可验证图像原值、寿命 / 系数分布及 ROI 信号分布。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| bins | int / str | 单帧函数默认 auto；界面可指定 | >0 或受支持策略 |
| value_range | float[2] / null | null | 有限且下界<上界 |
| complex_mode | enum | 单帧固定 abs | 迁移时显式选择 |
| frame_index | int | 工作台当前帧 | 选择由上游节点 / 检查操作提供 |

## 12. 有宽度线段空间剖面 `lc_line_profile`

- 版本 / 状态：旧矢量线 ROI，扩散分析的输入来源；CPU。
- 来源：[core/ROIdrawDialog.py::getPixelValues](<H:/Newera/programing/transient_data_processing/core/ROIdrawDialog.py:590>)；主画布另有 [core/ImageDisplayWindow.py::getPixelValues](<H:/Newera/programing/transient_data_processing/core/ImageDisplayWindow.py:2543>) 的同名实现。
- 公式 / 算法：沿线段 `p(r)=p0+r u` 采样，沿法向宽度逐点 round 到最近像素，将各位置的时间序列累计并平均；记录 r 为线段像素距离。
- 实际输入输出：`TYX → T×R×2` 的旧封装，末轴分别存距离和值；建议新契约为 TR 数据 + 独立 R 坐标。
- 边界：ROI 对话框版本的越界法向采样跳过，却仍除以预定 `int(width)+1`，边缘会偏低；取整会重复采同一像素。temporal_scale 参数未参与该函数计算；空间采样步长与物理像素尺寸不是同一概念。
- 迁移：独立空间采样节点，统一两个旧实现，记录端点、宽度、插值、有效计数和物理距离；保留真实 T 映射供高斯 / 扩散分析使用。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| endpoints | float[4] | 线段 ROI 指定 | 输入像素坐标 |
| width | float | 线段实例设置 | 法向带宽，需边界策略 |
| spatial_scale | float | 函数默认 1.0 | 旧函数中为像素坐标采样步长 |
| temporal_scale | float | 默认 1.0 | ROI 对话框函数实际未用 |
| 插值 | enum | 固定最近像素 round | 迁移时显式冻结 |

## 13. 寿命拟合候选筛选 `lc_decay_gate`

- 版本 / 状态：旧项目 1.1.1 寿命热图内核内置；区域曲线未执行同一前置筛选。
- 来源：[core/compute/algorithms/lifetime.py::has_correlated_window](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime.py:260>)、[core/compute/algorithms/lifetime_pipeline.py::_fit_lifetime_block](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime_pipeline.py:82>)。
- 公式 / 算法：取 `w=min(max_window,floor(T/2))`，滑动计算时间与信号的 Pearson 相关系数；存在任一 `abs(corr(t,I))≥threshold` 的窗口则通过。
- 实际输入输出：`T 信号 + T 时间 → bool`；热图逐像素执行，未通过时记录 NO_CORRELATED_WINDOW。
- 边界：这不是标准 g² 或两路信号关联，而是单路趋势筛选；正相关、负相关均可能通过。w<2、窗口非有限或恒定时不能通过，时间常量 / 异常也需检查。
- 迁移：可作为独立质量模块或拟合模块内明确可开关的判据，输出质量掩码；不能在所有曲线和图像路径默默套用。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| threshold | float | 0.8 | abs Pearson 系数阈值 |
| max_window | int | 10 | 实际窗口不超过 floor(T/2) |

## 14. 单指数寿命拟合 `lc_lifetime_single`

- 版本 / 状态：旧项目 1.1.1 冻结 CPU 参考；热图有批量 CUDA 路径，区域曲线使用 CPU。
- 来源：[core/compute/algorithms/lifetime.py::fit_lifetime](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime.py:127>)、[core/compute/algorithms/lifetime_pipeline.py::_fit_lifetime_block](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime_pipeline.py:82>)、[core/LifetimeCalculator.py::analyze_region](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:84>)、[core/compute/backends/lifetime_cuda.py::_solve_batch](<H:/Newera/programing/transient_data_processing/core/compute/backends/lifetime_cuda.py:148>)。
- 公式 / 算法：`I(t)=A exp(−t/τ)+C`；CPU 用 float64、受界 `curve_fit(method='trf')`，`ftol=xtol=gtol=1e−8`。质量 `R²=1−Σ(I−fit)²/Σ(I−mean I)²`。初值 A 为幅度跨度、τ 为拟合时间跨度 / 5、C 为最小值。
- 实际输入输出：`T + 显式 time_points → A,τ,C,R²,status,fit_curve`；旧时间映射须另核对，旧区域入口先做 ROI 均值。`TYX → YX` 寿命、R²、status 字段；单指数热图未单独导出 A/C 图。
- 边界：central negative / positive 数据先 abs；可从全输入或最大峰之后开始，后者重置时间零点。峰位置是**样本索引**而非峰值。τ 范围和 R² 是拟合后筛选，不是全部求解边界；τ 不通过 / R² 不通过可能以 0 表示，其他失败为 NaN 或未求值 0，必须结合 status。时间单位由调用方提供转换，不能把寿命 τ 与二阶关联延迟 τ 当同一参数。
- 迁移：区域归约与逐像素拟合共用同一内核，ROI 在上游；前后卷积独立排列。应统一输出参数、质量、状态和有效掩码，避免真实 τ=0 与失败混淆。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| from_start_cal | bool | false | false 从峰后拟合；true 从输入起点 |
| peak_range | int[2] | 界面 0…50 | 峰索引闭区间，按直接上游计数 |
| tau_range | float[2] | 界面 1e−3…1e3；类初始上限 1e2 | 筛选用严格开区间；以拟合时间单位计 |
| r_squared_min | float | 0.4 | 必须严格大于该值 |
| 求解边界 | bounds | A、C 无界；τ≥0 | 科学使用宜校验严格正 τ |
| pre_cov / post_cov | enum | 可无 | 原组合步骤；迁移为独立 19 / 20 节点 |

## 15. 双指数寿命拟合 `lc_lifetime_double`

- 版本 / 状态：旧项目 1.1.1 CPU 参考和批量 CUDA 路径；旧 UI 提示仍残留“双指数有问题不要用”，与较新的 README / 更新记录不一致。
- 来源：[core/compute/algorithms/lifetime.py::double_exponential](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime.py:50>)、[core/compute/algorithms/lifetime.py::fit_lifetime](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime.py:127>)、[core/compute/algorithms/lifetime_pipeline.py::_fit_lifetime_block](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime_pipeline.py:82>)。
- 公式 / 算法：`I(t)=A1 exp(−t/τ1)+A2 exp(−t/τ2)+C`；求解策略与 14 相同；CPU 初值在受界下移入可行区内部。
- 实际输入输出：`T → A1,τ1,A2,τ2,C,R²,status` 及拟合曲线；`TYX →` 两个寿命、两个振幅、基线、R² 和 status 的 YX 具名字段；默认主字段 τ1，不把两个寿命平均。
- 边界：旧冻结下界为 **A1≥0、τ1≥0、A2≥10、τ2≥10、C 无界**，后两项是带实验尺度含义的限制。τ 筛选只要求 τ1 或 τ2 **至少一个**落入范围；未按快 / 慢分量排序。共享入口只预检查至少 3 点，并不证明 5 参数双指数可辨识。热图对有限 fitted parameters 直接写入字段，即使 status 为 R² / τ 拒绝，也可能保留原拟合值；仅看数值图不能判断有效性。
- 迁移：参数边界、分量排序、可辨识性、初值、拟合协方差和有效输出规则必须重新确认，CPU / GPU 分别验收；原型可参考，不能直接视作通用稳健双指数求解器。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| 起点、峰范围、τ 范围、R² | 同 14 | 同 14 | 共享拟合配置 |
| lower_bounds | float[5] | 固定 [0,0,10,10,−∞] | 次序为 A1,τ1,A2,τ2,C |
| upper_bounds | float[5] | 全部 +∞ | 旧代码固定 |
| 分量排序 | enum | 无 | 新实现须明确 fast / slow 或保持参数顺序 |
| 输出有效性 | policy | 依 status 解释 | 新实现应同时输出原拟合与有效结果 |

## 16. 一维高斯剖面拟合 `lc_gaussian_profile_fit`

- 版本 / 状态：旧载流子扩散分析内部；CPU。
- 来源：[core/LifetimeCalculator.py::gaussian_func](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:248>)、[core/LifetimeCalculator.py::gaussian_fit](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:253>)、[core/LifetimeCalculator.py::diffusion_calculation](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:656>)。
- 公式 / 算法：`I(r)=a exp(−(r−μ)²/(2σ²))`；无基线项，调用 curve_fit；初值 `a=max(abs(I)), μ=median(r), σ=(max r−min r)/4`，保存 σ²。
- 实际输入输出：`R 距离 + R 强度 → a,μ,σ、covariance、拟合曲线`；逐时刻可得到 TR 拟合曲线和 T 上的 σ²。
- 边界：旧求解没有正 σ 边界，平方后失去符号；有背景、负峰、多峰和异常值时模型未必合适。旧循环拟合失败仍保留预分配零位，而成功曲线列表变短，存在字段长度不一致风险。
- 迁移：线段提取、物理距离标定与高斯拟合独立；输出逐样本 status，不用 0 冒充拟合失败，是否增加基线应由实验需求决定。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| position_scale | float | 调用方 space_unit | 旧流程将剖面像素距离相乘 |
| 初值 | float[3] | 自动估计 | 未暴露可编辑初值 |
| baseline | enum | 固定无基线 | 增加基线是新模型，需另记版本 |

## 17. 方差演化与扩散系数 `lc_diffusion_coefficient`

- 版本 / 状态：旧方差生成在处理线程，最终 D 的回归在结果绘图内执行；CPU。
- 来源：[core/LifetimeCalculator.py::diffusion_calculation](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:656>)、[core/ResultDisplayWidget.py::plot_variance_evolution](<H:/Newera/programing/transient_data_processing/core/ResultDisplayWidget.py:496>)。
- 公式 / 算法：对 16 得到的方差做 `σ²(t)=b+k t` 线性最小二乘，`D=k/2`；对应一维高斯扩散宽度演化假设。
- 实际输入输出：`T 时间 + T 方差 → D、斜率、截距、拟合曲线`；D 单位为长度²/时间。
- 边界：旧生成时间使用 `frame_idx×time_unit`，不是逐帧真实时间戳；绘图计算了去零副本 sigma_trim，却对未去零的全部方差 polyfit。滤除结果与导出时间数组也可能不等长。没有 R²、置信区间、异常值或负斜率保护；不能机械推广为任意二维 / 三维扩散系数公式。
- 迁移：回归必须成为显式处理节点，接真实 T 及上游 status；用户选择拟合区间要用时间裁剪或质量节点，保存物理模型及单位。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| 拟合阶数 | int | 固定 1 | 线性方差演化 |
| 扩散分母 | float | 固定 2 | 一维剖面模型 |
| time_unit / space_unit | float | 调用方提供 | 迁移为真实坐标和单位 |
| 有效样本 | mask | 旧回归实际未剔除失败零值 | 新实现明确 |

## 18. 寿命到传热系数 `lc_heat_transfer`

- 版本 / 状态：旧主程序实现；可接受已有寿命图，否则内部先拟合；派生计算 CPU。
- 来源：[core/LifetimeCalculator.py::heat_transfer_calculation](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:714>)、更新日志 1.10.12。
- 公式 / 算法：`H = where(τ≥0.1, 42.72/τ, 0)`，可再执行后卷积。
- 实际输入输出：`YX 寿命图 → YX 传热系数图`；旧方法可能接受 TYX 并隐藏执行寿命拟合。
- 边界：42.72 是硬编码经验常数，代码未给出完整量纲推导；0.1 同样依赖寿命单位。NaN 比较为 false 后会变 0，失去无效标记；双指数默认仅取所选主寿命字段，不能自动赋予唯一传热含义。
- 迁移：独立派生模块，仅接明确选定的寿命字段和质量；常数、阈值、单位与实验标定先确认，后滤波独立，禁止隐藏重拟合。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| coefficient | float | 固定 42.72 | 实验经验值，需量纲说明 |
| tau_min | float | 固定 0.1 | 按寿命时间单位 |
| 非有效输出 | policy | 旧为 0 | 新建议明确 NaN / status |
| lifetime_field | enum | 原默认主图 | 双指数时必须指定 |

## 19. 二维卷积核族 `lc_spatial_kernel`

- 版本 / 状态：旧寿命前 / 后处理；CPU。
- 来源：[core/LifetimeCalculator.py::apply_custom_kernel](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:131>)、[core/compute/algorithms/lifetime_pipeline.py::spatial_kernel](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime_pipeline.py:32>)。
- 公式 / 算法：`B[y,x]=Σ K[dy,dx] A[y−dy,x−dx]`。核边长 `s=2×half_size−1`；平均核 `1/s²`；高斯核 `exp(−r²/(2σ²))/ΣK`，`σ=max(0.3((s−1)/2−1)+0.8,0.1)`。
- 实际输入输出：`YX → YX`；热图前处理对 TYX 逐帧，后处理作用于寿命 / 系数图。float / 整数输出行为依实际输入数组及 ndimage 调用，旧直接前卷积块会写回块 dtype。
- 边界：旧边界 mode='mirror'。smooth 在 3×3 时中心 0.2、其余 0.1；更大尺寸为中心 2 到边缘线性递减后归一。直接旧函数卷积失败记录日志后返回原输入，迁移必须改为显式失败。
- 迁移：均值 / 高斯优先复用本项目 box_blur / gaussian，但要核对边界、核尺寸和无效值规则；其余作为通用核预设。任何后滤波结果均应标明是平滑后的估计图。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| kernel_type | enum | smooth | smooth / gaussian / sharpen / edge / laplacian / average；其他直接原样返回 |
| half_size | int | 2 | 实际边长 2h−1；管线最小夹到 1 |
| boundary | enum | 固定 mirror | 与新现有 gaussian 的边界不同 |

| 3×3 核名 | 旧具体系数 / 定义 | 其他尺寸说明 |
|---|---|---|
| smooth | 中心 0.2，其余 0.1 | 按到中心的距离线性减小，再归一 |
| average | 全部 1/9 | 全部 1/s² |
| sharpen | 中心 5、四邻 −1、角 0 | 中心 2，其余 −1/(s²−1) |
| edge | 中心 8，其余 −1 | 中心 −(s²−1)、其余 +1；符号发生变化 |
| laplacian | 中心 −4、四邻 +1、角 0 | 仍为中心及紧邻四点，外围为 0 |

## 20. 自定义时空卷积 `lc_spatiotemporal_convolution`

- 版本 / 状态：**1.1.2 开发中，未发布**；CPU 与隔离 CUDA 路径。源更新日志说明目标 GPU、真实大数据及 OpenGL/DPI 验收尚未完成。
- 来源：[core/convolution/model.py::for_execution](<H:/Newera/programing/transient_data_processing/core/convolution/model.py:46>)、[core/convolution/presets.py::generate](<H:/Newera/programing/transient_data_processing/core/convolution/presets.py:91>)、[core/compute/algorithms/convolution_pipeline.py::run_spatiotemporal_convolution](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/convolution_pipeline.py:96>)、[core/convolution/execution.py::preview_region](<H:/Newera/programing/transient_data_processing/core/convolution/execution.py:129>)。
- 公式 / 算法：`B[t,y,x]=Σ K[dt,dy,dx] A[t−dt,y−dy,x−dx]`；**正 dt 读取过去样本**。核可为稠密权重或有限稀疏坐标，按全局边界读取 halo、分 T/Y/X 块计算；预览与正式运行共享核及边界定义。
- 实际输入输出：`YX / TYX → 同形状浮点或复数`。二维输入禁止非零时间方向权重；零核输出零，单位核输出原值。兼容精度保留已有 float/complex，整数转 float64。
- 边界：工作台只允许 reflect / constant；旧二维族用 mirror，二者不能混写。非对称和偶数核须保存锚点 / origin；大跨度稀疏核仅 CPU，最多 512 个非零 tap，不能宣称 GPU 完整覆盖。
- 迁移：通用卷积模块 + 核编辑工具归属实例；先提供数值表 / 公式与预设，不必迁移整套 Qt/OpenGL 工作台。时间 tap 按直接上游样本计数，记录未来样本依赖和边缘语义。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| kernel cells | (dt,dy,dx,weight)[] | 中心 (0,0,0)=1 | 有限实权重；最多 32768 格，坐标绝对值≤4096 |
| boundary / cval | enum / float | 预设保存默认 reflect / 执行函数默认 0 | reflect / constant |
| origin / anchor | int[] | 显式零坐标为锚点 | 可含偶数核，不能假定几何中心 |
| radius | int[3] | 生成器默认 (0,1,1) | 每轴 0…32 |
| sigma | float[3] | (1,1,1) | 每轴有限正数 |
| causal | bool | false | 时间权重仅过去；拒绝冲突预设 |
| normalize | enum | 均值 / 高斯生成时归一 | 权重和或绝对权重和；近零和报错 |
| formula / parameters | str / dict | '1' / 空 | 有限 AST 白名单；≤16 个标量参数 |

预设包含 identity、mean、gaussian、difference、laplacian、sharpen、formula。差分预设的名称与通常导数方向不完全对应：旧 forward 为 `K0=1,K(+axis)=−1`；backward 为其相反数；central 为 `K(−axis)=−0.5,K(+axis)=+0.5`。应按上述卷积方向核对符号，并在物理导数需要时显式除以样本间距或间距²。核公式只允许坐标 dt/dh/dw、有限数字、算术及 exp/sqrt/sin/cos/abs，不执行任意 Python。

## 21. STFT 频带振幅 `lc_stft_amplitude`

- 版本 / 状态：旧项目 1.1.1 有界 CPU / CUDA 管线；核心支持单条或批量时间信号，旧应用入口按 TYX 像素序列处理。
- 来源：[core/compute/algorithms/stft.py::stft_frequency_trace](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/stft.py:45>)、[core/compute/algorithms/stft.py::select_frequency_indices](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/stft.py:7>)、[core/compute/backends/cpu.py::stft_block_cpu](<H:/Newera/programing/transient_data_processing/core/compute/backends/cpu.py:21>)、[core/DataProcessor.py::get_window](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:924>)。
- 公式 / 算法：`Z(k,m)=STFT_w(s)`，使用 scaling='spectrum'；输出 `a_m=mean_(k∈band)(c_k |Z(k,m)|)`。实信号单边谱除 DC 和偶数 nfft 的 Nyquist 外 c=2，其余 c=1；复信号双边谱全部 c=1。窗为对称窗 fftbins=false，Gaussian std=window_size/6。
- 实际输入输出：核心 `T / batch×T → T′ / batch×T′`；应用 `TYX → T′YX`，并保存频率、所选索引及 STFT 返回的时间。兼容输出 float32；CPU 核心已先 cast float32，外层选更高输出精度不恢复已丢失位数。
- 边界：band 为 f0±bandwidth/2，若区间无 bin 或宽度为 0 则取最近频率；平均的是幅值，不是功率或频带积分。旧调用未显式传 boundary / padded，需冻结库行为；时间边缘可能包含填充窗口。
- 迁移：单信号 / 逐像素共用模块，输入必须有均匀真实时间或显式重采样；窗宽和 hop 按直接上游样本。保留真实支持区间及频率轴，不把零填充当原采集样本。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| fps | float | 界面 EM_fps=360 | >0；迁移应由真实时间校验 |
| target_freq | float / int[] | 界面 30 Hz；核心也接受索引列表 | 与采样及单 / 双边频谱一致 |
| scale_range | float | 界面 1 Hz | 完整频带宽度，不是半宽 |
| window_size | int | 界面 128 | >0，且与输入长度兼容 |
| noverlap | int | 界面 120 | hop=window_size−noverlap>0 |
| nfft | int | 界面 360；应用入口取 max(custom_nfft,window_size) | 应≥有效 window_size |
| window | enum / array | 界面 hann | hann / hamming / gaussian / boxcar / blackman / blackmanharris；函数另支持 general_gaussian |

## 22. STFT 频谱诊断 `lc_stft_spectrum`

- 版本 / 状态：旧质量分析 CPU 路径；数值计算与绘图分在两个类。
- 来源：[core/DataProcessor.py::quality_stft](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:648>)、[core/ResultDisplayWidget.py::quality_avg](<H:/Newera/programing/transient_data_processing/core/ResultDisplayWidget.py:533>)。
- 公式 / 算法：先对全幅求空间均值 s(t)，再 `Z=STFT(s, scaling='psd', return_onesided=true)`；显示 `10 log10(abs(Z)²+1e−12)`。
- 实际输入输出：旧应用 `TYX → T 均值 → F×T′ 复系数`，显示另生成 dB 值。target_freq / scale_range 用于索引及图中标注，**原输出仍是全频谱**。
- 边界：不同于 21 的 spectrum 缩放和单边振幅修正；“质量”指诊断频谱，未计算 SNR、置信度或自动通过 / 拒绝分数。复数输入与硬编码单边选项需单独核对。
- 迁移：空间均值用上游 ROI 模块；STFT 频谱保留复系数 / 功率 / dB 的不同字段与单位，显示转换不改变处理输入。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| fps / window_size / noverlap / nfft / window | 同 21 | 同界面配置 | 同 21 |
| target_freq / scale_range | float | 30 / 1 Hz | 标注目标频带，非返回裁剪 |
| dB epsilon | float | 固定 1e−12 | 显示功率取 log 的保护值 |

## 23. CWT 频带振幅 `lc_cwt_amplitude`

- 版本 / 状态：旧项目 1.1.1 CPU / CUDA 管线；CPU 按尺度累计，减少完整系数体内存。
- 来源：[core/compute/algorithms/cwt.py::cwt_axes](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/cwt.py:197>)、[core/compute/algorithms/cwt.py::build_cwt_kernel_bank](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/cwt.py:48>)、[core/compute/algorithms/cwt.py::reduce_cwt_traces_from_kernels](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/cwt.py:141>)、[core/compute/backends/cpu.py::cwt_block_cpu](<H:/Newera/programing/transient_data_processing/core/compute/backends/cpu.py:43>)。
- 公式 / 算法：频率线性取点 `f_j=linspace(f0−B/2,f0+B/2,N)`，`scale_j=frequency2scale(wavelet,f_j/fps)`；输出 `a_t=mean_j(2 |CWT(scale_j,t)|/sqrt(scale_j))`。核参考为积分小波卷积后差分并居中截取，precision=10。
- 实际输入输出：核心 `T / batch×T → 同长度实数振幅`；应用 `TYX → TYX`，保存 scales、frequencies、wavelet。兼容输出 float32，可选择其他计算 / 输出精度。
- 边界：所有目标频率必须>0；宽带是有限点的幅值平均，不是频带能量积分。旧归约对实 / 复信号均乘 2，这是代码约定，迁移不能默认是通用物理振幅校准；核边缘与尺度影响有效支持区间。
- 迁移：保留小波、尺度、归约和边缘策略版本；旧 Julia 选择仅为 UI 选项，没有本次找到的可核验实现，不能作为已支持后端。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| fps / target_freq | float | 界面 360 / 30 Hz | >0 |
| scale_range | float | 界面 10 Hz | 整个目标频段须>0 |
| total_scales | int | 界面 256 | 信号内核≥1 |
| wavelet | enum | 界面 cmor3-3；部分函数默认 morl | cmor1-1.0 / cmor1.5-1.0 / cmor3-3 / cmor8-3 / cgau8 / mexh / morl |
| wavelet_precision | int | 10 | 构核离散精度；结果记录 PyWavelets 版本 |

## 24. CWT 频谱诊断 `lc_cwt_spectrum`

- 版本 / 状态：旧质量分析 CPU / CUDA 路径。
- 来源：[core/DataProcessor.py::quality_cwt](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:944>)、[core/compute/algorithms/cwt.py::cwt_quality_spectrum](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/cwt.py:163>)。
- 公式 / 算法：先全幅空间均值；`c=2×central_frequency(wavelet)×N`，`scales=c/arange(N,1,−1)`；输出 `abs(CWT)`，频率为 `scale2frequency(scales)×fps`。
- 实际输入输出：旧应用 `TYX → T → S×T` 实数系数幅值；S=N−1，而不是 N；保存频率及构核信息。
- 边界：旧按钮称“功率谱”，实际未平方、未作 23 的 `2/sqrt(scale)` 归约。target_freq / scale_range 不参与该尺度表生成，只保存 / 标注；不能宣称这是所选窄频带的同一输出。
- 迁移：上游 ROI 均值 + 独立全尺度诊断模块；幅值、功率及归约明确不同类型，使用真实 T 映射和尺度对应频率。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| totalscales | int | 界面 256 | 质量入口≥2；输出 N−1 行 |
| fps / wavelet | float / enum | 界面 360 / cmor3-3 | 与 23 同一已核验小波集合 |
| target_freq / scale_range | float | 界面 30 / 10 Hz | 旧质量方法不据此选尺度 |
| output_dtype | dtype | 兼容 float32 | 按精度策略 |

## 25. 时间均值投影 `lc_time_projection`

- 版本 / 状态：旧“累计时间振幅图”入口；CPU。
- 来源：[core/DataProcessor.py::accumulate_amplitude](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1370>)。
- 公式 / 算法：`B[y,x]=mean_t(A[t,y,x])`。实际为时间均值，**没有累加、取绝对值或乘时间间隔**。
- 实际输入输出：`TYX → YX`；有符号输入会相消；旧对象仍继承原 time_point，不能据此把输出视为时间视频。
- 边界：普通 mean 遇 NaN 会传播。时间积分 `Σ A_t Δt`、RMS、绝对振幅均值属于不同算法。
- 迁移：显式删除 T 的时间归约模块；已有 time_bin 能生成合帧样本，但“单个合帧样本”与“移除 T 的投影图”应在契约中区分。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| reduction | enum | 固定 mean | 其他统计应声明为扩展 |
| 时间范围 | upstream | 全部传入时间 | 使用上游时间裁剪，不新增任务级范围 |

## 26. 二维高斯拟合振幅与定位 `lc_gaussian_2d_fit`

- 版本 / 状态：旧单通道分析逐帧 CPU 实现。
- 来源：[core/DataProcessor.py::D2GaussFunction](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1382>)、[core/DataProcessor.py::twoD_gaussian_fit](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1399>)。
- 公式 / 算法：`G(x,y)=A exp(−((x−x0)²/(2σx²)+(y−y0)²/(2σy²)))+b`；旧流程可先三次插值，逐帧 curve_fit(maxfev=5000)。初始中心取最大像素，初始宽度均为 1，基线为帧均值。
- 实际输入输出：`TYX → T 振幅主结果`，另存 mean_signal / max_signal。旧代码计算 x/y 中心数组但没有放入结果，也未保存 σx/σy/b 图表；建议新模块输出全部参数和 status。
- 边界：启用阈值且整帧无值>thr 时用帧均值替代 A；RuntimeError 也用均值替代，未输出失败状态。固定 A≤100、b≥0 等边界具有实验尺度假设，放大后的中心尚未还原原坐标。
- 迁移：空间裁剪 / 重采样在上游，拟合只接明确图像；不要用“拟合振幅”字段混放回退均值。是否旋转高斯、允许负峰及背景范围须单独定义。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| zm | int | 函数 / 界面默认 2 | >1 才插值；迁移为 08 |
| thr_known / thr | bool / float | false / 2.5 | 先检查是否任一像素>thr |
| lower_bounds | float[6] | [0,0,0.1,0,0.1,0] | 次序 A,x0,σx,y0,σy,b |
| upper_bounds | float[6] | [100,Wz,(Wz/2)²,Hz,(Hz/2)²,max(Z)] | 旧宽度上界使用尺寸平方，需复核 |
| maxfev | int | 5000 | CPU 求解最大调用数 |

## 27. 阈值控制的最大值信号 `lc_threshold_max_signal`

- 版本 / 状态：旧“简单单通道提取”；CPU。
- 来源：[core/DataProcessor.py::simple_single_channel](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1515>)、[core/MainWindow.py::update_result_display](<H:/Newera/programing/transient_data_processing/core/MainWindow.py:2182>)。
- 公式 / 算法：令 `m_t=mean_YX(A_t), M_t=max_YX(A_t)`；thr_known=false 时输出 M；true 时 `s_t=where(M_t>thr,M_t,m_t)`。可在此前执行 08 插值。
- 实际输入输出：`TYX → T`，另存 mean / max / min；输出并非依据二态生理模型拟合的通道电流。
- 边界：像素 max 对噪声和插值过冲敏感。旧 UI 在未知阈值模式下还会根据滚动条修改结果数组，使用 ROI 均值 / whole_mean 的另一套规则；这与上述计算函数不是同一算法，迁移要消除显示控件暗改数值。
- 迁移：扩展 ROI max / mean 归约，必要时增加显式阈值选择模块；参数变化重新生成可追溯结果。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| thr_known | bool | false | 是否应用条件回退 |
| thr | float | 2.5 | 输入强度单位；旧界面 0.1…1000 |
| zm | int | 2 | 空间插值，建议独立节点 |

## 28. 二维傅里叶变换 `lc_fft2`

- 版本 / 状态：旧主程序逐帧 CPU 入口；注册表另有 CPU / CUDA 的 fft2 核心。
- 来源：[core/DataProcessor.py::twoD_fourier_transform](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1571>)、[core/compute/backends/cpu.py::fft2_block_cpu](<H:/Newera/programing/transient_data_processing/core/compute/backends/cpu.py:70>)。
- 公式 / 算法：`F(k_y,k_x)=Σ_yx A(y,x) exp(−2πi(k_y y/Y+k_x x/X))`。应用入口再 fftshift，保存完整复谱 F、abs(F)、angle(F)，主结果为 `log(1+abs(F))`；底层注册核未做 fftshift。
- 实际输入输出：`YX / TYX → 复数空间频域` 及幅度 / 相位 / 对数显示字段。旧应用复谱存 complex64，幅度等数组用浮点；底层可按输出 dtype 转换。
- 边界：旧单帧解析 / squeeze 会造成轴歧义；旧入口 pure_data[i] 对二维输入并非明确按整幅取值。应用入口依赖 ProcessedData 字段，Data 路径也有耦合问题，不能因注册了 fft2 就认定所有旧 UI 输入均可靠。
- 迁移：明确空间频率轴 F_y/F_x、fftshift、归一化、像素间距和复数精度。主算法输出复谱，幅度 / 相位 / log 用明确字段或后续模块产生。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| axes | int[2] | 固定空间两轴 | 不跨 T 做 FFT |
| shift | bool | 应用 true；注册核 false | 记录是否已中心化 |
| norm | enum | 旧调用未显式指定 | 迁移时冻结并与逆变换对应 |
| primary_field | enum | 应用 log_magnitude | 复谱不能由此显示字段反推 |

## 29. 二维傅里叶逆变换 `lc_ifft2`

- 版本 / 状态：旧主程序 CPU 函数；更新日志 0.13.0 提到空间频域滤波工作流。
- 来源：[core/DataProcessor.py::twoD_inverse_fourier_transform](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1629>)。
- 公式 / 算法：`B=ifft2(ifftshift(F))`，旧最终固定取 real(B)，写入 float32；没有重新从显示幅度补相位。
- 实际输入输出：复数空间频域 → YX / TYX 空间图；旧函数优先从 out_processed['twoD_FFT'] 读完整谱，否则读主数组。
- 边界：这个隐藏优先级可能忽略用户只修改的主显示数组；丢失相位仅打印警告仍尝试逆变换。`atleast_3d(YX)` 得 YX1，而非 1YX，单图处理有明确轴风险；real 会丢弃真正复输出。
- 迁移：逆变换必须接被明确编辑后的复谱，拒绝把 log 幅度当完整谱；输出 real / complex 策略显式设置并验证虚部，继承匹配的频率网格与归一化。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| input_field | reference | 隐式优先 twoD_FFT | 新模块改为显式连接 |
| inverse_shift | bool | 固定 true | 必须对应 28 的 shift |
| output_mode | enum | 固定 real | 新实现须支持并说明 complex / real 约束 |
| output_dtype | dtype | float32 | 按精度与输入配套 |

## 30. Welch 功率谱密度 `lc_welch_psd`

- 版本 / 状态：旧绘图控件中已实现的数值转换；CPU。
- 来源：[core/PlotGraphWidget.py::_apply_psd](<H:/Newera/programing/transient_data_processing/core/PlotGraphWidget.py:470>)。
- 公式 / 算法：Welch 分段加窗周期图平均；示意 `PSD(f)=mean_segments(|FFT(w·detrend(s))|²)/(fs Σw²)`，单 / 双边、窗、重叠和去趋势遵循实际库策略。旧调用仅显式指定 fs 与 nperseg。
- 实际输入输出：`T 时间 + T 信号 → F 频率、PSD`；图上替换显示数据，原曲线仍在 data_cache。
- 边界：旧以 `fs=1/mean(diff(t))` 估算均匀采样；时间换算仅处理 ms/us/μs/ns，ps 等单位未转换；不能迁入后把任意真实时间轴按均值强行当等间隔。
- 迁移：成为可保存、可导出的显式频谱节点，检查严格递增且均匀的真实时间轴；冻结库默认、窗能量、单位及有效样本策略。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| fs | float | 由平均 dt 估计；单点用 1 | 迁移后需时间轴验证 |
| nperseg | int | min(4096,T) | >0 |
| window / noverlap / detrend / scaling | mixed | SciPy 默认，旧代码未固定 | 接入时全部明确 |

## 31. 区间基线与峰参数 `lc_peak_metrics`

- 版本 / 状态：旧绘图交互中的区间最高峰分析；CPU。
- 来源：[core/PlotGraphWidget.py::_update_peak_analysis](<H:/Newera/programing/transient_data_processing/core/PlotGraphWidget.py:554>)。
- 公式 / 算法：用区间两端点作线性基线 b(x)，`y_c=y−b`；峰为 argmax(y_c)，净高 max(y_c)，面积为 `trapz(y_c,x)`。左右最近的 ≤半高采样点给出 `FWHM=x_right−x_left`。
- 实际输入输出：`T 坐标 + T 信号 → peak_x、原始 / 净峰高、面积、FWHM`；旧只展示悬浮文字与标记，没有独立处理结果。
- 边界：至少三点；选第一条可见曲线，未明确数据身份。面积包含整个选区的正负残差；半高点不插值、找不到交点则用边界。不是多峰检测，不含 prominence / 峰去重。
- 迁移：基线校正可独立节点，峰参数接显式信号；正峰 / 负峰、边界截断、单位、插值及有效状态需要定义。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| x_range | float[2] | 交互选区 | 原闭区间；新用上游时间裁剪 |
| baseline | enum | 端点线性插值 | 应与独立去背景区分 |
| half_height | float | 固定 0.5 | 相对扣基线后的正峰 |
| crossing_interpolation | enum | 无，直接取样点 | 提升精度属于明确新版本 |

## 32. 阈值事件幅值与持续时间 `lc_threshold_events`

- 版本 / 状态：旧静态辅助函数；本次未找到可靠的主界面调用链，不能列为已验收工作流。
- 来源：[core/DataProcessor.py::calculate_amp_dur](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1861>)。
- 公式 / 算法：open 取连续 `s>thr`，close 取连续 `s<thr`；每个连续区段事件振幅为区域**均值**，持续时间为 `事件样本数×0.65`。
- 实际输入输出：`T → K 振幅数组、K 持续时间数组`，未返回起止索引。
- 边界：等于阈值即终止；无迟滞、去抖、最小持续时间、基线扣除和首尾截断标记。0.65 是硬编码采样间隔且未附单位。
- 迁移：事件表输出应包含起止样本、真实时间、均值 / 峰值、边界截断和质量；时间由直接上游映射计算，不能继承 0.65。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| thr | float | 调用方提供 | 信号单位 |
| mode | enum | open | open / close |
| dt | float | 固定 0.65 | 接入时必须替换为真实时间定义 |

## 33. 事件边沿指数时间常数 `lc_event_exponential`

- 版本 / 状态：源码明确标注**未启用**，原型存在控制流问题。
- 来源：[core/DataProcessor.py::fit_exponential_time](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1917>)。
- 公式 / 算法：按阈值穿越取 n 点，上升片段作反转 / 基线变换，下降片段反转后拟合 `A exp(−x/τ)+b`，输出 τ 与 MSE；`x=arange(n)×0.65`。
- 实际输入输出：`T → K τ、K MSE`；事件身份和发生时间未保留。
- 边界：未触发阈值的循环也进入拟合段，y 可能未定义或沿用上次；不是每个返回结果都对应一次新事件。方法未声明 staticmethod，类实例调用方式也需修正。
- 迁移：作为重新设计的参考；先独立事件检测，再按事件窗口拟合，明确上升模型、采样时间、边界及求解状态。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| thr | float | 调用方提供 | 阈值穿越条件 |
| mode | enum | up | up / down |
| n | int | 5 | 拟合点数；需满足可辨识性 |
| dt | float | 固定 0.65 | 原型硬编码，必须移除 |

## 34. CLAHE 与轻度空间模糊 `lc_clahe_blur`

- 版本 / 状态：旧心肌光流的辅助预处理；独立试验 test2.py 另有类型转换步骤。
- 来源：[core/DataProcessor.py::clahe_and_blur](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1901>)、[test2.py::preprocess_image](<H:/Newera/programing/transient_data_processing/test2.py:30>)。
- 公式 / 算法：OpenCV CLAHE，随后 GaussianBlur(3×3,sigma=0)。试验脚本先 RGB / RGBA 转灰度，再将非 uint8 输入 min–max 到 0…255。
- 实际输入输出：单通道 YX → YX 纹理增强灰度图；主函数直接 clahe.apply(img)，没有试验脚本的统一转 uint8 步骤。
- 边界：OpenCV 输入类型有约束，不能把任意 float / complex 科学数组直接传入。该 CLAHE 是实际存在的光流辅助实现，不代表 02 已注释的导入 CLAHE 分支可用。逐帧增强会改变光强统计。
- 迁移：可拆为灰度转换、量化 / 固定标定、CLAHE、空间高斯四个节点，或作为明确声明的光流预设；用于追踪纹理，不能隐含用于强度关联。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| clipLimit | float | 固定 3.0 | OpenCV CLAHE 定义，非 02 的 clip_limit |
| tileGridSize | int[2] | 固定 (8,8) | 局部块网格 |
| Gaussian kernel / sigma | int[2] / float | (3,3) / 0 | sigma=0 由 OpenCV 推定 |
| dtype conversion | policy | 主函数无；试验脚本转 uint8 | 迁移时明确量化与固定尺度 |

## 35. Farneback 稠密光流与速度场 `lc_optical_flow`

- 版本 / 状态：旧心肌细胞运动分析 CPU 实现；test2.py 为两帧试验入口。
- 来源：[core/DataProcessor.py::heartbeat_movement](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1709>)、[test2.py::analyze_motion](<H:/Newera/programing/transient_data_processing/test2.py:73>)。
- 公式 / 算法：Farneback 得 `u(y,x),v(y,x)`，`m=sqrt(u²+v²), angle=atan2(v,u)`；旧物理速度 `speed=m×space_step×fps/(idx_current−idx_previous)`。固定参考模式将参考帧与指定序列配对；连续模式按指定序列相邻成员配对。
- 实际输入输出：`TYX → T′YXV` 位移场（V=2），附速度幅值、角度 TYX 和最大 / 平均速度 T′；旧 time_point 为相对帧差，未转换为真实秒。
- 边界：旧函数隐藏调用 34，硬编码 Farneback 参数；Δframe=0 强制改 1，而非拒绝无效配对；FPS 速度换算不适配真实非均匀采样。输出角度为 OpenCV cartToPolar 默认弧度。step 主要控制箭头显示 / 导出，不降低光流网格计算。
- 迁移：拆开预处理、帧配对光流与物理单位换算；保留每对真实时间 `Δt`，`velocity=(pixel_displacement×pixel_size)/Δt`。V 位移轴不能当 RGB；输出箭头、热图及曲线可共用结果窗口的不同字段。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| base_num | int | 调用方提供；−1 连续模式 | 指定参考或顺序帧对 |
| after_series | int[] | 调用方提供 | 需校验顺序、边界和非零 Δt |
| pyr_scale / levels | float / int | 固定 0.5 / 3 | Farneback 金字塔 |
| winsize / iterations | int | 固定 15 / 3 | 空间窗口及迭代 |
| poly_n / poly_sigma / flags | mixed | 固定 5 / 1.2 / 0 | 多项式邻域参数 |
| space_step / fps | float | 继承 metadata | 新实现优先真实像素尺度 / 时间 |
| step / scale / export_mode | mixed | 调用方提供 | 归属显示 / 导出，不混入数值处理 |

## 36. 安全多数组表达式 `lc_array_expression`

- 版本 / 状态：旧独立数据计算器，已有形状验证、输入槽、metadata 和后台调用；CPU。
- 来源：[core/calculator/engine.py::validate](<H:/Newera/programing/transient_data_processing/core/calculator/engine.py:146>)、[core/calculator/engine.py::_validate_call](<H:/Newera/programing/transient_data_processing/core/calculator/engine.py:256>)、[core/calculator/engine.py::execute](<H:/Newera/programing/transient_data_processing/core/calculator/engine.py:347>)、[core/calculator/model.py::CalculationPlan](<H:/Newera/programing/transient_data_processing/core/calculator/model.py:17>)。
- 公式 / 算法：AST 白名单解释执行 NumPy 运算；支持 `+ − × / ** %`、一元正负、单个比较、广播和整数 / 冒号切片；函数 mean / max / min / sum / std / abs / sqrt / log / clip / where / transpose；常量 pi/e。
- 实际输入输出：`A,B,… 多个标量或数值数组 → 根据表达式推导的真实形状及 dtype`；预切片、轴归约和转置逐步列出形状及内存估计。
- 边界：拒绝任意属性访问、未知函数与任意 Python；旧 execute 仍可能一次分配完整广播结果和多个中间数组，验证中的估计不能替代有界执行。相同 shape 不保证时间、坐标、单位或物理语义一致；metadata 继承由用户选择也不能自动证明正确。
- 迁移：未来统一多输入模块 / 表达式节点，显式绑定来源及上游结果字段、坐标对齐、单位和除零策略；常见操作优先注册专门模块。不能用表达式绕开强度 / 标签 / g² 的语义限制。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| expression | str | 用户填写 | 安全 AST 白名单 |
| operands | reference[] | A/B/C… 槽位 | 明确来源 / payload / axes |
| slice_text | str | 空 | 每个操作数独立预切片 |
| axis / keepdims | int / tuple / null、bool | 函数默认 axis=null、keepdims=false | 归约轴须有效且不重复 |
| transpose axes | int[] / null | null 反转轴顺序 | 明确完整排列 |
| metadata_source / overrides | reference / dict | 用户选择 | 须与实际形状和物理坐标相容 |

## 37. 分块 FFT 目标频率幅值 `lc_experiment_fft_trace`

- 版本 / 状态：**离线实验参考**，未接主程序。
- 来源：[others/all_signal_stft&cwt.py::extract_amplitude](<H:/Newera/programing/transient_data_processing/others/all_signal_stft&cwt.py:114>)、[others/simulate_signal_compare_stft&cwt.py::extract_150hz_amplitude](<H:/Newera/programing/transient_data_processing/others/simulate_signal_compare_stft&cwt.py:135>)。
- 公式 / 算法：T 样本均分 M 段，每段 L=floor(T/M)，尾部舍弃；各段 FFT 后取最接近 f0 的正频率 bin，按旧条件乘 2/L 或 1/L。
- 实际输入输出：`T → M 个幅值`；旧返回时间固定 `arange(M)/10`，幅值还统一减 2。
- 边界：固定 0.1 时间步与 −2 是实验假设，不由输入 fps / 窗口生成；旧正频率截取排除了 Nyquist，边缘 bin 的倍增判断不完全正确。一个脚本开放 f0，另一个固定 150 Hz。比较脚本还用 [more_compare.py::interp_to_t](<H:/Newera/programing/transient_data_processing/others/more_compare.py:82>) 的 `np.interp` 将短时谱曲线线性映射回原时间轴，仅供比较；这不会增加原有时间分辨率，接入时必须显式记录插值和边界。
- 迁移：可作为 21 的无重叠矩形窗特例 / 验证基线；重建真实段中心时间、窗幅值标定、DC / Nyquist 规则，不照搬偏置。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| sampling_rate | float | 调用方提供 | >0，均匀采样 |
| target_freq | float | 一个脚本传入，另一个固定 150 Hz | 必须与有效频率轴相容 |
| frame_count | int | 50 | M≤T；分段数 |
| time_step / amplitude_offset | float | 固定 0.1 / −2 | 实验遗留，不作为新默认 |

## 38. 合成脉冲、调制与噪声 `lc_experiment_signal_generator`

- 版本 / 状态：**离线实验参考**，用于 STFT / CWT 比较；包含根目录 teat3.py 旧试验。
- 来源：[others/simulate_signal_compare_stft&cwt.py::gen_base_square_wave](<H:/Newera/programing/transient_data_processing/others/simulate_signal_compare_stft&cwt.py:13>)、[others/simulate_signal_compare_stft&cwt.py::create_composite_signal](<H:/Newera/programing/transient_data_processing/others/simulate_signal_compare_stft&cwt.py:68>)、[others/HH-action_protential.py::generate_pulse_signal](<H:/Newera/programing/transient_data_processing/others/HH-action_protential.py:49>)、[others/first_compare.py::generate_signals](<H:/Newera/programing/transient_data_processing/others/first_compare.py:18>)。
- 公式 / 算法：随机开关方波 q(t) 取 1 / 0.3；正弦调制 `a sin(2πf t)`；高斯噪声 `η~N(noise_amp,noise_level²)`；组合 `s=q(t)×sinusoid+η`。另有周期矩形脉冲、非平稳信号和小波 / 窗参数比较。
- 实际输入输出：参数 → T 时间、T 基准及合成信号；适合提供真值、可控延迟和噪声的验证数据。
- 边界：旧 noise_amp 实为高斯**均值**，noise_level 为标准差；随机源未显式固定 seed。比较脚本有经验缩放如 /3.16，与 23 的冻结 CWT 振幅规则不同，不能混入生产算法。
- 迁移：可作为未来测试数据源；固定随机种子，分开真值 / 噪声 / 调制输出。生成操作属于数据源，不能伪装为需要输入视频的处理算法。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| fs / duration | float | 组合函数 1500 Hz / 5 s | >0 |
| mod_freq / mod_amp | float | 150 Hz / 1.0 | 应符合采样定理 |
| density | float | 1.0 | 调整随机关闭区间，须>0 |
| noise_amp / noise_level | float | 2 / 0.05 | 实为噪声均值 / 标准差 |
| 方波区段 | interval | 开 20…70 ms；关 50…100 ms / density | 旧采样取整 |
| 周期脉冲 | mixed | 周期 700 ms、宽 5 ms、幅度 5…6、步长 0.5 | HH 实验函数默认 |
| seed | int | 未提供 | 新生成器必须记录 |

## 39. OU 有色噪声 `lc_experiment_ou_noise`

- 版本 / 状态：**离线实验参考**，神经电生理仿真辅助。
- 来源：[others/HH-action_protential.py::generate_ou_noise](<H:/Newera/programing/transient_data_processing/others/HH-action_protential.py:10>)。
- 公式 / 算法：Euler–Maruyama：`I_(n+1)=I_n−(I_n−μ)Δt/τ+σ sqrt(2Δt/τ) ξ_n`，`ξ~N(0,1)`；初始 I0=μ。
- 实际输入输出：参数 → T 噪声序列；τ 和 dt 采用同一时间单位。
- 边界：没有 tau/dt/n_steps 的完整校验和 seed；离散近似不等同连续模型精确采样。HH 仿真中该序列随后被脉冲序列覆盖，因此“生成了 OU”不代表最终仿真实际使用 OU。
- 迁移：可供信号处理 / 关联方法的有色噪声测试；记录随机种子及积分步长，独立于图像处理工作模式。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| n_steps / dt | int / float | 调用方提供 | 正样本数 / 正时间间隔 |
| tau | float | 15.0 | 噪声相关时间 |
| sigma | float | 4.0 | 波动尺度 |
| mean_input | float | 3.0 | 均值 |
| seed | int | 未提供 | 接入验证源时显式记录 |

## 40. Hodgkin–Huxley 电生理仿真 `lc_experiment_hh_model`

- 版本 / 状态：**离线实验参考**；HH-pre.py 使用 odeint，HH-action_protential.py 另有显式迭代与抽样实现；不是机器学习模型。
- 来源：[others/HH-pre.py::hodgkin_huxley_model](<H:/Newera/programing/transient_data_processing/others/HH-pre.py:65>)、[others/HH-action_protential.py::simulate_pc12_spontaneous](<H:/Newera/programing/transient_data_processing/others/HH-action_protential.py:125>)。
- 公式 / 算法：`Cm dV/dt=I_ext−gNa m³h(V−ENa)−gK n⁴(V−EK)−gL(V−EL)`；门控变量满足 `dq/dt=αq(V)(1−q)−βq(V)`, q∈{m,h,n}，速率函数见源实现。
- 实际输入输出：模型参数 / 刺激 → T 上的电位和门控状态；显式版本返回抽样电位、仿真时间与驱动电流。
- 边界：采用脚本内固定电导 / 电位参数，函数名称中的 PC12 不构成细胞模型标定证据。HH-pre 速率分母在特定 V 有可去奇点而未显式处理；另一版本有奇点保护。后者生成 OU 后又覆盖为脉冲刺激，记录时间用 linspace 而不严格对应实际积分抽样点。
- 迁移：仅作为生物信号验证源候选，先核验数值积分、稳定性、刺激来源、单位和真实抽样映射；不作为图像算法默认预设。

| 参数 | 类型 | 旧默认 / 固定值 | 范围 / 选项 |
|---|---|---|---|
| Cm | float | 1.0 μF/cm² | 膜电容 |
| gNa / gK / gL | float | 120 / 36 / 0.3 mS/cm² | 脚本固定 |
| ENa / EK / EL | float | 50 / −77 / −54.387 mV | 脚本固定 |
| duration_ms / dt_sim | float | 显式仿真函数 1000 / 0.01 ms | 采样与积分步长分开 |
| target_fs | float | 2000 Hz | 取整抽样因子，需保存实际间隔 |
| I_ext | signal | HH-pre 为 10…11 ms 的 10 μA/cm² 脉冲 | 另一版本为独立脉冲发生器 |

## 旧注册表与本参考的对应

旧统一注册表只有 8 个条目，其他算法仍分布在 GUI、DataProcessor、LifetimeCalculator 和辅助脚本中。因此仅扫描注册表会漏掉 ROI、事件、光流、PSD、寻峰、计算器等功能。

来源：[core/compute/registry.py](<H:/Newera/programing/transient_data_processing/core/compute/registry.py:1>)。

| 旧 algorithm_id | 对应本清单 | 备注 |
|---|---|---|
| stft | 21 | 目标频带振幅，非全谱诊断 |
| cwt | 23 | 多尺度振幅归约 |
| cwt_quality | 24 | S×T 系数幅值，注册为不可单独配置的辅助项 |
| em_preprocess | 03 + 04 | 本清单按参考统计与相对信号拆分 |
| lifetime_single | 13 + 14 | 质量筛选与单指数拟合 |
| lifetime_double | 13 + 15 | 质量筛选与多字段双指数 |
| fft2 | 28 | 底层未 shift，与旧 UI 输出组织不同 |
| spatiotemporal_convolution | 20 | 含开发中工作台及边界 / origin 契约 |

注册表中 fit_status 写 int16，CPU 寿命管线实际分配 uint8；迁移应以状态枚举与真实 dtype 契约统一后再注册，不原样复制不一致描述。

## 供后续选择的接入分组

以下是迁移工作量和依赖判断，**不是已经确定的开发排期**。

| 接入分组 | 候选 | 对预览 / 处理闭环的帮助 | 需要先补的契约 |
|---|---|---|---|
| 优先复用已有模块 | 02、09、10、19、27 | 归一、裁剪、区域信号和滤波，减少重复算法 | 相同公式下的边界、dtype、无效值统一 |
| 较独立的通用能力 | 01、05/06、07、08、11、25、30、31 | 原值质量检查、ROI 编辑、投影与曲线验证 | 质量掩码、频率 / 分箱轴、坐标和状态输出 |
| 核心科学分析 | 03/04、12、14/15、21…24 | EM-iSCAT、寿命图、时频图及区域曲线 | 参考依赖、多输出、均匀时间检查、真实窗口支持 |
| 实验定义先确认 | 16…18、26、32/33、34/35 | 扩散、传热、单通道、事件及运动分析 | 物理单位、边界 / 初值、向量场、事件表 |
| 通用扩展基础 | 20、28/29、36 | 自定义卷积、频域编辑和跨来源数值处理 | 任意核、复数频域、多输入与资源有界执行 |
| 保留为验证素材 | 37…40 | 可控信号、噪声和仿真真值 | 修正硬编码，固定 seed、真实采样及模型版本 |

建议新组合的形状应可读，例如：

1. `时间裁剪 TYX → 背景参考 + 相对信号 TYX → ROI 均值 T → STFT 全谱 FT′`。
2. `时间裁剪 TYX → ROI 像素 TY′X′ → 单指数拟合 {τ,R²,status}:Y′X′ → 选择 τ 字段 → 直方图`。
3. `时间裁剪 TYX → 线段剖面 TR → 高斯宽度 σ²:T → 有效样本筛选 → 扩散回归 D`。
4. `TYX → 复数 FFT 空间频谱 → 同网格掩码 → IFFT → TYX → 原图 / 结果对比`。
5. `时间裁剪 TYX → 纹理预处理 TYX → 显式帧对 → 光流 T′YXV → 真实 Δt 速度场 → ROI 速度曲线`。

其中参考、多输出和新轴是待扩展接口；不能因为表达式形状能广播就认定上述所有流程已能在当前程序直接运行。单 / 双指数寿命参数 τ 与本项目关联延迟 τ 也应在界面中明确命名。

## 接入前逐项核验

| 类别 | 必须记录 / 验证的内容 | 已发现的旧实现差异 |
|---|---|---|
| 数据选择 | 原始 / 处理值、所选字段和来源 | image_backup、data_origin、data_processed、out_processed 在各入口混用 |
| 时间 | 真实样本时间、贡献区间、物理单位 | 固定 FPS、帧号×time_unit、0.65、arange/10、linspace 并存 |
| 空间 | ROI 原点、尺度、掩码、插值 | 圆形 quick_mask 半径为 size−1，线宽越界仍除固定采样数 |
| 统计 | 分母、有限值策略、分位 / 方差约定 | ROI 真均值与 ROI 外填零后的全幅均值不同 |
| 背景 | 参考范围、近零、缺失值、可追溯引用 | 中位数背景的安全替换由空间分块决定 |
| 数值 | 输入 / 计算 / 输出 dtype、量化、复数 | Anchor 整数截断、STFT 内部 float32、IFFT 固定 real / float32 |
| 拟合 | 初值、边界、单位、R²、失败及质量掩码 | 双指数特定下界、τ 范围 OR、失败零值与原始拟合共存 |
| 频谱 | 幅值 / 功率 / PSD、窗、边界、归一和相位 | STFT 两种缩放；CWT“功率”实际幅值；FFT 显示字段不是完整复谱 |
| 执行 | 分块无关、取消、预算、CPU / GPU 对照 | 开发中 GPU 自检不等于实验数据和目标设备验收 |
| 结果 | 独立字段、状态、参数与全流程来源 | 一些算法只在绘图时计算或由控件写回结果 |

若迁入新程序，首先用可手算小数组验证定义，再用实际数据对照；CPU 基线完整可用后才接可选 GPU。试算与正式运行共享同一实现，结果按矩阵、信号、频谱、字段集合、向量场或事件表显示和导出，不为每种算法新建独立工作模式。

## 阅读依据、保留边界与后续验证入口

- 功能与版本入口：[readme.md](<H:/Newera/programing/transient_data_processing/readme.md:1>)、[updatinglog.md](<H:/Newera/programing/transient_data_processing/updatinglog.md:1>)、[core/app_metadata.py](<H:/Newera/programing/transient_data_processing/core/app_metadata.py:1>)。更新记录保留历史行为；与现实现冲突时，本文按具体函数说明，不把新增记录当作现有可运行证据。
- 数值内核与执行入口：[core/DataProcessor.py](<H:/Newera/programing/transient_data_processing/core/DataProcessor.py:1>)、[core/LifetimeCalculator.py](<H:/Newera/programing/transient_data_processing/core/LifetimeCalculator.py:1>)、[core/compute/registry.py](<H:/Newera/programing/transient_data_processing/core/compute/registry.py:1>)、[core/compute/algorithms/lifetime.py](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime.py:1>)、[core/compute/algorithms/lifetime_pipeline.py](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/lifetime_pipeline.py:1>)、[core/compute/algorithms/stft.py](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/stft.py:1>)、[core/compute/algorithms/cwt.py](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/cwt.py:1>)、[core/compute/algorithms/preprocess_pipeline.py](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/preprocess_pipeline.py:1>)、[core/compute/algorithms/convolution_pipeline.py](<H:/Newera/programing/transient_data_processing/core/compute/algorithms/convolution_pipeline.py:1>)。
- 空间与数组操作：[core/DataManager.py](<H:/Newera/programing/transient_data_processing/core/DataManager.py:1>)、[core/ROIdrawDialog.py](<H:/Newera/programing/transient_data_processing/core/ROIdrawDialog.py:1>)、[core/SpatialExtractor.py](<H:/Newera/programing/transient_data_processing/core/SpatialExtractor.py:1>)、[core/calculator/engine.py](<H:/Newera/programing/transient_data_processing/core/calculator/engine.py:1>)、[core/convolution/model.py](<H:/Newera/programing/transient_data_processing/core/convolution/model.py:1>)、[core/convolution/presets.py](<H:/Newera/programing/transient_data_processing/core/convolution/presets.py:1>)。
- GUI 中的数值代码：[core/PlotGraphWidget.py](<H:/Newera/programing/transient_data_processing/core/PlotGraphWidget.py:1>)、[core/ResultDisplayWidget.py](<H:/Newera/programing/transient_data_processing/core/ResultDisplayWidget.py:1>)、[core/MainWindow.py](<H:/Newera/programing/transient_data_processing/core/MainWindow.py:1>)；这部分应提取为内核函数，不迁移其 Qt 控件和状态耦合。
- 测试参考：[tests/test_compute_contracts.py](<H:/Newera/programing/transient_data_processing/tests/test_compute_contracts.py:1>)、[tests/test_stft_pipeline.py](<H:/Newera/programing/transient_data_processing/tests/test_stft_pipeline.py:1>)、[tests/test_stage_d_pipelines.py](<H:/Newera/programing/transient_data_processing/tests/test_stage_d_pipelines.py:1>)（含 CWT 管线）、[tests/test_lifetime_pipeline.py](<H:/Newera/programing/transient_data_processing/tests/test_lifetime_pipeline.py:1>)、[tests/test_lifetime_result_fields.py](<H:/Newera/programing/transient_data_processing/tests/test_lifetime_result_fields.py:1>)、[tests/test_anchor_distribution.py](<H:/Newera/programing/transient_data_processing/tests/test_anchor_distribution.py:1>)、[tests/test_calculator_engine.py](<H:/Newera/programing/transient_data_processing/tests/test_calculator_engine.py:1>)、[tests/test_convolution_workbench.py](<H:/Newera/programing/transient_data_processing/tests/test_convolution_workbench.py:1>)。本次未执行这些旧测试。
- 根目录 final.py 为旧寿命 GUI 原型，test4.py / test5.py 为切片器原型，test2.py 为两帧光流试验，teat3.py 和 others/ 为离线比较 / 仿真。已被正式内核覆盖的重复算法未重新计数；旧脚本的额外经验缩放不覆盖正式契约。
- 本次未发现已接入的机器学习推理管线，亦未发现可核验的 Julia CWT 实现。寿命趋势筛选中的 Pearson 相关也不等于本项目标准二阶互关联 g²。
- 原项目文件保持只读；本次产物仅为参考文档和本项目索引。后续接入时每个模块仍须登记版本、参数、输入输出、执行预算及必要依赖。

### 关键来源内容指纹

以下为读取快照的 SHA-256，便于旧项目继续更新后定位本参考对应的实现；离线实验包含未跟踪文件，应按内容指纹而非仅按 Git HEAD 识别。

| 来源文件 | SHA-256 |
|---|---|
| readme.md | `503ea3af0e443f9c60e7ff1a2c6dc1a8f24e793c5d75bcd3f3679a4006ba8549` |
| updatinglog.md | `6a294d9476323f2180edd93dc795260e6bdf9bef5995eded3b202b282ea66752` |
| core/app_metadata.py | `ec522ecc75a3af2949bf6515ef12288c39336b44751513400913c4e34bb0e444` |
| core/DataProcessor.py | `3e6ac97d88eaf0924882cf8f0fe2aef1428a4fc792c1b974c8cb2d7658d97bf0` |
| core/LifetimeCalculator.py | `9195bd8fb6bea2064e476c0dd0aec0ad6266498966b7e142dc69ceab667855c0` |
| core/DataManager.py | `8f2615259d0114243bb56b4c8eaad2e8f1c68af3b43ee27bcb7da6ed48eaba82` |
| core/MainWindow.py | `5213a0c684c0fe28c91439f1d490993edb9ea577775ca780c26ca8696b3377ad` |
| core/PlotGraphWidget.py | `b89f005ec82fec433819b88d74b7a881629302b92b3375eed83fe3691223ba1b` |
| core/ResultDisplayWidget.py | `d47ee6ab3cb426e1930ad1702c68540008a4ac16488ce3447877cbc07fc19903` |
| core/compute/registry.py | `b7942416ffccf1ffe6bc8278372b60d2f27a6f6384377159f48ff9778b00d6a0` |
| core/compute/algorithms/lifetime.py | `a204e02fed5149bba84522be177c7cf8db21d384fcad25f0f77df1da01a8b918` |
| core/compute/algorithms/preprocess_pipeline.py | `4e8ffa15f2baebe6993b37b7eeb51b90adbf2211d402049910e770012c183af5` |
| core/compute/algorithms/stft.py | `915b4f7dee4f0db0e7631a69508a65751aa19fe51bc2a1990599bf5549cf2883` |
| core/compute/algorithms/cwt.py | `fddbbb5c1f30da3e76b45494a3c632a6b6d9ff6c79369da72a86afcf721c8eed` |
| core/compute/algorithms/convolution_pipeline.py | `19837170238dde09763c695d1f2f06f41c4ae2e1b46bf031201ff3af5eb200e5` |
| core/convolution/presets.py | `45906fd37bf4b86c46c92240aa03db194f8acfc8712642c17a8ebda5fca6a8ca` |
| core/calculator/engine.py | `18eaa19045cdd1c5a3f53b4718dc246f9709c60119d505e992cbb60f7867266b` |
| others/all_signal_stft&cwt.py | `f0c1cfe62d2b6b33a44ad452043ae219dc40c1a1e131819eee24298a3041c1aa` |
| others/simulate_signal_compare_stft&cwt.py | `cac5e4b9b5ed994868c6279bceaf9196aa03fa5e41bbeea6ca5fa7506a15d3fc` |
| others/HH-pre.py | `b6f3f9d13574002ad1a47d5fcdd1b009d1c10f51d3da8c142c9b6f90263db0c7` |
| others/HH-action_protential.py | `46ceb78c7afe1384da44b6b226d94818018e6b2aa5eea6fe485a5d99a97f0b76` |
