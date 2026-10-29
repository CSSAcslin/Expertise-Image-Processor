# Expertise Image Processor

**0.9.4 · 本机 Web 工作台 · Python 3.12–3.14**

面向科研数据的图像与视频工作台，将 **数据预览 → 模块处理 → 结果对比 → 导出与复用** 放在同一工作流中。浏览器负责交互，独立 Python 进程负责解码和计算。

![0.8.6 固定 τ、固定中心的平均窗长扫描 g² 曲线](docs/assets/correlation-window-0.8.6.png)

*截图使用验证用模拟序列，不代表实验结果。*

## 快速开始

主要验证平台为 Windows。先安装 Python 3.12–3.14，然后在终端中执行：

~~~powershell
git clone https://github.com/CSSAcslin/Expertise-Image-Processor.git
cd Expertise-Image-Processor
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_web.py
~~~

浏览器会打开 **http://127.0.0.1:8765**。后续可双击 [start.cmd](start.cmd)；保持启动窗口运行，关闭窗口停止服务。Web 前端无需 Node.js 或构建步骤。

~~~powershell
# 改用其他端口，不自动打开浏览器
.\.venv\Scripts\python.exe run_web.py --port 8766 --no-browser

# 指定独立工作目录
.\.venv\Scripts\python.exe run_web.py --workspace "D:\EIP-workspace"
~~~

Qt 兼容入口为 [start_desktop.cmd](start_desktop.cmd)，需另装 [requirements-desktop.txt](requirements-desktop.txt)。默认入口使用 Web。

## 能做什么

- **预览与定位**：图像、受支持的 TIFF 序列、常规视频和 Phantom CINE；逐帧定位、真实时间、播放、ROI 与高位深原值查询。
- **编辑模块流程**：添加、复制、启停、排序及撤销；实例独立设置参数，按轴、尺寸和科学含义校验输入输出。
- **处理科研数据**：空间与时间编辑、数值修饰、统计、固定 τ 系数图及 g²(τ) 扫描；试算与正式运行共用算法。
- **比较一维结果**：最多 32 条曲线叠加，独立显隐、命名、颜色与取值焦点；双幅时序比较可直接采集配准后的对应点 / 区域，保留左右时序并导出完整比较包。
- **扫描平均窗长**：固定 τ 和同一中心，计算 g² 随平均样本数 W 的曲线；支持正负 τ、窗长范围或自定义列表，保留真实配对时差并导出完整数值 / 图形。
- **统一管理数据**：左侧「数据 / 模块」并列；数据页内「输入 / 处理」纵向排列，分界可拖动，独立计数、折叠及批选。可找回历史导出文件夹，所选输入可运行，结果可合并导出、移除记录或逐项确认清理文件。
- **控制资源**：在运行设置中统一设置应用内存、缓存、显存和默认计算设备；大序列按需读取，结果使用磁盘映射。
- **定位运算瓶颈**：任务详情按需展开读取、计算、传输、来源映射与输出耗时，以及输入读取 / 输出吞吐；窗长扫描按 W 结果计数。
- **扩展算法**：统一模块契约与本机 Python 插件，保留完整参数、运行版本、样本映射和来源记录。

CPU 可完整运行。当前 27 个内置模块均可接入可选 CuPy / CUDA 后端，连续 GPU 节点复用设备数据，按预算批量传输；0.9.0 为 g² 提供 float64 融合内核和小信号批量累加，复用固定中心基准、上传分块和多页 TIFF 读取句柄。插件与旧序列统计仍使用 CPU，自动模式在 CUDA 不可用时回退。RTX 5070 Ti 已完成全部可见模块的真实 CUDA 对照和浏览器正式运行验收；安装、解释器选择及同机性能见 [GPU 落地说明](docs/GPU_DEPLOYMENT.md)，实现见 [0.9.0 运算加速](docs/ACCELERATION_0_9_0.md)。小任务可能 CPU 更快，不保证所有流程加速。

0.9.1 优化了后台执行：滑窗来源映射增量维护，压缩后的主路结果独立驻留，减少前序重复计算及大图缓存保留。代表性万帧流程保持原配置复跑，CPU 三次中位数为 46.76 秒、GPU 混合流程为 61.74 秒，均完整核对原数值和来源；旧单次基线分别为 565.18 / 700.64 秒。支持范围、预算和测量条件见 [0.9.1 后端说明](docs/BACKEND_EXECUTION_0_9_1.md)。

0.9.2 将 CPU、内存、GPU 运算占用和显存统一放到主界面右下角，以实时曲线和数值观察预览及处理时的设备状态；任务详情增加已覆盖 g² 融合内核的 CUDA 事件计时。布局与计时范围见 [设备监测说明](docs/DEVICE_MONITOR_0_9_2.md)。

0.9.3 为处理引擎增加通用执行计划：ROI、点、线和空间截面反推必要范围，逐点变换、固定配准、高斯邻域和时间编辑按依赖执行；紧凑结果按块接管主路，固定几何索引复用。CPU标量与GPU小信号g²可连续分块产出，并接续逐点后处理；显式设备、模块精度、旁路和完整输入校验保留。全部36个内置及兼容模块的能力与边界见 [模块审查](docs/MODULE_EXECUTION_CAPABILITIES.md)，实际范围见 [0.9.3执行说明](docs/PROCESSOR_EXECUTION_0_9_3.md)。

0.9.4 针对固定中心延迟扫描增加连续信号范围访问、CPU／GPU跨τ批量运算和有界真实时间数组，保留float64、实例精度与设备设置。原3001点完整任务CPU／原混合GPU三轮中位数2.89／3.74秒，GPU关联发射从474158次降到12次；全量856项测试通过。适用范围、完整数值来源核对和T／W复测见 [延迟扫描执行说明](docs/DELAY_SCAN_0_9_4.md)。

0.9.3验收时716项真实CUDA / CPU / Web回归通过；真实CINE39次及通用流程72次对照均核对完整结果。ROI完整任务同源码兼容 / 优化中位数为CPU 37.60 / 7.31秒、原混合GPU 40.64 / 6.56秒、连续GPU 9.37 / 4.21秒；全幅CPU控制组持平。数据、计时口径与逐轮结果见 [0.9.3实测](docs/PROCESSOR_EXECUTION_0_9_3.md#正式性能复测)。

## 操作顺序

1. 导入数据，定位帧并核对原值；需要选区时添加 ROI 裁剪模块。
2. 添加模块并调整顺序，检查输入输出，再试算或正式运行。
3. 在左侧 **数据 → 处理** 中点击结果，与原数据对比；下载所需输出或作为下一阶段输入。任务记录位于中央底部窄栏，按需展开。

显示窗宽、伽马和色图只改变显示。任务记录保留数值定义、坐标和真实时间；未完成结果明确标记。默认仅监听本机，原始文件通过路径引用，数据处理在本机完成。

## 文档

| 内容 | 入口 |
| --- | --- |
| 0.9.4 延迟扫描批量执行 | [连续信号、Q批量与真实时间摘要](docs/DELAY_SCAN_0_9_4.md) |
| 0.9.3 通用处理流执行与模块范围优化 | [统一执行计划、批量交接及科学边界](docs/PROCESSOR_EXECUTION_0_9_3.md) |
| 0.9.2 设备状态曲线与 GPU 计时 | [主界面设备监测](docs/DEVICE_MONITOR_0_9_2.md) |
| 独立 CPU / GPU 与工作台速度对照 | [两条专用流程、39轮实测与瓶颈](docs/DIRECT_FLOW_COMPARISON_0_9_2.md) |
| 0.9.1 来源映射与主路结果复用 | [后端执行与原流程复测](docs/BACKEND_EXECUTION_0_9_1.md) |
| GPU 安装、解释器与 RTX 5070 Ti 实测 | [GPU 落地与验收](docs/GPU_DEPLOYMENT.md) |
| 0.9.0 运算加速与性能诊断 | [融合 g²、读取复用与基准](docs/ACCELERATION_0_9_0.md) |
| 0.8.6 固定 τ 的 g² 窗长曲线 | [窗长扫描](docs/CORRELATION_WINDOW_0_8_6.md) |
| 完整操作、格式与兼容边界 | [使用指南](docs/USER_GUIDE.md) |
| 0.8.5 双幅时序与多曲线验证 | [算法与结果比较](docs/PAIRED_COMPARISON_0_8_5.md) |
| 0.8.4 大序列、滑窗复用与 GPU | [处理链与验证边界](docs/GPU_PIPELINE_0_8_4.md) |
| 0.8.3 左侧双选项卡与预览优先布局 | [布局与数据管理](docs/SIDEBAR_LAYOUT_0_8_3.md) |
| 结果导出、内存预算与 GPU | [0.8.0 说明](docs/RESULT_RESOURCE_GPU_0_8_0.md) |
| 模块公式、参数与输入输出 | [当前模块清单](docs/MODULE_INVENTORY_CURRENT.md) |
| 自定义算法示例 | [插件描述](examples/plugins/invert.json) · [Python 实现](examples/plugins/invert.py) |
| 开发与验证 | [贡献说明](CONTRIBUTING.md) |
| 版本变化与设计方向 | [更新日志](CHANGELOG.md) · [文档索引](docs/README.md) · [框架规划](FRAMEWORK_PLAN.md) |

## 当前边界

支持格式和科学含义以使用指南中的验证范围为准。当前不提供三维查看、编码视频导出、多用户服务、通用暂停恢复、模型训练或内置 ONNX 推理。第三方插件在本机执行；应用预算约束工作数组和缓存，进程还有解码与运行环境开销。

## 验证

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\benchmark_acceleration.py --backend both --output-dir .test-output
.\.venv\Scripts\python.exe tools\smoke_acceleration_0_9_0.py
.\.venv\Scripts\python.exe -m pytest tests\test_gpu_pipeline_0_8_4.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_paired_comparison_0_8_5.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_correlation_window.py -q
.\.venv\Scripts\python.exe tools\smoke_correlation_window.py
.\.venv\Scripts\python.exe tools\smoke_paired_comparison_0_8_5.py
.\.venv\Scripts\python.exe tools\smoke_gpu_pipeline_0_8_4.py
.\.venv\Scripts\python.exe tools\smoke_sidebar_data_0_8_3.py
.\.venv\Scripts\python.exe tools\smoke_result_interaction_0_8_1.py
~~~

浏览器验收在独立服务中生成模拟数据，Windows 默认使用已安装的 Edge；结果写入被忽略的 .test-output。GitHub Actions 在 Windows / Python 3.12 和 3.14 上执行 CPU 回归，并在 3.14 上验收侧栏与结果工作流。

## 许可

本仓库尚未指定开源许可证。
