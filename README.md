# Expertise Image Processor

**0.8.6 · 本机 Web 工作台 · Python 3.12–3.14**

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
- **扩展算法**：统一模块契约与本机 Python 插件，保留完整参数、运行版本、样本映射和来源记录。

CPU 可完整运行。当前 26 个内置模块均可接入可选 CuPy / CUDA 后端，连续 GPU 节点复用设备数据，按预算批量传输；时间滑窗、统计、配准及标定具有设备实现。插件与旧序列统计仍使用 CPU，自动模式在 CUDA 不可用时回退。支持范围、预算与验证边界见 [0.8.4 大序列处理与 GPU](docs/GPU_PIPELINE_0_8_4.md)；实际 CUDA 硬件上的数值和速度仍需验证。

## 操作顺序

1. 导入数据，定位帧并核对原值；需要选区时添加 ROI 裁剪模块。
2. 添加模块并调整顺序，检查输入输出，再试算或正式运行。
3. 在左侧 **数据 → 处理** 中点击结果，与原数据对比；下载所需输出或作为下一阶段输入。任务记录位于中央底部窄栏，按需展开。

显示窗宽、伽马和色图只改变显示。任务记录保留数值定义、坐标和真实时间；未完成结果明确标记。默认仅监听本机，原始文件通过路径引用，数据处理在本机完成。

## 文档

| 内容 | 入口 |
| --- | --- |
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
