# 贡献与开发

改动围绕 **视频数据预览** 与 **数据处理**。开始前阅读 [AGENTS.md](AGENTS.md)、[框架规划](FRAMEWORK_PLAN.md) 和 [当前模块清单](docs/MODULE_INVENTORY_CURRENT.md)。

## 本地环境

使用 Python 3.12–3.14，在仓库根目录执行：

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe run_web.py --no-browser
~~~

默认地址为 http://127.0.0.1:8765。前端位于 src/eip/web/static，直接加载，无需打包。Web 不依赖 Qt；需要验证 Qt 时安装 requirements-desktop.txt。可选 GPU 依赖通过项目 gpu 扩展安装，配置见 [资源与 GPU](docs/RESULT_RESOURCE_GPU_0_8_0.md)。

## 实现约束

- 普通播放只读取原始数据；显示调整不改变算法输入。
- 模块流程是执行配置的唯一来源。新增算法声明类型、轴、作用范围、参数、版本和科学含义，不能绕过公共执行器。
- 试算与正式处理共用实现；结果保留真实时间、贡献样本、坐标和来源。
- 大数据按需读取，工作数组和缓存有预算；CPU 保持完整可用，GPU 可选。
- 结果复用与清理保留引用检查和可追溯记录。

## 验证

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\smoke_data_management_0_8_2.py
.\.venv\Scripts\python.exe tools\smoke_result_interaction_0_8_1.py
.\.venv\Scripts\python.exe tools\smoke_result_resources_gpu.py
.\.venv\Scripts\python.exe tools\smoke_module_revision_0_7_4.py
~~~

选择与改动相关的回归；科学计算应以独立参考公式核对数值、无效值、精度、时间 / 坐标与来源，避免只检查界面文本。CPU 测试生成小型模拟数据；CUDA 测试在没有硬件时明确跳过。浏览器验收需要 Playwright，Windows 使用已安装的 Edge，服务和数据写入独立的 .test-output 子目录。

更新模块定义后，用 tools/export_module_inventory.py 同步清单，并在 CHANGELOG.md 记录可见变化。界面改动检查大 / 小窗口、键盘操作、图标提示及现有公共播放器。

## 提交与问题报告

在独立分支开发；本项目自动开发分支使用 codex/ 前缀。PR 说明具体触发条件、最终行为、验证结果和未验证边界。

报告问题时提供版本、Python / 系统、格式与轴尺寸、模块顺序 / 参数、最小复现步骤及错误日志。尽量使用模拟数据；可复现的最小样例比完整实验数据更有用。

本地实验数据、输出、虚拟环境、工作区 JSON、维护会话、凭据及缓存不进入版本库。自定义插件只有经确认可公开的代码和描述才提交。当前仓库未指定开源许可证。
