# GPU 安装、验收与 RTX 5070 Ti 实测

2026-10-27 在 Windows 11、Ryzen 9 9950X、RTX 5070 Ti（16303 MiB）上完成真实 CUDA 验证。项目使用 Python 3.12.10、CuPy 14.2.0、NumPy 2.4.6、驱动 616.56 和已安装的 CUDA 13.4 Toolkit。CuPy 构建 / 链接运行时为 13.2，实际发现的本机运行时和 NVRTC 为 13.4；完整版本保存在诊断记录中。

GPU 路径服务于同一预览、模块检查、试算、正式处理和结果导出流程。CPU 保持完整可用，显示映射不改变计算输入，g² 仍采用 float64 共同有效样本累计。

## 统一解释器

`start.cmd` 默认使用仓库内 `.venv\Scripts\python.exe`。父目录的 `.venv` 是另一个环境，在那里安装 CuPy 不会让仓库内解释器自动获得依赖。

已有系统 CUDA 13 Toolkit 时，在仓库根目录执行：

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu13.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools/check_gpu.py --output .test-output/gpu-check.json
.\start.cmd
~~~

项目也提供 `gpu13` 可选依赖；原有 `gpu` 仍对应 CUDA 12。一个环境只能安装一种 CuPy 分发包。没有系统 Toolkit 的新环境可使用 CuPy 官方提供的 `cupy-cuda13x[ctk]` 安装匹配的 CUDA 组件，详见 [CuPy 安装说明](https://docs.cupy.dev/en/stable/install.html)。不需要为了切换应用环境而修改另一份虚拟环境。

需要明确复用父目录的现成环境时，可在当前 PowerShell 窗口设置：

~~~powershell
$env:EIP_PYTHON = 'D:\coding\E\.venv\Scripts\python.exe'
.\start.cmd
~~~

该变量只影响当前终端及其子进程。启动窗口会打印实际解释器；`start.cmd` 也会转发 `--port`、`--workspace` 等参数。安装、诊断与测试均须使用同一个解释器。工作台的「运行设置 → 运行环境」也能核对路径。

## 启用与检查

1. 打开「运行设置」，点「重新检测」，应显示 `CUDA · NVIDIA GeForce RTX 5070 Ti`。
2. 选择默认设备 GPU 并保存；需要统一当前流程时使用「流程继承统一资源」。实例中明确指定 CPU 的节点继续使用 CPU。
3. 显存上限是预算，并不预先占满显存；正式任务与阶段检查按既有 3:1 份额执行。预算不足时调整预算、范围或显式选择 CPU。
4. 运行后检查任务记录中的各实例实际设备、资源计划、传输计数和关联实现；融合 g² 应记录 `cupy_fused_float64`。未知纯内核耗时继续保存为 null，不把主机等待当成内核时间。

本机已保存的资源设置本身选择了 GPU，本轮补齐了默认启动解释器中的依赖。运行中的旧服务可通过「重新检测」刷新设备发现。

`tools/check_gpu.py` 执行真实 float64 内核和 EIP 融合 g²，检查非连续双路输入、负 / 零 / 正延迟、共同有效样本数及独立公式；失败返回非零状态，不回退成 CPU。它不更改工作区设置或实验数据。

## 可重复验收

~~~powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q --require-cuda
.\.venv\Scripts\python.exe tools/smoke_acceleration_0_9_0.py --backend gpu
.\.venv\Scripts\python.exe tools/benchmark_acceleration.py --backend both --require-cuda --repeats 3 --output-dir .test-output
~~~

本机最终完整回归 **415 项通过、0 失败、0 跳过**，`pip check` 无依赖冲突。普通 CPU CI 仍允许跳过真实 CUDA 检查；`--require-cuda` 会在 CUDA 缺失或 float64 编译执行失败时直接报错。新增测试覆盖全部 27 个可见模块，比较真实 CuPy 与 CPU 的轴、形状、dtype、无效值、旁路输出和反向访问；既有融合内核专项另检查微小涨落、非连续批量、窗口移除及无效样本。

浏览器验收使用隔离工作区与模拟 TIFF，在 Edge 中检查 GPU 设置、实际解释器、驱动显存读数、正式 GPU g²(W)、独立进程的两个阶段检查、GPU 试算与正式结果完全一致、NPY 下载与独立公式、运行记录及大小窗口布局。默认 `--backend cpu` 入口也已复查。W 输出率与原始帧率保持区分。

## 同机性能结果

每行使用相同合成输入、模块、参数及同一解释器，CPU / GPU 各启动独立进程运行三次，取中位数。任务函数耗时包含读取、计算、CUDA 上下文准备、数值写盘及应有的 PNG / SVG 导出，排除 Python 进程启动；进程总耗时另列。系统文件缓存及 CUDA 磁盘编译缓存已热，不是冷盘或纯内核基准。

所有场景 τ=+1、固定中心、uint16 输入、每幅为下表尺寸，W 从 2 起、步长 2。主机预算 2048 MiB、预览 16 MiB、计算缓存 128 MiB；GPU 预算依次为 1024、2048、4096 MiB。

| 每幅尺寸 / 帧数 / 最大 W | 流程（均先固定双幅配准） | CPU 任务秒 | GPU 任务秒 | CPU/GPU | CPU / GPU 进程总秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| 128×128 / 769 / 600 | g²(W) 系数图 | 0.283 | 0.652 | 0.43× | 0.719 / 1.126 |
| 同上 | 取点 → g²(W) | 0.522 | 0.933 | 0.56× | 1.001 / 1.432 |
| 同上 | g²(W) → 像素空间均值 | 0.594 | 1.034 | 0.57× | 1.045 / 1.539 |
| 512×512 / 129 / 64 | g²(W) 系数图 | 0.638 | 0.523 | 1.22× | 1.078 / 0.992 |
| 同上 | 取点 → g²(W) | 0.614 | 0.782 | 0.79× | 1.071 / 1.293 |
| 同上 | g²(W) → 像素空间均值 | 0.927 | 0.825 | 1.12× | 1.394 / 1.334 |
| 768×768 / 65 / 62 | g²(W) 系数图 | 1.379 | 0.673 | 2.05× | 1.826 / 1.164 |

小图和取点任务在这台电脑上仍可能 CPU 更快；较大系数图才有明显 GPU 收益。不会据此自动移动模块、降低精度或更换用户选择的设备。三组共 42 次正式运行全部成功，按首 / 中 / 末 W 独立公式抽查，最大绝对误差 2.22×10⁻¹⁶、有限值掩码一致；这不是对全部实验数据或其他 GPU 的速度保证。

可共享的逐次耗时、资源和执行计数摘要见 [RTX 5070 Ti 基准记录](benchmarks/gpu_rtx5070ti_20261027.json)。完整运行包、诊断与浏览器截图保留在本机被忽略的 `.test-output` 目录。前两组默认三流程、第三组仅系数图，复测参数如下：

~~~powershell
.\.venv\Scripts\python.exe tools/benchmark_acceleration.py --backend both --require-cuda --repeats 3 --frames 769 --height 128 --width 128 --window-max 600 --gpu-mb 1024 --output-dir .test-output
.\.venv\Scripts\python.exe tools/benchmark_acceleration.py --backend both --require-cuda --repeats 3 --frames 129 --height 512 --width 512 --window-max 64 --gpu-mb 2048 --output-dir .test-output
.\.venv\Scripts\python.exe tools/benchmark_acceleration.py --backend both --require-cuda --repeats 3 --frames 65 --height 768 --width 768 --window-max 62 --cases image --gpu-mb 4096 --output-dir .test-output
~~~
