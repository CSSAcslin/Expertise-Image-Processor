# 文档索引

## 使用

| 文档 | 用途 |
| --- | --- |
| [项目首页](../README.md) | 安装、启动与功能概览 |
| [使用指南](USER_GUIDE.md) | 打开、定位、处理、对比、导出与兼容边界 |
| [0.9.4 延迟扫描批量执行](DELAY_SCAN_0_9_4.md) | 连续信号交接、CPU／GPU跨τ计算、真实时间数组及完整18轮复测 |
| [0.9.3 通用处理流执行](PROCESSOR_EXECUTION_0_9_3.md) | 所需范围、空间及时间编辑、CPU / GPU批量、阶段交接与验证边界 |
| [0.9.2 设备监测与 GPU 计时](DEVICE_MONITOR_0_9_2.md) | 右下角 CPU / 内存 / GPU / 显存曲线、数值口径及 CUDA 事件覆盖范围 |
| [固定流程 CPU / GPU 直写对照](DIRECT_FLOW_COMPARISON_0_9_2.md) | 单点与ROI非模块化独立代码、39轮实测、完整输出核对及执行规划瓶颈 |
| [延迟扫描纯流程对照与优化方案](DELAY_SCAN_OPTIMIZATION_PLAN.md) | 相同3001个延迟的独立CPU／GPU实测、逐值来源核对、信号块和Q批量优化目标 |
| [GPU 落地与验收](GPU_DEPLOYMENT.md) | CUDA 13 依赖、启动解释器、强制真机测试和 RTX 5070 Ti 同机性能 |
| [0.9.1 后端执行与来源索引](BACKEND_EXECUTION_0_9_1.md) | 主路数据驻留、增量来源映射、原配置复跑及性能记录 |
| [0.9.0 运算加速与性能诊断](ACCELERATION_0_9_0.md) | float64 融合 g²、读取与分块复用、分段耗时、CPU / CUDA 基准 |
| [0.8.6 固定 τ 的 g² 窗长扫描](CORRELATION_WINDOW_0_8_6.md) | 固定中心、平均点数扫描、W 横轴、完整数值和图形导出 |
| [0.8.5 双幅时序与多曲线比较](PAIRED_COMPARISON_0_8_5.md) | 对应点 / 区域、左右时序、曲线清单、取值焦点与完整比较包 |
| [0.8.4 大序列处理与 GPU](GPU_PIPELINE_0_8_4.md) | 设备数据复用、批量传输、滚动窗口、精确统计与验证边界 |
| [0.8.3 布局与数据管理](SIDEBAR_LAYOUT_0_8_3.md) | 当前左侧双选项卡、输入 / 处理子栏目、紧凑任务栏与独立批选 |
| [0.8.2 数据管理](DATA_MANAGEMENT_0_8_2.md) | 历史底部布局与引用移除；布局已被 0.8.3 替代 |
| [0.8.1 结果交互](RESULT_INTERACTION_0_8_1.md) | 图标、搜索、复用与逐项删除的历史说明 |
| [0.8.0 结果、资源与 GPU](RESULT_RESOURCE_GPU_0_8_0.md) | 内存 / 显存预算、运行包、引用保护与可选加速 |
| [模块清单](MODULE_INVENTORY_CURRENT.md) | 当前公式、参数、类型、轴与依赖 |
| [模块工作台](MODULE_WORKBENCH_0_7_0.md) | 模块窗口、截流导出与项目迁移 |

## 开发与科学定义

| 文档 | 用途 |
| --- | --- |
| [贡献说明](../CONTRIBUTING.md) · [长期指引](../AGENTS.md) | 开发流程与设计约束 |
| [框架规划](../FRAMEWORK_PLAN.md) | 两个核心、阶段目标与待验证范围 |
| [运算时间优化更新方案](EXECUTION_TIME_OPTIMIZATION_PLAN.md) | 来源映射增量维护、阶段结果驻留、依赖释放、分块流水与性能验收 |
| [模块化处理流执行优化方案](PROCESSOR_EXECUTION_OPTIMIZATION_PLAN.md) | 依据39次实测，规划所需像素范围、静态几何复用、GPU批量、内存交接与分阶段验收 |
| [全模块执行能力与优化边界](MODULE_EXECUTION_CAPABILITIES.md) | 36个内置及兼容模块的空间 / 时间依赖、完整输入校验、旁路与插件约束 |
| [空间 / 时间契约](MODULE_CONTRACTS_0_6_0.md) | 真实形状、样本、时间与坐标 |
| [分幅关联专项](SPLIT_INTENSITY_CORRELATION_PLAN.md) | 标准 g²、ROI 定义、双向延迟与存储边界 |
| [资源改造规划](RESULT_RESOURCE_GPU_PLAN.md) | 结果管理、预算和 GPU 的实施边界 |
| [算法参考](../FROM_LIFECALOT.md) | 旧工具箱参考；未注册算法不属于当前可用模块 |
| [更新日志](../CHANGELOG.md) | 当前更新与历史版本入口 |

历史设计文档描述其当时的实现；操作以当前版本的首页、使用指南和模块清单为准。
