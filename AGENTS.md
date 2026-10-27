# 工作区约定

- 从 `AI_START_HERE.md` 开始，按需查看 `docs/GPU_RUNBOOK.md`、`docs/INPUTS.md` 和 `docs/SOURCE_AUDIT.md`。
- 当前用户提供本机计算资源；原项目发起人没有 GPU。先完成本机运行与交付，再准备 Kaggle 验证。正式评分提交由当前用户决定。
- 基线是仓库保存的 V25 实际代码。先使用已发布模型/适配器完成推理；上游文字中的 0.651 不是本次验证结果。
- `references/upstream/` 与 `src/` 保留为原始快照，使用 `config/source-lock.json` 核对。兼容修复放在运行入口或派生副本，交付时附 diff。
- 可以自主处理环境、路径和批大小适配，记录改动。若改变精度、模型、数据、系数或后处理，另行标记实验，便于与基线比较。
- 运行完成后核对组件日志和输出。报告分别说明本地执行、格式检查、负载探测和官方评分的实际结果。
- 输入权限阻塞时记录具体资源与错误，继续可独立完成的工作。凭据和大型权重不进入 Git。
- 在 `deliverables/REPORT.md` 汇总结果，参照 `docs/DELIVERY.md` 和 `deliverables/REPORT_TEMPLATE.md` 整理交付物。
