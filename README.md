# NTL Part Two Complement · GPU AI 交接工作区

目标：复现 [fleongg/ntl-part-two-complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement)，修复输入权限、模型加载和完整推理问题，交回可在 Kaggle 隐藏测试重跑的 Notebook。**请 AI 先读 [AI_START_HERE.md](AI_START_HERE.md)。**

关键词：`NTL Part Two Complement`、`0.636`、`legacy_recovery_manifest.json`、`lora24_public`、`Qwen3.5`、`tokattn_q35`、`HF_DEACTIVATE_ASYNC_LOAD`、`ATLAS`、`Sinkhorn`、`submission.parquet`。

## 仓库交付范围

- 当前公开 V2 的完整 24 个单元格、原始 Python 导出、作者说明和官方输入元数据。
- AST 提取的 6 个辅助模块，包括完整 `comp_stack.py`；同名模块以最后一次写入为准，写入顺序有记录。
- 16 个 Dataset、8 个 Notebook Output、7 个 Model、1 个 Competition 的完整下载清单和 Kaggle 元数据模板。
- 官方 `train.csv`、三行公开 `test.csv`、`sample_submission.parquet`；来源和 SHA-256 随附。
- Linux GPU 运行入口、输入预检、下载工具、结果验证器、Notebook schema 修复版，以及既有 Qwen3.5 停滞诊断。
- 可获取的参考训练/特征提取 Notebook 源码、不可获取项的真实错误记录。

**源码交接完整不等于已包含所有权重。大型基础模型、第三方权重和文本池仍须下载；不能访问的原权重不能凭空补齐。本仓库没有 GPU 跑通证据，也不保证 0.636。** 逐项见 [DEPENDENCIES.md](docs/DEPENDENCIES.md)、[CURRENT_STATE.md](docs/CURRENT_STATE.md)。

## 开始

```bash
git clone https://github.com/Kidding-Ma/ntl-complement-gpu-handoff.git
cd ntl-complement-gpu-handoff
python scripts/verify_workspace.py
python scripts/fetch_assets.py --list
# 然后按 docs/GPU_RUNBOOK.md 准备 Linux GPU、下载输入并执行
```

对朋友的 AI：请直接把链接和根目录 `AI_START_HERE.md` 一起给它；不要仅凭 README 判断完成。完整验收要求在任务书内。

## 重要区别

作者称 0.636 来自 V1 提交 57551442。当前 V2 是发布版本：页面日志显示 CPU、9.1 秒、输出 0 B；API 却保留 enableGpu=true。两者按不同证据记录，均不能证明接收方成功复现。三行公开预测只验证格式，正式成绩需要 Kaggle 隐藏测试重新执行。

第三方代码、模型和数据分别保留原作者及原条款，见 [NOTICE.md](NOTICE.md)。
