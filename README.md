# Name That LLM Solution · GPU AI 交接工作区

目标：让有 GPU 的朋友及其 AI，复现用户指定的 [mohammedafreedpasha/name-that-llm-solution](https://www.kaggle.com/code/mohammedafreedpasha/name-that-llm-solution)，交回完整日志、可重跑 Notebook 和合格的 `submission.parquet`。

**朋友的 AI 从 [AI_START_HERE.md](AI_START_HERE.md) 开始。** 用户这边没有 GPU，GPU 执行由接收方完成。

这主要是**已有权重的推理与融合任务**，不是从零训练大模型。保留了原始 V25 的全部 7 个单元格、3 个模块、33 项输入引用，以及 Linux 入口、下载清单、预检、严格验收、Kaggle Notebook 和中文交付说明。Git 不包含大型权重和比赛数据。

> 说明文字声称 0.651，但实际代码使用 ATLAS 0.16、margin 0.38、Llama v5 权重 1.5、DeBERTa 1.3，且新增 GLM/Kimi 数据；与文字描述不一致。**本仓库没有 GPU 跑通或 0.651 成绩证据。** 详见 [源码审计](docs/SOURCE_AUDIT.md)。

## 快速开始

本工作区发布在既有仓库的独立分支，主分支保留原方案：

```bash
git clone --branch name-that-llm-solution-handoff --single-branch https://github.com/Kidding-Ma/ntl-complement-gpu-handoff.git name-that-llm-solution-handoff
cd name-that-llm-solution-handoff
python scripts/verify_workspace.py
python scripts/fetch_assets.py --list
# 然后读 AI_START_HERE.md 和 docs/GPU_RUNBOOK.md
```

Linux GPU 上的顺序：准备环境 → 下载/固定输入版本 → `preflight.py --gpu` → `run_pipeline.py` → `validate_submission.py` → Kaggle 断网重跑 → 按交付清单回传。没有 GPU 的电脑可以进行源码核验，不能据此验收模型运行。

## 文件导航

| 文件 | 用途 |
|---|---|
| [AI_START_HERE.md](AI_START_HERE.md) | 可直接交给朋友 AI 的任务书 |
| [docs/GPU_RUNBOOK.md](docs/GPU_RUNBOOK.md) | Linux / WSL2 GPU、环境、输入和完整执行 |
| [docs/INPUTS.md](docs/INPUTS.md) | 全部输入、版本限制、路径与下载 |
| [docs/SOURCE_AUDIT.md](docs/SOURCE_AUDIT.md) | 代码与文字差异、失败路径、已验证/未验证边界 |
| [docs/KAGGLE_SUBMISSION.md](docs/KAGGLE_SUBMISSION.md) | Notebook 导入、挂载输入、断网正式重跑 |
| [docs/DELIVERY.md](docs/DELIVERY.md) | 朋友应交回什么、怎样验收 |
| [notebooks/solution_strict.ipynb](notebooks/solution_strict.ipynb) | 自包含 Kaggle 派生版，加入错误检查，不改模型系数 |
| [references/upstream/source.ipynb](references/upstream/source.ipynb) | API 返回的原始 Notebook |
| [config/inputs.json](config/inputs.json) | 33 项机器可读输入清单 |
| [config/source-lock.json](config/source-lock.json) | 源码与单元格 SHA-256 |
| [NOTICE.md](NOTICE.md) | 第三方来源和许可边界 |

交接状态和本地检查证据见 [evidence/LOCAL_CHECKS.md](evidence/LOCAL_CHECKS.md)。源码快照日期：2026-10-27（北京时间）。
