# 有效提交：Public Score 0.636

本目录直接提供实际代码、输入清单与成绩证据。

| 项目 | 记录 |
|---|---|
| 正式提交号 | 57706812 |
| 状态 | COMPLETE |
| Public Score | 0.636 |
| Private Score | 尚未公开 |
| 固定 Notebook 版本 | `liuqingmu096/ntl-complement-fixed-3b8e333/1` |
| 源码 | [ntl_complement_fixed.ipynb](ntl_complement_fixed.ipynb) |
| 输入 | [kernel-metadata.json](kernel-metadata.json)、[MODEL_SOURCES.md](MODEL_SOURCES.md) |
| 成绩依据 | [submission_status.json](submission_status.json)、[submission_receipt.json](submission_receipt.json) |
| 运行证据 | [RESULTS.md](RESULTS.md)、[独立概率验收](independent-validation.json)、[交付验证](delivery_verification.json) |

公开预览中主分支四项、12 个 complement 成员及 ATLAS 完成，整套概率混合独立复算通过。双 T4 禁网公开预览约 2073.33 秒，**不是隐藏全量耗时**。`independent-validation.json` 产生在正式提交前，其中 `competition_submitted=false` 与后来的提交成功不矛盾。

复用时导入此目录 Notebook，按 metadata 挂载比赛、Dataset、Notebook Output 和模型，核对版本及兼容环境。基础模型和第三方权重从 MODEL_SOURCES 原渠道取得。根目录早期 ready Notebook 不是本次已评分固定版本。公开三行 parquet 不作为隐藏测试答案再次上传。

原报告保留当时 JST 时间说明，不改写原始回执。部分早期输入未固定历史版本，不能承诺与作者历史运行逐字节一致。
