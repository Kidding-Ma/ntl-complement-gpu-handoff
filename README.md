# Name That LLM：代码、权重和实验记录

比赛：[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)。

| 资料 | 内容与结果 |
|---|---|
| [有效提交](verified/score_0636/README.md) | 完整 Notebook、32 项输入来源、运行报告和分数回执；提交 57706812，Public Score **0.636** |
| [独立 LoRA24](methods/lora24/README.md) | 已训练权重、训练/推理代码、配置；交付报告 plain-head OOF BA **0.479341**，没有正式 LB；完整逐折预测尚缺 |
| [CPU 分类与风格特征](methods/cpu/README.md) | prompt 分组验证、文本/数值特征、数据审计；历史分组 OOF BA 约 **0.384484 / 0.388936** |
| [公开模型训练和特征提取代码](references/) | 编码器、LoRA、token attention 等原始 Notebook 和来源 metadata |
| [另一公开集成的固定源码](https://github.com/Kidding-Ma/ntl-complement-gpu-handoff/tree/debdb0781600566bd4d211ea7e2f324546124d92/src) | V25 实际代码；未验证该版本正式成绩，差异见 [源码审计](evidence/v25_source_audit.md) |
| [官方数据快照](data/competition/) | 2699 条训练数据、150 个 prompt、18 类；test.csv 仅 3 条公开预览，来源和校验见 provenance.json |

LoRA24 权重用 Git LFS 保存，下载和使用命令见其说明。基础模型和第三方输入从原始来源取得。

文件来源与 SHA-256：[curated_sources.json](manifests/curated_sources.json)。文件完整性检查：`python scripts/verify_workspace.py`。原作者与许可见 [NOTICE.md](NOTICE.md)。
