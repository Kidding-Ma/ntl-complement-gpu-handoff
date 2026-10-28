# 有效本地材料筛选

2026-10-28 从 `D:\Kaggle` 核查，仅新增与方案、实验和复现有关的内容。逐文件原位置、SHA-256 和用途见 [curated_sources.json](../manifests/curated_sources.json)。

| 本地材料 | 上传内容 | 用途 |
|---|---|---|
| Received-NTL-Complement-20261028 | 已评分 Notebook、输入 metadata、来源、成绩回执、运行报告、概率验收和环境 | 直接保存有效 0.636 提交证据 |
| Received-LoRA24-20261023/deliverables | final 权重 ZIP、校验、OOF 汇总、parity、集成修复 | 复用独立训练成果，权重通过 Git LFS |
| NameThatLLM-LoRA-Publish | 核心训练/推理/校验、配置、依赖和许可证 | 重训及独立新路线的可复用代码 |
| NameThatLLM-GoldRelease | 少量 CPU 控制、风格特征、审计及对应历史指标 | 独立对照或新方法补充 |
| name-that-llm-solution-handoff | 已有 GitHub 固定分支链接、简短源码审计 | 另一公开方案不重复复制 |

不新增：重复交接压缩包/clone/Upload-Ready 副本；截图和浏览器自动化脚本；缓存、`.deps`、`__pycache__`；论文 PDF/全文提取；旧提示词和过时泛泛研究报告；公开三行 submission 和成员小概率包；第三方基础模型/原始文本池。

这些内容或重复、或无法支持泛化结论、或与实验无关。第三方大型输入有原始下载地址，不重复托管。旧主分支的 upstream/references 和比赛快照保持原样，本次不新增原始比赛数据。

仍需明确的缺口：LoRA24 缺完整逐折预测/逐折权重；新 A/B/C 等路线尚未实现或实测；另一条 V25 未获得完整 GPU/正式评分证据；第三方输入权限与部分未固定历史版本需由执行环境核对。当前资料只证明已保存回执的有效提交，不冒充完整账号历史审计。

主要交付是[详细多方案计划](MULTI_METHOD_PLAN_ZH.md)，可以选择完全独立的新方法。
