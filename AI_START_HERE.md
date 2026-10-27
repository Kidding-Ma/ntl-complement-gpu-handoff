# 给接收方 AI 的完整任务书

你运行在朋友有 NVIDIA GPU 的电脑上。委托人没有 GPU。请实际执行并交付，不要只输出建议。

## 目标和成功条件

复现用户给定的 Kaggle Notebook 当前已固定的 V25 源码，完成已有模型/LoRA/编码器/词法统计/ATLAS/后处理的完整推理。优先复现本快照，避免追踪 latest 引入漂移。最终交回：完整运行证据、合格的提交文件、可在 Kaggle 隐藏测试断网重跑的 Notebook、输入版本与环境清单，以及中文结果报告。成绩只认实际 Kaggle 官方回执。

本仓库的原始说明有复制来的 0.651 声明，**不是本 V25 的已验证成绩**。先读 docs/SOURCE_AUDIT.md。第一阶段无需重训适配器，不得以换成其他作者配方代替完成任务。

## 执行顺序

1. `python scripts/verify_workspace.py`，再读 docs/GPU_RUNBOOK.md、docs/INPUTS.md、docs/DELIVERY.md。
2. 调查本机 GPU 型号/显存、驱动、CUDA、内存、磁盘；记录 `nvidia-smi`，确认 Python 和 torch 可用。先用已有硬件，不购买云资源。
3. 对照 `config/inputs.json`，下载 16 Dataset、9 Notebook Outputs、7 Model versions、1 Competition。自备有权限的 Kaggle 账号和本机凭据；不向委托人索要或上传密钥。首次参加比赛需由账号持有人接受规则。
4. 模型输入固定为清单里的版本 1。Dataset/Notebook 的挂载版本未由 pull API 公开，必须从 Input 面板/历史版本确认；无法确认时记录实际下载版本与 SHA-256，并明确写“版本未完全锁定”。下载 latest 不能冒充原版本。
5. `python scripts/preflight.py --input-root /absolute/input --gpu --output preflight.json`。缺文件/路径歧义时先解决，不直接启动数小时推理。Llama 必须是 base 3.1 8B，按缓存的验证报告检查字节身份。
6. `python scripts/run_pipeline.py --input-root /absolute/input --work-dir /absolute/run-001`。使用全新运行目录。保存标准输出、各组件日志、GPU 状态、环境 freeze。原始三个模块不可直接改写；兼容补丁单独记录。
7. 修复实际发生的环境/路径错误后重跑。不得让缺失的必需组件、ATLAS SKIPPED、COMP SKIPPED 或 traceback 被忽略。历史 `lora24_public` 缺权重在原注释中可跳过，但也必须记录；其他 LoRA 不得默默跳过。原代码有意关闭 Qwen3.5 token-attention，保留 `NTL_DROP_Q35=1`。
8. 用验证器检查 submission；检查 18 类、ID 对齐、无空值和重复、comp logits 形状与有限值、ATLAS 的实际混入。公开 test 只做格式检查，少于 18/50 行时部分后处理分支不会被覆盖。可用原 `--fake_test` 做负载探测，但不能称为成绩。
9. 导入 `notebooks/solution_strict.ipynb` 到 Kaggle，附齐来源，在可用 GPU 和限时内执行断网 Save & Run All。用正式测试入口，不带模拟开关。需要账号持有人授权/完成的比赛提交由其办理。保留准确 version/script ID、运行和评分回执。
10. 填写 deliverables/REPORT_TEMPLATE.md，按 docs/DELIVERY.md 返回源码、Notebook、日志、版本、hash、环境及结果。若确实被权重权限阻塞，返回可复查阻塞清单，状态为 blocked/incomplete。

## 自主处理范围

允许：在现有 GPU 上安装公开 Python 依赖、只读获取公开输入、修复路径/兼容性、完善日志、在独立分支保存补丁、做可复现的运行检查。

另开实验且先标记：改 batch/精度/模型深度、重训、改系数/阈值/数据、启用已禁用组件、删掉失败成员、换基础模型。先完成基线；OOM 可调 batch 但需列出差异并重跑验收。

不要把训练行自检当 GroupKFold OOF；如果后续研究新方案，按 prompt_id 分组验证，保留未调参评估集。不要承诺 0.651 或编造显存峰值和耗时。
