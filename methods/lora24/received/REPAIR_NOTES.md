# LoRA24 集成候选的修复记录

文件：`own_lora24_ensemble_fixed.ipynb`。尚无该候选完整运行或正式评分证据。

原 `find_checkpoint` 只按 model_type/hidden_size 查找，可能误选仅包含配置的 `own_lora24/final/base_config`。修复后的 Notebook 检查真实权重和全部 shard，排除 adapter/tokenizer/base_config 目录，核对 Qwen3-8B Base 的 36 层结构。

Notebook 包含输入预检、GPU 锚点 parity 检查、空文本池化保护和批次日志，保留原集成成员与融合系数。`repair_validation.json` 是当时的静态/路径检查结果；`final_parity.json` 属于收到的 final 权重重载结果，两者不代表整套候选取得 LB。

预检、ensemble 和内嵌 Gaussian 的完整代码均在 Notebook，不再保留重复导出文件。
