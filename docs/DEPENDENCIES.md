# 已评分 0.636 版本的输入说明

输入列表见 [已评分 metadata](../verified/score_0636/kernel-metadata.json) 和 [原始来源及版本](../verified/score_0636/MODEL_SOURCES.md)。以下是复用该固定配方时的路径与文件说明。Dataset/Notebook 的最新输出不一定等于当时输入。

## 高优先级输入

| 部件 | 原来源/路径契约 | 必须核实 |
|---|---|---|
| 官方输入 | `/kaggle/input/competitions/name-that-llm/{train.csv,test.csv,sample_submission.parquet}` | train 包含 id/prompt_id/text/model；test 为 id/text；正式测试由 Kaggle 挂载 |
| 主 two-stage part-two | `nishantdahal/ntl-handwriting-blend` | `pool_v4.parquet`、`lora_qwen3_14b/meta.json`、`lora_qwen25_14b/meta.json`、`deberta_v3_large/meta.json` 及全部关联 tensors/adapters/tokenizer |
| 主基础模型 | Qwen3 14B Base V1；Qwen2.5 14B V1 | config type/hidden/layers 符合原代码；Qwen3 EOS 必须读取训练 meta，不能默认 chat EOS |
| legacy LoRA24 | `fleongg/ntl-legacy-qwen24-recovery` | 唯一 `legacy_recovery_manifest.json`、其中全部 flat_file 与 SHA-256；原代码恢复到 `/kaggle/working/recovered_legacy/lora24/final` |
| complement 文本池 | `fleongg/ntl-quality-selected-public-svc-pool` 及其余附加文本集 | 源码使用的通配符可能读取多来源；实际匹配文件和行数须记录，不能任意换 pool |
| LoRA seed7 | `lucasboesen/name-that-llm-train-lora24-s7` Output | `lora_out/job_final/COMPLETE.json`、norms、valid_features、adapter、meta 等 |
| Claude LoRA | `zbigniewkirchner/ntl-qwen3-8b-lora-claude-head` | `**/final/COMPLETE.json`、heads、anchor_features、adapter 等 |
| Qwen2.5 7B instyle | `vnton37nguyn/ntl-train-lora-qwen25-instyle` Output | `lora_out/job_final/COMPLETE.json`、norms、valid_features 等 |
| frozen GLM | `lucasboesen/name-that-llm-glm4-9b-trunc25` | GLM4/4096 config 和真实权重/tokenizer |
| encoder 三组 | `name-that-llm-extract-encoders2`、electra/desklib outputs | `deberta-v3-large`、`electra-large-discriminator`、`desklib-detector` 子目录 |
| token attention 两组 | `name-that-llm-train-tokattn` / `tokattn2` | `tokattn_final.pt`、`tokattn_debl23_final.pt`；按 head 内 keep/layer/max_len 的原契约加载 |
| Qwen3.5 底座 | `bbucxi/qwen35-9b-safetensors` | config、权重 shards/index、tokenizer；加载日志与一条 forward |
| UltraBERT | `pjmathematician/ultrabert-380m-ntl` | `ultrabert-380m-ntl/fold*` 五折，模型代码/配置/权重和 id2label |
| ATLAS | `josephadamski91/ntl-atlas-qwen3-8b-v1` | 唯一 atlas_bundle.json、adapters.safetensors、ntl_atlas/model.py、ntl_rlcd/data.py 及各自 import 的完整包；Qwen3 8B Base V1 |
| 环境离线依赖 | `dhruvtibarewal/nlm-offline-wheels` | bitsandbytes wheel 与目标 Python/CUDA 兼容；线上准备和 Kaggle 禁网运行是两个阶段 |

## 原配方的融合核对项

main labels 顺序与 complement 的排序顺序不同；原融合有显式重排。保留重排、EOS、pool、截断层数、norms、head 和 blend 权重。精确参数见已评分 Notebook。

必需 complement：`stylegram, stylometry_lgb, debv3l, electl, desk, lora24_public, lora24_seed7, lora24_claude, lora20_q25_instyle, tokattn_q35, tokattn_deb, ultrabert`。

## 路径布局

基础模型示例：`/kaggle/input/models/qwen-lm/qwen-3/Transformers/8b-base/1/config.json`；Dataset 示例：`/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend/`；Notebook 输出示例：`/kaggle/input/notebooks/lucasboesen/name-that-llm-train-lora24-s7/lora_out/job_final/`。原代码包含依赖这些路径片段的发现逻辑，重新布局时需核对实际选中的文件。
