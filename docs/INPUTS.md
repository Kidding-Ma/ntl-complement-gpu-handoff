# 输入清单、版本与目录

机器可读权威清单：config/inputs.json。当前原 metadata 共 **16 Dataset + 9 Notebook Output + 7 Model + 1 Competition = 33**。这些是引用，不代表资产已经包含/下载/有权限。

## 版本锁定

七个 Model 引用明确为版本 1。Llama cache 在原 Markdown 明确指向 Notebook v1（script 362275911），清单按该版本下载。其余 Dataset/Notebook 的 attached selection 版本未在 pull API 暴露，version=null；不能拿“当前最新”冒充原挂载版本。参考源码的 currentVersionNumber 也不自动等于挂载的 output 版本。接收方可复制 inputs.json 到 config/inputs.resolved.json，逐项确认后填写 version，保留来源证据。

## 全部输入

| 类型 | 来源 | 版本 | 下载后相对 input 路径 | 用途/核对 |
|---|---|---|---|---|
| dataset | [dhruvtibarewal/nlm-l24-adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters) | 待解析 | `nlm-l24-adapters` | 旧 two-stage adapters：ts_q3b8/ts_q3b14/ts_q25_7b/ts_llama8、seed7、heads/pool；按实际文件核对 |
| dataset | [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) | 待解析 | `ntl-extra-api-passages` | 公开额外文本池，互补栈按文件名加载 |
| dataset | [abdullah1afify/ntl-generations](https://www.kaggle.com/datasets/abdullah1afify/ntl-generations) | 待解析 | `ntl-generations` | 公开额外文本池，互补栈按文件名加载 |
| dataset | [dophuchung/ntl-proxy-generations](https://www.kaggle.com/datasets/dophuchung/ntl-proxy-generations) | 待解析 | `ntl-proxy-generations` | 公开额外文本池，互补栈按文件名加载 |
| dataset | [zbigniewkirchner/ntl-qwen3-8b-lora-claude-head](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head) | 待解析 | `ntl-qwen3-8b-lora-claude-head` | final/COMPLETE.json 与 Claude LoRA 头 |
| dataset | [dhruvtibarewal/nlm-matched-clean](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean) | 待解析 | `nlm-matched-clean` | nlm_matched_clean.parquet，主栈 extra |
| dataset | [dhruvtibarewal/nlm-offline-wheels](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-offline-wheels) | 待解析 | `nlm-offline-wheels` | bitsandbytes 离线 wheel |
| dataset | [josephadamski91/ntl-atlas-qwen3-8b-v1](https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1) | 待解析 | `ntl-atlas-qwen3-8b-v1` | atlas_bundle.json、adapters.safetensors、ntl_atlas/ntl_rlcd Python 包 |
| dataset | [lucasboesen/ntl-free-proxy-generations](https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations) | 待解析 | `ntl-free-proxy-generations` | 公开额外文本池，互补栈按文件名加载 |
| dataset | [danielandresmiranda/ntl-openrouter-glm-kimi](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi) | 待解析 | `ntl-openrouter-glm-kimi` | ntl_openrouter_glm_kimi.csv；会改变主栈 extra |
| dataset | [kirilltym/ntl-spectral-style-heads-paired-4404](https://www.kaggle.com/datasets/kirilltym/ntl-spectral-style-heads-paired-4404) | 待解析 | `ntl-spectral-style-heads-paired-4404` | 互补统计/风格资源；保留原输入 |
| dataset | [nishantdahal/ntl-two-stage-lora](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora) | 待解析 | `ntl-two-stage-lora` | lora8b_v1/lora14b_v1 meta 和 weights |
| dataset | [nishantdahal/ntl-handwriting-blend](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend) | 待解析 | `ntl-handwriting-blend` | lora_qwen3_14b/lora_qwen25_14b/deberta_v3_large |
| dataset | [dhruvtibarewal/nlm-l24-adapters-v5](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5) | 待解析 | `nlm-l24-adapters-v5` | 18 个 q8v5/q14v5/ll8v5 fold/full adapters |
| dataset | [prvsiyan/prvsiyans-extended-name-that-llm-dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) | 待解析 | `prvsiyans-extended-name-that-llm-dataset` | 公开额外文本池，互补栈按文件名加载 |
| dataset | [pjmathematician/ultrabert-380m-ntl](https://www.kaggle.com/datasets/pjmathematician/ultrabert-380m-ntl) | 待解析 | `ultrabert-380m-ntl` | 5 个 fold/config + tokenizer + model shards |
| kernel | [lucasboesen/name-that-llm-glm4-9b-trunc25](https://www.kaggle.com/code/lucasboesen/name-that-llm-glm4-9b-trunc25) | 待解析 | `name-that-llm-glm4-9b-trunc25` | 互补编码器/GLM 权重或特征；保留输出目录 |
| kernel | [lucasboesen/name-that-llm-extract-encoders2](https://www.kaggle.com/code/lucasboesen/name-that-llm-extract-encoders2) | 待解析 | `name-that-llm-extract-encoders2` | 互补编码器/GLM 权重或特征；保留输出目录 |
| kernel | [lucasboesen/name-that-llm-electra-large-weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-electra-large-weights) | 待解析 | `name-that-llm-electra-large-weights` | 互补编码器/GLM 权重或特征；保留输出目录 |
| kernel | [lucasboesen/name-that-llm-desklib-weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-desklib-weights) | 待解析 | `name-that-llm-desklib-weights` | 互补编码器/GLM 权重或特征；保留输出目录 |
| kernel | [lucasboesen/name-that-llm-train-lora24-s7](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7) | 待解析 | `name-that-llm-train-lora24-s7` | lora_out/job_final/COMPLETE.json + adapter/norms/readouts/tokenizer/features |
| kernel | [lucasboesen/name-that-llm-train-tokattn](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn) | 待解析 | `name-that-llm-train-tokattn` | token-attention 输出；Qwen3.5 分支已按原配置关闭 |
| kernel | [lucasboesen/name-that-llm-train-tokattn2](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn2) | 待解析 | `name-that-llm-train-tokattn2` | tokattn_debl23_final.pt |
| kernel | [vnton37nguyn/ntl-train-lora-qwen25-instyle](https://www.kaggle.com/code/vnton37nguyn/ntl-train-lora-qwen25-instyle) | 待解析 | `ntl-train-lora-qwen25-instyle` | lora_out/job_final 的 Qwen2.5 instyle LoRA |
| kernel | [avazaku/name-that-llm-verified-llama-asset-cache](https://www.kaggle.com/code/avazaku/name-that-llm-verified-llama-asset-cache) | 1 | `name-that-llm-verified-llama-asset-cache` | base Llama 3.1 8B；llama-3.1/8b；验证报告和许可证 |
| model | [qwen-lm/qwen-3/Transformers/8b/1](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/8b/1) | 1 | `qwen-3/8b/transformers/1` | 基础模型 config/tokenizer/全部 shards |
| model | [qwen-lm/qwen-3/Transformers/14b/1](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/14b/1) | 1 | `qwen-3/14b/transformers/1` | 基础模型 config/tokenizer/全部 shards |
| model | [qwen-lm/qwen-3/Transformers/8b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/8b-base/1) | 1 | `qwen-3/8b-base/transformers/1` | 基础模型 config/tokenizer/全部 shards |
| model | [qwen-lm/qwen-3/Transformers/14b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/14b-base/1) | 1 | `qwen-3/14b-base/transformers/1` | 基础模型 config/tokenizer/全部 shards |
| model | [qwen-lm/qwen-3/Transformers/4b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/4b-base/1) | 1 | `qwen-3/4b-base/transformers/1` | 基础模型 config/tokenizer/全部 shards |
| model | [qwen-lm/qwen2.5/Transformers/7b/1](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/7b/1) | 1 | `qwen2.5/7b/transformers/1` | 基础模型 config/tokenizer/全部 shards |
| model | [qwen-lm/qwen2.5/Transformers/14b/1](https://www.kaggle.com/models/qwen-lm/qwen2.5/Transformers/14b/1) | 1 | `qwen2.5/14b/transformers/1` | 基础模型 config/tokenizer/全部 shards |
| competition | [name-that-llm](https://www.kaggle.com/competitions/name-that-llm) | 待解析 | `competitions/name-that-llm` | 官方 train.csv/test.csv/sample_submission |

## 目录要求

原源码直接 glob `/kaggle/input/**`，并用 `qwen-3/8b-base` 等子串发现模型。下载器为 Qwen 模型保留 `qwen-3/<variation>/transformers/1/`，Notebook 输出根保留原 slug，不能把所有文件平铺到一个目录。互补 LoRA 的 marker 硬编码了 `name-that-llm-train-lora24-s7/lora_out/job_final` 和 `ntl-train-lora-qwen25-instyle/lora_out/job_final`，不要改名。

Llama cache 应保留其输出中的 `llama-3.1/8b/`。该 cache 源码已在 references/ 下保存，下载应使用完成的输出，避免重新下载 16GB。其报告声称官方 base SHA 对齐；接收方必须读取并核对，不能把这个作者声称直接当本次校验完成。不得换成 Instruct、GGUF 或另一量化。

部分旧 adapter 即使不在最终 MODELS 里，cell 1 仍会查找，预检因此也要求它们。声明的输入中可能有算法未直接使用的冗余资源；第一轮保留全部附加来源，完成后另做依赖裁剪实验。

## 官方 CLI 命令示意

```bash
# dataset 可选第三段历史版本；接收方先确认版本
kaggle datasets download dhruvtibarewal/nlm-l24-adapters-v5 -p input/nlm-l24-adapters-v5 --unzip
# Notebook output 是训练产物，不是 kernels pull 下载源码
kaggle kernels output avazaku/name-that-llm-verified-llama-asset-cache/1 -p input/name-that-llm-verified-llama-asset-cache
# 模型是 instance VERSION 引用
kaggle models instances versions download qwen-lm/qwen-3/transformers/8b-base/1 -p input/qwen-3/8b-base/transformers/1 --untar
```

CLI 参数已按发送方已安装的 Kaggle 实现/--help 检查；认证和实际下载未完成。接收方需先确认自己的 CLI 支持版本引用。下载器默认不下载未锁定项；`--allow-unpinned` 仅允许显式使用 latest，报告仍标识未知版本，不赋予 exact reproduction 身份。每项下载后流式计算文件 SHA-256。

## 已知阻塞边界

- 发送方 CLI 没有可用 Kaggle 认证，因此未获取大型权重/比赛数据；接收方要用自己合法权限。
- public 源码可读不等于 output 仍存在或可以下载；API 失败/空输出必须留证据。
- avazaku 被引用的 .651 reproduction 源码 API 本次返回 404；Llama cache 源码可读。这些结果见 evidence/reference-fetch.json，不能假定引用的成绩已独立核验。
- 旧 lora24_public 历史权重缺失是原代码允许的特定 skip；其余成员必须核实完成。
- 额外数据的 label 真伪、prompt 独立性与许可没有在本次自动鉴定。
