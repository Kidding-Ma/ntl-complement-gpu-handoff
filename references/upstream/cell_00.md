# Name That LLM | 0.651 Verified Reproduction

**Our official public score is 0.651**, reproduced on `avazaku` using free Kaggle T4 x2 compute. The successful submission used [our exact version 3, script 362281064](https://www.kaggle.com/code/avazaku/name-that-llm-nlm-v27-reproduction?scriptVersionId=362281064). It was submitted on 26 October 2026 at 02:56:34 UTC and its successful score was verified in Kaggle's submission details.

This is a credited reproduction of [Dhruv Tibarewal's nlm-v27, scored version 1, script 362013743](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27?scriptVersionId=362013743), which also scored 0.651. **The model recipe and v5 adapters are his work.** Our work here verifies execution, restores a hash-checked Llama file dependency, and documents the result. It is not a new classification method or a model improvement over nlm-v27.

This documentation release preserves all six scored code cells byte for byte. The score belongs to version 3. A later documentation-only version has not received a separate official evaluation. Do not infer a new score from its version number or from the three visible example rows.

## What the ensemble does

The task is to identify one of 18 language models from a passage. The original recipe combines adapted neural representations with lexical and statistical models, then applies its fixed postprocessing. The implementation below is the authoritative recipe; the table is a guide to reading it.

| Stage | Preserved components | Role |
|---|---|---|
| Main neural ensemble | Qwen3-14B-Base, Qwen3-8B-Base and base Llama 3.1 8B, each with five v5 fold adapters plus the full-fit adapter | Average the released adapter predictions; main weights are 3, 2 and 1 |
| Additional main members | Nishant's Qwen3-14B, Qwen2.5-14B and DeBERTa | Add complementary representations with original weights 2.5, 1 and 1 |
| Statistical heads | Four-view lexical SVC, shared-prompt Gaussian heads and adaptive centred twins | Use the original training data, public pool and fixed settings |
| Complementary public stack | Stylegrams, stylometry, GLM/Qwen/encoder features, LoRA and token-attention checkpoints, five UltraBERT folds | Mix the original complementary predictions at 0.15 |
| ATLAS and final processing | Public ATLAS bundle on selected low-margin rows, then the original balancing and guards | ATLAS weight 0.10, top-two margin threshold 0.3; retain the original full-test Sinkhorn logic |

The program intentionally uses both CPU and GPU: neural inference uses the GPU path, while sklearn/SVC/PCA/Gaussian and parts of the ensemble use CPU. Replacing these components would create a different implementation. Model depth truncation in the source also explains the expected `UNEXPECTED` messages for unused higher layers and language-model heads during loading; these messages alone do not establish a failed member.

## Run the scored recipe

1. Copy this notebook with its attached inputs. Preserve their selected versions. The inventory below lists 31 references: 14 datasets, 7 model instances, 9 notebook outputs and the competition data.
2. Check your **free** Kaggle accelerator allowance. Choose **GPU T4 x2** and turn **Internet off**. No paid resource is needed. Start with the scored environment when available; the exact container is recorded below.
3. Reuse [Verified Llama Asset Cache version 1](https://www.kaggle.com/code/avazaku/name-that-llm-verified-llama-asset-cache?scriptVersionId=362275911). Attach its completed output rather than rerunning its downloads. It supplies the original base-model files at `llama-3.1/8b`, which the unchanged discovery code finds.
4. Use **Save Version / Save & Run All** for a saved execution. Leave the original code and coefficients unchanged for reproduction. The offline wheels input supplies bitsandbytes; the notebook does not need Internet access to install it.
5. Check completion logs for every required v5 adapter, the complementary stack and actual ATLAS mixing. Check `submission.parquet`, with one `id` and a valid `model` per test row. A completed notebook is insufficient if an exception was swallowed by a shell pipeline or the ATLAS error handler.
6. Submit the exact completed version once through the competition. Kaggle reruns it on the hidden test data. Only the successful official receipt establishes the public score. Publishing documentation does not require a second submission or a new hidden rerun.

The visible test is only a format check. In our execution it contained three rows. The source's small training self-checks are numerical checks on seen rows, not out-of-fold validation. Neither is evidence for 0.651 by itself.

## The file transport change

The original Meta Kaggle input was inaccessible to this account. We replaced only that transport with the same base Llama 3.1 8B files from a pinned public NousResearch distribution. This is not Llama Instruct, GGUF, a new quantization, or another fine-tuned model.

| Provenance | Pinned identity |
|---|---|
| Official reference | `meta-llama/Llama-3.1-8B` at `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` |
| Public transport | `NousResearch/Meta-Llama-3.1-8B` at `1f47e50cdbe801ad8a5174156ec3a0655108fb9f` |
| Cached output | Notebook version 1, script `362275911` |
| Verification | All 14 selected file identities match the official reference; downloaded bytes of all four weight shards passed SHA-256 guards |
| Configuration | Base Llama, hidden size 4096, 32 layers, vocabulary 128256, stored dtype bfloat16 |
| Saved files | 16,071,914,067 bytes including configuration, tokenizer and retained licence files |

The cache contains `llama_asset_verification.json`, `README.md`, `LICENSE`, `USE_POLICY.md` and `NOTICE`. The report exposes the expected and observed hashes so the claim can be checked without trusting a filename. The other 30 input references, all six inference cells, model precision, adapter weights and postprocessing were preserved.

## Audit and measured runtime

| Check | Observed result |
|---|---|
| Our official evaluation | **Succeeded, public score 0.651**, exact script `362281064` |
| Code identity | Six cells match scored nlm-v27 v1 byte for byte |
| Code SHA-256 | `ff22ff2b32ce759c9fdfaac9fe6229e3305d531891ed84de9ce04529b7b1b944` |
| Container | `gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435`, matching the parent |
| Saved visible execution | **4 h 4 min 16 s**, free GPU T4 x2, Internet off |
| Visible execution checks | Two GPUs, `smoke False`, all 18 required v5 fold/full-fit adapters completed in the original training self-check and test passes |
| Complement / ATLAS | Complement completed and loaded; ATLAS actually mixed at 0.10 |
| Optional historical member | `lora24_public` was skipped because its upstream checkpoint is unavailable; the scored parent's log has the identical skip |
| Memory | No independent peak RAM or VRAM measurement was recorded; post-cleanup zero-memory messages are not a peak estimate |

The parent's visible runtime was about 1 h 27 min. Our longer visible runtime is reported rather than hidden; model-loading and execution speed can vary. The official hidden-run runtime and queue time were not separately exposed in the receipt we recorded. The three-row timing is not a prediction of full-test latency. Allow several hours and consult the current Kaggle runtime limit and free quota before starting.

## Limits and troubleshooting

* **Version drift:** preserve input versions and the code digest. A newer public source or adapter release is not automatically the version that produced the quoted score.
* **Missing Llama files:** inspect the pinned cache output and its verification report. Keep the `llama-3.1/8b` directory structure. Do not replace it with an Instruct or quantized model just to remove an access warning.
* **Partial ensemble:** `COMP STACK DONE`, adapter completion lines and `atlas ... mixed at 0.1` provide evidence that components ran. `COMP SKIPPED`, `ATLAS SKIPPED`, a traceback or a missing required adapter needs investigation before claiming reproduction. The specific historical `lora24_public` skip above is part of the scored recipe's available inputs.
* **Balancing assumption:** the original Sinkhorn and prompt/group logic depend on competition structure. This notebook does not establish that those assumptions hold on arbitrary real-world text.
* **External data:** labels and source attribution in public pools are upstream claims. Public availability does not establish label authentication, prompt independence or freedom from source/topic confounding. We do not claim that an overlap audit authenticates those labels.
* **Scope:** this reproduces the released inference recipe, including its head fitting. It does not recreate or independently validate the full training process for every upstream adapter. There is no new component ablation or leakage-free OOF experiment in this release.

No coefficient, model, feature, pool, balancing rule or held-out choice was tuned here to obtain the matching score. The earlier reproduced 0.636 reference is a different public recipe; moving from that reference to this reproduced 0.651 does not prove a novel-method gain.

## Credits and licences

The principal recipe and v5 adapters are by **Dhruv Tibarewal**. Thanks to **Nishant Dahal**, **Lucas Boesen**, **Joseph Adamski**, **PJ Mathematician**, and the other authors linked in the inventory for their public checkpoints, data and complementary methods. The original code's comments and attribution remain intact. Foundation-model credit belongs to the upstream model authors, including Qwen, Meta and the encoder providers.

The original notebook code is declared Apache 2.0. That licence does **not** relicense attached model weights or datasets. Each input retains its own upstream terms, attribution and redistribution requirements. In particular, the handwriting-blend input declares CC BY-SA 4.0 and retains its upstream credits. Follow the terms on the linked input pages rather than applying the notebook's code licence to everything it reads.

**Built with Llama.** The Llama files are subject to the [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/LICENSE) and [Acceptable Use Policy](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/USE_POLICY.md). The cache retains both and the required notice. File transport through a public mirror does not remove these terms. If using Meta's gated distribution instead, obtain access through its official process; no access-control workaround or credential is embedded here.

## Attached input inventory

The scored run used the following public sources. The Kaggle Input tab records the mounted selections. Keep those versions when copying; the links below identify the upstream owners and their current source pages.

| Type | Upstream source |
|---|---|
| Dataset | [dhruvtibarewal/nlm-l24-adapters](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters) |
| Dataset | [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages) |
| Dataset | [abdullah1afify/ntl-generations](https://www.kaggle.com/datasets/abdullah1afify/ntl-generations) |
| Dataset | [dophuchung/ntl-proxy-generations](https://www.kaggle.com/datasets/dophuchung/ntl-proxy-generations) |
| Dataset | [zbigniewkirchner/ntl-qwen3-8b-lora-claude-head](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head) |
| Dataset | [dhruvtibarewal/nlm-matched-clean](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-matched-clean) |
| Dataset | [dhruvtibarewal/nlm-offline-wheels](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-offline-wheels) |
| Dataset | [josephadamski91/ntl-atlas-qwen3-8b-v1](https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1) |
| Dataset | [lucasboesen/ntl-free-proxy-generations](https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations) |
| Dataset | [nishantdahal/ntl-two-stage-lora](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora) |
| Dataset | [nishantdahal/ntl-handwriting-blend](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend) |
| Dataset | [dhruvtibarewal/nlm-l24-adapters-v5](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-l24-adapters-v5) |
| Dataset | [prvsiyan/prvsiyans-extended-name-that-llm-dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) |
| Dataset | [pjmathematician/ultrabert-380m-ntl](https://www.kaggle.com/datasets/pjmathematician/ultrabert-380m-ntl) |
| Notebook output | [lucasboesen/name-that-llm-glm4-9b-trunc25](https://www.kaggle.com/code/lucasboesen/name-that-llm-glm4-9b-trunc25) |
| Notebook output | [lucasboesen/name-that-llm-extract-encoders2](https://www.kaggle.com/code/lucasboesen/name-that-llm-extract-encoders2) |
| Notebook output | [lucasboesen/name-that-llm-electra-large-weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-electra-large-weights) |
| Notebook output | [lucasboesen/name-that-llm-desklib-weights](https://www.kaggle.com/code/lucasboesen/name-that-llm-desklib-weights) |
| Notebook output | [lucasboesen/name-that-llm-train-lora24-s7](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7) |
| Notebook output | [lucasboesen/name-that-llm-train-tokattn](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn) |
| Notebook output | [lucasboesen/name-that-llm-train-tokattn2](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-tokattn2) |
| Notebook output | [vnton37nguyn/ntl-train-lora-qwen25-instyle](https://www.kaggle.com/code/vnton37nguyn/ntl-train-lora-qwen25-instyle) |
| Notebook output | [avazaku/name-that-llm-verified-llama-asset-cache](https://www.kaggle.com/code/avazaku/name-that-llm-verified-llama-asset-cache?scriptVersionId=362275911) |
| Model | [qwen-lm/qwen-3/Transformers/14b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/14b-base/1) |
| Model | [qwen-lm/qwen-3/Transformers/8b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/8b-base/1) |
| Model | [qwen-lm/qwen-3/Transformers/4b-base/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/4b-base/1) |
| Model | [qwen-lm/qwen-3/Transformers/14b/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/14b/1) |
| Model | [qwen-lm/qwen-3/Transformers/8b/1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/8b/1) |
| Model | [qwen-lm/qwen2.5/Transformers/7b/1](https://www.kaggle.com/models/qwen-lm/qwen2.5/transformers/7b/1) |
| Model | [qwen-lm/qwen2.5/Transformers/14b/1](https://www.kaggle.com/models/qwen-lm/qwen2.5/transformers/14b/1) |
| Competition | [name-that-llm](https://www.kaggle.com/competitions/name-that-llm) |

The six unchanged implementation cells follow. No new experiment is started by this documentation release.
