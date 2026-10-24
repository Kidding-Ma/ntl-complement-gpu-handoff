# Name That LLM — Public Score 0.636

Our best confirmed competition submission is **57551442**, scored **0.636** using version1 of this notebook. This public release preserves every inference code cell of that scored version. The score is a Kaggle Public Score, not the three-row format preview. A new publication version is not a new scored submission.

The pipeline combines the public two-stage LoRA part-two main ensemble (SVC, Qwen3-14B, Qwen2.5-14B, DeBERTa), a15% complementary stack, and10% ATLAS on uncertain rows, followed by the original Sinkhorn balancing. Every input runs the actual models; there is no public-row prediction cache. All required components must succeed.

## Credits and reproducibility

Main recipe and new public checkpoints: [Nishant Dahal, two-stage LoRA part two](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two). Complementary model assets: [Dhruv Tibarewal](https://www.kaggle.com/dhruvtibarewal), [Lucas Boesen](https://www.kaggle.com/lucasboesen), [Vladislav Lavrentev/lightningv08](https://www.kaggle.com/lightningv08), PJ Mathematician, vnton37nguyn, and the original authors credited in source cells. ATLAS: [Joseph Adamski](https://www.kaggle.com/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1). Qwen base models are credited to the Qwen team. Original documentation and per-component credits remain below; metrics in that original documentation belong to their respective authors.

Some inputs are auxiliary public-origin archives maintained on this account; input visibility may limit fork execution. No privately generated validation passages or hidden competition data are included. No claim of full ensemble OOF validation is made for our final fixed mixture. Later experimental encoders scored0.634 and did not replace this0.636 model.

GPU execution was validated with all main models,12 complementary members, ATLAS, class alignment, training EOS, fusion math, and final parquet checks.


# New public two-stage main + complementary stack + ATLAS
Main: Nishant Dahal two-stage part two, public version2, original new checkpoints, pool, training EOS and weights. Add the same fixed15% complementary stack and10% ATLAS used in our scored0.612 method, before published full Sinkhorn. Required models fail closed. All actual inputs run the models; no public prediction cache. No full new ensemble OOF or improved Public Score is claimed. All source credits retained.


# Private reproduction: new public two-stage LoRA part two
Source: Nishant Dahal, https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two (public version2). Original documentation and credits retained below. Inference and fixed weights retained, including explicit training EOS and original pool. Added component evidence, required-member validation and byte-preserving adapter staging only. The three public example rows test execution and format, not accuracy. Our previous best Public Score is0.612; no new result is claimed before an actual competition evaluation.


# Two-stage LoRA, part two, LB 0.620

Our [first notebook](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool) scored 0.592 with two Qwen3 LoRAs trained on 35,000 public passages. This one scores 0.620 with the same recipe. Three things changed. The public pool got bigger and cleaner, two new members joined, and fixed a tokenizer bug that only shows up on Kaggle. Below is what each change did, on the same 5-fold CV split by prompt.

| Member | Trains on | First notebook | This notebook |
|---|---|---|---|
| n-gram SVC | train plus the public pool, fit here | 0.435 | 0.473 |
| Qwen3-14B LoRA, 27 of 40 layers | public pool, then train | 0.569 | 0.611 |
| Qwen2.5-14B LoRA, 32 of 48 layers | public pool, then train | | 0.569 |
| DeBERTa-v3-large, full fine tune | public pool, then train | | 0.546 |
| Blend with Sinkhorn, CV | | 0.593 | **0.642** |
| Public LB, with the centred twin | | 0.592 | **0.620** |

Member rows are balanced accuracy of that member alone, out of fold. The first notebook also had a Qwen3-8B LoRA at 0.569, which is gone now.

Everything the notebook needs is in the dataset [NTL two-stage LoRA part two](https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend). The base models are Kaggle's copies of Qwen3 14B Base and Qwen2.5 14B. Inference only. The commit run on the public test takes about 15 minutes on two T4s.

## 1. n-gram SVC

The same four TF-IDF views as the first notebook. Raw characters, words with punctuation, word shapes, and small words with every content word replaced by CONTENT. The same LinearSVC with C of 0.5 and balanced class weights. The only change is the pool it fits on, 75,783 public passages instead of 34,940. That alone moved it from 0.435 to 0.473.

## 2. The pool, the change that mattered most

Most of the gain came from data, not from models. On the old pool of 35,000 passages, four backbones finished at the same 0.569. Qwen3-8B, Qwen3-14B, Qwen2.5-7B and Qwen2.5-14B. The same Qwen3-14B with the same recipe on the new pool finished at 0.611.

You can see it before the host data even comes in. After stage one alone, trained only on public passages, the 14B's head scores 0.572 on the host passages. With the old pool it scored 0.493. The backbone did not change. The pool did.

The pool the 14B read in stage one has 83,827 passages from 26 public datasets. It keeps the filters of the first notebook. One paragraph, 30 to 300 words, no markdown, no duplicates, nothing identical to a train passage, and no punctuation mark that a model never uses in the host data. Two filters are new.

- **One source is out.** A set of claude-fable-5 passages written with a system prompt. They read nothing like the host's Fable passages.
- **Public passages too close to a train passage are out.** 1,153 passages whose content TF-IDF cosine with some train passage is 0.3 or more. Some public datasets answer prompts rebuilt from the training data. If stage one has already seen a host prompt, the out of fold score of stage two is too kind. This filter keeps the CV honest. It does not raise it.

The dataset ships the 75,783 passage version that the SVC fits on. It has the first filter but not the second.

## 3. Two new members

**Qwen2.5-14B LoRA.** The same recipe as the Qwen3 LoRAs, on the first 32 of 48 layers, pooled at layers 3, 5, 8, 11, 16, 21, 27 and 32. Stage one ran on the old 35,000 passage pool. Alone it ties the Qwen3 models at 0.569. Its top prediction agrees with theirs on about 77 percent of passages, which is about as often as they agree with each other. So it is not a new view, and it gets a small weight, 0.2.

**DeBERTa-v3-large, fully fine tuned.** No LoRA here, all the weights train. The same two stages. Stage one is 2 epochs over 61,486 public passages with class weights, at learning rate 0.000015 for the encoder and 0.001 for the head. Stage two is 3 epochs on train at 0.00001 and 0.0005. Max length 384, the mean of the last layer, one linear head. The bigger pool moved it from 0.512 to 0.546 alone.

## 4. LoRA inference, and a bug that only shows up on Kaggle

The inference code is the first notebook's. Keep the first layers, merge the adapter into the weights, average the hidden states at 8 depths without the first token, one LayerNorm per depth, one linear head. Two things are new.

**The end token.** Every passage gets an end token appended, in training and at inference. Our training tokenizer used `<|endoftext|>`, id 151643. Kaggle's copy of Qwen3-14B-Base ships the tokenizer config of the chat model, whose end token is `<|im_end|>`, id 151645. So on Kaggle every passage ended with a token the model never saw there in training, and the 14B's top prediction changed on about 14 percent of passages compared with the same checkpoint run locally. Nothing crashes and nothing warns. The fix is one line. Use the id saved in the checkpoint's meta.json. Kaggle's Qwen2.5-14B does not have this problem. If you run Qwen3-14B-Base from Kaggle Models, print which end token you get.

**T4 numerics and memory.**

- We tell cuBLAS to keep fp32 sums in fp16 matmuls. It costs nothing, so we set it.
- A batch that overflows fp16 is redone in bf16, as before.
- A batch that runs out of memory on the 15 GB T4 is redone one passage at a time.
- There is no try except block around the members. In an earlier submission a handler like that skipped a member that ran out of memory on the T4. The run still finished and scored, a few points lower, and we only found out from the log. A member that breaks should stop the notebook.

## 5. Centred twin

As in the first notebook. Cluster the test passages by content TF-IDF. In every cluster of 2 to 18 passages, subtract the cluster mean from each LoRA's features and read the rest with the centred head, at weight 0.1 per LoRA. It is not in the CV numbers above, because CV has no test clusters.

## 6. DeBERTa

The checkpoint is copied to local disk before loading. Loading it straight from the dataset mount took 16 minutes in our runs. The copy plus the load take 25 seconds.

## 7. Blend and Sinkhorn

A weighted sum of log probabilities. SVC 1.0, Qwen3-14B 0.4, Qwen2.5-14B 0.2, DeBERTa 0.25. These are round numbers chosen on out of fold predictions. Fitting them finer on 2,699 passages mostly fits noise. Then Sinkhorn balancing to equal class totals as in the first notebook, 300 iterations, skipped on the 3 row public test.

## CV says 0.642, LB says 0.620

The first notebook lost almost nothing between CV and LB, 0.593 against 0.592. This one loses 2.2 points. What changed between the two is mostly the pool. Our CV only tests on host passages, but the members learned much of what they know from public passages written with other settings. Some of that seems not to carry over to the hidden test. We have guesses and no clean answer yet. Part of the gap is plain noise too, since the public LB is only a slice of the test.

## What did not help

- **More parameters.** See section 2.
- **Averaging seeds**, or averaging the LoRA weights of several runs. 0.1 points at most.
- **A Gemma-4-12B backbone** with the same recipe.
- **Gates for one family**, for example a separate judge just for the Claude and GLM passages.
- **At most one passage per model inside a topic cluster.** Better in simulation, worse on the LB.
- **More stage two epochs**, or public passages mixed into stage two.
- **Stage two on short windows** cut from the host passages. Short passages are hard for every member, and the windows did not change that.
- **A stacker** instead of fixed weights.

None of it helped the two hardest pairs, `glm-5.3` against `glm-5.3-flash` and `gemini-3.7-flash` against `gemini-3.8-flash`. Every member gets stuck at about the same place on them.

## Credits

Public passages, CC BY 4.0 unless another license is given.

- [seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages)
- [willrice/name-that-llm-openrouter-harvest](https://www.kaggle.com/datasets/willrice/name-that-llm-openrouter-harvest)
- [lxdlxdlxd/ntl-synth-generations-v1](https://www.kaggle.com/datasets/lxdlxdlxd/ntl-synth-generations-v1)
- [prvsiyan/prvsiyans-extended-name-that-llm-dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset)
- [alperen5252525/ntl-vertex-gemini-proxy-gpt-generations](https://www.kaggle.com/datasets/alperen5252525/ntl-vertex-gemini-proxy-gpt-generations)
- [haileycheng/name-that-llm-same-prompt-passages](https://www.kaggle.com/datasets/haileycheng/name-that-llm-same-prompt-passages)
- [kaplinskiy/ntl-gpt-proxy-passages](https://www.kaggle.com/datasets/kaplinskiy/ntl-gpt-proxy-passages)
- [lucasboesen/ntl-free-proxy-generations](https://www.kaggle.com/datasets/lucasboesen/ntl-free-proxy-generations)
- [abdullah1afify/ntl-generations](https://www.kaggle.com/datasets/abdullah1afify/ntl-generations)
- [dophuchung/ntl-proxy-generations](https://www.kaggle.com/datasets/dophuchung/ntl-proxy-generations)
- [ziyuzhou1207/ntl-zz-proxy-generations](https://www.kaggle.com/datasets/ziyuzhou1207/ntl-zz-proxy-generations)
- [keshavatearth/ntl-gpt-new-prompts-v1](https://www.kaggle.com/datasets/keshavatearth/ntl-gpt-new-prompts-v1)
- [michaelsemenoff/ntl-own-extra-v1](https://www.kaggle.com/datasets/michaelsemenoff/ntl-own-extra-v1)
- [jadkutbi/ntl-own-proxy-generations](https://www.kaggle.com/datasets/jadkutbi/ntl-own-proxy-generations)
- [pardheev/ntl-proxy-extra-passages](https://www.kaggle.com/datasets/pardheev/ntl-proxy-extra-passages)
- [abhinavsaipodugu/ntl-proxy-gpt-gemini-passages](https://www.kaggle.com/datasets/abhinavsaipodugu/ntl-proxy-gpt-gemini-passages)
- [zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations](https://www.kaggle.com/datasets/zakirkhanaleemi/name-that-llm-benchmarks-proxy-generations)
- [zbigniewkirchner/ntl-extra-gpt-passages](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-extra-gpt-passages)
- [arxitekton/ntl-claude-gen-v1](https://www.kaggle.com/datasets/arxitekton/ntl-claude-gen-v1)
- [arxitekton/ntl-claude-gen-v2](https://www.kaggle.com/datasets/arxitekton/ntl-claude-gen-v2)
- [maniceet/name-that-llm-openrouter-openai-generations](https://www.kaggle.com/datasets/maniceet/name-that-llm-openrouter-openai-generations)
- [dedruck/ntl-proxy-gen](https://www.kaggle.com/datasets/dedruck/ntl-proxy-gen)
- [prvsiyan/ntl-own-prompt-generations-v1](https://www.kaggle.com/datasets/prvsiyan/ntl-own-prompt-generations-v1), CC BY-SA 4.0
- [drjcfmd/ntl-generated-v2](https://www.kaggle.com/datasets/drjcfmd/ntl-generated-v2), CC BY-SA 4.0
- [illiadolenko/ntl-generated-corpus-v1](https://www.kaggle.com/datasets/illiadolenko/ntl-generated-corpus-v1), CC0
- [aswathvanan/ntl-proxy-passages](https://www.kaggle.com/datasets/aswathvanan/ntl-proxy-passages), CC BY-SA 4.0, only in stage one of the Qwen3-14B LoRA

Ideas and recipes.

- The LoRA24 recipe from [lightningv08/ntl-lora24-fivefold-training](https://www.kaggle.com/code/lightningv08/ntl-lora24-fivefold-training) and [lucasboesen/name-that-llm-train-lora24-s7](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7)
- The style aware n-gram views from [yangkuangou](https://www.kaggle.com/yangkuangou)'s notebook
- Adaptive centering, the centred twin and Sinkhorn balancing from [lucasboesen/lora-x-adaptive-centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)

Models. Qwen3-14B-Base and Qwen2.5-14B by the Qwen team, Apache 2.0. DeBERTa-v3-large by Microsoft, MIT.