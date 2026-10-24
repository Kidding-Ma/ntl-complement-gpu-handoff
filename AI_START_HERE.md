# 给服务器 GPU 上的 AI：完整执行任务

用户希望使用本仓库复现并提交公开 Notebook `fleongg/ntl-part-two-complement`。你有服务器 GPU；用户本地没有完成本版本的 GPU 运行。目标不是重新写一个较简单的 baseline，也不是延续旧的 0.604 项目。**先复现本仓库的主融合 + 12 成员 complement + ATLAS，遇到真实阻塞再修复。**

## 第一步：核验交接

1. `python scripts/verify_workspace.py`；读取源码、`manifests/assets.json`、`docs/DEPENDENCIES.md`。
2. `python scripts/fetch_assets.py --list`。查看 `evidence/reference_fetch.json`，区分“能下载源码”和“能下载输出权重”。
3. 准备 Linux GPU 环境，记录 `nvidia-smi`、Python、CUDA、torch、transformers、peft、accelerate 和 `pip freeze`。版本候选见 requirements.txt，未经本机 GPU 验证，不能称为锁定可用环境。

## 第二步：取得全部真实输入

- 大模型不在 GitHub 中；从 manifests 原链接下载到 `/kaggle/input` 的指定相对路径。执行 `python scripts/fetch_assets.py`，失败项写入报告。使用接收方自己的 Kaggle 授权；若需接受比赛条款由账号持有人完成，不要索要用户的密钥写入仓库。
- `data/competition/` 中已包含官方训练和公开预览。可先 `mkdir -p /kaggle/input/competitions/name-that-llm && cp data/competition/*.csv data/competition/*.parquet /kaggle/input/competitions/name-that-llm/`。正式 Kaggle 提交时由比赛挂载替换，绝不能固定读仓库的三行 test。
- 优先查 `fleongg/ntl-legacy-qwen24-recovery` 和 `fleongg/ntl-quality-selected-public-svc-pool` 的可访问性。原代码必须校验 legacy manifest 的每个 SHA-256。
- 核对主模型 `pool_v4.parquet`、两个 14B LoRA、DeBERTa，和 ATLAS 的 `ntl_atlas/`、`ntl_rlcd/`、adapters；只看 marker 不代表全部权重已齐。
- 如果原资产确实不可访问，首先记录具体 URL、错误和缺失文件；查找原作者等价公开来源。若需要从源训练重建，保留独立实验、按 prompt_id 做 GroupKFold、输出完整权重结构和真实验证指标；这属于修改后的方案，不能声称逐字节复现 0.636。不要把自训 LoRA24 改名冒充原权重。

## 第三步：先单成员，再全融合

1. `python scripts/preflight.py --gpu`。预检是必要条件，未检查每一个张量，不能代替推理。
2. 对所有主模型、12 个必需 complement 成员和 ATLAS 做真实文本 forward；记录加载开始/结束、耗时、显存峰值、输出有限值、标签顺序及 EOS。不得因 OOM 或卡住跳过成员。
3. 重点关注 Qwen3.5 token-attention。`related/qwen35/FINDINGS.md` 是**旧 Notebook**的实测，不能直接认定本 V2 同样故障。先复现本版本；按 `docs/GPU_RUNBOOK.md` 单独测试 sequential load 候选。`--sync-load` 是未验证诊断开关。
4. `python -u scripts/run_gpu.py 2>&1 | tee /kaggle/working/run.log`，保留所有检查。该入口会执行原始单元格，并将子进程源码放在原目录；不是只运行提取出来的 comp_stack。
5. `python scripts/validate_submission.py`。检查 `enhanced_verification.json`、全部成员 NPZ、main 证据、ATLAS 覆盖、最终 parquet 的 ID/行数/18 类。

## 第四步：交回 Kaggle 可重跑版本

- 基线 `ntl_part_two_complement_ready.ipynb` 只修复 notebook schema，推理源码逐单元格相同。修复后的版本另存 `deliverables/ntl_complement_fixed.ipynb`，完整包含辅助模块和依赖逻辑，不依赖服务器本地绝对路径。
- 若上传自有新权重，给出 Kaggle Dataset 的下载/挂载结构与校验值，填写可用的 `kernel-metadata.json`；不可访问的旧源不能继续出现在成功版里。
- 固定 Python 环境/镜像、模型与 Dataset/Notebook 的具体版本和每个关键文件 hash。先保存版本运行，再在 Kaggle 正式评估。GPU 服务器本身拿不到隐藏测试集。

## 必须交付

1. `deliverables/ntl_complement_fixed.ipynb` 和需要的全部脚本。
2. `deliverables/RESULTS.md`：机器/GPU、环境、输入版本、实际成员、修复点、耗时、公开三行格式检查与真实 CV/LB 的明确区别。
3. 完整依赖到位报告，日志、错误栈、运行 timings、EOS/标签顺序证据、最终 validation JSON 和 hashes。
4. 任何重建训练的完整 config、训练/OOF 结果、adapter/head/norms/tokenizer/meta/COMPLETE 等所需文件；大文件通过合适的模型资产渠道提供可下载地址。
5. Kaggle 实际运行或提交结果；未执行的部分明确写“未验证”。只有语法检查和三行 parquet 不能宣布完成。

执行优先级：**获取/定位原资产 → 逐成员 GPU 验证 → 修复加载 → 完整融合 → 可移植 Notebook → Kaggle 隐藏重跑**。
