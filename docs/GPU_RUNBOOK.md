# GPU 执行手册

## 机器与资源

优先 Linux + NVIDIA GPU，Windows 朋友可用 WSL2 CUDA；避免在原生 Windows 跑这份以 Kaggle Linux 为目标的源码。上游文字称 T4×2 可运行，但这是另一条被引用的运行记录，**本快照没有硬件实测**。双 T4 合计 32GB、单卡 48/80GB 可作为排查起点，单卡 24GB 是否足够必须实测；`device_map='auto'` 会影响切分/CPU offload。建议准备 64–128GB 主存、300GB 以上可用 SSD，额外留出压缩包/解压/缓存空间；这些是保守规划估计，不是最小需求或资产大小统计。

未知 GPU 型号时先采集硬件，不能保证速度。原程序会串行加载多套基础模型、18 个 v5 adapters 和互补栈，公开三行也可能耗时很久。原任务按 Kaggle 9h 限时设计，完整隐藏测试耗时需测，不能由三行推算。

## 1. 环境

```bash
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# 按本机驱动从 pytorch.org/get-started/locally 选择 CUDA wheel。
# 下面只是 CUDA 12.6 示例；安装后必须确认 cuda.is_available()。
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-gpu-candidate.txt
python -m pip install kaggle
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

`requirements-gpu-candidate.txt` 是待 GPU 验证的候选范围，不是原作者锁定环境。先尝试 Kaggle 的原镜像/现成环境更接近原实现；metadata 记录了镜像 digest，但 `kaggle-private-byod` 不保证可从普通 Docker 拉取。不要声称单凭相同 digest 就复现了环境。Transformers 4/5 API、UltraBERT 支持、PEFT/torchao 冲突、dtype 参数和离线自定义模块可能需兼容补丁；保留完整 `pip freeze` 和报错，再做最小改动。

准备 Kaggle 认证按 `kaggle auth login` 的官方提示操作。凭据留本机，不能写进仓库。比赛规则接受/模型许可由账号持有人按官方流程处理。

## 2. 下载与检查

```bash
python scripts/verify_workspace.py
python scripts/fetch_assets.py --list
python scripts/fetch_assets.py --input-root "$PWD/input" --download --allow-unpinned
python scripts/preflight.py --input-root "$PWD/input" --gpu --output preflight.json
```

`--allow-unpinned` 明确接受 Dataset/Notebook 未解析的历史版本，默认拒绝下载这些 latest 引用；这不代表精确复现。先看 docs/INPUTS.md，用网页选择/下载准确版本或填 `config/inputs.resolved.json`（复制原清单并填写 version），再用 `--manifest config/inputs.resolved.json`。下载器不自动执行 Notebook 训练；仅下载其已完成的 output。API 无权限、被删除、无 output 时停止并记录，不拿空目录顶替。

清单全部下载可能很大，工具逐项执行、无自动付费、失败返回非零，并写 input/download-report.json。用 `--only owner/slug` 仅下载一个引用。不要在此机器上下载完整权重后再传给无 GPU 的委托人，优先交付清单与共享资产链接。

## 3. 跑完整流程

```bash
python scripts/run_pipeline.py --input-root "$PWD/input" --work-dir "$PWD/runs/run-001"
python scripts/validate_submission.py --test input/competitions/name-that-llm/test.csv --submission runs/run-001/submission.parquet --report runs/run-001/submission-check.json
```

work-dir 必须全新/为空。入口会复制原模块到 stage，显式替换 `/kaggle/input` 和 `/kaggle/working` 两类路径，先配置并装离线 bnb wheel，再跑 complement，再跑 infer。它保留原系数、相对 ATLAS 总预算、模型顺序与 `NTL_DROP_Q35=1`。`src/` 不会改写。

运行时不自动联网下载模型；确保 input 内含 tokenizer/config/shards/adapter/自定义代码/wheels。不要手工传 `--sim`、NTL_SMOKE、NTL_FAKE_TEST。入口检测模拟环境并拒绝验收。全新工作目录隔离旧 comp_logit/cache/submission，避免失败后误读旧文件。

日志：setup.log、complement.log、inference.log；证据：preflight.json、environment.json、pip-freeze.txt、nvidia-smi-before.txt / after.txt、gpu-monitor.csv、run-status.json、submission-check.json。GPU 采样只报告采样峰值，不能当连续精确峰值；主存峰值未自动测量。

## 4. 出错处理

| 情况 | 处理 |
|---|---|
| 无 CUDA / 驱动不匹配 | 先修 torch/驱动；不要降级 CPU 然后算完成 |
| 缺 adapter / meta / shards | 对照清单取正确输出和版本，保留路径结构 |
| 缺 Llama | 下载 avazaku cache v1 output，按其验证报告核对官方 identity；保留 llama-3.1/8b |
| OOM | 保存 traceback / 显存采样，另开运行目录试较小 batch；明确记录运行差异 |
| 模型反复停在 load | 看 CPU/RAM/磁盘和 shard 进度；不是 GPU 0% 就判死锁，不自动无限等待 |
| COMP/ATLAS SKIPPED | 严格入口会失败；查依赖、预算、模型代码，不能忽略 |
| sklearn/UltraBERT/PEFT API 错误 | 记录版本，做兼容补丁；原模块 immutable，补丁另存并生成 diff |
| 仅三行可跑 | 仅格式与小样本运行证据；继续做完整负载探测和 Kaggle 隐藏重跑 |

需要全规模计时可使用原 infer.py 的 `--fake_test N` 与 complement 的 NTL_FAKE_TEST/NTL_POOL 对齐同一 pool 与顺序；这是单独探测实验，主入口故意不接受假测试。保留独立脚本、参数和结果，不能提交 fake IDs。若 ATLAS 预算导致跳过，报告失败，不只把系数改成 0。

## 5. 最后交付

按 docs/KAGGLE_SUBMISSION.md 转回 Kaggle。委托人没有 GPU，因此只交一个本地 parquet 不够：必须能在正式隐藏 test 上重新计算。按 docs/DELIVERY.md 打包轻量源码、运行记录和最终 Notebook，重型权重通过合法可访问来源交接。
