# Linux GPU 操作手册

## 环境与容量

原作者主 part-two 说明使用 T4×2；完整 complement+ATLAS 的实际运行时间本次未测。准备可容纳 14B 截断模型和量化加载的 NVIDIA GPU。作为服务器准备余量可考虑 48GB 显存或多 GPU、64–128GB RAM、至少 200GB 可用 SSD；这是容量规划建议，不是实测最低配置。先根据实际下载体积和 staging 峰值调整。

使用 Linux，Python 3.13 更接近已有相关日志（本次 CPU 校验也为 3.13）。不要假设任意 torch/transformers 都兼容。先安装适合 CUDA 驱动的官方 torch，然后：

```bash
python -m venv .venv
source .venv/bin/activate
# 从 pytorch.org 对应 CUDA 的安装命令安装 torch，再安装下面依赖
python -m pip install -r requirements.txt
mkdir -p /kaggle/input /kaggle/working /kaggle/temp
nvidia-smi
python -m pip freeze > /kaggle/working/environment.txt
```

`requirements.txt` 中 transformers/accelerate 参考旧诊断实测版本。它不是本完整融合的已验证环境；运行成功后保存精确 lock。`trust_remote_code=True` 的模型应先检查下载源码再运行。

## 数据与模型

```bash
mkdir -p /kaggle/input/competitions/name-that-llm
cp data/competition/*.csv data/competition/*.parquet /kaggle/input/competitions/name-that-llm/
python scripts/fetch_assets.py --list
# 使用你自己的 Kaggle 授权，参照 Kaggle CLI 的 auth login/官方 API 设置。
python scripts/fetch_assets.py
python scripts/preflight.py --gpu
```

下载所有资源可能需要 Kaggle 登录/比赛授权。授权信息保存在接收方服务器，绝不 commit。不需要用户本机的 API key。失败项见 reports；`--only REF` 可重试一个来源。下载不等于版本锁定：另存 Dataset/Notebook version 与关键文件 SHA-256。

## 执行和验证

```bash
set -o pipefail
python -u scripts/run_gpu.py 2>&1 | tee /kaggle/working/run.log
python scripts/validate_submission.py
```

runner 按原始顺序执行每个单元格，并保留源码写出的完整辅助模块。其 `/kaggle/input`、working、temp 约定与原代码相同；没有改变模型数学。GPU 没有、预检缺失、开启伪 smoke/fake_test 都会阻止正式运行。

Qwen3.5 如果卡在加载，先阅读 related/qwen35/diagnose_qwen35.py，确认模型和 head 的路径匹配后，在干净进程、相同资源上做诊断。旧诊断脚本只识别 `--sync`，没有 help 模式；不要直接拿旧诊断当成本版本的证明。旧记录的是 mmap/并发加载候选，尚未证实本版本根因。可单独对照：

```bash
python -u scripts/run_gpu.py --sync-load 2>&1 | tee /kaggle/working/run_sync.log
```

这会在导入 transformers 前设置 `HF_DEACTIVATE_ASYNC_LOAD=1`；未经 GPU 验证，不是已修复承诺。优先只运行出问题的单成员，不要重复整个多小时流水线。检查 staging 的 scratch 余量及文件 hash，记录失败 traceback 和 CPU/GPU 使用情况。不要同时盲改 loader、max_len、层数和 blend。

## 回传与提交

保留原始日志、timings、environment、part_two_verification、enhanced_verification、token_audit、NPZ 和 handoff_validation。公开三行只证明格式和成员运行。若做开发验证，用 train 的 prompt_id 分组，绝不以三行预览推断 accuracy。

`python scripts/build_notebook.py` 生成 schema 正常的基线 Notebook（单元格 source 原样）。服务器修复另存 deliverables 下。填写模板中的 id/title/code_file 和真实可访问输入，复制成 `kernel-metadata.json`。先 review 修改版，再由 Kaggle CLI 或 UI 保存运行版本；正式隐藏测试必须由 Kaggle 执行。服务器生成的三行 parquet 不可用来代替隐藏测试。
