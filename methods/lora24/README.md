# 独立 LoRA24：现有权重与训练代码

报告为 Qwen3-8B Base 截断前 24 层、rank16、3 epoch 的独立训练；plain-head pooled OOF BA [0.479341](received/oof_report.json)，不是整套融合或正式 LB。原交付声称 B200 训练，本次未重训。

## 取得已有权重

可选 ZIP 通过 Git LFS 保存，137,703,403 字节：

```bash
git lfs install
git lfs pull --include="checkpoints/own_lora24_checkpoint.zip"
python -m zipfile -e checkpoints/own_lora24_checkpoint.zip outputs/lora24
```

以上命令从仓库根目录执行。ZIP SHA-256：`3d84cdadc2c2741c21c60cdead22a57076aca88563631da9719c4bba998de28b`。如果文件约 130 字节、以 `version https://git-lfs.github.com/spec/v1` 开头，则仍是 pointer，未取得真实权重。

解压后为 `outputs/lora24/own_lora24/final`，含 adapter、heads、readouts、tokenizer、完整性标记。**不含基础模型**，从 [Qwen 官方模型](https://huggingface.co/Qwen/Qwen3-8B-Base)下载完整 Base 并检查 shard。

## 使用与重新训练

下面从 `methods/lora24` 执行；先准备可用的 CUDA PyTorch、依赖与基础模型。当前无 GPU 电脑只验语法和完整性，不宣称 forward 成功。

```bash
python -m pip install -r requirements-gpu.txt
python scripts/validate_checkpoint.py --checkpoint ../../outputs/lora24/own_lora24/final
python scripts/check_parity.py --checkpoint ../../outputs/lora24/own_lora24/final --base-model /path/to/Qwen3-8B-Base --train-csv ../../data/competition/train.csv --output ../../outputs/lora24/parity.json
python scripts/predict.py --checkpoint ../../outputs/lora24/own_lora24/final --base-model /path/to/Qwen3-8B-Base --test-csv ../../data/competition/test.csv --output ../../outputs/lora24/plain-preview.parquet
```

这是 plain-head 公开接口预览；正式推理要使用 Kaggle 当次真实测试路径。

重新训练：

```bash
python scripts/prepare.py --train-csv ../../data/competition/train.csv --base-model /path/to/Qwen3-8B-Base --output ../../outputs/lora24-new/plan.json
python scripts/train.py --plan ../../outputs/lora24-new/plan.json --job fold0 --max-updates 2 --output ../../outputs/lora24-new/smoke
# 小步检查通过后，用新目录进行正式训练；设备按实际机器调整
python scripts/run_jobs.py --plan ../../outputs/lora24-new/plan.json --output-root ../../outputs/lora24-new/checkpoints --devices 0
python scripts/evaluate_oof.py --plan ../../outputs/lora24-new/plan.json --checkpoints ../../outputs/lora24-new/checkpoints --output ../../outputs/lora24-new/oof_report.json
```

每次新配置用新输出路径。脚本 `--help` 是参数依据，配置在 `configs/train.json`。原评估脚本只导出汇总，融合前需扩展保存逐行 OOF；当前 ZIP 只有 final，不能当未见官方验证标签的模型。

`received/own_lora24_ensemble_fixed.ipynb` 与修复模块是接入旧集成的候选。[修复说明](received/REPAIR_NOTES.md)主要避免误选只有配置的 base_config；本地检查不代表修改后完整集成已得分。

核心训练、预测和 ntl 模块与训练记录 SHA 匹配；辅助 preflight 是后续版本。具体见 [源码对照](../../manifests/lora_training_source_parity.json)。代码保留 [MIT LICENSE](LICENSE)，基础模型/衍生权重适用原模型条款，不继承公开原集成成绩。
