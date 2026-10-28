# 可复用 CPU 控制与风格特征

仅保留四个有直接用途的脚本：

- `style_aware_runner.py`：5-fold GroupKFold(prompt_id)、fold-local TF-IDF，导出 OOF margin 和逐类指标；历史 pooled BA 约 0.384484。
- `kaggle_submit.py`：四视图 TF-IDF + 25 数值风格统计 + LinearSVC；历史分组 OOF BA 约 0.388936，是 Kaggle 路径推理脚本。
- `cpu_blend_inference.py`：风格/原始视图融合和特征实现；历史约 0.393 probe 属于开发诊断，不是独立成绩。
- `audit_dataset.py`：schema、重复和分布审计。

历史指标在 `evidence/`。代码为本地当前快照，本次未重跑整套 CV，不声称这些数值是本次新测结果。

从本目录执行：

```bash
python -m pip install -r requirements.txt
python style_aware_runner.py --data-dir ../../data/competition --output-dir ../../outputs/cpu-control
```

`kaggle_submit.py` 旧接口主动拒绝少于 50 行 test；该人为门槛不代表官方隐藏数量，不要伪造更多样本绕过。新方法可复用特征函数并明确区分预览/正式接口。SVC margin 需要训练侧校准后融合。

保留原注释和来源归属；风格方案包括 Yang Kuang Ou 等公开实现的工作，见根目录 NOTICE。
