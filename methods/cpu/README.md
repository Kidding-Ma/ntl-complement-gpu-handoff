# CPU 分类与风格特征

- `style_aware_runner.py`：5-fold GroupKFold(prompt_id)、fold-local TF-IDF，输出 OOF margin 和逐类指标；历史 pooled BA 约 0.384484。
- `kaggle_submit.py`：四视图 TF-IDF + 25 数值风格统计 + LinearSVC；历史分组 OOF BA 约 0.388936，读取 Kaggle 比赛挂载。
- `audit_dataset.py`：数据 schema、重复和分布审计。

历史运行指标在 `evidence/`，不是正式 LB。以下命令从本目录执行：

```bash
python -m pip install -r requirements.txt
python style_aware_runner.py --data-dir ../../data/competition --output-dir ../../outputs/cpu-control
```

旧 `kaggle_submit.py` 的推理入口拒绝少于 50 行 test，因此不适用于仓库三行预览；该门槛不是官方隐藏样本数量。风格特征函数可独立复用，SVC margin 不是归一概率。
