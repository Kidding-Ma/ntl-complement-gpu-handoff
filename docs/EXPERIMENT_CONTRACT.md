# 实验结果与融合输入规范

每个新实验用独立输出目录，不覆盖旧结果。这是输出接口约定，不表示新实验已经生成。

```text
experiment/
  config.json
  environment.txt
  provenance.json
  classes.json
  folds.csv
  oof_scores.npz
  metrics.json
  report.md
  inference.py
  weights/             # 或明确、可访问的下载地址和 SHA-256
```

- `classes.json`：按分数列顺序保存完整标签数组。
- `folds.csv`：`id,prompt_id,fold`，ID 唯一且覆盖全部训练样本。
- `oof_scores.npz`：字符串 `ids` `(N,)`、浮点 `scores` `(N,C)`、整数 `fold` `(N,)`、字符串 `classes` `(C,)`；不使用 object pickle。注明 scores 是 logits、SVC margin 还是概率。
- `metrics.json`：pooled BA、每折 BA、每类 recall、混淆矩阵、切片、训练与推理成本；未测写 null。
- `provenance.json`：代码 commit、数据 SHA、外部来源/排除规则、模型/tokenizer revision、每折训练边界、权重 SHA、种子、精度、是否使用测试批量统计。
- `report.md`：控制是什么、改变什么、看到什么、剩余不确定性、是否继续；分开标记历史汇总、开发验证、独立验证、真实 LB。

OOF 每行的训练模型不能见过对应验证组的标签，包括公开微调适配器和外部重复文本。只重训分类头不足以修复底座泄漏。只有 summary 没有逐行预测时不能训练融合。

融合前核对 ID、行数、类别顺序、训练数据 hash、fold 和分数类型；校准器和融合器在外层评估以外拟合。全量 final 对真实测试动态预测；公开三行只验接口。

当前提交 schema 是 parquet 的 `id,model`，以最新官方 sample 为准；ID 顺序与输入一致，无重复，标签合法。保存 Notebook 版本和分数回执的对应关系。
