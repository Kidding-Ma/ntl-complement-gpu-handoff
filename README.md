# Name That LLM：多方案实验计划与有效成果

**主要交付：[多种可行方法的详细方案](docs/MULTI_METHOD_PLAN_ZH.md)。** 包括独立新方法、公开复现、数据路线及融合，不限定为改进已有方法。

比赛：[Name That LLM](https://www.kaggle.com/competitions/name-that-llm)。直接提供有用的源码、权重和有效成绩证据，不以账号提交页面或排行榜代替资料。

| 内容 | 入口 | 状态 |
|---|---|---|
| 八条路线的步骤、资源、对照和取舍 | [详细方案](docs/MULTI_METHOD_PLAN_ZH.md) | 本次主要交付；三个独立建模方向尚待实现/实验 |
| 已评分完整 Notebook 与回执 | [有效提交 0.636](verified/score_0636/README.md) | 隐藏重跑 COMPLETE，Public Score 0.636 |
| 独立 LoRA24 权重与训练代码 | [使用说明](methods/lora24/README.md) | 交付报告 OOF 0.479341，不是 LB；完整逐折预测缺失 |
| CPU 控制与风格特征 | [CPU 组件](methods/cpu/README.md) | 历史本地分组验证；适合对照和补充 |
| 另一公开集成的完整代码 | [已有固定分支](https://github.com/Kidding-Ma/ntl-complement-gpu-handoff/tree/debdb0781600566bd4d211ea7e2f324546124d92) | 未实测；文字 0.651 不代表该源码成绩 |
| 筛选依据与文件来源 | [筛选说明](docs/LOCAL_ASSET_SELECTION.md)、[来源清单](manifests/curated_sources.json) | 排除重复包、截图、缓存、旧提示词和无关材料 |
| 后续实验的统一输出 | [结果规范](docs/EXPERIMENT_CONTRACT.md) | 保留可信 OOF，支持模型比较和融合 |

只读方案/代码，可跳过可选的大权重：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Kidding-Ma/ntl-complement-gpu-handoff.git
cd ntl-complement-gpu-handoff
```

需要复用自训 LoRA24 时，按其说明通过 Git LFS 获取 `checkpoints/own_lora24_checkpoint.zip`。基础模型和其他第三方输入由所列原始渠道下载，不重复托管。

`upstream/`、`references/` 和旧 ready Notebook 保留历史原始来源。使用已有已评分基线时，选 `verified/score_0636/ntl_complement_fixed.ipynb`。历史交接文档的状态不覆盖这次收到的有效提交证据。

本次没有运行新的 GPU 实验或提交比赛。原作者、模型条款及代码许可见 [NOTICE.md](NOTICE.md)。
