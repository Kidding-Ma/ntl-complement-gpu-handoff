# 当前状态（2026-10-28）

当前主要交付为 [多方案详细计划](MULTI_METHOD_PLAN_ZH.md)，包括独立新方法、公开复现、数据路线和融合。

- 已收到并保存正式提交 57706812 的 COMPLETE 回执，Public Score 0.636，见 [有效提交](../verified/score_0636/README.md)。
- 该次公开预览全部成员和概率混合通过，约 2073.33 秒；不把它当隐藏全量耗时。
- 独立 LoRA24 的 final 权重和训练代码可用，原报告 OOF BA 0.479341；本次只做结构/hash 校验，未重新 GPU forward。完整逐折预测尚缺。
- 另一条 V25 公开大型集成有源码与输入清单，未确认该版本真实成绩。
- A/B/C 等新方法目前是详细实验设计，尚未实现或运行；不把计划当实测结论。

原始来源仍在 upstream/references，旧 GPU_RUNBOOK/DEPENDENCIES 仅供原始路线使用；已评分入口为 verified/score_0636 中的 fixed Notebook，不能混成根目录早期 ready Notebook。
