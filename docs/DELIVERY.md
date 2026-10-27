# GPU 接收方交付与验收

提交一个轻量结果包或可访问的 Git 分支，包含以下平行项目：

- 完成的 REPORT.md（按 deliverables/REPORT_TEMPLATE.md 填写），明确 success / incomplete / blocked。
- 最终 Notebook、完整 scripts/config、相对源快照的 diff、Git commit、文件 SHA-256。
- input inventory：每项 owner/ref/version、实际本地路径、下载状态、元数据/文件 hash；无法锁定的版本明确写 unknown。
- 环境：OS、GPU/显存、驱动、Python、CUDA、torch、pip freeze、启动命令；环境变更原因。
- 执行证据：从开始到结束的 setup/complement/inference 全日志、时间、退出码、run-status、显存采样、preflight。
- component table：18 个 v5 adapter 和其他主/互补成员的实际加载/完成；COMP STACK DONE、comp stack loaded、ATLAS mixed at 0.16、所有跳过项及理由。
- submission.parquet 和验证报告；ID 与运行当次 test 一一对应，无重复/空值、只含合法 18 类。
- comp_logit.npy / comp_meta.json（较小可另附），形状 `(n_test,18)`，有限值、类顺序与 ID 一致。
- Kaggle 隐藏重跑的精确版本/时间/结果或失败证据；有官方分数才填写，没有就写“未取得”。

不能验收：只有建议/截图/三行 submission；只有 `exit 0`；缺权重后删模型；ATLAS 超时后跳过；训练自检高准确率；输出继承旧缓存；拿 README 的 .651 充当本次分数；只给本地路径而接收方无法下载。

原注释允许的历史 lora24_public skip 单独披露；NTL_DROP_Q35=1 是本快照配置。其余缺失均不能默认为“原方案本来就这样”。即使可见运行全部通过，隐藏测试和时间分支仍需官方重跑验证。

大型基础模型/数据默认不进 Git，不必传给无 GPU 的委托人。通过 Kaggle 输入或其他合法共享源交接，保留模型/数据原许可。不上传 token、账号配置、SSH 私钥、个人信息或无关文件。
