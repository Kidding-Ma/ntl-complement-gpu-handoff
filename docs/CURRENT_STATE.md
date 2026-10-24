# 2026-10-24 交接状态

## 已确认

- 指定链接：`https://www.kaggle.com/code/fleongg/ntl-part-two-complement`。
- 浏览器当前公开版本 V2，scriptVersionId `361521141`；标题 0.636 来自作者说明中 V1 的提交 `57551442`。未从用户账号复现该成绩。
- V2 Logs 显示 `Successfully ran in 9.1s`、`Accelerator None`、`Output 0 B`。只有 nbconvert 输出，另有 markdown cell `id` 与旧 notebook schema 不匹配的转换错误。未看到真实模型推理日志。
- 公共 API 返回 V2 完整 24 单元格，metadata `enableGpu=true`、T4；字段与 UI 执行状态不同，不把它混作成功运行证据。API 来源和摘要见 `evidence/source_provenance.json`。
- 原代码是 inference：主模型 SVC、Qwen3-14B LoRA、Qwen2.5-14B LoRA、DeBERTa；以 85/15 概率混合 main 与 complement，margin<0.3 的行再混入 10% ATLAS，最后 Sinkhorn（>=50 行）。训练新的模型不是默认前置条件。
- 原代码有 12 个必需 complement 成员，结束时断言；还有 Gaussian/centred views。不能把 12 理解成只需要 12 份文件。
- Input 页列出 46.82 GB 的输入区，另有模型列表；不要将它当成全部基础模型加 scratch 的磁盘总需求。
- Notebook 侧栏曾显示 `[Model not available]`；Input 页能列出 7 个 Qwen 模型。作者明确说明部分辅助归档可见性可能限制 fork。**目前不能指定哪一个资源必定缺失，需接收者使用自己的账户逐项验证。**

## 本次没有完成

- 无本版本 GPU 推理、全融合、CV 或真实 Kaggle 提交。
- 无本版本全套权重/文本池离线副本；32 个输入 ref 已完整列出，但外部可访问性及版本一致性尚待验证。
- 本地 Kaggle CLI 未配置 API 授权；公开源码通过无需登录的官方 pull API 取得。浏览器登录与 CLI 凭据是两回事。
- requirements 是参考环境，不是 GPU 实测 lockfile。

## 历史上下文（避免混用）

用户旧的 `kiddingma/ntl-sota-0604-dual-lora-gaussian-hungarian` 方案曾缺 `lora24_public`，后来尝试自训 LoRA24；又遇到 Qwen3.5 加载停滞。这里只附相关诊断源码及记录，不把旧自训权重作为本次 exact legacy checkpoint，也不把旧方案 0.604 等成绩转移到本方案。

所有“已验证”范围均以本仓库检查报告或真实 GPU 日志为准。
