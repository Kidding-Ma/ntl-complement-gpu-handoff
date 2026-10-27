# 指定源码审计

证据：`references/upstream/api_response.json`、`metadata.json`、`source.ipynb`、`cell_00.md` 至 `cell_06.py`；SHA-256 在 `config/source-lock.json`。抓取的是指定 owner/slug 的 API 返回值，currentVersionNumber=25，不是把另一个方案替换进来。

## 文字与代码不能混为一谈

| 项目 | Markdown 声明 | V25 可执行代码 |
|---|---|---|
| 0.651 归属 | avazaku V3，script 362281064；父级 dhruvtibarewal nlm-v27 V1 | 指定链接为 mohammedafreedpasha V25；没有本次官方评分证据 |
| 输入数量 | 14 Dataset + 9 Notebook + 7 Model + 1 Competition = 31 | pull metadata 为 16 + 9 + 7 + 1 = **33** |
| Llama v5 权重 | 表格称 1 | `ll8v5` 为 **1.5**，另外保留 `ll8_ts` 1.0 |
| DeBERTa 权重 | 表格称 1 | `deb` 为 **1.3** |
| ATLAS | 0.10，margin 0.3 | **0.16，margin 0.38**；comp_w=0.15 |
| 数据 | 说明的 14 项表格 | 额外声明 danielandresmiranda/ntl-openrouter-glm-kimi 和 kirilltym/ntl-spectral-style-heads-paired-4404；有条件拼接 GLM/Kimi CSV |
| 容器 | 说明引用 `9a6bcf...` | 当前 metadata 为 `2757e0...`（完整值见 metadata.json） |
| 时间 | 说明含 10 月 26 日提交 | metadata.lastRunTime 显示 10 月 24 日，说明与执行身份需分别核对 |

不能继承文字中的 code digest、4h04m16s、所有适配器完成或 .651 作为本快照事实。README 的算法表也不是运行配置，实际代码优先。本地未独立核验其成绩回执。

## 原始执行内容

- cell 1：路径/权重发现、bitsandbytes wheel 安装、可选 GLM/Kimi 合并。
- cell 2–4：写出 `gauss.py`、`infer.py`、`comp_stack.py`。
- cell 5：`NTL_DROP_Q35=1`，独立进程运行 complement，统计耗时。
- cell 6：8 个主模型规格，18 个 v5 fold/full adapters，加上 Llama two-stage、seed7、Nishant 与编码器，叠加 SVC / adaptive twins / Gaussian / complement / ATLAS / Sinkhorn / guards / Hungarian；写 parquet 与 CSV。
- `ATLAS_BUDGET=int(8.3*3600-COMP_SEC-600)`，这是相对 infer.py 起点的时间上限，包含加载等已花时间，不能解读为独立的 8.3 小时 ATLAS。
- 短 test 的训练行自检为 in-sample；infer 部分在少于 18 行时不做 Sinkhorn，ATLAS margin 筛选在少于 50 行时不同。因此 3 行可运行不能证明隐藏分支或正式耗时。

## 失效风险与交接工具的处理

1. `!python ... | grep -vE 'Warning|warn'` 可以掩盖前一进程的退出码并过滤信息；严格入口改为 subprocess 逐行保留日志并检查返回码。
2. 原 `infer.py` 捕获 COMP/ATLAS 异常后继续。严格入口检查实际加载、mixed 标记和禁止词；退出码 0 不足以验收。
3. `glob(...)[0]`/`next(...)` 隐式选第一个路径。预检报告主路径歧义，清单下载保留目录结构；不要把多个版本混入 input。
4. 头文件齐全不代表基础模型 shard 齐全，预检检查 config 对应 shard index 以及单文件权重。
5. complement 的 `find_checkpoint` 按 model_type 和 hidden_size 选模型，可能选中不想要的 Qwen3 base/instruct；完整运行必须记录实际路径并与原挂载布局对照。
6. 上游历史 `lora24_public` 没有权重时代码会跳过。仅此历史项可作为记录在案的可用输入限制；其他 skip 会阻止严格验收。
7. `torch.load` / `trust_remote_code=True` 依赖上游模型代码和环境；接收方应使用授权输入，固定版本。变更兼容性不能偷偷替换模型。

严格版更改执行与验收，不修改三个算法模块和模型系数；路径替换仅发生在独立 stage 目录。`src/` 是从 %%writefile 后逐字提取的模块。

## 已验证的范围

发送方可以确认源码抓取、结构、静态语法、hash 与工具的合成数据验收；具体结果在 evidence/LOCAL_CHECKS.md。未下载完整 33 项大型资产，未运行 GPU，未建立精确上游 pip freeze，未完成 Kaggle 隐藏测试评分，也没有所有数据/适配器训练过程的复现证据。
