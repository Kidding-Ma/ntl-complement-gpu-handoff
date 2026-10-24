# Qwen3.5 加载停滞：复现与修复候选

日期：2026-10-24（Asia/Shanghai）。尚未修复验证通过。

## 已实测

- 原第 7 版 361482022 在 3299.1 秒打印 Qwen3.5 底座路径后长期无新增日志。
- 在同一笔记本新建独立 T4×2 草稿会话，未运行前面的大集成，直接加载相同 Qwen3.5。
- 环境：torch 2.11.0+cu128、transformers 5.16.1、accelerate 1.14.0；两张 Tesla T4，各 15360 MiB。
- attention checkpoint：keep=19、layer=18、max_len=512。
- 7.8 秒进入 AutoModel.from_pretrained；进度停在 0/589。30、60、90 秒时两张 GPU 利用率均为 0%，显存分别为 5399、5471 MiB。后续未见 LOAD_DONE。
- 独立诊断会话持续十多分钟后，通过 Active Events → Interactive Session → Stop Session 停止；UI 已确认 Cancelled。未停止原第 7 版。
- 重新加载编辑器持续出现 Loading Editor、Start session 禁用及“无法连接到 reCAPTCHA 服务”。因此无法执行后续对照，不声称已修复。

## 旧版本对比

版本历史显示第 1～6 版均失败，第 2 版约 1 小时 16 分钟后失败。其最后错误是：

`AssertionError: checkpoint for lora24_public not attached (**/lora24/final/COMPLETE.json)`

它没有执行到 Qwen3.5，不能用它的结束耗时作为第 7 版的成功时间参考。尚不能确定用户说的“5000多秒”具体指哪一个运行。

## 修复候选及依据

当前已安装版本的官方源码显示权重物化默认使用 ThreadPoolExecutor，设置 HF_DEACTIVATE_ASYNC_LOAD=1 会切换为顺序加载。

https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/core_model_loading.py

同版本 modeling_utils.py 记录某些 FUSE 挂载在 mmap + 并发缺页读取时可能死锁，但当前尚无调用栈证实 Kaggle 此处就是相同原因。这是待实测的候选解释，不是已确认根因。

https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py

候选笔记本：own_lora24_ensemble_sync_candidate.ipynb。

- 设置 HF_DEACTIVATE_ASYNC_LOAD=1。
- token-attention 添加 LOAD START / LOAD DONE / BATCH START / BATCH DONE 耗时日志。
- 使用 local_files_only=True；推理 use_cache=False。
- 保留原 token-attention 层数、512 token 上限、40 条自检、所有成员权重与最终 submission.parquet 逻辑。
- 同时包含已在浏览器版本使用的 PEFT torchao dispatcher 屏蔽及 parity 引用释放。
- 本地所有 code cell 和内嵌源码语法已通过；关键融合配置 AST 核对一致。
- 候选基于本地 own_lora24_ensemble_fixed.ipynb 构建，尚未与当前浏览器草稿完成逐单元格比较；不能直接声称它是当前第 7 版的完整同内容修订。

## 恢复连接后的验证顺序

先在同样输入和全新 GPU 会话运行 diagnose_qwen35.py --sync，只做同底座加载及一条真实训练文本 forward。对照默认加载的复现结果；通过才合入当前浏览器草稿。仍停滞时保留栈，针对 mmap 读取做 disable_mmap=True 的单独对照，并确认内存余量，不要同时盲改多个参数。最后再完整运行并核查提交文件。

当前没有提交比赛，也没有宣称预测准确率或评分已得到验证。
