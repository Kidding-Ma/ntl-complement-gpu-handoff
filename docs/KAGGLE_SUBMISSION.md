# Kaggle 断网提交

本比赛要求 Notebook 在隐藏测试上重新执行；本地可见 test 的 submission 不能代替正式推理。比赛信息以 [官方 overview](https://www.kaggle.com/competitions/name-that-llm/overview) 为准：当前公开要求 GPU/CPU 9h 内、Internet off，输出 `submission.parquet`。

1. 接收方先完成本地/自己 Kaggle 的依赖与 GPU 验收，再把 `notebooks/solution_strict.ipynb` 导入 Kaggle。它自包含，不需要运行时访问 GitHub。
2. 按 config/inputs.json 附齐 33 项输入，确认可访问和选中版本；特别是 Llama cache 输出、v5 adapters、ATLAS bundle、UltraBERT 5 folds、offline wheels。没有输入时只导入 ipynb 不能运行。
3. 选择可用 GPU（上游文字称 T4×2），Internet off。保留原路径 `/kaggle/input` 与 `/kaggle/working`。若账号需接受比赛/模型条款，由账号持有人操作。
4. 若使用 CLI，在 notebooks/ 下复制 kernel-metadata.template.json 为 kernel-metadata.json，填自己的 owner/slug、确认 sources 和权限，再 `kaggle kernels push -p notebooks`。模板本身不会上传或提交；API 不保证固定 Dataset/Notebook Output 历史版本，请在页面核查。
5. Save Version → Save & Run All。严格版对 complement/infer 返回码、结果、日志组件完成进行检查；仍需审查全部日志、路径和适配器完成。Notebook 成功只说明该次可见输入执行通过。
6. 确认没有模拟环境/假测试、没有已知失败被忽略，Notebook 已输出 `/kaggle/working/submission.parquet`，schema 为 id/model。`submission.csv` 仅供查看。
7. 正式提交前由账号持有人决定并操作/授权；不要替无授权账号报名或提交。记录精确 Notebook version、scriptVersionId、提交时间、状态、分数、运行失败日志。
8. 把最终已跑通的精确 Notebook 导出给委托人，附全部来源及版本。若委托人 Kaggle 缺某输入权限，应提前解决合法可访问的 transport，并列出差异和验证，不能声称“复制即可跑”。

原始 Notebook 另存于 references/upstream/source.ipynb；严格派生版移除了吞退出码的 `!` 调用，并加入校验。它保留当前 V25 实际系数，不把 README 中的 0.10/0.3 写回代码。如果需要比较 avazaku 原 .651 配方，另建实验并分别命名，不覆盖此次复现。
