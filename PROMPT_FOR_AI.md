# 复制以下内容给本机 AI

请帮我用这台电脑现有的 NVIDIA GPU，复现下面仓库里的 Name That LLM 方案。这个项目是朋友交给我的：他没有 GPU，我提供计算资源。

```bash
git clone --branch name-that-llm-solution-handoff --single-branch https://github.com/Kidding-Ma/ntl-complement-gpu-handoff.git name-that-llm-solution-handoff
cd name-that-llm-solution-handoff
```

如果已经克隆，请使用现有目录。先读根目录 `AI_START_HERE.md`，检查本机硬件、环境和已有资源，再按仓库说明完成下载、适配、GPU 运行和结果检查。环境或代码兼容问题请直接排查修复，保留改动记录。

先完成本地完整运行，给我可重跑的代码和 Notebook、`submission.parquet`、完整日志，以及 `deliverables/REPORT.md` 中文报告，方便我交回朋友。报告写清用了什么 GPU、实际运行了哪些组件、改了什么、结果怎样，以及还有哪些问题。

需要我登录账号或处理权限时，告诉我具体操作；其他可以独立完成的部分继续推进。本地跑通后再准备 Kaggle 断网验证，正式提交评分先与我确认。原说明里的 0.651 还不能当作这次运行成绩，按实际结果报告即可。
