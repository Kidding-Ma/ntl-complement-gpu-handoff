我这边没有 GPU，麻烦你让电脑上的 AI 克隆这个交接分支，先读根目录 AI_START_HERE.md，然后用你现有的 NVIDIA GPU 按步骤实际跑完。

这是我指定的 Name That LLM Solution V25 的完整交接，主要加载公开模型和适配器做推理/融合，第一阶段不用从零训练。数据、权重来源、环境、路径、命令、错误处理和验收都在仓库里。先核验输入和环境，再跑完整流程，最后给我能在 Kaggle 隐藏测试断网重跑的 Notebook、完整日志、版本记录、提交文件和中文报告。

不要只给运行建议，也不要把公开三行跑通当比赛完成。原说明里的 .651 和实际代码存在差异，没有本次官方回执就不要报这个成绩。缺少某个权重权限时，把准确来源、错误和待办发回；不要偷偷删模型。所有 token 和账号凭据留在你本机。

GitHub 工作区：https://github.com/Kidding-Ma/ntl-complement-gpu-handoff/tree/name-that-llm-solution-handoff
