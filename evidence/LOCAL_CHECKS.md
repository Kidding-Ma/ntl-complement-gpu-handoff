# 发送方检查记录

日期：2026-10-27（北京时间）。执行机器：Windows，Python 3.13；本次不使用 GPU。以下检查验证交接工具，**不构成模型/正式提交验证**。

| 检查 | 结果 |
|---|---|
| 指定源公开 pull API | HTTP 200；V25；7 cells，6 code；完整 API/blob/metadata 已存 |
| 输入引用 | 33 = 16 Dataset + 9 Notebook Output + 7 Model version + 1 Competition |
| 不可变源 SHA-256 | `verify_workspace.py` 的 13 个源码/元数据文件全部通过 |
| 提取模块和 helper 静态语法 | 全部通过；动态 setup 拼接也能解析 |
| 自包含派生 Notebook | 8 cells；nbformat schema 验证通过；没有执行模型 |
| CPU 合约测试 | 8 项全部通过，命令 `python -m unittest discover -s tests -v` |
| 验收测试覆盖 | 正常 schema、重复/错误 ID、错误 label、组件跳过、smoke 模式、缺 adapter 完成行、comp ID 错位、缺 shard、下载版本参数、Notebook 模块逐字一致 |
| 凭据模式扫描 | 未发现私钥头、GitHub/HF/Kaggle token 模式；只报告文件/类别，不打印任何凭据 |
| 参考源码抓取 | 10 份公开参考源码可取，avazaku 被引用的 reproduction 为 404；逐项结果见 reference-fetch.json |

开发中，模块逐字一致测试发现 Windows 文本换行转换；生成器改为 write_bytes 后测试通过。源模块未为修复此问题改写。

未完成：完整权重/比赛数据下载、GPU 环境安装验证、任何训练或神经推理、显存/主存峰值、完整隐藏负载、Dataset/Notebook 的历史挂载版本解析、官方成绩独立核验。发送方 Kaggle CLI 没有有效认证；交给接收方合法下载。pip 依赖范围只是候选环境，不是假造的上游 lock。

发布方式：原有 `Kidding-Ma/ntl-complement-gpu-handoff` 仓库的独立 `name-that-llm-solution-handoff` 分支。该分支根目录就是本工作区；不修改既有主分支，克隆时必须指定 branch。GitHub 页面建仓操作被工具因无法确认浏览器 URL 而中止，改用既有 SSH Git 连接发布本分支。
