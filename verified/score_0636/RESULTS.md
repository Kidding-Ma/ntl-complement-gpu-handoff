# Name That LLM 完整运行报告

更新：2026-10-28，日本时间。固定源码提交：`3b8e3331890281bcf4f41c5c2a37875ca8323811`。

## 当前结论

完整Notebook V1已在Kaggle双T4、禁网环境运行成功。12个必需complement成员、主分支四项、ATLAS及最终融合全部完成公开三行预览。独立验收重新计算了成员求和、标签重排、85/15融合、ATLAS混合、概率与最终预测，全部通过。

原流程计时为**2073.332秒，约34分33秒**，不含其计时器启动前的环境检查。已生成真实`submission.parquet`，三行，列为`id, model`，每条公开测试文本对应一个预测模型名。

已于**2026-10-28 01:09:59 JST**提交这个保存版本进行代码竞赛隐藏重跑，提交号**57706812**，最近状态为`SubmissionStatus.COMPLETE`。本次真实Kaggle **Public Score为0.636**；Private Score尚未公开。该数字来自本次提交回执，并非引用作者历史成绩。

## 四项交付

| 要求 | 文件 | 实际验收 |
|---|---|---|
| 修复后的完整Notebook | `ntl_complement_fixed.ipynb`、`kernel-metadata.json` | Kaggle保存版V1完整运行成功，13个原推理代码单元保留 |
| 模型权重及可用地址 | `MODEL_SOURCES.md`、`weights-manifests/` | 32项原输入已下载对账；本次Kaggle实际挂载；无需新训权重 |
| 运行报告 | 本文件、`delivery_verification.json`、完整运行日志 | 配置、修复、耗时、实际成员与限制均已记录 |
| 提交文件 | `submission.parquet` | 真实完整融合生成，3行公开预览，ID顺序/类别/预测核验通过 |

Notebook：https://www.kaggle.com/code/liuqingmu096/ntl-complement-fixed-3b8e333

提交通过`-k ... -v 1 -f submission.parquet`绑定Notebook版本，**没有把三行预览当成隐藏测试文件直接上传**。隐藏预测由Kaggle重跑动态生成。当前账号为`liuqingmu096`，旧模板`kiddingma`未作为本次发布账号。

## GPU与软件环境

- Kaggle Tesla T4×2，每卡15,636,037,632字节；CUDA12.8。
- Python3.13.15；torch2.11.0+cu128；Transformers5.16.1。
- peft0.20.0、accelerate1.14.0、numpy2.1.3、pandas2.3.3。
- sklearn1.6.1、scipy1.16.3、LightGBM4.6.0、tokenizers0.23.1、safetensors0.8.0。
- 实际镜像：`gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435`。
- 禁网，双T4，运行上限32400秒。保存版确认保留16个Dataset、8个Notebook输出、7个模型V1及比赛挂载。

原PPU服务器仍保留厂商torch2.4、Driver1.5.1/SDK1.6.1。PPU环境的Qwen3.5和UltraBERT兼容性限制并没有被伪装成已修复；完整交付选择了实测可用的Kaggle原生NVIDIA运行时。

## 修复与保持不变的内容

1. 旧Notebook schema转换错误：规范为nbformat4.5、补齐cell ID、清空历史输出，并用nbformat验证。
2. 运行时不支持新模型：先运行私有GPU诊断，确认原生TF5.16.1支持架构/API，再完整运行。未伪造torch版本或取消门槛。
3. 增加失败即停止的运行时检查及原版最终产物验证，保留所有原推理代码单元、权重、EOS、标签、层数、pool、norms及融合权重。
4. 原整包404资源改为官方固定版本逐文件取得，保留回执与SHA。实际Kaggle staging审计中**38个文件**与固定版本恢复回执逐字节一致，包含UltraBERT五折模型和两个主LoRA adapter；该数量不代表所有输入都已逐文件做历史版本对照。
5. PPU探针的分词器API、Python3.10哈希接口、设备初始化和命名空间问题在隔离脚本中处理；它们没有写入原模型资产或改变此次Kaggle原始推理数学。

## 全部必需成员实测

下面的时间是原日志中各成员完成时的**分支累计秒数**，并非每个模型的独立forward耗时。stylegram/stylometry及分类头拟合在CPU运行，神经模型在双T4环境执行。

| complement成员 | 完成时累计秒 | 结果 |
|---|---:|---|
| stylegram | 270 | 通过，CPU |
| stylometry_lgb | 390 | 通过，CPU |
| debv3l | 671 | 通过 |
| electl | 797 | 通过 |
| desk | 945 | 通过 |
| lora24_public | 972 | 通过，原anchor门槛保留 |
| lora24_seed7 | 980 | 通过，原anchor门槛保留 |
| lora24_claude | 990 | 通过，原anchor门槛保留 |
| lora20_q25_instyle | 1016 | 通过，原anchor门槛保留 |
| tokattn_q35 | 1298 | 通过，含40训练行自检 |
| tokattn_deb | 1313 | 通过，含40训练行自检 |
| ultrabert | 1369 | 五折全部通过 |

complement子进程总耗时1390.705秒，含其启动与退出。原自检值只属于训练拟合检查，不是CV。

| 主分支成员 | 主分支完成时累计秒 | 结果 |
|---|---:|---|
| SVC | 504 | 原完整拟合通过，CPU |
| Qwen3-14B LoRA | 561 | 通过 |
| Qwen2.5-14B LoRA | 608 | 通过 |
| DeBERTa | 620 | 通过 |

主分支所有成员log概率之和与最终logits一致。ATLAS完成三条预览的18个source打分；原标签重排、主/complement的85/15混合及ATLAS10%混合均独立复算通过。最终三行概率有限、每行归一，`argmax`模型名称与parquet逐行一致。

## 分组路径的额外验证

公开三行均为单例，不能覆盖完整分组行为。另用真实PPU特征验证了DeBERTa、Qwen3-4B、Qwen3-8B、GLM9B、Qwen2.5-7B的相关Gaussian/twin头：从拟合侧排除5个prompt，查询28行，分组大小1/2/3/4/18，概率检查通过。这是特征层诊断，不是完整OOF。

两个主14B LoRA及public/Claude两个complement LoRA另完成同prompt真实训练文本分组：28行plain、27行centred twin，所有概率检查通过。这些checkpoint可能见过这些训练文本，故同样不是CV或泛化分数。

## 资源、证据与限制

- 本轮没有新LoRA训练，也没有用替代模型冒充缺失成员。原权重来源见`MODEL_SOURCES.md`。
- 模型引用固定V1；四项修复Dataset版本为7/9/1/1。部分首次下载Dataset/Notebook输出的历史版本未固定，不能承诺逐字节复现作者评分输入。
- `kaggle-full-v1-evidence/independent-validation.json`、原`enhanced_verification.json`、全部成员NPZ和日志均在交付中。独立验证文件产生在正式提交之前，其中`competition_submitted=false`是该时点快照；最新提交状态以`submission_receipt.json`、`submission_status.json`为准。
- 完整GPU证据ZIP SHA256：`7a23aaa439bdd34a07ddbbea9a74f6eeedbe16dfb2d26bfead463d089bb107c2`，52个清单文件加清单本身，共53条，本地逐项核验通过。
- 原生CPU参考及Gaussian完整特征另存`native-and-group-validation-evidence.zip`，SHA256 `da0adacd602110b59ef033c8b5e9daeeea0db478fd0f9b14ccd1a0217db3f0f0`。
- 本次代码竞赛隐藏重跑已完成并得到公开分数；隐藏标签、逐行隐藏预测和Private Score未公开。未获取隐藏测试实际耗时，不把公开预览耗时当成隐藏全量耗时。
