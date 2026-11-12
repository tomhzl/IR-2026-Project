# 历史实验与可选方法

当前选用 **Qwen + 混合检索 v1**，入口为 `run_qwen_hybrid.py`，使用说明见 [README](../README.md)。本页保存历史实验；严格核验属于可选实验，不是默认流程。历史指标不能等同于重新运行当前代码的成绩。

## FTS5/BM25 检索基线

当前 `agent/solution.py` 使用本地 SQLite FTS5 对语料片段建立 BM25 索引，按问题中的多组关键词召回片段，用倒数排名融合得到最多 10 篇文档，再从原始正文摘取连续引用。默认不调用模型；对原文中明确陈述且能与问题主题对应的百分比，使用保守规则提取答案，其余题返回 `no_answer`。这是可复现的基线，仍需改进复杂问题的作答、检索与案例分析。

首次运行前先建索引。索引文件写入 `index/passages.sqlite3`，属于可重建缓存，不应打包提交。每次重建会用临时文件生成并原子替换旧索引。

```bash
python3 verify_data.py --full
python3 build_index.py
python3 run.py --split dev --agent agent.solution \
  --method-type retrieval --method-label fts5-bm25-answer-v9 --output output/fts5_bm25_dev
python3 check_results.py --split dev --run-dir output/fts5_bm25_dev --submission
python3 evaluate.py --split dev --run-dir output/fts5_bm25_dev
```

`run.py` 要求输出目录不存在或为空；重复实验请换一个新目录。索引构建时间、大小及查询耗时应在报告中单独记录。当前基线的检索日志由 `tools.record_retrieval()` 记录实际发生的索引查询、融合、原文读取与答案提取。

如需用模型处理其余问题，可在确认允许向所选服务发送问题与本地语料片段后，设置 `IR2026_API_BASE`（默认 `https://api.deepseek.com`）和 `IR2026_API_KEY`（默认地址也可使用 `DEEPSEEK_API_KEY`），并在 `run.py` 命令中添加 `--model-label`，填写该服务实际支持的模型标识。例如使用已配置的 DeepSeek 凭据：

```bash
python3 run.py --split dev --agent agent.solution \
  --method-type retrieval --method-label fts5-bm25-answer-v9 \
  --model-label deepseek-flash --output output/dev_with_model
python3 check_results.py --split dev --run-dir output/dev_with_model --submission
python3 evaluate.py --split dev --run-dir output/dev_with_model
```

实现使用兼容 Chat Completions 的 JSON 输出接口，不启用模型联网搜索；前 10 篇文档均提供候选片段。排名第一的文档保留 BM25 片段覆盖，其他文档结合 FTS 片段、开头相邻窗口，以及问题词在原文中的位置；每段引用不超过协议规定的 1200 字符。首轮弃答时，会用每篇文档的少量聚焦片段再尝试提取原文答案；回退答案须引用至少两篇不同文档的片段。弃答时返回最多 3 段来自不同候选文档的相关原文。只接受模型返回的片段编号，再从本地原文提供引用。现在对所有非空候选答案执行类型检查及独立核验，不再只检查数字编号题。核验要求核心关系成立、条件有原文支持；对已有多条真实证据但存在背景疑点的候选，最多追加一次目标关系复核。模型用量由 API 响应记录；调用失败或证据无效时保留 `no_answer` 和日志。模型运行仍属于固定的 `retrieval` 流程。验证集结果受检索召回限制；当前基线不代表已完成的课程方案。

已有训练集运行记录在 `output/deepseek_answer_train_v5/`，验证集改进结果见 `output/deepseek_answer_dev_v9/`。训练集 v5 GoldHit@10 为 0.25、AnswerEM 为 0.10；验证集 v9 GoldHit@10 为 0.24、AnswerEM 为 0.08，50 题中作答 4 题且均与标签一致。使用已有检索结果离线比较片段覆盖：命中标注文档的题中，训练集标注答案字面出现在片段中的题数从 20/25 增至 22/25，验证集从 6/12 增至 8/12。该字面检查不涵盖需要推导或格式转换的答案。模型返回的原文片段通过字面校验，但自动检查不保证引用在语义上支持答案。

## 混合检索

`agent/solution.py` 支持 `--retriever hybrid`。它将原有片段 BM25/RRF 排名与完整问题的全文 TF-IDF 排名融合；额外实现了 192 维 LSA 召回供消融对照。TF-IDF 使用 unigram、bigram 和次线性词频，索引覆盖全部 10,000 篇文档的全文。LSA 是本地语料训练的降维向量，并非预训练神经 embedding；方法说明见 [scikit-learn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD)。

在训练集上比较 12 组融合权重后，`configs/hybrid.json` 选择 BM25=0.5、TF-IDF=1.0、LSA=0.0。每路取前 100 篇，用加权 RRF（常数 30）融合，输出前 10 篇；LSA 权重为 0 时不加载或执行 LSA。索引训练只读取本地语料；问题和标签不进入索引，标签仅在离线实验生成排名后用于评分。

混合检索依赖当前 `requirements.txt` 中的数值计算库，需要 Python 3.12+（当前环境为 3.13）。首次使用：

```bash
python3 -m pip install -r requirements.txt
python3 build_index.py
python3 build_hybrid_index.py
```

新增索引为 `index/hybrid.npz`，本机构建耗时约 198 秒、大小约 144 MiB，详情在 `index/hybrid.build.json`。索引可重建，不应打包提交。两个索引均校验语料清单指纹，重建时原子替换，避免加载未完成的文件。

只评测检索命中率（不生成答案，结果为 `no_answer` 是预期行为）：

```bash
python3 run.py --split dev --agent agent.solution --retriever hybrid \
  --retrieval-only --method-label hybrid-bm25-tfidf-v1 --output output/my_hybrid_dev
python3 check_results.py --split dev --run-dir output/my_hybrid_dev --submission
python3 evaluate.py --split dev --run-dir output/my_hybrid_dev
```

接入现有 DeepSeek 答案和证据生成：

```bash
python3 run.py --split dev --agent agent.solution --retriever hybrid \
  --model-label deepseek-flash --method-label hybrid-answer-v1 --output output/my_hybrid_answer_dev
```

也可使用 SiliconFlow 中国站的免费 `Qwen/Qwen3.5-4B`。官方案例将其列为免费模型，实际可用性及限制以[平台说明](https://docs.siliconflow.cn/docs/usercases/use-siliconcloud-in-lightrans)和账号控制台为准。将 Key 配置到环境变量 `SILICONFLOW_API_KEY`，或复制 `.env.example` 为 `.env` 后填写；程序不会记录密钥。运行：

```bash
python3 run.py --split dev --agent agent.solution --retriever hybrid \
  --api-provider siliconflow --model-label Qwen/Qwen3.5-4B \
  --answer-policy verified --method-label hybrid-qwen35-4b-verified-v9 --output output/my_qwen35_hybrid_verified_dev
```

该配置使用 `https://api.siliconflow.cn/v1/chat/completions`、`enable_thinking=false` 和 JSON 输出，生成答案并校验引用编号。HTTP 429 最多重试两次；HTTP 401/402/403 会将当前题记为 `error` 并停止本轮，不再把账户错误记录为正常弃答。保留原参数并添加 `--resume` 可跳过已成功处理的题，重跑失败及未完成题；续跑会核对问题、模型、服务商、检索配置，避免混用不同实验结果。

```bash
python3 run.py --split dev --agent agent.solution --retriever hybrid \
  --api-provider siliconflow --model-label Qwen/Qwen3.5-4B \
  --answer-policy verified --method-label hybrid-qwen35-4b-verified-v9 --output output/my_qwen35_hybrid_verified_dev --resume
```

SiliconFlow 实验目录为 `output/siliconflow_qwen35_hybrid_dev_v1/`。此前 `output/hybrid_answer_dev_v1/` 的 DeepSeek 实验因 HTTP 402 中断：22 题已尝试，其中 13 题未发生 API 错误、9 题调用失败，其余 28 题未运行；详情在该目录的 `partial_summary.json`，不能将该部分结果作为完整验证集成绩。

SiliconFlow Qwen3.5-4B 已完成全部 50 道验证题并通过提交校验：GoldHit@10=0.32、EvidenceRecall@10=0.2147；24 题输出答案，其中 7 题与标签严格匹配，AnswerEM=0.14。平均每题约 14.55 秒，输入 579,764 tokens、输出 2,039 tokens，调用日志中无 API 错误。引用原文字面校验通过率为 100%，但这不等于答案或引用在语义上正确。例如第 478 题输出包含相同的体育场与日期，因顺序与标签不同而未通过严格匹配；另有多题输出了与标签不同的人物、作品或年份，需要进一步提高候选答案核对质量。

当前新增核验策略为 `source-grounded-v9`，实现位于 `agent/verification.py`：

- 所有候选均核验，包括普通段落及破折号列出的题；数值不能作为“谁”的答案。
- 答案相关文档与关键条件优先进入较短核验上下文，最多 14 段，保留同文档中使用代词的相邻证据。
- SiliconFlow 核验请求使用严格 JSON Schema，限制布尔字段、合法片段编号和输出长度；模型说 `supported=true` 仍需通过程序的证据检查。
- 条件与直接答案支持必须定位到本地原文连续片段。允许排版标点差异；明确省略号会展开成原文中的完整连续片段。词序、实词和数值不匹配则拒绝。正面核验若漏填实体短引文，仅可从其已引用的同一来源抽取原样实体，不生成新值。
- 至少两条真实证据且没有明确矛盾时，才允许一次额外目标关系复核；仍需经过相同的原文与类型校验。
- 未通过的候选保存在 `candidate_answer`，`verification.status` 和 `verification.reasons` 说明原因；正式 `answer` 留空。核验失败表示未能证实，不等于候选已被证伪。

生成与核验默认都关闭思考。可设置 `IR2026_VERIFY_THINKING=1`，仅在核验中启用 1024 token 思考预算；本轮试验没有采用该模式作为默认配置。

可对既有候选单独复核，不重做召回或生成：

```bash
python3 verify_run.py --run-dir output/siliconflow_qwen35_hybrid_dev_v1 \
  --output output/my_verified_dev
python3 check_results.py --split dev --run-dir output/my_verified_dev --submission
python3 evaluate.py --split dev --run-dir output/my_verified_dev
```

固定 v1 的 24 个候选进行核验后，结果在 `output/siliconflow_qwen35_hybrid_verified_dev_v9/`：保留 2 题，两题都严格匹配；原先 17 个不匹配候选均未保留，但也拒绝了 5 个标签正确的候选。AnswerEM 从 14% 降至 4%，说明当前策略减少了不可靠输出，同时存在明显的过度弃答，不能称为整体答案质量提升。Hit@10 保持 32%。其中一次核验请求超时后已补做，历史失败与真实重试用量保留在日志中；该题此前超时用量未知，因此累计 token 用量保留 `null`，不按零计。

该实验只比较同一批候选的核验效果，未重新评测当前完整生成流水线。派生结果保留真实原始检索和生成日志，再追加核验与原文对齐日志，耗时和用量累计统计；标签仅在运行后的 `evaluate.py` 中使用。中间 `verified_dev_v2` 为中止的试验，`verified_dev_v3` 和 `verified_dev_v8` 为过度弃答的对照，不是推荐结果。

以下表格只比较检索效果，模型作答指标应单独查看对应模型实验目录。使用固定训练集权重的完整实验如下：

| 检索方法 | Train Hit@10（100题） | Dev Hit@10（50题） | Dev EvidenceRecall@10 |
| --- | ---: | ---: | ---: |
| 原 BM25 基线 | 25% | 24% | 9.24% |
| 全文 TF-IDF | 39% | 34% | 20.55% |
| LSA | 21% | 30% | 14.71% |
| BM25 + TF-IDF 融合 | 45% | 32% | 21.47% |

融合的 Dev Hit@1 从基线的 6% 升至 14%。Dev Hit@10 比单独 TF-IDF 少命中 1 题；默认权重依据训练集选择，未用验证集重新调权。正式运行结果在 `output/hybrid_train_v1/`、`output/hybrid_dev_v1/`，单路消融和完整候选排名在 `output/hybrid_{train,dev}_ablation_v1/`。这些实验目录中的 AnswerEM 不用于比较答案质量。

复现实验（选权重会更新 `configs/hybrid.json`；训练集选权重后再跑验证集）：

```bash
python3 benchmark_hybrid.py --split train --select-config --output output/my_hybrid_train_ablation
python3 benchmark_hybrid.py --split dev --output output/my_hybrid_dev_ablation
```

`--retriever bm25` 可运行原基线。运行日志记录各路实际候选和分数、融合权重、最终排名及本地引用。
