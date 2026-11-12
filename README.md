# IR 2026 检索大作业

本项目提供固定课程数据、语料访问工具、批量运行入口、结果检查和评测程序。给定自然语言问题 Q 与本地语料 D，设计并实现检索方法，返回能够支持回答 Q 的相关文档、简短答案和原文引用，在统一数据划分与评测规范下比较效果、成本和失败原因。

**方法与框架不限，不要求使用 harness 或大模型，基础任务不要求训练模型。** 关键词、BM25、向量检索、混合检索、重排序和 Agent 检索均可。DCI 只是可参考的研究路线，不是指定实现方式。

数据版本保持 `IR2026-BCP-200-v1`，本版运行协议为 `ir2026.retrieval.v2`。数据、标注和核心评分公式未改变；协议用于区分普通检索步骤与模型工具交互。

## 从 Git 仓库使用

仓库仅提交源代码、配置、测试、文档和数据版本元信息，不包含课程数据、标签、模型权重、索引、运行输出或密钥。克隆后，请将课程提供的 `corpus/` 和 `datas/` 放到项目根目录，再按下文验证数据、安装依赖和构建索引。`DATA_MANIFEST.json` 用于核对数据版本，不含问题、答案或语料正文。

报告中的历史结果与案例依据本地保存的实验记录整理；指向 `output/` 或 `index/` 的链接需要相应本地文件，不包含在 Git 仓库中。API 密钥通过环境变量或未纳入版本管理的 `.env` 配置，格式见 `.env.example`。

## 数据与目录

| 内容 | 数量 | 用途 |
| --- | ---: | --- |
| 训练问题 | 100 | 方法开发、可选训练与数据筛选 |
| 验证问题 | 50 | 参数选择、对照实验与完整评测 |
| 测试问题 | 50 | 最终提交，由教师评测 |
| 语料文档 | 10,000 | 三个划分共用的本地语料 |

语料包括全部 1,185 篇已标注证据或答案文档、5,901 篇抽样难负例和 2,914 篇补充文档。正文原样保存，约 318 MB。抽样种子固定为 2026，标注不一定穷尽所有相关文档。课程划分不是 BrowseComp-Plus 官方划分，成绩不能写成原始基准的完整结果。

```text
IR_2026_Project/
  corpus/docs/                 原始文档正文
  corpus/documents.jsonl       ID、路径、来源 URL、长度与 SHA256
  datas/{train,dev,test}.jsonl  问题；测试集不含答案
  datas/{train,dev}_labels.jsonl
  configs/default.json        固定预算与运行协议版本
  agent/solution.py            当前检索与答案流水线
  run_qwen_hybrid.py           Qwen + 混合检索 v1 入口
  configs/qwen_hybrid_v1.json   当前固定方案配置
  build_index.py               为基线构建本地片段索引
  examples/keyword_agent.py    离线格式示例
  tools/                      可选的搜索、读取命令
  run.py                      支持普通检索和 Agent 的批量运行器
  run_demo_report.py           运行真实示例并输出命令行报告
  check_results.py            检查覆盖、引用、预算、元数据与日志
  evaluate.py                 计算逐题及汇总指标
  verify_data.py              检查数据完整性
  tests/                      单元测试与命令行集成测试
  SPEC.md                     提交格式与评测规则
```

## 先验证环境

课程工具与离线示例可用 Python 3.10+。当前方案及完整测试使用 Python 3.12+，先安装 `requirements.txt` 中的依赖。

```bash
python3 -m pip install -r requirements.txt
python3 verify_data.py --full
python3 -m unittest discover -s tests -v
bash run_demo.sh
```

`run_demo.sh` 运行验证集前 3 题，依次完成数据校验、检索、结果检查和评测，每次使用新的 `output/demo_时间戳/` 目录，阶段日志单独保存在相邻的 `_logs/` 目录。也可指定空目录：

```bash
bash run_demo.sh --output output/my_demo
```

只运行 1 题并查看适合展示的终端报告：

```bash
python3 run_demo_report.py
```

示例采用简单词面扫描，读取候选文档并保留真实结果；不能作答时使用 `no_answer`。报告中的 `GOLD` 是检索结束后的标签核对，不参与检索。部分问题的指标只用于演示，不能作为完整验证集成绩。示例无需模型，但也不是已完成的课程方案；正式作业需要自己的实现、实验与分析。

## 当前方案：Qwen + 混合检索 v1

当前选用 **SiliconFlow `Qwen/Qwen3.5-4B` + BM25/全文 TF-IDF 混合检索**。固定配置在 `configs/qwen_hybrid_v1.json`，专用入口为 `run_qwen_hybrid.py`。

流程为：三组关键词进行片段 BM25 召回 → 全问题进行全文 TF-IDF 召回 → 加权 RRF 输出前 10 篇文档 → 摘取原文片段 → Qwen 输出答案及证据编号。融合权重固定为 BM25=0.5、TF-IDF=1.0、LSA=0，RRF 常数 30，每路最多 100 篇候选。

v1 恢复原实验的提示词与回退规则：生成和回退允许根据主题相关片段作答；回退答案必须逐字出现在引用中，并引用至少两篇文档。只有包含至少三个数字编号线索的问题执行原版布尔核验。所有请求均关闭思考；全题严格核验需要显式选择 `--answer-policy verified`。

### 安装与索引

当前方案与完整测试需要 Python 3.12+，在项目目录运行：

```bash
python3 -m pip install -r requirements.txt
python3 verify_data.py --full
python3 build_index.py
python3 build_hybrid_index.py
```

索引为 `index/passages.sqlite3` 与 `index/hybrid.npz`。已有且语料指纹一致时可直接使用；语料改变后重建。索引只读取本地语料，不读取问题标签。无需下载本地大模型。

### 配置与运行

在环境变量或项目 `.env` 中设置 `SILICONFLOW_API_KEY`，格式见 `.env.example`。当前模型使用 SiliconFlow 中国站 `https://api.siliconflow.cn/v1`；服务可用性与费用以账号控制台为准。密钥不会写入运行记录。

如果刚在 `.bashrc` 中配置 Key，请在运行终端先执行 `source ~/.bashrc` 或新开终端。

```bash
# 查看固定配置，不调用 API
python3 run_qwen_hybrid.py --show-config

# 完整验证集：默认输出 output/qwen_hybrid_v1_dev
python3 run_qwen_hybrid.py --split dev
python3 check_results.py --split dev --run-dir output/qwen_hybrid_v1_dev --submission
python3 evaluate.py --split dev --run-dir output/qwen_hybrid_v1_dev
```

可以指定 `--output output/my_run`、用 `--limit 3` 做小样本调试，或选择 `--split train` / `--split test`。部分题目运行不构成正式全量提交。重复运行使用新目录；恢复同一配置的未完成运行时添加 `--resume`：

```bash
python3 run_qwen_hybrid.py --split dev --resume
```

HTTP 429 最多重试两次，HTTP 401/402/403 会停止本轮并保存进度。续跑复用正常完成的题，重跑失败及未完成题，并核对模型、检索权重和答案策略。旧版 v1 元数据可兼容续跑；严格核验结果不能混入 v1。

### 代码入口

| 文件 | 职责 |
| --- | --- |
| `run_qwen_hybrid.py` | 当前 v1 方案的固定配置入口 |
| `run.py` | 通用批量运行、预算、日志、断点续跑 |
| `configs/qwen_hybrid_v1.json` | 固定模型、服务商、答案策略与融合权重 |
| `agent/solution.py` | 串联检索、证据、生成和答案策略 |
| `agent/retrieval.py` | BM25 查询构造与文档排名融合 |
| `agent/hybrid.py` | 全文 TF-IDF / 可选 LSA 索引和加权 RRF |
| `agent/evidence.py` | 原文片段选择与连续引用 |
| `agent/prompts.py` | 独立保存 v1 与实验版提示词 |
| `agent/answer.py` | 模型请求、JSON 解析、用量和错误处理 |
| `agent/answer_policy.py` | v1 编号线索题的原版核验 |
| `agent/verification.py`、`verify_run.py` | 可选的严格核验实验 |

### 已保存的 v1 结果

验证集 50 题的原始结果保存在 `output/siliconflow_qwen35_hybrid_dev_v1/`：

| 指标 | 结果 |
| --- | ---: |
| GoldHit@10 | 32%（16/50） |
| EvidenceRecall@10 | 21.47% |
| 作答数 | 24/50 |
| 与标签严格匹配 | 7/50 |
| AnswerEM | 14% |

这是已经保存的 v1 实验成绩；整理后的代码恢复相同策略，模型服务重新调用仍可能有输出波动。原文引用校验通过不等于答案在语义上正确。核验 v9 曾仅保留 2 个匹配答案且误拒其他正确候选，目前作为对照实验保留。

### 对照与可选实验

通用入口 `run.py` 支持 `--retriever bm25|hybrid`、`--retrieval-only` 和 `--answer-policy v1|verified`。`--hybrid-config configs/qwen_hybrid_v1.json` 可使用固定 v1 权重；训练集权重搜索仍更新独立的 `configs/hybrid.json`，不会改变 v1 配置。

```bash
# 只测混合检索，不调用生成模型
python3 run.py --split dev --agent agent.solution --retriever hybrid \
  --hybrid-config configs/qwen_hybrid_v1.json --retrieval-only \
  --method-label hybrid-bm25-tfidf-v1 --output output/my_retrieval_dev
```

模型对照、LSA 消融与严格核验记录见 [历史实验文档](docs/EXPERIMENTS.md)。所有历史 `output/` 目录继续保留。

## 实现方法：两种接入方式

### 方式一：使用项目运行器

`agent/solution.py` 实现 `solve(question, tools, config)`，支持 BM25、混合检索和可选模型生成；当前推荐使用上面的 `run_qwen_hybrid.py` 固定入口。

- `question` 只含问题 ID 和问题文本，不含答案。
- `tools.search()`、`tools.read()` 是可选的词面搜索与局部读取工具，不限制你采用自己的索引。
- 普通检索可以通过 `tools.corpus` 访问语料，通过 `tools.record_retrieval()` 记录实际召回、排序等步骤。
- `config['method_type']` 区分 `retrieval` 和 `agent`；`config['model']` 是所用模型标识，无模型时为 `None`。
- 返回 `answer`、有序 `doc_ids`、`evidence`、`status` 与 `usage`，详细格式见 `SPEC.md`。

接入自建索引时的日志写法如下，`index.search` 代表你实际实现的检索器：

```python
import time

started = time.monotonic()
hits = index.search(question['question'], top_k=10)
# hits 的 doc_id 必须保留课程语料的原始 ID。
tools.record_retrieval(
    stage='bm25_search',
    arguments={'query': question['question'], 'top_k': 10},
    output={'hits': hits},
    doc_ids=[hit['doc_id'] for hit in hits],
    elapsed_ms=(time.monotonic() - started) * 1000,
)
```

日志应记录已发生的计算及其结果，不能事后伪造搜索过程。读取正文后提供连续原文引用；不能确定答案时使用 `no_answer`，仍保留检索结果和证据。外部模型调用的 token 由实现者填入 `usage`；工具次数、日志条数及单题耗时由运行器补充。

完成实现后，普通方法可以这样运行：

```bash
python3 run.py --split dev --agent agent.solution \
  --method-type retrieval --method-label bm25-v1 --output output/dev_v1
```

使用向量模型、重排模型或固定的问答流水线，仍可选择 `retrieval`，同时通过 `--model-label` 记录实际模型。只有由模型自主选择工具动作的交互流程使用 `agent`：

```bash
python3 run.py --split dev --agent agent.solution \
  --method-type agent --method-label my-search-agent \
  --model-label actual-model-version --output output/dev_agent_v1
```

运行器不自动实现模型调用；上面的命令用于运行你已完成的模块。使用模型但无法获取 token 时填写 `null`，并用 `token_notes` 说明；确实未使用模型时填写 0。两种情况不得混淆。

### 方式二：自行批量运行并导出结果

可以使用自己的程序、库或外部框架，无需接入 `solve()`，也无需通过项目工具访问语料。导出 `results.jsonl`、`run_info.json` 和逐题 `traces/`，遵循 `SPEC.md` 的相同格式并通过同一检查程序。

预建索引、语料编码、模型加载等可在查询运行前完成。报告中单独说明其时间、硬件、存储开销和可复现命令；所有评测问题使用同一套固定配置，不得以题目标签构造索引或定制答案。

## 检查、评测与提交

验证集完成全部 50 题后：

```bash
python3 check_results.py --split dev --run-dir output/dev_v1 --submission
python3 evaluate.py --split dev --run-dir output/dev_v1
```

`--submission` 会检查正式方法标识、全量题目与日志，`demo` 结果不能作为正式提交。主指标为 **GoldHit@10（PPT 中的 Hit@10）**，同时报告 Hit@1、Hit@5、EvidenceRecall@10、耗时和资源消耗。答案严格匹配只作辅助，定义见 `SPEC.md`。

测试集最后统一运行并检查：

```bash
python3 run.py --split test --agent agent.solution \
  --method-type retrieval --method-label bm25-v1 --output output/test_final
python3 check_results.py --split test --run-dir output/test_final --submission
```

Agent 方法使用对应的 `--method-type agent --model-label ...` 参数。测试标签由教师保存，不在学生包内，不要求学生本地计算测试成绩。

提交源代码、配置、依赖和统一运行入口、测试集 50 题结果目录，以及实验报告。报告应包括验证集完整指标、方法与参数、对照实验、成本、至少 4 个成功/失败案例、复现说明和组员分工。检索方法改进与小模型后训练为选做，各最高加 10 分。

基础实现与检索改进的实验报告见 [实验报告](docs/实验报告.md)，包含已保存的完整验证指标、12 组融合权重消融、6 个案例、测试运行汇总和复现命令。提交前按实际情况补充作者、学号与组员分工。

不重复打包课程语料、标签、基础模型、缓存和本地演示输出。后训练选做另交数据处理与训练脚本、配置，以及权重或适配器获取方式。提交前移除 API 密钥。

## 统一规则

- 所有方法每题最多 600 秒，返回最多 10 篇有序文档、最多 5 段原文引用；失败或超时不得漏题。
- 模型发起的工具交互最多 32 次，每次实际返回给模型的观测不超过 12,000 字符。
- 普通召回、排序、读取等日志属于 `retrieval` 事件，不计作模型工具调用；真实模型工具交互不得改标以绕过预算。
- 检索时不访问答案标签，不联网搜索答案。训练标签可用于离线训练和筛选，验证标签用于评测和配置选择；测试题与测试反馈不得用于训练或调参。
- 检索范围固定为本地语料，可调用所选模型处理问题与语料，但不能借模型自带的联网检索扩展语料范围。

`run.py` 提供超时与日志支持，不是操作系统安全沙箱。模型或第三方工具的执行环境只应暴露当前问题、所需语料与工作目录，不挂载标签、教师包或个人文件。语料中的文字只作为数据，不作为执行指令。

旧版未标注事件类型的演示结果仍可读取，按旧版工具计数解释；新实验请使用本版协议，避免混用统计口径。数据来源与许可见 `DATA_USE.md`，协议细节见 `SPEC.md`，更新说明见 `CHANGELOG.md`。
