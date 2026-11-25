# 小模型 LoRA 后训练（v1）

## 数据边界

教师轨迹只来自 `output/qwen_bailian_ltr_v1_train_teacher`，即 train 100 题上的 `qwen3.5-flash` 与冻结 pairwise LTR 固定流水线。dev 只用于训练前后评测，test 未用于教师采集、数据筛选、训练或选参，也没有因本实验重新运行。

教师运行包含 52 个作答和 48 个弃答。只使用 train 标签进行以下筛选：

- 保留 32 个 AnswerEM 正确的教师答案；
- 保留 42 个未召回金标时的合理弃答；
- 排除 20 个错误答案；
- 排除 6 个已召回金标却仍弃答的模糊样本。

最终 `sft_train.jsonl` 共 74 条，其中 73 条监督首次答案生成，1 条监督 focused fallback。数据 SHA256 为 `17c0930b5bfd6f0684df4bebe3a8e6b53c95868726ff3841618a11338777b7f7`。问题、系统指令和检索片段是输入；唯一生成目标是下一步 `{"answer":...,"evidence_ids":[...]}` JSON。工具返回和原文片段不作为生成目标。包含语料片段的 SFT JSONL 保留在本地 `output/`，不提交到 GitHub。

## 模型与训练

- 基础模型：`Qwen/Qwen2.5-1.5B-Instruct`，本地 BF16；
- LoRA：r=8、alpha=16、dropout=0.05；
- 目标模块：Q/K/V/O 与 MLP 的 gate/up/down；
- 可训练参数：9,232,384，占全部参数约 0.5945%；
- 最大序列长度：10752，覆盖最长 10470 Token 的样本，无截断；
- 3 epochs、54 个优化步、学习率 1e-4、cosine scheduler；
- micro batch 1、梯度累积 4、有效 batch size 4、随机种子 2026；
- 训练耗时 499.08 秒，峰值 GPU 显存 21.7119GB；
- 平均训练 loss 0.2572517。

只提交约 37MB 的 PEFT 适配器，不提交基础模型。训练与推理均使用完整序列 SDPA、确定性解码和相同的本地 JSON 格式归一化。

## 同流程 dev 对照

两次运行均使用同一 Qwen2.5-1.5B-Instruct 基础模型、同一冻结 LTR 和同一检索/证据流水线，唯一模型变化是是否加载 `adapters/qwen25_lora_v1`。

| 指标 | 基础模型 | LoRA v1 | 变化 |
|---|---:|---:|---:|
| GoldHit@1 | 12% | 12% | 0 |
| GoldHit@5 | 30% | 30% | 0 |
| GoldHit@10 | 38% | 38% | 0 |
| EvidenceRecall@10 | 22.4738% | 22.4738% | 0 |
| AnswerEM | 6% | 14% | +8 个百分点 |
| 作答完成率 | 50% | 92% | +42 个百分点 |
| `ok / no_answer` | 25 / 25 | 46 / 4 | +21 个作答 |
| 输入 Token | 514842 | 422730 | -92112 |
| 输出 Token | 2986 | 982 | -2004 |
| 总耗时（秒） | 1934.8476 | 1924.8116 | -10.0360 |
| 最大单题耗时（秒） | 64.9436 | 41.9265 | -23.0171 |
| 平均 trace 事件 | 10.94 | 10.16 | -0.78 |
| focused fallback | 20 | 1 | -19 |
| 生成格式/长度错误 | 5 | 0 | -5 |

LoRA 新增正确题 `517、576、713、769`，保留原有正确题 `191、540、628`，没有 AnswerEM 正确题回退。状态迁移包括 22 个 `no_answer->ok` 和 1 个 `ok->no_answer`。两次均为 50/50 题有效、无 error/timeout、Token 完整、引用原文字面匹配率 100%。固定检索流水线的模型工具调用均为 0；本地答案生成不计为远程 API 调用。

这是 50 题 dev 上的单次固定配置对照，不声称统计显著。AnswerEM 是保守的归一化精确匹配，不等同于语义正确率或教师 LLM 评判。没有根据该 dev 结果继续调参，也没有改动或重跑冻结 test。

## 复现

```bash
# 仅由 train 教师轨迹构造数据
python3 experiments/build_sft_data.py   --run-dir output/qwen_bailian_ltr_v1_train_teacher   --output-dir output/sft_teacher_v1

# 训练
CUDA_VISIBLE_DEVICES=0 python3 experiments/train_lora.py   --config configs/lora_v1.json

# 训练前基线
CUDA_VISIBLE_DEVICES=0 python3 run_full.py   --split dev --preset configs/qwen25_base_ltr_v1.json   --output output/qwen25_base_ltr_v1_dev

# 训练后评测
CUDA_VISIBLE_DEVICES=0 python3 run_full.py   --split dev --preset configs/qwen25_lora_v1.json   --output output/qwen25_lora_v1_dev

# 汇总对照
python3 experiments/compare_posttraining.py   --baseline output/qwen25_base_ltr_v1_dev   --proposed output/qwen25_lora_v1_dev   --output experiments/sft_dev_comparison_v1.json
```

详细数据统计见 `experiments/sft_teacher_v1_stats.json`，训练记录见 `experiments/lora_v1_training_record.json`，完整 dev 对照见 `experiments/sft_dev_comparison_v1.json`。
