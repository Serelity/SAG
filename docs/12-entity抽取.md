# 12-entity抽取

日期：2026-07-30

## 1. 目标

本阶段只做服务 SAG 检索的 entity 抽取，不做完整治理闭环 schema。

目标是把 12345 工单抽成可用于 SQL join / dynamic hyperedge 的实体：

```text
problem_object
problem_behavior
area
street
road
intersection
poi
case_type
time_month
department
lnglat
```

其中 `area` 默认只做过滤和排序，不做一跳扩展 frontier。

## 2. 为什么选择 Qwen3-8B

本任务是结构化抽取，不是复杂推理。默认模型使用 `Qwen/Qwen3-8B`，在抽取质量和本地部署成本之间更稳妥。

代码中从魔塔 ModelScope 下载模型到服务器本地。抽取时默认关闭 Qwen3 thinking mode，只要求模型输出 JSON。

你的服务器是 32GB V100，代码在 CUDA 可用时默认使用 `float16`，不使用 `bfloat16`。

## 3. 服务器准备

```bash
pip install -r requirements.sag.txt
pip install -r requirements.entity.txt
```

含义：

```text
requirements.sag.txt 安装 DuckDB。
requirements.entity.txt 安装 transformers、accelerate、modelscope 等大模型抽取依赖。
PyTorch 由服务器 CUDA 环境单独管理。
```

如果服务器环境还没有 PyTorch，请按你的 CUDA 版本安装对应的 PyTorch。不要在本项目 requirements 里固定 PyTorch，避免装错 CUDA wheel。

## 4. 下载模型到服务器本地

```bash
bash scripts/download_entity_model.sh
```

默认输出：

```text
models/Qwen3-8B
```

## 5. 先跑 100 行烟测

```bash
LIMIT=100 bash scripts/extract_entities_llm_100k.sh
LIMIT=100 bash scripts/build_sag_lite_llm_100k.sh
bash scripts/query_sag_lite_llm_stall_100k.sh
bash scripts/evaluate_sag_lite_llm_stall_100k.sh
```

确认没有显存、模型加载、JSON 解析问题后，再跑完整 100k。

## 6. 运行 LLM entity 抽取

```bash
bash scripts/extract_entities_llm_100k.sh
```

默认输出：

```text
outputs/sag_lite.entity_links.llm.100k.jsonl
outputs/sag_lite.entity_links.llm.rejects.100k.jsonl
```

## 7. 构建融合实体的 SAG 数据库

```bash
bash scripts/build_sag_lite_llm_100k.sh
```

默认输出：

```text
outputs/sag_lite.llm.100k.duckdb
```

## 8. 查询和评估

```bash
bash scripts/query_sag_lite_llm_stall_100k.sh
bash scripts/evaluate_sag_lite_llm_stall_100k.sh
```

默认输出：

```text
outputs/sag_lite.query.stall.llm.100k.json
outputs/sag_lite.eval_samples.stall.llm.100k.jsonl
outputs/sag_lite.entity_eval_samples.llm.100k.jsonl
```

## 9. 对比指标

和纯规则 SAG-lite 对比这些指标：

```text
seed_orders
expanded_orders
weak_precision@10
weak_precision@100
weak_recall@100
weak_recall@1000
metadata_street_missing recovery_rate
road / street / intersection / poi coverage
generic_entity_noise
人工标注后的 entity precision
人工标注后的 expansion precision
```

当前纯规则 SAG-lite 100k 基线：

```text
matched_orders: 3043
seed_orders: 1043
expanded_orders: 2000
weak_precision@10: 0.90
weak_precision@100: 0.99
weak_recall@100: 0.0261
weak_recall@1000: 0.2630
metadata_street_missing: 1385
metadata recovery_rate: 0.849097
```

## 10. 判断是否成功

成功标准：

```text
1. problem_object / problem_behavior 对隐含表达有补充。
2. road / poi 噪声低于规则版。
3. weak_recall@1000 相比规则版上升。
4. weak_precision@100 不明显下降。
5. stratified entity samples 不再全是 area。
```
