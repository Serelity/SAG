# 13-Qwen3-4B 工单级语义抽取

## 1. 目标与边界

本流程对每条已脱敏 12345 工单执行一次 Qwen3-4B 主调用，同时生成语义完整 event、开放式问题对象/行为、道路/路口/POI，以及 intent、emotion、satisfaction、urgency。只有确定性校验为 `repair_required` 的工单允许一次 repair 调用。

本地只开发、打包和运行不加载模型的单元测试。模型下载、真实脱敏数据、GPU 推理、性能和质量验收全部在服务器进行。不得把模型权重、工单 JSONL、响应、rejects、links、DuckDB 或日志提交 GitHub。

## 2. 服务器准备

```bash
nvidia-smi
df -h
pip install -r requirements.sag.txt
pip install -r requirements.entity.txt
```

PyTorch 必须按服务器 CUDA 环境单独安装，不在 requirements 中固定。模型目录默认：

```text
models/Qwen3-4B
```

生产输入必须是脱敏多视图 JSONL，默认：

```text
data/t_order_master.100k.multiview.jsonl
```

每行至少包含稳定 `doc_id`、`case_content_clean`，并可包含 `title_clean`、`case_goal_clean`、`address_detail_clean`、`metadata`。不要把原始 TSV 交给新语义抽取入口。

## 3. 上传交付包

Windows 本地打包：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_entity_extraction.ps1
```

把 `packages/sag-qwen3-4b-semantic-extraction.zip` 上传服务器，校验输出的 SHA-256 后解压。包中只包含源码、测试、配置、脚本、本文档和 requirements。

## 4. 10 条 smoke

```bash
LIMIT=10 bash scripts/extract_semantics_qwen3_4b.sh
python scripts/check_semantic_run.py \
  --semantic outputs/work_order_semantics.qwen3_4b.jsonl \
  --rejects outputs/work_order_semantics.rejects.jsonl \
  --run-report outputs/work_order_semantics.run.json \
  --quality-report outputs/work_order_semantics.quality.json
```

检查：无 OOM；processed 数量正确；每工单一次 primary；repair 不超过 repair-required 数量；`finish_reason=length` 可解释；报告中不含 prompt、正文、evidence 或原始响应。

## 5. 995 条样本

```bash
LIMIT=995 bash scripts/extract_semantics_qwen3_4b.sh
bash scripts/project_semantics_to_sag.sh
LIMIT=995 bash scripts/build_sag_semantic_100k.sh
```

人工抽样必须覆盖：开放领域主题、诉求动作/问题行为、road/POI、历史答复/当前立场、模板谢谢、对象态度/诉求人情绪和 discourse。验证器通过率不能当作准确率。

## 6. 100k 与恢复

995 样本质量通过后运行：

```bash
LIMIT=100000 bash scripts/extract_semantics_qwen3_4b.sh
bash scripts/project_semantics_to_sag.sh
LIMIT=100000 bash scripts/build_sag_semantic_100k.sh
```

中断恢复：

```bash
RESUME=1 LIMIT=100000 bash scripts/extract_semantics_qwen3_4b.sh
```

指定 doc_id 重跑：

```bash
DOC_ID_FILE=retry_doc_ids.txt RETRY_REJECTED=1 LIMIT=100000 \
  bash scripts/extract_semantics_qwen3_4b.sh
```

checkpoint 身份为 `(doc_id, content_hash, prompt_version, model_id)`。正文变化或 Prompt/模型变化不会被旧 checkpoint 错误跳过。

## 7. 产物

```text
outputs/work_order_semantics.qwen3_4b.jsonl
outputs/work_order_semantics.rejects.jsonl
outputs/work_order_semantics.run.json
outputs/work_order_semantics.quality.json
outputs/sag_events.qwen3_4b.jsonl
outputs/sag_event_entity_links.qwen3_4b.jsonl
outputs/sag_event_discourse.qwen3_4b.jsonl
outputs/sag_semantic.qwen3_4b.100k.duckdb
```

工单级 semantics JSONL 是权威审计中间产物；links/discourse/event rows 是可重复生成的 SAG 投影。不要用 event summary 替代完整脱敏 chunk。

## 8. 服务器性能记录

保留但不要提交：GPU 型号、dtype、batch size、input/output token 总量及 p50/p95、finish reason、repair/reject/OOM、elapsed seconds、orders/s、tokens/s、GPU 利用率和峰值显存。长短文本按配置分桶；若 Transformers 吞吐不足，可在 schema 稳定后增加 vLLM backend，不改变工单级 schema 和投影。

## 9. 故障恢复

- OOM：降低 `batch_size`，从 checkpoint `RESUME=1`；不要删除已完成 partial。
- 大量 `length`：提高服务器显存允许范围内的 `max_new_tokens`，先重跑 smoke。
- 大量 repair：按 warning 分布修 Prompt/validator，不要默认所有工单双调用。
- ID 对不上：确认 input、semantics、projection 使用同一稳定 `doc_id/content_hash`。
- 只需重投影：直接运行 `project_semantics_to_sag.sh` 和 `build_sag_semantic_100k.sh`，不加载模型。

回传时优先返回 checker 输出、run/quality 报告和哈希；原始响应或含正文产物只能通过受控安全通道传输。
