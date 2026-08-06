# SAG 语义评测、Issue 标注与 Oracle 投影

本阶段冻结 `sag_semantic_v7` 的 Prompt 和语义规则，先建立数据画像、候选决策审计、人工 gold 和 SAG 超边评测。目标不是提高字段非空率，而是验证语义抽取能否提升 SAG 检索并减少错误超边扩展。

## 1. 产物和隐私边界

可安全汇报的聚合产物：

- `input.profile.safe.json`：输入字段、长度、metadata、代理桶和 payload 重复率；
- `eval.manifest.report.safe.json`：抽样分布；
- `semantic.eval.safe.json`：mention、关系和超边指标；
- checker、run report、quality report、diagnostics summary。

以下均为本地私有产物，含脱敏正文、evidence、候选或人工 gold，不得提交、打包、粘贴到聊天或通过公开渠道传输。服务器只在 Qwen 推理时临时生成 semantic/reject 和同事务 candidate/decision ledger，随后通过私有通道取回本地审计：

- `*.private.jsonl`；
- candidate ledger；
- decision ledger；
- annotation packet；
- issue gold；
- gold 的 flat/issue-aware 投影；
- semantic/reject 原文。

仓库默认忽略 `outputs/` 和所有 `*.jsonl`，但仍须在运行和打包流程中主动检查。

## 2. 真实 multiview 画像

除 Qwen 推理外，画像、抽样、标注、验证、重放、Oracle 投影和 SAG 评估都在本地完成。真实 TSV 应优先在本地脱敏导出；如果当前版脱敏 multiview 只存在服务器，则通过私有通道复制到仓库外的本地私有目录。不要把旧版本地代理画像表述为当前生产画像。

本地运行示例：

```bash
PRIVATE_ROOT=/path/outside/repository/sag-private/current-v1
INPUT_JSONL="$PRIVATE_ROOT/t_order_master.100k.multiview.jsonl"
mkdir -p "$PRIVATE_ROOT/audit"

PYTHONPATH=src python scripts/profile_semantic_input.py \
  --input "$INPUT_JSONL" \
  --output "$PRIVATE_ROOT/audit/input.profile.safe.json" \
  --max-input-chars 2200 \
  --head-size 32
```

报告只含聚合数字。词法 proxy 仅用于抽样，不是 gold，不能将 `road_form`、`semantic_gap` 或情绪触发词数量解释为真实 precision/recall。

优先检查：

- 四个 clean field 的非空率；
- P50/P95/P99 长度；
- 受理类型和 type1 长尾；
- 前 32 条与全量的分布偏差；
- 当前、去除时间 metadata、仅语义 metadata、仅 clean field 四种 payload 的完全重复率。

若原 TSV 存在可靠标题或地址列但 multiview 未导出，应先修脱敏输入契约，再优化模型空间抽取。

## 3. 确定性抽样

先建立 48 条试标集：24 条生产分布、24 条挑战样本。指南稳定后再冻结 200 条生产分布集和 64 条挑战集。

```bash
PYTHONPATH=src python scripts/build_semantic_eval_manifest.py \
  --input "$INPUT_JSONL" \
  --semantic "$PRIVATE_ROOT/model/v7.semantic.private.jsonl" \
  --manifest "$PRIVATE_ROOT/audit/eval.manifest.private.jsonl" \
  --report "$PRIVATE_ROOT/audit/eval.manifest.report.safe.json" \
  --production-size 24 \
  --challenge-size 24 \
  --seed sag-eval-pilot-v1
```

manifest 不含正文，但含 `doc_id/content_hash`，只用于本地受控联接。生产集按 `service_object_type + 正文长度桶` 比例抽取；挑战集轮询覆盖空间、历史/当前、否定/失败、直接 discourse、repair 和 semantic gap 等高召回代理桶。没有本地 semantic 输出时可省略 `--semantic`，但此时 challenge 不包含 repair/semantic-gap 分层。

相同 seed 和输入产生相同 manifest；修改 gold 前应冻结 seed、输入 hash 和 manifest hash。

## 4. 私有标注包

```bash
PYTHONPATH=src python scripts/build_private_annotation_packet.py \
  --input "$INPUT_JSONL" \
  --manifest "$PRIVATE_ROOT/audit/eval.manifest.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/eval.annotation.private.jsonl"
```

脚本只在终端打印数量和文件 hash，不打印正文。每条记录包含 clean fields、metadata 和空标注骨架。

### 4.1 Issue 最小 schema

每条工单可以有多个 issue：

```json
{
  "mode": "problem|question|request|suggestion|praise|historical_response|current_stance",
  "time_scope": "current|historical",
  "objects": [
    {"surface": "", "field": "case_content_clean", "evidence": ""}
  ],
  "predicates": [
    {"surface": "", "field": "case_content_clean", "evidence": ""}
  ],
  "actions": [
    {"surface": "", "field": "case_goal_clean", "evidence": ""}
  ],
  "locations": [
    {"type": "road|intersection|poi", "surface": "", "field": "case_content_clean", "evidence": ""}
  ]
}
```

第一轮不强制标全局 canonical；surface、evidence、issue 关系和 current/history 是事实权威。每个非空 evidence 必须显式给出 `field`，且 evidence 必须逐字存在于该 clean field；surface 必须逐字包含在 evidence 中。

### 4.2 标注边界

- `problem`：原文明确已发生的问题、异常、状态或阻碍；
- `question`：正常咨询、查询、材料/流程问询，不制造 problem behavior；其 predicate 投影为 `issue_predicate` 属性成员；
- `request`：希望维修、要求退款、请求调查等诉求动作，不等同已发生异常；其 action 投影为 `request_action` 属性成员；
- `historical_response`：部门或历史工单的答复、处理结论；
- `current_stance`：对历史答复不认可、仍未解决、再次反映；
- location 只挂到它修饰的 issue；
- emotions 只标诉求人直接表达；
- satisfaction 必须同时标 direct evidence 和评价 target；
- urgency 只按当前风险或明确催办证据标注；
- 不因 metadata 类型强行修改正文事实；metadata 的 `service_object_type` 单独保留为 registered request type。

先由两名标注者在本地复制出的 A/B 文件中独立标 30–50 条，仲裁后修订指南。若 issue 边界无法稳定一致，应简化 schema，而不是直接交给模型。

### 4.3 本地验证、双标一致性与仲裁

完成一份标注后先做严格验证。safe report 不含正文、evidence、doc_id 或标注者姓名：

```bash
PYTHONPATH=src python scripts/validate_semantic_gold.py \
  --input "$PRIVATE_ROOT/audit/eval.annotator-a.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/eval.annotator-a.validation.safe.json" \
  --require-complete \
  --annotator annotator-a
```

两份标注必须具有完全相同的 `(doc_id, content_hash)` 集合且标注者不同。比较 grounded mention、issue attachment、完整 issue frame 和 discourse；冲突包含脱敏正文及双方标注，只能保存在本地私有目录：

```bash
PYTHONPATH=src python scripts/compare_semantic_annotations.py \
  --left "$PRIVATE_ROOT/audit/eval.annotator-a.private.jsonl" \
  --right "$PRIVATE_ROOT/audit/eval.annotator-b.private.jsonl" \
  --left-annotator annotator-a \
  --right-annotator annotator-b \
  --output "$PRIVATE_ROOT/audit/eval.agreement.safe.json" \
  --conflicts "$PRIVATE_ROOT/audit/eval.conflicts.private.jsonl"
```

一致率用于发现指南歧义，不设脱离数据的固定通过阈值。先审阅 `issue_frame`、`issue_attachment` 和 history/current 冲突，再由第三人仲裁；不得通过多数投票自动合并 evidence 或 issue 边。冲突包中的 `left/right/clean_fields/source_provenance` 不可修改，只在 `adjudication` 中填写完整最终 `issues/declared_intents/direct_emotions/satisfaction/urgency`，将状态改为 `resolved` 并填写非空说明。仲裁者必须不同于 A/B 标注者。

所有冲突解决后，在本地生成唯一 final gold：

```bash
PYTHONPATH=src python scripts/merge_adjudicated_gold.py \
  --left "$PRIVATE_ROOT/audit/eval.annotator-a.private.jsonl" \
  --right "$PRIVATE_ROOT/audit/eval.annotator-b.private.jsonl" \
  --conflicts "$PRIVATE_ROOT/audit/eval.conflicts.private.jsonl" \
  --left-annotator annotator-a \
  --right-annotator annotator-b \
  --adjudicator referee \
  --output "$PRIVATE_ROOT/audit/eval.gold.private.jsonl" \
  --report "$PRIVATE_ROOT/audit/eval.adjudication.safe.json"
```

合并器会重新计算 A/B 一致性，核对输入文件 hash、完整冲突身份集和冲突原始内容；一致记录确定性进入 final gold，冲突记录只能采用显式仲裁结果。输出再执行 gold v2 的文件级 provenance、enum、field/evidence 和 issue grounding 验证。

## 5. 私有候选和决策账本

服务器只在 Qwen 抽取事务内显式启用候选采集：

```bash
CANDIDATE_LEDGER=outputs/audit/candidates.private.jsonl \
DECISION_LEDGER=outputs/audit/decisions.private.jsonl \
BACKEND=vllm LIMIT=32 BATCH_SIZE=32 \
  bash scripts/extract_semantics_qwen3_4b.sh
```

candidate ledger 保存解析后、sanitation 前的结构化候选；不保存原始模型响应。decision ledger 保存 validator 前后状态、动作码和结构计数。两者是 append-only 的实际推理尝试历史，使用 `run_attempt_id + ledger_sequence` 区分崩溃恢复后的重跑；用于计算 gate 删除 precision 和修改 validator 后重放。推理结束后将 semantic、reject、ledger 和聚合报告通过私有通道复制回本地，服务器不承担后续画像、评测或 DuckDB 工作。

修改 validator 后，无需重跑 Qwen，在本地重放候选：

```bash
PYTHONPATH=src python scripts/replay_semantic_candidates.py \
  --input "$INPUT_JSONL" \
  --candidates "$PRIVATE_ROOT/model/candidates.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/replayed.private.jsonl" \
  --report "$PRIVATE_ROOT/audit/replayed.report.safe.json"
```

replay 输出仍含 evidence，必须保持私有；report 只含状态、动作和结构计数。只有 final adjudicated gold 完成后，才能评估 gate 删除是否正确：

```bash
PYTHONPATH=src python scripts/audit_semantic_ledger.py \
  --input "$INPUT_JSONL" \
  --gold "$PRIVATE_ROOT/audit/eval.gold.private.jsonl" \
  --candidates "$PRIVATE_ROOT/model/candidates.private.jsonl" \
  --decisions "$PRIVATE_ROOT/model/decisions.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/ledger.gold-audit.safe.json" \
  --traces "$PRIVATE_ROOT/audit/ledger.gold-audit.traces.private.jsonl"
```

审计器在每条 gold 工单的最新 `run_attempt_id` 内，按当前 validator 选择最后一个 terminal attempt；崩溃后仅有 primary、仍需要 repair 的记录计为 incomplete，不会退回使用旧运行 repair。报告分别统计 primary、repair 和 selected 的：

- raw/final mention precision、recall、F1；
- correctly/incorrectly kept；
- correctly/wrongly removed；
- correct/incorrect additions；
- deletion/addition precision；
- object、behavior、road、intersection、POI 分角色指标；
- original/current validator action counts、状态迁移和版本 provenance。

这些指标只评 SAG frontier mention；issue attachment 仍由正式 semantic gold evaluator 评估。删除 precision 只在实际发生删除时定义，否则为 `null`。safe report 不含 doc_id、mention、evidence；逐工单 raw/final/gold 集合只存在于 private trace。

独立版本：

- `prompt_version`：模型语义任务变化；
- `validator_version`：确定性裁决变化；
- `projection_version`：SAG 投影变化；
- `decoder_contract_version`：普通解码或 guided JSON 契约变化。

纯 instrumentation 不升级 Prompt version。

## 6. SAG 语义 gold 评估

当前平铺 semantic 与 issue gold 比较：

```bash
PYTHONPATH=src python scripts/evaluate_semantic_gold.py \
  --gold "$PRIVATE_ROOT/audit/eval.gold.private.jsonl" \
  --predictions "$PRIVATE_ROOT/model/v7.semantic.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/semantic.eval.safe.json"
```

使用人工 gold 构造“实体全部正确但全部挂在一个 order event”的 oracle flat：

```bash
PYTHONPATH=src python scripts/evaluate_semantic_gold.py \
  --gold "$PRIVATE_ROOT/audit/eval.gold.private.jsonl" \
  --oracle-flat \
  --output "$PRIVATE_ROOT/audit/oracle-flat.eval.safe.json"
```

核心指标：

- `mention_micro`：mention 是否存在；
- `mention_by_role`：problem object/behavior/road/intersection/POI；
- `object_behavior_attachment`：对象和行为是否属于同一 issue；
- `location_attachment`：地点是否挂到正确 issue；
- `issue_co_membership.false_co_membership_rate`：错误共边；
- `hyperedge_exact`：完整 issue 成员集合是否匹配。

mention F1 为 1 并不表示 SAG 正确；平铺 order event 仍可能产生大量错误共边。

## 7. Oracle flat 与 issue-aware 投影

`project_gold_issues.py` 可导出便于人工审查的私有 JSONL。正式检索实验使用隔离的本地 Oracle DuckDB：

```bash
PYTHONPATH=src python scripts/build_oracle_sag.py \
  --gold "$PRIVATE_ROOT/audit/eval.gold.private.jsonl" \
  --mode flat \
  --db "$PRIVATE_ROOT/audit/oracle-flat.duckdb" \
  --report "$PRIVATE_ROOT/audit/oracle-flat.build.safe.json"

PYTHONPATH=src python scripts/build_oracle_sag.py \
  --gold "$PRIVATE_ROOT/audit/eval.gold.private.jsonl" \
  --mode issue-aware \
  --db "$PRIVATE_ROOT/audit/oracle-issue.duckdb" \
  --report "$PRIVATE_ROOT/audit/oracle-issue.build.safe.json"
```

Oracle flat 将同一工单的全部 gold issue 合并成一个 event；issue-aware 每个 issue 一个 event。两者使用相同 mention，只改变共边关系。question/request 的 predicate/action 可以作为 seed 属性成员，但 `issue_predicate`、`request_action` 和 discourse 不能成为 expansion frontier。

### 7.1 私有查询 relevance 集

查询文件是本地私有 JSONL，最小结构：

```json
{
  "schema": "sag_oracle_query_relevance_v1",
  "private": true,
  "query_id": "local-query-id",
  "seed_entities": [
    {"entity_type": "problem_object", "values": ["路灯"]},
    {"entity_type": "problem_behavior", "values": ["不亮"]}
  ],
  "seed_group_operator": "AND",
  "expansion": {
    "enabled": true,
    "frontier_entity_types": ["road", "intersection", "poi"],
    "max_expanded_docs": 2000
  },
  "relevance": [
    {"doc_id": "private-doc-id", "grade": 3}
  ]
}
```

`grade` 为 1–3。Oracle gold 是封闭小语料；每个查询必须审阅全集或可证明召回充分的 pooling 结果，列全所有正相关工单。未列出的工单会被当作不相关，因此不完整 relevance 会虚增错误扩展率。查询应来自真实 12345 检索任务，不应为了证明 issue-aware 而刻意构造无人会发出的词组。

### 7.2 本地对照评估

```bash
PYTHONPATH=src python scripts/evaluate_oracle_sag.py \
  --flat-db "$PRIVATE_ROOT/audit/oracle-flat.duckdb" \
  --issue-db "$PRIVATE_ROOT/audit/oracle-issue.duckdb" \
  --queries "$PRIVATE_ROOT/audit/oracle.queries.private.jsonl" \
  --output "$PRIVATE_ROOT/audit/oracle.retrieval.safe.json" \
  --traces "$PRIVATE_ROOT/audit/oracle.retrieval.traces.private.jsonl" \
  --cutoffs 5,10
```

seed 的 `AND` 必须在同一 event 内满足；一跳 expansion 只使用命中 event 自身的 frontier，最后按工单聚合排名。报告包含：

- macro Precision@K、Recall@K、nDCG@K、MRR；
- seed recall、false seed rate；
- one-hop precision、错误扩展率；
- issue-aware 移除/丢失的 relevant/irrelevant seed、expansion 和结果数；
- event 成员数、每工单 event 数、实体→event/doc 度，用于观察 hub inflation。

Precision@K 固定以 K 为分母；结果不足 K 时空位视为未命中。无 expansion 的查询不参与 one-hop precision/error-rate 宏平均。safe report 不含 query_id、doc_id、正文或 evidence；`traces.private.jsonl` 包含这些信息，只能留在本地私有目录。

只有 gold issue-aware 投影先证明能改善 SAG 检索，且不会丢失有价值相关结果，才值得让 Qwen 输出 issue schema。

## 8. 实验顺序

1. 冻结 v7 baseline；
2. 当前 multiview 画像；
3. 48 条双人试标；
4. oracle flat vs issue-aware SAG；
5. 简单 XGrammar JSON schema，不使用 vLLM 0.8.5 XGrammar 不支持的 `maxItems/minItems/maxLength/uniqueItems`；
6. 当前平铺 schema vs 轻量 issue frame；
7. 空间软候选并集；
8. Prompt 压缩；
9. gold 和检索指标通过后才运行 995 质量扩展。

每次 A/B 只改变一个因素，并使用全新目录。最终上线判断必须同时考虑抽取、超边和 SAG 检索三层指标。
