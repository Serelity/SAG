# Qwen3-4B Work-Order Semantic Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a server-run Qwen3-4B pipeline that reads desensitized work orders once, emits one auditable event/entities/discourse record per order, validates or selectively repairs it, and deterministically projects it into SAG query tables.

**Architecture:** Keep the existing `sag_entity_llm.py` path as a legacy baseline and add focused modules for input identity, prompt construction, semantic schema, validation, projection, and orchestration. The model performs one primary call per work order; Python writes a work-order-level semantic record and expands it into event-entity and discourse rows without another model call. Local execution uses only synthetic fixtures and fake generators; real data, model loading, GPU inference, throughput measurement, and 995/100k acceptance runs occur on the server.

**Tech Stack:** Python 3 standard library, `unittest`, Hugging Face Transformers/Qwen3 on the server, DuckDB, Bash, PowerShell packaging, JSON/JSONL.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-04-qwen4b-work-order-semantic-extraction-design.md`.
- Do not download or load Qwen locally; all model/GPU commands are server-only.
- Do not read, stage, commit, or push real work-order data or generated model outputs.
- The production semantic extractor defaults to desensitized multiview JSONL; raw TSV support is legacy/explicit only.
- One work order receives exactly one primary model request; at most one repair request is allowed when validation returns `repair_required`.
- Preserve the full desensitized chunk and stable `doc_id`; do not treat the event summary as the sole source of truth.
- `problem_object` and `problem_behavior` are open-domain model outputs, not fixed-vocabulary classification.
- Model-extracted locations in v2 are `road`, `intersection`, and `poi`; metadata provides deterministic area/street entities.
- Discourse is stored as event attributes and is not a default SAG expansion frontier.
- Do not accept model-reported confidence. Program metadata and validation status are authoritative.
- Do not expose chain-of-thought. Prompts request final JSON only.
- Existing unrelated changes in `.gitattributes`, `.gitignore`, `LICENSE`, and `tests/fixtures/t_order_master_sample.tsv` must remain unstaged and unmodified by this work.
- `.superpowers/` brainstorming artifacts must never be included in feature commits or the final push.
- Use TDD for every behavior change and make focused commits after each task.
- Final GitHub push happens only after local non-model verification and server-run instructions are complete; return the branch/commit URL.

---

## File Structure

### New source modules

- `src/ragflow_style_pipeline/work_order_input.py` — parse v2 and legacy desensitized JSONL, validate clean fields, compute stable content identity, and read work orders.
- `src/ragflow_style_pipeline/sag_semantic_schema.py` — semantic output constants, default values, tolerant JSON extraction, and normalized in-memory record shapes.
- `src/ragflow_style_pipeline/sag_semantic_prompt.py` — long-text window selection, prompt/few-shot text, payload construction, and repair prompts.
- `src/ragflow_style_pipeline/sag_semantic_validation.py` — deterministic evidence, type, discourse, history, and template checks with stable warning codes.
- `src/ragflow_style_pipeline/sag_semantic_projection.py` — project one accepted work-order semantic record into SAG entity-link and discourse rows.
- `src/ragflow_style_pipeline/sag_semantic_llm.py` — backend adapter, batching, primary/repair orchestration, checkpoints, JSONL outputs, and reports.

### Modified source modules

- `src/ragflow_style_pipeline/sag_db.py` — accept projected semantic links/discourse, store semantic event summaries and expanded link provenance, and create `sag_event_discourse`.
- `src/ragflow_style_pipeline/sag_query.py` — leave discourse out of frontier expansion and optionally filter by discourse attributes.

### New/modified configuration and operations

- `configs/sag_semantic_extraction_qwen3_4b.json` — v2 prompt/schema/runtime limits.
- `scripts/extract_semantics_qwen3_4b.sh` — server extraction entry point using desensitized JSONL.
- `scripts/project_semantics_to_sag.sh` — projection-only rerun without loading Qwen.
- `scripts/build_sag_semantic_100k.sh` — build the v2 DuckDB.
- `scripts/check_semantic_run.py` — output count/hash/reject/truncation checks without printing work-order text.
- `scripts/package_entity_extraction.ps1` — include the new modules, config, tests, and scripts in the server upload archive.
- `docs/13-Qwen4B工单级语义抽取.md` — server setup, smoke/sample/full runs, resume/retry, acceptance, and artifact return.

### Tests

- `tests/test_work_order_input.py`
- `tests/test_sag_semantic_schema.py`
- `tests/test_sag_semantic_prompt.py`
- `tests/test_sag_semantic_validation.py`
- `tests/test_sag_semantic_projection.py`
- `tests/test_sag_semantic_llm.py`
- Modify: `tests/test_sag_db.py`
- Modify: `tests/test_sag_query.py`

---

### Task 1: Desensitized Work-Order Input and Stable Identity

**Files:**
- Create: `src/ragflow_style_pipeline/work_order_input.py`
- Create: `tests/test_work_order_input.py`

**Interfaces:**
- Produces: `WorkOrderInputError(ValueError)`.
- Produces: `normalize_work_order(document: dict) -> dict`.
- Produces: `read_work_orders(path: str | Path, limit: int | None = None) -> list[dict]`.
- Produces: `content_hash(order: dict) -> str` returning `sha256:<64 hex chars>`.
- Normalized orders expose `doc_id`, four `*_clean` fields, `metadata`, `content_hash`, and `chunk_text`.

- [ ] **Step 1: Write failing tests for v2 multiview input, legacy tagged text, and invalid empty content**

```python
import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.work_order_input import (
    WorkOrderInputError,
    normalize_work_order,
    read_work_orders,
)


class TestWorkOrderInput(unittest.TestCase):
    def test_normalizes_v2_desensitized_document_and_keeps_doc_id(self):
        order = normalize_work_order({
            "schema_version": "2.0",
            "doc_id": "order_safe_1",
            "title_clean": "路灯故障",
            "case_content_clean": "市民反映和平路路灯连续三天不亮。",
            "case_goal_clean": "希望维修",
            "address_detail_clean": "和平路",
            "metadata": {"service_object_type": "求助", "area_code_area": "钟楼区"},
        })
        self.assertEqual(order["doc_id"], "order_safe_1")
        self.assertEqual(order["case_content_clean"], "市民反映和平路路灯连续三天不亮。")
        self.assertTrue(order["content_hash"].startswith("sha256:"))
        self.assertIn("诉求内容：", order["chunk_text"])

    def test_adapts_legacy_tagged_text_without_silently_losing_content(self):
        order = normalize_work_order({
            "doc_id": "order_legacy_1",
            "text": "诉求类型：咨询\n诉求内容：咨询体检报告如何查询。\n诉求目标：希望告知查询方式\n所属区域：常州市 / 钟楼区",
            "metadata": {"service_object_type": "咨询", "area_code_area": "钟楼区"},
        })
        self.assertEqual(order["case_content_clean"], "咨询体检报告如何查询。")
        self.assertEqual(order["case_goal_clean"], "希望告知查询方式")

    def test_rejects_document_with_no_desensitized_semantic_text(self):
        with self.assertRaisesRegex(WorkOrderInputError, "empty_semantic_text"):
            normalize_work_order({"doc_id": "order_empty", "metadata": {}})

    def test_reads_jsonl_with_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "orders.jsonl"
            path.write_text("\n".join([
                json.dumps({"doc_id": "a", "case_content_clean": "第一条脱敏工单"}, ensure_ascii=False),
                json.dumps({"doc_id": "b", "case_content_clean": "第二条脱敏工单"}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            rows = read_work_orders(path, limit=1)
        self.assertEqual([row["doc_id"] for row in rows], ["a"])
```

- [ ] **Step 2: Run the focused test and confirm the module is missing**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_work_order_input -v
```

Expected: `ModuleNotFoundError: ragflow_style_pipeline.work_order_input`.

- [ ] **Step 3: Implement strict normalization and legacy tagged-text parsing**

```python
# src/ragflow_style_pipeline/work_order_input.py
import hashlib
import json
import re
from pathlib import Path

from ragflow_style_pipeline.sag_entities import clean_value

CLEAN_FIELDS = ("title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean")
LEGACY_LABELS = {
    "title_clean": "标题",
    "case_content_clean": "诉求内容",
    "case_goal_clean": "诉求目标",
    "address_detail_clean": "地址详情",
}


class WorkOrderInputError(ValueError):
    pass


def _legacy_field(text, label):
    labels = "|".join(re.escape(value) for value in ["标题", "诉求类型", "诉求内容", "诉求目标", "业务分类", "所属区域", "来电时间", "来源渠道", "地址详情"])
    match = re.search(rf"(?:^|\n){re.escape(label)}：(.*?)(?=\n(?:{labels})：|$)", text, re.S)
    return clean_value(match.group(1)) if match else ""


def content_hash(order):
    payload = {field: clean_value(order.get(field)) for field in CLEAN_FIELDS}
    payload["metadata"] = order.get("metadata") if isinstance(order.get("metadata"), dict) else {}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def normalize_work_order(document):
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    legacy_text = clean_value(document.get("text"))
    order = {
        "doc_id": clean_value(document.get("doc_id")),
        "metadata": metadata,
    }
    for field in CLEAN_FIELDS:
        value = clean_value(document.get(field))
        if not value and legacy_text:
            value = _legacy_field(legacy_text, LEGACY_LABELS[field])
        order[field] = value
    if not order["doc_id"]:
        raise WorkOrderInputError("missing_doc_id")
    if not any(order[field] for field in CLEAN_FIELDS):
        raise WorkOrderInputError("empty_semantic_text")
    order["chunk_text"] = "\n".join(
        f"{label}：{order[field]}" for field, label in LEGACY_LABELS.items() if order[field]
    )
    order["content_hash"] = content_hash(order)
    return order


def read_work_orders(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(normalize_work_order(json.loads(line)))
            except (json.JSONDecodeError, WorkOrderInputError) as exc:
                raise WorkOrderInputError(f"line_{line_number}:{exc}") from exc
            if limit is not None and len(rows) >= limit:
                break
    return rows
```

- [ ] **Step 4: Run focused tests and the existing database reader tests**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_work_order_input tests.test_sag_db.TestSagDbMapping -v
```

Expected: all tests pass; no model import or load occurs.

- [ ] **Step 5: Commit the input boundary**

```bash
git add src/ragflow_style_pipeline/work_order_input.py tests/test_work_order_input.py
git commit -m "feat: add desensitized work-order input"
```

---

### Task 2: Semantic Output Schema and Tolerant Parsing

**Files:**
- Create: `src/ragflow_style_pipeline/sag_semantic_schema.py`
- Create: `tests/test_sag_semantic_schema.py`

**Interfaces:**
- Produces: `parse_semantic_json(text: str) -> tuple[dict, list[str]]`.
- Produces: `normalize_semantic_output(value: dict) -> dict`.
- Produces stable groups: `problem_objects`, `problem_behaviors`, `roads`, `intersections`, `pois`.
- Produces stable discourse enums and defaults.

- [ ] **Step 1: Write failing schema tests**

```python
import unittest

from ragflow_style_pipeline.sag_semantic_schema import parse_semantic_json


class TestSemanticSchema(unittest.TestCase):
    def test_parses_fenced_json_and_supplies_safe_defaults(self):
        parsed, warnings = parse_semantic_json('''```json
        {"event_summary":"咨询体检报告查询方式","entities":{},"discourse":{}}
        ```''')
        self.assertEqual(parsed["event_summary"], "咨询体检报告查询方式")
        self.assertEqual(parsed["entities"]["roads"], [])
        self.assertEqual(parsed["discourse"]["satisfaction"]["label"], "unknown")
        self.assertEqual(parsed["discourse"]["urgency"]["level"], "normal")
        self.assertEqual(warnings, [])

    def test_rejects_invalid_enums_without_trusting_model_confidence(self):
        parsed, warnings = parse_semantic_json('''{
          "event_summary":"路灯故障",
          "entities":{"problem_objects":[{"surface":"路灯","canonical":"路灯","field":"case_content_clean","evidence":"路灯"}]},
          "discourse":{"satisfaction":{"label":"very_happy"},"urgency":{"level":"now"}},
          "confidence":0.99
        }''')
        self.assertNotIn("confidence", parsed)
        self.assertEqual(parsed["discourse"]["satisfaction"]["label"], "unknown")
        self.assertEqual(parsed["discourse"]["urgency"]["level"], "normal")
        self.assertIn("invalid_satisfaction_label", warnings)
        self.assertIn("invalid_urgency_level", warnings)

    def test_reports_json_parse_failure(self):
        parsed, warnings = parse_semantic_json('{"event_summary":')
        self.assertEqual(parsed["event_summary"], "")
        self.assertEqual(warnings, ["json_parse_failed"])
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_schema -v
```

Expected: module import fails.

- [ ] **Step 3: Implement schema normalization with stable enums and field names**

Implement these constants and behavior:

```python
ENTITY_GROUPS = ("problem_objects", "problem_behaviors", "roads", "intersections", "pois")
SOURCE_FIELDS = {"title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean"}
INTENTS = {"投诉", "举报", "求助", "咨询", "建议", "表扬", "催办", "反馈", "其他"}
EMOTIONS = {"愤怒", "不满", "焦虑", "无奈", "悲伤", "感谢", "认可"}
SATISFACTION_LABELS = {"satisfied", "dissatisfied", "mixed", "unknown"}
URGENCY_LEVELS = {"normal", "high", "critical"}
GROUP_LIMITS = {
    "problem_objects": 3,
    "problem_behaviors": 4,
    "roads": 4,
    "intersections": 2,
    "pois": 4,
}
```

`parse_semantic_json` must strip a single Markdown fence, extract the first complete outer JSON object, return a normalized empty result on parse failure, remove unknown top-level fields, remove model confidence fields recursively, coerce malformed arrays to empty arrays, and preserve machine-readable warnings.

- [ ] **Step 4: Run schema tests**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_schema -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the schema**

```bash
git add src/ragflow_style_pipeline/sag_semantic_schema.py tests/test_sag_semantic_schema.py
git commit -m "feat: define work-order semantic schema"
```

---

### Task 3: Prompt Engineering and Long-Text Windowing

**Files:**
- Create: `src/ragflow_style_pipeline/sag_semantic_prompt.py`
- Create: `tests/test_sag_semantic_prompt.py`
- Create: `configs/sag_semantic_extraction_qwen3_4b.json`

**Interfaces:**
- Consumes normalized orders from `normalize_work_order`.
- Produces: `select_content_windows(text: str, max_chars: int) -> dict`.
- Produces: `build_semantic_prompt(order: dict, config: dict) -> str`.
- Produces: `build_repair_prompt(order: dict, original_output: str, errors: list[str], config: dict) -> str`.

- [ ] **Step 1: Write failing tests for prompt boundaries and long-text retention**

```python
import unittest

from ragflow_style_pipeline.sag_semantic_prompt import build_semantic_prompt, select_content_windows


class TestSemanticPrompt(unittest.TestCase):
    def test_windowing_keeps_current_claim_near_tail(self):
        text = "前期反映" + ("历史答复" * 500) + "现服务对象表示仍未解决，再次要求拆除违建。"
        windows = select_content_windows(text, max_chars=300)
        self.assertTrue(windows["truncated"])
        self.assertIn("现服务对象表示仍未解决", windows["current_window"])
        self.assertLessEqual(len(windows["combined"]), 300)

    def test_prompt_is_open_domain_and_contains_difficult_boundaries(self):
        order = {
            "doc_id": "order_1",
            "title_clean": "",
            "case_content_clean": "港龙新港城北门口有电动车摆摊，请优先处理，谢谢！",
            "case_goal_clean": "希望处理",
            "address_detail_clean": "",
            "metadata": {"service_object_type": "求助", "area_code_area": "武进区"},
        }
        prompt = build_semantic_prompt(order, {"max_input_chars": 1500})
        self.assertIn("开放式识别", prompt)
        self.assertIn("港龙新港城", prompt)
        self.assertIn("不能判定满意", prompt)
        self.assertIn("诉求动作", prompt)
        self.assertIn('"problem_objects"', prompt)
        self.assertNotIn('"confidence"', prompt)
```

- [ ] **Step 2: Run the prompt test and verify failure**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_prompt -v
```

Expected: module import fails.

- [ ] **Step 3: Implement deterministic window selection**

Use current-claim markers:

```python
CURRENT_MARKERS = (
    "其不认可", "现服务对象表示", "现再次反映", "仍未解决", "再次要求", "希望部门", "现要求"
)
HISTORY_MARKERS = (
    "前期反映", "原工单", "处理结果", "部门答复", "答复如下"
)
```

For text over budget, reserve approximately 30% for the head, 40% for a window around the last current marker, and 30% for the tail; deduplicate overlapping slices and hard-cap the combined string. Return `head`, `current_window`, `tail`, `combined`, `truncated`, `original_chars`, and `kept_chars`.

- [ ] **Step 4: Implement the v2 semantic prompt and six cross-domain few-shots**

The final JSON skeleton in the prompt must be exactly:

```json
{
  "event_summary": "",
  "entities": {
    "problem_objects": [],
    "problem_behaviors": [],
    "roads": [],
    "intersections": [],
    "pois": []
  },
  "discourse": {
    "intents": [],
    "emotions": [],
    "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
    "urgency": {"level": "normal", "evidence": ""}
  }
}
```

Each entity item is `{surface, canonical, field, evidence}`. Include the six approved examples: road-light failure, POI-vs-road, request-action-vs-problem, polite-thanks-vs-satisfaction, target-attitude-vs-speaker-emotion, and historical-response-vs-current-stance. Instruct the model to reason internally but output JSON only.

- [ ] **Step 5: Add the server runtime config**

```json
{
  "schema_version": "2.0",
  "prompt_version": "sag_semantic_v2",
  "model_id": "Qwen/Qwen3-4B",
  "model_path": "models/Qwen3-4B",
  "backend": "transformers",
  "enable_thinking": false,
  "max_input_chars": 2200,
  "max_new_tokens": 512,
  "temperature": 0.0,
  "batch_size": 8,
  "progress_every": 50,
  "checkpoint_every": 50,
  "max_repairs_per_order": 1,
  "length_bucket_boundaries": [600, 1400],
  "default_output": "outputs/work_order_semantics.qwen3_4b.jsonl",
  "default_rejects": "outputs/work_order_semantics.rejects.jsonl"
}
```

- [ ] **Step 6: Run prompt/schema tests and commit**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_prompt tests.test_sag_semantic_schema -v
```

Expected: all tests pass.

```bash
git add src/ragflow_style_pipeline/sag_semantic_prompt.py tests/test_sag_semantic_prompt.py configs/sag_semantic_extraction_qwen3_4b.json
git commit -m "feat: add qwen semantic extraction prompt"
```

---

### Task 4: Deterministic Validation and Repair Decisions

**Files:**
- Create: `src/ragflow_style_pipeline/sag_semantic_validation.py`
- Create: `tests/test_sag_semantic_validation.py`

**Interfaces:**
- Consumes normalized order plus normalized semantic output.
- Produces: `validate_semantic_output(order: dict, semantic: dict, parse_warnings: list[str] | None = None) -> dict`.
- Result is `{"status": str, "warnings": list[str], "repair_fields": list[str]}`.
- Produces only statuses `accepted`, `accepted_with_warnings`, `repair_required`, `rejected`.

- [ ] **Step 1: Write failing tests for the observed failure classes**

```python
import unittest

from ragflow_style_pipeline.sag_semantic_validation import validate_semantic_output


def semantic_with(group, item):
    entities = {name: [] for name in ("problem_objects", "problem_behaviors", "roads", "intersections", "pois")}
    entities[group] = [item]
    return {
        "event_summary": "测试事件",
        "entities": entities,
        "discourse": {
            "intents": [], "emotions": [],
            "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
            "urgency": {"level": "normal", "evidence": ""},
        },
    }


class TestSemanticValidation(unittest.TestCase):
    def setUp(self):
        self.order = {
            "case_content_clean": "港龙新港城北门口有摊贩占道，希望清理，请优先处理，谢谢！",
            "case_goal_clean": "希望清理",
            "title_clean": "",
            "address_detail_clean": "",
        }

    def test_requires_repair_when_evidence_is_missing(self):
        result = validate_semantic_output(self.order, semantic_with("roads", {
            "surface": "和平路", "canonical": "和平路", "field": "case_content_clean", "evidence": "和平路"
        }))
        self.assertEqual(result["status"], "repair_required")
        self.assertIn("missing_evidence:entities.roads.0", result["warnings"])

    def test_flags_poi_gate_as_not_a_named_road(self):
        result = validate_semantic_output(self.order, semantic_with("roads", {
            "surface": "港龙新港城北门口", "canonical": "港龙新港城北门口", "field": "case_content_clean", "evidence": "港龙新港城北门口"
        }))
        self.assertEqual(result["status"], "repair_required")
        self.assertIn("road_poi_conflict:entities.roads.0", result["warnings"])

    def test_flags_request_action_as_problem_behavior(self):
        result = validate_semantic_output(self.order, semantic_with("problem_behaviors", {
            "surface": "清理", "canonical": "清理", "field": "case_goal_clean", "evidence": "清理"
        }))
        self.assertIn("request_action_as_behavior:entities.problem_behaviors.0", result["warnings"])

    def test_rejects_template_thanks_as_satisfaction_evidence(self):
        semantic = semantic_with("pois", {"surface":"港龙新港城","canonical":"港龙新港城","field":"case_content_clean","evidence":"港龙新港城"})
        semantic["discourse"]["satisfaction"] = {"label":"satisfied","target":"部门","evidence":"谢谢"}
        result = validate_semantic_output(self.order, semantic)
        self.assertEqual(result["status"], "repair_required")
        self.assertIn("template_politeness_as_satisfaction", result["warnings"])
```

- [ ] **Step 2: Run the validation test and verify failure**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_validation -v
```

Expected: module import fails.

- [ ] **Step 3: Implement stable warning codes and severity mapping**

Use exact machine-code prefixes:

```python
REPAIR_PREFIXES = (
    "json_parse_failed",
    "missing_evidence:",
    "invalid_source_field:",
    "road_poi_conflict:",
    "intersection_shape_conflict:",
    "request_action_as_behavior:",
    "canonical_evidence_conflict:",
    "satisfaction_missing_target_or_evidence",
    "template_politeness_as_satisfaction",
    "possible_history_contamination",
    "group_limit_exceeded:",
)
REJECT_PREFIXES = ("missing_doc_id", "empty_semantic_text", "repair_failed")
```

Validation must check exact evidence containment, group limits, deduplication, road naming shape, intersection shape, generic entities, case-goal request verbs, satisfaction target/evidence, urgency evidence, and current/history marker contradictions. Warnings that are informative but do not invalidate evidence produce `accepted_with_warnings`; repair prefixes produce `repair_required`.

- [ ] **Step 4: Run all semantic unit tests**

Run:

```bash
PYTHONPATH=src python -m unittest \
  tests.test_work_order_input \
  tests.test_sag_semantic_schema \
  tests.test_sag_semantic_prompt \
  tests.test_sag_semantic_validation -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the quality gate**

```bash
git add src/ragflow_style_pipeline/sag_semantic_validation.py tests/test_sag_semantic_validation.py
git commit -m "feat: validate semantic extraction outputs"
```

---

### Task 5: Deterministic SAG Projection and Database Schema

**Files:**
- Create: `src/ragflow_style_pipeline/sag_semantic_projection.py`
- Create: `tests/test_sag_semantic_projection.py`
- Modify: `src/ragflow_style_pipeline/sag_db.py`
- Modify: `tests/test_sag_db.py`

**Interfaces:**
- Produces: `project_semantic_record(record: dict, order: dict) -> tuple[dict, list[dict], dict]` returning event override, links, discourse.
- Produces: `project_semantics_file(input_path, orders_path, links_path, discourse_path) -> dict`.
- Modifies: `build_sag_db_from_orders(source_orders, db_path, extra_entity_links_by_doc=None, semantic_events_by_doc=None, discourse_by_doc=None) -> dict`.
- Adds DuckDB table `sag_event_discourse`.

- [ ] **Step 1: Write failing projection tests**

```python
import unittest

from ragflow_style_pipeline.sag_semantic_projection import project_semantic_record


class TestSemanticProjection(unittest.TestCase):
    def test_projects_entities_and_discourse_without_model_confidence(self):
        order = {
            "doc_id": "order_1",
            "metadata": {"service_object_type": "求助", "area_code_area": "钟楼区", "area_code_street": "永红街道"},
        }
        record = {
            "doc_id": "order_1",
            "event": {"summary": "市民反映和平路路灯连续三天不亮，希望维修"},
            "entities": {
                "problem_objects": [{"surface":"路灯","canonical":"路灯","source_field":"case_content_clean","evidence":"路灯"}],
                "problem_behaviors": [{"surface":"连续三天不亮","canonical":"照明故障","source_field":"case_content_clean","evidence":"连续三天不亮"}],
                "roads": [{"surface":"和平路","canonical":"和平路","source_field":"case_content_clean","evidence":"和平路"}],
                "intersections": [], "pois": [],
            },
            "discourse": {
                "intents": [{"label":"求助","evidence":"希望维修"}],
                "emotions": [],
                "satisfaction": {"label":"unknown","target":"","evidence":""},
                "urgency": {"level":"normal","evidence":""},
            },
            "validation": {"status":"accepted","warnings":[]},
            "model_run": {"prompt_version":"sag_semantic_v2"},
        }
        event, links, discourse = project_semantic_record(record, order)
        self.assertEqual(event["event_text"], record["event"]["summary"])
        self.assertEqual({(row["entity_type"], row["normalized_value"]) for row in links}, {
            ("problem_object", "路灯"), ("problem_behavior", "照明故障"), ("road", "和平路"),
        })
        self.assertTrue(all("confidence" not in row for row in links))
        self.assertEqual(discourse["declared_intent"], "求助")
        self.assertEqual(discourse["satisfaction"], "unknown")
```

- [ ] **Step 2: Run the projection test and verify failure**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_projection -v
```

Expected: module import fails.

- [ ] **Step 3: Implement group-to-type projection and projection-only CLI**

Use this fixed mapping:

```python
GROUP_TO_TYPE = {
    "problem_objects": "problem_object",
    "problem_behaviors": "problem_behavior",
    "roads": "road",
    "intersections": "intersection",
    "pois": "poi",
}
```

Projected semantic links contain `doc_id`, `entity_type`, `entity_value` (surface), `normalized_value` (canonical), `source_field`, `source_channel="semantic_llm"`, `matched_text` (evidence), `validation_status`, and `prompt_version`. They do not invent model confidence. The projection-only CLI reads work-order semantics plus normalized orders and writes links/discourse atomically.

- [ ] **Step 4: Write failing DuckDB tests for semantic events and discourse**

Append tests that call `build_sag_db_from_orders` with one semantic event override and discourse row, then assert:

```sql
select event_text from sag_events where doc_id = 'order_1';
select declared_intent, satisfaction, urgency from sag_event_discourse where doc_id = 'order_1';
```

Expected values are the semantic summary, `求助`, `unknown`, and `normal`.

- [ ] **Step 5: Extend `sag_db.py` without breaking legacy links**

- Add semantic provenance columns `surface_form`, `normalized_value`, `validation_status`, and `prompt_version` to `SAG_LINK_COLUMNS` while keeping legacy `confidence` nullable/blank for semantic links.
- Add `SAG_DISCOURSE_COLUMNS` with `event_id`, `doc_id`, `declared_intent`, `inferred_intents_json`, `intent_conflict`, `emotions_json`, `satisfaction`, `satisfaction_target`, `satisfaction_evidence`, `urgency`, and `urgency_evidence`.
- Let semantic event summaries override generated event text only when validation status is accepted/accepted_with_warnings.
- Keep rule and legacy LLM loaders backward-compatible.
- Create indexes on discourse `doc_id`, `event_id`, `satisfaction`, and `urgency`.

- [ ] **Step 6: Run projection/database tests**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_projection tests.test_sag_db -v
```

Expected: all pass; tests skip DuckDB cases only if DuckDB is absent.

- [ ] **Step 7: Commit projection and database changes**

```bash
git add \
  src/ragflow_style_pipeline/sag_semantic_projection.py \
  src/ragflow_style_pipeline/sag_db.py \
  tests/test_sag_semantic_projection.py \
  tests/test_sag_db.py
git commit -m "feat: project semantic records into sag tables"
```

---

### Task 6: Query-Time Discourse Filters Without Expansion Supernodes

**Files:**
- Modify: `src/ragflow_style_pipeline/sag_query.py`
- Modify: `tests/test_sag_query.py`

**Interfaces:**
- Existing `query_sag_db(db_path, config)` remains compatible.
- Adds optional filters: `intent`, `satisfaction`, `urgency_in`.
- Frontier still reads only `expansion.frontier_entity_types`; discourse fields cannot enter `_frontier_entities`.

- [ ] **Step 1: Write a failing query test for discourse filtering**

Create two events sharing `road=和平路`, with discourse rows `dissatisfied/high` and `satisfied/normal`. Query with:

```python
config = {
    "seed_entities": [{"entity_type": "road", "values": ["和平路"]}],
    "filters": {"satisfaction": "dissatisfied", "urgency_in": ["high", "critical"]},
    "expansion": {"enabled": False},
}
```

Assert only the dissatisfied/high event is returned.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_query -v
```

Expected: the new filter test returns both events or fails because discourse filtering is absent.

- [ ] **Step 3: Add parameterized discourse filter SQL**

Join `sag_event_discourse d` only when a discourse filter is present. Add parameterized clauses for exact `satisfaction`, membership in `urgency_in`, and membership in JSON-decoded inferred intents. Apply the same filters to seed and expanded event selection. Reject any `frontier_entity_types` value outside entity-link types before query execution; never translate discourse labels into entity IDs.

- [ ] **Step 4: Run query and database tests**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_query tests.test_sag_db -v
```

Expected: all pass.

- [ ] **Step 5: Commit query filters**

```bash
git add src/ragflow_style_pipeline/sag_query.py tests/test_sag_query.py
git commit -m "feat: filter sag events by discourse"
```

---

### Task 7: One-Call Orchestration, Selective Repair, Checkpoints, and Reports

**Files:**
- Create: `src/ragflow_style_pipeline/sag_semantic_llm.py`
- Create: `tests/test_sag_semantic_llm.py`

**Interfaces:**
- Produces: `run_semantic_extraction(input_path, output_path, rejects_path, run_report_path, quality_report_path, model_path, config, limit=None, resume=False, retry_rejected=False, generator=None) -> dict`.
- Generator protocol: `generator(prompts: list[str], max_new_tokens: int, temperature: float) -> list[dict]`, each dict containing `text`, `input_tokens`, `output_tokens`, `finish_reason`, and `latency_ms`.
- Produces: `load_transformers_generator(model_path, enable_thinking=False)` but tests inject a fake generator.

- [ ] **Step 1: Write failing fake-generator tests proving one primary read and selective repair**

```python
import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_llm import run_semantic_extraction


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, prompts, max_new_tokens, temperature):
        self.calls.append(list(prompts))
        rows = []
        for prompt in prompts:
            if "只修复" in prompt:
                text = json.dumps({
                    "event_summary": "市民反映港龙新港城北门有流动摊贩占道经营",
                    "entities": {
                        "problem_objects": [{"surface":"摊贩","canonical":"流动摊贩","field":"case_content_clean","evidence":"摊贩"}],
                        "problem_behaviors": [{"surface":"占道","canonical":"占道经营","field":"case_content_clean","evidence":"占道"}],
                        "roads": [], "intersections": [],
                        "pois": [{"surface":"港龙新港城","canonical":"港龙新港城","field":"case_content_clean","evidence":"港龙新港城"}],
                    },
                    "discourse": {"intents":[],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}},
                }, ensure_ascii=False)
            else:
                text = json.dumps({
                    "event_summary": "市民反映港龙新港城北门有摊贩占道",
                    "entities": {"problem_objects":[],"problem_behaviors":[],"roads":[{"surface":"港龙新港城北门口","canonical":"港龙新港城北门口","field":"case_content_clean","evidence":"港龙新港城北门口"}],"intersections":[],"pois":[]},
                    "discourse": {"intents":[],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}},
                }, ensure_ascii=False)
            rows.append({"text": text, "input_tokens": 100, "output_tokens": 80, "finish_reason": "stop", "latency_ms": 10})
        return rows


class TestSemanticLlm(unittest.TestCase):
    def test_one_primary_call_per_order_and_one_repair_only_for_invalid_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "orders.jsonl"
            input_path.write_text(json.dumps({
                "doc_id":"order_1", "case_content_clean":"港龙新港城北门口有摊贩占道。"
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            generator = RecordingGenerator()
            summary = run_semantic_extraction(
                input_path, tmp / "semantic.jsonl", tmp / "rejects.jsonl",
                tmp / "run.json", tmp / "quality.json", "unused", {
                    "prompt_version":"sag_semantic_v2", "batch_size":8,
                    "max_new_tokens":512, "temperature":0.0,
                    "max_repairs_per_order":1, "checkpoint_every":1,
                }, generator=generator,
            )
            record = json.loads((tmp / "semantic.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(summary["primary_requests"], 1)
        self.assertEqual(summary["repair_requests"], 1)
        self.assertTrue(record["validation"]["repair_attempted"])
        self.assertEqual(record["entities"]["pois"][0]["canonical"], "港龙新港城")
```

Also add tests that a valid primary result creates one generator request total, resume skips an identical `(doc_id, content_hash, prompt_version, model_id)`, and a changed content hash is not skipped.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_llm -v
```

Expected: module import fails.

- [ ] **Step 3: Implement backend result metadata and length-bucket batching**

- Copy the proven Qwen chat-template logic from legacy `sag_entity_llm.py` into the new backend adapter.
- Return per-response `text`, input/output token counts, finish reason (derive `length` when generated token count reaches the cap), and batch latency apportioned or recorded explicitly.
- Sort pending orders into configured character/token buckets while preserving a stable output sequence number.
- Do not import Transformers until `load_transformers_generator` is called.

- [ ] **Step 4: Implement primary → validate → optional repair orchestration**

For each order:

1. Build one primary prompt.
2. Parse and validate.
3. If accepted, write directly.
4. If `repair_required` and repair budget is one, build a repair prompt containing only the required clean fields, original output, machine warning codes, and repair field paths.
5. Parse and validate repaired output.
6. Write accepted results to semantic JSONL; write failed results with both model responses and warning codes to server-only rejects JSONL.
7. Programmatically add `doc_id`, `content_hash`, `schema_version`, `validation`, and `model_run`; ignore model attempts to provide them.

- [ ] **Step 5: Implement atomic checkpoints and resume identity**

Use identity:

```python
(doc_id, content_hash, prompt_version, model_id)
```

Write output to `*.partial.jsonl`, flush at `checkpoint_every`, and maintain `*.checkpoint.json` with completed identities and counters. On successful completion, atomically replace the final output. `--resume` loads both checkpoint and partial output. Projection remains separately rerunnable. Never treat an unterminated partial file as final.

- [ ] **Step 6: Implement run and quality reports without source text**

`run.json` must contain model/backend/dtype/config hash, input/output token totals and p50/p95 approximations, finish reason counts, truncation/parse/OOM counts, primary/repair requests, elapsed seconds, orders/sec, start/end timestamps, and checkpoint state. `quality.json` must contain status/warning/reject distributions, group coverage, entity-count distributions, canonical-vs-surface counts, intent conflict count, and template-politeness warning count. Neither report may contain chunk text, evidence, prompts, or raw responses.

- [ ] **Step 7: Add CLI arguments**

Support:

```text
--input --output --rejects --run-report --quality-report
--config --model-path --limit --resume --retry-rejected --doc-id-file
```

Default behavior must reject `.tsv` input with an explanation to use desensitized JSONL; add `--allow-raw-tsv` only if legacy compatibility is explicitly needed and document that it is server-controlled.

- [ ] **Step 8: Run orchestration and all semantic tests**

Run:

```bash
PYTHONPATH=src python -m unittest \
  tests.test_work_order_input \
  tests.test_sag_semantic_schema \
  tests.test_sag_semantic_prompt \
  tests.test_sag_semantic_validation \
  tests.test_sag_semantic_projection \
  tests.test_sag_semantic_llm -v
```

Expected: all pass with fake generators; no Qwen/Transformers model is loaded.

- [ ] **Step 9: Commit orchestration**

```bash
git add src/ragflow_style_pipeline/sag_semantic_llm.py tests/test_sag_semantic_llm.py
git commit -m "feat: orchestrate auditable semantic extraction"
```

---

### Task 8: Server Scripts, Packaging, and Operations Documentation

**Files:**
- Create: `scripts/extract_semantics_qwen3_4b.sh`
- Create: `scripts/project_semantics_to_sag.sh`
- Create: `scripts/build_sag_semantic_100k.sh`
- Create: `scripts/check_semantic_run.py`
- Modify: `scripts/package_entity_extraction.ps1`
- Create: `docs/13-Qwen4B工单级语义抽取.md`
- Test: `tests/test_sag_semantic_llm.py`
- Test: `tests/test_sag_semantic_projection.py`

**Interfaces:**
- Extraction defaults to `data/t_order_master.100k.multiview.jsonl`.
- All paths are overridable by environment variables.
- Projection-only and database-build scripts never load Qwen.

- [ ] **Step 1: Add a failing CLI/config test for safe defaults**

Add a test that parses CLI args or reads the script and asserts the default input ends with `.multiview.jsonl`, not `t_order_master.tsv`, and that outputs are under `outputs/`.

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_llm tests.test_sag_semantic_projection -v
```

Expected: safe-default test fails because scripts are absent.

- [ ] **Step 3: Create server extraction script**

Use these environment defaults:

```bash
INPUT_JSONL="${INPUT_JSONL:-data/t_order_master.100k.multiview.jsonl}"
CONFIG="${CONFIG:-configs/sag_semantic_extraction_qwen3_4b.json}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B}"
OUTPUT="${OUTPUT:-outputs/work_order_semantics.qwen3_4b.jsonl}"
REJECTS="${REJECTS:-outputs/work_order_semantics.rejects.jsonl}"
RUN_REPORT="${RUN_REPORT:-outputs/work_order_semantics.run.json}"
QUALITY_REPORT="${QUALITY_REPORT:-outputs/work_order_semantics.quality.json}"
LIMIT="${LIMIT:-100000}"
```

Forward optional `RESUME=1`, `RETRY_REJECTED=1`, and `DOC_ID_FILE` flags. Before model load, verify input/config/model paths and free disk space; print only paths and counts, never work-order text.

- [ ] **Step 4: Create projection-only and database scripts**

`project_semantics_to_sag.sh` invokes `sag_semantic_projection` and writes links/discourse. `build_sag_semantic_100k.sh` invokes `sag_db` with desensitized orders, projected links/discourse, and semantic event records. Both commands must work without importing Transformers.

- [ ] **Step 5: Create a privacy-safe post-run checker**

`scripts/check_semantic_run.py` accepts semantic/reject/run/quality paths and checks:

- JSONL parseability and final newline;
- unique `doc_id + content_hash + prompt_version + model` identities;
- line counts versus run report;
- status and warning counts;
- `finish_reason=length`, rejects, and repair counts;
- SHA-256 for each artifact;
- absence of raw responses/prompts in run and quality reports.

It prints counts and hashes only, not source/evidence text.

- [ ] **Step 6: Update PowerShell packaging**

Keep the existing staging approach, include new files, remove caches, and add a manifest containing package commit, included relative paths, and SHA-256. Explicitly fail packaging if staging contains `data/`, `outputs/`, `models/`, `.env`, `.jsonl`, `.duckdb`, `.zip`, or `.superpowers/`.

- [ ] **Step 7: Write server operations documentation**

Document exact commands for:

```bash
# Install server dependencies; PyTorch remains CUDA-environment specific.
pip install -r requirements.sag.txt
pip install -r requirements.entity.txt

# 10-row smoke run.
LIMIT=10 bash scripts/extract_semantics_qwen3_4b.sh
python scripts/check_semantic_run.py --semantic outputs/work_order_semantics.qwen3_4b.jsonl --rejects outputs/work_order_semantics.rejects.jsonl --run-report outputs/work_order_semantics.run.json --quality-report outputs/work_order_semantics.quality.json

# 995-row sample, then projection and DB build.
LIMIT=995 bash scripts/extract_semantics_qwen3_4b.sh
bash scripts/project_semantics_to_sag.sh
LIMIT=995 bash scripts/build_sag_semantic_100k.sh

# Resume interrupted extraction.
RESUME=1 LIMIT=100000 bash scripts/extract_semantics_qwen3_4b.sh

# Full run only after sample quality approval.
LIMIT=100000 bash scripts/extract_semantics_qwen3_4b.sh
```

Include GPU checks (`nvidia-smi`), disk checks, expected files, failure recovery, package upload/extract commands, and an explicit warning that local development must not run the model.

- [ ] **Step 8: Run script/config tests and syntax checks**

Run locally without Qwen:

```bash
PYTHONPATH=src python -m unittest tests.test_sag_semantic_llm tests.test_sag_semantic_projection -v
bash -n scripts/extract_semantics_qwen3_4b.sh
bash -n scripts/project_semantics_to_sag.sh
bash -n scripts/build_sag_semantic_100k.sh
PYTHONPATH=src python scripts/check_semantic_run.py --help
```

Expected: tests pass, Bash syntax checks exit 0, checker help exits 0.

- [ ] **Step 9: Commit server delivery assets**

```bash
git add \
  scripts/extract_semantics_qwen3_4b.sh \
  scripts/project_semantics_to_sag.sh \
  scripts/build_sag_semantic_100k.sh \
  scripts/check_semantic_run.py \
  scripts/package_entity_extraction.ps1 \
  docs/13-Qwen4B工单级语义抽取.md \
  tests/test_sag_semantic_llm.py \
  tests/test_sag_semantic_projection.py
git commit -m "docs: add semantic extraction server workflow"
```

---

### Task 9: Full Local Non-Model Verification and Privacy Audit

**Files:**
- Modify only files needed to fix failures caused by Tasks 1–8.
- Do not modify or stage unrelated pre-existing working-tree files.

**Interfaces:**
- Produces a verified commit set ready for server upload and later GitHub push.

- [ ] **Step 1: Run the full unittest suite**

Run:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Expected: all tests pass; DuckDB-dependent tests may skip only when DuckDB is not installed.

- [ ] **Step 2: Run syntax compilation without importing model dependencies**

Run:

```bash
PYTHONPATH=src python -m compileall -q src tests scripts/check_semantic_run.py
```

Expected: exit 0.

- [ ] **Step 3: Run diff and generated-file checks**

Run:

```bash
git diff --check
git status --short
git diff --name-only origin/main...HEAD
git ls-files | grep -E '(^|/)(data|outputs|models|packages|\.superpowers)/|\.jsonl$|\.duckdb$|\.zip$' && exit 1 || true
```

Expected: no whitespace errors; feature commits contain no data/model/output artifacts. Pre-existing unrelated modifications remain unstaged.

- [ ] **Step 4: Scan tracked feature changes for credentials and raw path leakage**

Run:

```bash
git diff origin/main...HEAD -- . ':!docs/superpowers/plans/*' ':!docs/superpowers/specs/*' | grep -Ein 'api[_-]?key|secret|password|bearer [A-Za-z0-9]|G:\\12345_pro_promax|/mnt/g/12345_pro_promax' && exit 1 || true
```

Expected: no matches.

- [ ] **Step 5: Build and inspect the upload archive without model execution**

Run on Windows/PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_entity_extraction.ps1
```

Inspect the generated manifest and ZIP entry list. Expected: source/config/tests/docs/scripts/requirements only; no `data`, `outputs`, `models`, `.env`, JSONL, DuckDB, ZIP-within-ZIP, or `.superpowers` entries.

- [ ] **Step 6: Record verification evidence in the final response**

Capture exact test counts, skipped tests, archive path/hash, commit list, and the fact that no model was run locally. Do not claim server extraction quality or speed until the server commands have actually run.

---

### Task 10: Server Acceptance Handoff and GitHub Delivery

**Files:**
- No source changes expected.
- Optional: update `docs/13-Qwen4B工单级语义抽取.md` only if server execution reveals a documentation error; commit that correction separately.

**Interfaces:**
- Consumes the verified package/commits from Task 9.
- Produces server run instructions, acceptance evidence supplied by the user/server, and a pushed GitHub branch/commit URL.

- [ ] **Step 1: Provide the package and exact server smoke command**

Give the package SHA-256 and the 10-row command from Task 8. Ask for the privacy-safe `run.json`, `quality.json`, checker summary, and GPU metrics—not raw work-order responses—in return.

- [ ] **Step 2: Gate larger server runs on smoke evidence**

Verify from returned evidence:

- processed count equals requested count;
- JSON/schema completion rate and finish-reason counts are plausible;
- no OOM;
- reject/repair reasons are visible;
- one primary request per processed work order;
- repair requests do not exceed repair-required orders;
- reports contain no source text.

Do not recommend the 995 or 100k run until these checks pass.

- [ ] **Step 3: Gate the 100k run on 995-sample quality review**

Require a stratified manual audit covering open-domain topics, request actions, road/POI ambiguity, historical responses, template thanks, and discourse. Record entity precision, event fidelity, evidence traceability, discourse accuracy, repair/reject rates, orders/sec, tokens/sec, GPU utilization, and expansion precision. Do not treat validator pass rate as accuracy.

- [ ] **Step 4: Re-run final Git checks immediately before push**

Run:

```bash
git status --short
git log --oneline origin/main..HEAD
git diff --check origin/main...HEAD
git diff --name-status origin/main...HEAD
```

Expected: only intended feature commits are ahead; pre-existing unrelated modifications remain unstaged and are not included in commits.

- [ ] **Step 5: Push the reviewed branch to GitHub**

After final user authorization and successful verification:

```bash
git push origin main
```

Expected: push succeeds without force. Never use `--force`.

- [ ] **Step 6: Return GitHub URLs and delivery summary**

Obtain the final SHA and construct the URL from evidence:

```bash
FINAL_SHA="$(git rev-parse HEAD)"
printf 'Branch: https://github.com/Serelity/SAG/tree/main\n'
printf 'Design commit: https://github.com/Serelity/SAG/commit/847d541\n'
printf 'Final commit: https://github.com/Serelity/SAG/commit/%s\n' "$FINAL_SHA"
printf 'Design spec: https://github.com/Serelity/SAG/blob/main/docs/superpowers/specs/2026-08-04-qwen4b-work-order-semantic-extraction-design.md\n'
printf 'Implementation guide: https://github.com/Serelity/SAG/blob/main/docs/13-Qwen4B工单级语义抽取.md\n'
```

Return those exact command outputs; do not invent a commit SHA.
