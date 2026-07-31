# Entity 抽取 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a server-runnable LLM-assisted entity extraction pipeline for SAG retrieval over 12345 work orders, keeping the entity set tightly scoped to SAG seed/filter/frontier entities.

**Architecture:** Keep the existing rule-based `sag_entities.py` as the high-precision baseline. Add a small local LLM extractor that outputs constrained JSON entity candidates, validate candidates against evidence spans and entity rules, merge them with rule links, and build a SAG DuckDB database that can be queried by the existing SAG retrieval code. Server scripts download the model to a local `models/` directory, run batch extraction, rebuild the SAG DB with merged entities, evaluate retrieval/entity quality, and package code without model weights.

**Tech Stack:** Python 3.10+, DuckDB, `unittest`, `transformers`, `accelerate`, PyTorch from the server CUDA environment, Qwen local instruct model, JSON/JSONL, Bash server scripts, PowerShell packaging script.

## Global Constraints

- Scope name is `entity抽取`.
- This phase serves SAG retrieval only; do not implement full governance workflow fields such as satisfaction, handling result, risk level, or督办 state.
- LLM output entity types are limited to `problem_object`, `problem_behavior`, `area`, `street`, `road`, `intersection`, and `poi`.
- Metadata-derived entity types remain supported: `time_month`, `area`, `street`, `case_type`, `department`, and `lnglat`.
- `area` is filter/score metadata and must not be added to the default expansion frontier.
- Default model is `Qwen/Qwen3-8B`, stored on the server under `models/Qwen3-8B`.
- Model weights are downloaded directly from ModelScope on the server and are not included in code packages.
- Qwen3 thinking mode must be disabled for this extraction task when the tokenizer supports `enable_thinking=False`.
- Extraction must support offline inference from a local model path after the first download.
- Raw sensitive work-order IDs remain hashed through the existing `stable_hash` path.
- LLM candidates without an evidence span present in the original text must be rejected.
- Generic spatial fragments such as `路`, `街`, `道路`, `小区`, `市场`, `关于道路`, `关于小区`, and `常州12345热线` must not enter SAG links.

---

## File Structure

Create or modify these files only.

```text
configs/sag_entity_extraction_qwen3_8b.json
  Runtime config for the LLM entity extraction job, including model path, output schema, allowed entity types, alias maps, and validation thresholds.

requirements.entity.txt
  Python dependencies for LLM entity extraction. Do not pin PyTorch here because the server CUDA environment should control the PyTorch build.

src/ragflow_style_pipeline/sag_entity_schema.py
  Shared entity schema, alias normalization, generic-noise filters, and validation helpers for LLM entity candidates.

src/ragflow_style_pipeline/sag_entity_llm.py
  Prompt construction, local model loading, JSON parsing, candidate validation, batch extraction CLI, and JSONL output writer.

src/ragflow_style_pipeline/sag_entities.py
  Reuse existing `SagEntityLink`; add conversion helpers only when needed. Do not move rule extraction logic into the LLM module.

src/ragflow_style_pipeline/sag_db.py
  Add optional `--entity-links-jsonl` input and merge LLM links with rule links during DuckDB build.

src/ragflow_style_pipeline/sag_eval.py
  Fix entity evaluation sampling to be stratified by entity type/source channel and add entity-noise counters.

tests/test_sag_entity_schema.py
  Unit tests for allowed entity types, normalization, aliasing, evidence validation, and generic-noise filtering.

tests/test_sag_entity_llm.py
  Unit tests for prompt shape, JSON parsing, candidate validation, and conversion to `SagEntityLink` without loading a real model.

tests/test_sag_db.py
  Extend DB tests to verify optional LLM entity links are merged and deduplicated.

tests/test_sag_eval.py
  Extend evaluation tests to verify stratified entity samples include multiple entity types.

scripts/download_entity_model.sh
  Server script that downloads the default model from ModelScope into `models/Qwen3-8B`.

scripts/extract_entities_llm_100k.sh
  Server script that runs LLM entity extraction on the first 100k source rows.

scripts/build_sag_lite_llm_100k.sh
  Server script that builds a merged rule+LLM SAG database.

scripts/query_sag_lite_llm_stall_100k.sh
  Server script that runs the existing stall query on the merged entity database.

scripts/evaluate_sag_lite_llm_stall_100k.sh
  Server script that writes retrieval samples, stratified entity samples, and quality counters.

scripts/package_entity_extraction.ps1
  Windows packaging script that zips code/config/scripts/docs for server upload without model weights or outputs.

docs/12-entity抽取.md
  Chinese note explaining the purpose, model choice, server run order, outputs, and evaluation interpretation.
```

Do not modify these files in this plan:

```text
src/ragflow_style_pipeline/topic_analysis.py
src/ragflow_style_pipeline/vector_search.py
src/ragflow_style_pipeline/local_search.py
src/ragflow_style_pipeline/bge_m3_embed.py
```

Those belong to the Hybrid/BGE baseline.

---

## Task 1: Entity Schema and Validation

**Files:**
- Create: `src/ragflow_style_pipeline/sag_entity_schema.py`
- Create: `tests/test_sag_entity_schema.py`

**Interfaces:**
- Produces: `ALLOWED_LLM_ENTITY_TYPES: set[str]`
- Produces: `GENERIC_ENTITY_VALUES: set[str]`
- Produces: `normalize_llm_entity_value(entity_type: str, value: str) -> str`
- Produces: `is_generic_entity_value(entity_type: str, value: str) -> bool`
- Produces: `evidence_exists(candidate: dict, order: dict) -> bool`
- Produces: `validate_llm_candidate(candidate: dict, order: dict, config: dict) -> tuple[bool, str]`
- Consumes: `clean_value` from `ragflow_style_pipeline.sag_entities`

- [ ] **Step 1: Write failing tests for allowed types and normalization**

Create `tests/test_sag_entity_schema.py` with:

```python
import unittest

from ragflow_style_pipeline.sag_entity_schema import (
    ALLOWED_LLM_ENTITY_TYPES,
    evidence_exists,
    is_generic_entity_value,
    normalize_llm_entity_value,
    validate_llm_candidate,
)


class TestSagEntitySchema(unittest.TestCase):
    def test_allowed_types_are_sag_retrieval_only(self):
        self.assertEqual(
            ALLOWED_LLM_ENTITY_TYPES,
            {"problem_object", "problem_behavior", "area", "street", "road", "intersection", "poi"},
        )

    def test_normalizes_aliases_for_problem_entities(self):
        self.assertEqual(normalize_llm_entity_value("problem_object", "卖菜摊子"), "流动摊贩")
        self.assertEqual(normalize_llm_entity_value("problem_behavior", "挡住人行道"), "占道经营")
        self.assertEqual(normalize_llm_entity_value("road", " 广成 路 "), "广成路")

    def test_filters_generic_spatial_noise(self):
        self.assertTrue(is_generic_entity_value("road", "路"))
        self.assertTrue(is_generic_entity_value("road", "关于道路"))
        self.assertTrue(is_generic_entity_value("poi", "关于小区"))
        self.assertTrue(is_generic_entity_value("poi", "本人要求市场"))
        self.assertFalse(is_generic_entity_value("road", "广成路"))
        self.assertFalse(is_generic_entity_value("poi", "清潭菜场"))

    def test_evidence_must_exist_in_source_text(self):
        order = {
            "case_content_clean": "市民反映广成路有流动摊贩占道经营。",
            "case_goal_clean": "希望城管处理",
            "title_clean": "",
            "address_detail_clean": "",
        }
        good = {"entity_type": "road", "entity_value": "广成路", "evidence_span": "广成路", "source_field": "case_content_clean"}
        bad = {"entity_type": "road", "entity_value": "江春路", "evidence_span": "江春路", "source_field": "case_content_clean"}

        self.assertTrue(evidence_exists(good, order))
        self.assertFalse(evidence_exists(bad, order))

    def test_validate_candidate_rejects_unsupported_and_generic_values(self):
        order = {"case_content_clean": "市民反映广成路有流动摊贩占道经营。"}
        config = {"min_confidence": 0.55}

        ok, reason = validate_llm_candidate(
            {
                "entity_type": "road",
                "entity_value": "广成路",
                "evidence_span": "广成路",
                "source_field": "case_content_clean",
                "confidence": 0.9,
            },
            order,
            config,
        )
        self.assertTrue(ok, reason)

        ok, reason = validate_llm_candidate(
            {
                "entity_type": "road",
                "entity_value": "道路",
                "evidence_span": "道路",
                "source_field": "case_content_clean",
                "confidence": 0.9,
            },
            order,
            config,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "generic_entity_value")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_entity_schema -v
```

Expected:

```text
ModuleNotFoundError: No module named 'ragflow_style_pipeline.sag_entity_schema'
```

- [ ] **Step 3: Implement schema helpers**

Create `src/ragflow_style_pipeline/sag_entity_schema.py`:

```python
"""SAG retrieval entity schema and validation for LLM candidates."""

from ragflow_style_pipeline.sag_entities import clean_value, normalize_entity_value


ALLOWED_LLM_ENTITY_TYPES = {
    "problem_object",
    "problem_behavior",
    "area",
    "street",
    "road",
    "intersection",
    "poi",
}

SOURCE_TEXT_FIELDS = ["title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean"]

PROBLEM_OBJECT_ALIASES = {
    "卖菜摊子": "流动摊贩",
    "卖菜摊": "流动摊贩",
    "路边摊": "流动摊贩",
    "流摊": "流动摊贩",
    "摊子": "流动摊贩",
}

PROBLEM_BEHAVIOR_ALIASES = {
    "挡住人行道": "占道经营",
    "堵住人行道": "占道经营",
    "占用道路": "占道经营",
    "占用人行道": "占道经营",
    "影响通行": "影响通行",
}

GENERIC_ENTITY_VALUES = {
    "路",
    "街",
    "道路",
    "关于道路",
    "政风热线",
    "常州12345热线",
    "小区",
    "关于小区",
    "导致小区",
    "该小区",
    "现小区",
    "市场",
    "要求市场",
    "本人要求市场",
    "其表示街道",
    "根据街道",
    "我街道",
    "关于街道",
}

GENERIC_PREFIXES = (
    "关于",
    "反映",
    "服务对象",
    "其表示",
    "本人要求",
    "希望",
    "要求",
    "导致",
    "现",
)


def normalize_llm_entity_value(entity_type, value):
    """Normalize an LLM entity candidate to the dictionary identity used by SAG."""
    entity_type = clean_value(entity_type)
    normalized = normalize_entity_value(value)
    if entity_type == "problem_object":
        return PROBLEM_OBJECT_ALIASES.get(normalized, normalized)
    if entity_type == "problem_behavior":
        return PROBLEM_BEHAVIOR_ALIASES.get(normalized, normalized)
    return normalized


def is_generic_entity_value(entity_type, value):
    """Return True when a value is too generic or fragmentary to be a SAG join key."""
    entity_type = clean_value(entity_type)
    normalized = normalize_llm_entity_value(entity_type, value)
    if not normalized:
        return True
    if normalized in GENERIC_ENTITY_VALUES:
        return True
    if entity_type in {"road", "street", "poi"} and len(normalized) <= 1:
        return True
    if entity_type == "road" and normalized in {"路", "街", "桥", "线", "巷", "弄"}:
        return True
    if entity_type == "poi":
        for prefix in GENERIC_PREFIXES:
            if normalized.startswith(prefix) and len(normalized) <= len(prefix) + 4:
                return True
    if entity_type == "street":
        for prefix in GENERIC_PREFIXES:
            if normalized.startswith(prefix):
                return True
    return False


def _source_text(order):
    return "\n".join(clean_value(order.get(field)) for field in SOURCE_TEXT_FIELDS if clean_value(order.get(field)))


def evidence_exists(candidate, order):
    """Return True when the candidate evidence text occurs in one source text field."""
    source_field = clean_value(candidate.get("source_field"))
    evidence_span = clean_value(candidate.get("evidence_span") or candidate.get("matched_text"))
    if not evidence_span:
        return False
    if source_field:
        return evidence_span in clean_value(order.get(source_field))
    return evidence_span in _source_text(order)


def validate_llm_candidate(candidate, order, config):
    """Validate one LLM candidate before converting it into a SAG entity link."""
    entity_type = clean_value(candidate.get("entity_type"))
    if entity_type not in ALLOWED_LLM_ENTITY_TYPES:
        return False, "unsupported_entity_type"
    entity_value = normalize_llm_entity_value(entity_type, candidate.get("entity_value"))
    if not entity_value:
        return False, "empty_entity_value"
    if is_generic_entity_value(entity_type, entity_value):
        return False, "generic_entity_value"
    confidence = float(candidate.get("confidence") or 0.0)
    min_confidence = float(config.get("min_confidence", 0.55))
    if confidence < min_confidence:
        return False, "low_confidence"
    if not evidence_exists(candidate, order):
        return False, "missing_evidence_span"
    return True, "ok"
```

- [ ] **Step 4: Run schema tests**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_entity_schema -v
```

Expected:

```text
Ran 5 tests
OK
```

- [ ] **Step 5: Commit schema task**

Run:

```bash
git add src/ragflow_style_pipeline/sag_entity_schema.py tests/test_sag_entity_schema.py
git commit -m "feat: define sag entity extraction schema"
```

---

## Task 2: LLM Entity Extraction Module

**Files:**
- Create: `src/ragflow_style_pipeline/sag_entity_llm.py`
- Create: `tests/test_sag_entity_llm.py`
- Create: `configs/sag_entity_extraction_qwen3_8b.json`

**Interfaces:**
- Consumes: `SagEntityLink` and `clean_value` from `sag_entities.py`
- Consumes: `validate_llm_candidate()` and `normalize_llm_entity_value()` from `sag_entity_schema.py`
- Produces: `build_extraction_prompt(order: dict, config: dict) -> str`
- Produces: `parse_llm_json(text: str) -> dict`
- Produces: `candidate_to_link(doc_id: str, candidate: dict) -> SagEntityLink`
- Produces: `extract_links_from_llm_response(order: dict, response_text: str, config: dict) -> tuple[list[SagEntityLink], list[dict]]`
- Produces CLI: `python -m ragflow_style_pipeline.sag_entity_llm --input ... --output ... --model-path ...`

- [ ] **Step 1: Write failing tests for prompt, parsing, and link conversion**

Create `tests/test_sag_entity_llm.py`:

```python
import unittest

from ragflow_style_pipeline.sag_entity_llm import (
    build_extraction_prompt,
    candidate_to_link,
    extract_links_from_llm_response,
    parse_llm_json,
)


class TestSagEntityLlm(unittest.TestCase):
    def test_prompt_limits_entity_types_to_sag_retrieval_schema(self):
        order = {
            "doc_id": "order_a",
            "title_clean": "流动摊贩占道",
            "case_content_clean": "市民反映广成路有流动摊贩占道经营。",
            "case_goal_clean": "希望处理",
            "address_detail_clean": "",
        }
        prompt = build_extraction_prompt(order, {"max_text_chars": 1000})

        self.assertIn("problem_object", prompt)
        self.assertIn("problem_behavior", prompt)
        self.assertIn("intersection", prompt)
        self.assertIn("不要输出满意度", prompt)
        self.assertIn("只输出 JSON", prompt)

    def test_parse_llm_json_extracts_first_json_object(self):
        parsed = parse_llm_json(
            '```json\n{"entities":[{"entity_type":"road","entity_value":"广成路","source_field":"case_content_clean","evidence_span":"广成路","confidence":0.9}]}\n```'
        )

        self.assertEqual(parsed["entities"][0]["entity_value"], "广成路")

    def test_extract_links_rejects_missing_evidence_and_generic_noise(self):
        order = {
            "doc_id": "order_a",
            "case_content_clean": "市民反映广成路有流动摊贩占道经营。",
            "case_goal_clean": "",
            "title_clean": "",
            "address_detail_clean": "",
        }
        response = """
        {
          "entities": [
            {"entity_type":"road","entity_value":"广成路","source_field":"case_content_clean","evidence_span":"广成路","confidence":0.91},
            {"entity_type":"road","entity_value":"道路","source_field":"case_content_clean","evidence_span":"道路","confidence":0.91},
            {"entity_type":"poi","entity_value":"不存在市场","source_field":"case_content_clean","evidence_span":"不存在市场","confidence":0.91}
          ]
        }
        """

        links, rejects = extract_links_from_llm_response(order, response, {"min_confidence": 0.55})

        self.assertEqual([(link.entity_type, link.entity_value) for link in links], [("road", "广成路")])
        self.assertEqual([reject["reason"] for reject in rejects], ["generic_entity_value", "missing_evidence_span"])

    def test_candidate_to_link_uses_llm_source_channel(self):
        link = candidate_to_link(
            "order_a",
            {
                "entity_type": "problem_behavior",
                "entity_value": "挡住人行道",
                "source_field": "case_content_clean",
                "evidence_span": "挡住人行道",
                "confidence": 0.8,
            },
        )

        self.assertEqual(link.entity_type, "problem_behavior")
        self.assertEqual(link.entity_value, "占道经营")
        self.assertEqual(link.source_channel, "llm")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_entity_llm -v
```

Expected:

```text
ModuleNotFoundError: No module named 'ragflow_style_pipeline.sag_entity_llm'
```

- [ ] **Step 3: Create extraction config**

Create `configs/sag_entity_extraction_qwen3_8b.json`:

```json
{
  "model_id": "Qwen/Qwen3-8B",
  "model_path": "models/Qwen3-8B",
  "download_backend": "modelscope",
  "enable_thinking": false,
  "max_text_chars": 2200,
  "max_new_tokens": 512,
  "temperature": 0.0,
  "min_confidence": 0.55,
  "batch_size": 1,
  "allowed_entity_types": [
    "problem_object",
    "problem_behavior",
    "area",
    "street",
    "road",
    "intersection",
    "poi"
  ],
  "default_output": "outputs/sag_lite.entity_links.llm.100k.jsonl",
  "default_rejects": "outputs/sag_lite.entity_links.llm.rejects.100k.jsonl"
}
```

- [ ] **Step 4: Implement prompt, parser, and response conversion**

Create `src/ragflow_style_pipeline/sag_entity_llm.py` with these core functions:

```python
"""LLM-assisted SAG entity extraction for 12345 work orders."""

import argparse
import json
import re
from pathlib import Path

from ragflow_style_pipeline.sag_db import read_source_rows
from ragflow_style_pipeline.sag_entities import SagEntityLink, clean_value, deduplicate_entity_links
from ragflow_style_pipeline.sag_entity_schema import (
    normalize_llm_entity_value,
    validate_llm_candidate,
)


ENTITY_INSTRUCTIONS = """你是 12345 工单的 SAG 检索实体抽取器。
任务只服务后续 SQL join 检索，不做完整治理分析。
只允许输出这些 entity_type：
- problem_object：问题对象，例如流动摊贩、商贩、车辆、垃圾、井盖
- problem_behavior：问题行为，例如占道经营、影响通行、扰民、乱停放、损坏
- area：区县/开发区
- street：街道/镇
- road：道路/街/大道/巷/桥/线的具体名称
- intersection：两个道路形成的路口/交叉口/交界处
- poi：小区、菜场、学校、医院、商场、公园等具体地点
不要输出满意度、处理结果、风险等级、部门职责、完整摘要。
实体必须来自原文，evidence_span 必须能在对应 source_field 的原文中找到。
不要把“道路”“路”“街”“小区”“关于小区”“关于道路”“常州12345热线”这类泛词作为实体。
只输出 JSON，不输出解释文字。
JSON 格式：
{"entities":[{"entity_type":"road","entity_value":"广成路","source_field":"case_content_clean","evidence_span":"广成路","confidence":0.9}]}
"""


def _truncate(value, max_chars):
    value = clean_value(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


def build_extraction_prompt(order, config):
    """Build a deterministic extraction prompt for one normalized source order."""
    max_text_chars = int(config.get("max_text_chars", 2200))
    fields = {
        "title_clean": _truncate(order.get("title_clean"), 300),
        "case_content_clean": _truncate(order.get("case_content_clean"), max_text_chars),
        "case_goal_clean": _truncate(order.get("case_goal_clean"), 500),
        "address_detail_clean": _truncate(order.get("address_detail_clean"), 500),
    }
    payload = json.dumps(fields, ensure_ascii=False, indent=2)
    return f"{ENTITY_INSTRUCTIONS}\n输入工单字段：\n{payload}\n只输出 JSON："


def parse_llm_json(text):
    """Parse the first JSON object from model output."""
    text = clean_value(text)
    text = re.sub(r"^```(?:json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"entities": []}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"entities": []}
    if not isinstance(parsed, dict):
        return {"entities": []}
    entities = parsed.get("entities")
    if not isinstance(entities, list):
        parsed["entities"] = []
    return parsed


def candidate_to_link(doc_id, candidate):
    """Convert one validated LLM candidate into a SagEntityLink."""
    entity_type = clean_value(candidate.get("entity_type"))
    entity_value = normalize_llm_entity_value(entity_type, candidate.get("entity_value"))
    source_field = clean_value(candidate.get("source_field")) or "case_content_clean"
    evidence_span = clean_value(candidate.get("evidence_span") or candidate.get("matched_text") or entity_value)
    confidence = float(candidate.get("confidence") or 0.0)
    return SagEntityLink(
        doc_id=doc_id,
        entity_type=entity_type,
        entity_value=entity_value,
        normalized_value=entity_value,
        source_field=source_field,
        source_channel="llm",
        confidence=confidence,
        matched_text=evidence_span,
    )


def extract_links_from_llm_response(order, response_text, config):
    """Validate model response and return SAG entity links plus rejected candidates."""
    parsed = parse_llm_json(response_text)
    links = []
    rejects = []
    doc_id = clean_value(order.get("doc_id"))
    for candidate in parsed.get("entities", []):
        if not isinstance(candidate, dict):
            rejects.append({"candidate": candidate, "reason": "candidate_not_object"})
            continue
        ok, reason = validate_llm_candidate(candidate, order, config)
        if ok:
            links.append(candidate_to_link(doc_id, candidate))
        else:
            rejects.append({"candidate": candidate, "reason": reason, "doc_id": doc_id})
    return deduplicate_entity_links(links), rejects
```

- [ ] **Step 5: Add local model inference wrapper and CLI**

Append to `src/ragflow_style_pipeline/sag_entity_llm.py`:

```python
def load_local_generator(model_path, enable_thinking=False):
    """Load a local causal LM for deterministic JSON extraction."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    def generate(prompt, max_new_tokens=512, temperature=0.0):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(enable_thinking),
                )
            except TypeError:
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = prompt
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        do_sample = float(temperature) > 0.0
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs.input_ids.shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return generate


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def link_to_json(link):
    """Serialize a SagEntityLink to JSONL."""
    return {
        "doc_id": link.doc_id,
        "entity_type": link.entity_type,
        "entity_value": link.entity_value,
        "normalized_value": link.normalized_value,
        "source_field": link.source_field,
        "source_channel": link.source_channel,
        "confidence": link.confidence,
        "matched_text": link.matched_text,
    }


def run_extraction(input_path, output_path, rejects_path, model_path, config, limit=None):
    """Run LLM entity extraction over source rows and write entity-link JSONL."""
    generator = load_local_generator(model_path, enable_thinking=bool(config.get("enable_thinking", False)))
    rows = read_source_rows(input_path, limit=limit)
    written_links = 0
    written_rejects = 0
    output_path = Path(output_path)
    rejects_path = Path(rejects_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file, rejects_path.open("w", encoding="utf-8") as rejects_file:
        for index, order in enumerate(rows, start=1):
            prompt = build_extraction_prompt(order, config)
            response = generator(
                prompt,
                max_new_tokens=int(config.get("max_new_tokens", 512)),
                temperature=float(config.get("temperature", 0.0)),
            )
            links, rejects = extract_links_from_llm_response(order, response, config)
            for link in links:
                output_file.write(json.dumps(link_to_json(link), ensure_ascii=False) + "\n")
                written_links += 1
            for reject in rejects:
                rejects_file.write(json.dumps(reject, ensure_ascii=False) + "\n")
                written_rejects += 1
            if index % 100 == 0:
                print(json.dumps({"processed": index, "links": written_links, "rejects": written_rejects}, ensure_ascii=False))

    return {"orders_processed": len(rows), "links_written": written_links, "rejects_written": written_rejects}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run local LLM SAG entity extraction.")
    parser.add_argument("--input", required=True, help="Input TSV or JSONL source rows.")
    parser.add_argument("--output", required=True, help="Output SagEntityLink JSONL.")
    parser.add_argument("--rejects", required=True, help="Rejected candidate JSONL.")
    parser.add_argument("--config", required=True, help="Extraction config JSON.")
    parser.add_argument("--model-path", required=True, help="Local model directory.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum source rows to process.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = run_extraction(args.input, args.output, args.rejects, args.model_path, config, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run LLM module tests**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_entity_llm -v
```

Expected:

```text
Ran 4 tests
OK
```

- [ ] **Step 7: Commit LLM extraction task**

Run:

```bash
git add src/ragflow_style_pipeline/sag_entity_llm.py tests/test_sag_entity_llm.py configs/sag_entity_extraction_qwen3_8b.json
git commit -m "feat: add llm sag entity extraction"
```

---

## Task 3: Dependencies and Server Model Download

**Files:**
- Create: `requirements.entity.txt`
- Create: `scripts/download_entity_model.sh`

**Interfaces:**
- Consumes: `configs/sag_entity_extraction_qwen3_8b.json`
- Produces local server directory: `models/Qwen3-8B`

- [ ] **Step 1: Create entity extraction dependency file**

Create `requirements.entity.txt`:

```text
accelerate>=1.8.0
modelscope>=1.27.0
transformers>=4.53.0
tqdm>=4.66.0
```

Do not add `torch` to this file. Install PyTorch in the server CUDA environment with the server's existing PyTorch policy.

- [ ] **Step 2: Create server model download script**

Create `scripts/download_entity_model.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-8B}"

mkdir -p "$(dirname "${MODEL_DIR}")"

python - <<PY
from modelscope import snapshot_download

model_id = "${MODEL_ID}"
model_dir = "${MODEL_DIR}"
snapshot_download(model_id, local_dir=model_dir)
print({"model_id": model_id, "model_dir": model_dir, "backend": "modelscope"})
PY
```

- [ ] **Step 3: Run syntax checks**

Run:

```bash
bash -n scripts/download_entity_model.sh
```

Expected:

```text
No output
```

- [ ] **Step 4: Document server install command in commit message context**

Server command after package upload:

```bash
pip install -r requirements.entity.txt
bash scripts/download_entity_model.sh
```

Expected model directory:

```text
models/Qwen3-8B/config.json
models/Qwen3-8B/tokenizer.json
```

- [ ] **Step 5: Commit dependency task**

Run:

```bash
git add requirements.entity.txt scripts/download_entity_model.sh
git commit -m "chore: add entity extraction model setup"
```

---

## Task 4: Merge LLM Links into SAG Database

**Files:**
- Modify: `src/ragflow_style_pipeline/sag_db.py`
- Modify: `tests/test_sag_db.py`

**Interfaces:**
- Consumes: LLM entity link JSONL rows with fields matching `SagEntityLink`
- Produces: `load_entity_links_jsonl(path: str | Path) -> dict[str, list[SagEntityLink]]`
- Modifies: `build_sag_db_from_orders(source_orders, db_path, extra_entity_links_by_doc=None)`
- Modifies CLI: adds optional `--entity-links-jsonl`

- [ ] **Step 1: Add failing DB merge test**

Append to `tests/test_sag_db.py`:

```python
    def test_build_db_merges_llm_entity_links(self):
        _skip_without_duckdb(self)
        import duckdb
        import json

        order = source_order_row(
            {
                "id": "1",
                "order_id": "ORD001",
                "case_content": "市民反映广成路有卖菜摊子挡住人行道。",
                "case_goal": "希望处理",
                "call_time": "2024-05-01 10:00:00",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            links_path = tmp / "llm_links.jsonl"
            links_path.write_text(
                json.dumps(
                    {
                        "doc_id": order["doc_id"],
                        "entity_type": "problem_object",
                        "entity_value": "流动摊贩",
                        "normalized_value": "流动摊贩",
                        "source_field": "case_content_clean",
                        "source_channel": "llm",
                        "confidence": 0.8,
                        "matched_text": "卖菜摊子",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            db_path = tmp / "sag.duckdb"
            build_sag_db_from_orders([order], db_path, extra_entity_links_by_doc=load_entity_links_jsonl(links_path))

            conn = duckdb.connect(str(db_path))
            rows = conn.execute(
                """
                select entity_type, entity_value, source_channel
                from sag_event_entity_links
                where doc_id = ? and source_channel = 'llm'
                """,
                [order["doc_id"]],
            ).fetchall()

        self.assertEqual(rows, [("problem_object", "流动摊贩", "llm")])
```

At the top of `tests/test_sag_db.py`, extend imports:

```python
from ragflow_style_pipeline.sag_db import (
    build_sag_db_from_orders,
    event_row,
    load_entity_links_jsonl,
    read_source_rows,
    source_order_row,
    stable_hash,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_db -v
```

Expected:

```text
ImportError: cannot import name 'load_entity_links_jsonl'
```

- [ ] **Step 3: Implement JSONL loader and DB merge**

Modify `src/ragflow_style_pipeline/sag_db.py`.

Add imports:

```python
from ragflow_style_pipeline.sag_entities import SagEntityLink
```

Add loader:

```python
def load_entity_links_jsonl(path):
    """Load externally extracted SagEntityLink JSONL rows grouped by doc_id."""
    path = Path(path)
    links_by_doc = {}
    if not path.exists():
        return links_by_doc
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            link = SagEntityLink(
                doc_id=clean_value(row.get("doc_id")),
                entity_type=clean_value(row.get("entity_type")),
                entity_value=clean_value(row.get("entity_value")),
                normalized_value=clean_value(row.get("normalized_value")) or clean_value(row.get("entity_value")),
                source_field=clean_value(row.get("source_field")) or "case_content_clean",
                source_channel=clean_value(row.get("source_channel")) or "llm",
                confidence=float(row.get("confidence") or 0.0),
                matched_text=clean_value(row.get("matched_text")) or clean_value(row.get("entity_value")),
            )
            if link.doc_id and link.entity_type and link.normalized_value:
                links_by_doc.setdefault(link.doc_id, []).append(link)
    return links_by_doc
```

Change the DB builder signature:

```python
def build_sag_db_from_orders(source_orders, db_path, extra_entity_links_by_doc=None):
```

Inside the loop that currently uses `extract_entities_from_order(order)`, replace:

```python
for link in extract_entities_from_order(order):
```

with:

```python
rule_links = extract_entities_from_order(order)
extra_links = (extra_entity_links_by_doc or {}).get(order["doc_id"], [])
for link in rule_links + extra_links:
```

Add CLI argument in `parse_args`:

```python
parser.add_argument("--entity-links-jsonl", default="", help="Optional LLM SagEntityLink JSONL to merge into the SAG database.")
```

Modify CLI build call:

```python
extra_links = load_entity_links_jsonl(args.entity_links_jsonl) if args.entity_links_jsonl else None
summary = build_sag_db_from_orders(rows, args.db, extra_entity_links_by_doc=extra_links)
```

- [ ] **Step 4: Run DB tests**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_db -v
```

Expected:

```text
OK
```

If local Python does not have DuckDB, expected skipped DuckDB integration tests are acceptable. Pure mapping tests must pass.

- [ ] **Step 5: Commit DB merge task**

Run:

```bash
git add src/ragflow_style_pipeline/sag_db.py tests/test_sag_db.py
git commit -m "feat: merge llm entity links into sag db"
```

---

## Task 5: Server Extraction and Query Scripts

**Files:**
- Create: `scripts/extract_entities_llm_100k.sh`
- Create: `scripts/build_sag_lite_llm_100k.sh`
- Create: `scripts/query_sag_lite_llm_stall_100k.sh`
- Create: `scripts/evaluate_sag_lite_llm_stall_100k.sh`

**Interfaces:**
- Consumes local model directory: `models/Qwen3-8B`
- Consumes input TSV: `data/t_order_master.tsv`
- Produces: `outputs/sag_lite.entity_links.llm.100k.jsonl`
- Produces: `outputs/sag_lite.llm.100k.duckdb`
- Produces: `outputs/sag_lite.query.stall.llm.100k.json`
- Produces: `outputs/sag_lite.eval_samples.stall.llm.100k.jsonl`
- Produces: `outputs/sag_lite.entity_eval_samples.llm.100k.jsonl`

- [ ] **Step 1: Create LLM extraction script**

Create `scripts/extract_entities_llm_100k.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${INPUT_TSV:-data/t_order_master.tsv}"
CONFIG="${CONFIG:-configs/sag_entity_extraction_qwen3_8b.json}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-8B}"
OUTPUT_LINKS="${OUTPUT_LINKS:-outputs/sag_lite.entity_links.llm.100k.jsonl}"
OUTPUT_REJECTS="${OUTPUT_REJECTS:-outputs/sag_lite.entity_links.llm.rejects.100k.jsonl}"
LIMIT="${LIMIT:-100000}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_entity_llm \
  --input "${INPUT_TSV}" \
  --output "${OUTPUT_LINKS}" \
  --rejects "${OUTPUT_REJECTS}" \
  --config "${CONFIG}" \
  --model-path "${MODEL_PATH}" \
  --limit "${LIMIT}"
```

- [ ] **Step 2: Create merged DB build script**

Create `scripts/build_sag_lite_llm_100k.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${INPUT_TSV:-data/t_order_master.tsv}"
ENTITY_LINKS_JSONL="${ENTITY_LINKS_JSONL:-outputs/sag_lite.entity_links.llm.100k.jsonl}"
OUTPUT_DB="${OUTPUT_DB:-outputs/sag_lite.llm.100k.duckdb}"
LIMIT="${LIMIT:-100000}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_db \
  --input "${INPUT_TSV}" \
  --db "${OUTPUT_DB}" \
  --limit "${LIMIT}" \
  --entity-links-jsonl "${ENTITY_LINKS_JSONL}"
```

- [ ] **Step 3: Create merged SAG query script**

Create `scripts/query_sag_lite_llm_stall_100k.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_DB="${INPUT_DB:-outputs/sag_lite.llm.100k.duckdb}"
CONFIG="${CONFIG:-configs/sag_query_stall.json}"
OUTPUT_REPORT="${OUTPUT_REPORT:-outputs/sag_lite.query.stall.llm.100k.json}"

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_query \
  --db "${INPUT_DB}" \
  --config "${CONFIG}" \
  --output "${OUTPUT_REPORT}"
```

- [ ] **Step 4: Create merged evaluation script**

Create `scripts/evaluate_sag_lite_llm_stall_100k.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_DB="${INPUT_DB:-outputs/sag_lite.llm.100k.duckdb}"
QUERY_REPORT="${QUERY_REPORT:-outputs/sag_lite.query.stall.llm.100k.json}"
MANUAL_SAMPLES="${MANUAL_SAMPLES:-outputs/sag_lite.eval_samples.stall.llm.100k.jsonl}"
ENTITY_SAMPLES="${ENTITY_SAMPLES:-outputs/sag_lite.entity_eval_samples.llm.100k.jsonl}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.sag_eval \
  --db "${INPUT_DB}" \
  --query-report "${QUERY_REPORT}" \
  --manual-samples "${MANUAL_SAMPLES}" \
  --entity-samples "${ENTITY_SAMPLES}"
```

- [ ] **Step 5: Run script syntax checks**

Run:

```bash
bash -n scripts/extract_entities_llm_100k.sh
bash -n scripts/build_sag_lite_llm_100k.sh
bash -n scripts/query_sag_lite_llm_stall_100k.sh
bash -n scripts/evaluate_sag_lite_llm_stall_100k.sh
```

Expected:

```text
No output
```

- [ ] **Step 6: Commit server script task**

Run:

```bash
git add scripts/extract_entities_llm_100k.sh scripts/build_sag_lite_llm_100k.sh scripts/query_sag_lite_llm_stall_100k.sh scripts/evaluate_sag_lite_llm_stall_100k.sh
git commit -m "feat: add server scripts for llm entity extraction"
```

---

## Task 6: Stratified Entity Evaluation

**Files:**
- Modify: `src/ragflow_style_pipeline/sag_eval.py`
- Modify: `tests/test_sag_eval.py`

**Interfaces:**
- Modifies: `build_entity_eval_samples(db_path, limit=200)` to stratify by entity type and source channel.
- Produces: `count_generic_entity_noise(db_path) -> dict[str, int]`

- [ ] **Step 1: Add failing stratified sample test**

Append to `tests/test_sag_eval.py`:

```python
    def test_build_entity_eval_samples_is_stratified(self):
        _skip_without_duckdb(self)
        from ragflow_style_pipeline.sag_db import load_entity_links_jsonl
        from ragflow_style_pipeline.sag_eval import build_entity_eval_samples
        import json

        orders = [
            source_order_row(
                {
                    "id": "1",
                    "case_content": "市民反映广成路有流动摊贩占道经营。",
                    "call_time": "2024-05-01 10:00:00",
                    "area_code_area": "钟楼区",
                }
            ),
            source_order_row(
                {
                    "id": "2",
                    "case_content": "市民反映清潭菜场附近有商贩摆摊。",
                    "call_time": "2024-05-02 10:00:00",
                    "area_code_area": "钟楼区",
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            links_path = tmp / "links.jsonl"
            links_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "doc_id": orders[0]["doc_id"],
                                "entity_type": "road",
                                "entity_value": "广成路",
                                "normalized_value": "广成路",
                                "source_field": "case_content_clean",
                                "source_channel": "llm",
                                "confidence": 0.9,
                                "matched_text": "广成路",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "doc_id": orders[1]["doc_id"],
                                "entity_type": "poi",
                                "entity_value": "清潭菜场",
                                "normalized_value": "清潭菜场",
                                "source_field": "case_content_clean",
                                "source_channel": "llm",
                                "confidence": 0.9,
                                "matched_text": "清潭菜场",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            db_path = tmp / "sag.duckdb"
            build_sag_db_from_orders(orders, db_path, extra_entity_links_by_doc=load_entity_links_jsonl(links_path))
            samples = build_entity_eval_samples(db_path, limit=20)

        observed_types = {sample["entity_type"] for sample in samples}
        self.assertIn("area", observed_types)
        self.assertIn("road", observed_types)
        self.assertIn("poi", observed_types)
```

- [ ] **Step 2: Run test to verify current bug**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_eval -v
```

Expected:

```text
FAIL because entity samples are ordered by entity_type and not stratified.
```

- [ ] **Step 3: Implement stratified sampling**

Modify `build_entity_eval_samples` in `src/ragflow_style_pipeline/sag_eval.py`:

```python
def build_entity_eval_samples(db_path, limit=200):
    """Return stratified entity extraction samples for manual review."""
    per_group_limit = max(1, int(limit) // 16)
    with _connect(db_path) as conn:
        groups = conn.execute(
            """
            select entity_type, source_channel, count(*) as n
            from sag_event_entity_links
            group by entity_type, source_channel
            order by entity_type, source_channel
            """
        ).fetchall()

        selected = []
        for entity_type, source_channel, _count in groups:
            rows = conn.execute(
                """
                select l.doc_id, l.entity_type, l.entity_value, l.source_field, l.source_channel,
                       l.confidence, l.matched_text, s.case_content_clean, s.address_detail_clean
                from sag_event_entity_links l
                left join source_orders s on s.doc_id = l.doc_id
                where l.entity_type = ? and l.source_channel = ?
                order by l.doc_id, l.entity_value
                limit ?
                """,
                [entity_type, source_channel, per_group_limit],
            ).fetchall()
            selected.extend(rows)
            if len(selected) >= int(limit):
                break

    return [
        {
            "doc_id": row[0],
            "entity_type": row[1],
            "entity_value": row[2],
            "source_field": row[3],
            "source_channel": row[4],
            "confidence": float(row[5] or 0.0),
            "matched_text": row[6],
            "case_content": row[7],
            "address_detail": row[8],
            "label": "",
            "label_reason": "",
        }
        for row in selected[: int(limit)]
    ]
```

- [ ] **Step 4: Add generic noise counters**

Add to `src/ragflow_style_pipeline/sag_eval.py`:

```python
GENERIC_NOISE_VALUES = ["路", "街", "道路", "关于道路", "常州12345热线", "政风热线", "关于小区", "导致小区", "本人要求市场"]


def count_generic_entity_noise(db_path):
    """Count known generic entity noise values in the SAG links table."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            select entity_type, entity_value, count(*) as n
            from sag_event_entity_links
            group by entity_type, entity_value
            """
        ).fetchall()
    counts = {}
    for entity_type, entity_value, count in rows:
        if entity_value in GENERIC_NOISE_VALUES:
            counts[f"{entity_type}:{entity_value}"] = int(count)
    return counts
```

In `main`, include the noise counts in printed output:

```python
"generic_entity_noise": count_generic_entity_noise(args.db),
```

- [ ] **Step 5: Run eval tests**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_eval -v
```

Expected:

```text
OK
```

If local Python does not have DuckDB, expected skipped DuckDB integration tests are acceptable. Pure metric tests must pass.

- [ ] **Step 6: Commit evaluation task**

Run:

```bash
git add src/ragflow_style_pipeline/sag_eval.py tests/test_sag_eval.py
git commit -m "test: stratify sag entity evaluation samples"
```

---

## Task 7: Documentation and Server Run Order

**Files:**
- Create: `docs/12-entity抽取.md`

**Interfaces:**
- Documents exact server commands.
- Documents output files and comparison metrics.

- [ ] **Step 1: Create Chinese note**

Create `docs/12-entity抽取.md`:

```markdown
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

## 4. 下载模型到服务器本地

```bash
bash scripts/download_entity_model.sh
```

默认输出：

```text
models/Qwen3-8B
```

## 5. 运行 LLM entity 抽取

```bash
bash scripts/extract_entities_llm_100k.sh
```

默认输出：

```text
outputs/sag_lite.entity_links.llm.100k.jsonl
outputs/sag_lite.entity_links.llm.rejects.100k.jsonl
```

## 6. 构建融合实体的 SAG 数据库

```bash
bash scripts/build_sag_lite_llm_100k.sh
```

默认输出：

```text
outputs/sag_lite.llm.100k.duckdb
```

## 7. 查询和评估

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

## 8. 对比指标

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

## 9. 判断是否成功

成功标准：

```text
1. problem_object / problem_behavior 对隐含表达有补充。
2. road / poi 噪声低于规则版。
3. weak_recall@1000 相比规则版上升。
4. weak_precision@100 不明显下降。
5. stratified entity samples 不再全是 area。
```
```

- [ ] **Step 2: Commit documentation task**

Run:

```bash
git add docs/12-entity抽取.md
git commit -m "docs: explain entity extraction pipeline"
```

---

## Task 8: Verification and Packaging

**Files:**
- Create: `scripts/package_entity_extraction.ps1`

**Interfaces:**
- Produces Windows package: `G:\RAG\packages\ragflow-learning-plan-entity-extraction.zip`
- Package includes code, tests, configs, scripts, docs, and requirements.
- Package excludes model weights, DuckDB outputs, and raw data.

- [ ] **Step 1: Create packaging script**

Create `scripts/package_entity_extraction.ps1`:

```powershell
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$RepoParent = Resolve-Path "$ProjectRoot\.."
$PackageDir = Join-Path $RepoParent "packages"
$PackagePath = Join-Path $PackageDir "ragflow-learning-plan-entity-extraction.zip"

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null

$Items = @(
  Join-Path $ProjectRoot "src",
  Join-Path $ProjectRoot "tests",
  Join-Path $ProjectRoot "configs",
  Join-Path $ProjectRoot "scripts",
  Join-Path $ProjectRoot "docs",
  Join-Path $ProjectRoot "requirements.sag.txt",
  Join-Path $ProjectRoot "requirements.entity.txt"
)

if (Test-Path $PackagePath) {
  Remove-Item -LiteralPath $PackagePath -Force
}

Compress-Archive -Path $Items -DestinationPath $PackagePath -Force
Write-Host "Wrote $PackagePath"
```

- [ ] **Step 2: Run Python unit tests locally**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_sag_entity_schema tests.test_sag_entity_llm tests.test_sag_db tests.test_sag_eval -v
```

Expected:

```text
OK
```

If local Python does not have DuckDB, DuckDB integration tests may skip. Schema and LLM parser tests must pass.

- [ ] **Step 3: Run full unit test suite**

Run:

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 4: Run script syntax checks on Linux server**

Run:

```bash
bash -n scripts/download_entity_model.sh
bash -n scripts/extract_entities_llm_100k.sh
bash -n scripts/build_sag_lite_llm_100k.sh
bash -n scripts/query_sag_lite_llm_stall_100k.sh
bash -n scripts/evaluate_sag_lite_llm_stall_100k.sh
```

Expected:

```text
No output
```

- [ ] **Step 5: Run a 100-row smoke test on server**

Run:

```bash
LIMIT=100 bash scripts/extract_entities_llm_100k.sh
LIMIT=100 bash scripts/build_sag_lite_llm_100k.sh
INPUT_DB=outputs/sag_lite.llm.100k.duckdb bash scripts/query_sag_lite_llm_stall_100k.sh
INPUT_DB=outputs/sag_lite.llm.100k.duckdb bash scripts/evaluate_sag_lite_llm_stall_100k.sh
```

Expected files:

```text
outputs/sag_lite.entity_links.llm.100k.jsonl
outputs/sag_lite.entity_links.llm.rejects.100k.jsonl
outputs/sag_lite.llm.100k.duckdb
outputs/sag_lite.query.stall.llm.100k.json
outputs/sag_lite.entity_eval_samples.llm.100k.jsonl
```

- [ ] **Step 6: Run package script from Windows**

Run from `G:\RAG\ragflow-learning-plan`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_entity_extraction.ps1
```

Expected package:

```text
G:\RAG\packages\ragflow-learning-plan-entity-extraction.zip
```

- [ ] **Step 7: Commit packaging task**

Run:

```bash
git add scripts/package_entity_extraction.ps1
git commit -m "chore: package entity extraction code"
```

---

## Final Server Run Order

Run these commands on the Linux server after uploading the package and placing `t_order_master.tsv` under `data/`.

```bash
cd ragflow-learning-plan
pip install -r requirements.sag.txt
pip install -r requirements.entity.txt
bash scripts/download_entity_model.sh
bash scripts/extract_entities_llm_100k.sh
bash scripts/build_sag_lite_llm_100k.sh
bash scripts/query_sag_lite_llm_stall_100k.sh
bash scripts/evaluate_sag_lite_llm_stall_100k.sh
```

Expected outputs:

```text
models/Qwen3-8B/
outputs/sag_lite.entity_links.llm.100k.jsonl
outputs/sag_lite.entity_links.llm.rejects.100k.jsonl
outputs/sag_lite.llm.100k.duckdb
outputs/sag_lite.query.stall.llm.100k.json
outputs/sag_lite.eval_samples.stall.llm.100k.jsonl
outputs/sag_lite.entity_eval_samples.llm.100k.jsonl
```

Compare against current pure SAG-lite outputs:

```text
outputs/sag_lite.query.stall.100k.json
outputs/sag_lite.eval_samples.stall.100k.jsonl
outputs/sag_lite.entity_eval_samples.100k.jsonl
```

Baseline numbers to preserve for comparison:

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

---

## Self-Review

Spec coverage:

```text
The plan is named entity抽取.
It uses Qwen3-8B as the local LLM.
It downloads the model directly from ModelScope on the server.
It keeps the entity schema scoped to SAG retrieval.
It packages code without model weights.
It includes tests, server scripts, evaluation, docs, and packaging.
```

Placeholder scan:

```text
No implementation step depends on unspecified file names, unspecified commands, or unspecified output paths.
```

Type consistency:

```text
LLM extraction outputs SagEntityLink-compatible JSONL.
sag_db.py consumes SagEntityLink-compatible JSONL through load_entity_links_jsonl.
sag_eval.py samples from sag_event_entity_links after the merged DB build.
```
