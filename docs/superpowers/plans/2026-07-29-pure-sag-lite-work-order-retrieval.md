# Pure SAG-lite Work Order Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure SAG-lite baseline for 12345 work orders: source rows become events, events connect to extracted entities, SQL joins perform seed retrieval and 1-hop expansion, and reports include evaluation metrics.

**Architecture:** Add four focused Python modules: `sag_entities.py` extracts rule-based entities, `sag_db.py` builds DuckDB tables, `sag_query.py` performs pure SQL event-entity retrieval, and `sag_eval.py` computes weak-label/manual-sample evaluation outputs. Existing Hybrid Retrieval code remains untouched so it stays a clean baseline.

**Tech Stack:** Python standard library, `unittest`, optional `duckdb`, JSON/JSONL, Bash scripts for Linux server execution.

---

## File Structure

Create or modify these files only.

```text
src/ragflow_style_pipeline/sag_entities.py
  Rule-based entity extraction from source rows / multiview rows.

src/ragflow_style_pipeline/sag_db.py
  TSV/JSONL input normalization and DuckDB schema creation.

src/ragflow_style_pipeline/sag_query.py
  Pure SAG seed retrieval, 1-hop expansion, scoring, statistics, and report generation.

src/ragflow_style_pipeline/sag_eval.py
  Weak-label metrics, manual evaluation sample export, entity evaluation sample export.

tests/test_sag_entities.py
  Unit tests for entity normalization and extraction.

tests/test_sag_db.py
  Unit tests for source row mapping and DuckDB table building.

tests/test_sag_query.py
  Unit tests for seed retrieval, 1-hop expansion, scoring, and report aggregation.

tests/test_sag_eval.py
  Unit tests for weak labels, precision@K, recall@K, and sample generation.

configs/sag_query_stall.json
  Pure SAG query config for 流动摆摊 / 占道经营.

scripts/build_sag_lite_100k.sh
  Server script to build the SAG-lite DuckDB database.

scripts/query_sag_lite_stall_100k.sh
  Server script to run pure SAG query.

scripts/evaluate_sag_lite_stall_100k.sh
  Server script to compute evaluation outputs.

docs/11-纯SAG工单检索实验.md
  Chinese learning note explaining the design, commands, and interpretation.
```

Do not modify:

```text
src/ragflow_style_pipeline/topic_analysis.py
src/ragflow_style_pipeline/vector_search.py
src/ragflow_style_pipeline/local_search.py
```

Those files belong to the Hybrid Retrieval baseline.

---

## Task 1: Entity Extraction Module

**Files:**

- Create: `src/ragflow_style_pipeline/sag_entities.py`
- Create: `tests/test_sag_entities.py`

### Purpose

Extract SAG entities from one source order. This implements the paper-inspired `chunk -> entities` step for your work-order rows.

### API to implement

```python
@dataclass(frozen=True)
class SagEntityLink:
    doc_id: str
    entity_type: str
    entity_value: str
    normalized_value: str
    source_field: str
    source_channel: str
    confidence: float
    matched_text: str

def normalize_entity_value(value):
    """Normalize entity text for dictionary identity."""

def extract_entities_from_order(order):
    """Return deduplicated SagEntityLink objects for one order dictionary."""

def deduplicate_entity_links(links):
    """Deduplicate links by doc_id, entity_type, normalized_value, source_field."""
```

### Steps

- [ ] **Step 1: Write failing tests for metadata and text entity extraction**

Create `tests/test_sag_entities.py`:

```python
import unittest

from ragflow_style_pipeline.sag_entities import (
    SagEntityLink,
    deduplicate_entity_links,
    extract_entities_from_order,
    normalize_entity_value,
)


class TestSagEntities(unittest.TestCase):
    def test_normalize_entity_value_removes_spaces(self):
        self.assertEqual(normalize_entity_value(" 永红街道 "), "永红街道")
        self.assertEqual(normalize_entity_value("广成 路"), "广成路")

    def test_extracts_metadata_entities(self):
        order = {
            "doc_id": "order_a",
            "call_month": "2024-05",
            "area_code_area": "钟楼区",
            "area_code_street": "永红街道",
            "type3": "无照经营游商",
            "case_content_clean": "",
            "case_goal_clean": "",
            "title_clean": "",
            "address_detail_clean": "",
        }

        links = extract_entities_from_order(order)
        observed = {(link.entity_type, link.entity_value, link.source_field) for link in links}

        self.assertIn(("time_month", "2024-05", "call_month"), observed)
        self.assertIn(("area", "钟楼区", "area_code_area"), observed)
        self.assertIn(("street", "永红街道", "area_code_street"), observed)
        self.assertIn(("case_type", "无照经营游商", "type3"), observed)

    def test_extracts_case_content_space_and_problem_entities(self):
        order = {
            "doc_id": "order_b",
            "call_month": "2024-05",
            "area_code_area": "",
            "area_code_street": "",
            "type3": "",
            "case_content_clean": "市民反映钟楼区永红街道广成路和江春路交叉口有流动摊贩占道经营，影响通行。",
            "case_goal_clean": "希望城管处理",
            "title_clean": "流动摊贩占道",
            "address_detail_clean": "广成路与江春路交界处",
        }

        links = extract_entities_from_order(order)
        observed = {(link.entity_type, link.entity_value) for link in links}

        self.assertIn(("area", "钟楼区"), observed)
        self.assertIn(("street", "永红街道"), observed)
        self.assertIn(("road", "广成路"), observed)
        self.assertIn(("road", "江春路"), observed)
        self.assertIn(("intersection", "广成路和江春路交叉口"), observed)
        self.assertIn(("problem_object", "流动摊贩"), observed)
        self.assertIn(("problem_behavior", "占道经营"), observed)
        self.assertIn(("problem_behavior", "影响通行"), observed)

    def test_deduplicates_same_entity_from_same_source_field(self):
        links = [
            SagEntityLink("order_a", "road", "广成路", "广成路", "case_content_clean", "case_content", 0.9, "广成路"),
            SagEntityLink("order_a", "road", "广成路", "广成路", "case_content_clean", "case_content", 0.9, "广成路"),
            SagEntityLink("order_a", "road", "广成路", "广成路", "address_detail_clean", "address_detail", 0.9, "广成路"),
        ]

        deduped = deduplicate_entity_links(links)

        self.assertEqual(len(deduped), 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_sag_entities -v
```

Meaning:

```text
运行 sag_entities 的单元测试。
现在文件还不存在，所以预期失败。
```

Expected:

```text
ModuleNotFoundError: No module named 'ragflow_style_pipeline.sag_entities'
```

- [ ] **Step 3: Implement minimal entity extraction**

Create `src/ragflow_style_pipeline/sag_entities.py` with:

```python
"""Pure SAG-lite entity extraction for 12345 work orders."""

import re
from dataclasses import dataclass


NULLISH_VALUES = {"", "NULL", "null", "None", "none", "\\N"}

AREAS = [
    "钟楼区",
    "天宁区",
    "新北区",
    "武进区",
    "金坛区",
    "溧阳市",
    "常州市经济开发区",
    "经开区",
    "市本级",
]

PROBLEM_OBJECT_TERMS = [
    "流动摊贩",
    "游商摊贩",
    "夜市摊贩",
    "摊贩",
    "小摊",
    "商贩",
]

PROBLEM_BEHAVIOR_TERMS = [
    "占道经营",
    "无照经营",
    "店外经营",
    "影响通行",
    "摆摊",
    "设摊",
    "扰民",
    "油烟",
]

ROAD_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{1,12}(?:路|街|大道|巷|弄|桥|线)")
STREET_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{1,12}(?:街道|镇)")
INTERSECTION_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{1,12}(?:路|街|大道|巷|弄|桥|线))"
    r"(?:和|与|及|、)"
    r"([\u4e00-\u9fffA-Za-z0-9]{1,12}(?:路|街|大道|巷|弄|桥|线))"
    r"(?:交叉口|交界处|路口)"
)
POI_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{1,16}(?:小区|市场|学校|广场|商场|夜市|公园|医院|菜场|地铁站)"
)


@dataclass(frozen=True)
class SagEntityLink:
    doc_id: str
    entity_type: str
    entity_value: str
    normalized_value: str
    source_field: str
    source_channel: str
    confidence: float
    matched_text: str


def clean_value(value):
    if value is None:
        return ""
    value = str(value).strip()
    if value in NULLISH_VALUES:
        return ""
    return value


def normalize_entity_value(value):
    return re.sub(r"\s+", "", clean_value(value))


def _link(doc_id, entity_type, value, source_field, source_channel, confidence, matched_text=None):
    value = clean_value(value)
    normalized = normalize_entity_value(value)
    if not normalized:
        return None
    return SagEntityLink(
        doc_id=doc_id,
        entity_type=entity_type,
        entity_value=value,
        normalized_value=normalized,
        source_field=source_field,
        source_channel=source_channel,
        confidence=float(confidence),
        matched_text=clean_value(matched_text if matched_text is not None else value),
    )


def deduplicate_entity_links(links):
    seen = set()
    deduped = []
    for link in links:
        key = (link.doc_id, link.entity_type, link.normalized_value, link.source_field)
        if key not in seen:
            seen.add(key)
            deduped.append(link)
    return deduped


def _append_link(links, *args):
    link = _link(*args)
    if link is not None:
        links.append(link)


def _extract_known_terms(doc_id, text, source_field, source_channel, terms, entity_type, confidence):
    links = []
    for term in terms:
        if term and term in text:
            _append_link(links, doc_id, entity_type, term, source_field, source_channel, confidence, term)
    return links


def _extract_text_entities(doc_id, text, source_field, source_channel):
    links = []
    if not text:
        return links

    links.extend(_extract_known_terms(doc_id, text, source_field, source_channel, AREAS, "area", 0.9))
    links.extend(_extract_known_terms(doc_id, text, source_field, source_channel, PROBLEM_OBJECT_TERMS, "problem_object", 0.7))
    links.extend(_extract_known_terms(doc_id, text, source_field, source_channel, PROBLEM_BEHAVIOR_TERMS, "problem_behavior", 0.7))

    for match in STREET_PATTERN.finditer(text):
        _append_link(links, doc_id, "street", match.group(0), source_field, source_channel, 0.9, match.group(0))
    for match in INTERSECTION_PATTERN.finditer(text):
        _append_link(links, doc_id, "intersection", match.group(0), source_field, source_channel, 0.9, match.group(0))
    for match in ROAD_PATTERN.finditer(text):
        _append_link(links, doc_id, "road", match.group(0), source_field, source_channel, 0.9, match.group(0))
    for match in POI_PATTERN.finditer(text):
        _append_link(links, doc_id, "poi", match.group(0), source_field, source_channel, 0.6, match.group(0))

    return links


def extract_entities_from_order(order):
    doc_id = clean_value(order.get("doc_id"))
    links = []

    metadata_fields = [
        ("call_month", "time_month"),
        ("area_code_area", "area"),
        ("area_code_street", "street"),
        ("case_lnglat", "lnglat"),
        ("type1", "case_type"),
        ("type2", "case_type"),
        ("type3", "case_type"),
        ("type4", "case_type"),
        ("type5", "case_type"),
        ("deptName", "department"),
        ("orgName", "department"),
        ("belong_dept", "department"),
    ]
    for field_name, entity_type in metadata_fields:
        _append_link(links, doc_id, entity_type, order.get(field_name), field_name, "metadata", 1.0)

    text_fields = [
        ("case_content_clean", "case_content"),
        ("address_detail_clean", "address_detail"),
        ("title_clean", "title"),
        ("case_goal_clean", "case_goal"),
    ]
    for field_name, channel in text_fields:
        confidence = 0.8 if channel in {"title", "case_goal"} else 0.9
        text = clean_value(order.get(field_name))
        text_links = _extract_text_entities(doc_id, text, field_name, channel)
        links.extend(
            SagEntityLink(
                doc_id=link.doc_id,
                entity_type=link.entity_type,
                entity_value=link.entity_value,
                normalized_value=link.normalized_value,
                source_field=link.source_field,
                source_channel=link.source_channel,
                confidence=min(link.confidence, confidence) if link.entity_type.startswith("problem_") else link.confidence,
                matched_text=link.matched_text,
            )
            for link in text_links
        )

    return deduplicate_entity_links(links)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_sag_entities -v
```

Expected:

```text
Ran 4 tests
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ragflow_style_pipeline/sag_entities.py tests/test_sag_entities.py
git commit -m "feat: extract sag lite entities"
```

Meaning:

```text
只提交 SAG 实体抽取模块和对应测试。
```

---

## Task 2: Source Row Mapping

**Files:**

- Create: `src/ragflow_style_pipeline/sag_db.py`
- Create: `tests/test_sag_db.py`

### Purpose

Convert raw TSV rows or multiview JSONL documents into `source_orders` rows and `sag_events` rows. This keeps the design tied to `t_order_master.tsv`.

### API to implement

```python
def stable_hash(value):
    """Return short stable hash for sensitive source identifiers."""

def source_order_row(row):
    """Map one raw TSV or flattened multiview row into source_orders shape."""

def event_row(source_order):
    """Build one sag_events row from one source order row."""

def read_source_rows(input_path, limit=None):
    """Read .tsv or .jsonl and return source_order rows."""
```

### Steps

- [ ] **Step 1: Write failing tests for source mapping**

Create `tests/test_sag_db.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_db import event_row, read_source_rows, source_order_row, stable_hash


class TestSagDbMapping(unittest.TestCase):
    def test_stable_hash_does_not_return_raw_value(self):
        hashed = stable_hash("ORD001")

        self.assertNotEqual(hashed, "ORD001")
        self.assertEqual(hashed, stable_hash("ORD001"))
        self.assertEqual(len(hashed), 16)

    def test_source_order_row_maps_raw_tsv_fields(self):
        raw = {
            "id": "1",
            "order_id": "ORD001",
            "title": "广成路流动摊贩",
            "case_content": "市民反映钟楼区永红街道广成路有流动摊贩占道经营",
            "case_goal": "希望处理",
            "address_detail": "广成路与江春路交界处",
            "call_time": "2024-05-01 10:00:00",
            "area_code_city": "常州市",
            "area_code_area": "钟楼区",
            "area_code_street": "",
            "case_lnglat": "119.95,31.78",
            "case_accord_type_one_name": "城乡建设",
            "case_accord_type_two_name": "市容管理",
            "case_accord_type_three_name": "无照经营游商",
            "case_accord_type_four_name": "流动摊贩",
            "case_accord_type_five_name": "",
            "case_accord_code": "ABC",
            "order_source": "电话",
            "order_type": "个人",
            "order_status": "100",
            "service_object_type": "投诉举报",
        }

        row = source_order_row(raw)

        self.assertTrue(row["doc_id"].startswith("order_"))
        self.assertEqual(row["case_content_clean"], raw["case_content"])
        self.assertEqual(row["title_clean"], raw["title"])
        self.assertEqual(row["address_detail_clean"], raw["address_detail"])
        self.assertEqual(row["call_month"], "2024-05")
        self.assertEqual(row["type3"], "无照经营游商")
        self.assertEqual(row["type4"], "流动摊贩")
        self.assertEqual(row["raw_id_hash"], stable_hash("1"))
        self.assertEqual(row["order_id_hash"], stable_hash("ORD001"))

    def test_event_row_preserves_complete_event_semantics(self):
        source = source_order_row(
            {
                "id": "1",
                "order_id": "ORD001",
                "title": "广成路流动摊贩",
                "case_content": "市民反映钟楼区永红街道广成路有流动摊贩占道经营",
                "case_goal": "希望处理",
                "address_detail": "广成路与江春路交界处",
                "call_time": "2024-05-01 10:00:00",
                "area_code_area": "钟楼区",
                "area_code_street": "永红街道",
                "case_accord_type_three_name": "无照经营游商",
                "order_status": "100",
            }
        )

        event = event_row(source)

        self.assertEqual(event["doc_id"], source["doc_id"])
        self.assertEqual(event["event_month"], "2024-05")
        self.assertIn("广成路流动摊贩", event["event_text"])
        self.assertIn("无照经营游商", event["event_text"])
        self.assertIn("永红街道", event["event_text"])

    def test_read_source_rows_supports_tsv_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tsv_path = tmp / "orders.tsv"
            tsv_path.write_text(
                "id\torder_id\tcase_content\tcase_goal\tcall_time\n"
                "1\tORD001\t流动摊贩占道经营\t希望处理\t2024-05-01 10:00:00\n",
                encoding="utf-8",
            )
            jsonl_path = tmp / "orders.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "doc_id": "order_x",
                        "case_content_clean": "流动摊贩占道经营",
                        "case_goal_clean": "希望处理",
                        "metadata": {"call_time": "2024-05-01 10:00:00", "call_month": "2024-05"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            tsv_rows = read_source_rows(tsv_path)
            jsonl_rows = read_source_rows(jsonl_path)

        self.assertEqual(len(tsv_rows), 1)
        self.assertEqual(len(jsonl_rows), 1)
        self.assertEqual(tsv_rows[0]["call_month"], "2024-05")
        self.assertEqual(jsonl_rows[0]["doc_id"], "order_x")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_sag_db -v
```

Expected:

```text
ImportError or AttributeError for missing sag_db functions
```

- [ ] **Step 3: Implement source mapping**

Create `src/ragflow_style_pipeline/sag_db.py` with the mapping functions first. Do not create DuckDB tables in this step.

Use these exact column lists:

```python
SOURCE_ORDER_COLUMNS = [
    "doc_id",
    "raw_id_hash",
    "order_id_hash",
    "title_clean",
    "case_content_clean",
    "case_goal_clean",
    "address_detail_clean",
    "call_time",
    "call_month",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "case_lnglat",
    "type1",
    "type2",
    "type3",
    "type4",
    "type5",
    "case_accord_code",
    "order_source",
    "order_type",
    "order_status",
    "service_object_type",
]

SAG_EVENT_COLUMNS = [
    "event_id",
    "doc_id",
    "event_text",
    "event_time",
    "event_month",
    "event_source",
    "event_status",
]
```

Implementation requirements:

```text
stable_hash uses sha256 and returns first 16 hex characters.
doc_id uses raw doc_id if present; otherwise uses hash(id or order_id).
source_order_row maps raw TSV fields and multiview JSONL fields.
event_text joins non-empty title/case_content/case_goal/address/category/area/time lines.
read_source_rows detects .tsv and .jsonl by suffix.
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_sag_db -v
```

Expected:

```text
Ran 4 tests
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ragflow_style_pipeline/sag_db.py tests/test_sag_db.py
git commit -m "feat: map source orders to sag events"
```

---

## Task 3: DuckDB SAG Database Builder

**Files:**

- Modify: `src/ragflow_style_pipeline/sag_db.py`
- Modify: `tests/test_sag_db.py`

### Purpose

Create the physical SAG-lite database:

```text
source_orders
sag_events
sag_entities
sag_event_entity_links
```

### API to add

```python
def build_sag_db_from_orders(source_orders, db_path):
    """Create DuckDB tables from normalized source orders."""

def build_sag_db(input_path, db_path, limit=None):
    """Read input path and build DuckDB database."""
```

### Steps

- [ ] **Step 1: Add failing DuckDB build test**

Append this test class to `tests/test_sag_db.py`:

```python
class TestSagDbBuild(unittest.TestCase):
    def test_build_sag_db_from_orders_creates_events_entities_and_links(self):
        import duckdb

        orders = [
            source_order_row(
                {
                    "id": "1",
                    "order_id": "ORD001",
                    "title": "流动摊贩占道",
                    "case_content": "市民反映钟楼区永红街道广成路有流动摊贩占道经营",
                    "case_goal": "希望处理",
                    "address_detail": "广成路",
                    "call_time": "2024-05-01 10:00:00",
                    "area_code_area": "钟楼区",
                    "area_code_street": "",
                    "case_accord_type_three_name": "无照经营游商",
                }
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sag.duckdb"
            report = build_sag_db_from_orders(orders, db_path)
            with duckdb.connect(str(db_path)) as conn:
                event_count = conn.execute("select count(*) from sag_events").fetchone()[0]
                entity_count = conn.execute("select count(*) from sag_entities").fetchone()[0]
                road_count = conn.execute(
                    "select count(*) from sag_event_entity_links where entity_type = 'road'"
                ).fetchone()[0]

        self.assertEqual(report["events_loaded"], 1)
        self.assertEqual(event_count, 1)
        self.assertGreater(entity_count, 0)
        self.assertGreaterEqual(road_count, 1)
```

Also update the import:

```python
from ragflow_style_pipeline.sag_db import (
    build_sag_db_from_orders,
    event_row,
    read_source_rows,
    source_order_row,
    stable_hash,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_sag_db.TestSagDbBuild -v
```

Expected:

```text
ImportError or NameError for build_sag_db_from_orders
```

- [ ] **Step 3: Implement DuckDB builder**

Add to `src/ragflow_style_pipeline/sag_db.py`:

```text
DuckDB dependency is imported inside functions.
Drop and recreate the four SAG tables.
Insert source_orders.
Insert sag_events.
Use extract_entities_from_order() from sag_entities.py.
Build sag_entities dictionary by (entity_type, normalized_value).
Insert sag_event_entity_links.
Create indexes on doc_id, event_id, entity_type, normalized_value, event_month.
Return counts: source_orders_loaded, events_loaded, entities_loaded, links_loaded.
```

Implementation note:

```python
def _entity_id(entity_type, normalized_value):
    return "entity_" + stable_hash(f"{entity_type}:{normalized_value}")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_sag_db -v
```

Expected:

```text
Ran 5 tests
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ragflow_style_pipeline/sag_db.py tests/test_sag_db.py
git commit -m "feat: build sag lite duckdb"
```

---

## Task 4: Pure SAG Seed Query and 1-hop Expansion

**Files:**

- Create: `src/ragflow_style_pipeline/sag_query.py`
- Create: `tests/test_sag_query.py`

### Purpose

Implement the core pure SAG retrieval:

```text
seed entity match
  -> dynamic SQL expansion through shared spatial entities
  -> structural scoring
```

### API to implement

```python
def load_sag_query_config(config_path):
    """Load JSON query config."""

def query_sag_db(db_path, config):
    """Return ordered pure SAG result dictionaries."""

def score_sag_result(match_stage, matched_seed_count, matched_space_count, confidence_sum):
    """Return structural score without semantic similarity."""
```

### Steps

- [ ] **Step 1: Write failing query tests**

Create `tests/test_sag_query.py`:

```python
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_db import build_sag_db_from_orders, source_order_row
from ragflow_style_pipeline.sag_query import query_sag_db, score_sag_result


def _build_test_db(tmpdir):
    orders = [
        source_order_row(
            {
                "id": "1",
                "order_id": "ORD001",
                "case_content": "市民反映钟楼区永红街道广成路有流动摊贩占道经营，影响通行。",
                "case_goal": "希望处理",
                "address_detail": "广成路",
                "call_time": "2024-05-01 10:00:00",
                "area_code_area": "钟楼区",
                "area_code_street": "",
                "case_accord_type_three_name": "无照经营游商",
            }
        ),
        source_order_row(
            {
                "id": "2",
                "order_id": "ORD002",
                "case_content": "市民反映广成路附近有夜间噪声。",
                "case_goal": "希望处理",
                "address_detail": "广成路",
                "call_time": "2024-05-02 10:00:00",
                "area_code_area": "钟楼区",
                "area_code_street": "",
                "case_accord_type_three_name": "社会生活噪声",
            }
        ),
        source_order_row(
            {
                "id": "3",
                "order_id": "ORD003",
                "case_content": "咨询医保办理条件。",
                "case_goal": "希望了解政策",
                "call_time": "2024-05-03 10:00:00",
                "area_code_area": "天宁区",
                "case_accord_type_three_name": "职工医疗保险",
            }
        ),
    ]
    db_path = Path(tmpdir) / "sag.duckdb"
    build_sag_db_from_orders(orders, db_path)
    return db_path


class TestSagQuery(unittest.TestCase):
    def test_score_prioritizes_seed_over_expansion(self):
        seed_score = score_sag_result("seed_entity", 2, 0, 1.4)
        expanded_score = score_sag_result("one_hop_expansion", 0, 2, 1.8)

        self.assertGreater(seed_score, expanded_score)

    def test_query_sag_db_returns_seed_and_one_hop_expansion(self):
        config = {
            "query_name": "stall",
            "seed_entities": [
                {"entity_type": "problem_object", "values": ["流动摊贩", "摊贩"], "operator": "OR"},
                {"entity_type": "problem_behavior", "values": ["占道经营"], "operator": "OR"},
            ],
            "seed_group_operator": "AND",
            "filters": {"call_month_gte": "2024-01", "call_month_lte": "2024-12"},
            "expansion": {
                "enabled": True,
                "max_hops": 1,
                "frontier_entity_types": ["road"],
                "max_expanded_events": 10,
            },
            "representative_limit": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _build_test_db(tmpdir)
            results = query_sag_db(db_path, config)

        self.assertGreaterEqual(len(results), 2)
        self.assertEqual(results[0]["match_stage"], "seed_entity")
        self.assertIn("problem_object", results[0]["matched_entities"])
        stages = {result["match_stage"] for result in results}
        self.assertIn("one_hop_expansion", stages)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_sag_query -v
```

Expected:

```text
ModuleNotFoundError: No module named 'ragflow_style_pipeline.sag_query'
```

- [ ] **Step 3: Implement minimal pure SAG query**

Create `src/ragflow_style_pipeline/sag_query.py`.

Implementation requirements:

```text
Read config dictionaries directly.
Seed retrieval:
  For each seed group, query doc_id/event_id where entity_type matches and normalized_value is in normalized config values.
  OR inside group.
  AND across groups by intersecting event sets.
Apply month filter using sag_events.event_month.
Expansion:
  From seed event ids, collect frontier entities of configured types.
  Find other events sharing those entity ids.
  Exclude seed events from expansion stage.
  Limit to max_expanded_events.
Scoring:
  Seed score > expansion score.
Return sorted result dicts.
```

Result dictionary shape:

```python
{
    "rank": 1,
    "doc_id": "order_x",
    "event_id": "event_x",
    "score": 21.4,
    "match_stage": "seed_entity",
    "matched_entities": {"problem_object": ["流动摊贩"], "problem_behavior": ["占道经营"]},
    "explanation": {"reason": "matched seed entities"},
}
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_sag_query -v
```

Expected:

```text
Ran 2 tests
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ragflow_style_pipeline/sag_query.py tests/test_sag_query.py
git commit -m "feat: query sag lite event entities"
```

---

## Task 5: SAG Report Aggregation

**Files:**

- Modify: `src/ragflow_style_pipeline/sag_query.py`
- Modify: `tests/test_sag_query.py`

### Purpose

Turn raw SAG results into the report required by the design:

```text
statistics
entity_coverage
metadata_recovery
conflict_report
representative_cases
retrieval
```

### API to add

```python
def analyze_sag_query(db_path, config):
    """Return full pure SAG-lite report."""
```

### Steps

- [ ] **Step 1: Add failing report test**

Append to `tests/test_sag_query.py`:

```python
from ragflow_style_pipeline.sag_query import analyze_sag_query


class TestSagReport(unittest.TestCase):
    def test_analyze_sag_query_reports_statistics_and_metadata_recovery(self):
        config = {
            "query_name": "stall",
            "seed_entities": [
                {"entity_type": "problem_object", "values": ["流动摊贩", "摊贩"], "operator": "OR"},
                {"entity_type": "problem_behavior", "values": ["占道经营"], "operator": "OR"},
            ],
            "seed_group_operator": "AND",
            "filters": {"call_month_gte": "2024-01", "call_month_lte": "2024-12"},
            "expansion": {
                "enabled": True,
                "max_hops": 1,
                "frontier_entity_types": ["road"],
                "max_expanded_events": 10,
            },
            "representative_limit": 5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _build_test_db(tmpdir)
            report = analyze_sag_query(db_path, config)

        self.assertEqual(report["query"]["query_name"], "stall")
        self.assertGreaterEqual(report["matched_orders"], 2)
        self.assertIn("by_month", report["statistics"])
        self.assertIn("road", report["entity_coverage"])
        self.assertIn("metadata_street_missing", report["metadata_recovery"])
        self.assertGreaterEqual(report["metadata_recovery"]["metadata_street_missing_but_text_road_found"], 1)
        self.assertGreaterEqual(len(report["representative_cases"]), 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_sag_query.TestSagReport -v
```

Expected:

```text
ImportError or AttributeError for analyze_sag_query
```

- [ ] **Step 3: Implement report aggregation**

Add to `src/ragflow_style_pipeline/sag_query.py`.

Implementation requirements:

```text
statistics.by_month counts source_orders.call_month for matched docs.
statistics.by_area_metadata counts source_orders.area_code_area.
statistics.by_street_metadata counts source_orders.area_code_street.
statistics.by_street_entity counts entity_type = street.
statistics.by_road_entity counts entity_type = road.
statistics.by_intersection_entity counts entity_type = intersection.
statistics.by_poi_entity counts entity_type = poi.
statistics.by_problem_object counts entity_type = problem_object.
statistics.by_problem_behavior counts entity_type = problem_behavior.
entity_coverage reports events_total, events_with_entity, coverage, source_breakdown.
metadata_recovery focuses on source_orders.area_code_street missing but text-derived street/road/intersection/poi found.
representative_cases include case_content_clean, case_goal_clean, area_code_area, area_code_street, matched_entities, explanation.
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests.test_sag_query -v
```

Expected:

```text
Ran 3 tests
OK
```

- [ ] **Step 5: Commit**

Run:

```bash
git add src/ragflow_style_pipeline/sag_query.py tests/test_sag_query.py
git commit -m "feat: report sag lite statistics"
```

---

## Task 6: Evaluation Metrics

**Files:**

- Create: `src/ragflow_style_pipeline/sag_eval.py`
- Create: `tests/test_sag_eval.py`

### Purpose

Add evaluation so results are not judged only by “looks good”. This implements weak-label metrics, artificial labeling samples, and entity sampling.

### API to implement

```python
def precision_at_k(results, gold_doc_ids, k):
    """Return fraction of top-k results whose doc_id is in gold_doc_ids."""

def recall_at_k(results, gold_doc_ids, k):
    """Return fraction of gold_doc_ids covered by top-k results."""

def build_weak_gold_doc_ids(db_path, config):
    """Build weak gold set for the query from category and keyword rules."""

def evaluate_sag_results(db_path, config, results):
    """Return weak precision/recall metrics."""

def build_manual_eval_samples(db_path, results, limit=100):
    """Return JSONL-ready manual evaluation samples."""
```

### Steps

- [ ] **Step 1: Write failing tests**

Create `tests/test_sag_eval.py`:

```python
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_db import build_sag_db_from_orders, source_order_row
from ragflow_style_pipeline.sag_eval import (
    build_manual_eval_samples,
    build_weak_gold_doc_ids,
    evaluate_sag_results,
    precision_at_k,
    recall_at_k,
)
from ragflow_style_pipeline.sag_query import query_sag_db


class TestSagEval(unittest.TestCase):
    def test_precision_and_recall_at_k(self):
        results = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]
        gold = {"a", "c", "x"}

        self.assertAlmostEqual(precision_at_k(results, gold, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(results, gold, 2), 1 / 3)
        self.assertAlmostEqual(recall_at_k(results, gold, 3), 2 / 3)

    def test_evaluate_sag_results_uses_weak_gold(self):
        orders = [
            source_order_row(
                {
                    "id": "1",
                    "order_id": "ORD001",
                    "case_content": "流动摊贩占道经营",
                    "case_goal": "希望处理",
                    "call_time": "2024-05-01 10:00:00",
                    "case_accord_type_three_name": "无照经营游商",
                }
            ),
            source_order_row(
                {
                    "id": "2",
                    "order_id": "ORD002",
                    "case_content": "咨询医保办理条件",
                    "case_goal": "希望了解",
                    "call_time": "2024-05-02 10:00:00",
                    "case_accord_type_three_name": "职工医疗保险",
                }
            ),
        ]
        config = {
            "query_name": "stall",
            "seed_entities": [
                {"entity_type": "problem_object", "values": ["流动摊贩", "摊贩"], "operator": "OR"},
                {"entity_type": "problem_behavior", "values": ["占道经营"], "operator": "OR"},
            ],
            "seed_group_operator": "AND",
            "filters": {"call_month_gte": "2024-01", "call_month_lte": "2024-12"},
            "expansion": {"enabled": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sag.duckdb"
            build_sag_db_from_orders(orders, db_path)
            results = query_sag_db(db_path, config)
            gold = build_weak_gold_doc_ids(db_path, config)
            metrics = evaluate_sag_results(db_path, config, results)
            samples = build_manual_eval_samples(db_path, results, limit=1)

        self.assertEqual(len(gold), 1)
        self.assertEqual(metrics["weak_precision@10"], 1.0)
        self.assertEqual(metrics["weak_recall@100"], 1.0)
        self.assertEqual(len(samples), 1)
        self.assertIn("label", samples[0])
        self.assertEqual(samples[0]["label"], "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_sag_eval -v
```

Expected:

```text
ModuleNotFoundError: No module named 'ragflow_style_pipeline.sag_eval'
```

- [ ] **Step 3: Implement evaluation**

Create `src/ragflow_style_pipeline/sag_eval.py`.

Implementation requirements:

```text
precision_at_k returns 0.0 when k <= 0 or results empty.
recall_at_k returns 0.0 when gold_doc_ids empty.
Weak gold uses type3 and keywords:
  type3 in 无照经营游商, 店外经营, 无证照餐饮店
  or case_content/title/case_goal contains 流动摊贩, 游商摊贩, 摆摊, 设摊, 占道经营, 无照经营, 店外经营
evaluate_sag_results computes:
  weak_precision@10
  weak_precision@50
  weak_precision@100
  weak_recall@100
  weak_recall@500
  weak_recall@1000
  weak_gold_count
build_manual_eval_samples returns result rows with empty label and label_reason.
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests.test_sag_eval -v
```

Expected:

```text
Ran 2 tests
OK
```

- [ ] **Step 5: Integrate evaluation into analyze_sag_query**

Modify `src/ragflow_style_pipeline/sag_query.py`:

```text
Import evaluate_sag_results from sag_eval inside analyze_sag_query to avoid circular imports.
Add report["evaluation"] = evaluate_sag_results(db_path, config, results).
```

- [ ] **Step 6: Run affected tests**

Run:

```bash
python -m unittest tests.test_sag_query tests.test_sag_eval -v
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit**

Run:

```bash
git add src/ragflow_style_pipeline/sag_eval.py src/ragflow_style_pipeline/sag_query.py tests/test_sag_eval.py
git commit -m "feat: evaluate sag lite retrieval"
```

---

## Task 7: CLI Entrypoints, Config, and Server Scripts

**Files:**

- Modify: `src/ragflow_style_pipeline/sag_db.py`
- Modify: `src/ragflow_style_pipeline/sag_query.py`
- Modify: `src/ragflow_style_pipeline/sag_eval.py`
- Create: `configs/sag_query_stall.json`
- Create: `scripts/build_sag_lite_100k.sh`
- Create: `scripts/query_sag_lite_stall_100k.sh`
- Create: `scripts/evaluate_sag_lite_stall_100k.sh`

### Purpose

Make the implementation runnable on the Linux GPU server, even though this pure SAG phase does not use GPU.

### Steps

- [ ] **Step 1: Add CLI parsing to sag_db.py**

Add:

```text
python -m ragflow_style_pipeline.sag_db \
  --input data/t_order_master.tsv \
  --db outputs/sag_lite.100k.duckdb \
  --limit 100000
```

Expected JSON output:

```json
{
  "input_path": "data/t_order_master.tsv",
  "db_path": "outputs/sag_lite.100k.duckdb",
  "source_orders_loaded": 100000,
  "events_loaded": 100000,
  "entities_loaded": 0,
  "links_loaded": 0
}
```

The exact entity/link counts will be non-zero on real data.

- [ ] **Step 2: Add CLI parsing to sag_query.py**

Add:

```text
python -m ragflow_style_pipeline.sag_query \
  --db outputs/sag_lite.100k.duckdb \
  --config configs/sag_query_stall.json \
  --output outputs/sag_lite.query.stall.100k.json
```

Expected behavior:

```text
Load config.
Run analyze_sag_query.
Write JSON report with ensure_ascii=False and indent=2.
Print the same JSON summary.
```

- [ ] **Step 3: Add CLI parsing to sag_eval.py**

Add:

```text
python -m ragflow_style_pipeline.sag_eval \
  --db outputs/sag_lite.100k.duckdb \
  --query-report outputs/sag_lite.query.stall.100k.json \
  --manual-samples outputs/sag_lite.eval_samples.stall.100k.jsonl \
  --entity-samples outputs/sag_lite.entity_eval_samples.100k.jsonl
```

Expected behavior:

```text
Read the SAG query report.
Export manual evaluation JSONL.
Export entity evaluation JSONL.
Print counts.
```

- [ ] **Step 4: Create stall query config**

Create `configs/sag_query_stall.json`:

```json
{
  "query_name": "stall",
  "seed_entities": [
    {
      "entity_type": "problem_object",
      "values": ["流动摊贩", "游商摊贩", "夜市摊贩", "摊贩", "小摊", "商贩"],
      "operator": "OR"
    },
    {
      "entity_type": "problem_behavior",
      "values": ["摆摊", "设摊", "占道经营", "无照经营", "店外经营"],
      "operator": "OR"
    }
  ],
  "seed_group_operator": "AND",
  "space_entities": [],
  "filters": {
    "call_month_gte": "2024-01",
    "call_month_lte": "2024-12"
  },
  "expansion": {
    "enabled": true,
    "max_hops": 1,
    "frontier_entity_types": ["street", "road", "intersection", "poi"],
    "max_expanded_events": 2000
  },
  "representative_limit": 20
}
```

Note:

```text
第一版默认不把 area 放进 expansion frontier。
原因：area 粒度太粗，容易把结果膨胀成“同区所有问题”。
```

- [ ] **Step 5: Create server build script**

Create `scripts/build_sag_lite_100k.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

python -m ragflow_style_pipeline.sag_db \
  --input data/t_order_master.tsv \
  --db outputs/sag_lite.100k.duckdb \
  --limit 100000
```

Command meaning for docs:

```text
mkdir -p outputs：如果 outputs 文件夹不存在，就创建它。
python -m ragflow_style_pipeline.sag_db：用 Python 模块方式运行建库程序。
--input：指定原始 TSV 数据位置。
--db：指定输出 DuckDB 数据库位置。
--limit：只读取前 100000 行，方便先做 10 万行实验。
```

- [ ] **Step 6: Create server query script**

Create `scripts/query_sag_lite_stall_100k.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m ragflow_style_pipeline.sag_query \
  --db outputs/sag_lite.100k.duckdb \
  --config configs/sag_query_stall.json \
  --output outputs/sag_lite.query.stall.100k.json
```

- [ ] **Step 7: Create server evaluation script**

Create `scripts/evaluate_sag_lite_stall_100k.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

python -m ragflow_style_pipeline.sag_eval \
  --db outputs/sag_lite.100k.duckdb \
  --query-report outputs/sag_lite.query.stall.100k.json \
  --manual-samples outputs/sag_lite.eval_samples.stall.100k.jsonl \
  --entity-samples outputs/sag_lite.entity_eval_samples.100k.jsonl
```

- [ ] **Step 8: Run syntax checks**

Run:

```bash
python -m py_compile src/ragflow_style_pipeline/sag_entities.py src/ragflow_style_pipeline/sag_db.py src/ragflow_style_pipeline/sag_query.py src/ragflow_style_pipeline/sag_eval.py
```

Expected:

```text
No output
```

Run on Linux server or WSL:

```bash
bash -n scripts/build_sag_lite_100k.sh
bash -n scripts/query_sag_lite_stall_100k.sh
bash -n scripts/evaluate_sag_lite_stall_100k.sh
```

Expected:

```text
No output
```

- [ ] **Step 9: Commit**

Run:

```bash
git add src/ragflow_style_pipeline/sag_db.py src/ragflow_style_pipeline/sag_query.py src/ragflow_style_pipeline/sag_eval.py configs/sag_query_stall.json scripts/build_sag_lite_100k.sh scripts/query_sag_lite_stall_100k.sh scripts/evaluate_sag_lite_stall_100k.sh
git commit -m "feat: add sag lite cli scripts"
```

---

## Task 8: Learning Note and Final Verification

**Files:**

- Create: `docs/11-纯SAG工单检索实验.md`

### Purpose

Explain the pure SAG pipeline in Chinese and provide beginner-friendly server commands.

### Steps

- [ ] **Step 1: Create learning note**

Create `docs/11-纯SAG工单检索实验.md` with sections:

```text
1. 为什么把 Hybrid Retrieval 固定为 baseline
2. 纯 SAG-lite 和 SAG 论文怎么对应
3. t_order_master.tsv 哪些字段进入 event
4. 哪些字段进入 entity
5. 为什么 case_content 是核心
6. 为什么 area_code_street 缺失时不能硬过滤
7. 怎么运行建库脚本
8. 怎么运行查询脚本
9. 怎么运行评估脚本
10. 怎么读结果
11. 当前限制和下一步
```

Include these beginner command explanations:

```bash
bash scripts/build_sag_lite_100k.sh
```

Meaning:

```text
用 Bash 执行建库脚本，把 data/t_order_master.tsv 的前 10 万行转成 SAG-lite DuckDB 数据库。
```

```bash
bash scripts/query_sag_lite_stall_100k.sh
```

Meaning:

```text
运行纯 SAG 查询，不使用 embedding，只通过 event-entity 关系找“流动摆摊/占道经营”相关工单。
```

```bash
bash scripts/evaluate_sag_lite_stall_100k.sh
```

Meaning:

```text
根据弱标签和抽样文件评估纯 SAG 结果，帮助判断结果是否真的有用。
```

- [ ] **Step 2: Run full unit tests**

Run:

```bash
python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected for SAG files:

```text
Only intended SAG-lite files are modified or untracked.
```

There may be older unrelated modified files from previous work. Do not stage unrelated files.

- [ ] **Step 4: Commit docs**

Run:

```bash
git add docs/11-纯SAG工单检索实验.md
git commit -m "docs: explain pure sag lite experiment"
```

- [ ] **Step 5: Package server files**

Run from `G:\RAG` on Windows PowerShell:

```powershell
Compress-Archive -Path .\ragflow-learning-plan\src,.\ragflow-learning-plan\tests,.\ragflow-learning-plan\configs,.\ragflow-learning-plan\scripts,.\ragflow-learning-plan\docs,.\ragflow-learning-plan\requirements.embedding.txt -DestinationPath .\packages\ragflow-learning-plan-pure-sag-lite.zip -Force
```

Meaning:

```text
把服务器需要的代码、测试、配置、脚本、文档打成 zip 包。
-Force 表示如果同名 zip 已存在，就覆盖。
```

---

## Final Verification Checklist

Before claiming completion:

- [ ] `python -m unittest tests.test_sag_entities -v` passes.
- [ ] `python -m unittest tests.test_sag_db -v` passes.
- [ ] `python -m unittest tests.test_sag_query -v` passes.
- [ ] `python -m unittest tests.test_sag_eval -v` passes.
- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] SAG query report contains `evaluation`.
- [ ] SAG query report contains `metadata_recovery`.
- [ ] SAG query report separates `seed_orders` and `expanded_orders`.
- [ ] `configs/sag_query_stall.json` does not include embedding/BM25/LLM settings.
- [ ] Server scripts do not require Docker.
- [ ] Server scripts do not require GPU.
- [ ] Existing Hybrid Retrieval files are not changed by this feature.

---

## Server Run Order After Implementation

The user should run these commands on the Linux server.

```bash
cd ragflow-learning-plan
```

Meaning:

```text
进入项目目录。
```

```bash
cp /你的实际路径/t_order_master.tsv data/t_order_master.tsv
```

Meaning:

```text
把原始 TSV 数据复制到项目的 data 目录。
左边 `/你的实际路径/t_order_master.tsv` 要换成服务器上真实的数据文件路径。
右边 `data/t_order_master.tsv` 是脚本默认读取的位置。
```

```bash
bash scripts/build_sag_lite_100k.sh
```

Meaning:

```text
构建纯 SAG-lite DuckDB 数据库。
```

```bash
bash scripts/query_sag_lite_stall_100k.sh
```

Meaning:

```text
运行“流动摆摊/占道经营”纯 SAG 查询。
```

```bash
bash scripts/evaluate_sag_lite_stall_100k.sh
```

Meaning:

```text
生成评估样本和指标文件。
```

Expected output files:

```text
outputs/sag_lite.100k.duckdb
outputs/sag_lite.query.stall.100k.json
outputs/sag_lite.eval_samples.stall.100k.jsonl
outputs/sag_lite.entity_eval_samples.100k.jsonl
```
