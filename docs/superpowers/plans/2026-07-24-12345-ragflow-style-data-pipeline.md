# 12345 RAGFlow-style Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming TSV-to-JSONL preprocessing pipeline that converts 12345 order records into redacted RAG documents with metadata.

**Architecture:** The pipeline is a small Python package with separate modules for schema/config, PII redaction, document construction, streaming TSV reading, JSONL export, and quality reporting. It uses standard-library Python first so it can run in the existing `cz12345` conda environment without extra installs.

**Tech Stack:** Python 3.11, standard library `csv`, `json`, `re`, `argparse`, `pathlib`, `unittest`.

---

### Task 1: Repository structure and ignore rules

**Files:**

- Modify: `.gitignore`
- Create: `src/ragflow_style_pipeline/__init__.py`
- Create: `tests/fixtures/t_order_master_sample.tsv`

- [ ] **Step 1: Ignore generated outputs**

Add these lines to `.gitignore`:

```gitignore
outputs/
*.jsonl
*.quality.json
```

- [ ] **Step 2: Create package marker**

Create `src/ragflow_style_pipeline/__init__.py`:

```python
"""RAGFlow-style preprocessing pipeline for 12345 order TSV data."""
```

- [ ] **Step 3: Create a tiny TSV fixture**

Create `tests/fixtures/t_order_master_sample.tsv` with a header containing the fields used by the first pipeline version and three rows:

```text
id	order_id	service_object_type	case_content	case_goal	area_code_city	area_code_area	area_code_street	case_accord_type_one_name	case_accord_type_two_name	case_accord_type_three_name	order_source	order_type	order_status	call_time
1	ORD001	求助	市民反映手机号一三八零零一三八零零附近夜间摆摊扰民	希望处理占道经营	常州市	武进区	湖塘镇	城乡建设	市容管理	无照经营游商	电话	个人	100	2025-01-02 10:11:12
2	ORD002	咨询	咨询身份证三二零四零零一九九零零一零一一二三四能否办理医保	希望了解医保政策	常州市	市本级		民生保障	社会保障	职工医疗保险	电话	个人	100	2025-02-03 09:00:00
3	ORD003	投诉举报	NULL	希望处理噪声问题	常州市	天宁区	茶山街道	环境保护	噪声污染	社会生活噪声	互联网	个人	31	2025-03-04 20:30:00
```

### Task 2: PII redaction

**Files:**

- Create: `src/ragflow_style_pipeline/pii_redactor.py`
- Create: `tests/test_pii_redactor.py`

- [ ] **Step 1: Write tests**

```python
from ragflow_style_pipeline.pii_redactor import redact_text


def test_redacts_mainland_phone_number():
    text, counts = redact_text("联系电话" + "138" + "0013" + "8000" + "，请处理")
    assert text == "联系电话[手机号]，请处理"
    assert counts["phone"] == 1


def test_redacts_18_digit_id_number():
    text, counts = redact_text("身份证" + "320400" + "19900101" + "1234" + "可以办理吗")
    assert text == "身份证[身份证号]可以办理吗"
    assert counts["id_card"] == 1


def test_none_becomes_empty_text():
    text, counts = redact_text(None)
    assert text == ""
    assert counts["phone"] == 0
    assert counts["id_card"] == 0
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
& 'D:\anaconda\set\envs\cz12345\python.exe' -m unittest discover -s tests
```

Expected: import failure because module does not exist yet.

- [ ] **Step 3: Implement redactor**

```python
import re
from collections import Counter

PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def redact_text(value):
    if value is None:
        value = ""
    text = str(value)
    counts = Counter()
    text, phone_count = PHONE_RE.subn("[手机号]", text)
    text, id_count = ID_CARD_RE.subn("[身份证号]", text)
    counts["phone"] += phone_count
    counts["id_card"] += id_count
    return text, counts
```

- [ ] **Step 4: Run tests and confirm pass**

Run:

```powershell
& 'D:\anaconda\set\envs\cz12345\python.exe' -m unittest discover -s tests
```

Expected: all redactor tests pass.

### Task 3: Document builder

**Files:**

- Create: `src/ragflow_style_pipeline/document_builder.py`
- Create: `tests/test_document_builder.py`

- [ ] **Step 1: Write tests**

```python
from ragflow_style_pipeline.document_builder import build_document


def test_builds_document_text_and_metadata():
    row = {
        "id": "1",
        "order_id": "ORD001",
        "service_object_type": "求助",
        "case_content": "市民反映手机号" + "138" + "0013" + "8000" + "附近夜间摆摊扰民",
        "case_goal": "希望处理占道经营",
        "area_code_city": "常州市",
        "area_code_area": "武进区",
        "area_code_street": "湖塘镇",
        "case_accord_type_one_name": "城乡建设",
        "case_accord_type_two_name": "市容管理",
        "case_accord_type_three_name": "无照经营游商",
        "order_source": "电话",
        "order_type": "个人",
        "order_status": "100",
        "call_time": "2025-01-02 10:11:12",
    }

    doc, counts = build_document(row)

    assert doc["doc_id"] == "1"
    assert "诉求内容：市民反映手机号[手机号]附近夜间摆摊扰民" in doc["text"]
    assert doc["metadata"]["area_code_area"] == "武进区"
    assert doc["metadata"]["type3"] == "无照经营游商"
    assert doc["metadata"]["call_month"] == "2025-01"
    assert counts["phone"] == 1


def test_skips_null_text_fields():
    row = {
        "id": "3",
        "order_id": "ORD003",
        "service_object_type": "投诉举报",
        "case_content": "NULL",
        "case_goal": "希望处理噪声问题",
        "area_code_city": "常州市",
        "area_code_area": "天宁区",
        "area_code_street": "茶山街道",
        "case_accord_type_one_name": "环境保护",
        "case_accord_type_two_name": "噪声污染",
        "case_accord_type_three_name": "社会生活噪声",
        "order_source": "互联网",
        "order_type": "个人",
        "order_status": "31",
        "call_time": "2025-03-04 20:30:00",
    }

    doc, counts = build_document(row)

    assert "诉求内容：NULL" not in doc["text"]
    assert "诉求目标：希望处理噪声问题" in doc["text"]
    assert counts["phone"] == 0
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
& 'D:\anaconda\set\envs\cz12345\python.exe' -m unittest discover -s tests
```

Expected: import failure because document builder does not exist yet.

- [ ] **Step 3: Implement document builder**

Implement `is_nullish`, `clean_value`, `build_text`, `build_metadata`, and `build_document` using the field list in the design spec.

- [ ] **Step 4: Run tests and confirm pass**

Run:

```powershell
& 'D:\anaconda\set\envs\cz12345\python.exe' -m unittest discover -s tests
```

Expected: all document builder tests pass.

### Task 4: Streaming exporter CLI

**Files:**

- Create: `src/ragflow_style_pipeline/export_jsonl.py`
- Create: `tests/test_export_jsonl.py`

- [ ] **Step 1: Write tests**

Test that the fixture TSV exports two documents when `--limit 2` is used, writes JSONL, writes quality JSON, and redacts phone/ID values.

- [ ] **Step 2: Implement CLI**

The CLI must support:

```powershell
& 'D:\anaconda\set\envs\cz12345\python.exe' -m ragflow_style_pipeline.export_jsonl `
  --input 'G:\12345_pro_promax\data\t_order_master.tsv' `
  --output 'outputs\t_order_master.sample.jsonl' `
  --quality-report 'outputs\t_order_master.sample.quality.json' `
  --limit 1000
```

Required behavior:

- Read UTF-8 TSV with tab delimiter.
- Stream rows with `csv.DictReader`.
- Skip rows whose field count does not match header count.
- Use `build_document`.
- Write one JSON object per line with `ensure_ascii=False`.
- Write quality report with read rows, output docs, skipped rows, and redaction counts.

- [ ] **Step 3: Run tests and confirm pass**

Run:

```powershell
& 'D:\anaconda\set\envs\cz12345\python.exe' -m unittest discover -s tests
```

Expected: all tests pass.

### Task 5: Generate first safe sample

**Files:**

- Output only: `outputs/t_order_master.sample.jsonl`
- Output only: `outputs/t_order_master.sample.quality.json`

- [ ] **Step 1: Run sample export**

Run:

```powershell
& 'D:\anaconda\set\envs\cz12345\python.exe' -m ragflow_style_pipeline.export_jsonl `
  --input 'G:\12345_pro_promax\data\t_order_master.tsv' `
  --output 'outputs\t_order_master.sample.jsonl' `
  --quality-report 'outputs\t_order_master.sample.quality.json' `
  --limit 1000
```

Expected: JSONL and quality report are created under `outputs/`.

- [ ] **Step 2: Verify no obvious PII patterns in sample**

Run:

```powershell
rg -n "1[3-9]\d{9}|\d{17}[\dXx]" outputs\t_order_master.sample.jsonl
```

Expected: no matches.

- [ ] **Step 3: Inspect quality report**

Run:

```powershell
Get-Content outputs\t_order_master.sample.quality.json
```

Expected: includes read row count, output document count, skipped row count, and redaction counts.

### Task 6: Commit implementation

**Files:**

- All source and test files.
- No generated outputs.

- [ ] **Step 1: Check status**

Run:

```powershell
git status --short
```

Expected: source, tests, configs, and docs only. No `outputs/` files staged.

- [ ] **Step 2: Commit**

Run:

```powershell
git add .gitignore src tests docs
git commit -m "feat: add ragflow-style tsv preprocessing pipeline"
```

Expected: commit succeeds.
