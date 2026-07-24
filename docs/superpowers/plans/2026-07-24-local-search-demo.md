# Local Search Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local BM25-style retrieval demo for redacted 12345 order JSONL documents.

**Architecture:** The feature is split into tokenizer, in-memory search index, and CLI entry point. The tokenizer has no external dependencies; the index only stores document metadata, document lengths, and a posting list for query-time scoring.

**Tech Stack:** Python 3.12 standard library, `unittest`, JSONL files produced by the existing export pipeline.

---

### Task 1: Tokenizer

**Files:**
- Create: `src/ragflow_style_pipeline/text_tokenizer.py`
- Test: `tests/test_text_tokenizer.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:

```python
tokens = tokenize("武进区 夜间摆摊扰民 GPT-5 2024")
assert "武进" in tokens
assert "进区" in tokens
assert "夜间" in tokens
assert "摆摊" in tokens
assert "扰民" in tokens
assert "gpt" in tokens
assert "5" in tokens
assert "2024" in tokens
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_text_tokenizer
```

Expected: import failure because `text_tokenizer.py` does not exist.

- [ ] **Step 3: Implement tokenizer**

Create `tokenize(text)` with:

- lowercase ASCII tokens.
- numeric tokens.
- Chinese bigrams for contiguous Chinese text.
- one-character token only for single-character Chinese segments.

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_text_tokenizer
```

Expected: all tokenizer tests pass.

### Task 2: Search index

**Files:**
- Create: `src/ragflow_style_pipeline/local_search.py`
- Test: `tests/test_local_search.py`

- [ ] **Step 1: Write failing tests**

Add tests for:

- `load_documents()` loading JSONL rows and respecting `limit`.
- `build_index()` creating a searchable index.
- `search()` returning salary-related document first for query `拖欠工资`.
- `search()` supporting exact metadata filter such as `area_code_area=武进区`.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_local_search
```

Expected: import failure because `local_search.py` does not exist.

- [ ] **Step 3: Implement search index**

Use BM25-style scoring:

```python
idf = log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
score += idf * (term_frequency * (k1 + 1)) / (term_frequency + k1 * (1 - b + b * doc_length / avg_doc_length))
```

Default parameters:

- `k1=1.5`
- `b=0.75`

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_local_search
```

Expected: all local search tests pass.

### Task 3: CLI

**Files:**
- Create: `src/ragflow_style_pipeline/search_jsonl.py`
- Test: `tests/test_search_jsonl.py`

- [ ] **Step 1: Write failing tests**

Add a CLI test that:

- creates a small temporary JSONL file.
- runs `main([...])` with `--query 拖欠工资 --top-k 1`.
- checks that stdout contains `Rank 1`.
- checks that optional `--output` writes JSON results.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
PYTHONPATH=src python3 -m unittest tests.test_search_jsonl
```

Expected: import failure because `search_jsonl.py` does not exist.

- [ ] **Step 3: Implement CLI**

Support:

```text
--input
--query
--top-k
--limit
--area
--type1
--type2
--type3
--month
--output
```

- [ ] **Step 4: Run all tests**

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

Expected: all tests pass.

### Task 4: Demo verification

Run 1000 sample:

```bash
PYTHONPATH=src python3 -m ragflow_style_pipeline.search_jsonl --input outputs/t_order_master.sample.jsonl --query "武进区 夜间 摆摊 扰民" --top-k 5
```

Run 100k sample:

```bash
PYTHONPATH=src python3 -m ragflow_style_pipeline.search_jsonl --input outputs/t_order_master.100k.jsonl --query "拖欠工资 工地 工资 未发" --top-k 5
```

Run filtered search:

```bash
PYTHONPATH=src python3 -m ragflow_style_pipeline.search_jsonl --input outputs/t_order_master.100k.jsonl --query "占道经营 摆摊" --area 武进区 --top-k 5
```

Expected: each command prints ranked results with score, doc id, metadata summary, and a short snippet.

