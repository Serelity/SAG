# 12345 Work Order Document Views and Semantic Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从原始 `t_order_master.tsv` 重新构建适合 RAG 的“多视图工单文档”，并实现第一类能力：先用 `case_content` 语义检索命中相关工单，再对命中集合做结构化统计。

**Architecture:** 一条工单不再只有一个 `text` 字段，而是拆成 `case_content_clean`、`case_goal_clean`、`embedding_text`、`display_text`、`metadata`、`derived`。`embedding_text` 用于 BGE-M3 向量检索，`metadata` 用于过滤和统计，`display_text` 用于结果展示和后续 LLM 阅读。第一版只做“语义检索后统计”，不做“从正文抽结构化标签”和“不做相似正文聚类”。

**Tech Stack:** Python 3.11/3.12, JSONL, NumPy, DuckDB, FlagEmbedding BGE-M3, Bash, unittest.

---

## 0. 本计划的范围

本次只实现第一种：

```text
用户主题查询
  -> BGE-M3 编码查询
  -> 用 case_content/case_goal 向量召回 TopN 工单
  -> 用 metadata 做时间、区域、分类统计
  -> 输出 JSON 统计报告
```

本次不实现：

```text
从 case_content 抽结构化标签
相似 case_content 聚类
LLM 自然语言解析查询条件
LLM 自动总结统计结果
```

这几个能力后面可以继续加，但不能塞进第一版，否则容易把数据地基做乱。

---

## 1. 文件结构

### 新增文件

- `src/ragflow_style_pipeline/embedding_text.py`
  - 负责从文档中取出真正送进 embedding 模型的文本。
  - 优先读取新字段 `embedding_text`。
  - 兼容旧 JSONL：如果没有 `embedding_text`，从旧 `text` 中抽取 `诉求内容` 和 `诉求目标`。

- `src/ragflow_style_pipeline/analysis_db.py`
  - 负责把多视图 JSONL 转成 DuckDB 分析数据库。
  - DuckDB 只保存结构化字段和展示文本，不保存大向量。

- `src/ragflow_style_pipeline/topic_analysis.py`
  - 负责执行“语义检索后统计”。
  - 输入：JSON 配置、文档 JSONL、向量 `.npy`、向量 sidecar `.jsonl`、BGE-M3 模型路径。
  - 输出：统计分析 JSON。

- `configs/topic_analysis_stall.json`
  - 第一版示例配置。
  - 用来模拟未来 LLM 解析后的结构化查询。

- `scripts/export_multiview_100k.sh`
  - 服务器脚本：从原始 TSV 导出 10 万条多视图文档。

- `scripts/build_analysis_db_100k.sh`
  - 服务器脚本：把 10 万条多视图文档写入 DuckDB。

- `scripts/embed_bge_m3_100k_multiview.sh`
  - 服务器脚本：对多视图文档生成 BGE-M3 embedding。

- `scripts/analyze_topic_stall_100k.sh`
  - 服务器脚本：运行“流动摆摊”专题语义检索后统计。

- `docs/10-工单文档表示层与语义统计.md`
  - 中文学习笔记。

- `tests/test_embedding_text.py`
  - 测试 embedding 文本选择逻辑。

- `tests/test_analysis_db.py`
  - 测试 JSONL 到 DuckDB 的结构化入库。

- `tests/test_topic_analysis.py`
  - 测试语义检索结果的过滤和统计。

### 修改文件

- `src/ragflow_style_pipeline/document_builder.py`
  - 增加多视图字段。
  - 保留旧字段 `text`，让旧检索脚本仍能工作。

- `src/ragflow_style_pipeline/export_jsonl.py`
  - 保持 CLI 不变。
  - 因为 `build_document()` 输出变丰富，导出的 JSONL 自动升级为多视图格式。

- `src/ragflow_style_pipeline/bge_m3_embed.py`
  - 如果当前代码库没有该文件，从服务器包同步回来。
  - 写 sidecar 时保留 `display_text`、`embedding_text` 和 `metadata`。

- `src/ragflow_style_pipeline/vector_search.py`
  - 如果当前代码库没有该文件，从服务器包同步回来。
  - 搜索结果优先展示 `display_text`，兼容旧 `text`。

- `requirements.embedding.txt`
  - 增加 `duckdb`。

- `.gitignore`
  - 确认忽略 `.duckdb`、`.npy`、大型输出文件。

---

## Task 1: 多视图文档结构

**Files:**

- Modify: `src/ragflow_style_pipeline/document_builder.py`
- Test: `tests/test_document_builder.py`

### 设计目标

一条工单输出为：

```json
{
  "doc_id": "order_xxx",
  "case_content_clean": "用户诉求正文",
  "case_goal_clean": "用户诉求目标",
  "embedding_text": "诉求内容：...\n诉求目标：...",
  "display_text": "诉求类型：...\n诉求内容：...\n诉求目标：...\n业务分类：...\n所属区域：...\n来电时间：...\n来源渠道：...",
  "text": "兼容旧代码，内容等于 display_text",
  "metadata": {},
  "derived": {
    "topic_tags": [],
    "keywords": [],
    "semantic_cluster_id": "",
    "problem_object": "",
    "problem_behavior": "",
    "location_mention": "",
    "appeal_action": ""
  }
}
```

`derived` 第一版只放空结构，表示后面可以扩展；本次不抽标签。

- [ ] **Step 1: 先写失败测试**

修改 `tests/test_document_builder.py`，在 `TestDocumentBuilder` 类里增加这个测试：

```python
    def test_builds_multiview_document_fields(self):
        phone = "138" + "0013" + "8000"
        row = {
            "id": "10",
            "order_id": "ORD010",
            "service_object_type": "投诉举报",
            "case_content": "市民反映手机号" + phone + "附近有流动摊贩占道经营",
            "case_goal": "希望执法部门清理流动摊贩",
            "area_code_city": "常州市",
            "area_code_area": "武进区",
            "area_code_street": "丁堰街道",
            "case_accord_type_one_name": "城乡建设",
            "case_accord_type_two_name": "市容管理",
            "case_accord_type_three_name": "无照经营游商",
            "order_source": "互联网",
            "order_type": "个人",
            "order_status": "25",
            "call_time": "2024-06-11 20:51:18",
        }

        doc, counts = build_document(row)

        self.assertEqual(
            doc["case_content_clean"],
            "市民反映手机号[手机号]附近有流动摊贩占道经营",
        )
        self.assertEqual(doc["case_goal_clean"], "希望执法部门清理流动摊贩")
        self.assertEqual(
            doc["embedding_text"],
            (
                "诉求内容：市民反映手机号[手机号]附近有流动摊贩占道经营\n"
                "诉求目标：希望执法部门清理流动摊贩"
            ),
        )
        self.assertIn("业务分类：城乡建设 / 市容管理 / 无照经营游商", doc["display_text"])
        self.assertIn("所属区域：常州市 / 武进区 / 丁堰街道", doc["display_text"])
        self.assertEqual(doc["text"], doc["display_text"])
        self.assertEqual(doc["metadata"]["area_code_area"], "武进区")
        self.assertEqual(doc["metadata"]["call_month"], "2024-06")
        self.assertEqual(doc["derived"]["topic_tags"], [])
        self.assertEqual(doc["derived"]["semantic_cluster_id"], "")
        self.assertEqual(counts["phone"], 1)
```

- [ ] **Step 2: 运行测试，确认失败**

在 Windows PowerShell 运行：

```powershell
wsl -d RAGFlow-Ubuntu --cd /home/wx/projects/ragflow-learning-plan -- env PYTHONPATH=src PYTHONIOENCODING=utf-8 python3 -m unittest tests.test_document_builder
```

命令含义：

- `wsl -d RAGFlow-Ubuntu`：进入你的 Ubuntu 开发环境。
- `--cd /home/wx/projects/ragflow-learning-plan`：进入项目目录。
- `PYTHONPATH=src`：让 Python 能找到 `src/ragflow_style_pipeline`。
- `PYTHONIOENCODING=utf-8`：避免中文输出乱码。
- `python3 -m unittest tests.test_document_builder`：运行文档构建测试。

预期结果：

```text
FAILED
KeyError: 'case_content_clean'
```

- [ ] **Step 3: 实现多视图字段**

修改 `src/ragflow_style_pipeline/document_builder.py`。

保留文件顶部已有 import 和工具函数，增加以下函数：

```python
def build_case_content(row):
    """Return the normalized raw complaint/request content."""
    return clean_value(row.get("case_content"))


def build_case_goal(row):
    """Return the normalized request goal."""
    return clean_value(row.get("case_goal"))


def build_embedding_text_from_parts(case_content, case_goal):
    """Build the text that dense embedding models should encode."""
    lines = []
    if case_content:
        lines.append(f"诉求内容：{case_content}")
    if case_goal:
        lines.append(f"诉求目标：{case_goal}")
    return "\n".join(lines)


def build_display_text(row):
    """Create the text shown to humans and later provided to LLM context."""
    lines = []

    service_object_type = clean_value(row.get("service_object_type"))
    case_content = build_case_content(row)
    case_goal = build_case_goal(row)

    if service_object_type:
        lines.append(f"诉求类型：{service_object_type}")
    if case_content:
        lines.append(f"诉求内容：{case_content}")
    if case_goal:
        lines.append(f"诉求目标：{case_goal}")

    category = _join_non_empty(
        [
            clean_value(row.get("case_accord_type_one_name")),
            clean_value(row.get("case_accord_type_two_name")),
            clean_value(row.get("case_accord_type_three_name")),
        ],
        " / ",
    )
    if category:
        lines.append(f"业务分类：{category}")

    area = _join_non_empty(
        [
            clean_value(row.get("area_code_city")),
            clean_value(row.get("area_code_area")),
            clean_value(row.get("area_code_street")),
        ],
        " / ",
    )
    if area:
        lines.append(f"所属区域：{area}")

    call_time = clean_value(row.get("call_time"))
    if call_time:
        lines.append(f"来电时间：{call_time}")

    order_source = clean_value(row.get("order_source"))
    if order_source:
        lines.append(f"来源渠道：{order_source}")

    return "\n".join(lines)


def build_derived():
    """Return reserved derived fields for future text analytics."""
    return {
        "topic_tags": [],
        "keywords": [],
        "semantic_cluster_id": "",
        "problem_object": "",
        "problem_behavior": "",
        "location_mention": "",
        "appeal_action": "",
    }
```

把旧的 `build_text(row)` 改成兼容包装：

```python
def build_text(row):
    """Backward-compatible full display text for one order row."""
    return build_display_text(row)
```

把 `build_document(row)` 替换为：

```python
def build_document(row):
    """Build one JSONL-ready multi-view RAG document and redaction statistics."""
    counts = Counter()

    case_content, case_content_counts = redact_text(build_case_content(row))
    counts.update(case_content_counts)

    case_goal, case_goal_counts = redact_text(build_case_goal(row))
    counts.update(case_goal_counts)

    display_text, display_counts = redact_text(build_display_text(row))
    counts.update(display_counts)

    embedding_text = build_embedding_text_from_parts(case_content, case_goal)

    metadata = build_metadata(row)
    redacted_metadata = {}
    for key, value in metadata.items():
        redacted_value, value_counts = redact_text(value)
        redacted_metadata[key] = redacted_value
        counts.update(value_counts)

    return (
        {
            "doc_id": build_doc_id(row),
            "case_content_clean": case_content,
            "case_goal_clean": case_goal,
            "embedding_text": embedding_text,
            "display_text": display_text,
            "text": display_text,
            "metadata": redacted_metadata,
            "derived": build_derived(),
        },
        counts,
    )
```

注意：这里 `display_text` 会再次对正文脱敏，所以 `case_content_clean` 和 `display_text` 都不会暴露手机号。

- [ ] **Step 4: 运行测试，确认通过**

```powershell
wsl -d RAGFlow-Ubuntu --cd /home/wx/projects/ragflow-learning-plan -- env PYTHONPATH=src PYTHONIOENCODING=utf-8 python3 -m unittest tests.test_document_builder
```

预期结果：

```text
Ran 4 tests
OK
```

- [ ] **Step 5: 提交**

```bash
git add src/ragflow_style_pipeline/document_builder.py tests/test_document_builder.py
git commit -m "feat: add multiview work order documents"
```

---

## Task 2: Embedding 文本选择逻辑

**Files:**

- Create or Modify: `src/ragflow_style_pipeline/embedding_text.py`
- Test: `tests/test_embedding_text.py`

### 设计目标

embedding 模型只看语义主体：

```text
诉求内容
诉求目标
```

默认不看：

```text
区域
时间
来源
分类
状态
```

这些字段进入 metadata，用于过滤和统计。

- [ ] **Step 1: 写失败测试**

创建或替换 `tests/test_embedding_text.py`：

```python
import unittest

from ragflow_style_pipeline.embedding_text import embedding_text


class TestEmbeddingText(unittest.TestCase):
    def test_embedding_text_prefers_new_embedding_text_field(self):
        document = {
            "embedding_text": "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
            "display_text": (
                "诉求内容：流动摊贩占道经营\n"
                "业务分类：城乡建设 / 市容管理 / 无照经营游商\n"
                "所属区域：常州市 / 武进区"
            ),
            "text": "旧字段",
        }

        self.assertEqual(
            embedding_text(document),
            "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
        )

    def test_embedding_text_falls_back_to_old_text_prefixes(self):
        document = {
            "text": (
                "诉求类型：求助\n"
                "诉求内容：服务对象反映房屋漏水。\n"
                "诉求目标：希望维修。\n"
                "业务分类：住房保障 / 物业管理\n"
                "所属区域：常州市 / 新北区"
            ),
            "metadata": {"type2": "物业管理"},
        }

        text = embedding_text(document)

        self.assertIn("诉求内容：服务对象反映房屋漏水。", text)
        self.assertIn("诉求目标：希望维修。", text)
        self.assertNotIn("业务分类", text)
        self.assertNotIn("所属区域", text)

    def test_embedding_text_falls_back_to_full_display_text_when_body_missing(self):
        document = {"display_text": "服务对象反映老板不给工资。", "metadata": {}}

        self.assertEqual(embedding_text(document), "服务对象反映老板不给工资。")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认失败**

```powershell
wsl -d RAGFlow-Ubuntu --cd /home/wx/projects/ragflow-learning-plan -- env PYTHONPATH=src PYTHONIOENCODING=utf-8 python3 -m unittest tests.test_embedding_text
```

预期结果：

```text
FAILED
```

如果本地旧代码库没有 `embedding_text.py`，预期错误是：

```text
ModuleNotFoundError: No module named 'ragflow_style_pipeline.embedding_text'
```

- [ ] **Step 3: 实现 embedding 文本选择**

创建或替换 `src/ragflow_style_pipeline/embedding_text.py`：

```python
"""Build semantic text for embedding models."""

EMBEDDING_PREFIXES = ("诉求内容：", "诉求目标：")


def embedding_text(document):
    """Return case-content-first text for dense embedding."""
    explicit_text = str(document.get("embedding_text", "")).strip()
    if explicit_text:
        return explicit_text

    text = str(document.get("text") or document.get("display_text") or "")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith(EMBEDDING_PREFIXES)
    ]
    if lines:
        return "\n".join(lines)
    return text.strip()
```

- [ ] **Step 4: 运行测试，确认通过**

```powershell
wsl -d RAGFlow-Ubuntu --cd /home/wx/projects/ragflow-learning-plan -- env PYTHONPATH=src PYTHONIOENCODING=utf-8 python3 -m unittest tests.test_embedding_text
```

预期结果：

```text
Ran 3 tests
OK
```

- [ ] **Step 5: 提交**

```bash
git add src/ragflow_style_pipeline/embedding_text.py tests/test_embedding_text.py
git commit -m "feat: prefer explicit embedding text"
```

---

## Task 3: BGE-M3 向量生成兼容多视图文档

**Files:**

- Create or Modify: `src/ragflow_style_pipeline/bge_m3_embed.py`
- Test: `tests/test_bge_m3_embed.py`

### 设计目标

向量 sidecar 不只保存旧的 `text`，还要保存：

```text
embedding_text
display_text
case_content_clean
case_goal_clean
metadata
```

这样后续分析命中工单时，不需要再回读原始 TSV。

- [ ] **Step 1: 写失败测试**

创建或替换 `tests/test_bge_m3_embed.py`：

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ragflow_style_pipeline.bge_m3_embed import write_embedding_outputs


class TestBgeM3Embed(unittest.TestCase):
    def test_write_embedding_outputs_preserves_multiview_sidecar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_path = Path(tmpdir) / "vectors.npy"
            meta_path = Path(tmpdir) / "vectors.meta.jsonl"
            documents = [
                {
                    "doc_id": "order_a",
                    "case_content_clean": "流动摊贩占道经营",
                    "case_goal_clean": "希望清理",
                    "embedding_text": "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
                    "display_text": "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
                    "text": "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
                    "metadata": {"area_code_area": "武进区", "type3": "无照经营游商"},
                }
            ]
            vectors = np.array([[1.0, 0.0]], dtype=np.float32)

            write_embedding_outputs(documents, vectors, vector_path, meta_path)

            saved_vectors = np.load(vector_path)
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8").strip())

            self.assertEqual(saved_vectors.shape, (1, 2))
            self.assertEqual(saved_meta["doc_id"], "order_a")
            self.assertEqual(saved_meta["case_content_clean"], "流动摊贩占道经营")
            self.assertEqual(saved_meta["embedding_text"], "诉求内容：流动摊贩占道经营\n诉求目标：希望清理")
            self.assertEqual(saved_meta["display_text"], "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区")
            self.assertEqual(saved_meta["text"], saved_meta["display_text"])
            self.assertEqual(saved_meta["metadata"]["type3"], "无照经营游商")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认失败**

```powershell
wsl -d RAGFlow-Ubuntu --cd /home/wx/projects/ragflow-learning-plan -- env PYTHONPATH=src PYTHONIOENCODING=utf-8 python3 -m unittest tests.test_bge_m3_embed
```

预期结果：

```text
FAILED
```

如果当前代码库没有 `bge_m3_embed.py`，预期错误是：

```text
ModuleNotFoundError: No module named 'ragflow_style_pipeline.bge_m3_embed'
```

- [ ] **Step 3: 实现或同步 BGE-M3 embedding 脚本**

创建或替换 `src/ragflow_style_pipeline/bge_m3_embed.py`：

```python
"""Generate bge-m3 dense embeddings for exported RAG JSONL documents."""

import argparse
import json
from pathlib import Path

import numpy as np

from ragflow_style_pipeline.embedding_text import embedding_text
from ragflow_style_pipeline.local_search import load_documents


def parse_args(argv=None):
    """Parse bge-m3 embedding CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate bge-m3 dense embeddings.")
    parser.add_argument("--input", required=True, help="Input redacted document JSONL.")
    parser.add_argument("--vectors", required=True, help="Output .npy vector path.")
    parser.add_argument("--meta", required=True, help="Output metadata JSONL sidecar path.")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model name or local path.")
    parser.add_argument("--device", default="cuda", help="Embedding device, such as cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum token length for bge-m3.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum documents to encode.")
    return parser.parse_args(argv)


def normalize_dense_output(output):
    """Return dense vectors from a FlagEmbedding output as float32 NumPy array."""
    return np.asarray(output["dense_vecs"], dtype=np.float32)


def _load_encoder(model_name, device):
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(model_name, use_fp16=device == "cuda", device=device)


def _encode_batch(model, texts, batch_size, max_length):
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return normalize_dense_output(output)


def _sidecar_document(document):
    display_text = str(document.get("display_text") or document.get("text") or "")
    return {
        "doc_id": document["doc_id"],
        "case_content_clean": document.get("case_content_clean", ""),
        "case_goal_clean": document.get("case_goal_clean", ""),
        "embedding_text": document.get("embedding_text", ""),
        "display_text": display_text,
        "text": display_text,
        "metadata": document.get("metadata", {}),
        "derived": document.get("derived", {}),
    }


def write_embedding_outputs(documents, vectors, vector_path, meta_path):
    """Write vectors and matching document sidecar."""
    vector_path = Path(vector_path)
    meta_path = Path(meta_path)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(vector_path, vectors)
    with meta_path.open("w", encoding="utf-8") as output_file:
        for document in documents:
            output_file.write(json.dumps(_sidecar_document(document), ensure_ascii=False) + "\n")


def main(argv=None):
    """Generate and save dense vectors for exported documents."""
    args = parse_args(argv)
    documents = load_documents(args.input, limit=args.limit)
    texts = [embedding_text(document) for document in documents]
    model = _load_encoder(args.model, args.device)
    vectors = _encode_batch(model, texts, args.batch_size, args.max_length)
    write_embedding_outputs(documents, vectors, args.vectors, args.meta)

    print(
        json.dumps(
            {
                "documents": len(documents),
                "model": args.model,
                "device": args.device,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "vectors": args.vectors,
                "meta": args.meta,
                "shape": list(vectors.shape),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试，确认通过**

```powershell
wsl -d RAGFlow-Ubuntu --cd /home/wx/projects/ragflow-learning-plan -- env PYTHONPATH=src PYTHONIOENCODING=utf-8 python3 -m unittest tests.test_bge_m3_embed
```

预期结果：

```text
Ran 1 test
OK
```

- [ ] **Step 5: 提交**

```bash
git add src/ragflow_style_pipeline/bge_m3_embed.py tests/test_bge_m3_embed.py
git commit -m "feat: preserve multiview embedding sidecar"
```

---

## Task 4: 向量检索模块兼容多视图文档

**Files:**

- Create or Modify: `src/ragflow_style_pipeline/vector_search.py`
- Test: `tests/test_vector_search.py`

- [ ] **Step 1: 写失败测试**

创建或替换 `tests/test_vector_search.py`：

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ragflow_style_pipeline.vector_search import load_vector_index, vector_search


class TestVectorSearch(unittest.TestCase):
    def test_vector_search_returns_nearest_multiview_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_path = Path(tmpdir) / "vectors.npy"
            meta_path = Path(tmpdir) / "vectors.meta.jsonl"
            np.save(vector_path, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            meta_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "doc_id": "stall",
                                "display_text": "诉求内容：流动摊贩占道经营",
                                "embedding_text": "诉求内容：流动摊贩占道经营",
                                "case_content_clean": "流动摊贩占道经营",
                                "metadata": {"type3": "无照经营游商", "area_code_area": "武进区"},
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "doc_id": "noise",
                                "display_text": "诉求内容：噪音扰民",
                                "embedding_text": "诉求内容：噪音扰民",
                                "case_content_clean": "噪音扰民",
                                "metadata": {"type3": "社会生活噪声", "area_code_area": "天宁区"},
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            index = load_vector_index(vector_path, meta_path)
            results = vector_search(index, np.array([0.9, 0.1], dtype=np.float32), top_k=1)

            self.assertEqual(results[0]["doc_id"], "stall")
            self.assertEqual(results[0]["retriever"], "vector")
            self.assertEqual(results[0]["text"], "诉求内容：流动摊贩占道经营")
            self.assertEqual(results[0]["case_content_clean"], "流动摊贩占道经营")
            self.assertIn("vector_score", results[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认失败**

```powershell
wsl -d RAGFlow-Ubuntu --cd /home/wx/projects/ragflow-learning-plan -- env PYTHONPATH=src PYTHONIOENCODING=utf-8 python3 -m unittest tests.test_vector_search
```

预期结果：

```text
FAILED
```

- [ ] **Step 3: 实现向量检索模块**

创建或替换 `src/ragflow_style_pipeline/vector_search.py`：

```python
"""Vector search over cached embedding arrays."""

import json
from pathlib import Path

import numpy as np

from ragflow_style_pipeline.local_search import _matches_filters


def _l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_vector_index(vector_path, meta_path):
    """Load embedding vectors and matching JSONL metadata sidecar."""
    vectors = np.load(vector_path).astype(np.float32)
    documents = [
        json.loads(line)
        for line in Path(meta_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(vectors) != len(documents):
        raise ValueError(f"vector/document count mismatch: {len(vectors)} != {len(documents)}")
    return {"vectors": _l2_normalize(vectors), "documents": documents}


def _result_text(document):
    return str(document.get("display_text") or document.get("text") or "")


def vector_search(index, query_vector, top_k=5, filters=None):
    """Search cached vectors with cosine similarity."""
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    query = _l2_normalize(query)[0]
    scores = index["vectors"] @ query
    ranked_indices = np.argsort(-scores)

    results = []
    for document_index in ranked_indices:
        document = index["documents"][int(document_index)]
        if not _matches_filters(document, filters):
            continue
        score = float(scores[int(document_index)])
        results.append(
            {
                "doc_id": document["doc_id"],
                "score": round(score, 6),
                "vector_score": round(score, 6),
                "text": _result_text(document),
                "display_text": _result_text(document),
                "embedding_text": document.get("embedding_text", ""),
                "case_content_clean": document.get("case_content_clean", ""),
                "case_goal_clean": document.get("case_goal_clean", ""),
                "metadata": document.get("metadata", {}),
                "derived": document.get("derived", {}),
                "retriever": "vector",
            }
        )
        if len(results) >= top_k:
            break
    return results
```

- [ ] **Step 4: 运行测试，确认通过**

```powershell
wsl -d RAGFlow-Ubuntu --cd /home/wx/projects/ragflow-learning-plan -- env PYTHONPATH=src PYTHONIOENCODING=utf-8 python3 -m unittest tests.test_vector_search
```

预期结果：

```text
Ran 1 test
OK
```

- [ ] **Step 5: 提交**

```bash
git add src/ragflow_style_pipeline/vector_search.py tests/test_vector_search.py
git commit -m "feat: support multiview vector search"
```

---

## Task 5: 构建 DuckDB 分析数据库

**Files:**

- Create: `src/ragflow_style_pipeline/analysis_db.py`
- Test: `tests/test_analysis_db.py`
- Modify: `requirements.embedding.txt`

### 设计目标

DuckDB 表只保存便于过滤和统计的内容：

```text
doc_id
case_content_clean
case_goal_clean
embedding_text
display_text
call_time
call_month
area_code_city
area_code_area
area_code_street
type1
type2
type3
order_source
order_type
order_status
service_object_type
```

向量仍然保存为 `.npy`，不塞进 DuckDB。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_analysis_db.py`：

```python
import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from ragflow_style_pipeline.analysis_db import build_analysis_db


class TestAnalysisDb(unittest.TestCase):
    def test_build_analysis_db_loads_multiview_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "orders.jsonl"
            db_path = Path(tmpdir) / "orders.duckdb"
            documents = [
                {
                    "doc_id": "order_a",
                    "case_content_clean": "流动摊贩占道经营",
                    "case_goal_clean": "希望清理",
                    "embedding_text": "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
                    "display_text": "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
                    "metadata": {
                        "call_time": "2024-06-11 20:51:18",
                        "call_month": "2024-06",
                        "area_code_city": "常州市",
                        "area_code_area": "武进区",
                        "area_code_street": "丁堰街道",
                        "type1": "城乡建设",
                        "type2": "市容管理",
                        "type3": "无照经营游商",
                        "order_source": "互联网",
                        "order_type": "个人",
                        "order_status": "25",
                        "service_object_type": "投诉举报",
                    },
                }
            ]
            jsonl_path.write_text(
                "\n".join(json.dumps(document, ensure_ascii=False) for document in documents) + "\n",
                encoding="utf-8",
            )

            report = build_analysis_db(jsonl_path, db_path)

            self.assertEqual(report["documents_loaded"], 1)
            with duckdb.connect(str(db_path), read_only=True) as conn:
                rows = conn.execute(
                    "select doc_id, area_code_area, area_code_street, type3, call_month from work_orders"
                ).fetchall()
            self.assertEqual(rows, [("order_a", "武进区", "丁堰街道", "无照经营游商", "2024-06")])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 安装 DuckDB 后运行测试，确认失败**

服务器或 WSL 环境运行：

```bash
pip install duckdb
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_analysis_db
```

命令含义：

- `pip install duckdb`：安装本地分析数据库依赖。
- `PYTHONPATH=src`：让 Python 找到项目源码。
- `python -m unittest tests.test_analysis_db`：运行数据库构建测试。

预期结果：

```text
FAILED
ModuleNotFoundError: No module named 'ragflow_style_pipeline.analysis_db'
```

- [ ] **Step 3: 实现 DuckDB 构建模块**

创建 `src/ragflow_style_pipeline/analysis_db.py`：

```python
"""Build a DuckDB analysis database from multi-view work-order JSONL."""

import argparse
import json
from pathlib import Path

import duckdb


WORK_ORDER_COLUMNS = [
    "doc_id",
    "case_content_clean",
    "case_goal_clean",
    "embedding_text",
    "display_text",
    "call_time",
    "call_month",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "type1",
    "type2",
    "type3",
    "order_source",
    "order_type",
    "order_status",
    "service_object_type",
]


def _document_row(document):
    metadata = document.get("metadata", {})
    display_text = str(document.get("display_text") or document.get("text") or "")
    return {
        "doc_id": document.get("doc_id", ""),
        "case_content_clean": document.get("case_content_clean", ""),
        "case_goal_clean": document.get("case_goal_clean", ""),
        "embedding_text": document.get("embedding_text", ""),
        "display_text": display_text,
        "call_time": metadata.get("call_time", ""),
        "call_month": metadata.get("call_month", ""),
        "area_code_city": metadata.get("area_code_city", ""),
        "area_code_area": metadata.get("area_code_area", ""),
        "area_code_street": metadata.get("area_code_street", ""),
        "type1": metadata.get("type1", ""),
        "type2": metadata.get("type2", ""),
        "type3": metadata.get("type3", ""),
        "order_source": metadata.get("order_source", ""),
        "order_type": metadata.get("order_type", ""),
        "order_status": metadata.get("order_status", ""),
        "service_object_type": metadata.get("service_object_type", ""),
    }


def _read_rows(jsonl_path):
    rows = []
    with Path(jsonl_path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                rows.append(_document_row(json.loads(line)))
    return rows


def build_analysis_db(jsonl_path, db_path):
    """Create or replace the work_orders table from a JSONL document file."""
    rows = _read_rows(jsonl_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as conn:
        conn.execute("drop table if exists work_orders")
        column_sql = ", ".join(f"{column} varchar" for column in WORK_ORDER_COLUMNS)
        conn.execute(f"create table work_orders ({column_sql})")
        if rows:
            placeholders = ", ".join(["?"] * len(WORK_ORDER_COLUMNS))
            conn.executemany(
                f"insert into work_orders values ({placeholders})",
                [[row[column] for column in WORK_ORDER_COLUMNS] for row in rows],
            )
        conn.execute("create index if not exists idx_work_orders_doc_id on work_orders(doc_id)")
        conn.execute("create index if not exists idx_work_orders_call_month on work_orders(call_month)")
        conn.execute("create index if not exists idx_work_orders_area on work_orders(area_code_area)")
        conn.execute("create index if not exists idx_work_orders_street on work_orders(area_code_street)")
        conn.execute("create index if not exists idx_work_orders_type3 on work_orders(type3)")

    return {"jsonl": str(jsonl_path), "db": str(db_path), "documents_loaded": len(rows)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build DuckDB analysis database from work-order JSONL.")
    parser.add_argument("--input", required=True, help="Input multi-view document JSONL.")
    parser.add_argument("--db", required=True, help="Output DuckDB database path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_analysis_db(args.input, args.db)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 更新依赖文件**

如果 `requirements.embedding.txt` 不存在，创建它。确保包含：

```text
FlagEmbedding
torch
numpy
duckdb
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_analysis_db
```

预期结果：

```text
Ran 1 test
OK
```

- [ ] **Step 6: 提交**

```bash
git add src/ragflow_style_pipeline/analysis_db.py tests/test_analysis_db.py requirements.embedding.txt
git commit -m "feat: add work order analysis database"
```

---

## Task 6: 语义检索后统计

**Files:**

- Create: `src/ragflow_style_pipeline/topic_analysis.py`
- Test: `tests/test_topic_analysis.py`
- Create: `configs/topic_analysis_stall.json`

### 设计目标

第一版配置文件模拟未来 LLM 解析结果：

```json
{
  "query": "流动摆摊 占道经营 游商摊贩",
  "top_n": 1000,
  "score_threshold": 0.0,
  "filters": {
    "call_month_gte": "2024-01",
    "call_month_lte": "2024-12",
    "area_code_area_in": []
  }
}
```

`area_code_area_in` 为空表示不限制区域。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_topic_analysis.py`：

```python
import unittest

from ragflow_style_pipeline.topic_analysis import aggregate_results, apply_result_filters


class TestTopicAnalysis(unittest.TestCase):
    def test_apply_result_filters_keeps_month_range_and_area(self):
        results = [
            {
                "doc_id": "a",
                "score": 0.91,
                "metadata": {
                    "call_month": "2024-06",
                    "area_code_area": "武进区",
                    "area_code_street": "丁堰街道",
                    "type3": "无照经营游商",
                },
            },
            {
                "doc_id": "b",
                "score": 0.88,
                "metadata": {
                    "call_month": "2023-12",
                    "area_code_area": "武进区",
                    "area_code_street": "湖塘镇",
                    "type3": "无照经营游商",
                },
            },
            {
                "doc_id": "c",
                "score": 0.87,
                "metadata": {
                    "call_month": "2024-07",
                    "area_code_area": "天宁区",
                    "area_code_street": "茶山街道",
                    "type3": "社会生活噪声",
                },
            },
        ]

        filtered = apply_result_filters(
            results,
            {
                "call_month_gte": "2024-01",
                "call_month_lte": "2024-12",
                "area_code_area_in": ["武进区"],
            },
        )

        self.assertEqual([result["doc_id"] for result in filtered], ["a"])

    def test_aggregate_results_counts_month_area_street_and_type3(self):
        results = [
            {
                "doc_id": "a",
                "score": 0.91,
                "case_content_clean": "流动摊贩占道经营",
                "text": "诉求内容：流动摊贩占道经营",
                "metadata": {
                    "call_month": "2024-06",
                    "area_code_area": "武进区",
                    "area_code_street": "丁堰街道",
                    "type3": "无照经营游商",
                },
            },
            {
                "doc_id": "b",
                "score": 0.88,
                "case_content_clean": "夜市摊贩扰民",
                "text": "诉求内容：夜市摊贩扰民",
                "metadata": {
                    "call_month": "2024-06",
                    "area_code_area": "武进区",
                    "area_code_street": "丁堰街道",
                    "type3": "无照经营游商",
                },
            },
            {
                "doc_id": "c",
                "score": 0.80,
                "case_content_clean": "学校门口摆摊",
                "text": "诉求内容：学校门口摆摊",
                "metadata": {
                    "call_month": "2024-07",
                    "area_code_area": "天宁区",
                    "area_code_street": "茶山街道",
                    "type3": "无照经营游商",
                },
            },
        ]

        report = aggregate_results(
            query="流动摆摊",
            filters={"call_month_gte": "2024-01", "call_month_lte": "2024-12"},
            results=results,
            representative_limit=2,
        )

        self.assertEqual(report["matched_orders"], 3)
        self.assertEqual(report["statistics"]["by_month"][0], {"value": "2024-06", "count": 2})
        self.assertEqual(report["statistics"]["by_area"][0], {"value": "武进区", "count": 2})
        self.assertEqual(report["statistics"]["by_street"][0], {"value": "丁堰街道", "count": 2})
        self.assertEqual(report["statistics"]["by_type3"][0], {"value": "无照经营游商", "count": 3})
        self.assertEqual(len(report["representative_cases"]), 2)
        self.assertEqual(report["representative_cases"][0]["doc_id"], "a")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_topic_analysis
```

预期结果：

```text
FAILED
ModuleNotFoundError: No module named 'ragflow_style_pipeline.topic_analysis'
```

- [ ] **Step 3: 实现统计模块**

创建 `src/ragflow_style_pipeline/topic_analysis.py`：

```python
"""Semantic retrieval followed by structured statistics for work-order topics."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ragflow_style_pipeline.bge_m3_embed import _encode_batch, _load_encoder
from ragflow_style_pipeline.vector_search import load_vector_index, vector_search


COUNT_FIELDS = {
    "by_month": "call_month",
    "by_area": "area_code_area",
    "by_street": "area_code_street",
    "by_type3": "type3",
}


def load_config(config_path):
    """Load a topic analysis config JSON file."""
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _in_month_range(month, filters):
    if not month:
        return False
    month_gte = filters.get("call_month_gte", "")
    month_lte = filters.get("call_month_lte", "")
    if month_gte and month < month_gte:
        return False
    if month_lte and month > month_lte:
        return False
    return True


def _in_allowed_values(value, allowed_values):
    if not allowed_values:
        return True
    return value in allowed_values


def apply_result_filters(results, filters):
    """Apply structured metadata filters to vector search results."""
    filters = filters or {}
    area_values = filters.get("area_code_area_in") or []
    street_values = filters.get("area_code_street_in") or []
    type3_values = filters.get("type3_in") or []
    score_threshold = float(filters.get("score_threshold", 0.0))

    filtered = []
    for result in results:
        metadata = result.get("metadata", {})
        if float(result.get("score", 0.0)) < score_threshold:
            continue
        if not _in_month_range(str(metadata.get("call_month", "")), filters):
            continue
        if not _in_allowed_values(str(metadata.get("area_code_area", "")), area_values):
            continue
        if not _in_allowed_values(str(metadata.get("area_code_street", "")), street_values):
            continue
        if not _in_allowed_values(str(metadata.get("type3", "")), type3_values):
            continue
        filtered.append(result)
    return filtered


def _counter_items(results, metadata_field):
    counter = Counter()
    for result in results:
        value = str(result.get("metadata", {}).get(metadata_field, "")).strip()
        if value:
            counter[value] += 1
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _representative_cases(results, limit):
    cases = []
    for result in results[:limit]:
        metadata = result.get("metadata", {})
        cases.append(
            {
                "doc_id": result.get("doc_id", ""),
                "score": result.get("score", 0.0),
                "call_month": metadata.get("call_month", ""),
                "area": metadata.get("area_code_area", ""),
                "street": metadata.get("area_code_street", ""),
                "type3": metadata.get("type3", ""),
                "case_content": result.get("case_content_clean", ""),
                "text": result.get("text", ""),
            }
        )
    return cases


def aggregate_results(query, filters, results, representative_limit=10):
    """Aggregate semantic retrieval results by metadata fields."""
    return {
        "query": query,
        "filters": filters or {},
        "matched_orders": len(results),
        "statistics": {
            output_key: _counter_items(results, metadata_field)
            for output_key, metadata_field in COUNT_FIELDS.items()
        },
        "representative_cases": _representative_cases(results, representative_limit),
    }


def encode_query(model_path, device, query, batch_size=1, max_length=1024):
    """Encode one query with BGE-M3 and return a dense vector."""
    model = _load_encoder(model_path, device)
    vectors = _encode_batch(model, [query], batch_size=batch_size, max_length=max_length)
    return np.asarray(vectors[0], dtype=np.float32)


def analyze_topic(config, vector_path, meta_path, model_path, device):
    """Run semantic retrieval and aggregate the matched results."""
    query = config["query"]
    top_n = int(config.get("top_n", 1000))
    representative_limit = int(config.get("representative_limit", 10))
    filters = dict(config.get("filters", {}))
    if "score_threshold" in config:
        filters["score_threshold"] = config["score_threshold"]

    index = load_vector_index(vector_path, meta_path)
    query_vector = encode_query(model_path, device, query)
    raw_results = vector_search(index, query_vector, top_k=top_n, filters=None)
    filtered_results = apply_result_filters(raw_results, filters)
    report = aggregate_results(query, filters, filtered_results, representative_limit)
    report["retrieval"] = {
        "top_n": top_n,
        "raw_retrieved": len(raw_results),
        "after_filters": len(filtered_results),
        "vector_path": str(vector_path),
        "meta_path": str(meta_path),
        "model_path": str(model_path),
        "device": device,
    }
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a work-order topic after semantic retrieval.")
    parser.add_argument("--config", required=True, help="Topic analysis config JSON.")
    parser.add_argument("--vectors", required=True, help="Embedding .npy vector path.")
    parser.add_argument("--meta", required=True, help="Embedding sidecar JSONL.")
    parser.add_argument("--model", default=".cache/models/BAAI/bge-m3", help="BGE-M3 model path.")
    parser.add_argument("--device", default="cuda", help="Embedding device.")
    parser.add_argument("--output", required=True, help="Output analysis JSON.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    report = analyze_topic(config, args.vectors, args.meta, args.model, args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 写示例配置**

创建 `configs/topic_analysis_stall.json`：

```json
{
  "query": "流动摆摊 占道经营 游商摊贩 夜市摊贩 无照经营",
  "top_n": 1000,
  "score_threshold": 0.0,
  "representative_limit": 10,
  "filters": {
    "call_month_gte": "2024-01",
    "call_month_lte": "2024-12",
    "area_code_area_in": [],
    "area_code_street_in": [],
    "type3_in": []
  }
}
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m unittest tests.test_topic_analysis
```

预期结果：

```text
Ran 2 tests
OK
```

- [ ] **Step 6: 提交**

```bash
git add src/ragflow_style_pipeline/topic_analysis.py tests/test_topic_analysis.py configs/topic_analysis_stall.json
git commit -m "feat: add semantic topic statistics"
```

---

## Task 7: 服务器运行脚本

**Files:**

- Create: `scripts/export_multiview_100k.sh`
- Create: `scripts/build_analysis_db_100k.sh`
- Create: `scripts/embed_bge_m3_100k_multiview.sh`
- Create: `scripts/analyze_topic_stall_100k.sh`

- [ ] **Step 1: 写导出 10 万条多视图文档脚本**

创建 `scripts/export_multiview_100k.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_TSV="${INPUT_TSV:-data/t_order_master.tsv}"
OUTPUT_JSONL="${OUTPUT_JSONL:-outputs/t_order_master.100k.multiview.jsonl}"
QUALITY_REPORT="${QUALITY_REPORT:-outputs/t_order_master.100k.multiview.quality.json}"
LIMIT="${LIMIT:-100000}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.export_jsonl \
  --input "${INPUT_TSV}" \
  --output "${OUTPUT_JSONL}" \
  --quality-report "${QUALITY_REPORT}" \
  --limit "${LIMIT}"
```

命令含义：

- `INPUT_TSV`：原始 TSV 文件路径，默认 `data/t_order_master.tsv`。
- `OUTPUT_JSONL`：导出的多视图 JSONL。
- `QUALITY_REPORT`：质量报告。
- `LIMIT=100000`：先导出 10 万行，避免一开始处理全量 98 万行。

- [ ] **Step 2: 写 DuckDB 构建脚本**

创建 `scripts/build_analysis_db_100k.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-outputs/t_order_master.100k.multiview.jsonl}"
OUTPUT_DB="${OUTPUT_DB:-outputs/work_orders.100k.duckdb}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.analysis_db \
  --input "${INPUT_JSONL}" \
  --db "${OUTPUT_DB}"
```

命令含义：

- `INPUT_JSONL`：多视图文档。
- `OUTPUT_DB`：DuckDB 分析数据库文件。

- [ ] **Step 3: 写 BGE-M3 向量生成脚本**

创建 `scripts/embed_bge_m3_100k_multiview.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${INPUT_JSONL:-outputs/t_order_master.100k.multiview.jsonl}"
VECTOR_OUTPUT="${VECTOR_OUTPUT:-outputs/embeddings.100k.multiview.bge-m3.npy}"
META_OUTPUT="${META_OUTPUT:-outputs/embeddings.100k.multiview.bge-m3.meta.jsonl}"
BGE_M3_MODEL="${BGE_M3_MODEL:-.cache/models/BAAI/bge-m3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.bge_m3_embed \
  --input "${INPUT_JSONL}" \
  --vectors "${VECTOR_OUTPUT}" \
  --meta "${META_OUTPUT}" \
  --model "${BGE_M3_MODEL}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}"
```

命令含义：

- `BGE_M3_MODEL`：服务器本地模型路径。你现在能用的是 `.cache/models/BAAI/bge-m3`。
- `DEVICE=cuda`：使用 NVIDIA GPU。
- `BATCH_SIZE=4`：V100 32GB 可以后面调大，但第一版先稳。

- [ ] **Step 4: 写专题分析脚本**

创建 `scripts/analyze_topic_stall_100k.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/topic_analysis_stall.json}"
VECTOR_INPUT="${VECTOR_INPUT:-outputs/embeddings.100k.multiview.bge-m3.npy}"
META_INPUT="${META_INPUT:-outputs/embeddings.100k.multiview.bge-m3.meta.jsonl}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/topic_analysis.stall.100k.json}"
BGE_M3_MODEL="${BGE_M3_MODEL:-.cache/models/BAAI/bge-m3}"
DEVICE="${DEVICE:-cuda}"

mkdir -p outputs

PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m ragflow_style_pipeline.topic_analysis \
  --config "${CONFIG}" \
  --vectors "${VECTOR_INPUT}" \
  --meta "${META_INPUT}" \
  --model "${BGE_M3_MODEL}" \
  --device "${DEVICE}" \
  --output "${OUTPUT_JSON}"
```

命令含义：

- `CONFIG`：专题分析配置，第一版是流动摆摊。
- `VECTOR_INPUT`：10 万条多视图文档向量。
- `META_INPUT`：向量对应的工单 sidecar。
- `OUTPUT_JSON`：统计分析结果。

- [ ] **Step 5: 给脚本加执行权限**

在服务器运行：

```bash
chmod +x scripts/export_multiview_100k.sh scripts/build_analysis_db_100k.sh scripts/embed_bge_m3_100k_multiview.sh scripts/analyze_topic_stall_100k.sh
```

命令含义：

- `chmod +x`：让 `.sh` 文件可以直接作为脚本执行。

- [ ] **Step 6: 提交**

```bash
git add scripts/export_multiview_100k.sh scripts/build_analysis_db_100k.sh scripts/embed_bge_m3_100k_multiview.sh scripts/analyze_topic_stall_100k.sh
git commit -m "feat: add server scripts for semantic statistics"
```

---

## Task 8: 中文学习笔记

**Files:**

- Create: `docs/10-工单文档表示层与语义统计.md`

- [ ] **Step 1: 创建笔记**

创建 `docs/10-工单文档表示层与语义统计.md`：

```markdown
# 10 - 工单文档表示层与语义统计

## 这一阶段解决什么问题

前面已经证明 BGE-M3 对 12345 工单检索有效。下一步不能只继续调 embedding，而要回到源头：一条原始工单应该如何变成适合 RAG 的文档。

## 为什么不能一个 text 字段走到底

如果把诉求内容、地区、时间、分类、来源全部拼进一个 text，embedding 模型会同时学习正文语义和 metadata 信息。

这会带来问题：

- 相同地区可能压过真实问题语义。
- 相同分类可能压过用户诉求正文。
- 检索看起来相关，但实际 case_content 不一定最匹配。
- 后续统计时难以区分“正文命中”和“字段命中”。

## 多视图文档

第一版把一条工单拆成：

```text
case_content_clean：脱敏后的用户诉求正文
case_goal_clean：脱敏后的诉求目标
embedding_text：给 BGE-M3 编码的文本
display_text：给人和 LLM 看的完整上下文
metadata：时间、区域、分类、来源、状态等结构化字段
derived：未来从正文派生出来的标签和聚类字段
```

## 字段职责

| 字段 | 主要用途 |
| --- | --- |
| case_content_clean | 核心语义、相似问题判断 |
| case_goal_clean | 诉求目的补充 |
| embedding_text | 向量检索 |
| display_text | 展示、LLM 上下文 |
| metadata | 过滤、统计、分组 |
| derived | 后续正文标签、关键词、聚类 |

## case_content 怎么参与统计

case_content 是非结构化文本，不能直接像区县、月份一样 group by。

第一版采用：

```text
case_content/case_goal 语义检索
  -> 得到某个问题相关的工单集合
  -> 对这个集合按 metadata 统计
```

也就是说，case_content 负责定义“哪些工单属于这个问题”，metadata 负责统计“这些工单在时间、区域、分类上怎么分布”。

## 这次先做什么

本次只做第一种能力：

```text
语义检索后统计
```

暂时不做：

```text
从 case_content 抽结构化标签
相似 case_content 聚类
LLM 自动解析自然语言查询
```

## 服务器运行顺序

```bash
bash scripts/export_multiview_100k.sh
bash scripts/build_analysis_db_100k.sh
bash scripts/embed_bge_m3_100k_multiview.sh
bash scripts/analyze_topic_stall_100k.sh
```

这四步的含义：

1. 从原始 TSV 导出 10 万条多视图工单文档。
2. 把文档写入 DuckDB，方便结构化统计。
3. 用 BGE-M3 对 embedding_text 生成向量。
4. 对“流动摆摊”做语义检索后统计。
```

- [ ] **Step 2: 提交**

```bash
git add docs/10-工单文档表示层与语义统计.md
git commit -m "docs: explain work order document views"
```

---

## Task 9: 服务器端验证流程

**Files:**

- No code files.

- [ ] **Step 1: 准备原始数据**

服务器项目目录里放：

```text
data/t_order_master.tsv
```

如果没有 `data` 目录，运行：

```bash
mkdir -p data
```

命令含义：

- `mkdir -p data`：创建 `data` 目录；如果目录已经存在，不报错。

- [ ] **Step 2: 确认模型目录存在**

运行：

```bash
ls -lh .cache/models/BAAI/bge-m3
```

命令含义：

- `ls -lh`：查看目录内容和文件大小。
- `.cache/models/BAAI/bge-m3`：你已经从魔搭下载好的 BGE-M3 模型路径。

预期：能看到模型文件，例如：

```text
config.json
model.safetensors
tokenizer.json
```

- [ ] **Step 3: 导出多视图 JSONL**

运行：

```bash
bash scripts/export_multiview_100k.sh
```

命令含义：

- `bash`：用 Bash 执行脚本。
- `scripts/export_multiview_100k.sh`：从 TSV 导出 10 万条多视图文档。

预期输出里包含：

```json
{
  "documents_written": 99133
}
```

实际数量可能不是正好 100000，因为坏行会被跳过。

- [ ] **Step 4: 构建 DuckDB**

运行：

```bash
bash scripts/build_analysis_db_100k.sh
```

预期输出：

```json
{
  "documents_loaded": 99133
}
```

- [ ] **Step 5: 生成 BGE-M3 向量**

运行：

```bash
bash scripts/embed_bge_m3_100k_multiview.sh
```

预期输出：

```json
{
  "device": "cuda",
  "shape": [
    99133,
    1024
  ]
}
```

- [ ] **Step 6: 运行流动摆摊专题统计**

运行：

```bash
bash scripts/analyze_topic_stall_100k.sh
```

预期生成：

```text
outputs/topic_analysis.stall.100k.json
```

结果里应该包含：

```json
{
  "query": "流动摆摊 占道经营 游商摊贩 夜市摊贩 无照经营",
  "matched_orders": 1000,
  "statistics": {
    "by_month": [],
    "by_area": [],
    "by_street": [],
    "by_type3": []
  },
  "representative_cases": []
}
```

`matched_orders` 不一定永远是 1000，因为过滤条件和分数阈值会影响数量。

---

## Self-Review

### Spec coverage

- 多视图文档：Task 1。
- case_content 作为核心语义字段：Task 1、Task 2。
- metadata 作为辅助过滤和统计字段：Task 1、Task 5、Task 6。
- 语义检索后统计：Task 6。
- 服务器运行脚本：Task 7、Task 9。
- 中文学习笔记：Task 8。
- 暂不做正文抽标签和聚类：第 0 节明确排除。

### Placeholder scan

本文没有使用 `TBD`、`TODO`、`implement later` 作为实现占位。未来能力被明确标为不在本次范围内。

### Type consistency

- 文档字段统一使用 `case_content_clean`、`case_goal_clean`、`embedding_text`、`display_text`、`metadata`、`derived`。
- 统计配置统一使用 `call_month_gte`、`call_month_lte`、`area_code_area_in`、`area_code_street_in`、`type3_in`。
- 向量 sidecar 和检索结果统一保留 `doc_id`、`score`、`case_content_clean`、`text`、`metadata`。

