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
