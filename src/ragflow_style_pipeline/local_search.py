"""Local BM25-style retrieval for JSONL RAG documents."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ragflow_style_pipeline.text_tokenizer import tokenize


def load_documents(jsonl_path, limit=None):
    """Load JSONL documents from disk.

    Each line must contain at least `doc_id`, `text`, and `metadata`.
    """
    documents = []
    with Path(jsonl_path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if limit is not None and len(documents) >= limit:
                break
            if not line.strip():
                continue
            document = json.loads(line)
            text = str(document.get("text") or document.get("display_text") or "")
            documents.append(
                {
                    "doc_id": str(document.get("doc_id", f"line_{line_number}")),
                    "text": text,
                    "display_text": str(document.get("display_text") or text),
                    "embedding_text": str(document.get("embedding_text", "")),
                    "case_content_clean": str(document.get("case_content_clean", "")),
                    "case_goal_clean": str(document.get("case_goal_clean", "")),
                    "metadata": dict(document.get("metadata", {})),
                    "derived": dict(document.get("derived", {})),
                }
            )
    return documents


def _index_text(document):
    metadata = document.get("metadata", {})
    metadata_text = " ".join(
        str(metadata.get(field, ""))
        for field in [
            "service_object_type",
            "area_code_city",
            "area_code_area",
            "area_code_street",
            "type1",
            "type2",
            "type3",
            "call_month",
        ]
    )
    return f"{document.get('text', '')}\n{metadata_text}"


def build_index(documents):
    """Build an in-memory inverted index for local search."""
    documents = list(documents)
    postings = defaultdict(dict)
    document_lengths = {}
    document_tokens = {}

    for document_index, document in enumerate(documents):
        tokens = tokenize(_index_text(document))
        counts = Counter(tokens)
        document_lengths[document_index] = sum(counts.values())
        document_tokens[document_index] = counts
        for token, count in counts.items():
            postings[token][document_index] = count

    total_length = sum(document_lengths.values())
    average_document_length = total_length / len(documents) if documents else 0.0

    return {
        "documents": documents,
        "postings": dict(postings),
        "document_lengths": document_lengths,
        "document_tokens": document_tokens,
        "average_document_length": average_document_length,
    }


def _matches_filters(document, filters):
    if not filters:
        return True
    metadata = document.get("metadata", {})
    for field, expected_value in filters.items():
        if expected_value and metadata.get(field) != expected_value:
            return False
    return True


def search(index, query, top_k=5, filters=None, k1=1.5, b=0.75):
    """Search an index and return ranked result dictionaries."""
    query_tokens = tokenize(query)
    documents = index["documents"]
    postings = index["postings"]
    document_lengths = index["document_lengths"]
    average_document_length = index["average_document_length"] or 1.0
    document_count = len(documents)

    if not query_tokens or not documents:
        return []

    scores = defaultdict(float)
    unique_query_tokens = set(query_tokens)

    for token in unique_query_tokens:
        token_postings = postings.get(token)
        if not token_postings:
            continue
        document_frequency = len(token_postings)
        inverse_document_frequency = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

        for document_index, term_frequency in token_postings.items():
            document = documents[document_index]
            if not _matches_filters(document, filters):
                continue
            document_length = document_lengths[document_index]
            denominator = term_frequency + k1 * (
                1 - b + b * document_length / average_document_length
            )
            scores[document_index] += inverse_document_frequency * (
                term_frequency * (k1 + 1)
            ) / denominator

    ranked = sorted(scores.items(), key=lambda item: (-item[1], documents[item[0]]["doc_id"]))
    results = []
    for document_index, score in ranked[:top_k]:
        document = documents[document_index]
        results.append(
            {
                "doc_id": document["doc_id"],
                "score": round(score, 6),
                "text": document["text"],
                "metadata": document.get("metadata", {}),
            }
        )
    return results
