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
