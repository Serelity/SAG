"""Command-line local search over exported RAG JSONL documents."""

import argparse
import json
import time
from pathlib import Path

from ragflow_style_pipeline.local_search import build_index, load_documents, search
from ragflow_style_pipeline.pii_redactor import redact_text


FILTER_ARGUMENTS = {
    "area": "area_code_area",
    "type1": "type1",
    "type2": "type2",
    "type3": "type3",
    "month": "call_month",
}

SAFE_METADATA_FIELDS = [
    "service_object_type",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "type1",
    "type2",
    "type3",
    "call_month",
]


def _build_filters(args):
    filters = {}
    for argument_name, metadata_field in FILTER_ARGUMENTS.items():
        value = getattr(args, argument_name)
        if value:
            filters[metadata_field] = value
    return filters


def _snippet(text, max_length=220):
    text, _counts = redact_text(text)
    text = " ".join(str(text).split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _metadata_summary(metadata):
    fields = [
        metadata.get("area_code_area", ""),
        metadata.get("area_code_street", ""),
        metadata.get("type1", ""),
        metadata.get("type2", ""),
        metadata.get("type3", ""),
        metadata.get("call_month", ""),
    ]
    return " | ".join(field for field in fields if field)


def format_results(results):
    """Format ranked results for terminal output."""
    if not results:
        return "No results."

    blocks = []
    for rank, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Rank {rank} | score={result['score']:.6f} | doc_id={result['doc_id']}",
                    f"Metadata: {_metadata_summary(result.get('metadata', {}))}",
                    f"Snippet: {_snippet(result.get('text', ''))}",
                ]
            )
        )
    return "\n\n".join(blocks)


def safe_result(result):
    """Return a result payload safe for display or local JSON reports."""
    metadata = result.get("metadata", {})
    return {
        "doc_id": result["doc_id"],
        "score": result["score"],
        "metadata": {
            field: metadata.get(field, "")
            for field in SAFE_METADATA_FIELDS
            if metadata.get(field, "")
        },
        "snippet": _snippet(result.get("text", "")),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Search redacted RAG JSONL documents with a local BM25-style index."
    )
    parser.add_argument("--input", required=True, help="Path to the redacted JSONL file.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to print.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum documents to load.")
    parser.add_argument("--area", default="", help="Filter by metadata.area_code_area.")
    parser.add_argument("--type1", default="", help="Filter by metadata.type1.")
    parser.add_argument("--type2", default="", help="Filter by metadata.type2.")
    parser.add_argument("--type3", default="", help="Filter by metadata.type3.")
    parser.add_argument("--month", default="", help="Filter by metadata.call_month, such as 2024-06.")
    parser.add_argument("--output", default="", help="Optional JSON result output path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    start_time = time.perf_counter()
    documents = load_documents(args.input, limit=args.limit)
    index = build_index(documents)
    results = search(index, args.query, top_k=args.top_k, filters=_build_filters(args))
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Loaded documents: {len(documents)}")
    print(f"Query: {args.query}")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print()
    print(format_results(results))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input": args.input,
            "query": args.query,
            "top_k": args.top_k,
            "filters": _build_filters(args),
            "documents_loaded": len(documents),
            "elapsed_seconds": round(elapsed_seconds, 6),
            "results": [safe_result(result) for result in results],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
