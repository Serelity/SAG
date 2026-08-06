"""Streaming TSV to JSONL exporter for 12345 order data."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from ragflow_style_pipeline.document_builder import build_document
from ragflow_style_pipeline.sag_semantic_versions import MULTIVIEW_INPUT_VERSION
from ragflow_style_pipeline.work_order_input import CLEAN_FIELDS, content_hash


def _row_has_expected_field_count(row):
    return None not in row and all(value is not None for value in row.values())


def _file_sha256_and_size(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return "sha256:" + digest.hexdigest(), size


def _document_has_polluted_text(document):
    values = [
        document.get("text", ""),
        document.get("title_clean", ""),
        document.get("case_content_clean", ""),
        document.get("case_goal_clean", ""),
        document.get("address_detail_clean", ""),
    ]
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        values.extend(metadata.values())
    return any(
        "\t" in value or "\\t" in value
        for value in values if isinstance(value, str)
    )


def export_tsv_to_jsonl(input_path, output_path, quality_report_path, limit=None):
    """Stream TSV rows into redacted JSONL documents and a quality report."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    quality_report_path = Path(quality_report_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    documents_written = 0
    rows_skipped_bad_field_count = 0
    rows_skipped_empty_semantic_text = 0
    rows_skipped_polluted_text = 0
    redactions = Counter()

    output_temporary = output_path.with_name(output_path.name + ".tmp")
    quality_temporary = quality_report_path.with_name(quality_report_path.name + ".tmp")
    try:
        with input_path.open("r", encoding="utf-8", newline="") as input_file, output_temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as output_file:
            reader = csv.DictReader(input_file, delimiter="\t")
            for row in reader:
                if limit is not None and rows_read >= limit:
                    break

                rows_read += 1
                if not _row_has_expected_field_count(row):
                    rows_skipped_bad_field_count += 1
                    continue

                document, counts = build_document(row)
                if not any(document.get(field) for field in CLEAN_FIELDS):
                    rows_skipped_empty_semantic_text += 1
                    continue
                if _document_has_polluted_text(document):
                    rows_skipped_polluted_text += 1
                    continue

                document["input_schema"] = MULTIVIEW_INPUT_VERSION
                document["content_hash"] = content_hash(document)
                output_file.write(json.dumps(document, ensure_ascii=False) + "\n")
                documents_written += 1
                redactions.update(counts)
            output_file.flush()
            os.fsync(output_file.fileno())

        output_sha256, output_bytes = _file_sha256_and_size(output_temporary)
        report = {
            "schema": MULTIVIEW_INPUT_VERSION,
            "clean_fields": list(CLEAN_FIELDS),
            "content_hash_contract": "clean_fields_plus_metadata_v1",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "rows_read": rows_read,
            "documents_written": documents_written,
            "rows_skipped_bad_field_count": rows_skipped_bad_field_count,
            "rows_skipped_empty_semantic_text": rows_skipped_empty_semantic_text,
            "rows_skipped_polluted_text": rows_skipped_polluted_text,
            "redactions": dict(redactions),
            "output_bytes": output_bytes,
            "output_sha256": output_sha256,
        }
        with quality_temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(output_temporary, output_path)
        os.replace(quality_temporary, quality_report_path)
        return report
    except BaseException:
        output_temporary.unlink(missing_ok=True)
        quality_temporary.unlink(missing_ok=True)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert 12345 order TSV rows into redacted RAG JSONL documents."
    )
    parser.add_argument("--input", required=True, help="Path to the source TSV file.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    parser.add_argument(
        "--quality-report",
        required=True,
        help="Path to the output quality report JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of input rows to read. Omit it to export all rows.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = export_tsv_to_jsonl(
        input_path=args.input,
        output_path=args.output,
        quality_report_path=args.quality_report,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
