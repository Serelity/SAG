"""Streaming TSV to JSONL exporter for 12345 order data."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ragflow_style_pipeline.document_builder import build_document


def _row_has_expected_field_count(row):
    return None not in row and all(value is not None for value in row.values())


def _document_has_polluted_text(document):
    text = document.get("text", "")
    return "\t" in text or "\\t" in text


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
    rows_skipped_polluted_text = 0
    redactions = Counter()

    with input_path.open("r", encoding="utf-8", newline="") as input_file, output_path.open(
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
            if _document_has_polluted_text(document):
                rows_skipped_polluted_text += 1
                continue

            output_file.write(json.dumps(document, ensure_ascii=False) + "\n")
            documents_written += 1
            redactions.update(counts)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_read": rows_read,
        "documents_written": documents_written,
        "rows_skipped_bad_field_count": rows_skipped_bad_field_count,
        "rows_skipped_polluted_text": rows_skipped_polluted_text,
        "redactions": dict(redactions),
    }
    quality_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


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
