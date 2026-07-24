"""Safety scanner for exported JSONL documents.

This tool reports counts only. It intentionally avoids printing raw document
text so that verification does not leak sensitive data into logs.
"""

import argparse
import json
import re
from pathlib import Path


PHONE_RE = re.compile(r"(?<!\d)0?1[3-9]\d{9,10}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
NAME_LABEL_RE = re.compile(r"姓名[:：]\s*(?!\[姓名\])[\u4e00-\u9fff·]{2,8}")


def scan_jsonl_safety(jsonl_path):
    """Return count-only safety findings for one JSONL export."""
    report = {
        "input_path": str(jsonl_path),
        "documents_scanned": 0,
        "possible_unredacted_phone": 0,
        "possible_unredacted_id_card": 0,
        "possible_unredacted_name_label": 0,
    }

    with Path(jsonl_path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            document = json.loads(line)
            text = str(document.get("text", ""))
            report["documents_scanned"] += 1
            report["possible_unredacted_phone"] += len(PHONE_RE.findall(text))
            report["possible_unredacted_id_card"] += len(ID_CARD_RE.findall(text))
            report["possible_unredacted_name_label"] += len(NAME_LABEL_RE.findall(text))

    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan exported RAG JSONL documents for count-only sensitive-pattern findings."
    )
    parser.add_argument("--input", required=True, help="Path to the redacted JSONL file.")
    parser.add_argument("--output", default="", help="Optional JSON report output path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = scan_jsonl_safety(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
