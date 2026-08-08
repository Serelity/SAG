"""Synthetic-only tests for strict TSV preparation and PII boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ragflow_style_pipeline.constants import REQUIRED_TSV_COLUMNS
from ragflow_style_pipeline.pii_redactor import redact_text, residual_pii_codes
from ragflow_style_pipeline.work_order import (
    PrepareError,
    clean_content_hash,
    iter_prepared_documents,
    prepare,
    stable_doc_id,
)


class WorkOrderTests(unittest.TestCase):
    def _row(self, **changes):
        row = {column: "" for column in REQUIRED_TSV_COLUMNS}
        row.update(
            {
                "id": "SYNTHETIC-1",
                "title": "路灯问题",
                "case_content": "市民反映道路路灯不亮",
                "case_goal": "希望维修路灯",
                "address_detail": "幸福路口",
                "service_object_type": "求助",
                "area_code_city": "示例市",
                "area_code_area": "示例区",
                "area_code_street": "示例街道",
                "case_accord_type_one_name": "城乡建设",
                "case_accord_type_two_name": "市政设施",
                "case_accord_type_three_name": "路灯",
                "order_source": "电话",
                "order_type": "个人",
                "order_status": "受理",
                "call_time": "2025-01-02 03:04:05",
            }
        )
        row.update(changes)
        return row

    def _write(self, root: str, rows: list[dict], header=None, suffix="\n") -> Path:
        columns = list(header or REQUIRED_TSV_COLUMNS)
        path = Path(root) / "synthetic.tsv"
        lines = ["\t".join(columns)]
        lines.extend("\t".join(row.get(column, "") for column in columns) for row in rows)
        path.write_text("\n".join(lines) + suffix, encoding="utf-8-sig", newline="")
        return path

    def test_redacts_before_document_and_omits_source_identity(self):
        phone = "138" + "0013" + "8000"
        identity = "320400" + "19900101" + "123" + "X"
        with tempfile.TemporaryDirectory() as root:
            path = self._write(
                root,
                [
                    self._row(
                        case_content="联系电话" + phone + "，姓名：张三",
                        case_goal="查询证件" + identity,
                    )
                ],
            )
            document = list(iter_prepared_documents(path))[0]
        serialized = json.dumps(document, ensure_ascii=False)
        self.assertNotIn(phone, serialized)
        self.assertNotIn(identity, serialized)
        self.assertNotIn("SYNTHETIC-1", serialized)
        self.assertIn("[手机号]", document["case_content_clean"])
        self.assertIn("[姓名]", document["case_content_clean"])
        self.assertEqual((), residual_pii_codes(serialized))
        self.assertEqual(stable_doc_id("SYNTHETIC-1"), document["doc_id"])

    def test_content_hash_binds_only_complete_clean_fields(self):
        fields = {
            "title_clean": "标题",
            "case_content_clean": "内容",
            "case_goal_clean": "目标",
            "address_detail_clean": "地址",
        }
        first = clean_content_hash(fields)
        with_extra = {**fields, "metadata": {"call_time": "changed"}}
        self.assertEqual(first, clean_content_hash(with_extra))
        self.assertNotEqual(first, clean_content_hash({**fields, "case_goal_clean": "另一个目标"}))

    def test_bad_rows_are_skipped_with_aggregate_codes(self):
        rows = [
            self._row(id="", order_id=""),
            self._row(id="DUP"),
            self._row(id="DUP", case_content="另一个内容"),
            self._row(id="EMPTY", title="", case_content="", case_goal="", address_detail=""),
            self._row(id="SHIFT", case_content=r"污染\t后续字段"),
            self._row(id="META", call_time="不是时间"),
            self._row(id="GOOD"),
        ]
        with tempfile.TemporaryDirectory() as root:
            source = self._write(root, rows)
            run_dir = Path(root) / "run"
            report = prepare(source, run_dir)
            documents = [
                json.loads(line)
                for line in (run_dir / "documents.private.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(7, report["rows_read"])
        self.assertEqual(2, report["documents_written"])
        self.assertEqual(1, report["rejection_counts"]["missing_source_id"])
        self.assertEqual(1, report["rejection_counts"]["duplicate_source_id"])
        self.assertEqual(1, report["rejection_counts"]["missing_semantic_text"])
        self.assertEqual(1, report["rejection_counts"]["suspected_field_shift"])
        self.assertEqual(1, report["rejection_counts"]["polluted_structured_field"])
        self.assertEqual(2, len(documents))

    def test_bad_field_count_is_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, [self._row(id="GOOD")])
            with path.open("a", encoding="utf-8", newline="") as output:
                output.write("too\tfew\n")
            run_dir = Path(root) / "run"
            report = prepare(path, run_dir)
        self.assertEqual(2, report["rows_read"])
        self.assertEqual(1, report["rejection_counts"]["bad_field_count"])

    def test_invalid_utf8_data_row_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, [self._row(id="GOOD")])
            header = list(REQUIRED_TSV_COLUMNS)
            bad_cells = ["" for _ in header]
            bad_cells[header.index("id")] = "BAD"
            bad_cells[header.index("case_content")] = "placeholder"
            with path.open("ab") as output:
                output.write("\t".join(bad_cells).encode("utf-8") + b"\xff\n")
            run_dir = Path(root) / "run"
            report = prepare(path, run_dir)
        self.assertEqual(2, report["rows_read"])
        self.assertEqual(1, report["documents_written"])
        self.assertEqual(1, report["rejection_counts"]["bad_field_count"])

    def test_header_contract_is_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            missing = [column for column in REQUIRED_TSV_COLUMNS if column != "case_goal"]
            path = self._write(root, [self._row()], header=missing)
            with self.assertRaisesRegex(PrepareError, "missing_required_columns"):
                list(iter_prepared_documents(path))
            duplicate = [*REQUIRED_TSV_COLUMNS, "id"]
            path = self._write(root, [self._row()], header=duplicate)
            with self.assertRaisesRegex(PrepareError, "duplicate_header_column"):
                list(iter_prepared_documents(path))

    def test_limit_counts_source_rows_and_prepare_requires_fresh_run(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write(root, [self._row(id="A"), self._row(id="B")])
            run_dir = Path(root) / "run"
            report = prepare(path, run_dir, limit=1)
            self.assertEqual(1, report["rows_read"])
            self.assertTrue(report["limit_applied"])
            with self.assertRaisesRegex(PrepareError, "run_dir_not_fresh"):
                prepare(path, run_dir)

    def test_pii_rule_order_and_residual_scan(self):
        raw = "邮箱a@example.com，身份证11010519491231002X，联系人：李四，电话010-12345678"
        cleaned, counts = redact_text(raw)
        self.assertEqual(1, counts["email"])
        self.assertEqual(1, counts["id_card"])
        self.assertEqual(1, counts["contact_name"])
        self.assertEqual(1, counts["landline"])
        self.assertEqual((), residual_pii_codes(cleaned))


if __name__ == "__main__":
    unittest.main()
