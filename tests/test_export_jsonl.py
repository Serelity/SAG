import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragflow_style_pipeline.export_jsonl import export_tsv_to_jsonl
from ragflow_style_pipeline.work_order_input import normalize_work_order


class TestExportJsonl(unittest.TestCase):
    def test_exports_limited_jsonl_and_quality_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "sample.jsonl"
            quality_report = Path(tmp_dir) / "sample.quality.json"

            report = export_tsv_to_jsonl(
                input_path=Path("tests/fixtures/t_order_master_sample.tsv"),
                output_path=output,
                quality_report_path=quality_report,
                limit=2,
            )

            output_bytes = output.read_bytes()
            docs = [json.loads(line) for line in output_bytes.decode("utf-8").splitlines()]
            quality = json.loads(quality_report.read_text(encoding="utf-8"))

        self.assertEqual(len(docs), 2)
        self.assertTrue(docs[0]["doc_id"].startswith("order_"))
        self.assertEqual(docs[0]["input_schema"], "sag_multiview_input_v2")
        self.assertEqual(docs[0]["redaction_version"], "sag_pii_redaction_v2")
        self.assertEqual(docs[0]["content_hash"], normalize_work_order(docs[0])["content_hash"])
        self.assertNotIn("order_id", docs[0]["metadata"])
        self.assertIn("诉求内容：市民反映附近夜间摆摊扰民", docs[0]["text"])
        self.assertEqual(report["rows_read"], 2)
        self.assertEqual(report["documents_written"], 2)
        self.assertEqual(report["rows_skipped_bad_field_count"], 0)
        self.assertEqual(quality["documents_written"], 2)
        self.assertEqual(quality["schema"], "sag_multiview_input_v2")
        self.assertEqual(quality["redaction_version"], "sag_pii_redaction_v2")
        self.assertEqual(quality["clean_fields"], [
            "title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean",
        ])
        self.assertEqual(quality["output_bytes"], len(output_bytes))
        self.assertEqual(
            quality["output_sha256"],
            "sha256:" + hashlib.sha256(output_bytes).hexdigest(),
        )

    def test_export_failure_preserves_existing_outputs_and_removes_temporaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.tsv"
            output = Path(tmp_dir) / "sample.jsonl"
            quality_report = Path(tmp_dir) / "sample.quality.json"
            input_path.write_text(
                "id\torder_id\tcase_content\tcase_goal\n"
                "1\tORD001\t内容一\t目标一\n"
                "2\tORD002\t内容二\t目标二\n",
                encoding="utf-8",
            )
            output.write_bytes(b"existing-jsonl\n")
            quality_report.write_bytes(b"existing-report\n")
            calls = 0

            def fail_on_second(row):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated")
                from ragflow_style_pipeline.document_builder import build_document
                return build_document(row)

            with patch(
                "ragflow_style_pipeline.export_jsonl.build_document",
                side_effect=fail_on_second,
            ), self.assertRaisesRegex(RuntimeError, "simulated"):
                export_tsv_to_jsonl(input_path, output, quality_report)

            self.assertEqual(output.read_bytes(), b"existing-jsonl\n")
            self.assertEqual(quality_report.read_bytes(), b"existing-report\n")
            self.assertFalse(output.with_name(output.name + ".tmp").exists())
            self.assertFalse(quality_report.with_name(quality_report.name + ".tmp").exists())

    def test_redacts_phone_and_id_in_exported_jsonl(self):
        phone = "138" + "0013" + "8000"
        id_card = "320400" + "19900101" + "123" + "X"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.tsv"
            output = Path(tmp_dir) / "sample.jsonl"
            quality_report = Path(tmp_dir) / "sample.quality.json"
            input_path.write_text(
                "\t".join(
                    [
                        "id",
                        "order_id",
                        "service_object_type",
                        "case_content",
                        "case_goal",
                        "area_code_city",
                        "area_code_area",
                        "area_code_street",
                        "case_accord_type_one_name",
                        "case_accord_type_two_name",
                        "case_accord_type_three_name",
                        "order_source",
                        "order_type",
                        "order_status",
                        "call_time",
                    ]
                )
                + "\n"
                + "\t".join(
                    [
                        "9",
                        "ORD009",
                        "咨询",
                        "联系电话" + phone + "，身份证" + id_card,
                        "希望回访",
                        "常州市",
                        "武进区",
                        "湖塘镇",
                        "民生保障",
                        "社会保障",
                        "职工医疗保险",
                        "电话",
                        "个人",
                        "100",
                        "2025-01-02 10:11:12",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = export_tsv_to_jsonl(
                input_path=input_path,
                output_path=output,
                quality_report_path=quality_report,
            )

            exported_text = output.read_text(encoding="utf-8")

        self.assertNotIn(phone, exported_text)
        self.assertNotIn(id_card, exported_text)
        self.assertIn("[手机号]", exported_text)
        self.assertIn("[身份证号]", exported_text)
        self.assertEqual(report["redactions"]["phone"], 1)
        self.assertEqual(report["redactions"]["id_card"], 1)

    def test_exports_redacted_title_and_address_fields(self):
        phone = "139" + "0013" + "8000"
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input.tsv"
            output = Path(tmp_dir) / "sample.jsonl"
            quality_report = Path(tmp_dir) / "sample.quality.json"
            input_path.write_text(
                "id\torder_id\ttitle\tcase_content\tcase_goal\taddress_detail\n"
                "1\tORD001\t姓名：张三，咨询\t正常内容\t希望答复\t人民路，电话" + phone + "\n",
                encoding="utf-8",
            )

            export_tsv_to_jsonl(input_path, output, quality_report)
            document = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(document["title_clean"], "姓名：[姓名]，咨询")
        self.assertEqual(document["address_detail_clean"], "人民路，电话[手机号]")
        self.assertNotIn(phone, json.dumps(document, ensure_ascii=False))

    def test_skips_only_fully_empty_semantic_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "empty.tsv"
            output = Path(tmp_dir) / "sample.jsonl"
            quality_report = Path(tmp_dir) / "sample.quality.json"
            input_path.write_text(
                "id\torder_id\ttitle\tcase_content\tcase_goal\taddress_detail\tservice_object_type\n"
                "1\tORD001\t\t\t\t\t咨询\n"
                "2\tORD002\t标题有效\t\t\t\t咨询\n"
                "3\tORD003\t\t\t\t地址有效\t咨询\n",
                encoding="utf-8",
            )

            report = export_tsv_to_jsonl(input_path, output, quality_report)
            docs = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["rows_read"], 3)
        self.assertEqual(report["documents_written"], 2)
        self.assertEqual(report["rows_skipped_empty_semantic_text"], 1)
        self.assertEqual([doc["title_clean"] for doc in docs], ["标题有效", ""])
        self.assertEqual([doc["address_detail_clean"] for doc in docs], ["", "地址有效"])

    def test_skips_rows_with_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "bad.tsv"
            output = Path(tmp_dir) / "sample.jsonl"
            quality_report = Path(tmp_dir) / "sample.quality.json"
            input_path.write_text(
                "id\torder_id\tservice_object_type\tcase_content\tcase_goal\n"
                "1\tORD001\t求助\t内容\t目标\n"
                "2\tORD002\t咨询\n",
                encoding="utf-8",
            )

            report = export_tsv_to_jsonl(
                input_path=input_path,
                output_path=output,
                quality_report_path=quality_report,
            )

            docs = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(docs), 1)
        self.assertEqual(report["rows_read"], 2)
        self.assertEqual(report["documents_written"], 1)
        self.assertEqual(report["rows_skipped_bad_field_count"], 1)

    def test_skips_pollution_in_address_clean_field(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "polluted-address.tsv"
            output = Path(tmp_dir) / "sample.jsonl"
            quality_report = Path(tmp_dir) / "sample.quality.json"
            input_path.write_text(
                "id\torder_id\tcase_content\tcase_goal\taddress_detail\n"
                "1\tORD001\t正常内容\t目标\t正常地址\n"
                "2\tORD002\t正常内容\t目标\t污染地址\\\\t后续字段\n",
                encoding="utf-8",
            )

            report = export_tsv_to_jsonl(input_path, output, quality_report)

        self.assertEqual(report["documents_written"], 1)
        self.assertEqual(report["rows_skipped_polluted_text"], 1)

    def test_skips_documents_with_tab_polluted_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "polluted.tsv"
            output = Path(tmp_dir) / "sample.jsonl"
            quality_report = Path(tmp_dir) / "sample.quality.json"
            input_path.write_text(
                "id\torder_id\tservice_object_type\tcase_content\tcase_goal\n"
                "1\tORD001\t求助\t正常内容\t目标\n"
                "2\tORD002\t求助\t污染内容\\\\t后续字段混入\t目标\n",
                encoding="utf-8",
            )

            report = export_tsv_to_jsonl(
                input_path=input_path,
                output_path=output,
                quality_report_path=quality_report,
            )

            docs = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(docs), 1)
        self.assertEqual(report["rows_read"], 2)
        self.assertEqual(report["documents_written"], 1)
        self.assertEqual(report["rows_skipped_polluted_text"], 1)


if __name__ == "__main__":
    unittest.main()
