import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.export_jsonl import export_tsv_to_jsonl


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

            docs = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            quality = json.loads(quality_report.read_text(encoding="utf-8"))

        self.assertEqual(len(docs), 2)
        self.assertTrue(docs[0]["doc_id"].startswith("order_"))
        self.assertIn("诉求内容：市民反映附近夜间摆摊扰民", docs[0]["text"])
        self.assertEqual(report["rows_read"], 2)
        self.assertEqual(report["documents_written"], 2)
        self.assertEqual(report["rows_skipped_bad_field_count"], 0)
        self.assertEqual(quality["documents_written"], 2)

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
