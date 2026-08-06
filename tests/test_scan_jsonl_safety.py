import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.scan_jsonl_safety import scan_jsonl_safety


class TestScanJsonlSafety(unittest.TestCase):
    def test_scans_clean_fields_and_metadata_without_double_counting_display_text(self):
        phone = "138" + "0013" + "8000"
        id_card = "320400" + "19900101" + "123" + "X"
        document = {
            "doc_id": "doc_1",
            "text": "地址详情中复制的电话" + phone,
            "title_clean": "",
            "case_content_clean": "正常正文",
            "case_goal_clean": "正常目标",
            "address_detail_clean": "地址电话" + phone,
            "metadata": {"private_test_field": "证件" + id_card},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "docs.jsonl"
            path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")

            report = scan_jsonl_safety(path)

        self.assertEqual(report["documents_scanned"], 1)
        self.assertEqual(report["possible_unredacted_phone"], 1)
        self.assertEqual(report["possible_unredacted_id_card"], 1)

    def test_counts_unredacted_sensitive_patterns_without_returning_raw_text(self):
        document = {
            "doc_id": "doc_1",
            "text": "姓名：张三，电话13800138000，身份证32040019900101123X",
            "metadata": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "docs.jsonl"
            path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")

            report = scan_jsonl_safety(path)

            self.assertEqual(report["documents_scanned"], 1)
            self.assertEqual(report["possible_unredacted_phone"], 1)
            self.assertEqual(report["possible_unredacted_id_card"], 1)
            self.assertEqual(report["possible_unredacted_name_label"], 1)
            self.assertNotIn("张三", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
