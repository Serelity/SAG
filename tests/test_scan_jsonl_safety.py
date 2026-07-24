import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.scan_jsonl_safety import scan_jsonl_safety


class TestScanJsonlSafety(unittest.TestCase):
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
