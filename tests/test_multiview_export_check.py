import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.export_jsonl import export_tsv_to_jsonl
from ragflow_style_pipeline.multiview_export_check import check_multiview_export


class TestMultiviewExportCheck(unittest.TestCase):
    def _export(self, root):
        source = root / "source.tsv"
        output = root / "orders.private.jsonl"
        quality = root / "quality.safe.json"
        source.write_text(
            "id\torder_id\ttitle\tcase_content\tcase_goal\taddress_detail\n"
            "private-id-1\tprivate-order-1\t\t私有正文一\t私有目标一\t私有地址一\n"
            "private-id-2\tprivate-order-2\t\t私有正文二\t私有目标二\t私有地址二\n",
            encoding="utf-8",
        )
        export_tsv_to_jsonl(source, output, quality)
        return output, quality

    def test_valid_export_report_is_aggregate_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output, quality = self._export(Path(tmpdir))
            report = check_multiview_export(output, quality)
        self.assertFalse(report["errors_present"])
        self.assertEqual(report["records"], 2)
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("private-id", serialized)
        self.assertNotIn("私有正文", serialized)
        self.assertNotIn("私有地址", serialized)

    def test_detects_duplicate_identity_hash_drift_and_quality_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output, quality = self._export(Path(tmpdir))
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            rows[0]["content_hash"] = "sha256:" + "0" * 64
            rows[1]["doc_id"] = rows[0]["doc_id"]
            rows[1]["content_hash"] = rows[0]["content_hash"]
            output.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = check_multiview_export(output, quality)
        self.assertTrue(report["errors_present"])
        self.assertEqual(report["error_counts"]["duplicate_doc_id"], 1)
        self.assertEqual(report["error_counts"]["duplicate_identity"], 1)
        self.assertEqual(report["error_counts"]["content_hash_mismatch"], 2)
        self.assertEqual(report["error_counts"]["quality_sha256_mismatch"], 1)


if __name__ == "__main__":
    unittest.main()
