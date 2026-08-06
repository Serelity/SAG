import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.semantic_inference_packet import build_inference_packet
from ragflow_style_pipeline.work_order_input import normalize_work_order


class TestSemanticInferencePacket(unittest.TestCase):
    def _source_and_manifest(self, root):
        source_rows = [
            {
                "input_schema": "sag_multiview_input_v2",
                "redaction_version": "sag_pii_redaction_v2",
                "doc_id": "private-order-1",
                "title_clean": "",
                "case_content_clean": "私有正文一",
                "case_goal_clean": "私有目标一",
                "address_detail_clean": "私有地址一",
                "metadata": {"service_object_type": "咨询"},
                "text": "不应复制的展示正文",
                "display_text": "不应复制的展示正文",
                "embedding_text": "不应复制的向量正文",
                "derived": {"private": "不应复制"},
            },
            {"doc_id": "invalid-unrelated", "case_content_clean": ""},
            {
                "input_schema": "sag_multiview_input_v2",
                "redaction_version": "sag_pii_redaction_v2",
                "doc_id": "private-order-2",
                "title_clean": "",
                "case_content_clean": "私有正文二",
                "case_goal_clean": "私有目标二",
                "address_detail_clean": "",
                "metadata": {"service_object_type": "求助"},
            },
        ]
        normalized = [normalize_work_order(source_rows[0]), normalize_work_order(source_rows[2])]
        for source, order in zip((source_rows[0], source_rows[2]), normalized):
            source["content_hash"] = order["content_hash"]
        source = root / "source.private.jsonl"
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in source_rows),
            encoding="utf-8",
        )
        manifest = root / "manifest.private.jsonl"
        manifest_rows = [
            {
                "schema": "sag_semantic_eval_manifest_v2",
                "subset": "challenge",
                "doc_id": order["doc_id"],
                "content_hash": order["content_hash"],
            }
            for order in reversed(normalized)
        ]
        manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
            encoding="utf-8",
        )
        return source, manifest

    def test_builds_minimal_packet_in_manifest_order_with_safe_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, manifest = self._source_and_manifest(root)
            output = root / "inference.private.jsonl"
            report = build_inference_packet(source, manifest, output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["doc_id"] for row in rows], ["private-order-2", "private-order-1"])
        self.assertEqual(report["records_scanned"], 3)
        self.assertEqual(report["ignored_invalid_records"], 1)
        self.assertEqual(report["records_written"], 2)
        self.assertEqual(set(rows[0]), set(report["fields"]))
        for row in rows:
            self.assertEqual(row["redaction_version"], "sag_pii_redaction_v2")
            self.assertEqual(row["inference_packet_schema"], "sag_semantic_inference_packet_v1")
            self.assertNotIn("text", row)
            self.assertNotIn("display_text", row)
            self.assertNotIn("embedding_text", row)
            self.assertNotIn("derived", row)
            self.assertEqual(normalize_work_order(row)["content_hash"], row["content_hash"])
        serialized_report = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("private-order", serialized_report)
        self.assertNotIn("私有正文", serialized_report)

    def test_rejects_path_overlap_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, manifest = self._source_and_manifest(root)
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "paths_must_differ"):
                build_inference_packet(source, manifest, source)
            self.assertEqual(source.read_bytes(), original)

    def test_rejects_target_hash_drift_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, manifest = self._source_and_manifest(root)
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            rows[0]["content_hash"] = "sha256:" + "0" * 64
            manifest.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[1]) + "\n")
            output = root / "inference.private.jsonl"
            with self.assertRaisesRegex(ValueError, "target_content_hash_mismatch"):
                build_inference_packet(source, manifest, output)
            self.assertFalse(output.exists())

    def test_rejects_duplicate_manifest_doc_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, manifest = self._source_and_manifest(root)
            row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
            manifest.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate_doc_id"):
                build_inference_packet(source, manifest, root / "inference.private.jsonl")


if __name__ == "__main__":
    unittest.main()
