import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_annotation_workbench import (
    AnnotationStore,
    AnnotationStoreError,
)
from ragflow_style_pipeline.sag_semantic_audit import (
    build_eval_manifest,
    build_private_annotation_packet,
    prepare_annotation_round,
    validate_gold_annotations,
)


class TestAnnotationWorkbench(unittest.TestCase):
    def _round_file(self, root):
        source = root / "orders.jsonl"
        rows = [
            {
                "doc_id": "private-order-1",
                "case_content_clean": "服务对象咨询许可证需要哪些材料",
                "case_goal_clean": "希望告知办理材料",
                "metadata": {"service_object_type": "咨询", "type1": "政务"},
            },
            {
                "doc_id": "private-order-2",
                "case_content_clean": "人民路路灯损坏",
                "case_goal_clean": "要求维修",
                "metadata": {"service_object_type": "求助", "type1": "市政"},
            },
        ]
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest, _report = build_eval_manifest(
            source, production_size=2, challenge_size=0, seed="workbench"
        )
        manifest_path = root / "manifest.private.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
            encoding="utf-8",
        )
        packet = build_private_annotation_packet(source, manifest_path)
        packet_path = root / "packet.private.jsonl"
        packet_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet),
            encoding="utf-8",
        )
        left, _right, _round_report = prepare_annotation_round(
            packet_path, "annotator-a", "annotator-b"
        )
        path = root / "annotator-a.private.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in left),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _payload(evidence="需要哪些材料", status="completed"):
        return {
            "issues": [{
                "mode": "question",
                "time_scope": "current",
                "objects": [{
                    "surface": "许可证",
                    "field": "case_content_clean",
                    "evidence": "许可证需要哪些材料",
                }],
                "predicates": [{
                    "surface": "需要哪些材料",
                    "field": "case_content_clean",
                    "evidence": evidence,
                }],
                "actions": [],
                "locations": [],
            }],
            "declared_intents": [{
                "label": "咨询",
                "field": "case_content_clean",
                "evidence": "咨询许可证需要哪些材料",
            }],
            "direct_emotions": [],
            "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
            "urgency": {"level": "normal", "evidence": ""},
            "status": status,
            "notes": "independent annotation",
        }

    def test_record_omits_identity_and_invalid_save_does_not_touch_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._round_file(Path(tmpdir))
            original = path.read_bytes()
            store = AnnotationStore(path, expected_annotator="annotator-a")
            record = store.record(0)
            self.assertNotIn("doc_id", record)
            self.assertNotIn("content_hash", record)
            self.assertNotIn("manifest_provenance", record)
            result = store.save(
                0, store.revision, self._payload(evidence="不存在的证据")
            )
            self.assertFalse(result["saved"])
            self.assertIn("issue_member_evidence_not_in_field", result["validation"]["errors"])
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_name(path.name + ".bak").exists())

    def test_valid_save_is_atomic_backed_up_and_preserves_private_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._round_file(Path(tmpdir))
            original = path.read_bytes()
            original_rows = [json.loads(line) for line in original.decode().splitlines()]
            store = AnnotationStore(path, expected_annotator="annotator-a")
            old_revision = store.revision
            result = store.save(0, old_revision, self._payload())
            self.assertTrue(result["saved"])
            self.assertNotEqual(result["revision"], old_revision)
            self.assertEqual(result["status_counts"]["completed"], 1)
            self.assertEqual(path.with_name(path.name + ".bak").read_bytes(), original)
            updated = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for key in (
                "doc_id", "content_hash", "clean_fields", "metadata",
                "manifest_provenance", "annotation_round_provenance",
            ):
                self.assertEqual(updated[0][key], original_rows[0][key])
            self.assertEqual(updated[0]["annotation"]["status"], "completed")
            validation = validate_gold_annotations(path, expected_annotator="annotator-a")
            self.assertFalse(validation["errors_present"])
            with self.assertRaisesRegex(AnnotationStoreError, "annotation_revision_conflict"):
                store.save(1, old_revision, self._payload(status="in_progress"))

    def test_rejects_external_change_and_payload_source_injection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._round_file(Path(tmpdir))
            store = AnnotationStore(path, expected_annotator="annotator-a")
            injected = self._payload(status="in_progress")
            injected["doc_id"] = "replacement"
            with self.assertRaisesRegex(AnnotationStoreError, "annotation_payload_invalid"):
                store.save(0, store.revision, injected)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(AnnotationStoreError, "changed_externally"):
                store.save(0, store.revision, self._payload(status="in_progress"))


if __name__ == "__main__":
    unittest.main()
