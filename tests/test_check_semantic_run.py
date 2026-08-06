import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.check_semantic_run import check


class TestCheckSemanticRun(unittest.TestCase):
    def _write_artifacts(self, root, manifest_records=1):
        semantic = root / "semantic.jsonl"
        rejects = root / "rejects.jsonl"
        run_report = root / "run.json"
        quality_report = root / "quality.json"
        semantic.write_text(json.dumps({
            "doc_id": "private-doc-id",
            "content_hash": "sha256:content",
            "model_run": {
                "prompt_version": "sag_semantic_v7",
                "model": "Qwen/Qwen3-4B",
                "finish_reason": "stop",
            },
            "validation": {
                "status": "accepted",
                "warnings": [],
                "repair_attempted": False,
            },
        }) + "\n", encoding="utf-8")
        rejects.write_text("", encoding="utf-8")
        run_report.write_text(json.dumps({
            "records_written": 1,
            "rejects_written": 0,
            "identity_manifest_enabled": True,
            "identity_manifest_records": manifest_records,
            "identity_manifest_sha256": "sha256:" + "a" * 64,
            "input_records_scanned": 99,
            "ignored_invalid_input_records": 2,
            "orders_input": 1,
        }) + "\n", encoding="utf-8")
        quality_report.write_text(json.dumps({"records": 1}) + "\n", encoding="utf-8")
        return SimpleNamespace(
            semantic=str(semantic),
            rejects=str(rejects),
            run_report=str(run_report),
            quality_report=str(quality_report),
        )

    def test_checks_identity_manifest_provenance_without_returning_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._write_artifacts(Path(tmpdir))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = check(args)
        self.assertEqual(result["identity_manifest"], {
            "enabled": True,
            "records": 1,
            "input_records_scanned": 99,
            "ignored_invalid_input_records": 2,
            "sha256": "sha256:" + "a" * 64,
        })
        self.assertNotIn("private-doc-id", output.getvalue())

    def test_rejects_identity_manifest_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._write_artifacts(Path(tmpdir), manifest_records=2)
            with self.assertRaisesRegex(ValueError, "identity_manifest_count_mismatch"):
                check(args)


if __name__ == "__main__":
    unittest.main()
