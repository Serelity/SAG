import json
import tempfile
import unittest
from pathlib import Path

from scripts.split_semantic_eval_manifest import split_manifest


class TestSplitSemanticEvalManifest(unittest.TestCase):
    def _source(self, root):
        path = root / "source.private.jsonl"
        rows = [
            {
                "schema": "sag_semantic_eval_manifest_v2",
                "doc_id": f"private-{index}",
                "content_hash": f"sha256:{index:064x}",
                "subset": "production" if index < 6 else "challenge",
            }
            for index in range(12)
        ]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_split_is_deterministic_stratified_disjoint_and_report_is_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = self._source(root)
            dev1, hold1, report1 = root / "dev1.jsonl", root / "hold1.jsonl", root / "report1.json"
            dev2, hold2, report2 = root / "dev2.jsonl", root / "hold2.jsonl", root / "report2.json"
            first = split_manifest(source, dev1, hold1, report1, dev_size=4)
            second = split_manifest(source, dev2, hold2, report2, dev_size=4)
            development = [json.loads(line) for line in dev1.read_text().splitlines()]
            holdout = [json.loads(line) for line in hold1.read_text().splitlines()]
            safe_text = report1.read_text(encoding="utf-8")
            dev1_bytes, dev2_bytes = dev1.read_bytes(), dev2.read_bytes()
            hold1_bytes, hold2_bytes = hold1.read_bytes(), hold2.read_bytes()
        self.assertEqual(dev1_bytes, dev2_bytes)
        self.assertEqual(hold1_bytes, hold2_bytes)
        self.assertEqual(first, second)
        self.assertEqual(first["development_subset_counts"], {"challenge": 2, "production": 2})
        self.assertEqual(first["holdout_subset_counts"], {"challenge": 4, "production": 4})
        self.assertEqual(first["identity_overlap"], 0)
        self.assertEqual(len(development), 4)
        self.assertEqual(len(holdout), 8)
        self.assertNotIn("private-", safe_text)
        self.assertNotIn("doc_id", safe_text)
        self.assertNotIn("content_hash", safe_text)

    def test_rejects_overlapping_paths_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = self._source(root)
            with self.assertRaisesRegex(ValueError, "manifest_paths_must_be_distinct"):
                split_manifest(source, source, root / "hold.jsonl", root / "report.json", dev_size=4)
            first = source.read_text(encoding="utf-8").splitlines()[0]
            with source.open("a", encoding="utf-8") as target:
                target.write(first + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate_identity"):
                split_manifest(source, root / "dev.jsonl", root / "hold.jsonl", root / "report.json", dev_size=4)


if __name__ == "__main__":
    unittest.main()
