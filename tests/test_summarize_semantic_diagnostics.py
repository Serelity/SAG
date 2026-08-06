import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_semantic_diagnostics import summarize


class TestSummarizeSemanticDiagnostics(unittest.TestCase):
    def test_summarizes_privacy_safe_events_without_order_content(self):
        rows = [
            {
                "event": "model_result",
                "phase": "primary",
                "input_tokens": 100,
                "output_tokens": 40,
                "finish_reason": "stop",
                "latency_ms": 125.5,
                "repair_requested": False,
                "validation_before": {
                    "status": "repair_required",
                    "warnings": ["missing_evidence:entities.roads.0"],
                    "repair_fields": ["entities.roads.0"],
                },
                "validation_after": {
                    "status": "accepted_with_warnings",
                    "warnings": ["dropped_invalid_candidate:entities.roads.0"],
                    "repair_fields": [],
                },
                "sanitation_warnings": ["dropped_invalid_candidate:entities.roads.0"],
                "semantic_counts": {
                    "entities": {
                        "problem_objects": 1,
                        "problem_behaviors": 1,
                        "roads": 0,
                        "intersections": 0,
                        "pois": 1,
                    }
                },
            },
            {
                "event": "batch_memory",
                "current_allocated_gb": 8.0,
                "current_reserved_gb": 8.5,
                "peak_allocated_gb": 12.0,
                "peak_reserved_gb": 13.0,
            },
            {
                "event": "run_completed",
                "records_written": 1,
                "rejects_written": 0,
                "primary_requests": 1,
                "repair_requests": 0,
                "primary_batches": 1,
                "repair_batches": 0,
                "elapsed_seconds": 1.5,
                "stage_seconds": {"model_load": 1.0, "generation_wall": 0.4},
                "run_attempt_id": "safe-attempt-id",
                "candidate_entries_written": 1,
                "decision_entries_written": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "diagnostics.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = summarize(path)

        self.assertEqual(result["phase_counts"], {"primary": 1})
        self.assertEqual(result["validation_before_statuses"], {"repair_required": 1})
        self.assertEqual(result["validation_after_statuses"], {"accepted_with_warnings": 1})
        self.assertEqual(result["sanitation_actions"], {"dropped_invalid_candidate:entities.roads.0": 1})
        self.assertEqual(result["input_tokens_total"], 100)
        self.assertEqual(result["output_tokens_total"], 40)
        self.assertEqual(result["memory_by_batch"][0]["peak_reserved_gb"], 13.0)
        self.assertEqual(result["failure_counts"], {})
        self.assertEqual(result["last_event"], "run_completed")
        self.assertEqual(result["run_completed"]["primary_batches"], 1)
        self.assertEqual(result["run_completed"]["repair_batches"], 0)
        self.assertEqual(result["run_completed"]["stage_seconds"]["model_load"], 1.0)
        self.assertEqual(result["run_completed"]["candidate_entries_written"], 1)
        self.assertEqual(json.loads(output.getvalue()), result)


if __name__ == "__main__":
    unittest.main()
