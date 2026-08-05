import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_llm import parse_args, run_semantic_extraction


class RecordingGenerator:
    def __init__(self, repair_primary=True):
        self.calls = []
        self.repair_primary = repair_primary

    def __call__(self, prompts, max_new_tokens, temperature):
        self.calls.append(list(prompts))
        rows = []
        for prompt in prompts:
            repair = "只修复" in prompt
            if repair:
                entities = {
                    "problem_objects": [{"surface":"摊贩","canonical":"流动摊贩","field":"case_content_clean","evidence":"摊贩"}],
                    "problem_behaviors": [{"surface":"占道","canonical":"占道经营","field":"case_content_clean","evidence":"占道"}],
                    "roads": [], "intersections": [],
                    "pois": [{"surface":"港龙新港城","canonical":"港龙新港城","field":"case_content_clean","evidence":"港龙新港城"}],
                }
            elif self.repair_primary:
                rows.append({"text":"{\"event_summary\":","input_tokens":100,"output_tokens":20,"finish_reason":"stop","latency_ms":10})
                continue
            else:
                entities = {"problem_objects":[{"surface":"摊贩","canonical":"流动摊贩","field":"case_content_clean","evidence":"摊贩"}],"problem_behaviors":[],"roads":[],"intersections":[],"pois":[]}
            text = json.dumps({
                "event_summary":"市民反映港龙新港城北门有摊贩占道",
                "entities": entities,
                "discourse":{"intents":[],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}},
            }, ensure_ascii=False)
            rows.append({"text":text,"input_tokens":100,"output_tokens":80,"finish_reason":"stop","latency_ms":10})
        return rows


class TestSemanticLlm(unittest.TestCase):
    def _input(self, path, text="港龙新港城北门口有摊贩占道。"):
        path.write_text(json.dumps({"doc_id":"order_1","case_content_clean":text}, ensure_ascii=False) + "\n", encoding="utf-8")

    def test_one_primary_call_and_one_selective_repair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp / "orders.jsonl"; self._input(source)
            generator = RecordingGenerator()
            summary = run_semantic_extraction(
                source, tmp/"semantic.jsonl", tmp/"rejects.jsonl", tmp/"run.json", tmp/"quality.json", "unused",
                {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v3","batch_size":8,"max_new_tokens":512,"temperature":0.0,"max_repairs_per_order":1,"checkpoint_every":1},
                generator=generator,
            )
            record = json.loads((tmp/"semantic.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(summary["primary_requests"], 1)
        self.assertEqual(summary["repair_requests"], 1)
        self.assertTrue(record["validation"]["repair_attempted"])
        self.assertEqual(record["entities"]["pois"][0]["canonical"], "港龙新港城")

    def test_invalid_optional_candidate_is_dropped_without_repair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; self._input(source)

            class InvalidCandidateGenerator:
                def __init__(self): self.calls = []
                def __call__(self, prompts, max_new_tokens, temperature):
                    self.calls.append(prompts)
                    return [{"text": json.dumps({
                        "event_summary":"市民反映摊贩占道",
                        "entities":{"problem_objects":[{
                            "surface":"不存在","canonical":"不存在","field":"case_content_clean","evidence":"不存在"
                        }],"problem_behaviors":[],"roads":[],"intersections":[],"pois":[]},
                        "discourse":{"intents":[],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}},
                    }, ensure_ascii=False),"input_tokens":20,"output_tokens":30,"finish_reason":"stop","latency_ms":1}]

            generator = InvalidCandidateGenerator()
            diagnostic = tmp/"diagnostics.jsonl"
            summary = run_semantic_extraction(
                source, tmp/"semantic.jsonl", tmp/"rejects.jsonl", tmp/"run.json", tmp/"quality.json", "unused",
                {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v3","batch_size":1,"checkpoint_every":1},
                generator=generator, diagnostic_path=diagnostic,
            )
            record = json.loads((tmp/"semantic.jsonl").read_text(encoding="utf-8"))
            diagnostics = [json.loads(line) for line in diagnostic.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary["repair_requests"], 0)
        self.assertEqual(record["validation"]["status"], "accepted_with_warnings")
        self.assertEqual(record["entities"]["problem_objects"], [])
        self.assertIn("dropped_invalid_candidate:entities.problem_objects.0", record["validation"]["warnings"])

        def all_keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from all_keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from all_keys(item)

        serialized = json.dumps(diagnostics, ensure_ascii=False)
        for forbidden_text in ("港龙新港城", "市民反映"):
            self.assertNotIn(forbidden_text, serialized)
        forbidden_keys = {"prompt", "prompts", "evidence", "raw_response", "primary_response", "repair_response", "chunk_text"}
        self.assertFalse(set(all_keys(diagnostics)) & forbidden_keys)
        self.assertEqual(diagnostics[0]["schema"], "privacy_safe_diagnostics_v1")
        self.assertIn("model_call_started", {row["event"] for row in diagnostics})
        self.assertIn("run_completed", {row["event"] for row in diagnostics})

    def test_input_failure_is_logged_without_exception_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"broken.jsonl"
            source.write_text("{not-json}\n", encoding="utf-8")
            diagnostic = tmp/"diagnostics.jsonl"
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                run_semantic_extraction(
                    source, tmp/"semantic.jsonl", tmp/"rejects.jsonl", tmp/"run.json", tmp/"quality.json", "unused",
                    {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v3","batch_size":1},
                    generator=RecordingGenerator(False), diagnostic_path=diagnostic,
                )
            diagnostics = [json.loads(line) for line in diagnostic.read_text(encoding="utf-8").splitlines()]
        failure = diagnostics[-1]
        self.assertEqual(failure["event"], "run_failed")
        self.assertEqual(failure["stage"], "input_read")
        self.assertNotIn("message", failure)
        self.assertNotIn("not-json", json.dumps(diagnostics, ensure_ascii=False))

    def test_generator_failure_is_logged_without_exception_message_or_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; self._input(source)
            diagnostic = tmp/"diagnostics.jsonl"

            class FailingGenerator:
                def __call__(self, prompts, max_new_tokens, temperature):
                    raise RuntimeError("港龙新港城：模拟含正文的异常消息")

            with self.assertRaises(RuntimeError):
                run_semantic_extraction(
                    source, tmp/"semantic.jsonl", tmp/"rejects.jsonl", tmp/"run.json", tmp/"quality.json", "unused",
                    {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v3","batch_size":1},
                    generator=FailingGenerator(), diagnostic_path=diagnostic,
                )
            diagnostics = [json.loads(line) for line in diagnostic.read_text(encoding="utf-8").splitlines()]
        failure = next(row for row in diagnostics if row["event"] == "model_call_failed")
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertNotIn("message", failure)
        self.assertNotIn("港龙新港城", json.dumps(diagnostics, ensure_ascii=False))

    def test_valid_primary_uses_one_request_and_resume_skips_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; self._input(source)
            paths = [tmp/name for name in ("semantic.jsonl","rejects.jsonl","run.json","quality.json")]
            config = {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v3","batch_size":8,"checkpoint_every":1}
            first = RecordingGenerator(False)
            run_semantic_extraction(source, *paths, "unused", config, generator=first)
            second = RecordingGenerator(False)
            report = run_semantic_extraction(source, *paths, "unused", config, resume=True, generator=second)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(second.calls, [])
        self.assertEqual(report["orders_processed"], 0)

    def test_cli_exposes_safe_arguments(self):
        args = parse_args(["--input","safe.multiview.jsonl","--output","outputs/a.jsonl","--rejects","outputs/r.jsonl","--run-report","outputs/run.json","--quality-report","outputs/q.json","--config","config.json","--model-path","models/Qwen3-4B"])
        self.assertTrue(args.input.endswith(".multiview.jsonl"))
        self.assertEqual(args.diagnostic_log, "")
        self.assertIsNone(args.batch_size)
        overridden = parse_args([
            "--input","safe.multiview.jsonl","--output","outputs/a.jsonl","--rejects","outputs/r.jsonl",
            "--run-report","outputs/run.json","--quality-report","outputs/q.json","--config","config.json",
            "--model-path","models/Qwen3-4B","--batch-size","1","--diagnostic-log","outputs/d.jsonl",
        ])
        self.assertEqual(overridden.batch_size, 1)
        self.assertEqual(overridden.diagnostic_log, "outputs/d.jsonl")


if __name__ == "__main__":
    unittest.main()
