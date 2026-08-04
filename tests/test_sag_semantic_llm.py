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
                entities = {"problem_objects":[],"problem_behaviors":[],"roads":[{
                    "surface":"港龙新港城北门口","canonical":"港龙新港城北门口","field":"case_content_clean","evidence":"港龙新港城北门口"
                }],"intersections":[],"pois":[]}
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
                {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v2","batch_size":8,"max_new_tokens":512,"temperature":0.0,"max_repairs_per_order":1,"checkpoint_every":1},
                generator=generator,
            )
            record = json.loads((tmp/"semantic.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(summary["primary_requests"], 1)
        self.assertEqual(summary["repair_requests"], 1)
        self.assertTrue(record["validation"]["repair_attempted"])
        self.assertEqual(record["entities"]["pois"][0]["canonical"], "港龙新港城")

    def test_valid_primary_uses_one_request_and_resume_skips_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; self._input(source)
            paths = [tmp/name for name in ("semantic.jsonl","rejects.jsonl","run.json","quality.json")]
            config = {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v2","batch_size":8,"checkpoint_every":1}
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


if __name__ == "__main__":
    unittest.main()
