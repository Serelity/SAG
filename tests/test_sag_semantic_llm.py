import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ragflow_style_pipeline.sag_semantic_llm import (
    _load_configured_generator,
    _validate_with_sanitation,
    load_vllm_generator,
    parse_args,
    run_semantic_extraction,
)


class RecordingGenerator:
    def __init__(self, repair_primary=True):
        self.calls = []
        self.token_limits = []
        self.repair_primary = repair_primary

    def __call__(self, prompts, max_new_tokens, temperature):
        self.calls.append(list(prompts))
        self.token_limits.append(max_new_tokens)
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
                {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v5","batch_size":8,"max_new_tokens":640,"repair_max_new_tokens":768,"temperature":0.0,"max_repairs_per_order":1,"checkpoint_every":1},
                generator=generator,
            )
            record = json.loads((tmp/"semantic.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(generator.token_limits, [640, 768])
        self.assertEqual(summary["primary_requests"], 1)
        self.assertEqual(summary["repair_requests"], 1)
        self.assertTrue(record["validation"]["repair_attempted"])
        self.assertEqual(record["entities"]["pois"][0]["canonical"], "港龙新港城")

    def test_complete_safe_json_recovery_does_not_consume_model_repair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; self._input(source)

            class TrailingCommaGenerator:
                def __init__(self): self.calls = 0
                def __call__(self, prompts, max_new_tokens, temperature):
                    self.calls += 1
                    text = json.dumps({
                        "event_summary":"市民反映港龙新港城北门有摊贩占道",
                        "entities":{
                            "problem_objects":[{
                                "surface":"摊贩", "canonical":"流动摊贩",
                                "field":"case_content_clean", "evidence":"摊贩",
                            }],
                            "problem_behaviors":[{
                                "surface":"占道", "canonical":"占道经营",
                                "field":"case_content_clean", "evidence":"占道",
                            }],
                            "roads":[], "intersections":[],
                            "pois":[{
                                "surface":"港龙新港城", "canonical":"港龙新港城",
                                "field":"case_content_clean", "evidence":"港龙新港城",
                            }],
                        },
                        "discourse":{
                            "intents":[], "emotions":[],
                            "satisfaction":{"label":"unknown", "target":"", "evidence":""},
                            "urgency":{"level":"normal", "evidence":""},
                        },
                    }, ensure_ascii=False)
                    text = text[:-1] + ",}"
                    return [{
                        "text":text, "input_tokens":100, "output_tokens":80,
                        "finish_reason":"stop", "latency_ms":1,
                    } for _ in prompts]

            generator = TrailingCommaGenerator()
            report = run_semantic_extraction(
                source, tmp/"semantic.jsonl", tmp/"rejects.jsonl", tmp/"run.json", tmp/"quality.json", "unused",
                {
                    "model_id":"Qwen/Qwen3-4B", "prompt_version":"sag_semantic_v7",
                    "batch_size":1, "checkpoint_every":1,
                },
                generator=generator,
            )
            record = json.loads((tmp/"semantic.jsonl").read_text(encoding="utf-8"))
            quality = json.loads((tmp/"quality.json").read_text(encoding="utf-8"))
        self.assertEqual(generator.calls, 1)
        self.assertEqual(report["repair_requests"], 0)
        self.assertEqual(record["validation"]["status"], "accepted_with_warnings")
        self.assertIn("json_recovered_trailing_comma", record["validation"]["warnings"])
        self.assertEqual(quality["json_recovery_count"], 1)
        self.assertEqual(quality["semantic_gap_counts"], {})

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
                {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v5","batch_size":1,"checkpoint_every":1},
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
                    {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v5","batch_size":1},
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
                    {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v5","batch_size":1},
                    generator=FailingGenerator(), diagnostic_path=diagnostic,
                )
            diagnostics = [json.loads(line) for line in diagnostic.read_text(encoding="utf-8").splitlines()]
        failure = next(row for row in diagnostics if row["event"] == "model_call_failed")
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertNotIn("message", failure)
        self.assertNotIn("港龙新港城", json.dumps(diagnostics, ensure_ascii=False))

    def test_fixed_point_sanitation_resolves_new_canonical_conflict_without_repair(self):
        order = {
            "doc_id":"order_2", "title_clean":"", "case_content_clean":"路灯连续三天不亮。",
            "case_goal_clean":"", "address_detail_clean":"",
        }
        semantic = {
            "event_summary":"路灯连续三天不亮",
            "entities":{
                "problem_objects":[],
                "problem_behaviors":[{
                    "surface":"照明故障", "canonical":"照明故障",
                    "source_field":"case_content_clean", "evidence":"连续三天不亮",
                }],
                "roads":[], "intersections":[], "pois":[],
            },
            "discourse":{
                "intents":[], "emotions":[],
                "satisfaction":{"label":"unknown", "target":"", "evidence":""},
                "urgency":{"level":"normal", "evidence":""},
            },
        }
        cleaned, validation, trace = _validate_with_sanitation(order, semantic, [])
        candidate = cleaned["entities"]["problem_behaviors"][0]
        self.assertEqual(validation["status"], "accepted_with_warnings")
        self.assertEqual(candidate["surface"], "连续三天不亮")
        self.assertEqual(candidate["canonical"], "连续三天不亮")
        self.assertEqual(trace["sanitation_warnings"], [
            "aligned_surface_to_evidence:entities.problem_behaviors.0",
            "aligned_canonical_to_surface:entities.problem_behaviors.0",
        ])

    def test_invalid_overflow_candidate_does_not_crowd_out_later_valid_item(self):
        order = {
            "doc_id":"order_overflow", "title_clean":"",
            "case_content_clean":"路灯、灯杆、配电箱均有故障",
            "case_goal_clean":"", "address_detail_clean":"",
        }
        semantic = {
            "event_summary":"照明设施故障",
            "entities":{
                "problem_objects":[
                    {"surface":"虚构对象", "canonical":"虚构对象", "source_field":"case_content_clean", "evidence":"虚构对象"},
                    {"surface":"路灯", "canonical":"路灯", "source_field":"case_content_clean", "evidence":"路灯"},
                    {"surface":"灯杆", "canonical":"灯杆", "source_field":"case_content_clean", "evidence":"灯杆"},
                    {"surface":"配电箱", "canonical":"配电箱", "source_field":"case_content_clean", "evidence":"配电箱"},
                ],
                "problem_behaviors":[], "roads":[], "intersections":[], "pois":[],
            },
            "discourse":{
                "intents":[], "emotions":[],
                "satisfaction":{"label":"unknown", "target":"", "evidence":""},
                "urgency":{"level":"normal", "evidence":""},
            },
        }
        cleaned, validation, trace = _validate_with_sanitation(
            order, semantic, ["group_limit_exceeded:problem_objects"]
        )
        self.assertEqual(
            [item["canonical"] for item in cleaned["entities"]["problem_objects"]],
            ["路灯", "灯杆", "配电箱"],
        )
        self.assertEqual(validation["status"], "accepted_with_warnings")
        self.assertIn(
            "dropped_invalid_candidate:entities.problem_objects.0",
            trace["sanitation_warnings"],
        )

    def test_repairs_are_aggregated_across_primary_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"
            source.write_text("".join(
                json.dumps({
                    "doc_id":f"order_{index}",
                    "case_content_clean":"港龙新港城北门口有摊贩占道。",
                }, ensure_ascii=False) + "\n"
                for index in range(3)
            ), encoding="utf-8")
            generator = RecordingGenerator(True)
            report = run_semantic_extraction(
                source, tmp/"semantic.jsonl", tmp/"rejects.jsonl", tmp/"run.json", tmp/"quality.json", "unused",
                {
                    "model_id":"Qwen/Qwen3-4B", "prompt_version":"sag_semantic_v6",
                    "batch_size":1, "repair_batch_size":3, "checkpoint_every":1,
                    "max_new_tokens":640, "repair_max_new_tokens":768,
                },
                generator=generator,
            )
            rows = [json.loads(line) for line in (tmp/"semantic.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([len(call) for call in generator.calls], [1, 1, 1, 3])
        self.assertEqual(generator.token_limits, [640, 640, 640, 768])
        self.assertEqual(report["primary_requests"], 3)
        self.assertEqual(report["repair_requests"], 3)
        self.assertEqual(report["primary_batches"], 3)
        self.assertEqual(report["repair_batches"], 1)
        self.assertEqual(report["repair_batch_size"], 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["validation"]["repair_attempted"] for row in rows))

    def test_incremental_checkpoint_preserves_all_batched_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"
            source.write_text("".join(
                json.dumps({"doc_id":f"order_{index}", "case_content_clean":"港龙新港城北门口有摊贩占道。"}, ensure_ascii=False) + "\n"
                for index in range(5)
            ), encoding="utf-8")
            generator = RecordingGenerator(False)
            report = run_semantic_extraction(
                source, tmp/"semantic.jsonl", tmp/"rejects.jsonl", tmp/"run.json", tmp/"quality.json", "unused",
                {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v5","batch_size":2,"checkpoint_every":3},
                generator=generator,
            )
            rows = [json.loads(line) for line in (tmp/"semantic.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(generator.calls), 3)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({row["doc_id"] for row in rows}), 5)
        self.assertEqual(report["batch_size"], 2)
        self.assertGreater(report["output_tokens_per_second"], 0)

    def test_valid_primary_uses_one_request_and_resume_skips_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; self._input(source)
            paths = [tmp/name for name in ("semantic.jsonl","rejects.jsonl","run.json","quality.json")]
            config = {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v5","batch_size":8,"checkpoint_every":1}
            first = RecordingGenerator(False)
            run_semantic_extraction(source, *paths, "unused", config, generator=first)
            second = RecordingGenerator(False)
            report = run_semantic_extraction(source, *paths, "unused", config, resume=True, generator=second)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(second.calls, [])
        self.assertEqual(report["orders_processed"], 0)

    def test_resume_identity_is_backend_independent_for_same_model_and_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; self._input(source)
            paths = [tmp/name for name in ("semantic.jsonl","rejects.jsonl","run.json","quality.json")]
            base = {"model_id":"Qwen/Qwen3-4B","prompt_version":"sag_semantic_v5","batch_size":8,"checkpoint_every":1}
            run_semantic_extraction(source, *paths, "unused", {**base, "backend":"transformers"}, generator=RecordingGenerator(False))
            vllm_generator = RecordingGenerator(False)
            report = run_semantic_extraction(
                source, *paths, "unused", {**base, "backend":"vllm"},
                resume=True, generator=vllm_generator,
            )
        self.assertEqual(vllm_generator.calls, [])
        self.assertEqual(report["orders_processed"], 0)

    def test_vllm_generator_uses_offline_batch_api_and_reports_backend(self):
        captured = {}

        class FakeCuda:
            @staticmethod
            def is_available(): return False

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured.setdefault("templates", []).append((messages, kwargs))
                return "templated:" + messages[0]["content"]

        class FakeChoice:
            text = '{"event_summary":"完成"}'
            token_ids = [1, 2, 3]
            finish_reason = "stop"

        class FakeOutput:
            outputs = [FakeChoice()]
            prompt_token_ids = [4, 5]

        class FakeLLM:
            def __init__(self, **kwargs): captured["engine"] = kwargs
            def get_tokenizer(self): return FakeTokenizer()
            def generate(self, prompts, sampling, use_tqdm):
                captured["generate"] = (prompts, sampling.kwargs, use_tqdm)
                return [FakeOutput() for _ in prompts]

        class FakeSamplingParams:
            def __init__(self, **kwargs): self.kwargs = kwargs

        fake_torch = types.SimpleNamespace(cuda=FakeCuda())
        fake_vllm = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            sys.modules, {"torch": fake_torch, "vllm": fake_vllm},
        ), patch.dict(os.environ, {}, clear=False):
            generator = load_vllm_generator(tmpdir, enable_thinking=False)
            rows = generator(["工单一", "工单二"], max_new_tokens=640, temperature=0.0)
        self.assertEqual(captured["engine"]["dtype"], "float16")
        self.assertEqual(captured["engine"]["max_num_seqs"], 64)
        self.assertFalse(captured["engine"]["enable_prefix_caching"])
        self.assertFalse(captured["engine"]["enable_chunked_prefill"])
        self.assertFalse(captured["engine"]["enforce_eager"])
        self.assertEqual(captured["generate"][0], ["templated:工单一", "templated:工单二"])
        self.assertEqual(captured["generate"][1], {"temperature":0.0, "max_tokens":640})
        self.assertFalse(captured["generate"][2])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input_tokens"], 2)
        self.assertEqual(rows[0]["output_tokens"], 3)
        self.assertEqual(generator.cache_implementation, "paged")
        self.assertFalse(generator.prefix_caching)
        self.assertFalse(generator.chunked_prefill)
        self.assertFalse(generator.enforce_eager)

    def test_configured_generator_routes_vllm_without_importing_it_locally(self):
        sentinel = object()
        config = {
            "backend":"vllm", "enable_thinking":False,
            "vllm_gpu_memory_utilization":0.85, "vllm_max_model_len":4096,
            "vllm_max_num_seqs":64, "vllm_enable_prefix_caching":False,
            "vllm_enable_chunked_prefill":False, "vllm_enforce_eager":False,
        }
        environment = {
            "VLLM_GPU_MEMORY_UTILIZATION":"0.8",
            "VLLM_MAX_MODEL_LEN":"3072",
            "VLLM_MAX_NUM_SEQS":"32",
            "VLLM_ENABLE_PREFIX_CACHING":"true",
            "VLLM_ENABLE_CHUNKED_PREFILL":"false",
            "VLLM_ENFORCE_EAGER":"true",
        }
        with patch.dict(os.environ, environment), patch(
            "ragflow_style_pipeline.sag_semantic_llm.load_vllm_generator", return_value=sentinel,
        ) as loader:
            result = _load_configured_generator("models/Qwen3-4B", config)
        self.assertIs(result, sentinel)
        loader.assert_called_once_with(
            "models/Qwen3-4B", enable_thinking=False,
            gpu_memory_utilization=0.8, max_model_len=3072,
            max_num_seqs=32, enable_prefix_caching=True,
            enable_chunked_prefill=False, enforce_eager=True,
        )
        with self.assertRaisesRegex(ValueError, "unsupported_backend"):
            _load_configured_generator("unused", {"backend":"unknown"})

    def test_cli_exposes_safe_arguments(self):
        args = parse_args(["--input","safe.multiview.jsonl","--output","outputs/a.jsonl","--rejects","outputs/r.jsonl","--run-report","outputs/run.json","--quality-report","outputs/q.json","--config","config.json","--model-path","models/Qwen3-4B"])
        self.assertTrue(args.input.endswith(".multiview.jsonl"))
        self.assertEqual(args.diagnostic_log, "")
        self.assertIsNone(args.batch_size)
        overridden = parse_args([
            "--input","safe.multiview.jsonl","--output","outputs/a.jsonl","--rejects","outputs/r.jsonl",
            "--run-report","outputs/run.json","--quality-report","outputs/q.json","--config","config.json",
            "--model-path","models/Qwen3-4B","--batch-size","1","--repair-batch-size","4",
            "--backend","vllm","--diagnostic-log","outputs/d.jsonl",
        ])
        self.assertEqual(overridden.batch_size, 1)
        self.assertEqual(overridden.repair_batch_size, 4)
        self.assertEqual(overridden.backend, "vllm")
        self.assertEqual(overridden.diagnostic_log, "outputs/d.jsonl")


if __name__ == "__main__":
    unittest.main()
