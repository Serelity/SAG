import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_db import build_sag_db
from ragflow_style_pipeline.sag_query import query_sag_db
from ragflow_style_pipeline.sag_semantic_audit import replay_candidate_ledger
from ragflow_style_pipeline.sag_semantic_llm import run_semantic_extraction
from ragflow_style_pipeline.sag_semantic_projection import project_semantics_file


def output_value():
    return {
        "event_summary": "人民路路灯不亮；幸福小区垃圾堆积",
        "issues": [
            {
                "time_scope": "current",
                "objects": [{"surface": "路灯", "field": "case_content_clean", "evidence": "路灯"}],
                "problem_behaviors": [{"surface": "不亮", "field": "case_content_clean", "evidence": "不亮"}],
                "question_focus": [],
                "request_actions": [{"surface": "维修路灯", "field": "case_content_clean", "evidence": "维修路灯"}],
                "locations": [{"type": "road", "surface": "人民路", "field": "case_content_clean", "evidence": "人民路"}],
            },
            {
                "time_scope": "current",
                "objects": [{"surface": "垃圾", "field": "case_content_clean", "evidence": "垃圾"}],
                "problem_behaviors": [{"surface": "堆积", "field": "case_content_clean", "evidence": "堆积"}],
                "question_focus": [],
                "request_actions": [{"surface": "清理垃圾", "field": "case_content_clean", "evidence": "清理垃圾"}],
                "locations": [{"type": "poi", "surface": "幸福小区", "field": "case_content_clean", "evidence": "幸福小区"}],
            },
        ],
        "discourse": {
            "intents": [{"label": "求助", "field": "case_content_clean", "evidence": "希望"}],
            "emotions": [],
            "satisfaction": {"label": "unknown", "target": "", "field": "", "evidence": ""},
            "urgency": {"level": "normal", "field": "", "evidence": ""},
        },
    }


class Generator:
    def __init__(self, fail_primary=False):
        self.calls = []
        self.fail_primary = fail_primary

    def __call__(self, prompts, max_new_tokens, temperature):
        self.calls.append((list(prompts), max_new_tokens))
        prompt_text = "\n".join(
            item.get("content", "") for item in prompts[0]
        ) if isinstance(prompts[0], list) else str(prompts[0])
        repair = "你是 JSON 修复器" in prompt_text
        rows = []
        for _prompt in prompts:
            text = '{"event_summary":' if self.fail_primary and not repair else json.dumps(output_value(), ensure_ascii=False)
            rows.append({
                "text": text, "input_tokens": 300, "output_tokens": 180,
                "finish_reason": "stop", "latency_ms": 1,
            })
        return rows


class TestIssuePipeline(unittest.TestCase):
    def config(self):
        return {
            "schema_version": "3.0",
            "output_schema_version": "sag_semantic_issue_output_v1",
            "prompt_version": "sag_semantic_v8_dev1",
            "model_id": "Qwen/Qwen3-4B", "batch_size": 1,
            "repair_batch_size": 1, "max_new_tokens": 1024,
            "repair_max_new_tokens": 768, "max_repairs_per_order": 1,
            "checkpoint_every": 1,
        }

    def test_one_primary_and_at_most_one_whole_record_repair(self):
        text = "人民路路灯不亮，希望维修路灯；幸福小区垃圾堆积，希望清理垃圾"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "input.jsonl"
            source.write_text(json.dumps({"doc_id": "d1", "case_content_clean": text}, ensure_ascii=False) + "\n", encoding="utf-8")
            generator = Generator(fail_primary=True)
            report = run_semantic_extraction(
                source, root / "semantic.jsonl", root / "rejects.jsonl",
                root / "run.json", root / "quality.json", "unused",
                self.config(), generator=generator,
            )
            record = json.loads((root / "semantic.jsonl").read_text(encoding="utf-8"))
            quality = json.loads((root / "quality.json").read_text(encoding="utf-8"))
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual([call[1] for call in generator.calls], [1024, 768])
        self.assertEqual(report["primary_requests"], 1)
        self.assertEqual(report["repair_requests"], 1)
        self.assertEqual(report["output_schema_version"], "sag_semantic_issue_output_v1")
        self.assertEqual(report["validator_version"], "sag_semantic_issue_validator_v1")
        self.assertEqual(report["projection_version"], "sag_semantic_issue_projection_v1")
        self.assertEqual(record["output_schema_version"], "sag_semantic_issue_output_v1")
        self.assertEqual(len(record["issues"]), 2)
        self.assertNotIn("entities", record)
        self.assertEqual(quality["issue_count_distribution"], {"2": 1})

    def test_v8_candidate_ledger_replays_with_issue_validator(self):
        text = "人民路路灯不亮，希望维修路灯；幸福小区垃圾堆积，希望清理垃圾"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "input.jsonl"
            source.write_text(json.dumps({"doc_id": "d1", "case_content_clean": text}, ensure_ascii=False) + "\n", encoding="utf-8")
            ledger = root / "candidates.private.jsonl"
            run_semantic_extraction(
                source, root / "semantic.jsonl", root / "rejects.jsonl",
                root / "run.json", root / "quality.json", "unused",
                self.config(), generator=Generator(), candidate_ledger_path=ledger,
            )
            candidate = json.loads(ledger.read_text(encoding="utf-8"))
            rows, report = replay_candidate_ledger(source, ledger)
        self.assertEqual(candidate["output_schema_version"], "sag_semantic_issue_output_v1")
        self.assertEqual(rows[0]["validator_version"], "sag_semantic_issue_validator_v1")
        self.assertEqual(rows[0]["output_schema_version"], "sag_semantic_issue_output_v1")
        self.assertEqual(len(rows[0]["semantic"]["issues"]), 2)
        self.assertEqual(report["records_replayed"], 1)

    def test_issue_projection_prevents_cross_issue_and_seed(self):
        text = "人民路路灯不亮，希望维修路灯；幸福小区垃圾堆积，希望清理垃圾"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "input.jsonl"
            source.write_text(json.dumps({
                "doc_id": "d1", "case_content_clean": text,
                "metadata": {"area_code_area": "钟楼区"},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            run_semantic_extraction(
                source, root / "semantic.jsonl", root / "rejects.jsonl",
                root / "run.json", root / "quality.json", "unused",
                self.config(), generator=Generator(),
            )
            projection = project_semantics_file(
                root / "semantic.jsonl", source, root / "links.jsonl",
                root / "disc.jsonl", root / "events.jsonl",
            )
            build_sag_db(
                source, root / "sag.duckdb", entity_links_jsonl=root / "links.jsonl",
                semantic_events_jsonl=root / "events.jsonl", discourse_jsonl=root / "disc.jsonl",
            )
            wrong = query_sag_db(root / "sag.duckdb", {
                "seed_entities": [
                    {"entity_type": "problem_object", "values": ["路灯"]},
                    {"entity_type": "problem_behavior", "values": ["堆积"]},
                ],
                "seed_group_operator": "AND", "expansion": {"enabled": False},
            })
            right = query_sag_db(root / "sag.duckdb", {
                "seed_entities": [
                    {"entity_type": "problem_object", "values": ["路灯"]},
                    {"entity_type": "problem_behavior", "values": ["不亮"]},
                ],
                "seed_group_operator": "AND", "expansion": {"enabled": False},
            })
            import duckdb
            with duckdb.connect(str(root / "sag.duckdb")) as conn:
                rows = conn.execute(
                    "select event_id, entity_type, normalized_value, source_channel "
                    "from sag_event_entity_links order by event_id, entity_type, normalized_value"
                ).fetchall()
        self.assertEqual(projection["events"], 2)
        self.assertEqual(projection["issue_events"], 2)
        self.assertEqual(wrong, [])
        self.assertEqual(len(right), 1)
        self.assertTrue(right[0]["event_id"].endswith("::issue::1"))
        # Metadata area is copied to both issues; regex text extraction is not.
        self.assertEqual(sum(row[1:] == ("area", "钟楼区", "metadata") for row in rows), 2)


if __name__ == "__main__":
    unittest.main()
