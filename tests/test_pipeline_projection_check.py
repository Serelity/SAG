"""Synthetic fake-generator tests for durable extraction, resume and replay."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ragflow_style_pipeline.check import CheckError, assert_safe_value, check
from ragflow_style_pipeline.constants import REQUIRED_TSV_COLUMNS
from ragflow_style_pipeline.pipeline import PipelineError, extract, load_config
from ragflow_style_pipeline.projection import project
from ragflow_style_pipeline.work_order import prepare


VALID_ENTITY = json.dumps(
    {
        "issues": [
            {
                "objects": ["路灯"],
                "problems": ["不亮"],
                "questions": [],
                "locations": ["幸福路"],
                "requests": ["维修路灯"],
            }
        ]
    },
    ensure_ascii=False,
)
VALID_QUESTION = json.dumps(
    {
        "issues": [
            {
                "objects": ["公交卡"],
                "problems": [],
                "questions": ["办理条件"],
                "locations": [],
                "requests": [],
            }
        ]
    },
    ensure_ascii=False,
)


class QueueGenerator:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def generate(self, prompts, max_tokens):
        self.calls.append((len(prompts), max_tokens))
        if not self.batches:
            raise AssertionError("unexpected generation call")
        outputs = self.batches.pop(0)
        if isinstance(outputs, Exception):
            raise outputs
        return outputs


class PipelineProjectionCheckTests(unittest.TestCase):
    def _config(self, root: str) -> dict:
        source = Path(__file__).parents[1] / "configs" / "entity_extraction_v1.json"
        config_path = Path(root) / "config.json"
        config_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return load_config(config_path)

    def _model(self, root: str) -> Path:
        model = Path(root) / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
        (model / "tokenizer_config.json").write_text('{"chat_template":"synthetic"}', encoding="utf-8")
        (model / "model.safetensors.index.json").write_text('{"weight_map":{}}', encoding="utf-8")
        (model / "model-00001-of-00001.safetensors").write_bytes(b"synthetic-weight-name-size-only")
        return model

    def _row(self, identity, *, question=False):
        row = {column: "" for column in REQUIRED_TSV_COLUMNS}
        row.update(
            {
                "id": identity,
                "title": "公交卡咨询" if question else "路灯不亮",
                "case_content": "咨询公交卡办理条件" if question else "幸福路路灯不亮",
                "case_goal": "了解办理条件" if question else "希望维修路灯",
                "address_detail": "" if question else "幸福路",
                "service_object_type": "咨询" if question else "求助",
                "call_time": "2025-01-02 03:04:05",
            }
        )
        return row

    def _prepare(self, root: str, rows=None) -> Path:
        rows = rows or [self._row("A"), self._row("B", question=True)]
        tsv = Path(root) / "synthetic.tsv"
        header = list(REQUIRED_TSV_COLUMNS)
        lines = ["\t".join(header)]
        lines.extend("\t".join(row[column] for column in header) for row in rows)
        tsv.write_text("\n".join(lines) + "\n", encoding="utf-8-sig", newline="")
        run_dir = Path(root) / "run"
        prepare(tsv, run_dir)
        return run_dir

    def _telemetry(self, text):
        return {
            "text": text,
            "finish_reason": "stop",
            "input_tokens": 100,
            "output_tokens": 20,
            "latency_share_ms": 10.0,
            "gpu_peak_allocated_gb": 1.0,
            "gpu_peak_reserved_gb": 2.0,
        }

    def test_primary_selective_repair_projection_and_check(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._prepare(root)
            config = self._config(root)
            model = self._model(root)
            generator = QueueGenerator(
                [
                    [self._telemetry(VALID_ENTITY), self._telemetry("not json")],
                    [self._telemetry(VALID_QUESTION)],
                ]
            )
            summary = extract(run_dir, config, model, generator=generator)
            projection = project(run_dir)
            report = check(run_dir)
            entities = (run_dir / "entities.private.jsonl").read_text(encoding="utf-8").splitlines()
            rejects = (run_dir / "rejects.private.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([(2, 768), (1, 512)], generator.calls)
        self.assertEqual(1, summary["repairs_started_this_invocation"])
        self.assertEqual(2, len(entities))
        self.assertEqual([], rejects)
        self.assertEqual(2, report["entity_document_count"])
        self.assertGreater(projection["link_count"], 0)

    def test_repair_still_empty_writes_private_reject(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._prepare(root, [self._row("A")])
            config = self._config(root)
            model = self._model(root)
            empty = json.dumps({"issues": []})
            generator = QueueGenerator([["bad"], [empty]])
            extract(run_dir, config, model, generator=generator)
            project(run_dir)
            report = check(run_dir)
            reject = json.loads((run_dir / "rejects.private.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(1, report["reject_count"])
        self.assertEqual(2, reject["attempt_count"])
        self.assertEqual(2, len(reject["error_codes"]))
        self.assertNotIn("bad", json.dumps(reject))

    def test_resume_does_not_repeat_interrupted_primary(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._prepare(root, [self._row("A")])
            config = self._config(root)
            model = self._model(root)
            failing = QueueGenerator([[RuntimeError("simulated")][0]])
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                extract(run_dir, config, model, generator=failing)
            resumed = QueueGenerator([[self._telemetry(VALID_ENTITY)]])
            summary = extract(run_dir, config, model, generator=resumed, resume=True)
            project(run_dir)
            report = check(run_dir)
        self.assertEqual([(1, 768)], failing.calls)
        self.assertEqual([(1, 512)], resumed.calls)
        self.assertEqual(1, summary["repairs_started_this_invocation"])
        self.assertEqual(1, report["entity_document_count"])

    def test_resume_after_interrupted_repair_rejects_without_third_generation(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._prepare(root, [self._row("A")])
            config = self._config(root)
            model = self._model(root)
            failing = QueueGenerator([["bad"], RuntimeError("repair-crash")])
            with self.assertRaisesRegex(RuntimeError, "repair-crash"):
                extract(run_dir, config, model, generator=failing)
            no_calls = QueueGenerator([])
            extract(run_dir, config, model, generator=no_calls, resume=True)
            project(run_dir)
            report = check(run_dir)
        self.assertEqual([], no_calls.calls)
        self.assertEqual(1, report["reject_count"])

    def test_resume_truncates_only_unterminated_final_fragments(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._prepare(root, [self._row("A")])
            config = self._config(root)
            model = self._model(root)
            failing = QueueGenerator([RuntimeError("primary-crash")])
            with self.assertRaisesRegex(RuntimeError, "primary-crash"):
                extract(run_dir, config, model, generator=failing)
            with (run_dir / "entities.private.jsonl").open("ab") as output:
                output.write(b'{"partial":')
            resumed = QueueGenerator([[self._telemetry(VALID_ENTITY)]])
            extract(run_dir, config, model, generator=resumed, resume=True)
            project(run_dir)
            report = check(run_dir)
        self.assertEqual([(1, 512)], resumed.calls)
        self.assertEqual(1, report["entity_document_count"])

    def test_resume_rejects_changed_contract(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._prepare(root, [self._row("A")])
            config = self._config(root)
            model = self._model(root)
            generator = QueueGenerator([[self._telemetry(VALID_ENTITY)]])
            extract(run_dir, config, model, generator=generator)
            changed = dict(config)
            changed["batch_size"] = 16
            with self.assertRaisesRegex(PipelineError, "resume_contract_mismatch"):
                extract(run_dir, changed, model, generator=QueueGenerator([]), resume=True)

    def test_check_detects_tampered_grounding(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._prepare(root, [self._row("A")])
            config = self._config(root)
            model = self._model(root)
            extract(
                run_dir,
                config,
                model,
                generator=QueueGenerator([[self._telemetry(VALID_ENTITY)]]),
            )
            project(run_dir)
            entity_path = run_dir / "entities.private.jsonl"
            entity = json.loads(entity_path.read_text(encoding="utf-8"))
            entity["issues"][0]["objects"][0]["mentions"][0]["end"] += 1
            entity_path.write_text(json.dumps(entity, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CheckError, "ungrounded_mention"):
                check(run_dir)

    def test_safe_validator_rejects_identity_text_and_private_paths(self):
        for value in (
            {"doc_id": "secret"},
            {"nested": {"surface": "secret"}},
            {"artifact": "documents.private.jsonl"},
        ):
            with self.subTest(value=value), self.assertRaises(CheckError):
                assert_safe_value(value)


if __name__ == "__main__":
    unittest.main()
