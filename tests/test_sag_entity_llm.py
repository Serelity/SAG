import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ragflow_style_pipeline.sag_entity_llm import (
    build_extraction_prompt,
    candidate_to_link,
    extract_links_from_llm_response,
    parse_llm_json,
    run_extraction,
)


class TestSagEntityLlm(unittest.TestCase):
    def test_prompt_limits_entity_types_to_sag_retrieval_schema(self):
        order = {
            "doc_id": "order_a",
            "title_clean": "流动摊贩占道",
            "case_content_clean": "市民反映广成路有流动摊贩占道经营。",
            "case_goal_clean": "希望处理",
            "address_detail_clean": "",
        }
        prompt = build_extraction_prompt(order, {"max_text_chars": 1000})

        self.assertIn("problem_object", prompt)
        self.assertIn("problem_behavior", prompt)
        self.assertIn("intersection", prompt)
        self.assertIn("不要输出满意度", prompt)
        self.assertIn("只输出 JSON", prompt)

    def test_parse_llm_json_extracts_first_json_object(self):
        parsed = parse_llm_json(
            '```json\n{"entities":[{"entity_type":"road","entity_value":"广成路","source_field":"case_content_clean","evidence_span":"广成路","confidence":0.9}]}\n```'
        )

        self.assertEqual(parsed["entities"][0]["entity_value"], "广成路")

    def test_extract_links_rejects_missing_evidence_and_generic_noise(self):
        order = {
            "doc_id": "order_a",
            "case_content_clean": "市民反映广成路有流动摊贩占道经营。",
            "case_goal_clean": "",
            "title_clean": "",
            "address_detail_clean": "",
        }
        response = """
        {
          "entities": [
            {"entity_type":"road","entity_value":"广成路","source_field":"case_content_clean","evidence_span":"广成路","confidence":0.91},
            {"entity_type":"road","entity_value":"道路","source_field":"case_content_clean","evidence_span":"道路","confidence":0.91},
            {"entity_type":"poi","entity_value":"不存在市场","source_field":"case_content_clean","evidence_span":"不存在市场","confidence":0.91}
          ]
        }
        """

        links, rejects = extract_links_from_llm_response(order, response, {"min_confidence": 0.55})

        self.assertEqual([(link.entity_type, link.entity_value) for link in links], [("road", "广成路")])
        self.assertEqual([reject["reason"] for reject in rejects], ["generic_entity_value", "missing_evidence_span"])

    def test_candidate_to_link_uses_llm_source_channel(self):
        link = candidate_to_link(
            "order_a",
            {
                "entity_type": "problem_behavior",
                "entity_value": "挡住人行道",
                "source_field": "case_content_clean",
                "evidence_span": "挡住人行道",
                "confidence": 0.8,
            },
        )

        self.assertEqual(link.entity_type, "problem_behavior")
        self.assertEqual(link.entity_value, "占道经营")
        self.assertEqual(link.source_channel, "llm")

    def test_run_extraction_batches_generation_and_reports_progress(self):
        calls = []

        def fake_generator(prompts, max_new_tokens=512, temperature=0.0):
            calls.append(list(prompts))
            return [
                json.dumps(
                    {
                        "entities": [
                            {
                                "entity_type": "poi",
                                "entity_value": "SharedPlace",
                                "source_field": "case_content_clean",
                                "evidence_span": "SharedPlace",
                                "confidence": 0.9,
                            }
                        ]
                    }
                )
                for _prompt in prompts
            ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "orders.tsv"
            output_path = tmp / "links.jsonl"
            rejects_path = tmp / "rejects.jsonl"
            input_path.write_text(
                "id\tcase_content\n"
                "1\tSharedPlace has one issue\n"
                "2\tSharedPlace has two issues\n"
                "3\tSharedPlace has three issues\n"
                "4\tSharedPlace has four issues\n"
                "5\tSharedPlace has five issues\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()
            config = {
                "batch_size": 2,
                "progress_every": 2,
                "max_text_chars": 100,
                "max_new_tokens": 64,
                "temperature": 0.0,
                "min_confidence": 0.55,
            }

            with patch("ragflow_style_pipeline.sag_entity_llm.load_local_generator", return_value=fake_generator):
                with redirect_stdout(stdout):
                    summary = run_extraction(input_path, output_path, rejects_path, "unused-model", config, limit=5)

            progress_rows = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
            written_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([len(call) for call in calls], [2, 2, 1])
        self.assertEqual(summary["orders_processed"], 5)
        self.assertEqual(summary["links_written"], 5)
        self.assertEqual(len(written_rows), 5)
        self.assertTrue(any(row["processed"] == 2 and row["total"] == 5 for row in progress_rows))
        self.assertTrue(any(row["processed"] == 5 and row["done"] for row in progress_rows))
        self.assertTrue(all("eta_seconds" in row for row in progress_rows))


if __name__ == "__main__":
    unittest.main()
