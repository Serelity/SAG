import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_db import build_sag_db_from_orders, load_entity_links_jsonl, source_order_row
from ragflow_style_pipeline.sag_eval import (
    build_entity_eval_samples,
    build_manual_eval_samples,
    build_weak_gold_doc_ids,
    evaluate_sag_results,
    precision_at_k,
    recall_at_k,
)
from ragflow_style_pipeline.sag_query import query_sag_db


def _skip_without_duckdb(testcase):
    try:
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        testcase.skipTest("duckdb is not installed in the local test runtime")


class TestSagEval(unittest.TestCase):
    def test_precision_and_recall_at_k(self):
        results = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]
        gold = {"a", "c", "x"}

        self.assertAlmostEqual(precision_at_k(results, gold, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(results, gold, 2), 1 / 3)
        self.assertAlmostEqual(recall_at_k(results, gold, 3), 2 / 3)

    def test_evaluate_sag_results_uses_weak_gold(self):
        _skip_without_duckdb(self)
        orders = [
            source_order_row(
                {
                    "id": "1",
                    "order_id": "ORD001",
                    "case_content": "流动摊贩占道经营",
                    "case_goal": "希望处理",
                    "call_time": "2024-05-01 10:00:00",
                    "case_accord_type_three_name": "无照经营游商",
                }
            ),
            source_order_row(
                {
                    "id": "2",
                    "order_id": "ORD002",
                    "case_content": "咨询医保办理条件",
                    "case_goal": "希望了解",
                    "call_time": "2024-05-02 10:00:00",
                    "case_accord_type_three_name": "职工医疗保险",
                }
            ),
        ]
        config = {
            "query_name": "stall",
            "seed_entities": [
                {"entity_type": "problem_object", "values": ["流动摊贩", "摊贩"], "operator": "OR"},
                {"entity_type": "problem_behavior", "values": ["占道经营"], "operator": "OR"},
            ],
            "seed_group_operator": "AND",
            "filters": {"call_month_gte": "2024-01", "call_month_lte": "2024-12"},
            "expansion": {"enabled": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sag.duckdb"
            build_sag_db_from_orders(orders, db_path)
            results = query_sag_db(db_path, config)
            gold = build_weak_gold_doc_ids(db_path, config)
            metrics = evaluate_sag_results(db_path, config, results)
            samples = build_manual_eval_samples(db_path, results, limit=1)

        self.assertEqual(len(gold), 1)
        self.assertEqual(metrics["weak_precision@10"], 1.0)
        self.assertEqual(metrics["weak_recall@100"], 1.0)
        self.assertEqual(len(samples), 1)
        self.assertIn("label", samples[0])
        self.assertEqual(samples[0]["label"], "")

    def test_build_entity_eval_samples_is_stratified(self):
        _skip_without_duckdb(self)
        orders = [
            source_order_row(
                {
                    "id": "1",
                    "case_content": "市民反映广成路有流动摊贩占道经营。",
                    "call_time": "2024-05-01 10:00:00",
                    "area_code_area": "钟楼区",
                }
            ),
            source_order_row(
                {
                    "id": "2",
                    "case_content": "市民反映清潭菜场附近有商贩摆摊。",
                    "call_time": "2024-05-02 10:00:00",
                    "area_code_area": "钟楼区",
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            links_path = tmp / "links.jsonl"
            links_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "doc_id": orders[0]["doc_id"],
                                "entity_type": "road",
                                "entity_value": "广成路",
                                "normalized_value": "广成路",
                                "source_field": "case_content_clean",
                                "source_channel": "llm",
                                "confidence": 0.9,
                                "matched_text": "广成路",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "doc_id": orders[1]["doc_id"],
                                "entity_type": "poi",
                                "entity_value": "清潭菜场",
                                "normalized_value": "清潭菜场",
                                "source_field": "case_content_clean",
                                "source_channel": "llm",
                                "confidence": 0.9,
                                "matched_text": "清潭菜场",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            db_path = tmp / "sag.duckdb"
            build_sag_db_from_orders(orders, db_path, extra_entity_links_by_doc=load_entity_links_jsonl(links_path))
            samples = build_entity_eval_samples(db_path, limit=20)

        observed_types = {sample["entity_type"] for sample in samples}
        self.assertIn("area", observed_types)
        self.assertIn("road", observed_types)
        self.assertIn("poi", observed_types)


if __name__ == "__main__":
    unittest.main()
