import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_db import build_sag_db_from_orders, source_order_row
from ragflow_style_pipeline.sag_eval import (
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


if __name__ == "__main__":
    unittest.main()
