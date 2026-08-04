import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_db import build_sag_db_from_orders, source_order_row
from ragflow_style_pipeline.sag_query import analyze_sag_query, query_sag_db, score_sag_result


def _skip_without_duckdb(testcase):
    try:
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        testcase.skipTest("duckdb is not installed in the local test runtime")


def _build_test_db(tmpdir):
    orders = [
        source_order_row(
            {
                "id": "1",
                "order_id": "ORD001",
                "case_content": "市民反映钟楼区永红街道广成路有流动摊贩占道经营，影响通行。",
                "case_goal": "希望处理",
                "address_detail": "广成路",
                "call_time": "2024-05-01 10:00:00",
                "area_code_area": "钟楼区",
                "area_code_street": "",
                "case_accord_type_three_name": "无照经营游商",
            }
        ),
        source_order_row(
            {
                "id": "2",
                "order_id": "ORD002",
                "case_content": "市民反映广成路附近有夜间噪声。",
                "case_goal": "希望处理",
                "address_detail": "广成路",
                "call_time": "2024-05-02 10:00:00",
                "area_code_area": "钟楼区",
                "area_code_street": "",
                "case_accord_type_three_name": "社会生活噪声",
            }
        ),
        source_order_row(
            {
                "id": "3",
                "order_id": "ORD003",
                "case_content": "咨询医保办理条件。",
                "case_goal": "希望了解政策",
                "call_time": "2024-05-03 10:00:00",
                "area_code_area": "天宁区",
                "case_accord_type_three_name": "职工医疗保险",
            }
        ),
    ]
    db_path = Path(tmpdir) / "sag.duckdb"
    build_sag_db_from_orders(orders, db_path)
    return db_path


class TestSagQuery(unittest.TestCase):
    def test_score_prioritizes_seed_over_expansion(self):
        seed_score = score_sag_result("seed_entity", 2, 0, 1.4)
        expanded_score = score_sag_result("one_hop_expansion", 0, 2, 1.8)

        self.assertGreater(seed_score, expanded_score)

    def test_query_sag_db_returns_seed_and_one_hop_expansion(self):
        _skip_without_duckdb(self)
        config = {
            "query_name": "stall",
            "seed_entities": [
                {"entity_type": "problem_object", "values": ["流动摊贩", "摊贩"], "operator": "OR"},
                {"entity_type": "problem_behavior", "values": ["占道经营"], "operator": "OR"},
            ],
            "seed_group_operator": "AND",
            "filters": {"call_month_gte": "2024-01", "call_month_lte": "2024-12"},
            "expansion": {
                "enabled": True,
                "max_hops": 1,
                "frontier_entity_types": ["road"],
                "max_expanded_events": 10,
            },
            "representative_limit": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _build_test_db(tmpdir)
            results = query_sag_db(db_path, config)

        self.assertGreaterEqual(len(results), 2)
        self.assertEqual(results[0]["match_stage"], "seed_entity")
        self.assertIn("problem_object", results[0]["matched_entities"])
        stages = {result["match_stage"] for result in results}
        self.assertIn("one_hop_expansion", stages)

    def test_query_filters_seed_events_by_discourse(self):
        _skip_without_duckdb(self)
        orders = [
            source_order_row({"id":"11","order_id":"D11","case_content":"和平路路灯不亮","address_detail":"和平路"}),
            source_order_row({"id":"12","order_id":"D12","case_content":"和平路道路咨询","address_detail":"和平路"}),
        ]
        discourse = {
            orders[0]["doc_id"]: {"doc_id":orders[0]["doc_id"],"inferred_intents_json":"[\"投诉\"]","satisfaction":"dissatisfied","urgency":"high"},
            orders[1]["doc_id"]: {"doc_id":orders[1]["doc_id"],"inferred_intents_json":"[\"咨询\"]","satisfaction":"satisfied","urgency":"normal"},
        }
        config = {
            "seed_entities":[{"entity_type":"road","values":["和平路"]}],
            "filters":{"intent":"投诉","satisfaction":"dissatisfied","urgency_in":["high","critical"]},
            "expansion":{"enabled":False},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "discourse.duckdb"
            build_sag_db_from_orders(orders, db_path, discourse_by_doc=discourse)
            results = query_sag_db(db_path, config)
        self.assertEqual([row["doc_id"] for row in results], [orders[0]["doc_id"]])

    def test_discourse_cannot_be_used_as_frontier(self):
        _skip_without_duckdb(self)
        config = {
            "seed_entities":[{"entity_type":"road","values":["广成路"]}],
            "expansion":{"enabled":True,"frontier_entity_types":["satisfaction"]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _build_test_db(tmpdir)
            with self.assertRaisesRegex(ValueError, "invalid_frontier_entity_types:satisfaction"):
                query_sag_db(db_path, config)


class TestSagReport(unittest.TestCase):
    def test_analyze_sag_query_reports_statistics_and_metadata_recovery(self):
        _skip_without_duckdb(self)
        config = {
            "query_name": "stall",
            "seed_entities": [
                {"entity_type": "problem_object", "values": ["流动摊贩", "摊贩"], "operator": "OR"},
                {"entity_type": "problem_behavior", "values": ["占道经营"], "operator": "OR"},
            ],
            "seed_group_operator": "AND",
            "filters": {"call_month_gte": "2024-01", "call_month_lte": "2024-12"},
            "expansion": {
                "enabled": True,
                "max_hops": 1,
                "frontier_entity_types": ["road"],
                "max_expanded_events": 10,
            },
            "representative_limit": 5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = _build_test_db(tmpdir)
            report = analyze_sag_query(db_path, config)

        self.assertEqual(report["query"]["query_name"], "stall")
        self.assertGreaterEqual(report["matched_orders"], 2)
        self.assertIn("by_month", report["statistics"])
        self.assertIn("road", report["entity_coverage"])
        self.assertIn("metadata_street_missing", report["metadata_recovery"])
        self.assertGreaterEqual(report["metadata_recovery"]["metadata_street_missing_but_text_road_found"], 1)
        self.assertGreaterEqual(len(report["representative_cases"]), 1)


if __name__ == "__main__":
    unittest.main()
