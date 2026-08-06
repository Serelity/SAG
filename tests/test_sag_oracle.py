import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_oracle import (
    build_oracle_sag_db,
    evaluate_oracle_retrieval,
    load_oracle_queries,
    query_oracle_graph,
)


def _skip_without_duckdb(testcase):
    try:
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        testcase.skipTest("duckdb is not installed in the local test runtime")


class TestSagOracle(unittest.TestCase):
    def _gold_row(self, doc_id, content, issues, records=3):
        return {
            "schema": "sag_issue_gold_v2",
            "private": True,
            "subset": "challenge",
            "doc_id": doc_id,
            "content_hash": f"sha256:{doc_id}",
            "manifest_provenance": {
                "schema": "sag_semantic_eval_manifest_v2",
                "records": records,
                "content_sha256": "sha256:" + "a" * 64,
            },
            "clean_fields": {
                "title_clean": "",
                "case_content_clean": content,
                "case_goal_clean": "",
                "address_detail_clean": "",
            },
            "metadata": {"service_object_type": "求助"},
            "issues": issues,
            "declared_intents": [],
            "direct_emotions": [],
            "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
            "urgency": {"level": "normal", "evidence": ""},
            "annotation": {
                "annotator": "adjudicator", "status": "adjudicated", "notes": "",
            },
        }

    @staticmethod
    def _mention(surface):
        return {
            "surface": surface,
            "field": "case_content_clean",
            "evidence": surface,
        }

    def _write_gold_and_queries(self, root):
        cross = self._gold_row(
            "a-cross-issue",
            "甲路广告牌脱落，乙路路灯不亮",
            [
                {
                    "mode": "problem", "time_scope": "current",
                    "objects": [self._mention("广告牌")],
                    "predicates": [self._mention("脱落")],
                    "actions": [],
                    "locations": [{**self._mention("甲路"), "type": "road"}],
                },
                {
                    "mode": "problem", "time_scope": "current",
                    "objects": [self._mention("路灯")],
                    "predicates": [self._mention("不亮")],
                    "actions": [],
                    "locations": [{**self._mention("乙路"), "type": "road"}],
                },
            ],
        )
        relevant = self._gold_row(
            "b-relevant",
            "丙路广告牌不亮",
            [{
                "mode": "problem", "time_scope": "current",
                "objects": [self._mention("广告牌")],
                "predicates": [self._mention("不亮")],
                "actions": [],
                "locations": [{**self._mention("丙路"), "type": "road"}],
            }],
        )
        noise = self._gold_row(
            "c-expansion-noise",
            "甲路井盖破损",
            [{
                "mode": "problem", "time_scope": "current",
                "objects": [self._mention("井盖")],
                "predicates": [self._mention("破损")],
                "actions": [],
                "locations": [{**self._mention("甲路"), "type": "road"}],
            }],
        )
        gold_path = root / "gold.private.jsonl"
        gold_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (cross, relevant, noise)),
            encoding="utf-8",
        )
        query = {
            "schema": "sag_oracle_query_relevance_v1",
            "private": True,
            "query_id": "cross-issue-query",
            "seed_entities": [
                {"entity_type": "problem_object", "values": ["广告牌"]},
                {"entity_type": "problem_behavior", "values": ["不亮"]},
            ],
            "seed_group_operator": "AND",
            "expansion": {
                "enabled": True,
                "frontier_entity_types": ["road"],
                "max_expanded_docs": 10,
            },
            "relevance": [{"doc_id": "b-relevant", "grade": 3}],
        }
        query_path = root / "queries.private.jsonl"
        query_path.write_text(json.dumps(query, ensure_ascii=False) + "\n", encoding="utf-8")
        return gold_path, query_path

    def test_oracle_issue_projection_removes_cross_issue_seed_and_expansion(self):
        _skip_without_duckdb(self)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_path, query_path = self._write_gold_and_queries(root)
            flat_db = root / "flat.duckdb"
            issue_db = root / "issue.duckdb"
            flat_build = build_oracle_sag_db(gold_path, flat_db, "flat")
            issue_build = build_oracle_sag_db(gold_path, issue_db, "issue-aware")
            query = load_oracle_queries(query_path)[0]
            flat_mode, flat_results = query_oracle_graph(flat_db, query)
            issue_mode, issue_results = query_oracle_graph(issue_db, query)
            report, traces = evaluate_oracle_retrieval(
                flat_db, issue_db, query_path, cutoffs=(1, 3)
            )

        self.assertEqual(flat_build["events"], 3)
        self.assertEqual(issue_build["events"], 4)
        self.assertEqual(flat_mode, "flat")
        self.assertEqual(issue_mode, "issue_aware")
        self.assertEqual(
            [(row["doc_id"], row["match_stage"]) for row in flat_results],
            [
                ("a-cross-issue", "seed_entity"),
                ("b-relevant", "seed_entity"),
                ("c-expansion-noise", "one_hop_expansion"),
            ],
        )
        self.assertEqual(
            [(row["doc_id"], row["match_stage"]) for row in issue_results],
            [("b-relevant", "seed_entity")],
        )
        self.assertEqual(report["flat"]["precision@1"], 0.0)
        self.assertEqual(report["issue_aware"]["precision@1"], 1.0)
        self.assertEqual(report["flat"]["false_seed_rate"], 0.5)
        self.assertEqual(report["issue_aware"]["false_seed_rate"], 0.0)
        self.assertEqual(report["flat"]["erroneous_expansion_rate"], 1.0)
        self.assertIsNone(report["issue_aware"]["erroneous_expansion_rate"])
        self.assertEqual(
            report["paired_retrieval_effects"]["removed_irrelevant_flat_seed_docs"], 1
        )
        self.assertEqual(
            report["paired_retrieval_effects"]["lost_relevant_flat_seed_docs"], 0
        )
        self.assertEqual(
            report["paired_retrieval_effects"]["removed_irrelevant_flat_expansion_docs"], 1
        )
        self.assertEqual(report["graph_structure"]["flat"]["events_per_doc"]["max"], 1)
        self.assertEqual(report["graph_structure"]["issue_aware"]["events_per_doc"]["max"], 2)
        self.assertEqual(len(traces), 1)
        self.assertTrue(traces[0]["private"])
        serialized_report = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("a-cross-issue", serialized_report)
        self.assertNotIn("b-relevant", serialized_report)
        self.assertNotIn("cross-issue-query", serialized_report)

    def test_oracle_query_validation_rejects_discourse_frontier(self):
        query = {
            "schema": "sag_oracle_query_relevance_v1",
            "private": True,
            "query_id": "q1",
            "seed_entities": [{"entity_type": "problem_object", "values": ["路灯"]}],
            "expansion": {"frontier_entity_types": ["satisfaction"]},
            "relevance": [{"doc_id": "d1", "grade": 1}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.private.jsonl"
            path.write_text(json.dumps(query, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid_frontier_types"):
                load_oracle_queries(path)
            query["expansion"] = {"enabled": "false"}
            path.write_text(json.dumps(query, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid_expansion_enabled"):
                load_oracle_queries(path)


if __name__ == "__main__":
    unittest.main()
