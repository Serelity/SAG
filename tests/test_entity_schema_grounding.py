"""Synthetic tests for the sole model schema and exact Python grounding."""

from __future__ import annotations

import unittest

from ragflow_style_pipeline.entity_prompt import model_view, prompt_fingerprint
from ragflow_style_pipeline.entity_schema import EntitySchemaError, parse_model_output
from ragflow_style_pipeline.grounding import ground_payload


EMPTY_ISSUE = {
    "objects": [],
    "problems": [],
    "questions": [],
    "locations": [],
    "requests": [],
}


class EntitySchemaGroundingTests(unittest.TestCase):
    def _document(self):
        return {
            "doc_id": "order_synthetic",
            "content_hash": "sha256:synthetic",
            "title_clean": "路灯不亮咨询",
            "case_content_clean": "幸福路路灯不亮，幸福路另一路灯不亮",
            "case_goal_clean": "希望维修路灯，并咨询维修进度",
            "address_detail_clean": "幸福路",
        }

    def test_accepts_only_five_string_arrays(self):
        parsed = parse_model_output(
            '{"issues":[{"objects":["路灯"],"problems":["不亮"],'
            '"questions":["维修进度"],"locations":["幸福路"],"requests":["维修路灯"]}]}'
        )
        self.assertEqual("路灯", parsed["issues"][0]["objects"][0])
        invalid = [
            '{"issues":[{"objects":[],"problems":[],"questions":[],"locations":[]}]}',
            '{"issues":[{"objects":[{"text":"路灯"}],"problems":[],"questions":[],"locations":[],"requests":[]}]}',
            '{"issues":[],"summary":"extra"}',
            '{"issues":NaN}',
            "{'issues': []}",
            '{"issues":[}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(EntitySchemaError):
                parse_model_output(raw)

    def test_only_conservative_json_recovery(self):
        raw = """```json
        {"issues":[{"objects":["路灯,"],"problems":[],"questions":[],"locations":[],"requests":[],},],}
        ```"""
        parsed = parse_model_output(raw)
        self.assertEqual("路灯,", parsed["issues"][0]["objects"][0])
        with self.assertRaisesRegex(EntitySchemaError, "trailing_content"):
            parse_model_output('{"issues":[]} explanation')
        with self.assertRaisesRegex(EntitySchemaError, "duplicate_json_key"):
            parse_model_output('{"issues":[],"issues":[]}')

    def test_exact_offsets_all_occurrences_and_evidence(self):
        payload = {
            "issues": [
                {
                    "objects": ["路灯", "路灯"],
                    "problems": ["不亮", "原文没有"],
                    "questions": ["维修进度"],
                    "locations": ["幸福路"],
                    "requests": ["维修路灯"],
                }
            ]
        }
        grounded = ground_payload(self._document(), payload)
        issue = grounded["issues"][0]
        locations = issue["locations"][0]["mentions"]
        self.assertEqual(3, len(locations))
        for mention in locations:
            field_text = self._document()[mention["field"]]
            self.assertEqual("幸福路", field_text[mention["start"] : mention["end"]])
            self.assertEqual("幸福路", mention["evidence"])
        self.assertEqual(1, grounded["grounding_stats"]["dropped_candidates"])
        self.assertEqual(1, grounded["grounding_stats"]["duplicate_candidates"])

    def test_drops_pii_placeholders_punctuation_and_empty_issues(self):
        document = self._document()
        document["case_content_clean"] += "，联系电话[手机号]"
        issue = dict(EMPTY_ISSUE)
        issue["objects"] = ["[手机号]", "手机号", "，", " ", "不存在"]
        grounded = ground_payload(document, {"issues": [issue]})
        self.assertEqual([], grounded["issues"])
        self.assertEqual(1, grounded["grounding_stats"]["empty_issues"])

    def test_issue_id_ignores_issue_order_and_member_order(self):
        first = {
            "issues": [
                {
                    "objects": ["路灯"],
                    "problems": ["不亮"],
                    "questions": [],
                    "locations": ["幸福路"],
                    "requests": [],
                }
            ]
        }
        second = {
            "issues": [
                {
                    "objects": ["路灯"],
                    "problems": ["不亮"],
                    "questions": [],
                    "locations": ["幸福路"],
                    "requests": [],
                },
                dict(EMPTY_ISSUE),
            ]
        }
        first_id = ground_payload(self._document(), first)["issues"][0]["issue_id"]
        second_id = ground_payload(self._document(), second)["issues"][0]["issue_id"]
        self.assertEqual(first_id, second_id)

    def test_model_view_shares_budget_across_nonempty_fields(self):
        view = model_view(self._document(), 80)
        self.assertLessEqual(len(view), 80)
        self.assertIn("标题：", view)
        self.assertIn("诉求内容：", view)
        self.assertIn("诉求目标：", view)
        self.assertIn("地址详情：", view)
        self.assertRegex(prompt_fingerprint(), r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
