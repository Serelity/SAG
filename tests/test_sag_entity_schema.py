import unittest

from ragflow_style_pipeline.sag_entity_schema import (
    ALLOWED_LLM_ENTITY_TYPES,
    evidence_exists,
    is_generic_entity_value,
    normalize_llm_entity_value,
    validate_llm_candidate,
)


class TestSagEntitySchema(unittest.TestCase):
    def test_allowed_types_are_sag_retrieval_only(self):
        self.assertEqual(
            ALLOWED_LLM_ENTITY_TYPES,
            {"problem_object", "problem_behavior", "area", "street", "road", "intersection", "poi"},
        )

    def test_normalizes_aliases_for_problem_entities(self):
        self.assertEqual(normalize_llm_entity_value("problem_object", "卖菜摊子"), "流动摊贩")
        self.assertEqual(normalize_llm_entity_value("problem_behavior", "挡住人行道"), "占道经营")
        self.assertEqual(normalize_llm_entity_value("road", " 广成 路 "), "广成路")

    def test_filters_generic_spatial_noise(self):
        self.assertTrue(is_generic_entity_value("road", "路"))
        self.assertTrue(is_generic_entity_value("road", "关于道路"))
        self.assertTrue(is_generic_entity_value("poi", "关于小区"))
        self.assertTrue(is_generic_entity_value("poi", "本人要求市场"))
        self.assertFalse(is_generic_entity_value("road", "广成路"))
        self.assertFalse(is_generic_entity_value("poi", "清潭菜场"))

    def test_evidence_must_exist_in_source_text(self):
        order = {
            "case_content_clean": "市民反映广成路有流动摊贩占道经营。",
            "case_goal_clean": "希望城管处理",
            "title_clean": "",
            "address_detail_clean": "",
        }
        good = {"entity_type": "road", "entity_value": "广成路", "evidence_span": "广成路", "source_field": "case_content_clean"}
        bad = {"entity_type": "road", "entity_value": "江春路", "evidence_span": "江春路", "source_field": "case_content_clean"}

        self.assertTrue(evidence_exists(good, order))
        self.assertFalse(evidence_exists(bad, order))

    def test_validate_candidate_rejects_unsupported_and_generic_values(self):
        order = {"case_content_clean": "市民反映广成路有流动摊贩占道经营。"}
        config = {"min_confidence": 0.55}

        ok, reason = validate_llm_candidate(
            {
                "entity_type": "road",
                "entity_value": "广成路",
                "evidence_span": "广成路",
                "source_field": "case_content_clean",
                "confidence": 0.9,
            },
            order,
            config,
        )
        self.assertTrue(ok, reason)

        ok, reason = validate_llm_candidate(
            {
                "entity_type": "road",
                "entity_value": "道路",
                "evidence_span": "道路",
                "source_field": "case_content_clean",
                "confidence": 0.9,
            },
            order,
            config,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "generic_entity_value")


if __name__ == "__main__":
    unittest.main()
