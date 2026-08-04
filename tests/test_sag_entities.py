import unittest

from ragflow_style_pipeline.sag_entities import (
    SagEntityLink,
    deduplicate_entity_links,
    extract_entities_from_order,
    normalize_entity_value,
)


class TestSagEntities(unittest.TestCase):
    def test_keeps_road_name_starting_with_connector_character(self):
        links = extract_entities_from_order(
            {"doc_id": "order_road", "case_content_clean": "和平路路灯不亮"}
        )
        roads = {link.normalized_value for link in links if link.entity_type == "road"}

        self.assertIn("和平路", roads)
        self.assertNotIn("平路", roads)

    def test_normalize_entity_value_removes_spaces(self):
        self.assertEqual(normalize_entity_value(" 永红街道 "), "永红街道")
        self.assertEqual(normalize_entity_value("广成 路"), "广成路")

    def test_extracts_metadata_entities(self):
        order = {
            "doc_id": "order_a",
            "call_month": "2024-05",
            "area_code_area": "钟楼区",
            "area_code_street": "永红街道",
            "type3": "无照经营游商",
            "case_content_clean": "",
            "case_goal_clean": "",
            "title_clean": "",
            "address_detail_clean": "",
        }

        links = extract_entities_from_order(order)
        observed = {(link.entity_type, link.entity_value, link.source_field) for link in links}

        self.assertIn(("time_month", "2024-05", "call_month"), observed)
        self.assertIn(("area", "钟楼区", "area_code_area"), observed)
        self.assertIn(("street", "永红街道", "area_code_street"), observed)
        self.assertIn(("case_type", "无照经营游商", "type3"), observed)

    def test_extracts_case_content_space_and_problem_entities(self):
        order = {
            "doc_id": "order_b",
            "call_month": "2024-05",
            "area_code_area": "",
            "area_code_street": "",
            "type3": "",
            "case_content_clean": "市民反映钟楼区永红街道广成路和江春路交叉口有流动摊贩占道经营，影响通行。",
            "case_goal_clean": "希望城管处理",
            "title_clean": "流动摊贩占道",
            "address_detail_clean": "广成路与江春路交界处",
        }

        links = extract_entities_from_order(order)
        observed = {(link.entity_type, link.entity_value) for link in links}

        self.assertIn(("area", "钟楼区"), observed)
        self.assertIn(("street", "永红街道"), observed)
        self.assertIn(("road", "广成路"), observed)
        self.assertIn(("road", "江春路"), observed)
        self.assertIn(("intersection", "广成路和江春路交叉口"), observed)
        self.assertIn(("intersection", "广成路与江春路交界处"), observed)
        self.assertIn(("problem_object", "流动摊贩"), observed)
        self.assertIn(("problem_behavior", "占道经营"), observed)
        self.assertIn(("problem_behavior", "影响通行"), observed)

    def test_deduplicates_same_entity_from_same_source_field(self):
        links = [
            SagEntityLink("order_a", "road", "广成路", "广成路", "case_content_clean", "case_content", 0.9, "广成路"),
            SagEntityLink("order_a", "road", "广成路", "广成路", "case_content_clean", "case_content", 0.9, "广成路"),
            SagEntityLink("order_a", "road", "广成路", "广成路", "address_detail_clean", "address_detail", 0.9, "广成路"),
        ]

        deduped = deduplicate_entity_links(links)

        self.assertEqual(len(deduped), 2)


if __name__ == "__main__":
    unittest.main()
