import unittest

from ragflow_style_pipeline.topic_analysis import aggregate_results, apply_result_filters


class TestTopicAnalysis(unittest.TestCase):
    def test_apply_result_filters_keeps_month_range_and_area(self):
        results = [
            {
                "doc_id": "a",
                "score": 0.91,
                "metadata": {
                    "call_month": "2024-06",
                    "area_code_area": "武进区",
                    "area_code_street": "丁堰街道",
                    "type3": "无照经营游商",
                },
            },
            {
                "doc_id": "b",
                "score": 0.88,
                "metadata": {
                    "call_month": "2023-12",
                    "area_code_area": "武进区",
                    "area_code_street": "湖塘镇",
                    "type3": "无照经营游商",
                },
            },
            {
                "doc_id": "c",
                "score": 0.87,
                "metadata": {
                    "call_month": "2024-07",
                    "area_code_area": "天宁区",
                    "area_code_street": "茶山街道",
                    "type3": "社会生活噪声",
                },
            },
        ]

        filtered = apply_result_filters(
            results,
            {
                "call_month_gte": "2024-01",
                "call_month_lte": "2024-12",
                "area_code_area_in": ["武进区"],
            },
        )

        self.assertEqual([result["doc_id"] for result in filtered], ["a"])

    def test_aggregate_results_counts_month_area_street_and_type3(self):
        results = [
            {
                "doc_id": "a",
                "score": 0.91,
                "case_content_clean": "流动摊贩占道经营",
                "text": "诉求内容：流动摊贩占道经营",
                "metadata": {
                    "call_month": "2024-06",
                    "area_code_area": "武进区",
                    "area_code_street": "丁堰街道",
                    "type3": "无照经营游商",
                },
            },
            {
                "doc_id": "b",
                "score": 0.88,
                "case_content_clean": "夜市摊贩扰民",
                "text": "诉求内容：夜市摊贩扰民",
                "metadata": {
                    "call_month": "2024-06",
                    "area_code_area": "武进区",
                    "area_code_street": "丁堰街道",
                    "type3": "无照经营游商",
                },
            },
            {
                "doc_id": "c",
                "score": 0.80,
                "case_content_clean": "学校门口摆摊",
                "text": "诉求内容：学校门口摆摊",
                "metadata": {
                    "call_month": "2024-07",
                    "area_code_area": "天宁区",
                    "area_code_street": "茶山街道",
                    "type3": "无照经营游商",
                },
            },
        ]

        report = aggregate_results(
            query="流动摆摊",
            filters={"call_month_gte": "2024-01", "call_month_lte": "2024-12"},
            results=results,
            representative_limit=2,
        )

        self.assertEqual(report["matched_orders"], 3)
        self.assertEqual(report["statistics"]["by_month"][0], {"value": "2024-06", "count": 2})
        self.assertEqual(report["statistics"]["by_area"][0], {"value": "武进区", "count": 2})
        self.assertEqual(report["statistics"]["by_street"][0], {"value": "丁堰街道", "count": 2})
        self.assertEqual(report["statistics"]["by_type3"][0], {"value": "无照经营游商", "count": 3})
        self.assertEqual(len(report["representative_cases"]), 2)
        self.assertEqual(report["representative_cases"][0]["doc_id"], "a")


if __name__ == "__main__":
    unittest.main()
