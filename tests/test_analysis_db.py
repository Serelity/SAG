import unittest

from ragflow_style_pipeline.analysis_db import document_row


class TestAnalysisDb(unittest.TestCase):
    def test_document_row_flattens_multiview_document(self):
        document = {
            "doc_id": "order_a",
            "case_content_clean": "流动摊贩占道经营",
            "case_goal_clean": "希望清理",
            "embedding_text": "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
            "display_text": "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
            "metadata": {
                "call_time": "2024-06-11 20:51:18",
                "call_month": "2024-06",
                "area_code_city": "常州市",
                "area_code_area": "武进区",
                "area_code_street": "丁堰街道",
                "type1": "城乡建设",
                "type2": "市容管理",
                "type3": "无照经营游商",
                "order_source": "互联网",
                "order_type": "个人",
                "order_status": "25",
                "service_object_type": "投诉举报",
            },
        }

        row = document_row(document)

        self.assertEqual(row["doc_id"], "order_a")
        self.assertEqual(row["case_content_clean"], "流动摊贩占道经营")
        self.assertEqual(row["area_code_area"], "武进区")
        self.assertEqual(row["area_code_street"], "丁堰街道")
        self.assertEqual(row["type3"], "无照经营游商")
        self.assertEqual(row["call_month"], "2024-06")


if __name__ == "__main__":
    unittest.main()
