import unittest
import re

from ragflow_style_pipeline.document_builder import build_document


class TestDocumentBuilder(unittest.TestCase):
    def test_builds_document_text_and_metadata(self):
        phone = "138" + "0013" + "8000"
        row = {
            "id": "1",
            "order_id": "ORD001",
            "service_object_type": "求助",
            "case_content": "市民反映手机号" + phone + "附近夜间摆摊扰民",
            "case_goal": "希望处理占道经营",
            "area_code_city": "常州市",
            "area_code_area": "武进区",
            "area_code_street": "湖塘镇",
            "case_accord_type_one_name": "城乡建设",
            "case_accord_type_two_name": "市容管理",
            "case_accord_type_three_name": "无照经营游商",
            "order_source": "电话",
            "order_type": "个人",
            "order_status": "100",
            "call_time": "2025-01-02 10:11:12",
        }

        doc, counts = build_document(row)

        self.assertTrue(doc["doc_id"].startswith("order_"))
        self.assertIn("诉求内容：市民反映手机号[手机号]附近夜间摆摊扰民", doc["text"])
        self.assertIn("业务分类：城乡建设 / 市容管理 / 无照经营游商", doc["text"])
        self.assertIn("所属区域：常州市 / 武进区 / 湖塘镇", doc["text"])
        self.assertEqual(doc["metadata"]["area_code_area"], "武进区")
        self.assertEqual(doc["metadata"]["type3"], "无照经营游商")
        self.assertEqual(doc["metadata"]["call_month"], "2025-01")
        self.assertEqual(counts["phone"], 1)

    def test_builds_multiview_document_fields(self):
        phone = "138" + "0013" + "8000"
        row = {
            "id": "10",
            "order_id": "ORD010",
            "service_object_type": "投诉举报",
            "case_content": "市民反映手机号" + phone + "附近有流动摊贩占道经营",
            "case_goal": "希望执法部门清理流动摊贩",
            "area_code_city": "常州市",
            "area_code_area": "武进区",
            "area_code_street": "丁堰街道",
            "case_accord_type_one_name": "城乡建设",
            "case_accord_type_two_name": "市容管理",
            "case_accord_type_three_name": "无照经营游商",
            "order_source": "互联网",
            "order_type": "个人",
            "order_status": "25",
            "call_time": "2024-06-11 20:51:18",
        }

        doc, counts = build_document(row)

        self.assertEqual(
            doc["case_content_clean"],
            "市民反映手机号[手机号]附近有流动摊贩占道经营",
        )
        self.assertEqual(doc["case_goal_clean"], "希望执法部门清理流动摊贩")
        self.assertEqual(
            doc["embedding_text"],
            (
                "诉求内容：市民反映手机号[手机号]附近有流动摊贩占道经营\n"
                "诉求目标：希望执法部门清理流动摊贩"
            ),
        )
        self.assertIn("业务分类：城乡建设 / 市容管理 / 无照经营游商", doc["display_text"])
        self.assertIn("所属区域：常州市 / 武进区 / 丁堰街道", doc["display_text"])
        self.assertEqual(doc["text"], doc["display_text"])
        self.assertEqual(doc["metadata"]["area_code_area"], "武进区")
        self.assertEqual(doc["metadata"]["call_month"], "2024-06")
        self.assertEqual(doc["derived"]["topic_tags"], [])
        self.assertEqual(doc["derived"]["semantic_cluster_id"], "")
        self.assertEqual(counts["phone"], 1)

    def test_skips_null_text_fields(self):
        row = {
            "id": "3",
            "order_id": "ORD003",
            "service_object_type": "投诉举报",
            "case_content": "NULL",
            "case_goal": "希望处理噪声问题",
            "area_code_city": "常州市",
            "area_code_area": "天宁区",
            "area_code_street": "茶山街道",
            "case_accord_type_one_name": "环境保护",
            "case_accord_type_two_name": "噪声污染",
            "case_accord_type_three_name": "社会生活噪声",
            "order_source": "互联网",
            "order_type": "个人",
            "order_status": "31",
            "call_time": "2025-03-04 20:30:00",
        }

        doc, counts = build_document(row)

        self.assertNotIn("诉求内容：NULL", doc["text"])
        self.assertIn("诉求目标：希望处理噪声问题", doc["text"])
        self.assertEqual(doc["metadata"]["call_month"], "2025-03")
        self.assertEqual(counts["phone"], 0)

    def test_hashes_long_numeric_doc_id(self):
        raw_id = "1800" + "6964" + "8507" + "1659" + "009"
        row = {
            "id": raw_id,
            "order_id": "ORD001",
            "service_object_type": "求助",
            "case_content": "市民反映附近夜间摆摊扰民",
            "case_goal": "希望处理占道经营",
            "call_time": "2025-01-02 10:11:12",
        }

        doc, _counts = build_document(row)
        same_doc, _same_counts = build_document(row)

        self.assertNotEqual(doc["doc_id"], raw_id)
        self.assertTrue(doc["doc_id"].startswith("order_"))
        self.assertEqual(doc["doc_id"], same_doc["doc_id"])
        self.assertRegex(doc["doc_id"], r"^order_[a-z]+$")


if __name__ == "__main__":
    unittest.main()
