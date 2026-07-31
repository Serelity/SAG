import unittest

from ragflow_style_pipeline.embedding_text import embedding_text


class TestEmbeddingText(unittest.TestCase):
    def test_embedding_text_prefers_new_embedding_text_field(self):
        document = {
            "embedding_text": "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
            "display_text": (
                "诉求内容：流动摊贩占道经营\n"
                "业务分类：城乡建设 / 市容管理 / 无照经营游商\n"
                "所属区域：常州市 / 武进区"
            ),
            "text": "旧字段",
        }

        self.assertEqual(
            embedding_text(document),
            "诉求内容：流动摊贩占道经营\n诉求目标：希望清理",
        )

    def test_embedding_text_falls_back_to_old_text_prefixes(self):
        document = {
            "text": (
                "诉求类型：求助\n"
                "诉求内容：服务对象反映房屋漏水。\n"
                "诉求目标：希望维修。\n"
                "业务分类：住房保障 / 物业管理\n"
                "所属区域：常州市 / 新北区"
            ),
            "metadata": {"type2": "物业管理"},
        }

        text = embedding_text(document)

        self.assertIn("诉求内容：服务对象反映房屋漏水。", text)
        self.assertIn("诉求目标：希望维修。", text)
        self.assertNotIn("业务分类", text)
        self.assertNotIn("所属区域", text)

    def test_embedding_text_falls_back_to_full_display_text_when_body_missing(self):
        document = {"display_text": "服务对象反映老板不给工资。", "metadata": {}}

        self.assertEqual(embedding_text(document), "服务对象反映老板不给工资。")


if __name__ == "__main__":
    unittest.main()
