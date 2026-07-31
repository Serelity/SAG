import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.local_search import build_index, load_documents, search


def _doc(doc_id, text, **metadata):
    return {
        "doc_id": doc_id,
        "text": text,
        "metadata": {
            "area_code_area": metadata.get("area_code_area", ""),
            "type1": metadata.get("type1", ""),
            "type2": metadata.get("type2", ""),
            "type3": metadata.get("type3", ""),
            "call_month": metadata.get("call_month", ""),
        },
    }


class TestLocalSearch(unittest.TestCase):
    def test_load_documents_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "docs.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(_doc("doc_1", "第一条工单"), ensure_ascii=False),
                        json.dumps(_doc("doc_2", "第二条工单"), ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            documents = load_documents(path, limit=1)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["doc_id"], "doc_1")

    def test_load_documents_preserves_multiview_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "docs.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "doc_id": "order_a",
                        "case_content_clean": "流动摊贩占道经营",
                        "case_goal_clean": "希望清理",
                        "embedding_text": "诉求内容：流动摊贩占道经营",
                        "display_text": "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
                        "text": "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
                        "metadata": {"area_code_area": "武进区"},
                        "derived": {"topic_tags": []},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            documents = load_documents(path)

            self.assertEqual(documents[0]["case_content_clean"], "流动摊贩占道经营")
            self.assertEqual(documents[0]["embedding_text"], "诉求内容：流动摊贩占道经营")
            self.assertEqual(
                documents[0]["display_text"],
                "诉求内容：流动摊贩占道经营\n所属区域：常州市 / 武进区",
            )
            self.assertEqual(documents[0]["derived"], {"topic_tags": []})

    def test_search_ranks_more_relevant_document_first(self):
        documents = [
            _doc("salary", "诉求内容：工地拖欠工资，工资一直未发。业务分类：民生保障"),
            _doc("salary_policy", "诉求内容：咨询工资标准。业务分类：民生保障"),
            _doc("noise", "诉求内容：夜间广场舞噪音扰民。业务分类：环境保护"),
        ]
        index = build_index(documents)

        results = search(index, "拖欠工资 工地 未发", top_k=2)

        self.assertEqual(results[0]["doc_id"], "salary")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_search_applies_metadata_filters(self):
        documents = [
            _doc("wujin", "诉求内容：占道经营摆摊", area_code_area="武进区"),
            _doc("tianning", "诉求内容：占道经营摆摊", area_code_area="天宁区"),
        ]
        index = build_index(documents)

        results = search(index, "占道经营 摆摊", top_k=5, filters={"area_code_area": "武进区"})

        self.assertEqual([result["doc_id"] for result in results], ["wujin"])


if __name__ == "__main__":
    unittest.main()
