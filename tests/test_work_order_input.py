import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.work_order_input import (
    WorkOrderInputError,
    normalize_work_order,
    read_work_orders,
)


class TestWorkOrderInput(unittest.TestCase):
    def test_normalizes_v2_desensitized_document_and_keeps_doc_id(self):
        order = normalize_work_order({
            "schema_version": "2.0",
            "doc_id": "order_safe_1",
            "title_clean": "路灯故障",
            "case_content_clean": "市民反映和平路路灯连续三天不亮。",
            "case_goal_clean": "希望维修",
            "address_detail_clean": "和平路",
            "metadata": {"service_object_type": "求助", "area_code_area": "钟楼区"},
        })
        self.assertEqual(order["doc_id"], "order_safe_1")
        self.assertEqual(order["case_content_clean"], "市民反映和平路路灯连续三天不亮。")
        self.assertTrue(order["content_hash"].startswith("sha256:"))
        self.assertIn("诉求内容：", order["chunk_text"])

    def test_adapts_legacy_tagged_text_without_silently_losing_content(self):
        order = normalize_work_order({
            "doc_id": "order_legacy_1",
            "text": "诉求类型：咨询\n诉求内容：咨询体检报告如何查询。\n诉求目标：希望告知查询方式\n所属区域：常州市 / 钟楼区",
            "metadata": {"service_object_type": "咨询", "area_code_area": "钟楼区"},
        })
        self.assertEqual(order["case_content_clean"], "咨询体检报告如何查询。")
        self.assertEqual(order["case_goal_clean"], "希望告知查询方式")

    def test_rejects_document_with_no_desensitized_semantic_text(self):
        with self.assertRaisesRegex(WorkOrderInputError, "empty_semantic_text"):
            normalize_work_order({"doc_id": "order_empty", "metadata": {}})

    def test_reads_jsonl_with_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "orders.jsonl"
            path.write_text("\n".join([
                json.dumps({"doc_id": "a", "case_content_clean": "第一条脱敏工单"}, ensure_ascii=False),
                json.dumps({"doc_id": "b", "case_content_clean": "第二条脱敏工单"}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            rows = read_work_orders(path, limit=1)
        self.assertEqual([row["doc_id"] for row in rows], ["a"])
