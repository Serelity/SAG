import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_db import (
    build_sag_db_from_orders,
    event_row,
    load_entity_links_jsonl,
    read_source_rows,
    source_order_row,
    stable_hash,
)


def _skip_without_duckdb(testcase):
    try:
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        testcase.skipTest("duckdb is not installed in the local test runtime")


class TestSagDbMapping(unittest.TestCase):
    def test_stable_hash_does_not_return_raw_value(self):
        hashed = stable_hash("ORD001")

        self.assertNotEqual(hashed, "ORD001")
        self.assertEqual(hashed, stable_hash("ORD001"))
        self.assertEqual(len(hashed), 16)

    def test_source_order_row_maps_raw_tsv_fields(self):
        raw = {
            "id": "1",
            "order_id": "ORD001",
            "title": "广成路流动摊贩",
            "case_content": "市民反映钟楼区永红街道广成路有流动摊贩占道经营",
            "case_goal": "希望处理",
            "address_detail": "广成路与江春路交界处",
            "call_time": "2024-05-01 10:00:00",
            "area_code_city": "常州市",
            "area_code_area": "钟楼区",
            "area_code_street": "",
            "case_lnglat": "119.95,31.78",
            "case_accord_type_one_name": "城乡建设",
            "case_accord_type_two_name": "市容管理",
            "case_accord_type_three_name": "无照经营游商",
            "case_accord_type_four_name": "流动摊贩",
            "case_accord_type_five_name": "",
            "case_accord_code": "ABC",
            "order_source": "电话",
            "order_type": "个人",
            "order_status": "100",
            "service_object_type": "投诉举报",
        }

        row = source_order_row(raw)

        self.assertTrue(row["doc_id"].startswith("order_"))
        self.assertEqual(row["case_content_clean"], raw["case_content"])
        self.assertEqual(row["title_clean"], raw["title"])
        self.assertEqual(row["address_detail_clean"], raw["address_detail"])
        self.assertEqual(row["call_month"], "2024-05")
        self.assertEqual(row["type3"], "无照经营游商")
        self.assertEqual(row["type4"], "流动摊贩")
        self.assertEqual(row["raw_id_hash"], stable_hash("1"))
        self.assertEqual(row["order_id_hash"], stable_hash("ORD001"))

    def test_event_row_preserves_complete_event_semantics(self):
        source = source_order_row(
            {
                "id": "1",
                "order_id": "ORD001",
                "title": "广成路流动摊贩",
                "case_content": "市民反映钟楼区永红街道广成路有流动摊贩占道经营",
                "case_goal": "希望处理",
                "address_detail": "广成路与江春路交界处",
                "call_time": "2024-05-01 10:00:00",
                "area_code_area": "钟楼区",
                "area_code_street": "永红街道",
                "case_accord_type_three_name": "无照经营游商",
                "order_status": "100",
            }
        )

        event = event_row(source)

        self.assertEqual(event["doc_id"], source["doc_id"])
        self.assertEqual(event["event_month"], "2024-05")
        self.assertIn("广成路流动摊贩", event["event_text"])
        self.assertIn("无照经营游商", event["event_text"])
        self.assertIn("永红街道", event["event_text"])

    def test_read_source_rows_supports_tsv_and_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tsv_path = tmp / "orders.tsv"
            tsv_path.write_text(
                "id\torder_id\tcase_content\tcase_goal\tcall_time\n"
                "1\tORD001\t流动摊贩占道经营\t希望处理\t2024-05-01 10:00:00\n",
                encoding="utf-8",
            )
            jsonl_path = tmp / "orders.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "doc_id": "order_x",
                        "case_content_clean": "流动摊贩占道经营",
                        "case_goal_clean": "希望处理",
                        "metadata": {"call_time": "2024-05-01 10:00:00", "call_month": "2024-05"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            tsv_rows = read_source_rows(tsv_path)
            jsonl_rows = read_source_rows(jsonl_path)

        self.assertEqual(len(tsv_rows), 1)
        self.assertEqual(len(jsonl_rows), 1)
        self.assertEqual(tsv_rows[0]["call_month"], "2024-05")
        self.assertEqual(jsonl_rows[0]["doc_id"], "order_x")


class TestSagDbBuild(unittest.TestCase):
    def test_build_sag_db_from_orders_creates_events_entities_and_links(self):
        _skip_without_duckdb(self)
        import duckdb

        orders = [
            source_order_row(
                {
                    "id": "1",
                    "order_id": "ORD001",
                    "title": "流动摊贩占道",
                    "case_content": "市民反映钟楼区永红街道广成路有流动摊贩占道经营",
                    "case_goal": "希望处理",
                    "address_detail": "广成路",
                    "call_time": "2024-05-01 10:00:00",
                    "area_code_area": "钟楼区",
                    "area_code_street": "",
                    "case_accord_type_three_name": "无照经营游商",
                }
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sag.duckdb"
            report = build_sag_db_from_orders(orders, db_path)
            with duckdb.connect(str(db_path)) as conn:
                event_count = conn.execute("select count(*) from sag_events").fetchone()[0]
                entity_count = conn.execute("select count(*) from sag_entities").fetchone()[0]
                road_count = conn.execute(
                    "select count(*) from sag_event_entity_links where entity_type = 'road'"
                ).fetchone()[0]

        self.assertEqual(report["events_loaded"], 1)
        self.assertEqual(event_count, 1)
        self.assertGreater(entity_count, 0)
        self.assertGreaterEqual(road_count, 1)

    def test_build_db_merges_llm_entity_links(self):
        _skip_without_duckdb(self)
        import duckdb

        order = source_order_row(
            {
                "id": "1",
                "order_id": "ORD001",
                "case_content": "市民反映广成路有卖菜摊子挡住人行道。",
                "case_goal": "希望处理",
                "call_time": "2024-05-01 10:00:00",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            links_path = tmp / "llm_links.jsonl"
            links_path.write_text(
                json.dumps(
                    {
                        "doc_id": order["doc_id"],
                        "entity_type": "problem_object",
                        "entity_value": "流动摊贩",
                        "normalized_value": "流动摊贩",
                        "source_field": "case_content_clean",
                        "source_channel": "llm",
                        "confidence": 0.8,
                        "matched_text": "卖菜摊子",
                        "projection_version": "sag_semantic_projection_v1",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            db_path = tmp / "sag.duckdb"
            build_sag_db_from_orders([order], db_path, extra_entity_links_by_doc=load_entity_links_jsonl(links_path))

            conn = duckdb.connect(str(db_path))
            rows = conn.execute(
                """
                select entity_type, entity_value, source_channel, projection_version
                from sag_event_entity_links
                where doc_id = ? and source_channel = 'llm'
                """,
                [order["doc_id"]],
            ).fetchall()

        self.assertEqual(rows, [(
            "problem_object", "流动摊贩", "llm", "sag_semantic_projection_v1"
        )])

    def test_build_db_stores_semantic_event_and_discourse(self):
        _skip_without_duckdb(self)
        import duckdb

        order = source_order_row({"id":"9","order_id":"ORD009","case_content":"和平路路灯不亮","service_object_type":"求助"})
        semantic_event = {
            "doc_id": order["doc_id"],
            "event": {"summary": "市民反映和平路路灯不亮"},
            "validation": {"status": "accepted"},
            "projection_version": "sag_semantic_projection_v1",
        }
        discourse = {
            "doc_id": order["doc_id"], "declared_intent":"求助",
            "inferred_intents_json":"[\"求助\"]", "intent_conflict":"false",
            "emotions_json":"[]", "satisfaction":"unknown", "urgency":"normal",
            "projection_version":"sag_semantic_projection_v1",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "semantic.duckdb"
            build_sag_db_from_orders(
                [order], db_path,
                semantic_events_by_doc={order["doc_id"]: semantic_event},
                discourse_by_doc={order["doc_id"]: discourse},
            )
            with duckdb.connect(str(db_path)) as conn:
                event_text, event_projection = conn.execute(
                    "select event_text, projection_version from sag_events where doc_id = ?",
                    [order["doc_id"]],
                ).fetchone()
                values = conn.execute(
                    "select declared_intent, satisfaction, urgency, projection_version "
                    "from sag_event_discourse where doc_id = ?",
                    [order["doc_id"]],
                ).fetchone()
        self.assertEqual(event_text, "市民反映和平路路灯不亮")
        self.assertEqual(event_projection, "sag_semantic_projection_v1")
        self.assertEqual(values, ("求助", "unknown", "normal", "sag_semantic_projection_v1"))


if __name__ == "__main__":
    unittest.main()
