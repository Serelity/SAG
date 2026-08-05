import unittest

from ragflow_style_pipeline.sag_semantic_validation import (
    enrich_semantic_output,
    sanitize_semantic_output,
    validate_semantic_output,
)


def semantic_with(group, item):
    entities = {name: [] for name in ("problem_objects", "problem_behaviors", "roads", "intersections", "pois")}
    entities[group] = [item]
    return {"event_summary": "测试事件", "entities": entities, "discourse": {
        "intents": [], "emotions": [],
        "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
        "urgency": {"level": "normal", "evidence": ""},
    }}


class TestSemanticValidation(unittest.TestCase):
    def setUp(self):
        self.order = {"doc_id": "order_1", "case_content_clean": "港龙新港城北门口有摊贩占道，希望清理，请优先处理，谢谢！", "case_goal_clean": "希望清理", "title_clean": "", "address_detail_clean": ""}

    def test_requires_repair_when_evidence_is_missing(self):
        result = validate_semantic_output(self.order, semantic_with("roads", {"surface":"和平路","canonical":"和平路","source_field":"case_content_clean","evidence":"和平路"}))
        self.assertEqual(result["status"], "repair_required")
        self.assertIn("missing_evidence:entities.roads.0", result["warnings"])

    def test_flags_poi_gate_as_not_a_named_road(self):
        result = validate_semantic_output(self.order, semantic_with("roads", {"surface":"港龙新港城北门口","canonical":"港龙新港城北门口","source_field":"case_content_clean","evidence":"港龙新港城北门口"}))
        self.assertIn("road_poi_conflict:entities.roads.0", result["warnings"])

    def test_flags_request_action_as_problem_behavior(self):
        result = validate_semantic_output(self.order, semantic_with("problem_behaviors", {"surface":"清理","canonical":"清理","source_field":"case_goal_clean","evidence":"清理"}))
        self.assertIn("request_action_as_behavior:entities.problem_behaviors.0", result["warnings"])

    def test_rejects_template_thanks_as_satisfaction_evidence(self):
        semantic = semantic_with("pois", {"surface":"港龙新港城","canonical":"港龙新港城","source_field":"case_content_clean","evidence":"港龙新港城"})
        semantic["discourse"]["satisfaction"] = {"label":"satisfied","target":"部门","evidence":"谢谢"}
        result = validate_semantic_output(self.order, semantic)
        self.assertEqual(result["status"], "repair_required")
        self.assertIn("template_politeness_as_satisfaction", result["warnings"])

    def test_accepts_valid_semantic_and_reports_parse_warning(self):
        semantic = semantic_with("problem_objects", {"surface":"摊贩","canonical":"流动摊贩","source_field":"case_content_clean","evidence":"摊贩"})
        self.assertEqual(validate_semantic_output(self.order, semantic)["status"], "accepted")
        result = validate_semantic_output(self.order, semantic, ["json_parse_failed"])
        self.assertEqual(result["status"], "repair_required")

    def test_history_contamination_and_urgency_evidence(self):
        order = dict(self.order, case_content_clean="部门答复已处理。现服务对象表示其不认可，仍未解决。")
        semantic = semantic_with("problem_objects", {"surface":"问题","canonical":"业务","source_field":"case_content_clean","evidence":"仍未解决"})
        semantic["event_summary"] = "部门已处理问题"
        semantic["discourse"]["urgency"] = {"level":"high","evidence":""}
        result = validate_semantic_output(order, semantic)
        self.assertIn("possible_history_contamination", result["warnings"])
        self.assertIn("urgency_missing_evidence", result["warnings"])

    def test_aligns_surface_to_verified_source_evidence_without_dropping_entity(self):
        semantic = semantic_with("problem_behaviors", {
            "surface":"占道经营", "canonical":"占道经营",
            "source_field":"case_content_clean", "evidence":"摊贩占道",
        })
        validation = validate_semantic_output(self.order, semantic)
        self.assertIn("surface_evidence_mismatch:entities.problem_behaviors.0", validation["warnings"])
        cleaned, actions = sanitize_semantic_output(semantic, validation["warnings"])
        revalidated = validate_semantic_output(self.order, cleaned)
        self.assertEqual(cleaned["entities"]["problem_behaviors"][0]["surface"], "摊贩占道")
        self.assertEqual(cleaned["entities"]["problem_behaviors"][0]["canonical"], "占道经营")
        self.assertEqual(revalidated["status"], "accepted")
        self.assertEqual(actions, ["aligned_surface_to_evidence:entities.problem_behaviors.0"])

    def test_sanitizes_invalid_optional_candidates_and_safe_discourse_defaults(self):
        semantic = semantic_with("problem_objects", {
            "surface":"摊贩", "canonical":"流动摊贩",
            "source_field":"case_content_clean", "evidence":"不存在的证据",
        })
        semantic["discourse"]["intents"] = [{"label":"投诉", "evidence":"不存在的意图证据"}]
        semantic["discourse"]["emotions"] = [{"label":"不满", "intensity":2, "evidence":"不存在的情绪证据"}]
        semantic["discourse"]["satisfaction"] = {"label":"satisfied", "target":"部门", "evidence":"谢谢"}
        semantic["discourse"]["urgency"] = {"level":"high", "evidence":"请优先处理"}
        validation = validate_semantic_output(self.order, semantic)
        cleaned, actions = sanitize_semantic_output(semantic, validation["warnings"])
        revalidated = validate_semantic_output(self.order, cleaned)

        self.assertEqual(cleaned["entities"]["problem_objects"], [])
        self.assertEqual(cleaned["discourse"]["intents"], [])
        self.assertEqual(cleaned["discourse"]["emotions"], [])
        self.assertEqual(cleaned["discourse"]["satisfaction"], {"label":"unknown", "target":"", "evidence":""})
        self.assertEqual(cleaned["discourse"]["urgency"], {"level":"normal", "evidence":""})
        self.assertEqual(revalidated["status"], "accepted")
        self.assertIn("dropped_invalid_candidate:entities.problem_objects.0", actions)
        self.assertIn("dropped_unverified_evidence:discourse.intents.0", actions)
        self.assertIn("dropped_unverified_evidence:discourse.emotions.0", actions)
        self.assertIn("reset_unverified_satisfaction", actions)
        self.assertIn("reset_unverified_urgency", actions)

    def test_recovers_verified_string_entities_but_not_unverified_values(self):
        semantic = semantic_with("roads", {
            "surface":"人民路", "canonical":"人民路",
            "source_field":"", "evidence":"人民路",
        })
        semantic["entities"]["roads"].append({
            "surface":"虚构路", "canonical":"虚构路",
            "source_field":"", "evidence":"虚构路",
        })
        order = dict(self.order, case_content_clean="人民路与花东街交叉口有乱摆摊")
        enriched, actions = enrich_semantic_output(
            order, semantic,
            ["coerced_entity_string:roads:0", "coerced_entity_string:roads:1"],
        )
        self.assertEqual(
            enriched["entities"]["roads"][0]["source_field"],
            "case_content_clean",
        )
        validation = validate_semantic_output(order, enriched)
        cleaned, sanitation = sanitize_semantic_output(
            enriched, validation["warnings"], order=order,
        )
        self.assertEqual([item["canonical"] for item in cleaned["entities"]["roads"]], ["人民路"])
        self.assertIn("recovered_entity_string:entities.roads.0", actions)
        self.assertIn("dropped_invalid_candidate:entities.roads.1", sanitation)

    def test_recovers_literal_intent_evidence_and_missing_explicit_intent(self):
        order = dict(self.order, case_content_clean="服务对象投诉工作人员删除工单，要求核实")
        semantic = semantic_with("problem_objects", {
            "surface":"工单", "canonical":"工单", "source_field":"case_content_clean", "evidence":"工单",
        })
        semantic["discourse"]["intents"] = [{"label":"投诉", "evidence":"服务对象要求核实问题"}]
        enriched, actions = enrich_semantic_output(order, semantic)
        self.assertEqual(enriched["discourse"]["intents"], [{"label":"投诉", "evidence":"投诉"}])
        self.assertIn("recovered_intent_evidence:discourse.intents.0", actions)
        self.assertEqual(validate_semantic_output(order, enriched)["status"], "accepted")

        semantic["discourse"]["intents"] = []
        enriched, actions = enrich_semantic_output(order, semantic)
        self.assertEqual(enriched["discourse"]["intents"], [{"label":"投诉", "evidence":"投诉"}])
        self.assertIn("recovered_explicit_intent:投诉", actions)

    def test_recovers_only_direct_requester_emotion_evidence(self):
        direct_order = dict(self.order, case_content_clean="服务对象对此非常不满，其不认可处理结果")
        semantic = semantic_with("problem_objects", {
            "surface":"处理结果", "canonical":"处理结果",
            "source_field":"case_content_clean", "evidence":"处理结果",
        })
        enriched, actions = enrich_semantic_output(direct_order, semantic)
        self.assertEqual(enriched["discourse"]["emotions"], [{
            "label":"不满", "intensity":3, "evidence":"非常不满",
        }])
        self.assertIn("recovered_explicit_emotion:不满", actions)

        for text in (
            "工作人员态度恶劣，要求处理",
            "服务对象反映商家很不满意处理要求",
        ):
            with self.subTest(text=text):
                object_attitude = dict(self.order, case_content_clean=text)
                enriched, _ = enrich_semantic_output(object_attitude, semantic)
                self.assertEqual(enriched["discourse"]["emotions"], [])

    def test_strict_intersection_requires_two_named_roads(self):
        order = dict(
            self.order,
            case_content_clean="劳动东路北侧往污水厂交叉口侧石破损；人民路与花东街交叉口拥堵",
        )
        single = semantic_with("intersections", {
            "surface":"劳动东路北侧往污水厂交叉口",
            "canonical":"劳动东路北侧往污水厂交叉口",
            "source_field":"case_content_clean", "evidence":"劳动东路北侧往污水厂交叉口",
        })
        dual = semantic_with("intersections", {
            "surface":"人民路与花东街交叉口", "canonical":"人民路与花东街交叉口",
            "source_field":"case_content_clean", "evidence":"人民路与花东街交叉口",
        })
        self.assertIn(
            "intersection_shape_conflict:entities.intersections.0",
            validate_semantic_output(order, single)["warnings"],
        )
        self.assertNotIn(
            "intersection_shape_conflict:entities.intersections.0",
            validate_semantic_output(order, dual)["warnings"],
        )

    def test_drops_request_only_refund_behavior_but_keeps_unfulfilled_fact(self):
        order = dict(
            self.order,
            case_content_clean="培训机构闭店，服务对象希望协调退款或者销课",
            case_goal_clean="希望协调退款或者销课",
        )
        request = semantic_with("problem_behaviors", {
            "surface":"希望协调退款或者销课", "canonical":"未退款",
            "source_field":"case_content_clean", "evidence":"希望协调退款或者销课",
        })
        result = validate_semantic_output(order, request)
        self.assertIn("request_action_as_behavior:entities.problem_behaviors.0", result["warnings"])

        request["entities"]["problem_behaviors"][0] = {
            "surface":"要求拆除违法建筑", "canonical":"拆除",
            "source_field":"case_content_clean", "evidence":"要求拆除违法建筑",
        }
        order["case_content_clean"] = "服务对象要求拆除违法建筑"
        self.assertIn(
            "request_action_as_behavior:entities.problem_behaviors.0",
            validate_semantic_output(order, request)["warnings"],
        )

        request["entities"]["problem_behaviors"][0] = {
            "surface":"希望退还未消费课时费用", "canonical":"退款",
            "source_field":"case_content_clean", "evidence":"希望退还未消费课时费用",
        }
        order["case_content_clean"] = "服务对象希望退还未消费课时费用"
        self.assertIn(
            "request_action_as_behavior:entities.problem_behaviors.0",
            validate_semantic_output(order, request)["warnings"],
        )

        order["case_content_clean"] = "培训机构闭店，学费仍未退款"
        fact = semantic_with("problem_behaviors", {
            "surface":"仍未退款", "canonical":"退款未到账",
            "source_field":"case_content_clean", "evidence":"仍未退款",
        })
        self.assertNotIn(
            "request_action_as_behavior:entities.problem_behaviors.0",
            validate_semantic_output(order, fact)["warnings"],
        )

    def test_detects_canonical_target_conflict_and_invalid_satisfaction_target(self):
        order = dict(self.order, case_content_clean="严重影响自己家的采光，其不认可")
        semantic = semantic_with("problem_behaviors", {
            "surface":"严重影响自己家的采光", "canonical":"严重影响通风",
            "source_field":"case_content_clean", "evidence":"严重影响自己家的采光",
        })
        semantic["discourse"]["satisfaction"] = {
            "label":"dissatisfied", "target":"服务对象", "evidence":"其不认可",
        }
        result = validate_semantic_output(order, semantic)
        self.assertIn("canonical_evidence_conflict:entities.problem_behaviors.0", result["warnings"])
        self.assertIn("invalid_satisfaction_target", result["warnings"])
        cleaned, actions = sanitize_semantic_output(semantic, result["warnings"], order=order)
        self.assertEqual(
            cleaned["entities"]["problem_behaviors"][0]["canonical"],
            "严重影响自己家的采光",
        )
        self.assertEqual(cleaned["discourse"]["satisfaction"]["label"], "unknown")
        self.assertIn("reset_unverified_satisfaction", actions)


if __name__ == "__main__":
    unittest.main()
