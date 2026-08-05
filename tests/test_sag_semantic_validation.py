import unittest

from ragflow_style_pipeline.sag_semantic_validation import sanitize_semantic_output, validate_semantic_output


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


if __name__ == "__main__":
    unittest.main()
