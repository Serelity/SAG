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

    def test_synthesizes_only_verified_two_road_intersection(self):
        order = dict(
            self.order,
            case_content_clean="人民路与花东街交叉口有乱摆摊",
        )
        semantic = semantic_with("problem_behaviors", {
            "surface":"乱摆摊", "canonical":"乱摆摊",
            "source_field":"case_content_clean", "evidence":"乱摆摊",
        })
        semantic["entities"]["roads"] = [
            {
                "surface":"人民路", "canonical":"人民路",
                "source_field":"case_content_clean", "evidence":"人民路",
            },
            {
                "surface":"花东街", "canonical":"花东街",
                "source_field":"case_content_clean", "evidence":"花东街",
            },
        ]
        enriched, actions = enrich_semantic_output(order, semantic)
        self.assertEqual(enriched["entities"]["intersections"], [{
            "surface":"人民路与花东街交叉口",
            "canonical":"人民路与花东街交叉口",
            "source_field":"case_content_clean",
            "evidence":"人民路与花东街交叉口",
        }])
        self.assertIn("synthesized_intersection:entities.intersections.0", actions)
        validation = validate_semantic_output(order, enriched)
        self.assertEqual(validation["status"], "accepted_with_warnings")
        self.assertIn("semantic_gap:problem_objects", validation["warnings"])

        one_road = dict(order, case_content_clean="劳动东路北侧往污水厂交叉口侧石破损")
        semantic["entities"]["roads"] = [{
            "surface":"劳动东路", "canonical":"劳动东路",
            "source_field":"case_content_clean", "evidence":"劳动东路",
        }]
        enriched, _ = enrich_semantic_output(one_road, semantic)
        self.assertEqual(enriched["entities"]["intersections"], [])

        two_roads_without_junction = dict(order, case_content_clean="车辆从人民路驶向花东街")
        semantic["entities"]["roads"] = [
            {
                "surface":"人民路", "canonical":"人民路",
                "source_field":"case_content_clean", "evidence":"人民路",
            },
            {
                "surface":"花东街", "canonical":"花东街",
                "source_field":"case_content_clean", "evidence":"花东街",
            },
        ]
        enriched, _ = enrich_semantic_output(two_roads_without_junction, semantic)
        self.assertEqual(enriched["entities"]["intersections"], [])

    def test_drops_admin_division_and_bare_address_pois(self):
        order = dict(
            self.order,
            case_content_clean="金坛区薛埠镇东环路37号鑫城汽修厂涉嫌垄断",
        )
        semantic = semantic_with("pois", {
            "surface":"薛埠镇", "canonical":"薛埠镇",
            "source_field":"case_content_clean", "evidence":"薛埠镇",
        })
        semantic["entities"]["pois"].extend([
            {
                "surface":"东环路37号", "canonical":"东环路37号",
                "source_field":"case_content_clean", "evidence":"东环路37号",
            },
            {
                "surface":"鑫城汽修厂", "canonical":"鑫城汽修厂",
                "source_field":"case_content_clean", "evidence":"鑫城汽修厂",
            },
        ])
        validation = validate_semantic_output(order, semantic)
        self.assertIn("poi_shape_conflict:entities.pois.0", validation["warnings"])
        self.assertIn("poi_shape_conflict:entities.pois.1", validation["warnings"])
        cleaned, actions = sanitize_semantic_output(semantic, validation["warnings"], order=order)
        self.assertEqual(
            [item["canonical"] for item in cleaned["entities"]["pois"]],
            ["鑫城汽修厂"],
        )
        self.assertIn("dropped_invalid_candidate:entities.pois.0", actions)
        self.assertIn("dropped_invalid_candidate:entities.pois.1", actions)

        order["case_content_clean"] += "；路劲城小区物业问题"
        semantic["entities"]["pois"] = [{
            "surface":"路劲城小区", "canonical":"路劲城小区",
            "source_field":"case_content_clean", "evidence":"路劲城小区",
        }]
        self.assertNotIn(
            "poi_shape_conflict:entities.pois.0",
            validate_semantic_output(order, semantic)["warnings"],
        )

    def test_recovers_single_named_road_from_direction_or_address(self):
        order = dict(
            self.order,
            case_content_clean="河海西路复康路方向车辆占道；东环路37号鑫城汽修厂",
        )
        semantic = semantic_with("roads", {
            "surface":"复康路方向", "canonical":"复康路方向",
            "source_field":"case_content_clean", "evidence":"河海西路复康路方向",
        })
        semantic["entities"]["roads"].extend([
            {
                "surface":"东环路37号", "canonical":"东环路37号",
                "source_field":"case_content_clean", "evidence":"东环路37号",
            },
            {
                "surface":"一路顺风有限公司", "canonical":"一路顺风有限公司",
                "source_field":"case_content_clean", "evidence":"一路顺风有限公司",
            },
        ])
        order["case_content_clean"] += "；一路顺风有限公司"
        enriched, actions = enrich_semantic_output(order, semantic)
        self.assertEqual(
            [item["canonical"] for item in enriched["entities"]["roads"]],
            ["复康路", "东环路", "一路顺风有限公司"],
        )
        self.assertIn("recovered_named_road:entities.roads.0", actions)
        self.assertIn("recovered_named_road:entities.roads.1", actions)
        self.assertNotIn("recovered_named_road:entities.roads.2", actions)

    def test_filters_normal_service_actions_but_keeps_observed_failure(self):
        cases = (
            ("咨询办理", "咨询办理"),
            ("注册有限责任公司", "注册有限责任公司"),
            ("希望转入盲童学校", "希望转入盲童学校"),
        )
        for evidence, canonical in cases:
            with self.subTest(evidence=evidence):
                order = dict(self.order, case_content_clean=evidence, case_goal_clean=evidence)
                semantic = semantic_with("problem_behaviors", {
                    "surface":evidence, "canonical":canonical,
                    "source_field":"case_content_clean", "evidence":evidence,
                })
                result = validate_semantic_output(order, semantic)
                self.assertTrue(any(
                    warning.startswith((
                        "normal_service_action_as_behavior:",
                        "request_action_as_behavior:",
                    ))
                    for warning in result["warnings"]
                ))

        observed_cases = (
            ("现在确按照湖南的系数给我办理退休", "按湖南系数办理退休"),
            ("网上办理失败", "办理失败"),
            ("公司注册受阻", "注册受阻"),
            ("窗口拒绝办理", "拒绝办理"),
            ("申请一直不通过", "申请不通过"),
        )
        for observed, canonical in observed_cases:
            with self.subTest(observed=observed):
                order = dict(
                    self.order,
                    case_content_clean=observed,
                    case_goal_clean="要求按常州政策办理",
                )
                semantic = semantic_with("problem_behaviors", {
                    "surface":observed, "canonical":canonical,
                    "source_field":"case_content_clean", "evidence":observed,
                })
                self.assertFalse(any(
                    warning.startswith((
                        "normal_service_action_as_behavior:",
                        "request_action_as_behavior:",
                    ))
                    for warning in validate_semantic_output(order, semantic)["warnings"]
                ))

    def test_rejects_unsupported_canonical_state_and_deduplicates_variants(self):
        order = dict(self.order, case_content_clean="皮蛋属于三无食品；严重影响自己家的采光")
        semantic = semantic_with("problem_behaviors", {
            "surface":"三无食品", "canonical":"销售过期食品",
            "source_field":"case_content_clean", "evidence":"三无食品",
        })
        result = validate_semantic_output(order, semantic)
        self.assertIn("canonical_evidence_conflict:entities.problem_behaviors.0", result["warnings"])

        semantic["entities"]["problem_behaviors"] = [
            {
                "surface":"严重影响自己家的采光", "canonical":"严重影响采光",
                "source_field":"case_content_clean", "evidence":"严重影响自己家的采光",
            },
            {
                "surface":"严重影响自己家的采光", "canonical":"严重影响自己家的采光",
                "source_field":"case_content_clean", "evidence":"严重影响自己家的采光",
            },
        ]
        enriched, actions = enrich_semantic_output(order, semantic)
        self.assertEqual(len(enriched["entities"]["problem_behaviors"]), 1)
        self.assertEqual(
            enriched["entities"]["problem_behaviors"][0]["canonical"],
            "严重影响采光",
        )
        self.assertIn("deduplicated_entity_variant:entities.problem_behaviors.1", actions)

    def test_keeps_one_reliable_intent_and_replaces_weak_help_label(self):
        order = dict(self.order, case_content_clean="服务对象投诉商家违法，要求查处")
        semantic = semantic_with("problem_objects", {
            "surface":"商家", "canonical":"商家",
            "source_field":"case_content_clean", "evidence":"商家",
        })
        semantic["discourse"]["intents"] = [{"label":"求助", "evidence":"要求"}]
        enriched, actions = enrich_semantic_output(order, semantic)
        self.assertEqual(enriched["discourse"]["intents"], [{"label":"投诉", "evidence":"投诉"}])
        self.assertIn("replaced_weak_intent:投诉", actions)

        order["case_content_clean"] = "服务对象建议加强培训，同时咨询长途客车处理方式"
        semantic["discourse"]["intents"] = [
            {"label":"求助", "evidence":"要求"},
            {"label":"建议", "evidence":"建议"},
        ]
        enriched, _ = enrich_semantic_output(order, semantic)
        self.assertEqual(enriched["discourse"]["intents"], [{"label":"建议", "evidence":"建议"}])

    def test_requires_direct_emotion_and_satisfaction_evidence(self):
        order = dict(self.order, case_content_clean="公司威胁不给工资，服务对象要求处理")
        semantic = semantic_with("problem_behaviors", {
            "surface":"威胁不给工资", "canonical":"拖欠工资",
            "source_field":"case_content_clean", "evidence":"威胁不给工资",
        })
        semantic["discourse"]["emotions"] = [{
            "label":"不满", "intensity":2, "evidence":"威胁不给",
        }]
        semantic["discourse"]["satisfaction"] = {
            "label":"dissatisfied", "target":"公司", "evidence":"不给",
        }
        validation = validate_semantic_output(order, semantic)
        self.assertIn("unsupported_emotion_evidence:discourse.emotions.0", validation["warnings"])
        self.assertIn("unsupported_satisfaction_evidence", validation["warnings"])
        cleaned, _ = sanitize_semantic_output(semantic, validation["warnings"], order=order)
        self.assertEqual(cleaned["discourse"]["emotions"], [])
        self.assertEqual(cleaned["discourse"]["satisfaction"]["label"], "unknown")

        direct_order = dict(self.order, case_content_clean="服务对象对此不满意，其不认可处理结果")
        semantic["discourse"]["emotions"] = [{
            "label":"不满", "intensity":2, "evidence":"服务对象对此不满意",
        }]
        semantic["discourse"]["satisfaction"] = {
            "label":"dissatisfied", "target":"处理结果", "evidence":"其不认可",
        }
        validation = validate_semantic_output(direct_order, semantic)
        self.assertNotIn("unsupported_emotion_evidence:discourse.emotions.0", validation["warnings"])
        self.assertNotIn("unsupported_satisfaction_evidence", validation["warnings"])

        semantic["discourse"]["emotions"] = [{
            "label":"认可", "intensity":2, "evidence":"不认可",
        }]
        semantic["discourse"]["satisfaction"] = {
            "label":"satisfied", "target":"处理结果", "evidence":"不认可",
        }
        validation = validate_semantic_output(direct_order, semantic)
        self.assertIn("unsupported_emotion_evidence:discourse.emotions.0", validation["warnings"])
        self.assertIn("unsupported_satisfaction_evidence", validation["warnings"])

    def test_semantic_gap_warnings_are_audit_only(self):
        order = dict(
            self.order,
            case_content_clean="天宁区政务服务中心疑似有人盗用手机号办理医保业务",
        )
        semantic = semantic_with("problem_objects", {
            "surface":"手机号", "canonical":"手机号",
            "source_field":"case_content_clean", "evidence":"手机号",
        })
        semantic["entities"]["problem_objects"] = []
        result = validate_semantic_output(order, semantic)
        self.assertIn("semantic_gap:problem_objects", result["warnings"])
        self.assertIn("semantic_gap:problem_behaviors", result["warnings"])
        self.assertIn("semantic_gap:pois", result["warnings"])
        self.assertEqual(result["status"], "accepted_with_warnings")

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
