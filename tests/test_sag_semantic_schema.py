import unittest

from ragflow_style_pipeline.sag_semantic_schema import (
    ENTITY_GROUPS,
    GROUP_LIMITS,
    normalize_semantic_output,
    parse_semantic_json,
)


class TestSemanticSchema(unittest.TestCase):
    def test_parses_fenced_json_and_supplies_safe_defaults(self):
        parsed, warnings = parse_semantic_json('''```json
        {"event_summary":"咨询体检报告查询方式","entities":{},"discourse":{}}
        ```''')
        self.assertEqual(parsed["event_summary"], "咨询体检报告查询方式")
        self.assertEqual(parsed["entities"]["roads"], [])
        self.assertEqual(parsed["discourse"]["satisfaction"], {
            "label": "unknown", "target": "", "evidence": ""
        })
        self.assertEqual(parsed["discourse"]["urgency"], {
            "level": "normal", "evidence": ""
        })
        self.assertEqual(warnings, [])

    def test_rejects_invalid_enums_without_trusting_model_confidence(self):
        parsed, warnings = parse_semantic_json('''{
          "event_summary":"路灯故障",
          "entities":{"problem_objects":[{"surface":"路灯","canonical":"路灯","field":"case_content_clean","evidence":"路灯","confidence":0.8}]},
          "discourse":{"satisfaction":{"label":"very_happy","confidence":0.7},"urgency":{"level":"now"}},
          "confidence":0.99
        }''')
        self.assertNotIn("confidence", repr(parsed))
        self.assertEqual(
            parsed["entities"]["problem_objects"][0]["source_field"],
            "case_content_clean",
        )
        self.assertEqual(parsed["discourse"]["satisfaction"]["label"], "unknown")
        self.assertEqual(parsed["discourse"]["urgency"]["level"], "normal")
        self.assertEqual(
            warnings,
            ["invalid_satisfaction_label", "invalid_urgency_level"],
        )

    def test_recovers_only_complete_safe_json_variants(self):
        cases = (
            (
                '{"event_summary":"ok","entities":{},"discourse":{},}',
                "json_recovered_trailing_comma",
            ),
            (
                "{'event_summary':'ok','entities':{},'discourse':{}}",
                "json_recovered_python_literal",
            ),
            (
                '{"event_summary":"line\nbreak","entities":{},"discourse":{}}',
                "json_recovered_control_character",
            ),
        )
        for raw, expected_warning in cases:
            with self.subTest(expected_warning=expected_warning):
                parsed, warnings = parse_semantic_json(raw)
                self.assertTrue(parsed["event_summary"])
                self.assertEqual(warnings, [expected_warning])

    def test_python_literal_recovery_rejects_non_json_types(self):
        invalid_values = (
            "{'event_summary':'bad','entities':{'roads':set()},'discourse':{}}",
            '{"event_summary":"bad","entities":{},"discourse":{},"score":NaN}',
            '{"event_summary":"bad","entities":{},"discourse":{},"score":1e400}',
        )
        for raw in invalid_values:
            with self.subTest(raw=raw):
                parsed, warnings = parse_semantic_json(raw)
                self.assertEqual(parsed, normalize_semantic_output({}))
                self.assertEqual(warnings, ["json_parse_failed"])

    def test_reports_json_parse_failure(self):
        parsed, warnings = parse_semantic_json('{"event_summary":')
        self.assertEqual(parsed, normalize_semantic_output({}))
        self.assertEqual(warnings, ["json_parse_failed"])

    def test_does_not_recover_nested_object_from_truncated_outer_object(self):
        parsed, warnings = parse_semantic_json(
            '{"event_summary":"truncated","entities":{"roads":[]}'
        )
        self.assertEqual(parsed, normalize_semantic_output({}))
        self.assertEqual(warnings, ["json_parse_failed"])

    def test_extracts_first_complete_object_from_surrounding_text(self):
        parsed, warnings = parse_semantic_json(
            '模型说明 result={"event_summary":"含有 {括号} 和 \\"引号\\"",'
            '"entities":{},"discourse":{}} trailing {"event_summary":"第二个"}'
        )
        self.assertEqual(parsed["event_summary"], '含有 {括号} 和 "引号"')
        self.assertEqual(warnings, [])

    def test_skips_balanced_invalid_braces_before_valid_object(self):
        parsed, warnings = parse_semantic_json(
            '说明 {not-json} result='
            '{"event_summary":"ok","entities":{},"discourse":{}}'
        )
        self.assertEqual(parsed["event_summary"], "ok")
        self.assertEqual(warnings, [])

    def test_normalizes_legacy_window_source_paths_to_case_content(self):
        for field in (
            "case_content_windows",
            "case_content_windows.head",
            "case_content_windows.current_window",
            "case_content_windows.tail",
            "case_content_windows.combined",
        ):
            with self.subTest(field=field):
                parsed, warnings = parse_semantic_json(__import__("json").dumps({
                    "entities": {"problem_objects": [{
                        "surface": "路灯", "canonical": "路灯",
                        "field": field, "evidence": "路灯",
                    }]},
                }, ensure_ascii=False))
                self.assertEqual(
                    parsed["entities"]["problem_objects"][0]["source_field"],
                    "case_content_clean",
                )
                self.assertEqual(warnings, [])

    def test_normalizes_stable_entity_groups_items_and_limits(self):
        value = {
            "event_summary": 123,
            "unknown_top_level": "removed",
            "entities": {
                "problem_objects": [
                    {"surface": "对象%d" % index, "canonical": "", "source_field": "bad_field", "evidence": None}
                    for index in range(GROUP_LIMITS["problem_objects"] + 1)
                ],
                "problem_behaviors": "not-an-array",
                "roads": ["not-an-object", {"surface": "青洋路", "field": "title_clean"}],
                "unknown_group": [{"confidence": 1}],
            },
            "discourse": {},
        }
        parsed, warnings = parse_semantic_json(__import__("json").dumps(value, ensure_ascii=False))

        self.assertEqual(parsed["event_summary"], "")
        self.assertEqual(tuple(parsed["entities"]), ENTITY_GROUPS)
        self.assertEqual(len(parsed["entities"]["problem_objects"]), 3)
        self.assertEqual(parsed["entities"]["problem_objects"][0], {
            "surface": "对象0", "canonical": "", "source_field": "bad_field", "evidence": ""
        })
        self.assertEqual(parsed["entities"]["roads"], [
            {
                "surface": "not-an-object", "canonical": "not-an-object",
                "source_field": "", "evidence": "not-an-object",
            },
            {
                "surface": "青洋路", "canonical": "",
                "source_field": "title_clean", "evidence": "",
            },
        ])
        self.assertEqual(parsed["entities"]["intersections"], [])
        self.assertEqual(warnings, [
            "group_limit_exceeded:problem_objects",
            "malformed_entity_group:problem_behaviors",
            "coerced_entity_string:roads:0",
        ])

    def test_normalizes_discourse_arrays_enums_defaults_and_limits(self):
        value = {
            "discourse": {
                "intents": [
                    {"label": "咨询", "evidence": "怎么查"},
                    {"label": "非法", "evidence": "x"},
                    {"label": "投诉", "evidence": 3},
                    {"label": "建议", "evidence": "建议"},
                    {"label": "反馈", "evidence": "反馈"},
                ],
                "emotions": [
                    {"label": "焦虑", "intensity": 5, "evidence": "着急"},
                    {"label": "高兴", "intensity": 2},
                    {"label": "感谢", "intensity": 3, "evidence": "谢谢"},
                    {"label": "认可", "intensity": 2},
                ],
                "satisfaction": "bad",
                "urgency": [],
            }
        }
        parsed, warnings = parse_semantic_json(__import__("json").dumps(value, ensure_ascii=False))

        self.assertEqual(parsed["discourse"]["intents"], [
            {"label": "咨询", "evidence": "怎么查"},
            {"label": "投诉", "evidence": ""},
            {"label": "建议", "evidence": "建议"},
        ])
        self.assertEqual(parsed["discourse"]["emotions"], [
            {"label": "焦虑", "intensity": 1, "evidence": "着急"},
            {"label": "感谢", "intensity": 3, "evidence": "谢谢"},
        ])
        self.assertEqual(parsed["discourse"]["satisfaction"]["label"], "unknown")
        self.assertEqual(parsed["discourse"]["urgency"]["level"], "normal")
        self.assertEqual(warnings, [
            "invalid_intent_label",
            "intents_limit_exceeded",
            "invalid_emotion_intensity",
            "invalid_emotion_label",
            "emotions_limit_exceeded",
            "malformed_satisfaction",
            "malformed_urgency",
        ])

    def test_warns_for_malformed_top_level_containers_in_stable_order(self):
        parsed, warnings = parse_semantic_json(
            '{"entities":"bad","discourse":[]}'
        )
        defaults = normalize_semantic_output({})
        self.assertEqual(parsed["entities"], defaults["entities"])
        self.assertEqual(parsed["discourse"], defaults["discourse"])
        self.assertEqual(warnings, ["malformed_entities", "malformed_discourse"])

    def test_coerces_malformed_discourse_arrays(self):
        parsed, warnings = parse_semantic_json('''{
          "entities": {},
          "discourse": {
            "intents": {},
            "emotions": "angry",
            "satisfaction": {"label": "bad"},
            "urgency": {"level": "bad"}
          }
        }''')
        self.assertEqual(parsed["discourse"]["intents"], [])
        self.assertEqual(parsed["discourse"]["emotions"], [])
        self.assertEqual(warnings, [
            "malformed_intents",
            "malformed_emotions",
            "invalid_satisfaction_label",
            "invalid_urgency_level",
        ])

    def test_deduplicates_warnings_in_first_discovery_order(self):
        _, warnings = parse_semantic_json('''{
          "discourse": {
            "intents": [{"label":"坏标签"}, {"label":"仍是坏标签"}],
            "emotions": [
              {"label":"焦虑", "intensity":0},
              {"label":"感谢", "intensity":4}
            ],
            "satisfaction": {"label":"unknown"},
            "urgency": {"level":"normal"}
          }
        }''')
        self.assertEqual(warnings, [
            "invalid_intent_label",
            "invalid_emotion_intensity",
        ])

    def test_normalize_semantic_output_removes_unknown_and_confidence_recursively(self):
        normalized = normalize_semantic_output({
            "event_summary": "摘要",
            "entities": {"pois": [{
                "surface": "公园", "canonical": "公园", "field": "address_detail_clean",
                "evidence": "公园", "confidence": 0.9, "extra": {"confidence": 1},
            }]},
            "discourse": {
                "intents": [{"label": "咨询", "evidence": "咨询", "confidence": 1}],
                "emotions": [{"label": "感谢", "intensity": 2, "evidence": "谢谢", "confidence": 1}],
                "satisfaction": {"label": "satisfied", "target": "答复", "evidence": "满意"},
                "urgency": {"level": "high", "evidence": "尽快"},
            },
            "confidence": 1,
            "extra": True,
        })
        self.assertEqual(set(normalized), {"event_summary", "entities", "discourse"})
        self.assertNotIn("confidence", repr(normalized))
        self.assertEqual(normalized["discourse"]["emotions"][0]["intensity"], 2)

    def test_clears_evidence_for_normal_urgency(self):
        parsed, warnings = parse_semantic_json('''{
          "discourse":{"urgency":{"level":"normal","evidence":"请优先处理"}}
        }''')
        self.assertEqual(parsed["discourse"]["urgency"], {
            "level": "normal", "evidence": "",
        })
        self.assertEqual(warnings, ["cleared_normal_urgency_evidence"])

    def test_overflow_can_be_preserved_only_for_order_aware_extraction(self):
        raw = __import__("json").dumps({
            "entities": {
                "problem_objects": [
                    {
                        "surface": f"对象{index}", "canonical": f"对象{index}",
                        "field": "case_content_clean", "evidence": f"对象{index}",
                    }
                    for index in range(4)
                ]
            }
        }, ensure_ascii=False)
        stable, _ = parse_semantic_json(raw)
        extraction, warnings = parse_semantic_json(raw, preserve_overflow=True)
        self.assertEqual(len(stable["entities"]["problem_objects"]), 3)
        self.assertEqual(len(extraction["entities"]["problem_objects"]), 4)
        self.assertEqual(warnings, ["group_limit_exceeded:problem_objects"])

    def test_preserves_string_entity_for_order_aware_verification(self):
        parsed, warnings = parse_semantic_json('''{
          "entities":{"roads":["人民路","花东街"],"pois":[null]}
        }''')
        self.assertEqual(parsed["entities"]["roads"], [
            {
                "surface": "人民路", "canonical": "人民路",
                "source_field": "", "evidence": "人民路",
            },
            {
                "surface": "花东街", "canonical": "花东街",
                "source_field": "", "evidence": "花东街",
            },
        ])
        self.assertEqual(warnings, [
            "coerced_entity_string:roads:0",
            "coerced_entity_string:roads:1",
            "malformed_entity_item:pois:0",
        ])


if __name__ == "__main__":
    unittest.main()
