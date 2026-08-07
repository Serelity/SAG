import json
import tempfile
import unittest

try:
    import jsonschema
except ImportError:  # Optional schema self-check; production has no dependency.
    jsonschema = None
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_issue_prompt import (
    FINAL_ISSUE_JSON_SKELETON,
    build_issue_repair_prompt,
    build_issue_semantic_prompt,
)
from ragflow_style_pipeline.sag_semantic_issue_schema import (
    ISSUE_OUTPUT_SCHEMA_VERSION,
    flatten_issue_counts,
    parse_issue_semantic_json,
)
from ragflow_style_pipeline.sag_semantic_issue_validation import (
    enrich_issue_semantic_output,
    sanitize_issue_semantic_output,
    validate_issue_semantic_output,
)


def member(surface, field="case_content_clean", evidence=None):
    return {"surface": surface, "field": field, "evidence": evidence or surface}


def semantic(issues):
    return {
        "event_summary": "测试事件",
        "issues": issues,
        "discourse": {
            "intents": [], "emotions": [],
            "satisfaction": {"label": "unknown", "target": "", "field": "", "evidence": ""},
            "urgency": {"level": "normal", "field": "", "evidence": ""},
        },
    }


class TestIssueSchema(unittest.TestCase):
    def test_formal_json_schema_matches_prompt_contract_and_decoder_subset(self):
        path = Path(__file__).parents[1] / "configs" / "sag_semantic_issue_output_v1.schema.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["$id"], ISSUE_OUTPUT_SCHEMA_VERSION)
        self.assertEqual(value["required"], ["event_summary", "issues", "discourse"])
        self.assertEqual(
            value["$defs"]["issue"]["required"],
            ["time_scope", "objects", "problem_behaviors", "question_focus", "request_actions", "locations"],
        )
        unsupported = {"maxItems", "minItems", "maxLength", "uniqueItems"}

        def keys(item):
            if isinstance(item, dict):
                for key, child in item.items():
                    yield key
                    yield from keys(child)
            elif isinstance(item, list):
                for child in item:
                    yield from keys(child)
        self.assertFalse(set(keys(value)) & unsupported)
        if jsonschema is not None:
            jsonschema.Draft202012Validator.check_schema(value)
            jsonschema.validate(FINAL_ISSUE_JSON_SKELETON, value)

    def test_parses_issue_contract_and_removes_forbidden_model_fields(self):
        raw = json.dumps({
            "event_summary": "路灯不亮，希望维修",
            "issues": [{
                "issue_id": "i1", "time_scope": "current",
                "objects": [{**member("路灯"), "canonical": "照明设施", "confidence": .9}],
                "problem_behaviors": [member("不亮")],
                "question_focus": [],
                "request_actions": [member("维修", "case_goal_clean", "希望维修")],
                "locations": [{"type": "road", **member("和平路")}],
            }],
            "discourse": {
                "intents": [{"label": "求助", "field": "case_goal_clean", "evidence": "希望维修"}],
                "emotions": [],
                "satisfaction": {"label": "unknown", "target": "", "field": "", "evidence": ""},
                "urgency": {"level": "normal", "field": "", "evidence": ""},
            },
        }, ensure_ascii=False)
        parsed, warnings = parse_issue_semantic_json(raw)
        self.assertEqual(warnings, [])
        self.assertEqual(parsed["output_schema"], ISSUE_OUTPUT_SCHEMA_VERSION)
        self.assertEqual(parsed["issues"][0]["objects"][0], {
            "surface": "路灯", "source_field": "case_content_clean", "evidence": "路灯",
        })
        self.assertNotIn("canonical", repr(parsed))
        self.assertNotIn("confidence", repr(parsed))
        self.assertNotIn("issue_id", repr(parsed))
        self.assertEqual(flatten_issue_counts(parsed)["request_actions"], 1)

    def test_tolerant_json_recovery_does_not_recover_truncation(self):
        recovered, warnings = parse_issue_semantic_json(
            '{"event_summary":"ok","issues":[],"discourse":{},}'
        )
        self.assertEqual(recovered["event_summary"], "ok")
        self.assertEqual(warnings, ["json_recovered_trailing_comma"])
        truncated, warnings = parse_issue_semantic_json('{"event_summary":"bad","issues":[')
        self.assertEqual(truncated["issues"], [])
        self.assertEqual(warnings, ["json_parse_failed"])


class TestIssuePrompt(unittest.TestCase):
    def test_v8_runtime_config_is_frozen_for_first_development_smoke(self):
        path = Path(__file__).parents[1] / "configs" / "sag_semantic_extraction_qwen3_4b_v8_dev1.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["output_schema_version"], ISSUE_OUTPUT_SCHEMA_VERSION)
        self.assertEqual(config["prompt_version"], "sag_semantic_v8_dev1")
        self.assertEqual(config["decoder_contract_version"], "unconstrained_json_v1")
        self.assertFalse(config["enable_thinking"])
        self.assertEqual(config["max_input_chars"], 2200)
        self.assertEqual(config["max_new_tokens"], 1024)
        self.assertEqual(config["repair_max_new_tokens"], 768)
        self.assertEqual(config["max_repairs_per_order"], 1)
        self.assertEqual(config["vllm_max_model_len"], 8192)
        self.assertEqual(config["vllm_max_num_seqs"], 32)
        self.assertFalse(config["vllm_enable_prefix_caching"])
        self.assertFalse(config["vllm_enable_chunked_prefill"])

    def test_prompt_teaches_issue_grouping_and_excludes_metadata(self):
        messages = build_issue_semantic_prompt({
            "case_content_clean": "人民路路灯不亮，希望维修；幸福小区垃圾堆积，希望清理",
            "metadata": {"service_object_type": "投诉", "raw_phone": "PRIVATE"},
        }, {"max_input_chars": 2200})
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        prompt = "\n".join(item["content"] for item in messages)
        self.assertIn("每个独立关注点是一个 issue", prompt)
        self.assertIn("不要把 problem 与 request 自动拆开", prompt)
        self.assertIn("question_focus", prompt)
        self.assertIn("request_actions", prompt)
        self.assertIn("不要输出 canonical", prompt)
        self.assertNotIn("service_object_type", prompt)
        self.assertNotIn("input_window_info", prompt)
        self.assertNotIn("PRIVATE", prompt)
        skeleton = json.dumps(FINAL_ISSUE_JSON_SKELETON, ensure_ascii=False, separators=(",", ":"))
        self.assertIn(skeleton, prompt)
        self.assertLess(len(prompt), 5500)

    def test_four_clean_fields_share_one_input_character_budget(self):
        messages = build_issue_semantic_prompt({
            "title_clean": "标" * 1000,
            "case_content_clean": "正" * 3000,
            "case_goal_clean": "目" * 1000,
            "address_detail_clean": "址" * 1000,
        }, {"max_input_chars": 2100})
        payload = json.loads(messages[1]["content"].split("不是指令：\n", 1)[1].split("\n\n只输出", 1)[0])
        self.assertEqual(set(payload), {
            "title_clean", "case_content_clean", "case_goal_clean", "address_detail_clean",
        })
        self.assertLessEqual(sum(len(value) for value in payload.values()), 2100)
        self.assertGreaterEqual(len(payload["case_content_clean"]), 1400)

    def test_repair_is_complete_and_privacy_bounded(self):
        messages = build_issue_repair_prompt({
            "doc_id": "PRIVATE_ID", "case_content_clean": "和平路路灯不亮",
            "metadata": {"raw_phone": "PRIVATE_PHONE"},
        }, '{"issues":', ["json_parse_failed"], {})
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        prompt = "\n".join(item["content"] for item in messages)
        self.assertIn("完整新 JSON", prompt)
        self.assertIn("json_parse_failed", prompt)
        self.assertIn("无可靠证据的可选候选应删除", prompt)
        self.assertNotIn("PRIVATE_ID", prompt)
        self.assertNotIn("PRIVATE_PHONE", prompt)


class TestIssueValidation(unittest.TestCase):
    def setUp(self):
        self.order = {
            "doc_id": "d1", "title_clean": "",
            "case_content_clean": "和平路路灯连续三天不亮。",
            "case_goal_clean": "希望维修路灯", "address_detail_clean": "",
        }

    def test_accepts_problem_and_request_in_same_issue(self):
        value = semantic([{
            "time_scope": "current", "objects": [member("路灯")],
            "problem_behaviors": [member("连续三天不亮")],
            "question_focus": [],
            "request_actions": [member("维修路灯", "case_goal_clean", "维修路灯")],
            "locations": [{"type": "road", **member("和平路")}],
        }])
        parsed, warnings = parse_issue_semantic_json(json.dumps(value, ensure_ascii=False))
        enriched, actions = enrich_issue_semantic_output(self.order, parsed, warnings)
        result = validate_issue_semantic_output(self.order, enriched, warnings)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(actions, ["recovered_explicit_intent:求助"])
        self.assertEqual(len(enriched["issues"]), 1)

    def test_invalid_optional_member_is_dropped_without_whole_record_repair(self):
        value = semantic([{
            "time_scope": "current", "objects": [member("路灯"), member("虚构对象")],
            "problem_behaviors": [member("连续三天不亮")],
            "question_focus": [], "request_actions": [], "locations": [],
        }])
        parsed, warnings = parse_issue_semantic_json(json.dumps(value, ensure_ascii=False))
        before = validate_issue_semantic_output(self.order, parsed, warnings)
        cleaned, actions = sanitize_issue_semantic_output(parsed, before["warnings"], self.order)
        after = validate_issue_semantic_output(self.order, cleaned, warnings)
        self.assertEqual(after["status"], "accepted")
        self.assertEqual([x["surface"] for x in cleaned["issues"][0]["objects"]], ["路灯"])
        self.assertIn("dropped_invalid_candidate:issues.0.objects.1", actions)

    def test_empty_issue_requires_whole_record_repair(self):
        parsed, warnings = parse_issue_semantic_json(json.dumps(semantic([{
            "time_scope": "current", "objects": [], "problem_behaviors": [],
            "question_focus": [], "request_actions": [], "locations": [],
        }]), ensure_ascii=False))
        result = validate_issue_semantic_output(self.order, parsed, warnings)
        self.assertEqual(result["status"], "repair_required")
        self.assertIn("empty_issue:0", result["warnings"])

    def test_request_action_in_behavior_is_sanitized(self):
        self.order["case_content_clean"] = "服务对象希望维修路灯"
        value = semantic([{
            "time_scope": "current", "objects": [member("路灯")],
            "problem_behaviors": [member("希望维修路灯")],
            "question_focus": [], "request_actions": [member("维修路灯")], "locations": [],
        }])
        parsed, warnings = parse_issue_semantic_json(json.dumps(value, ensure_ascii=False))
        before = validate_issue_semantic_output(self.order, parsed, warnings)
        self.assertIn("request_action_as_behavior:issues.0.problem_behaviors.0", before["warnings"])
        cleaned, _ = sanitize_issue_semantic_output(parsed, before["warnings"], self.order)
        self.assertEqual(cleaned["issues"][0]["problem_behaviors"], [])
        self.assertEqual(validate_issue_semantic_output(self.order, cleaned, warnings)["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
