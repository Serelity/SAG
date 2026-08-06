import json
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_prompt import (
    CURRENT_MARKERS,
    HISTORY_MARKERS,
    FINAL_JSON_SKELETON,
    build_repair_prompt,
    build_semantic_prompt,
    select_content_windows,
)


class TestSemanticPrompt(unittest.TestCase):
    def test_windowing_keeps_current_claim_near_tail(self):
        text = "前期反映" + ("历史答复" * 500) + "现服务对象表示仍未解决，再次要求拆除违建。"
        windows = select_content_windows(text, max_chars=300)
        self.assertTrue(windows["truncated"])
        self.assertIn("现服务对象表示仍未解决", windows["current_window"])
        self.assertLessEqual(len(windows["combined"]), 300)
        self.assertEqual(windows["kept_chars"], len(windows["combined"]))
        self.assertEqual(windows["original_chars"], len(text))

    def test_windowing_uses_the_last_current_marker(self):
        text = "H" * 250 + "仍未解决" + "M" * 250 + "再次要求" + "T" * 250
        windows = select_content_windows(text, max_chars=100)
        self.assertIn("再次要求", windows["current_window"])
        self.assertNotIn("仍未解决", windows["current_window"])

    def test_overlapping_slices_are_deduplicated_and_hard_capped(self):
        text = "a" * 10 + "现要求" + "b" * 97
        windows = select_content_windows(text, max_chars=100)
        self.assertTrue(windows["truncated"])
        self.assertEqual(
            windows["combined"],
            windows["head"] + windows["current_window"] + windows["tail"],
        )
        self.assertLessEqual(len(windows["combined"]), 100)
        # Source is homogeneous around the overlaps, so lengths prove that no
        # source position has been repeated between the returned slices.
        self.assertLessEqual(
            len(windows["head"]) + len(windows["current_window"]) + len(windows["tail"]),
            100,
        )

    def test_zero_and_negative_budgets_are_safe_and_deterministic(self):
        expected = {
            "head": "",
            "current_window": "",
            "tail": "",
            "combined": "",
            "truncated": True,
            "original_chars": 4,
            "kept_chars": 0,
        }
        self.assertEqual(select_content_windows("正文内容", 0), expected)
        self.assertEqual(select_content_windows("正文内容", -9), expected)
        self.assertEqual(select_content_windows("正文内容", "bad"), expected)
        self.assertFalse(select_content_windows("", 0)["truncated"])

    def test_short_text_is_retained_once(self):
        windows = select_content_windows("短文本", max_chars=20)
        self.assertEqual(windows, {
            "head": "短文本",
            "current_window": "",
            "tail": "",
            "combined": "短文本",
            "truncated": False,
            "original_chars": 3,
            "kept_chars": 3,
        })

    def test_no_current_marker_keeps_head_and_tail_without_false_current_window(self):
        text = "前期反映" + "x" * 120 + "部门答复已处理" + "y" * 120
        windows = select_content_windows(text, max_chars=100)
        self.assertEqual(windows["current_window"], "")
        self.assertTrue(windows["head"].startswith("前期反映"))
        self.assertTrue(windows["tail"].endswith("y" * 20))
        self.assertEqual(len(windows["combined"]), 100)

    def test_prompt_is_open_domain_and_contains_difficult_boundaries(self):
        order = {
            "doc_id": "order_1",
            "title_clean": "",
            "case_content_clean": "港龙新港城北门口有电动车摆摊，请优先处理，谢谢！",
            "case_goal_clean": "希望处理",
            "address_detail_clean": "",
            "metadata": {"service_object_type": "求助", "area_code_area": "武进区"},
        }
        prompt = build_semantic_prompt(order, {"max_input_chars": 1500})
        self.assertIn("开放式识别", prompt)
        self.assertIn("港龙新港城", prompt)
        self.assertIn("不能判定满意", prompt)
        self.assertIn("诉求动作", prompt)
        self.assertIn('"problem_objects"', prompt)
        self.assertIn("intents 必须是最多3个 {label,evidence} 对象", prompt)
        self.assertIn("禁止输出字符串数组、英文标签或“投诉举报”等组合标签", prompt)
        self.assertNotIn('"confidence"', prompt)

    def test_prompt_contains_all_approved_cross_domain_examples(self):
        prompt = build_semantic_prompt({"case_content_clean": "合成工单"}, {})
        for expected in (
            "和平路路灯连续三天不亮",
            "港龙新港城北门口",
            "树枝遮挡交通标志",
            "培训机构突然闭店",
            "收费员拒绝开票",
            "部门答复称车位已清理",
            "咨询办理医疗器械经营许可证",
            "人民路与花东街交叉口乱摆摊",
        ):
            self.assertIn(expected, prompt)
        self.assertEqual(prompt.count("示例 "), 8)

    def test_prompt_has_exact_final_skeleton_and_entity_item_field(self):
        prompt = build_semantic_prompt({"case_content_clean": "正文"}, {})
        skeleton = json.dumps(FINAL_JSON_SKELETON, ensure_ascii=False, separators=(",", ":"))
        self.assertIn(skeleton, prompt)
        self.assertEqual(FINAL_JSON_SKELETON, {
            "event_summary": "",
            "entities": {
                "problem_objects": [],
                "problem_behaviors": [],
                "roads": [],
                "intersections": [],
                "pois": [],
            },
            "discourse": {
                "intents": [],
                "emotions": [],
                "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
                "urgency": {"level": "normal", "evidence": ""},
            },
        })
        self.assertIn("{surface, canonical, field, evidence}", prompt)
        self.assertNotIn("source_field", prompt)
        self.assertIn("只输出最终 JSON", prompt)
        self.assertIn("不要输出分析过程、推理步骤或思维链", prompt)

    def test_prompt_keeps_windowed_body_under_valid_source_field(self):
        order = {
            "case_content_clean": "前期反映" + "历史答复" * 100 + "现服务对象表示仍未解决。",
        }
        prompt = build_semantic_prompt(order, {"max_input_chars": 80})
        payload_marker = "当前脱敏工单 payload：\n"
        payload_text = prompt.split(payload_marker, 1)[1].split("\n\n请在内部", 1)[0]
        payload = json.loads(payload_text)

        self.assertIn("case_content_clean", payload)
        self.assertNotIn("case_content_windows", payload)
        self.assertLessEqual(len(payload["case_content_clean"]), 80)
        self.assertTrue(payload["input_window_info"]["truncated"])
        self.assertIn("绝不能作为 field/evidence", prompt)

    def test_prompt_labels_history_and_current_markers_without_rule_assertion(self):
        order = {
            "case_content_clean": "前期反映噪声。部门答复已处理，现再次反映仍未解决。",
        }
        prompt = build_semantic_prompt(order, {"max_input_chars": 10})
        for marker in ("前期反映", "部门答复"):
            self.assertIn(marker, HISTORY_MARKERS)
            self.assertIn(marker, prompt)
        for marker in ("现再次反映", "仍未解决"):
            self.assertIn(marker, CURRENT_MARKERS)
            self.assertIn(marker, prompt)
        self.assertIn('"truncated":true', prompt)

    def test_primary_prompt_only_includes_allowlisted_metadata_context(self):
        prompt = build_semantic_prompt({
            "case_content_clean": "合成工单",
            "metadata": {
                "service_object_type": "咨询",
                "area_code_area": "钟楼区",
                "raw_phone": "SHOULD_NOT_APPEAR",
                "raw_identity": {"name": "PRIVATE_NAME"},
            },
        }, {})
        self.assertIn("service_object_type", prompt)
        self.assertIn("钟楼区", prompt)
        self.assertNotIn("SHOULD_NOT_APPEAR", prompt)
        self.assertNotIn("PRIVATE_NAME", prompt)
        self.assertNotIn("raw_phone", prompt)

    def test_repair_prompt_only_repairs_and_excludes_raw_metadata(self):
        order = {
            "doc_id": "safe_order_id",
            "title_clean": "路灯故障",
            "case_content_clean": "和平路路灯不亮，希望维修。",
            "case_goal_clean": "希望维修",
            "address_detail_clean": "和平路",
            "metadata": {
                "area_code_area": "钟楼区",
                "raw_phone": "PRIVATE_PHONE",
                "raw_address": "PRIVATE_ADDRESS",
            },
        }
        original = '{"event_summary":"路灯不亮","entities":{"roads":[]}}'
        errors = ["missing_evidence:entities.roads.0", "invalid_source_field:entities.roads.0"]
        prompt = build_repair_prompt(order, original, errors, {"max_input_chars": 100})
        self.assertIn("只修复错误码指向的字段", prompt)
        self.assertIn(original, prompt)
        self.assertIn(errors[0], prompt)
        self.assertIn(errors[1], prompt)
        self.assertIn("必要 clean fields", prompt)
        self.assertIn("只输出最终 JSON", prompt)
        self.assertNotIn("PRIVATE_PHONE", prompt)
        self.assertNotIn("PRIVATE_ADDRESS", prompt)
        self.assertNotIn("metadata", prompt)
        self.assertNotIn("safe_order_id", prompt)
        self.assertNotIn('"confidence"', prompt)
        self.assertIn('"case_content_clean"', prompt)
        self.assertNotIn('"case_content_windows"', prompt)

    def test_runtime_config_matches_approved_values_exactly(self):
        path = Path(__file__).parents[1] / "configs" / "sag_semantic_extraction_qwen3_4b.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config, {
            "schema_version": "2.0",
            "prompt_version": "sag_semantic_v7",
            "model_id": "Qwen/Qwen3-4B",
            "model_path": "models/Qwen3-4B",
            "backend": "transformers",
            "attn_implementation": "sdpa",
            "cache_implementation": "dynamic",
            "vllm_gpu_memory_utilization": 0.85,
            "vllm_max_model_len": 4096,
            "vllm_max_num_seqs": 64,
            "vllm_enable_prefix_caching": False,
            "vllm_enable_chunked_prefill": False,
            "vllm_enforce_eager": False,
            "enable_thinking": False,
            "max_input_chars": 2200,
            "max_new_tokens": 640,
            "repair_max_new_tokens": 768,
            "temperature": 0.0,
            "batch_size": 8,
            "repair_batch_size": 8,
            "progress_every": 50,
            "checkpoint_every": 50,
            "max_repairs_per_order": 1,
            "length_bucket_boundaries": [600, 1400],
            "empty_cache_between_batches": False,
            "default_output": "outputs/work_order_semantics.qwen3_4b.jsonl",
            "default_rejects": "outputs/work_order_semantics.rejects.jsonl",
        })


if __name__ == "__main__":
    unittest.main()
