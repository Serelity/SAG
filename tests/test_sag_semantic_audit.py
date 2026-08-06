import json
import tempfile
import unittest
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_audit import (
    audit_candidate_ledger_against_gold,
    build_eval_manifest,
    build_private_annotation_packet,
    compare_gold_annotations,
    evaluate_semantic_gold,
    merge_adjudicated_gold,
    prepare_annotation_round,
    profile_semantic_input,
    project_gold_issues,
    replay_candidate_ledger,
    validate_gold_annotations,
)
from ragflow_style_pipeline.work_order_input import normalize_work_order


class TestSemanticAudit(unittest.TestCase):
    def _write_orders(self, path):
        rows = [
            {
                "doc_id": "order_secret_1",
                "case_content_clean": "服务对象咨询许可证如何办理",
                "case_goal_clean": "希望告知办理材料",
                "metadata": {
                    "service_object_type": "咨询",
                    "type1": "政务",
                    "type2": "审批",
                    "type3": "许可证",
                    "call_time": "2026-01-01 10:00:00",
                    "call_month": "2026-01",
                },
            },
            {
                "doc_id": "order_secret_2",
                "case_content_clean": "人民路路灯损坏，要求维修",
                "case_goal_clean": "要求维修",
                "metadata": {
                    "service_object_type": "求助",
                    "type1": "城乡建设",
                    "type2": "市政",
                    "type3": "路灯",
                    "call_time": "2026-01-02 10:00:00",
                    "call_month": "2026-01",
                },
            },
            {
                "doc_id": "order_secret_3",
                "case_content_clean": "部门答复已经处理，现服务对象不认可，花园小区问题仍未解决",
                "case_goal_clean": "再次要求处理",
                "metadata": {
                    "service_object_type": "投诉举报",
                    "type1": "住房保障",
                    "type2": "物业",
                    "type3": "小区管理",
                    "call_time": "2026-01-03 10:00:00",
                    "call_month": "2026-01",
                },
            },
            {
                "doc_id": "order_secret_4",
                "case_content_clean": "某公司注册失败",
                "case_goal_clean": "请求协助",
                "metadata": {
                    "service_object_type": "求助",
                    "type1": "经济综合",
                    "type2": "企业登记",
                    "type3": "注册",
                    "call_time": "2026-01-04 10:00:00",
                    "call_month": "2026-01",
                },
            },
            {
                "doc_id": "order_secret_5",
                "case_content_clean": "人民路与和平路交叉口拥堵",
                "case_goal_clean": "建议疏导",
                "metadata": {
                    "service_object_type": "意见建议",
                    "type1": "交通出行",
                    "type2": "道路",
                    "type3": "拥堵",
                    "call_time": "2026-01-05 10:00:00",
                    "call_month": "2026-01",
                },
            },
            {
                "doc_id": "order_secret_6",
                "case_content_clean": "查询医保业务进度",
                "case_goal_clean": "希望告知结果",
                "metadata": {
                    "service_object_type": "咨询",
                    "type1": "民生保障",
                    "type2": "医保",
                    "type3": "查询",
                    "call_time": "2026-01-06 10:00:00",
                    "call_month": "2026-01",
                },
            },
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_profile_is_aggregate_only_and_measures_payload_reuse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "orders.jsonl"
            self._write_orders(source)
            report = profile_semantic_input(source, head_size=2)
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertEqual(report["valid_records"], 6)
        self.assertEqual(report["invalid_records"], 0)
        self.assertEqual(report["field_presence"]["case_content_clean"]["count"], 6)
        self.assertIn("without_time", report["inference_payload_reuse"])
        self.assertTrue(report["source_sha256"].startswith("sha256:"))
        self.assertEqual(len(report["source_sha256"]), 71)
        self.assertEqual(report["head_population_drift"]["head"]["records"], 2)
        self.assertNotIn("服务对象咨询许可证如何办理", serialized)
        self.assertNotIn("order_secret_1", serialized)

    def test_manifest_is_deterministic_text_free_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "orders.jsonl"
            self._write_orders(source)
            first, first_report = build_eval_manifest(
                source, production_size=3, challenge_size=2, seed="fixed"
            )
            second, second_report = build_eval_manifest(
                source, production_size=3, challenge_size=2, seed="fixed"
            )
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertTrue(first_report["source_sha256"].startswith("sha256:"))
        self.assertTrue(first_report["manifest_content_sha256"].startswith("sha256:"))
        self.assertEqual(first_report["semantic_source_sha256"], "")
        self.assertEqual(len(first), 5)
        production = {
            (row["doc_id"], row["content_hash"])
            for row in first if row["subset"] == "production"
        }
        challenge = {
            (row["doc_id"], row["content_hash"])
            for row in first if row["subset"] == "challenge"
        }
        self.assertTrue(production.isdisjoint(challenge))
        self.assertTrue(all("case_content_clean" not in row for row in first))
        self.assertTrue(any(row["challenge_reasons"] for row in first if row["subset"] == "challenge"))
        self.assertTrue(any(
            "rare_type3" in row["challenge_reasons"]
            for row in first if row["subset"] == "challenge"
        ))

    def test_private_annotation_packet_requires_exact_manifest_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; manifest_path = tmp/"manifest.jsonl"
            self._write_orders(source)
            manifest, _ = build_eval_manifest(
                source, production_size=2, challenge_size=1, seed="packet"
            )
            manifest_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
                encoding="utf-8",
            )
            packet = build_private_annotation_packet(source, manifest_path)
            broken = [dict(manifest[0], content_hash="sha256:wrong")]
            manifest_path.write_text(
                json.dumps(broken[0], ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "manifest_identity_mismatch"):
                build_private_annotation_packet(source, manifest_path)
            old_schema = dict(manifest[0], schema="sag_semantic_eval_manifest_v1")
            manifest_path.write_text(
                json.dumps(old_schema, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "invalid_manifest_record"):
                build_private_annotation_packet(source, manifest_path)
        self.assertEqual(len(packet), 3)
        self.assertTrue(all(row["private"] for row in packet))
        self.assertTrue(all(row["schema"] == "sag_issue_gold_v2" for row in packet))
        self.assertTrue(all(row["manifest_provenance"]["records"] == 3 for row in packet))
        self.assertEqual(len({
            row["manifest_provenance"]["content_sha256"] for row in packet
        }), 1)
        self.assertTrue(all(row["annotation"]["status"] == "unlabeled" for row in packet))
        self.assertTrue(all(row["clean_fields"]["case_content_clean"] for row in packet))

    def test_prepare_annotation_round_requires_pristine_packet_and_isolates_copies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "orders.jsonl"
            manifest_path = root / "manifest.private.jsonl"
            packet_path = root / "packet.private.jsonl"
            self._write_orders(source)
            manifest, _report = build_eval_manifest(
                source, production_size=2, challenge_size=1, seed="annotation-round"
            )
            manifest_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
                encoding="utf-8",
            )
            packet = build_private_annotation_packet(source, manifest_path)
            packet_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet),
                encoding="utf-8",
            )
            left, right, report = prepare_annotation_round(
                packet_path, "annotator-a", "annotator-b"
            )
            packet[0]["annotation"]["status"] = "in_progress"
            packet_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "annotation_packet_not_pristine"):
                prepare_annotation_round(packet_path, "annotator-a", "annotator-b")
        self.assertEqual(len(left), 3)
        self.assertEqual(len(right), 3)
        self.assertEqual(report["records_per_annotator"], 3)
        self.assertEqual(report["annotator_count"], 2)
        self.assertEqual({row["annotation"]["annotator"] for row in left}, {"annotator-a"})
        self.assertEqual({row["annotation"]["annotator"] for row in right}, {"annotator-b"})
        self.assertEqual({row["annotation"]["status"] for row in left + right}, {"in_progress"})
        self.assertEqual(
            {row["annotation_round_provenance"]["round_id"] for row in left + right},
            {report["round_id"]},
        )
        left[0]["issues"].append({"private": "left-only"})
        self.assertEqual(right[0]["issues"], [])
        serialized_report = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("annotator-a", serialized_report)
        self.assertNotIn("order_secret", serialized_report)

    def _completed_annotation(self, annotator="annotator-a"):
        return {
            "schema": "sag_issue_gold_v2",
            "private": True,
            "subset": "challenge",
            "doc_id": "annotation-secret-1",
            "content_hash": "sha256:annotation-1",
            "manifest_provenance": {
                "schema": "sag_semantic_eval_manifest_v2",
                "records": 1,
                "content_sha256": "sha256:" + "a" * 64,
            },
            "annotation_round_provenance": {
                "schema": "sag_issue_annotation_round_v1",
                "round_id": "test-round",
                "source_packet_sha256": "sha256:" + "b" * 64,
            },
            "clean_fields": {
                "title_clean": "",
                "case_content_clean": "和平路路灯不亮，服务对象对此不满意",
                "case_goal_clean": "要求维修",
                "address_detail_clean": "",
            },
            "metadata": {"service_object_type": "求助"},
            "issues": [{
                "mode": "problem",
                "time_scope": "current",
                "objects": [{
                    "surface": "路灯", "field": "case_content_clean", "evidence": "路灯",
                }],
                "predicates": [{
                    "surface": "不亮", "field": "case_content_clean", "evidence": "不亮",
                }],
                "actions": [{
                    "surface": "维修", "field": "case_goal_clean", "evidence": "维修",
                }],
                "locations": [{
                    "type": "road", "surface": "和平路",
                    "field": "case_content_clean", "evidence": "和平路",
                }],
            }],
            "declared_intents": [{
                "label": "求助", "field": "case_goal_clean", "evidence": "要求维修",
            }],
            "direct_emotions": [{
                "label": "不满", "intensity": 2,
                "field": "case_content_clean", "evidence": "不满意",
            }],
            "satisfaction": {
                "label": "dissatisfied", "target": "路灯问题",
                "field": "case_content_clean", "evidence": "不满意",
            },
            "urgency": {"level": "normal", "evidence": ""},
            "annotation": {
                "annotator": annotator, "status": "completed", "notes": "",
            },
        }

    def test_gold_validation_is_safe_and_checks_grounding(self):
        valid = self._completed_annotation()
        invalid = json.loads(json.dumps(valid, ensure_ascii=False))
        invalid["doc_id"] = "annotation-secret-2"
        invalid["content_hash"] = "sha256:annotation-2"
        valid["manifest_provenance"]["records"] = 2
        invalid["manifest_provenance"]["records"] = 2
        invalid["issues"][0]["predicates"][0]["evidence"] = "不存在的证据"
        invalid["urgency"] = {"level": "normal", "evidence": "路灯"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gold.private.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (valid, invalid)),
                encoding="utf-8",
            )
            report = validate_gold_annotations(
                path, require_complete=True, expected_annotator="annotator-a"
            )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertEqual(report["records_read"], 2)
        self.assertEqual(report["valid_records"], 1)
        self.assertEqual(report["error_records"], 1)
        self.assertEqual(report["error_counts"]["issue_member_evidence_not_in_field"], 1)
        self.assertEqual(report["error_counts"]["normal_urgency_has_evidence"], 1)
        self.assertTrue(report["errors_present"])
        self.assertFalse(report["ready_for_agreement"])
        self.assertNotIn("annotation-secret", serialized)
        self.assertNotIn("不存在的证据", serialized)

    def test_gold_validation_rejects_same_doc_id_with_different_hash(self):
        first = self._completed_annotation()
        second = self._completed_annotation()
        first["manifest_provenance"]["records"] = 2
        second["manifest_provenance"]["records"] = 2
        second["content_hash"] = "sha256:different"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "duplicate-doc.private.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (first, second)),
                encoding="utf-8",
            )
            report = validate_gold_annotations(path, require_complete=True)
        self.assertEqual(report["duplicate_identities"], 0)
        self.assertEqual(report["duplicate_doc_ids"], 1)
        self.assertEqual(report["error_counts"]["duplicate_doc_id"], 1)
        self.assertTrue(report["errors_present"])

    def test_gold_validation_detects_missing_manifest_record(self):
        row = self._completed_annotation()
        row["manifest_provenance"]["records"] = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "truncated.private.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            report = validate_gold_annotations(path, require_complete=True)
        self.assertEqual(
            report["file_error_counts"]["manifest_record_count_mismatch"], 1
        )
        self.assertTrue(report["errors_present"])
        self.assertFalse(report["ready_for_evaluation"])

    def test_annotation_agreement_returns_safe_metrics_and_private_conflicts(self):
        left = self._completed_annotation("annotator-a")
        right = self._completed_annotation("annotator-b")
        right["issues"][0]["locations"] = []
        right["direct_emotions"] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "left.private.jsonl"
            right_path = Path(tmpdir) / "right.private.jsonl"
            left_path.write_text(json.dumps(left, ensure_ascii=False) + "\n", encoding="utf-8")
            right_path.write_text(json.dumps(right, ensure_ascii=False) + "\n", encoding="utf-8")
            report, conflicts = compare_gold_annotations(
                left_path, right_path,
                left_annotator="annotator-a", right_annotator="annotator-b",
            )
            wrong = json.loads(json.dumps(right, ensure_ascii=False))
            wrong["content_hash"] = "sha256:wrong"
            right_path.write_text(json.dumps(wrong, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "annotation_identity_sets_differ"):
                compare_gold_annotations(left_path, right_path)
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertEqual(report["shared_records"], 1)
        self.assertEqual(report["exact_records"], 0)
        self.assertEqual(report["conflict_records"], 1)
        self.assertIn("grounded_mentions", report["conflict_reason_counts"])
        self.assertIn("issue_attachment", report["conflict_reason_counts"])
        self.assertIn("discourse", report["conflict_reason_counts"])
        self.assertLess(report["grounded_mention_agreement"]["dice_f1"], 1.0)
        self.assertNotIn("annotation-secret", serialized)
        self.assertEqual(conflicts[0]["doc_id"], "annotation-secret-1")
        self.assertTrue(conflicts[0]["private"])
        self.assertEqual(conflicts[0]["adjudication"]["status"], "pending")

    def test_adjudication_merge_requires_explicit_untampered_resolution(self):
        exact_left = self._completed_annotation("annotator-a")
        exact_left["manifest_provenance"]["records"] = 2
        conflict_left = self._completed_annotation("annotator-a")
        conflict_left["doc_id"] = "annotation-secret-2"
        conflict_left["content_hash"] = "sha256:annotation-2"
        conflict_left["manifest_provenance"]["records"] = 2
        exact_right = json.loads(json.dumps(exact_left, ensure_ascii=False))
        exact_right["annotation"]["annotator"] = "annotator-b"
        conflict_right = json.loads(json.dumps(conflict_left, ensure_ascii=False))
        conflict_right["annotation"]["annotator"] = "annotator-b"
        conflict_right["issues"][0]["locations"] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left_path = root / "left.private.jsonl"
            right_path = root / "right.private.jsonl"
            conflicts_path = root / "conflicts.private.jsonl"
            left_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (
                    exact_left, conflict_left
                )), encoding="utf-8",
            )
            right_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (
                    exact_right, conflict_right
                )), encoding="utf-8",
            )
            _agreement, conflicts = compare_gold_annotations(
                left_path, right_path,
                left_annotator="annotator-a", right_annotator="annotator-b",
            )
            decision = conflicts[0]["adjudication"]
            decision.update({
                "status": "resolved",
                "adjudicator": "referee",
                "issues": conflict_left["issues"],
                "declared_intents": conflict_left["declared_intents"],
                "direct_emotions": conflict_left["direct_emotions"],
                "satisfaction": conflict_left["satisfaction"],
                "urgency": conflict_left["urgency"],
                "notes": "manual resolution",
            })
            conflicts_path.write_text(
                json.dumps(conflicts[0], ensure_ascii=False) + "\n", encoding="utf-8"
            )
            without_notes = json.loads(json.dumps(conflicts[0], ensure_ascii=False))
            without_notes["adjudication"]["notes"] = ""
            conflicts_path.write_text(
                json.dumps(without_notes, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "conflict_resolution_missing_notes"):
                merge_adjudicated_gold(
                    left_path, right_path, conflicts_path, adjudicator="referee"
                )
            conflicts_path.write_text(
                json.dumps(conflicts[0], ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "adjudicator_not_independent"):
                merge_adjudicated_gold(
                    left_path, right_path, conflicts_path, adjudicator="annotator-a"
                )
            rows, report = merge_adjudicated_gold(
                left_path, right_path, conflicts_path,
                adjudicator="referee",
                left_annotator="annotator-a", right_annotator="annotator-b",
            )
            final_path = root / "gold.private.jsonl"
            final_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            validation = validate_gold_annotations(
                final_path, require_complete=True, expected_annotator="referee"
            )
            tampered = json.loads(json.dumps(conflicts[0], ensure_ascii=False))
            tampered["clean_fields"]["case_content_clean"] = "tampered"
            conflicts_path.write_text(
                json.dumps(tampered, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "conflict_source_payload_changed"):
                merge_adjudicated_gold(
                    left_path, right_path, conflicts_path, adjudicator="referee"
                )
        self.assertEqual(len(rows), 2)
        self.assertEqual(report["exact_agreements"], 1)
        self.assertEqual(report["resolved_conflicts"], 1)
        self.assertTrue(validation["ready_for_evaluation"])
        self.assertEqual({row["annotation"]["status"] for row in rows}, {"adjudicated"})
        self.assertEqual(
            {row["adjudication_provenance"]["resolution"] for row in rows},
            {"exact_agreement", "explicit_conflict"},
        )

    def test_candidate_ledger_gold_audit_uses_latest_terminal_attempt(self):
        documents = [
            {
                "doc_id": "ledger-doc-1",
                "case_content_clean": "和平路方向路灯不亮",
                "case_goal_clean": "要求维修",
                "metadata": {"service_object_type": "求助"},
            },
            {
                "doc_id": "ledger-doc-2",
                "case_content_clean": "乙路井盖破损",
                "case_goal_clean": "请求处理",
                "metadata": {"service_object_type": "求助"},
            },
        ]
        normalized = [normalize_work_order(row) for row in documents]

        def mention(surface, field="case_content_clean"):
            return {"surface": surface, "field": field, "evidence": surface}

        gold_rows = []
        for order, objects, predicates, road in (
            (normalized[0], [mention("路灯")], [mention("不亮")], "和平路"),
            (normalized[1], [mention("井盖")], [mention("破损")], "乙路"),
        ):
            gold_rows.append({
                "schema": "sag_issue_gold_v2", "private": True, "subset": "challenge",
                "doc_id": order["doc_id"], "content_hash": order["content_hash"],
                "manifest_provenance": {
                    "schema": "sag_semantic_eval_manifest_v2", "records": 2,
                    "content_sha256": "sha256:" + "a" * 64,
                },
                "clean_fields": {
                    key: order.get(key, "")
                    for key in (
                        "title_clean", "case_content_clean", "case_goal_clean",
                        "address_detail_clean",
                    )
                },
                "metadata": order["metadata"],
                "issues": [{
                    "mode": "problem", "time_scope": "current",
                    "objects": objects, "predicates": predicates, "actions": [],
                    "locations": [{"type": "road", **mention(road)}],
                }],
                "declared_intents": [], "direct_emotions": [],
                "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
                "urgency": {"level": "normal", "evidence": ""},
                "annotation": {
                    "annotator": "referee", "status": "adjudicated", "notes": "done",
                },
            })

        def empty_semantic(summary=""):
            return {
                "event_summary": summary,
                "entities": {
                    "problem_objects": [], "problem_behaviors": [], "roads": [],
                    "intersections": [], "pois": [],
                },
                "discourse": {
                    "intents": [], "emotions": [],
                    "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
                    "urgency": {"level": "normal", "evidence": ""},
                },
            }

        primary = empty_semantic("")
        primary["entities"]["problem_objects"] = [{
            "surface": "路灯", "canonical": "路灯",
            "source_field": "case_content_clean", "evidence": "路灯",
        }]
        primary["entities"]["problem_behaviors"] = [{
            "surface": "要求维修", "canonical": "要求维修",
            "source_field": "case_goal_clean", "evidence": "要求维修",
        }]
        primary["entities"]["roads"] = [{
            "surface": "和平路方向", "canonical": "和平路方向",
            "source_field": "case_content_clean", "evidence": "和平路方向",
        }]
        primary["entities"]["pois"] = [{
            "surface": "和平路", "canonical": "和平路",
            "source_field": "case_content_clean", "evidence": "和平路",
        }]
        repair = empty_semantic("路灯不亮")
        for group, value in (
            ("problem_objects", "路灯"),
            ("problem_behaviors", "不亮"),
            ("roads", "和平路"),
        ):
            repair["entities"][group] = [{
                "surface": value, "canonical": value,
                "source_field": "case_content_clean", "evidence": value,
            }]
        incomplete = empty_semantic("")

        candidate_rows = []
        for sequence, order, phase, run_id, candidate in (
            (1, normalized[0], "primary", "run-a", primary),
            (2, normalized[0], "repair", "run-a", repair),
            (3, normalized[1], "primary", "run-b", incomplete),
        ):
            candidate_rows.append({
                "schema": "sag_semantic_candidate_ledger_v1", "private": True,
                "doc_id": order["doc_id"], "content_hash": order["content_hash"],
                "phase": phase, "run_attempt_id": run_id,
                "ledger_sequence": sequence, "model": "Qwen/Qwen3-4B",
                "prompt_version": "sag_semantic_v7",
                "decoder_contract_version": "unconstrained_json_v1",
                "candidate": candidate, "parse_warnings": [],
                "generation": {},
            })
        decision = {
            "schema": "sag_semantic_decision_ledger_v1", "private": True,
            "validator_version": "sag_semantic_validator_v1",
            "doc_id": normalized[0]["doc_id"],
            "content_hash": normalized[0]["content_hash"],
            "phase": "repair", "run_attempt_id": "run-a", "ledger_sequence": 2,
            "parse_warnings": [], "validation_before": {}, "actions": [],
            "validation_after": {"status": "accepted", "warnings": []},
            "final_counts": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.jsonl"
            gold_path = root / "gold.private.jsonl"
            candidates_path = root / "candidates.private.jsonl"
            decisions_path = root / "decisions.private.jsonl"
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in documents),
                encoding="utf-8",
            )
            gold_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold_rows),
                encoding="utf-8",
            )
            candidates_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidate_rows),
                encoding="utf-8",
            )
            decisions_path.write_text(
                json.dumps(decision, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            report, traces = audit_candidate_ledger_against_gold(
                input_path, gold_path, candidates_path, decisions_path
            )
        selected = report["scopes"]["selected"]["micro"]
        primary_metrics = report["scopes"]["primary"]["micro"]
        self.assertEqual(report["candidate_entries_total"], 3)
        self.assertEqual(report["records_with_any_candidates"], 2)
        self.assertEqual(report["records_with_terminal_candidates"], 1)
        self.assertEqual(report["incomplete_latest_attempts"], 1)
        self.assertEqual(report["gold_records_without_terminal_candidates"], 1)
        self.assertEqual(report["phase_records"], {"primary": 2, "repair": 1})
        self.assertEqual(selected["final_tp"], 3)
        self.assertEqual(selected["final_fp"], 0)
        self.assertEqual(selected["final_fn"], 0)
        self.assertGreaterEqual(primary_metrics["correctly_removed"], 2)
        self.assertGreaterEqual(primary_metrics["correct_additions"], 1)
        self.assertEqual(report["selected_attempts_missing_decisions"], 0)
        self.assertEqual(report["original_validator_status_counts"], {"accepted": 1})
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("ledger-doc", serialized)
        self.assertNotIn("和平路", serialized)
        self.assertTrue(all(row["private"] for row in traces))

    def test_candidate_replay_selects_latest_attempt_without_invoking_model(self):
        order_document = {
            "doc_id": "d-replay",
            "case_content_clean": "和平路路灯不亮",
            "metadata": {"service_object_type": "求助"},
        }
        order = normalize_work_order(order_document)
        base = {
            "event_summary": "路灯不亮",
            "entities": {
                "problem_objects": [], "problem_behaviors": [],
                "roads": [], "intersections": [], "pois": [],
            },
            "discourse": {
                "intents": [], "emotions": [],
                "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
                "urgency": {"level": "normal", "evidence": ""},
            },
        }
        repaired = json.loads(json.dumps(base, ensure_ascii=False))
        repaired["entities"]["roads"] = [{
            "surface": "和平路", "canonical": "和平路",
            "source_field": "case_content_clean", "evidence": "和平路",
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir); source = tmp/"orders.jsonl"; ledger = tmp/"candidates.private.jsonl"
            source.write_text(json.dumps(order_document, ensure_ascii=False) + "\n", encoding="utf-8")
            ledger.write_text("".join(
                json.dumps({
                    "schema": "sag_semantic_candidate_ledger_v1", "private": True,
                    "doc_id": order["doc_id"], "content_hash": order["content_hash"],
                    "phase": phase, "model": "m", "prompt_version": "p",
                    "decoder_contract_version": "d", "candidate": candidate,
                    "parse_warnings": [],
                }, ensure_ascii=False) + "\n"
                for phase, candidate in (("primary", base), ("repair", repaired))
            ), encoding="utf-8")
            rows, report = replay_candidate_ledger(source, ledger)
            with ledger.open("a", encoding="utf-8") as output:
                output.write(json.dumps({
                    "schema": "sag_semantic_candidate_ledger_v1", "private": True,
                    "doc_id": order["doc_id"], "content_hash": order["content_hash"],
                    "phase": "primary", "run_attempt_id": "new-attempt",
                    "ledger_sequence": 3, "model": "m", "prompt_version": "p",
                    "decoder_contract_version": "d", "candidate": base,
                    "parse_warnings": [],
                }, ensure_ascii=False) + "\n")
            latest_rows, latest_report = replay_candidate_ledger(source, ledger)
        self.assertEqual(report["candidates_read"], 2)
        self.assertEqual(report["records_replayed"], 1)
        self.assertEqual(rows[0]["selected_phase"], "repair")
        self.assertEqual(rows[0]["semantic"]["entities"]["roads"][0]["canonical"], "和平路")
        self.assertEqual(latest_report["candidates_read"], 3)
        self.assertEqual(latest_rows[0]["selected_phase"], "primary")
        self.assertEqual(latest_rows[0]["semantic"]["entities"]["roads"], [])

    def test_non_problem_predicates_are_not_projected_as_problem_behaviors(self):
        gold = {
            "doc_id": "question-1",
            "issues": [{
                "mode": "question", "time_scope": "current",
                "objects": [{"surface":"经营许可证"}],
                "predicates": [{"surface":"需要哪些材料"}],
                "actions": [], "locations": [],
            }],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gold.jsonl"
            path.write_text(json.dumps(gold, ensure_ascii=False) + "\n", encoding="utf-8")
            _orders, issues, links = project_gold_issues(path, flat=False)
            report = evaluate_semantic_gold(path, oracle_flat=True)
        self.assertEqual(issues[0]["mode"], "question")
        self.assertEqual(
            {(row["role"], row["entity_type"]) for row in links},
            {("object", "problem_object"), ("issue_predicate", "issue_predicate")},
        )
        self.assertNotIn("problem_behavior", report["mention_by_role"])

    def test_sag_evaluation_detects_false_flat_hyperedge_membership(self):
        gold_record = {
            "doc_id": "d1",
            "declared_intents": [{"label":"投诉", "evidence":"投诉"}],
            "direct_emotions": [],
            "satisfaction": {"label":"unknown", "target":"", "evidence":""},
            "urgency": {"level":"normal", "evidence":""},
            "issues": [
                {
                    "mode": "problem",
                    "objects": [{"surface": "公司"}],
                    "predicates": [{"surface": "拖欠工资"}],
                    "locations": [],
                },
                {
                    "mode": "problem",
                    "objects": [{"surface": "路灯"}],
                    "predicates": [{"surface": "不亮"}],
                    "locations": [{"type": "road", "surface": "和平路"}],
                },
            ],
        }
        flat_prediction = {
            "doc_id": "d1",
            "entities": {
                "problem_objects": [
                    {"surface": "公司"},
                    {"surface": "路灯"},
                ],
                "problem_behaviors": [
                    {"surface": "拖欠工资"},
                    {"surface": "不亮"},
                ],
                "roads": [{"surface": "和平路"}],
                "intersections": [],
                "pois": [],
            },
        }
        issue_prediction = {
            "doc_id": "d1", "issues": gold_record["issues"],
            "discourse": {
                "intents": [{"label":"投诉", "evidence":"投诉"}],
                "emotions": [],
                "satisfaction": {"label":"unknown", "target":"", "evidence":""},
                "urgency": {"level":"normal", "evidence":""},
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gold_path = tmp / "gold.jsonl"
            flat_path = tmp / "flat.jsonl"
            issue_path = tmp / "issue.jsonl"
            gold_path.write_text(json.dumps(gold_record, ensure_ascii=False) + "\n", encoding="utf-8")
            flat_path.write_text(json.dumps(flat_prediction, ensure_ascii=False) + "\n", encoding="utf-8")
            issue_path.write_text(json.dumps(issue_prediction, ensure_ascii=False) + "\n", encoding="utf-8")
            flat = evaluate_semantic_gold(gold_path, flat_path)
            issue = evaluate_semantic_gold(gold_path, issue_path)
            oracle_flat = evaluate_semantic_gold(gold_path, oracle_flat=True)
        self.assertEqual(flat["mention_micro"]["f1"], 1.0)
        self.assertGreater(flat["issue_co_membership"]["false_co_membership_rate"], 0.0)
        self.assertGreater(flat["object_behavior_attachment"]["false_positive"], 0)
        self.assertGreater(flat["location_attachment"]["false_positive"], 0)
        self.assertEqual(issue["issue_co_membership"]["f1"], 1.0)
        self.assertEqual(issue["object_behavior_attachment"]["f1"], 1.0)
        self.assertEqual(issue["location_attachment"]["f1"], 1.0)
        self.assertEqual(issue["hyperedge_exact"]["precision"], 1.0)
        self.assertEqual(issue["issue_frame_exact"]["precision"], 1.0)
        self.assertEqual(issue["discourse"]["intent_grounded"]["f1"], 1.0)
        self.assertEqual(issue["satisfaction_exact_accuracy"], 1.0)
        self.assertEqual(issue["urgency_exact_accuracy"], 1.0)
        self.assertEqual(oracle_flat["mention_micro"]["f1"], 1.0)
        self.assertGreater(oracle_flat["issue_co_membership"]["false_co_membership_rate"], 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            gold_path = Path(tmpdir) / "gold.jsonl"
            gold_path.write_text(json.dumps(gold_record, ensure_ascii=False) + "\n", encoding="utf-8")
            _orders, flat_issues, flat_links = project_gold_issues(gold_path, flat=True)
            _orders, issue_aware, issue_links = project_gold_issues(gold_path, flat=False)
        self.assertEqual(len(flat_issues), 1)
        self.assertEqual(len(issue_aware), 2)
        self.assertEqual(len(flat_links), len(issue_links))
        self.assertEqual({row["projection"] for row in flat_issues}, {"flat"})
        self.assertEqual({row["projection"] for row in issue_aware}, {"issue_aware"})


if __name__ == "__main__":
    unittest.main()
