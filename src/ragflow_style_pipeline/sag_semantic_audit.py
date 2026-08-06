"""Privacy-safe profiling, sampling, and SAG-oriented semantic evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_prompt import _semantic_payload
from ragflow_style_pipeline.sag_semantic_versions import (
    ADJUDICATION_MERGE_VERSION,
    ANNOTATION_AGREEMENT_VERSION,
    CANDIDATE_LEDGER_VERSION,
    DECISION_LEDGER_VERSION,
    EVAL_MANIFEST_VERSION,
    EVALUATION_VERSION,
    GOLD_SCHEMA_VERSION,
    GOLD_VALIDATION_VERSION,
    INPUT_PROFILE_VERSION,
    LEDGER_GOLD_AUDIT_VERSION,
    VALIDATOR_REPLAY_VERSION,
    VALIDATOR_VERSION,
)
from ragflow_style_pipeline.work_order_input import (
    CLEAN_FIELDS,
    WorkOrderInputError,
    normalize_work_order,
)

_METADATA_FIELDS = (
    "service_object_type",
    "type1",
    "type2",
    "type3",
    "area_code_area",
    "area_code_street",
    "order_source",
    "order_type",
    "order_status",
    "call_month",
)

_PROXY_PATTERNS = {
    "history_or_response": re.compile(
        r"部门答复|处理结果|答复如下|经核实|经调查|前期反映|原工单|回复称|已答复"
    ),
    "current_disagreement": re.compile(
        r"不认可|仍未解决|再次要求|现再次反映|有异议|不接受答复|再次来电|仍然|至今"
    ),
    "explicit_complaint": re.compile(r"投诉|举报|维权"),
    "consultation_question": re.compile(
        r"咨询|请问|如何|怎么|是否|哪些材料|办理流程|查询"
    ),
    "request_language": re.compile(
        r"希望|要求|请求|建议|请相关|请尽快|督促|协调|处理|解决"
    ),
    "negative_or_failure": re.compile(
        r"未|没有|尚未|拒绝|不予|无法|不能|失败|受阻|不通过|被拒"
    ),
    "road_form": re.compile(
        r"[\u4e00-\u9fffA-Za-z0-9]{2,18}(?:大道|公路|路|街|巷|弄|线)"
    ),
    "intersection_form": re.compile(
        r"[\u4e00-\u9fffA-Za-z0-9]{1,16}(?:大道|公路|路|街|巷|弄|线)"
        r"(?:与|和|及|/|、)"
        r"[\u4e00-\u9fffA-Za-z0-9]{1,16}(?:大道|公路|路|街|巷|弄|线)"
        r"(?:交叉口|路口|交界处|交汇处)?"
    ),
    "poi_form": re.compile(
        r"小区|新村|花园|家园|公园|学校|幼儿园|医院|市场|商场|广场|"
        r"服务中心|公司|工厂|汽修厂|产业园|工业园|园区|机构|派出所|"
        r"超市|大厦|体育馆"
    ),
    "doorplate_form": re.compile(
        r"(?:大道|公路|路|街|巷|弄)\d+(?:号|弄|栋|幢|单元)"
    ),
    "direct_dissatisfaction": re.compile(
        r"不满意|不认可|有异议|不接受答复|满意度不高"
    ),
    "direct_emotion": re.compile(r"气愤|愤怒|生气|焦虑|着急|担心|无奈|难过|悲伤"),
    "high_urgency": re.compile(r"催办|再次要求|仍未解决|长期未解决|至今未"),
    "critical_risk": re.compile(
        r"人身安全|火灾|燃气泄漏|漏气|坍塌|倒塌|触电|爆炸|生命危险"
    ),
}

# Frozen with GOLD_SCHEMA_VERSION. Do not import these from the model-output
# schema: changing the extraction contract must not silently change gold v1.
_GOLD_INTENTS = {"投诉", "举报", "求助", "咨询", "建议", "表扬", "催办", "反馈", "其他"}
_GOLD_EMOTIONS = {"愤怒", "不满", "焦虑", "无奈", "悲伤", "感谢", "认可"}
_GOLD_SATISFACTION_LABELS = {"satisfied", "dissatisfied", "mixed", "unknown"}
_GOLD_URGENCY_LEVELS = {"normal", "high", "critical"}
_ISSUE_MODES = {
    "problem", "question", "request", "suggestion", "praise",
    "historical_response", "current_stance",
}
_TIME_SCOPES = {"current", "historical"}
_LOCATION_TYPES = {"road", "intersection", "poi"}
_ANNOTATION_STATUSES = {"unlabeled", "in_progress", "completed", "adjudicated"}
_COMPLETE_ANNOTATION_STATUSES = {"completed", "adjudicated"}

_LENGTH_BUCKETS = (
    ("000-099", 99),
    ("100-199", 199),
    ("200-399", 399),
    ("400-599", 599),
    ("600-1399", 1399),
    ("1400+", math.inf),
)


def _text(value):
    return value if isinstance(value, str) else ""


def _sha(value):
    encoded = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _percentile(values, fraction):
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(
        ordered[lower] * (upper - position) + ordered[upper] * (position - lower),
        1,
    )


def _length_summary(values):
    if not values:
        return {key: 0 for key in ("min", "p50", "p75", "p90", "p95", "p99", "max", "mean")}
    return {
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "mean": round(sum(values) / len(values), 1),
    }


def _length_bucket(length):
    for name, upper in _LENGTH_BUCKETS:
        if length <= upper:
            return name
    return _LENGTH_BUCKETS[-1][0]


def _metadata_shape(values, records):
    total = sum(values.values())
    if not total:
        return {
            "cardinality": 0,
            "empty_rate": 0.0,
            "top1_share": 0.0,
            "top5_share": 0.0,
            "top20_share": 0.0,
            "singleton_values": 0,
        }

    def share(limit):
        return round(sum(count for _value, count in values.most_common(limit)) / total, 4)

    return {
        "cardinality": len(values),
        "empty_rate": round(values.get("<EMPTY>", 0) / records, 4) if records else 0.0,
        "top1_share": share(1),
        "top5_share": share(5),
        "top20_share": share(20),
        "singleton_values": sum(count == 1 for count in values.values()),
    }


def _payload_variants(order, max_input_chars):
    current = _semantic_payload(order, {"max_input_chars": max_input_chars}, include_metadata=True)
    metadata = current.get("metadata_context", {})
    return {
        "current": current,
        "without_time": {
            **current,
            "metadata_context": {
                key: value
                for key, value in metadata.items()
                if key not in {"call_time", "call_month"}
            },
        },
        "semantic_metadata_only": {
            **current,
            "metadata_context": {
                key: value
                for key, value in metadata.items()
                if key in {"service_object_type", "type1", "type2", "type3"}
            },
        },
        "clean_fields_only": {
            key: value for key, value in current.items() if key != "metadata_context"
        },
    }


def _descriptor(order, semantic_index=None):
    content = _text(order.get("case_content_clean"))
    goal = _text(order.get("case_goal_clean"))
    searchable = "\n".join(value for value in (content, goal) if value)
    metadata = order.get("metadata") if isinstance(order.get("metadata"), dict) else {}
    flags = sorted(name for name, pattern in _PROXY_PATTERNS.items() if pattern.search(searchable))
    if len([part for part in re.split(r"[。！？!?；;\n]+", content) if part.strip()]) >= 3:
        flags.append("multi_sentence_3plus")
    if semantic_index:
        semantic = semantic_index.get(order.get("doc_id"), {})
        validation = semantic.get("validation") if isinstance(semantic.get("validation"), dict) else {}
        if validation.get("repair_attempted"):
            flags.append("model_repair")
        flags.extend(
            "model_" + warning.replace(":", "_")
            for warning in validation.get("warnings", [])
            if isinstance(warning, str) and warning.startswith("semantic_gap:")
        )
    return {
        "doc_id": order.get("doc_id", ""),
        "content_hash": order.get("content_hash", ""),
        "service_object_type": _text(metadata.get("service_object_type")) or "<EMPTY>",
        "type1": _text(metadata.get("type1")) or "<EMPTY>",
        "type2": _text(metadata.get("type2")) or "<EMPTY>",
        "type3": _text(metadata.get("type3")) or "<EMPTY>",
        "length_bucket": _length_bucket(len(content)),
        "challenge_flags": sorted(set(flags)),
    }


def iter_normalized_orders(path):
    """Yield normalized orders while keeping raw text out of caller-visible errors."""
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                yield normalize_work_order(value)
            except (json.JSONDecodeError, WorkOrderInputError) as error:
                yield {
                    "_profile_error": type(error).__name__,
                    "_profile_line": line_number,
                }


def profile_semantic_input(path, max_input_chars=2200, head_size=32):
    """Return aggregate-only input statistics safe to share for diagnosis."""
    lengths = {field: [] for field in (*CLEAN_FIELDS, "chunk_text")}
    presence = Counter()
    metadata_counts = {field: Counter() for field in _METADATA_FIELDS}
    proxy_counts = Counter()
    payload_hashes = {name: Counter() for name in (
        "current", "without_time", "semantic_metadata_only", "clean_fields_only"
    )}
    payload_lengths = {name: [] for name in payload_hashes}
    failure_counts = Counter()
    descriptors = []
    records_read = 0

    for order in iter_normalized_orders(path):
        records_read += 1
        if "_profile_error" in order:
            failure_counts[order["_profile_error"]] += 1
            continue
        descriptors.append(_descriptor(order))
        for field in CLEAN_FIELDS:
            value = _text(order.get(field))
            presence[field] += bool(value)
            lengths[field].append(len(value))
        chunk = _text(order.get("chunk_text"))
        lengths["chunk_text"].append(len(chunk))
        searchable = "\n".join(_text(order.get(field)) for field in CLEAN_FIELDS)
        for name, pattern in _PROXY_PATTERNS.items():
            proxy_counts[name] += bool(pattern.search(searchable))
        content = _text(order.get("case_content_clean"))
        proxy_counts["multi_sentence_3plus"] += (
            len([part for part in re.split(r"[。！？!?；;\n]+", content) if part.strip()]) >= 3
        )
        proxy_counts["content_over_600"] += len(content) > 600
        proxy_counts["content_over_1400"] += len(content) > 1400
        proxy_counts["content_over_input_limit"] += len(content) > max_input_chars
        metadata = order.get("metadata") if isinstance(order.get("metadata"), dict) else {}
        for field, counts in metadata_counts.items():
            value = _text(metadata.get(field)) or "<EMPTY>"
            counts[value] += 1
        for name, payload in _payload_variants(order, max_input_chars).items():
            serialized = _stable_json(payload)
            payload_hashes[name][_sha(serialized)] += 1
            payload_lengths[name].append(len(serialized))

    valid_records = len(descriptors)
    service_counts = metadata_counts["service_object_type"]

    def descriptor_profile(rows):
        flags = Counter(flag for row in rows for flag in row["challenge_flags"])
        services = Counter(row["service_object_type"] for row in rows)
        lengths_by_bucket = Counter(row["length_bucket"] for row in rows)
        return {
            "records": len(rows),
            "service_object_type_counts": dict(services),
            "length_bucket_counts": dict(lengths_by_bucket),
            "proxy_counts": dict(flags),
        }

    payload_report = {}
    for name, counts in payload_hashes.items():
        duplicate_records = sum(count - 1 for count in counts.values() if count > 1)
        payload_report[name] = {
            "unique_payloads": len(counts),
            "duplicate_records_beyond_first": duplicate_records,
            "duplicate_rate": round(duplicate_records / valid_records, 4) if valid_records else 0.0,
            "duplicate_groups": sum(count > 1 for count in counts.values()),
            "largest_group": max(counts.values(), default=0),
            "serialized_chars": _length_summary(payload_lengths[name]),
        }

    return {
        "schema": INPUT_PROFILE_VERSION,
        "source_file": Path(path).name,
        "source_sha256": _file_sha256(path),
        "records_read": records_read,
        "valid_records": valid_records,
        "invalid_records": records_read - valid_records,
        "failure_type_counts": dict(failure_counts),
        "field_presence": {
            field: {
                "count": presence[field],
                "rate": round(presence[field] / valid_records, 4) if valid_records else 0.0,
            }
            for field in CLEAN_FIELDS
        },
        "field_length_chars": {
            field: _length_summary(values) for field, values in lengths.items()
        },
        "proxy_notice": "High-recall lexical proxies are sampling aids, not gold labels.",
        "proxy_counts": {
            name: {
                "count": count,
                "rate": round(count / valid_records, 4) if valid_records else 0.0,
            }
            for name, count in sorted(proxy_counts.items())
        },
        "metadata_shape": {
            field: _metadata_shape(counts, valid_records)
            for field, counts in metadata_counts.items()
        },
        "safe_distributions": {
            "service_object_type": dict(service_counts),
        },
        "inference_payload_reuse": payload_report,
        "head_population_drift": {
            "head_size": min(max(0, int(head_size)), valid_records),
            "head": descriptor_profile(descriptors[:max(0, int(head_size))]),
            "population": descriptor_profile(descriptors),
        },
    }


def build_private_annotation_packet(input_path, manifest_path):
    """Join a text-free manifest to desensitized fields for private annotation."""
    manifest = []
    with Path(manifest_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if (
                not isinstance(value, dict)
                or value.get("schema") != EVAL_MANIFEST_VERSION
                or not _text(value.get("doc_id"))
                or not _text(value.get("content_hash"))
                or _text(value.get("subset")) not in {"production", "challenge"}
            ):
                raise ValueError(f"line_{line_number}:invalid_manifest_record")
            manifest.append(value)

    manifest_provenance = {
        "schema": _text(manifest[0].get("schema")) if manifest else "",
        "records": len(manifest),
        "content_sha256": "sha256:" + _sha("\n".join(
            _stable_json(row) for row in manifest
        )),
    }
    target_ids = {str(row["doc_id"]) for row in manifest}
    if len(target_ids) != len(manifest):
        raise ValueError("duplicate_manifest_doc_id")
    orders = {}
    for order in iter_normalized_orders(input_path):
        if "_profile_error" in order:
            continue
        if order["doc_id"] in target_ids:
            orders[order["doc_id"]] = order
            if len(orders) == len(target_ids):
                break

    packet = []
    for row in manifest:
        doc_id = str(row["doc_id"])
        order = orders.get(doc_id)
        if order is None or row.get("content_hash") != order.get("content_hash"):
            raise ValueError(f"manifest_identity_mismatch:{doc_id}")
        packet.append({
            "schema": GOLD_SCHEMA_VERSION,
            "private": True,
            "subset": row.get("subset", ""),
            "doc_id": doc_id,
            "content_hash": order.get("content_hash", ""),
            "manifest_provenance": manifest_provenance,
            "clean_fields": {
                field: _text(order.get(field)) for field in CLEAN_FIELDS
            },
            "metadata": order.get("metadata") if isinstance(order.get("metadata"), dict) else {},
            "issues": [],
            "declared_intents": [],
            "direct_emotions": [],
            "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
            "urgency": {"level": "normal", "evidence": ""},
            "annotation": {
                "annotator": "",
                "status": "unlabeled",
                "notes": "",
            },
        })
    return packet


def _load_annotation_rows(path):
    rows = []
    parse_errors = Counter()
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                parse_errors["invalid_json"] += 1
                continue
            if not isinstance(value, dict):
                parse_errors["record_not_object"] += 1
                continue
            rows.append(value)
    return rows, parse_errors


def _evidence_field(item, clean_fields):
    field = _text(item.get("field") or item.get("source_field"))
    evidence = _text(item.get("evidence"))
    if field:
        return field
    matches = [
        name for name in CLEAN_FIELDS
        if evidence and evidence in _text(clean_fields.get(name))
    ]
    return matches[0] if len(matches) == 1 else ""


def _validate_evidence(item, clean_fields, errors, prefix, require_surface=False):
    if not isinstance(item, dict):
        errors.append(f"{prefix}_not_object")
        return
    evidence = _text(item.get("evidence"))
    field = _text(item.get("field") or item.get("source_field"))
    surface = _text(item.get("surface") or item.get("text"))
    if require_surface and not surface:
        errors.append(f"{prefix}_missing_surface")
    if not evidence:
        errors.append(f"{prefix}_missing_evidence")
    if evidence and not field:
        errors.append(f"{prefix}_missing_field")
    if field and field not in CLEAN_FIELDS:
        errors.append(f"{prefix}_invalid_field")
        return
    if field:
        if evidence and evidence not in _text(clean_fields.get(field)):
            errors.append(f"{prefix}_evidence_not_in_field")
    elif evidence and not any(evidence in _text(clean_fields.get(name)) for name in CLEAN_FIELDS):
        errors.append(f"{prefix}_evidence_not_in_clean_fields")
    if require_surface and surface and evidence and surface not in evidence:
        errors.append(f"{prefix}_surface_not_in_evidence")


def _validate_gold_row(row, require_complete=False, expected_annotator=""):
    errors = []
    warnings = []
    if row.get("schema") != GOLD_SCHEMA_VERSION:
        errors.append("invalid_gold_schema")
    if row.get("private") is not True:
        errors.append("private_marker_missing")
    if not _text(row.get("doc_id")):
        errors.append("missing_doc_id")
    if not _text(row.get("content_hash")):
        errors.append("missing_content_hash")
    if _text(row.get("subset")) not in {"production", "challenge"}:
        errors.append("invalid_subset")
    provenance = row.get("manifest_provenance")
    if not isinstance(provenance, dict):
        errors.append("manifest_provenance_not_object")
    else:
        if not _text(provenance.get("schema")):
            errors.append("manifest_provenance_missing_schema")
        records = provenance.get("records")
        if not isinstance(records, int) or isinstance(records, bool) or records < 1:
            errors.append("manifest_provenance_invalid_records")
        content_sha256 = _text(provenance.get("content_sha256"))
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", content_sha256):
            errors.append("manifest_provenance_invalid_hash")
    clean_fields = row.get("clean_fields")
    if not isinstance(clean_fields, dict):
        errors.append("clean_fields_not_object")
        clean_fields = {}
    else:
        for field in CLEAN_FIELDS:
            if field not in clean_fields:
                errors.append("clean_field_missing")
            elif not isinstance(clean_fields.get(field), str):
                errors.append("clean_field_not_string")
    if not isinstance(row.get("metadata"), dict):
        errors.append("metadata_not_object")
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        errors.append("annotation_not_object")
        annotation = {}
    status = _text(annotation.get("status"))
    annotator = _text(annotation.get("annotator"))
    if status not in _ANNOTATION_STATUSES:
        errors.append("invalid_annotation_status")
    if require_complete and status not in _COMPLETE_ANNOTATION_STATUSES:
        errors.append("annotation_not_complete")
    if status in _COMPLETE_ANNOTATION_STATUSES and not annotator:
        errors.append("completed_annotation_missing_annotator")
    if expected_annotator and annotator != expected_annotator:
        errors.append("unexpected_annotator")

    issues = row.get("issues")
    if not isinstance(issues, list):
        errors.append("issues_not_array")
        issues = []
    if status in _COMPLETE_ANNOTATION_STATUSES and not issues:
        errors.append("completed_annotation_without_issue")
    seen_issues = set()
    for issue in issues:
        if not isinstance(issue, dict):
            errors.append("issue_not_object")
            continue
        mode = _text(issue.get("mode"))
        time_scope = _text(issue.get("time_scope"))
        if mode not in _ISSUE_MODES:
            errors.append("issue_invalid_mode")
        if time_scope not in _TIME_SCOPES:
            errors.append("issue_invalid_time_scope")
        if mode == "historical_response" and time_scope != "historical":
            errors.append("historical_response_not_historical")
        if mode == "current_stance" and time_scope != "current":
            errors.append("current_stance_not_current")
        issue_key = _stable_json(issue)
        if issue_key in seen_issues:
            errors.append("duplicate_issue")
        seen_issues.add(issue_key)
        member_count = 0
        for group in ("objects", "predicates", "actions"):
            members = issue.get(group)
            if not isinstance(members, list):
                errors.append(f"issue_{group}_not_array")
                continue
            seen_members = set()
            for member in members:
                _validate_evidence(member, clean_fields, errors, "issue_member", require_surface=True)
                if isinstance(member, dict):
                    key = (
                        _normalized_mention(member),
                        _text(member.get("field") or member.get("source_field")),
                        _text(member.get("evidence")),
                    )
                    if key in seen_members:
                        errors.append("duplicate_issue_member")
                    seen_members.add(key)
                member_count += 1
        if mode in {"problem", "question", "historical_response", "current_stance"}:
            predicates = issue.get("predicates")
            if isinstance(predicates, list) and not predicates:
                warnings.append(f"{mode}_without_predicate")
        if mode in {"request", "suggestion"}:
            actions = issue.get("actions")
            if isinstance(actions, list) and not actions:
                warnings.append(f"{mode}_without_action")
        locations = issue.get("locations")
        if not isinstance(locations, list):
            errors.append("issue_locations_not_array")
            locations = []
        seen_locations = set()
        for location in locations:
            _validate_evidence(location, clean_fields, errors, "issue_location", require_surface=True)
            if isinstance(location, dict):
                if _text(location.get("type")) not in _LOCATION_TYPES:
                    errors.append("issue_location_invalid_type")
                key = (
                    _text(location.get("type")),
                    _normalized_mention(location),
                    _text(location.get("field") or location.get("source_field")),
                    _text(location.get("evidence")),
                )
                if key in seen_locations:
                    errors.append("duplicate_issue_location")
                seen_locations.add(key)
            member_count += 1
        if not member_count:
            errors.append("empty_issue")

    intents = row.get("declared_intents")
    if not isinstance(intents, list):
        errors.append("declared_intents_not_array")
        intents = []
    for intent in intents:
        if not isinstance(intent, dict) or _text(intent.get("label")) not in _GOLD_INTENTS:
            errors.append("intent_invalid_label")
        _validate_evidence(intent, clean_fields, errors, "intent")

    emotions = row.get("direct_emotions")
    if not isinstance(emotions, list):
        errors.append("direct_emotions_not_array")
        emotions = []
    for emotion in emotions:
        if not isinstance(emotion, dict) or _text(emotion.get("label")) not in _GOLD_EMOTIONS:
            errors.append("emotion_invalid_label")
        intensity = emotion.get("intensity") if isinstance(emotion, dict) else None
        if not isinstance(intensity, int) or isinstance(intensity, bool) or not 1 <= intensity <= 3:
            errors.append("emotion_invalid_intensity")
        _validate_evidence(emotion, clean_fields, errors, "emotion")

    satisfaction = row.get("satisfaction")
    if not isinstance(satisfaction, dict):
        errors.append("satisfaction_not_object")
        satisfaction = {}
    satisfaction_label = _text(satisfaction.get("label"))
    if satisfaction_label not in _GOLD_SATISFACTION_LABELS:
        errors.append("satisfaction_invalid_label")
    if satisfaction_label == "unknown":
        if any(_text(satisfaction.get(key)) for key in (
            "target", "evidence", "field", "source_field"
        )):
            errors.append("unknown_satisfaction_has_grounding")
    elif satisfaction_label in _GOLD_SATISFACTION_LABELS:
        if not _text(satisfaction.get("target")):
            errors.append("satisfaction_missing_target")
        _validate_evidence(satisfaction, clean_fields, errors, "satisfaction")

    urgency = row.get("urgency")
    if not isinstance(urgency, dict):
        errors.append("urgency_not_object")
        urgency = {}
    urgency_level = _text(urgency.get("level"))
    if urgency_level not in _GOLD_URGENCY_LEVELS:
        errors.append("urgency_invalid_level")
    if urgency_level == "normal":
        if any(_text(urgency.get(key)) for key in ("evidence", "field", "source_field")):
            errors.append("normal_urgency_has_evidence")
    elif urgency_level in _GOLD_URGENCY_LEVELS:
        _validate_evidence(urgency, clean_fields, errors, "urgency")

    if status == "in_progress":
        warnings.append("annotation_in_progress")
    if status == "unlabeled":
        warnings.append("annotation_unlabeled")
    return errors, warnings


def validate_gold_annotations(path, require_complete=False, expected_annotator=""):
    """Validate private issue annotation structure without exposing identifiers or text."""
    rows, parse_errors = _load_annotation_rows(path)
    error_counts = Counter(parse_errors)
    warning_counts = Counter()
    error_records = sum(parse_errors.values())
    status_counts = Counter()
    annotators = set()
    identities = set()
    doc_ids = set()
    duplicate_identities = 0
    duplicate_doc_ids = 0
    completed_records = 0
    valid_records = 0
    for row in rows:
        annotation = row.get("annotation") if isinstance(row.get("annotation"), dict) else {}
        status = _text(annotation.get("status")) or "<INVALID>"
        status_counts[status] += 1
        annotator = _text(annotation.get("annotator"))
        if annotator:
            annotators.add(annotator)
        completed_records += status in _COMPLETE_ANNOTATION_STATUSES
        doc_id = _text(row.get("doc_id"))
        identity = (doc_id, _text(row.get("content_hash")))
        duplicate = identity in identities
        duplicate_doc_id = doc_id in doc_ids
        if duplicate:
            duplicate_identities += 1
        if duplicate_doc_id:
            duplicate_doc_ids += 1
        identities.add(identity)
        doc_ids.add(doc_id)
        errors, warnings = _validate_gold_row(
            row,
            require_complete=require_complete,
            expected_annotator=expected_annotator,
        )
        if duplicate:
            errors.append("duplicate_record_identity")
        if duplicate_doc_id:
            errors.append("duplicate_doc_id")
        if errors:
            error_records += 1
            error_counts.update(errors)
        else:
            valid_records += 1
        warning_counts.update(warnings)
    total_records = len(rows) + sum(parse_errors.values())
    provenance_values = {
        _stable_json(row.get("manifest_provenance"))
        for row in rows if isinstance(row.get("manifest_provenance"), dict)
    }
    file_error_counts = Counter()
    if len(provenance_values) != 1:
        file_error_counts["manifest_provenance_not_unique"] += 1
    elif rows:
        expected_records = rows[0]["manifest_provenance"].get("records")
        if isinstance(expected_records, int) and not isinstance(expected_records, bool):
            if expected_records != len(rows):
                file_error_counts["manifest_record_count_mismatch"] += 1
    errors_present = bool(error_records or file_error_counts)
    return {
        "schema": GOLD_VALIDATION_VERSION,
        "gold_schema": GOLD_SCHEMA_VERSION,
        "private_input": True,
        "require_complete": bool(require_complete),
        "records_read": total_records,
        "parsed_records": len(rows),
        "unique_identities": len(identities),
        "duplicate_identities": duplicate_identities,
        "unique_doc_ids": len(doc_ids),
        "duplicate_doc_ids": duplicate_doc_ids,
        "completed_records": completed_records,
        "valid_records": valid_records,
        "error_records": error_records,
        "status_counts": dict(status_counts),
        "annotator_cardinality": len(annotators),
        "error_counts": dict(error_counts),
        "file_error_counts": dict(file_error_counts),
        "errors_present": errors_present,
        "warning_counts": dict(warning_counts),
        "ready_for_evaluation": (
            total_records > 0
            and not errors_present
            and completed_records == total_records
        ),
        "ready_for_agreement": (
            total_records > 0
            and not errors_present
            and completed_records == total_records
            and len(annotators) == 1
        ),
    }


def _grounded_member(item, role, clean_fields, entity_type=""):
    item = item if isinstance(item, dict) else {}
    return (
        role,
        entity_type,
        _normalized_mention(item),
        _evidence_field(item, clean_fields),
        _text(item.get("evidence")),
    )


def _grounded_issue_frames(row):
    clean_fields = row.get("clean_fields") if isinstance(row.get("clean_fields"), dict) else {}
    frames = set()
    for issue in row.get("issues", []) if isinstance(row.get("issues"), list) else []:
        if not isinstance(issue, dict):
            continue
        mode = _text(issue.get("mode"))
        predicate_role = "problem_behavior" if mode == "problem" else "issue_predicate"
        members = set()
        for item in issue.get("objects", []) if isinstance(issue.get("objects"), list) else []:
            members.add(_grounded_member(item, "problem_object", clean_fields))
        for item in issue.get("predicates", []) if isinstance(issue.get("predicates"), list) else []:
            members.add(_grounded_member(item, predicate_role, clean_fields))
        for item in issue.get("actions", []) if isinstance(issue.get("actions"), list) else []:
            members.add(_grounded_member(item, "request_action", clean_fields))
        for item in issue.get("locations", []) if isinstance(issue.get("locations"), list) else []:
            location_type = _text(item.get("type")) if isinstance(item, dict) else ""
            members.add(_grounded_member(item, "location", clean_fields, location_type))
        frames.add((frozenset(members), mode, _text(issue.get("time_scope"))))
    return frames


def _grounded_discourse(row):
    clean_fields = row.get("clean_fields") if isinstance(row.get("clean_fields"), dict) else {}

    def grounded(items, include_intensity=False):
        values = set()
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            base = (
                _text(item.get("label")),
                _evidence_field(item, clean_fields),
                _text(item.get("evidence")),
            )
            values.add(base + ((item.get("intensity"),) if include_intensity else ()))
        return frozenset(values)

    satisfaction = row.get("satisfaction") if isinstance(row.get("satisfaction"), dict) else {}
    urgency = row.get("urgency") if isinstance(row.get("urgency"), dict) else {}
    return {
        "intents": grounded(row.get("declared_intents")),
        "emotions": grounded(row.get("direct_emotions"), include_intensity=True),
        "satisfaction": (
            _text(satisfaction.get("label")),
            _text(satisfaction.get("target")),
            _evidence_field(satisfaction, clean_fields),
            _text(satisfaction.get("evidence")),
        ),
        "urgency": (
            _text(urgency.get("level")),
            _evidence_field(urgency, clean_fields),
            _text(urgency.get("evidence")),
        ),
    }


def _agreement_counts(left, right):
    return {
        "matched": len(left & right),
        "left_only": len(left - right),
        "right_only": len(right - left),
        "dice_f1": round(2 * len(left & right) / (len(left) + len(right)), 4)
        if left or right else 1.0,
    }


def _private_conflict_row(left, right, reasons, source_provenance):
    return {
        "schema": ANNOTATION_AGREEMENT_VERSION,
        "private": True,
        "doc_id": left.get("doc_id", ""),
        "content_hash": left.get("content_hash", ""),
        "subset": left.get("subset", ""),
        "manifest_provenance": left.get("manifest_provenance", {}),
        "source_provenance": source_provenance,
        "clean_fields": left.get("clean_fields", {}),
        "metadata": left.get("metadata", {}),
        "conflict_reasons": sorted(reasons),
        "left": {
            key: left.get(key)
            for key in (
                "issues", "declared_intents", "direct_emotions",
                "satisfaction", "urgency", "annotation",
            )
        },
        "right": {
            key: right.get(key)
            for key in (
                "issues", "declared_intents", "direct_emotions",
                "satisfaction", "urgency", "annotation",
            )
        },
        "adjudication": {
            "status": "pending",
            "adjudicator": "",
            "issues": [],
            "declared_intents": [],
            "direct_emotions": [],
            "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
            "urgency": {"level": "normal", "evidence": ""},
            "notes": "",
        },
    }


def compare_gold_annotations(left_path, right_path, left_annotator="", right_annotator=""):
    """Compare two complete private annotation files and return safe metrics plus conflicts."""
    left_validation = validate_gold_annotations(
        left_path, require_complete=True, expected_annotator=left_annotator
    )
    right_validation = validate_gold_annotations(
        right_path, require_complete=True, expected_annotator=right_annotator
    )
    if not left_validation["ready_for_agreement"]:
        raise ValueError("left_annotations_not_ready")
    if not right_validation["ready_for_agreement"]:
        raise ValueError("right_annotations_not_ready")
    left_rows, _ = _load_annotation_rows(left_path)
    right_rows, _ = _load_annotation_rows(right_path)
    left_index = {
        (_text(row.get("doc_id")), _text(row.get("content_hash"))): row
        for row in left_rows
    }
    right_index = {
        (_text(row.get("doc_id")), _text(row.get("content_hash"))): row
        for row in right_rows
    }
    if left_index.keys() != right_index.keys():
        raise ValueError("annotation_identity_sets_differ")
    left_annotators = {
        _text(row.get("annotation", {}).get("annotator")) for row in left_rows
    }
    right_annotators = {
        _text(row.get("annotation", {}).get("annotator")) for row in right_rows
    }
    if left_annotators == right_annotators:
        raise ValueError("annotators_not_distinct")

    source_provenance = {
        "schema": ANNOTATION_AGREEMENT_VERSION,
        "gold_schema": GOLD_SCHEMA_VERSION,
        "left_sha256": _file_sha256(left_path),
        "right_sha256": _file_sha256(right_path),
    }
    shared = sorted(left_index.keys() & right_index.keys())
    issue_counts = Counter()
    mention_counts = Counter()
    pair_counts = Counter()
    discourse_exact = Counter()
    exact_records = 0
    reason_counts = Counter()
    conflicts = []
    for identity in shared:
        left = left_index[identity]
        right = right_index[identity]
        if any(left.get(key) != right.get(key) for key in (
            "subset", "manifest_provenance", "clean_fields", "metadata"
        )):
            raise ValueError("annotation_source_payloads_differ")
        left_frames = _grounded_issue_frames(left)
        right_frames = _grounded_issue_frames(right)
        left_mentions = set().union(*(frame[0] for frame in left_frames)) if left_frames else set()
        right_mentions = set().union(*(frame[0] for frame in right_frames)) if right_frames else set()
        left_pairs, _ = _pairs([set(frame[0]) for frame in left_frames])
        right_pairs, _ = _pairs([set(frame[0]) for frame in right_frames])
        left_discourse = _grounded_discourse(left)
        right_discourse = _grounded_discourse(right)
        for label, values_left, values_right in (
            ("issue", left_frames, right_frames),
            ("mention", left_mentions, right_mentions),
            ("pair", left_pairs, right_pairs),
        ):
            counts = issue_counts if label == "issue" else mention_counts if label == "mention" else pair_counts
            counts["matched"] += len(values_left & values_right)
            counts["left_only"] += len(values_left - values_right)
            counts["right_only"] += len(values_right - values_left)
        for key in ("intents", "emotions", "satisfaction", "urgency"):
            discourse_exact[key] += left_discourse[key] == right_discourse[key]

        reasons = set()
        if left_mentions != right_mentions:
            reasons.add("grounded_mentions")
        if left_pairs != right_pairs:
            reasons.add("issue_attachment")
        if left_frames != right_frames:
            reasons.add("issue_frame")
        if left_discourse != right_discourse:
            reasons.add("discourse")
        if not reasons:
            exact_records += 1
        else:
            reason_counts.update(reasons)
            conflicts.append(_private_conflict_row(
                left, right, reasons, source_provenance
            ))

    def aggregate(counts):
        matched = counts["matched"]
        total = 2 * matched + counts["left_only"] + counts["right_only"]
        return {
            "matched": matched,
            "left_only": counts["left_only"],
            "right_only": counts["right_only"],
            "dice_f1": round(2 * matched / total, 4) if total else 1.0,
        }

    shared_count = len(shared)
    report = {
        "schema": ANNOTATION_AGREEMENT_VERSION,
        "gold_schema": GOLD_SCHEMA_VERSION,
        "private_inputs": True,
        "source_provenance": source_provenance,
        "left_records": len(left_index),
        "right_records": len(right_index),
        "shared_records": shared_count,
        "left_only_identities": len(left_index.keys() - right_index.keys()),
        "right_only_identities": len(right_index.keys() - left_index.keys()),
        "exact_records": exact_records,
        "exact_record_rate": round(exact_records / shared_count, 4) if shared_count else 0.0,
        "conflict_records": len(conflicts),
        "conflict_reason_counts": dict(reason_counts),
        "grounded_mention_agreement": aggregate(mention_counts),
        "issue_attachment_agreement": aggregate(pair_counts),
        "issue_frame_agreement": aggregate(issue_counts),
        "discourse_exact_rates": {
            key: round(value / shared_count, 4) if shared_count else 0.0
            for key, value in discourse_exact.items()
        },
        "left_validation": left_validation,
        "right_validation": right_validation,
    }
    return report, conflicts


def merge_adjudicated_gold(
    left_path,
    right_path,
    conflicts_path,
    adjudicator,
    left_annotator="",
    right_annotator="",
):
    """Merge exact agreements and explicitly resolved conflicts into final gold."""
    adjudicator = _text(adjudicator).strip()
    if not adjudicator:
        raise ValueError("missing_adjudicator")
    agreement, expected_conflicts = compare_gold_annotations(
        left_path,
        right_path,
        left_annotator=left_annotator,
        right_annotator=right_annotator,
    )
    left_rows, _ = _load_annotation_rows(left_path)
    right_rows, _ = _load_annotation_rows(right_path)
    provided_conflicts, parse_errors = _load_annotation_rows(conflicts_path)
    if parse_errors:
        raise ValueError("invalid_conflict_jsonl")

    def index(rows, label):
        values = {}
        for row in rows:
            identity = (_text(row.get("doc_id")), _text(row.get("content_hash")))
            if not all(identity) or identity in values:
                raise ValueError(f"invalid_or_duplicate_{label}_identity")
            values[identity] = row
        return values

    left_index = index(left_rows, "left")
    right_index = index(right_rows, "right")
    expected_index = index(expected_conflicts, "expected_conflict")
    provided_index = index(provided_conflicts, "provided_conflict")
    if expected_index.keys() != provided_index.keys():
        raise ValueError("conflict_identity_sets_differ")

    left_name = next(iter({
        _text(row.get("annotation", {}).get("annotator")) for row in left_rows
    }))
    right_name = next(iter({
        _text(row.get("annotation", {}).get("annotator")) for row in right_rows
    }))
    if adjudicator in {left_name, right_name}:
        raise ValueError("adjudicator_not_independent")
    final_rows = []
    for identity, left in left_index.items():
        right = right_index[identity]
        if identity not in expected_index:
            semantic = {
                key: left.get(key)
                for key in (
                    "issues", "declared_intents", "direct_emotions",
                    "satisfaction", "urgency",
                )
            }
            notes = "exact_double_annotation_agreement"
        else:
            expected = expected_index[identity]
            provided = provided_index[identity]
            immutable_keys = (
                "schema", "private", "doc_id", "content_hash", "subset",
                "manifest_provenance", "source_provenance", "clean_fields",
                "metadata", "conflict_reasons", "left", "right",
            )
            if any(provided.get(key) != expected.get(key) for key in immutable_keys):
                raise ValueError("conflict_source_payload_changed")
            decision = provided.get("adjudication")
            if not isinstance(decision, dict) or decision.get("status") != "resolved":
                raise ValueError("conflict_not_resolved")
            if _text(decision.get("adjudicator")) != adjudicator:
                raise ValueError("conflict_adjudicator_mismatch")
            if not _text(decision.get("notes")).strip():
                raise ValueError("conflict_resolution_missing_notes")
            semantic = {
                key: decision.get(key)
                for key in (
                    "issues", "declared_intents", "direct_emotions",
                    "satisfaction", "urgency",
                )
            }
            notes = _text(decision.get("notes"))

        final = {
            "schema": GOLD_SCHEMA_VERSION,
            "private": True,
            "subset": left.get("subset", ""),
            "doc_id": identity[0],
            "content_hash": identity[1],
            "manifest_provenance": left.get("manifest_provenance", {}),
            "clean_fields": left.get("clean_fields", {}),
            "metadata": left.get("metadata", {}),
            **semantic,
            "annotation": {
                "annotator": adjudicator,
                "status": "adjudicated",
                "notes": notes,
            },
            "adjudication_provenance": {
                "schema": ADJUDICATION_MERGE_VERSION,
                "left_annotator": left_name,
                "right_annotator": right_name,
                "left_sha256": agreement["source_provenance"]["left_sha256"],
                "right_sha256": agreement["source_provenance"]["right_sha256"],
                "conflicts_sha256": _file_sha256(conflicts_path),
                "resolution": "explicit_conflict" if identity in expected_index else "exact_agreement",
            },
        }
        errors, _warnings = _validate_gold_row(
            final, require_complete=True, expected_annotator=adjudicator
        )
        if errors:
            raise ValueError("invalid_adjudicated_record:" + sorted(errors)[0])
        final_rows.append(final)

    report = {
        "schema": ADJUDICATION_MERGE_VERSION,
        "gold_schema": GOLD_SCHEMA_VERSION,
        "private_inputs": True,
        "records": len(final_rows),
        "exact_agreements": len(final_rows) - len(expected_index),
        "resolved_conflicts": len(expected_index),
        "agreement_source_sha256": {
            "left": agreement["source_provenance"]["left_sha256"],
            "right": agreement["source_provenance"]["right_sha256"],
        },
        "conflicts_source_sha256": _file_sha256(conflicts_path),
    }
    return final_rows, report


def _read_semantic_index(path):
    if not path:
        return {}
    rows = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict) and value.get("doc_id"):
                rows[str(value["doc_id"])] = value
    return rows


def _largest_remainder_quotas(counts, total):
    population = sum(counts.values())
    if population <= 0 or total <= 0:
        return {key: 0 for key in counts}
    total = min(total, population)
    raw = {key: total * count / population for key, count in counts.items()}
    quotas = {key: min(counts[key], int(math.floor(value))) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (raw[key] - math.floor(raw[key]), counts[key], str(key)),
        reverse=True,
    )
    for key in order:
        if not remaining:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    return quotas


def _stable_rank(row, seed):
    return _sha(f"{seed}\u241f{row['doc_id']}\u241f{row['content_hash']}")


def build_eval_manifest(
    input_path,
    production_size=200,
    challenge_size=64,
    seed="sag-eval-v1",
    semantic_path=None,
):
    """Build deterministic production-like and challenge manifests without text."""
    semantic_index = _read_semantic_index(semantic_path)
    rows = [
        _descriptor(order, semantic_index)
        for order in iter_normalized_orders(input_path)
        if "_profile_error" not in order
    ]
    type3_counts = Counter(row["type3"] for row in rows if row["type3"] != "<EMPTY>")
    for row in rows:
        if row["type3"] != "<EMPTY>" and type3_counts[row["type3"]] <= 10:
            row["challenge_flags"] = sorted(set(row["challenge_flags"] + ["rare_type3"]))
    rows.sort(key=lambda row: _stable_rank(row, seed))

    challenge_candidates = [row for row in rows if row["challenge_flags"]]
    by_flag = defaultdict(list)
    for row in challenge_candidates:
        for flag in row["challenge_flags"]:
            by_flag[flag].append(row)
    for flag in by_flag:
        by_flag[flag].sort(key=lambda row: _stable_rank(row, seed + ":challenge:" + flag))

    selected_challenge = []
    selected_ids = set()
    positions = Counter()
    flags = sorted(by_flag)
    while len(selected_challenge) < min(challenge_size, len(challenge_candidates)):
        progress = False
        for flag in flags:
            candidates = by_flag[flag]
            while positions[flag] < len(candidates):
                candidate = candidates[positions[flag]]
                positions[flag] += 1
                identity = (candidate["doc_id"], candidate["content_hash"])
                if identity in selected_ids:
                    continue
                selected_ids.add(identity)
                selected_challenge.append(candidate)
                progress = True
                break
            if len(selected_challenge) >= challenge_size:
                break
        if not progress:
            break

    production_pool = [
        row for row in rows
        if (row["doc_id"], row["content_hash"]) not in selected_ids
    ]
    strata = defaultdict(list)
    for row in production_pool:
        strata[(row["service_object_type"], row["length_bucket"])].append(row)
    quotas = _largest_remainder_quotas(
        {stratum: len(values) for stratum, values in strata.items()},
        min(production_size, len(production_pool)),
    )
    selected_production = []
    for stratum in sorted(strata):
        values = sorted(
            strata[stratum], key=lambda row: _stable_rank(row, seed + ":production")
        )
        selected_production.extend(values[:quotas[stratum]])
    selected_production.sort(key=lambda row: _stable_rank(row, seed + ":output"))

    manifest = []
    for subset, selected in (
        ("production", selected_production),
        ("challenge", selected_challenge),
    ):
        for row in selected:
            manifest.append({
                "schema": EVAL_MANIFEST_VERSION,
                "subset": subset,
                "doc_id": row["doc_id"],
                "content_hash": row["content_hash"],
                "service_object_type": row["service_object_type"],
                "type1": row["type1"],
                "type2": row["type2"],
                "type3": row["type3"],
                "length_bucket": row["length_bucket"],
                "challenge_reasons": row["challenge_flags"] if subset == "challenge" else [],
            })

    report = {
        "schema": EVAL_MANIFEST_VERSION,
        "seed": seed,
        "source_sha256": _file_sha256(input_path),
        "semantic_source_sha256": _file_sha256(semantic_path) if semantic_path else "",
        "manifest_content_sha256": "sha256:" + _sha("\n".join(
            _stable_json(row) for row in manifest
        )),
        "source_records": len(rows),
        "production_requested": production_size,
        "production_selected": len(selected_production),
        "challenge_requested": challenge_size,
        "challenge_selected": len(selected_challenge),
        "subsets": {},
    }
    for subset in ("production", "challenge"):
        subset_rows = [row for row in manifest if row["subset"] == subset]
        report["subsets"][subset] = {
            "records": len(subset_rows),
            "service_object_type_counts": dict(Counter(
                row["service_object_type"] for row in subset_rows
            )),
            "length_bucket_counts": dict(Counter(
                row["length_bucket"] for row in subset_rows
            )),
            "type1_cardinality": len({row["type1"] for row in subset_rows}),
            "challenge_reason_counts": dict(Counter(
                reason for row in subset_rows for reason in row["challenge_reasons"]
            )),
        }
    return manifest, report


def _normalized_mention(value):
    if isinstance(value, dict):
        value = value.get("surface") or value.get("text") or value.get("canonical") or ""
    value = _text(value)
    return re.sub(r"[\s，,。；;：:、（）()\[\]【】]", "", value).casefold()


def _issue_members(issue, include_non_frontier=False):
    if not isinstance(issue, dict):
        return set()
    members = set()
    mode = _text(issue.get("mode"))
    object_groups = (
        issue.get("objects"), issue.get("object_mentions"), issue.get("problem_objects")
    )
    predicate_groups = (
        issue.get("predicates"), issue.get("predicate_mentions"), issue.get("problem_behaviors")
    )
    for group in object_groups:
        for item in group if isinstance(group, list) else []:
            mention = _normalized_mention(item)
            if mention:
                members.add(("problem_object", mention))
    predicate_role = "problem_behavior" if mode in {"", "problem"} else "issue_predicate"
    if predicate_role == "problem_behavior" or include_non_frontier:
        for group in predicate_groups:
            for item in group if isinstance(group, list) else []:
                mention = _normalized_mention(item)
                if mention:
                    members.add((predicate_role, mention))
    if include_non_frontier:
        for group in (issue.get("actions"), issue.get("action_mentions")):
            for item in group if isinstance(group, list) else []:
                mention = _normalized_mention(item)
                if mention:
                    members.add(("request_action", mention))
    for item in issue.get("locations", []) if isinstance(issue.get("locations"), list) else []:
        if not isinstance(item, dict):
            continue
        location_type = _text(item.get("type"))
        if location_type not in {"road", "intersection", "poi"}:
            continue
        mention = _normalized_mention(item)
        if mention:
            members.add((location_type, mention))
    return members


def _issue_descriptors(record, oracle_flat=False):
    issues = record.get("issues") if isinstance(record, dict) else None
    if isinstance(issues, list) and not oracle_flat:
        return [
            (
                frozenset(_issue_members(issue, include_non_frontier=True)),
                _text(issue.get("mode")) if isinstance(issue, dict) else "",
                _text(issue.get("time_scope")) if isinstance(issue, dict) else "",
            )
            for issue in issues
            if _issue_members(issue, include_non_frontier=True)
        ]
    flat = _record_issues(record, oracle_flat=oracle_flat)
    return [(frozenset(members), "", "") for members in flat]


def _record_issues(record, oracle_flat=False):
    issues = record.get("issues") if isinstance(record, dict) else None
    if isinstance(issues, list) and not oracle_flat:
        return [_issue_members(issue) for issue in issues if _issue_members(issue)]
    entities = record.get("entities") if isinstance(record, dict) else {}
    members = set()
    mapping = {
        "problem_objects": "problem_object",
        "problem_behaviors": "problem_behavior",
        "roads": "road",
        "intersections": "intersection",
        "pois": "poi",
    }
    if isinstance(entities, dict):
        for group, role in mapping.items():
            for item in entities.get(group, []) if isinstance(entities.get(group), list) else []:
                mention = _normalized_mention(item)
                if mention:
                    members.add((role, mention))
    if not members and isinstance(issues, list):
        for issue in issues:
            members.update(_issue_members(issue))
    return [members] if members else []


def _pairs(issues):
    pairs = set()
    hyperedges = set()
    for members in issues:
        frozen = frozenset(members)
        if frozen:
            hyperedges.add(frozen)
        ordered = sorted(members)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1:]:
                pairs.add((left, right))
    return pairs, hyperedges


def _attachment_pairs(pairs, attachment):
    if attachment == "object_behavior":
        roles = {"problem_object", "problem_behavior"}
        return {
            pair for pair in pairs
            if {pair[0][0], pair[1][0]} == roles
        }
    if attachment == "location":
        locations = {"road", "intersection", "poi"}
        semantic_roles = {"problem_object", "problem_behavior"}
        return {
            pair for pair in pairs
            if (
                pair[0][0] in locations and pair[1][0] in semantic_roles
            ) or (
                pair[1][0] in locations and pair[0][0] in semantic_roles
            )
        }
    return set()


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _discourse_sets(record, gold=False):
    record = record if isinstance(record, dict) else {}
    discourse = record.get("discourse") if isinstance(record.get("discourse"), dict) else {}
    intents = record.get("declared_intents") if gold else discourse.get("intents")
    emotions = record.get("direct_emotions") if gold else discourse.get("emotions")
    satisfaction = record.get("satisfaction") if gold else discourse.get("satisfaction")
    urgency = record.get("urgency") if gold else discourse.get("urgency")
    intents = intents if isinstance(intents, list) else []
    emotions = emotions if isinstance(emotions, list) else []
    satisfaction = satisfaction if isinstance(satisfaction, dict) else {}
    urgency = urgency if isinstance(urgency, dict) else {}
    return {
        "intent_labels": {
            _text(item.get("label"))
            for item in intents if isinstance(item, dict) and _text(item.get("label"))
        },
        "intent_grounded": {
            (_text(item.get("label")), _text(item.get("evidence")))
            for item in intents if isinstance(item, dict) and _text(item.get("label"))
        },
        "emotion_labels": {
            _text(item.get("label"))
            for item in emotions if isinstance(item, dict) and _text(item.get("label"))
        },
        "emotion_grounded": {
            (
                _text(item.get("label")),
                int(item.get("intensity")) if isinstance(item.get("intensity"), int) else 0,
                _text(item.get("evidence")),
            )
            for item in emotions if isinstance(item, dict) and _text(item.get("label"))
        },
        "satisfaction": (
            _text(satisfaction.get("label")) or "unknown",
            _text(satisfaction.get("target")),
            _text(satisfaction.get("evidence")),
        ),
        "urgency": (
            _text(urgency.get("level")) or "normal",
            _text(urgency.get("evidence")),
        ),
    }


def _read_jsonl_index(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("doc_id"):
                raise ValueError(f"line_{line_number}:invalid_record")
            rows[str(value["doc_id"])] = value
    return rows


def _semantic_frontier_mentions(semantic):
    entities = semantic.get("entities") if isinstance(semantic, dict) else {}
    mapping = {
        "problem_objects": "problem_object",
        "problem_behaviors": "problem_behavior",
        "roads": "road",
        "intersections": "intersection",
        "pois": "poi",
    }
    mentions = set()
    if not isinstance(entities, dict):
        return mentions
    for group, role in mapping.items():
        for item in entities.get(group, []) if isinstance(entities.get(group), list) else []:
            mention = _normalized_mention(item)
            if mention:
                mentions.add((role, mention))
    return mentions


def _audit_set_counts(raw, final, gold):
    removed = raw - final
    added = final - raw
    return Counter({
        "raw_tp": len(raw & gold),
        "raw_fp": len(raw - gold),
        "raw_fn": len(gold - raw),
        "final_tp": len(final & gold),
        "final_fp": len(final - gold),
        "final_fn": len(gold - final),
        "correctly_kept": len(raw & final & gold),
        "incorrectly_kept": len((raw & final) - gold),
        "correctly_removed": len(removed - gold),
        "wrongly_removed": len(removed & gold),
        "correct_additions": len(added & gold),
        "incorrect_additions": len(added - gold),
    })


def audit_candidate_ledger_against_gold(
    input_path,
    gold_path,
    candidate_ledger_path,
    decision_ledger_path=None,
):
    """Audit pre/post-validator frontier mentions against completed issue gold."""
    from ragflow_style_pipeline.sag_semantic_llm import _validate_with_sanitation

    gold_validation = validate_gold_annotations(gold_path, require_complete=True)
    if not gold_validation["ready_for_evaluation"]:
        raise ValueError("gold_annotations_not_ready")
    gold_index = _read_jsonl_index(gold_path)
    target_ids = set(gold_index)
    orders = {}
    for order in iter_normalized_orders(input_path):
        if "_profile_error" not in order and order["doc_id"] in target_ids:
            orders[(order["doc_id"], order["content_hash"])] = order
    expected = {
        (doc_id, _text(row.get("content_hash"))) for doc_id, row in gold_index.items()
    }
    if expected - orders.keys():
        raise ValueError("gold_order_identity_mismatch")

    attempts = []
    candidate_entries_total = 0
    candidate_model_counts = Counter()
    candidate_prompt_version_counts = Counter()
    candidate_decoder_version_counts = Counter()
    candidate_sequences = set()
    with Path(candidate_ledger_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if (
                not isinstance(value, dict)
                or value.get("schema") != CANDIDATE_LEDGER_VERSION
                or value.get("private") is not True
            ):
                raise ValueError(f"line_{line_number}:invalid_candidate_ledger")
            candidate_entries_total += 1
            identity = (_text(value.get("doc_id")), _text(value.get("content_hash")))
            phase = _text(value.get("phase"))
            sequence = value.get("ledger_sequence")
            if (
                phase not in {"primary", "repair"}
                or not _text(value.get("run_attempt_id"))
                or not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1
                or sequence in candidate_sequences
                or not isinstance(value.get("candidate"), dict)
            ):
                raise ValueError(f"line_{line_number}:invalid_candidate_attempt")
            candidate_sequences.add(sequence)
            if identity not in expected:
                continue
            candidate_model_counts[_text(value.get("model")) or "<EMPTY>"] += 1
            candidate_prompt_version_counts[
                _text(value.get("prompt_version")) or "<EMPTY>"
            ] += 1
            candidate_decoder_version_counts[
                _text(value.get("decoder_contract_version")) or "<EMPTY>"
            ] += 1
            attempts.append((identity, phase, sequence, value))

    decisions = {}
    decision_entries_total = 0
    decision_entries_matched = 0
    decision_validator_version_counts = Counter()
    decision_sequences = set()
    if decision_ledger_path:
        with Path(decision_ledger_path).open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if (
                    not isinstance(value, dict)
                    or value.get("schema") != DECISION_LEDGER_VERSION
                    or value.get("private") is not True
                ):
                    raise ValueError(f"line_{line_number}:invalid_decision_ledger")
                decision_entries_total += 1
                identity = (_text(value.get("doc_id")), _text(value.get("content_hash")))
                phase = _text(value.get("phase"))
                sequence = value.get("ledger_sequence")
                if (
                    phase not in {"primary", "repair"}
                    or not _text(value.get("run_attempt_id"))
                    or not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1
                    or sequence in decision_sequences
                ):
                    raise ValueError(f"line_{line_number}:invalid_decision_attempt")
                decision_sequences.add(sequence)
                if identity not in expected:
                    continue
                decision_entries_matched += 1
                decision_validator_version_counts[
                    _text(value.get("validator_version")) or "<EMPTY>"
                ] += 1
                key = (identity, phase, _text(value.get("run_attempt_id")))
                current = decisions.get(key)
                if current is None or sequence > current[0]:
                    decisions[key] = (sequence, value)

    validation_cache = {}

    def current_result(identity, candidate_entry):
        key = (
            identity,
            candidate_entry.get("phase"),
            candidate_entry.get("ledger_sequence"),
        )
        if key not in validation_cache:
            validation_cache[key] = _validate_with_sanitation(
                orders[identity],
                candidate_entry["candidate"],
                candidate_entry.get("parse_warnings")
                if isinstance(candidate_entry.get("parse_warnings"), list) else [],
            )
        return validation_cache[key]

    latest_any = {}
    for identity, _phase, sequence, value in attempts:
        if identity not in latest_any or sequence > latest_any[identity][0]:
            latest_any[identity] = (sequence, value)

    latest_run_attempt = {
        identity: _text(value.get("run_attempt_id"))
        for identity, (_sequence, value) in latest_any.items()
    }
    latest_by_phase = {}
    latest_final = {}
    for identity, phase, sequence, value in attempts:
        if _text(value.get("run_attempt_id")) != latest_run_attempt.get(identity):
            continue
        phase_key = (identity, phase)
        if phase_key not in latest_by_phase or sequence > latest_by_phase[phase_key][0]:
            latest_by_phase[phase_key] = (sequence, value)
        _final, validation, _trace = current_result(identity, value)
        terminal = phase == "repair" or validation.get("status") != "repair_required"
        if terminal and (identity not in latest_final or sequence > latest_final[identity][0]):
            latest_final[identity] = (sequence, value)
    incomplete_latest_attempts = len(latest_any) - len(latest_final)

    scope_counts = {name: Counter() for name in ("primary", "repair", "selected")}
    role_counts = {
        scope: defaultdict(Counter) for scope in ("primary", "repair", "selected")
    }
    current_action_counts = Counter()
    original_action_counts = Counter()
    original_status_counts = Counter()
    current_status_counts = Counter()
    validator_status_transitions = Counter()
    missing_decisions = 0
    traces = []

    def audit_one(scope, identity, candidate_entry, selected_for_final):
        nonlocal missing_decisions
        candidate = candidate_entry["candidate"]
        final, validation, trace = current_result(identity, candidate_entry)
        gold_issues = _record_issues(gold_index[identity[0]])
        gold_mentions = set().union(*gold_issues) if gold_issues else set()
        raw_mentions = _semantic_frontier_mentions(candidate)
        final_mentions = _semantic_frontier_mentions(final)
        counts = _audit_set_counts(raw_mentions, final_mentions, gold_mentions)
        scope_counts[scope].update(counts)
        for role in sorted({item[0] for item in raw_mentions | final_mentions | gold_mentions}):
            role_counts[scope][role].update(_audit_set_counts(
                {item for item in raw_mentions if item[0] == role},
                {item for item in final_mentions if item[0] == role},
                {item for item in gold_mentions if item[0] == role},
            ))
        if scope == "selected":
            current_action_counts.update(trace.get("sanitation_warnings", []))
            current_status_counts[validation.get("status", "<EMPTY>")] += 1
            decision_key = (
                identity,
                candidate_entry.get("phase"),
                _text(candidate_entry.get("run_attempt_id")),
            )
            original = decisions.get(decision_key)
            if decision_ledger_path and original is None:
                missing_decisions += 1
            elif original:
                decision = original[1]
                original_action_counts.update(decision.get("actions", []))
                after = decision.get("validation_after")
                original_status = (
                    _text(after.get("status")) if isinstance(after, dict) else "<EMPTY>"
                ) or "<EMPTY>"
                original_status_counts[original_status] += 1
                validator_status_transitions[
                    f"{original_status}->{validation.get('status', '<EMPTY>')}"
                ] += 1
        traces.append({
            "schema": LEDGER_GOLD_AUDIT_VERSION,
            "private": True,
            "doc_id": identity[0],
            "content_hash": identity[1],
            "phase": candidate_entry.get("phase"),
            "scope": scope,
            "selected_for_final": selected_for_final,
            "run_attempt_id": candidate_entry.get("run_attempt_id", ""),
            "ledger_sequence": candidate_entry.get("ledger_sequence"),
            "raw_mentions": [list(item) for item in sorted(raw_mentions)],
            "final_mentions": [list(item) for item in sorted(final_mentions)],
            "gold_mentions": [list(item) for item in sorted(gold_mentions)],
            "counts": dict(counts),
            "current_validator_status": validation.get("status", ""),
            "current_validator_actions": trace.get("sanitation_warnings", []),
        })

    for (identity, phase), (_sequence, entry) in sorted(latest_by_phase.items()):
        audit_one(phase, identity, entry, latest_final.get(identity, (None, None))[1] is entry)
    for identity, (_sequence, entry) in sorted(latest_final.items()):
        audit_one("selected", identity, entry, True)

    def summarize(counts):
        raw = _prf(counts["raw_tp"], counts["raw_fp"], counts["raw_fn"])
        final = _prf(counts["final_tp"], counts["final_fp"], counts["final_fn"])
        removed = counts["correctly_removed"] + counts["wrongly_removed"]
        added = counts["correct_additions"] + counts["incorrect_additions"]
        return {
            **dict(counts),
            "raw_prf": raw,
            "final_prf": final,
            "deletion_precision": round(counts["correctly_removed"] / removed, 4)
            if removed else None,
            "addition_precision": round(counts["correct_additions"] / added, 4)
            if added else None,
        }

    report = {
        "schema": LEDGER_GOLD_AUDIT_VERSION,
        "gold_schema": GOLD_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "private_inputs": True,
        "gold_records": len(expected),
        "input_source_sha256": _file_sha256(input_path),
        "gold_source_sha256": _file_sha256(gold_path),
        "candidate_source_sha256": _file_sha256(candidate_ledger_path),
        "decision_source_sha256": _file_sha256(decision_ledger_path)
        if decision_ledger_path else "",
        "candidate_entries_total": candidate_entries_total,
        "candidate_attempts_matched": len(attempts),
        "candidate_attempts_unmatched": candidate_entries_total - len(attempts),
        "records_with_any_candidates": len(latest_any),
        "records_with_terminal_candidates": len(latest_final),
        "gold_records_without_terminal_candidates": len(expected - latest_final.keys()),
        "incomplete_latest_attempts": incomplete_latest_attempts,
        "selection_policy": "latest_current_validator_terminal_attempt",
        "candidate_model_counts": dict(candidate_model_counts),
        "candidate_prompt_version_counts": dict(candidate_prompt_version_counts),
        "candidate_decoder_version_counts": dict(candidate_decoder_version_counts),
        "phase_records": dict(Counter(phase for _identity, phase in latest_by_phase)),
        "scopes": {
            scope: {
                "micro": summarize(scope_counts[scope]),
                "by_role": {
                    role: summarize(counts)
                    for role, counts in sorted(role_counts[scope].items())
                },
            }
            for scope in ("primary", "repair", "selected")
        },
        "current_validator_action_counts": dict(current_action_counts),
        "current_validator_status_counts": dict(current_status_counts),
        "decision_entries_total": decision_entries_total,
        "decision_entries_matched": decision_entries_matched,
        "decision_entries_unmatched": decision_entries_total - decision_entries_matched,
        "decision_validator_version_counts": dict(decision_validator_version_counts),
        "selected_attempts_missing_decisions": missing_decisions,
        "original_validator_action_counts": dict(original_action_counts),
        "validator_action_count_delta": {
            action: current_action_counts[action] - original_action_counts[action]
            for action in sorted(set(current_action_counts) | set(original_action_counts))
        },
        "original_validator_status_counts": dict(original_status_counts),
        "validator_status_transitions": dict(validator_status_transitions),
    }
    return report, traces


def replay_candidate_ledger(input_path, candidate_ledger_path):
    """Re-run the current deterministic validator without invoking a model."""
    from ragflow_style_pipeline.sag_semantic_llm import _semantic_counts, _validate_with_sanitation

    orders = {
        (order["doc_id"], order["content_hash"]): order
        for order in iter_normalized_orders(input_path)
        if "_profile_error" not in order
    }
    candidates = []
    with Path(candidate_ledger_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("private"):
                raise ValueError(f"line_{line_number}:invalid_private_candidate")
            identity = (
                _text(value.get("doc_id")),
                _text(value.get("content_hash")),
            )
            phase = _text(value.get("phase"))
            if phase not in {"primary", "repair"}:
                raise ValueError(f"line_{line_number}:invalid_candidate_phase")
            sequence = value.get("ledger_sequence", line_number)
            if not isinstance(sequence, int) or sequence < 1:
                raise ValueError(f"line_{line_number}:invalid_ledger_sequence")
            candidates.append((identity, sequence, value))

    selected = {}
    for identity, sequence, value in candidates:
        if identity not in orders:
            raise ValueError("candidate_order_identity_mismatch")
        current = selected.get(identity)
        if current is None or sequence > current[0]:
            selected[identity] = (sequence, value)

    rows = []
    status_counts = Counter()
    action_counts = Counter()
    for identity in sorted(selected):
        _sequence, candidate = selected[identity]
        order = orders[identity]
        semantic, validation, trace = _validate_with_sanitation(
            order,
            candidate.get("candidate") if isinstance(candidate.get("candidate"), dict) else {},
            candidate.get("parse_warnings") if isinstance(candidate.get("parse_warnings"), list) else [],
        )
        status_counts[validation["status"]] += 1
        action_counts.update(trace.get("sanitation_warnings", []))
        rows.append({
            "schema": VALIDATOR_REPLAY_VERSION,
            "private": True,
            "validator_version": VALIDATOR_VERSION,
            "doc_id": identity[0],
            "content_hash": identity[1],
            "selected_phase": candidate.get("phase"),
            "model": candidate.get("model", ""),
            "prompt_version": candidate.get("prompt_version", ""),
            "decoder_contract_version": candidate.get("decoder_contract_version", ""),
            "semantic": semantic,
            "validation": validation,
            "trace": trace,
        })
    report = {
        "schema": VALIDATOR_REPLAY_VERSION,
        "private_input": True,
        "candidates_read": len(candidates),
        "records_replayed": len(rows),
        "status_counts": dict(status_counts),
        "action_counts": dict(action_counts),
        "semantic_count_totals": {
            group: sum(
                _semantic_counts(row["semantic"])["entities"][group]
                for row in rows
            )
            for group in ("problem_objects", "problem_behaviors", "roads", "intersections", "pois")
        },
    }
    return rows, report


def project_gold_issues(gold_path, flat=False):
    """Project private issue gold into deterministic SAG issue/member rows."""
    gold = _read_jsonl_index(gold_path)
    order_events = []
    issue_events = []
    member_links = []
    seen_links = set()
    for doc_id, record in gold.items():
        order_events.append({
            "doc_id": doc_id,
            "content_hash": record.get("content_hash", ""),
            "subset": record.get("subset", ""),
            "gold_schema": GOLD_SCHEMA_VERSION,
        })
        raw_issues = record.get("issues") if isinstance(record.get("issues"), list) else []
        if flat:
            issue_groups = [("flat", raw_issues)] if raw_issues else []
        else:
            issue_groups = [(str(index + 1), [issue]) for index, issue in enumerate(raw_issues)]
        for suffix, issue_group in issue_groups:
            issue_id = f"{doc_id}::issue::{suffix}"
            modes = sorted({
                _text(issue.get("mode"))
                for issue in issue_group if isinstance(issue, dict) and _text(issue.get("mode"))
            })
            time_scopes = sorted({
                _text(issue.get("time_scope"))
                for issue in issue_group
                if isinstance(issue, dict) and _text(issue.get("time_scope"))
            })
            issue_events.append({
                "issue_id": issue_id,
                "doc_id": doc_id,
                "mode": modes[0] if len(modes) == 1 else "mixed",
                "time_scope": time_scopes[0] if len(time_scopes) == 1 else "mixed",
                "projection": "flat" if flat else "issue_aware",
                "gold_schema": GOLD_SCHEMA_VERSION,
            })
            for issue in issue_group:
                if not isinstance(issue, dict):
                    continue
                issue_mode = _text(issue.get("mode"))
                predicate_role = "behavior" if issue_mode in {"", "problem"} else "issue_predicate"
                predicate_type = "problem_behavior" if issue_mode in {"", "problem"} else "issue_predicate"
                role_groups = (
                    ("object", "problem_object", issue.get("objects") or issue.get("object_mentions")),
                    (predicate_role, predicate_type, issue.get("predicates") or issue.get("predicate_mentions")),
                    ("request_action", "request_action", issue.get("actions") or issue.get("action_mentions")),
                )
                for role, entity_type, items in role_groups:
                    for item in items if isinstance(items, list) else []:
                        item = item if isinstance(item, dict) else {"surface": _text(item)}
                        surface = _text(item.get("surface") or item.get("text"))
                        if not surface:
                            continue
                        normalized = _text(item.get("canonical")) or surface
                        source_field = _text(item.get("source_field") or item.get("field"))
                        evidence = _text(item.get("evidence"))
                        link_key = (
                            issue_id, role, entity_type, surface,
                            normalized, source_field, evidence,
                        )
                        if link_key in seen_links:
                            continue
                        seen_links.add(link_key)
                        member_links.append({
                            "issue_id": issue_id,
                            "doc_id": doc_id,
                            "role": role,
                            "entity_type": entity_type,
                            "surface": surface,
                            "normalized_value": normalized,
                            "source_field": source_field,
                            "evidence": evidence,
                            "gold_schema": GOLD_SCHEMA_VERSION,
                        })
                for item in issue.get("locations", []) if isinstance(issue.get("locations"), list) else []:
                    if not isinstance(item, dict):
                        continue
                    entity_type = _text(item.get("type"))
                    surface = _text(item.get("surface") or item.get("text"))
                    if entity_type not in {"road", "intersection", "poi"} or not surface:
                        continue
                    normalized = _text(item.get("canonical")) or surface
                    source_field = _text(item.get("source_field") or item.get("field"))
                    evidence = _text(item.get("evidence"))
                    link_key = (
                        issue_id, "location", entity_type, surface,
                        normalized, source_field, evidence,
                    )
                    if link_key in seen_links:
                        continue
                    seen_links.add(link_key)
                    member_links.append({
                        "issue_id": issue_id,
                        "doc_id": doc_id,
                        "role": "location",
                        "entity_type": entity_type,
                        "surface": surface,
                        "normalized_value": normalized,
                        "source_field": source_field,
                        "evidence": evidence,
                        "gold_schema": GOLD_SCHEMA_VERSION,
                    })
    return order_events, issue_events, member_links


def evaluate_semantic_gold(gold_path, prediction_path=None, oracle_flat=False):
    """Evaluate mentions and SAG issue co-membership without returning work-order text."""
    gold = _read_jsonl_index(gold_path)
    predictions = _read_jsonl_index(prediction_path) if prediction_path else gold
    mention_counts = Counter()
    role_counts = defaultdict(Counter)
    pair_counts = Counter()
    attachment_counts = {
        "object_behavior": Counter(),
        "location": Counter(),
    }
    exact_hyperedges = 0
    exact_issue_frames = 0
    gold_hyperedges_total = 0
    predicted_hyperedges_total = 0
    gold_issue_frames_total = 0
    predicted_issue_frames_total = 0
    discourse_counts = {
        name: Counter()
        for name in (
            "intent_labels", "intent_grounded",
            "emotion_labels", "emotion_grounded",
        )
    }
    satisfaction_exact = 0
    urgency_exact = 0
    evaluated = 0
    missing_predictions = 0

    for doc_id, gold_record in gold.items():
        prediction = predictions.get(doc_id)
        if prediction is None:
            prediction = {}
            missing_predictions += 1
        gold_issues = _record_issues(gold_record)
        predicted_issues = _record_issues(
            gold_record if oracle_flat else prediction,
            oracle_flat=oracle_flat,
        )
        gold_mentions = set().union(*gold_issues) if gold_issues else set()
        predicted_mentions = set().union(*predicted_issues) if predicted_issues else set()
        mention_counts["tp"] += len(gold_mentions & predicted_mentions)
        mention_counts["fp"] += len(predicted_mentions - gold_mentions)
        mention_counts["fn"] += len(gold_mentions - predicted_mentions)
        for role in {value[0] for value in gold_mentions | predicted_mentions}:
            gold_role = {value for value in gold_mentions if value[0] == role}
            predicted_role = {value for value in predicted_mentions if value[0] == role}
            role_counts[role]["tp"] += len(gold_role & predicted_role)
            role_counts[role]["fp"] += len(predicted_role - gold_role)
            role_counts[role]["fn"] += len(gold_role - predicted_role)

        gold_pairs, gold_edges = _pairs(gold_issues)
        predicted_pairs, predicted_edges = _pairs(predicted_issues)
        gold_frames = set(_issue_descriptors(gold_record))
        predicted_frames = set(_issue_descriptors(
            gold_record if oracle_flat else prediction,
            oracle_flat=oracle_flat,
        ))
        pair_counts["tp"] += len(gold_pairs & predicted_pairs)
        pair_counts["fp"] += len(predicted_pairs - gold_pairs)
        pair_counts["fn"] += len(gold_pairs - predicted_pairs)
        for attachment, counts in attachment_counts.items():
            gold_attachment = _attachment_pairs(gold_pairs, attachment)
            predicted_attachment = _attachment_pairs(predicted_pairs, attachment)
            counts["tp"] += len(gold_attachment & predicted_attachment)
            counts["fp"] += len(predicted_attachment - gold_attachment)
            counts["fn"] += len(gold_attachment - predicted_attachment)
        exact_hyperedges += len(gold_edges & predicted_edges)
        exact_issue_frames += len(gold_frames & predicted_frames)
        gold_hyperedges_total += len(gold_edges)
        predicted_hyperedges_total += len(predicted_edges)
        gold_issue_frames_total += len(gold_frames)
        predicted_issue_frames_total += len(predicted_frames)

        gold_discourse = _discourse_sets(gold_record, gold=True)
        predicted_discourse = (
            _discourse_sets(gold_record, gold=True)
            if oracle_flat else _discourse_sets(prediction, gold=False)
        )
        for name, counts in discourse_counts.items():
            gold_values = gold_discourse[name]
            predicted_values = predicted_discourse[name]
            counts["tp"] += len(gold_values & predicted_values)
            counts["fp"] += len(predicted_values - gold_values)
            counts["fn"] += len(gold_values - predicted_values)
        satisfaction_exact += gold_discourse["satisfaction"] == predicted_discourse["satisfaction"]
        urgency_exact += gold_discourse["urgency"] == predicted_discourse["urgency"]
        evaluated += 1

    mention_prf = _prf(mention_counts["tp"], mention_counts["fp"], mention_counts["fn"])
    pair_prf = _prf(pair_counts["tp"], pair_counts["fp"], pair_counts["fn"])
    return {
        "schema": EVALUATION_VERSION,
        "gold_schema": GOLD_SCHEMA_VERSION,
        "mode": "oracle_flat" if oracle_flat else "prediction",
        "gold_records": len(gold),
        "prediction_records": len(predictions),
        "records_evaluated": evaluated,
        "missing_predictions": missing_predictions,
        "mention_micro": mention_prf,
        "mention_by_role": {
            role: _prf(counts["tp"], counts["fp"], counts["fn"])
            for role, counts in sorted(role_counts.items())
        },
        "issue_co_membership": {
            **pair_prf,
            "false_co_membership_rate": round(
                pair_counts["fp"] / (pair_counts["tp"] + pair_counts["fp"]), 4
            ) if pair_counts["tp"] + pair_counts["fp"] else 0.0,
        },
        "object_behavior_attachment": _prf(
            attachment_counts["object_behavior"]["tp"],
            attachment_counts["object_behavior"]["fp"],
            attachment_counts["object_behavior"]["fn"],
        ),
        "location_attachment": _prf(
            attachment_counts["location"]["tp"],
            attachment_counts["location"]["fp"],
            attachment_counts["location"]["fn"],
        ),
        "hyperedge_exact": {
            "matched": exact_hyperedges,
            "gold": gold_hyperedges_total,
            "predicted": predicted_hyperedges_total,
            "precision": round(exact_hyperedges / predicted_hyperedges_total, 4)
            if predicted_hyperedges_total else 0.0,
            "recall": round(exact_hyperedges / gold_hyperedges_total, 4)
            if gold_hyperedges_total else 0.0,
        },
        "issue_frame_exact": {
            "matched": exact_issue_frames,
            "gold": gold_issue_frames_total,
            "predicted": predicted_issue_frames_total,
            "precision": round(exact_issue_frames / predicted_issue_frames_total, 4)
            if predicted_issue_frames_total else 0.0,
            "recall": round(exact_issue_frames / gold_issue_frames_total, 4)
            if gold_issue_frames_total else 0.0,
        },
        "discourse": {
            name: _prf(counts["tp"], counts["fp"], counts["fn"])
            for name, counts in discourse_counts.items()
        },
        "satisfaction_exact_accuracy": round(satisfaction_exact / evaluated, 4)
        if evaluated else 0.0,
        "urgency_exact_accuracy": round(urgency_exact / evaluated, 4)
        if evaluated else 0.0,
    }
