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
    EVAL_MANIFEST_VERSION,
    EVALUATION_VERSION,
    GOLD_SCHEMA_VERSION,
    INPUT_PROFILE_VERSION,
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
            if not isinstance(value, dict) or not value.get("doc_id"):
                raise ValueError(f"line_{line_number}:invalid_manifest_record")
            manifest.append(value)

    target_ids = {str(row["doc_id"]) for row in manifest}
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
