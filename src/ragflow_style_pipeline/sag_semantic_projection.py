"""Deterministic projection from v7 flat or v8 issue semantics to SAG rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_versions import (
    ISSUE_PROJECTION_VERSION,
    PROJECTION_VERSION,
)
from ragflow_style_pipeline.work_order_input import read_work_orders

GROUP_TO_TYPE = {
    "problem_objects": "problem_object", "problem_behaviors": "problem_behavior",
    "roads": "road", "intersections": "intersection", "pois": "poi",
}
ISSUE_GROUP_TO_TYPE = {
    "objects": "problem_object",
    "problem_behaviors": "problem_behavior",
    "question_focus": "issue_predicate",
    "request_actions": "request_action",
}
_ACCEPTED = {"accepted", "accepted_with_warnings"}


def _text(value):
    return value if isinstance(value, str) else ""


def _event_base(doc_id):
    return "event_" + hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16]


def _discourse_row(record, order, event_id, projection_version):
    discourse_value = record.get("discourse") if isinstance(record.get("discourse"), dict) else {}
    metadata = order.get("metadata") if isinstance(order.get("metadata"), dict) else {}
    intents = discourse_value.get("intents") if isinstance(discourse_value.get("intents"), list) else []
    inferred = [_text(item.get("label")) for item in intents if isinstance(item, dict) and _text(item.get("label"))]
    declared = _text(metadata.get("service_object_type")) or _text(order.get("service_object_type"))
    emotions = discourse_value.get("emotions") if isinstance(discourse_value.get("emotions"), list) else []
    satisfaction = discourse_value.get("satisfaction") if isinstance(discourse_value.get("satisfaction"), dict) else {}
    urgency = discourse_value.get("urgency") if isinstance(discourse_value.get("urgency"), dict) else {}
    return {
        "event_id": event_id,
        "doc_id": _text(record.get("doc_id")) or _text(order.get("doc_id")),
        "declared_intent": declared,
        "inferred_intents_json": json.dumps(inferred, ensure_ascii=False),
        "intent_conflict": "true" if declared and inferred and declared not in inferred else "false",
        "emotions_json": json.dumps(emotions, ensure_ascii=False, separators=(",", ":")),
        "satisfaction": _text(satisfaction.get("label")) or "unknown",
        "satisfaction_target": _text(satisfaction.get("target")),
        "satisfaction_evidence": _text(satisfaction.get("evidence")),
        "urgency": _text(urgency.get("level")) or "normal",
        "urgency_evidence": _text(urgency.get("evidence")),
        "projection_version": projection_version,
    }


def _issue_event_text(issue):
    parts = []
    for label, group in (
        ("对象", "objects"), ("问题", "problem_behaviors"),
        ("咨询", "question_focus"), ("诉求", "request_actions"),
    ):
        values = [
            _text(item.get("surface")) for item in issue.get(group, [])
            if isinstance(item, dict) and _text(item.get("surface"))
        ] if isinstance(issue.get(group), list) else []
        if values:
            parts.append(label + "：" + "、".join(values))
    locations = [
        _text(item.get("surface")) for item in issue.get("locations", [])
        if isinstance(item, dict) and _text(item.get("surface"))
    ] if isinstance(issue.get("locations"), list) else []
    if locations:
        parts.append("地点：" + "、".join(locations))
    return "；".join(parts)


def _link(doc_id, event_id, entity_type, item, status, prompt_version, projection_version, role):
    surface = _text(item.get("surface")) if isinstance(item, dict) else ""
    if not surface:
        return None
    # The v8 model does not generate canonical.  Surface is a reversible local
    # fallback, not a claim that global entity linking has been solved.
    normalized = _text(item.get("canonical")) or surface
    return {
        "event_id": event_id,
        "doc_id": doc_id,
        "entity_type": entity_type,
        "entity_value": surface,
        "surface_form": surface,
        "normalized_value": normalized,
        "source_field": _text(item.get("source_field") or item.get("field")),
        "source_channel": "semantic_llm",
        "matched_text": _text(item.get("evidence")),
        "semantic_role": role,
        "validation_status": status,
        "prompt_version": prompt_version,
        "projection_version": projection_version,
    }


def project_semantic_record_rows(record, order):
    """Return all event, member, and discourse rows for one semantic record."""
    record = record if isinstance(record, dict) else {}
    order = order if isinstance(order, dict) else {}
    doc_id = _text(record.get("doc_id")) or _text(order.get("doc_id"))
    validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
    status = _text(validation.get("status"))
    event_value = record.get("event") if isinstance(record.get("event"), dict) else {}
    summary = _text(event_value.get("summary")) or _text(record.get("event_summary"))
    model_run = record.get("model_run") if isinstance(record.get("model_run"), dict) else {}
    prompt_version = _text(model_run.get("prompt_version"))
    issues = record.get("issues") if isinstance(record.get("issues"), list) else []
    accepted = status in _ACCEPTED
    events, links, discourses = [], [], []

    if issues:
        projection_version = ISSUE_PROJECTION_VERSION
        for index, issue in enumerate(issues, 1):
            if not isinstance(issue, dict):
                continue
            event_id = f"{_event_base(doc_id)}::issue::{index}"
            events.append({
                "event_id": event_id,
                "doc_id": doc_id,
                "event_text": _issue_event_text(issue) or summary,
                "event_kind": "issue",
                "issue_index": str(index),
                "time_scope": _text(issue.get("time_scope")) or "current",
                "validation_status": status,
                "projection_version": projection_version,
            })
            if accepted:
                for group, entity_type in ISSUE_GROUP_TO_TYPE.items():
                    values = issue.get(group) if isinstance(issue.get(group), list) else []
                    for item in values:
                        link = _link(
                            doc_id, event_id, entity_type, item, status,
                            prompt_version, projection_version, group,
                        )
                        if link:
                            links.append(link)
                locations = issue.get("locations") if isinstance(issue.get("locations"), list) else []
                for item in locations:
                    entity_type = _text(item.get("type")) if isinstance(item, dict) else ""
                    if entity_type not in {"road", "intersection", "poi"}:
                        continue
                    link = _link(
                        doc_id, event_id, entity_type, item, status,
                        prompt_version, projection_version, "location",
                    )
                    if link:
                        links.append(link)
            discourses.append(_discourse_row(record, order, event_id, projection_version))
        return events, links, discourses

    projection_version = PROJECTION_VERSION
    event_id = _event_base(doc_id)
    events.append({
        "event_id": event_id,
        "doc_id": doc_id,
        "event_text": summary,
        "event_kind": "order",
        "issue_index": "",
        "time_scope": "current",
        "validation_status": status,
        "projection_version": projection_version,
    })
    entities = record.get("entities") if isinstance(record.get("entities"), dict) else {}
    if accepted:
        for group, entity_type in GROUP_TO_TYPE.items():
            values = entities.get(group) if isinstance(entities.get(group), list) else []
            for item in values:
                link = _link(
                    doc_id, event_id, entity_type, item, status,
                    prompt_version, projection_version, group,
                )
                if link:
                    links.append(link)
    discourses.append(_discourse_row(record, order, event_id, projection_version))
    return events, links, discourses


def project_semantic_record(record, order):
    """Backward-compatible one-event helper used by existing integrations."""
    events, links, discourses = project_semantic_record_rows(record, order)
    event = events[0] if events else {
        "doc_id": _text(record.get("doc_id")), "event_text": "",
        "validation_status": "", "projection_version": PROJECTION_VERSION,
    }
    discourse = discourses[0] if discourses else {}
    return event, links, discourse


def _atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def project_semantics_file(input_path, orders_path, links_path, discourse_path, events_path=None):
    orders = {row["doc_id"]: row for row in read_work_orders(orders_path)}
    links, discourse, events = [], [], []
    accepted = issue_events = 0
    with Path(input_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = _text(record.get("doc_id"))
            if doc_id not in orders:
                raise ValueError(f"line_{line_number}:unknown_doc_id:{doc_id}")
            record_events, record_links, record_discourse = project_semantic_record_rows(
                record, orders[doc_id]
            )
            events.extend(record_events)
            links.extend(record_links)
            discourse.extend(record_discourse)
            issue_events += sum(row.get("event_kind") == "issue" for row in record_events)
            if _text((record.get("validation") or {}).get("status")) in _ACCEPTED:
                accepted += 1
    _atomic_jsonl(links_path, links)
    _atomic_jsonl(discourse_path, discourse)
    if events_path:
        _atomic_jsonl(events_path, events)
    return {
        "records": len({row["doc_id"] for row in events}),
        "events": len(events),
        "issue_events": issue_events,
        "accepted": accepted,
        "links": len(links),
        "discourse": len(discourse),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Project semantic JSONL into SAG rows without loading a model."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--orders", required=True)
    parser.add_argument("--links", required=True)
    parser.add_argument("--discourse", required=True)
    parser.add_argument("--events", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(
        project_semantics_file(
            args.input, args.orders, args.links, args.discourse, args.events or None
        ), ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
