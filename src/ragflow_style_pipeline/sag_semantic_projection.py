"""Deterministic projection from work-order semantics to SAG rows."""

import argparse
import json
import os
from pathlib import Path

from ragflow_style_pipeline.sag_semantic_versions import PROJECTION_VERSION
from ragflow_style_pipeline.work_order_input import read_work_orders

GROUP_TO_TYPE = {
    "problem_objects": "problem_object", "problem_behaviors": "problem_behavior",
    "roads": "road", "intersections": "intersection", "pois": "poi",
}
_ACCEPTED = {"accepted", "accepted_with_warnings"}


def _text(value):
    return value if isinstance(value, str) else ""


def project_semantic_record(record, order):
    """Return semantic event override, entity links, and event discourse."""
    record = record if isinstance(record, dict) else {}
    order = order if isinstance(order, dict) else {}
    doc_id = _text(record.get("doc_id")) or _text(order.get("doc_id"))
    validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
    status = _text(validation.get("status"))
    event_value = record.get("event") if isinstance(record.get("event"), dict) else {}
    summary = _text(event_value.get("summary")) or _text(record.get("event_summary"))
    event = {
        "doc_id": doc_id,
        "event_text": summary,
        "validation_status": status,
        "projection_version": PROJECTION_VERSION,
    }
    prompt_version = _text((record.get("model_run") or {}).get("prompt_version")) if isinstance(record.get("model_run"), dict) else ""
    links = []
    entities = record.get("entities") if isinstance(record.get("entities"), dict) else {}
    if status in _ACCEPTED:
        for group, entity_type in GROUP_TO_TYPE.items():
            for item in entities.get(group, []) if isinstance(entities.get(group), list) else []:
                if not isinstance(item, dict):
                    continue
                surface = _text(item.get("surface"))
                canonical = _text(item.get("canonical")) or surface
                if not canonical:
                    continue
                links.append({
                    "doc_id": doc_id, "entity_type": entity_type,
                    "entity_value": surface or canonical, "surface_form": surface or canonical,
                    "normalized_value": canonical,
                    "source_field": _text(item.get("source_field") or item.get("field")),
                    "source_channel": "semantic_llm", "matched_text": _text(item.get("evidence")),
                    "validation_status": status, "prompt_version": prompt_version,
                    "projection_version": PROJECTION_VERSION,
                })
    discourse_value = record.get("discourse") if isinstance(record.get("discourse"), dict) else {}
    metadata = order.get("metadata") if isinstance(order.get("metadata"), dict) else {}
    intents = discourse_value.get("intents") if isinstance(discourse_value.get("intents"), list) else []
    inferred = [_text(item.get("label")) for item in intents if isinstance(item, dict) and _text(item.get("label"))]
    declared = _text(metadata.get("service_object_type")) or _text(order.get("service_object_type"))
    emotions = discourse_value.get("emotions") if isinstance(discourse_value.get("emotions"), list) else []
    satisfaction = discourse_value.get("satisfaction") if isinstance(discourse_value.get("satisfaction"), dict) else {}
    urgency = discourse_value.get("urgency") if isinstance(discourse_value.get("urgency"), dict) else {}
    discourse = {
        "doc_id": doc_id, "declared_intent": declared,
        "inferred_intents_json": json.dumps(inferred, ensure_ascii=False),
        "intent_conflict": "true" if declared and inferred and declared not in inferred else "false",
        "emotions_json": json.dumps(emotions, ensure_ascii=False, separators=(",", ":")),
        "satisfaction": _text(satisfaction.get("label")) or "unknown",
        "satisfaction_target": _text(satisfaction.get("target")),
        "satisfaction_evidence": _text(satisfaction.get("evidence")),
        "urgency": _text(urgency.get("level")) or "normal",
        "urgency_evidence": _text(urgency.get("evidence")),
        "projection_version": PROJECTION_VERSION,
    }
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
    accepted = 0
    with Path(input_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = _text(record.get("doc_id"))
            if doc_id not in orders:
                raise ValueError(f"line_{line_number}:unknown_doc_id:{doc_id}")
            event, event_links, event_discourse = project_semantic_record(record, orders[doc_id])
            events.append(event); links.extend(event_links); discourse.append(event_discourse)
            if _text((record.get("validation") or {}).get("status")) in _ACCEPTED:
                accepted += 1
    _atomic_jsonl(links_path, links)
    _atomic_jsonl(discourse_path, discourse)
    if events_path:
        _atomic_jsonl(events_path, events)
    return {"records": len(events), "accepted": accepted, "links": len(links), "discourse": len(discourse)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Project semantic work-order JSONL into SAG rows without loading a model.")
    parser.add_argument("--input", required=True, help="Work-order semantic JSONL")
    parser.add_argument("--orders", required=True, help="Desensitized multiview work-order JSONL")
    parser.add_argument("--links", required=True, help="Output SAG entity links JSONL")
    parser.add_argument("--discourse", required=True, help="Output event discourse JSONL")
    parser.add_argument("--events", default="", help="Optional output semantic event JSONL")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(project_semantics_file(args.input, args.orders, args.links, args.discourse, args.events or None), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
