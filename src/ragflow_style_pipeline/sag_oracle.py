"""Local Oracle SAG projection and retrieval evaluation over adjudicated issue gold."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from ragflow_style_pipeline.sag_entities import normalize_entity_value
from ragflow_style_pipeline.sag_semantic_audit import (
    project_gold_issues,
    validate_gold_annotations,
)
from ragflow_style_pipeline.sag_semantic_versions import (
    GOLD_SCHEMA_VERSION,
    ORACLE_EVALUATION_VERSION,
    ORACLE_PROJECTION_VERSION,
    ORACLE_QUERY_SCHEMA_VERSION,
)

_QUERY_ENTITY_TYPES = {
    "problem_object", "problem_behavior", "road", "intersection", "poi",
    "issue_predicate", "request_action",
}
_DEFAULT_FRONTIER_TYPES = {
    "problem_object", "problem_behavior", "road", "intersection", "poi",
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _connect(db_path, read_only=False):
    import duckdb

    return duckdb.connect(str(db_path), read_only=read_only)


def build_oracle_sag_db(gold_path, db_path, mode):
    """Build a private flat or issue-aware Oracle graph from completed gold."""
    if mode not in {"flat", "issue-aware"}:
        raise ValueError("invalid_oracle_projection_mode")
    validation = validate_gold_annotations(gold_path, require_complete=True)
    if not validation["ready_for_evaluation"]:
        raise ValueError("gold_annotations_not_ready")

    started = time.perf_counter()
    order_rows, issue_rows, member_rows = project_gold_issues(
        gold_path, flat=mode == "flat"
    )
    event_rows = [
        {
            "event_id": row["issue_id"],
            "doc_id": row["doc_id"],
            "mode": row["mode"],
            "time_scope": row["time_scope"],
            "projection_mode": row["projection"],
            "projection_version": ORACLE_PROJECTION_VERSION,
        }
        for row in issue_rows
    ]
    links = []
    seen = set()
    for row in member_rows:
        normalized = normalize_entity_value(row.get("normalized_value") or row.get("surface"))
        key = (row.get("issue_id"), row.get("entity_type"), normalized)
        if not normalized or key in seen:
            continue
        seen.add(key)
        links.append({
            "event_id": row.get("issue_id", ""),
            "doc_id": row.get("doc_id", ""),
            "entity_type": row.get("entity_type", ""),
            "normalized_value": normalized,
            "surface": row.get("surface", ""),
            "role": row.get("role", ""),
            "projection_version": ORACLE_PROJECTION_VERSION,
        })

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute("drop table if exists oracle_orders")
        conn.execute("drop table if exists oracle_events")
        conn.execute("drop table if exists oracle_members")
        conn.execute(
            "create table oracle_orders (doc_id varchar, content_hash varchar, subset varchar, "
            "gold_schema varchar, projection_version varchar)"
        )
        conn.execute(
            "create table oracle_events (event_id varchar, doc_id varchar, mode varchar, "
            "time_scope varchar, projection_mode varchar, projection_version varchar)"
        )
        conn.execute(
            "create table oracle_members (event_id varchar, doc_id varchar, entity_type varchar, "
            "normalized_value varchar, surface varchar, role varchar, projection_version varchar)"
        )
        if order_rows:
            conn.executemany(
                "insert into oracle_orders values (?, ?, ?, ?, ?)",
                [[
                    row.get("doc_id", ""), row.get("content_hash", ""), row.get("subset", ""),
                    GOLD_SCHEMA_VERSION, ORACLE_PROJECTION_VERSION,
                ] for row in order_rows],
            )
        if event_rows:
            conn.executemany(
                "insert into oracle_events values (?, ?, ?, ?, ?, ?)",
                [[row[key] for key in (
                    "event_id", "doc_id", "mode", "time_scope",
                    "projection_mode", "projection_version",
                )] for row in event_rows],
            )
        if links:
            conn.executemany(
                "insert into oracle_members values (?, ?, ?, ?, ?, ?, ?)",
                [[row[key] for key in (
                    "event_id", "doc_id", "entity_type", "normalized_value",
                    "surface", "role", "projection_version",
                )] for row in links],
            )
        conn.execute("create index idx_oracle_events_event on oracle_events(event_id)")
        conn.execute("create index idx_oracle_events_doc on oracle_events(doc_id)")
        conn.execute("create index idx_oracle_members_event on oracle_members(event_id)")
        conn.execute("create index idx_oracle_members_entity on oracle_members(entity_type, normalized_value)")

    return {
        "schema": ORACLE_PROJECTION_VERSION,
        "private": True,
        "mode": mode,
        "gold_sha256": _file_sha256(gold_path),
        "orders": len(order_rows),
        "events": len(event_rows),
        "members": len(links),
        "entity_type_counts": dict(Counter(row["entity_type"] for row in links)),
        "build_seconds": round(time.perf_counter() - started, 4),
    }


def load_oracle_queries(path):
    """Load and validate a private query/relevance JSONL without returning source text."""
    queries = []
    query_ids = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("schema") != ORACLE_QUERY_SCHEMA_VERSION:
                raise ValueError(f"line_{line_number}:invalid_oracle_query_schema")
            if value.get("private") is not True:
                raise ValueError(f"line_{line_number}:private_marker_missing")
            query_id = str(value.get("query_id") or "").strip()
            if not query_id or query_id in query_ids:
                raise ValueError(f"line_{line_number}:invalid_or_duplicate_query_id")
            query_ids.add(query_id)
            groups = value.get("seed_entities")
            if not isinstance(groups, list) or not groups:
                raise ValueError(f"line_{line_number}:missing_seed_entities")
            normalized_groups = []
            for group in groups:
                if not isinstance(group, dict):
                    raise ValueError(f"line_{line_number}:invalid_seed_group")
                entity_type = str(group.get("entity_type") or "").strip()
                raw_values = group.get("values")
                if (
                    not isinstance(raw_values, list)
                    or any(not isinstance(item, str) for item in raw_values)
                ):
                    raise ValueError(f"line_{line_number}:invalid_seed_group")
                values = sorted({
                    normalize_entity_value(item)
                    for item in raw_values if normalize_entity_value(item)
                })
                if entity_type not in _QUERY_ENTITY_TYPES or not values:
                    raise ValueError(f"line_{line_number}:invalid_seed_group")
                normalized_groups.append({"entity_type": entity_type, "values": values})
            operator = str(value.get("seed_group_operator") or "AND").upper()
            if operator not in {"AND", "OR"}:
                raise ValueError(f"line_{line_number}:invalid_seed_operator")
            relevance = value.get("relevance")
            if not isinstance(relevance, list) or not relevance:
                raise ValueError(f"line_{line_number}:missing_relevance")
            grades = {}
            for item in relevance:
                doc_id = str(item.get("doc_id") or "").strip() if isinstance(item, dict) else ""
                grade = item.get("grade") if isinstance(item, dict) else None
                if (
                    not doc_id or doc_id in grades or not isinstance(grade, int)
                    or isinstance(grade, bool) or not 1 <= grade <= 3
                ):
                    raise ValueError(f"line_{line_number}:invalid_relevance")
                grades[doc_id] = grade
            raw_expansion = value.get("expansion")
            if raw_expansion is not None and not isinstance(raw_expansion, dict):
                raise ValueError(f"line_{line_number}:invalid_expansion")
            expansion = raw_expansion or {}
            enabled = expansion.get("enabled", True)
            max_expanded_docs = expansion.get("max_expanded_docs", 2000)
            if not isinstance(enabled, bool):
                raise ValueError(f"line_{line_number}:invalid_expansion_enabled")
            if (
                not isinstance(max_expanded_docs, int)
                or isinstance(max_expanded_docs, bool)
                or max_expanded_docs < 0
            ):
                raise ValueError(f"line_{line_number}:invalid_max_expanded_docs")
            frontier_types = expansion.get("frontier_entity_types")
            if frontier_types is None:
                frontier_types = sorted(_DEFAULT_FRONTIER_TYPES)
            if not isinstance(frontier_types, list):
                raise ValueError(f"line_{line_number}:invalid_frontier_types")
            frontier_types = sorted({str(item) for item in frontier_types})
            if set(frontier_types) - _DEFAULT_FRONTIER_TYPES:
                raise ValueError(f"line_{line_number}:invalid_frontier_types")
            queries.append({
                "query_id": query_id,
                "seed_entities": normalized_groups,
                "seed_group_operator": operator,
                "expansion": {
                    "enabled": enabled,
                    "frontier_entity_types": frontier_types,
                    "max_expanded_docs": max_expanded_docs,
                },
                "grades": grades,
            })
    if not queries:
        raise ValueError("empty_oracle_query_set")
    return queries


def _number_summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": 0, "p50": 0.0, "p95": 0.0, "max": 0, "mean": 0.0}

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return round(
            values[lower] * (upper - position) + values[upper] * (position - lower), 4
        )

    return {
        "count": len(values),
        "min": values[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": values[-1],
        "mean": round(sum(values) / len(values), 4),
    }


def _graph_statistics(events):
    entity_events = {}
    entity_docs = {}
    frontier_events = {}
    frontier_docs = {}
    events_by_doc = Counter()
    for event_id, event in events.items():
        events_by_doc[event["doc_id"]] += 1
        for member in event["members"]:
            entity_events.setdefault(member, set()).add(event_id)
            entity_docs.setdefault(member, set()).add(event["doc_id"])
            if member[0] in _DEFAULT_FRONTIER_TYPES:
                frontier_events.setdefault(member, set()).add(event_id)
                frontier_docs.setdefault(member, set()).add(event["doc_id"])
    return {
        "events": len(events),
        "event_member_count": _number_summary([len(event["members"]) for event in events.values()]),
        "events_per_doc": _number_summary(list(events_by_doc.values())),
        "entity_event_degree": _number_summary([len(value) for value in entity_events.values()]),
        "entity_doc_degree": _number_summary([len(value) for value in entity_docs.values()]),
        "frontier_entity_event_degree": _number_summary([
            len(value) for value in frontier_events.values()
        ]),
        "frontier_entity_doc_degree": _number_summary([
            len(value) for value in frontier_docs.values()
        ]),
    }


def _load_graph(db_path):
    with _connect(db_path, read_only=True) as conn:
        mode_rows = conn.execute(
            "select distinct projection_mode, projection_version from oracle_events"
        ).fetchall()
        order_schema_rows = conn.execute(
            "select distinct gold_schema, projection_version from oracle_orders"
        ).fetchall()
        if (
            len(mode_rows) != 1
            or mode_rows[0][1] != ORACLE_PROJECTION_VERSION
            or order_schema_rows != [(GOLD_SCHEMA_VERSION, ORACLE_PROJECTION_VERSION)]
        ):
            raise ValueError("invalid_oracle_database")
        order_ids = {row[0] for row in conn.execute("select doc_id from oracle_orders").fetchall()}
        event_rows = conn.execute("select event_id, doc_id from oracle_events").fetchall()
        member_rows = conn.execute(
            "select event_id, entity_type, normalized_value from oracle_members"
        ).fetchall()
    events = {event_id: {"doc_id": doc_id, "members": set()} for event_id, doc_id in event_rows}
    for event_id, entity_type, normalized in member_rows:
        if event_id in events:
            events[event_id]["members"].add((entity_type, normalized))
    return mode_rows[0][0], order_ids, events


def query_oracle_graph(db_path, query):
    """Run one deterministic same-event seed plus one-hop Oracle query."""
    mode, _order_ids, events = _load_graph(db_path)
    groups = [
        {(group["entity_type"], value) for value in group["values"]}
        for group in query["seed_entities"]
    ]
    operator = query["seed_group_operator"]
    seed_events = {}
    for event_id, event in events.items():
        matched_groups = [event["members"] & group for group in groups]
        qualifies = any(matched_groups) if operator == "OR" else all(matched_groups)
        if qualifies:
            matched = set().union(*matched_groups)
            seed_events[event_id] = {
                "doc_id": event["doc_id"],
                "matched": matched,
                "score": 10.0 * len(matched),
            }

    expansion = query["expansion"]
    expanded_events = {}
    if expansion["enabled"] and seed_events:
        frontier_types = set(expansion["frontier_entity_types"])
        frontier = {
            member
            for event_id in seed_events
            for member in events[event_id]["members"]
            if member[0] in frontier_types
        }
        for event_id, event in events.items():
            if event_id in seed_events:
                continue
            matched = event["members"] & frontier
            if matched:
                expanded_events[event_id] = {
                    "doc_id": event["doc_id"],
                    "matched": matched,
                    "score": 3.0 * len(matched) - 5.0,
                }

    docs = {}
    for stage, rows in (("seed_entity", seed_events), ("one_hop_expansion", expanded_events)):
        for event_id, item in rows.items():
            doc = docs.setdefault(item["doc_id"], {
                "doc_id": item["doc_id"],
                "score": item["score"],
                "match_stage": stage,
                "seed_events": 0,
                "expanded_events": 0,
                "matched_entities": set(),
            })
            if stage == "seed_entity":
                doc["match_stage"] = "seed_entity"
                doc["seed_events"] += 1
            else:
                doc["expanded_events"] += 1
            doc["score"] = max(doc["score"], item["score"])
            doc["matched_entities"].update(item["matched"])

    seed_docs = [value for value in docs.values() if value["match_stage"] == "seed_entity"]
    expanded_docs = [value for value in docs.values() if value["match_stage"] == "one_hop_expansion"]
    seed_docs.sort(key=lambda item: (-item["score"], item["doc_id"]))
    expanded_docs.sort(key=lambda item: (-item["score"], item["doc_id"]))
    expanded_docs = expanded_docs[:expansion["max_expanded_docs"]]
    results = seed_docs + expanded_docs
    results.sort(key=lambda item: (
        -item["score"], 0 if item["match_stage"] == "seed_entity" else 1, item["doc_id"]
    ))
    for rank, item in enumerate(results, 1):
        item["rank"] = rank
        item["matched_entities"] = [list(value) for value in sorted(item["matched_entities"])]
    return mode, results


def _metrics(results, grades, cutoffs):
    relevant = set(grades)
    seed_docs = {row["doc_id"] for row in results if row["match_stage"] == "seed_entity"}
    expanded_docs = {row["doc_id"] for row in results if row["match_stage"] == "one_hop_expansion"}
    metrics = {}
    for cutoff in cutoffs:
        top = results[:cutoff]
        hits = sum(row["doc_id"] in relevant for row in top)
        dcg = sum(
            ((2 ** grades.get(row["doc_id"], 0)) - 1) / math.log2(rank + 1)
            for rank, row in enumerate(top, 1)
        )
        ideal = sorted(grades.values(), reverse=True)[:cutoff]
        idcg = sum(
            ((2 ** grade) - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(ideal, 1)
        )
        metrics[f"precision@{cutoff}"] = round(hits / cutoff, 4)
        metrics[f"recall@{cutoff}"] = round(hits / len(relevant), 4)
        metrics[f"ndcg@{cutoff}"] = round(dcg / idcg, 4) if idcg else 0.0
    first_relevant = next(
        (rank for rank, row in enumerate(results, 1) if row["doc_id"] in relevant), None
    )
    relevant_seed = len(seed_docs & relevant)
    irrelevant_seed = len(seed_docs - relevant)
    relevant_expanded = len(expanded_docs & relevant)
    irrelevant_expanded = len(expanded_docs - relevant)
    return {
        **metrics,
        "mrr": round(1 / first_relevant, 4) if first_relevant else 0.0,
        "relevant_docs": len(relevant),
        "retrieved_docs": len(results),
        "seed_docs": len(seed_docs),
        "expanded_docs": len(expanded_docs),
        "seed_recall": round(relevant_seed / len(relevant), 4),
        "false_seed_rate": round(irrelevant_seed / len(seed_docs), 4) if seed_docs else 0.0,
        "one_hop_precision": round(relevant_expanded / len(expanded_docs), 4)
        if expanded_docs else None,
        "erroneous_expansion_rate": round(irrelevant_expanded / len(expanded_docs), 4)
        if expanded_docs else None,
    }


def evaluate_oracle_retrieval(flat_db, issue_db, query_path, cutoffs=(5, 10)):
    """Compare flat and issue-aware Oracle retrieval without leaking query or document IDs."""
    queries = load_oracle_queries(query_path)
    flat_mode, flat_orders, flat_events = _load_graph(flat_db)
    issue_mode, issue_orders, issue_events = _load_graph(issue_db)
    if flat_mode != "flat" or issue_mode != "issue_aware":
        raise ValueError("oracle_database_modes_mismatch")
    if flat_orders != issue_orders:
        raise ValueError("oracle_database_order_sets_differ")
    for query in queries:
        if set(query["grades"]) - flat_orders:
            raise ValueError("relevance_doc_not_in_oracle_gold")

    cutoffs = tuple(sorted({int(value) for value in cutoffs if int(value) > 0}))
    if not cutoffs:
        raise ValueError("empty_metric_cutoffs")
    per_mode = {"flat": [], "issue_aware": []}
    traces = []
    paired = Counter()
    for query in queries:
        _mode, flat_results = query_oracle_graph(flat_db, query)
        _mode, issue_results = query_oracle_graph(issue_db, query)
        grades = query["grades"]
        flat_metrics = _metrics(flat_results, grades, cutoffs)
        issue_metrics = _metrics(issue_results, grades, cutoffs)
        per_mode["flat"].append(flat_metrics)
        per_mode["issue_aware"].append(issue_metrics)
        relevant = set(grades)
        flat_seed = {row["doc_id"] for row in flat_results if row["match_stage"] == "seed_entity"}
        issue_seed = {row["doc_id"] for row in issue_results if row["match_stage"] == "seed_entity"}
        flat_expanded = {
            row["doc_id"] for row in flat_results
            if row["match_stage"] == "one_hop_expansion"
        }
        issue_expanded = {
            row["doc_id"] for row in issue_results
            if row["match_stage"] == "one_hop_expansion"
        }
        flat_retrieved = {row["doc_id"] for row in flat_results}
        issue_retrieved = {row["doc_id"] for row in issue_results}
        flat_only_seed = flat_seed - issue_seed
        flat_only_expansion = flat_expanded - issue_retrieved
        issue_only_expansion = issue_expanded - flat_retrieved
        flat_only_results = flat_retrieved - issue_retrieved
        issue_only_results = issue_retrieved - flat_retrieved
        paired["flat_only_seed_docs"] += len(flat_only_seed)
        paired["removed_irrelevant_flat_seed_docs"] += len(flat_only_seed - relevant)
        paired["lost_relevant_flat_seed_docs"] += len(flat_only_seed & relevant)
        paired["flat_expansion_promoted_to_issue_seed"] += len(flat_expanded & issue_seed)
        paired["removed_irrelevant_flat_expansion_docs"] += len(flat_only_expansion - relevant)
        paired["lost_relevant_flat_expansion_docs"] += len(flat_only_expansion & relevant)
        paired["new_irrelevant_issue_expansion_docs"] += len(issue_only_expansion - relevant)
        paired["new_relevant_issue_expansion_docs"] += len(issue_only_expansion & relevant)
        paired["removed_irrelevant_flat_result_docs"] += len(flat_only_results - relevant)
        paired["lost_relevant_flat_result_docs"] += len(flat_only_results & relevant)
        paired["new_irrelevant_issue_result_docs"] += len(issue_only_results - relevant)
        paired["new_relevant_issue_result_docs"] += len(issue_only_results & relevant)
        traces.append({
            "schema": ORACLE_EVALUATION_VERSION,
            "private": True,
            "query_id": query["query_id"],
            "query": {
                "seed_entities": query["seed_entities"],
                "seed_group_operator": query["seed_group_operator"],
                "expansion": query["expansion"],
            },
            "relevance": [
                {"doc_id": doc_id, "grade": grade}
                for doc_id, grade in sorted(grades.items())
            ],
            "flat_results": flat_results,
            "issue_aware_results": issue_results,
            "flat_metrics": flat_metrics,
            "issue_aware_metrics": issue_metrics,
        })

    metric_names = [
        *(name for cutoff in cutoffs for name in (
            f"precision@{cutoff}", f"recall@{cutoff}", f"ndcg@{cutoff}",
        )),
        "mrr", "seed_recall", "false_seed_rate",
        "one_hop_precision", "erroneous_expansion_rate",
    ]

    def aggregate(rows):
        output = {}
        for name in metric_names:
            values = [row[name] for row in rows if row.get(name) is not None]
            output[name] = round(sum(values) / len(values), 4) if values else None
        output["queries_with_expansion"] = sum(row["expanded_docs"] > 0 for row in rows)
        output["retrieved_docs_total"] = sum(row["retrieved_docs"] for row in rows)
        output["seed_docs_total"] = sum(row["seed_docs"] for row in rows)
        output["expanded_docs_total"] = sum(row["expanded_docs"] for row in rows)
        return output

    flat_aggregate = aggregate(per_mode["flat"])
    issue_aggregate = aggregate(per_mode["issue_aware"])
    deltas = {
        name: round(issue_aggregate[name] - flat_aggregate[name], 4)
        for name in metric_names
        if issue_aggregate.get(name) is not None and flat_aggregate.get(name) is not None
    }
    report = {
        "schema": ORACLE_EVALUATION_VERSION,
        "query_schema": ORACLE_QUERY_SCHEMA_VERSION,
        "projection_version": ORACLE_PROJECTION_VERSION,
        "private_query_input": True,
        "query_source_sha256": _file_sha256(query_path),
        "queries": len(queries),
        "cutoffs": list(cutoffs),
        "flat": flat_aggregate,
        "issue_aware": issue_aggregate,
        "issue_aware_minus_flat": deltas,
        "graph_structure": {
            "flat": _graph_statistics(flat_events),
            "issue_aware": _graph_statistics(issue_events),
        },
        "paired_retrieval_effects": dict(paired),
    }
    return report, traces
