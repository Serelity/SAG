"""Pure SAG-lite querying over event-entity DuckDB tables."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from ragflow_style_pipeline.sag_entities import normalize_entity_value


STAT_ENTITY_TYPES = ["street", "road", "intersection", "poi", "problem_object", "problem_behavior"]
ALLOWED_FRONTIER_ENTITY_TYPES = {
    "problem_object", "problem_behavior", "area", "street", "road",
    "intersection", "poi", "case_type", "time_month", "department", "lnglat",
}

COVERAGE_ENTITY_TYPES = [
    "area",
    "street",
    "road",
    "intersection",
    "poi",
    "problem_object",
    "problem_behavior",
    "case_type",
    "lnglat",
]


def load_sag_query_config(config_path):
    """Load a SAG query config JSON file."""
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def score_sag_result(match_stage, matched_seed_count, matched_space_count, confidence_sum):
    """Return structural score without semantic similarity."""
    seed_score = float(matched_seed_count) * 10.0
    space_score = float(matched_space_count) * 3.0
    confidence_score = float(confidence_sum)
    expansion_penalty = 5.0 if match_stage == "one_hop_expansion" else 0.0
    return round(seed_score + space_score + confidence_score - expansion_penalty, 6)


def _connect(db_path):
    import duckdb

    return duckdb.connect(str(db_path), read_only=False)


def _event_filter_sql(filters, event_alias="e", discourse_alias="d"):
    """Build parameterized event/discourse filters and whether discourse is needed."""
    filters = filters or {}
    clauses, params = _month_filter_sql(filters, event_alias)
    needs_discourse = False
    satisfaction = filters.get("satisfaction")
    if satisfaction:
        clauses.append(f"{discourse_alias}.satisfaction = ?")
        params.append(str(satisfaction))
        needs_discourse = True
    urgency_values = [str(value) for value in (filters.get("urgency_in") or []) if str(value)]
    if urgency_values:
        clauses.append(f"{discourse_alias}.urgency in ({', '.join(['?'] * len(urgency_values))})")
        params.extend(urgency_values)
        needs_discourse = True
    intent = filters.get("intent")
    if intent:
        clauses.append(
            f"exists (select 1 from json_each({discourse_alias}.inferred_intents_json) j where trim(cast(j.value as varchar), '\"') = ?)"
        )
        params.append(str(intent))
        needs_discourse = True
    return clauses, params, needs_discourse


def _month_filter_sql(filters, alias="e"):
    filters = filters or {}
    clauses = []
    params = []
    month_gte = filters.get("call_month_gte")
    month_lte = filters.get("call_month_lte")
    if month_gte:
        clauses.append(f"{alias}.event_month >= ?")
        params.append(month_gte)
    if month_lte:
        clauses.append(f"{alias}.event_month <= ?")
        params.append(month_lte)
    return clauses, params


def _seed_event_ids(conn, config):
    seed_groups = config.get("seed_entities") or []
    if not seed_groups:
        return set(), {}

    group_event_sets = []
    matched_by_event = {}
    confidence_by_event = Counter()
    for group in seed_groups:
        entity_type = group["entity_type"]
        values = [normalize_entity_value(value) for value in group.get("values", []) if normalize_entity_value(value)]
        if not values:
            group_event_sets.append(set())
            continue

        placeholders = ", ".join(["?"] * len(values))
        rows = conn.execute(
            f"""
            select l.event_id, l.doc_id, l.entity_type, l.entity_value, l.confidence
            from sag_event_entity_links l
            join sag_events e on e.event_id = l.event_id
            where l.entity_type = ?
              and replace(l.entity_value, ' ', '') in ({placeholders})
            """,
            [entity_type] + values,
        ).fetchall()

        event_ids = set()
        for event_id, _doc_id, matched_type, entity_value, confidence in rows:
            event_ids.add(event_id)
            matched_by_event.setdefault(event_id, {}).setdefault(matched_type, set()).add(entity_value)
            confidence_by_event[event_id] += float(confidence or 0.0)
        group_event_sets.append(event_ids)

    if str(config.get("seed_group_operator", "AND")).upper() == "OR":
        seed_ids = set().union(*group_event_sets) if group_event_sets else set()
    else:
        seed_ids = set.intersection(*group_event_sets) if group_event_sets else set()

    filters = config.get("filters") or {}
    filter_clauses, filter_params, needs_discourse = _event_filter_sql(filters, "e", "d")
    if filter_clauses and seed_ids:
        placeholders = ", ".join(["?"] * len(seed_ids))
        discourse_join = "join sag_event_discourse d on d.event_id = e.event_id" if needs_discourse else ""
        filtered_ids = {
            row[0]
            for row in conn.execute(
                f"""
                select e.event_id
                from sag_events e
                {discourse_join}
                where e.event_id in ({placeholders})
                  and {" and ".join(filter_clauses)}
                """,
                list(seed_ids) + filter_params,
            ).fetchall()
        }
        seed_ids = filtered_ids

    seed_info = {
        event_id: {
            "matched_entities": {key: sorted(value) for key, value in matched_by_event.get(event_id, {}).items()},
            "confidence_sum": confidence_by_event[event_id],
        }
        for event_id in seed_ids
    }
    return seed_ids, seed_info


def _event_doc_map(conn, event_ids):
    if not event_ids:
        return {}
    placeholders = ", ".join(["?"] * len(event_ids))
    rows = conn.execute(
        f"select event_id, doc_id from sag_events where event_id in ({placeholders})",
        list(event_ids),
    ).fetchall()
    return {event_id: doc_id for event_id, doc_id in rows}


def _frontier_entities(conn, seed_ids, entity_types):
    if not seed_ids or not entity_types:
        return []
    seed_placeholders = ", ".join(["?"] * len(seed_ids))
    type_placeholders = ", ".join(["?"] * len(entity_types))
    return conn.execute(
        f"""
        select distinct entity_id, entity_type, entity_value
        from sag_event_entity_links
        where event_id in ({seed_placeholders})
          and entity_type in ({type_placeholders})
        """,
        list(seed_ids) + list(entity_types),
    ).fetchall()


def _expanded_events(conn, seed_ids, frontier_rows, config):
    if not frontier_rows:
        return {}
    entity_ids = [row[0] for row in frontier_rows]
    entity_id_to_type_value = {row[0]: (row[1], row[2]) for row in frontier_rows}
    placeholders = ", ".join(["?"] * len(entity_ids))
    max_expanded = int((config.get("expansion") or {}).get("max_expanded_events", 2000))
    filters = config.get("filters") or {}
    filter_clauses, filter_params, needs_discourse = _event_filter_sql(filters, "e", "d")
    discourse_join = "join sag_event_discourse d on d.event_id = e.event_id" if needs_discourse else ""
    filter_sql = " and " + " and ".join(filter_clauses) if filter_clauses else ""
    rows = conn.execute(
        f"""
        select l.event_id, l.doc_id, l.entity_id, l.confidence
        from sag_event_entity_links l
        join sag_events e on e.event_id = l.event_id
        {discourse_join}
        where l.entity_id in ({placeholders})
        {filter_sql}
        """,
        entity_ids + filter_params,
    ).fetchall()

    expanded = {}
    for event_id, doc_id, entity_id, confidence in rows:
        if event_id in seed_ids:
            continue
        entity_type, entity_value = entity_id_to_type_value[entity_id]
        info = expanded.setdefault(
            event_id,
            {"doc_id": doc_id, "matched_entities": {}, "confidence_sum": 0.0, "matched_space_count": 0},
        )
        info["matched_entities"].setdefault(entity_type, set()).add(entity_value)
        info["confidence_sum"] += float(confidence or 0.0)
        info["matched_space_count"] += 1

    sorted_items = sorted(expanded.items(), key=lambda item: (-item[1]["matched_space_count"], item[0]))
    return dict(sorted_items[:max_expanded])


def query_sag_db(db_path, config):
    """Return ordered pure SAG result dictionaries."""
    with _connect(db_path) as conn:
        seed_ids, seed_info = _seed_event_ids(conn, config)
        doc_by_event = _event_doc_map(conn, seed_ids)

        results = []
        for event_id in seed_ids:
            info = seed_info[event_id]
            matched_seed_count = sum(len(values) for values in info["matched_entities"].values())
            results.append(
                {
                    "doc_id": doc_by_event.get(event_id, ""),
                    "event_id": event_id,
                    "score": score_sag_result("seed_entity", matched_seed_count, 0, info["confidence_sum"]),
                    "match_stage": "seed_entity",
                    "matched_entities": info["matched_entities"],
                    "explanation": {"reason": "matched seed entities"},
                }
            )

        expansion = config.get("expansion") or {}
        if expansion.get("enabled", False) and int(expansion.get("max_hops", 1)) >= 1:
            frontier_types = expansion.get("frontier_entity_types") or []
            invalid_types = sorted(set(frontier_types) - ALLOWED_FRONTIER_ENTITY_TYPES)
            if invalid_types:
                raise ValueError("invalid_frontier_entity_types:" + ",".join(invalid_types))
            frontier_rows = _frontier_entities(conn, seed_ids, frontier_types)
            expanded = _expanded_events(conn, seed_ids, frontier_rows, config)
            for event_id, info in expanded.items():
                matched_entities = {key: sorted(value) for key, value in info["matched_entities"].items()}
                results.append(
                    {
                        "doc_id": info["doc_id"],
                        "event_id": event_id,
                        "score": score_sag_result(
                            "one_hop_expansion",
                            0,
                            info["matched_space_count"],
                            info["confidence_sum"],
                        ),
                        "match_stage": "one_hop_expansion",
                        "matched_entities": matched_entities,
                        "explanation": {"reason": "shared spatial entities with seed events"},
                    }
                )

    results = sorted(results, key=lambda result: (-result["score"], result["match_stage"], result["doc_id"]))
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def _counter_items(values):
    counter = Counter(value for value in values if value)
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _matched_doc_ids(results):
    return [result["doc_id"] for result in results if result.get("doc_id")]


def _rows_by_doc(conn, doc_ids):
    if not doc_ids:
        return {}
    placeholders = ", ".join(["?"] * len(doc_ids))
    rows = conn.execute(f"select * from source_orders where doc_id in ({placeholders})", doc_ids).fetchall()
    columns = [desc[0] for desc in conn.description]
    return {row[columns.index("doc_id")]: dict(zip(columns, row)) for row in rows}


def _entity_values_for_docs(conn, doc_ids, entity_type, source_channel=None):
    if not doc_ids:
        return []
    placeholders = ", ".join(["?"] * len(doc_ids))
    params = list(doc_ids) + [entity_type]
    channel_sql = ""
    if source_channel:
        channel_sql = " and source_channel = ?"
        params.append(source_channel)
    rows = conn.execute(
        f"""
        select entity_value
        from sag_event_entity_links
        where doc_id in ({placeholders})
          and entity_type = ?
          {channel_sql}
        """,
        params,
    ).fetchall()
    return [row[0] for row in rows]


def _statistics(conn, doc_ids, rows_by_doc):
    return {
        "by_month": _counter_items(row.get("call_month", "") for row in rows_by_doc.values()),
        "by_area_metadata": _counter_items(row.get("area_code_area", "") for row in rows_by_doc.values()),
        "by_street_metadata": _counter_items(row.get("area_code_street", "") for row in rows_by_doc.values()),
        "by_street_entity": _counter_items(_entity_values_for_docs(conn, doc_ids, "street")),
        "by_road_entity": _counter_items(_entity_values_for_docs(conn, doc_ids, "road")),
        "by_intersection_entity": _counter_items(_entity_values_for_docs(conn, doc_ids, "intersection")),
        "by_poi_entity": _counter_items(_entity_values_for_docs(conn, doc_ids, "poi")),
        "by_problem_object": _counter_items(_entity_values_for_docs(conn, doc_ids, "problem_object")),
        "by_problem_behavior": _counter_items(_entity_values_for_docs(conn, doc_ids, "problem_behavior")),
    }


def _entity_coverage(conn, doc_ids):
    total = len(set(doc_ids))
    coverage = {}
    if total == 0:
        return {
            entity_type: {"events_total": 0, "events_with_entity": 0, "coverage": 0.0, "source_breakdown": []}
            for entity_type in COVERAGE_ENTITY_TYPES
        }
    placeholders = ", ".join(["?"] * len(doc_ids))
    for entity_type in COVERAGE_ENTITY_TYPES:
        rows = conn.execute(
            f"""
            select source_channel, count(distinct doc_id)
            from sag_event_entity_links
            where doc_id in ({placeholders})
              and entity_type = ?
            group by source_channel
            """,
            list(doc_ids) + [entity_type],
        ).fetchall()
        events_with_entity = len(
            {
                row[0]
                for row in conn.execute(
                    f"""
                    select distinct doc_id
                    from sag_event_entity_links
                    where doc_id in ({placeholders})
                      and entity_type = ?
                    """,
                    list(doc_ids) + [entity_type],
                ).fetchall()
            }
        )
        coverage[entity_type] = {
            "events_total": total,
            "events_with_entity": events_with_entity,
            "coverage": round(events_with_entity / total, 6),
            "source_breakdown": [{"source_channel": row[0], "count": row[1]} for row in rows],
        }
    return coverage


def _metadata_recovery(conn, doc_ids, rows_by_doc):
    missing_docs = [doc_id for doc_id, row in rows_by_doc.items() if not str(row.get("area_code_street", "")).strip()]
    if not missing_docs:
        return {
            "metadata_street_missing": 0,
            "metadata_street_missing_but_text_street_found": 0,
            "metadata_street_missing_but_text_road_found": 0,
            "metadata_street_missing_but_text_intersection_found": 0,
            "metadata_street_missing_but_text_poi_found": 0,
            "recovery_rate": 0.0,
        }

    def count_text_docs(entity_type):
        placeholders = ", ".join(["?"] * len(missing_docs))
        rows = conn.execute(
            f"""
            select distinct doc_id
            from sag_event_entity_links
            where doc_id in ({placeholders})
              and entity_type = ?
              and source_channel in ('case_content', 'address_detail', 'title', 'case_goal')
            """,
            missing_docs + [entity_type],
        ).fetchall()
        return {row[0] for row in rows}

    street_docs = count_text_docs("street")
    road_docs = count_text_docs("road")
    intersection_docs = count_text_docs("intersection")
    poi_docs = count_text_docs("poi")
    recovered_docs = street_docs | road_docs | intersection_docs | poi_docs
    return {
        "metadata_street_missing": len(missing_docs),
        "metadata_street_missing_but_text_street_found": len(street_docs),
        "metadata_street_missing_but_text_road_found": len(road_docs),
        "metadata_street_missing_but_text_intersection_found": len(intersection_docs),
        "metadata_street_missing_but_text_poi_found": len(poi_docs),
        "recovery_rate": round(len(recovered_docs) / len(missing_docs), 6),
    }


def _representative_cases(results, rows_by_doc, limit):
    cases = []
    for result in results[:limit]:
        row = rows_by_doc.get(result["doc_id"], {})
        cases.append(
            {
                "rank": result["rank"],
                "doc_id": result["doc_id"],
                "score": result["score"],
                "match_stage": result["match_stage"],
                "case_content": row.get("case_content_clean", ""),
                "case_goal": row.get("case_goal_clean", ""),
                "metadata_area": row.get("area_code_area", ""),
                "metadata_street": row.get("area_code_street", ""),
                "matched_entities": result.get("matched_entities", {}),
                "explanation": result.get("explanation", {}),
            }
        )
    return cases


def _conflict_report(conn, doc_ids, rows_by_doc, limit=20):
    conflicts = []
    for doc_id, row in rows_by_doc.items():
        metadata_area = row.get("area_code_area", "")
        if not metadata_area:
            continue
        text_areas = set(
            _entity_values_for_docs(conn, [doc_id], "area", "case_content")
            + _entity_values_for_docs(conn, [doc_id], "area", "address_detail")
            + _entity_values_for_docs(conn, [doc_id], "area", "title")
        )
        different = sorted(area for area in text_areas if area and area != metadata_area)
        if different:
            conflicts.append({"doc_id": doc_id, "metadata_area": metadata_area, "text_area": different})
        if len(conflicts) >= limit:
            break
    return {"conflict_count": len(conflicts), "examples": conflicts}


def analyze_sag_query(db_path, config):
    """Run pure SAG-lite retrieval and return a full analysis report."""
    started_at = time.time()
    results = query_sag_db(db_path, config)
    doc_ids = _matched_doc_ids(results)

    with _connect(db_path) as conn:
        rows_by_doc = _rows_by_doc(conn, doc_ids)
        report = {
            "query": config,
            "matched_orders": len(doc_ids),
            "seed_orders": sum(1 for result in results if result["match_stage"] == "seed_entity"),
            "expanded_orders": sum(1 for result in results if result["match_stage"] == "one_hop_expansion"),
            "statistics": _statistics(conn, doc_ids, rows_by_doc),
            "entity_coverage": _entity_coverage(conn, doc_ids),
            "metadata_recovery": _metadata_recovery(conn, doc_ids, rows_by_doc),
            "conflict_report": _conflict_report(conn, doc_ids, rows_by_doc),
            "representative_cases": _representative_cases(
                results,
                rows_by_doc,
                int(config.get("representative_limit", 10)),
            ),
            "results": results,
            "retrieval": {
                "db_path": str(db_path),
                "elapsed_ms": round((time.time() - started_at) * 1000, 3),
                "mode": "pure_sag_lite",
            },
        }

    try:
        from ragflow_style_pipeline.sag_eval import evaluate_sag_results

        report["evaluation"] = evaluate_sag_results(db_path, config, results)
    except ModuleNotFoundError:
        report["evaluation"] = {}
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run pure SAG-lite work-order query.")
    parser.add_argument("--db", required=True, help="SAG-lite DuckDB database path.")
    parser.add_argument("--config", required=True, help="SAG query config JSON.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_sag_query_config(args.config)
    report = analyze_sag_query(args.db, config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
