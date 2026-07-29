"""Pure SAG-lite entity extraction for 12345 work orders."""

import re
from dataclasses import dataclass


NULLISH_VALUES = {"", "NULL", "null", "None", "none", "\\N"}

AREAS = [
    "钟楼区",
    "天宁区",
    "新北区",
    "武进区",
    "金坛区",
    "溧阳市",
    "常州市经济开发区",
    "经开区",
    "市本级",
]

PROBLEM_OBJECT_TERMS = [
    "流动摊贩",
    "游商摊贩",
    "夜市摊贩",
    "摊贩",
    "小摊",
    "商贩",
]

PROBLEM_BEHAVIOR_TERMS = [
    "占道经营",
    "无照经营",
    "店外经营",
    "影响通行",
    "摆摊",
    "设摊",
    "扰民",
    "油烟",
]

ROAD_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{1,12}(?:路|街|大道|巷|弄|桥|线)")
STREET_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{1,12}(?:街道|镇)")
INTERSECTION_PATTERN = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{1,12}(?:路|街|大道|巷|弄|桥|线))"
    r"(?:和|与|及|、)"
    r"([\u4e00-\u9fffA-Za-z0-9]{1,12}(?:路|街|大道|巷|弄|桥|线))"
    r"(?:交叉口|交界处|路口)"
)
POI_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{1,16}(?:小区|市场|学校|广场|商场|夜市|公园|医院|菜场|地铁站)"
)


@dataclass(frozen=True)
class SagEntityLink:
    doc_id: str
    entity_type: str
    entity_value: str
    normalized_value: str
    source_field: str
    source_channel: str
    confidence: float
    matched_text: str


def clean_value(value):
    """Normalize one source value into a plain string."""
    if value is None:
        return ""
    value = str(value).strip()
    if value in NULLISH_VALUES:
        return ""
    return value


def normalize_entity_value(value):
    """Normalize entity text for dictionary identity."""
    return re.sub(r"\s+", "", clean_value(value))


def deduplicate_entity_links(links):
    """Deduplicate links by document, type, normalized value, and source field."""
    seen = set()
    deduped = []
    for link in links:
        key = (link.doc_id, link.entity_type, link.normalized_value, link.source_field)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)
    return deduped


def _make_link(doc_id, entity_type, value, source_field, source_channel, confidence, matched_text=None):
    value = clean_value(value)
    normalized_value = normalize_entity_value(value)
    if not doc_id or not normalized_value:
        return None
    return SagEntityLink(
        doc_id=doc_id,
        entity_type=entity_type,
        entity_value=value,
        normalized_value=normalized_value,
        source_field=source_field,
        source_channel=source_channel,
        confidence=float(confidence),
        matched_text=clean_value(matched_text if matched_text is not None else value),
    )


def _append_link(links, doc_id, entity_type, value, source_field, source_channel, confidence, matched_text=None):
    link = _make_link(doc_id, entity_type, value, source_field, source_channel, confidence, matched_text)
    if link is not None:
        links.append(link)


def _extract_known_terms(doc_id, text, source_field, source_channel, terms, entity_type, confidence):
    links = []
    for term in terms:
        if term in text:
            _append_link(links, doc_id, entity_type, term, source_field, source_channel, confidence, term)
    return links


def _strip_after_known_area(value):
    value = clean_value(value)
    for area in AREAS:
        if area in value:
            value = value.split(area)[-1]
    return value


def _clean_street_name(value):
    value = _strip_after_known_area(value)
    if "街道" in value:
        return value.rsplit("街道", 1)[0][-8:] + "街道"
    if "镇" in value:
        return value.rsplit("镇", 1)[0][-8:] + "镇"
    return value


def _clean_road_name(value):
    value = _strip_after_known_area(value)
    for marker in ["街道", "镇"]:
        if marker in value:
            value = value.split(marker)[-1]
    for connector in ["和", "与", "及", "、"]:
        if connector in value:
            value = value.split(connector)[-1]
    return value


def _intersection_value(match):
    first_road = _clean_road_name(match.group(1))
    second_road = _clean_road_name(match.group(2))
    matched = match.group(0)
    connector = "与" if "与" in matched else "和"
    suffix = "交界处" if "交界处" in matched else "路口" if "路口" in matched else "交叉口"
    return f"{first_road}{connector}{second_road}{suffix}"


def _extract_text_entities(doc_id, text, source_field, source_channel):
    links = []
    text = clean_value(text)
    if not text:
        return links

    links.extend(_extract_known_terms(doc_id, text, source_field, source_channel, AREAS, "area", 0.9))
    links.extend(
        _extract_known_terms(
            doc_id,
            text,
            source_field,
            source_channel,
            PROBLEM_OBJECT_TERMS,
            "problem_object",
            0.7,
        )
    )
    links.extend(
        _extract_known_terms(
            doc_id,
            text,
            source_field,
            source_channel,
            PROBLEM_BEHAVIOR_TERMS,
            "problem_behavior",
            0.7,
        )
    )

    for match in INTERSECTION_PATTERN.finditer(text):
        intersection = _intersection_value(match)
        _append_link(links, doc_id, "intersection", intersection, source_field, source_channel, 0.9, match.group(0))
        _append_link(links, doc_id, "road", _clean_road_name(match.group(1)), source_field, source_channel, 0.9, match.group(1))
        _append_link(links, doc_id, "road", _clean_road_name(match.group(2)), source_field, source_channel, 0.9, match.group(2))
    for match in STREET_PATTERN.finditer(text):
        _append_link(links, doc_id, "street", _clean_street_name(match.group(0)), source_field, source_channel, 0.9, match.group(0))
    for match in ROAD_PATTERN.finditer(text):
        _append_link(links, doc_id, "road", _clean_road_name(match.group(0)), source_field, source_channel, 0.9, match.group(0))
    for match in POI_PATTERN.finditer(text):
        _append_link(links, doc_id, "poi", match.group(0), source_field, source_channel, 0.6)

    return links


def extract_entities_from_order(order):
    """Return deduplicated SAG entity links for one normalized source order."""
    doc_id = clean_value(order.get("doc_id"))
    links = []

    for source_field, entity_type in [
        ("call_month", "time_month"),
        ("area_code_area", "area"),
        ("area_code_street", "street"),
        ("case_lnglat", "lnglat"),
        ("type1", "case_type"),
        ("type2", "case_type"),
        ("type3", "case_type"),
        ("type4", "case_type"),
        ("type5", "case_type"),
        ("belong_dept", "department"),
        ("deptName", "department"),
        ("orgName", "department"),
    ]:
        _append_link(
            links,
            doc_id,
            entity_type,
            order.get(source_field),
            source_field,
            "metadata",
            1.0,
        )

    for source_field, source_channel in [
        ("case_content_clean", "case_content"),
        ("address_detail_clean", "address_detail"),
        ("title_clean", "title"),
        ("case_goal_clean", "case_goal"),
    ]:
        links.extend(_extract_text_entities(doc_id, order.get(source_field), source_field, source_channel))

    return deduplicate_entity_links(links)
