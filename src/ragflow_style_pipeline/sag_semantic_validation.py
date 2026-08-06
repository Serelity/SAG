"""Deterministic quality gates for normalized work-order semantics."""

from copy import deepcopy
import re

from ragflow_style_pipeline.sag_semantic_schema import (
    ENTITY_GROUPS,
    GROUP_LIMITS,
    SOURCE_FIELDS,
    SOURCE_FIELD_ORDER,
)

STATUSES = {"accepted", "accepted_with_warnings", "repair_required", "rejected"}
REPAIR_PREFIXES = (
    "json_parse_failed", "empty_event_summary", "missing_evidence:", "invalid_source_field:",
    "surface_evidence_mismatch:", "empty_canonical:", "generic_entity:",
    "duplicate_entity:", "road_poi_conflict:", "intersection_shape_conflict:",
    "poi_shape_conflict:", "request_action_as_behavior:",
    "normal_service_action_as_behavior:", "canonical_evidence_conflict:",
    "satisfaction_missing_target_or_evidence", "invalid_satisfaction_target",
    "unsupported_satisfaction_evidence", "template_politeness_as_satisfaction",
    "object_attitude_as_emotion:", "unsupported_emotion_evidence:",
    "urgency_missing_evidence", "template_priority_as_urgency",
    "possible_history_contamination",
)
REJECT_PREFIXES = ("missing_doc_id", "empty_semantic_text", "repair_failed")
_GENERIC = {"问题", "情况", "事情", "相关部门", "工作人员", "道路", "马路边", "小区", "地点"}
_REQUEST_ACTIONS = {
    "处理", "清理", "维修", "修剪", "拆除", "解决", "调查", "协调", "整改",
    "回复", "答复", "退款", "退费", "销课", "注销", "恢复", "核实", "查处",
    "处罚", "赔偿", "公示", "告知", "办理", "申请", "咨询", "查询", "注册",
    "转入", "就读", "了解",
}
_NORMAL_SERVICE_ACTIONS = (
    "咨询", "查询", "办理", "注册", "申请", "转入", "就读", "了解",
)
_NORMAL_SERVICE_PREFIXES = (
    "希望", "要求", "请求", "想要", "想", "需要", "咨询", "如何", "怎么", "是否",
)
_REQUEST_MARKERS = (
    "希望", "要求", "请求", "建议", "请", "督促", "协调",
    "何时", "如何", "怎么", "是否", "需要哪些", "需要什么",
)
_FACT_ACTION_PREFIXES = (
    "未", "没有", "尚未", "仍未", "一直未", "至今未", "还没有",
    "拒绝", "不予", "无法", "不能",
)
_FACT_ACTION_SUFFIXES = ("未完成", "未到账", "失败", "受阻", "未通过", "不通过", "被拒")
_POI_GAP_HINTS = (
    "小区", "新村", "花园", "学校", "幼儿园", "医院", "商场", "广场",
    "服务中心", "产业园", "工业园", "园区", "派出所", "超市", "体育馆", "学堂",
)
_POI_HINTS = (
    "小区", "新村", "花园", "家园", "公园", "学校", "幼儿园", "医院", "市场",
    "商场", "广场", "服务中心", "有限公司", "公司", "工厂", "汽修厂", "店", "馆",
    "产业园", "工业园", "园区", "机构", "工作室", "派出所", "超市", "大厦",
    "华苑", "社区", "学堂", "体育馆", "北门", "南门", "东门", "西门",
)
_ROAD_SUFFIXES = ("路", "街", "大道", "巷", "弄", "线")
_ROAD_CONFLICT_HINTS = (
    "小区", "新村", "花园", "家园", "公园", "学校", "医院", "市场", "商场", "广场",
    "北门", "南门", "东门", "西门",
)
_POI_SUFFIXES = (
    "苑", "城", "府", "村", "园", "中心", "公司", "厂", "店", "馆", "市场", "商场",
    "广场", "学校", "医院", "机构", "工作室", "派出所", "超市", "大厦", "社区", "学堂",
)
_ADMIN_DIVISION = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9]{1,24}(?:省|市|区|县|镇|乡|街道)$")
_NON_ADMIN_PLACE_SUFFIXES = ("小区", "园区", "景区", "校区", "社区", "厂区")
_BARE_ROAD_ADDRESS = re.compile(
    r"^[\u4e00-\u9fffA-Za-z0-9]{1,24}(?:大道|公路|路|街|巷|弄)\d+(?:号|弄|栋|幢|单元)?$"
)
_ROAD_NAME = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{1,16}?(?:大道|公路|道路|路|街|巷|弄|线)")
_ROAD_WITH_QUALIFIER = re.compile(
    r"^(?P<road>[\u4e00-\u9fffA-Za-z0-9]{1,16}?(?:大道|公路|路|街|巷|弄|线))"
    r"(?:方向|路段|沿线|附近|\d+(?:号|弄|栋|幢|单元)?)$"
)
_INTERSECTION_CONNECTOR = re.compile(r"(?:与|和|及|/|、).{0,8}?(?:[\u4e00-\u9fffA-Za-z0-9]{1,16}?(?:大道|公路|道路|路|街|巷|弄|线))")
_HISTORY_MARKERS = ("部门答复", "处理结果", "前期反映", "原工单", "答复如下")
_CURRENT_MARKERS = ("其不认可", "仍未解决", "现服务对象表示", "再次要求", "现再次反映")
_INVALID_SATISFACTION_TARGETS = {"服务对象", "诉求人", "市民", "群众", "本人", "自己", "其"}
_INTENT_TRIGGERS = {
    "投诉": ("投诉", "维权", "要求查处", "要求处罚", "请求查处", "请求处罚"),
    "举报": ("举报",),
    "求助": ("希望相关部门", "要求相关部门", "请求", "需要帮助", "希望", "要求"),
    "咨询": (
        "咨询", "如何", "怎么", "是否", "如何办理", "怎么办理", "办理流程",
        "需要准备什么", "需要哪些材料", "查询方式", "不清楚", "是否需要",
        "需向哪些部门", "核实是否",
    ),
    "建议": ("建议",),
    "表扬": ("表扬",),
    "催办": ("催办", "再次要求", "仍未解决"),
    "反馈": ("反馈",),
}
_LITERAL_INTENT_TRIGGERS = {
    "投诉": ("投诉", "维权", "要求查处", "要求处罚", "请求查处", "请求处罚"),
    "举报": ("举报",),
    "咨询": ("咨询",),
    "建议": ("建议",),
    "表扬": ("表扬",),
    "催办": ("催办",),
}
_REQUESTER_MARKERS = (
    "服务对象", "诉求人", "市民", "群众", "本人", "自己", "业主", "来电人", "反映人",
)
_REQUESTER_EMOTION_BRIDGE = re.compile(
    r"^[，,。；;：:\s]*(?:对此|表示|感到|认为|觉得|很|非常|十分|相当|表示自己|对此表示)?[，,\s]*$"
)
_OBJECT_ATTITUDE_MARKERS = (
    "工作人员态度恶劣", "商家态度恶劣", "客服态度恶劣", "服务态度恶劣",
    "态度恶劣", "服务态度差",
)
_EMOTION_TRIGGERS = {
    "愤怒": (("非常气愤", 3), ("十分气愤", 3), ("气愤", 2), ("愤怒", 2), ("生气", 2)),
    "不满": (("非常不满", 3), ("很不满意", 3), ("不满意", 2), ("其不认可", 2), ("不认可", 2)),
    "焦虑": (("非常着急", 3), ("十分着急", 3), ("焦虑", 2), ("着急", 2), ("担心", 1)),
    "无奈": (("非常无奈", 3), ("无奈", 2)),
    "悲伤": (("非常难过", 3), ("难过", 2), ("悲伤", 2)),
    "感谢": (("非常感谢", 3), ("感谢", 2), ("谢谢", 1)),
    "认可": (("非常认可", 3), ("认可", 2)),
}
_SATISFACTION_TRIGGERS = {
    "satisfied": ("满意", "认可处理结果", "认可答复"),
    "dissatisfied": (
        "不是很满意", "不怎么满意", "不太满意", "满意度不高", "谈不上满意",
        "并不满意", "不满意", "其不认可", "并不认可", "不认可", "未认可",
        "没有认可", "不能认可", "无法认可", "有异议", "不接受答复",
    ),
}
_NEGATED_RECOGNITION = ("其不认可", "并不认可", "不认可", "未认可", "没有认可", "不能认可", "无法认可")
_UNSUPPORTED_CANONICAL_STATES = {
    "过期": ("过期",),
    "变质": ("变质", "腐败"),
    "伪造": ("伪造", "作伪"),
    "假冒": ("假冒", "冒充"),
    "未整改": ("未整改", "没有整改", "未有任何整改", "尚未整改", "至今未有任何整改"),
    "未退款": ("未退款", "没有退款", "未退", "没有退", "尚未退", "未到账"),
    "未处理": ("未处理", "没有处理", "未予处理", "尚未处理", "无人处理", "不处理", "仍无处理"),
    "泄漏": ("泄漏", "漏气", "漏水"),
    "坍塌": ("坍塌", "倒塌"),
}
_ANOMALY_MARKERS = (
    "盗用", "冒名", "拖欠", "破损", "损坏", "毁坏", "占用", "堵塞", "污染", "违法",
    "违规", "不立案", "不解答", "未处理", "未解决", "拒绝", "泄漏", "坍塌", "体罚",
    "诈骗", "无人", "未取得", "随地小便", "摆摊", "不清理", "不通过", "被拒",
    "三无", "态度不好",
)


def _text(value):
    return value if isinstance(value, str) else ""


def _add(items, value):
    if value and value not in items:
        items.append(value)


def _source_field(item):
    return _text(item.get("source_field") or item.get("field"))


def _repair_field(warning):
    if ":" in warning:
        suffix = warning.split(":", 1)[1]
        if suffix.startswith("entities.") or suffix.startswith("discourse."):
            return suffix
    if warning.startswith("satisfaction_") or "satisfaction" in warning:
        return "discourse.satisfaction"
    if "urgency" in warning:
        return "discourse.urgency"
    if warning in {"empty_event_summary", "possible_history_contamination"}:
        return "event_summary"
    return ""


def _contains(source, evidence):
    return bool(evidence and evidence in source)


def _evidence_field(order, evidence):
    evidence = _text(evidence)
    if not evidence:
        return ""
    for field in SOURCE_FIELD_ORDER:
        if evidence in _text(order.get(field)):
            return field
    return ""


def _intent_fallback(order, label, literal_only=False):
    triggers = _LITERAL_INTENT_TRIGGERS.get(label, ()) if literal_only else _INTENT_TRIGGERS.get(label, ())
    for trigger in triggers:
        if _evidence_field(order, trigger):
            return trigger
    return ""


def _intent_evidence_supported(order, label, evidence):
    del order
    if not evidence:
        return False
    if label == "其他":
        return True
    return any(trigger in evidence for trigger in _INTENT_TRIGGERS.get(label, ()))


def _primary_intent_fallback(order):
    for label in ("投诉", "举报", "催办", "建议", "咨询", "表扬"):
        evidence = _intent_fallback(order, label, literal_only=True)
        if evidence:
            return label, evidence
    for label in ("咨询", "催办", "求助"):
        evidence = _intent_fallback(order, label)
        if evidence:
            return label, evidence
    return "", ""


def _requester_owns_emotion(source, position, evidence):
    if evidence == "其不认可":
        return True
    for marker in _REQUESTER_MARKERS:
        marker_start = source.rfind(marker, max(0, position - 18), position)
        if marker_start < 0:
            continue
        bridge = source[marker_start + len(marker):position]
        if len(bridge) <= 8 and _REQUESTER_EMOTION_BRIDGE.fullmatch(bridge):
            return True
    return False


def _emotion_fallback(order, label):
    for evidence, intensity in _EMOTION_TRIGGERS.get(label, ()):
        for field in SOURCE_FIELD_ORDER:
            source = _text(order.get(field))
            search_from = 0
            while evidence and (position := source.find(evidence, search_from)) >= 0:
                if label == "认可" and any(
                    source.startswith(negated, position - len(negated) + len(evidence))
                    for negated in _NEGATED_RECOGNITION
                    if len(negated) >= len(evidence) and position >= len(negated) - len(evidence)
                ):
                    search_from = position + len(evidence)
                    continue
                if _requester_owns_emotion(source, position, evidence):
                    return evidence, intensity
                search_from = position + len(evidence)
    return "", 1


def _emotion_evidence_supported(order, label, evidence):
    if not evidence:
        return False
    triggers = _EMOTION_TRIGGERS.get(label, ())
    for trigger, _intensity in triggers:
        position_in_evidence = evidence.find(trigger)
        if position_in_evidence < 0:
            continue
        for field in SOURCE_FIELD_ORDER:
            source = _text(order.get(field))
            evidence_start = source.find(evidence)
            if evidence_start < 0:
                continue
            position = evidence_start + position_in_evidence
            if label == "认可" and any(negated in evidence for negated in _NEGATED_RECOGNITION):
                continue
            if _requester_owns_emotion(source, position, trigger):
                return True
    return False


def _satisfaction_fallback(order, label):
    if label not in _SATISFACTION_TRIGGERS:
        return ""
    for field in SOURCE_FIELD_ORDER:
        source = _text(order.get(field))
        positive, negative = _satisfaction_polarities(source)
        supported = (
            (label == "satisfied" and positive and not negative)
            or (label == "dissatisfied" and negative and not positive)
        )
        if not supported:
            continue
        for trigger in _SATISFACTION_TRIGGERS[label]:
            if trigger in source:
                return trigger
    return ""


def _satisfaction_polarities(evidence):
    negative_tokens = _SATISFACTION_TRIGGERS["dissatisfied"]
    negative = any(token in evidence for token in negative_tokens)
    positive_text = evidence
    for token in sorted(negative_tokens, key=len, reverse=True):
        positive_text = positive_text.replace(token, "")
    positive = any(
        token in positive_text for token in _SATISFACTION_TRIGGERS["satisfied"]
    )
    return positive, negative


def _satisfaction_evidence_supported(label, evidence):
    if label == "unknown":
        return True
    positive, negative = _satisfaction_polarities(evidence)
    if label == "mixed":
        return positive and negative
    if label == "satisfied":
        return positive and not negative
    if label == "dissatisfied":
        return negative and not positive
    return False


def _road_names(text):
    return {
        road.lstrip("与和及")
        for road in _ROAD_NAME.findall(_text(text))
        if road.lstrip("与和及")
    }


def _is_named_road(text):
    roads = _road_names(text)
    return (
        text.endswith(_ROAD_SUFFIXES)
        and len(roads) == 1
        and not any(hint in text for hint in _ROAD_CONFLICT_HINTS)
        and text not in {"马路", "道路", "公路"}
    )


def _recover_road_name(item):
    if not isinstance(item, dict):
        return ""
    for value in (_text(item.get("canonical")), _text(item.get("surface"))):
        match = _ROAD_WITH_QUALIFIER.fullmatch(value.strip())
        road = match.group("road") if match else ""
        if road and _is_named_road(road):
            return road
    return ""


def _is_invalid_poi_shape(text):
    compact = _text(text).strip()
    is_admin = bool(_ADMIN_DIVISION.fullmatch(compact)) and not compact.endswith(
        _NON_ADMIN_PLACE_SUFFIXES
    )
    return bool(compact and (is_admin or _BARE_ROAD_ADDRESS.fullmatch(compact)))


def _recoverable_string_shape(group, text):
    if group == "roads":
        return _is_named_road(text)
    if group == "intersections":
        return _is_strict_intersection(text)
    if group == "pois":
        return (
            not _is_invalid_poi_shape(text)
            and (any(hint in text for hint in _POI_HINTS) or text.endswith(_POI_SUFFIXES))
        )
    return True


def _intersection_span(evidence, first_road, second_road):
    suffix = r"(?:交叉口|路口|交叉处|交界处|交汇处)"
    connector = r"(?:与|和|及|/|、)"
    for left, right in ((first_road, second_road), (second_road, first_road)):
        patterns = (
            re.escape(left) + r"\s*" + connector + r"\s*" + re.escape(right) + suffix,
            re.escape(left) + r"\s*" + re.escape(right) + suffix,
        )
        for pattern in patterns:
            match = re.search(pattern, evidence)
            if match and _is_strict_intersection(match.group(0)):
                return match.group(0)
    return ""


def _synthesize_intersection(order, entities, actions):
    intersections = entities.get("intersections")
    roads = entities.get("roads")
    if not isinstance(intersections, list) or not isinstance(roads, list):
        return
    existing = {
        (_source_field(item), _text(item.get("evidence")))
        for item in intersections if isinstance(item, dict)
    }
    for left_index, left in enumerate(roads):
        if not isinstance(left, dict) or not _is_named_road(_text(left.get("canonical"))):
            continue
        for right in roads[left_index + 1:]:
            if not isinstance(right, dict) or not _is_named_road(_text(right.get("canonical"))):
                continue
            field = _source_field(left)
            source = _text(order.get(field))
            left_evidence = _text(left.get("evidence"))
            right_evidence = _text(right.get("evidence"))
            left_road = _text(left.get("canonical"))
            right_road = _text(right.get("canonical"))
            if (
                field not in SOURCE_FIELDS or field != _source_field(right)
                or not _contains(source, left_evidence)
                or not _contains(source, right_evidence)
                or left_road not in (_text(left.get("surface")) + left_evidence)
                or right_road not in (_text(right.get("surface")) + right_evidence)
            ):
                continue
            span = _intersection_span(source, left_road, right_road)
            if not span or (field, span) in existing:
                continue
            intersections.append({
                "surface": span,
                "canonical": span,
                "source_field": field,
                "evidence": span,
            })
            existing.add((field, span))
            _add(actions, f"synthesized_intersection:entities.intersections.{len(intersections) - 1}")
            if len(intersections) >= GROUP_LIMITS["intersections"]:
                return


def _deduplicate_entity_variants(entities, actions):
    for group in ENTITY_GROUPS:
        items = entities.get(group)
        if not isinstance(items, list):
            continue
        kept = []
        positions = {}
        for original_index, item in enumerate(items):
            if not isinstance(item, dict):
                kept.append(item)
                continue
            key = (
                _source_field(item), _text(item.get("evidence")), _text(item.get("surface"))
            )
            canonical = _text(item.get("canonical"))
            previous_position = positions.get(key)
            if previous_position is None:
                positions[key] = len(kept)
                kept.append(item)
                continue
            previous = kept[previous_position]
            previous_canonical = _text(previous.get("canonical"))
            variants_overlap = (
                canonical == _text(item.get("surface"))
                or previous_canonical == _text(previous.get("surface"))
                or canonical in previous_canonical
                or previous_canonical in canonical
            )
            if not variants_overlap:
                kept.append(item)
                continue
            preferred = min(
                (previous, item),
                key=lambda candidate: (
                    _text(candidate.get("canonical")) == _text(candidate.get("surface")),
                    len(_text(candidate.get("canonical"))),
                ),
            )
            kept[previous_position] = preferred
            _add(actions, f"deduplicated_entity_variant:entities.{group}.{original_index}")
        entities[group] = kept


def enrich_semantic_output(order, semantic, parse_warnings=None):
    """Recover only candidates directly provable from desensitized clean fields."""
    order = order if isinstance(order, dict) else {}
    cleaned = deepcopy(semantic if isinstance(semantic, dict) else {})
    entities = cleaned.get("entities") if isinstance(cleaned.get("entities"), dict) else {}
    discourse = cleaned.get("discourse") if isinstance(cleaned.get("discourse"), dict) else {}
    actions = []
    coerced = {}
    for warning in parse_warnings or []:
        if not isinstance(warning, str) or not warning.startswith("coerced_entity_string:"):
            continue
        parts = warning.split(":")
        if len(parts) != 3:
            continue
        try:
            coerced.setdefault(parts[1], set()).add(int(parts[2]))
        except ValueError:
            pass

    for group, indexes in coerced.items():
        items = entities.get(group)
        if not isinstance(items, list):
            continue
        for index in sorted(indexes):
            if not (0 <= index < len(items)) or not isinstance(items[index], dict):
                continue
            evidence = _text(items[index].get("evidence"))
            field = _evidence_field(order, evidence)
            if field and _recoverable_string_shape(group, evidence):
                items[index]["source_field"] = field
                _add(actions, f"recovered_entity_string:entities.{group}.{index}")

    roads = entities.get("roads")
    if isinstance(roads, list):
        for index, item in enumerate(roads):
            if not isinstance(item, dict) or _is_named_road(_text(item.get("canonical"))):
                continue
            field = _source_field(item)
            evidence = _text(item.get("evidence"))
            road = _recover_road_name(item)
            if field in SOURCE_FIELDS and _contains(_text(order.get(field)), evidence) and road:
                item["surface"] = road
                item["canonical"] = road
                _add(actions, f"recovered_named_road:entities.roads.{index}")

    raw_intents = discourse.get("intents") if isinstance(discourse.get("intents"), list) else []
    intents = []
    for index, item in enumerate(raw_intents):
        if not isinstance(item, dict):
            continue
        candidate = deepcopy(item)
        label = _text(candidate.get("label"))
        evidence = _text(candidate.get("evidence"))
        if not _evidence_field(order, evidence) or not _intent_evidence_supported(order, label, evidence):
            fallback = _intent_fallback(order, label)
            if fallback:
                candidate["evidence"] = fallback
                evidence = fallback
                _add(actions, f"recovered_intent_evidence:discourse.intents.{index}")
            else:
                _add(actions, f"dropped_unsupported_intent:discourse.intents.{index}")
                continue
        if not any(
            _text(existing.get("label")) == label
            for existing in intents if isinstance(existing, dict)
        ):
            intents.append(candidate)

    explicit_label, explicit_evidence = "", ""
    for label in ("投诉", "举报", "催办", "建议", "咨询", "表扬"):
        evidence = _intent_fallback(order, label, literal_only=True)
        if evidence:
            explicit_label, explicit_evidence = label, evidence
            break
    weak_evidence = {"希望", "要求", "请求", "希望相关部门", "要求相关部门"}
    if explicit_label:
        without_weak_help = [
            item for item in intents
            if not (
                _text(item.get("label")) == "求助"
                and _text(item.get("evidence")) in weak_evidence
            )
        ]
        if len(without_weak_help) != len(intents):
            intents = without_weak_help
            _add(actions, f"replaced_weak_intent:{explicit_label}")
        if not any(_text(item.get("label")) == explicit_label for item in intents):
            intents.insert(0, {"label": explicit_label, "evidence": explicit_evidence})
            _add(actions, f"recovered_explicit_intent:{explicit_label}")
        intents = intents[:3]
    elif not intents:
        label, evidence = _primary_intent_fallback(order)
        if label:
            intents.append({"label": label, "evidence": evidence})
            _add(actions, f"recovered_explicit_intent:{label}")

    raw_emotions = discourse.get("emotions") if isinstance(discourse.get("emotions"), list) else []
    emotions = []
    for index, item in enumerate(raw_emotions):
        if not isinstance(item, dict):
            continue
        candidate = deepcopy(item)
        label = _text(candidate.get("label"))
        evidence = _text(candidate.get("evidence"))
        if not _evidence_field(order, evidence) or not _emotion_evidence_supported(order, label, evidence):
            fallback, intensity = _emotion_fallback(order, label)
            if fallback:
                candidate["evidence"] = fallback
                candidate["intensity"] = intensity
                _add(actions, f"recovered_emotion_evidence:discourse.emotions.{index}")
            else:
                _add(actions, f"dropped_unsupported_emotion:discourse.emotions.{index}")
                continue
        emotions.append(candidate)
    if not emotions:
        for label in ("愤怒", "不满", "焦虑", "无奈", "悲伤"):
            evidence, intensity = _emotion_fallback(order, label)
            if evidence:
                emotions.append({"label": label, "intensity": intensity, "evidence": evidence})
                _add(actions, f"recovered_explicit_emotion:{label}")
                break

    satisfaction = discourse.get("satisfaction") if isinstance(discourse.get("satisfaction"), dict) else {}
    satisfaction_label = _text(satisfaction.get("label")) or "unknown"
    satisfaction_evidence = _text(satisfaction.get("evidence"))
    if (
        satisfaction_label != "unknown"
        and not _satisfaction_evidence_supported(satisfaction_label, satisfaction_evidence)
    ):
        fallback = _satisfaction_fallback(order, satisfaction_label)
        if fallback:
            satisfaction["evidence"] = fallback
            _add(actions, "recovered_satisfaction_evidence")

    urgency = discourse.get("urgency") if isinstance(discourse.get("urgency"), dict) else {}
    if (_text(urgency.get("level")) or "normal") == "normal" and _text(urgency.get("evidence")):
        discourse["urgency"] = {"level": "normal", "evidence": ""}
        _add(actions, "cleared_normal_urgency_evidence")

    _synthesize_intersection(order, entities, actions)
    _deduplicate_entity_variants(entities, actions)
    cleaned["entities"] = entities
    discourse["intents"] = intents
    discourse["emotions"] = emotions
    cleaned["discourse"] = discourse
    return cleaned, actions


def _action_has_failure(evidence, actions):
    if any(
        prefix + action in evidence
        for prefix in _FACT_ACTION_PREFIXES
        for action in actions
    ):
        return True
    return any(
        re.search(
            re.escape(action) + r"[^，,。；;]{0,6}?" + re.escape(suffix),
            evidence,
        )
        for action in actions
        for suffix in _FACT_ACTION_SUFFIXES
    )


def _is_normal_service_action_behavior(evidence, surface, canonical):
    comparison = " ".join((evidence, surface, canonical))
    actions = [action for action in _NORMAL_SERVICE_ACTIONS if action in comparison]
    if not actions:
        return False
    has_failure = _action_has_failure(evidence, actions)
    if has_failure:
        return False
    stripped = (canonical or surface or evidence).strip()
    starts_as_service = stripped.startswith(_NORMAL_SERVICE_ACTIONS)
    question_or_request = any(
        evidence.strip().startswith(prefix) for prefix in _NORMAL_SERVICE_PREFIXES
    ) and any(action in evidence for action in actions)
    return starts_as_service or question_or_request


def _is_request_only_behavior(order, field, evidence, surface, canonical):
    goal = _text(order.get("case_goal_clean"))
    text = evidence or surface or canonical
    goal_only = field == "case_goal_clean" or bool(text and text in goal)
    has_request = goal_only or any(marker in text for marker in _REQUEST_MARKERS)
    actions = [action for action in _REQUEST_ACTIONS if action in canonical]
    has_action = bool(actions)
    # Trust factual negation only when it is present in verified evidence.
    # A hallucinated canonical such as “未退款” must not turn a pure
    # “希望退款” request into an observed failure.
    has_factual_action = _action_has_failure(evidence, actions)
    return bool(has_request and has_action and not has_factual_action)


def _is_strict_intersection(text):
    roads = {
        road.lstrip("与和及")
        for road in _ROAD_NAME.findall(text)
        if road.lstrip("与和及")
    }
    relationship = bool(_INTERSECTION_CONNECTOR.search(text)) or any(
        marker in text for marker in ("路口", "交叉口", "交界处", "交汇处")
    )
    return len(roads) >= 2 and relationship


def _canonical_target_conflicts(canonical, comparison):
    # Relation prefixes are too generic to prove the target.  For example,
    # “严重影响采光” cannot justify canonical “严重影响通风” merely because
    # both strings share “严重影响”.  Keep this intentionally small; arbitrary
    # open-domain synonym canonicalization is still left to the model.
    for prefix in ("严重影响", "影响", "妨碍", "遮挡", "堵塞", "占用", "损坏", "破坏", "污染", "拖欠"):
        if canonical.startswith(prefix):
            target = canonical[len(prefix):].strip()
            if len(target) >= 2 and target not in comparison:
                return True
    for state, supports in _UNSUPPORTED_CANONICAL_STATES.items():
        if state in canonical and not any(support in comparison for support in supports):
            return True
    return False


def sanitize_semantic_output(semantic, warnings, order=None):
    """Conservatively remove invalid optional candidates without losing an event.

    A bad optional entity or discourse attribute should not reject an otherwise
    useful work order.  Parse failures and history contamination are not
    sanitized because they require a model repair or rejection.
    """
    order = order if isinstance(order, dict) else {}
    cleaned = deepcopy(semantic if isinstance(semantic, dict) else {})
    entities = cleaned.get("entities") if isinstance(cleaned.get("entities"), dict) else {}
    discourse = cleaned.get("discourse") if isinstance(cleaned.get("discourse"), dict) else {}
    drop_entities = {}
    drop_discourse = {"intents": set(), "emotions": set()}
    reset_satisfaction = False
    reset_urgency = False
    sanitation_warnings = []
    align_surfaces = {}
    align_canonicals = {}
    entity_drop_codes = {
        "invalid_source_field", "missing_evidence", "empty_canonical",
        "generic_entity", "duplicate_entity", "road_poi_conflict",
        "intersection_shape_conflict", "poi_shape_conflict",
        "request_action_as_behavior", "normal_service_action_as_behavior",
    }

    for warning in warnings or []:
        if not isinstance(warning, str) or ":" not in warning:
            if warning in {
                "satisfaction_missing_target_or_evidence",
                "invalid_satisfaction_target",
                "unsupported_satisfaction_evidence",
                "template_politeness_as_satisfaction",
            }:
                reset_satisfaction = True
            elif warning in {"urgency_missing_evidence", "template_priority_as_urgency"}:
                reset_urgency = True
            continue
        code, path = warning.split(":", 1)
        parts = path.split(".")
        if code in entity_drop_codes and len(parts) == 3 and parts[0] == "entities":
            try:
                index = int(parts[2])
            except ValueError:
                continue
            items = entities.get(parts[1])
            item = items[index] if isinstance(items, list) and 0 <= index < len(items) else None
            evidence = _text(item.get("evidence")) if isinstance(item, dict) else ""
            if code == "invalid_source_field":
                field = _evidence_field(order, evidence)
                if field and _recoverable_string_shape(parts[1], evidence):
                    item["source_field"] = field
                    _add(sanitation_warnings, f"recovered_entity_source:entities.{parts[1]}.{index}")
                    continue
            if code == "missing_evidence" and isinstance(item, dict):
                field = _source_field(item)
                if field in SOURCE_FIELDS and _contains(_text(order.get(field)), evidence):
                    continue
            drop_entities.setdefault(parts[1], set()).add(index)
        elif code == "surface_evidence_mismatch" and len(parts) == 3 and parts[0] == "entities":
            try:
                align_surfaces.setdefault(parts[1], set()).add(int(parts[2]))
            except ValueError:
                pass
        elif code == "canonical_evidence_conflict" and len(parts) == 3 and parts[0] == "entities":
            try:
                align_canonicals.setdefault(parts[1], set()).add(int(parts[2]))
            except ValueError:
                pass
        elif code in {"object_attitude_as_emotion", "unsupported_emotion_evidence"} and len(parts) == 3 and parts[:2] == ["discourse", "emotions"]:
            try:
                drop_discourse["emotions"].add(int(parts[2]))
            except ValueError:
                pass
        elif code == "missing_evidence" and len(parts) == 3 and parts[0] == "discourse":
            if parts[1] in drop_discourse:
                try:
                    index = int(parts[2])
                except ValueError:
                    continue
                items = discourse.get(parts[1])
                item = items[index] if isinstance(items, list) and 0 <= index < len(items) else None
                if parts[1] == "intents" and isinstance(item, dict):
                    fallback = _intent_fallback(order, _text(item.get("label")))
                    if fallback:
                        item["evidence"] = fallback
                        _add(sanitation_warnings, f"recovered_intent_evidence:discourse.intents.{index}")
                        continue
                drop_discourse[parts[1]].add(index)
        elif code == "missing_evidence" and path == "discourse.satisfaction":
            reset_satisfaction = True
        elif code == "missing_evidence" and path == "discourse.urgency":
            reset_urgency = True

    for group, indexes in align_surfaces.items():
        items = entities.get(group)
        dropped = drop_entities.get(group, set())
        if not isinstance(items, list):
            continue
        for index in sorted(indexes):
            if index in dropped or not (0 <= index < len(items)) or not isinstance(items[index], dict):
                continue
            evidence = _text(items[index].get("evidence"))
            if evidence:
                items[index]["surface"] = evidence
                _add(sanitation_warnings, f"aligned_surface_to_evidence:entities.{group}.{index}")

    for group, indexes in align_canonicals.items():
        items = entities.get(group)
        dropped = drop_entities.get(group, set())
        if not isinstance(items, list):
            continue
        for index in sorted(indexes):
            if index in dropped or not (0 <= index < len(items)) or not isinstance(items[index], dict):
                continue
            surface = _text(items[index].get("surface")) or _text(items[index].get("evidence"))
            if surface:
                items[index]["canonical"] = surface
                _add(sanitation_warnings, f"aligned_canonical_to_surface:entities.{group}.{index}")

    for group, indexes in drop_entities.items():
        items = entities.get(group)
        if not isinstance(items, list):
            continue
        for index in sorted(indexes, reverse=True):
            if 0 <= index < len(items):
                del items[index]
                _add(sanitation_warnings, f"dropped_invalid_candidate:entities.{group}.{index}")

    for group, indexes in drop_discourse.items():
        items = discourse.get(group)
        if not isinstance(items, list):
            continue
        for index in sorted(indexes, reverse=True):
            if 0 <= index < len(items):
                del items[index]
                _add(sanitation_warnings, f"dropped_unverified_evidence:discourse.{group}.{index}")

    _deduplicate_entity_variants(entities, sanitation_warnings)
    for group, limit in GROUP_LIMITS.items():
        items = entities.get(group)
        if isinstance(items, list) and len(items) > limit:
            del items[limit:]
            _add(sanitation_warnings, f"truncated_group_to_limit:{group}")

    if reset_satisfaction:
        discourse["satisfaction"] = {"label": "unknown", "target": "", "evidence": ""}
        _add(sanitation_warnings, "reset_unverified_satisfaction")
    if reset_urgency:
        discourse["urgency"] = {"level": "normal", "evidence": ""}
        _add(sanitation_warnings, "reset_unverified_urgency")
    cleaned["entities"] = entities
    cleaned["discourse"] = discourse
    return cleaned, sanitation_warnings


def validate_semantic_output(order, semantic, parse_warnings=None):
    """Validate normalized semantics and return stable status/warnings/repair paths."""
    order = order if isinstance(order, dict) else {}
    semantic = semantic if isinstance(semantic, dict) else {}
    warnings = []
    for warning in parse_warnings or []:
        _add(warnings, _text(warning))

    if not _text(order.get("doc_id")) and "doc_id" in order:
        _add(warnings, "missing_doc_id")
    if not any(_text(order.get(field)) for field in SOURCE_FIELDS):
        _add(warnings, "empty_semantic_text")
    if not _text(semantic.get("event_summary")):
        _add(warnings, "empty_event_summary")

    entities = semantic.get("entities") if isinstance(semantic.get("entities"), dict) else {}
    seen = set()
    for group in ENTITY_GROUPS:
        items = entities.get(group, [])
        if not isinstance(items, list):
            _add(warnings, f"malformed_entity_group:{group}")
            continue
        if len(items) > GROUP_LIMITS[group]:
            _add(warnings, f"group_limit_exceeded:{group}")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                _add(warnings, f"malformed_entity_item:{group}:{index}")
                continue
            path = f"entities.{group}.{index}"
            field = _source_field(item)
            evidence = _text(item.get("evidence"))
            surface = _text(item.get("surface"))
            canonical = _text(item.get("canonical"))
            if field not in SOURCE_FIELDS:
                _add(warnings, f"invalid_source_field:{path}")
                source = ""
            else:
                source = _text(order.get(field))
            if not _contains(source, evidence):
                _add(warnings, f"missing_evidence:{path}")
            if surface and evidence and surface not in evidence and evidence not in surface:
                _add(warnings, f"surface_evidence_mismatch:{path}")
            if not canonical:
                _add(warnings, f"empty_canonical:{path}")
            if canonical in _GENERIC or surface in _GENERIC:
                _add(warnings, f"generic_entity:{path}")
            key = (group, canonical, evidence)
            if key in seen:
                _add(warnings, f"duplicate_entity:{path}")
            seen.add(key)
            if group == "roads":
                # Evidence may legitimately be a longer address containing both
                # a road and a POI.  Type the candidate from surface/canonical;
                # otherwise “下城路” would be lost merely because its evidence
                # also says “路劲城小区”.
                if not _is_named_road(canonical):
                    _add(warnings, f"road_poi_conflict:{path}")
            if group == "intersections":
                text = evidence or surface or canonical
                if not _is_strict_intersection(text):
                    _add(warnings, f"intersection_shape_conflict:{path}")
            if group == "pois" and (
                _is_invalid_poi_shape(surface) or _is_invalid_poi_shape(canonical)
            ):
                _add(warnings, f"poi_shape_conflict:{path}")
            if group == "problem_behaviors":
                if _is_request_only_behavior(order, field, evidence, surface, canonical):
                    _add(warnings, f"request_action_as_behavior:{path}")
                elif _is_normal_service_action_behavior(evidence, surface, canonical):
                    _add(warnings, f"normal_service_action_as_behavior:{path}")
            comparison = surface + evidence
            no_character_support = canonical and len(canonical) > 2 and not any(
                ch in comparison for ch in canonical
            )
            if canonical and evidence and (
                no_character_support or _canonical_target_conflicts(canonical, comparison)
            ):
                _add(warnings, f"canonical_evidence_conflict:{path}")

    discourse = semantic.get("discourse") if isinstance(semantic.get("discourse"), dict) else {}
    for group in ("intents", "emotions"):
        for index, item in enumerate(discourse.get(group, []) if isinstance(discourse.get(group), list) else []):
            if not isinstance(item, dict):
                continue
            evidence = _text(item.get("evidence"))
            if not evidence or not any(evidence in _text(order.get(field)) for field in SOURCE_FIELDS):
                _add(warnings, f"missing_evidence:discourse.{group}.{index}")
            if group == "intents" and not _intent_evidence_supported(
                order, _text(item.get("label")), evidence
            ):
                _add(warnings, f"unsupported_intent_evidence:discourse.intents.{index}")
            if group == "emotions":
                if any(marker in evidence for marker in _OBJECT_ATTITUDE_MARKERS):
                    _add(warnings, f"object_attitude_as_emotion:discourse.emotions.{index}")
                elif not _emotion_evidence_supported(order, _text(item.get("label")), evidence):
                    _add(warnings, f"unsupported_emotion_evidence:discourse.emotions.{index}")

    satisfaction = discourse.get("satisfaction") if isinstance(discourse.get("satisfaction"), dict) else {}
    sat_label = _text(satisfaction.get("label")) or "unknown"
    sat_target = _text(satisfaction.get("target"))
    sat_evidence = _text(satisfaction.get("evidence"))
    if sat_label != "unknown" and (not sat_target or not sat_evidence):
        _add(warnings, "satisfaction_missing_target_or_evidence")
    if sat_label != "unknown" and sat_target in _INVALID_SATISFACTION_TARGETS:
        _add(warnings, "invalid_satisfaction_target")
    if sat_label != "unknown" and sat_evidence in {"谢谢", "感谢", "感谢转交", "请优先处理，谢谢"}:
        _add(warnings, "template_politeness_as_satisfaction")
    if sat_label != "unknown" and not _satisfaction_evidence_supported(sat_label, sat_evidence):
        _add(warnings, "unsupported_satisfaction_evidence")
    if sat_evidence and not any(sat_evidence in _text(order.get(field)) for field in SOURCE_FIELDS):
        _add(warnings, "missing_evidence:discourse.satisfaction")

    urgency = discourse.get("urgency") if isinstance(discourse.get("urgency"), dict) else {}
    urgency_level = _text(urgency.get("level")) or "normal"
    urgency_evidence = _text(urgency.get("evidence"))
    if urgency_level != "normal" and not urgency_evidence:
        _add(warnings, "urgency_missing_evidence")
    if urgency_evidence and not any(urgency_evidence in _text(order.get(field)) for field in SOURCE_FIELDS):
        _add(warnings, "missing_evidence:discourse.urgency")
    if urgency_level != "normal" and urgency_evidence in {"优先处理", "请优先处理"}:
        _add(warnings, "template_priority_as_urgency")

    content = _text(order.get("case_content_clean"))
    summary = _text(semantic.get("event_summary"))
    searchable = "".join(_text(order.get(field)) for field in SOURCE_FIELD_ORDER)
    has_anomaly = any(marker in searchable for marker in _ANOMALY_MARKERS)
    if has_anomaly and not entities.get("problem_objects"):
        _add(warnings, "semantic_gap:problem_objects")
    if has_anomaly and not entities.get("problem_behaviors"):
        _add(warnings, "semantic_gap:problem_behaviors")
    if any(hint in searchable for hint in _POI_GAP_HINTS) and not entities.get("pois"):
        _add(warnings, "semantic_gap:pois")
    if any(marker in content for marker in _HISTORY_MARKERS) and any(marker in content for marker in _CURRENT_MARKERS):
        if any(token in summary for token in ("已处理", "已解决")) and not any(token in summary for token in ("不认可", "仍未", "再次")):
            _add(warnings, "possible_history_contamination")

    if any(warning.startswith(REJECT_PREFIXES) for warning in warnings):
        status = "rejected"
    elif any(warning.startswith(REPAIR_PREFIXES) for warning in warnings):
        status = "repair_required"
    elif warnings:
        status = "accepted_with_warnings"
    else:
        status = "accepted"
    repair_fields = []
    for warning in warnings:
        field = _repair_field(warning)
        _add(repair_fields, field)
    return {"status": status, "warnings": warnings, "repair_fields": repair_fields}
