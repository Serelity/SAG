"""Prompt construction and deterministic long-text windowing for semantic extraction.

This module contains no model imports.  It only prepares already-desensitized work
orders for the server-side semantic extraction backend.
"""

import json


CURRENT_MARKERS = (
    "其不认可",
    "现服务对象表示",
    "现再次反映",
    "仍未解决",
    "再次要求",
    "希望部门",
    "现要求",
)
HISTORY_MARKERS = (
    "前期反映",
    "原工单",
    "处理结果",
    "部门答复",
    "答复如下",
)

_CLEAN_FIELDS = (
    "title_clean",
    "case_content_clean",
    "case_goal_clean",
    "address_detail_clean",
)
_METADATA_FIELDS = (
    "service_object_type",
    "area_code_city",
    "area_code_area",
    "area_code_street",
    "type1",
    "type2",
    "type3",
    "call_time",
    "call_month",
)

FINAL_JSON_SKELETON = {
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
}


def _text(value):
    return value if isinstance(value, str) else ""


def _max_input_chars(config):
    value = config.get("max_input_chars", 2200) if isinstance(config, dict) else 2200
    if type(value) is not int:
        return 2200
    return max(0, value)


def _last_marker(text, markers):
    latest_position = -1
    latest_marker = ""
    for marker in markers:
        position = text.rfind(marker)
        if position > latest_position:
            latest_position = position
            latest_marker = marker
    return latest_position, latest_marker


def _present_markers(text, markers):
    return [marker for marker in markers if marker in text]


def select_content_windows(text, max_chars):
    """Select bounded, non-overlapping source slices from a work-order body.

    A long body assigns approximately 30 percent of the budget to its head,
    40 percent to a window around the *last* current-claim marker, and 30
    percent to its tail.  The current window has priority where slices overlap,
    so the returned ``head + current_window + tail`` contains no duplicated
    source characters.  With no current marker, the unused middle budget is
    moved to the tail rather than treating a historical-response marker as a
    current claim.
    """
    text = _text(text)
    original_chars = len(text)
    if type(max_chars) is not int or max_chars <= 0:
        return {
            "head": "",
            "current_window": "",
            "tail": "",
            "combined": "",
            "truncated": original_chars > 0,
            "original_chars": original_chars,
            "kept_chars": 0,
        }

    if original_chars <= max_chars:
        return {
            "head": text,
            "current_window": "",
            "tail": "",
            "combined": text,
            "truncated": False,
            "original_chars": original_chars,
            "kept_chars": original_chars,
        }

    head_budget = max_chars * 3 // 10
    current_budget = max_chars * 4 // 10
    tail_budget = max_chars - head_budget - current_budget
    marker_position, marker = _last_marker(text, CURRENT_MARKERS)

    if marker_position < 0 or current_budget == 0:
        # Without explicit current-claim evidence, retain more of the tail but
        # do not relabel a historical marker or arbitrary tail text as current.
        tail_budget += current_budget
        head = text[:head_budget]
        tail = text[original_chars - tail_budget:] if tail_budget else ""
        combined = head + tail
        return {
            "head": head,
            "current_window": "",
            "tail": tail,
            "combined": combined[:max_chars],
            "truncated": True,
            "original_chars": original_chars,
            "kept_chars": min(len(combined), max_chars),
        }

    before_marker = current_budget // 3
    current_start = max(0, marker_position - before_marker)
    current_end = min(original_chars, current_start + current_budget)
    if current_end - current_start < current_budget:
        current_start = max(0, current_end - current_budget)

    # If the budget can hold it, ensure the complete marker remains inside the
    # selected current window after boundary adjustment.
    marker_end = marker_position + len(marker)
    if len(marker) <= current_budget and marker_end > current_end:
        current_end = min(original_chars, marker_end)
        current_start = max(0, current_end - current_budget)

    # Give the semantically important current window priority over edge slices.
    # This clips overlaps instead of repeating the same source characters.
    head_end = min(head_budget, current_start)
    tail_start = max(original_chars - tail_budget, current_end)
    head = text[:head_end]
    current_window = text[current_start:current_end]
    tail = text[tail_start:]
    combined = head + current_window + tail
    if len(combined) > max_chars:  # Defensive hard cap for all future changes.
        combined = combined[:max_chars]

    return {
        "head": head,
        "current_window": current_window,
        "tail": tail,
        "combined": combined,
        "truncated": True,
        "original_chars": original_chars,
        "kept_chars": len(combined),
    }


def _window_payload(order, max_chars):
    content = _text(order.get("case_content_clean")) if isinstance(order, dict) else ""
    windows = select_content_windows(content, max_chars)
    return {
        "head": windows["head"],
        "current_window": windows["current_window"],
        "tail": windows["tail"],
        "input_truncated": windows["truncated"],
        "original_chars": windows["original_chars"],
        "kept_chars": windows["kept_chars"],
        "history_markers_present": _present_markers(content, HISTORY_MARKERS),
        "current_markers_present": _present_markers(content, CURRENT_MARKERS),
    }


def _safe_metadata(order):
    metadata = order.get("metadata") if isinstance(order, dict) else None
    if not isinstance(metadata, dict):
        return {}
    return {
        field: metadata[field]
        for field in _METADATA_FIELDS
        if isinstance(metadata.get(field), (str, int, float, bool))
    }


def _semantic_payload(order, config, include_metadata=True):
    order = order if isinstance(order, dict) else {}
    payload = {
        "title_clean": _text(order.get("title_clean")),
        "case_content_windows": _window_payload(order, _max_input_chars(config)),
        "case_goal_clean": _text(order.get("case_goal_clean")),
        "address_detail_clean": _text(order.get("address_detail_clean")),
    }
    if include_metadata:
        payload["metadata_context"] = _safe_metadata(order)
    return payload


_RULES = """你是面向 SAG 检索的 12345 工单语义结构化器，不是普通关键词抽取器。请对任意业务领域做开放式识别。

字段语义：
- title_clean 是脱敏标题；case_content_windows 是诉求正文的有界窗口；case_goal_clean 是诉求人希望采取的动作；address_detail_clean 是地址补充。
- metadata_context 只作登记背景，不得覆盖正文当前事实、当前立场或推断意图。
- event_summary 应语义完整地概括当前核心事件，优先最新事实、立场和诉求，建议不超过 80 个汉字；咨询、建议、表扬不得强行制造问题。

实体边界：
- problem_objects 与 problem_behaviors 均为开放式识别，不受固定词表限制；只保留具有跨工单检索价值的具体概念。
- problem_objects 是问题、诉求或咨询所指向的领域对象；不要输出“相关部门、工作人员、事情、情况、问题”等泛词。
- problem_behaviors 是对象的问题行为、异常现象、状态或核心关系。要求处理、希望维修、请求清理、建议拆除、修剪等纯诉求动作不能作为 problem behavior。
- road 只接受具体命名道路；“马路边、道路、消防通道、小区北门口”不是 road。
- intersection 必须有明确路口表达或明确道路组合；poi 是具体小区、市场、学校、医院、商场、公园或机构。“港龙新港城北门口”应抽取 poi“港龙新港城”，不是 road。
- 空间 canonical 必须保守，不猜测原文未给出的全称。
- 每个实体项严格使用 {surface, canonical, field, evidence}；field 只能是 title_clean、case_content_clean、case_goal_clean、address_detail_clean；evidence 必须是该字段中的连续原文。

Discourse 边界：
- intents 最多 3 个，label 只能为：投诉、举报、求助、咨询、建议、表扬、催办、反馈、其他。
- emotions 最多 2 个，label 只能为：愤怒、不满、焦虑、无奈、悲伤、感谢、认可；intensity 只能为 1、2、3。没有诉求人直接情绪证据时输出空数组。
- 被投诉对象“态度恶劣”不等同诉求人表达愤怒。
- satisfaction label 只能为 satisfied、dissatisfied、mixed、unknown；非 unknown 必须有明确 target 和直接 evidence。
- 模板“谢谢”“感谢转交”“请优先处理”不能判定满意，“优先处理”也不能单独推断紧急。
- urgency level 只能为 normal、high、critical；明确催办、长期未解决或影响扩大可为 high，当前人身安全、火灾、燃气泄漏、坍塌等紧迫风险才可为 critical。
- 历史答复中的“已处理、已解决”不能覆盖当前“不认可、仍未解决”等最新立场。

判断顺序：先识别核心事件和历史/当前边界，再判断最新事实与要求、对象、问题行为、纯诉求动作、空间类型，最后只抽取有直接证据的 discourse，并复核 evidence 与检索价值。
数量上限：problem_objects 3，problem_behaviors 4，roads 4，intersections 2，pois 4，intents 3，emotions 2。上限不是最低数量；无可靠证据时使用空数组。
"""

_FEW_SHOTS = """六个跨领域边界示例：

示例 1（路灯故障；维修是诉求动作）
输入：case_content_clean=“和平路路灯连续三天不亮。”；case_goal_clean=“希望维修”
输出：{"event_summary":"市民反映和平路路灯连续三天不亮，希望维修","entities":{"problem_objects":[{"surface":"路灯","canonical":"路灯","field":"case_content_clean","evidence":"路灯"}],"problem_behaviors":[{"surface":"连续三天不亮","canonical":"照明故障","field":"case_content_clean","evidence":"连续三天不亮"}],"roads":[{"surface":"和平路","canonical":"和平路","field":"case_content_clean","evidence":"和平路"}],"intersections":[],"pois":[]},"discourse":{"intents":[{"label":"求助","evidence":"希望维修"}],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}}}

示例 2（小区北门是 POI，不是道路）
输入：case_content_clean=“港龙新港城北门口有电动车摆摊占道。”；case_goal_clean=“希望清理”
输出：{"event_summary":"市民反映港龙新港城北门有摊贩占道经营，希望清理","entities":{"problem_objects":[{"surface":"电动车摆摊","canonical":"流动摊贩","field":"case_content_clean","evidence":"电动车摆摊"}],"problem_behaviors":[{"surface":"摆摊占道","canonical":"占道经营","field":"case_content_clean","evidence":"摆摊占道"}],"roads":[],"intersections":[],"pois":[{"surface":"港龙新港城","canonical":"港龙新港城","field":"case_content_clean","evidence":"港龙新港城北门口"}]},"discourse":{"intents":[{"label":"求助","evidence":"希望清理"}],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}}}

示例 3（遮挡是问题；修剪是诉求动作）
输入：case_content_clean=“学校门前行道树树枝遮挡交通标志。”；case_goal_clean=“建议尽快修剪”
输出：{"event_summary":"市民反映学校门前行道树枝遮挡交通标志，建议修剪","entities":{"problem_objects":[{"surface":"行道树","canonical":"行道树","field":"case_content_clean","evidence":"行道树"},{"surface":"交通标志","canonical":"交通标志","field":"case_content_clean","evidence":"交通标志"}],"problem_behaviors":[{"surface":"树枝遮挡交通标志","canonical":"遮挡交通标志","field":"case_content_clean","evidence":"树枝遮挡交通标志"}],"roads":[],"intersections":[],"pois":[{"surface":"学校","canonical":"学校","field":"case_content_clean","evidence":"学校门前"}]},"discourse":{"intents":[{"label":"建议","evidence":"建议尽快修剪"}],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}}}

示例 4（礼貌致谢不代表满意）
输入：case_content_clean=“培训机构突然闭店，学费还没有退。”；case_goal_clean=“请优先处理退款，谢谢！”
输出：{"event_summary":"市民反映培训机构闭店且学费未退，请求处理退款","entities":{"problem_objects":[{"surface":"培训机构","canonical":"培训机构","field":"case_content_clean","evidence":"培训机构"},{"surface":"学费","canonical":"培训费","field":"case_content_clean","evidence":"学费"}],"problem_behaviors":[{"surface":"突然闭店","canonical":"机构闭店","field":"case_content_clean","evidence":"突然闭店"},{"surface":"还没有退","canonical":"退款未到账","field":"case_content_clean","evidence":"还没有退"}],"roads":[],"intersections":[],"pois":[]},"discourse":{"intents":[{"label":"求助","evidence":"请优先处理退款"}],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}}}

示例 5（对象态度不等于诉求人情绪）
输入：case_content_clean=“停车场收费员拒绝开票，且态度恶劣。”；case_goal_clean=“要求调查处理”
输出：{"event_summary":"市民反映停车场收费员拒绝开票且态度恶劣，要求调查处理","entities":{"problem_objects":[{"surface":"停车场收费员","canonical":"停车收费服务","field":"case_content_clean","evidence":"停车场收费员"},{"surface":"票","canonical":"收费票据","field":"case_content_clean","evidence":"开票"}],"problem_behaviors":[{"surface":"拒绝开票","canonical":"拒绝提供票据","field":"case_content_clean","evidence":"拒绝开票"},{"surface":"态度恶劣","canonical":"服务态度恶劣","field":"case_content_clean","evidence":"态度恶劣"}],"roads":[],"intersections":[],"pois":[]},"discourse":{"intents":[{"label":"投诉","evidence":"要求调查处理"}],"emotions":[],"satisfaction":{"label":"unknown","target":"","evidence":""},"urgency":{"level":"normal","evidence":""}}}

示例 6（历史答复不能覆盖当前立场）
输入：case_content_clean=“部门答复称车位已清理。现服务对象表示其不认可，通道仍被占用，再次要求处理。”
输出：{"event_summary":"市民不认可前次清理答复，反映通道仍被占用并再次要求处理","entities":{"problem_objects":[{"surface":"通道","canonical":"通道","field":"case_content_clean","evidence":"通道"}],"problem_behaviors":[{"surface":"仍被占用","canonical":"通道被占用","field":"case_content_clean","evidence":"仍被占用"}],"roads":[],"intersections":[],"pois":[]},"discourse":{"intents":[{"label":"催办","evidence":"再次要求处理"}],"emotions":[{"label":"不满","intensity":2,"evidence":"其不认可"}],"satisfaction":{"label":"dissatisfied","target":"前次处理答复","evidence":"其不认可"},"urgency":{"level":"high","evidence":"仍被占用，再次要求处理"}}}
"""


def build_semantic_prompt(order, config):
    """Build the primary extraction prompt from one normalized work order."""
    payload = _semantic_payload(order, config, include_metadata=True)
    skeleton = json.dumps(FINAL_JSON_SKELETON, ensure_ascii=False, indent=2)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        _RULES
        + "\n"
        + _FEW_SHOTS
        + "\n最终输出结构必须与以下 JSON 骨架完全一致；实体数组有值时才放入规定的实体项：\n"
        + skeleton
        + "\n\n当前脱敏工单 payload：\n"
        + payload_json
        + "\n\n请在内部完成上述判断，不要输出分析过程、推理步骤或思维链。只输出最终 JSON，不要输出 Markdown、解释或其他文字。"
    )


def build_repair_prompt(order, original_output, errors, config):
    """Build a constrained one-shot repair prompt without raw metadata fields."""
    clean_payload = _semantic_payload(order, config, include_metadata=False)
    clean_json = json.dumps(clean_payload, ensure_ascii=False, indent=2, sort_keys=True)
    error_codes = [error for error in errors if isinstance(error, str)] if isinstance(errors, list) else []
    errors_json = json.dumps(error_codes, ensure_ascii=False, indent=2)
    original_output = _text(original_output)
    skeleton = json.dumps(FINAL_JSON_SKELETON, ensure_ascii=False, indent=2)
    return f"""你是 JSON 结果修复器。下面仅提供修复所需的脱敏 clean fields、原始模型输出和验证器机器错误码。
只修复错误码指向的字段；其他正确字段保持原意不变。不得新增原文没有的事实，所有 evidence 必须是相应 clean field 的连续原文，实体项使用 {{surface, canonical, field, evidence}}。
输出必须符合给定 JSON 骨架，不要添加运行元数据。请在内部检查，但不要输出分析过程、推理步骤或思维链。

必要 clean fields：
{clean_json}

原始模型输出：
{original_output}

机器错误码：
{errors_json}

JSON 骨架：
{skeleton}

这是一次只修复任务。只输出最终 JSON，不要输出 Markdown、解释或其他文字。"""
