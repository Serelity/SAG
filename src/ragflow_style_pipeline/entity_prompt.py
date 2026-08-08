"""Small, single-version prompts for literal issue-oriented extraction."""

from __future__ import annotations

import hashlib

from .constants import CLEAN_FIELDS, CLEAN_FIELD_LABELS, PROMPT_SCHEMA_VERSION


SYSTEM_INSTRUCTION = """你是12345工单实体抽取器。只输出一个JSON对象：
{"issues":[{"objects":[],"problems":[],"questions":[],"locations":[],"requests":[]}]}
规则：
1. issues中的每项都必须且只能含上述五个字符串数组。
2. 每个字符串必须逐字出现在输入中；不得改写、概括、补全、归一化或推断。
3. objects=现实对象或服务对象；problems=问题、故障或异常现象；questions=咨询事项；locations=文本明确提到的地点；requests=希望办理、处理或提供的动作。
4. 一个现实业务关注点一个issue。只有合并后会让无关对象、问题、地点或诉求产生原文没有的关系时才拆issue；不要仅因problem/question/request话语角色不同而拆分。
5. 纯咨询放questions，不要伪造成problem。历史答复、历史处置、情绪、满意度和紧急程度不抽取为当前实体。
6. 不确定候选直接省略。无有效实体时输出{"issues":[]}。
禁止输出field、evidence、offset、canonical、confidence、id、metadata、summary、discourse、Markdown或解释。"""
PRIMARY_PREFIX = "\n输入：\n"
REPAIR_PREFIX = "\n上次输出无法解析或没有任何可逐字定位实体。请重新输出严格JSON，宁可省略不确定项。\n输入：\n"
INPUT_BUDGET_ALGORITHM = "fair_shared_unicode_character_budget_v1"


def _allocate_lengths(values: list[str], budget: int) -> list[int]:
    """Give every non-empty field a fair deterministic share, then use spare budget."""
    lengths = [0] * len(values)
    active = [index for index, value in enumerate(values) if value]
    remaining = max(0, budget)
    while active and remaining:
        share = max(1, remaining // len(active))
        next_active = []
        for index in active:
            available = len(values[index]) - lengths[index]
            take = min(available, share, remaining)
            lengths[index] += take
            remaining -= take
            if lengths[index] < len(values[index]):
                next_active.append(index)
            if remaining == 0:
                break
        active = next_active
    return lengths


def model_view(document: dict, max_input_chars: int) -> str:
    if type(max_input_chars) is not int or max_input_chars <= 0:
        raise ValueError("invalid_max_input_chars")
    prefixes = [CLEAN_FIELD_LABELS[field] + "：" for field in CLEAN_FIELDS]
    values = [str(document.get(field, "")) for field in CLEAN_FIELDS]
    nonempty = [index for index, value in enumerate(values) if value]
    prefix_budget = sum(len(prefixes[index]) + 1 for index in nonempty)
    text_budget = max(0, max_input_chars - prefix_budget)
    lengths = _allocate_lengths(values, text_budget)
    return "\n".join(
        prefixes[index] + values[index][: lengths[index]]
        for index in nonempty
        if lengths[index] > 0
    )


def primary_prompt(document: dict, max_input_chars: int) -> str:
    return SYSTEM_INSTRUCTION + PRIMARY_PREFIX + model_view(document, max_input_chars)


def repair_prompt(document: dict, max_input_chars: int) -> str:
    return SYSTEM_INSTRUCTION + REPAIR_PREFIX + model_view(document, max_input_chars)


def prompt_fingerprint() -> str:
    payload = "\0".join(
        (
            PROMPT_SCHEMA_VERSION,
            SYSTEM_INSTRUCTION,
            PRIMARY_PREFIX,
            REPAIR_PREFIX,
            INPUT_BUDGET_ALGORITHM,
        )
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
