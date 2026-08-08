"""Deterministic PII redaction applied before text enters a prompt."""

from __future__ import annotations

from collections import Counter
import re


EMAIL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.+-])[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+"
    r"(?:\.[A-Za-z0-9-]+)+"
)
LABELED_LANDLINE_RE = re.compile(
    r"((?:电话|联系电话|联系方式|座机)\s*[:：]?\s*)0\d{2,3}[-－—\s]?\d{7,8}(?!\d)"
)
LABELED_QQ_RE = re.compile(r"((?:QQ|扣扣)\s*[:：]?\s*)[1-9]\d{4,11}(?!\d)", re.I)
LABELED_WECHAT_RE = re.compile(
    r"((?:微信号?|wechat)\s*[:：]?\s*)[A-Za-z][A-Za-z0-9_-]{5,19}", re.I
)
CONTACT_NAME_RE = re.compile(
    r"((?:联系人|联络人)(?:姓名)?\s*[:：]\s*)"
    r"(?!\[姓名\])([\u4e00-\u9fff·]{2,4})"
    r"(?=[，,。；;、\s]|\[手机号\]|\d|联系电话|联系|电话|手机|$)"
)
CONTACT_NAME_BEFORE_PHONE_RE = re.compile(
    r"((?:联系人|联络人)(?:姓名)?\s*)"
    r"(?![:：\s]|\[姓名\])([\u4e00-\u9fff·]{2,4})"
    r"(?=\s*[，,]?\s*(?:0?1[3-9]\d{9,10}|\[手机号\]))"
)
NAME_LABEL_RE = re.compile(r"((?:姓名|市民姓名|来电人姓名)\s*[:：]\s*)(?!\[姓名\])[\u4e00-\u9fff·]{2,8}")
PHONE_RE = re.compile(r"(?<!\d)0?1[3-9]\d{9,10}(?!\d)")
ALNUM_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,6}\d{10,}[A-Za-z0-9]*(?![A-Za-z0-9])")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{18,}(?!\d)")
NUMERIC_ID_RE = re.compile(r"(?<!\d)\d{13,17}(?!\d)")

PLACEHOLDERS = frozenset(
    {
        "[邮箱]",
        "[姓名]",
        "[座机]",
        "[QQ号]",
        "[微信号]",
        "[手机号]",
        "[业务编号]",
        "[身份证号]",
        "[长数字编号]",
        "[数字编号]",
    }
)
PLACEHOLDER_RE = re.compile(
    r"^(?:" + "|".join(re.escape(item) for item in sorted(PLACEHOLDERS)) + r")$"
)

# Ordered replacements are part of the redaction contract. More specific rules run first.
_RULES = (
    ("email", EMAIL_RE, "[邮箱]"),
    ("contact_name", CONTACT_NAME_RE, r"\1[姓名]"),
    ("contact_name", CONTACT_NAME_BEFORE_PHONE_RE, r"\1[姓名]"),
    ("name", NAME_LABEL_RE, r"\1[姓名]"),
    ("landline", LABELED_LANDLINE_RE, r"\1[座机]"),
    ("qq", LABELED_QQ_RE, r"\1[QQ号]"),
    ("wechat", LABELED_WECHAT_RE, r"\1[微信号]"),
    ("phone", PHONE_RE, "[手机号]"),
    ("alnum_id", ALNUM_ID_RE, "[业务编号]"),
    ("id_card", ID_CARD_RE, "[身份证号]"),
    ("long_number", LONG_NUMBER_RE, "[长数字编号]"),
    ("numeric_id", NUMERIC_ID_RE, "[数字编号]"),
)


def redact_text(value: object) -> tuple[str, Counter[str]]:
    """Return redacted text and aggregate replacement counts."""
    text = "" if value is None else str(value)
    counts: Counter[str] = Counter()
    for name, pattern, replacement in _RULES:
        text, count = pattern.subn(replacement, text)
        counts[name] += count
    return text, counts


def residual_pii_codes(value: object) -> tuple[str, ...]:
    """Return residual recognisable PII classes after redaction."""
    text = "" if value is None else str(value)
    codes = []
    for name, pattern, _replacement in _RULES:
        if pattern.search(text):
            codes.append(name)
    return tuple(dict.fromkeys(codes))
