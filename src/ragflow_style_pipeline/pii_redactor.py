"""Utilities for removing obvious personal identifiers from order text."""

from collections import Counter
import re


PHONE_RE = re.compile(r"(?<!\d)0?1[3-9]\d{9,10}(?!\d)")
ALNUM_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{1,6}\d{10,}[A-Za-z0-9]*(?![A-Za-z0-9])")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{18,}(?!\d)")
NUMERIC_ID_RE = re.compile(r"(?<!\d)\d{13,17}(?!\d)")
NAME_LABEL_RE = re.compile(r"(姓名[:：]\s*)[\u4e00-\u9fff·]{2,8}")


def redact_text(value):
    """Return text with obvious phone and ID-card patterns replaced.

    The function returns a pair:
    - redacted text
    - replacement counts, grouped by sensitive type
    """
    if value is None:
        value = ""

    counts = Counter()
    text = str(value)
    text, phone_count = PHONE_RE.subn("[手机号]", text)
    text, alnum_id_count = ALNUM_ID_RE.subn("[业务编号]", text)
    text, id_card_count = ID_CARD_RE.subn("[身份证号]", text)
    text, long_number_count = LONG_NUMBER_RE.subn("[长数字编号]", text)
    text, numeric_id_count = NUMERIC_ID_RE.subn("[数字编号]", text)
    text, name_count = NAME_LABEL_RE.subn(r"\1[姓名]", text)
    counts["phone"] += phone_count
    counts["alnum_id"] += alnum_id_count
    counts["id_card"] += id_card_count
    counts["long_number"] += long_number_count
    counts["numeric_id"] += numeric_id_count
    counts["name"] += name_count
    return text, counts
