"""Utilities for removing obvious personal identifiers from order text."""

from collections import Counter
import re


PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{18,}(?!\d)")


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
    text, id_card_count = ID_CARD_RE.subn("[身份证号]", text)
    text, long_number_count = LONG_NUMBER_RE.subn("[长数字编号]", text)
    counts["phone"] += phone_count
    counts["id_card"] += id_card_count
    counts["long_number"] += long_number_count
    return text, counts
