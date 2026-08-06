"""Utilities for removing obvious personal identifiers from order text."""

from collections import Counter
import re


PHONE_RE = re.compile(r"(?<!\d)0?1[3-9]\d{9,10}(?!\d)")
EMAIL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.+-])[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+"
    r"(?:\.[A-Za-z0-9-]+)+"
)
LABELED_LANDLINE_RE = re.compile(
    r"((?:电话|联系电话|联系方式|座机)\s*[:：]?\s*)0\d{2,3}[-－—\s]?\d{7,8}(?!\d)"
)
LABELED_QQ_RE = re.compile(r"((?:QQ|扣扣)\s*[:：]?\s*)[1-9]\d{4,11}(?!\d)", re.I)
LABELED_WECHAT_RE = re.compile(
    r"((?:微信|微信号|wechat)\s*[:：]?\s*)[A-Za-z][A-Za-z0-9_-]{5,19}", re.I
)
CONTACT_NAME_RE = re.compile(
    r"((?:联系人|联络人)(?:姓名)?\s*[:：]\s*)"
    r"(?!\[姓名\])[\u4e00-\u9fff·]{2,4}"
    r"(?=[，,。；;、\s]|\[手机号\]|\d|联系电话|联系|电话|手机|$)"
)
CONTACT_NAME_BEFORE_PHONE_RE = re.compile(
    r"((?:联系人|联络人)(?:姓名)?\s*)"
    r"(?![:：\s]|\[姓名\])[\u4e00-\u9fff·]{2,4}"
    r"(?=\s*[，,]?\s*(?:0?1[3-9]\d{9,10}|\[手机号\]))"
)
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
    text, email_count = EMAIL_RE.subn("[邮箱]", text)
    text, contact_name_count = CONTACT_NAME_RE.subn(r"\1[姓名]", text)
    text, contact_phone_name_count = CONTACT_NAME_BEFORE_PHONE_RE.subn(r"\1[姓名]", text)
    text, landline_count = LABELED_LANDLINE_RE.subn(r"\1[座机]", text)
    text, qq_count = LABELED_QQ_RE.subn(r"\1[QQ号]", text)
    text, wechat_count = LABELED_WECHAT_RE.subn(r"\1[微信号]", text)
    text, phone_count = PHONE_RE.subn("[手机号]", text)
    text, alnum_id_count = ALNUM_ID_RE.subn("[业务编号]", text)
    text, id_card_count = ID_CARD_RE.subn("[身份证号]", text)
    text, long_number_count = LONG_NUMBER_RE.subn("[长数字编号]", text)
    text, numeric_id_count = NUMERIC_ID_RE.subn("[数字编号]", text)
    text, name_count = NAME_LABEL_RE.subn(r"\1[姓名]", text)
    counts["email"] += email_count
    counts["contact_name"] += contact_name_count + contact_phone_name_count
    counts["landline"] += landline_count
    counts["qq"] += qq_count
    counts["wechat"] += wechat_count
    counts["phone"] += phone_count
    counts["alnum_id"] += alnum_id_count
    counts["id_card"] += id_card_count
    counts["long_number"] += long_number_count
    counts["numeric_id"] += numeric_id_count
    counts["name"] += name_count
    return text, counts
