"""Small tokenizer for local keyword retrieval.

The first local-search demo intentionally avoids external dependencies.
Chinese text is represented by character bigrams, while ASCII words and
numbers are represented by lowercased contiguous tokens.
"""

import re


_ASCII_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")


def _is_chinese_character(character):
    codepoint = ord(character)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2A6DF
    )


def _flush_chinese_segment(segment, tokens):
    if not segment:
        return
    if len(segment) == 1:
        tokens.append("".join(segment))
        return
    for index in range(len(segment) - 1):
        tokens.append("".join(segment[index : index + 2]))


def tokenize(text):
    """Tokenize mixed Chinese/ASCII text for simple local retrieval."""
    if not text or not str(text).strip():
        return []

    tokens = []
    chinese_segment = []
    ascii_segment = []

    def flush_ascii_segment():
        if not ascii_segment:
            return
        segment = "".join(ascii_segment).lower()
        tokens.extend(_ASCII_TOKEN_RE.findall(segment))
        ascii_segment.clear()

    def flush_chinese_segment():
        _flush_chinese_segment(chinese_segment, tokens)
        chinese_segment.clear()

    for character in str(text):
        if _is_chinese_character(character):
            flush_ascii_segment()
            chinese_segment.append(character)
        elif character.isascii() and character.isalnum():
            flush_chinese_segment()
            ascii_segment.append(character)
        else:
            flush_chinese_segment()
            flush_ascii_segment()

    flush_chinese_segment()
    flush_ascii_segment()
    return tokens
