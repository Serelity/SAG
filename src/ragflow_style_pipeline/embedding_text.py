"""Build semantic text for embedding models."""

EMBEDDING_PREFIXES = ("诉求内容：", "诉求目标：")


def embedding_text(document):
    """Return case-content-first text for dense embedding."""
    explicit_text = str(document.get("embedding_text", "")).strip()
    if explicit_text:
        return explicit_text

    text = str(document.get("text") or document.get("display_text") or "")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith(EMBEDDING_PREFIXES)
    ]
    if lines:
        return "\n".join(lines)
    return text.strip()
