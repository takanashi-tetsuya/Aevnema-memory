"""Bounded Source evidence views used by answer verification."""

from __future__ import annotations

import re


_EVIDENCE_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]{2,}"
)


def source_excerpt(
    raw_text: str,
    episode_text: str,
    participants: list[str],
    max_chars: int,
) -> str:
    """Select nearby complete records without changing persisted Source text."""

    if max_chars <= 0 or len(raw_text) <= max_chars:
        return raw_text
    keywords = {
        value.strip()
        for value in participants
        if len(value.strip()) >= 2 and value.strip() not in {"???", "[USERNAME]"}
    }
    keywords.update(_EVIDENCE_TOKEN_RE.findall(episode_text))
    blocks = raw_text.split("\n\n")
    scored = [
        (sum(1 for keyword in keywords if keyword in block), index)
        for index, block in enumerate(blocks)
    ]
    relevant = [index for score, index in scored if score > 0]
    if not relevant:
        half = max(1, (max_chars - 40) // 2)
        return (
            raw_text[:half]
            + "\n\n[...Source 中段省略...]\n\n"
            + raw_text[-half:]
        )
    ranked = sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)
    selected: set[int] = {0}
    for score, index in ranked:
        if score <= 0 or len(selected) >= 12:
            break
        selected.update(
            neighbor
            for neighbor in (index - 1, index, index + 1)
            if 0 <= neighbor < len(blocks)
        )
    output: list[str] = []
    used = 0
    previous: int | None = None
    for index in sorted(selected):
        block = blocks[index]
        separator = (
            "\n\n[...省略若干记录...]\n\n"
            if previous is not None and index > previous + 1
            else "\n\n"
        )
        addition = (separator if output else "") + block
        if output and used + len(addition) > max_chars:
            continue
        output.append(addition)
        used += len(addition)
        previous = index
    return "".join(output)[:max_chars]
