"""Corpus-agnostic source ordering and timeline cohort helpers."""

from __future__ import annotations

from pathlib import Path
import re


def infer_timeline_scope(source_key: str) -> str:
    """Use the source's containing folders as its default chronology cohort."""

    parts = Path(source_key).parts
    parent_parts = parts[:-1]
    if not parent_parts:
        return "root"
    return ":".join(str(value).casefold() for value in parent_parts)


def natural_path_sort_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    """Sort numbered corpus files in human order without extra dependencies."""

    tokens = re.split(r"(\d+)", path.as_posix().casefold())
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token)
        for token in tokens
        if token
    )
