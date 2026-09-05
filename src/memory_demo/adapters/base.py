from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from memory_demo.types import (
    NormalizedBlock,
    normalize_epistemic_status,
    normalize_evidence_origin,
)


class InputAdapter(ABC):
    @abstractmethod
    def supports(self, path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_blocks(self, path: Path) -> list[NormalizedBlock]:
        raise NotImplementedError


def read_text_document(path: Path) -> str:
    """Decode common knowledge-file encodings without silent replacement."""

    payload = path.read_bytes()
    if not payload:
        return ""
    if payload.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        encodings = ("utf-32",)
    elif payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    elif payload.startswith(b"\xef\xbb\xbf"):
        encodings = ("utf-8-sig",)
    else:
        encodings = ("utf-8", "gb18030")
    errors: list[str] = []
    for encoding in encodings:
        try:
            text = payload.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        invalid_controls = sorted(
            {ord(character) for character in text if ord(character) < 32}
            - {9, 10, 13}
        )
        if invalid_controls:
            rendered = ", ".join(f"0x{value:02x}" for value in invalid_controls)
            errors.append(
                f"{encoding}: decoded text contains binary control bytes: {rendered}"
            )
            continue
        return text
    raise UnicodeError(
        f"cannot decode {path} as any supported text encoding: " + "; ".join(errors)
    )


def split_text_chunks(
    text: str, *, max_chars: int = 2_000, minimum_boundary: int = 1_000
) -> list[str]:
    """Bound a single logical record while preferring sentence boundaries."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    remaining = str(text).strip()
    while len(remaining) > max_chars:
        boundary = max(
            remaining.rfind(mark, 0, max_chars)
            for mark in ("。", "！", "？", ".", "!", "?", "\n")
        )
        if boundary < min(minimum_boundary, max_chars):
            boundary = max_chars
        else:
            boundary += 1
        chunk = remaining[:boundary].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def normalize_memory_evidence(value: Any) -> dict[str, Any]:
    """Normalize an explicit importer annotation without inventing defaults."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    if "evidence_origin" in value or "origin" in value:
        result["evidence_origin"] = normalize_evidence_origin(
            value.get("evidence_origin", value.get("origin")), "unknown"
        )
    if "epistemic_status" in value or "status" in value:
        result["epistemic_status"] = normalize_epistemic_status(
            value.get("epistemic_status", value.get("status")), "unknown"
        )
    if "generation" in value:
        try:
            result["evidence_generation"] = max(0, int(value["generation"]))
        except (TypeError, ValueError):
            result["evidence_generation"] = 0
    note = value.get("epistemic_note", value.get("note"))
    if note is not None and str(note).strip():
        result["epistemic_note"] = " ".join(str(note).split())
    return result


def logical_source_key(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = path.name
    category = root.name.casefold()
    if category in {"main", "favor", "event"}:
        return f"{category}/{relative}"
    return relative
