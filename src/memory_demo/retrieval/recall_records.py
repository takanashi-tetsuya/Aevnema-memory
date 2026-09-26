"""Reversible, language-selected views of Source records.

No text is translated, normalized, or stripped of inline markup.  Model-facing
record IDs select locally held evidence; they are not themselves evidence of a
claim.  Callers still own semantic review and Source/Episode database binding.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re


_RECORD = re.compile(r"(?m)^\[record:[ \t]*(\d+)\][ \t]*\r?$")
_LANGUAGE = re.compile(r"(?m)^([a-z]{2,3}(?:-[A-Za-z0-9]+)*|unknown):[ \t]*")
_LEGEND = re.compile(r"(?m)^\[speaker_alias_legend\][ \t]*\r?$")


class RecordReferenceError(ValueError):
    """A reference is invisible, stale, ambiguous, or outside its raw span."""


@dataclass(frozen=True)
class RecordProjection:
    records: list[dict]
    consumed_end: int
    incomplete_ranges: list[dict]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_id(source_id: int, digest: str, start: int, end: int) -> str:
    return "r_" + _sha(json.dumps([source_id, digest, start, end], separators=(",", ":")))


def _window(raw: str, start: int, end: int | None) -> tuple[int, int]:
    if not isinstance(raw, str):
        raise RecordReferenceError("Source text must be a string")
    end = len(raw) if end is None else end
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(raw):
        raise RecordReferenceError("invalid Source window")
    return start, end


def _spans(raw: str) -> list[tuple[int | None, int, int]]:
    markers = list(_RECORD.finditer(raw))
    if markers:
        result = []
        for i, marker in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
            # Separator whitespace is not part of a record's text.  It remains
            # in the Source and is never silently joined into another quote.
            end = len(raw[:end].rstrip())
            result.append((int(marker.group(1)), marker.start(), end))
        return result
    # Ordinary text has no declared record structure: use complete paragraphs,
    # retaining every character inside them, including CRLF and inline markup.
    result = []
    start = 0
    for boundary in re.finditer(r"\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*", raw):
        if raw[start:boundary.start()].strip():
            result.append((None, start, boundary.start()))
        start = boundary.end()
    if raw[start:].strip():
        result.append((None, start, len(raw)))
    return result


def _alias_values(value: str) -> dict[str, str]:
    aliases = {}
    for part in value.split(" | "):
        language, separator, name = part.partition("=")
        if separator and language.strip() and name.strip():
            aliases[language.strip()] = name.strip()
    return aliases


def _legend(raw: str, header_end: int) -> dict | None:
    header = raw[:header_end]
    marker = _LEGEND.search(header)
    if marker is None:
        return None
    start = marker.start()
    cursor = marker.end()
    if header[cursor:cursor + 1] == "\n":
        cursor += 1
    end = cursor
    aliases = {}
    for line in header[cursor:].splitlines(keepends=True):
        if not line.strip() or line.startswith("["):
            break
        speaker, separator, value = line.rstrip("\r\n").partition(": ")
        if separator:
            aliases[speaker] = _alias_values(value)
        end += len(line)
    end = len(header[:end].rstrip("\r\n"))
    return {"start": start, "end": end, "text": raw[start:end], "aliases": aliases}


def _single_field(raw_record: str, name: str) -> tuple[str, int, int] | None:
    match = re.search(r"(?m)^\[" + re.escape(name) + r":[ \t]*(.*)\][ \t]*\r?$", raw_record)
    return (match.group(1), match.start(1), match.end(1)) if match else None


def _display_span(raw: str, lo: int, hi: int, language: str, formatted: bool) -> tuple[str, int, int]:
    if not formatted:
        return "raw", lo, hi
    block = raw[lo:hi]
    markers = list(_LANGUAGE.finditer(block))
    choices = []
    for i, marker in enumerate(markers):
        start = lo + marker.end()
        end = lo + (markers[i + 1].start() if i + 1 < len(markers) else len(block))
        end = len(raw[:end].rstrip("\r\n"))
        if raw[start:end].strip():
            choices.append((marker.group(1), start, end))
    preferred = next((item for item in choices if item[0].casefold() == language.casefold()), None)
    if preferred:
        return preferred
    # The stored original script is preferable to guessing a translation.  A
    # multi-line script with no unambiguous single-line field falls back to an
    # available language (or the entire record), never a synthetic joined text.
    script = _single_field(block, "script_raw")
    if script and script[0].strip():
        return "script_raw", lo + script[1], lo + script[2]
    if choices:
        return next((item for item in choices if item[0] == "unknown"), choices[0])
    return "raw", lo, hi


def project_window(source_id: int, raw: str, episode_ids: list[int], *,
                   start: int = 0, end: int | None = None,
                   language: str = "zh-CN") -> RecordProjection:
    """Project complete visible records and report incomplete boundary records.

    ``consumed_end`` stops at the earliest incomplete record's start.  If the
    requested window starts inside a record it can be *before* ``start``: the
    caller must reread that prefix, rather than mark the missing text consumed.
    The Source's existing alias legend is attached as metadata with its own raw
    range; it is not a separately citable dialogue record.
    """
    start, end = _window(raw, start, end)
    if type(source_id) is not int or source_id <= 0:
        raise RecordReferenceError("source_id must be a positive integer")
    if not isinstance(episode_ids, (list, tuple)) or not episode_ids or any(type(e) is not int or e <= 0 for e in episode_ids):
        raise RecordReferenceError("episode_ids must contain positive integers")
    if not isinstance(language, str) or not language.strip():
        raise RecordReferenceError("language must be nonempty")
    if start == end:
        return RecordProjection([], end, [])
    spans = _spans(raw)
    digest = _sha(raw)
    legend = _legend(raw, spans[0][1]) if spans and spans[0][0] is not None else None
    records, incomplete = [], []
    for index, lo, hi in spans:
        if hi <= start or lo >= end:
            continue
        if lo < start or hi > end:
            incomplete.append({"record_index": index, "start": lo, "end": hi,
                               "missing_prefix": lo < start, "missing_suffix": hi > end})
            continue
        selected_language, text_start, text_end = _display_span(raw, lo, hi, language, index is not None)
        block = raw[lo:hi]
        speaker = _single_field(block, "speaker_raw") if index is not None else None
        speaker_raw = speaker[0] if speaker else None
        aliases = deepcopy((legend or {}).get("aliases", {}).get(speaker_raw, {}))
        inline_aliases = _single_field(block, "speaker_aliases") if index is not None else None
        if inline_aliases:
            aliases.update(_alias_values(inline_aliases[0]))
        record = {
            "record_id": _record_id(source_id, digest, lo, hi),
            "record_index": index, "source_id": source_id, "source_sha256": digest,
            "episode_ids": sorted(set(episode_ids)), "language": selected_language,
            "text": raw[text_start:text_end], "start": text_start, "end": text_end,
            "context_start": lo, "context_end": hi, "raw_record": block,
            "raw_record_sha256": _sha(block), "speaker_raw": speaker_raw,
            "speaker_aliases": aliases, "speaker_alias_legend": deepcopy(legend),
        }
        records.append(record)
    return RecordProjection(records, min([end] + [item["start"] for item in incomplete]), incomplete)


def project_source(source_id: int, raw: str, episode_ids: list[int], *,
                   start: int = 0, end: int | None = None,
                   language: str = "zh-CN") -> list[dict]:
    """Convenience view; use project_window when committing read progress."""
    return project_window(source_id, raw, episode_ids, start=start, end=end, language=language).records


def align_window(raw: str, start: int = 0, end: int | None = None) -> tuple[int, int]:
    """Expand a proposed window to include each intersecting complete record.

    A single large record can exceed the original window budget.  The caller
    must account for the expanded size before sending it to a model.
    """
    start, end = _window(raw, start, end)
    if start == end:
        return start, end
    intersecting = [(lo, hi) for _, lo, hi in _spans(raw) if lo < end and hi > start]
    if not intersecting:
        return start, end
    return min(start, intersecting[0][0]), max(end, intersecting[-1][1])


def model_records(records: list[dict]) -> list[dict]:
    """Compact provider input; full local records must be kept for resolution."""
    fields = ("record_id", "record_index", "source_id", "episode_ids", "language",
              "text", "start", "end", "speaker_raw", "speaker_aliases")
    return [{key: deepcopy(record[key]) for key in fields} for record in records]


def resolve_record(record_id: str, visible_records: list[dict], *, episode_id: int,
                   start: int | None = None, end: int | None = None,
                   raw: str | None = None) -> dict:
    """Resolve a locally projected record, optionally to an absolute raw span.

    The optional span must lie entirely inside the displayed language text.
    Pass the current Source ``raw`` to detect changes since projection.  Never
    pass model-created dictionaries as ``visible_records``.
    """
    if not isinstance(record_id, str) or not record_id or type(episode_id) is not int or episode_id <= 0:
        raise RecordReferenceError("invalid record or Episode ID")
    matches = [r for r in visible_records if r.get("record_id") == record_id and episode_id in r.get("episode_ids", [])]
    if not matches:
        raise RecordReferenceError("record is not visible for this Episode")
    record = matches[0]
    if any((r["source_id"], r["source_sha256"], r["start"], r["end"], r["text"]) !=
           (record["source_id"], record["source_sha256"], record["start"], record["end"], record["text"]) for r in matches[1:]):
        raise RecordReferenceError("record has ambiguous visible projections")
    lo, hi = record["context_start"], record["context_end"]
    if record_id != _record_id(record["source_id"], record["source_sha256"], lo, hi):
        raise RecordReferenceError("record identity changed")
    block = record["raw_record"]
    a, b = record["start"], record["end"]
    if not lo <= a < b <= hi or len(block) != hi - lo or _sha(block) != record["raw_record_sha256"] or block[a-lo:b-lo] != record["text"]:
        raise RecordReferenceError("record text no longer matches its raw span")
    if raw is not None and (not isinstance(raw, str) or _sha(raw) != record["source_sha256"] or raw[lo:hi] != block):
        raise RecordReferenceError("Source changed since record projection")
    if (start is None) != (end is None):
        raise RecordReferenceError("both span endpoints are required")
    start, end = (a, b) if start is None else (start, end)
    if type(start) is not int or type(end) is not int or not a <= start < end <= b:
        raise RecordReferenceError("reference span is outside the displayed text")
    quote = block[start-lo:end-lo]
    if not quote.strip():
        raise RecordReferenceError("reference quote is empty")
    return {"source_id": record["source_id"], "episode_id": episode_id,
            "source_sha256": record["source_sha256"], "start": start, "end": end,
            "quote": quote, "context_start": lo, "context_end": hi,
            "record_id": record_id, "record_index": record["record_index"],
            "language": record["language"]}
