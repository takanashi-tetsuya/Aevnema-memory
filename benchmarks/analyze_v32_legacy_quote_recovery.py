"""Read-only feasibility analysis for recovering locators from K_diag.

This tool deliberately does *not* write evidence quotes, re-extract Episodes,
or use any question/gold artefact.  It reports only deterministic locator
leads available from the legacy Episode, its Source byte string, and its
existing epistemic note.  A recovered locator is not an accepted semantic
claim and is never made runtime eligible by this analysis.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

SCHEMA = "aevnema.v3_2.legacy_quote_recovery_feasibility.v1"
_RECORD_HEADER_RE = re.compile(r"(?m)^\[record:\s*(\d+)\]\s*$")
_NOTE_RECORD_RE = re.compile(
    r"(?:source\s*)?(?:record|记录)\s*(\d+)(?:\s*(?:-|–|~|至|到)\s*(\d+))?",
    flags=re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_NOTE_RANGE = 24


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normalise(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _source_record_blocks(raw_text: str) -> dict[str, str]:
    """Parse the immutable record envelope locally, rejecting duplicate IDs.

    This duplicate-safe parser intentionally mirrors the delivery contract's
    simple record framing but avoids importing the application (and therefore
    any provider-client dependency) during an offline SQLite analysis.
    """

    matches = list(_RECORD_HEADER_RE.finditer(raw_text))
    blocks: dict[str, str] = {}
    duplicates: set[str] = set()
    for index, match in enumerate(matches):
        record_id = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        if record_id in blocks:
            duplicates.add(record_id)
            continue
        blocks[record_id] = raw_text[match.start():end].strip()
    for record_id in duplicates:
        blocks.pop(record_id, None)
    return blocks


def _note_record_ids(note: str, available: set[str]) -> tuple[list[str], list[str]]:
    """Return bounded record locators and explicit reasons not to trust one."""

    values: list[str] = []
    reasons: list[str] = []
    for match in _NOTE_RECORD_RE.finditer(note):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            reasons.append("descending_note_record_range")
            continue
        if end - start + 1 > _MAX_NOTE_RANGE:
            reasons.append("note_record_range_exceeds_bounded_feasibility_limit")
            continue
        for number in range(start, end + 1):
            record_id = str(number)
            if record_id in available:
                values.append(record_id)
            else:
                reasons.append("note_record_not_present_in_current_raw_source")
    return list(dict.fromkeys(values)), list(dict.fromkeys(reasons))


def _literal_episode_record_ids(episode_text: str, blocks: Mapping[str, str]) -> list[str]:
    """Find only whole-summary literal containment in one raw record block."""

    target = _normalise(episode_text)
    if len(target) < 12:
        return []
    return [record_id for record_id, block in blocks.items() if target in _normalise(block)]


def analyze(*, database: Path, output_dir: Path) -> dict[str, Any]:
    database = database.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    if not database.is_file():
        raise FileNotFoundError(database)
    uri = f"{database.as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    try:
        totals = connection.execute(
            """
            SELECT
              COUNT(*) AS episodes,
              SUM(CASE WHEN COALESCE(evidence_quotes_json, '[]') != '[]' THEN 1 ELSE 0 END) AS quoted,
              SUM(CASE WHEN COALESCE(evidence_spans_json, '[]') != '[]' THEN 1 ELSE 0 END) AS spanned,
              SUM(CASE WHEN evidence_origin = 'source' THEN 1 ELSE 0 END) AS source_origin
            FROM episode
            """
        ).fetchone()
        rows = connection.execute(
            """
            SELECT e.id, e.source_id, e.source_key, e.segment_index, e.text,
                   e.epistemic_note, e.evidence_origin, e.evidence_basis,
                   e.evidence_quotes_json, e.evidence_spans_json, s.raw_text
            FROM episode AS e
            JOIN source AS s ON s.id = e.source_id
            ORDER BY e.id
            """
        ).fetchall()
    finally:
        connection.close()

    outcomes: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in rows:
        raw = str(row[10])
        blocks = _source_record_blocks(raw)
        available = set(blocks)
        note = str(row[5] or "")
        note_ids, note_reasons = _note_record_ids(note, available)
        literal_ids = _literal_episode_record_ids(str(row[4]), blocks)
        if literal_ids and note_ids:
            category = "literal_episode_text_and_explicit_note_locator"
        elif literal_ids:
            category = "literal_episode_text_single_raw_record_candidate"
        elif note_ids:
            category = "explicit_note_record_locator_candidate"
        elif not blocks:
            category = "raw_source_has_no_unambiguous_record_blocks"
        elif note_reasons:
            category = "note_locator_unusable"
        else:
            category = "no_deterministic_literal_or_note_locator"
        category_counts[category] += 1
        reason_counts.update(note_reasons)
        outcomes.append(
            {
                "episode_id": int(row[0]),
                "source_id": int(row[1]),
                "source_key": str(row[2]),
                "segment_index": int(row[3]),
                "legacy_evidence_origin": str(row[6]),
                "legacy_evidence_basis": str(row[7]),
                "persisted_quote_present": str(row[8]) != "[]",
                "persisted_span_present": str(row[9]) != "[]",
                "source_raw_sha256": "sha256:" + sha256(raw.encode("utf-8")).hexdigest(),
                "available_raw_record_count": len(blocks),
                "literal_episode_record_ids": literal_ids,
                "explicit_note_record_ids": note_ids,
                "note_locator_reasons": note_reasons,
                "feasibility_category": category,
                "semantic_claim_status": "not_assessed",
                "derived_evidence_quote_written": False,
                "runtime_eligibility": "forbidden",
            }
        )

    total_episodes = int(totals["episodes"] or 0)
    strict_bound = int(totals["quoted"] or 0) and int(totals["spanned"] or 0)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "read_only_feasibility_complete",
        "created_at": _utc_now(),
        "paid_provider_calls_issued": 0,
        "database": {
            "path": str(database),
            "sha256": _sha256_file(database),
            "opened_mode": "sqlite_read_only",
        },
        "learning_data_eligibility": {
            "episode_count": total_episodes,
            "source_origin_episode_count": int(totals["source_origin"] or 0),
            "episodes_with_persisted_evidence_quotes": int(totals["quoted"] or 0),
            "episodes_with_persisted_evidence_spans": int(totals["spanned"] or 0),
            "strict_source_bound_creation_eligible": bool(strict_bound),
            "strict_source_bound_creation_eligibility_reason": (
                "not_eligible_no_episode_has_both_persisted_quote_and_span"
                if not strict_bound
                else "aggregate_only_check_not_a_semantic_approval"
            ),
            "ordinary_legacy_retrieval": "possible_for_diagnostic_only_when_source_rows_exist",
            "legacy_retrieval_is_new_edge_eligibility": False,
        },
        "recovery_rules": {
            "inputs_allowed": ["legacy_episode_text", "legacy_epistemic_note", "source_id", "unchanged_raw_source"],
            "literal_rule": "The complete legacy Episode text must occur in exactly one raw record block.",
            "note_rule": "Only explicit bounded record/记录 locators already present in that Source are reported.",
            "not_done": ["no_quote_write", "no_reextraction", "no_gold_lookup", "no_runtime_enablement", "no_semantic_entailment_inference"],
        },
        "summary": {
            "categories": dict(sorted(category_counts.items())),
            "note_locator_failure_reasons": dict(sorted(reason_counts.items())),
            "deterministic_locator_candidate_count": sum(
                1
                for item in outcomes
                if item["literal_episode_record_ids"] or item["explicit_note_record_ids"]
            ),
            "no_deterministic_locator_count": sum(
                1
                for item in outcomes
                if item["feasibility_category"] == "no_deterministic_literal_or_note_locator"
            ),
        },
        "episodes_full_local": outcomes,
    }
    output_dir.mkdir(parents=True)
    full = output_dir / "legacy_quote_recovery_feasibility.full_local.json"
    summary = output_dir / "legacy_quote_recovery_feasibility.json"
    full.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {key: value for key, value in payload.items() if key != "episodes_full_local"}
    summary.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    categories = payload["summary"]["categories"]
    assert isinstance(categories, Mapping)
    lines = [
        "# K_diag literal-locator recovery feasibility",
        "",
        "This was a SQLite read-only analysis. No evidence quote, span, Episode, vector, gold item, or runtime state was written; no provider request was made.",
        "",
        f"- Episodes: `{total_episodes}`",
        f"- Persisted evidence quotes: `{int(totals['quoted'] or 0)}`",
        f"- Persisted evidence spans: `{int(totals['spanned'] or 0)}`",
        f"- Deterministic locator candidates: `{payload['summary']['deterministic_locator_candidate_count']}`",
        f"- No deterministic locator: `{payload['summary']['no_deterministic_locator_count']}`",
        "",
        "A locator candidate only identifies a raw record to inspect. It does not prove a legacy summary's meaning, speaker, scope, or causal claim, and it does not create a source-bound Episode or an eligible learning edge.",
        "",
        "| Feasibility category | Episodes |",
        "| --- | ---: |",
        *[f"| {name} | {count} |" for name, count in categories.items()],
        "",
        "Strict source-bound edge creation remains ineligible in this snapshot because no Episode carries both persisted quote and span evidence. Ordinary legacy retrieval can remain a diagnostic route only; it is not a substitute for the strict creation contract.",
    ]
    (output_dir / "LEGACY_QUOTE_RECOVERY_FEASIBILITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"full_local": str(full), "summary": str(summary), "episodes": total_episodes, "locator_candidates": payload["summary"]["deterministic_locator_candidate_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
