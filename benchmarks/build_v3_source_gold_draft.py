"""Re-anchor the historical Stage3 scorer as an explicitly provisional v3 draft.

This tool is evaluator-only.  It reads a SQLite snapshot in read-only mode and
creates a review worklist with source-segment and Episode-text hashes.  It does
not claim that historical Episode groups are Source-level gold: a row remains
unscorable until a reviewer adds a real source record/span and its raw hash.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from typing import Any, Iterable

from benchmarks.support.evidence_scorer import CRITERIA


DRAFT_STATUS = "draft_pending_source_span_review"
GOLD_SCHEMA_VERSION = "aevnema.source-gold.draft.v1"
SPLIT_SCHEMA_VERSION = "aevnema.gold-split.draft.v1"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_text(path: Path, value: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _readonly_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _fetch_episode_candidates(
    connection: sqlite3.Connection, episode_ids: Iterable[int]
) -> list[dict[str, Any]]:
    ids = [int(value) for value in episode_ids]
    if not ids:
        raise ValueError("legacy alternative group must not be empty")
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
        f"""
        SELECT
            episode.id,
            episode.source_id,
            episode.source_key,
            episode.segment_index,
            episode.text,
            episode.evidence_spans_json,
            episode.evidence_quotes_json,
            source.raw_text
        FROM episode
        JOIN source ON source.id = episode.source_id
        WHERE episode.id IN ({placeholders})
        """,
        ids,
    ).fetchall()
    by_id = {int(row["id"]): row for row in rows}
    missing = sorted(set(ids).difference(by_id))
    if missing:
        raise ValueError(f"snapshot is missing historical Episode IDs: {missing}")
    candidates: list[dict[str, Any]] = []
    for episode_id in ids:
        row = by_id[episode_id]
        try:
            existing_spans = json.loads(str(row["evidence_spans_json"]))
            existing_quotes = json.loads(str(row["evidence_quotes_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Episode {episode_id} has invalid legacy evidence JSON") from exc
        candidates.append(
            {
                "episode_id": episode_id,
                "source_key": str(row["source_key"]),
                "segment_index": int(row["segment_index"]),
                "episode_text_sha256": _text_sha256(str(row["text"])),
                "source_segment_sha256": _text_sha256(str(row["raw_text"])),
                "record_locator": None,
                "span_start": None,
                "span_end": None,
                "raw_span_sha256": None,
                "legacy_episode_evidence_span_count": len(existing_spans),
                "legacy_episode_evidence_quote_count": len(existing_quotes),
                "review_status": "pending_source_span_review",
                "usable_for_scoring": False,
            }
        )
    return candidates


def _questions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError("questions file must contain an array")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not item.get("id") or not item.get("question"):
            raise ValueError("each question must have id and question")
        question_id = str(item["id"])
        if question_id in result:
            raise ValueError(f"duplicate question id: {question_id}")
        result[question_id] = item
    return result


def build_draft(
    *,
    database: Path,
    questions_path: Path,
    legacy_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Build a non-scorable re-anchoring worklist and its source-connected split."""

    questions = _questions_by_id(questions_path)
    legacy_manifest = _read_json(legacy_manifest_path)
    manifest_questions = dict(legacy_manifest.get("questions", {}))
    database_sha256 = _file_sha256(database)
    snapshot_id = f"legacy-{database.stem}"
    families: list[dict[str, Any]] = []
    all_source_keys: set[str] = set()
    connection = _readonly_connection(database)
    try:
        for question_id, criterion in CRITERIA.items():
            question = questions.get(question_id)
            if question is None:
                raise ValueError(f"questions file is missing legacy question: {question_id}")
            historical = manifest_questions.get(question_id, {})
            required_groups = criterion["required_episode_groups"]
            claim_groups: list[dict[str, Any]] = []
            for index, alternatives in enumerate(required_groups, start=1):
                candidates = _fetch_episode_candidates(connection, alternatives)
                source_keys = sorted({item["source_key"] for item in candidates})
                unexpected_sources = sorted(
                    set(source_keys).difference(criterion["required_sources"])
                )
                if unexpected_sources:
                    raise ValueError(
                        "snapshot-local legacy Episode IDs no longer map to the "
                        f"expected source closure for {question_id}: {unexpected_sources}"
                    )
                all_source_keys.update(source_keys)
                claim_groups.append(
                    {
                        "claim_group_id": f"{question_id}:legacy-slot-{index}",
                        "coverage_clause": {
                            "operator": "any_of",
                            "required": True,
                            "members": [
                                f"{question_id}:legacy-slot-{index}:candidate-{candidate['episode_id']}"
                                for candidate in candidates
                            ],
                        },
                        "legacy_episode_alternatives": [int(value) for value in alternatives],
                        "candidate_source_keys": source_keys,
                        "evidence_atoms": candidates,
                        "review_status": "pending_source_span_review",
                        "usable_for_scoring": False,
                    }
                )
            families.append(
                {
                    "family_id": f"stage3:{question_id}",
                    "question_id": question_id,
                    "question_sha256": _text_sha256(str(question["question"])),
                    "question_file": str(questions_path).replace("\\", "/"),
                    "query_mode": "factual",
                    "legacy_status": "legacy_calibration_only",
                    "question_coverage": {
                        "operator": "all_of",
                        "members": [group["claim_group_id"] for group in claim_groups],
                    },
                    "legacy_required_sources": list(criterion["required_sources"]),
                    "legacy_required_facts": list(historical.get("required_facts", [])),
                    "legacy_anchors": list(historical.get("anchors", [])),
                    "claim_groups": claim_groups,
                    "hard_negatives": [],
                    "review_status": "pending_source_span_review",
                    "usable_for_scoring": False,
                }
            )
    finally:
        connection.close()
    draft = {
        "schema_version": GOLD_SCHEMA_VERSION,
        "visibility": "evaluator_only",
        "status": DRAFT_STATUS,
        "source_manifest_sha256": _file_sha256(legacy_manifest_path),
        "question_manifest_sha256": _file_sha256(questions_path),
        "legacy_keyword_diagnostic": {
            "path": "benchmarks/manifests/original_five_question_quality_benchmark.json",
            "description": "Preserved separately as the historical five-question, fifteen-keyword diagnostic; it is not Source claim gold.",
        },
        "snapshots": [
            {
                "snapshot_id": snapshot_id,
                "database_filename": database.name,
                "database_sha256": database_sha256,
                "availability": {"K_old": True, "K_full": "unmapped"},
                "episode_id_stability": "snapshot_local_only",
            }
        ],
        "families": families,
        "limitations": [
            "Every evidence atom is pending source record/span review and is unusable for score gating.",
            "Legacy Episode IDs are only re-anchoring candidates and must not be reused across snapshots without hash verification.",
            "No source file hash or raw span hash is available until a reviewer anchors a source record/span.",
            "This asset is evaluator-only and production modules must not read it.",
        ],
    }
    component_id = "stage3-main-source-connected-1"
    split = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "visibility": "evaluator_only",
        "status": DRAFT_STATUS,
        "gold_manifest_sha256": _text_sha256(_canonical_json(draft).decode("utf-8")),
        "grouping_policy": {
            "unit": "source_connected_component",
            "random_question_split_allowed": False,
            "reason": "The three historical questions share source keys through a single connected component.",
        },
        "assignments": [
            {
                "family_id": family["family_id"],
                "split": "legacy_calibration",
                "source_component_id": component_id,
                "source_keys": sorted(
                    {
                        source_key
                        for group in family["claim_groups"]
                        for source_key in group["candidate_source_keys"]
                    }
                ),
                "availability": {"K_old": True, "K_full": "unmapped"},
                "rationale": "Historical Stage3 family; not eligible as an independent holdout until source-span review and source-disjoint families exist.",
            }
            for family in families
        ],
        "unassigned_holdout_reason": "No source-disjoint reviewed family is available in this draft.",
        "all_source_keys": sorted(all_source_keys),
    }
    review = "\n".join(
        [
            "# Aevnema v3 Source-Gold Draft Review",
            "",
            "Status: **draft_pending_source_span_review**. This is an evaluator-only re-anchoring worklist, not an accepted Source-level gold set.",
            "",
            "## What was preserved",
            "",
            "- The three historical Stage3 questions and their nine Episode alternative groups were retained as legacy candidate slots.",
            "- The original five-question / fifteen-keyword benchmark remains separately frozen as a diagnostic and was not promoted to gold.",
            "- Each candidate records its snapshot-local Episode ID, source key, segment index, Episode text hash, and source-segment hash.",
            "",
            "## Why this draft cannot score systems",
            "",
            "- Historical candidates have no reviewed source record locator, span start/end, source-file hash, or raw span hash.",
            "- Each evidence atom is marked `pending_source_span_review` and `usable_for_scoring: false`.",
            "- The historical questions are one source-connected component and are all assigned to `legacy_calibration`; none is a blind holdout.",
            "",
            "## Required review before promotion",
            "",
            "1. Anchor every retained claim to a source record/JSONPath and exact source span.",
            "2. Store raw span and source-file hashes, then independently review the claim's epistemic strength.",
            "3. Express any joint claim with an explicit `all_of` clause and any alternative with `any_of`.",
            "4. Create source-disjoint reviewed families before declaring a calibration/holdout split.",
            "5. Keep this material evaluator-only; do not feed it into `MemoryApplication`, `QueryEngine`, or model prompts.",
            "",
        ]
    )
    return draft, split, review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("validation/evaluation-questions-stage3-deep.json"),
    )
    parser.add_argument(
        "--legacy-manifest",
        type=Path,
        default=Path("validation/stage3-deep-evidence-manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing draft artifacts in --output-dir",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    draft, split, review = build_draft(
        database=args.database,
        questions_path=args.questions,
        legacy_manifest_path=args.legacy_manifest,
    )
    output = args.output_dir
    _atomic_write(output / "gold_manifest.draft.json", draft, overwrite=args.overwrite)
    _atomic_write(output / "split_manifest.draft.json", split, overwrite=args.overwrite)
    _atomic_write_text(output / "gold_review.md", review, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": draft["status"],
                "family_count": len(draft["families"]),
                "claim_group_count": sum(len(item["claim_groups"]) for item in draft["families"]),
                "output_dir": str(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
