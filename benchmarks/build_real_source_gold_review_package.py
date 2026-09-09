"""Build a local, non-scorable review package for a draft Source-gold set.

The historical Stage3 draft records episode summaries and rendered Source
segments from a legacy SQLite snapshot.  Those are useful *leads*, but not
Source-level claims.  This evaluator-only tool recovers their record windows,
reads the currently authorised Blue Archive JSON files, and prepares one
review item per logical draft atom.

It intentionally never writes a gold manifest, changes review statuses, calls
a model, imports a database, or invokes the formal benchmark.  Every emitted
item remains ``pending_source_span_review`` and ``usable_for_scoring: false``.
The full-local package is for a human reviewer to choose, narrow, split, or
reject exact raw ``TextCn`` spans before a separate approved-gold workflow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping

from memory_demo.adapters.blue_archive import BlueArchiveJsonAdapter, clean_game_text
from memory_demo.config import SegmentConfig
from memory_demo.ingestion.segmenter import NaturalSegmenter


PACKAGE_SCHEMA = "aevnema.v3.source_gold_review_package.v1"
FROZEN_SOURCE_MANIFEST_SCHEMA = "aevnema.source-gold.frozen-source-manifest.v1"
DRAFT_STATUS = "draft_pending_source_span_review"
PENDING_STATUS = "pending_source_span_review"
_RECORD_HEADER = re.compile(r"(?m)^\[record:\s*(\d+)\]\s*$")
_RENDERED_RECORD = re.compile(
    r"(?ms)^\[record:\s*(\d+)\]\s*\n(.*?)(?=^\[record:\s*\d+\]\s*$|\Z)"
)
_ZH_CN_LINE = re.compile(r"(?m)^zh-CN:\s?(.*)$")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON input: {path.name}") from error


def _readonly_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA trusted_schema = OFF")
    return connection


def _required_atoms(draft: Mapping[str, object]) -> Iterable[tuple[dict[str, object], dict[str, object], dict[str, object]]]:
    families = draft.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("draft must contain at least one family")
    for raw_family in families:
        if not isinstance(raw_family, dict):
            raise ValueError("draft family must be an object")
        groups = raw_family.get("claim_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("draft family must contain claim groups")
        for raw_group in groups:
            if not isinstance(raw_group, dict):
                raise ValueError("draft claim group must be an object")
            atoms = raw_group.get("evidence_atoms")
            if not isinstance(atoms, list) or not atoms:
                raise ValueError("draft claim group must contain evidence atoms")
            for raw_atom in atoms:
                if not isinstance(raw_atom, dict):
                    raise ValueError("draft evidence atom must be an object")
                yield raw_family, raw_group, raw_atom


def _atom_identity(family: Mapping[str, object], group: Mapping[str, object], atom: Mapping[str, object]) -> str:
    family_id = str(family.get("family_id", "")).strip()
    group_id = str(group.get("claim_group_id", "")).strip()
    episode_id = atom.get("episode_id")
    if not family_id or not group_id or isinstance(episode_id, bool) or not isinstance(episode_id, int):
        raise ValueError("draft atom needs family_id, claim_group_id, and integer episode_id")
    return f"{family_id}/{group_id}/candidate-{episode_id}"


def _load_legacy_rows(
    snapshot: Path,
    atoms: Iterable[tuple[dict[str, object], dict[str, object], dict[str, object]]],
) -> dict[int, dict[str, object]]:
    wanted: dict[int, tuple[str, int, str, str]] = {}
    for family, group, atom in atoms:
        _atom_identity(family, group, atom)
        episode_id = atom["episode_id"]
        assert isinstance(episode_id, int)
        source_key = atom.get("source_key")
        segment_index = atom.get("segment_index")
        episode_hash = atom.get("episode_text_sha256")
        segment_hash = atom.get("source_segment_sha256")
        if (
            not isinstance(source_key, str)
            or not source_key
            or isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or not isinstance(episode_hash, str)
            or not isinstance(segment_hash, str)
        ):
            raise ValueError(f"draft episode {episode_id} is missing legacy binding fields")
        expected = (source_key, segment_index, episode_hash, segment_hash)
        prior = wanted.setdefault(episode_id, expected)
        if prior != expected:
            raise ValueError(f"draft reuses episode {episode_id} with incompatible bindings")

    connection = _readonly_connection(snapshot)
    try:
        placeholders = ",".join("?" for _ in wanted)
        rows = connection.execute(
            f"""
            SELECT e.id, e.source_key, e.segment_index, e.text, s.raw_text
            FROM episode AS e
            JOIN source AS s ON s.id = e.source_id
            WHERE e.id IN ({placeholders})
            """,
            tuple(sorted(wanted)),
        ).fetchall()
    except sqlite3.DatabaseError as error:
        raise ValueError("legacy snapshot does not expose the required episode/source tables") from error
    finally:
        connection.close()
    found = {int(row["id"]): row for row in rows}
    missing = sorted(set(wanted).difference(found))
    if missing:
        raise ValueError(f"legacy snapshot is missing draft episode IDs: {missing}")

    resolved: dict[int, dict[str, object]] = {}
    for episode_id, expected in wanted.items():
        source_key, segment_index, episode_hash, segment_hash = expected
        row = found[episode_id]
        actual_source_key = str(row["source_key"])
        actual_segment_index = int(row["segment_index"])
        episode_text = str(row["text"])
        segment_text = str(row["raw_text"])
        if actual_source_key != source_key or actual_segment_index != segment_index:
            raise ValueError(f"legacy snapshot binding differs for episode {episode_id}")
        if _sha256_text(episode_text) != episode_hash:
            raise ValueError(f"legacy episode text hash differs for episode {episode_id}")
        if _sha256_text(segment_text) != segment_hash:
            raise ValueError(f"legacy source segment hash differs for episode {episode_id}")
        record_indices = [int(value) for value in _RECORD_HEADER.findall(segment_text)]
        if not record_indices or len(record_indices) != len(set(record_indices)):
            raise ValueError(f"legacy source segment has no unambiguous record window for episode {episode_id}")
        rendered_records: dict[int, str] = {}
        for raw_index, body in _RENDERED_RECORD.findall(segment_text):
            match = _ZH_CN_LINE.search(body)
            if match is not None:
                rendered_records[int(raw_index)] = match.group(1)
        resolved[episode_id] = {
            "episode_id": episode_id,
            "source_key": source_key,
            "segment_index": segment_index,
            "episode_text": episode_text,
            "episode_text_sha256": episode_hash,
            "source_segment_sha256": segment_hash,
            "record_indices": record_indices,
            "rendered_zh_cn": rendered_records,
        }
    return resolved


def _source_key_path(source_root: Path, source_key: str) -> Path:
    relative = Path(source_key)
    if (
        not source_key
        or relative.is_absolute()
        or relative.drive
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("draft source_key is not a safe relative Source path")
    target = source_root.joinpath(*relative.parts)
    try:
        target.resolve(strict=True).relative_to(source_root.resolve())
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"draft source file is unavailable: {source_key}") from error
    if not target.is_file():
        raise ValueError(f"draft source path is not a file: {source_key}")
    return target


def _source_document(path: Path) -> tuple[dict[str, object], bytes]:
    raw_bytes = path.read_bytes()
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Source file is not UTF-8 JSON: {path.name}") from error
    if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), list):
        raise ValueError(f"Source file is not a Blue Archive content document: {path.name}")
    return parsed, raw_bytes


def _record_material(
    document: Mapping[str, object],
    record_index: int,
    *,
    source_key: str,
    in_legacy_window: bool,
) -> dict[str, object]:
    content = document.get("content")
    assert isinstance(content, list)
    if record_index < 0 or record_index >= len(content):
        raise ValueError(f"legacy record index {record_index} is unavailable in {source_key}")
    record = content[record_index]
    if not isinstance(record, dict):
        raise ValueError(f"legacy record index {record_index} is not an object in {source_key}")
    text_cn = record.get("TextCn")
    if (not isinstance(text_cn, str) or not text_cn) and in_legacy_window:
        raise ValueError(f"legacy record index {record_index} has no raw TextCn in {source_key}")
    text_jp = record.get("TextJp")
    script_kr = record.get("ScriptKr")
    if not isinstance(text_cn, str) or not text_cn:
        return {
            "record_index": record_index,
            "record_pointer": f"/content/{record_index}",
            "record_locator": None,
            "full_record_span": None,
            "source_original": {
                "TextJp": text_jp if isinstance(text_jp, str) else "",
                "TextCn": "",
                "ScriptKr": script_kr if isinstance(script_kr, str) else "",
            },
            "field_sha256": {
                "TextJp": _sha256_text(text_jp) if isinstance(text_jp, str) else None,
                "TextCn": None,
                "ScriptKr": _sha256_text(script_kr) if isinstance(script_kr, str) else None,
            },
            "in_legacy_window": in_legacy_window,
            "context_only_reason": "record has no raw TextCn and cannot be a formal span candidate",
        }
    locator = {
        "kind": "blue_archive.content_record.v1",
        "pointer": f"/content/{record_index}/TextCn",
        "record_index": record_index,
        "record_pointer": f"/content/{record_index}",
    }
    return {
        "record_index": record_index,
        "record_locator": locator,
        "full_record_span": {
            "span_start": 0,
            "span_end": len(text_cn),
            "raw_span_sha256": _sha256_text(text_cn),
            "text": text_cn,
        },
        "source_original": {
            "TextJp": text_jp if isinstance(text_jp, str) else "",
            "TextCn": text_cn,
            "ScriptKr": script_kr if isinstance(script_kr, str) else "",
        },
        "field_sha256": {
            "TextJp": _sha256_text(text_jp) if isinstance(text_jp, str) else None,
            "TextCn": _sha256_text(text_cn),
            "ScriptKr": _sha256_text(script_kr) if isinstance(script_kr, str) else None,
        },
        "in_legacy_window": in_legacy_window,
    }


def _current_segment_material(path: Path, source_key: str, legacy_segment_index: int, expected_hash: str) -> dict[str, object]:
    adapter = BlueArchiveJsonAdapter()
    segments = NaturalSegmenter(SegmentConfig()).segment(source_key, adapter.read_blocks(path))
    if legacy_segment_index < 0 or legacy_segment_index >= len(segments):
        return {
            "status": "unavailable_at_legacy_index",
            "legacy_segment_index": legacy_segment_index,
            "current_segment_count": len(segments),
        }
    current = segments[legacy_segment_index]
    digest = _sha256_text(current.raw_text)
    return {
        "status": "available",
        "legacy_segment_index": legacy_segment_index,
        "current_segment_count": len(segments),
        "first_record": current.first_record,
        "last_record": current.last_record,
        "current_rendered_segment_sha256": digest,
        "matches_legacy_rendered_segment": digest == expected_hash,
    }


def _claim_boundary(family: Mapping[str, object], group: Mapping[str, object]) -> dict[str, object]:
    return {
        "question_id": str(family.get("question_id", "")),
        "legacy_required_facts": list(family.get("legacy_required_facts", [])),
        "question_coverage": dict(family.get("question_coverage", {})),
        "claim_group_id": str(group.get("claim_group_id", "")),
        "claim_group_coverage_clause": dict(group.get("coverage_clause", {})),
        "candidate_source_keys": list(group.get("candidate_source_keys", [])),
        "boundary_rule": (
            "This is a legacy retrieval-candidate slot. A reviewer must choose a direct "
            "Source claim, narrow it to exact raw TextCn span(s), or split/reject it; the "
            "legacy any_of slot is not itself approved Source-claim logic."
        ),
    }


def _render_markdown(package: Mapping[str, object]) -> str:
    lines = [
        "# Aevnema v3 Real Source-Gold Review Package",
        "",
        f"Status: **{PENDING_STATUS}** — this package is non-scorable and cannot promote gold.",
        "",
        "Each full-record span below is a review candidate only. It is not an automatically accepted claim.",
        "",
        "## Reviewer decision vocabulary",
        "",
        "Choose one of: `accept_narrowed_span`, `split_into_new_atoms`, `reject_candidate`, or `needs_more_context`.",
        "",
    ]
    atoms = package.get("atoms")
    assert isinstance(atoms, list)
    for item in atoms:
        assert isinstance(item, dict)
        lines.extend(
            [
                f"## {item['atom_id']}",
                "",
                f"- Status: `{item['review_status']}`; usable for scoring: `false`.",
                f"- Source: `{item['source']['source_key']}`; file SHA-256: `{item['source']['source_file_sha256']}`.",
                f"- Legacy Episode: `{item['legacy_candidate']['episode_id']}` (summary is a non-authoritative lead).",
                "",
                "### Claim boundary",
                "",
                str(item["claim_boundary"]["boundary_rule"]),
                "",
                "### Legacy summary (non-authoritative)",
                "",
                str(item["legacy_candidate"]["episode_text"]),
                "",
                "### Candidate Source records and local context",
                "",
            ]
        )
        context = item["context_records"]
        assert isinstance(context, list)
        for record in context:
            assert isinstance(record, dict)
            locator = record["record_locator"]
            span = record["full_record_span"]
            original = record["source_original"]
            assert isinstance(original, dict)
            designation = "candidate window" if record["in_legacy_window"] else "adjacent context"
            if locator is None or span is None:
                lines.extend(
                    [
                        f"#### `{record['record_pointer']}` ({designation}; non-text context)",
                        "",
                        "This record has no raw `TextCn`; it is retained only as context and cannot be selected as a formal span.",
                        "",
                    ]
                )
                continue
            assert isinstance(locator, dict)
            assert isinstance(span, dict)
            lines.extend(
                [
                    f"#### `{locator['pointer']}` ({designation})",
                    "",
                    f"- Full raw `TextCn` candidate span: `{span['span_start']}..{span['span_end']}`",
                    f"- Raw span SHA-256: `{span['raw_span_sha256']}`",
                    "",
                    "Japanese original:",
                    "",
                    f"> {original['TextJp']}",
                    "",
                    "Raw Chinese field used by the formal locator:",
                    "",
                    f"> {original['TextCn']}",
                    "",
                ]
            )
        notes = item["dispute_notes"]
        assert isinstance(notes, list)
        lines.extend(["### Dispute / review notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.extend(
            [
                "",
                "### Human decision (leave pending until a reviewer signs)",
                "",
                "- Decision: `pending_source_span_review`",
                "- Accepted locator/span: _not selected_",
                "- Reviewer rationale: _not supplied_",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def build_review_package(
    *,
    draft_path: Path,
    source_root: Path,
    legacy_snapshot: Path,
    context_radius: int = 1,
) -> tuple[dict[str, object], dict[str, object], str]:
    """Build review material without modifying the draft or Source files."""

    if context_radius < 0:
        raise ValueError("context_radius must be non-negative")
    draft_value = _read_json(draft_path)
    if not isinstance(draft_value, dict):
        raise ValueError("draft must be a JSON object")
    if draft_value.get("status") != DRAFT_STATUS:
        raise ValueError("review package accepts only the pending Source-gold draft")
    if draft_value.get("usable_for_scoring") is True:
        raise ValueError("review package refuses a scorable draft")
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)

    atoms = list(_required_atoms(draft_value))
    legacy_rows = _load_legacy_rows(legacy_snapshot, atoms)
    source_cache: dict[str, tuple[Path, dict[str, object], bytes]] = {}
    current_segment_cache: dict[tuple[str, int, str], dict[str, object]] = {}
    output_atoms: list[dict[str, object]] = []

    for family, group, atom in atoms:
        atom_id = _atom_identity(family, group, atom)
        episode_id = atom["episode_id"]
        assert isinstance(episode_id, int)
        legacy = legacy_rows[episode_id]
        source_key = str(legacy["source_key"])
        if source_key not in source_cache:
            source_path = _source_key_path(source_root, source_key)
            document, raw_bytes = _source_document(source_path)
            source_cache[source_key] = (source_path, document, raw_bytes)
        source_path, document, raw_bytes = source_cache[source_key]
        record_indices = list(legacy["record_indices"])
        assert all(isinstance(index, int) for index in record_indices)
        first_record = min(record_indices)
        last_record = max(record_indices)
        content = document["content"]
        assert isinstance(content, list)
        context_indices = range(
            max(0, first_record - context_radius),
            min(len(content), last_record + context_radius + 1),
        )
        candidate_records = [
            _record_material(
                document,
                index,
                source_key=source_key,
                in_legacy_window=True,
            )
            for index in record_indices
        ]
        context_records = [
            _record_material(
                document,
                index,
                source_key=source_key,
                in_legacy_window=index in record_indices,
            )
            for index in context_indices
        ]
        rendered_zh_cn = legacy["rendered_zh_cn"]
        assert isinstance(rendered_zh_cn, dict)
        normalized_mismatch_indices = [
            index
            for index in record_indices
            if str(rendered_zh_cn.get(index, ""))
            != clean_game_text(str(content[index].get("TextCn", "")))
        ]
        segment_index = int(legacy["segment_index"])
        expected_segment_hash = str(legacy["source_segment_sha256"])
        cache_key = (source_key, segment_index, expected_segment_hash)
        if cache_key not in current_segment_cache:
            current_segment_cache[cache_key] = _current_segment_material(
                source_path, source_key, segment_index, expected_segment_hash
            )
        current_segment = current_segment_cache[cache_key]
        segment_matches = current_segment.get("matches_legacy_rendered_segment") is True
        dispute_notes = [
            "This atom is a legacy Episode-summary retrieval lead, not an accepted Source claim.",
            "Full-record spans are supplied only as human-review candidates; a reviewer must narrow, split, or reject them.",
            "The historical segment hash is verified against the read-only legacy snapshot.",
            (
                "All recovered record TextCn values match the legacy rendered records after the importer normalisation."
                if not normalized_mismatch_indices
                else "Recovered record text differs from the legacy rendered window at record indexes: "
                + ", ".join(str(index) for index in normalized_mismatch_indices)
            ),
            (
                "Current segmenter output at the historical segment index matches the historical rendering."
                if segment_matches
                else "Current segmenter output at the historical segment index differs from the historical rendering; record locators, not the old segment hash/index, are the proposed new anchors."
            ),
            "No human acceptance is present in this package; usable_for_scoring remains false.",
        ]
        output_atoms.append(
            {
                "atom_id": atom_id,
                "review_status": PENDING_STATUS,
                "usable_for_scoring": False,
                "promotion_prohibited": True,
                "claim_boundary": _claim_boundary(family, group),
                "legacy_candidate": {
                    "episode_id": episode_id,
                    "episode_text": legacy["episode_text"],
                    "episode_text_sha256": legacy["episode_text_sha256"],
                    "source_segment_index": segment_index,
                    "source_segment_sha256": expected_segment_hash,
                    "legacy_record_indices": record_indices,
                    "legacy_evidence_span_count": atom.get("legacy_episode_evidence_span_count"),
                    "legacy_evidence_quote_count": atom.get("legacy_episode_evidence_quote_count"),
                    "authority": "historical non-authoritative re-anchoring lead",
                },
                "source": {
                    "source_key": source_key,
                    "source_file_sha256": _sha256_bytes(raw_bytes),
                    "source_file_bytes": len(raw_bytes),
                    "historical_window": {"first_record": first_record, "last_record": last_record},
                    "current_segment_at_historical_index": current_segment,
                },
                "candidate_records": candidate_records,
                "context_records": context_records,
                "dispute_notes": dispute_notes,
                "reviewer_decision": {
                    "status": PENDING_STATUS,
                    "accepted_locator": None,
                    "span_start": None,
                    "span_end": None,
                    "raw_span_sha256": None,
                    "rationale": None,
                },
            }
        )

    sources = [
        {
            "source_key": source_key,
            "source_file_sha256": _sha256_bytes(raw_bytes),
        }
        for source_key, (_path, _document, raw_bytes) in sorted(source_cache.items())
    ]
    frozen_manifest = {"schema": FROZEN_SOURCE_MANIFEST_SCHEMA, "sources": sources}
    package = {
        "schema": PACKAGE_SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": DRAFT_STATUS,
        "scoring_eligible": False,
        "promotion_prohibited": True,
        "visibility": "full_local",
        "draft": {
            "path": draft_path.name,
            "sha256": _sha256_bytes(draft_path.read_bytes()),
            "source_gold_status": draft_value.get("status"),
        },
        "legacy_snapshot": {
            "path": legacy_snapshot.name,
            "sha256": _sha256_bytes(legacy_snapshot.read_bytes()),
            "mode": "read_only_re-anchoring evidence",
        },
        "source_root": {"path": str(source_root.resolve()), "source_count": len(sources)},
        "atom_count": len(output_atoms),
        "atoms": output_atoms,
        "reviewer_instructions": [
            "Do not approve a full-record candidate merely because it is locatable or hashed.",
            "Preserve speaker, epistemic modality, and whether a statement is direct dialogue, a report, a question, or an inference.",
            "Split multi-fact summaries into separate Source atoms before any scoring promotion.",
            "A later approved artifact must bind selected raw TextCn code-point spans and a frozen Source manifest through validate_source_gold.py.",
        ],
    }
    package["candidate_frozen_source_manifest_sha256"] = _sha256_bytes(
        _canonical_json_bytes(frozen_manifest)
    )
    markdown = _render_markdown(package)
    return package, frozen_manifest, markdown


def _write_atomic(path: Path, payload: bytes) -> None:
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_review_package(
    output_dir: Path,
    package: Mapping[str, object],
    frozen_manifest: Mapping[str, object],
    markdown: str,
) -> None:
    """Write a new review directory; never overwrite a prior review package."""

    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite review package: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    documents = {
        "source_gold_review.full_local.json": _canonical_json_bytes(package),
        "frozen_source_manifest.candidate.json": _canonical_json_bytes(frozen_manifest),
        "source_gold_review.md": markdown.encode("utf-8"),
    }
    for name, content in documents.items():
        _write_atomic(output_dir / name, content)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--legacy-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-radius", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    package, frozen_manifest, markdown = build_review_package(
        draft_path=args.draft,
        source_root=args.source_root,
        legacy_snapshot=args.legacy_snapshot,
        context_radius=args.context_radius,
    )
    write_review_package(args.output_dir, package, frozen_manifest, markdown)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "status": package["status"],
                "scoring_eligible": package["scoring_eligible"],
                "atom_count": package["atom_count"],
                "source_count": package["source_root"]["source_count"],  # type: ignore[index]
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
