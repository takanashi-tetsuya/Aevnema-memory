from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import re
from types import SimpleNamespace

import numpy as np

from memory_demo.adapters import BlueArchiveJsonAdapter, TextAdapter
from memory_demo.adapters.base import logical_source_key
from memory_demo.config import AppConfig
from memory_demo.database import SCHEMA_VERSION
from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.ingestion.ordering import natural_path_sort_key
from memory_demo.ingestion.segmenter import NaturalSegmenter


_PARTICIPANT_SEPARATOR = re.compile(r"\s*[/／]\s*|[()（）]")
_DIALOGUE_LABEL = re.compile(r"^([^:\n]{2,80}):(?:\s|$)")
_NON_NAME_LABELS = {
    "unknown", "zh-cn", "zh-tw", "ja", "en", "ko", "th",
}
_UNKNOWN_PARTICIPANTS = {
    "???", "[username]", "未标注发言者", "未知发言者", "unknown speaker",
}


def _character_script(value: str) -> str:
    codepoint = ord(value)
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana"
    if 0x30A0 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF:
        return "katakana"
    if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF:
        return "han"
    if 0xAC00 <= codepoint <= 0xD7AF:
        return "hangul"
    if value.isascii() and value.isalnum():
        return "ascii_alnum"
    return "other"


def _contains_name_literal(text: str, name: str) -> bool:
    """Match multilingual names without accepting same-script substrings.

    Unicode ``\b`` does not distinguish Japanese names such as ミカ from the
    suffix of ウミカ. Script-aware boundaries retain normal forms such as
    ミカちゃん while rejecting the same-script containment false positive.
    """

    if not name:
        return False
    start = 0
    while (index := text.find(name, start)) >= 0:
        end = index + len(name)
        first_script = _character_script(name[0])
        last_script = _character_script(name[-1])
        bounded_scripts = {"katakana", "ascii_alnum", "hangul"}
        left_ok = (
            first_script not in bounded_scripts
            or index == 0
            or _character_script(text[index - 1]) != first_script
        )
        right_ok = (
            last_script not in bounded_scripts
            or end == len(text)
            or _character_script(text[end]) != last_script
        )
        if left_ok and right_ok:
            return True
        start = index + 1
    return False


def _dialogue_speaker_groups(text: str) -> list[tuple[str, set[str]]]:
    raw_labels = [
        *re.findall(r"\[speaker_raw:\s*([^\]]+)\]", text),
        *(
            matched.group(1)
            for line in text.splitlines()
            if not line.lstrip().startswith("[")
            and (matched := _DIALOGUE_LABEL.match(line.strip()))
        ),
    ]
    groups: list[tuple[str, set[str]]] = []
    seen: set[tuple[str, ...]] = set()
    for raw_label in raw_labels:
        label = raw_label.strip()
        if not label or label.casefold() in _NON_NAME_LABELS:
            continue
        aliases = {
            part.strip()
            for part in _PARTICIPANT_SEPARATOR.split(label)
            if len(part.strip()) >= 2
            and part.strip().casefold() not in _NON_NAME_LABELS
        }
        if len(label) >= 2:
            aliases.add(label)
        key = tuple(sorted(alias.casefold() for alias in aliases))
        if not aliases or key in seen:
            continue
        seen.add(key)
        groups.append((label, aliases))
    return groups


def audit_episode_evidence(
    episodes: list[sqlite3.Row] | list[dict],
    source_text_by_id: dict[int, str],
    *,
    require_evidence: bool,
) -> tuple[list[dict], list[dict]]:
    """Validate the durable Episode-to-Source evidence contract."""

    evidence_issues: list[dict] = []
    scoped_participant_issues: list[dict] = []
    line_cache: dict[int, list[str]] = {}
    unknown = {value.casefold() for value in _UNKNOWN_PARTICIPANTS}

    for row in episodes:
        episode_id = int(row["id"])
        source_id = int(row["source_id"])
        basis = str(row["evidence_basis"] or "")
        try:
            quotes = json.loads(str(row["evidence_quotes_json"]))
            if not isinstance(quotes, list) or not all(
                isinstance(value, str) and bool(value.strip()) for value in quotes
            ):
                raise TypeError("evidence_quotes_json must be non-empty strings")
            spans = json.loads(str(row["evidence_spans_json"]))
            if not isinstance(spans, list) or not all(
                isinstance(value, list)
                and len(value) == 2
                and all(
                    isinstance(index, int) and not isinstance(index, bool)
                    for index in value
                )
                for value in spans
            ):
                raise TypeError(
                    "evidence_spans_json must be integer [start, end] pairs"
                )
        except (json.JSONDecodeError, TypeError) as exc:
            evidence_issues.append(
                {
                    "episode_id": episode_id,
                    "source_id": source_id,
                    "reason": "invalid_evidence_json",
                    "error": str(exc),
                }
            )
            continue

        if basis == "legacy_unavailable":
            if require_evidence:
                evidence_issues.append(
                    {
                        "episode_id": episode_id,
                        "source_id": source_id,
                        "reason": "evidence_required_but_unavailable",
                    }
                )
            continue
        if basis != "reasoning_view_nonempty_lines_v1":
            evidence_issues.append(
                {
                    "episode_id": episode_id,
                    "source_id": source_id,
                    "reason": "unsupported_evidence_basis",
                    "basis": basis,
                }
            )
            continue
        if not quotes or not spans or len(quotes) != len(spans):
            evidence_issues.append(
                {
                    "episode_id": episode_id,
                    "source_id": source_id,
                    "reason": "evidence_quote_span_count_mismatch",
                    "quotes": len(quotes),
                    "spans": len(spans),
                }
            )
            continue

        lines = line_cache.get(source_id)
        if lines is None:
            reasoning_source = MemoryExtractor.compact_source_for_reasoning(
                source_text_by_id.get(source_id, "")
            )
            lines = [
                line for line in reasoning_source.splitlines() if line.strip()
            ]
            line_cache[source_id] = lines
        reconstructed: list[str] = []
        for span_index, (start, end) in enumerate(spans):
            if start < 1 or end < start or end > len(lines):
                evidence_issues.append(
                    {
                        "episode_id": episode_id,
                        "source_id": source_id,
                        "reason": "evidence_span_out_of_range",
                        "span_index": span_index,
                        "span": [start, end],
                        "line_count": len(lines),
                    }
                )
                continue
            reconstructed.append("\n".join(lines[start - 1 : end]))
        if len(reconstructed) != len(quotes):
            continue
        for span_index, (stored, rebuilt) in enumerate(
            zip(quotes, reconstructed, strict=True)
        ):
            if stored != rebuilt:
                evidence_issues.append(
                    {
                        "episode_id": episode_id,
                        "source_id": source_id,
                        "reason": "evidence_quote_does_not_match_span",
                        "span_index": span_index,
                    }
                )

        evidence_folded = "\n".join(quotes).casefold()
        try:
            participants = json.loads(str(row["participants_json"]))
        except (TypeError, json.JSONDecodeError):
            participants = []
        if not isinstance(participants, list):
            participants = []
        for participant in participants:
            if not isinstance(participant, str) or not participant.strip():
                continue
            normalized = participant.strip()
            parts = {
                value.strip()
                for value in _PARTICIPANT_SEPARATOR.split(normalized)
                if value.strip()
            }
            grounded = (
                normalized.casefold() in unknown
                or normalized.casefold() in evidence_folded
                or (
                    bool(parts)
                    and all(
                        value.casefold() in unknown
                        or value.casefold() in evidence_folded
                        for value in parts
                    )
                )
            )
            if not grounded:
                scoped_participant_issues.append(
                    {
                        "episode_id": episode_id,
                        "source_id": source_id,
                        "participant": normalized,
                    }
                )
    return evidence_issues, scoped_participant_issues


def audit_episode_evidence_coverage(
    episodes: list[sqlite3.Row] | list[dict],
    source_text_by_id: dict[int, str],
) -> list[dict]:
    """Re-run the deterministic long-gap gate from persisted evidence."""

    episodes_by_source: dict[int, list[SimpleNamespace]] = {}
    for row in episodes:
        try:
            spans = json.loads(str(row["evidence_spans_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(spans, list):
            continue
        normalized: list[tuple[int, int]] = []
        for value in spans:
            if (
                isinstance(value, list)
                and len(value) == 2
                and all(isinstance(index, int) for index in value)
            ):
                normalized.append((int(value[0]), int(value[1])))
        episodes_by_source.setdefault(int(row["source_id"]), []).append(
            SimpleNamespace(evidence_spans=normalized)
        )

    issues: list[dict] = []
    for source_id, source_text in source_text_by_id.items():
        reasoning_source = MemoryExtractor.compact_source_for_reasoning(source_text)
        source_lines = [
            line for line in reasoning_source.splitlines() if line.strip()
        ]
        errors = MemoryExtractor._single_pass_coverage_errors(
            source_lines,
            episodes_by_source.get(source_id, []),
        )
        for error in errors:
            match = re.search(r"lines (\d+)-(\d+)", error)
            sample_lines: list[str] = []
            if match:
                start, end = (int(value) for value in match.groups())
                sample_lines = source_lines[start - 1 : min(end, start + 11)]
            issues.append(
                {
                    "source_id": source_id,
                    "reason": error,
                    "sample_lines": sample_lines,
                }
            )
    return issues


def audit_cross_source_evidence_duplicates(
    episodes: list[sqlite3.Row] | list[dict],
    *,
    containment_threshold: float = 0.8,
) -> list[dict]:
    """Find duplicate events caused by copying overlap into another Source."""

    items: list[dict] = []
    for row in episodes:
        try:
            quotes = json.loads(str(row["evidence_quotes_json"]))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(quotes, list):
            continue
        lines = {
            re.sub(r"\s+", "", line).casefold()
            for quote in quotes
            if isinstance(quote, str)
            for line in quote.splitlines()
            if line.strip() and not line.lstrip().startswith("[")
        }
        if lines:
            items.append(
                {
                    "id": int(row["id"]),
                    "source_id": int(row["source_id"]),
                    "source_key": str(row["source_key"]),
                    "lines": lines,
                }
            )
    issues: list[dict] = []
    for right_index, right in enumerate(items):
        for left in items[:right_index]:
            if (
                left["source_id"] == right["source_id"]
                or left["source_key"] != right["source_key"]
            ):
                continue
            common = len(left["lines"].intersection(right["lines"]))
            denominator = min(len(left["lines"]), len(right["lines"]))
            ratio = common / denominator if denominator else 0.0
            if ratio < containment_threshold:
                continue
            issues.append(
                {
                    "left_episode_id": left["id"],
                    "right_episode_id": right["id"],
                    "left_source_id": left["source_id"],
                    "right_source_id": right["source_id"],
                    "source_key": left["source_key"],
                    "evidence_containment": round(ratio, 6),
                    "common_lines": common,
                    "smaller_evidence_lines": denominator,
                }
            )
    return issues


def audit_episode_participants(
    episodes: list[sqlite3.Row] | list[dict],
    source_text_by_id: dict[int, str],
) -> tuple[list[dict], list[dict]]:
    """Check participant evidence without using a corpus-specific name list.

    Grounding verifies that persisted participant labels were copied from the
    same Source. Completeness is deliberately narrower: it only considers
    literal speaker labels from structured dialogue and only when that literal
    label also occurs in Episode prose. This catches omissions without turning
    the audit into an entity recognizer or importing outside knowledge.
    """

    grounding_issues: list[dict] = []
    completeness_issues: list[dict] = []
    speaker_groups_by_source = {
        source_id: _dialogue_speaker_groups(source_text)
        for source_id, source_text in source_text_by_id.items()
    }

    for row in episodes:
        episode_id = int(row["id"])
        source_id = int(row["source_id"])
        source_text = source_text_by_id.get(source_id, "")
        source_folded = source_text.casefold()
        try:
            participants = json.loads(str(row["participants_json"]))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(participants, list):
            continue
        participant_parts: set[str] = set()
        for participant in participants:
            if not isinstance(participant, str):
                continue
            normalized = participant.strip()
            parts = {
                part.strip()
                for part in _PARTICIPANT_SEPARATOR.split(normalized)
                if part.strip()
            }
            participant_parts.update(part.casefold() for part in parts)
            grounded = (
                normalized.casefold() in source_folded
                or normalized.casefold() in _UNKNOWN_PARTICIPANTS
                or (
                    bool(parts)
                    and all(
                        part.casefold() in source_folded
                        or part.casefold() in _UNKNOWN_PARTICIPANTS
                        for part in parts
                    )
                )
            )
            if not grounded:
                grounding_issues.append(
                    {
                        "episode_id": episode_id,
                        "source_id": source_id,
                        "participant": normalized,
                    }
                )

        episode_text = str(row["text"])
        evidence_groups: list[tuple[str, set[str]]] = []
        try:
            evidence_quotes = json.loads(str(row["evidence_quotes_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            evidence_quotes = []
        if isinstance(evidence_quotes, list):
            evidence_text = "\n".join(
                value for value in evidence_quotes if isinstance(value, str)
            )
            if evidence_text.strip():
                evidence_groups = _dialogue_speaker_groups(evidence_text)
        groups = evidence_groups or speaker_groups_by_source.get(source_id, [])
        for label, aliases in groups:
            mentioned = sorted(
                (
                    alias
                    for alias in aliases
                    if _contains_name_literal(episode_text, alias)
                ),
                key=len,
                reverse=True,
            )
            if not mentioned:
                continue
            if any(alias.casefold() in participant_parts for alias in aliases):
                continue
            completeness_issues.append(
                {
                    "episode_id": episode_id,
                    "source_id": source_id,
                    "speaker_label": label,
                    "matched_text": mentioned,
                }
            )
    return grounding_issues, completeness_issues


def temporal_cycle_audit(episodes, associations) -> dict[str, dict]:
    scope_by_episode = {
        int(row["id"]): str(row["timeline_scope"] or "") for row in episodes
    }
    scopes = sorted(set(scope_by_episode.values()))
    scope_nodes = [
        (
            scope,
            {
                episode_id
                for episode_id, episode_scope in scope_by_episode.items()
                if episode_scope == scope
            },
        )
        for scope in scopes
    ]
    # Cross-file/cross-domain temporal edges are still ordering claims. Audit a
    # second global projection so a cycle cannot hide merely because the two
    # endpoints were assigned different timeline_scope strings.
    scope_nodes.append(("__global__", set(scope_by_episode)))
    result: dict[str, dict] = {}
    for scope, nodes in scope_nodes:
        adjacency = {node: set() for node in nodes}
        indegree = {node: 0 for node in nodes}
        used_edges: list[int] = []
        skipped_edges: list[int] = []
        non_ordering_edges: list[int] = []
        for edge in associations:
            if (
                edge["relation_type"] != "temporal"
                or edge["from_type"] != "episode"
                or edge["to_type"] != "episode"
                or int(edge["polarity"]) <= 0
            ):
                continue
            left, right = int(edge["from_id"]), int(edge["to_id"])
            if left not in nodes or right not in nodes:
                continue
            key = str(edge["relation_key"]).casefold()
            if key == "after" or key.endswith("_after"):
                left, right = right, left
            elif key in {"same_time", "same_time_as", "simultaneous"}:
                non_ordering_edges.append(int(edge["id"]))
                continue
            elif key not in {"before", "precedes"} and not key.endswith("_before"):
                skipped_edges.append(int(edge["id"]))
                continue
            used_edges.append(int(edge["id"]))
            if right not in adjacency[left]:
                adjacency[left].add(right)
                indegree[right] += 1
        available = sorted(node for node in nodes if indegree[node] == 0)
        ordered: list[int] = []
        while available:
            node = available.pop(0)
            ordered.append(node)
            for neighbor in sorted(adjacency[node]):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    available.append(neighbor)
                    available.sort()
        remaining = sorted(nodes.difference(ordered))
        result[scope] = {
            "episode_count": len(nodes),
            "temporal_edge_count": len(used_edges),
            "has_cycle": bool(remaining),
            "cycle_or_blocked_episode_ids": remaining,
            "non_ordering_temporal_edge_ids": non_ordering_edges,
            "skipped_temporal_edge_ids": skipped_edges,
        }
    return result


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Read-only memory database audit")
    parser.add_argument("database")
    parser.add_argument("--duplicate-threshold", type=float, default=0.94)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="skip O(N^2) duplicate comparison and omit row-level details",
    )
    parser.add_argument("--output", help="optional UTF-8 JSON report path")
    parser.add_argument(
        "--source-root",
        action="append",
        help=(
            "input directory used to reconcile expected segments; repeat this "
            "option when one database contains multiple corpora"
        ),
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 2 when integrity or import-completeness checks fail",
    )
    parser.add_argument(
        "--require-episode-evidence",
        action="store_true",
        help=(
            "fail when an Episode lacks persisted, reproducible Source spans; "
            "use for single-pass audited imports"
        ),
    )
    args = parser.parse_args()
    database = Path(args.database)
    if not database.is_file():
        raise SystemExit(f"database does not exist: {database}")
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row

    episodes = connection.execute(
        """
        SELECT id, source_id, source_key, segment_index, text, participants_json,
               event_type, story_time_text, story_order, timeline_scope,
               confidence, evidence_origin, epistemic_status, generation,
               epistemic_note, evidence_quotes_json, evidence_spans_json,
               evidence_basis, embedding
        FROM episode ORDER BY id
        """
    ).fetchall()
    source_text_by_id = {
        int(row["id"]): str(row["raw_text"])
        for row in connection.execute(
            "SELECT id, raw_text FROM source ORDER BY id"
        )
    }
    (
        participant_grounding_issues,
        participant_completeness_issues,
    ) = audit_episode_participants(episodes, source_text_by_id)
    episode_evidence_issues, evidence_participant_grounding_issues = (
        audit_episode_evidence(
            episodes,
            source_text_by_id,
            require_evidence=args.require_episode_evidence,
        )
    )
    cross_source_evidence_duplicates = audit_cross_source_evidence_duplicates(
        episodes
    )
    episode_evidence_coverage_issues = audit_episode_evidence_coverage(
        episodes, source_text_by_id
    )
    concepts = connection.execute(
        """
        SELECT id, canonical_name, description, embedding_text,
               confidence, status, canonical_concept_id, embedding
        FROM concept ORDER BY id
        """
    ).fetchall()
    aliases_by_concept: dict[int, list[dict]] = {}
    alias_rows = connection.execute(
        """
        SELECT concept_id, alias, language, normalized_alias
        FROM concept_alias ORDER BY concept_id, id
        """
    ).fetchall()
    for row in alias_rows:
        aliases_by_concept.setdefault(int(row["concept_id"]), []).append(
            {"alias": row["alias"], "language": row["language"]}
        )
    canonical_id_by_concept = {
        int(row["id"]): int(row["canonical_concept_id"] or row["id"])
        for row in concepts
    }
    active_concept_ids = {
        int(row["id"]) for row in concepts if row["status"] == "active"
    }
    alias_owners: dict[str, set[int]] = {}
    alias_examples: dict[str, set[str]] = {}
    for row in alias_rows:
        canonical_id = canonical_id_by_concept[int(row["concept_id"])]
        if canonical_id not in active_concept_ids:
            continue
        normalized = str(row["normalized_alias"])
        alias_owners.setdefault(normalized, set()).add(canonical_id)
        alias_examples.setdefault(normalized, set()).add(str(row["alias"]))
    concept_alias_collisions = [
        {
            "normalized_alias": normalized,
            "aliases": sorted(alias_examples[normalized]),
            "active_concept_ids": sorted(owners),
        }
        for normalized, owners in sorted(alias_owners.items())
        if len(owners) > 1
    ]
    concept_output_issues: list[dict] = []
    concept_meta_markers = ("此处修正", "应提取", "未提及", "候选条目", "更核心")
    for row in concepts:
        canonical_name = str(row["canonical_name"])
        combined = f'{row["description"]}\n{row["embedding_text"]}'
        if re.search(r"\s+[/／]\s+", canonical_name):
            concept_output_issues.append(
                {
                    "concept_id": int(row["id"]),
                    "reason": "alias_chain_in_canonical_name",
                    "canonical_name": canonical_name,
                }
            )
        marker = next((item for item in concept_meta_markers if item in combined), None)
        if marker is not None:
            concept_output_issues.append(
                {
                    "concept_id": int(row["id"]),
                    "reason": "model_meta_commentary",
                    "marker": marker,
                    "canonical_name": canonical_name,
                }
            )
    associations = connection.execute(
        """
        SELECT id, from_type, from_id, to_type, to_id, relation_type,
               relation_key, relation_text, polarity, weight, confidence,
               generation, evidence_count, evidence_json, audit_json,
               created_reason
        FROM association ORDER BY id
        """
    ).fetchall()
    tasks = connection.execute(
        "SELECT stage, status, COUNT(*) AS count FROM extraction_task GROUP BY stage, status"
    ).fetchall()
    episode_counts_by_source = [
        int(row["count"])
        for row in connection.execute(
            "SELECT COUNT(*) AS count FROM episode GROUP BY source_id ORDER BY source_id"
        ).fetchall()
    ]

    integrity_check = str(
        connection.execute("PRAGMA integrity_check").fetchone()[0]
    )
    schema_row = connection.execute(
        "SELECT schema_version FROM schema_meta LIMIT 1"
    ).fetchone()
    actual_schema_version = int(schema_row[0]) if schema_row is not None else None
    foreign_key_issues = [
        list(row) for row in connection.execute("PRAGMA foreign_key_check")
    ]
    running_runs = [
        dict(row)
        for row in connection.execute(
            "SELECT id, status FROM extraction_run WHERE status != 'completed'"
        )
    ]
    noncompleted_tasks = [
        dict(row)
        for row in connection.execute(
            """
            SELECT id, run_id, source_key, segment_index, stage, status, error_summary
            FROM extraction_task WHERE status != 'completed' ORDER BY id
            """
        )
    ]

    expected_blob_length = 0
    try:
        expected_blob_length = int(
            json.loads(
                connection.execute(
                    "SELECT config_snapshot FROM extraction_run ORDER BY id DESC LIMIT 1"
                ).fetchone()[0]
            )["model"]["embedding_dimension"]
        ) * 4
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    embedding_issues: list[dict] = []
    for table in ("episode", "concept", "paragraph"):
        for row in connection.execute(
            f"SELECT id, embedding FROM {table} ORDER BY id"
        ):
            blob = row["embedding"]
            if expected_blob_length and len(blob) != expected_blob_length:
                embedding_issues.append(
                    {
                        "table": table,
                        "id": int(row["id"]),
                        "reason": "unexpected_blob_length",
                        "length": len(blob),
                        "expected": expected_blob_length,
                    }
                )
                continue
            vector = np.frombuffer(blob, dtype="<f4")
            if not np.isfinite(vector).all() or not np.any(vector):
                embedding_issues.append(
                    {
                        "table": table,
                        "id": int(row["id"]),
                        "reason": "non_finite_or_zero_vector",
                    }
                )
                continue
            norm = float(np.linalg.norm(vector))
            if abs(norm - 1.0) > 1e-3:
                embedding_issues.append(
                    {
                        "table": table,
                        "id": int(row["id"]),
                        "reason": "not_l2_normalized",
                        "norm": norm,
                    }
                )

    orphan_associations: list[int] = []
    for row in associations:
        missing = False
        for prefix in ("from", "to"):
            node_type = str(row[f"{prefix}_type"])
            node_id = int(row[f"{prefix}_id"])
            if node_type not in {"episode", "concept"}:
                missing = True
                break
            exists = connection.execute(
                f"SELECT 1 FROM {node_type} WHERE id = ?", (node_id,)
            ).fetchone()
            if exists is None:
                missing = True
                break
        if missing:
            orphan_associations.append(int(row["id"]))

    sources_without_episodes = [
        int(row["id"])
        for row in connection.execute(
            """
            SELECT s.id FROM source s
            LEFT JOIN episode e ON e.source_id = s.id
            GROUP BY s.id HAVING COUNT(e.id) = 0
            ORDER BY s.id
            """
        )
    ]
    concepts_without_episode_links = [
        int(row["id"])
        for row in connection.execute(
            """
            SELECT c.id FROM concept c
            WHERE NOT EXISTS (
                SELECT 1 FROM association a
                WHERE (a.from_type = 'concept' AND a.from_id = c.id
                       AND a.to_type = 'episode')
                   OR (a.to_type = 'concept' AND a.to_id = c.id
                       AND a.from_type = 'episode')
            )
            ORDER BY c.id
            """
        )
    ]
    empty_content_issues: list[dict] = []
    for table, column in (
        ("source", "raw_text"),
        ("episode", "text"),
        ("concept", "canonical_name"),
        ("concept", "embedding_text"),
        ("paragraph", "text"),
        ("association", "relation_key"),
        ("association", "relation_text"),
    ):
        for row in connection.execute(
            f"SELECT id FROM {table} WHERE trim({column}) = '' ORDER BY id"
        ):
            empty_content_issues.append(
                {"table": table, "id": int(row["id"]), "column": column}
            )

    json_payload_issues: list[dict] = []
    for row in episodes:
        try:
            participants = json.loads(str(row["participants_json"]))
            if not isinstance(participants, list) or not all(
                isinstance(item, str) for item in participants
            ):
                raise TypeError("participants_json must be an array of strings")
        except (json.JSONDecodeError, TypeError) as exc:
            json_payload_issues.append(
                {
                    "table": "episode",
                    "id": int(row["id"]),
                    "column": "participants_json",
                    "error": str(exc),
                }
            )
    for row in associations:
        for column in ("evidence_json", "audit_json"):
            try:
                value = json.loads(str(row[column]))
                if not isinstance(value, list):
                    raise TypeError(f"{column} must be an array")
            except (json.JSONDecodeError, TypeError) as exc:
                json_payload_issues.append(
                    {
                        "table": "association",
                        "id": int(row["id"]),
                        "column": column,
                        "error": str(exc),
                    }
                )

    fts_count_issues: list[dict] = []
    for content_table, fts_table in (
        ("episode", "episode_fts"),
        ("episode", "episode_bigram_fts"),
        ("source", "source_fts"),
        ("source", "source_bigram_fts"),
    ):
        expected_count = int(
            connection.execute(f"SELECT COUNT(*) FROM {content_table}").fetchone()[0]
        )
        actual_count = int(
            connection.execute(f"SELECT COUNT(*) FROM {fts_table}").fetchone()[0]
        )
        if actual_count != expected_count:
            fts_count_issues.append(
                {
                    "table": fts_table,
                    "expected": expected_count,
                    "actual": actual_count,
                }
            )
    reference_coverage_issues: list[dict] = []
    explicit_evidence_boundary_issues: list[dict] = []
    explicit_evidence_mapping_ambiguities: list[dict] = []
    inferential_type_markers = (
        "推测",
        "推论",
        "考据",
        "社区观点",
        "剧情解读",
    )
    for row in connection.execute(
        """
        SELECT s.id, s.raw_text, COUNT(e.id) AS episode_count
        FROM source s LEFT JOIN episode e ON e.source_id = s.id
        GROUP BY s.id ORDER BY s.id
        """
    ):
        fact_records = len(
            re.findall(r"\[资料类型[：:]", str(row["raw_text"]))
        )
        if fact_records >= 2 and int(row["episode_count"]) < fact_records:
            reference_coverage_issues.append(
                {
                    "source_id": int(row["id"]),
                    "fact_records": fact_records,
                    "episodes": int(row["episode_count"]),
                }
            )
        reference_types = re.findall(
            r"\[资料类型[：:]\s*([^\]|]+)", str(row["raw_text"])
        )
        source_episodes = connection.execute(
            """
            SELECT id, evidence_origin, epistemic_status, generation
            FROM episode WHERE source_id = ? ORDER BY id
            """,
            (int(row["id"]),),
        ).fetchall()
        if reference_types and len(reference_types) == len(source_episodes):
            episodes_to_check = [
                episode
                for episode, reference_type in zip(
                    source_episodes, reference_types, strict=True
                )
                if any(
                    marker in reference_type
                    for marker in inferential_type_markers
                )
            ]
        elif reference_types and all(
            any(marker in reference_type for marker in inferential_type_markers)
            for reference_type in reference_types
        ):
            episodes_to_check = list(source_episodes)
        elif reference_types and any(
            any(marker in reference_type for marker in inferential_type_markers)
            for reference_type in reference_types
        ):
            episodes_to_check = []
            explicit_evidence_mapping_ambiguities.append(
                {
                    "source_id": int(row["id"]),
                    "fact_records": len(reference_types),
                    "episodes": len(source_episodes),
                }
            )
        else:
            episodes_to_check = []
        for episode in episodes_to_check:
                if (
                    episode["evidence_origin"] != "importer"
                    or episode["epistemic_status"] != "speculative"
                    or int(episode["generation"] or 0) < 1
                ):
                    explicit_evidence_boundary_issues.append(
                        {
                            "source_id": int(row["id"]),
                            "episode_id": int(episode["id"]),
                            "evidence_origin": episode["evidence_origin"],
                            "epistemic_status": episode["epistemic_status"],
                            "generation": int(episode["generation"] or 0),
                        }
                    )

    episode_generation = {
        int(row["id"]): int(row["generation"] or 0) for row in episodes
    }
    association_generation_issues: list[dict] = []
    for row in associations:
        endpoint_generations = [
            episode_generation[int(row[f"{prefix}_id"])]
            for prefix in ("from", "to")
            if row[f"{prefix}_type"] == "episode"
            and int(row[f"{prefix}_id"]) in episode_generation
        ]
        minimum_generation = max(endpoint_generations, default=0)
        if int(row["generation"] or 0) < minimum_generation:
            association_generation_issues.append(
                {
                    "association_id": int(row["id"]),
                    "generation": int(row["generation"] or 0),
                    "minimum_endpoint_generation": minimum_generation,
                }
            )

    segment_reconciliation: dict | None = None
    if args.source_root:
        source_roots = [Path(value).resolve() for value in args.source_root]
        for source_root in source_roots:
            if not source_root.is_dir():
                raise SystemExit(f"source root is not a directory: {source_root}")
        config = AppConfig.from_env(args.env_file)
        adapters = [BlueArchiveJsonAdapter(), TextAdapter()]
        segmenter = NaturalSegmenter(config.segment)
        expected_segments: set[tuple[str, int]] = set()
        expected_owners: dict[tuple[str, int], str] = {}
        collisions: list[dict[str, str | int]] = []
        supported_files = 0
        root_reports: list[dict[str, str | int]] = []
        for source_root in source_roots:
            root_files = 0
            root_segments = 0
            for path in sorted(
                (item for item in source_root.rglob("*") if item.is_file()),
                key=natural_path_sort_key,
            ):
                adapter = next(
                    (candidate for candidate in adapters if candidate.supports(path)),
                    None,
                )
                if adapter is None:
                    continue
                supported_files += 1
                root_files += 1
                source_key = logical_source_key(path, source_root)
                for segment in segmenter.segment(source_key, adapter.read_blocks(path)):
                    key = (source_key, int(segment.segment_index))
                    owner = str(path)
                    previous_owner = expected_owners.get(key)
                    if previous_owner is not None and previous_owner != owner:
                        collisions.append(
                            {
                                "source_key": source_key,
                                "segment_index": int(segment.segment_index),
                                "first_path": previous_owner,
                                "second_path": owner,
                            }
                        )
                    expected_owners[key] = owner
                    expected_segments.add(key)
                    root_segments += 1
            root_reports.append(
                {
                    "root": str(source_root),
                    "supported_files": root_files,
                    "expected_segments": root_segments,
                }
            )
        actual_segments = {
            (str(row["source_key"]), int(row["segment_index"]))
            for row in connection.execute(
                "SELECT DISTINCT source_key, segment_index FROM episode"
            )
        }
        missing_segments = sorted(expected_segments - actual_segments)
        unexpected_segments = sorted(actual_segments - expected_segments)
        segment_reconciliation = {
            "supported_files": supported_files,
            "expected_segments": len(expected_segments),
            "actual_segments": len(actual_segments),
            "roots": root_reports,
            "source_key_collisions": collisions,
            "missing_segments": missing_segments,
            "unexpected_segments": unexpected_segments,
        }

    duplicate_pairs: list[dict] = []
    if not args.summary_only and len(episodes) >= 2:
        matrix = np.stack(
            [np.frombuffer(row["embedding"], dtype="<f4") for row in episodes]
        )
        similarities = matrix @ matrix.T
        for left in range(len(episodes)):
            for right in range(left):
                score = float(similarities[left, right])
                if score >= args.duplicate_threshold:
                    duplicate_pairs.append(
                        {
                            "score": round(score, 5),
                            "left_id": int(episodes[left]["id"]),
                            "right_id": int(episodes[right]["id"]),
                            "left_text": str(episodes[left]["text"]),
                            "right_text": str(episodes[right]["text"]),
                        }
                    )
    duplicate_pairs.sort(key=lambda item: item["score"], reverse=True)

    temporal_cycles = temporal_cycle_audit(episodes, associations)
    report = {
        "database": str(database.resolve()),
        "counts": {
            "sources": connection.execute("SELECT COUNT(*) FROM source").fetchone()[0],
            "episodes": len(episodes),
            "concepts": len(concepts),
            "associations": len(associations),
        },
        "embedding_blob_lengths": {
            "episode": sorted({len(row["embedding"]) for row in episodes}),
            "concept": sorted({len(row["embedding"]) for row in concepts}),
            "paragraph": sorted(
                {
                    len(row["embedding"])
                    for row in connection.execute("SELECT embedding FROM paragraph")
                }
            ),
        },
        "integrity": {
            "schema_version": actual_schema_version,
            "expected_schema_version": SCHEMA_VERSION,
            "sqlite": integrity_check,
            "foreign_key_issues": foreign_key_issues,
            "orphan_association_ids": orphan_associations,
            "concepts_without_episode_links": concepts_without_episode_links,
            "embedding_issues": embedding_issues,
            "empty_content_issues": empty_content_issues,
            "json_payload_issues": json_payload_issues,
            "fts_count_issues": fts_count_issues,
            "sources_without_episodes": sources_without_episodes,
            "reference_coverage_issues": reference_coverage_issues,
            "explicit_evidence_boundary_issues": explicit_evidence_boundary_issues,
            "explicit_evidence_mapping_ambiguities": explicit_evidence_mapping_ambiguities,
            "association_generation_issues": association_generation_issues,
            "episode_evidence_issues": episode_evidence_issues,
            "noncompleted_runs": running_runs,
            "noncompleted_tasks": noncompleted_tasks,
        },
        "segment_reconciliation": segment_reconciliation,
        "tasks": [dict(row) for row in tasks],
        "episodes_per_source": {
            "sources_with_episodes": len(episode_counts_by_source),
            "minimum": min(episode_counts_by_source, default=0),
            "mean": (
                round(sum(episode_counts_by_source) / len(episode_counts_by_source), 2)
                if episode_counts_by_source
                else 0
            ),
            "maximum": max(episode_counts_by_source, default=0),
            "counts": episode_counts_by_source if not args.summary_only else None,
        },
        "association_types": dict(Counter(row["relation_type"] for row in associations)),
        "quality": {
            "concept_alias_collisions": concept_alias_collisions,
            "concept_output_issues": concept_output_issues,
            "participant_grounding_issues": participant_grounding_issues,
            "participant_completeness_issues": participant_completeness_issues,
            "evidence_participant_grounding_issues": (
                evidence_participant_grounding_issues
            ),
            "cross_source_evidence_duplicates": (
                cross_source_evidence_duplicates
            ),
            "episode_evidence_coverage_issues": (
                episode_evidence_coverage_issues
            ),
        },
        "temporal_cycles": temporal_cycles,
        "negative_associations": sum(int(row["polarity"] < 0) for row in associations),
        "duplicate_episode_pairs": duplicate_pairs,
    }
    if not args.summary_only:
        report.update(
            {
                "associations": [dict(row) for row in associations],
                "episodes": [
                    {
                        key: row[key]
                        for key in (
                            "id", "source_id", "source_key", "segment_index", "text",
                            "participants_json", "event_type", "story_time_text",
                            "story_order", "timeline_scope", "confidence",
                            "evidence_quotes_json", "evidence_spans_json",
                            "evidence_basis",
                        )
                    }
                    for row in episodes
                ],
                "concepts": [
                    {
                        **{
                            key: row[key]
                            for key in (
                                "id", "canonical_name", "description", "confidence", "status"
                            )
                        },
                        "aliases": aliases_by_concept.get(int(row["id"]), []),
                    }
                    for row in concepts
                ],
            }
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    strict_issues = bool(
        actual_schema_version != SCHEMA_VERSION
        or integrity_check != "ok"
        or foreign_key_issues
        or orphan_associations
        or concepts_without_episode_links
        or embedding_issues
        or empty_content_issues
        or json_payload_issues
        or fts_count_issues
        or sources_without_episodes
        or reference_coverage_issues
        or explicit_evidence_boundary_issues
        or explicit_evidence_mapping_ambiguities
        or association_generation_issues
        or episode_evidence_issues
        or running_runs
        or noncompleted_tasks
        or participant_grounding_issues
        or participant_completeness_issues
        or evidence_participant_grounding_issues
        or cross_source_evidence_duplicates
        or episode_evidence_coverage_issues
        or any(scope["has_cycle"] for scope in temporal_cycles.values())
        or bool(
            temporal_cycles.get("__global__", {}).get(
                "non_ordering_temporal_edge_ids", []
            )
        )
        or (
            segment_reconciliation
            and (
                segment_reconciliation["missing_segments"]
                or segment_reconciliation["unexpected_segments"]
                or segment_reconciliation["source_key_collisions"]
            )
        )
    )
    connection.close()
    return 2 if args.strict and strict_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
