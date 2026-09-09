"""A benchmark-only, source-validated evidence cache.

This module is deliberately not a retrieval or association implementation.  It
models the ordinary B3 baseline from the v3.2 program: a prior request may
remember an exact request key and its ``EvidenceRef`` records, but every reuse
must re-open the current SQLite snapshot and prove that the same Episode,
Source identity, Source bytes, quoted evidence, and delivery budget still
produce the same source-bound excerpt.  It holds no answer prose, cue, edge,
receipt, manifest, or selected-ID result from an association matcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence

from memory_demo.embeddings import normalize_query_text
from memory_demo.retrieval.context import (
    source_excerpt,
    verified_evidence_view_outcomes,
)


SCHEMA = "aevnema.v3_2.source_validated_evidence_cache.v1"


class SourceValidatedCacheMiss(ValueError):
    """A cache entry failed an exact or source-provenance validation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _digest(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


def _json_digest(value: object) -> str:
    return _digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _request_key(*, question: str, domain: str, scope_hash: str) -> str:
    normalized = normalize_query_text(question)
    if not normalized:
        raise ValueError("question is required")
    if not str(domain).strip() or not str(scope_hash).strip():
        raise ValueError("domain and scope_hash are required")
    return _digest("\0".join((normalized, str(domain).strip(), str(scope_hash).strip())))


def _open_readonly(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0
    )
    connection.row_factory = sqlite3.Row
    return connection


def _decode_quotes(value: object) -> list[str]:
    try:
        decoded = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError) as error:
        raise SourceValidatedCacheMiss("episode_evidence_quotes_invalid") from error
    if not isinstance(decoded, list):
        raise SourceValidatedCacheMiss("episode_evidence_quotes_invalid")
    quotes = [item.strip() for item in decoded if isinstance(item, str) and item.strip()]
    if len(quotes) != len(decoded):
        raise SourceValidatedCacheMiss("episode_evidence_quotes_invalid")
    if not quotes:
        raise SourceValidatedCacheMiss("episode_has_no_persisted_evidence_quotes")
    return quotes


def _decode_participants(value: object) -> list[str]:
    try:
        decoded = json.loads(str(value or "[]"))
    except (TypeError, json.JSONDecodeError) as error:
        raise SourceValidatedCacheMiss("episode_participants_invalid") from error
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise SourceValidatedCacheMiss("episode_participants_invalid")
    return list(decoded)


def _source_bound_excerpt(
    *, raw_text: str, episode_text: str, participants: list[str], quotes: list[str], max_chars: int
) -> tuple[str, list[str]]:
    outcomes = verified_evidence_view_outcomes(raw_text, quotes)
    verified = [str(item["view"]) for item in outcomes if item.get("view") is not None]
    if len(verified) != len(quotes):
        raise SourceValidatedCacheMiss("quote_verification_failed")
    excerpt = source_excerpt(
        raw_text, episode_text, participants, max_chars, evidence_quotes=quotes
    )
    if any(quote not in excerpt for quote in verified):
        raise SourceValidatedCacheMiss("verified_quote_exceeds_excerpt_budget")
    return excerpt, [str(item.get("reason", "not_observed")) for item in outcomes]


def _row_materialization(
    connection: sqlite3.Connection,
    *,
    episode_id: int,
    max_chars: int,
) -> dict[str, object]:
    row = connection.execute(
        """
        SELECT e.id AS episode_id, e.source_id, e.source_key, e.segment_index,
               e.text AS episode_text, e.participants_json, e.evidence_quotes_json,
               e.evidence_origin, e.epistemic_status, s.raw_text
        FROM episode AS e
        JOIN source AS s ON s.id = e.source_id
        WHERE e.id = ?
        """,
        (episode_id,),
    ).fetchone()
    if row is None:
        raise SourceValidatedCacheMiss("episode_missing")
    if str(row["evidence_origin"]) != "source":
        raise SourceValidatedCacheMiss("episode_not_source_origin")
    raw_text = str(row["raw_text"])
    quotes = _decode_quotes(row["evidence_quotes_json"])
    participants = _decode_participants(row["participants_json"])
    excerpt, verification_reasons = _source_bound_excerpt(
        raw_text=raw_text,
        episode_text=str(row["episode_text"]),
        participants=participants,
        quotes=quotes,
        max_chars=max_chars,
    )
    episode_identity = {
        "episode_id": int(row["episode_id"]),
        "source_id": int(row["source_id"]),
        "source_key": str(row["source_key"]),
        "segment_index": int(row["segment_index"]),
        "episode_text_sha256": _digest(str(row["episode_text"])),
        "participants_sha256": _json_digest(participants),
        "evidence_quotes_sha256": _json_digest(quotes),
        "evidence_origin": str(row["evidence_origin"]),
        "epistemic_status": str(row["epistemic_status"]),
        "source_raw_sha256": _digest(raw_text),
    }
    return {
        "identity": episode_identity,
        "source_evidence_delivery": "source_bound",
        "source_evidence_delivery_reason": "all_quotes_verified_and_delivered",
        "source_evidence_verification_reasons": verification_reasons,
        "source_evidence_quote_count": len(quotes),
        "source_excerpt": excerpt,
        "source_excerpt_sha256": _digest(excerpt),
    }


@dataclass(frozen=True)
class SourceValidatedEvidenceCache:
    """A serializable exact-key B3 cache entry plus its read-only validator."""

    payload: Mapping[str, object]

    @classmethod
    def seed(
        cls,
        *,
        database: Path,
        question: str,
        domain: str,
        scope_hash: str,
        episode_ids: Sequence[int],
        source_excerpt_chars: int,
    ) -> "SourceValidatedEvidenceCache":
        if source_excerpt_chars <= 0:
            raise ValueError("source_excerpt_chars must be positive")
        ids = list(dict.fromkeys(int(value) for value in episode_ids))
        if not ids:
            raise ValueError("at least one Q1 EvidenceRef episode id is required")
        connection = _open_readonly(database)
        try:
            refs = [
                _row_materialization(
                    connection, episode_id=episode_id, max_chars=source_excerpt_chars
                )
                for episode_id in ids
            ]
        finally:
            connection.close()
        return cls(
            {
                "schema": SCHEMA,
                "entry_type": "ordinary_source_validated_evidence_cache",
                "answer_prose_stored": False,
                "association_state_stored": False,
                "request_key": _request_key(
                    question=question, domain=domain, scope_hash=scope_hash
                ),
                "domain": str(domain).strip(),
                "scope_hash": str(scope_hash).strip(),
                "source_excerpt_chars": int(source_excerpt_chars),
                "evidence_refs": refs,
            }
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SourceValidatedEvidenceCache":
        if payload.get("schema") != SCHEMA:
            raise ValueError("unexpected source validated cache schema")
        if payload.get("entry_type") != "ordinary_source_validated_evidence_cache":
            raise ValueError("unexpected source validated cache entry type")
        if payload.get("answer_prose_stored") is not False:
            raise ValueError("source cache must not contain answer prose")
        if payload.get("association_state_stored") is not False:
            raise ValueError("source cache must not contain association state")
        if not isinstance(payload.get("evidence_refs"), list):
            raise ValueError("source cache lacks evidence refs")
        return cls(dict(payload))

    def as_dict(self) -> dict[str, object]:
        return dict(self.payload)

    def replay(
        self,
        *,
        database: Path,
        question: str,
        domain: str,
        scope_hash: str,
        source_excerpt_chars: int,
    ) -> dict[str, object]:
        payload = self.payload
        if payload.get("request_key") != _request_key(
            question=question, domain=domain, scope_hash=scope_hash
        ):
            raise SourceValidatedCacheMiss("exact_request_scope_or_domain_mismatch")
        if int(payload.get("source_excerpt_chars", -1)) != int(source_excerpt_chars):
            raise SourceValidatedCacheMiss("source_excerpt_budget_changed")
        cached_refs = payload.get("evidence_refs")
        if not isinstance(cached_refs, list) or not cached_refs:
            raise SourceValidatedCacheMiss("cache_evidence_refs_missing")
        connection = _open_readonly(database)
        try:
            replayed: list[dict[str, object]] = []
            for cached in cached_refs:
                if not isinstance(cached, Mapping):
                    raise SourceValidatedCacheMiss("cache_evidence_ref_invalid")
                expected = cached.get("identity")
                if not isinstance(expected, Mapping):
                    raise SourceValidatedCacheMiss("cache_identity_missing")
                # Check the stored Source revision before parsing a quote.  A
                # changed Source can itself make a formerly well-formed
                # projection fail the parser; reporting that as a malformed
                # cache quote would conceal the actual invalidation cause.
                raw_row = connection.execute(
                    """
                    SELECT s.raw_text
                    FROM episode AS e JOIN source AS s ON s.id = e.source_id
                    WHERE e.id = ?
                    """,
                    (int(expected.get("episode_id", -1)),),
                ).fetchone()
                if raw_row is None:
                    raise SourceValidatedCacheMiss("source_or_episode_revision_changed")
                if _digest(str(raw_row["raw_text"])) != expected.get("source_raw_sha256"):
                    raise SourceValidatedCacheMiss("source_or_episode_revision_changed")
                current = _row_materialization(
                    connection,
                    episode_id=int(expected.get("episode_id", -1)),
                    max_chars=source_excerpt_chars,
                )
                if current.get("identity") != dict(expected):
                    raise SourceValidatedCacheMiss("source_or_episode_revision_changed")
                if current.get("source_excerpt_sha256") != cached.get("source_excerpt_sha256"):
                    raise SourceValidatedCacheMiss("source_excerpt_changed")
                replayed.append(current)
        finally:
            connection.close()
        return {
            "cache_status": "hit_source_validated",
            "request_key": payload["request_key"],
            "evidence_state": "complete",
            "materialized_source_refs": replayed,
            "executed_modules": {
                "cache_key_validation": True,
                "source_identity_validation": True,
                "source_quote_validation": True,
                "source_materialization": True,
                "planner": False,
                "embedding": False,
                "retrieval": False,
                "reranker": False,
                "contextual_matcher": False,
                "answer_generation": False,
                "answer_audit": False,
            },
        }
