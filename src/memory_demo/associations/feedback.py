"""Explicit feedback and source-checked, untyped recall shortcuts.

This module does not decide whether an inference is correct. The caller must
independently review the quoted Source text and supply that verdict. Hashes and
exact spans bind that review to the database version that is actually learned;
neither vector similarity nor traversal/use counts are accepted as a verdict.
The sidecar tables deliberately do not publish or alter V3 learning receipts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import sqlite3
from typing import Iterable, Sequence

from memory_demo.database import Database, utc_now


def _positive_id(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty_text(value: str, name: str, *, limit: int = 8192) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return value.strip()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_sha256(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    episode_id: int
    source_id: int
    source_sha256: str
    start: int
    end: int
    quote: str

    def __post_init__(self) -> None:
        _positive_id(self.episode_id, "episode_id")
        _positive_id(self.source_id, "source_id")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (self.start, self.end)):
            raise ValueError("source span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source span must be nonempty and nonnegative")
        if not isinstance(self.quote, str) or not self.quote.strip():
            raise ValueError("source quote must be nonempty")
        if self.end - self.start != len(self.quote):
            raise ValueError("source span must have the same length as its exact quote")


@dataclass(frozen=True, slots=True)
class VerifiedRecallLink:
    from_episode_id: int
    to_episode_id: int
    evidence: tuple[SourceEvidence, ...]
    verifier: str
    rationale: str
    verified: bool = False

    def __post_init__(self) -> None:
        _positive_id(self.from_episode_id, "from_episode_id")
        _positive_id(self.to_episode_id, "to_episode_id")
        if self.from_episode_id == self.to_episode_id:
            raise ValueError("a recall shortcut must join distinct episodes")
        if self.verified is not True:
            raise ValueError("an explicit successful source-review verdict is required")
        _nonempty_text(self.verifier, "verifier", limit=256)
        _nonempty_text(self.rationale, "rationale")
        evidence = tuple(self.evidence)
        if not evidence or not all(isinstance(item, SourceEvidence) for item in evidence):
            raise ValueError("source evidence is required")
        endpoints = {self.from_episode_id, self.to_episode_id}
        if not endpoints.issubset({item.episode_id for item in evidence}):
            raise ValueError("both shortcut endpoints require source evidence")
        object.__setattr__(self, "evidence", evidence)


@dataclass(frozen=True, slots=True)
class WeightChange:
    association_id: int
    before: float | None
    after: float
    created: bool = False


@dataclass(frozen=True, slots=True)
class FeedbackResult:
    feedback_id: str
    applied: bool
    changes: tuple[WeightChange, ...]

    @property
    def association_ids(self) -> tuple[int, ...]:
        return tuple(change.association_id for change in self.changes)


class RecallFeedbackService:
    """Opt-in learning with exact retry semantics and bounded strengths.

    ``association_ids`` must contain only edges attributed to the delivered
    answer. The service cannot reconstruct this attribution from explored paths.
    Positive feedback uses w + rate * (1 - w), negative feedback w * (1 - rate).
    There is no clock-driven decay. New shortcuts start at ``learning_rate`` and
    receive no second reinforcement in their creation event.
    """

    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS recall_feedback_event(
                feedback_id TEXT PRIMARY KEY,
                request_sha256 TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('user_positive','user_negative','verified')),
                request_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS recall_feedback_change(
                feedback_id TEXT NOT NULL REFERENCES recall_feedback_event(feedback_id),
                association_id INTEGER NOT NULL REFERENCES association(id),
                before_weight REAL,
                after_weight REAL NOT NULL CHECK(after_weight BETWEEN 0 AND 1),
                was_created INTEGER NOT NULL CHECK(was_created IN (0,1)),
                PRIMARY KEY(feedback_id, association_id)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS recall_link_evidence(
                association_id INTEGER PRIMARY KEY REFERENCES association(id),
                feedback_id TEXT NOT NULL REFERENCES recall_feedback_event(feedback_id),
                from_episode_id INTEGER NOT NULL REFERENCES episode(id),
                to_episode_id INTEGER NOT NULL REFERENCES episode(id),
                evidence_json TEXT NOT NULL,
                verifier TEXT NOT NULL,
                rationale TEXT NOT NULL,
                verified_at TEXT NOT NULL
            )
        """)

    def ensure_schema(self) -> None:
        with self.db.transaction() as connection:
            self._ensure_schema(connection)

    @staticmethod
    def _ids(values: Iterable[int]) -> tuple[int, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError("association_ids must be an iterable of positive integers")
        try:
            return tuple(sorted({_positive_id(value, "association_id") for value in values}))
        except TypeError as error:
            raise ValueError("association_ids must be an iterable of positive integers") from error

    @staticmethod
    def _rate(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("learning_rate must be numeric")
        rate = float(value)
        if not math.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        return rate

    @staticmethod
    def _validate_evidence(connection: sqlite3.Connection, evidence: Sequence[SourceEvidence]) -> None:
        sources: dict[int, str] = {}
        for item in evidence:
            row = connection.execute(
                "SELECT source_id FROM episode WHERE id = ?", (item.episode_id,)
            ).fetchone()
            if row is None or int(row["source_id"]) != item.source_id:
                raise ValueError("source evidence no longer belongs to the stated episode")
            if item.source_id not in sources:
                source = connection.execute(
                    "SELECT raw_text FROM source WHERE id = ?", (item.source_id,)
                ).fetchone()
                if source is None:
                    raise ValueError("source evidence no longer exists")
                sources[item.source_id] = str(source["raw_text"])
            raw_text = sources[item.source_id]
            if source_sha256(raw_text) != item.source_sha256:
                raise ValueError("source changed since verification")
            if raw_text[item.start:item.end] != item.quote:
                raise ValueError("source quote does not match its exact span")

    @classmethod
    def _validate_stored_link(cls, connection: sqlite3.Connection, edge) -> None:
        stored = connection.execute(
            "SELECT * FROM recall_link_evidence WHERE association_id = ?", (int(edge["id"]),)
        ).fetchone()
        if stored is None:
            raise ValueError("simple recall edge has no source-review receipt")
        if (
            edge["from_type"] != "episode" or edge["to_type"] != "episode"
            or int(edge["from_id"]) != int(stored["from_episode_id"])
            or int(edge["to_id"]) != int(stored["to_episode_id"])
        ):
            raise ValueError("simple recall endpoints no longer match their receipt")
        evidence = tuple(SourceEvidence(**item) for item in json.loads(stored["evidence_json"]))
        if not {int(edge["from_id"]), int(edge["to_id"])}.issubset(
            {item.episode_id for item in evidence}
        ):
            raise ValueError("simple recall receipt must support both endpoints")
        cls._validate_evidence(connection, evidence)

    def valid_learned_edge_ids(self) -> set[int]:
        """Read current source validity; absent sidecars return empty without DDL."""
        with self.db.connection() as connection:
            connection.execute("BEGIN")
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='recall_link_evidence'"
            ).fetchone() is None:
                return set()
            valid: set[int] = set()
            for edge in connection.execute(
                "SELECT * FROM association WHERE association_mode = 'simple_recall' "
                "AND lifecycle_state = 'active'"
            ).fetchall():
                try:
                    self._validate_stored_link(connection, edge)
                except (ValueError, TypeError, KeyError):
                    continue
                valid.add(int(edge["id"]))
            return valid

    def apply_user_feedback(
        self, feedback_id: str, association_ids: Iterable[int], *,
        positive: bool, learning_rate: float = 0.2,
    ) -> FeedbackResult:
        if not isinstance(positive, bool):
            raise ValueError("positive must be a boolean")
        ids = self._ids(association_ids)
        if not ids:
            raise ValueError("feedback requires at least one attributed association")
        return self._apply(
            feedback_id, ids, "user_positive" if positive else "user_negative",
            self._rate(learning_rate), (),
        )

    def learn_verified(
        self, feedback_id: str, links: Sequence[VerifiedRecallLink], *,
        association_ids: Iterable[int] = (), learning_rate: float = 0.2,
    ) -> FeedbackResult:
        links = tuple(links)
        if not links or not all(isinstance(link, VerifiedRecallLink) for link in links):
            raise ValueError("learning requires source-reviewed recall links")
        pairs = [tuple(sorted((link.from_episode_id, link.to_episode_id))) for link in links]
        if len(pairs) != len(set(pairs)):
            raise ValueError("provide each shortcut endpoint pair only once per event")
        links = tuple(link for _, link in sorted(zip(pairs, links), key=lambda item: item[0]))
        return self._apply(feedback_id, self._ids(association_ids), "verified", self._rate(learning_rate), links)

    def verified_receipt(
        self, feedback_id: str, links: Sequence[VerifiedRecallLink], *, learning_rate: float = 0.2,
    ) -> FeedbackResult | None:
        """Read a committed learning event without DDL or a second update.

        The complete request and current Source evidence must still match. This
        closes the commit/checkpoint gap after cancellation or process restart.
        """
        links = tuple(sorted(links, key=lambda link: tuple(sorted((link.from_episode_id, link.to_episode_id)))))
        payload = _json({"version": 1, "kind": "verified", "association_ids": (),
                         "learning_rate": self._rate(learning_rate), "links": [asdict(link) for link in links]})
        with self.db.connection() as connection:
            connection.execute("BEGIN")
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recall_feedback_event'").fetchone() is None:
                return None
            prior = connection.execute("SELECT request_sha256 FROM recall_feedback_event WHERE feedback_id=?", (feedback_id,)).fetchone()
            if prior is None:
                return None
            if prior["request_sha256"] != source_sha256(payload):
                raise ValueError("committed recall receipt belongs to a different payload")
            for link in links:
                self._validate_evidence(connection, link.evidence)
            changes = tuple(WeightChange(int(row["association_id"]), row["before_weight"],
                                        float(row["after_weight"]), bool(row["was_created"]))
                            for row in connection.execute("SELECT * FROM recall_feedback_change WHERE feedback_id=? ORDER BY association_id", (feedback_id,)))
            return FeedbackResult(feedback_id, False, changes)

    def _apply(
        self, feedback_id: str, ids: tuple[int, ...], kind: str,
        rate: float, links: tuple[VerifiedRecallLink, ...],
    ) -> FeedbackResult:
        feedback_id = _nonempty_text(feedback_id, "feedback_id", limit=256)
        payload = _json({"version": 1, "kind": kind, "association_ids": ids,
                         "learning_rate": rate, "links": [asdict(link) for link in links]})
        request_hash = source_sha256(payload)
        with self.db.transaction() as connection:
            self._ensure_schema(connection)
            prior = connection.execute(
                "SELECT request_sha256 FROM recall_feedback_event WHERE feedback_id = ?", (feedback_id,)
            ).fetchone()
            if prior is not None and prior["request_sha256"] != request_hash:
                raise ValueError("feedback_id already belongs to a different feedback payload")
            # Validate even an exact retry: a stale receipt is never re-presented
            # as a current source verification after the underlying text changes.
            for link in links:
                self._validate_evidence(connection, link.evidence)
            edges = {}
            for association_id in ids:
                edge = connection.execute(
                    "SELECT * FROM association WHERE id = ?", (association_id,)
                ).fetchone()
                if edge is None:
                    raise ValueError(f"unknown association_id: {association_id}")
                if edge["association_mode"] == "simple_recall":
                    self._validate_stored_link(connection, edge)
                edges[association_id] = edge
            if prior is not None:
                changes = tuple(WeightChange(
                    int(row["association_id"]), row["before_weight"],
                    float(row["after_weight"]), bool(row["was_created"]),
                ) for row in connection.execute(
                    "SELECT * FROM recall_feedback_change WHERE feedback_id = ? ORDER BY association_id",
                    (feedback_id,),
                ))
                return FeedbackResult(feedback_id, False, changes)
            now = utc_now()
            connection.execute(
                "INSERT INTO recall_feedback_event VALUES (?, ?, ?, ?, ?)",
                (feedback_id, request_hash, kind, payload, now),
            )
            created_ids: set[int] = set()
            for link in links:
                first, second = sorted((link.from_episode_id, link.to_episode_id))
                edge = connection.execute(
                    "SELECT * FROM association WHERE from_type='episode' AND from_id=? "
                    "AND to_type='episode' AND to_id=? AND relation_type='retrieval' "
                    "AND relation_key='simple_recall' AND polarity=1", (first, second),
                ).fetchone()
                if edge is None:
                    cursor = connection.execute("""
                        INSERT INTO association(
                            from_type, from_id, to_type, to_id, relation_type, relation_key,
                            relation_text, association_mode, weight, confidence, claim_level,
                            created_reason, created_at, updated_at
                        ) VALUES ('episode', ?, 'episode', ?, 'retrieval', 'simple_recall',
                                  '', 'simple_recall', ?, 1.0, 'retrieval_only',
                                  'source_reviewed_recall', ?, ?)
                    """, (first, second, rate, now, now))
                    association_id = int(cursor.lastrowid)
                    created_ids.add(association_id)
                    edge = connection.execute("SELECT * FROM association WHERE id=?", (association_id,)).fetchone()
                elif edge["association_mode"] != "simple_recall":
                    raise ValueError("shortcut key belongs to an incompatible association mode")
                association_id = int(edge["id"])
                edges[association_id] = edge
                connection.execute("""
                    INSERT INTO recall_link_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(association_id) DO UPDATE SET
                        feedback_id=excluded.feedback_id, evidence_json=excluded.evidence_json,
                        verifier=excluded.verifier, rationale=excluded.rationale,
                        verified_at=excluded.verified_at
                """, (association_id, feedback_id, first, second,
                      _json([asdict(item) for item in link.evidence]), link.verifier, link.rationale, now))
            changes = []
            for association_id, edge in sorted(edges.items()):
                before = float(edge["weight"])
                created = association_id in created_ids
                after = rate if created else (
                    before * (1 - rate) if kind == "user_negative" else before + rate * (1 - before)
                )
                connection.execute(
                    "UPDATE association SET weight=?, updated_at=? WHERE id=?", (after, now, association_id)
                )
                change = WeightChange(association_id, None if created else before, after, created)
                changes.append(change)
                connection.execute(
                    "INSERT INTO recall_feedback_change VALUES (?, ?, ?, ?, ?)",
                    (feedback_id, association_id, change.before, change.after, int(created)),
                )
            return FeedbackResult(feedback_id, True, tuple(changes))
