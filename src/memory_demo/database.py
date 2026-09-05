from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from threading import RLock
from time import sleep
from typing import Iterator
import unicodedata

import numpy as np


SCHEMA_VERSION = 13
SQLITE_BUSY_TIMEOUT_MS = 30_000
_SQLITE_LOCK_RETRY_DELAYS_SECONDS = (0.05, 0.10, 0.25, 0.50, 1.00)


_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_WORD_RE = re.compile(r"[a-z0-9_]+")


class DatabaseBusyError(RuntimeError):
    """A retryable SQLite writer contention that outlived the local wait."""


def _is_sqlite_lock_error(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).casefold()
    return "database is locked" in message or "database is busy" in message


def fts_bigram_tokens(value: str | None) -> str:
    """Pre-tokenize CJK bigrams while retaining Latin words for FTS5."""
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    tokens = [word for word in _WORD_RE.findall(normalized) if len(word) >= 2]
    for run in _CJK_RUN_RE.findall(normalized):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return " ".join(tokens)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Connection factory. Connections and cursors are never shared globally."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # SQLite permits only one writer at a time.  Import workers may still
        # prepare/model-extract concurrently; this guard queues their short
        # persistence transactions instead of allowing them to contend until a
        # connection's busy timeout becomes a permanent file failure.
        self._write_lock = RLock()
        self._journal_mode_lock = RLock()
        self._journal_mode_initialized = False

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000
        )
        connection.row_factory = sqlite3.Row
        connection.create_function(
            "memory_bigram_tokens", 1, fts_bigram_tokens, deterministic=True
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        # journal_mode is persistent for a database file.  Re-applying WAL on
        # every connection is itself a write-capable PRAGMA and was racing the
        # 16 file workers.  Configure it just once for this Database instance.
        with self._journal_mode_lock:
            if not self._journal_mode_initialized:
                self._enable_wal_with_retry(connection)
                self._journal_mode_initialized = True
        return connection

    @staticmethod
    def _enable_wal_with_retry(connection: sqlite3.Connection) -> None:
        for delay in (*_SQLITE_LOCK_RETRY_DELAYS_SECONDS, None):
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_lock_error(exc):
                    raise
                if delay is None:
                    raise DatabaseBusyError(
                        "SQLite remained locked while enabling WAL"
                    ) from exc
                sleep(delay)

    @staticmethod
    def _begin_immediate_with_retry(connection: sqlite3.Connection) -> None:
        for delay in (*_SQLITE_LOCK_RETRY_DELAYS_SECONDS, None):
            try:
                connection.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                if not _is_sqlite_lock_error(exc):
                    raise
                if delay is None:
                    raise DatabaseBusyError(
                        "SQLite writer stayed locked after bounded retries"
                    ) from exc
                sleep(delay)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        # Keep modelling/extraction concurrent, but serialize only the small
        # local SQLite write sections within this process.
        with self._write_lock:
            with self.connection() as connection:
                try:
                    self._begin_immediate_with_retry(connection)
                    yield connection
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    def initialize(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        with self.connection() as connection:
            connection.executescript(schema)
            row = connection.execute(
                "SELECT schema_version FROM schema_meta LIMIT 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_meta(schema_version, created_at) VALUES(?, ?)",
                    (SCHEMA_VERSION, utc_now()),
                )
                self._ensure_v11_indexes(connection)
                connection.commit()
                return
            version = int(row["schema_version"])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"schema version {version} is newer than supported {SCHEMA_VERSION}"
                )
            while version < SCHEMA_VERSION:
                if version == 1:
                    self._migrate_v1_to_v2(connection)
                    version = 2
                    continue
                if version == 2:
                    self._migrate_v2_to_v3(connection)
                    version = 3
                    continue
                if version == 3:
                    self._migrate_v3_to_v4(connection)
                    version = 4
                    continue
                if version == 4:
                    self._migrate_v4_to_v5(connection)
                    version = 5
                    continue
                if version == 5:
                    self._migrate_v5_to_v6(connection)
                    version = 6
                    continue
                if version == 6:
                    self._migrate_v6_to_v7(connection)
                    version = 7
                    continue
                if version == 7:
                    self._migrate_v7_to_v8(connection)
                    version = 8
                    continue
                if version == 8:
                    self._migrate_v8_to_v9(connection)
                    version = 9
                    continue
                if version == 9:
                    self._migrate_v9_to_v10(connection)
                    version = 10
                    continue
                if version == 10:
                    self._migrate_v10_to_v11(connection)
                    version = 11
                    continue
                if version == 11:
                    self._migrate_v11_to_v12(connection)
                    version = 12
                    continue
                if version == 12:
                    self._migrate_v12_to_v13(connection)
                    version = 13
                    continue
                raise RuntimeError(
                    f"no migration path from schema version {version}"
                )
            connection.execute(
                "UPDATE schema_meta SET schema_version = ?",
                (SCHEMA_VERSION,),
            )
            self._ensure_v11_indexes(connection)
            connection.commit()

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(association)")
        }
        additions = {
            "claim_level": (
                "TEXT NOT NULL DEFAULT 'direct_fact' "
                "CHECK(claim_level IN ('direct_fact', 'supported_inference', "
                "'historical_context'))"
            ),
            "audit_status": (
                "TEXT NOT NULL DEFAULT 'not_required' "
                "CHECK(audit_status IN ('not_required', 'dual_accepted'))"
            ),
            "evidence_json": "TEXT NOT NULL DEFAULT '[]'",
            "audit_json": "TEXT NOT NULL DEFAULT '[]'",
        }
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE association ADD COLUMN {name} {declaration}"
                )

    @staticmethod
    def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS paragraph(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                source_key TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                paragraph_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES source(id) ON DELETE CASCADE,
                UNIQUE(source_id, paragraph_index)
            );
            CREATE INDEX IF NOT EXISTS idx_paragraph_source
                ON paragraph(source_id, paragraph_index);
            CREATE INDEX IF NOT EXISTS idx_paragraph_source_key
                ON paragraph(source_key, segment_index);
            """
        )

    @staticmethod
    def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(association)")
        }
        if "generation" not in columns:
            connection.execute(
                "ALTER TABLE association ADD COLUMN generation "
                "INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0)"
            )
        # Older databases did not record inference depth.  The only defensible
        # reconstruction is 0 for directly extracted/manual edges and 1 for
        # known LLM-derived edges.  No legacy row is guessed to be generation 2+.
        connection.execute(
            """
            UPDATE association
            SET generation = 1
            WHERE claim_level IN ('supported_inference', 'historical_context')
               OR created_reason IN (
                    '导入阶段 Episode 候选关系判断',
                    '新 Concept 与已有 Concept 的关系判断'
               )
               OR created_reason LIKE '查询中自主增长：%'
            """
        )

    @staticmethod
    def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
        """Populate external-content FTS indexes for pre-v5 databases."""
        connection.execute("INSERT INTO episode_fts(episode_fts) VALUES('rebuild')")
        connection.execute("INSERT INTO source_fts(source_fts) VALUES('rebuild')")

    @staticmethod
    def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
        """Backfill the CJK bigram companion indexes."""
        connection.execute("DELETE FROM episode_bigram_fts")
        connection.execute(
            """
            INSERT INTO episode_bigram_fts(rowid, tokens)
            SELECT id, memory_bigram_tokens(text || ' ' || source_key)
            FROM episode
            """
        )
        connection.execute("DELETE FROM source_bigram_fts")
        connection.execute(
            """
            INSERT INTO source_bigram_fts(rowid, tokens)
            SELECT id, memory_bigram_tokens(raw_text)
            FROM source
            """
        )

    @staticmethod
    def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
        """Add orthogonal provenance, epistemic status and inference depth."""
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(episode)")
        }
        additions = {
            "evidence_origin": (
                "TEXT NOT NULL DEFAULT 'source' "
                "CHECK(evidence_origin IN "
                "('source', 'importer', 'system', 'mixed', 'unknown'))"
            ),
            "epistemic_status": (
                "TEXT NOT NULL DEFAULT 'unknown' "
                "CHECK(epistemic_status IN "
                "('observed', 'reported', 'speculative', 'mixed', 'unknown'))"
            ),
            "generation": "INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0)",
            "epistemic_note": "TEXT NOT NULL DEFAULT ''",
        }
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE episode ADD COLUMN {name} {declaration}"
                )

    @staticmethod
    def _migrate_v7_to_v8(connection: sqlite3.Connection) -> None:
        """Add asserted and make it the default for unannotated documents."""
        for trigger in (
            "episode_fts_insert",
            "episode_bigram_fts_insert",
            "episode_fts_delete",
            "episode_bigram_fts_delete",
            "episode_fts_update",
            "episode_bigram_fts_update",
        ):
            connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        for index in (
            "idx_episode_source",
            "idx_episode_source_key",
            "idx_episode_timeline",
        ):
            connection.execute(f"DROP INDEX IF EXISTS {index}")
        connection.execute(
            """
            CREATE TABLE episode_v8(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                source_key TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                participants_json TEXT NOT NULL DEFAULT '[]',
                event_type TEXT NOT NULL DEFAULT '',
                location_text TEXT NOT NULL DEFAULT '',
                story_time_text TEXT NOT NULL DEFAULT '',
                story_order REAL,
                timeline_scope TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.5
                    CHECK(confidence >= 0 AND confidence <= 1),
                evidence_origin TEXT NOT NULL DEFAULT 'source'
                    CHECK(evidence_origin IN
                        ('source', 'importer', 'system', 'mixed', 'unknown')),
                epistemic_status TEXT NOT NULL DEFAULT 'asserted'
                    CHECK(epistemic_status IN
                        ('observed', 'asserted', 'reported', 'speculative',
                         'mixed', 'unknown')),
                generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
                epistemic_note TEXT NOT NULL DEFAULT '',
                embedding BLOB NOT NULL,
                extraction_run_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES source(id),
                FOREIGN KEY(extraction_run_id) REFERENCES extraction_run(id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO episode_v8(
                id, source_id, source_key, segment_index, text,
                participants_json, event_type, location_text, story_time_text,
                story_order, timeline_scope, confidence, evidence_origin,
                epistemic_status, generation, epistemic_note, embedding,
                extraction_run_id, created_at, updated_at
            )
            SELECT
                id, source_id, source_key, segment_index, text,
                participants_json, event_type, location_text, story_time_text,
                story_order, timeline_scope, confidence, evidence_origin,
                CASE
                    WHEN evidence_origin = 'source'
                     AND epistemic_status = 'unknown'
                    THEN 'asserted'
                    ELSE epistemic_status
                END,
                generation, epistemic_note, embedding, extraction_run_id,
                created_at, updated_at
            FROM episode
            """
        )
        connection.execute("DROP TABLE episode")
        connection.execute("ALTER TABLE episode_v8 RENAME TO episode")
        connection.execute(
            "CREATE INDEX idx_episode_source ON episode(source_id)"
        )
        connection.execute(
            "CREATE INDEX idx_episode_source_key "
            "ON episode(source_key, segment_index)"
        )
        connection.execute(
            "CREATE INDEX idx_episode_timeline "
            "ON episode(timeline_scope, story_order)"
        )
        connection.execute(
            """
            CREATE TRIGGER episode_fts_insert AFTER INSERT ON episode BEGIN
                INSERT INTO episode_fts(rowid, text, source_key)
                VALUES (new.id, new.text, new.source_key);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER episode_bigram_fts_insert AFTER INSERT ON episode BEGIN
                INSERT INTO episode_bigram_fts(rowid, tokens)
                VALUES (
                    new.id,
                    memory_bigram_tokens(new.text || ' ' || new.source_key)
                );
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER episode_fts_delete AFTER DELETE ON episode BEGIN
                INSERT INTO episode_fts(episode_fts, rowid, text, source_key)
                VALUES ('delete', old.id, old.text, old.source_key);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER episode_bigram_fts_delete AFTER DELETE ON episode BEGIN
                DELETE FROM episode_bigram_fts WHERE rowid = old.id;
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER episode_fts_update
            AFTER UPDATE OF text, source_key ON episode BEGIN
                INSERT INTO episode_fts(episode_fts, rowid, text, source_key)
                VALUES ('delete', old.id, old.text, old.source_key);
                INSERT INTO episode_fts(rowid, text, source_key)
                VALUES (new.id, new.text, new.source_key);
            END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER episode_bigram_fts_update
            AFTER UPDATE OF text, source_key ON episode BEGIN
                DELETE FROM episode_bigram_fts WHERE rowid = old.id;
                INSERT INTO episode_bigram_fts(rowid, tokens)
                VALUES (
                    new.id,
                    memory_bigram_tokens(new.text || ' ' || new.source_key)
                );
            END
            """
        )
        connection.execute("INSERT INTO episode_fts(episode_fts) VALUES('rebuild')")
        connection.execute("DELETE FROM episode_bigram_fts")
        connection.execute(
            """
            INSERT INTO episode_bigram_fts(rowid, tokens)
            SELECT id, memory_bigram_tokens(text || ' ' || source_key)
            FROM episode
            """
        )

    @staticmethod
    def _migrate_v8_to_v9(connection: sqlite3.Connection) -> None:
        """Persist the exact evidence accepted for every new Episode.

        Existing rows cannot be reconstructed safely from their summaries, so
        they remain explicitly marked as legacy evidence instead of receiving
        guessed spans.
        """

        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(episode)")
        }
        additions = {
            "evidence_quotes_json": "TEXT NOT NULL DEFAULT '[]'",
            "evidence_spans_json": "TEXT NOT NULL DEFAULT '[]'",
            "evidence_basis": "TEXT NOT NULL DEFAULT 'legacy_unavailable'",
        }
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE episode ADD COLUMN {name} {declaration}"
                )

    @staticmethod
    def _migrate_v9_to_v10(connection: sqlite3.Connection) -> None:
        """Persist learned-relation vectors so restarts do not re-embed them."""

        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(association)")
        }
        if "cue_embedding" not in columns:
            connection.execute(
                "ALTER TABLE association ADD COLUMN cue_embedding BLOB"
            )
        if "cue_embedding_text" not in columns:
                connection.execute(
                    "ALTER TABLE association ADD COLUMN cue_embedding_text "
                    "TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod
    def _migrate_v10_to_v11(connection: sqlite3.Connection) -> None:
        """Add local double-key contextual-association storage.

        SQLite cannot add a foreign-key constraint to an existing table with
        ``ALTER TABLE`` portably, so cue references are checked atomically by
        the repository.  The migration is deliberately additive and idempotent
        so old knowledge databases remain usable with the feature disabled.
        """
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(association)")
        }
        additions = {
            "association_mode": "TEXT NOT NULL DEFAULT 'semantic'",
            "context_cue_id": "INTEGER",
            "need_cue_id": "INTEGER",
            "utility_weight": "REAL NOT NULL DEFAULT 0.0",
            "utility_successes": "INTEGER NOT NULL DEFAULT 0",
            "utility_noops": "INTEGER NOT NULL DEFAULT 0",
            "utility_harms": "INTEGER NOT NULL DEFAULT 0",
            "distinct_query_count": "INTEGER NOT NULL DEFAULT 0",
            "lifecycle_state": "TEXT NOT NULL DEFAULT 'active'",
            "expires_at": "TEXT",
            "last_evaluated_at": "TEXT",
            "source_request_hash": "TEXT NOT NULL DEFAULT ''",
            "utility_query_hashes": "TEXT NOT NULL DEFAULT '[]'",
        }
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE association ADD COLUMN {name} {declaration}"
                )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS association_cue_prototype(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                cue_kind TEXT NOT NULL CHECK(cue_kind IN ('context', 'need')),
                model_id TEXT NOT NULL,
                dimension INTEGER NOT NULL CHECK(dimension > 0),
                dtype TEXT NOT NULL DEFAULT 'float32' CHECK(dtype = 'float32'),
                vector_blob BLOB NOT NULL,
                text_hash TEXT NOT NULL,
                display_text TEXT NOT NULL DEFAULT '',
                source_request_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(domain, cue_kind, model_id, text_hash)
            );
            """
        )

    @staticmethod
    def _ensure_v11_indexes(connection: sqlite3.Connection) -> None:
        """Create v11 indexes only after columns have been migrated."""
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS association_mode_state_idx
                ON association(association_mode, lifecycle_state);
            CREATE INDEX IF NOT EXISTS association_context_cue_idx
                ON association(context_cue_id);
            CREATE INDEX IF NOT EXISTS association_need_cue_idx
                ON association(need_cue_id);
            CREATE INDEX IF NOT EXISTS association_endpoint_mode_idx
                ON association(from_type, from_id, association_mode);
            CREATE INDEX IF NOT EXISTS association_cue_domain_kind_idx
                ON association_cue_prototype(domain, cue_kind);
            """
        )

    @staticmethod
    def _migrate_v11_to_v12(connection: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(association)")
        }
        if "utility_query_hashes" not in columns:
            connection.execute(
                "ALTER TABLE association ADD COLUMN utility_query_hashes TEXT NOT NULL DEFAULT '[]'"
            )

    @staticmethod
    def _migrate_v12_to_v13(connection: sqlite3.Connection) -> None:
        """Allow retrieval-only contextual edges without loosening fact rows."""
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(association)")
        }
        if "association_mode" not in columns:
            # A v12 database should already have passed the previous migration;
            # fail explicitly rather than silently producing a partial table.
            raise RuntimeError("association contextual columns are missing")
        for index in (
            "idx_association_from",
            "idx_association_to",
            "idx_association_relation",
            "idx_association_temporal_before",
            "idx_association_temporal_after",
            "association_mode_state_idx",
            "association_context_cue_idx",
            "association_need_cue_idx",
            "association_endpoint_mode_idx",
        ):
            connection.execute(f"DROP INDEX IF EXISTS {index}")
        connection.execute(
            """
            CREATE TABLE association_v13(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_type TEXT NOT NULL CHECK(from_type IN ('episode', 'concept')),
                from_id INTEGER NOT NULL,
                to_type TEXT NOT NULL CHECK(to_type IN ('episode', 'concept')),
                to_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                relation_key TEXT NOT NULL,
                relation_text TEXT NOT NULL,
                cue_embedding BLOB,
                cue_embedding_text TEXT NOT NULL DEFAULT '',
                association_mode TEXT NOT NULL DEFAULT 'semantic',
                context_cue_id INTEGER,
                need_cue_id INTEGER,
                utility_weight REAL NOT NULL DEFAULT 0.0,
                utility_successes INTEGER NOT NULL DEFAULT 0,
                utility_noops INTEGER NOT NULL DEFAULT 0,
                utility_harms INTEGER NOT NULL DEFAULT 0,
                distinct_query_count INTEGER NOT NULL DEFAULT 0,
                lifecycle_state TEXT NOT NULL DEFAULT 'active',
                expires_at TEXT,
                last_evaluated_at TEXT,
                source_request_hash TEXT NOT NULL DEFAULT '',
                utility_query_hashes TEXT NOT NULL DEFAULT '[]',
                polarity INTEGER NOT NULL DEFAULT 1 CHECK(polarity IN (-1, 0, 1)),
                weight REAL NOT NULL DEFAULT 0.5 CHECK(weight >= 0 AND weight <= 1),
                confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
                generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
                evidence_count INTEGER NOT NULL DEFAULT 1,
                claim_level TEXT NOT NULL DEFAULT 'direct_fact'
                    CHECK(claim_level IN ('direct_fact', 'supported_inference', 'historical_context', 'retrieval_only')),
                audit_status TEXT NOT NULL DEFAULT 'not_required'
                    CHECK(audit_status IN ('not_required', 'dual_accepted')),
                evidence_json TEXT NOT NULL DEFAULT '[]',
                audit_json TEXT NOT NULL DEFAULT '[]',
                created_reason TEXT NOT NULL DEFAULT '',
                last_used TEXT,
                use_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(from_type, from_id, to_type, to_id, relation_type, relation_key, polarity)
            )
            """
        )
        # Explicit column names keep this migration independent of historical
        # column ordering changes between v9/v10/v12.
        connection.execute(
            """
            INSERT INTO association_v13(
                id, from_type, from_id, to_type, to_id, relation_type, relation_key,
                relation_text, cue_embedding, cue_embedding_text, association_mode,
                context_cue_id, need_cue_id, utility_weight, utility_successes,
                utility_noops, utility_harms, distinct_query_count, lifecycle_state,
                expires_at, last_evaluated_at, source_request_hash, utility_query_hashes,
                polarity, weight, confidence, generation, evidence_count, claim_level,
                audit_status, evidence_json, audit_json, created_reason, last_used,
                use_count, created_at, updated_at
            )
            SELECT
                id, from_type, from_id, to_type, to_id, relation_type, relation_key,
                relation_text, cue_embedding, cue_embedding_text, association_mode,
                context_cue_id, need_cue_id, utility_weight, utility_successes,
                utility_noops, utility_harms, distinct_query_count, lifecycle_state,
                expires_at, last_evaluated_at, source_request_hash, utility_query_hashes,
                polarity, weight, confidence, generation, evidence_count, claim_level,
                audit_status, evidence_json, audit_json, created_reason, last_used,
                use_count, created_at, updated_at
            FROM association
            """
        )
        connection.execute("DROP TABLE association")
        connection.execute("ALTER TABLE association_v13 RENAME TO association")
        Database._ensure_v11_indexes(connection)

    def backup_to(self, target: str | Path) -> None:
        target_path = Path(target)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_uri = f"{self.path.resolve().as_uri()}?mode=ro"
        source = sqlite3.connect(source_uri, uri=True, timeout=5.0)
        try:
            destination = sqlite3.connect(target_path)
            try:
                source.backup(destination)
            finally:
                destination.close()
        finally:
            source.close()

    @staticmethod
    def validate_float32_vector(
        vector, dimension: int, *, require_normalized: bool = False
    ) -> np.ndarray:
        """Validate the on-disk contract used by cue prototypes.

        Embeddings are intentionally always float32 in both SQLite and RAM.
        Rejecting malformed blobs at the persistence boundary prevents a bad
        prototype from poisoning every future contextual lookup.
        """
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != int(dimension):
            raise ValueError("embedding dimension does not match prototype")
        if not np.all(np.isfinite(array)):
            raise ValueError("embedding contains non-finite values")
        norm = float(np.linalg.norm(array))
        if norm <= 0.0:
            raise ValueError("embedding must be non-zero")
        if require_normalized and abs(norm - 1.0) > 1e-3:
            raise ValueError("embedding must be L2 normalized")
        return np.ascontiguousarray(array, dtype=np.float32)

    def audit_contextual_storage(self) -> dict[str, int | bool]:
        """Check cue-vector and contextual-edge referential integrity offline."""
        result: dict[str, int | bool] = {
            "orphan_context_cues": 0,
            "orphan_need_cues": 0,
            "invalid_cue_kinds": 0,
            "invalid_vectors": 0,
            "cross_domain_edges": 0,
            "invalid_targets": 0,
            "invalid_claim_level": 0,
        }
        with self.connection() as connection:
            prototypes = connection.execute(
                "SELECT * FROM association_cue_prototype"
            ).fetchall()
            for row in prototypes:
                try:
                    blob = bytes(row["vector_blob"])
                    if len(blob) != int(row["dimension"]) * 4:
                        raise ValueError
                    self.validate_float32_vector(
                        np.frombuffer(blob, dtype=np.float32),
                        int(row["dimension"]),
                        require_normalized=True,
                    )
                except (TypeError, ValueError):
                    result["invalid_vectors"] += 1
            edges = connection.execute(
                """
                SELECT a.*, cc.domain AS context_domain, nc.domain AS need_domain
                FROM association a
                LEFT JOIN association_cue_prototype cc ON cc.id = a.context_cue_id
                LEFT JOIN association_cue_prototype nc ON nc.id = a.need_cue_id
                WHERE a.association_mode = 'contextual_recall'
                """
            ).fetchall()
            for row in edges:
                if row["context_domain"] is None:
                    result["orphan_context_cues"] += 1
                if row["need_domain"] is None:
                    result["orphan_need_cues"] += 1
                # The LEFT JOIN above also exposes a cue of the wrong kind;
                # retain this as a separate diagnostic from a missing row.
                if row["context_domain"] is not None or row["need_domain"] is not None:
                    cue_rows = connection.execute(
                        "SELECT cue_kind FROM association_cue_prototype WHERE id IN (?, ?)",
                        (row["context_cue_id"], row["need_cue_id"]),
                    ).fetchall()
                    kinds = [str(item["cue_kind"]) for item in cue_rows]
                    if len(kinds) != 2 or kinds.count("context") != 1 or kinds.count("need") != 1:
                        result["invalid_cue_kinds"] += 1
                if (
                    row["context_domain"] is not None
                    and row["need_domain"] is not None
                    and row["context_domain"] != row["need_domain"]
                ):
                    result["cross_domain_edges"] += 1
                if row["to_type"] != "episode" or connection.execute(
                    "SELECT 1 FROM episode WHERE id = ?", (row["to_id"],)
                ).fetchone() is None:
                    result["invalid_targets"] += 1
                if str(row["claim_level"]) != "retrieval_only":
                    result["invalid_claim_level"] += 1
        result["ok"] = not any(
            int(result[key])
            for key in (
                "orphan_context_cues",
                "orphan_need_cues",
                "invalid_vectors",
                "cross_domain_edges",
                "invalid_targets",
                "invalid_claim_level",
                "invalid_cue_kinds",
            )
        )
        return result
