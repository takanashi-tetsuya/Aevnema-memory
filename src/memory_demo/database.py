from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import RLock
from time import sleep
from typing import Callable, Iterator
import unicodedata

import numpy as np


SCHEMA_VERSION = 22
SQLITE_BUSY_TIMEOUT_MS = 30_000
_SQLITE_LOCK_RETRY_DELAYS_SECONDS = (0.05, 0.10, 0.25, 0.50, 1.00)
_transaction_liveness_hooks: ContextVar[tuple[Callable[[], None], ...]] = (
    ContextVar("memory_demo_transaction_liveness_hooks", default=())
)


_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_WORD_RE = re.compile(r"[a-z0-9_]+")
_OPAQUE_CONTRACT_DIGEST_RE = re.compile(
    r"^(?:[a-z][a-z0-9_.-]*:)?sha256:[0-9a-f]{64}$"
)
_SAFE_CONTRACT_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$"
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class DatabaseBusyError(RuntimeError):
    """A retryable SQLite writer contention that outlived the local wait."""


@contextmanager
def transaction_liveness(require_live: Callable[[], None]) -> Iterator[None]:
    """Bind a request's liveness rule to transactions in this context.

    A query can finish remote work just as its deadline expires.  Repository
    APIs remain compatible, but their SQLite transactions still need to
    reject such late results at both the write-entry and durable-COMMIT
    boundaries.  ContextVars keep concurrent requests and worker threads
    isolated without making a deadline a process-global policy.
    """

    if not callable(require_live):
        raise TypeError("transaction liveness hook must be callable")
    prior = _transaction_liveness_hooks.get()
    token = _transaction_liveness_hooks.set((*prior, require_live))
    try:
        yield
    finally:
        _transaction_liveness_hooks.reset(token)


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


def _is_opaque_contract_digest(value: object) -> int:
    """SQLite CHECK helper for redacted v16 revisit-contract fields."""

    return int(
        isinstance(value, str)
        and _OPAQUE_CONTRACT_DIGEST_RE.fullmatch(value.strip()) is not None
    )


def _is_safe_contract_identifier(value: object) -> int:
    """Allow bounded metadata IDs but never arbitrary contract prose."""

    return int(
        isinstance(value, str)
        and _SAFE_CONTRACT_IDENTIFIER_RE.fullmatch(value.strip()) is not None
    )


def _has_opaque_contract_bindings(value: object) -> int:
    """Validate the fixed, opaque JSON shape used for slot/need bindings."""

    if not isinstance(value, str) or len(value) > 65_536:
        return 0
    try:
        bindings = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0
    if not isinstance(bindings, list) or not bindings:
        return 0
    slot_ids: set[str] = set()
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {
            "slot_id",
            "need_query_id",
            "need_hash",
        }:
            return 0
        values = tuple(binding.values())
        if any(
            not isinstance(item, str)
            or _OPAQUE_CONTRACT_DIGEST_RE.fullmatch(item.strip()) is None
            for item in values
        ):
            return 0
        slot_id = str(binding["slot_id"])
        if slot_id in slot_ids:
            return 0
        slot_ids.add(slot_id)
    return 1


def _has_single_opaque_contract_binding(value: object) -> int:
    """Accept exactly one redacted v16 slot/need binding for a v17 seed."""

    if _has_opaque_contract_bindings(value) != 1:
        return 0
    try:
        bindings = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0
    return int(isinstance(bindings, list) and len(bindings) == 1)


def _is_sha256_hex(value: object) -> int:
    """Validate a bare content hash without admitting arbitrary text."""

    return int(
        isinstance(value, str)
        and _SHA256_HEX_RE.fullmatch(value.strip().casefold()) is not None
    )


def _is_utc_instant(value: object) -> int:
    """SQLite CHECK helper for canonical UTC-only manifest timestamps."""

    if not isinstance(value, str) or not value.strip():
        return 0
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError:
        return 0
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        return 0
    return int(parsed.astimezone(timezone.utc).isoformat() == text)


def _contextual_revisit_ready_fingerprint(
    receipt_id: object,
    association_id: object,
    context_cue_id: object,
    need_cue_id: object,
    domain: object,
    model_id: object,
    embedding_space_id: object,
    dimension: object,
    dtype: object,
    durable_artifact_hash: object,
    ready_at: object,
    ready_index_epoch: object,
    status: object,
    verification_status: object,
) -> str:
    """Compute the existing v16 receipt-publication fingerprint in SQLite.

    This lets the v17 ready-transition trigger reject a fabricated opaque
    fingerprint even if someone issues raw SQL instead of using the repository.
    It serializes only receipt metadata and hashes, never source/query text.
    """

    try:
        normalized = {
            "association_id": int(association_id),
            "context_cue_id": int(context_cue_id),
            "creation_receipt_id": int(receipt_id),
            "dimension": int(dimension),
            "domain": str(domain or ""),
            "dtype": str(dtype or ""),
            "durable_artifact_hash": str(durable_artifact_hash or ""),
            "embedding_space_id": str(embedding_space_id or ""),
            "model_id": str(model_id or ""),
            "need_cue_id": int(need_cue_id),
            "ready_at": str(ready_at or ""),
            "ready_index_epoch": int(ready_index_epoch),
            "status": str(status or ""),
            "verification_status": str(verification_status or ""),
            "version": "contextual-ready-publication-v1",
        }
    except (TypeError, ValueError):
        return ""
    if any(
        int(normalized[field_name]) <= 0
        for field_name in (
            "association_id",
            "context_cue_id",
            "creation_receipt_id",
            "dimension",
            "need_cue_id",
            "ready_index_epoch",
        )
    ):
        return ""
    material = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "revisit-publication:sha256:" + hashlib.sha256(material).hexdigest()


def _contextual_revisit_vector_fingerprint(value: object) -> str:
    """Return a non-reversible cue-blob digest; never expose vector bytes."""

    try:
        blob = bytes(value)
    except (TypeError, ValueError):
        return ""
    if not blob:
        return ""
    return "cue-vector:sha256:" + hashlib.sha256(blob).hexdigest()


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
        connection.create_function(
            "contextual_revisit_opaque_digest",
            1,
            _is_opaque_contract_digest,
            deterministic=True,
        )
        connection.create_function(
            "contextual_revisit_safe_identifier",
            1,
            _is_safe_contract_identifier,
            deterministic=True,
        )
        connection.create_function(
            "contextual_revisit_opaque_bindings",
            1,
            _has_opaque_contract_bindings,
            deterministic=True,
        )
        connection.create_function(
            "contextual_revisit_single_opaque_binding",
            1,
            _has_single_opaque_contract_binding,
            deterministic=True,
        )
        connection.create_function(
            "contextual_revisit_sha256_hex",
            1,
            _is_sha256_hex,
            deterministic=True,
        )
        connection.create_function(
            "contextual_revisit_utc_instant",
            1,
            _is_utc_instant,
            deterministic=True,
        )
        connection.create_function(
            "contextual_revisit_ready_fingerprint",
            14,
            _contextual_revisit_ready_fingerprint,
            deterministic=True,
        )
        connection.create_function(
            "contextual_revisit_vector_fingerprint",
            1,
            _contextual_revisit_vector_fingerprint,
            deterministic=True,
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
    def transaction(
        self,
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> Iterator[sqlite3.Connection]:
        # Keep modelling/extraction concurrent, but serialize only the small
        # local SQLite write sections within this process.
        liveness_hooks = _transaction_liveness_hooks.get()

        def require_context_live() -> None:
            for hook in liveness_hooks:
                hook()

        def require_before_commit() -> None:
            require_context_live()
            if before_commit is not None:
                before_commit()

        with self._write_lock:
            with self.connection() as connection:
                try:
                    require_context_live()
                    self._begin_immediate_with_retry(connection)
                    yield connection
                    # Some request-scoped writers must prove that the caller
                    # is still live at the only point where SQLite makes all
                    # earlier mutations durable.  Raising here rolls the
                    # transaction back rather than publishing a late result.
                    require_before_commit()
                    connection.commit()
                except BaseException:
                    # KeyboardInterrupt/SystemExit must not leave a pending
                    # write transaction to be committed implicitly later.
                    connection.rollback()
                    raise

    def initialize(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        # The checked-in V17 block is retained in schema.sql as a readable
        # bootstrap description, but its baseline trigger set is intentionally
        # weaker than the UDF-backed guard installed below.  Never execute it
        # during a live initialization: V17 is created only by the atomic
        # `_ensure_v17_schema` transaction after all earlier schema objects
        # exist.  This also makes a crash during an old-version migration
        # fail closed (no runtime-manifest table) rather than leave a weak one.
        v17_start = schema.find("-- V17 adds a recovery seed")
        fts_start = schema.find("-- FTS is a retrieval index", v17_start)
        contextual_start = schema.find("-- Contextual recall creation")
        if v17_start < 0 or fts_start < 0 or contextual_start < 0:
            raise RuntimeError("schema.sql is missing the V17 bootstrap boundary")
        pre_v17_schema = schema[:v17_start] + schema[fts_start:]
        # An old database must receive only the historical core tables before
        # its ordered migrations run.  In particular, do not install later
        # contextual triggers against an association table that may not yet
        # have ``association_mode`` (or replay their weaker baseline guards).
        # The migrations themselves introduce V11--V16 in order, and V17 is
        # still installed exclusively by the strict installer below.
        legacy_bootstrap_schema = schema[:contextual_start] + schema[fts_start:]
        with self.connection() as connection:
            # Replaying schema.sql on every startup used to DROP and recreate
            # the baseline V17 trigger before the stricter guard installer
            # ran.  A current database already owns all of those objects, so
            # avoid that destructive no-op entirely.  Older databases still
            # receive the historical baseline before their ordered migrations
            # because several pre-V17 migration fixtures intentionally start
            # with only a subset of tables.
            schema_meta_exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'schema_meta'
                """
            ).fetchone() is not None
            row = (
                connection.execute(
                    "SELECT schema_version FROM schema_meta LIMIT 1"
                ).fetchone()
                if schema_meta_exists
                else None
            )
            if row is None:
                # A database without version metadata is not ready for any
                # ordinary writer.  Build its baseline first, then install
                # every strict guard before publishing schema_meta below.
                connection.executescript(pre_v17_schema)
                self._ensure_v11_indexes(connection)
                self._ensure_v14_schema(connection)
                self._ensure_v15_schema(connection)
                self._ensure_v16_schema(connection)
                self._ensure_v17_schema(connection)
                self._ensure_v18_schema(connection)
                self._ensure_v19_schema(connection)
                self._ensure_v20_schema(connection)
                self._ensure_v21_schema(connection)
                self._ensure_v22_schema(connection)
                connection.execute(
                    "INSERT INTO schema_meta(schema_version, created_at) VALUES(?, ?)",
                    (SCHEMA_VERSION, utc_now()),
                )
                connection.commit()
                return
            version = int(row["schema_version"])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"schema version {version} is newer than supported {SCHEMA_VERSION}"
                )
            if version < SCHEMA_VERSION:
                # Keep older migration fixtures and genuine legacy stores
                # usable without giving them future contextual DDL early.
                connection.executescript(legacy_bootstrap_schema)
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
                if version == 13:
                    self._migrate_v13_to_v14(connection)
                    version = 14
                    continue
                if version == 14:
                    self._migrate_v14_to_v15(connection)
                    version = 15
                    continue
                if version == 15:
                    self._migrate_v15_to_v16(connection)
                    version = 16
                    continue
                if version == 16:
                    self._migrate_v16_to_v17(connection)
                    version = 17
                    continue
                if version == 17:
                    self._migrate_v17_to_v18(connection)
                    version = 18
                    continue
                if version == 18:
                    self._migrate_v18_to_v19(connection)
                    version = 19
                    continue
                if version == 19:
                    self._migrate_v19_to_v20(connection)
                    version = 20
                    continue
                if version == 20:
                    self._migrate_v20_to_v21(connection)
                    version = 21
                    continue
                if version == 21:
                    self._migrate_v21_to_v22(connection)
                    version = 22
                    continue
                raise RuntimeError(
                    f"no migration path from schema version {version}"
                )
            self._ensure_v11_indexes(connection)
            self._ensure_v14_schema(connection)
            self._ensure_v15_schema(connection)
            self._ensure_v16_schema(connection)
            self._ensure_v17_schema(connection)
            # `_ensure_v22_schema` verifies every V20/V21 field as part of
            # its own atomic installer.  Do not replay an older trigger set
            # here: on an already-V22 table that would briefly weaken the
            # one permitted pending-to-ready guard transition.
            self._ensure_v22_schema(connection)
            # Publish the schema version only after the final hardening pass
            # has atomically installed every V22 guard.  ``executescript``
            # may commit an already-open SQLite transaction, so updating this
            # marker first could otherwise advertise a capability before its
            # immutable sidecar objects exist.
            connection.execute(
                "UPDATE schema_meta SET schema_version = ?",
                (SCHEMA_VERSION,),
            )
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

    @staticmethod
    def _migrate_v13_to_v14(connection: sqlite3.Connection) -> None:
        """Add immutable contextual-creation receipts without rebuilding data.

        v13 knowledge stores are already in active use, so this migration is
        intentionally additive: no existing Association or cue row is
        rewritten, and initialization remains safe even though ``schema.sql``
        is executed before the recorded-version migration loop.
        """

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(association_cue_prototype)"
            )
        }
        if "embedding_space_id" not in columns:
            connection.execute(
                "ALTER TABLE association_cue_prototype "
                "ADD COLUMN embedding_space_id TEXT NOT NULL DEFAULT ''"
            )
        Database._ensure_v14_schema(connection)

    @staticmethod
    def _ensure_v14_schema(connection: sqlite3.Connection) -> None:
        """Install the receipt/index-readiness tables and guards idempotently."""

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS contextual_index_publication(
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                index_epoch INTEGER NOT NULL DEFAULT 0 CHECK(index_epoch >= 0),
                embedding_space_id TEXT NOT NULL DEFAULT '',
                context_cue_count INTEGER NOT NULL DEFAULT 0
                    CHECK(context_cue_count >= 0),
                need_cue_count INTEGER NOT NULL DEFAULT 0
                    CHECK(need_cue_count >= 0),
                published_at TEXT NOT NULL DEFAULT ''
            );

            INSERT OR IGNORE INTO contextual_index_publication(
                singleton, index_epoch, context_cue_count, need_cue_count,
                published_at
            ) VALUES(1, 0, 0, 0, '');

            CREATE TABLE IF NOT EXISTS contextual_creation_receipt(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creation_request_id TEXT NOT NULL UNIQUE,
                creation_request_hash TEXT NOT NULL,
                candidate_fingerprint TEXT NOT NULL,
                association_id INTEGER NOT NULL,
                context_cue_id INTEGER NOT NULL,
                need_cue_id INTEGER NOT NULL,
                domain TEXT NOT NULL,
                model_id TEXT NOT NULL,
                embedding_space_id TEXT NOT NULL,
                dimension INTEGER NOT NULL CHECK(dimension > 0),
                dtype TEXT NOT NULL CHECK(dtype = 'float32'),
                source_facts_json TEXT NOT NULL,
                verification_refs_json TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                anchor_provenance_json TEXT NOT NULL,
                target_provenance_json TEXT NOT NULL,
                source_request_hash TEXT NOT NULL DEFAULT '',
                context_vector_ref TEXT NOT NULL DEFAULT '',
                need_vector_ref TEXT NOT NULL DEFAULT '',
                anchor_vector_ref TEXT NOT NULL DEFAULT '',
                anchor_contribution_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL CHECK(status IN (
                    'committed_pending_index',
                    'ready',
                    'legacy_pending_verification'
                )),
                ready_index_epoch INTEGER,
                durable_artifact_hash TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                ready_at TEXT,
                FOREIGN KEY(association_id) REFERENCES association(id),
                FOREIGN KEY(context_cue_id) REFERENCES association_cue_prototype(id),
                FOREIGN KEY(need_cue_id) REFERENCES association_cue_prototype(id)
            );

            CREATE INDEX IF NOT EXISTS contextual_creation_receipt_edge_idx
                ON contextual_creation_receipt(association_id);
            CREATE INDEX IF NOT EXISTS contextual_creation_receipt_status_idx
                ON contextual_creation_receipt(status, id);

            DROP TRIGGER IF EXISTS contextual_creation_receipt_immutable;
            CREATE TRIGGER contextual_creation_receipt_immutable
            BEFORE UPDATE ON contextual_creation_receipt
            WHEN
                NEW.creation_request_id <> OLD.creation_request_id
                OR NEW.creation_request_hash <> OLD.creation_request_hash
                OR NEW.candidate_fingerprint <> OLD.candidate_fingerprint
                OR NEW.association_id <> OLD.association_id
                OR NEW.context_cue_id <> OLD.context_cue_id
                OR NEW.need_cue_id <> OLD.need_cue_id
                OR NEW.domain <> OLD.domain
                OR NEW.model_id <> OLD.model_id
                OR NEW.embedding_space_id <> OLD.embedding_space_id
                OR NEW.dimension <> OLD.dimension
                OR NEW.dtype <> OLD.dtype
                OR NEW.source_facts_json <> OLD.source_facts_json
                OR NEW.verification_refs_json <> OLD.verification_refs_json
                OR NEW.verification_status <> OLD.verification_status
                OR NEW.anchor_provenance_json <> OLD.anchor_provenance_json
                OR NEW.target_provenance_json <> OLD.target_provenance_json
                OR NEW.source_request_hash <> OLD.source_request_hash
                OR NEW.context_vector_ref <> OLD.context_vector_ref
                OR NEW.need_vector_ref <> OLD.need_vector_ref
                OR NEW.anchor_vector_ref <> OLD.anchor_vector_ref
                OR NEW.anchor_contribution_id <> OLD.anchor_contribution_id
                OR NEW.durable_artifact_hash <> OLD.durable_artifact_hash
                OR NEW.created_at <> OLD.created_at
                OR (
                    OLD.status = 'ready'
                    AND (
                        NEW.status <> 'ready'
                        OR NEW.ready_index_epoch IS NOT OLD.ready_index_epoch
                        OR NEW.ready_at IS NOT OLD.ready_at
                    )
                )
                OR (
                    OLD.status = 'legacy_pending_verification'
                    AND NEW.status <> 'legacy_pending_verification'
                )
                OR (
                    OLD.status = 'committed_pending_index'
                    AND NEW.status NOT IN ('committed_pending_index', 'ready')
                )
                OR (
                    OLD.status = 'committed_pending_index'
                    AND NEW.status = 'committed_pending_index'
                    AND (
                        NEW.ready_index_epoch IS NOT OLD.ready_index_epoch
                        OR NEW.ready_at IS NOT OLD.ready_at
                    )
                )
                OR (
                    NEW.status = 'ready'
                    AND (NEW.ready_index_epoch IS NULL OR NEW.ready_index_epoch <= 0)
                )
            BEGIN
                SELECT RAISE(ABORT, 'contextual creation receipt is immutable');
            END;
            """
        )
        publication_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_index_publication)"
            )
        }
        if "embedding_space_id" not in publication_columns:
            connection.execute(
                "ALTER TABLE contextual_index_publication "
                "ADD COLUMN embedding_space_id TEXT NOT NULL DEFAULT ''"
            )
        receipt_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_creation_receipt)"
            )
        }
        for column_name in (
            "source_request_hash",
            "context_vector_ref",
            "need_vector_ref",
            "anchor_vector_ref",
            "anchor_contribution_id",
        ):
            if column_name not in receipt_columns:
                connection.execute(
                    "ALTER TABLE contextual_creation_receipt "
                    f"ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod
    def _migrate_v14_to_v15(connection: sqlite3.Connection) -> None:
        """Add the immutable utility ledger without inventing legacy history.

        Historical association counters and the bounded v11 query-hash cache
        are intentionally retained for read compatibility only.  They cannot
        prove a source-bound creation receipt, a fixed V3 selector universe,
        or a durable independent observation, so this migration never creates
        ledger rows from them.
        """

        Database._ensure_v15_schema(connection)

    @staticmethod
    def _ensure_v15_schema(connection: sqlite3.Connection) -> None:
        """Install append-only source-bound contextual utility storage."""

        # Early v15 previews predated this explicit factual-support bit.  Add
        # it before creating the v15 guards that reference it.  ``0`` means
        # unverified/unknown, never a retroactive claim that historical gains
        # were factual; the later promotion query excludes such old rows.
        existing_ledger_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_utility_ledger)"
            )
        }
        if (
            existing_ledger_columns
            and "factual_support_verified" not in existing_ledger_columns
        ):
            connection.execute(
                "ALTER TABLE contextual_utility_ledger "
                "ADD COLUMN factual_support_verified INTEGER NOT NULL DEFAULT 0 "
                "CHECK(factual_support_verified IN (0, 1))"
            )

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS contextual_utility_ledger(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id TEXT NOT NULL UNIQUE,
                family_id TEXT NOT NULL,
                association_id INTEGER NOT NULL,
                creation_receipt_id INTEGER NOT NULL,
                evaluation_as_of TEXT NOT NULL,
                candidate_universe_fingerprint TEXT NOT NULL,
                requirements_fingerprint TEXT NOT NULL,
                budget_fingerprint TEXT NOT NULL,
                input_fingerprint TEXT NOT NULL,
                treatment_fingerprint TEXT NOT NULL,
                masked_fingerprint TEXT NOT NULL,
                single_edge_fingerprint TEXT NOT NULL,
                leave_one_out_fingerprint TEXT NOT NULL,
                factual_support_verified INTEGER NOT NULL
                    CHECK(factual_support_verified IN (0, 1)),
                treatment_episode_count INTEGER NOT NULL DEFAULT 0
                    CHECK(treatment_episode_count >= 0),
                masked_episode_count INTEGER NOT NULL DEFAULT 0
                    CHECK(masked_episode_count >= 0),
                treatment_required_count INTEGER NOT NULL DEFAULT 0
                    CHECK(treatment_required_count >= 0),
                masked_required_count INTEGER NOT NULL DEFAULT 0
                    CHECK(masked_required_count >= 0),
                treatment_gain_count INTEGER NOT NULL DEFAULT 0
                    CHECK(treatment_gain_count >= 0),
                treatment_loss_count INTEGER NOT NULL DEFAULT 0
                    CHECK(treatment_loss_count >= 0),
                single_edge_gain_count INTEGER NOT NULL DEFAULT 0
                    CHECK(single_edge_gain_count >= 0),
                single_edge_loss_count INTEGER NOT NULL DEFAULT 0
                    CHECK(single_edge_loss_count >= 0),
                leave_one_out_gain_count INTEGER NOT NULL DEFAULT 0
                    CHECK(leave_one_out_gain_count >= 0),
                leave_one_out_loss_count INTEGER NOT NULL DEFAULT 0
                    CHECK(leave_one_out_loss_count >= 0),
                recall_gain INTEGER NOT NULL DEFAULT 0 CHECK(recall_gain >= 0),
                work_metric TEXT NOT NULL DEFAULT '' CHECK(work_metric IN (
                    '', 'provider_receipt_delta'
                )),
                treatment_work INTEGER,
                masked_work INTEGER,
                work_saved INTEGER,
                provider_receipt_refs_json TEXT NOT NULL DEFAULT '[]',
                harm INTEGER NOT NULL DEFAULT 0 CHECK(harm IN (0, 1)),
                is_shadow INTEGER NOT NULL DEFAULT 0 CHECK(is_shadow IN (0, 1)),
                outcome TEXT NOT NULL CHECK(outcome IN (
                    'recall_gain', 'equal_quality_faster', 'no_op', 'harmful'
                )),
                created_at TEXT NOT NULL,
                FOREIGN KEY(association_id) REFERENCES association(id),
                FOREIGN KEY(creation_receipt_id) REFERENCES contextual_creation_receipt(id),
                UNIQUE(family_id, association_id),
                CHECK(
                    (work_metric = '' AND treatment_work IS NULL
                        AND masked_work IS NULL AND work_saved IS NULL)
                    OR
                    (work_metric = 'provider_receipt_delta'
                        AND treatment_work IS NOT NULL AND masked_work IS NOT NULL
                        AND work_saved IS NOT NULL
                        AND treatment_work >= 0 AND masked_work >= 0
                        AND work_saved >= 0
                        AND work_saved = masked_work - treatment_work)
                ),
                CHECK((harm = 1 AND outcome = 'harmful')
                      OR (harm = 0 AND outcome <> 'harmful')),
                CHECK(
                    (harm = 1 AND (
                        treatment_loss_count > 0 OR single_edge_loss_count > 0
                        OR leave_one_out_loss_count > 0
                    ))
                    OR
                    (harm = 0 AND treatment_loss_count = 0
                        AND single_edge_loss_count = 0 AND leave_one_out_loss_count = 0)
                ),
                CHECK(recall_gain = 0 OR (harm = 0 AND outcome = 'recall_gain')),
                CHECK(
                    factual_support_verified = 1
                    OR (
                        recall_gain = 0
                        AND treatment_gain_count = 0
                        AND single_edge_gain_count = 0
                        AND leave_one_out_gain_count = 0
                    )
                ),
                CHECK(recall_gain <= MAX(
                    treatment_gain_count, single_edge_gain_count, leave_one_out_gain_count
                )),
                CHECK(outcome <> 'recall_gain' OR recall_gain > 0),
                CHECK(outcome <> 'equal_quality_faster' OR work_saved > 0)
            );

            CREATE INDEX IF NOT EXISTS contextual_utility_ledger_edge_idx
                ON contextual_utility_ledger(association_id, is_shadow, harm, recall_gain);
            CREATE INDEX IF NOT EXISTS contextual_utility_ledger_receipt_idx
                ON contextual_utility_ledger(creation_receipt_id, id);

            DROP TRIGGER IF EXISTS contextual_utility_ledger_source_guard;
            CREATE TRIGGER contextual_utility_ledger_source_guard
            BEFORE INSERT ON contextual_utility_ledger
            WHEN NOT EXISTS (
                SELECT 1
                FROM contextual_creation_receipt AS receipt
                JOIN association AS edge ON edge.id = receipt.association_id
                WHERE receipt.id = NEW.creation_receipt_id
                  AND receipt.association_id = NEW.association_id
                  AND edge.association_mode = 'contextual_recall'
                  AND receipt.status = 'ready'
                  AND receipt.verification_status IN ('verified', 'source_bound')
                  AND receipt.id = (
                      SELECT MIN(candidate_receipt.id)
                      FROM contextual_creation_receipt AS candidate_receipt
                      WHERE candidate_receipt.association_id = NEW.association_id
                        AND candidate_receipt.status = 'ready'
                        AND candidate_receipt.verification_status IN ('verified', 'source_bound')
                  )
            )
            BEGIN
                SELECT RAISE(ABORT, 'utility ledger needs canonical ready source-bound creation receipt');
            END;

            DROP TRIGGER IF EXISTS contextual_utility_ledger_no_replace;
            CREATE TRIGGER contextual_utility_ledger_no_replace
            BEFORE INSERT ON contextual_utility_ledger
            WHEN EXISTS (
                SELECT 1
                FROM contextual_utility_ledger AS existing
                WHERE existing.id = NEW.id
                   OR existing.observation_id = NEW.observation_id
                   OR (
                       existing.family_id = NEW.family_id
                       AND existing.association_id = NEW.association_id
                   )
            )
            BEGIN
                SELECT RAISE(ABORT, 'contextual utility ledger rows cannot be replaced');
            END;

            DROP TRIGGER IF EXISTS contextual_utility_ledger_factual_guard;
            CREATE TRIGGER contextual_utility_ledger_factual_guard
            BEFORE INSERT ON contextual_utility_ledger
            WHEN NEW.factual_support_verified = 0
             AND (
                 NEW.recall_gain <> 0
                 OR NEW.treatment_gain_count <> 0
                 OR NEW.single_edge_gain_count <> 0
                 OR NEW.leave_one_out_gain_count <> 0
             )
            BEGIN
                SELECT RAISE(ABORT, 'relevance-only utility cannot claim required coverage');
            END;

            DROP TRIGGER IF EXISTS contextual_utility_ledger_no_update;
            CREATE TRIGGER contextual_utility_ledger_no_update
            BEFORE UPDATE ON contextual_utility_ledger
            BEGIN
                SELECT RAISE(ABORT, 'contextual utility ledger is append-only');
            END;

            DROP TRIGGER IF EXISTS contextual_utility_ledger_no_delete;
            CREATE TRIGGER contextual_utility_ledger_no_delete
            BEFORE DELETE ON contextual_utility_ledger
            BEGIN
                SELECT RAISE(ABORT, 'contextual utility ledger is append-only');
            END;
            """
        )
        ledger_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_utility_ledger)"
            )
        }
        required_ledger_columns = {
            "observation_id",
            "family_id",
            "association_id",
            "creation_receipt_id",
            "factual_support_verified",
            "candidate_universe_fingerprint",
            "requirements_fingerprint",
            "budget_fingerprint",
            "input_fingerprint",
            "treatment_fingerprint",
            "masked_fingerprint",
            "single_edge_fingerprint",
            "leave_one_out_fingerprint",
        }
        missing_ledger_columns = required_ledger_columns - ledger_columns
        if missing_ledger_columns:
            raise RuntimeError(
                "contextual utility ledger is missing required immutable columns: "
                + ", ".join(sorted(missing_ledger_columns))
            )

    @staticmethod
    def _migrate_v15_to_v16(connection: sqlite3.Connection) -> None:
        """Add empty revisit contracts without inventing historic inputs.

        A v14/v15 receipt can establish that an edge was ready, but it cannot
        reconstruct the redacted selector inputs required for a later revisit.
        The additive migration therefore creates no contract rows and leaves
        all receipt/utility/association fields untouched.
        """

        Database._ensure_v16_schema(connection)

    @staticmethod
    def _ensure_v16_schema(connection: sqlite3.Connection) -> None:
        """Install immutable, source-bound, redacted revisit contracts."""

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS contextual_revisit_contract(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creation_receipt_id INTEGER NOT NULL UNIQUE,
                association_id INTEGER NOT NULL,
                context_cue_id INTEGER NOT NULL,
                need_cue_id INTEGER NOT NULL,
                domain TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(domain) = 1),
                model_id TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(model_id) = 1),
                embedding_space_id TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(embedding_space_id) = 1),
                dimension INTEGER NOT NULL CHECK(dimension > 0),
                dtype TEXT NOT NULL CHECK(dtype = 'float32'),
                context_hash TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(context_hash) = 1),
                slot_need_bindings_json TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_bindings(slot_need_bindings_json) = 1),
                requirements_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(requirements_fingerprint) = 1),
                source_closure_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(source_closure_fingerprint) = 1),
                retrieval_policy_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(retrieval_policy_fingerprint) = 1),
                budget_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(budget_fingerprint) = 1),
                anchor_manifest_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(anchor_manifest_fingerprint) = 1),
                source_fact_roles_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(source_fact_roles_fingerprint) = 1),
                source_fact_refs_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(source_fact_refs_fingerprint) = 1),
                ready_index_epoch INTEGER NOT NULL CHECK(ready_index_epoch > 0),
                ready_publication_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(ready_publication_fingerprint) = 1),
                contract_version TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(contract_version) = 1),
                contract_fingerprint TEXT NOT NULL UNIQUE
                    CHECK(contextual_revisit_opaque_digest(contract_fingerprint) = 1),
                created_at TEXT NOT NULL,
                FOREIGN KEY(creation_receipt_id)
                    REFERENCES contextual_creation_receipt(id),
                FOREIGN KEY(association_id) REFERENCES association(id),
                FOREIGN KEY(context_cue_id) REFERENCES association_cue_prototype(id),
                FOREIGN KEY(need_cue_id) REFERENCES association_cue_prototype(id)
            );

            CREATE INDEX IF NOT EXISTS contextual_revisit_contract_edge_idx
                ON contextual_revisit_contract(association_id, id);

            DROP TRIGGER IF EXISTS contextual_revisit_contract_no_replace;
            CREATE TRIGGER contextual_revisit_contract_no_replace
            BEFORE INSERT ON contextual_revisit_contract
            WHEN EXISTS (
                SELECT 1
                FROM contextual_revisit_contract AS existing
                WHERE existing.id = NEW.id
                   OR existing.creation_receipt_id = NEW.creation_receipt_id
                   OR existing.contract_fingerprint = NEW.contract_fingerprint
            )
            BEGIN
                SELECT RAISE(ABORT, 'contextual revisit contracts cannot be replaced');
            END;

            DROP TRIGGER IF EXISTS contextual_revisit_contract_redaction_guard;
            CREATE TRIGGER contextual_revisit_contract_redaction_guard
            BEFORE INSERT ON contextual_revisit_contract
            WHEN contextual_revisit_safe_identifier(NEW.domain) = 0
              OR contextual_revisit_safe_identifier(NEW.model_id) = 0
              OR contextual_revisit_safe_identifier(NEW.embedding_space_id) = 0
              OR contextual_revisit_safe_identifier(NEW.contract_version) = 0
              OR contextual_revisit_opaque_digest(NEW.context_hash) = 0
              OR contextual_revisit_opaque_bindings(NEW.slot_need_bindings_json) = 0
              OR contextual_revisit_opaque_digest(NEW.requirements_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.source_closure_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.retrieval_policy_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.budget_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.anchor_manifest_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.source_fact_roles_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.source_fact_refs_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.ready_publication_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.contract_fingerprint) = 0
            BEGIN
                SELECT RAISE(ABORT, 'contextual revisit contract must remain redacted');
            END;

            DROP TRIGGER IF EXISTS contextual_revisit_contract_source_guard;
            CREATE TRIGGER contextual_revisit_contract_source_guard
            BEFORE INSERT ON contextual_revisit_contract
            WHEN NOT EXISTS (
                SELECT 1
                FROM contextual_creation_receipt AS receipt
                JOIN association AS edge ON edge.id = receipt.association_id
                JOIN association_cue_prototype AS context_cue
                    ON context_cue.id = receipt.context_cue_id
                JOIN association_cue_prototype AS need_cue
                    ON need_cue.id = receipt.need_cue_id
                JOIN contextual_index_publication AS publication
                    ON publication.singleton = 1
                WHERE receipt.id = NEW.creation_receipt_id
                  AND receipt.association_id = NEW.association_id
                  AND receipt.context_cue_id = NEW.context_cue_id
                  AND receipt.need_cue_id = NEW.need_cue_id
                  AND receipt.domain = NEW.domain
                  AND receipt.model_id = NEW.model_id
                  AND receipt.embedding_space_id = NEW.embedding_space_id
                  AND receipt.dimension = NEW.dimension
                  AND receipt.dtype = NEW.dtype
                  AND receipt.ready_index_epoch = NEW.ready_index_epoch
                  AND edge.association_mode = 'contextual_recall'
                  AND edge.context_cue_id = receipt.context_cue_id
                  AND edge.need_cue_id = receipt.need_cue_id
                  AND context_cue.cue_kind = 'context'
                  AND need_cue.cue_kind = 'need'
                  AND context_cue.domain = receipt.domain
                  AND need_cue.domain = receipt.domain
                  AND context_cue.model_id = receipt.model_id
                  AND need_cue.model_id = receipt.model_id
                  AND context_cue.embedding_space_id = receipt.embedding_space_id
                  AND need_cue.embedding_space_id = receipt.embedding_space_id
                  AND context_cue.dimension = receipt.dimension
                  AND need_cue.dimension = receipt.dimension
                  AND context_cue.dtype = receipt.dtype
                  AND need_cue.dtype = receipt.dtype
                  AND receipt.status = 'ready'
                  AND receipt.verification_status IN ('verified', 'source_bound')
                  AND publication.embedding_space_id = receipt.embedding_space_id
                  AND publication.index_epoch >= receipt.ready_index_epoch
                  AND receipt.id = COALESCE(
                      (
                          SELECT MIN(existing_contract.creation_receipt_id)
                          FROM contextual_revisit_contract AS existing_contract
                          WHERE existing_contract.association_id = NEW.association_id
                      ),
                      (
                          SELECT MIN(existing_ledger.creation_receipt_id)
                          FROM contextual_utility_ledger AS existing_ledger
                          WHERE existing_ledger.association_id = NEW.association_id
                      ),
                      (
                          SELECT MIN(candidate_receipt.id)
                          FROM contextual_creation_receipt AS candidate_receipt
                          WHERE candidate_receipt.association_id = NEW.association_id
                            AND candidate_receipt.status = 'ready'
                            AND candidate_receipt.verification_status IN ('verified', 'source_bound')
                      )
                  )
            )
            BEGIN
                SELECT RAISE(ABORT, 'revisit contract needs canonical ready source-bound creation receipt');
            END;

            DROP TRIGGER IF EXISTS contextual_revisit_contract_no_update;
            CREATE TRIGGER contextual_revisit_contract_no_update
            BEFORE UPDATE ON contextual_revisit_contract
            BEGIN
                SELECT RAISE(ABORT, 'contextual revisit contract is immutable');
            END;

            DROP TRIGGER IF EXISTS contextual_revisit_contract_no_delete;
            CREATE TRIGGER contextual_revisit_contract_no_delete
            BEFORE DELETE ON contextual_revisit_contract
            BEGIN
                SELECT RAISE(ABORT, 'contextual revisit contract is immutable');
            END;

            -- Once either a durable contract or a ledger row selected a
            -- ready receipt, later readiness of an older pending receipt may
            -- not rewrite that selection.  This preserves exact retries and
            -- makes the first durable consumer the local canonical origin.
            DROP TRIGGER IF EXISTS contextual_utility_ledger_source_guard;
            CREATE TRIGGER contextual_utility_ledger_source_guard
            BEFORE INSERT ON contextual_utility_ledger
            WHEN NOT EXISTS (
                SELECT 1
                FROM contextual_creation_receipt AS receipt
                JOIN association AS edge ON edge.id = receipt.association_id
                WHERE receipt.id = NEW.creation_receipt_id
                  AND receipt.association_id = NEW.association_id
                  AND edge.association_mode = 'contextual_recall'
                  AND receipt.status = 'ready'
                  AND receipt.verification_status IN ('verified', 'source_bound')
                  AND receipt.id = COALESCE(
                      (
                          SELECT MIN(existing_contract.creation_receipt_id)
                          FROM contextual_revisit_contract AS existing_contract
                          WHERE existing_contract.association_id = NEW.association_id
                      ),
                      (
                          SELECT MIN(existing_ledger.creation_receipt_id)
                          FROM contextual_utility_ledger AS existing_ledger
                          WHERE existing_ledger.association_id = NEW.association_id
                      ),
                      (
                          SELECT MIN(candidate_receipt.id)
                          FROM contextual_creation_receipt AS candidate_receipt
                          WHERE candidate_receipt.association_id = NEW.association_id
                            AND candidate_receipt.status = 'ready'
                            AND candidate_receipt.verification_status IN ('verified', 'source_bound')
                      )
                  )
            )
            BEGIN
                SELECT RAISE(ABORT, 'utility ledger needs canonical ready source-bound creation receipt');
            END;
            """
        )
        contract_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_revisit_contract)"
            )
        }
        required_contract_columns = {
            "creation_receipt_id",
            "association_id",
            "context_cue_id",
            "need_cue_id",
            "context_hash",
            "slot_need_bindings_json",
            "requirements_fingerprint",
            "source_closure_fingerprint",
            "retrieval_policy_fingerprint",
            "budget_fingerprint",
            "anchor_manifest_fingerprint",
            "source_fact_roles_fingerprint",
            "source_fact_refs_fingerprint",
            "ready_index_epoch",
            "ready_publication_fingerprint",
            "contract_version",
            "contract_fingerprint",
        }
        missing_contract_columns = required_contract_columns - contract_columns
        if missing_contract_columns:
            raise RuntimeError(
                "contextual revisit contract is missing required immutable columns: "
                + ", ".join(sorted(missing_contract_columns))
            )

    @staticmethod
    def _migrate_v16_to_v17(connection: sqlite3.Connection) -> None:
        """Install empty runtime-manifest storage without inferring old Q2 data.

        A V16 contract is deliberately redacted enough that it cannot recreate
        a public Q2 request, its safe runtime slot names, or its scope binding.
        Therefore migration adds only schema/guards; it never backfills a row.
        """

        Database._ensure_v17_schema(connection)

    @staticmethod
    def _ensure_v17_schema(connection: sqlite3.Connection) -> None:
        """Install immutable pending-to-ready runtime manifests for V17.

        All columns are fixed scalar IDs, hashes, numeric limits or timestamps.
        The persistent vector remains solely in association_cue_prototype;
        question/source/answer prose has no column here.
        """

        try:
            # ``executescript`` otherwise commits each DDL replacement at its
            # own boundary.  Keep every DROP/CREATE in this V17 hardening
            # batch under one writer transaction so other connections never
            # observe a missing or weaker runtime-manifest guard.
            connection.executescript(
                """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS contextual_revisit_runtime_manifest(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creation_receipt_id INTEGER NOT NULL UNIQUE,
                association_id INTEGER NOT NULL,
                context_cue_id INTEGER NOT NULL,
                need_cue_id INTEGER NOT NULL,
                domain TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(domain) = 1),
                model_id TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(model_id) = 1),
                embedding_space_id TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(embedding_space_id) = 1),
                dimension INTEGER NOT NULL CHECK(dimension > 0),
                dtype TEXT NOT NULL CHECK(dtype = 'float32'),
                context_scope_hash TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(context_scope_hash) = 1),
                context_hash TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(context_hash) = 1),
                source_request_hash TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(source_request_hash) = 1),
                context_cue_text_hash TEXT NOT NULL
                    CHECK(contextual_revisit_sha256_hex(context_cue_text_hash) = 1),
                need_cue_text_hash TEXT NOT NULL
                    CHECK(contextual_revisit_sha256_hex(need_cue_text_hash) = 1),
                context_cue_vector_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(context_cue_vector_fingerprint) = 1),
                need_cue_vector_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(need_cue_vector_fingerprint) = 1),
                slot_need_bindings_json TEXT NOT NULL
                    CHECK(contextual_revisit_single_opaque_binding(slot_need_bindings_json) = 1),
                requirements_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(requirements_fingerprint) = 1),
                source_closure_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(source_closure_fingerprint) = 1),
                retrieval_policy_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(retrieval_policy_fingerprint) = 1),
                budget_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(budget_fingerprint) = 1),
                anchor_manifest_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(anchor_manifest_fingerprint) = 1),
                source_fact_refs_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(source_fact_refs_fingerprint) = 1),
                anchor_episode_id INTEGER NOT NULL CHECK(anchor_episode_id > 0),
                anchor_activation REAL NOT NULL
                    CHECK(anchor_activation > 0.0 AND anchor_activation = anchor_activation),
                anchor_source_fact_id TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(anchor_source_fact_id) = 1),
                target_episode_id INTEGER NOT NULL CHECK(target_episode_id > 0),
                target_source_fact_id TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(target_source_fact_id) = 1),
                target_mapping_ref TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(target_mapping_ref) = 1),
                runtime_slot_ref TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(runtime_slot_ref) = 1),
                runtime_query_ref TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(runtime_query_ref) = 1),
                runtime_clause_ref TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(runtime_clause_ref) = 1),
                endpoint_limit INTEGER NOT NULL CHECK(endpoint_limit > 0),
                episode_limit INTEGER NOT NULL CHECK(episode_limit > 0),
                source_fact_limit INTEGER CHECK(
                    source_fact_limit IS NULL OR source_fact_limit >= 0
                ),
                delivery_token_limit INTEGER CHECK(
                    delivery_token_limit IS NULL OR delivery_token_limit >= 0
                ),
                support_mode TEXT NOT NULL CHECK(support_mode = 'alternative'),
                contract_version TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(contract_version) = 1),
                manifest_version TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(manifest_version) = 1),
                source_fact_roles_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(source_fact_roles_fingerprint) = 1),
                not_before_at TEXT NOT NULL
                    CHECK(contextual_revisit_utc_instant(not_before_at) = 1),
                expires_at TEXT NOT NULL
                    CHECK(contextual_revisit_utc_instant(expires_at) = 1),
                seed_fingerprint TEXT NOT NULL UNIQUE
                    CHECK(contextual_revisit_opaque_digest(seed_fingerprint) = 1),
                binding_fingerprint TEXT NOT NULL UNIQUE
                    CHECK(contextual_revisit_opaque_digest(binding_fingerprint) = 1),
                state TEXT NOT NULL CHECK(state IN ('pending_index', 'ready')),
                ready_index_epoch INTEGER,
                ready_at TEXT,
                ready_publication_fingerprint TEXT,
                contract_id INTEGER UNIQUE,
                contract_fingerprint TEXT UNIQUE,
                manifest_fingerprint TEXT UNIQUE,
                created_at TEXT NOT NULL
                    CHECK(contextual_revisit_utc_instant(created_at) = 1),
                FOREIGN KEY(creation_receipt_id)
                    REFERENCES contextual_creation_receipt(id),
                FOREIGN KEY(association_id) REFERENCES association(id),
                FOREIGN KEY(context_cue_id) REFERENCES association_cue_prototype(id),
                FOREIGN KEY(need_cue_id) REFERENCES association_cue_prototype(id),
                FOREIGN KEY(contract_id) REFERENCES contextual_revisit_contract(id),
                CHECK(anchor_episode_id <> target_episode_id),
                CHECK(expires_at > not_before_at),
                CHECK(
                    (state = 'pending_index'
                        AND ready_index_epoch IS NULL AND ready_at IS NULL
                        AND ready_publication_fingerprint IS NULL
                        AND contract_id IS NULL AND contract_fingerprint IS NULL
                        AND manifest_fingerprint IS NULL)
                    OR
                    (state = 'ready'
                        AND ready_index_epoch IS NOT NULL AND ready_index_epoch > 0
                        AND ready_at IS NOT NULL
                        AND ready_publication_fingerprint IS NOT NULL
                        AND contract_id IS NOT NULL
                        AND contract_fingerprint IS NOT NULL
                        AND manifest_fingerprint IS NOT NULL)
                )
            );

            CREATE INDEX IF NOT EXISTS contextual_revisit_runtime_manifest_lookup_idx
                ON contextual_revisit_runtime_manifest(
                    state, domain, context_scope_hash, context_hash,
                    source_request_hash, model_id, embedding_space_id, dimension, dtype
                );
            CREATE INDEX IF NOT EXISTS contextual_revisit_runtime_manifest_edge_idx
                ON contextual_revisit_runtime_manifest(association_id, id);

            DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_no_replace;
            CREATE TRIGGER contextual_revisit_runtime_manifest_no_replace
            BEFORE INSERT ON contextual_revisit_runtime_manifest
            WHEN EXISTS (
                SELECT 1
                FROM contextual_revisit_runtime_manifest AS existing
                WHERE existing.id = NEW.id
                   OR existing.creation_receipt_id = NEW.creation_receipt_id
                   OR existing.seed_fingerprint = NEW.seed_fingerprint
                   OR existing.binding_fingerprint = NEW.binding_fingerprint
                   OR (
                       NEW.manifest_fingerprint IS NOT NULL
                       AND existing.manifest_fingerprint = NEW.manifest_fingerprint
                   )
            )
            BEGIN
                SELECT RAISE(ABORT, 'contextual runtime manifests cannot be replaced');
            END;

            DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_redaction_guard;
            CREATE TRIGGER contextual_revisit_runtime_manifest_redaction_guard
            BEFORE INSERT ON contextual_revisit_runtime_manifest
            WHEN NEW.state <> 'pending_index'
              OR contextual_revisit_safe_identifier(NEW.domain) = 0
              OR contextual_revisit_safe_identifier(NEW.model_id) = 0
              OR contextual_revisit_safe_identifier(NEW.embedding_space_id) = 0
              OR contextual_revisit_safe_identifier(NEW.contract_version) = 0
              OR contextual_revisit_safe_identifier(NEW.manifest_version) = 0
              OR contextual_revisit_opaque_digest(NEW.context_scope_hash) = 0
              OR contextual_revisit_opaque_digest(NEW.context_hash) = 0
              OR contextual_revisit_opaque_digest(NEW.source_request_hash) = 0
              OR contextual_revisit_sha256_hex(NEW.context_cue_text_hash) = 0
              OR contextual_revisit_sha256_hex(NEW.need_cue_text_hash) = 0
              OR contextual_revisit_opaque_digest(NEW.context_cue_vector_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.need_cue_vector_fingerprint) = 0
              OR contextual_revisit_single_opaque_binding(NEW.slot_need_bindings_json) = 0
              OR contextual_revisit_opaque_digest(NEW.requirements_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.source_closure_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.retrieval_policy_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.budget_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.anchor_manifest_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.source_fact_refs_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.anchor_source_fact_id) = 0
              OR contextual_revisit_opaque_digest(NEW.target_source_fact_id) = 0
              OR contextual_revisit_opaque_digest(NEW.target_mapping_ref) = 0
              OR contextual_revisit_opaque_digest(NEW.runtime_slot_ref) = 0
              OR contextual_revisit_opaque_digest(NEW.runtime_query_ref) = 0
              OR contextual_revisit_opaque_digest(NEW.runtime_clause_ref) = 0
              OR contextual_revisit_opaque_digest(NEW.source_fact_roles_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.seed_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.binding_fingerprint) = 0
              OR contextual_revisit_utc_instant(NEW.not_before_at) = 0
              OR contextual_revisit_utc_instant(NEW.expires_at) = 0
              OR contextual_revisit_utc_instant(NEW.created_at) = 0
            BEGIN
                SELECT RAISE(ABORT, 'contextual runtime manifest must remain redacted');
            END;

            DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_source_guard;
            CREATE TRIGGER contextual_revisit_runtime_manifest_source_guard
            BEFORE INSERT ON contextual_revisit_runtime_manifest
            WHEN NOT EXISTS (
                SELECT 1
                FROM contextual_creation_receipt AS receipt
                JOIN association AS edge ON edge.id = receipt.association_id
                JOIN association_cue_prototype AS context_cue
                    ON context_cue.id = receipt.context_cue_id
                JOIN association_cue_prototype AS need_cue
                    ON need_cue.id = receipt.need_cue_id
                JOIN episode AS anchor ON anchor.id = edge.from_id
                JOIN episode AS target ON target.id = edge.to_id
                WHERE receipt.id = NEW.creation_receipt_id
                  AND receipt.association_id = NEW.association_id
                  AND receipt.context_cue_id = NEW.context_cue_id
                  AND receipt.need_cue_id = NEW.need_cue_id
                  AND receipt.domain = NEW.domain
                  AND receipt.model_id = NEW.model_id
                  AND receipt.embedding_space_id = NEW.embedding_space_id
                  AND receipt.dimension = NEW.dimension
                  AND receipt.dtype = NEW.dtype
                  AND receipt.source_request_hash = NEW.source_request_hash
                  AND receipt.status = 'committed_pending_index'
                  AND receipt.verification_status IN ('verified', 'source_bound')
                  AND edge.association_mode = 'contextual_recall'
                  AND edge.from_type = 'episode'
                  AND edge.from_id = NEW.anchor_episode_id
                  AND edge.to_type = 'episode'
                  AND edge.to_id = NEW.target_episode_id
                  AND edge.context_cue_id = receipt.context_cue_id
                  AND edge.need_cue_id = receipt.need_cue_id
                  AND edge.expires_at = NEW.expires_at
                  AND anchor.generation = 0
                  AND target.generation = 0
                  AND context_cue.cue_kind = 'context'
                  AND need_cue.cue_kind = 'need'
                  AND context_cue.domain = receipt.domain
                  AND need_cue.domain = receipt.domain
                  AND context_cue.model_id = receipt.model_id
                  AND need_cue.model_id = receipt.model_id
                  AND context_cue.embedding_space_id = receipt.embedding_space_id
                  AND need_cue.embedding_space_id = receipt.embedding_space_id
                  AND context_cue.dimension = receipt.dimension
                  AND need_cue.dimension = receipt.dimension
                  AND context_cue.dtype = receipt.dtype
                  AND need_cue.dtype = receipt.dtype
                  AND context_cue.text_hash = NEW.context_cue_text_hash
                  AND need_cue.text_hash = NEW.need_cue_text_hash
                  AND NEW.context_cue_vector_fingerprint =
                      contextual_revisit_vector_fingerprint(context_cue.vector_blob)
                  AND NEW.need_cue_vector_fingerprint =
                      contextual_revisit_vector_fingerprint(need_cue.vector_blob)
                  AND NEW.not_before_at = receipt.created_at
                  AND receipt.id = (
                      SELECT MIN(candidate_receipt.id)
                      FROM contextual_creation_receipt AS candidate_receipt
                      WHERE candidate_receipt.association_id = NEW.association_id
                        AND candidate_receipt.status IN ('committed_pending_index', 'ready')
                        AND candidate_receipt.verification_status IN ('verified', 'source_bound')
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM contextual_revisit_contract AS existing_contract
                      WHERE existing_contract.association_id = NEW.association_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM contextual_utility_ledger AS existing_ledger
                      WHERE existing_ledger.association_id = NEW.association_id
                  )
            )
            BEGIN
                SELECT RAISE(ABORT, 'runtime manifest needs canonical pending source-bound receipt');
            END;

            DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_ready_transition;
            CREATE TRIGGER contextual_revisit_runtime_manifest_ready_transition
            BEFORE UPDATE ON contextual_revisit_runtime_manifest
            WHEN OLD.state <> 'pending_index'
              OR NEW.state <> 'ready'
              OR NEW.creation_receipt_id IS NOT OLD.creation_receipt_id
              OR NEW.association_id IS NOT OLD.association_id
              OR NEW.context_cue_id IS NOT OLD.context_cue_id
              OR NEW.need_cue_id IS NOT OLD.need_cue_id
              OR NEW.domain IS NOT OLD.domain
              OR NEW.model_id IS NOT OLD.model_id
              OR NEW.embedding_space_id IS NOT OLD.embedding_space_id
              OR NEW.dimension IS NOT OLD.dimension
              OR NEW.dtype IS NOT OLD.dtype
              OR NEW.context_scope_hash IS NOT OLD.context_scope_hash
              OR NEW.context_hash IS NOT OLD.context_hash
              OR NEW.source_request_hash IS NOT OLD.source_request_hash
              OR NEW.context_cue_text_hash IS NOT OLD.context_cue_text_hash
              OR NEW.need_cue_text_hash IS NOT OLD.need_cue_text_hash
              OR NEW.context_cue_vector_fingerprint IS NOT OLD.context_cue_vector_fingerprint
              OR NEW.need_cue_vector_fingerprint IS NOT OLD.need_cue_vector_fingerprint
              OR NEW.slot_need_bindings_json IS NOT OLD.slot_need_bindings_json
              OR NEW.requirements_fingerprint IS NOT OLD.requirements_fingerprint
              OR NEW.source_closure_fingerprint IS NOT OLD.source_closure_fingerprint
              OR NEW.retrieval_policy_fingerprint IS NOT OLD.retrieval_policy_fingerprint
              OR NEW.budget_fingerprint IS NOT OLD.budget_fingerprint
              OR NEW.anchor_manifest_fingerprint IS NOT OLD.anchor_manifest_fingerprint
              OR NEW.source_fact_refs_fingerprint IS NOT OLD.source_fact_refs_fingerprint
              OR NEW.anchor_episode_id IS NOT OLD.anchor_episode_id
              OR NEW.anchor_activation IS NOT OLD.anchor_activation
              OR NEW.anchor_source_fact_id IS NOT OLD.anchor_source_fact_id
              OR NEW.target_episode_id IS NOT OLD.target_episode_id
              OR NEW.target_source_fact_id IS NOT OLD.target_source_fact_id
              OR NEW.target_mapping_ref IS NOT OLD.target_mapping_ref
              OR NEW.runtime_slot_ref IS NOT OLD.runtime_slot_ref
              OR NEW.runtime_query_ref IS NOT OLD.runtime_query_ref
              OR NEW.runtime_clause_ref IS NOT OLD.runtime_clause_ref
              OR NEW.endpoint_limit IS NOT OLD.endpoint_limit
              OR NEW.episode_limit IS NOT OLD.episode_limit
              OR NEW.source_fact_limit IS NOT OLD.source_fact_limit
              OR NEW.delivery_token_limit IS NOT OLD.delivery_token_limit
              OR NEW.support_mode IS NOT OLD.support_mode
              OR NEW.contract_version IS NOT OLD.contract_version
              OR NEW.manifest_version IS NOT OLD.manifest_version
              OR NEW.source_fact_roles_fingerprint IS NOT OLD.source_fact_roles_fingerprint
              OR NEW.not_before_at IS NOT OLD.not_before_at
              OR NEW.expires_at IS NOT OLD.expires_at
              OR NEW.seed_fingerprint IS NOT OLD.seed_fingerprint
              OR NEW.binding_fingerprint IS NOT OLD.binding_fingerprint
              OR NEW.created_at IS NOT OLD.created_at
              OR NEW.ready_index_epoch IS NULL
              OR NEW.ready_at IS NULL
              OR NEW.ready_publication_fingerprint IS NULL
              OR NEW.contract_id IS NULL
              OR NEW.contract_fingerprint IS NULL
              OR NEW.manifest_fingerprint IS NULL
              OR contextual_revisit_opaque_digest(NEW.ready_publication_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.contract_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.manifest_fingerprint) = 0
              OR contextual_revisit_utc_instant(NEW.ready_at) = 0
              OR NEW.ready_at < NEW.not_before_at
              OR NOT EXISTS (
                  SELECT 1
                  FROM contextual_creation_receipt AS receipt
                  JOIN contextual_index_publication AS publication
                    ON publication.singleton = 1
                  JOIN contextual_revisit_contract AS contract
                    ON contract.id = NEW.contract_id
                  WHERE receipt.id = NEW.creation_receipt_id
                    AND receipt.association_id = NEW.association_id
                    AND receipt.context_cue_id = NEW.context_cue_id
                    AND receipt.need_cue_id = NEW.need_cue_id
                    AND receipt.status = 'ready'
                    AND receipt.verification_status IN ('verified', 'source_bound')
                    AND receipt.ready_index_epoch = NEW.ready_index_epoch
                    AND receipt.ready_at = NEW.ready_at
                    AND publication.embedding_space_id = NEW.embedding_space_id
                    AND publication.index_epoch >= NEW.ready_index_epoch
                    AND NEW.ready_publication_fingerprint =
                        contextual_revisit_ready_fingerprint(
                            receipt.id, receipt.association_id, receipt.context_cue_id,
                            receipt.need_cue_id, receipt.domain, receipt.model_id,
                            receipt.embedding_space_id, receipt.dimension, receipt.dtype,
                            receipt.durable_artifact_hash, receipt.ready_at,
                            receipt.ready_index_epoch, receipt.status,
                            receipt.verification_status
                        )
                    AND contract.creation_receipt_id = NEW.creation_receipt_id
                    AND contract.association_id = NEW.association_id
                    AND contract.context_cue_id = NEW.context_cue_id
                    AND contract.need_cue_id = NEW.need_cue_id
                    AND contract.domain = NEW.domain
                    AND contract.model_id = NEW.model_id
                    AND contract.embedding_space_id = NEW.embedding_space_id
                    AND contract.dimension = NEW.dimension
                    AND contract.dtype = NEW.dtype
                    AND contract.context_hash = NEW.context_hash
                    AND contract.slot_need_bindings_json = NEW.slot_need_bindings_json
                    AND contract.requirements_fingerprint = NEW.requirements_fingerprint
                    AND contract.source_closure_fingerprint = NEW.source_closure_fingerprint
                    AND contract.retrieval_policy_fingerprint = NEW.retrieval_policy_fingerprint
                    AND contract.budget_fingerprint = NEW.budget_fingerprint
                    AND contract.anchor_manifest_fingerprint = NEW.anchor_manifest_fingerprint
                    AND contract.source_fact_roles_fingerprint = NEW.source_fact_roles_fingerprint
                    AND contract.source_fact_refs_fingerprint = NEW.source_fact_refs_fingerprint
                    AND contract.ready_index_epoch = NEW.ready_index_epoch
                    AND contract.ready_publication_fingerprint = NEW.ready_publication_fingerprint
                    AND contract.contract_version = NEW.contract_version
                    AND contract.contract_fingerprint = NEW.contract_fingerprint
              )
            BEGIN
                SELECT RAISE(ABORT, 'contextual runtime manifest permits only one verified ready transition');
            END;

            DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_no_delete;
            CREATE TRIGGER contextual_revisit_runtime_manifest_no_delete
            BEFORE DELETE ON contextual_revisit_runtime_manifest
            BEGIN
                SELECT RAISE(ABORT, 'contextual runtime manifest is immutable');
            END;
            COMMIT;
            """
            )
        except sqlite3.DatabaseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        manifest_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_revisit_runtime_manifest)"
            )
        }
        required_manifest_columns = {
            "creation_receipt_id",
            "association_id",
            "context_cue_id",
            "need_cue_id",
            "context_scope_hash",
            "context_hash",
            "source_request_hash",
            "context_cue_text_hash",
            "need_cue_text_hash",
            "context_cue_vector_fingerprint",
            "need_cue_vector_fingerprint",
            "slot_need_bindings_json",
            "anchor_episode_id",
            "anchor_activation",
            "anchor_source_fact_id",
            "target_episode_id",
            "target_source_fact_id",
            "target_mapping_ref",
            "runtime_slot_ref",
            "runtime_query_ref",
            "runtime_clause_ref",
            "not_before_at",
            "expires_at",
            "seed_fingerprint",
            "binding_fingerprint",
            "state",
            "ready_index_epoch",
            "ready_publication_fingerprint",
            "contract_id",
            "contract_fingerprint",
            "manifest_fingerprint",
        }
        missing_manifest_columns = required_manifest_columns - manifest_columns
        if missing_manifest_columns:
            raise RuntimeError(
                "contextual runtime manifest is missing required immutable columns: "
                + ", ".join(sorted(missing_manifest_columns))
            )

    @staticmethod
    def _migrate_v17_to_v18(connection: sqlite3.Connection) -> None:
        """Install empty restricted-rewrite guards without inventing Q1 data.

        A legacy V17 manifest intentionally lacks the trusted, fully consumed
        grammar IR and its HMAC commitment.  Backfilling would therefore turn
        an old exact request into a guessed rewrite capability, so V18 only
        creates the immutable table and its guards.
        """

        Database._ensure_v18_schema(connection)

    @staticmethod
    def _ensure_v18_schema(connection: sqlite3.Connection) -> None:
        """Install immutable, HMAC-only T16 restricted-rewrite sidecars.

        The sidecar is written only beside a pending V17 runtime seed.  It has
        no mutable ready state: a read must join the already-authoritative
        ready manifest and re-run that manifest's source/cue/contract checks.
        """

        try:
            connection.executescript(
                """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS contextual_restricted_rewrite_guard(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                creation_receipt_id INTEGER NOT NULL UNIQUE,
                association_id INTEGER NOT NULL,
                domain TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(domain) = 1),
                context_scope_hash TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(context_scope_hash) = 1),
                rewrite_commitment TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(rewrite_commitment) = 1),
                commitment_key_id TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(commitment_key_id) = 1),
                grammar_version TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(grammar_version) = 1),
                seed_fingerprint TEXT NOT NULL
                    CHECK(contextual_revisit_opaque_digest(seed_fingerprint) = 1),
                signature_version TEXT NOT NULL
                    CHECK(contextual_revisit_safe_identifier(signature_version) = 1),
                created_at TEXT NOT NULL
                    CHECK(contextual_revisit_utc_instant(created_at) = 1),
                guard_fingerprint TEXT NOT NULL UNIQUE
                    CHECK(contextual_revisit_opaque_digest(guard_fingerprint) = 1),
                FOREIGN KEY(creation_receipt_id)
                    REFERENCES contextual_creation_receipt(id),
                FOREIGN KEY(association_id) REFERENCES association(id)
            );

            CREATE INDEX IF NOT EXISTS contextual_restricted_rewrite_guard_lookup_idx
                ON contextual_restricted_rewrite_guard(
                    association_id, domain, context_scope_hash,
                    rewrite_commitment, commitment_key_id, grammar_version,
                    signature_version
                );
            CREATE INDEX IF NOT EXISTS contextual_restricted_rewrite_guard_root_lookup_idx
                ON contextual_restricted_rewrite_guard(
                    domain, context_scope_hash, rewrite_commitment,
                    commitment_key_id, grammar_version, signature_version, id
                );

            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_replace;
            CREATE TRIGGER contextual_restricted_rewrite_guard_no_replace
            BEFORE INSERT ON contextual_restricted_rewrite_guard
            WHEN EXISTS (
                SELECT 1
                FROM contextual_restricted_rewrite_guard AS existing
                WHERE existing.id = NEW.id
                   OR existing.creation_receipt_id = NEW.creation_receipt_id
                   OR existing.guard_fingerprint = NEW.guard_fingerprint
            )
            BEGIN
                SELECT RAISE(ABORT, 'contextual restricted rewrite guards cannot be replaced');
            END;

            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_redaction_guard;
            CREATE TRIGGER contextual_restricted_rewrite_guard_redaction_guard
            BEFORE INSERT ON contextual_restricted_rewrite_guard
            WHEN contextual_revisit_safe_identifier(NEW.domain) = 0
              OR contextual_revisit_opaque_digest(NEW.context_scope_hash) = 0
              OR contextual_revisit_opaque_digest(NEW.rewrite_commitment) = 0
              OR contextual_revisit_safe_identifier(NEW.commitment_key_id) = 0
              OR contextual_revisit_safe_identifier(NEW.grammar_version) = 0
              OR contextual_revisit_opaque_digest(NEW.seed_fingerprint) = 0
              OR contextual_revisit_safe_identifier(NEW.signature_version) = 0
              OR contextual_revisit_utc_instant(NEW.created_at) = 0
              OR contextual_revisit_opaque_digest(NEW.guard_fingerprint) = 0
            BEGIN
                SELECT RAISE(ABORT, 'contextual restricted rewrite guard must remain redacted');
            END;

            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_source_guard;
            CREATE TRIGGER contextual_restricted_rewrite_guard_source_guard
            BEFORE INSERT ON contextual_restricted_rewrite_guard
            WHEN NOT EXISTS (
                SELECT 1
                FROM contextual_creation_receipt AS receipt
                JOIN contextual_revisit_runtime_manifest AS manifest
                  ON manifest.creation_receipt_id = receipt.id
                WHERE receipt.id = NEW.creation_receipt_id
                  AND receipt.association_id = NEW.association_id
                  AND receipt.domain = NEW.domain
                  AND receipt.status = 'committed_pending_index'
                  AND receipt.verification_status IN ('verified', 'source_bound')
                  AND receipt.created_at = NEW.created_at
                  AND manifest.association_id = NEW.association_id
                  AND manifest.domain = NEW.domain
                  AND manifest.context_scope_hash = NEW.context_scope_hash
                  AND manifest.seed_fingerprint = NEW.seed_fingerprint
                  AND manifest.state = 'pending_index'
                  AND manifest.created_at = NEW.created_at
                  AND manifest.not_before_at = NEW.created_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'restricted rewrite guard needs canonical pending runtime manifest');
            END;

            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update;
            CREATE TRIGGER contextual_restricted_rewrite_guard_no_update
            BEFORE UPDATE ON contextual_restricted_rewrite_guard
            BEGIN
                SELECT RAISE(ABORT, 'contextual restricted rewrite guard is immutable');
            END;

            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_delete;
            CREATE TRIGGER contextual_restricted_rewrite_guard_no_delete
            BEFORE DELETE ON contextual_restricted_rewrite_guard
            BEGIN
                SELECT RAISE(ABORT, 'contextual restricted rewrite guard is immutable');
            END;
            COMMIT;
            """
            )
        except sqlite3.DatabaseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        guard_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        required_guard_columns = {
            "creation_receipt_id",
            "association_id",
            "domain",
            "context_scope_hash",
            "rewrite_commitment",
            "commitment_key_id",
            "grammar_version",
            "seed_fingerprint",
            "signature_version",
            "created_at",
            "guard_fingerprint",
        }
        missing_guard_columns = required_guard_columns - guard_columns
        if missing_guard_columns:
            raise RuntimeError(
                "contextual restricted rewrite guard is missing required immutable columns: "
                + ", ".join(sorted(missing_guard_columns))
            )

    @staticmethod
    def _migrate_v18_to_v19(connection: sqlite3.Connection) -> None:
        """Add an HMAC seed binding without authorizing legacy V18 rows.

        V18 roots were deliberately opaque but could be copied alongside a
        recomputed public guard fingerprint by a database-only attacker.  The
        V19 column is therefore added empty for pre-existing rows rather than
        guessed/backfilled; the typed reader rejects those rows, while every
        new insert must carry the second process-key HMAC.
        """

        if connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND name = 'contextual_restricted_rewrite_guard_v18_legacy'
            """
        ).fetchone() is not None:
            raise RuntimeError(
                "interrupted V19 restricted-rewrite migration detected; "
                "restore the database snapshot before retrying"
            )
        Database._ensure_v18_schema(connection)
        Database._ensure_v19_schema(connection)

    @staticmethod
    def _restricted_rewrite_guard_has_unique_association(
        connection: sqlite3.Connection,
    ) -> bool:
        """Detect the V18-only one-guard-per-edge index without guessing names."""

        for index in connection.execute(
            "PRAGMA index_list(contextual_restricted_rewrite_guard)"
        ):
            if not bool(index["unique"]):
                continue
            index_name = str(index["name"]).replace('"', '""')
            index_columns = [
                str(item["name"])
                for item in connection.execute(f'PRAGMA index_info("{index_name}")')
            ]
            if index_columns == ["association_id"]:
                return True
        return False

    @staticmethod
    def _restricted_rewrite_guard_objects_sql(
        *,
        require_v20_provenance: bool,
        require_v21_manifest_binding: bool = False,
        require_v22_ready_manifest_binding: bool = False,
    ) -> str:
        """Return the complete immutable sidecar object set for one schema cut.

        Callers place this fragment inside one explicit SQLite writer
        transaction.  In particular, a table rebuild must not publish the new
        table before its indexes and every insert/update/delete guard exist.
        """

        provenance_redaction = """
              OR contextual_revisit_opaque_digest(NEW.context_cue_vector_fingerprint) = 0
              OR contextual_revisit_opaque_digest(NEW.need_cue_vector_fingerprint) = 0
              OR contextual_revisit_safe_identifier(NEW.model_id) = 0
              OR contextual_revisit_safe_identifier(NEW.embedding_space_id) = 0
              OR NEW.dimension <= 0
              OR NEW.dtype <> 'float32'
        """ if require_v20_provenance else ""
        provenance_source = """
                  AND NEW.context_cue_vector_fingerprint =
                      manifest.context_cue_vector_fingerprint
                  AND NEW.need_cue_vector_fingerprint =
                      manifest.need_cue_vector_fingerprint
                  AND NEW.model_id = manifest.model_id
                  AND NEW.embedding_space_id = manifest.embedding_space_id
                  AND NEW.dimension = manifest.dimension
                  AND NEW.dtype = manifest.dtype
                  AND NEW.context_cue_vector_fingerprint =
                      contextual_revisit_vector_fingerprint(context_cue.vector_blob)
                  AND NEW.need_cue_vector_fingerprint =
                      contextual_revisit_vector_fingerprint(need_cue.vector_blob)
                  AND context_cue.model_id = NEW.model_id
                  AND need_cue.model_id = NEW.model_id
                  AND context_cue.embedding_space_id = NEW.embedding_space_id
                  AND need_cue.embedding_space_id = NEW.embedding_space_id
                  AND context_cue.dimension = NEW.dimension
                  AND need_cue.dimension = NEW.dimension
                  AND context_cue.dtype = NEW.dtype
                  AND need_cue.dtype = NEW.dtype
        """ if require_v20_provenance else ""
        manifest_binding_redaction = """
              OR contextual_revisit_opaque_digest(NEW.manifest_binding_commitment) = 0
        """ if require_v21_manifest_binding else ""
        ready_manifest_insert_redaction = """
              OR NEW.ready_manifest_commitment <> ''
        """ if require_v22_ready_manifest_binding else ""
        ready_manifest_insert_source = """
                  AND NEW.ready_manifest_commitment = ''
        """ if require_v22_ready_manifest_binding else ""
        update_guard_sql = """
            CREATE TRIGGER contextual_restricted_rewrite_guard_no_update
            BEFORE UPDATE ON contextual_restricted_rewrite_guard
            BEGIN
                SELECT RAISE(ABORT, 'contextual restricted rewrite guard is immutable');
            END;
        """
        if require_v22_ready_manifest_binding:
            # V22 permits one and only one trusted publication transition:
            # the opaque ready HMAC plus its public guard fingerprint may be
            # filled after the matching manifest is ready.  Every other
            # guard field remains immutable.  The engine still recomputes the
            # HMAC itself, so a raw database writer cannot use this structural
            # exception to forge authorization.
            update_guard_sql = """
                CREATE TRIGGER contextual_restricted_rewrite_guard_no_update
                BEFORE UPDATE ON contextual_restricted_rewrite_guard
                WHEN NOT (
                    NEW.id = OLD.id
                    AND NEW.creation_receipt_id = OLD.creation_receipt_id
                    AND NEW.association_id = OLD.association_id
                    AND NEW.domain = OLD.domain
                    AND NEW.context_scope_hash = OLD.context_scope_hash
                    AND NEW.rewrite_commitment = OLD.rewrite_commitment
                    AND NEW.binding_commitment = OLD.binding_commitment
                    AND NEW.manifest_binding_commitment = OLD.manifest_binding_commitment
                    AND NEW.context_cue_vector_fingerprint = OLD.context_cue_vector_fingerprint
                    AND NEW.need_cue_vector_fingerprint = OLD.need_cue_vector_fingerprint
                    AND NEW.model_id = OLD.model_id
                    AND NEW.embedding_space_id = OLD.embedding_space_id
                    AND NEW.dimension = OLD.dimension
                    AND NEW.dtype = OLD.dtype
                    AND NEW.commitment_key_id = OLD.commitment_key_id
                    AND NEW.grammar_version = OLD.grammar_version
                    AND NEW.seed_fingerprint = OLD.seed_fingerprint
                    AND NEW.signature_version = OLD.signature_version
                    AND NEW.created_at = OLD.created_at
                    AND OLD.ready_manifest_commitment = ''
                    AND NEW.ready_manifest_commitment <> ''
                    AND contextual_revisit_opaque_digest(NEW.ready_manifest_commitment) = 1
                    AND NEW.guard_fingerprint <> OLD.guard_fingerprint
                    AND contextual_revisit_opaque_digest(NEW.guard_fingerprint) = 1
                    AND NEW.signature_version = 'contextual-restricted-rewrite-guard-v5'
                    AND EXISTS (
                        SELECT 1
                        FROM contextual_creation_receipt AS receipt
                        JOIN contextual_revisit_runtime_manifest AS manifest
                          ON manifest.creation_receipt_id = receipt.id
                        WHERE receipt.id = OLD.creation_receipt_id
                          AND receipt.association_id = OLD.association_id
                          AND receipt.domain = OLD.domain
                          AND receipt.status = 'ready'
                          AND manifest.association_id = OLD.association_id
                          AND manifest.domain = OLD.domain
                          AND manifest.context_scope_hash = OLD.context_scope_hash
                          AND manifest.seed_fingerprint = OLD.seed_fingerprint
                          AND manifest.state = 'ready'
                          AND manifest.manifest_fingerprint IS NOT NULL
                    )
                )
                BEGIN
                    SELECT RAISE(ABORT, 'contextual restricted rewrite guard is immutable');
                END;
            """
        return f"""
            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_replace;
            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_redaction_guard;
            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_source_guard;
            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update;
            DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_delete;
            DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_lookup_idx;
            DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_root_lookup_idx;

            CREATE INDEX contextual_restricted_rewrite_guard_lookup_idx
                ON contextual_restricted_rewrite_guard(
                    association_id, domain, context_scope_hash,
                    rewrite_commitment, commitment_key_id, grammar_version,
                    signature_version
                );
            CREATE INDEX contextual_restricted_rewrite_guard_root_lookup_idx
                ON contextual_restricted_rewrite_guard(
                    domain, context_scope_hash, rewrite_commitment,
                    commitment_key_id, grammar_version, signature_version, id
                );

            CREATE TRIGGER contextual_restricted_rewrite_guard_no_replace
            BEFORE INSERT ON contextual_restricted_rewrite_guard
            WHEN EXISTS (
                SELECT 1
                FROM contextual_restricted_rewrite_guard AS existing
                WHERE existing.id = NEW.id
                   OR existing.creation_receipt_id = NEW.creation_receipt_id
                   OR existing.guard_fingerprint = NEW.guard_fingerprint
            )
            BEGIN
                SELECT RAISE(ABORT, 'contextual restricted rewrite guards cannot be replaced');
            END;

            CREATE TRIGGER contextual_restricted_rewrite_guard_redaction_guard
            BEFORE INSERT ON contextual_restricted_rewrite_guard
            WHEN contextual_revisit_safe_identifier(NEW.domain) = 0
              OR contextual_revisit_opaque_digest(NEW.context_scope_hash) = 0
              OR contextual_revisit_opaque_digest(NEW.rewrite_commitment) = 0
              OR contextual_revisit_opaque_digest(NEW.binding_commitment) = 0
              {manifest_binding_redaction}
              {ready_manifest_insert_redaction}
              OR contextual_revisit_safe_identifier(NEW.commitment_key_id) = 0
              OR contextual_revisit_safe_identifier(NEW.grammar_version) = 0
              OR contextual_revisit_opaque_digest(NEW.seed_fingerprint) = 0
              OR contextual_revisit_safe_identifier(NEW.signature_version) = 0
              OR contextual_revisit_utc_instant(NEW.created_at) = 0
              OR contextual_revisit_opaque_digest(NEW.guard_fingerprint) = 0
              {provenance_redaction}
            BEGIN
                SELECT RAISE(ABORT, 'contextual restricted rewrite guard must remain redacted');
            END;

            CREATE TRIGGER contextual_restricted_rewrite_guard_source_guard
            BEFORE INSERT ON contextual_restricted_rewrite_guard
            WHEN NOT EXISTS (
                SELECT 1
                FROM contextual_creation_receipt AS receipt
                JOIN contextual_revisit_runtime_manifest AS manifest
                  ON manifest.creation_receipt_id = receipt.id
                JOIN association_cue_prototype AS context_cue
                  ON context_cue.id = manifest.context_cue_id
                JOIN association_cue_prototype AS need_cue
                  ON need_cue.id = manifest.need_cue_id
                WHERE receipt.id = NEW.creation_receipt_id
                  AND receipt.association_id = NEW.association_id
                  AND receipt.domain = NEW.domain
                  AND receipt.status = 'committed_pending_index'
                  AND receipt.verification_status IN ('verified', 'source_bound')
                  AND receipt.created_at = NEW.created_at
                  AND manifest.association_id = NEW.association_id
                  AND manifest.domain = NEW.domain
                  AND manifest.context_scope_hash = NEW.context_scope_hash
                  AND manifest.seed_fingerprint = NEW.seed_fingerprint
                  AND manifest.state = 'pending_index'
                  AND manifest.created_at = NEW.created_at
                  AND manifest.not_before_at = NEW.created_at
                  {provenance_source}
                  {ready_manifest_insert_source}
            )
            BEGIN
                SELECT RAISE(ABORT, 'restricted rewrite guard needs canonical pending runtime manifest');
            END;

            {update_guard_sql}

            CREATE TRIGGER contextual_restricted_rewrite_guard_no_delete
            BEFORE DELETE ON contextual_restricted_rewrite_guard
            BEGIN
                SELECT RAISE(ABORT, 'contextual restricted rewrite guard is immutable');
            END;
        """

    @staticmethod
    def _ensure_v19_schema(connection: sqlite3.Connection) -> None:
        """Require a root-to-seed HMAC while preserving receipt history.

        V18 did not have a binding HMAC and also over-constrained durable edges
        to one sidecar.  A V19 rebuild is atomic: the table, indexes, and all
        guards become visible together.  Legacy rows intentionally receive an
        empty binding, which the typed reader treats as a normal cache miss.
        """

        interrupted_legacy = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND name = 'contextual_restricted_rewrite_guard_v18_legacy'
            """
        ).fetchone()
        if interrupted_legacy is not None:
            raise RuntimeError(
                "interrupted V19 restricted-rewrite migration detected; "
                "restore the database snapshot before retrying"
            )
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        needs_rebuild = (
            "binding_commitment" not in columns
            or Database._restricted_rewrite_guard_has_unique_association(connection)
        )
        binding_expression = (
            "binding_commitment" if "binding_commitment" in columns else "''"
        )
        if needs_rebuild:
            script = f"""
                BEGIN IMMEDIATE;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_replace;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_redaction_guard;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_source_guard;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_delete;
                DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_lookup_idx;
                DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_root_lookup_idx;
                ALTER TABLE contextual_restricted_rewrite_guard
                    RENAME TO contextual_restricted_rewrite_guard_v18_legacy;
                CREATE TABLE contextual_restricted_rewrite_guard(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    creation_receipt_id INTEGER NOT NULL UNIQUE,
                    association_id INTEGER NOT NULL,
                    domain TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(domain) = 1),
                    context_scope_hash TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(context_scope_hash) = 1),
                    rewrite_commitment TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(rewrite_commitment) = 1),
                    binding_commitment TEXT NOT NULL DEFAULT '',
                    commitment_key_id TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(commitment_key_id) = 1),
                    grammar_version TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(grammar_version) = 1),
                    seed_fingerprint TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(seed_fingerprint) = 1),
                    signature_version TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(signature_version) = 1),
                    created_at TEXT NOT NULL
                        CHECK(contextual_revisit_utc_instant(created_at) = 1),
                    guard_fingerprint TEXT NOT NULL UNIQUE
                        CHECK(contextual_revisit_opaque_digest(guard_fingerprint) = 1),
                    FOREIGN KEY(creation_receipt_id)
                        REFERENCES contextual_creation_receipt(id),
                    FOREIGN KEY(association_id) REFERENCES association(id)
                );
                INSERT INTO contextual_restricted_rewrite_guard(
                    id, creation_receipt_id, association_id, domain,
                    context_scope_hash, rewrite_commitment, binding_commitment,
                    commitment_key_id, grammar_version, seed_fingerprint,
                    signature_version, created_at, guard_fingerprint
                )
                SELECT id, creation_receipt_id, association_id, domain,
                       context_scope_hash, rewrite_commitment, {binding_expression},
                       commitment_key_id, grammar_version, seed_fingerprint,
                       signature_version, created_at, guard_fingerprint
                FROM contextual_restricted_rewrite_guard_v18_legacy;
                DROP TABLE contextual_restricted_rewrite_guard_v18_legacy;
                {Database._restricted_rewrite_guard_objects_sql(require_v20_provenance=False)}
                COMMIT;
            """
        else:
            script = f"""
                BEGIN IMMEDIATE;
                {Database._restricted_rewrite_guard_objects_sql(require_v20_provenance=False)}
                COMMIT;
            """
        try:
            connection.executescript(script)
        except sqlite3.DatabaseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        guard_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        if "binding_commitment" not in guard_columns:
            raise RuntimeError(
                "contextual restricted rewrite guard is missing V19 HMAC binding"
            )

    @staticmethod
    def _migrate_v19_to_v20(connection: sqlite3.Connection) -> None:
        """Bind a T16 capability to the two origin cue-vector digests.

        The V19 root-to-seed HMAC prevents a sidecar transplant, but a
        database-only attacker could still replace cue blobs and recompute the
        public manifest hashes.  V20 adds two opaque origin vector digests to
        the HMAC-protected sidecar.  Pre-existing rows are not backfilled or
        re-signed: they deliberately lose the optional rewrite capability.
        """

        if connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND name = 'contextual_restricted_rewrite_guard_v19_legacy'
            """
        ).fetchone() is not None:
            raise RuntimeError(
                "interrupted V20 restricted-rewrite migration detected; "
                "restore the database snapshot before retrying"
            )
        Database._ensure_v19_schema(connection)
        Database._ensure_v20_schema(connection)

    @staticmethod
    def _ensure_v20_schema(connection: sqlite3.Connection) -> None:
        """Require the V20 cue provenance HMAC fields on every new sidecar."""

        interrupted_legacy = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND name = 'contextual_restricted_rewrite_guard_v19_legacy'
            """
        ).fetchone()
        if interrupted_legacy is not None:
            raise RuntimeError(
                "interrupted V20 restricted-rewrite migration detected; "
                "restore the database snapshot before retrying"
            )
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        if "binding_commitment" not in columns:
            raise RuntimeError(
                "V20 restricted-rewrite migration requires V19 HMAC binding"
            )
        needs_rebuild = (
            "context_cue_vector_fingerprint" not in columns
            or "need_cue_vector_fingerprint" not in columns
            or "model_id" not in columns
            or "embedding_space_id" not in columns
            or "dimension" not in columns
            or "dtype" not in columns
            or Database._restricted_rewrite_guard_has_unique_association(connection)
        )
        context_expression = (
            "context_cue_vector_fingerprint"
            if "context_cue_vector_fingerprint" in columns
            else "''"
        )
        need_expression = (
            "need_cue_vector_fingerprint"
            if "need_cue_vector_fingerprint" in columns
            else "''"
        )
        if needs_rebuild:
            script = f"""
                BEGIN IMMEDIATE;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_replace;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_redaction_guard;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_source_guard;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_delete;
                DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_lookup_idx;
                DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_root_lookup_idx;
                ALTER TABLE contextual_restricted_rewrite_guard
                    RENAME TO contextual_restricted_rewrite_guard_v19_legacy;
                CREATE TABLE contextual_restricted_rewrite_guard(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    creation_receipt_id INTEGER NOT NULL UNIQUE,
                    association_id INTEGER NOT NULL,
                    domain TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(domain) = 1),
                    context_scope_hash TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(context_scope_hash) = 1),
                    rewrite_commitment TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(rewrite_commitment) = 1),
                    binding_commitment TEXT NOT NULL DEFAULT '',
                    context_cue_vector_fingerprint TEXT NOT NULL DEFAULT '',
                    need_cue_vector_fingerprint TEXT NOT NULL DEFAULT '',
                    model_id TEXT NOT NULL DEFAULT '',
                    embedding_space_id TEXT NOT NULL DEFAULT '',
                    dimension INTEGER NOT NULL DEFAULT 0,
                    dtype TEXT NOT NULL DEFAULT '',
                    commitment_key_id TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(commitment_key_id) = 1),
                    grammar_version TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(grammar_version) = 1),
                    seed_fingerprint TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(seed_fingerprint) = 1),
                    signature_version TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(signature_version) = 1),
                    created_at TEXT NOT NULL
                        CHECK(contextual_revisit_utc_instant(created_at) = 1),
                    guard_fingerprint TEXT NOT NULL UNIQUE
                        CHECK(contextual_revisit_opaque_digest(guard_fingerprint) = 1),
                    FOREIGN KEY(creation_receipt_id)
                        REFERENCES contextual_creation_receipt(id),
                    FOREIGN KEY(association_id) REFERENCES association(id)
                );
                INSERT INTO contextual_restricted_rewrite_guard(
                    id, creation_receipt_id, association_id, domain,
                    context_scope_hash, rewrite_commitment, binding_commitment,
                    context_cue_vector_fingerprint, need_cue_vector_fingerprint,
                    model_id, embedding_space_id, dimension, dtype,
                    commitment_key_id, grammar_version, seed_fingerprint,
                    signature_version, created_at, guard_fingerprint
                )
                SELECT id, creation_receipt_id, association_id, domain,
                       context_scope_hash, rewrite_commitment, binding_commitment,
                       {context_expression}, {need_expression},
                       {'model_id' if 'model_id' in columns else "''"},
                       {'embedding_space_id' if 'embedding_space_id' in columns else "''"},
                       {'dimension' if 'dimension' in columns else '0'},
                       {'dtype' if 'dtype' in columns else "''"},
                       commitment_key_id, grammar_version, seed_fingerprint,
                       signature_version, created_at, guard_fingerprint
                FROM contextual_restricted_rewrite_guard_v19_legacy;
                DROP TABLE contextual_restricted_rewrite_guard_v19_legacy;
                {Database._restricted_rewrite_guard_objects_sql(require_v20_provenance=True)}
                COMMIT;
            """
        else:
            script = f"""
                BEGIN IMMEDIATE;
                {Database._restricted_rewrite_guard_objects_sql(require_v20_provenance=True)}
                COMMIT;
            """
        try:
            connection.executescript(script)
        except sqlite3.DatabaseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        guard_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        required = {
            "binding_commitment",
            "context_cue_vector_fingerprint",
            "need_cue_vector_fingerprint",
            "model_id",
            "embedding_space_id",
            "dimension",
            "dtype",
        }
        if missing := required - guard_columns:
            raise RuntimeError(
                "contextual restricted rewrite guard is missing V20 provenance fields: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _migrate_v20_to_v21(connection: sqlite3.Connection) -> None:
        """Bind each T16 sidecar to its canonical manifest lifecycle authority.

        V20 authenticated the semantic root and cue provenance, but its first
        HMAC did not cover the repository-built runtime-manifest binding.  A
        V21 guard therefore receives a second, process-keyed commitment over
        that binding.  Existing V20 rows are retained for audit with an empty
        field and intentionally become ordinary rewrite misses; no historical
        sidecar is ever backfilled or re-signed.
        """

        if connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND name = 'contextual_restricted_rewrite_guard_v20_legacy'
            """
        ).fetchone() is not None:
            raise RuntimeError(
                "interrupted V21 restricted-rewrite migration detected; "
                "restore the database snapshot before retrying"
            )
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        required_v20 = {
            "binding_commitment",
            "context_cue_vector_fingerprint",
            "need_cue_vector_fingerprint",
            "model_id",
            "embedding_space_id",
            "dimension",
            "dtype",
        }
        if missing := required_v20 - columns:
            raise RuntimeError(
                "V21 restricted-rewrite migration requires V20 provenance fields: "
                + ", ".join(sorted(missing))
            )
        Database._ensure_v21_schema(connection)

    @staticmethod
    def _ensure_v21_schema(connection: sqlite3.Connection) -> None:
        """Require a second HMAC over the canonical runtime-manifest binding."""

        interrupted_legacy = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table'
              AND name = 'contextual_restricted_rewrite_guard_v20_legacy'
            """
        ).fetchone()
        if interrupted_legacy is not None:
            raise RuntimeError(
                "interrupted V21 restricted-rewrite migration detected; "
                "restore the database snapshot before retrying"
            )
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        required_v20 = {
            "binding_commitment",
            "context_cue_vector_fingerprint",
            "need_cue_vector_fingerprint",
            "model_id",
            "embedding_space_id",
            "dimension",
            "dtype",
        }
        if missing := required_v20 - columns:
            raise RuntimeError(
                "V21 restricted-rewrite migration requires V20 provenance fields: "
                + ", ".join(sorted(missing))
            )
        needs_rebuild = (
            "manifest_binding_commitment" not in columns
            or Database._restricted_rewrite_guard_has_unique_association(connection)
        )
        manifest_binding_expression = (
            "manifest_binding_commitment"
            if "manifest_binding_commitment" in columns
            else "''"
        )
        if needs_rebuild:
            script = f"""
                BEGIN IMMEDIATE;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_replace;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_redaction_guard;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_source_guard;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_update;
                DROP TRIGGER IF EXISTS contextual_restricted_rewrite_guard_no_delete;
                DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_lookup_idx;
                DROP INDEX IF EXISTS contextual_restricted_rewrite_guard_root_lookup_idx;
                ALTER TABLE contextual_restricted_rewrite_guard
                    RENAME TO contextual_restricted_rewrite_guard_v20_legacy;
                CREATE TABLE contextual_restricted_rewrite_guard(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    creation_receipt_id INTEGER NOT NULL UNIQUE,
                    association_id INTEGER NOT NULL,
                    domain TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(domain) = 1),
                    context_scope_hash TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(context_scope_hash) = 1),
                    rewrite_commitment TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(rewrite_commitment) = 1),
                    binding_commitment TEXT NOT NULL DEFAULT '',
                    manifest_binding_commitment TEXT NOT NULL DEFAULT '',
                    context_cue_vector_fingerprint TEXT NOT NULL DEFAULT '',
                    need_cue_vector_fingerprint TEXT NOT NULL DEFAULT '',
                    model_id TEXT NOT NULL DEFAULT '',
                    embedding_space_id TEXT NOT NULL DEFAULT '',
                    dimension INTEGER NOT NULL DEFAULT 0,
                    dtype TEXT NOT NULL DEFAULT '',
                    commitment_key_id TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(commitment_key_id) = 1),
                    grammar_version TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(grammar_version) = 1),
                    seed_fingerprint TEXT NOT NULL
                        CHECK(contextual_revisit_opaque_digest(seed_fingerprint) = 1),
                    signature_version TEXT NOT NULL
                        CHECK(contextual_revisit_safe_identifier(signature_version) = 1),
                    created_at TEXT NOT NULL
                        CHECK(contextual_revisit_utc_instant(created_at) = 1),
                    guard_fingerprint TEXT NOT NULL UNIQUE
                        CHECK(contextual_revisit_opaque_digest(guard_fingerprint) = 1),
                    FOREIGN KEY(creation_receipt_id)
                        REFERENCES contextual_creation_receipt(id),
                    FOREIGN KEY(association_id) REFERENCES association(id)
                );
                INSERT INTO contextual_restricted_rewrite_guard(
                    id, creation_receipt_id, association_id, domain,
                    context_scope_hash, rewrite_commitment, binding_commitment,
                    manifest_binding_commitment,
                    context_cue_vector_fingerprint, need_cue_vector_fingerprint,
                    model_id, embedding_space_id, dimension, dtype,
                    commitment_key_id, grammar_version, seed_fingerprint,
                    signature_version, created_at, guard_fingerprint
                )
                SELECT id, creation_receipt_id, association_id, domain,
                       context_scope_hash, rewrite_commitment, binding_commitment,
                       {manifest_binding_expression},
                       context_cue_vector_fingerprint, need_cue_vector_fingerprint,
                       model_id, embedding_space_id, dimension, dtype,
                       commitment_key_id, grammar_version, seed_fingerprint,
                       signature_version, created_at, guard_fingerprint
                FROM contextual_restricted_rewrite_guard_v20_legacy;
                DROP TABLE contextual_restricted_rewrite_guard_v20_legacy;
                {Database._restricted_rewrite_guard_objects_sql(
                    require_v20_provenance=True,
                    require_v21_manifest_binding=True,
                )}
                COMMIT;
            """
        else:
            script = f"""
                BEGIN IMMEDIATE;
                {Database._restricted_rewrite_guard_objects_sql(
                    require_v20_provenance=True,
                    require_v21_manifest_binding=True,
                )}
                COMMIT;
            """
        try:
            connection.executescript(script)
        except sqlite3.DatabaseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        guard_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        if "manifest_binding_commitment" not in guard_columns:
            raise RuntimeError(
                "contextual restricted rewrite guard is missing V21 manifest HMAC"
            )

    @staticmethod
    def _migrate_v21_to_v22(connection: sqlite3.Connection) -> None:
        """Require a third HMAC for the ready manifest/contract publication.

        V21's second HMAC protects the pending lifecycle binding, but does not
        authenticate the later ready manifest fingerprint.  V22 retains every
        old sidecar with an empty field for audit and intentionally never
        backfills/re-signs it: only a fresh trusted pending-to-ready
        transaction may create the third commitment.
        """

        Database._ensure_v22_schema(connection)

    @staticmethod
    def _ensure_v22_schema(connection: sqlite3.Connection) -> None:
        """Install the one-shot ready-manifest HMAC guard atomically."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        required_v21 = {
            "binding_commitment",
            "manifest_binding_commitment",
            "context_cue_vector_fingerprint",
            "need_cue_vector_fingerprint",
            "model_id",
            "embedding_space_id",
            "dimension",
            "dtype",
        }
        if missing := required_v21 - columns:
            raise RuntimeError(
                "V22 restricted-rewrite migration requires V21 fields: "
                + ", ".join(sorted(missing))
            )
        if Database._restricted_rewrite_guard_has_unique_association(connection):
            raise RuntimeError(
                "V22 restricted-rewrite guard has an unsupported unique association index"
            )
        if "ready_manifest_commitment" not in columns:
            script = f"""
                BEGIN IMMEDIATE;
                ALTER TABLE contextual_restricted_rewrite_guard
                    ADD COLUMN ready_manifest_commitment TEXT NOT NULL DEFAULT '';
                {Database._restricted_rewrite_guard_objects_sql(
                    require_v20_provenance=True,
                    require_v21_manifest_binding=True,
                    require_v22_ready_manifest_binding=True,
                )}
                COMMIT;
            """
        else:
            script = f"""
                BEGIN IMMEDIATE;
                {Database._restricted_rewrite_guard_objects_sql(
                    require_v20_provenance=True,
                    require_v21_manifest_binding=True,
                    require_v22_ready_manifest_binding=True,
                )}
                COMMIT;
            """
        try:
            connection.executescript(script)
        except sqlite3.DatabaseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        guard_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(contextual_restricted_rewrite_guard)"
            )
        }
        if "ready_manifest_commitment" not in guard_columns:
            raise RuntimeError(
                "contextual restricted rewrite guard is missing V22 ready HMAC"
            )

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
