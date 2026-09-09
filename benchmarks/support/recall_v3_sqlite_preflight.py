from __future__ import annotations

"""Read-only SQLite evidence closure preflight for a v3 recall run.

This is deliberately a *second* preflight layer.  :mod:`benchmarks.run_recall_v3`
only reads JSON and deliberately does not import ``sqlite3``; this module is
the narrow place where a frozen run specification may be checked against the
knowledge-base snapshot it names.

It never imports ``MemoryApplication`` or ``Database``, never runs a schema
migration, and opens the supplied database with SQLite's ``mode=ro`` URI.  The
only filesystem writes it can perform are none: the command-line entry point
prints a report to stdout rather than creating an output artifact.

The additional, explicit ``sqlite_preflight`` portion of a run spec is:

.. code-block:: json

  {
    "sqlite_preflight": {
      "database_sha256": "sha256:<main-db-bytes>",
      "source_manifest_sha256": "sha256:<external-manifest-bytes>",
      "scope_sha256": "sha256:<external-scope-bytes>",
      "schema_version": 22,
      "evaluation_as_of": "2026-09-06T00:00:00+00:00",
      "source_keys": ["main/example.json"],
      "embedding_space": {
        "id": "bge-m3/normalized-v1", "model_id": "bge-m3",
        "preprocess_version": "normalized-v1", "dimension": 1024,
        "normalized": true, "dtype": "float32"
      },
      "expected_counts": {
        "source": 1, "paragraph": 1, "episode": 1, "concept": 0,
        "association": 1, "cue_prototype": 2, "contextual_edge": 1
      }
    }
  }

``--source-manifest`` and ``--scope-manifest`` are required so their declared
hashes are attested from real local files, rather than merely copied between
JSON documents.  Their application-specific payloads are intentionally not
parsed here: this layer only proves byte identity, while the run specification
provides the explicit source-key scope.  A contextual arm additionally binds each declared cue and
edge to real database IDs using ``database_cue_id`` / ``database_association_id``.
An edge must list both cue names in ``cue_ids``.  These extra fields are
ignored by the pure JSON preflight and therefore do not weaken it.

No source text, episode text, prompts, credentials, or vectors are emitted in
the report.  Source keys and numeric database identifiers are deliberately
limited provenance information needed to explain a closure failure.
"""

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Any


PREFLIGHT_SCHEMA = "aevnema.recall-v3.sqlite-preflight.v1"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_VECTOR_TABLES = ("episode", "paragraph", "concept")
_COUNT_TABLES = {
    "source": "source",
    "paragraph": "paragraph",
    "episode": "episode",
    "concept": "concept",
    "association": "association",
    "cue_prototype": "association_cue_prototype",
}
# These are the only frozen SQLite layouts this reader understands.  Version
# 13 is the historical baseline used by the original V3 artifacts; version
# 22 is the application's current layout.  The V14--V21 migrations are not
# accepted by range: although their retrieval columns may overlap, a frozen
# replay must not assume an interrupted or otherwise unreviewed intermediate
# layout has the same read contract.  Keep this list static rather than
# importing ``memory_demo.database.SCHEMA_VERSION`` so this module remains a
# read-only boundary with no application initialisation or migration surface.
_REQUIRED_COLUMNS_V13: dict[str, frozenset[str]] = {
    "schema_meta": frozenset({"schema_version", "created_at"}),
    "source": frozenset({"id", "raw_text"}),
    "paragraph": frozenset({"id", "source_id", "source_key", "embedding"}),
    "episode": frozenset(
        {"id", "source_id", "source_key", "embedding", "generation"}
    ),
    "concept": frozenset({"id", "embedding"}),
    "association": frozenset(
        {
            "id",
            "from_type",
            "from_id",
            "to_type",
            "to_id",
            "relation_type",
            "relation_key",
            "polarity",
            "association_mode",
            "context_cue_id",
            "need_cue_id",
            "lifecycle_state",
            "expires_at",
            "cue_embedding",
        }
    ),
    "association_cue_prototype": frozenset(
        {"id", "domain", "cue_kind", "model_id", "dimension", "dtype", "vector_blob"}
    ),
    "episode_fts": frozenset(),
    "source_fts": frozenset(),
}

# V14--V22 add contextual runtime/receipt state, but the frozen replay reader
# intentionally reads only the v13 retrieval closure above.  Listing V22
# separately is deliberate: a future schema bump must be explicitly audited
# before a snapshot can be accepted, even when its columns happen to be a
# superset of this reader's needs.
_READ_SCHEMA_CONTRACTS: dict[int, dict[str, frozenset[str]]] = {
    13: _REQUIRED_COLUMNS_V13,
    22: _REQUIRED_COLUMNS_V13,
}
_SUPPORTED_SCHEMA_VERSIONS = frozenset(_READ_SCHEMA_CONTRACTS)
_SCHEMA_META_COLUMNS = {"schema_meta": _REQUIRED_COLUMNS_V13["schema_meta"]}
_SIDECAR_FLAGS = ("wal_present", "shm_present", "journal_present")


def _issue(code: str, message: str, *, path: str) -> dict[str, str]:
    return {"scope": "sqlite", "code": code, "path": path, "message": message}


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_utc_timestamp(value: object) -> datetime | None:
    """Parse a strict, timezone-aware ISO-8601 UTC timestamp."""
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed.astimezone(timezone.utc)


def _sha256_file(path: Path) -> tuple[str, bool]:
    """Hash a local file and say whether it was stable while read.

    The stability bit closes the most common race without mutating the source.
    It is intentionally based on size/mtime as a cheap guard around the byte
    hash; a caller still receives the exact resulting digest.
    """
    before = path.stat()
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    after = path.stat()
    stable = (
        before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ino == after.st_ino
    )
    return "sha256:" + digest.hexdigest(), stable


def _file_digest_report(path: Path, *, label: str, issues: list[dict[str, str]]) -> str | None:
    try:
        if not path.is_file():
            issues.append(
                _issue(
                    "snapshot_file_missing",
                    f"{label} must name an existing regular file",
                    path=label,
                )
            )
            return None
        digest, stable = _sha256_file(path)
    except OSError as exc:
        issues.append(
            _issue("snapshot_file_unreadable", f"cannot hash {label}: {exc}", path=label)
        )
        return None
    if not stable:
        issues.append(
            _issue(
                "snapshot_file_changed_during_hash",
                f"{label} changed while its digest was calculated",
                path=label,
            )
        )
    return digest


def _normalise_source_keys(
    value: object, issues: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Accept compact strings or explicit source-key requirement objects."""
    if not isinstance(value, list) or not value:
        issues.append(
            _issue(
                "source_key_scope_missing",
                "sqlite_preflight.source_keys must be a non-empty list",
                path="spec.sqlite_preflight.source_keys",
            )
        )
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        path = f"spec.sqlite_preflight.source_keys[{index}]"
        if isinstance(raw, str):
            source_key = raw.strip()
            require_episode = False
        elif _is_mapping(raw):
            source_key = str(raw.get("source_key", "")).strip()
            require_episode = raw.get("require_episode", False)
            if not isinstance(require_episode, bool):
                issues.append(
                    _issue(
                        "source_key_requirement_invalid",
                        "require_episode must be a boolean",
                        path=f"{path}.require_episode",
                    )
                )
                require_episode = False
        else:
            source_key = ""
            require_episode = False
        if not source_key:
            issues.append(
                _issue(
                    "source_key_invalid",
                    "each source-key requirement needs a non-empty source_key",
                    path=path,
                )
            )
            continue
        if source_key in seen:
            issues.append(
                _issue(
                    "source_key_duplicate",
                    "source keys must be unique in a frozen scope",
                    path=path,
                )
            )
            continue
        seen.add(source_key)
        result.append({"source_key": source_key, "require_episode": require_episode})
    return result


def _read_config(run_spec: object, issues: list[dict[str, str]]) -> dict[str, Any]:
    if not _is_mapping(run_spec):
        issues.append(_issue("run_spec_not_object", "run spec must be an object", path="spec"))
        return {}
    raw = run_spec.get("sqlite_preflight")
    if not _is_mapping(raw):
        issues.append(
            _issue(
                "sqlite_preflight_missing",
                "v3 SQLite closure needs an explicit sqlite_preflight object",
                path="spec.sqlite_preflight",
            )
        )
        return {}
    config = dict(raw)
    for name in ("database_sha256", "source_manifest_sha256", "scope_sha256"):
        if not _SHA256_RE.fullmatch(str(config.get(name, ""))):
            issues.append(
                _issue(
                    "snapshot_digest_invalid",
                    f"{name} must be a lowercase sha256: digest",
                    path=f"spec.sqlite_preflight.{name}",
                )
            )
    schema_version = config.get("schema_version")
    if not _is_positive_int(schema_version):
        issues.append(
            _issue(
                "schema_version_invalid",
                "schema_version must be a positive integer",
                path="spec.sqlite_preflight.schema_version",
            )
        )
    elif int(schema_version) not in _SUPPORTED_SCHEMA_VERSIONS:
        issues.append(
            _issue(
                "unsupported_schema_version",
                "this preflight supports frozen read schemas "
                + ", ".join(str(version) for version in sorted(_SUPPORTED_SCHEMA_VERSIONS)),
                path="spec.sqlite_preflight.schema_version",
            )
        )
    evaluation_as_of = _parse_utc_timestamp(config.get("evaluation_as_of"))
    if evaluation_as_of is None:
        issues.append(
            _issue(
                "evaluation_as_of_invalid",
                "evaluation_as_of must be an ISO-8601 UTC timestamp",
                path="spec.sqlite_preflight.evaluation_as_of",
            )
        )
    else:
        # Keep a parsed, deterministic instant internally.  The original text
        # is preserved in the spec and surfaced in the final report below.
        config["_evaluation_as_of"] = evaluation_as_of

    embedding = config.get("embedding_space")
    if not _is_mapping(embedding):
        issues.append(
            _issue(
                "embedding_space_missing",
                "sqlite_preflight needs embedding_space provenance",
                path="spec.sqlite_preflight.embedding_space",
            )
        )
    else:
        dimension = embedding.get("dimension")
        if not _is_positive_int(dimension):
            issues.append(
                _issue(
                    "embedding_dimension_invalid",
                    "embedding_space.dimension must be a positive integer",
                    path="spec.sqlite_preflight.embedding_space.dimension",
                )
            )
        for name in ("id", "model_id", "preprocess_version"):
            if not _is_nonempty_string(embedding.get(name)):
                issues.append(
                    _issue(
                        "embedding_provenance_missing",
                        f"embedding_space.{name} must be non-empty",
                        path=f"spec.sqlite_preflight.embedding_space.{name}",
                    )
                )
        if embedding.get("normalized") is not True:
            issues.append(
                _issue(
                    "embedding_normalization_missing",
                    "embedding_space.normalized must explicitly be true",
                    path="spec.sqlite_preflight.embedding_space.normalized",
                )
            )
        if embedding.get("dtype") != "float32":
            issues.append(
                _issue(
                    "embedding_dtype_invalid",
                    "only float32 SQLite embedding blobs are supported",
                    path="spec.sqlite_preflight.embedding_space.dtype",
                )
            )

    config["_source_key_requirements"] = _normalise_source_keys(
        config.get("source_keys"), issues
    )
    expected_counts = config.get("expected_counts")
    expected_names = {*_COUNT_TABLES, "contextual_edge"}
    if not _is_mapping(expected_counts):
        issues.append(
            _issue(
                "expected_counts_missing",
                "sqlite_preflight must pin exact relevant table counts",
                path="spec.sqlite_preflight.expected_counts",
            )
        )
    else:
        names = {str(name) for name in expected_counts}
        missing = sorted(expected_names - names)
        unexpected = sorted(names - expected_names)
        if missing:
            issues.append(
                _issue(
                    "expected_counts_incomplete",
                    "expected_counts must include every relevant table count",
                    path="spec.sqlite_preflight.expected_counts",
                )
            )
        if unexpected:
            issues.append(
                _issue(
                    "expected_counts_unknown_name",
                    "expected_counts contains an unsupported count name",
                    path="spec.sqlite_preflight.expected_counts",
                )
            )
        for name, value in expected_counts.items():
            if str(name) in expected_names and not _is_nonnegative_int(value):
                issues.append(
                    _issue(
                        "expected_count_invalid",
                        "expected counts must be non-negative integers",
                        path=f"spec.sqlite_preflight.expected_counts.{name}",
                    )
                )
    return config


def _sqlite_uri(path: Path) -> str:
    """Build an explicit read-only SQLite URI without using a write-capable API."""
    return path.resolve(strict=True).as_uri() + "?mode=ro"


def _connect_readonly(path: Path) -> sqlite3.Connection:
    # ``mode=ro`` ensures no database is created.  query_only is a second,
    # connection-local defence in case code below is accidentally changed.
    connection = sqlite3.connect(_sqlite_uri(path), uri=True, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA trusted_schema = OFF")
    return connection


def _table_columns(connection: sqlite3.Connection, name: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({name})")
    }


def _schema_checks(
    connection: sqlite3.Connection,
    required_by_table: Mapping[str, frozenset[str]],
    issues: list[dict[str, str]],
) -> bool:
    tables = {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    complete = True
    for table, required_columns in required_by_table.items():
        if table not in tables:
            complete = False
            issues.append(
                _issue(
                    "required_table_missing",
                    f"database does not contain required table {table}",
                    path=f"database.schema.{table}",
                )
            )
            continue
        missing = sorted(required_columns - _table_columns(connection, table))
        if missing:
            complete = False
            issues.append(
                _issue(
                    "required_column_missing",
                    f"required columns are missing from {table}",
                    path=f"database.schema.{table}",
                )
            )
    return complete


def _pragma_checks(
    connection: sqlite3.Connection, issues: list[dict[str, str]]
) -> dict[str, Any]:
    # Do not include arbitrary pragma payloads in the report.  They may carry
    # page-level details unrelated to a benchmark decision.
    integrity = [str(row[0]) for row in connection.execute("PRAGMA integrity_check")]
    if integrity != ["ok"]:
        issues.append(
            _issue(
                "sqlite_integrity_check_failed",
                "PRAGMA integrity_check did not return exactly ok",
                path="database.integrity_check",
            )
        )
    foreign_key_rows = list(connection.execute("PRAGMA foreign_key_check"))
    if foreign_key_rows:
        issues.append(
            _issue(
                "sqlite_foreign_key_check_failed",
                "PRAGMA foreign_key_check reported one or more violations",
                path="database.foreign_key_check",
            )
        )
    return {
        "integrity_check": "ok" if integrity == ["ok"] else "failed",
        "foreign_key_violation_count": len(foreign_key_rows),
    }


def _read_schema_version(
    connection: sqlite3.Connection, issues: list[dict[str, str]]
) -> int | None:
    rows = list(connection.execute("SELECT schema_version FROM schema_meta"))
    if len(rows) != 1:
        issues.append(
            _issue(
                "schema_meta_invalid",
                "schema_meta must contain exactly one schema version row",
                path="database.schema_meta",
            )
        )
        return None
    value = rows[0]["schema_version"]
    if not _is_positive_int(value):
        issues.append(
            _issue(
                "schema_meta_invalid",
                "schema_meta.schema_version must be a positive integer",
                path="database.schema_meta.schema_version",
            )
        )
        return None
    return int(value)


def _count(
    connection: sqlite3.Connection, table: str, *, predicate: str = "", params: Sequence[Any] = ()
) -> int:
    query = f"SELECT COUNT(*) AS count FROM {table}"
    if predicate:
        query += " WHERE " + predicate
    return int(connection.execute(query, tuple(params)).fetchone()["count"])


def _snapshot_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts = {name: _count(connection, table) for name, table in _COUNT_TABLES.items()}
    counts["contextual_edge"] = _count(
        connection, "association", predicate="association_mode = 'contextual_recall'"
    )
    # FTS counts are independent integrity signals.  They are reported but not
    # declared in expected_counts because they must equal their authoritative
    # source tables for a schema-v13 snapshot.
    counts["episode_fts"] = _count(connection, "episode_fts")
    counts["source_fts"] = _count(connection, "source_fts")
    return counts


def _vector_checks(
    connection: sqlite3.Connection,
    *,
    dimension: int,
    model_id: str,
    evaluation_as_of: datetime | None,
    issues: list[dict[str, str]],
) -> dict[str, int]:
    expected_bytes = dimension * 4
    invalid: dict[str, int] = {}
    for table in _VECTOR_TABLES:
        invalid[table] = _count(
            connection,
            table,
            predicate="typeof(embedding) != 'blob' OR length(embedding) != ?",
            params=(expected_bytes,),
        )
        if invalid[table]:
            issues.append(
                _issue(
                    "embedding_blob_shape_mismatch",
                    f"{table} has an embedding blob with the wrong float32 shape",
                    path=f"database.{table}.embedding",
                )
            )
    invalid["association_cue_embedding"] = _count(
        connection,
        "association",
        predicate=(
            "cue_embedding IS NOT NULL AND "
            "(typeof(cue_embedding) != 'blob' OR length(cue_embedding) != ?)"
        ),
        params=(expected_bytes,),
    )
    if invalid["association_cue_embedding"]:
        issues.append(
            _issue(
                "association_cue_blob_shape_mismatch",
                "association cue embeddings do not match the pinned float32 shape",
                path="database.association.cue_embedding",
            )
        )
    invalid["cue_prototype"] = _count(
        connection,
        "association_cue_prototype",
        predicate=(
            "dimension != ? OR dtype != 'float32' OR typeof(vector_blob) != 'blob' "
            "OR length(vector_blob) != ?"
        ),
        params=(dimension, expected_bytes),
    )
    if invalid["cue_prototype"]:
        issues.append(
            _issue(
                "cue_prototype_shape_mismatch",
                "cue prototypes do not match the pinned model dimension/dtype",
                path="database.association_cue_prototype",
            )
        )
    invalid["cue_prototype_model"] = _count(
        connection,
        "association_cue_prototype",
        predicate="model_id != ?",
        params=(model_id,),
    )
    if invalid["cue_prototype_model"]:
        issues.append(
            _issue(
                "cue_prototype_model_mismatch",
                "cue prototype model_id does not match the pinned embedding model",
                path="database.association_cue_prototype.model_id",
            )
        )
    invalid["active_contextual_edge_structure"] = _count(
        connection,
        "association AS a",
        predicate=(
            "a.association_mode = 'contextual_recall' AND "
            "a.lifecycle_state IN ('probation', 'active') AND "
            "("
            "a.to_type != 'episode' OR a.relation_type != 'retrieval' OR "
            "a.relation_key != 'contextual_recall' OR a.polarity != 1 OR "
            "NOT EXISTS (SELECT 1 FROM episode e WHERE e.id = a.to_id AND e.generation = 0) OR "
            "(a.from_type = 'episode' AND NOT EXISTS "
            "(SELECT 1 FROM episode ae WHERE ae.id = a.from_id AND ae.generation = 0)) OR "
            "(a.from_type = 'concept' AND NOT EXISTS "
            "(SELECT 1 FROM concept ac WHERE ac.id = a.from_id)) OR "
            "a.from_type NOT IN ('episode', 'concept') OR "
            "NOT EXISTS (SELECT 1 FROM association_cue_prototype cp "
            "WHERE cp.id = a.context_cue_id AND cp.cue_kind = 'context') OR "
            "NOT EXISTS (SELECT 1 FROM association_cue_prototype np "
            "WHERE np.id = a.need_cue_id AND np.cue_kind = 'need') OR "
            "(SELECT cp.domain FROM association_cue_prototype cp "
            "WHERE cp.id = a.context_cue_id) != "
            "(SELECT np.domain FROM association_cue_prototype np "
            "WHERE np.id = a.need_cue_id)"
            ")"
        ),
    )
    if invalid["active_contextual_edge_structure"]:
        issues.append(
            _issue(
                "contextual_edge_database_closure_broken",
                "an active contextual association has missing or incompatible endpoints/cues",
                path="database.association.contextual_recall",
            )
        )
    expired_count = 0
    invalid_expiry_count = 0
    if evaluation_as_of is not None:
        for row in connection.execute(
            """
            SELECT expires_at FROM association
            WHERE association_mode = 'contextual_recall'
              AND lifecycle_state IN ('probation', 'active')
              AND expires_at IS NOT NULL
            """
        ):
            expires_at = _parse_utc_timestamp(row["expires_at"])
            if expires_at is None:
                invalid_expiry_count += 1
            elif expires_at <= evaluation_as_of:
                expired_count += 1
    invalid["active_contextual_edge_expired_as_of"] = expired_count
    invalid["active_contextual_edge_invalid_expiry"] = invalid_expiry_count
    if invalid_expiry_count:
        issues.append(
            _issue(
                "contextual_edge_expiry_invalid",
                "an active contextual edge has a non-UTC or malformed expiry timestamp",
                path="database.association.contextual_recall.expires_at",
            )
        )
    return invalid


def _source_key_checks(
    connection: sqlite3.Connection,
    requirements: Sequence[Mapping[str, Any]],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Check the declared source scope without reading source or episode text."""
    summary: list[dict[str, Any]] = []
    for requirement in requirements:
        source_key = str(requirement["source_key"])
        episode_count = _count(
            connection, "episode", predicate="source_key = ?", params=(source_key,)
        )
        paragraph_count = _count(
            connection, "paragraph", predicate="source_key = ?", params=(source_key,)
        )
        present = episode_count > 0 or paragraph_count > 0
        if not present:
            issues.append(
                _issue(
                    "declared_source_key_missing",
                    "a source key declared by the run spec has no source-derived record",
                    path="spec.sqlite_preflight.source_keys",
                )
            )
        if requirement["require_episode"] and episode_count == 0:
            issues.append(
                _issue(
                    "declared_source_key_episode_missing",
                    "this source key is required to have at least one Episode",
                    path="spec.sqlite_preflight.source_keys",
                )
            )
        summary.append(
            {
                "source_key": source_key,
                "episode_count": episode_count,
                "paragraph_count": paragraph_count,
                "present": present,
                "require_episode": bool(requirement["require_episode"]),
            }
        )
    return summary


def _parse_target_episode(target_ref: object) -> int | None:
    if not isinstance(target_ref, str):
        return None
    matched = re.fullmatch(r"episode:([1-9][0-9]*)", target_ref)
    return int(matched.group(1)) if matched else None


def _arm_bindings(
    run_spec: Mapping[str, Any],
    config: Mapping[str, Any],
    connection: sqlite3.Connection,
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Bind each arm's declared contextual graph to the real SQLite graph."""
    raw_arms = run_spec.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        issues.append(_issue("arms_missing", "run spec needs at least one arm", path="spec.arms"))
        return []
    configured_keys = {
        str(item["source_key"]) for item in config.get("_source_key_requirements", [])
    }
    embedding = config.get("embedding_space") if _is_mapping(config.get("embedding_space")) else {}
    expected_model = str(embedding.get("model_id", ""))
    expected_dimension = embedding.get("dimension")
    evaluation_as_of = config.get("_evaluation_as_of")
    if not isinstance(evaluation_as_of, datetime):
        evaluation_as_of = None
    arm_summaries: list[dict[str, Any]] = []
    seen_arm_ids: set[str] = set()
    for index, raw_arm in enumerate(raw_arms):
        path = f"spec.arms[{index}]"
        arm_issues_before = len(issues)
        if not _is_mapping(raw_arm):
            issues.append(_issue("arm_not_object", "arm must be an object", path=path))
            continue
        arm = dict(raw_arm)
        arm_id = str(arm.get("arm_id", "")).strip()
        if not arm_id:
            issues.append(_issue("arm_id_missing", "arm_id must be non-empty", path=f"{path}.arm_id"))
            arm_id = f"arm-{index}"
        if arm_id in seen_arm_ids:
            issues.append(_issue("arm_id_duplicate", "arm_id values must be unique", path=f"{path}.arm_id"))
        seen_arm_ids.add(arm_id)

        snapshot = arm.get("snapshot")
        if not _is_mapping(snapshot):
            issues.append(
                _issue(
                    "arm_snapshot_missing",
                    "every database-backed arm needs a snapshot provenance object",
                    path=f"{path}.snapshot",
                )
            )
        else:
            for key in ("database_sha256", "source_manifest_sha256", "scope_sha256"):
                if snapshot.get(key) != config.get(key):
                    issues.append(
                        _issue(
                            "arm_snapshot_provenance_mismatch",
                            f"arm snapshot {key} must match sqlite_preflight",
                            path=f"{path}.snapshot.{key}",
                        )
                    )
            arm_space = snapshot.get("embedding_space")
            if (
                not _is_mapping(arm_space)
                or arm_space.get("id") != embedding.get("id")
                or arm_space.get("dimension") != expected_dimension
                or arm_space.get("model_id") != expected_model
                or arm_space.get("preprocess_version") != embedding.get("preprocess_version")
                or arm_space.get("normalized") is not True
            ):
                issues.append(
                    _issue(
                        "arm_embedding_space_mismatch",
                        "arm embedding space must match the pinned SQLite snapshot",
                        path=f"{path}.snapshot.embedding_space",
                    )
                )

        requires_contextual = arm.get("requires_contextual") is True
        candidate_state = None
        if _is_mapping(arm.get("candidate_inventory")):
            candidate_state = arm["candidate_inventory"].get("state")
        if not requires_contextual:
            arm_summaries.append(
                {
                    "arm_id": arm_id,
                    "requires_contextual": False,
                    "candidate_state": candidate_state,
                    "status": "ready" if len(issues) == arm_issues_before else "invalid_setup",
                    "execution_state": "valid_no_hit" if candidate_state == "known_empty" else "ready_to_execute",
                }
            )
            continue

        closure = arm.get("contextual")
        if not _is_mapping(closure):
            issues.append(
                _issue(
                    "contextual_closure_missing",
                    "contextual arm must declare its graph closure",
                    path=f"{path}.contextual",
                )
            )
            arm_summaries.append(
                {
                    "arm_id": arm_id,
                    "requires_contextual": True,
                    "candidate_state": candidate_state,
                    "status": "invalid_setup",
                    "execution_state": "not_executable",
                }
            )
            continue
        closure = dict(closure)
        raw_cues = closure.get("cues")
        cue_ids: dict[str, int] = {}
        if not isinstance(raw_cues, list) or not raw_cues:
            issues.append(
                _issue(
                    "database_cue_binding_missing",
                    "contextual arm needs non-empty, database-bound cues",
                    path=f"{path}.contextual.cues",
                )
            )
        else:
            for cue_index, raw_cue in enumerate(raw_cues):
                cue_path = f"{path}.contextual.cues[{cue_index}]"
                if not _is_mapping(raw_cue):
                    issues.append(_issue("cue_not_object", "cue must be an object", path=cue_path))
                    continue
                cue_name = str(raw_cue.get("cue_id", "")).strip()
                database_cue_id = raw_cue.get("database_cue_id")
                expected_kind = raw_cue.get("cue_kind")
                if not cue_name or not _is_positive_int(database_cue_id):
                    issues.append(
                        _issue(
                            "database_cue_binding_missing",
                            "cue needs cue_id and positive database_cue_id",
                            path=cue_path,
                        )
                    )
                    continue
                if expected_kind not in {"context", "need"}:
                    issues.append(
                        _issue(
                            "database_cue_kind_missing",
                            "cue must explicitly be context or need",
                            path=f"{cue_path}.cue_kind",
                        )
                    )
                if cue_name in cue_ids or int(database_cue_id) in cue_ids.values():
                    issues.append(
                        _issue(
                            "database_cue_binding_duplicate",
                            "cue names and database cue IDs must be unique",
                            path=cue_path,
                        )
                    )
                    continue
                cue_ids[cue_name] = int(database_cue_id)
                row = connection.execute(
                    """
                    SELECT id, cue_kind, model_id, dimension, dtype, length(vector_blob) AS vector_bytes
                    FROM association_cue_prototype WHERE id = ?
                    """,
                    (int(database_cue_id),),
                ).fetchone()
                expected_bytes = int(expected_dimension) * 4 if _is_positive_int(expected_dimension) else -1
                if row is None:
                    issues.append(
                        _issue(
                            "database_cue_not_found",
                            "declared database cue ID is absent from this snapshot",
                            path=f"{cue_path}.database_cue_id",
                        )
                    )
                elif (
                    row["cue_kind"] != expected_kind
                    or row["model_id"] != expected_model
                    or row["dimension"] != expected_dimension
                    or row["dtype"] != "float32"
                    or row["vector_bytes"] != expected_bytes
                ):
                    issues.append(
                        _issue(
                            "database_cue_provenance_mismatch",
                            "database cue does not match its declared kind/model/shape",
                            path=f"{cue_path}.database_cue_id",
                        )
                    )

        raw_edges = closure.get("edges")
        if not isinstance(raw_edges, list) or not raw_edges:
            issues.append(
                _issue(
                    "database_contextual_edge_binding_missing",
                    "contextual arm needs non-empty, database-bound edges",
                    path=f"{path}.contextual.edges",
                )
            )
        else:
            declared_edge_ids: set[int] = set()
            for edge_index, raw_edge in enumerate(raw_edges):
                edge_path = f"{path}.contextual.edges[{edge_index}]"
                if not _is_mapping(raw_edge):
                    issues.append(_issue("edge_not_object", "edge must be an object", path=edge_path))
                    continue
                association_id = raw_edge.get("database_association_id")
                target_episode_id = _parse_target_episode(raw_edge.get("target_ref"))
                raw_cue_names = raw_edge.get("cue_ids")
                if (
                    not _is_positive_int(association_id)
                    or target_episode_id is None
                    or not isinstance(raw_cue_names, list)
                    or len(raw_cue_names) != 2
                    or not all(isinstance(name, str) for name in raw_cue_names)
                ):
                    issues.append(
                        _issue(
                            "database_contextual_edge_binding_missing",
                            "edge needs database_association_id, episode target_ref, and two cue_ids",
                            path=edge_path,
                        )
                    )
                    continue
                association_id = int(association_id)
                if association_id in declared_edge_ids:
                    issues.append(
                        _issue(
                            "database_contextual_edge_binding_duplicate",
                            "database_association_id may appear only once per arm",
                            path=f"{edge_path}.database_association_id",
                        )
                    )
                    continue
                declared_edge_ids.add(association_id)
                referenced_cues = {str(name) for name in raw_cue_names}
                if len(referenced_cues) != len(raw_cue_names) or not referenced_cues.issubset(cue_ids):
                    issues.append(
                        _issue(
                            "edge_cue_binding_closure_missing",
                            "edge cue_ids must name two distinct declared cues",
                            path=f"{edge_path}.cue_ids",
                        )
                    )
                    continue
                row = connection.execute(
                    """
                    SELECT a.id, a.from_type, a.from_id, a.to_type, a.to_id,
                           a.relation_type, a.relation_key, a.polarity,
                           a.association_mode, a.context_cue_id, a.need_cue_id,
                           a.lifecycle_state, a.expires_at, e.source_key,
                           e.generation AS target_generation,
                           anchor_episode.id AS anchor_episode_id,
                           anchor_episode.generation AS anchor_episode_generation,
                           anchor_concept.id AS anchor_concept_id,
                           cp.domain AS context_domain, np.domain AS need_domain
                    FROM association a
                    LEFT JOIN episode e ON e.id = a.to_id
                    LEFT JOIN episode anchor_episode
                      ON a.from_type = 'episode' AND anchor_episode.id = a.from_id
                    LEFT JOIN concept anchor_concept
                      ON a.from_type = 'concept' AND anchor_concept.id = a.from_id
                    LEFT JOIN association_cue_prototype cp ON cp.id = a.context_cue_id
                    LEFT JOIN association_cue_prototype np ON np.id = a.need_cue_id
                    WHERE a.id = ?
                    """,
                    (association_id,),
                ).fetchone()
                if row is None:
                    issues.append(
                        _issue(
                            "database_contextual_edge_not_found",
                            "declared contextual association ID is absent from this snapshot",
                            path=f"{edge_path}.database_association_id",
                        )
                    )
                    continue
                required_db_cues = {cue_ids[name] for name in referenced_cues}
                actual_db_cues = {row["context_cue_id"], row["need_cue_id"]}
                expiry = (
                    _parse_utc_timestamp(row["expires_at"])
                    if row["expires_at"] is not None
                    else None
                )
                expiry_valid = row["expires_at"] is None or expiry is not None
                expired = expiry is not None and evaluation_as_of is not None and expiry <= evaluation_as_of
                anchor_valid = (
                    (row["from_type"] == "episode" and row["anchor_episode_id"] is not None
                     and row["anchor_episode_generation"] == 0)
                    or (row["from_type"] == "concept" and row["anchor_concept_id"] is not None)
                )
                if not expiry_valid:
                    issues.append(
                        _issue(
                            "database_contextual_edge_expiry_invalid",
                            "declared edge has a non-UTC or malformed expiry timestamp",
                            path=f"{edge_path}.database_association_id",
                        )
                    )
                elif expired:
                    issues.append(
                        _issue(
                            "database_contextual_edge_expired_as_of",
                            "declared edge is not eligible at the pinned evaluation_as_of instant",
                            path=f"{edge_path}.database_association_id",
                        )
                    )
                if not anchor_valid:
                    issues.append(
                        _issue(
                            "database_contextual_edge_anchor_missing",
                            "declared contextual edge has no valid database anchor",
                            path=f"{edge_path}.database_association_id",
                        )
                    )
                valid_edge = (
                    row["association_mode"] == "contextual_recall"
                    and row["relation_type"] == "retrieval"
                    and row["relation_key"] == "contextual_recall"
                    and row["polarity"] == 1
                    and row["to_type"] == "episode"
                    and row["to_id"] == target_episode_id
                    and row["target_generation"] == 0
                    and row["lifecycle_state"] in {"probation", "active"}
                    and expiry_valid
                    and not expired
                    and anchor_valid
                    and actual_db_cues == required_db_cues
                    and row["context_domain"] is not None
                    and row["context_domain"] == row["need_domain"]
                    and row["source_key"] in configured_keys
                )
                if not valid_edge:
                    issues.append(
                        _issue(
                            "database_contextual_edge_closure_mismatch",
                            "declared edge does not close over active database cues, target, and source scope",
                            path=f"{edge_path}.database_association_id",
                        )
                    )

        status = "ready" if len(issues) == arm_issues_before else "invalid_setup"
        if status != "ready":
            execution_state = "not_executable"
        elif candidate_state == "known_empty":
            # A complete, real graph can legitimately return no candidates for
            # a frozen request.  This is distinct from a graph that was absent.
            execution_state = "valid_no_hit"
        else:
            execution_state = "ready_to_execute"
        arm_summaries.append(
            {
                "arm_id": arm_id,
                "requires_contextual": True,
                "candidate_state": candidate_state,
                "status": status,
                "execution_state": execution_state,
            }
        )
    return arm_summaries


def _sidecar_snapshot(
    database: Path, issues: list[dict[str, str]]
) -> dict[str, Any]:
    """Record every live SQLite sidecar for a quiescent-snapshot check.

    A WAL can carry committed logical data not present in the main database
    file, while SHM and rollback-journal files prove the file set is live.
    The v1 run-spec pins only ``database_sha256`` for the main database, so a
    replay must require a quiescent/checkpointed snapshot instead of treating
    any sidecar as an ignorable auxiliary file.
    """
    result: dict[str, Any] = {name: False for name in _SIDECAR_FLAGS}
    for suffix, name in (
        ("-wal", "wal_present"),
        ("-shm", "shm_present"),
        ("-journal", "journal_present"),
    ):
        sidecar = Path(str(database) + suffix)
        if sidecar.exists():
            result[name] = True
            digest = _file_digest_report(sidecar, label=f"database{suffix}", issues=issues)
            if digest is not None:
                result[f"{name}_sha256"] = digest
    return result


def _has_live_sqlite_sidecar(snapshot: Mapping[str, Any]) -> bool:
    return any(snapshot.get(name) is True for name in _SIDECAR_FLAGS)


def _external_digest_checks(
    config: Mapping[str, Any],
    *,
    source_manifest: Path | None,
    scope_manifest: Path | None,
    issues: list[dict[str, str]],
) -> dict[str, str | None]:
    result: dict[str, str | None] = {"source_manifest_sha256": None, "scope_sha256": None}
    inputs = (
        ("source_manifest_sha256", source_manifest, "source_manifest"),
        ("scope_sha256", scope_manifest, "scope_manifest"),
    )
    for expected_name, path, label in inputs:
        if path is None:
            issues.append(
                _issue(
                    "external_snapshot_artifact_missing",
                    f"{label} file is required to attest {expected_name}",
                    path=label,
                )
            )
            continue
        digest = _file_digest_report(path, label=label, issues=issues)
        result[expected_name] = digest
        if digest is not None and digest != config.get(expected_name):
            issues.append(
                _issue(
                    "external_snapshot_digest_mismatch",
                    f"{label} bytes do not match the pinned digest",
                    path=label,
                )
            )
    return result


def preflight_sqlite_evidence(
    run_spec: object,
    database: str | Path,
    *,
    source_manifest: str | Path | None = None,
    scope_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Attest a frozen SQLite snapshot without altering it.

    The report covers only SQLite/source-scope closure.  Call
    ``benchmarks.run_recall_v3.preflight_run`` as well before executing a full
    benchmark: this function intentionally does not inspect evaluator-only
    gold or split files.
    """
    issues: list[dict[str, str]] = []
    config = _read_config(run_spec, issues)
    database_path = Path(database)
    database_digest = _file_digest_report(database_path, label="database", issues=issues)
    if database_digest is not None and database_digest != config.get("database_sha256"):
        issues.append(
            _issue(
                "database_digest_mismatch",
                "database bytes do not match sqlite_preflight.database_sha256",
                path="database",
            )
        )
    sidecars_before = _sidecar_snapshot(database_path, issues)
    external_digests = _external_digest_checks(
        config,
        source_manifest=Path(source_manifest) if source_manifest is not None else None,
        scope_manifest=Path(scope_manifest) if scope_manifest is not None else None,
        issues=issues,
    )

    database_summary: dict[str, Any] = {
        "database_sha256": database_digest,
        "schema_version": None,
        "counts": {},
        "vector_shape_invalid_counts": {},
        "integrity": None,
        "sidecars": {"before": sidecars_before},
    }
    source_key_summary: list[dict[str, Any]] = []
    arm_summaries: list[dict[str, Any]] = []
    connection: sqlite3.Connection | None = None
    initial_digest = database_digest
    declared_schema_version = (
        int(config["schema_version"])
        if _is_positive_int(config.get("schema_version"))
        else None
    )
    declared_schema_contract = (
        _READ_SCHEMA_CONTRACTS.get(declared_schema_version)
        if declared_schema_version is not None
        else None
    )
    if _has_live_sqlite_sidecar(sidecars_before):
        issues.append(
            _issue(
                "sqlite_sidecar_snapshot_not_quiescent",
                "frozen SQLite preflight requires no WAL, SHM, or rollback-journal sidecar",
                path="database-sidecars",
            )
        )
    try:
        # Do not even open a snapshot with live sidecars.  SQLite can need to
        # recover a hot journal, which is incompatible with this no-write,
        # main-file-only attestation contract.
        if database_digest is not None and not _has_live_sqlite_sidecar(sidecars_before):
            connection = _connect_readonly(database_path)
            schema_meta_ready = _schema_checks(
                connection, _SCHEMA_META_COLUMNS, issues
            )
            database_summary["integrity"] = _pragma_checks(connection, issues)
            if schema_meta_ready:
                schema_version = _read_schema_version(connection, issues)
                database_summary["schema_version"] = schema_version
                database_schema_contract = _READ_SCHEMA_CONTRACTS.get(schema_version)
                if schema_version is not None and database_schema_contract is None:
                    issues.append(
                        _issue(
                            "database_schema_version_unsupported",
                            "database schema version is not an explicitly supported frozen read schema",
                            path="database.schema_meta.schema_version",
                        )
                    )
                if (
                    schema_version is not None
                    and declared_schema_version is not None
                    and schema_version != declared_schema_version
                ):
                    issues.append(
                        _issue(
                            "schema_version_mismatch",
                            "database schema version does not match the frozen run spec",
                            path="database.schema_meta.schema_version",
                        )
                    )
                if (
                    database_schema_contract is not None
                    and declared_schema_contract is not None
                    and schema_version == declared_schema_version
                    and _schema_checks(connection, database_schema_contract, issues)
                ):
                    counts = _snapshot_counts(connection)
                    database_summary["counts"] = counts
                    expected_counts = config.get("expected_counts")
                    if _is_mapping(expected_counts):
                        for name in {*_COUNT_TABLES, "contextual_edge"}:
                            if name in expected_counts and counts.get(name) != expected_counts[name]:
                                issues.append(
                                    _issue(
                                        "database_count_mismatch",
                                        "database count does not match the frozen snapshot",
                                        path=f"database.counts.{name}",
                                    )
                                )
                    if (
                        counts["episode_fts"] != counts["episode"]
                        or counts["source_fts"] != counts["source"]
                    ):
                        issues.append(
                            _issue(
                                "fts_count_mismatch",
                                "FTS row counts do not match their authoritative tables",
                                path="database.fts",
                            )
                        )
                    embedding = config.get("embedding_space")
                    if _is_mapping(embedding) and _is_positive_int(embedding.get("dimension")):
                        database_summary["vector_shape_invalid_counts"] = _vector_checks(
                            connection,
                            dimension=int(embedding["dimension"]),
                            model_id=str(embedding.get("model_id", "")),
                            evaluation_as_of=(
                                config.get("_evaluation_as_of")
                                if isinstance(config.get("_evaluation_as_of"), datetime)
                                else None
                            ),
                            issues=issues,
                        )
                    source_key_summary = _source_key_checks(
                        connection,
                        config.get("_source_key_requirements", []),
                        issues,
                    )
                    if _is_mapping(run_spec):
                        arm_summaries = _arm_bindings(run_spec, config, connection, issues)
    except (OSError, sqlite3.Error, ValueError) as exc:
        issues.append(
            _issue(
                "readonly_database_open_or_query_failed",
                f"read-only SQLite preflight could not complete: {exc}",
                path="database",
            )
        )
    finally:
        if connection is not None:
            connection.close()

    # A second digest makes a changing database snapshot an explicit refusal;
    # no checkpoint, backup, or retry is attempted because each could mask the
    # very snapshot race the preflight is supposed to detect.
    final_digest = _file_digest_report(database_path, label="database", issues=issues)
    if initial_digest is not None and final_digest is not None and initial_digest != final_digest:
        issues.append(
            _issue(
                "database_changed_during_preflight",
                "database bytes changed while read-only checks were running",
                path="database",
            )
        )
    database_summary["database_sha256_after"] = final_digest
    sidecars_after = _sidecar_snapshot(database_path, issues)
    database_summary["sidecars"]["after"] = sidecars_after
    if (
        not _has_live_sqlite_sidecar(sidecars_before)
        and _has_live_sqlite_sidecar(sidecars_after)
    ):
        issues.append(
            _issue(
                "sqlite_sidecar_snapshot_not_quiescent",
                "a WAL, SHM, or rollback-journal sidecar appeared during preflight",
                path="database-sidecars",
            )
        )
    if sidecars_before.get("wal_present") or sidecars_after.get("wal_present"):
        issues.append(
            _issue(
                "wal_snapshot_not_pinned",
                "a WAL is present but the v1 snapshot contract pins only main database bytes",
                path="database-wal",
            )
        )
    if sidecars_before != sidecars_after:
        issues.append(
            _issue(
                "database_sidecar_changed_during_preflight",
                "database journal sidecars changed while read-only checks were running",
                path="database-sidecars",
            )
        )

    status = "ready" if not issues else "invalid_setup"
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "can_execute": status == "ready",
        "preflight_layer": "sqlite_evidence_only",
        "requires_json_preflight": True,
        "evaluation_as_of": (
            config.get("evaluation_as_of")
            if isinstance(config.get("evaluation_as_of"), str)
            else None
        ),
        "network_calls": 0,
        "model_calls": 0,
        "database_writes": 0,
        "external_digests": external_digests,
        "external_attestation": "byte_digest_only",
        "database": database_summary,
        "source_key_closure": source_key_summary,
        "arms": arm_summaries,
        "issues": issues,
    }


class _JsonInputError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _JsonInputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> object:
    raise _JsonInputError(f"non-standard JSON number: {value}")


def _load_json(path: Path) -> object:
    try:
        raw = path.read_bytes()
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _JsonInputError) as exc:
        raise _JsonInputError(f"cannot safely load run spec: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attest a v3 SQLite snapshot without writing to it."
    )
    parser.add_argument("--spec", type=Path, required=True, help="v3 run-spec JSON")
    parser.add_argument("--database", type=Path, required=True, help="SQLite database snapshot")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--scope-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = _load_json(args.spec)
        report = preflight_sqlite_evidence(
            spec,
            args.database,
            source_manifest=args.source_manifest,
            scope_manifest=args.scope_manifest,
        )
    except _JsonInputError as exc:
        report = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "invalid_setup",
            "can_execute": False,
            "preflight_layer": "sqlite_evidence_only",
            "requires_json_preflight": True,
            "evaluation_as_of": None,
            "network_calls": 0,
            "model_calls": 0,
            "database_writes": 0,
            "external_digests": {},
            "external_attestation": "byte_digest_only",
            "database": {},
            "source_key_closure": [],
            "arms": [],
            "issues": [_issue("input_load_failed", str(exc), path="cli")],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
