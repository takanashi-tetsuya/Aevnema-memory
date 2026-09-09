"""Run one isolated, non-scorable real Q1 -> Q2 diagnostic pilot.

This runner is deliberately separate from :mod:`run_recall_v3`.  It never
loads a gold manifest, split, formal benchmark spec, or trace replay ticket;
therefore it cannot produce a formal score.  Its only mutable state is a set
of SQLite clones under the supplied output directory.

The Q1 worker uses the normal ``QueryEngine.query(... contextual_learning``
path.  A request-local finalizer observer records the public finalizer's
boundary without pausing it.  A separately cloned pre-Q1 state supplies a
clearly labelled before-commit baseline only after Q1 has a terminal result.
Every terminal outcome is written once, including a failed finalizer, a
timeout, no eligible candidate, or no V17 runtime projection.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from threading import Lock
from time import monotonic, perf_counter
import traceback
from typing import Any, Callable, Mapping

import numpy as np

from memory_demo.app import MemoryApplication
from memory_demo.association_overlay import AssociationOverlay
from memory_demo.associations.growth import AssociationGrowthEngine
from memory_demo.associations.traversal import GraphTraverser
from memory_demo.chronology import ChronologyService
from memory_demo.config import AppConfig
from memory_demo.database import Database
from memory_demo.embeddings import normalize_query_text
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher


PILOT_SCHEMA = "aevnema.v3.q1_q2_diagnostic_pilot.v2"
CASE_SCHEMA = "aevnema.v3.q1_q2_diagnostic_source_slice.v1"
FULL_LOCAL_NAME = "q1_q2_case.full_local.json"
CASE_REPORT_NAME = "q1_q2_case.report.md"
Q2_ARM_NAMES = (
    "before_commit_probe",
    "post_send_immediate_q2",
    "learning_ready_q2",
    "edge_masked_q2",
    "ordinary_embedding_cache_q2",
    "edge_restart_q2",
    "source_changed_q2",
)


def _pilot_scope_hash(scope_label: str) -> str:
    """Derive the opaque V17 scope key from a diagnostic-only scope label.

    The input manifest deliberately carries a readable case label so humans
    can review that Q1 and Q2 are in the same isolated diagnostic scope. The
    exact-revisit contract API, however, accepts only an opaque digest. Keep
    the conversion at this runner boundary, with a fixed namespace and the
    same input for every arm, rather than weakening the engine's scope guard.
    """

    normalized = str(scope_label or "").strip()
    if not normalized:
        raise ValueError("diagnostic pilot scope label is required")
    digest = sha256(
        b"aevnema/v3/q1-q2-diagnostic-scope/v1\0"
        + normalized.encode("utf-8")
    ).hexdigest()
    return "pilot-scope:sha256:" + digest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        # Learned runtime receipts carry opaque maps keyed by both string
        # fingerprints and integer episode ids.  JSON permits both after
        # encoding, but ``sort_keys=True`` compares them before encoding and
        # can crash the terminal artifact write after a successful Q1.
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _collect_answer_evidence_checkpoints(output_dir: Path) -> dict[str, object]:
    """Embed the experiment-local answer boundary receipts in full-local trace."""

    files = sorted((output_dir / "logs" / "q1").rglob("*.answer-evidence.jsonl"))
    records: list[dict[str, object]] = []
    parse_errors: list[dict[str, object]] = []
    relative_files: list[str] = []
    for path in files:
        relative_files.append(str(path.relative_to(output_dir)))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                parse_errors.append(
                    {
                        "file": str(path.relative_to(output_dir)),
                        "line": line_number,
                        "error": str(error),
                    }
                )
                continue
            if isinstance(record, dict):
                records.append(record)
    return {
        "files": relative_files,
        "records": records,
        "parse_errors": parse_errors,
        "sharing_default": "excluded_by_default_from_jsonl_summary_and_architecture_audit_tools",
    }


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


@dataclass(frozen=True, slots=True)
class PilotCase:
    q1_text: str
    q2_text: str
    domain: str
    scope_hash: str
    manifest_path: Path
    manifest_sha256: str

    @classmethod
    def from_manifest(cls, path: Path) -> "PilotCase":
        payload = _read_object(path)
        if payload.get("schema") != CASE_SCHEMA:
            raise ValueError("diagnostic pilot input has an unexpected schema")
        if payload.get("formal_scoring_eligible") is not False:
            raise ValueError("diagnostic pilot input must be explicitly non-scorable")
        if payload.get("promotion_prohibited") is not True:
            raise ValueError("diagnostic pilot input must explicitly prohibit promotion")
        if any(key in payload for key in ("gold", "expected_answer", "score")):
            raise ValueError("diagnostic pilot input must not carry scoring material")
        raw_input = payload.get("pilot_input")
        if not isinstance(raw_input, dict):
            raise ValueError("diagnostic pilot input needs pilot_input")
        if any(key in raw_input for key in ("gold", "answer", "expected", "label")):
            raise ValueError("pilot_input must not carry Q2 gold, answer, or label")
        q1 = str(raw_input.get("q1_text", "")).strip()
        q2 = str(raw_input.get("q2_text", "")).strip()
        domain = str(raw_input.get("contextual_domain", "")).strip()
        scope_label = str(raw_input.get("contextual_revisit_scope", "")).strip()
        if not q1 or not q2 or not domain or not scope_label:
            raise ValueError("pilot input needs non-empty Q1/Q2, domain, and scope")
        # V17 automatic revisit is only meaningful for the same normalized
        # raw request.  Reject a near-but-not-equal test rather than letting a
        # report imply that a paraphrase had an automatic revisit opportunity.
        if normalize_query_text(q1) != normalize_query_text(q2):
            raise ValueError("V17 diagnostic Q1 and Q2 must normalize identically")
        return cls(
            q1_text=q1,
            q2_text=q2,
            domain=domain,
            scope_hash=_pilot_scope_hash(scope_label),
            manifest_path=path.resolve(),
            manifest_sha256=_sha256_file(path),
        )


class ProviderLedger:
    """Per-request provider observations, with no prompts or responses."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.logical_batches: list[dict[str, object]] = []
        self.observations: list[dict[str, object]] = []
        self.fallbacks: list[dict[str, object]] = []

    def logical_batch_started(self, **kwargs: object) -> None:
        with self._lock:
            self.logical_batches.append(dict(kwargs))

    def provider_call_finished(self, observation: object) -> None:
        if is_dataclass(observation):
            payload = asdict(observation)
        elif isinstance(observation, Mapping):
            payload = dict(observation)
        else:
            raise TypeError("provider observation must be structured")
        with self._lock:
            self.observations.append(payload)

    def fallback_used(self, **kwargs: object) -> None:
        with self._lock:
            self.fallbacks.append(dict(kwargs))

    def export(self) -> dict[str, object]:
        with self._lock:
            observations = [dict(item) for item in self.observations]
            successful_statuses = {"success", "succeeded"}
            return {
                "logical_batches": [dict(item) for item in self.logical_batches],
                "observations": observations,
                "fallbacks": [dict(item) for item in self.fallbacks],
                "counts": {
                    "logical_batches": len(self.logical_batches),
                    "http_attempts": len(observations),
                    "sent": sum(bool(item.get("sent")) for item in observations),
                    "succeeded": sum(
                        item.get("status") in successful_statuses
                        for item in observations
                    ),
                    "failed_or_rejected": sum(
                        item.get("status") not in successful_statuses
                        for item in observations
                    ),
                    "fallbacks": len(self.fallbacks),
                },
            }


class FinalizerObserver:
    """Observe the public finalizer without delaying Q1's commit path."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.before_finalizer_at: float | None = None
        self.finalizer_returned_at: float | None = None
        self.finalizer_calls = 0

    def wrap(self, finalizer: Callable[..., object]) -> Callable[..., object]:
        def observed(event, plan, cue_materializations):
            with self._lock:
                self.finalizer_calls += 1
                if self.finalizer_calls != 1:
                    raise RuntimeError("pilot finalizer observer was entered more than once")
                self.before_finalizer_at = monotonic()
            result = finalizer(event, plan, cue_materializations)
            with self._lock:
                self.finalizer_returned_at = monotonic()
            return result

        return observed

    def export(self, *, started_at: float) -> dict[str, object]:
        def elapsed(value: float | None) -> float | None:
            return None if value is None else round((value - started_at) * 1000.0, 3)

        with self._lock:
            return {
                "finalizer_calls": self.finalizer_calls,
                "before_public_finalizer_at_ms": elapsed(self.before_finalizer_at),
                "public_finalizer_returned_at_ms": elapsed(self.finalizer_returned_at),
                "q1_pause_injected": False,
            }


def _clone_sqlite(source: Path, target: Path) -> None:
    """Back up a source read-only and normalize only the new destination."""

    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    read_connection = sqlite3.connect(source_uri, uri=True, timeout=5.0)
    try:
        write_connection = sqlite3.connect(target, timeout=5.0)
        try:
            read_connection.backup(write_connection)
            write_connection.execute("PRAGMA journal_mode=DELETE")
            write_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            # Some V3 schemas contain application-owned validation UDFs in
            # their DDL.  Bare sqlite connections cannot resolve those UDFs
            # during ``quick_check`` even though the copied database is
            # sound; use checks that neither invoke schema expressions nor
            # mutate the clone.  A normal read of both primary corpus tables
            # proves the backup is queryable for this runner.
            foreign_key_errors = list(write_connection.execute("PRAGMA foreign_key_check"))
            if foreign_key_errors:
                raise RuntimeError("new pilot SQLite clone has foreign-key errors")
            for table in ("source", "episode"):
                write_connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        finally:
            write_connection.close()
    finally:
        read_connection.close()


def _receipt_and_edge_ids(app: MemoryApplication) -> tuple[set[int], set[int]]:
    with app.db.connection() as connection:
        receipts = {
            int(row[0])
            for row in connection.execute("SELECT id FROM contextual_creation_receipt")
        }
        associations = {
            int(row[0]) for row in connection.execute("SELECT id FROM association")
        }
    return receipts, associations


def _source_evidence_stats(app: MemoryApplication) -> dict[str, int]:
    with app.db.connection() as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM episode").fetchone()[0])
        bound = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM episode
                WHERE evidence_origin = 'source'
                  AND COALESCE(evidence_quotes_json, '[]') != '[]'
                  AND COALESCE(evidence_spans_json, '[]') != '[]'
                """
            ).fetchone()[0]
        )
    return {
        "episodes": total,
        "episodes_with_literal_source_evidence": bound,
        "episodes_without_literal_source_evidence": total - bound,
    }


def _pilot_config(env_file: Path, database_path: Path, log_dir: Path) -> AppConfig:
    config = AppConfig.from_env(env_file)
    config.database_path = database_path
    config.log_dir = log_dir
    # Request-scoped controls make any observed Q2 difference attributable to
    # the Q1 receipt, not to Q2 growth or an association text/cue shortcut.
    config.retrieval.contextual_association_enabled = True
    config.retrieval.contextual_association_shadow = False
    config.retrieval.growth_max_rounds = 0
    # With zero growth rounds there are no provisional rows to retain or
    # prune.  Disable the unused cleanup branch uniformly for every arm so a
    # read-only edge overlay never attempts a repository restore operation.
    config.retrieval.growth_persist_only_used = False
    config.retrieval.association_cue_enabled = False
    config.retrieval.association_cue_fast_path_enabled = False
    config.retrieval.contextual_promotion_enabled = False
    config.retrieval.contextual_restricted_rewrite_enabled = False
    # The diagnostic does not speculate with the follow-up planner: a second
    # planner call is allowed only when the initial local candidate pass has
    # an empty evidence slot.  No provider model or deadline is changed.
    config.retrieval.followup_planning_mode = "missing_slots"
    # This case asks one atomic factual question. Keep every planner
    # paraphrase for retrieval, but freeze only its first atomic question as
    # the delivery denominator so paraphrases do not become fictitious
    # independent claims. The public engine records that denominator before
    # any candidate retrieval.
    config.retrieval.rerank_atomic_query_limit = 1
    # The same single-fact diagnostic needs one answer Episode plus one
    # independently retrieved source-bound anchor for a possible public
    # learning plan.  This constrains only the prompt-facing delivery set;
    # candidate discovery and the 11 planned retrieval vectors remain intact.
    config.retrieval.answer_episode_limit = 2
    # V6 observed that the optional compressor repeated the initial
    # source-bound selection exactly (episode 32); the deterministic floor
    # then added the same independent anchor (episode 21).  Do not spend a
    # second cloud call to reproduce that identical delivery decision in the
    # next diagnostic version. This does not weaken source verification or
    # alter discovery, selection, answer audit, or the request deadline.
    config.retrieval.rerank_audit_enabled = False
    # This experiment explicitly retains answer/audit evidence at the model
    # boundary.  Production/default profiles leave the companion disabled.
    config.retrieval.answer_evidence_checkpoint_enabled = True
    # Preserve time for the already-authorized transport fallback.  Without
    # this pilot-local bound, one stalled primary HTTP request consumes the
    # entire outer Q1 deadline and makes the configured fallback unreachable.
    # Successful primary calls are unchanged; a timeout has no retry storm
    # and may make exactly one DeepSeek -> GLM-4.5V transport fallback while
    # the original 120-second total budget remains authoritative.
    config.model.timeout_seconds = 25.0
    config.model.max_retries = 0
    config.retrieval.validate()
    return config


def _open_app(env_file: Path, db_path: Path, log_dir: Path) -> tuple[MemoryApplication, AppConfig]:
    config = _pilot_config(env_file, db_path, log_dir)
    app = MemoryApplication(config)
    app.rebuild_indexes()
    app.rebuild_contextual_indexes()
    return app, config


def _trace_context(model: object, ledger: ProviderLedger):
    trace_request = getattr(model, "trace_request", None)
    return trace_request(ledger) if callable(trace_request) else nullcontext()


def _call_query(
    engine: object,
    question: str,
    *,
    domain: str,
    scope_hash: str,
    deadline_seconds: float,
    query_embeddings_override: dict[str, np.ndarray] | None = None,
    frozen_plan: dict[str, object] | None = None,
    query_vector_bundle: object | None = None,
    strict_vector_bundle: bool = False,
    contextual_learning: bool = False,
    learning_request_id: str | None = None,
    generate_answer: bool = True,
    stop_after: str | None = None,
) -> dict[str, object]:
    ledger = ProviderLedger()
    started_wall = _utc_now()
    started = perf_counter()
    try:
        with _trace_context(engine.model, ledger):
            result = engine.query(
                question,
                generate_answer=generate_answer,
                stop_after=stop_after,
                deadline_seconds=deadline_seconds,
                query_embeddings_override=query_embeddings_override,
                frozen_plan=frozen_plan,
                query_vector_bundle=query_vector_bundle,
                strict_vector_bundle=strict_vector_bundle,
                contextual_domain=domain,
                contextual_revisit_scope_hash=scope_hash,
                contextual_learning=contextual_learning,
                learning_request_id=learning_request_id,
            )
        return {
            "status": "completed",
            "started_at": started_wall,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "result": result,
            "provider": ledger.export(),
        }
    except BaseException as error:  # preserve every actual terminal outcome
        return {
            "status": "failed",
            "started_at": started_wall,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "provider": ledger.export(),
        }


def _call_public_exact_revisit(
    engine: object,
    question: str,
    *,
    domain: str,
    scope_hash: str,
    deadline_seconds: float,
) -> dict[str, object]:
    """Exercise the public, provider-free Q2 reuse boundary exactly once."""

    ledger = ProviderLedger()
    started_wall = _utc_now()
    started = perf_counter()
    try:
        with _trace_context(engine.model, ledger):
            result = engine.try_contextual_revisit(
                question,
                contextual_domain=domain,
                contextual_revisit_scope_hash=scope_hash,
                deadline_seconds=deadline_seconds,
            )
        return {
            "status": "completed",
            "started_at": started_wall,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "result": result,
            "revisit_status": "hit" if isinstance(result, Mapping) else "miss",
            "provider": ledger.export(),
        }
    except BaseException as error:  # preserve every actual terminal outcome
        return {
            "status": "failed",
            "started_at": started_wall,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "provider": ledger.export(),
        }


def _mask_engine_edge(engine: object, hidden_association_id: int) -> None:
    """Hide one Q1 association everywhere the Q2 engine can resolve it."""

    overlay = AssociationOverlay(engine.associations, hidden_ids=[hidden_association_id])
    engine.associations = overlay
    engine.traverser = GraphTraverser(overlay)
    engine.growth = AssociationGrowthEngine(
        engine.model, overlay, engine.episodes, engine.config.weights, engine.logger
    )
    engine.chronology = ChronologyService(engine.episodes, overlay, engine.logger)
    matcher = engine.contextual_matcher
    if matcher is not None:
        engine.contextual_matcher = ContextualAssociationMatcher(
            matcher.context_index,
            matcher.need_index,
            overlay,
            context_threshold=matcher.context_threshold,
            need_threshold=matcher.need_threshold,
            combine_mode=matcher.combine_mode,
            context_top_k=matcher.context_top_k,
            need_top_k=matcher.need_top_k,
            edge_top_k=matcher.edge_top_k,
            endpoint_limit=matcher.endpoint_limit,
            embedding_space_id=matcher.embedding_space_id,
        )


def _result_summary(record: Mapping[str, object]) -> dict[str, object]:
    if record.get("status") != "completed":
        error = record.get("error")
        return {
            "run_status": str(record.get("status", "unknown")),
            "elapsed_ms": record.get("elapsed_ms"),
            "error_type": error.get("type") if isinstance(error, Mapping) else None,
            "provider_counts": (
                record.get("provider", {}).get("counts", {})
                if isinstance(record.get("provider"), Mapping)
                else {}
            ),
        }
    if not isinstance(record.get("result"), Mapping):
        return {
            "run_status": "exact_revisit_miss",
            "elapsed_ms": record.get("elapsed_ms"),
            "candidate_episode_ids": [],
            "candidate_count": 0,
            "selected_episode_ids": [],
            "selected_count": 0,
            "delivered_episode_ids": [],
            "delivered_coverage": False,
            "actual_delivery_episode_count": 0,
            "actual_delivery_source_fact_count": 0,
            "actual_delivery_token_cost": 0,
            "delivery_loss": None,
            "missing_required_clauses": [],
            "answer_generation_skipped": True,
            "contextual_attached_edges": [],
            "contextual_reason": "public_exact_revisit_miss",
            "exact_revisit_executed_modules": {},
            "actual_timings": {},
            "embedding_cache": {},
            "provider_counts": (
                record.get("provider", {}).get("counts", {})
                if isinstance(record.get("provider"), Mapping)
                else {}
            ),
        }
    result = record["result"]
    candidate_ids = result.get("candidate_episode_ids", [])
    delivered_ids = result.get("episode_ids", [])
    evidence_trace = result.get("evidence_slot_trace")
    selector = (
        evidence_trace.get("slot_selector_v3")
        if isinstance(evidence_trace, Mapping)
        else None
    )
    treatment = (
        selector.get("treatment_selector")
        if isinstance(selector, Mapping) and isinstance(selector.get("treatment_selector"), Mapping)
        else None
    )
    contextual = result.get("contextual_association")
    exact = result.get("exact_revisit")
    timings = result.get("timings")
    delivered_rows = result.get("evidence_episodes")
    return {
        "run_status": "completed",
        "elapsed_ms": record.get("elapsed_ms"),
        "candidate_episode_ids": list(candidate_ids) if isinstance(candidate_ids, list) else [],
        "candidate_count": len(candidate_ids) if isinstance(candidate_ids, list) else 0,
        "selected_episode_ids": (
            list(treatment.get("selected_episode_ids", [])) if isinstance(treatment, Mapping) else []
        ),
        "selected_count": (
            len(treatment.get("selected_episode_ids", [])) if isinstance(treatment, Mapping) and isinstance(treatment.get("selected_episode_ids"), list) else 0
        ),
        "delivered_episode_ids": list(delivered_ids) if isinstance(delivered_ids, list) else [],
        "delivered_coverage": bool(
            (
                result.get("answer_generation_skipped") is False
                and isinstance(result.get("answer"), str)
                and result.get("answer", "").strip()
            )
            or (
                result.get("answer_generation_skipped") is True
                and isinstance(delivered_ids, list)
                and bool(delivered_ids)
            )
        ),
        "actual_delivery_episode_count": result.get("actual_delivery_episode_count"),
        "actual_delivery_source_fact_count": result.get("actual_delivery_source_fact_count"),
        "actual_delivery_token_cost": result.get("actual_delivery_token_cost"),
        "delivery_loss": result.get("delivery_loss"),
        "missing_required_clauses": result.get("missing_required_clauses", []),
        "answer_generation_skipped": result.get("answer_generation_skipped"),
        "contextual_attached_edges": (
            list(contextual.get("attached_edges", [])) if isinstance(contextual, Mapping) else []
        ),
        "contextual_reason": contextual.get("reason") if isinstance(contextual, Mapping) else None,
        "exact_revisit_executed_modules": (
            exact.get("executed_modules") if isinstance(exact, Mapping) else None
        ),
        "actual_timings": timings if isinstance(timings, Mapping) else {},
        "embedding_cache": result.get("query_embedding_cache", {}),
        "provider_counts": (
            record.get("provider", {}).get("counts", {})
            if isinstance(record.get("provider"), Mapping)
            else {}
        ),
    }


def _q2_arm(
    name: str,
    *,
    env_file: Path,
    database: Path,
    output_dir: Path,
    case: PilotCase,
    deadline_seconds: float,
    hidden_association_id: int | None = None,
    rebuild_before_query: bool = False,
) -> dict[str, object]:
    arm_log_dir = output_dir / "logs" / name
    app, config = _open_app(env_file, database, arm_log_dir)
    if rebuild_before_query:
        app.rebuild_indexes()
        app.rebuild_contextual_indexes()
    engine = app.query_engine(config=config)
    if hidden_association_id is not None:
        _mask_engine_edge(engine, hidden_association_id)
    record = _call_public_exact_revisit(
        engine,
        case.q2_text,
        domain=case.domain,
        scope_hash=case.scope_hash,
        deadline_seconds=deadline_seconds,
    )
    return {
        "arm": name,
        "database": str(database),
        "edge_mask": {
            "enabled": hidden_association_id is not None,
            "hidden_association_id": hidden_association_id,
            "scope": "one Q1 association only; ordinary base retrieval remains available",
        },
        "restart_rebuild": rebuild_before_query,
        "execution_mode": "public_exact_revisit_only_zero_model",
        "record": record,
        "summary": _result_summary(record),
    }


def _invalidate_source_clone(database: Path) -> dict[str, object]:
    """Change only a pilot-owned clone to exercise Source invalidation."""

    marker = "\n[diagnostic_source_revision: changed_for_public_q2_invalidation]"
    # Source writes fire FTS triggers that call ``memory_bigram_tokens``.
    # A bare sqlite3 connection omits that project UDF, which used to abort
    # this final Q2 arm after Q1 had already succeeded and before the terminal
    # case package could be written.  The standard connection factory gives
    # this isolated clone the same trigger contract as all application writes.
    clone_database = Database(database)
    with clone_database.transaction() as connection:
        before = int(connection.execute("SELECT COUNT(*) FROM source").fetchone()[0])
        connection.execute("UPDATE source SET raw_text = raw_text || ?", (marker,))
    return {
        "changed_source_rows": before,
        "mutation": "append_nonsemantic_diagnostic_revision_marker_to_pilot_clone",
    }


def _terminal_q2_arms(reason: str) -> dict[str, dict[str, object]]:
    """Give every dependent Q2 arm an immediate, explicit terminal state."""

    return {
        name: {
            "arm": name,
            "not_run_reason": reason,
            "edge_mask": {
                "enabled": name == "edge_masked_q2"
            },
            "summary": {"run_status": "not_run", "provider_counts": {}},
        }
        for name in Q2_ARM_NAMES
    }


def _q2_arm_checkpoint_path(output_dir: Path, name: str) -> Path:
    """Return the full-local, independently durable result for one Q2 arm."""

    if name not in Q2_ARM_NAMES:
        raise ValueError("unknown Q2 arm checkpoint name")
    return output_dir / f"q2_{name}.full_local.json"


def _write_q2_arm_checkpoint(
    *,
    output_dir: Path,
    name: str,
    case: PilotCase,
    q1_learning: Mapping[str, object],
    q1_new_material: Mapping[str, object],
    status: str,
    arm: Mapping[str, object] | None = None,
    source_mutation: Mapping[str, object] | None = None,
) -> None:
    """Atomically preserve one Q2 arm without inferring unobserved outcomes.

    The pilot's aggregate full-local trace remains the canonical case package.
    These small, arm-specific full-local files close the narrow post-Q1 crash
    gap: a started arm is explicitly pending until its public result returns.
    They intentionally carry no answer cache, planner output, or synthetic
    reuse material.
    """

    if status not in {
        "q2_arm_pending_public_preflight",
        "q2_source_mutation_complete_pending_public_preflight",
        "q2_arm_terminal",
        "q2_arm_not_run",
    }:
        raise ValueError("invalid Q2 arm checkpoint status")
    payload: dict[str, object] = {
        "schema": PILOT_SCHEMA,
        "status": status,
        "written_at": _utc_now(),
        "case": {
            "q2_text": case.q2_text,
            "domain": case.domain,
            "scope_hash": case.scope_hash,
            "normalized_questions_equal": normalize_query_text(case.q1_text)
            == normalize_query_text(case.q2_text),
        },
        "q1_dependency": {
            "learning_status": q1_learning.get("status"),
            "learning_reason": q1_learning.get("reason"),
            "new_receipt_ids": q1_new_material.get("new_receipt_ids"),
            "new_association_ids": q1_new_material.get("new_association_ids"),
            "new_receipts": q1_new_material.get("new_receipts"),
        },
        "arm": arm
        if arm is not None
        else {
            "arm": name,
            "observation": "not_observed_yet",
            "summary": {"run_status": "not_observed", "provider_counts": {}},
        },
    }
    if source_mutation is not None:
        payload["source_mutation"] = source_mutation
    _write_json(_q2_arm_checkpoint_path(output_dir, name), payload)


def _run_q2_arm_checkpointed(
    name: str,
    *,
    env_file: Path,
    database: Path,
    output_dir: Path,
    case: PilotCase,
    deadline_seconds: float,
    q1_learning: Mapping[str, object],
    q1_new_material: Mapping[str, object],
    hidden_association_id: int | None = None,
    rebuild_before_query: bool = False,
    source_mutation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run exactly one public Q2 preflight with durable before/after state."""

    pending_status = (
        "q2_source_mutation_complete_pending_public_preflight"
        if source_mutation is not None
        else "q2_arm_pending_public_preflight"
    )
    _write_q2_arm_checkpoint(
        output_dir=output_dir,
        name=name,
        case=case,
        q1_learning=q1_learning,
        q1_new_material=q1_new_material,
        status=pending_status,
        source_mutation=source_mutation,
    )
    arm = _q2_arm(
        name,
        env_file=env_file,
        database=database,
        output_dir=output_dir,
        case=case,
        deadline_seconds=deadline_seconds,
        hidden_association_id=hidden_association_id,
        rebuild_before_query=rebuild_before_query,
    )
    if source_mutation is not None:
        arm["source_mutation"] = dict(source_mutation)
    _write_q2_arm_checkpoint(
        output_dir=output_dir,
        name=name,
        case=case,
        q1_learning=q1_learning,
        q1_new_material=q1_new_material,
        status="q2_arm_terminal",
        arm=arm,
        source_mutation=source_mutation,
    )
    return arm


def _new_receipt_material(
    app: MemoryApplication,
    *,
    before_receipts: set[int],
    before_edges: set[int],
) -> dict[str, object]:
    after_receipts, after_edges = _receipt_and_edge_ids(app)
    new_receipts = sorted(after_receipts - before_receipts)
    rows: list[dict[str, object]] = []
    for receipt_id in new_receipts:
        receipt = app.associations.get_contextual_creation_receipt(receipt_id)
        if receipt is None:
            continue
        association_id = int(receipt["association_id"])
        runtime = app.associations.load_contextual_revisit_runtime_manifest(receipt_id)
        rows.append(
            {
                "receipt": receipt,
                "association_id_is_new": association_id not in before_edges,
                "runtime_manifest": runtime,
            }
        )
    return {
        "before_receipt_count": len(before_receipts),
        "after_receipt_count": len(after_receipts),
        "new_receipt_ids": new_receipts,
        "new_association_ids": sorted(after_edges - before_edges),
        "new_receipts": rows,
    }


def _write_case_report(
    path: Path,
    *,
    payload: Mapping[str, object],
) -> None:
    """Readable report with no raw provider payload, evidence text, or vectors."""

    case = payload["case"]
    q1 = payload["q1"]
    arms = payload["arms"]
    assert isinstance(case, Mapping) and isinstance(q1, Mapping) and isinstance(arms, Mapping)
    lines = [
        "# V3 Q1→Q2 diagnostic pilot case report",
        "",
        "- Status: diagnostic only; unapproved gold was not loaded and no formal score was calculated.",
        "- Promotion/canary: closed.",
        f"- Input question hash: `sha256:{sha256(str(case['q2_text']).encode('utf-8')).hexdigest()}`",
        f"- Q1/Q2 normalized equality: `{case['normalized_questions_equal']}`",
        f"- Full-local raw trace: `{FULL_LOCAL_NAME}` (local-only; contains raw Q1/Q2 results and provider observations).",
        "",
        "## Q1 learning boundary",
        "",
        f"- Q1 status: `{q1.get('record', {}).get('status')}`",
        f"- Learning status/reason: `{q1.get('learning_status')}` / `{q1.get('learning_reason')}`",
        f"- Local answer-boundary checkpoint records: `{len(q1.get('answer_evidence_checkpoints', {}).get('records', [])) if isinstance(q1.get('answer_evidence_checkpoints'), Mapping) else 'not_observed'}`.",
        f"- Public finalizer observed: `{q1.get('finalizer_observer', {}).get('before_public_finalizer_at_ms')}` ms; returned: `{q1.get('finalizer_observer', {}).get('public_finalizer_returned_at_ms')}` ms. Q1 pause injected: `{q1.get('finalizer_observer', {}).get('q1_pause_injected')}`.",
        f"- New receipts: `{q1.get('new_material', {}).get('new_receipt_ids', [])}`; new edges: `{q1.get('new_material', {}).get('new_association_ids', [])}`.",
        "",
        "## Same-Q2 arms",
        "",
        "| Arm | Run | Candidate | Selected | Delivered coverage | Edge mask | Exact executed modules | Provider HTTP attempts | Elapsed ms |",
        "|---|---|---:|---:|---|---|---|---:|---:|",
    ]
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            continue
        summary = arm.get("summary", {})
        if not isinstance(summary, Mapping):
            summary = {}
        edge_mask = arm.get("edge_mask", {})
        if not isinstance(edge_mask, Mapping):
            edge_mask = {}
        provider = summary.get("provider_counts", {})
        if not isinstance(provider, Mapping):
            provider = {}
        modules = summary.get("exact_revisit_executed_modules")
        if isinstance(modules, Mapping):
            executed = [str(key) for key, value in modules.items() if value is True]
            skipped = [str(key) for key, value in modules.items() if value is False]
            rendered_modules = (
                "executed: " + ", ".join(executed or ["none"])
                + "; skipped: " + ", ".join(skipped or ["none"])
            )
        elif isinstance(modules, list):
            rendered_modules = ", ".join(map(str, modules))
        else:
            rendered_modules = "not observed"
        lines.append(
            "| {name} | {status} | {candidate} | {selected} | {delivered} | {masked} | {modules} | {calls} | {elapsed} |".format(
                name=name,
                status=summary.get("run_status", "not_run"),
                candidate=summary.get("candidate_count", 0),
                selected=summary.get("selected_count", 0),
                delivered=summary.get("delivered_coverage", False),
                masked=edge_mask.get("enabled", False),
                modules=rendered_modules,
                calls=provider.get("http_attempts", 0),
                elapsed=summary.get("elapsed_ms", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The table reports observed execution only. Every executed Q2 arm uses the public provider-free exact-revisit entry; it cannot fall through to ordinary planning, embedding, reranking, answering, or learning. `before_commit_probe` runs against an isolated pre-Q1 database clone after Q1 reaches a terminal state; it is a state baseline, not a concurrent pause. A masked arm hides the single Q1 association across its engine and contextual matcher, while retaining independent base retrieval. The ordinary-cache/full-answer comparison is explicitly deferred; no future-Q2 cache was prewarmed. The source-changed arm mutates only its pilot-owned clone. An absent exact module is not described as a skip unless the raw trace reports it.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic_pilot(
    *,
    source_database: Path,
    case_manifest: Path,
    output_dir: Path,
    env_file: Path,
    deadline_seconds: float = 120.0,
) -> dict[str, object]:
    """Execute a single no-retry Q1/Q2 pilot and persist all observed arms."""

    source_database = source_database.resolve()
    case_manifest = case_manifest.resolve()
    env_file = env_file.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("pilot output directory must be new; never overwrite a case")
    if not source_database.is_file() or not case_manifest.is_file() or not env_file.is_file():
        raise FileNotFoundError("source database, case manifest, and env file must exist")
    if any((source_database.with_name(source_database.name + suffix)).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("pilot source database must be a static snapshot without WAL/SHM sidecars")
    case = PilotCase.from_manifest(case_manifest)
    output_dir.mkdir(parents=True)
    work_dir = output_dir / "work"
    q1_database = work_dir / "q1_learning.sqlite"
    before_commit_database = work_dir / "before_commit_state.sqlite"
    _clone_sqlite(source_database, q1_database)
    _clone_sqlite(source_database, before_commit_database)

    started_at = _utc_now()
    started = monotonic()
    app, config = _open_app(env_file, q1_database, output_dir / "logs" / "q1")
    source_evidence = _source_evidence_stats(app)
    before_receipts, before_edges = _receipt_and_edge_ids(app)
    q1_engine = app.query_engine(config=config)
    # No future-Q2 embedding or other cache state is created before Q1.  The
    # zero-model Q2 preflight below must be attributable solely to any public
    # receipt/edge/manifest produced by this Q1.
    cache_warmup = {
        "status": "not_run",
        "reason": "future_q2_preheat_prohibited_for_this_diagnostic",
    }
    observer = FinalizerObserver()
    original_finalizer = q1_engine.contextual_learning_finalizer
    if not callable(original_finalizer):
        raise RuntimeError("Q1 engine lacks the application contextual finalizer")
    q1_engine.contextual_learning_finalizer = observer.wrap(original_finalizer)
    # Persist the same full-local case envelope before the first Q1 request.
    # If the host stops the worker mid-request, this leaves an auditable
    # ``not_observed`` boundary rather than a directory containing only a
    # partial JSONL.  The normal terminal write below atomically replaces this
    # in-progress envelope; it is not a general journal or a second result.
    in_progress_reason = (
        "Q1 was started but no terminal Q1 record has been observed yet; "
        "dependent Q2 arms are not run."
    )
    in_progress_payload: dict[str, object] = {
        "schema": PILOT_SCHEMA,
        "status": "in_progress_q1",
        "started_at": started_at,
        "finished_at": "not_observed",
        "elapsed_ms": "not_observed",
        "formal_scoring": {
            "status": "not_scored_unapproved_gold",
            "formal_score": False,
            "gold_loaded": False,
            "promotion_prohibited": True,
            "canary_prohibited": True,
        },
        "source_database": {
            "path": str(source_database),
            "sha256": _sha256_file(source_database),
            "sqlite_clone_mode": "read-only sqlite backup into pilot-owned destinations",
            "source_evidence_eligibility": source_evidence,
        },
        "case": {
            "manifest_path": str(case.manifest_path),
            "manifest_sha256": case.manifest_sha256,
            "q1_text": case.q1_text,
            "q2_text": case.q2_text,
            "domain": case.domain,
            "scope_hash": case.scope_hash,
            "normalized_questions_equal": normalize_query_text(case.q1_text) == normalize_query_text(case.q2_text),
            "need_origin": "live_q2_planner_or_private_v17_reconstruction_only",
        },
        "pilot_config": {
            "growth_max_rounds": config.retrieval.growth_max_rounds,
            "growth_persist_only_used": config.retrieval.growth_persist_only_used,
            "association_cue_enabled": config.retrieval.association_cue_enabled,
            "association_cue_fast_path_enabled": config.retrieval.association_cue_fast_path_enabled,
            "contextual_association_enabled": config.retrieval.contextual_association_enabled,
            "contextual_association_shadow": config.retrieval.contextual_association_shadow,
            "contextual_promotion_enabled": config.retrieval.contextual_promotion_enabled,
            "followup_planning_mode": config.retrieval.followup_planning_mode,
            "rerank_atomic_query_limit": config.retrieval.rerank_atomic_query_limit,
            "answer_episode_limit": config.retrieval.answer_episode_limit,
            "rerank_audit_enabled": config.retrieval.rerank_audit_enabled,
            "model_timeout_seconds": config.model.timeout_seconds,
            "model_max_retries": config.model.max_retries,
        },
        "q1": {
            "record": {"status": "not_observed_q1_in_progress"},
            "finalizer_observer": observer.export(started_at=started),
            "learning_status": "not_observed",
            "learning_reason": "not_observed",
            "new_material": {
                "before_receipt_count": len(before_receipts),
                "after_receipt_count": "not_observed",
                "new_receipt_ids": "not_observed",
                "new_association_ids": "not_observed",
                "new_receipts": "not_observed",
            },
            "cache_warmup_before_q1": cache_warmup,
            "answer_evidence_checkpoints": {"status": "not_observed_q1_in_progress"},
        },
        "arms": _terminal_q2_arms(in_progress_reason),
        "no_retry_policy": "Every Q1/Q2 arm is attempted at most once. Failures, timeouts, and no-benefit outcomes remain in this case package.",
        "arm_timing_contract": {
            "before_commit_probe": "post-Q1, pre-Q1-state-clone baseline; never pauses Q1",
            "post_send_immediate_q2": "first Q2 arm after the public Q1 call returns",
            "learning_ready_q2": "separate Q2 arm after the same new receipt reports ready",
        },
        "interruption_recovery": {
            "checkpoint_written_before_q1": True,
            "unobserved_terminal_rule": "If this envelope remains in_progress_q1, Q1's terminal result and all dependent-arm outcomes are not_observed rather than inferred from logs.",
        },
    }
    _write_json(output_dir / FULL_LOCAL_NAME, in_progress_payload)
    _write_case_report(output_dir / CASE_REPORT_NAME, payload=in_progress_payload)
    q1_record = _call_query(
        q1_engine,
        case.q1_text,
        domain=case.domain,
        scope_hash=case.scope_hash,
        deadline_seconds=deadline_seconds,
        contextual_learning=True,
        learning_request_id="v3-real-q1-q2-diagnostic-" + sha256(
            case.q1_text.encode("utf-8")
        ).hexdigest()[:20],
    )
    answer_evidence_checkpoints = _collect_answer_evidence_checkpoints(output_dir)

    arms: dict[str, object] = {}
    new_material = _new_receipt_material(
        app, before_receipts=before_receipts, before_edges=before_edges
    )
    q1_result = q1_record.get("result")
    learning = q1_result.get("contextual_learning") if isinstance(q1_result, Mapping) else {}
    learning = learning if isinstance(learning, Mapping) else {}
    ready_rows = [
        item for item in new_material["new_receipts"]
        if isinstance(item, Mapping)
        and isinstance(item.get("receipt"), Mapping)
        and item["receipt"].get("status") == "ready"
        and item.get("association_id_is_new") is True
    ]
    q1_completed_publicly = (
        q1_record.get("status") == "completed"
        and observer.finalizer_calls == 1
        and bool(ready_rows)
    )
    # Q1 is now terminal and its public finalizer observation, receipt state,
    # and answer-boundary checkpoint are available.  Replace the initial
    # in-progress envelope before any dependent clone/mask/restart work.  If
    # a later diagnostic arm faults or the host stops, this same full-local
    # file retains observed Q1 facts and labels all Q2 outcomes not_observed;
    # it is an atomic pilot recovery checkpoint, not a general journal.
    q1_terminal_payload: dict[str, object] = {
        "schema": PILOT_SCHEMA,
        "status": "q1_terminal_q2_pending",
        "started_at": started_at,
        "finished_at": "not_observed_q2_pending",
        "elapsed_ms": round((monotonic() - started) * 1000.0, 3),
        "formal_scoring": {
            "status": "not_scored_unapproved_gold",
            "formal_score": False,
            "gold_loaded": False,
            "promotion_prohibited": True,
            "canary_prohibited": True,
        },
        "source_database": {
            "path": str(source_database),
            "sha256": _sha256_file(source_database),
            "sqlite_clone_mode": "read-only sqlite backup into pilot-owned destinations",
            "source_evidence_eligibility": source_evidence,
        },
        "case": {
            "manifest_path": str(case.manifest_path),
            "manifest_sha256": case.manifest_sha256,
            "q1_text": case.q1_text,
            "q2_text": case.q2_text,
            "domain": case.domain,
            "scope_hash": case.scope_hash,
            "normalized_questions_equal": normalize_query_text(case.q1_text) == normalize_query_text(case.q2_text),
            "need_origin": "live_q2_planner_or_private_v17_reconstruction_only",
        },
        "pilot_config": {
            "growth_max_rounds": config.retrieval.growth_max_rounds,
            "growth_persist_only_used": config.retrieval.growth_persist_only_used,
            "association_cue_enabled": config.retrieval.association_cue_enabled,
            "association_cue_fast_path_enabled": config.retrieval.association_cue_fast_path_enabled,
            "contextual_association_enabled": config.retrieval.contextual_association_enabled,
            "contextual_association_shadow": config.retrieval.contextual_association_shadow,
            "contextual_promotion_enabled": config.retrieval.contextual_promotion_enabled,
            "followup_planning_mode": config.retrieval.followup_planning_mode,
            "rerank_atomic_query_limit": config.retrieval.rerank_atomic_query_limit,
            "answer_episode_limit": config.retrieval.answer_episode_limit,
            "rerank_audit_enabled": config.retrieval.rerank_audit_enabled,
            "model_timeout_seconds": config.model.timeout_seconds,
            "model_max_retries": config.model.max_retries,
        },
        "q1": {
            "record": q1_record,
            "finalizer_observer": observer.export(started_at=started),
            "learning_status": learning.get("status"),
            "learning_reason": learning.get("reason"),
            "new_material": new_material,
            "cache_warmup_before_q1": cache_warmup,
            "answer_evidence_checkpoints": answer_evidence_checkpoints,
        },
        "arms": _terminal_q2_arms(
            "Q1 reached a terminal state, but this Q2 arm was not observed yet."
        ),
        "no_retry_policy": "Every Q1/Q2 arm is attempted at most once. Failures, timeouts, and no-benefit outcomes remain in this case package.",
        "arm_timing_contract": {
            "before_commit_probe": "post-Q1, pre-Q1-state-clone baseline; never pauses Q1",
            "post_send_immediate_q2": "first Q2 arm after the public Q1 call returns",
            "learning_ready_q2": "separate Q2 arm after the same new receipt reports ready",
        },
        "interruption_recovery": {
            "checkpoint_written_before_q1": True,
            "q1_terminal_checkpoint_written_before_q2": True,
            "unobserved_terminal_rule": "If this envelope remains q1_terminal_q2_pending, Q1 facts are observed and every Q2 arm remains not_observed rather than inferred from clone files or logs.",
        },
    }
    _write_json(output_dir / FULL_LOCAL_NAME, q1_terminal_payload)
    _write_case_report(output_dir / CASE_REPORT_NAME, payload=q1_terminal_payload)
    if q1_completed_publicly:
        q1_edge_id = int(ready_rows[0]["receipt"]["association_id"])
        # This is intentionally scheduled after the public Q1 call has a
        # terminal state.  Its database is a pre-Q1 clone, so it is a genuine
        # no-edge state but not a concurrent mechanism that can hold Q1.
        arms["before_commit_probe"] = _run_q2_arm_checkpointed(
            "before_commit_probe", env_file=env_file, database=before_commit_database,
            output_dir=output_dir, case=case, deadline_seconds=deadline_seconds,
            q1_learning=learning, q1_new_material=new_material,
        )
        immediate_database = work_dir / "post_send_immediate.sqlite"
        _clone_sqlite(q1_database, immediate_database)
        arms["post_send_immediate_q2"] = _run_q2_arm_checkpointed(
            "post_send_immediate_q2", env_file=env_file, database=immediate_database,
            output_dir=output_dir, case=case, deadline_seconds=deadline_seconds,
            q1_learning=learning, q1_new_material=new_material,
        )
        ready_database = work_dir / "learning_ready.sqlite"
        _clone_sqlite(q1_database, ready_database)
        arms["learning_ready_q2"] = _run_q2_arm_checkpointed(
            "learning_ready_q2", env_file=env_file, database=ready_database,
            output_dir=output_dir, case=case, deadline_seconds=deadline_seconds,
            q1_learning=learning, q1_new_material=new_material,
        )
        masked_database = work_dir / "q1_ready_masked.sqlite"
        _clone_sqlite(q1_database, masked_database)
        arms["edge_masked_q2"] = _run_q2_arm_checkpointed(
            "edge_masked_q2", env_file=env_file, database=masked_database,
            output_dir=output_dir, case=case, deadline_seconds=deadline_seconds,
            q1_learning=learning, q1_new_material=new_material,
            hidden_association_id=q1_edge_id,
        )
        # Cache-control/full-answer comparison is deliberately deferred until
        # this no-model evidence-reuse layer has succeeded.  It must not
        # smuggle a future Q2 vector or answer state into this pilot.
        arms["ordinary_embedding_cache_q2"] = {
            "arm": "ordinary_embedding_cache_q2",
            "not_run_reason": "deferred_until_public_exact_revisit_evidence_layer_passes",
            "edge_mask": {"enabled": False},
            "execution_mode": "not_run_no_future_q2_preheat",
            "summary": {"run_status": "not_run", "provider_counts": {}},
        }
        _write_q2_arm_checkpoint(
            output_dir=output_dir,
            name="ordinary_embedding_cache_q2",
            case=case,
            q1_learning=learning,
            q1_new_material=new_material,
            status="q2_arm_not_run",
            arm=arms["ordinary_embedding_cache_q2"],
        )
        restart_database = work_dir / "q1_ready_restart.sqlite"
        _clone_sqlite(q1_database, restart_database)
        arms["edge_restart_q2"] = _run_q2_arm_checkpointed(
            "edge_restart_q2", env_file=env_file, database=restart_database,
            output_dir=output_dir, case=case, deadline_seconds=deadline_seconds,
            q1_learning=learning, q1_new_material=new_material,
            rebuild_before_query=True,
        )
        source_changed_database = work_dir / "q1_ready_source_changed.sqlite"
        _clone_sqlite(q1_database, source_changed_database)
        # Record the arm before its isolated Source mutation.  The mutation
        # has to commit successfully before the public preflight is allowed
        # to observe the expected source-version rejection.
        _write_q2_arm_checkpoint(
            output_dir=output_dir,
            name="source_changed_q2",
            case=case,
            q1_learning=learning,
            q1_new_material=new_material,
            status="q2_arm_pending_public_preflight",
        )
        source_mutation = _invalidate_source_clone(source_changed_database)
        arms["source_changed_q2"] = _run_q2_arm_checkpointed(
            "source_changed_q2", env_file=env_file,
            database=source_changed_database, output_dir=output_dir, case=case,
            deadline_seconds=deadline_seconds,
            q1_learning=learning, q1_new_material=new_material,
            source_mutation=source_mutation,
        )
    else:
        terminal_reason = (
            "Q1 did not complete through the public finalizer with a new ready source-bound association receipt"
        )
        arms.update(_terminal_q2_arms(terminal_reason))
        for name, arm in arms.items():
            if not isinstance(arm, Mapping):
                continue
            _write_q2_arm_checkpoint(
                output_dir=output_dir,
                name=name,
                case=case,
                q1_learning=learning,
                q1_new_material=new_material,
                status="q2_arm_not_run",
                arm=arm,
            )

    payload: dict[str, object] = {
        "schema": PILOT_SCHEMA,
        "status": "diagnostic_complete",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_ms": round((monotonic() - started) * 1000.0, 3),
        "formal_scoring": {
            "status": "not_scored_unapproved_gold",
            "formal_score": False,
            "gold_loaded": False,
            "promotion_prohibited": True,
            "canary_prohibited": True,
        },
        "source_database": {
            "path": str(source_database),
            "sha256": _sha256_file(source_database),
            "sqlite_clone_mode": "read-only sqlite backup into pilot-owned destinations",
            "source_evidence_eligibility": source_evidence,
        },
        "case": {
            "manifest_path": str(case.manifest_path),
            "manifest_sha256": case.manifest_sha256,
            "q1_text": case.q1_text,
            "q2_text": case.q2_text,
            "domain": case.domain,
            "scope_hash": case.scope_hash,
            "normalized_questions_equal": normalize_query_text(case.q1_text) == normalize_query_text(case.q2_text),
            "need_origin": "live_q2_planner_or_private_v17_reconstruction_only",
        },
        "pilot_config": {
            "growth_max_rounds": config.retrieval.growth_max_rounds,
            "growth_persist_only_used": config.retrieval.growth_persist_only_used,
            "association_cue_enabled": config.retrieval.association_cue_enabled,
            "association_cue_fast_path_enabled": config.retrieval.association_cue_fast_path_enabled,
            "contextual_association_enabled": config.retrieval.contextual_association_enabled,
            "contextual_association_shadow": config.retrieval.contextual_association_shadow,
            "contextual_promotion_enabled": config.retrieval.contextual_promotion_enabled,
            "followup_planning_mode": config.retrieval.followup_planning_mode,
            "rerank_atomic_query_limit": config.retrieval.rerank_atomic_query_limit,
            "answer_episode_limit": config.retrieval.answer_episode_limit,
            "rerank_audit_enabled": config.retrieval.rerank_audit_enabled,
            "model_timeout_seconds": config.model.timeout_seconds,
            "model_max_retries": config.model.max_retries,
        },
        "q1": {
            "record": q1_record,
            "finalizer_observer": observer.export(started_at=started),
            "learning_status": learning.get("status"),
            "learning_reason": learning.get("reason"),
            "new_material": new_material,
            "cache_warmup_before_q1": cache_warmup,
            "answer_evidence_checkpoints": answer_evidence_checkpoints,
        },
        "arms": arms,
        "no_retry_policy": "Every Q1/Q2 arm is attempted at most once. Failures, timeouts, and no-benefit outcomes remain in this case package.",
        "arm_timing_contract": {
            "before_commit_probe": "post-Q1, pre-Q1-state-clone baseline; never pauses Q1",
            "post_send_immediate_q2": "first Q2 arm after the public Q1 call returns",
            "learning_ready_q2": "separate Q2 arm after the same new receipt reports ready",
        },
    }
    full_local = output_dir / FULL_LOCAL_NAME
    _write_json(full_local, payload)
    _write_case_report(output_dir / CASE_REPORT_NAME, payload=payload)
    return {
        "output_dir": str(output_dir),
        "full_local": str(full_local),
        "case_report": str(output_dir / CASE_REPORT_NAME),
        "q1_learning_status": learning.get("status"),
        "q1_learning_reason": learning.get("reason"),
        "new_receipt_ids": new_material["new_receipt_ids"],
        "new_association_ids": new_material["new_association_ids"],
        "formal_score": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated V3 Q1/Q2 diagnostic pilot")
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(run_diagnostic_pilot(
        source_database=args.source_database,
        case_manifest=args.case_manifest,
        output_dir=args.output_dir,
        env_file=args.env_file,
        deadline_seconds=args.deadline_seconds,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
