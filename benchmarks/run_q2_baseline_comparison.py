"""Compare N11d exact evidence reuse with ordinary same-question baselines.

This is a companion diagnostic, not a change to the frozen N11d run.  It
uses fresh clones of the original static Source slice, never writes an edge,
receipt, contract, manifest, answer cache, or gold artifact.  The only cache
control is one normal whole-question embedding generated once inside its own
ordinary Q2 request and supplied to that same request's ordinary retrieval.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import traceback
from typing import Mapping

import numpy as np

from benchmarks.run_q1_q2_diagnostic_pilot import (
    CASE_SCHEMA,
    PilotCase,
    ProviderLedger,
    _call_query,
    _clone_sqlite,
    _pilot_config,
    _sha256_file,
    _trace_context,
    _write_json,
)
from memory_demo.app import MemoryApplication
from memory_demo.embeddings import normalize_query_text


BASELINE_SCHEMA = "aevnema.v3.q2_baseline_comparison.v1"
N11D_SCHEMA = "aevnema.v3.q1_q2_diagnostic_pilot.v2"
ARM_NAMES = (
    "pre_edge_exact_reuse",
    "ordinary_search",
    "ordinary_query_embedding_cache",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _ordinary_app(
    *,
    env_file: Path,
    database: Path,
    log_dir: Path,
) -> tuple[MemoryApplication, object]:
    """Open a clone whose Q2 cannot use association or exact-revisit paths."""

    config = _pilot_config(env_file, database, log_dir)
    config.retrieval.contextual_association_enabled = False
    config.retrieval.contextual_association_shadow = False
    config.retrieval.contextual_promotion_enabled = False
    config.retrieval.validate()
    app = MemoryApplication(config)
    app.rebuild_indexes()
    app.rebuild_contextual_indexes()
    return app, config


def _combined_provider_counts(*ledgers: Mapping[str, object]) -> dict[str, int]:
    fields = (
        "logical_batches",
        "http_attempts",
        "sent",
        "succeeded",
        "failed_or_rejected",
        "fallbacks",
    )
    result = {field: 0 for field in fields}
    for ledger in ledgers:
        counts = ledger.get("counts") if isinstance(ledger, Mapping) else None
        if not isinstance(counts, Mapping):
            continue
        for field in fields:
            result[field] += int(counts.get(field, 0) or 0)
    return result


def _cache_seed(
    engine: object,
    question: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create one ordinary request-local vector, retaining no answer state."""

    ledger = ProviderLedger()
    started = perf_counter()
    try:
        with _trace_context(engine.model, ledger):
            vector = engine.embed_query_text(question)
        return vector, {
            "status": "completed",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "vector_origin": "ordinary_q2_whole_question_request_cache",
            "vector_text_sha256": "sha256:" + sha256(
                normalize_query_text(question).encode("utf-8")
            ).hexdigest(),
            "provider": ledger.export(),
        }
    except BaseException as error:
        return np.empty((0,), dtype=np.float32), {
            "status": "failed",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "provider": ledger.export(),
        }


def _answer_evidence_quality(log_dir: Path) -> dict[str, object]:
    """Summarise retained evidence without copying it into the comparison UI."""

    files = sorted(log_dir.glob("*.answer-evidence.jsonl"))
    rows: list[dict[str, object]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                rows.append(loaded)
    selected: list[dict[str, object]] = []
    audit_valid: bool | None = None
    assertions: list[Mapping[str, object]] = []
    stages: list[str] = []
    for row in rows:
        stage = str(row.get("stage", ""))
        if stage:
            stages.append(stage)
        raw_assertions = row.get("prompt_evidence_assertions")
        if isinstance(raw_assertions, Mapping):
            assertions.append(raw_assertions)
        if stage == "answer_input":
            raw_selected = row.get("selected_evidence")
            if isinstance(raw_selected, list):
                for evidence in raw_selected:
                    if not isinstance(evidence, Mapping):
                        continue
                    selected.append(
                        {
                            "episode_id": evidence.get("episode_id"),
                            "source_key": evidence.get("source_key"),
                            "source_evidence_delivery": evidence.get(
                                "source_evidence_delivery"
                            ),
                            "source_evidence_delivery_reason": evidence.get(
                                "source_evidence_delivery_reason"
                            ),
                            "source_excerpt_hash": evidence.get("source_excerpt_hash"),
                            "source_excerpt_characters": len(
                                str(evidence.get("source_excerpt", ""))
                            ),
                        }
                    )
        if stage == "audit_result":
            audit = row.get("audit")
            if isinstance(audit, Mapping) and isinstance(audit.get("valid"), bool):
                audit_valid = bool(audit["valid"])
    return {
        "companion_files": [path.name for path in files],
        "stages": stages,
        "selected_source_evidence": selected,
        "audit_valid": audit_valid,
        "prompt_evidence_assertions": assertions,
    }


def _ordinary_execution_summary(record: Mapping[str, object]) -> dict[str, object]:
    """Report observed Q2 work; never infer it from Q1 timings."""

    result = record.get("result")
    if not isinstance(result, Mapping):
        return {
            "run_status": str(record.get("status", "unknown")),
            "execution_modules": {},
            "evidence_acquisition_ms": None,
            "phase_seconds": {},
            "candidate_episode_ids": [],
            "delivered_episode_ids": [],
            "answer_cache_used": None,
            "embedding_cache": {},
        }
    timings = result.get("timings")
    phases = (
        timings.get("phases_seconds", {}) if isinstance(timings, Mapping) else {}
    )
    phases = phases if isinstance(phases, Mapping) else {}
    evidence_phase_names = (
        "intent_parse",
        "initial_embedding",
        "initial_retrieval",
        "initial_graph_expansion",
        "contextual_association",
        "followup_planning",
        "followup_embedding",
        "followup_retrieval",
        "followup_graph_expansion",
        "source_cohort",
        "evidence_preparation",
        "evidence_rerank",
        "final_selection",
    )
    evidence_acquisition_ms = round(
        sum(float(phases.get(name, 0.0) or 0.0) for name in evidence_phase_names)
        * 1000.0,
        3,
    )
    rerank = result.get("rerank_trace")
    audits = result.get("answer_audits")
    return {
        "run_status": "completed",
        "execution_modules": {
            "planner": "intent_parse" in phases,
            "embedding": bool(result.get("query_embedding_cache", {}).get("miss_count", 0)),
            "retriever": "initial_retrieval" in phases,
            "reranker": bool(rerank.get("enabled")) if isinstance(rerank, Mapping) else False,
            "answer_generation": bool(str(result.get("answer", "")).strip()),
            "answer_audit": bool(audits),
            "contextual_association": False,
            "exact_revisit": False,
        },
        "evidence_acquisition_ms": evidence_acquisition_ms,
        "phase_seconds": dict(phases),
        "candidate_episode_ids": list(result.get("candidate_episode_ids", [])),
        "reranked_episode_ids": list(result.get("reranked_episode_ids", [])),
        "delivered_episode_ids": list(result.get("episode_ids", [])),
        "answer_cache_used": (
            rerank.get("answer_cache_used") if isinstance(rerank, Mapping) else None
        ),
        "embedding_cache": result.get("query_embedding_cache", {}),
    }


def _run_ordinary_arm(
    *,
    name: str,
    output_dir: Path,
    database: Path,
    env_file: Path,
    case: PilotCase,
    deadline_seconds: float,
    cache_control: bool,
) -> dict[str, object]:
    app, config = _ordinary_app(
        env_file=env_file, database=database, log_dir=output_dir / "logs" / name
    )
    engine = app.query_engine(config=config)
    cache_seed: dict[str, object] | None = None
    override: dict[str, np.ndarray] | None = None
    if cache_control:
        vector, cache_seed = _cache_seed(engine, case.q2_text)
        if cache_seed["status"] != "completed":
            return {
                "arm": name,
                "status": "cache_seed_failed",
                "cache_seed": cache_seed,
                "record": None,
                "provider_counts": _combined_provider_counts(cache_seed["provider"]),
                "evidence_quality": _answer_evidence_quality(output_dir / "logs" / name),
            }
        override = {case.q2_text: vector}
    record = _call_query(
        engine,
        case.q2_text,
        domain=case.domain,
        scope_hash=case.scope_hash,
        deadline_seconds=deadline_seconds,
        query_embeddings_override=override,
        contextual_learning=False,
    )
    query_provider = record.get("provider")
    ledgers = [query_provider] if isinstance(query_provider, Mapping) else []
    if cache_seed is not None and isinstance(cache_seed.get("provider"), Mapping):
        ledgers.insert(0, cache_seed["provider"])
    return {
        "arm": name,
        "status": "terminal",
        "execution_mode": (
            "ordinary_search_with_one_request_local_whole_question_embedding"
            if cache_control
            else "ordinary_search_no_edge_no_exact_revisit_no_cache"
        ),
        "cache_seed": cache_seed,
        "record": record,
        "summary": _ordinary_execution_summary(record),
        "provider_counts": _combined_provider_counts(*ledgers),
        "evidence_quality": _answer_evidence_quality(output_dir / "logs" / name),
        "safety_assertions": {
            "contextual_association_enabled": False,
            "contextual_learning_requested": False,
            "answer_cache_used": False,
            "edge_or_manifest_read": False,
        },
    }


def _write_arm(output_dir: Path, name: str, payload: Mapping[str, object]) -> None:
    _write_json(output_dir / f"{name}.full_local.json", payload)


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    arms = payload.get("arms", {})
    assert isinstance(arms, Mapping)
    lines = [
        "# Same-Q2 ordinary baseline comparison",
        "",
        "Diagnostic only: no gold, formal scoring, promotion, or canary.",
        "",
        "| Arm | Status | Delivered | Provider HTTP | Evidence time |",
        "|---|---|---|---:|---:|",
    ]
    for name in ARM_NAMES:
        arm = arms.get(name, {})
        if not isinstance(arm, Mapping):
            continue
        summary = arm.get("summary", {})
        summary = summary if isinstance(summary, Mapping) else {}
        counts = arm.get("provider_counts", {})
        counts = counts if isinstance(counts, Mapping) else {}
        lines.append(
            "| {name} | {status} | {delivered} | {calls} | {elapsed} |".format(
                name=name,
                status=summary.get("run_status", arm.get("status")),
                delivered=summary.get("delivered_episode_ids", "—"),
                calls=counts.get("http_attempts", 0),
                elapsed=summary.get("evidence_acquisition_ms", "—"),
            )
        )
    lines.extend(
        [
            "",
            "Evidence time is the sum of this Q2 request's emitted planning/retrieval/selection phases, excluding answer generation. It is never derived by dividing a Q1 duration by a Q2 duration.",
            "",
            "The pre-edge row is copied from the immutable N11d public pre-commit exact-reuse probe. The two ordinary rows run on new, edge-free static-Source clones with contextual association disabled. The cache control stores one normal whole-question embedding only; it supplies no Q1 plan, edge, manifest, answer, selected IDs, or gold data.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_baseline_comparison(
    *,
    source_database: Path,
    case_manifest: Path,
    n11d_full_local: Path,
    output_dir: Path,
    env_file: Path,
    deadline_seconds: float = 120.0,
) -> dict[str, object]:
    """Run two new ordinary controls and preserve N11d's pre-edge observation."""

    source_database = source_database.resolve()
    case_manifest = case_manifest.resolve()
    n11d_full_local = n11d_full_local.resolve()
    output_dir = output_dir.resolve()
    env_file = env_file.resolve()
    if output_dir.exists():
        raise FileExistsError("baseline output directory must be new")
    if not all(path.is_file() for path in (source_database, case_manifest, n11d_full_local, env_file)):
        raise FileNotFoundError("source database, case manifest, N11d trace, and env file must exist")
    if any(source_database.with_name(source_database.name + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("source database must be a static snapshot without WAL/SHM sidecars")

    case = PilotCase.from_manifest(case_manifest)
    n11d = _read_json(n11d_full_local)
    if n11d.get("schema") != N11D_SCHEMA or n11d.get("status") != "diagnostic_complete":
        raise ValueError("N11d input must be its preserved complete diagnostic trace")
    source_info = n11d.get("source_database")
    n11d_case = n11d.get("case")
    if not isinstance(source_info, Mapping) or not isinstance(n11d_case, Mapping):
        raise ValueError("N11d trace lacks source/case binding")
    if str(source_info.get("sha256", "")).casefold() != _sha256_file(source_database).casefold():
        raise ValueError("N11d and ordinary controls must use the same static Source slice")
    if normalize_query_text(str(n11d_case.get("q2_text", ""))) != normalize_query_text(case.q2_text):
        raise ValueError("N11d and ordinary controls must use the same Q2")
    n11d_arms = n11d.get("arms")
    if not isinstance(n11d_arms, Mapping) or not isinstance(n11d_arms.get("before_commit_probe"), Mapping):
        raise ValueError("N11d trace lacks the observed pre-edge control")

    output_dir.mkdir(parents=True)
    work_dir = output_dir / "work"
    started = _utc_now()
    pre_edge = dict(n11d_arms["before_commit_probe"])
    pre_edge_payload: dict[str, object] = {
        "schema": BASELINE_SCHEMA,
        "status": "observed_in_preserved_n11d_trace",
        "source_trace": str(n11d_full_local),
        "source_trace_sha256": _sha256_file(n11d_full_local),
        "arm": pre_edge,
        "provider_counts": (
            pre_edge.get("summary", {}).get("provider_counts", {})
            if isinstance(pre_edge.get("summary"), Mapping)
            else {}
        ),
        "summary": {
            "run_status": "exact_revisit_miss",
            "delivered_episode_ids": [],
            "evidence_acquisition_ms": None,
            "execution_modules": {},
        },
        "evidence_quality": {"status": "not_delivered_pre_edge_exact_miss"},
    }
    _write_arm(output_dir, "pre_edge_exact_reuse", pre_edge_payload)

    ordinary_database = work_dir / "ordinary_search.sqlite"
    cache_database = work_dir / "ordinary_query_embedding_cache.sqlite"
    _clone_sqlite(source_database, ordinary_database)
    _clone_sqlite(source_database, cache_database)
    _write_arm(
        output_dir,
        "ordinary_search",
        {
            "schema": BASELINE_SCHEMA,
            "status": "ordinary_q2_pending",
            "observation": "not_observed_yet",
            "execution_mode": "ordinary_search_no_edge_no_exact_revisit_no_cache",
        },
    )
    ordinary = _run_ordinary_arm(
        name="ordinary_search",
        output_dir=output_dir,
        database=ordinary_database,
        env_file=env_file,
        case=case,
        deadline_seconds=deadline_seconds,
        cache_control=False,
    )
    _write_arm(output_dir, "ordinary_search", ordinary)
    _write_arm(
        output_dir,
        "ordinary_query_embedding_cache",
        {
            "schema": BASELINE_SCHEMA,
            "status": "ordinary_q2_pending",
            "observation": "not_observed_yet",
            "execution_mode": "ordinary_search_with_one_request_local_whole_question_embedding",
        },
    )
    cached = _run_ordinary_arm(
        name="ordinary_query_embedding_cache",
        output_dir=output_dir,
        database=cache_database,
        env_file=env_file,
        case=case,
        deadline_seconds=deadline_seconds,
        cache_control=True,
    )
    _write_arm(output_dir, "ordinary_query_embedding_cache", cached)

    payload: dict[str, object] = {
        "schema": BASELINE_SCHEMA,
        "status": "diagnostic_complete",
        "started_at": started,
        "finished_at": _utc_now(),
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
        },
        "case": {
            "q2_text": case.q2_text,
            "domain": case.domain,
            "scope_hash": case.scope_hash,
        },
        "n11d_trace": {
            "path": str(n11d_full_local),
            "sha256": _sha256_file(n11d_full_local),
        },
        "arms": {
            "pre_edge_exact_reuse": pre_edge_payload,
            "ordinary_search": ordinary,
            "ordinary_query_embedding_cache": cached,
        },
    }
    _write_json(output_dir / "q2_baseline_comparison.full_local.json", payload)
    _write_report(output_dir / "q2_baseline_comparison.report.md", payload)
    return {
        "output_dir": str(output_dir),
        "full_local": str(output_dir / "q2_baseline_comparison.full_local.json"),
        "report": str(output_dir / "q2_baseline_comparison.report.md"),
        "ordinary_status": ordinary.get("status"),
        "cached_status": cached.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N11d ordinary-Q2 baseline comparison")
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--n11d-full-local", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run_baseline_comparison(
                source_database=args.source_database,
                case_manifest=args.case_manifest,
                n11d_full_local=args.n11d_full_local,
                output_dir=args.output_dir,
                env_file=args.env_file,
                deadline_seconds=args.deadline_seconds,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
