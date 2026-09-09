"""Run a Q2-only, public evidence-delivery comparison from a frozen Q1 state.

This runner deliberately has no Q1 path.  It accepts only an already durable
snapshot plus an explicit ready edge, clones that snapshot per arm, and calls
the public QueryEngine.query(..., stop_after="evidence") endpoint.  Therefore
it cannot manufacture a receipt, edge, manifest, answer cache, or Q2 learning
event while preparing an edge-available/masked evidence comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from time import monotonic, perf_counter
from typing import Mapping

from benchmarks.run_q1_q2_diagnostic_pilot import (
    ProviderLedger,
    _clone_sqlite,
    _open_app,
    _sha256_file,
    _trace_context,
    _utc_now as _pilot_utc_now,
    _write_json,
)
from benchmarks.run_same_chapter_learning_matrix import (
    FULL_LOCAL_NAME,
    MATRIX_SCHEMA,
    _q2_summary,
    _run_q2_arm,
    load_frozen_matrix_case,
)


SCHEMA = "aevnema.v3_2.frozen_evidence_matrix.v1"
REPORT_NAME = "frozen_evidence_matrix.report.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_shared_plan(
    *,
    variant_id: str,
    question: str,
    source_snapshot: Path,
    output_dir: Path,
    env_file: Path,
    treatment_edge_id: int,
) -> dict[str, object]:
    """Freeze one ordinary Q2 input before either edge condition runs.

    The public ``build_query_plan`` entry point owns any planner, embedding,
    and rerank work.  The resulting plan is passed unchanged to both cloned
    Q2 conditions, so preparation is not accidentally charged once per edge
    condition and a masked arm cannot observe a different stochastic plan.
    """

    database = output_dir / "work" / f"prepare__{variant_id}.sqlite"
    log_dir = output_dir / "logs" / "preparation" / variant_id
    plan_path = output_dir / "preparation" / f"{variant_id}.frozen_query_plan.full_local.json"
    _clone_sqlite(source_snapshot, database)
    ledger = ProviderLedger()
    started_at = _pilot_utc_now()
    started = perf_counter()
    try:
        app, config = _open_app(env_file, database, log_dir)
        engine = app.query_engine(config=config)
        with _trace_context(engine.model, ledger):
            plan, query_vector_bundle = engine.build_frozen_query_input(question)
        if not isinstance(plan, dict):
            raise TypeError("build_query_plan did not return an object")
        cue_ids = {
            int(value)
            for value in plan.get("association_cue_association_ids", [])
            if str(value).strip()
        }
        if int(treatment_edge_id) in cue_ids:
            raise ValueError(
                "frozen base plan contains the treatment edge through association cues"
            )
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(plan_path, plan)
        return {
            "status": "completed",
            "started_at": started_at,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "database": str(database),
            "plan_path": str(plan_path),
            "plan_sha256": _sha256_file(plan_path),
            "plan_id": plan.get("plan_id", "not_observed"),
            "vector_material": {
                "initial_query_count": len(plan.get("initial_queries", [])),
                "initial_embedding_rows": len(
                    plan.get("initial_query_embeddings_float32", [])
                ),
                "followup_query_count": len(plan.get("followup_queries", [])),
                "followup_embedding_rows": len(
                    plan.get("followup_query_embeddings_float32", [])
                ),
                "mode": "public_frozen_query_plan_no_custom_vector_override",
                "bundle_metadata": query_vector_bundle.metadata(),
                "restore_contract": "public QueryEngine.restore_frozen_query_vectors",
            },
            "provider": ledger.export(),
            # This immutable in-process object is intentionally kept out of
            # JSON.  The persisted plan contains its original float32 rows,
            # and a resumed harness must use the named public restore method
            # instead of fabricating a bundle or re-embedding the Q2 text.
            "_runtime_query_vector_bundle": query_vector_bundle,
        }
    except BaseException as error:
        return {
            "status": "failed",
            "started_at": started_at,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "error": {"type": type(error).__name__, "message": str(error)},
            "provider": ledger.export(),
        }


def _restore_shared_plan(
    *,
    variant_id: str,
    question: str,
    source_snapshot: Path,
    output_dir: Path,
    env_file: Path,
    treatment_edge_id: int,
    prepared_plan_dir: Path,
) -> dict[str, object]:
    """Load an earlier public plan and restore its checked vectors locally.

    This is intentionally a public recovery path, not an answer cache or a
    manually assembled matcher input.  It is useful when a prior diagnostic
    captured a complete frozen plan but a later runtime-binding defect kept
    the contextual lane from receiving its logical requirement vectors.
    """

    plan_path = prepared_plan_dir / f"{variant_id}.frozen_query_plan.full_local.json"
    database = output_dir / "work" / f"restore__{variant_id}.sqlite"
    log_dir = output_dir / "logs" / "restoration" / variant_id
    ledger = ProviderLedger()
    started_at = _pilot_utc_now()
    started = perf_counter()
    try:
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise TypeError("stored frozen plan is not an object")
        _clone_sqlite(source_snapshot, database)
        app, config = _open_app(env_file, database, log_dir)
        engine = app.query_engine(config=config)
        with _trace_context(engine.model, ledger):
            query_vector_bundle = engine.restore_frozen_query_vectors(question, plan)
        cue_ids = {
            int(value)
            for value in plan.get("association_cue_association_ids", [])
            if str(value).strip()
        }
        if int(treatment_edge_id) in cue_ids:
            raise ValueError(
                "restored base plan contains the treatment edge through association cues"
            )
        return {
            "status": "completed",
            "started_at": started_at,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "database": str(database),
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": _sha256_file(plan_path),
            "plan_id": plan.get("plan_id", "not_observed"),
            "source_plan_origin": "pre_existing_public_frozen_query_plan",
            "vector_material": {
                "mode": "public_restore_frozen_query_vectors_no_model_call",
                "bundle_metadata": query_vector_bundle.metadata(),
                "restore_contract": "public QueryEngine.restore_frozen_query_vectors",
            },
            "provider": ledger.export(),
            "_runtime_query_vector_bundle": query_vector_bundle,
        }
    except BaseException as error:
        return {
            "status": "failed",
            "started_at": started_at,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "error": {"type": type(error).__name__, "message": str(error)},
            "provider": ledger.export(),
        }


def _snapshot_edge_receipt(snapshot: Path, association_id: int) -> dict[str, object]:
    """Read, but never alter, the exact edge/receipt/manifest triplet."""

    uri = f"{snapshot.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    try:
        edge = connection.execute(
            "SELECT id, from_type, from_id, to_type, to_id, relation_key, lifecycle_state "
            "FROM association WHERE id = ?",
            (association_id,),
        ).fetchone()
        receipt = connection.execute(
            "SELECT id, status, verification_status, durable_artifact_hash "
            "FROM contextual_creation_receipt "
            "WHERE association_id = ? AND status = 'ready'",
            (association_id,),
        ).fetchone()
        manifest = connection.execute(
            "SELECT id, state, creation_receipt_id, manifest_fingerprint "
            "FROM contextual_revisit_runtime_manifest "
            "WHERE association_id = ? AND state = 'ready'",
            (association_id,),
        ).fetchone()
    finally:
        connection.close()
    if edge is None:
        raise ValueError(f"association {association_id} is absent from frozen snapshot")
    if receipt is None:
        raise ValueError(f"association {association_id} lacks a ready creation receipt")
    if manifest is None:
        raise ValueError(f"association {association_id} lacks a ready runtime manifest")
    if int(manifest["creation_receipt_id"]) != int(receipt["id"]):
        raise ValueError("runtime manifest does not bind the ready receipt")
    return {
        "association": dict(edge),
        "receipt": dict(receipt),
        "runtime_manifest": dict(manifest),
    }


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    arms = payload.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    lines = [
        "# Frozen Q2 evidence-delivery matrix",
        "",
        "This is a non-scoring Q2-only observation. The Q1 state was read from "
        "a frozen, pre-existing snapshot; this run made no Q1, answer generation, "
        "answer audit, Q2-learning, association-growth, or edge-use write. "
        "A live-preparation run may still report an evidence-coverage audit as "
        "part of retrieval selection; it is not an answer audit.",
        "",
        "| Variant | Condition | Run | Evidence | Edge state | Delivered episodes | HTTP in arm | Evidence ms |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for key, arm in arms.items():
        if not isinstance(arm, Mapping):
            continue
        summary = arm.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        stages = summary.get("provider_stages")
        stages = stages if isinstance(stages, Mapping) else {}
        attempts = sum(
            int(value.get("http_attempts", 0) or 0)
            for value in stages.values()
            if isinstance(value, Mapping)
        )
        lines.append(
            "| {variant} | {condition} | {status} | {evidence} | {edge} | {delivered} | {http} | {elapsed} |".format(
                variant=arm.get("variant_id", key),
                condition=arm.get("condition", "not_observed"),
                status=summary.get("run_status", "not_observed"),
                evidence=summary.get("evidence_state", "not_observed"),
                edge=summary.get("this_run_edge_participated", "not_observed"),
                delivered=summary.get("delivered_episode_ids", []),
                http=attempts,
                elapsed=summary.get("evidence_acquisition_ms", "not_observed"),
            )
        )
    lines.extend(
        [
            "",
            "A `source_bound` delivery state only identifies a provenance-bound "
            "Source excerpt. It is not a formal correctness or generalization score; "
            "gold was not loaded.",
            "",
            "Preparation records, when present, are separate from both conditions. "
            "A `frozen_query_plan` is shared only when its plan hash is identical "
            "in both arm records; otherwise the result is a live-input observation, "
            "not a frozen-input comparison.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_frozen_evidence_matrix(
    *,
    learned_snapshot: Path,
    matrix_manifest: Path,
    output_dir: Path,
    env_file: Path,
    treatment_edge_id: int,
    deadline_seconds: float = 120.0,
    freeze_inputs: bool = True,
    prepared_plan_dir: Path | None = None,
) -> dict[str, object]:
    """Run each frozen Q2 treatment once from independently cloned snapshots."""

    learned_snapshot = learned_snapshot.resolve()
    matrix_manifest = matrix_manifest.resolve()
    output_dir = output_dir.resolve()
    env_file = env_file.resolve()
    prepared_plan_dir = prepared_plan_dir.resolve() if prepared_plan_dir else None
    if output_dir.exists():
        raise FileExistsError("frozen evidence matrix output directory must be new")
    if not all(path.is_file() for path in (learned_snapshot, matrix_manifest, env_file)):
        raise FileNotFoundError("snapshot, matrix manifest, and env file are required")
    if prepared_plan_dir is not None and not prepared_plan_dir.is_dir():
        raise FileNotFoundError("prepared plan directory is required when supplied")
    if prepared_plan_dir is not None:
        origin_trace = prepared_plan_dir.parent / FULL_LOCAL_NAME
        if not origin_trace.is_file():
            raise FileNotFoundError(
                "prepared plan directory must retain its originating full-local trace"
            )
        origin = json.loads(origin_trace.read_text(encoding="utf-8"))
        origin_snapshot = (
            origin.get("q1", {}).get("snapshot", {})
            if isinstance(origin, Mapping)
            else {}
        )
        if not isinstance(origin_snapshot, Mapping) or origin_snapshot.get("sha256") != _sha256_file(learned_snapshot):
            raise ValueError("prepared plans are not bound to this frozen learned snapshot")
    if any(
        learned_snapshot.with_name(learned_snapshot.name + suffix).exists()
        for suffix in ("-wal", "-shm")
    ):
        raise ValueError("frozen learned snapshot must not have WAL/SHM sidecars")
    case = load_frozen_matrix_case(matrix_manifest)
    edge_binding = _snapshot_edge_receipt(learned_snapshot, int(treatment_edge_id))
    output_dir.mkdir(parents=True)
    started = monotonic()
    started_at = _utc_now()
    snapshot = {
        "path": str(learned_snapshot),
        "sha256": _sha256_file(learned_snapshot),
        "origin": "pre_existing_legal_q1_snapshot_read_only",
        "edge_binding": edge_binding,
    }
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": started_at,
        "finished_at": "not_observed",
        "formal_scoring": {"status": "not_scored_unapproved_gold", "gold_loaded": False},
        "q1": {"status": "pre_existing_not_reissued", "snapshot": snapshot},
        "case": {
            "manifest_path": str(matrix_manifest),
            "manifest_sha256": _sha256_file(matrix_manifest),
            "q2_variants": [
                {
                    "id": item.identifier,
                    "kind": item.kind,
                    "text": item.text,
                    "context_mode": item.context_mode,
                    "reference_basis": item.reference_basis,
                }
                for item in case.variants
            ],
            "domain": case.domain,
            "scope_hash": case.scope_hash,
        },
        "execution_profile": (
            "public_query_stop_after_evidence_shared_frozen_input"
            if freeze_inputs
            else "public_query_stop_after_evidence_live_input"
        ),
        "input_preparation": {
            "mode": (
                "shared_public_frozen_query_plan_per_variant"
                if freeze_inputs
                else "live_input_per_condition"
            ),
            "records": {},
            "prepared_plan_dir": str(prepared_plan_dir) if prepared_plan_dir else None,
        },
        "arms": {},
        "no_retry_policy": "Each condition is attempted once; terminal outcomes remain.",
    }
    _write_json(output_dir / FULL_LOCAL_NAME, payload)
    _write_report(output_dir / REPORT_NAME, payload)
    arms: dict[str, object] = {}
    for variant in case.variants:
        prepared_plan: dict[str, object] | None = None
        if freeze_inputs:
            preparation = (
                _restore_shared_plan(
                    variant_id=variant.identifier,
                    question=variant.text,
                    source_snapshot=learned_snapshot,
                    output_dir=output_dir,
                    env_file=env_file,
                    treatment_edge_id=int(treatment_edge_id),
                    prepared_plan_dir=prepared_plan_dir,
                )
                if prepared_plan_dir is not None
                else _prepare_shared_plan(
                    variant_id=variant.identifier,
                    question=variant.text,
                    source_snapshot=learned_snapshot,
                    output_dir=output_dir,
                    env_file=env_file,
                    treatment_edge_id=int(treatment_edge_id),
                )
            )
            runtime_query_vector_bundle = preparation.pop(
                "_runtime_query_vector_bundle", None
            )
            preparation_records = payload["input_preparation"]["records"]
            assert isinstance(preparation_records, dict)
            preparation_records[variant.identifier] = preparation
            _write_json(output_dir / FULL_LOCAL_NAME, payload)
            _write_report(output_dir / REPORT_NAME, payload)
            if preparation.get("status") != "completed":
                for condition in ("edge_available", "this_run_edge_masked"):
                    arms[f"{variant.identifier}__{condition}"] = {
                        "variant_id": variant.identifier,
                        "variant_kind": variant.kind,
                        "condition": condition,
                        "not_run_reason": "frozen_input_preparation_failed",
                        "preparation": preparation,
                        "summary": {
                            "run_status": "not_run",
                            "reason": "frozen_input_preparation_failed",
                        },
                    }
                payload["arms"] = arms
                payload["elapsed_ms"] = round((monotonic() - started) * 1000.0, 3)
                _write_json(output_dir / FULL_LOCAL_NAME, payload)
                _write_report(output_dir / REPORT_NAME, payload)
                continue
            plan_path = Path(str(preparation["plan_path"]))
            prepared = json.loads(plan_path.read_text(encoding="utf-8"))
            if not isinstance(prepared, dict):
                raise TypeError("stored frozen plan is not an object")
            prepared_plan = prepared
        for condition in ("edge_available", "this_run_edge_masked"):
            arm = _run_q2_arm(
                variant=variant,
                condition=condition,
                source_snapshot=learned_snapshot,
                output_dir=output_dir,
                env_file=env_file,
                domain=case.domain,
                scope_hash=case.scope_hash,
                deadline_seconds=deadline_seconds,
                this_run_edge_id=int(treatment_edge_id),
                snapshot=snapshot,
                evidence_only=True,
                frozen_plan=prepared_plan,
                query_vector_bundle=runtime_query_vector_bundle,
            )
            if freeze_inputs:
                arm["shared_input_preparation"] = {
                    "variant_id": variant.identifier,
                    "plan_id": prepared_plan.get("plan_id", "not_observed")
                    if prepared_plan is not None
                    else "not_observed",
                    "plan_sha256": preparation.get("plan_sha256"),
                    "prepared_once_for_both_conditions": True,
                    "vector_bundle_supplied": runtime_query_vector_bundle is not None,
                }
            arms[f"{variant.identifier}__{condition}"] = arm
            payload["arms"] = arms
            payload["elapsed_ms"] = round((monotonic() - started) * 1000.0, 3)
            _write_json(output_dir / FULL_LOCAL_NAME, payload)
            _write_report(output_dir / REPORT_NAME, payload)
    payload.update(
        {
            "status": "completed",
            "finished_at": _utc_now(),
            "elapsed_ms": round((monotonic() - started) * 1000.0, 3),
            "snapshot_sha256_after_all_arms": _sha256_file(learned_snapshot),
        }
    )
    payload["snapshot_unchanged_after_all_arms"] = (
        payload["snapshot_sha256_after_all_arms"] == snapshot["sha256"]
    )
    _write_json(output_dir / FULL_LOCAL_NAME, payload)
    _write_report(output_dir / REPORT_NAME, payload)
    return {
        "output_dir": str(output_dir),
        "full_local": str(output_dir / FULL_LOCAL_NAME),
        "report": str(output_dir / REPORT_NAME),
        "snapshot_unchanged_after_all_arms": payload[
            "snapshot_unchanged_after_all_arms"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a public Q2 evidence-only matrix from a frozen Q1 snapshot"
    )
    parser.add_argument("--learned-snapshot", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--treatment-edge-id", type=int, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    parser.add_argument(
        "--live-input",
        action="store_true",
        help="Do not prepare one shared frozen query plan per Q2 variant",
    )
    parser.add_argument(
        "--prepared-plan-dir",
        type=Path,
        help="Restore one saved public frozen plan per variant without model calls",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_frozen_evidence_matrix(
                learned_snapshot=args.learned_snapshot,
                matrix_manifest=args.matrix_manifest,
                output_dir=args.output_dir,
                env_file=args.env_file,
                treatment_edge_id=args.treatment_edge_id,
                deadline_seconds=args.deadline_seconds,
                freeze_inputs=not args.live_input,
                prepared_plan_dir=args.prepared_plan_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
