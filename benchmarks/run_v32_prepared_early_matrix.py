"""Run W06's prepared-early shadow comparison through public Q2 entry.

This is a local, non-scoring diagnostic over an already frozen N12b snapshot.
It restores each stored Q2 vector bundle once per independent SQLite clone and
uses only ``QueryEngine.query(..., stop_after='evidence')``.  The runner never
issues Q1, creates a receipt/edge/manifest, invokes an answer/audit model, or
changes the source snapshot.  Each arm is written immediately and is never
retried by this command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any, Mapping

from benchmarks.run_q1_q2_diagnostic_pilot import (
    ProviderLedger,
    _clone_sqlite,
    _mask_engine_edge,
    _open_app,
    _sha256_file,
    _trace_context,
    _write_json,
)
from benchmarks.run_same_chapter_learning_matrix import (
    _provider_partition,
    load_frozen_matrix_case,
)


SCHEMA = "aevnema.v3_2.prepared_early_shadow_matrix.v1"
CONDITIONS: tuple[tuple[str, bool, bool], ...] = (
    # name, prepared-early enabled, hide only the learned contextual edge
    ("P0_no_edge_ordinary", False, True),
    ("P1_late_residual", False, False),
    ("P2_prepared_early_shadow", True, False),
    ("P2M_prepared_early_shadow_masked", True, True),
)
NONEXACT_KINDS = frozenset(
    {"paraphrase", "partial_clue", "near_neighbor_counterexample"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _compact_result(value: object) -> dict[str, Any]:
    """Extract the observed W06 contract without suppressing raw full-local."""

    result = _as_mapping(value)
    contextual = _as_mapping(result.get("contextual_association"))
    prepared = _as_mapping(contextual.get("prepared_early"))
    evidence = _as_mapping(result.get("evidence_result"))
    timings = _as_mapping(result.get("timings"))
    prepared_modules = _as_mapping(prepared.get("executed_modules"))
    return {
        "prepared_early": {
            "stage": prepared.get("stage", "not_observed"),
            "enabled": prepared.get("enabled", False),
            "shadow": prepared.get("shadow", False),
            "need_origin": prepared.get("need_origin", "not_observed"),
            "vector_origin": prepared.get("vector_origin", "not_observed"),
            "key_scoring_mode": prepared.get("key_scoring_mode", "not_observed"),
            "context_need_vector_relationship": prepared.get(
                "context_need_vector_relationship", "not_observed"
            ),
            "context_need_vector_degenerate": prepared.get(
                "context_need_vector_degenerate", "not_observed"
            ),
            "independent_anchor_episode_ids": prepared.get(
                "independent_anchor_episode_ids", []
            ),
            "proposal_generated": prepared.get("proposal_generated", False),
            "source_validated": prepared.get("source_validated", False),
            "source_metadata_validated": prepared.get(
                "source_metadata_validated", False
            ),
            "source_binding_validated": prepared.get(
                "source_binding_validated", False
            ),
            "source_binding_status": prepared.get(
                "source_binding_status", "not_observed"
            ),
            "current_requirement_supported": prepared.get(
                "current_requirement_supported", False
            ),
            "current_requirement_support_status": prepared.get(
                "current_requirement_support_status", "not_observed"
            ),
            "early_stop_allowed": prepared.get("early_stop_allowed", False),
            "candidate_pool_mutated": prepared.get("candidate_pool_mutated", None),
            "candidate_pool_requested": prepared.get(
                "candidate_pool_requested", None
            ),
            "candidate_pool_validated_episode_ids": prepared.get(
                "candidate_pool_validated_episode_ids", []
            ),
            "candidate_pool_eligible_episode_ids": prepared.get(
                "candidate_pool_eligible_episode_ids", []
            ),
            "candidate_pool_injected_episode_ids": prepared.get(
                "candidate_pool_injected_episode_ids", []
            ),
            "candidate_pool_injection_reason": prepared.get(
                "candidate_pool_injection_reason", "not_observed"
            ),
            "candidate_pool_strict_source_status": prepared.get(
                "candidate_pool_strict_source_status", []
            ),
            "candidate_pool_dependency_partition": prepared.get(
                "candidate_pool_dependency_partition", {}
            ),
            "rerank_input_reuse": prepared.get("rerank_input_reuse", {}),
            "ordinary_retrieval_continues": prepared.get(
                "ordinary_retrieval_continues", None
            ),
            "retention_reason": prepared.get("retention_reason", "not_observed"),
            "reason": prepared.get("reason", "not_observed"),
            "pre_target_candidate_order": prepared.get(
                "pre_target_candidate_order", []
            ),
            "accepted_candidate_order": prepared.get(
                "accepted_candidate_order", []
            ),
            "target_gate": prepared.get("target_gate", {}),
            "executed_modules": dict(prepared_modules),
        },
        "late_residual": {
            "reason": contextual.get("reason", "not_observed"),
            "attached_edges": contextual.get("attached_edges", []),
            "attached_episode_ids": contextual.get("attached_episode_ids", []),
            "base_endpoint_manifest": contextual.get(
                "base_endpoint_manifest", []
            ),
            "prepared_early_contextual_endpoint_manifest": contextual.get(
                "prepared_early_contextual_endpoint_manifest", []
            ),
            "selected_contextual_contribution_ids": contextual.get(
                "selected_contextual_contribution_ids", []
            ),
        },
        "evidence_delivery": {
            "state": evidence.get("evidence_state", "not_observed"),
            "selected_refs": evidence.get("selected_refs", []),
            "materialized_source_refs": evidence.get(
                "materialized_source_refs", []
            ),
            "participation": evidence.get("participation", {}),
            "executed_modules": evidence.get("executed_modules", {}),
            "actual_skipped_modules": evidence.get("actual_skipped_modules", []),
        },
        "phase_seconds": timings.get("phases_seconds", {}),
        "query_vector_bundle": result.get("query_vector_bundle"),
        "query_plan_id": result.get("query_plan_id"),
    }


def _run_arm(
    *,
    snapshot: Path,
    output_dir: Path,
    variant_id: str,
    question: str,
    plan_path: Path,
    domain: str,
    evaluation_as_of: str,
    condition: str,
    prepared_early_enabled: bool,
    edge_masked: bool,
    env_file: Path,
    treatment_edge_id: int,
    deadline_seconds: float,
    scoring_mode: str = "cn",
    candidate_pool_enabled: bool = False,
) -> dict[str, Any]:
    database = output_dir / "work" / f"{variant_id}__{condition}.sqlite"
    _clone_sqlite(snapshot, database)
    app, config = _open_app(
        env_file, database, output_dir / "logs" / variant_id / condition
    )
    # This is a post-plan runtime diagnostic switch, deliberately outside the
    # frozen plan configuration.  It cannot change its text/vectors/seeds or
    # make the proposal a delivery decision.
    config.retrieval.contextual_prepared_early_enabled = prepared_early_enabled
    config.retrieval.contextual_prepared_early_shadow = True
    config.retrieval.contextual_prepared_early_scoring_mode = scoring_mode
    config.retrieval.contextual_prepared_early_candidate_pool_enabled = (
        candidate_pool_enabled
    )
    config.retrieval.validate()
    engine = app.query_engine(config=config)
    if edge_masked:
        _mask_engine_edge(engine, treatment_edge_id)
    plan = _read(plan_path)
    restore = getattr(engine, "restore_frozen_query_vectors", None)
    if not callable(restore):
        raise TypeError("public engine lacks restore_frozen_query_vectors")
    bundle = restore(question, plan)
    ledger = ProviderLedger()
    started_at = _utc_now()
    started = perf_counter()
    try:
        with _trace_context(engine.model, ledger):
            result = engine.query(
                question,
                generate_answer=False,
                stop_after="evidence",
                frozen_plan=plan,
                query_vector_bundle=bundle,
                strict_vector_bundle=True,
                contextual_domain=domain,
                contextual_evaluation_as_of=evaluation_as_of,
                contextual_learning=False,
                deadline_seconds=deadline_seconds,
            )
        provider = ledger.export()
        counts = _as_mapping(provider.get("counts"))
        if int(counts.get("http_attempts", 0) or 0) != 0:
            raise RuntimeError("prepared-early evidence replay issued a provider request")
        return {
            "status": "completed",
            "started_at": started_at,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "condition": condition,
            "database_clone": str(database),
            "plan_path": str(plan_path),
            "plan_sha256": _sha256_file(plan_path),
            "frozen_input": {
                "query_plan_id": plan.get("plan_id"),
                "vector_bundle_restored_by_public_api": True,
                "evaluation_as_of": evaluation_as_of,
            },
            "edge_mask": {
                "enabled": edge_masked,
                "association_id": treatment_edge_id if edge_masked else None,
                "scope": "memory overlay only; base retrieval remains available",
            },
            "prepared_early_runtime": {
                "enabled": prepared_early_enabled,
                "shadow": True,
                "key_scoring_mode": scoring_mode,
                "candidate_pool_enabled": candidate_pool_enabled,
                "early_stop_permitted": False,
            },
            "result": result,
            "compact": _compact_result(result),
            "provider": provider,
            "provider_stage_partition": _provider_partition({"provider": provider}),
            "writes": {
                "q1_created": False,
                "receipt_created": False,
                "edge_created": False,
                "manifest_created": False,
                "answer_generated": False,
                "answer_audited": False,
                "q2_learning": False,
                "edge_use_written": False,
            },
        }
    except BaseException as error:
        provider = ledger.export()
        return {
            "status": "failed",
            "started_at": started_at,
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "condition": condition,
            "database_clone": str(database),
            "plan_path": str(plan_path),
            "prepared_early_runtime": {
                "enabled": prepared_early_enabled,
                "shadow": True,
                "key_scoring_mode": scoring_mode,
                "candidate_pool_enabled": candidate_pool_enabled,
                "early_stop_permitted": False,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "provider": provider,
            "provider_stage_partition": _provider_partition({"provider": provider}),
        }


def _report(payload: Mapping[str, Any]) -> str:
    arms = _as_mapping(payload.get("arms"))
    lines = [
        "# W06 prepared-early shadow matrix",
        "",
        "This local, non-scoring run uses N12b's frozen snapshot and stored per-Q2 vectors. P2 is a shadow observation only: it cannot add a candidate, claim current-requirement support, stop ordinary retrieval, or write learning/usage state. No answer, audit, Q1, receipt, edge, manifest, or provider HTTP is authorized here.",
        "",
        "| Q2 | Arm | Prepared stage | Proposal | Source-valid candidate | Retained as | Late branch | Delivered Episodes | HTTP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for key, arm_value in arms.items():
        arm = _as_mapping(arm_value)
        compact = _as_mapping(arm.get("compact"))
        prepared = _as_mapping(compact.get("prepared_early"))
        late = _as_mapping(compact.get("late_residual"))
        delivery = _as_mapping(compact.get("evidence_delivery"))
        provider = _as_mapping(arm.get("provider"))
        counts = _as_mapping(provider.get("counts"))
        refs = delivery.get("selected_refs", [])
        episode_ids = [
            item.get("episode_id")
            for item in refs
            if isinstance(item, Mapping)
        ]
        variant, condition = str(key).rsplit("__", 1)
        lines.append(
            "| {variant} | {condition} | {stage} | {proposal} | {valid} | {reason} | {late} | {episodes} | {http} |".format(
                variant=variant,
                condition=condition,
                stage=prepared.get("stage", "not_observed"),
                proposal=prepared.get("proposal_generated", "not_observed"),
                valid=prepared.get("source_validated", "not_observed"),
                reason=prepared.get("retention_reason", "not_observed"),
                late=late.get("reason", "not_observed"),
                episodes=episode_ids,
                http=counts.get("http_attempts", "not_observed"),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: this matrix establishes runtime position and local candidate validation only. It does not establish recall, answer quality, utility, generalization, or a P2 early-stop benefit. Any failed or not-observed arm remains represented in the full-local package.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    learned_snapshot: Path,
    matrix_manifest: Path,
    prepared_plan_dir: Path,
    output_dir: Path,
    env_file: Path,
    treatment_edge_id: int,
    deadline_seconds: float = 120.0,
) -> dict[str, str]:
    learned_snapshot = learned_snapshot.resolve()
    matrix_manifest = matrix_manifest.resolve()
    prepared_plan_dir = prepared_plan_dir.resolve()
    output_dir = output_dir.resolve()
    env_file = env_file.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    if any(
        learned_snapshot.with_name(learned_snapshot.name + suffix).exists()
        for suffix in ("-wal", "-shm")
    ):
        raise ValueError("frozen snapshot must not have WAL/SHM sidecars")
    case = load_frozen_matrix_case(matrix_manifest)
    variants = [item for item in case.variants if item.kind in NONEXACT_KINDS]
    if not variants:
        raise ValueError("matrix has no non-exact variants for W06")
    output_dir.mkdir(parents=True)
    evaluation_as_of = _utc_now()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": _utc_now(),
        "formal_scoring": "disabled_unapproved_gold",
        "paid_provider_calls_issued": 0,
        "q1": "pre_existing_n12b_snapshot_not_reissued",
        "snapshot": {
            "path": str(learned_snapshot),
            "sha256": _sha256_file(learned_snapshot),
        },
        "input": {
            "matrix_manifest": str(matrix_manifest),
            "matrix_manifest_sha256": _sha256_file(matrix_manifest),
            "prepared_plan_dir": str(prepared_plan_dir),
            "evaluation_as_of": evaluation_as_of,
            "variants": [item.identifier for item in variants],
        },
        "conditions": {
            name: {
                "prepared_early_enabled": early,
                "edge_masked": masked,
                "entry": "public QueryEngine.query(stop_after='evidence')",
            }
            for name, early, masked in CONDITIONS
        },
        "arms": {},
        "no_retry_policy": "Each independent local arm runs once; terminal outcomes are retained.",
    }
    full_local = output_dir / "prepared_early_shadow_matrix.full_local.json"
    _write_json(full_local, payload)
    for variant in variants:
        plan_path = prepared_plan_dir / f"{variant.identifier}.frozen_query_plan.full_local.json"
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        for condition, early_enabled, edge_masked in CONDITIONS:
            key = f"{variant.identifier}__{condition}"
            payload["arms"][key] = {
                "status": "pending",
                "condition": condition,
                "variant_kind": variant.kind,
            }
            _write_json(full_local, payload)
            arm = _run_arm(
                snapshot=learned_snapshot,
                output_dir=output_dir,
                variant_id=variant.identifier,
                question=variant.text,
                plan_path=plan_path,
                domain=case.domain,
                evaluation_as_of=evaluation_as_of,
                condition=condition,
                prepared_early_enabled=early_enabled,
                edge_masked=edge_masked,
                env_file=env_file,
                treatment_edge_id=treatment_edge_id,
                deadline_seconds=deadline_seconds,
            )
            payload["arms"][key] = {
                "variant_id": variant.identifier,
                "variant_kind": variant.kind,
                **arm,
            }
            _write_json(full_local, payload)
            _write_json(output_dir / f"{key}.full_local.json", payload["arms"][key])
    payload["status"] = "completed"
    payload["finished_at"] = _utc_now()
    payload["snapshot_sha256_after"] = _sha256_file(learned_snapshot)
    payload["snapshot_unchanged"] = (
        payload["snapshot_sha256_after"] == payload["snapshot"]["sha256"]
    )
    _write_json(full_local, payload)
    report = output_dir / "W06_PREPARED_EARLY_SHADOW_MATRIX.md"
    report.write_text(_report(payload) + "\n", encoding="utf-8")
    return {"full_local": str(full_local), "report": str(report)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-snapshot", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--prepared-plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--treatment-edge-id", type=int, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
