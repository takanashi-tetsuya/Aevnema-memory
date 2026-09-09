"""Run one local candidate-pool comparison for the prepared P2 proposal.

The experiment restores stored Q2 plans/vectors and makes no provider calls.
P2 may add a source-validated, non-base edge target to normal graph seeds, but
it cannot answer, stop retrieval, create learning state, or reclassify that
target as independent base evidence. Every arm has its own SQLite clone and is
written once immediately.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from benchmarks.run_q1_q2_diagnostic_pilot import _sha256_file, _write_json
from benchmarks.run_same_chapter_learning_matrix import load_frozen_matrix_case
from benchmarks.run_v32_prepared_early_matrix import (
    NONEXACT_KINDS,
    _as_mapping,
    _run_arm,
)


SCHEMA = "aevnema.v3_2.prepared_early_candidate_pool_matrix.v1"
CONDITIONS: tuple[tuple[str, bool, bool, bool], ...] = (
    # name, prepared-early enabled, learned-edge masked, pool enabled
    ("P0_no_edge_ordinary", False, True, False),
    ("P1_late_residual", False, False, False),
    ("P2_prepared_early_candidate_pool", True, False, True),
    ("P2M_prepared_early_candidate_pool_masked", True, True, True),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _report(payload: Mapping[str, Any]) -> str:
    arms = _as_mapping(payload.get("arms"))
    lines = [
        "# Prepared-early candidate-pool matrix",
        "",
        "This is a local, zero-model diagnostic over N12b's frozen snapshot and stored Q2 vectors. P2 may only seed ordinary graph expansion with a source-validated, non-base target. It cannot answer, audit, stop ordinary retrieval, create a receipt/edge/manifest, learn, or treat that target as independent base evidence.",
        "",
        "| Q2 | Arm | Validated IDs | Injected IDs | Excluded from base route | Delivered Episodes | HTTP |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for key, value in arms.items():
        arm = _as_mapping(value)
        compact = _as_mapping(arm.get("compact"))
        prepared = _as_mapping(compact.get("prepared_early"))
        late = _as_mapping(compact.get("late_residual"))
        delivery = _as_mapping(compact.get("evidence_delivery"))
        provider = _as_mapping(arm.get("provider"))
        counts = _as_mapping(provider.get("counts"))
        delivered = [
            item.get("episode_id")
            for item in delivery.get("selected_refs", [])
            if isinstance(item, Mapping)
        ]
        variant, condition = str(key).rsplit("__", 1)
        lines.append(
            "| {variant} | {condition} | {validated} | {injected} | {excluded} | {delivered} | {http} |".format(
                variant=variant,
                condition=condition,
                validated=prepared.get("candidate_pool_validated_episode_ids", []),
                injected=prepared.get("candidate_pool_injected_episode_ids", []),
                excluded=late.get(
                    "prepared_early_contextual_endpoint_manifest", []
                ),
                delivered=delivered,
                http=counts.get("http_attempts", "not_observed"),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: a materialized candidate is not proof of factual support or a retrieval-quality gain. The full-local records retain every executed module, candidate trace, source gate result, and any non-beneficial outcome.",
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
        raise ValueError("matrix has no non-exact variants")
    output_dir.mkdir(parents=True)
    evaluation_as_of = _utc_now()
    snapshot_hash = _sha256_file(learned_snapshot)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": _utc_now(),
        "formal_scoring": "disabled_unapproved_gold",
        "paid_provider_calls_issued": 0,
        "q1": "pre_existing_n12b_snapshot_not_reissued",
        "snapshot": {"path": str(learned_snapshot), "sha256": snapshot_hash},
        "input": {
            "matrix_manifest": str(matrix_manifest),
            "matrix_manifest_sha256": _sha256_file(matrix_manifest),
            "prepared_plan_dir": str(prepared_plan_dir),
            "evaluation_as_of": evaluation_as_of,
            "variants": [item.identifier for item in variants],
        },
        "conditions": {
            name: {
                "prepared_early_enabled": prepared,
                "edge_masked": masked,
                "candidate_pool_enabled": pool,
                "entry": "public QueryEngine.query(stop_after='evidence')",
            }
            for name, prepared, masked, pool in CONDITIONS
        },
        "arms": {},
        "no_retry_policy": "Each independent local arm runs once; terminal outcomes are retained.",
    }
    full_local = output_dir / "prepared_early_candidate_pool_matrix.full_local.json"
    _write_json(full_local, payload)
    for variant in variants:
        plan_path = prepared_plan_dir / f"{variant.identifier}.frozen_query_plan.full_local.json"
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        for condition, prepared, masked, pool in CONDITIONS:
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
                prepared_early_enabled=prepared,
                edge_masked=masked,
                candidate_pool_enabled=pool,
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
        payload["snapshot_sha256_after"] == snapshot_hash
    )
    _write_json(full_local, payload)
    report = output_dir / "W08_PREPARED_EARLY_CANDIDATE_POOL_MATRIX.md"
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
