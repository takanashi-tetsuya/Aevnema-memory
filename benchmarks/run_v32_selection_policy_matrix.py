"""Run the R0--R5 local selection-policy B/S/T/M comparison once.

Every arm restores the same frozen Q2 plan and vector bundle from an existing
N12b snapshot.  It does not issue Q1, generate an answer, audit, learn, write
usage, call a provider, or modify the source snapshot.  The four conditions
differ only in whether the already-persisted edge is visible to the prepared
proposal and final request-local contribution selector.
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


SCHEMA = "aevnema.v3_2.r0_r5_selection_policy_matrix.v1"
CONDITIONS: tuple[tuple[str, bool, bool, bool, bool], ...] = (
    # name, prepared early, mask edge, candidate pool, selector shadow
    ("B_common_base_edge_masked", False, True, False, False),
    # The frozen Q2 plan fixes the final selector to non-shadow.  S remains
    # an observe-only prepared proposal (no candidate seeding), while the
    # plan's ordinary late selector retains its frozen behavior.
    ("S_prepared_observe_only", True, False, False, False),
    ("T_prepared_edge_treatment", True, False, True, False),
    ("M_prepared_edge_masked", True, True, True, False),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _selected_ids(arm: Mapping[str, Any]) -> list[object]:
    result = _as_mapping(arm.get("result"))
    evidence = _as_mapping(result.get("evidence_result"))
    return [
        _as_mapping(item).get("episode_id")
        for item in evidence.get("selected_refs", [])
        if isinstance(item, Mapping)
    ]


def _report(payload: Mapping[str, Any]) -> str:
    arms = _as_mapping(payload.get("arms"))
    lines = [
        "# R0--R5 selection-policy evidence matrix",
        "",
        "This is one local, evidence-only B/S/T/M pass over three frozen N12b Q2 variants. It uses the public QueryEngine request path and restored vectors. No provider call, Q1, answer, answer audit, learning, receipt, edge/manifest creation, usage write, promotion, or formal score is included.",
        "",
        "| Q2 | Arm | Input-stage classification | Base support mapping | Selected Episodes | Source delivery | Requirement coverage | Edge selection state | HTTP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    excerpts: dict[str, tuple[str, str]] = {}
    for key, raw_arm in arms.items():
        arm = _as_mapping(raw_arm)
        result = _as_mapping(arm.get("result"))
        contextual = _as_mapping(result.get("contextual_association"))
        prepared = _as_mapping(contextual.get("prepared_early"))
        input_stage = _as_mapping(prepared.get("input_stage_replay"))
        evidence = _as_mapping(result.get("evidence_result"))
        participation = _as_mapping(evidence.get("participation"))
        provider = _as_mapping(arm.get("provider"))
        counts = _as_mapping(provider.get("counts"))
        variant, condition = str(key).rsplit("__", 1)
        lines.append(
            "| {variant} | {condition} | {stage} | {mapping} | {selected} | {delivery} | {coverage} | {edge} | {http} |".format(
                variant=variant,
                condition=condition,
                stage=input_stage.get("classification", "not_observed"),
                mapping=contextual.get("base_support_mapping_status", "not_observed"),
                selected=_selected_ids(arm),
                delivery=evidence.get("source_delivery_state", evidence.get("evidence_state", "not_observed")),
                coverage=evidence.get("requirement_coverage_state", "not_observed"),
                edge=participation.get("edge_state", "not_observed"),
                http=counts.get("http_attempts", "not_observed"),
            )
        )
        for source_ref in evidence.get("materialized_source_refs", []):
            ref = _as_mapping(source_ref)
            excerpt = str(ref.get("source_excerpt", ""))
            digest = str(ref.get("source_excerpt_hash", ""))
            if excerpt and digest:
                excerpts.setdefault(
                    digest,
                    (
                        "Episode {episode} — {source} segment {segment}".format(
                            episode=ref.get("episode_id", "not_observed"),
                            source=ref.get("source_key", "not_observed"),
                            segment=ref.get("segment_index", "not_observed"),
                        ),
                        excerpt,
                    ),
                )
    lines.extend(
        [
            "",
            "## Actual delivered Source excerpts",
            "",
            "Each distinct delivered excerpt is included below once, keyed by its local content hash. Source-bound describes quote/span provenance only; it is not a semantic entailment or a gold result.",
            "",
        ]
    )
    if excerpts:
        for digest, (label, excerpt) in sorted(excerpts.items()):
            lines.extend([f"### {label}", "", f"`{digest}`", "", "```text", excerpt, "```", ""])
    else:
        lines.extend(["No source excerpt was materialized in a completed arm.", ""])
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "B and M hide only the persisted edge; independent base retrieval remains available. S observes a prepared proposal without seeding the candidate pool, but its ordinary late selector remains exactly as fixed by the frozen Q2 plan (non-shadow); S is therefore not an independent masked-selection arm. T enables the same prepared candidate-pool lane. A matching output, a no-benefit result, a harmful selection, a timeout, or a missing observation is retained rather than retried.",
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
    snapshot_hash = _sha256_file(learned_snapshot)
    evaluation_as_of = _utc_now()
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
            "shared_policy": "source-bound delivery plus request-local ranking-only contextual priority",
        },
        "conditions": {
            name: {
                "prepared_early_enabled": prepared,
                "edge_masked": masked,
                "candidate_pool_enabled": candidate_pool,
                "selector_shadow": selector_shadow,
                "entry": "public QueryEngine.query(stop_after='evidence')",
            }
            for name, prepared, masked, candidate_pool, selector_shadow in CONDITIONS
        },
        "arms": {},
        "no_retry_policy": "Each independent local arm runs once; every terminal outcome is retained.",
    }
    full_local = output_dir / "r0_r5_selection_policy_matrix.full_local.json"
    _write_json(full_local, payload)
    for variant in variants:
        plan_path = prepared_plan_dir / f"{variant.identifier}.frozen_query_plan.full_local.json"
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        for condition, prepared, masked, candidate_pool, selector_shadow in CONDITIONS:
            key = f"{variant.identifier}__{condition}"
            payload["arms"][key] = {"status": "pending", "condition": condition}
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
                candidate_pool_enabled=candidate_pool,
                selection_shadow=selector_shadow,
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
    payload["snapshot_unchanged"] = payload["snapshot_sha256_after"] == snapshot_hash
    _write_json(full_local, payload)
    report = output_dir / "R0_R5_SELECTION_POLICY_MATRIX.md"
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
