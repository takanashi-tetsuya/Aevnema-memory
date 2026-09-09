"""Run W07's W/C/N/CN prepared-early key ablation through public Q2 calls.

The input is a completed W06 shadow package.  Only variants which actually
produced a source-valid P2 candidate there are eligible; this command does not
manufacture a missing slot, change a Q2 text/vector, or inspect a gold answer.
Each condition is a fresh clone of the same frozen learned snapshot and calls
the normal evidence-only QueryEngine entry once, with no model provider calls.
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
    _as_mapping,
    _compact_result,
    _read,
    _run_arm,
)


SCHEMA = "aevnema.v3_2.prepared_early_key_ablation.v1"
KEY_MODES = ("w", "c", "n", "cn")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_variants(observation: Mapping[str, Any]) -> list[str]:
    """Return only W06 variants with an actually observed valid candidate."""

    arms = _as_mapping(observation.get("arms"))
    variants: set[str] = set()
    for arm in arms.values():
        arm_map = _as_mapping(arm)
        if str(arm_map.get("condition", "")) != "P2_prepared_early_shadow":
            continue
        compact = _as_mapping(arm_map.get("compact"))
        prepared = _as_mapping(compact.get("prepared_early"))
        if bool(prepared.get("proposal_generated")) and bool(
            prepared.get("source_validated")
        ):
            variant_id = str(arm_map.get("variant_id", "")).strip()
            if variant_id:
                variants.add(variant_id)
    return sorted(variants)


def _shuffle_control_eligibility(snapshot: Path) -> dict[str, object]:
    """State whether a nontrivial RAM-only endpoint shuffle is possible.

    A one-edge corpus has no permutation that changes an endpoint while
    retaining the edge/cue count and distribution.  We report that explicitly
    rather than adding a synthetic edge or writing a replacement into SQLite.
    """

    import sqlite3

    uri = f"{snapshot.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        rows = connection.execute(
            """
            SELECT id, from_id, to_id, context_cue_id, need_cue_id
            FROM association
            WHERE association_mode = 'contextual_recall'
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()
    count = len(rows)
    if count < 2:
        return {
            "status": "not_run",
            "reason": "fewer_than_two_contextual_edges_no_nontrivial_permutation",
            "contextual_edge_count": count,
            "database_mutated": False,
        }
    return {
        "status": "not_run",
        "reason": "ram_only_shuffle_not_implemented_in_this_frozen_followup",
        "contextual_edge_count": count,
        "database_mutated": False,
    }


def _report(payload: Mapping[str, Any]) -> str:
    arms = _as_mapping(payload.get("arms"))
    lines = [
        "# W07 prepared-early key ablation",
        "",
        "This is a ranking/candidate diagnostic, not a claim that any key alone proves a requirement. Every arm uses the same public evidence-only Q2 path, current-Q2 target/source gate, frozen plan/vector bundle, source snapshot, and no-early-stop shadow policy. W retains edge/anchor/lifecycle strength only; C and N require their respective cue gates; CN is the production double-key rule.",
        "",
        "| Q2 | Key | Proposal | Source-valid candidate | First association | First target | Score | Vector relationship | Delivery | HTTP |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for key, arm_value in arms.items():
        arm = _as_mapping(arm_value)
        compact = _as_mapping(arm.get("compact"))
        prepared = _as_mapping(compact.get("prepared_early"))
        accepted = prepared.get("accepted_candidate_order", [])
        first = accepted[0] if isinstance(accepted, list) and accepted and isinstance(accepted[0], Mapping) else {}
        delivery = _as_mapping(compact.get("evidence_delivery"))
        provider = _as_mapping(arm.get("provider"))
        counts = _as_mapping(provider.get("counts"))
        variant, mode = str(key).rsplit("__", 1)
        lines.append(
            "| {variant} | {mode} | {proposal} | {valid} | {edge} | {target} | {score} | {relationship} | {delivery} | {http} |".format(
                variant=variant,
                mode=mode,
                proposal=prepared.get("proposal_generated", "not_observed"),
                valid=prepared.get("source_validated", "not_observed"),
                edge=first.get("association_id", "—"),
                target=first.get("target_episode_id", "—"),
                score=first.get("combined_score", "—"),
                relationship=prepared.get(
                    "context_need_vector_relationship", "not_observed"
                ),
                delivery=delivery.get("state", "not_observed"),
                http=counts.get("http_attempts", "not_observed"),
            )
        )
    shuffle = _as_mapping(payload.get("shuffled_endpoint_control"))
    lines.extend(
        [
            "",
            "Shuffle control: {status} — {reason}.".format(
                status=shuffle.get("status", "not_observed"),
                reason=shuffle.get("reason", "not_observed"),
            ),
            "",
            "Interpretation boundary: when context and need share one physical vector, C/N/CN are explicitly degenerate for this case and cannot demonstrate independent double-key value. This output preserves that limitation rather than fabricating a second vector or endpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    learned_snapshot: Path,
    matrix_manifest: Path,
    prepared_plan_dir: Path,
    candidate_observation: Path,
    output_dir: Path,
    env_file: Path,
    treatment_edge_id: int,
    deadline_seconds: float = 120.0,
) -> dict[str, str]:
    learned_snapshot = learned_snapshot.resolve()
    matrix_manifest = matrix_manifest.resolve()
    prepared_plan_dir = prepared_plan_dir.resolve()
    candidate_observation = candidate_observation.resolve()
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
    prior = _read(candidate_observation)
    variants = _candidate_variants(prior)
    if not variants:
        raise ValueError("W06 observation has no source-valid P2 candidate")
    variant_by_id = {item.identifier: item for item in case.variants}
    if any(item not in variant_by_id for item in variants):
        raise ValueError("W06 candidate variant is absent from frozen matrix")
    prior_snapshot = _as_mapping(prior.get("snapshot"))
    if str(prior_snapshot.get("sha256", "")) != _sha256_file(learned_snapshot):
        raise ValueError("W06 observation used a different learned snapshot")

    output_dir.mkdir(parents=True)
    evaluation_as_of = _utc_now()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": _utc_now(),
        "formal_scoring": "disabled_unapproved_gold",
        "paid_provider_calls_issued": 0,
        "snapshot": {
            "path": str(learned_snapshot),
            "sha256": _sha256_file(learned_snapshot),
        },
        "input": {
            "matrix_manifest": str(matrix_manifest),
            "matrix_manifest_sha256": _sha256_file(matrix_manifest),
            "prepared_plan_dir": str(prepared_plan_dir),
            "candidate_observation": str(candidate_observation),
            "candidate_observation_sha256": _sha256_file(candidate_observation),
            "eligible_variants_observed_in_W06": variants,
            "evaluation_as_of": evaluation_as_of,
        },
        "key_modes": list(KEY_MODES),
        "shuffled_endpoint_control": _shuffle_control_eligibility(learned_snapshot),
        "arms": {},
        "no_retry_policy": "Each key/variant arm runs once; terminal outcomes remain.",
    }
    full_local = output_dir / "prepared_early_key_ablation.full_local.json"
    _write_json(full_local, payload)
    for variant_id in variants:
        variant = variant_by_id[variant_id]
        plan_path = prepared_plan_dir / f"{variant.identifier}.frozen_query_plan.full_local.json"
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        for mode in KEY_MODES:
            key = f"{variant.identifier}__{mode}"
            payload["arms"][key] = {"status": "pending", "key_scoring_mode": mode}
            _write_json(full_local, payload)
            arm = _run_arm(
                snapshot=learned_snapshot,
                output_dir=output_dir,
                variant_id=variant.identifier,
                question=variant.text,
                plan_path=plan_path,
                domain=case.domain,
                evaluation_as_of=evaluation_as_of,
                condition=f"key_{mode}",
                prepared_early_enabled=True,
                edge_masked=False,
                env_file=env_file,
                treatment_edge_id=treatment_edge_id,
                deadline_seconds=deadline_seconds,
                scoring_mode=mode,
            )
            compact = _compact_result(arm.get("result")) if arm.get("status") == "completed" else {}
            payload["arms"][key] = {
                "variant_id": variant.identifier,
                "variant_kind": variant.kind,
                "key_scoring_mode": mode,
                **arm,
                "compact": compact,
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
    report = output_dir / "W07_PREPARED_EARLY_KEY_ABLATION.md"
    report.write_text(_report(payload) + "\n", encoding="utf-8")
    return {"full_local": str(full_local), "report": str(report)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-snapshot", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--prepared-plan-dir", type=Path, required=True)
    parser.add_argument("--candidate-observation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--treatment-edge-id", type=int, required=True)
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
