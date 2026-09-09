"""Run the v3.2 W06/W07 public-entry position replay without model calls.

The replay is intentionally narrow: it uses the pre-existing N12b snapshot,
the four stored public frozen Q2 plans, and public QueryEngine entry points.
It does not issue Q1, create a receipt/edge/manifest, call the matcher
directly, invent a missing slot, or run answer/audit/learning.  A preflight
miss has no private diagnostic payload, so the report preserves that limit as
``not_observed_by_public_preflight`` rather than making up a rejection cause.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
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
from benchmarks.run_same_chapter_learning_matrix import _provider_partition, load_frozen_matrix_case


SCHEMA = "aevnema.v3_2.public_position_replay.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _compact_result(value: object) -> dict[str, Any]:
    """Persist the public branch state needed for W06/W07, not raw Source."""

    result = value if isinstance(value, Mapping) else {}
    evidence = result.get("evidence_result")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    contextual = result.get("contextual_association")
    contextual = contextual if isinstance(contextual, Mapping) else {}
    revisit = result.get("exact_revisit")
    revisit = revisit if isinstance(revisit, Mapping) else {}
    timings = result.get("timings")
    timings = timings if isinstance(timings, Mapping) else {}
    return {
        "exact_revisit_status": revisit.get("status", "not_observed"),
        "exact_revisit_kind": revisit.get("kind", "not_observed"),
        "contextual_association_reason": contextual.get("reason", "not_observed"),
        "contextual_matcher_executed": (
            bool(contextual.get("hits") or contextual.get("pre_target_proposals"))
            if contextual
            else False
        ),
        "contextual_attached_edges": contextual.get("attached_edges", []),
        "contextual_selected_contribution_ids": contextual.get(
            "selected_contextual_contribution_ids", []
        ),
        "delivered_episode_ids": [
            item.get("episode_id")
            for item in evidence.get("selected_refs", [])
            if isinstance(item, Mapping)
        ],
        "participation": evidence.get("participation", "not_observed"),
        "executed_modules": evidence.get("executed_modules", "not_observed"),
        "actual_skipped_modules": evidence.get("actual_skipped_modules", "not_observed"),
        "phase_seconds": timings.get("phases_seconds", "not_observed"),
    }


def _restore_plan(engine: object, *, question: str, path: Path) -> tuple[dict[str, Any], object]:
    plan = _read(path)
    restore = getattr(engine, "restore_frozen_query_vectors", None)
    if not callable(restore):
        raise TypeError("public engine lacks restore_frozen_query_vectors")
    bundle = restore(question, plan)
    return dict(plan), bundle


def _run_condition(
    *,
    snapshot: Path,
    output_dir: Path,
    variant_id: str,
    question: str,
    condition: str,
    domain: str,
    scope_hash: str,
    plan_path: Path,
    env_file: Path,
    treatment_edge_id: int,
    deadline_seconds: float,
) -> dict[str, Any]:
    database = output_dir / "work" / f"{variant_id}__{condition}.sqlite"
    _clone_sqlite(snapshot, database)
    app, config = _open_app(env_file, database, output_dir / "logs" / variant_id / condition)
    engine = app.query_engine(config=config)
    if condition == "edge_masked":
        _mask_engine_edge(engine, treatment_edge_id)
    ledger = ProviderLedger()
    started = perf_counter()
    with _trace_context(engine.model, ledger):
        # This is the public early automatic-revisit boundary. It returns
        # None on a miss by contract, so it cannot explain a non-exact miss.
        preflight = engine.try_contextual_revisit(
            question,
            contextual_domain=domain,
            contextual_revisit_scope_hash=scope_hash,
            deadline_seconds=deadline_seconds,
        )
        plan, bundle = _restore_plan(engine, question=question, path=plan_path)
        result = engine.query(
            question,
            generate_answer=False,
            stop_after="evidence",
            frozen_plan=plan,
            query_vector_bundle=bundle,
            strict_vector_bundle=True,
            contextual_domain=domain,
            contextual_revisit_scope_hash=scope_hash,
            contextual_learning=False,
            deadline_seconds=deadline_seconds,
        )
    provider = ledger.export()
    counts = provider.get("counts") if isinstance(provider, Mapping) else {}
    counts = counts if isinstance(counts, Mapping) else {}
    if int(counts.get("http_attempts", 0) or 0) != 0:
        raise RuntimeError("position replay unexpectedly issued a provider request")
    return {
        "status": "completed",
        "condition": condition,
        "database_clone": str(database),
        "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "preflight": {
            "entry": "public QueryEngine.try_contextual_revisit",
            "status": "hit" if isinstance(preflight, Mapping) else "miss",
            "miss_reason": (
                "not_observed_by_public_preflight"
                if preflight is None
                else None
            ),
            "result": _compact_result(preflight),
        },
        "late_evidence_entry": {
            "entry": "public QueryEngine.query(stop_after='evidence')",
            "result": _compact_result(result),
        },
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


def _report(payload: Mapping[str, Any]) -> str:
    arms = payload.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    lines = [
        "# W06/W07 public position replay",
        "",
        "This is a local, Q2-only, non-scoring replay. It used pre-existing N12b state and stored Q2 vectors. No model HTTP, Q1, answer/audit, receipt, edge, manifest, or usage write occurred.",
        "",
        "| Variant | Condition | Early public preflight | Early miss reason | Late branch reason | Matcher executed | Delivered Episodes | HTTP |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            continue
        preflight = arm.get("preflight") if isinstance(arm.get("preflight"), Mapping) else {}
        late = arm.get("late_evidence_entry") if isinstance(arm.get("late_evidence_entry"), Mapping) else {}
        result = late.get("result") if isinstance(late.get("result"), Mapping) else {}
        provider = arm.get("provider") if isinstance(arm.get("provider"), Mapping) else {}
        counts = provider.get("counts") if isinstance(provider.get("counts"), Mapping) else {}
        lines.append(
            "| {variant} | {condition} | {early} | {miss} | {late_reason} | {matcher} | {episodes} | {http} |".format(
                variant=name.rsplit("__", 1)[0],
                condition=arm.get("condition", "not_observed"),
                early=preflight.get("status", "not_observed"),
                miss=preflight.get("miss_reason", "—") or "—",
                late_reason=result.get("contextual_association_reason", "not_observed"),
                matcher=result.get("contextual_matcher_executed", "not_observed"),
                episodes=result.get("delivered_episode_ids", []),
                http=counts.get("http_attempts", "not_observed"),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation boundary: the early public preflight supports automatic exact/restricted revisit only. A public preflight miss intentionally exposes no reason, so it cannot establish a non-exact proposal rejection. The late public evidence path records the actual contextual branch; if base delivery leaves no required slot unresolved, the matcher is correctly not entered. This replay does not manufacture a missing slot or claim a new cross-topic result.",
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
    if any(learned_snapshot.with_name(learned_snapshot.name + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("frozen snapshot must not have WAL/SHM sidecars")
    case = load_frozen_matrix_case(matrix_manifest)
    output_dir.mkdir(parents=True)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "started_at": _utc_now(),
        "paid_provider_calls_issued": 0,
        "q1": "pre_existing_n12b_snapshot_not_reissued",
        "formal_scoring": "disabled_unapproved_gold",
        "snapshot": {"path": str(learned_snapshot), "sha256": _sha256_file(learned_snapshot)},
        "input": {"matrix_manifest": str(matrix_manifest), "prepared_plan_dir": str(prepared_plan_dir)},
        "position_contract": {
            "early": "public try_contextual_revisit automatic exact/restricted preflight",
            "late": "public query stop_after=evidence ordinary selection with contextual residual branch",
            "nonexact_early_public_entry": "not_available_in_current_public_api",
        },
        "arms": {},
        "no_retry_policy": "Each new local replay condition runs once; historical outcomes are not replaced.",
    }
    _write_json(output_dir / "public_position_replay.full_local.json", payload)
    for variant in case.variants:
        plan = prepared_plan_dir / f"{variant.identifier}.frozen_query_plan.full_local.json"
        if not plan.is_file():
            raise FileNotFoundError(plan)
        for condition in ("edge_available", "edge_masked"):
            key = f"{variant.identifier}__{condition}"
            payload["arms"][key] = _run_condition(
                snapshot=learned_snapshot,
                output_dir=output_dir,
                variant_id=variant.identifier,
                question=variant.text,
                condition=condition,
                domain=case.domain,
                scope_hash=case.scope_hash,
                plan_path=plan,
                env_file=env_file,
                treatment_edge_id=treatment_edge_id,
                deadline_seconds=deadline_seconds,
            )
            _write_json(output_dir / "public_position_replay.full_local.json", payload)
    payload["status"] = "completed"
    payload["finished_at"] = _utc_now()
    payload["snapshot_sha256_after"] = _sha256_file(learned_snapshot)
    payload["snapshot_unchanged"] = payload["snapshot_sha256_after"] == payload["snapshot"]["sha256"]
    _write_json(output_dir / "public_position_replay.full_local.json", payload)
    (output_dir / "W06_W07_PUBLIC_POSITION_REPLAY.md").write_text(_report(payload) + "\n", encoding="utf-8")
    return {"full_local": str(output_dir / "public_position_replay.full_local.json"), "report": str(output_dir / "W06_W07_PUBLIC_POSITION_REPLAY.md")}


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
