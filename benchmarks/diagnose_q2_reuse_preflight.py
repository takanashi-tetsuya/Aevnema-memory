"""Record one public, provider-free Q2 exact-reuse preflight.

This is deliberately an evaluator artifact, not a new retrieval shortcut.  It
calls :meth:`QueryEngine.try_contextual_revisit`, the same public automatic
revisit entrance used before ordinary ``query`` work.  A miss remains a miss:
the script neither constructs a ticket, invokes a matcher directly, creates a
contract/edge, nor falls through to a provider-backed query unless the caller
explicitly asks for the one small full-answer control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from time import perf_counter
from typing import Mapping

from benchmarks.run_q1_q2_diagnostic_pilot import (
    ProviderLedger,
    _call_query,
    _clone_sqlite,
    _open_app,
    _read_object,
    _result_summary,
    _trace_context,
    _write_json,
)


SCHEMA = "aevnema.v3.q2_reuse_preflight.v1"


def _json_array_size(value: object) -> int:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def _database_state(
    database: Path,
    *,
    receipt_id: int,
    association_id: int,
) -> dict[str, object]:
    """Read durable identifiers without making a matcher or model call."""

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        receipt = connection.execute(
            """
            SELECT id, association_id, context_cue_id, need_cue_id, domain,
                   model_id, embedding_space_id, dimension, dtype,
                   verification_status, source_facts_json,
                   verification_refs_json, source_request_hash, status,
                   ready_index_epoch, created_at, ready_at
            FROM contextual_creation_receipt
            WHERE id = ?
            """,
            (receipt_id,),
        ).fetchone()
        edge = connection.execute(
            """
            SELECT id, from_type, from_id, to_type, to_id, association_mode,
                   lifecycle_state, source_request_hash, claim_level,
                   created_reason, created_at, expires_at
            FROM association
            WHERE id = ?
            """,
            (association_id,),
        ).fetchone()
        contract_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM contextual_revisit_contract "
                "WHERE creation_receipt_id = ?",
                (receipt_id,),
            ).fetchone()[0]
        )
        manifest_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM contextual_revisit_runtime_manifest "
                "WHERE creation_receipt_id = ?",
                (receipt_id,),
            ).fetchone()[0]
        )
        episode_rows = []
        if edge is not None:
            for episode_id in (int(edge["from_id"]), int(edge["to_id"])):
                row = connection.execute(
                    """
                    SELECT id, source_id, source_key, segment_index,
                           evidence_origin, evidence_quotes_json,
                           evidence_spans_json, evidence_basis
                    FROM episode WHERE id = ?
                    """,
                    (episode_id,),
                ).fetchone()
                if row is None:
                    episode_rows.append({"episode_id": episode_id, "present": False})
                    continue
                episode_rows.append(
                    {
                        "episode_id": int(row["id"]),
                        "present": True,
                        "source_id": int(row["source_id"]),
                        "source_key": str(row["source_key"]),
                        "segment_index": int(row["segment_index"]),
                        "evidence_origin": str(row["evidence_origin"]),
                        "literal_quote_count": _json_array_size(
                            row["evidence_quotes_json"]
                        ),
                        "literal_span_count": _json_array_size(
                            row["evidence_spans_json"]
                        ),
                        "evidence_basis": str(row["evidence_basis"] or ""),
                    }
                )
    finally:
        connection.close()

    receipt_payload: dict[str, object] | None = None
    if receipt is not None:
        source_facts = str(receipt["source_facts_json"] or "[]")
        verification_refs = str(receipt["verification_refs_json"] or "[]")
        receipt_payload = {
            "id": int(receipt["id"]),
            "association_id": int(receipt["association_id"]),
            "context_cue_id": int(receipt["context_cue_id"]),
            "need_cue_id": int(receipt["need_cue_id"]),
            "domain": str(receipt["domain"]),
            "model_id": str(receipt["model_id"]),
            "embedding_space_id": str(receipt["embedding_space_id"]),
            "dimension": int(receipt["dimension"]),
            "dtype": str(receipt["dtype"]),
            "verification_status": str(receipt["verification_status"]),
            "source_fact_count": _json_array_size(source_facts),
            "source_facts_sha256": "sha256:"
            + hashlib.sha256(source_facts.encode("utf-8")).hexdigest(),
            "verification_ref_count": _json_array_size(verification_refs),
            "source_request_hash": str(receipt["source_request_hash"]),
            "status": str(receipt["status"]),
            "ready_index_epoch": int(receipt["ready_index_epoch"]),
            "created_at": str(receipt["created_at"]),
            "ready_at": str(receipt["ready_at"]),
        }
    edge_payload = (
        {
            "id": int(edge["id"]),
            "from": {"type": str(edge["from_type"]), "id": int(edge["from_id"])},
            "to": {"type": str(edge["to_type"]), "id": int(edge["to_id"])},
            "association_mode": str(edge["association_mode"]),
            "lifecycle_state": str(edge["lifecycle_state"]),
            "source_request_hash": str(edge["source_request_hash"]),
            "claim_level": str(edge["claim_level"]),
            "created_reason": str(edge["created_reason"]),
            "created_at": str(edge["created_at"]),
            "expires_at": str(edge["expires_at"]),
        }
        if edge is not None
        else None
    )
    return {
        "receipt": receipt_payload,
        "edge": edge_payload,
        "contract_count": contract_count,
        "runtime_manifest_count": manifest_count,
        "endpoint_source_records": episode_rows,
    }


def _existing_q1_projection_observation(case_payload: Mapping[str, object]) -> dict[str, object]:
    q1 = case_payload.get("q1")
    record = q1.get("record") if isinstance(q1, Mapping) else None
    result = record.get("result") if isinstance(record, Mapping) else None
    learning = result.get("contextual_learning") if isinstance(result, Mapping) else None
    requirements = (
        result.get("authoritative_requirements") if isinstance(result, Mapping) else None
    )
    required = requirements.get("requirements") if isinstance(requirements, Mapping) else []
    slot = required[0] if isinstance(required, list) and required else {}
    if not isinstance(slot, Mapping):
        slot = {}
    fields = {
        name: slot.get(name)
        for name in (
            "subject_terms",
            "object_terms",
            "relation_hint",
            "temporal_hint",
            "epistemic_hint",
            "negation_hint",
            "modality_hint",
        )
    }
    non_template_fields = [
        name
        for name, value in fields.items()
        if bool(value) and value != []
    ]
    return {
        "q1_learning_status": learning.get("status") if isinstance(learning, Mapping) else None,
        "q1_revisit_contract_outcomes": (
            list(learning.get("revisit_contracts", []))
            if isinstance(learning, Mapping) and isinstance(learning.get("revisit_contracts"), list)
            else []
        ),
        "required_slot_shape": fields,
        "template_ineligible_fields": non_template_fields,
        "analysis_boundary": (
            "These are saved Q1 fields compared with the exact-template "
            "shape. They explain the absent contract/manifest; no missing "
            "Q1 event is reconstructed."
        ),
    }


def _existing_q2_observation(case_payload: Mapping[str, object], arm: str) -> dict[str, object]:
    arms = case_payload.get("arms")
    entry = arms.get(arm) if isinstance(arms, Mapping) else None
    record = entry.get("record") if isinstance(entry, Mapping) else None
    result = record.get("result") if isinstance(record, Mapping) else None
    contextual = result.get("contextual_association") if isinstance(result, Mapping) else None
    contextual_reason, reason_source = _contextual_reason(result, contextual)
    return {
        "arm": arm,
        "record_status": record.get("status") if isinstance(record, Mapping) else None,
        "exact_revisit": result.get("exact_revisit") if isinstance(result, Mapping) else None,
        "contextual_reason": contextual_reason,
        "contextual_reason_source": reason_source if contextual_reason is not None else "not_observed",
        "attached_edges": (
            list(contextual.get("attached_edges", []))
            if isinstance(contextual, Mapping) else []
        ),
        "matcher_backend": (
            contextual.get("matcher_backend") if isinstance(contextual, Mapping) else None
        ),
        "selected_contextual_contribution_ids": (
            list(contextual.get("selected_contextual_contribution_ids", []))
            if isinstance(contextual, Mapping) else []
        ),
        "provider_counts": (
            record.get("provider", {}).get("counts", {})
            if isinstance(record, Mapping) and isinstance(record.get("provider"), Mapping)
            else {}
        ),
    }


def _contextual_reason(
    result: Mapping[str, object] | None,
    contextual: Mapping[str, object] | None,
) -> tuple[object | None, str]:
    """Read the saved selector reason without inventing one when absent."""

    reason = contextual.get("reason") if isinstance(contextual, Mapping) else None
    if reason is not None:
        return reason, "contextual_association.reason"
    slot_trace = result.get("evidence_slot_trace") if isinstance(result, Mapping) else None
    selector_v3 = (
        slot_trace.get("slot_selector_v3") if isinstance(slot_trace, Mapping) else None
    )
    if isinstance(selector_v3, Mapping) and selector_v3.get("reason") is not None:
        return selector_v3.get("reason"), "evidence_slot_trace.slot_selector_v3.reason"
    return None, "not_observed"


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    state = payload["durable_state"]
    preflight = payload["public_preflight"]
    prior = payload["existing_q2_observation"]
    full = payload.get("full_answer_control")
    assert isinstance(state, Mapping) and isinstance(preflight, Mapping)
    assert isinstance(prior, Mapping)
    receipt = state.get("receipt")
    edge = state.get("edge")
    lines = [
        "# Q2 reuse preflight diagnostic",
        "",
        "- This run uses `QueryEngine.try_contextual_revisit`, the public automatic reuse entry. It made no provider calls and does not invoke the matcher directly.",
        "- The diagnostic cloned the supplied database before reading it; no receipt, edge, cue, contract, or source row was changed.",
        "",
        "## Durable route",
        "",
        f"- Receipt: `{receipt.get('id') if isinstance(receipt, Mapping) else None}`; status `{receipt.get('status') if isinstance(receipt, Mapping) else None}`; source verification `{receipt.get('verification_status') if isinstance(receipt, Mapping) else None}`.",
        f"- Edge: `{edge.get('id') if isinstance(edge, Mapping) else None}`; `{edge.get('from') if isinstance(edge, Mapping) else None}` → `{edge.get('to') if isinstance(edge, Mapping) else None}`.",
        f"- V16 contracts for receipt: `{state.get('contract_count')}`. V17 runtime manifests for receipt: `{state.get('runtime_manifest_count')}`.",
        "",
        "## Public preflight event",
        "",
        f"- Result: `{preflight.get('result_status')}`. Provider HTTP attempts: `{preflight.get('provider_counts', {}).get('http_attempts') if isinstance(preflight.get('provider_counts'), Mapping) else None}`.",
        f"- First ordinary-fallback reason: `{preflight.get('first_fallback_reason')}`.",
        "",
        "The preflight reached no contract/source-closure/matcher/selector/answer stage because the receipt has no matching ready V17 runtime manifest. This is `not_entered`, not an edge rejection after matching and not a measured no-benefit result.",
        "",
        "## Existing v10 full Q2 record",
        "",
        f"- Exact revisit: `{prior.get('exact_revisit')}`; contextual selector reason: `{prior.get('contextual_reason')}`; attached edges: `{prior.get('attached_edges')}`.",
        "- The ordinary selector had no unresolved required slot, so it did not invoke the matcher. Its full provider count is historical only and is not attributed to edge reuse.",
    ]
    if isinstance(full, Mapping):
        summary = full.get("summary")
        lines.extend(
            [
                "",
                "## One full-answer control",
                "",
                f"- Run status: `{summary.get('run_status') if isinstance(summary, Mapping) else None}`; HTTP attempts: `{summary.get('provider_counts', {}).get('http_attempts') if isinstance(summary, Mapping) and isinstance(summary.get('provider_counts'), Mapping) else None}`.",
                f"- Exact revisit: `{full.get('exact_revisit')}`; contextual attached edges: `{full.get('contextual_attached_edges')}`; reason: `{full.get('contextual_reason')}`.",
                "- This is one control attempt, not a replacement for the v10 trial and not a search for a better result.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    source_database: Path,
    case_path: Path,
    env_file: Path,
    output_dir: Path,
    receipt_id: int,
    association_id: int,
    existing_arm: str,
    run_full_answer: bool,
    deadline_seconds: float,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    case_payload = _read_object(case_path)
    case = case_payload.get("case")
    if not isinstance(case, Mapping):
        raise ValueError("case package has no case metadata")
    question = str(case.get("q2_text", "")).strip()
    domain = str(case.get("domain", "")).strip()
    scope_hash = str(case.get("scope_hash", "")).strip()
    if not question or not domain or not scope_hash:
        raise ValueError("case package has incomplete Q2 request metadata")

    output_dir.mkdir(parents=True)
    work_database = output_dir / "work" / "public_preflight.sqlite"
    _clone_sqlite(source_database, work_database)
    state = _database_state(
        work_database, receipt_id=receipt_id, association_id=association_id
    )
    app, config = _open_app(env_file, work_database, output_dir / "logs" / "preflight")
    engine = app.query_engine(config=config)
    ledger = ProviderLedger()
    started = perf_counter()
    with _trace_context(engine.model, ledger):
        reuse_result = engine.try_contextual_revisit(
            question,
            contextual_domain=domain,
            contextual_revisit_scope_hash=scope_hash,
            deadline_seconds=deadline_seconds,
        )
    provider = ledger.export()
    counts = provider["counts"]
    if int(counts["http_attempts"]) != 0 or int(counts["logical_batches"]) != 0:
        raise RuntimeError("public preflight unexpectedly made a provider call")

    manifest_count = int(state["runtime_manifest_count"])
    fallback_reason = (
        "runtime_manifest_lookup_miss"
        if reuse_result is None and manifest_count == 0
        else "not_observed" if reuse_result is None else "not_applicable_hit"
    )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "mode": "diagnostic_only_non_scorable",
        "source_database": str(source_database.resolve()),
        "case_path": str(case_path.resolve()),
        "request": {
            "question_sha256": "sha256:"
            + hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "domain": domain,
            "scope_hash": scope_hash,
            "receipt_id": receipt_id,
            "association_id": association_id,
        },
        "durable_state": state,
        "q1_projection_observation": _existing_q1_projection_observation(case_payload),
        "public_preflight": {
            "entry": "QueryEngine.try_contextual_revisit",
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 3),
            "result_status": "hit" if reuse_result is not None else "miss",
            "result": reuse_result,
            "provider_counts": counts,
            "first_fallback_reason": fallback_reason,
            "stage_boundary": (
                "runtime_manifest_lookup" if reuse_result is None else "exact_revisit_delivery"
            ),
        },
        "existing_q2_observation": _existing_q2_observation(case_payload, existing_arm),
        "inference_boundary": (
            "The public preflight's miss is an observed result. The recorded Q1 "
            "slot shape is only an explanation for the absent optional runtime "
            "projection, not a reconstructed event."
        ),
        "formal_scoring": False,
    }
    if run_full_answer:
        full_database = output_dir / "work" / "full_answer_control.sqlite"
        _clone_sqlite(source_database, full_database)
        full_app, full_config = _open_app(
            env_file, full_database, output_dir / "logs" / "full_answer_control"
        )
        record = _call_query(
            full_app.query_engine(config=full_config),
            question,
            domain=domain,
            scope_hash=scope_hash,
            deadline_seconds=deadline_seconds,
        )
        result = record.get("result") if isinstance(record, Mapping) else None
        contextual = result.get("contextual_association") if isinstance(result, Mapping) else None
        contextual_reason, reason_source = _contextual_reason(result, contextual)
        payload["full_answer_control"] = {
            "record": record,
            "summary": _result_summary(record),
            "exact_revisit": result.get("exact_revisit") if isinstance(result, Mapping) else None,
            "contextual_attached_edges": (
                list(contextual.get("attached_edges", []))
                if isinstance(contextual, Mapping) else []
            ),
            "contextual_reason": contextual_reason,
            "contextual_reason_source": reason_source,
            "attempt_policy": "one_control_attempt_only",
        }

    _write_json(output_dir / "q2_reuse_preflight.full_local.json", payload)
    _write_report(output_dir / "q2_reuse_preflight.report.md", payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--case-path", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt-id", type=int, default=1)
    parser.add_argument("--association-id", type=int, default=121)
    parser.add_argument("--existing-arm", default="learning_ready_q2")
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    parser.add_argument("--run-full-answer", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run(
        source_database=args.source_database,
        case_path=args.case_path,
        env_file=args.env_file,
        output_dir=args.output_dir,
        receipt_id=args.receipt_id,
        association_id=args.association_id,
        existing_arm=args.existing_arm,
        run_full_answer=bool(args.run_full_answer),
        deadline_seconds=float(args.deadline_seconds),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "preflight": payload["public_preflight"]["result_status"],
                "provider_http_attempts": payload["public_preflight"]["provider_counts"]["http_attempts"],
                "full_answer_control": bool(args.run_full_answer),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
