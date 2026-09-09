"""Run one frozen, same-chapter Q1→Q2 learning diagnostic matrix.

This runner is intentionally separate from the frozen N11d exact-revisit
pilot.  It makes one normal public Q1 learning request, freezes the resulting
SQLite state once, and runs every Q2 variant from a fresh clone of that exact
state.  The paired masked arm hides only the association created by that Q1.

It is a diagnostic harness, not a gold evaluator.  It never reads a gold
manifest, expected answer, expected Episode, or source span hint.  All Q2
requests use the public ``QueryEngine.query`` entry point with
``contextual_learning=False``.  No caller-supplied vector override is used:
the ordinary request-local coordinator owns duplicate-text reuse, while an
automatic same-text revisit may correctly require no new embedding at all.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

from benchmarks.run_q1_q2_diagnostic_pilot import (
    FinalizerObserver,
    ProviderLedger,
    _call_query,
    _clone_sqlite,
    _mask_engine_edge,
    _new_receipt_material,
    _open_app,
    _receipt_and_edge_ids,
    _sha256_file,
    _source_evidence_stats,
    _utc_now,
    _write_json,
)
from memory_demo.embeddings import normalize_query_text


MATRIX_SCHEMA = "aevnema.v3.same_chapter_learning_matrix.v1"
FULL_LOCAL_NAME = "same_chapter_learning_matrix.full_local.json"
REPORT_NAME = "same_chapter_learning_matrix.report.md"
REQUIRED_KINDS = frozenset(
    {
        "same_text",
        "paraphrase",
        "partial_clue",
        "near_neighbor_counterexample",
    }
)
FORBIDDEN_HINT_KEYS = frozenset(
    {
        "gold",
        "answer",
        "expected",
        "expected_answer",
        "expected_episode",
        "expected_episode_id",
        "episode_id",
        "evidence",
        "evidence_id",
        "hint",
        "label",
        "target_episode",
        "target_episode_id",
        "hidden_target",
        "source_span",
        "source_locator",
    }
)
EVIDENCE_PHASES = (
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
ANSWER_PURPOSES = frozenset({"answer_generation", "answer_correction"})
AUDIT_PURPOSES = frozenset(
    {"answer_evidence_audit", "event_continuity_audit"}
)


@dataclass(frozen=True, slots=True)
class Q2Variant:
    identifier: str
    kind: str
    text: str
    context_mode: str
    reference_basis: str


@dataclass(frozen=True, slots=True)
class FrozenMatrixCase:
    q1_text: str
    domain: str
    scope_hash: str
    variants: tuple[Q2Variant, ...]
    manifest_path: Path
    manifest_sha256: str


def _scope_hash(label: str) -> str:
    normalized = str(label or "").strip()
    if not normalized:
        raise ValueError("contextual revisit scope is required")
    digest = sha256(
        b"aevnema/v3/same-chapter-learning-matrix/v1\0"
        + normalized.encode("utf-8")
    ).hexdigest()
    return "pilot-scope:sha256:" + digest


def _read_object(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("matrix manifest must be a JSON object")
    return loaded


def _forbidden_key_path(value: object, path: str = "$") -> str | None:
    """Return a forbidden hint-key path without inspecting ordinary text."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in FORBIDDEN_HINT_KEYS:
                return f"{path}.{key}"
            nested = _forbidden_key_path(child, f"{path}.{key}")
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, child in enumerate(value):
            nested = _forbidden_key_path(child, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def load_frozen_matrix_case(path: Path) -> FrozenMatrixCase:
    """Validate the frozen matrix without introducing a scoring target."""

    payload = _read_object(path)
    if payload.get("schema") != MATRIX_SCHEMA:
        raise ValueError("unexpected same-chapter matrix schema")
    if payload.get("formal_scoring_eligible") is not False:
        raise ValueError("matrix must explicitly be non-scorable")
    if payload.get("promotion_prohibited") is not True:
        raise ValueError("matrix must explicitly prohibit promotion")
    forbidden = _forbidden_key_path(payload)
    if forbidden is not None:
        raise ValueError(f"matrix must not contain a gold/answer/hint field: {forbidden}")

    q1 = payload.get("q1")
    if not isinstance(q1, Mapping):
        raise ValueError("matrix needs a q1 object")
    q1_text = str(q1.get("text", "")).strip()
    domain = str(q1.get("contextual_domain", "")).strip()
    scope_label = str(q1.get("contextual_revisit_scope", "")).strip()
    if not q1_text or not domain or not scope_label:
        raise ValueError("matrix q1 needs text, domain, and scope")

    raw_variants = payload.get("q2_variants")
    if not isinstance(raw_variants, list):
        raise ValueError("matrix needs q2_variants")
    variants: list[Q2Variant] = []
    identifiers: set[str] = set()
    kinds: set[str] = set()
    normalized_texts: set[str] = set()
    normalized_q1 = normalize_query_text(q1_text)
    for raw in raw_variants:
        if not isinstance(raw, Mapping):
            raise ValueError("every Q2 variant must be an object")
        identifier = str(raw.get("id", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        text = str(raw.get("text", "")).strip()
        context_mode = str(raw.get("context_mode", "absent")).strip()
        reference_basis = str(raw.get("reference_basis", "query_text_only")).strip()
        if not identifier or not kind or not text:
            raise ValueError("every Q2 variant needs id, kind, and text")
        if identifier in identifiers or kind in kinds:
            raise ValueError("Q2 variant ids and kinds must be unique")
        if kind not in REQUIRED_KINDS:
            raise ValueError(f"unsupported Q2 variant kind: {kind}")
        if context_mode != "absent" or reference_basis != "query_text_only":
            raise ValueError("Q2 variants must use query text only and no prior context")
        if kind == "partial_clue":
            prohibited = {
                "context_text",
                "antecedent",
                "referent",
                "hidden_target",
                "prior_turns",
            }.intersection(map(str, raw))
            if prohibited:
                raise ValueError(
                    "partial clue may not carry contextual or hidden-reference fields"
                )
        normalized = normalize_query_text(text)
        if not normalized or normalized in normalized_texts:
            raise ValueError("Q2 texts must be non-empty and distinct after normalization")
        if kind == "same_text":
            if normalized != normalized_q1:
                raise ValueError("same_text Q2 must equal Q1 after normalization")
        elif normalized == normalized_q1:
            raise ValueError("only same_text may equal Q1 after normalization")
        identifiers.add(identifier)
        kinds.add(kind)
        normalized_texts.add(normalized)
        variants.append(
            Q2Variant(
                identifier=identifier,
                kind=kind,
                text=text,
                context_mode=context_mode,
                reference_basis=reference_basis,
            )
        )
    if kinds != REQUIRED_KINDS:
        raise ValueError("matrix must include each required Q2 variant exactly once")
    return FrozenMatrixCase(
        q1_text=q1_text,
        domain=domain,
        scope_hash=_scope_hash(scope_label),
        variants=tuple(variants),
        manifest_path=path.resolve(),
        manifest_sha256=_sha256_file(path),
    )


def _provider_partition(record: Mapping[str, object]) -> dict[str, object]:
    provider = record.get("provider")
    observations = (
        provider.get("observations", []) if isinstance(provider, Mapping) else []
    )
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in observations:
        if not isinstance(item, Mapping):
            continue
        purpose = str(item.get("purpose", ""))
        if purpose in ANSWER_PURPOSES:
            group = "answer"
        elif purpose in AUDIT_PURPOSES:
            group = "audit"
        else:
            group = "evidence_acquisition"
        grouped[group].append(item)

    def summarize(items: list[Mapping[str, object]]) -> dict[str, object]:
        succeeded = {"success", "succeeded"}
        return {
            "http_attempts": len(items),
            "sent": sum(bool(item.get("sent")) for item in items),
            "succeeded": sum(item.get("status") in succeeded for item in items),
            "failed_or_rejected": sum(
                item.get("status") not in succeeded for item in items
            ),
            "fallbacks_into_group": 0,
            "provider_total_ms": round(
                sum(float(item.get("total_ms", 0.0) or 0.0) for item in items), 3
            ),
            "attempts": [
                {
                    key: item.get(key)
                    for key in (
                        "role",
                        "purpose",
                        "requested_model",
                        "actual_model",
                        "status",
                        "total_ms",
                        "network_ms",
                        "error_class",
                    )
                }
                for item in items
            ],
        }

    result = {name: summarize(grouped[name]) for name in (
        "evidence_acquisition", "answer", "audit"
    )}
    fallbacks = provider.get("fallbacks", []) if isinstance(provider, Mapping) else []
    result["fallbacks"] = list(fallbacks) if isinstance(fallbacks, list) else []
    return result


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _answer_checkpoint_summary(log_dir: Path) -> dict[str, object]:
    """Read only the local Q2 boundary receipts; missing values stay absent."""

    rows: list[Mapping[str, object]] = []
    parse_errors: list[str] = []
    files = sorted(log_dir.glob("*.answer-evidence.jsonl"))
    for path in files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                parse_errors.append(f"{path.name}:{line_number}: {error}")
                continue
            if isinstance(value, Mapping):
                rows.append(value)
    stages = [str(row.get("stage", "")) for row in rows if row.get("stage")]
    stage_times = {
        stage: next(
            (row.get("timestamp") for row in rows if row.get("stage") == stage),
            "not_observed",
        )
        for stage in ("answer_input", "initial_answer", "audit_result", "audit_exception")
    }
    answer_started = _parse_timestamp(stage_times["answer_input"])
    initial_answer = _parse_timestamp(stage_times["initial_answer"])
    audit_terminal = _parse_timestamp(
        stage_times["audit_result"]
        if stage_times["audit_result"] != "not_observed"
        else stage_times["audit_exception"]
    )
    source_rows: list[dict[str, object]] = []
    audit_valid: bool | str = "not_observed"
    for row in rows:
        if row.get("stage") == "answer_input":
            selected = row.get("selected_evidence")
            if isinstance(selected, list):
                for evidence in selected:
                    if isinstance(evidence, Mapping):
                        source_rows.append(
                            {
                                "episode_id": evidence.get("episode_id"),
                                "source_key": evidence.get("source_key"),
                                "source_evidence_delivery": evidence.get(
                                    "source_evidence_delivery"
                                ),
                                "source_evidence_delivery_reason": evidence.get(
                                    "source_evidence_delivery_reason"
                                ),
                                "source_excerpt_hash": evidence.get(
                                    "source_excerpt_hash"
                                ),
                                "source_excerpt_characters": len(
                                    str(evidence.get("source_excerpt", ""))
                                ),
                            }
                        )
        if row.get("stage") == "audit_result":
            audit = row.get("audit")
            if isinstance(audit, Mapping) and isinstance(audit.get("valid"), bool):
                audit_valid = bool(audit["valid"])
    return {
        "files": [path.name for path in files],
        "stages": stages,
        "stage_timestamps": stage_times,
        "answer_input_to_initial_answer_ms": (
            round((initial_answer - answer_started).total_seconds() * 1000.0, 3)
            if answer_started is not None and initial_answer is not None
            else "not_observed"
        ),
        "initial_answer_to_audit_terminal_ms": (
            round((audit_terminal - initial_answer).total_seconds() * 1000.0, 3)
            if audit_terminal is not None and initial_answer is not None
            else "not_observed"
        ),
        "selected_source_evidence": source_rows,
        "audit_valid": audit_valid,
        "parse_errors": parse_errors,
    }


def _edge_ids_from_result(result: Mapping[str, object]) -> list[int]:
    found: set[int] = set()

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                if child_key in {"attached_edges", "association_ids"} and isinstance(child, list):
                    for item in child:
                        try:
                            found.add(int(item))
                        except (TypeError, ValueError):
                            pass
                elif child_key in {"association_id", "edge_id"}:
                    try:
                        found.add(int(child))
                    except (TypeError, ValueError):
                        pass
                elif key in {"exact_revisit", "contextual_association", "evidence_slot_trace"}:
                    visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)

    for top_level in ("exact_revisit", "contextual_association", "evidence_slot_trace"):
        visit(result.get(top_level), top_level)
    return sorted(found)


def _runtime_manifest_state(value: object) -> str | None:
    """Read the public manifest state without assuming its representation.

    The repository returns a frozen ``ContextualRevisitRuntimeManifest`` record
    during a live run, while the durable full-local trace serializes that same
    record as an object.  Both representations are part of this harness's
    public observation surface; neither should be silently discarded merely
    because it is not a mapping.
    """

    if isinstance(value, Mapping):
        state = value.get("state")
    else:
        state = getattr(value, "state", None)
    return str(state) if isinstance(state, str) else None


def _ready_receipt_rows(new_material: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return newly persisted, publicly loadable ready receipt rows."""

    rows = new_material.get("new_receipts", [])
    if not isinstance(rows, list):
        return []
    return [
        item
        for item in rows
        if isinstance(item, Mapping)
        and isinstance(item.get("receipt"), Mapping)
        and item["receipt"].get("status") == "ready"
        and item.get("association_id_is_new") is True
        and _runtime_manifest_state(item.get("runtime_manifest")) == "ready"
    ]


def _q2_summary(
    record: Mapping[str, object],
    *,
    log_dir: Path,
    this_run_edge_id: int,
) -> dict[str, object]:
    """Summarise only the current Q2 record, without an answer correctness claim."""

    checkpoints = _answer_checkpoint_summary(log_dir)
    if record.get("status") != "completed" or not isinstance(record.get("result"), Mapping):
        error = record.get("error")
        return {
            "run_status": str(record.get("status", "unknown")),
            "elapsed_ms": record.get("elapsed_ms"),
            "error": error if isinstance(error, Mapping) else "not_observed",
            "provider_stages": _provider_partition(record),
            "checkpoints": checkpoints,
            "ordinary_answer_status": "audit_unavailable",
            "final_delivery_status": "not_observed_due_to_terminal_exception",
            "this_run_edge_participated": "not_observed_due_to_terminal_exception",
            "actual_edge_path": "not_observed_due_to_terminal_exception",
        }
    result = record["result"]
    timings = result.get("timings")
    phases = (
        timings.get("phases_seconds", {}) if isinstance(timings, Mapping) else {}
    )
    phases = dict(phases) if isinstance(phases, Mapping) else {}
    evidence_ms = round(
        sum(float(phases.get(name, 0.0) or 0.0) for name in EVIDENCE_PHASES)
        * 1000.0,
        3,
    )
    edge_path = _edge_ids_from_result(result)
    exact = result.get("exact_revisit")
    modules = (
        exact.get("executed_modules") if isinstance(exact, Mapping) else "not_observed"
    )
    skipped = (
        [name for name, value in modules.items() if value is False]
        if isinstance(modules, Mapping)
        else []
    )
    audit_valid = checkpoints["audit_valid"]
    audit_status = (
        "audit_valid"
        if audit_valid is True
        else "audit_invalid"
        if audit_valid is False
        else "audit_unavailable"
    )
    evidence_only = result.get("evidence_result")
    if isinstance(evidence_only, Mapping):
        execution = evidence_only.get("executed_modules")
        execution = execution if isinstance(execution, Mapping) else {}
        participation = evidence_only.get("participation")
        participation = participation if isinstance(participation, Mapping) else {}
        return {
            "run_status": "completed",
            "execution_profile": result.get("execution_profile", "live_evidence"),
            "elapsed_ms": record.get("elapsed_ms"),
            "evidence_acquisition_ms": evidence_ms,
            "phase_seconds": phases,
            "answer_generation_phase_seconds": "not_run_evidence_only",
            "audit_engine_phase_seconds": "not_run_evidence_only",
            "provider_stages": _provider_partition(record),
            "candidate_episode_ids": list(result.get("candidate_episode_ids", [])),
            "delivered_episode_ids": [
                item.get("episode_id")
                for item in evidence_only.get("materialized_source_refs", [])
                if isinstance(item, Mapping)
            ],
            "evidence_state": evidence_only.get("evidence_state"),
            "runtime_coverage_status": evidence_only.get(
                "runtime_coverage_status", "not_observed"
            ),
            "query_vector_bundle": result.get("query_vector_bundle", "not_observed"),
            "exact_revisit_modules": modules,
            "work_explicitly_skipped": list(
                evidence_only.get("actual_skipped_modules", [])
            ),
            "actual_edge_path": edge_path,
            "this_run_edge_participated": participation.get(
                "edge_state", "not_observed"
            ),
            "checkpoints": checkpoints,
            "ordinary_answer_status": "not_run_evidence_only",
            "answer_cache_used": False,
            "q2_learning_requested": False,
            "association_usage_recorded": result.get(
                "association_usage_recorded", "not_observed"
            ),
            "evidence_materialized": evidence_only.get(
                "delivery_states", {}
            ).get("evidence_materialized")
            if isinstance(evidence_only.get("delivery_states"), Mapping)
            else "not_observed",
            "executed_modules": dict(execution),
        }
    return {
        "run_status": "completed",
        "elapsed_ms": record.get("elapsed_ms"),
        "evidence_acquisition_ms": evidence_ms,
        "phase_seconds": phases,
        "answer_generation_phase_seconds": phases.get("answer_generation", "not_observed"),
        "audit_engine_phase_seconds": "not_observed_separate_phase",
        "provider_stages": _provider_partition(record),
        "candidate_episode_ids": list(result.get("candidate_episode_ids", [])),
        "delivered_episode_ids": list(result.get("episode_ids", [])),
        "query_vector_bundle": result.get("query_vector_bundle", "not_observed"),
        "exact_revisit_modules": modules,
        "work_explicitly_skipped": skipped,
        "actual_edge_path": edge_path,
        "this_run_edge_participated": this_run_edge_id in edge_path,
        "checkpoints": checkpoints,
        "ordinary_answer_status": audit_status,
        "answer_cache_used": (
            result.get("rerank_trace", {}).get("answer_cache_used")
            if isinstance(result.get("rerank_trace"), Mapping)
            else "not_observed"
        ),
        "q2_learning_requested": False,
    }


def _arm_path(output_dir: Path, variant: Q2Variant, condition: str) -> Path:
    return output_dir / f"q2_{variant.identifier}__{condition}.full_local.json"


def _write_arm(
    output_dir: Path,
    *,
    variant: Q2Variant,
    condition: str,
    snapshot: Mapping[str, object],
    this_run_edge_id: int | str,
    status: str,
    arm: Mapping[str, object] | None = None,
    reason: str | None = None,
) -> None:
    _write_json(
        _arm_path(output_dir, variant, condition),
        {
            "schema": MATRIX_SCHEMA,
            "status": status,
            "written_at": _utc_now(),
            "variant": {
                "id": variant.identifier,
                "kind": variant.kind,
                "text": variant.text,
                "context_mode": variant.context_mode,
                "reference_basis": variant.reference_basis,
            },
            "condition": condition,
            "snapshot": dict(snapshot),
            "this_run_edge_id": this_run_edge_id,
            "reason": reason,
            "arm": dict(arm) if isinstance(arm, Mapping) else {
                "observation": "not_observed_yet"
            },
        },
    )


def _run_q2_arm(
    *,
    variant: Q2Variant,
    condition: str,
    source_snapshot: Path,
    output_dir: Path,
    env_file: Path,
    domain: str,
    scope_hash: str,
    deadline_seconds: float,
    this_run_edge_id: int,
    snapshot: Mapping[str, object],
    evidence_only: bool,
    frozen_plan: dict[str, object] | None = None,
    query_vector_bundle: object | None = None,
) -> dict[str, object]:
    database = output_dir / "work" / f"{variant.identifier}__{condition}.sqlite"
    _clone_sqlite(source_snapshot, database)
    log_dir = output_dir / "logs" / variant.identifier / condition
    _write_arm(
        output_dir,
        variant=variant,
        condition=condition,
        snapshot=snapshot,
        this_run_edge_id=this_run_edge_id,
        status="q2_arm_pending",
    )
    app, config = _open_app(env_file, database, log_dir)
    engine = app.query_engine(config=config)
    if condition == "this_run_edge_masked":
        _mask_engine_edge(engine, this_run_edge_id)
    record = _call_query(
        engine,
        variant.text,
        domain=domain,
        scope_hash=scope_hash,
        deadline_seconds=deadline_seconds,
        frozen_plan=frozen_plan,
        query_vector_bundle=query_vector_bundle,
        strict_vector_bundle=query_vector_bundle is not None,
        contextual_learning=False,
        generate_answer=not evidence_only,
        stop_after="evidence" if evidence_only else None,
    )
    arm = {
        "variant_id": variant.identifier,
        "variant_kind": variant.kind,
        "condition": condition,
        "database": str(database),
        "execution_mode": (
            "public_query_frozen_input_evidence_only_no_q2_learning_no_override"
            if evidence_only and frozen_plan is not None
            else "public_query_evidence_only_no_q2_learning_no_override"
            if evidence_only
            else "public_query_no_q2_learning_no_override"
        ),
        "frozen_input": {
            "enabled": frozen_plan is not None,
            "plan_id": (
                str(frozen_plan.get("plan_id", "not_observed"))
                if frozen_plan is not None
                else "not_observed"
            ),
            "vector_bundle_supplied": query_vector_bundle is not None,
        },
        "edge_mask": {
            "enabled": condition == "this_run_edge_masked",
            "hidden_association_id": this_run_edge_id
            if condition == "this_run_edge_masked"
            else None,
            "scope": "this Q1 association only; base retrieval remains available",
        },
        "record": record,
        "summary": _q2_summary(
            record,
            log_dir=log_dir,
            this_run_edge_id=this_run_edge_id,
        ),
    }
    _write_arm(
        output_dir,
        variant=variant,
        condition=condition,
        snapshot=snapshot,
        this_run_edge_id=this_run_edge_id,
        status="q2_arm_terminal",
        arm=arm,
    )
    return arm


def _empty_arms(case: FrozenMatrixCase, reason: str) -> dict[str, object]:
    return {
        f"{variant.identifier}__{condition}": {
            "variant_id": variant.identifier,
            "condition": condition,
            "not_run_reason": reason,
            "summary": {"run_status": "not_run", "reason": reason},
        }
        for variant in case.variants
        for condition in ("edge_available", "this_run_edge_masked")
    }


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    q1 = payload.get("q1", {})
    arms = payload.get("arms", {})
    arms = arms if isinstance(arms, Mapping) else {}
    lines = [
        "# Same-chapter different-requirement learning diagnostic",
        "",
        "Diagnostic only. No gold was loaded; no row is a correctness, recall, or generalization score.",
        "",
        "## Q1 boundary",
        "",
        f"- Q1 status: `{q1.get('record', {}).get('status') if isinstance(q1, Mapping) and isinstance(q1.get('record'), Mapping) else 'not_observed'}`",
        f"- Public finalizer calls: `{q1.get('finalizer_observer', {}).get('finalizer_calls') if isinstance(q1, Mapping) and isinstance(q1.get('finalizer_observer'), Mapping) else 'not_observed'}`",
        f"- New receipt IDs: `{q1.get('new_material', {}).get('new_receipt_ids') if isinstance(q1, Mapping) and isinstance(q1.get('new_material'), Mapping) else 'not_observed'}`",
        f"- Q1 provenance: `{q1.get('continuation_provenance', {}).get('mode', 'new_public_q1') if isinstance(q1, Mapping) and isinstance(q1.get('continuation_provenance'), Mapping) else 'new_public_q1'}`",
        f"- This-run masked edge: `{payload.get('this_run_edge_id', 'not_observed')}`",
        "",
        "## Q2 observations",
        "",
        "| Variant | Condition | Run | Edge participated | Delivered | Evidence ms | Provider HTTP (evidence / answer / audit) | Audit |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for key, arm in arms.items():
        if not isinstance(arm, Mapping):
            continue
        summary = arm.get("summary", {})
        summary = summary if isinstance(summary, Mapping) else {}
        provider = summary.get("provider_stages", {})
        provider = provider if isinstance(provider, Mapping) else {}
        calls = []
        for stage in ("evidence_acquisition", "answer", "audit"):
            item = provider.get(stage, {})
            calls.append(str(item.get("http_attempts", 0)) if isinstance(item, Mapping) else "0")
        lines.append(
            "| {variant} | {condition} | {status} | {edge} | {delivered} | {time} | {calls} | {audit} |".format(
                variant=arm.get("variant_id", key),
                condition=arm.get("condition", "not_observed"),
                status=summary.get("run_status", "not_run"),
                edge=summary.get("this_run_edge_participated", False),
                delivered=summary.get("delivered_episode_ids", []),
                time=summary.get("evidence_acquisition_ms", "not_observed"),
                calls=" / ".join(calls),
                audit=summary.get("ordinary_answer_status", "not_observed"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The paired rows preserve whether the this-run edge actually appeared in the result path, source-bound delivery, provider calls, and Q2 timing. They do not automatically label an answer ‘correct’: ordinary-search success, edge-used/no-benefit, same-delivery acceleration, and degradation remain separately reviewable observations. A timeout, audit exception, no-edge result, or no-benefit result remains in its own arm artifact and is never retried for a better result.",
            "",
            "The partial-clue row has `context_mode=absent` and `reference_basis=query_text_only`; no prior turn, answer, expected Episode, or hidden referent is supplied. Same-text automatic exact reuse may consume no new embedding, which is recorded as not-required rather than a cache hit. Other requests rely on their own request-local embedding coordinator; no per-edge cloud call or supplied vector override is used.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_saved_q1_continuation(
    completed_q1_run: Path,
    *,
    case: FrozenMatrixCase,
    source_database: Path,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    """Validate an unconsumed public Q1 before running its first Q2 matrix.

    This is deliberately a harness-only continuation path.  It never writes to
    the preserved Q1 directory, calls no model, and requires that every Q2 arm
    in the original trace is still ``not_run``.  The copied Q1 SQLite state is
    subsequently revalidated against the static pre-Q1 source before any Q2
    clone is made.
    """

    completed_q1_run = completed_q1_run.resolve()
    trace_path = completed_q1_run / FULL_LOCAL_NAME
    q1_database = completed_q1_run / "work" / "q1_learning.sqlite"
    if not trace_path.is_file() or not q1_database.is_file():
        raise FileNotFoundError(
            "continued Q1 requires its preserved full-local trace and q1_learning.sqlite"
        )
    prior = _read_object(trace_path)
    if prior.get("schema") != MATRIX_SCHEMA:
        raise ValueError("continued Q1 trace has an unexpected matrix schema")
    prior_case = prior.get("case")
    prior_source = prior.get("source_database")
    if not isinstance(prior_case, Mapping) or not isinstance(prior_source, Mapping):
        raise ValueError("continued Q1 trace lacks its frozen case or source binding")
    if prior_case.get("manifest_sha256") != case.manifest_sha256:
        raise ValueError("continued Q1 used a different frozen matrix manifest")
    if prior_case.get("q1_text") != case.q1_text:
        raise ValueError("continued Q1 text differs from the frozen matrix")
    if prior_case.get("domain") != case.domain or prior_case.get("scope_hash") != case.scope_hash:
        raise ValueError("continued Q1 domain or scope differs from the frozen matrix")
    if prior_source.get("sha256") != _sha256_file(source_database):
        raise ValueError("continued Q1 source database differs from the static source snapshot")
    q1 = prior.get("q1")
    if not isinstance(q1, Mapping) or not isinstance(q1.get("record"), Mapping):
        raise ValueError("continued Q1 trace lacks a terminal Q1 record")
    finalizer = q1.get("finalizer_observer")
    if not isinstance(finalizer, Mapping):
        raise ValueError("continued Q1 trace lacks its public-finalizer observation")
    arms = prior.get("arms")
    if not isinstance(arms, Mapping) or not arms:
        raise ValueError("continued Q1 trace lacks Q2 arm state")
    for arm in arms.values():
        summary = arm.get("summary") if isinstance(arm, Mapping) else None
        if not isinstance(summary, Mapping) or summary.get("run_status") != "not_run":
            raise ValueError("continued Q1 already has a Q2 observation and cannot be rerun")
    return (
        dict(q1),
        q1_database,
        {
            "mode": "continue_preserved_public_q1",
            "prior_output_dir": str(completed_q1_run),
            "prior_full_local_sha256": _sha256_file(trace_path),
            "prior_q1_database_sha256": _sha256_file(q1_database),
            "prior_q2_arms": "all_not_run",
            "writes_to_prior_directory": False,
            "new_q1_model_calls": 0,
        },
    )


def run_same_chapter_learning_matrix(
    *,
    source_database: Path,
    matrix_manifest: Path,
    output_dir: Path,
    env_file: Path,
    deadline_seconds: float = 120.0,
    continue_from_q1_run: Path | None = None,
    evidence_only: bool = False,
    defer_q2: bool = False,
) -> dict[str, object]:
    """Run the frozen matrix once, preserving each observed terminal arm.

    ``continue_from_q1_run`` is only for a preserved public Q1 whose Q2 arms
    never started because of a harness fault.  It validates and clones that
    Q1's durable state; it does not issue a replacement Q1 or mutate the
    original run.
    """

    source_database = source_database.resolve()
    matrix_manifest = matrix_manifest.resolve()
    output_dir = output_dir.resolve()
    env_file = env_file.resolve()
    continue_from_q1_run = (
        continue_from_q1_run.resolve() if continue_from_q1_run is not None else None
    )
    if output_dir.exists():
        raise FileExistsError("matrix output directory must be new")
    if not all(path.is_file() for path in (source_database, matrix_manifest, env_file)):
        raise FileNotFoundError("source database, matrix manifest, and env file are required")
    if any(
        source_database.with_name(source_database.name + suffix).exists()
        for suffix in ("-wal", "-shm")
    ):
        raise ValueError("matrix source database must be a static snapshot without WAL/SHM")
    case = load_frozen_matrix_case(matrix_manifest)
    continued_q1: dict[str, object] | None = None
    continued_q1_database: Path | None = None
    continuation_provenance: dict[str, object] | None = None
    if continue_from_q1_run is not None:
        (
            continued_q1,
            continued_q1_database,
            continuation_provenance,
        ) = _load_saved_q1_continuation(
            continue_from_q1_run,
            case=case,
            source_database=source_database,
        )
    output_dir.mkdir(parents=True)
    work_dir = output_dir / "work"
    q1_database = work_dir / "q1_learning.sqlite"
    _clone_sqlite(
        continued_q1_database if continued_q1_database is not None else source_database,
        q1_database,
    )

    started_at = _utc_now()
    started = monotonic()
    app, config = _open_app(env_file, q1_database, output_dir / "logs" / "q1")
    if continued_q1 is None:
        before_receipts, before_edges = _receipt_and_edge_ids(app)
    else:
        baseline_database = work_dir / "pre_q1_static_source.sqlite"
        _clone_sqlite(source_database, baseline_database)
        baseline_app, _ = _open_app(
            env_file, baseline_database, output_dir / "logs" / "pre_q1_static_source"
        )
        before_receipts, before_edges = _receipt_and_edge_ids(baseline_app)
    observer = FinalizerObserver()
    engine = None
    if continued_q1 is None:
        engine = app.query_engine(config=config)
        finalizer = engine.contextual_learning_finalizer
        if not callable(finalizer):
            raise RuntimeError("Q1 engine lacks the public contextual finalizer")
        engine.contextual_learning_finalizer = observer.wrap(finalizer)
    source_evidence = _source_evidence_stats(app)

    def render(status: str, *, q1: Mapping[str, object], arms: Mapping[str, object], snapshot: Mapping[str, object], edge: int | str) -> dict[str, object]:
        return {
            "schema": MATRIX_SCHEMA,
            "status": status,
            "started_at": started_at,
            "finished_at": _utc_now() if status == "diagnostic_complete" else "not_observed",
            "elapsed_ms": round((monotonic() - started) * 1000.0, 3),
            "formal_scoring": {
                "status": "not_scored_unapproved_gold",
                "formal_score": False,
                "gold_loaded": False,
                "promotion_prohibited": True,
                "canary_prohibited": True,
            },
            "classification": "same_chapter_different_requirement_diagnostic",
            "q2_execution_profile": (
                "evidence_only" if evidence_only else "full_answer"
            ),
            "source_database": {
                "path": str(source_database),
                "sha256": _sha256_file(source_database),
                "source_evidence_eligibility": source_evidence,
            },
            "case": {
                "manifest_path": str(case.manifest_path),
                "manifest_sha256": case.manifest_sha256,
                "q1_text": case.q1_text,
                "domain": case.domain,
                "scope_hash": case.scope_hash,
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
            },
            "pilot_config": {
                "association_cue_enabled": config.retrieval.association_cue_enabled,
                "association_cue_fast_path_enabled": config.retrieval.association_cue_fast_path_enabled,
                "contextual_association_enabled": config.retrieval.contextual_association_enabled,
                "contextual_association_shadow": config.retrieval.contextual_association_shadow,
                "contextual_promotion_enabled": config.retrieval.contextual_promotion_enabled,
                "followup_planning_mode": config.retrieval.followup_planning_mode,
                "growth_max_rounds": config.retrieval.growth_max_rounds,
                "model_timeout_seconds": config.model.timeout_seconds,
                "model_max_retries": config.model.max_retries,
            },
            "q1": dict(q1),
            "learned_snapshot": dict(snapshot),
            "this_run_edge_id": edge,
            "arms": dict(arms),
            "cache_controls": {
                "current_ordinary_query_embedding_control": "not_part_of_this_matrix; historical same-Q2 control is request-local whole-question vector reuse, not a source-validated evidence cache",
                "ordinary_source_validated_evidence_cache": "not_run_independent_control",
            },
            "no_retry_policy": "Each Q1 and each Q2 condition is attempted at most once. Failures, no-benefit observations, and timeouts remain preserved.",
        }

    initial_q1 = {
        "record": {"status": "not_observed_q1_in_progress"},
        "finalizer_observer": observer.export(started_at=started),
        "learning_status": "not_observed",
        "learning_reason": "not_observed",
        "new_material": {"new_receipt_ids": "not_observed", "new_association_ids": "not_observed"},
    }
    snapshot: dict[str, object] = {"status": "not_observed_q1_in_progress"}
    payload = render(
        "in_progress_q1", q1=initial_q1,
        arms=_empty_arms(case, "Q1 has not reached a terminal state"),
        snapshot=snapshot, edge="not_observed",
    )
    _write_json(output_dir / FULL_LOCAL_NAME, payload)
    _write_report(output_dir / REPORT_NAME, payload)

    if continued_q1 is None:
        if engine is None:
            raise RuntimeError("new Q1 requires a public query engine")
        q1_record = _call_query(
            engine,
            case.q1_text,
            domain=case.domain,
            scope_hash=case.scope_hash,
            deadline_seconds=deadline_seconds,
            contextual_learning=True,
            learning_request_id="same-chapter-matrix-"
            + sha256(case.q1_text.encode("utf-8")).hexdigest()[:20],
        )
    else:
        q1_record = dict(continued_q1["record"])
    new_material = _new_receipt_material(
        app, before_receipts=before_receipts, before_edges=before_edges
    )
    q1_result = q1_record.get("result")
    learning = (
        q1_result.get("contextual_learning", {})
        if isinstance(q1_result, Mapping)
        else {}
    )
    learning = learning if isinstance(learning, Mapping) else {}
    if continued_q1 is None:
        finalizer_observation: Mapping[str, object] = observer.export(started_at=started)
        finalizer_calls = observer.finalizer_calls
        q1 = {
            "record": q1_record,
            "finalizer_observer": finalizer_observation,
            "learning_status": learning.get("status"),
            "learning_reason": learning.get("reason"),
            "new_material": new_material,
        }
    else:
        saved_finalizer = continued_q1.get("finalizer_observer")
        if not isinstance(saved_finalizer, Mapping):
            raise ValueError("continued Q1 lacks a saved public-finalizer observation")
        raw_finalizer_calls = saved_finalizer.get("finalizer_calls")
        if isinstance(raw_finalizer_calls, bool):
            finalizer_calls = 0
        else:
            try:
                finalizer_calls = int(raw_finalizer_calls)
            except (TypeError, ValueError):
                finalizer_calls = 0
        finalizer_observation = dict(saved_finalizer)
        q1 = dict(continued_q1)
        q1.update(
            {
                "record": q1_record,
                "finalizer_observer": finalizer_observation,
                "learning_status": learning.get("status", q1.get("learning_status")),
                "learning_reason": learning.get("reason", q1.get("learning_reason")),
                "new_material": new_material,
                "continuation_provenance": continuation_provenance,
            }
        )
    ready_rows = _ready_receipt_rows(new_material)
    legal_q1 = (
        q1_record.get("status") == "completed"
        and finalizer_calls == 1
        and len(ready_rows) == 1
    )
    if not legal_q1:
        reason = (
            "Q1 did not produce exactly one new ready receipt/edge/runtime manifest "
            "through the public finalizer"
        )
        arms = _empty_arms(case, reason)
        snapshot = {"status": "not_created", "reason": reason}
        for variant in case.variants:
            for condition in ("edge_available", "this_run_edge_masked"):
                _write_arm(
                    output_dir,
                    variant=variant,
                    condition=condition,
                    snapshot=snapshot,
                    this_run_edge_id="not_observed_single_this_run_edge",
                    status="q2_arm_not_run",
                    reason=reason,
                    arm=arms[f"{variant.identifier}__{condition}"],
                )
        payload = render(
            "diagnostic_complete", q1=q1, arms=arms, snapshot=snapshot,
            edge="not_observed_single_this_run_edge",
        )
        _write_json(output_dir / FULL_LOCAL_NAME, payload)
        _write_report(output_dir / REPORT_NAME, payload)
        return {"output_dir": str(output_dir), "full_local": str(output_dir / FULL_LOCAL_NAME), "q1_legal": False}

    this_run_edge_id = int(ready_rows[0]["receipt"]["association_id"])
    learned_snapshot = work_dir / "learned_snapshot.sqlite"
    _clone_sqlite(q1_database, learned_snapshot)
    snapshot = {
        "status": "frozen",
        "path": str(learned_snapshot),
        "sha256": _sha256_file(learned_snapshot),
        "basis": "one clone of the post-public-Q1 learning state; every Q2 condition is cloned from this file",
    }
    arms = _empty_arms(case, "Q2 not observed yet")
    payload = render("q1_terminal_q2_pending", q1=q1, arms=arms, snapshot=snapshot, edge=this_run_edge_id)
    _write_json(output_dir / FULL_LOCAL_NAME, payload)
    _write_report(output_dir / REPORT_NAME, payload)

    if defer_q2:
        # A fresh Q1 may be legal before the controlled Q2 stage has a
        # shared frozen input.  Preserve that state as terminal and let the
        # dedicated public frozen-evidence runner consume its snapshot; do
        # not spend one independent preparation per edge condition here.
        deferred_reason = (
            "Q2 deliberately deferred to the shared-frozen-input evidence runner; "
            "no Q2 request was issued by this Q1 execution"
        )
        arms = _empty_arms(case, deferred_reason)
        payload = render(
            "diagnostic_complete", q1=q1, arms=arms,
            snapshot=snapshot, edge=this_run_edge_id,
        )
        _write_json(output_dir / FULL_LOCAL_NAME, payload)
        _write_report(output_dir / REPORT_NAME, payload)
        return {
            "output_dir": str(output_dir),
            "full_local": str(output_dir / FULL_LOCAL_NAME),
            "report": str(output_dir / REPORT_NAME),
            "q1_legal": True,
            "q2_deferred": True,
            "this_run_edge_id": this_run_edge_id,
            "learned_snapshot_sha256": snapshot["sha256"],
        }

    for variant in case.variants:
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
                this_run_edge_id=this_run_edge_id,
                snapshot=snapshot,
                evidence_only=evidence_only,
            )
            arms[f"{variant.identifier}__{condition}"] = arm
            payload = render(
                "q1_terminal_q2_pending", q1=q1, arms=arms,
                snapshot=snapshot, edge=this_run_edge_id,
            )
            _write_json(output_dir / FULL_LOCAL_NAME, payload)
            _write_report(output_dir / REPORT_NAME, payload)

    snapshot_after = _sha256_file(learned_snapshot)
    snapshot["sha256_after_all_q2"] = snapshot_after
    snapshot["unchanged_after_all_q2"] = snapshot_after == snapshot["sha256"]
    payload = render(
        "diagnostic_complete", q1=q1, arms=arms,
        snapshot=snapshot, edge=this_run_edge_id,
    )
    _write_json(output_dir / FULL_LOCAL_NAME, payload)
    _write_report(output_dir / REPORT_NAME, payload)
    return {
        "output_dir": str(output_dir),
        "full_local": str(output_dir / FULL_LOCAL_NAME),
        "report": str(output_dir / REPORT_NAME),
        "q1_legal": True,
        "this_run_edge_id": this_run_edge_id,
        "learned_snapshot_sha256": snapshot["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a frozen same-chapter Q1→Q2 learning diagnostic matrix"
    )
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--deadline-seconds", type=float, default=120.0)
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help=(
            "stop each public Q2 after actual Source materialization; do not "
            "generate/audit answers, learn, grow associations, or mark edges used"
        ),
    )
    parser.add_argument(
        "--continue-from-q1-run",
        type=Path,
        help=(
            "preserved Q1 run whose Q2 arms are all not_run; validates and "
            "clones it without issuing a replacement Q1"
        ),
    )
    parser.add_argument(
        "--defer-q2",
        action="store_true",
        help=(
            "stop after a legal Q1 snapshot and leave every Q2 arm explicitly "
            "not_run for the shared-frozen-input evidence runner"
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_same_chapter_learning_matrix(
                source_database=args.source_database,
                matrix_manifest=args.matrix_manifest,
                output_dir=args.output_dir,
                env_file=args.env_file,
                deadline_seconds=args.deadline_seconds,
                continue_from_q1_run=args.continue_from_q1_run,
                evidence_only=args.evidence_only,
                defer_q2=args.defer_q2,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
