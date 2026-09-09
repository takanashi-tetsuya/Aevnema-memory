"""Export a strictly observed postmortem from a saved Q1→Q2 diagnostic case.

The exporter does not invoke a model, open a corpus database, or amend the
input case.  It correlates the case's provider ledger with its redacted JSONL
events and names absent fields ``not_observed`` instead of reconstructing
prompts, responses, budgets, evidence selection, or retry rationale.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "aevnema.v3.q1_q2_case_postmortem.v2"
FULL_LOCAL_NAME = "q1_q2_case.postmortem.full_local.json"
REPORT_NAME = "q1_q2_case.postmortem.report.md"
NOT_OBSERVED = "not_observed"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"JSON object required at {path}:{line_number}")
        rows.append(value)
    return rows


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _elapsed_ms(started_at: object, event_at: object) -> float | str:
    started = _timestamp(started_at)
    event = _timestamp(event_at)
    if started is None or event is None:
        return NOT_OBSERVED
    return round((event - started).total_seconds() * 1000.0, 3)


def _first(rows: list[dict[str, Any]], event: str, phase: str | None = None) -> dict[str, Any] | None:
    for row in rows:
        if row.get("event") != event:
            continue
        if phase is not None and row.get("phase") != phase:
            continue
        return row
    return None


def _request_index_for_observation(
    observation: Mapping[str, Any],
    rows: list[dict[str, Any]],
    consumed: set[int],
) -> int | None:
    """Locate one saved request without treating a batch as one HTTP call.

    A transport fallback keeps its parent's logical batch ID.  Pairing every
    ledger observation to the first event in that batch silently attributes the
    fallback's timing and response to the timed-out primary call.  Prefer the
    durable call/request ID, then use an unconsumed same-batch request whose
    recorded model agrees with the ledger.
    """

    call_id = observation.get("call_id")
    if isinstance(call_id, str) and call_id:
        for index, row in enumerate(rows):
            if row.get("event") == "llm_request" and row.get("request_id") == call_id:
                return index
    batch_id = observation.get("logical_batch_id")
    if not isinstance(batch_id, str):
        return None
    wanted_models = {
        value for value in (
            observation.get("actual_model"), observation.get("requested_model")
        ) if isinstance(value, str) and value
    }
    candidates = [
        index for index, row in enumerate(rows)
        if index not in consumed
        and row.get("event") == "llm_request"
        and row.get("logical_batch_id") == batch_id
    ]
    for index in candidates:
        payload = rows[index].get("payload")
        if isinstance(payload, Mapping) and payload.get("model") in wanted_models:
            return index
    return candidates[0] if candidates else None


def _first_in_window(
    rows: list[dict[str, Any]],
    *,
    start: int,
    end: int,
    event: str,
    phase: str | None = None,
) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    for index in range(start, end):
        row = rows[index]
        if row.get("event") == event and (phase is None or row.get("phase") == phase):
            return index, row
    return None, None


def _call_window_end(rows: list[dict[str, Any]], request_index: int) -> int:
    for index in range(request_index + 1, len(rows)):
        row = rows[index]
        # A fallback request is the next request in the event stream and
        # therefore closes the parent's HTTP window.  So does a request in a
        # new logical batch; otherwise a later fallback could be attributed
        # to an unrelated successful call.
        if row.get("event") == "llm_request":
            return index
    return len(rows)


def _fallback_before_request(
    rows: list[dict[str, Any]], request_index: int
) -> dict[str, Any] | str:
    payload = rows[request_index].get("payload")
    model = payload.get("model") if isinstance(payload, Mapping) else None
    for index in range(request_index - 1, -1, -1):
        row = rows[index]
        if row.get("event") == "fallback" and row.get("to_model") == model:
            return {
                "timestamp": row.get("timestamp", NOT_OBSERVED),
                "from_model": row.get("from_model", NOT_OBSERVED),
                "to_model": row.get("to_model", NOT_OBSERVED),
                "relation": "fallback_into_this_http_attempt",
            }
        if row.get("event") == "llm_request" and row.get("logical_batch_id") != rows[request_index].get("logical_batch_id"):
            break
    return NOT_OBSERVED


def _http_attempt_record(
    observation: Mapping[str, Any],
    rows: list[dict[str, Any]],
    request_index: int | None,
    q1_started_at: object,
    q1_terminal_at: object,
) -> dict[str, Any]:
    """Render one saved HTTP attempt, preserving all unavailable fields."""

    if request_index is None:
        return {
            "call_id": observation.get("call_id", NOT_OBSERVED),
            "logical_batch_id": observation.get("logical_batch_id", NOT_OBSERVED),
            "purpose": observation.get("purpose", NOT_OBSERVED),
            "status": observation.get("status", NOT_OBSERVED),
            "request_event": NOT_OBSERVED,
            "reason": "no_matching_saved_llm_request",
        }
    request = rows[request_index]
    end = _call_window_end(rows, request_index)
    batch_started = next(
        (
            row for row in reversed(rows[:request_index + 1])
            if row.get("event") == "provider_call"
            and row.get("phase") == "logical_batch_started"
            and row.get("logical_batch_id") == request.get("logical_batch_id")
        ),
        None,
    )
    attempt_index, attempt = _first_in_window(
        rows, start=request_index, end=end, event="provider_call", phase="http_attempt"
    )
    result_index, result = _first_in_window(
        rows,
        start=(attempt_index + 1 if attempt_index is not None else request_index),
        end=end,
        event="provider_call",
        phase="http_result",
    )
    _response_index, response = _first_in_window(
        rows,
        start=(result_index + 1 if result_index is not None else request_index),
        end=end,
        event="llm_response",
    )
    _error_index, error = _first_in_window(
        rows,
        start=(result_index + 1 if result_index is not None else request_index),
        end=end,
        event="llm_error",
    )
    _fallback_index, fallback_after = _first_in_window(
        rows,
        start=(result_index + 1 if result_index is not None else request_index),
        end=end,
        event="fallback",
    )
    attempt_at = attempt.get("timestamp") if attempt else NOT_OBSERVED
    result_at = result.get("timestamp") if result else NOT_OBSERVED
    payload = request.get("payload") if isinstance(request.get("payload"), Mapping) else {}
    return {
        "call_id": observation.get("call_id", request.get("request_id", NOT_OBSERVED)),
        "logical_batch_id": observation.get("logical_batch_id", request.get("logical_batch_id", NOT_OBSERVED)),
        "purpose": observation.get("purpose", NOT_OBSERVED),
        "operation": observation.get("operation", request.get("operation", NOT_OBSERVED)),
        "ledger_requested_model": observation.get("requested_model", NOT_OBSERVED),
        "request_model": payload.get("model", NOT_OBSERVED),
        "ledger_actual_model": observation.get("actual_model", NOT_OBSERVED),
        "ledger_status": observation.get("status", NOT_OBSERVED),
        "logical_batch_start_time": batch_started.get("timestamp") if batch_started else NOT_OBSERVED,
        "http_attempt_time": attempt_at,
        "http_result_time": result_at,
        "end_time": response.get("timestamp") if response else (error.get("timestamp") if error else result_at),
        "elapsed_since_q1_start_ms_at_http_attempt": _elapsed_ms(q1_started_at, attempt_at),
        "remaining_budget_ms": NOT_OBSERVED,
        "remaining_budget_basis": "not persisted at the HTTP-attempt boundary",
        "remaining_until_observed_q1_terminal_ms_at_http_attempt": _elapsed_ms(attempt_at, q1_terminal_at),
        "fallback_into_this_attempt": _fallback_before_request(rows, request_index),
        "fallback_after_this_attempt": {
            "timestamp": fallback_after.get("timestamp", NOT_OBSERVED),
            "from_model": fallback_after.get("from_model", NOT_OBSERVED),
            "to_model": fallback_after.get("to_model", NOT_OBSERVED),
            "trigger_error": error.get("error", NOT_OBSERVED) if error else NOT_OBSERVED,
        } if fallback_after else NOT_OBSERVED,
        "input_state": {
            "request_event_observed": True,
            "metadata": payload,
            "raw_input": NOT_OBSERVED,
            "content_logged": request.get("content_logged", NOT_OBSERVED),
        },
        "output_state": {
            "response_metadata": response.get("payload", NOT_OBSERVED) if response else NOT_OBSERVED,
            "raw_output": NOT_OBSERVED,
            "content_logged": response.get("content_logged", NOT_OBSERVED) if response else NOT_OBSERVED,
            "logged_error": error.get("error", NOT_OBSERVED) if error else NOT_OBSERVED,
            "http_outcome": result.get("outcome", NOT_OBSERVED) if result else NOT_OBSERVED,
        },
        "ledger_transport": {
            key: observation.get(key, NOT_OBSERVED)
            for key in (
                "endpoint", "http_status", "error_class", "late_result_discarded",
                "network_ms", "total_ms", "queue_ms", "prompt_tokens", "completion_tokens",
            )
        },
    }


def _candidate_timeline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report candidate IDs only; the non-scorable case declares no required ID."""

    first_candidates: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != "sparse_retrieval":
            continue
        queries = row.get("queries")
        rankings = row.get("episode_rankings")
        if not isinstance(queries, list) or not isinstance(rankings, list):
            continue
        for index, ranking in enumerate(rankings):
            if not isinstance(ranking, list):
                continue
            for rank, candidate in enumerate(ranking, 1):
                if not isinstance(candidate, Mapping):
                    continue
                try:
                    episode_id = int(candidate["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                first_candidates.setdefault(
                    episode_id,
                    {
                        "episode_id": episode_id,
                        "first_candidate_time": row.get("timestamp", NOT_OBSERVED),
                        "retrieval_query_index": index,
                        "retrieval_query_is_redacted_in_case_report": True,
                        "candidate_rank": rank,
                        "candidate_score": candidate.get("score", NOT_OBSERVED),
                        "first_selected_time": NOT_OBSERVED,
                        "first_final_prompt_time": NOT_OBSERVED,
                    },
                )
    rerank = _first(rows, "evidence_reranked")
    trace = rerank.get("trace") if isinstance(rerank, Mapping) and isinstance(rerank.get("trace"), Mapping) else {}
    selected_ids = trace.get("final_episode_ids", NOT_OBSERVED)
    answer_request = next(
        (
            row for row in rows
            if row.get("event") == "llm_request" and row.get("operation") == "chat_text"
        ),
        None,
    )
    return {
        "declared_required_evidence": NOT_OBSERVED,
        "declared_required_evidence_basis": (
            "The saved pilot manifest is explicitly non-scorable and contains no gold or expected answer."
        ),
        "candidate_evidence_first_appearance": list(first_candidates.values()),
        "selection_first_appearance": rerank.get("timestamp", NOT_OBSERVED) if rerank else NOT_OBSERVED,
        "selected_episode_ids": selected_ids if isinstance(selected_ids, list) else NOT_OBSERVED,
        "selection_basis": (
            "Saved evidence_reranked trace; this is a retrieval/selection observation, not a completed public Q1 result."
            if rerank else "not_observed"
        ),
        "final_prompt_first_appearance": NOT_OBSERVED,
        "final_prompt_basis": (
            "A final chat_text request is timestamped at "
            + str(answer_request.get("timestamp"))
            + ", but its content was not logged; evidence IDs in that prompt are not observable."
            if answer_request is not None
            else "No final chat_text request was observed."
        ),
    }


def _answer_audit_observations(
    rows: list[dict[str, Any]], observations: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Keep saved audit decisions without manufacturing absent answer text."""

    first_audit = _first(rows, "answer_evidence_audit")
    final_audit_failure = _first(rows, "answer_audit_failed")
    late_result = next(
        (row for row in reversed(rows) if row.get("event") == "answer_generated"),
        None,
    )
    result = late_result.get("result") if isinstance(late_result, Mapping) and isinstance(late_result.get("result"), Mapping) else {}
    correction_attempts = [
        {
            "call_id": item.get("call_id", NOT_OBSERVED),
            "logical_batch_id": item.get("logical_batch_id", NOT_OBSERVED),
            "requested_model": item.get("requested_model", NOT_OBSERVED),
            "actual_model": item.get("actual_model", NOT_OBSERVED),
            "status": item.get("status", NOT_OBSERVED),
        }
        for item in observations if item.get("purpose") == "answer_correction"
    ]
    return {
        "initial_answer": {
            "raw_content": NOT_OBSERVED,
            "content_hash": NOT_OBSERVED,
            "redacted_metadata": result.get("answer", NOT_OBSERVED),
            "basis": "Only redacted answer metadata was saved in the late diagnostic result.",
        },
        "first_audit": {
            "timestamp": first_audit.get("timestamp", NOT_OBSERVED) if first_audit else NOT_OBSERVED,
            "audit_index": first_audit.get("audit_index", NOT_OBSERVED) if first_audit else NOT_OBSERVED,
            "decision": first_audit.get("audit", NOT_OBSERVED) if first_audit else NOT_OBSERVED,
            "answer_revision_hash": NOT_OBSERVED,
            "evidence_set_hash": NOT_OBSERVED,
            "requirements_hash": NOT_OBSERVED,
        },
        "correction": {
            "raw_input": NOT_OBSERVED,
            "raw_output": NOT_OBSERVED,
            "input_hash": NOT_OBSERVED,
            "output_hash": NOT_OBSERVED,
            "transport_attempts": correction_attempts or NOT_OBSERVED,
            "basis": "The provider ledger identifies answer_correction attempts; request and response content are redacted.",
        },
        "final_audit": {
            "timestamp": final_audit_failure.get("timestamp", NOT_OBSERVED) if final_audit_failure else NOT_OBSERVED,
            "audit_index": final_audit_failure.get("audit_index", NOT_OBSERVED) if final_audit_failure else NOT_OBSERVED,
            "error": final_audit_failure.get("error", NOT_OBSERVED) if final_audit_failure else NOT_OBSERVED,
            "raw_decision": NOT_OBSERVED,
            "answer_revision_hash": NOT_OBSERVED,
            "evidence_set_hash": NOT_OBSERVED,
            "requirements_hash": NOT_OBSERVED,
        },
        "late_diagnostic_result": {
            "timestamp": late_result.get("timestamp", NOT_OBSERVED) if late_result else NOT_OBSERVED,
            "authoritative_requirements": result.get("authoritative_requirements", NOT_OBSERVED),
            "selected_episode_ids": result.get("episode_ids", NOT_OBSERVED),
            "evidence_episodes": result.get("evidence_episodes", NOT_OBSERVED),
            "timings": result.get("timings", NOT_OBSERVED),
            "public_completion_claim": False,
        },
    }


def build_postmortem(case_path: Path, q1_jsonl_path: Path) -> dict[str, Any]:
    """Return a no-network, read-only reconstruction of one saved Q1 case."""

    case = _read_object(case_path)
    rows = _read_jsonl(q1_jsonl_path)
    q1 = case.get("q1")
    if not isinstance(q1, Mapping) or not isinstance(q1.get("record"), Mapping):
        raise ValueError("case lacks q1.record")
    record = q1["record"]
    provider = record.get("provider")
    if not isinstance(provider, Mapping):
        raise ValueError("case lacks q1 provider ledger")
    observations = provider.get("observations")
    if not isinstance(observations, list):
        raise ValueError("case provider ledger lacks observations")
    observed_attempts = [item for item in observations if isinstance(item, Mapping)]
    q1_started_at = record.get("started_at", NOT_OBSERVED)
    q1_terminal_at = (
        _first(rows, "provider_call", "http_result").get("timestamp")
        if _first(rows, "provider_call", "http_result") is not None
        else NOT_OBSERVED
    )
    # The terminal request is the last observed Q1 provider result, not the
    # first warm-up result.  This remains a timestamp-only derivation.
    q1_http_rows = [
        row for row in rows
        if row.get("event") == "provider_call" and row.get("phase") == "http_result"
        and isinstance(row.get("logical_batch_id"), str)
        and row.get("logical_batch_id") in {
            item.get("logical_batch_id") for item in observations if isinstance(item, Mapping)
        }
    ]
    if q1_http_rows:
        q1_terminal_at = q1_http_rows[-1].get("timestamp", NOT_OBSERVED)
    consumed_requests: set[int] = set()
    replay: list[dict[str, Any]] = []
    for item in observed_attempts:
        request_index = _request_index_for_observation(item, rows, consumed_requests)
        if request_index is not None:
            consumed_requests.add(request_index)
        replay.append(_http_attempt_record(
            item, rows, request_index, q1_started_at, q1_terminal_at
        ))
    logical_batches: dict[str, dict[str, Any]] = {}
    for item in replay:
        batch_id = str(item.get("logical_batch_id", NOT_OBSERVED))
        batch = logical_batches.setdefault(batch_id, {
            "logical_batch_id": batch_id,
            "purposes": [],
            "operations": [],
            "call_ids": [],
            "attempt_statuses": [],
        })
        for key, destination in (("purpose", "purposes"), ("operation", "operations"), ("call_id", "call_ids"), ("ledger_status", "attempt_statuses")):
            value = item.get(key, NOT_OBSERVED)
            if value not in batch[destination]:
                batch[destination].append(value)
    rerank = _first(rows, "evidence_reranked")
    final_answer_request = next(
        (row for row in rows if row.get("event") == "llm_request" and row.get("operation") == "chat_text"),
        None,
    )
    raw_counts = provider.get("counts", {})
    observed_counts = {
        "http_attempts": len(replay),
        "succeeded": sum(item.get("ledger_status") in {"success", "succeeded"} for item in replay),
        "failed_or_rejected": sum(item.get("ledger_status") not in {"success", "succeeded"} for item in replay),
    }
    return {
        "schema": SCHEMA,
        "mode": "offline_saved_artifact_reconstruction",
        "input_case": {"path": str(case_path.resolve())},
        "input_q1_jsonl": {"path": str(q1_jsonl_path.resolve()), "event_count": len(rows)},
        "q1": {
            "record_status": record.get("status", NOT_OBSERVED),
            "record_started_at": q1_started_at,
            "record_elapsed_ms": record.get("elapsed_ms", NOT_OBSERVED),
            "observed_last_provider_result_at": q1_terminal_at,
            "terminal_error": record.get("error", NOT_OBSERVED),
            "http_attempts": replay,
            "logical_batches": list(logical_batches.values()),
            "historical_ledger_count_comparison": {
                "saved_counts": raw_counts,
                "recomputed_from_saved_observations": observed_counts,
                "status": (
                    "inconsistent_saved_aggregate"
                    if isinstance(raw_counts, Mapping) and any(raw_counts.get(key) != value for key, value in observed_counts.items())
                    else "consistent"
                ),
                "original_case_unchanged": True,
            },
        },
        "evidence_appearance": _candidate_timeline(rows),
        "rerank_observation": {
            "event_time": rerank.get("timestamp", NOT_OBSERVED) if rerank else NOT_OBSERVED,
            "trace": rerank.get("trace", NOT_OBSERVED) if rerank else NOT_OBSERVED,
            "failure_event": (
                _first(rows, "evidence_rerank_failed").get("error", NOT_OBSERVED)
                if _first(rows, "evidence_rerank_failed") else NOT_OBSERVED
            ),
        },
        "late_answer_request": {
            "request_time": final_answer_request.get("timestamp", NOT_OBSERVED) if final_answer_request else NOT_OBSERVED,
            "request_metadata": final_answer_request.get("payload", NOT_OBSERVED) if final_answer_request else NOT_OBSERVED,
            "raw_prompt": NOT_OBSERVED,
            "reason_for_continuation": NOT_OBSERVED,
        },
        "answer_audit_observations": _answer_audit_observations(rows, observed_attempts),
        "missing_observations": [
            {
                "field": field,
                "status": NOT_OBSERVED,
                "reason": reason,
            }
            for field, reason in (
                ("initial_answer_raw_content", "answer event stored only redacted metadata"),
                ("initial_answer_content_hash", "raw answer was not persisted"),
                ("correction_raw_input_and_output", "provider content logging was disabled"),
                ("answer_revision_hash", "not recorded in the saved event stream"),
                ("evidence_set_hash", "not recorded in the saved event stream"),
                ("requirements_hash", "not recorded in the saved event stream"),
                ("audit_prompt_source_spans", "only audit commentary, not the sent source excerpt, was persisted"),
                ("per_attempt_remaining_budget", "not recorded at the HTTP-attempt boundary"),
                ("final_audit_decision", "the final audit was late-discarded before a response was accepted"),
                ("final_prompt_evidence_membership", "chat request content was not logged"),
            )
        ],
        "non_inference_rule": (
            "Values absent from the case ledger or redacted JSONL are emitted as not_observed; this export does not recreate content."
        ),
    }


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    q1 = payload["q1"]
    rerank = payload["rerank_observation"]
    late = payload["late_answer_request"]
    evidence = payload["evidence_appearance"]
    audit = payload["answer_audit_observations"]
    lines = [
        "# Q1→Q2 saved-case postmortem",
        "",
        "- Mode: offline reconstruction from the saved case and redacted Q1 JSONL; no model or database call was made.",
        "- The original failed trial remains unchanged.",
        "- `not_observed` means no value was persisted; it has not been reconstructed.",
        "",
        "## Q1 HTTP chronology",
        "",
        "| Purpose | Request model | Ledger status | HTTP start | HTTP end | Q1 elapsed at start (ms) | Fallback relation |",
        "|---|---|---|---|---|---:|---|",
    ]
    for item in q1["http_attempts"]:
        fallback = item.get("fallback_into_this_attempt", NOT_OBSERVED)
        if fallback == NOT_OBSERVED:
            fallback = item.get("fallback_after_this_attempt", NOT_OBSERVED)
        lines.append(
            "| {purpose} | {model} | {status} | {start} | {end} | {elapsed} | {fallback} |".format(
                purpose=item["purpose"], model=item.get("request_model", NOT_OBSERVED),
                status=item["ledger_status"], start=item["http_attempt_time"],
                end=item["http_result_time"],
                elapsed=item["elapsed_since_q1_start_ms_at_http_attempt"], fallback=fallback,
            )
        )
    lines.extend([
        "",
        "## Observed failure boundary",
        "",
        f"- Rerank trace time: `{rerank['event_time']}`; logged rerank failure: `{rerank['failure_event']}`.",
        f"- Final answer request time: `{late['request_time']}`; its raw prompt is `{late['raw_prompt']}`.",
        f"- Declared required evidence: `{evidence['declared_required_evidence']}`. Selection event: `{evidence['selection_first_appearance']}`; final prompt membership remains `{evidence['final_prompt_first_appearance']}`.",
        f"- First audit time: `{audit['first_audit']['timestamp']}`; its raw answer revision remains `{audit['first_audit']['answer_revision_hash']}`.",
        f"- Final audit: `{audit['final_audit']['error']}`.",
        "",
        "## Missing saved observations",
        "",
    ])
    for item in payload["missing_observations"]:
        lines.append(f"- `{item['field']}`: `{item['status']}` — {item['reason']}.")
    lines.extend([
        "",
        "## Historical aggregate check",
        "",
        f"- Saved ledger aggregate status: `{q1['historical_ledger_count_comparison']['status']}`. The saved input is not edited by this exporter.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_postmortem(*, case_path: Path, q1_jsonl_path: Path, output_dir: Path) -> dict[str, str]:
    if output_dir.exists():
        raise FileExistsError("postmortem output directory must be new")
    if not case_path.is_file() or not q1_jsonl_path.is_file():
        raise FileNotFoundError("saved case and Q1 JSONL are required")
    output_dir.mkdir(parents=True)
    payload = build_postmortem(case_path, q1_jsonl_path)
    full_local = output_dir / FULL_LOCAL_NAME
    full_local.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = output_dir / REPORT_NAME
    _write_report(report, payload)
    return {"full_local": str(full_local), "report": str(report)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an offline Q1/Q2 case postmortem")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--q1-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_postmortem(case_path=args.case, q1_jsonl_path=args.q1_jsonl, output_dir=args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
