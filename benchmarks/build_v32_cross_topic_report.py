"""Export the preserved v3.2 fresh-Q1 trials as readable local evidence.

The exporter is intentionally read-only.  It never reruns a Q1, invents a
missing stage from a later log, or turns an answer-input excerpt into a
source-bound learning result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "aevnema.v3_2.cross_topic_q1_report.v1"
TRIALS = {
    "event_moon_festival": "E32-05a_event_q1",
    "main_rest": "E32-05b_main_q1",
    "favor_competition": "E32-05c_favor_q1",
}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _provider_stages(record: Mapping[str, object]) -> list[dict[str, object]]:
    provider = record.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    observations = provider.get("observations")
    if not isinstance(observations, list):
        return []
    rows: list[dict[str, object]] = []
    for item in observations:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "role": item.get("role", "not_observed"),
                "purpose": item.get("purpose", "not_observed"),
                "requested_model": item.get("requested_model", "not_observed"),
                "actual_model": item.get("actual_model", "not_observed"),
                "status": item.get("status", "not_observed"),
                "sent": item.get("sent", "not_observed"),
                "network_ms": item.get("network_ms", "not_observed"),
                "total_ms": item.get("total_ms", "not_observed"),
                "http_status": item.get("http_status", "not_observed"),
                "error_class": item.get("error_class", "not_observed"),
            }
        )
    return rows


def _checkpoint(path: Path) -> dict[str, object]:
    """Extract only actual answer-boundary entries from local-only logs."""

    checkpoint: dict[str, object] = {
        "answer_input": "not_observed",
        "revisions": [],
        "selected_evidence": [],
    }
    for log_path in sorted((path / "logs" / "q1").glob("*.answer-evidence.jsonl")):
        for raw_line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping):
                continue
            if event.get("event") == "answer_input_checkpoint":
                checkpoint["answer_input"] = "observed"
                checkpoint["version_binding"] = event.get("version_binding", {})
                checkpoint["prompt_evidence_assertions"] = event.get(
                    "prompt_evidence_assertions", {}
                )
                selected = event.get("selected_evidence")
                if isinstance(selected, list):
                    # These are the actual Source excerpts given to the answer
                    # boundary.  They are local-only and may be long, but must
                    # remain available for a proper forensic report.
                    checkpoint["selected_evidence"] = selected
            elif event.get("event") == "answer_revision_checkpoint":
                checkpoint["revisions"].append(
                    {
                        "stage": event.get("stage", "not_observed"),
                        "answer": event.get("answer", "not_observed"),
                        "audit_index": event.get("audit_index", "not_observed"),
                        "terminal_reason": event.get("terminal_reason", "not_observed"),
                        "error_type": event.get("error_type", "not_observed"),
                        "remaining_budget_seconds": event.get(
                            "remaining_budget_seconds", "not_observed"
                        ),
                    }
                )
    return checkpoint


def _trial(path: Path, *, label: str) -> dict[str, object]:
    trace_path = path / "same_chapter_learning_matrix.full_local.json"
    if not trace_path.is_file():
        return {
            "label": label,
            "status": "not_started",
            "reason": "paid_provider_branch_paused_after_two_terminal_Q1_technical_failures",
            "artifact": str(trace_path),
        }
    payload = _load(trace_path)
    q1 = payload.get("q1")
    q1 = q1 if isinstance(q1, Mapping) else {}
    record = q1.get("record")
    record = record if isinstance(record, Mapping) else {}
    result = record.get("result")
    result = result if isinstance(result, Mapping) else {}
    error = record.get("error")
    error = error if isinstance(error, Mapping) else {}
    finalizer = q1.get("finalizer_observer")
    finalizer = finalizer if isinstance(finalizer, Mapping) else {}
    material = q1.get("new_material")
    material = material if isinstance(material, Mapping) else {}
    case = payload.get("case")
    case = case if isinstance(case, Mapping) else {}
    source = payload.get("source_database")
    source = source if isinstance(source, Mapping) else {}
    arms = payload.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    q2_states = {
        str(summary.get("run_status"))
        for arm in arms.values()
        if isinstance(arm, Mapping)
        and isinstance((summary := arm.get("summary")), Mapping)
        and summary.get("run_status") is not None
    }
    return {
        "label": label,
        "artifact": str(trace_path),
        "trace_sha256": _sha(trace_path),
        "matrix_status": payload.get("status", "not_observed"),
        "q1": {
            "text": case.get("q1_text", "not_observed"),
            "status": record.get("status", "not_observed"),
            "started_at": record.get("started_at", "not_observed"),
            "elapsed_ms": record.get("elapsed_ms", "not_observed"),
            "error": {"type": error.get("type", "not_observed"), "message": error.get("message", "not_observed")},
            "provider_stages": _provider_stages(record),
            "provider_counts": (
                record.get("provider", {}).get("counts", {})
                if isinstance(record.get("provider"), Mapping)
                else {}
            ),
            "candidate_episode_ids": result.get("candidate_episode_ids", "not_observed"),
            "reranked_episode_ids": result.get("reranked_episode_ids", "not_observed"),
            "selected_episode_ids": result.get("episode_ids", "not_observed"),
            "evidence_episodes": result.get("evidence_episodes", "not_observed"),
            "answer_terminal_state": result.get("answer_terminal_state", "not_observed"),
            "answer_audits": result.get("answer_audits", "not_observed"),
            "contextual_learning": result.get("contextual_learning", "not_observed"),
        },
        "answer_boundary": _checkpoint(path),
        "learning": {
            "status": q1.get("learning_status", "not_observed"),
            "reason": q1.get("learning_reason", "not_observed"),
            "public_finalizer_calls": finalizer.get("finalizer_calls", "not_observed"),
            "new_receipt_ids": material.get("new_receipt_ids", "not_observed"),
            "new_association_ids": material.get("new_association_ids", "not_observed"),
            "runtime_manifest": (
                "not_created" if not material.get("new_receipt_ids") else "inspect_receipt"
            ),
        },
        "q2": {
            "states": sorted(q2_states) or ["not_observed"],
            "reason": (
                "dependent Q2 did not run because this Q1 did not create exactly one ready receipt/edge/runtime manifest"
                if q2_states == {"not_run"}
                else "not_observed"
            ),
        },
        "source_database": {"path": source.get("path", "not_observed"), "sha256": source.get("sha256", "not_observed")},
        "formal_scoring": "not_scored_unapproved_gold",
    }


def build_report(*, experiments_dir: Path, output_dir: Path) -> dict[str, object]:
    experiments_dir = experiments_dir.resolve()
    output_dir = output_dir.resolve()
    trials = [
        _trial(experiments_dir / directory, label=label)
        for label, directory in TRIALS.items()
    ]
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "formal_scoring": "disabled_unapproved_gold",
        "paid_provider_branch": {
            "status": "paused",
            "reason": "two consecutive terminal fresh-Q1 technical failures; no retry and no favor Q1",
        },
        "trials": trials,
        "interpretation": [
            "A result present at answer_input_checkpoint is materialized Source context for an attempted answer, not source-bound learning evidence.",
            "No Q2 state is interpreted as a retrieval miss when no ready receipt/edge/runtime manifest was created.",
            "The trials use the preserved diagnostic snapshot and are not K_full or formal cross-topic metrics.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cross_topic_results.full_local.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# v3.2 cross-topic Q1 report",
        "",
        "This is a non-scoring diagnostic on a preserved legacy-evidence snapshot. It is not a cross-topic recall, utility, or generalization result.",
        "",
        "| Topic | Q1 | Answer boundary | Learning | Q2 | Provider HTTP |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for trial in trials:
        q1 = trial.get("q1", {}) if isinstance(trial, Mapping) else {}
        q1 = q1 if isinstance(q1, Mapping) else {}
        boundary = trial.get("answer_boundary", {}) if isinstance(trial, Mapping) else {}
        boundary = boundary if isinstance(boundary, Mapping) else {}
        learning = trial.get("learning", {}) if isinstance(trial, Mapping) else {}
        learning = learning if isinstance(learning, Mapping) else {}
        q2 = trial.get("q2", {}) if isinstance(trial, Mapping) else {}
        q2 = q2 if isinstance(q2, Mapping) else {}
        counts = q1.get("provider_counts", {}) if isinstance(q1, Mapping) else {}
        counts = counts if isinstance(counts, Mapping) else {}
        lines.append(
            f"| {trial['label']} | `{q1.get('status', trial.get('status', 'not_observed'))}` | `{boundary.get('answer_input', 'not_observed')}` | `{learning.get('status', 'not_observed')}` / finalizer `{learning.get('public_finalizer_calls', 'not_observed')}` | `{', '.join(map(str, q2.get('states', ['not_observed'])))}` | {counts.get('http_attempts', 'not_observed')} |"
        )
    lines.extend(
        [
            "",
            "## Observed boundaries",
            "",
            "- **event_moon_festival:** answer-input and answer-revision checkpoints are observed: two legacy heuristic Source excerpts were assembled, an answer attempt was recorded, and the audit checkpoint reports `ModelDeadlineExceeded`. The public query then elapsed before `post_query_result`, so no result-object candidate/selector/learning state is inferred from those earlier checkpoints.",
            "- **main_rest:** the answer-input checkpoint contains actual Source excerpts from `main/32040.json`, including speaker and raw script fields. Its two selected entries are `legacy_heuristic_excerpt` / `no_persisted_evidence_quote`; an answer attempt was recorded, followed by an audit exception. Learning is `selector_delivery_incomplete`, with zero finalizer calls and no receipt/edge/manifest.",
            "- **favor_competition:** not started after the frozen paid-provider stop rule. It is neither a miss nor a failed Q1.",
            "",
            "The companion `cross_topic_results.full_local.json` contains the complete locally authorized Source excerpts, answer-boundary events, answer attempt, versions, and provider-stage observations. It contains no credentials or request headers.",
            "",
        ]
    )
    (output_dir / "RETRIEVAL_TRANSFER_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build_report(experiments_dir=args.experiments_dir, output_dir=args.output_dir)
    print(json.dumps({"schema": payload["schema"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
