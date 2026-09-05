from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Iterable


def _normalized_evidence_text(value: Any) -> str:
    return "\n".join(
        line.rstrip() for line in str(value or "").replace("\r\n", "\n").split("\n")
    ).strip()


def validate_frozen_candidate_alignment(
    source_rows: Iterable[dict[str, Any]],
    database_episode_texts: dict[int, str],
) -> dict[str, Any]:
    """Reject a frozen-ID replay when IDs now refer to different Episodes.

    ID existence alone is insufficient after a corpus rebuild because SQLite
    can reuse the same integer for unrelated content.  Historical retrieval
    reports persist the selected evidence text, which acts as a semantic
    fingerprint for the candidate-ID namespace.
    """

    rows = list(source_rows)
    candidate_ids = {
        int(value)
        for row in rows
        for value in (
            row.get("result", {})
            .get("rerank_trace", {})
            .get("candidate_episode_ids", [])
        )
    }
    missing_candidate_ids = sorted(candidate_ids.difference(database_episode_texts))
    checked_fingerprints: set[tuple[int, str]] = set()
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row.get("id", ""))
        for evidence in row.get("result", {}).get("evidence_episodes", []):
            if not isinstance(evidence, dict) or evidence.get("id") is None:
                continue
            episode_id = int(evidence["id"])
            expected = _normalized_evidence_text(evidence.get("text"))
            if not expected:
                continue
            fingerprint = (episode_id, expected)
            if fingerprint in checked_fingerprints:
                continue
            checked_fingerprints.add(fingerprint)
            actual_raw = database_episode_texts.get(episode_id)
            actual = _normalized_evidence_text(actual_raw)
            if actual_raw is None or actual != expected:
                mismatches.append(
                    {
                        "question_id": question_id,
                        "episode_id": episode_id,
                        "expected_text_prefix": expected[:180],
                        "database_text_prefix": actual[:180],
                        "missing": actual_raw is None,
                    }
                )
    return {
        "candidate_id_count": len(candidate_ids),
        "candidate_ids_present": len(candidate_ids) - len(missing_candidate_ids),
        "missing_candidate_ids": missing_candidate_ids,
        "evidence_fingerprint_count": len(checked_fingerprints),
        "evidence_text_mismatch_count": len(mismatches),
        "evidence_text_mismatches": mismatches,
        "passed": not missing_candidate_ids and not mismatches,
    }


def _coverage_episode_ids(payload: Any) -> set[int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("coverage"), list):
        return set()
    result: set[int] = set()
    for item in payload["coverage"]:
        if not isinstance(item, dict) or not isinstance(item.get("episode_ids"), list):
            continue
        for value in item["episode_ids"]:
            try:
                result.add(int(value))
            except (TypeError, ValueError):
                continue
    return result


def score_fixed_candidate_run(
    *,
    selected_episode_ids: Iterable[int],
    candidate_episode_ids: Iterable[int],
    required_episode_groups: Iterable[Iterable[int]],
    rerank_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one stochastic rerank over a frozen candidate pool.

    The evidence manifest defines independent fact slots.  Each slot may list
    several interchangeable Episode IDs; hitting any one of them covers the
    slot.  Coverage-scout votes are diagnostic only and never change the score.
    """

    selected = [int(value) for value in selected_episode_ids]
    candidates = [int(value) for value in candidate_episode_ids]
    trace = rerank_trace if isinstance(rerank_trace, dict) else {}
    initial_ids = _coverage_episode_ids(trace.get("initial"))
    independent_ids = _coverage_episode_ids(
        trace.get("independent_coverage", trace.get("coverage_audit"))
    )

    groups: list[dict[str, Any]] = []
    for group_index, values in enumerate(required_episode_groups):
        alternatives = sorted({int(value) for value in values})
        allowed = set(alternatives)
        selected_matches = [value for value in selected if value in allowed]
        candidate_matches = [value for value in candidates if value in allowed]
        initial_vote = bool(initial_ids.intersection(allowed))
        independent_vote = bool(independent_ids.intersection(allowed))
        groups.append(
            {
                "group_index": group_index,
                "alternatives": alternatives,
                "candidate_available": bool(candidate_matches),
                "candidate_matches": candidate_matches,
                "selected": bool(selected_matches),
                "selected_matches": selected_matches,
                "first_rank": next(
                    (
                        rank
                        for rank, episode_id in enumerate(selected, start=1)
                        if episode_id in allowed
                    ),
                    None,
                ),
                "initial_coverage_vote": initial_vote,
                "independent_coverage_vote": independent_vote,
                "coverage_vote_disagreement": initial_vote != independent_vote,
            }
        )

    required_count = len(groups)
    hit_count = sum(bool(group["selected"]) for group in groups)
    candidate_hit_count = sum(
        bool(group["candidate_available"]) for group in groups
    )
    model_call_count = 0
    if trace.get("enabled") and trace.get("backend", "llm") == "llm":
        model_call_count = 1
        model_call_count += int(bool(trace.get("coverage_audit_performed")))
        model_call_count += int(bool(trace.get("compressor_performed")))
        # Historical traces predate the explicit flags but contain the payloads.
        if "coverage_audit_performed" not in trace:
            model_call_count += int(trace.get("coverage_audit") is not None)
        if "compressor_performed" not in trace:
            model_call_count += int(trace.get("audit") is not None)

    return {
        "selected_hit_count": hit_count,
        "required_group_count": required_count,
        "recall_at_20": hit_count / required_count if required_count else 0.0,
        "candidate_hit_count": candidate_hit_count,
        "candidate_recall_at_100": (
            candidate_hit_count / required_count if required_count else 0.0
        ),
        "coverage_vote_disagreement_count": sum(
            bool(group["coverage_vote_disagreement"]) for group in groups
        ),
        "estimated_model_calls": model_call_count,
        "groups": groups,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def aggregate_stability_runs(
    rows: Iterable[dict[str, Any]],
    *,
    expected_repeats: int,
    pass_threshold: float = 0.95,
) -> dict[str, Any]:
    """Aggregate repeated frozen-candidate reranks by question and fact slot."""

    materialized = list(rows)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        by_question[str(row["question_id"])].append(row)

    question_rows: list[dict[str, Any]] = []
    unstable_slot_count = 0
    total_slot_count = 0
    for question_id in sorted(by_question):
        runs = sorted(
            by_question[question_id], key=lambda item: int(item["repeat_index"])
        )
        recalls = [float(item["score"]["recall_at_20"]) for item in runs]
        elapsed = [float(item.get("elapsed_seconds", 0.0)) for item in runs]
        group_count = max(
            (len(item["score"].get("groups", [])) for item in runs),
            default=0,
        )
        groups: list[dict[str, Any]] = []
        for group_index in range(group_count):
            observations = [
                item["score"]["groups"][group_index]
                for item in runs
                if group_index < len(item["score"].get("groups", []))
            ]
            hit_count = sum(bool(item["selected"]) for item in observations)
            observation_count = len(observations)
            hit_rate = hit_count / observation_count if observation_count else 0.0
            if 0.0 < hit_rate < 1.0:
                unstable_slot_count += 1
            total_slot_count += 1
            groups.append(
                {
                    "group_index": group_index,
                    "alternatives": (
                        observations[0]["alternatives"] if observations else []
                    ),
                    "candidate_available_in_all_runs": all(
                        bool(item["candidate_available"]) for item in observations
                    ),
                    "hit_count": hit_count,
                    "observation_count": observation_count,
                    "hit_rate": hit_rate,
                    "missed_repeat_indices": [
                        int(run["repeat_index"])
                        for run, observation in zip(runs, observations, strict=False)
                        if not observation["selected"]
                    ],
                    "initial_coverage_vote_rate": (
                        sum(bool(item["initial_coverage_vote"]) for item in observations)
                        / observation_count
                        if observation_count
                        else 0.0
                    ),
                    "independent_coverage_vote_rate": (
                        sum(
                            bool(item["independent_coverage_vote"])
                            for item in observations
                        )
                        / observation_count
                        if observation_count
                        else 0.0
                    ),
                    "coverage_vote_disagreement_rate": (
                        sum(
                            bool(item["coverage_vote_disagreement"])
                            for item in observations
                        )
                        / observation_count
                        if observation_count
                        else 0.0
                    ),
                }
            )
        question_rows.append(
            {
                "question_id": question_id,
                "question": str(runs[0].get("question", "")) if runs else "",
                "completed_repeats": len(runs),
                "expected_repeats": int(expected_repeats),
                "recall_at_20_mean": mean(recalls) if recalls else 0.0,
                "recall_at_20_minimum": min(recalls, default=0.0),
                "recall_at_20_maximum": max(recalls, default=0.0),
                "runs_above_threshold": sum(
                    value > pass_threshold for value in recalls
                ),
                "run_pass_rate": (
                    sum(value > pass_threshold for value in recalls) / len(recalls)
                    if recalls
                    else 0.0
                ),
                "latency_seconds_mean": mean(elapsed) if elapsed else 0.0,
                "latency_seconds_p95": _percentile(elapsed, 0.95),
                "groups": groups,
            }
        )

    recalls = [float(item["score"]["recall_at_20"]) for item in materialized]
    candidate_recalls = [
        float(item["score"]["candidate_recall_at_100"])
        for item in materialized
    ]
    elapsed = [float(item.get("elapsed_seconds", 0.0)) for item in materialized]
    model_calls = sum(
        int(item["score"].get("estimated_model_calls", 0))
        for item in materialized
    )
    return {
        "expected_repeats": int(expected_repeats),
        "completed_runs": len(materialized),
        "question_count": len(by_question),
        "candidate_recall_at_100_mean": (
            mean(candidate_recalls) if candidate_recalls else 0.0
        ),
        "candidate_recall_at_100_minimum": min(candidate_recalls, default=0.0),
        "selected_recall_at_20_mean": mean(recalls) if recalls else 0.0,
        "selected_recall_at_20_median": median(recalls) if recalls else 0.0,
        "selected_recall_at_20_minimum": min(recalls, default=0.0),
        "runs_above_95_percent": sum(value > pass_threshold for value in recalls),
        "run_pass_rate": (
            sum(value > pass_threshold for value in recalls) / len(recalls)
            if recalls
            else 0.0
        ),
        "unstable_required_slot_count": unstable_slot_count,
        "required_slot_count": total_slot_count,
        "unstable_required_slot_rate": (
            unstable_slot_count / total_slot_count if total_slot_count else 0.0
        ),
        "coverage_vote_disagreement_count": sum(
            int(item["score"].get("coverage_vote_disagreement_count", 0))
            for item in materialized
        ),
        "estimated_model_calls": model_calls,
        "latency_seconds_mean": mean(elapsed) if elapsed else 0.0,
        "latency_seconds_p95": _percentile(elapsed, 0.95),
        "questions": question_rows,
    }
