from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from memory_demo.retrieval import QueryEngine
from benchmarks.support.stage5 import load_json, write_json


def _unique(values: Iterable[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _slot_one_ids(slots: Iterable[dict[str, Any]]) -> list[int]:
    """Choose at most one non-duplicate candidate from each planned slot."""

    selected: list[int] = []
    for slot in slots:
        candidates = slot.get(
            "planned_floor_episode_ids",
            slot.get("floor_episode_ids", slot.get("candidate_episode_ids", [])),
        )
        for value in candidates:
            episode_id = int(value)
            if episode_id not in selected:
                selected.append(episode_id)
                break
    return selected


def _score(selected_ids: list[int], criterion: dict[str, Any]) -> dict[str, Any]:
    selected = set(int(value) for value in selected_ids)
    groups = []
    for group_index, alternatives in enumerate(criterion["required_episode_groups"]):
        allowed = {int(value) for value in alternatives}
        matches = [value for value in selected_ids if int(value) in allowed]
        groups.append(
            {
                "group_index": group_index,
                "alternatives": sorted(allowed),
                "matched_episode_ids": matches,
                "selected": bool(selected.intersection(allowed)),
            }
        )
    hits = sum(bool(group["selected"]) for group in groups)
    return {
        "matched_group_count": hits,
        "required_group_count": len(groups),
        "recall_at_20": hits / len(groups) if groups else 0.0,
        "groups": groups,
    }


def _policy_floor_ids(trace: dict[str, Any]) -> dict[str, list[int]]:
    deterministic = trace.get("deterministic_evidence_floor", {})
    constraint_slots = deterministic.get("constraint_slots", [])
    atomic_slots = deterministic.get("atomic_slots", [])
    constraint = _unique(
        value
        for slot in constraint_slots
        for value in slot.get(
            "planned_floor_episode_ids", slot.get("floor_episode_ids", [])
        )
    )
    atomic = _unique(deterministic.get("atomic_floor_ids", []))
    sparse = _unique(deterministic.get("whole_question_sparse_ids", []))
    neighbors = _unique(trace.get("answer_slot_neighbor_floor_ids", []))
    slot_one = _unique(
        [
            *_slot_one_ids(constraint_slots),
            *_slot_one_ids(atomic_slots),
            *neighbors,
        ]
    )
    current = _unique(trace.get("required_evidence_floor_ids", []))
    return {
        "none": [],
        "answer_neighbors": neighbors,
        "constraint": _unique([*constraint, *neighbors]),
        "atomic": _unique([*atomic, *neighbors]),
        "constraint_atomic": _unique([*constraint, *atomic, *neighbors]),
        "slot_one": slot_one,
        "slot_one_sparse4": _unique([*slot_one, *sparse[:4]]),
        "current": current,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline counterfactual audit of deterministic evidence floors."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--budgets",
        default="4,6,8,10,12,16,20",
        help="comma-separated hard-floor budgets",
    )
    args = parser.parse_args()

    report = load_json(args.input)
    manifest = load_json(args.manifest)["questions"]
    budgets = sorted(
        {max(0, int(value)) for value in args.budgets.split(",") if value.strip()}
    )
    rows: list[dict[str, Any]] = []
    for row in report["rows"]:
        question_id = str(row["id"])
        criterion = manifest[question_id]
        trace = row["result"]["rerank_trace"]
        audit = trace.get("audit") or {}
        pre_floor_ids = _unique(
            audit.get("final_episode_ids")
            or trace.get("shortlist_episode_ids", [])[:20]
        )[:20]
        pre_floor_ids, coverage_changes = QueryEngine._enforce_coverage_selection(
            trace.get("merged_coverage", {}),
            pre_floor_ids,
            set(int(value) for value in trace.get("shortlist_episode_ids", [])),
            20,
        )
        allowed_ids = {
            int(value) for value in trace.get("rerank_input_episode_ids", [])
        }
        policies = _policy_floor_ids(trace)
        arms: dict[str, Any] = {
            "base_result": {
                "selected_episode_ids": [
                    int(value) for value in row["result"].get("episode_ids", [])
                ],
                "score": _score(
                    [int(value) for value in row["result"].get("episode_ids", [])],
                    criterion,
                ),
            },
            "coverage_only": {
                "selected_episode_ids": pre_floor_ids,
                "score": _score(pre_floor_ids, criterion),
            },
        }
        for policy_name, floor_ids in policies.items():
            for budget in budgets:
                selected, changes = QueryEngine._enforce_selection_floor(
                    pre_floor_ids,
                    floor_ids[:budget],
                    allowed_ids,
                    20,
                )
                key = f"{policy_name}@{budget}"
                arms[key] = {
                    "selected_episode_ids": selected,
                    "floor_episode_ids": floor_ids[:budget],
                    "floor_count": min(len(floor_ids), budget),
                    "floor_replacements": changes,
                    "score": _score(selected, criterion),
                }
                audit_ids = _unique(
                    audit.get("final_episode_ids")
                    or trace.get("shortlist_episode_ids", [])[:20]
                )[:20]
                final_selected, final_floor_changes, final_coverage_changes = (
                    QueryEngine._enforce_final_selection_constraints(
                        audit_ids,
                        floor_ids[:budget],
                        trace.get("merged_coverage", {}),
                        allowed_ids,
                        set(
                            int(value)
                            for value in trace.get("shortlist_episode_ids", [])
                        ),
                        20,
                    )
                )
                final_key = f"{policy_name}@{budget}:coverage_last"
                arms[final_key] = {
                    "selected_episode_ids": final_selected,
                    "floor_episode_ids": floor_ids[:budget],
                    "floor_count": min(len(floor_ids), budget),
                    "floor_replacements": final_floor_changes,
                    "coverage_replacements": final_coverage_changes,
                    "score": _score(final_selected, criterion),
                }
        rows.append(
            {
                "question_id": question_id,
                "question": str(row["question"]),
                "coverage_replacements": coverage_changes,
                "policy_floor_ids": policies,
                "arms": arms,
            }
        )

    arm_names = sorted({name for row in rows for name in row["arms"]})
    summaries = []
    for arm_name in arm_names:
        values = [
            float(row["arms"][arm_name]["score"]["recall_at_20"])
            for row in rows
        ]
        summaries.append(
            {
                "arm": arm_name,
                "mean_recall_at_20": mean(values) if values else 0.0,
                "minimum_recall_at_20": min(values, default=0.0),
                "questions_at_100_percent": sum(value == 1.0 for value in values),
                "questions_above_95_percent": sum(value > 0.95 for value in values),
            }
        )
    summaries.sort(
        key=lambda item: (
            -float(item["minimum_recall_at_20"]),
            -float(item["mean_recall_at_20"]),
            str(item["arm"]),
        )
    )
    write_json(
        args.output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": str(args.input.resolve()),
            "manifest": str(args.manifest.resolve()),
            "budgets": budgets,
            "summary": summaries,
            "rows": rows,
        },
    )
    for item in summaries[:20]:
        print(
            item["arm"],
            f"mean={item['mean_recall_at_20']:.4f}",
            f"min={item['minimum_recall_at_20']:.4f}",
            f"perfect={item['questions_at_100_percent']}/{len(rows)}",
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
