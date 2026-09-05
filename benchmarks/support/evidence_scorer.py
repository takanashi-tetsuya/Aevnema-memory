from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


CRITERIA: dict[str, dict[str, Any]] = {
    "makeup_club_hidden_purpose": {
        "required_episode_groups": [[247, 248], [273], [277, 279]],
        "required_sources": ["main/31060.json", "main/31090.json"],
        "required_terms": ["渚", "退学", "圣三一", "叛徒", "伊甸条约"],
    },
    "paradise_paradox_political_projection": {
        "required_episode_groups": [[62, 63, 64, 65], [273, 277, 279], [307, 309]],
        "required_sources": [
            "main/33030.json",
            "main/31090.json",
            "main/32170.json",
        ],
        "required_terms": ["乐园", "渚", "补习部", "叛徒", "未花", "政变", "阿里乌斯"],
    },
    "old_cathedral_attack_causal_chain": {
        "required_episode_groups": [[320], [332, 341], [307, 309]],
        "required_sources": ["main/33070.json", "main/32170.json"],
        "required_terms": ["阿里乌斯", "未花"],
        "required_term_groups": [["推论", "推断", "前置条件", "暗示"]],
    },
}


def score_result(
    question_id: str,
    result: dict[str, Any],
    mode: str,
    status: str,
    criteria: dict[str, dict[str, Any]] = CRITERIA,
) -> dict[str, Any]:
    criterion = criteria[question_id]
    episode_ids = {int(value) for value in result.get("episode_ids", [])}
    sources = {
        str(row.get("source_key", ""))
        for row in result.get("evidence_episodes", [])
    }
    answer = str(result.get("answer", ""))
    audits = result.get("answer_audits", [])
    last_audit_valid = bool(audits and audits[-1].get("valid") is True)
    changed_ids = {
        *[int(value) for value in result.get("new_association_ids", [])],
        *[int(value) for value in result.get("reinforced_association_ids", [])],
    }
    used_ids = {int(value) for value in result.get("association_ids", [])}
    counterfactual = result.get("growth_counterfactual_utility", {})
    staging = result.get("growth_staging", {})
    safe_zero_write = bool(
        mode == "graph_growing"
        and not changed_ids
        and isinstance(counterfactual, dict)
        and counterfactual.get("enabled") is True
        and counterfactual.get("causal_utility_observed") is False
        and not counterfactual.get("persistable_changed_ids", [])
        and isinstance(staging, dict)
        and staging.get("enabled") is True
        and staging.get("committed") is False
    )
    mode_behavior_passed = (
        not used_ids
        if mode == "vector_only"
        else bool(used_ids)
        if mode == "graph_static"
        else (
            bool(changed_ids) and changed_ids.issubset(used_ids)
        ) or safe_zero_write
    )
    episode_groups = [
        {
            "alternatives": group,
            "matched": sorted(episode_ids.intersection(group)),
            "passed": bool(episode_ids.intersection(group)),
        }
        for group in criterion["required_episode_groups"]
    ]
    source_checks = {
        source: source in sources for source in criterion["required_sources"]
    }
    term_checks = {term: term in answer for term in criterion["required_terms"]}
    term_group_checks = [
        {
            "alternatives": group,
            "matched": [term for term in group if term in answer],
            "passed": any(term in answer for term in group),
        }
        for group in criterion.get("required_term_groups", [])
    ]
    changed_edges_used = changed_ids.issubset(used_ids)
    passed = (
        status == "completed"
        and all(item["passed"] for item in episode_groups)
        and all(source_checks.values())
        and all(term_checks.values())
        and all(item["passed"] for item in term_group_checks)
        and last_audit_valid
        and changed_edges_used
        and mode_behavior_passed
    )
    return {
        "passed": passed,
        "last_answer_audit_valid": last_audit_valid,
        "answer_revision_count": int(result.get("answer_revision_count", 0)),
        "episode_groups": episode_groups,
        "required_sources": source_checks,
        "required_terms": term_checks,
        "required_term_groups": term_group_checks,
        "atomic_anchor_episode_ids": result.get("atomic_anchor_episode_ids", []),
        "new_association_ids": result.get("new_association_ids", []),
        "reinforced_association_ids": result.get("reinforced_association_ids", []),
        "changed_associations_used_in_answer_paths": changed_edges_used,
        "safe_counterfactual_zero_write": safe_zero_write,
        "mode_behavior_passed": mode_behavior_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the three Stage 3 deep tests.")
    parser.add_argument("evaluation_report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional remapped evidence manifest; defaults to the historical IDs",
    )
    args = parser.parse_args()

    report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    criteria = CRITERIA
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        criteria = manifest["questions"]
    mode_scores: dict[str, list[dict[str, Any]]] = {}
    for mode in report["mode_order"]:
        rows: list[dict[str, Any]] = []
        for item in report["modes"].get(mode, []):
            row = {
                "question_id": item["id"],
                "status": item["status"],
                **score_result(
                    item["id"],
                    item.get("result", {}),
                    mode,
                    item["status"],
                    criteria,
                ),
            }
            rows.append(row)
        mode_scores[mode] = rows

    all_rows = [row for rows in mode_scores.values() for row in rows]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_report": str(args.evaluation_report),
        "evidence_manifest": str(args.manifest) if args.manifest else None,
        "prompt_version": report.get("configuration", {}).get("prompt_version"),
        "passed": bool(all_rows) and all(row["passed"] for row in all_rows),
        "passed_count": sum(bool(row["passed"]) for row in all_rows),
        "total_count": len(all_rows),
        "modes": mode_scores,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
