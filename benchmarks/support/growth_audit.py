from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any


def _json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def audit_growth(report_path: str | Path) -> dict[str, Any]:
    report_file = Path(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    rows = report.get("modes", {}).get("graph_growing", [])
    database_path = report_file.parent / "graph_growing.db"

    prior_changed: set[int] = set()
    changed_ids: set[int] = set()
    question_audits: list[dict[str, Any]] = []
    for row in rows:
        result = row.get("result", {}) if isinstance(row, dict) else {}
        created = {int(value) for value in result.get("new_association_ids", [])}
        reinforced = {
            int(value) for value in result.get("reinforced_association_ids", [])
        }
        current_changed = created | reinforced
        used = {int(value) for value in result.get("association_ids", [])}
        question_audits.append(
            {
                "question_id": row.get("id"),
                "status": row.get("status"),
                "changed_associations_used": current_changed.issubset(used),
                "reused_prior_association_ids": sorted(prior_changed & used),
                "created_association_ids": sorted(created),
                "reinforced_association_ids": sorted(reinforced),
            }
        )
        changed_ids.update(current_changed)
        prior_changed.update(current_changed)

    edge_audits: list[dict[str, Any]] = []
    if database_path.exists() and changed_ids:
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        try:
            placeholders = ",".join("?" for _ in changed_ids)
            database_rows = {
                int(row["id"]): row
                for row in connection.execute(
                    f"SELECT * FROM association WHERE id IN ({placeholders})",
                    sorted(changed_ids),
                )
            }
        finally:
            connection.close()
        for association_id in sorted(changed_ids):
            row = database_rows.get(association_id)
            if row is None:
                edge_audits.append(
                    {
                        "association_id": association_id,
                        "passed": False,
                        "issues": ["changed association is missing from mode database"],
                    }
                )
                continue
            evidence = _json_list(row["evidence_json"])
            audits = _json_list(row["audit_json"])
            issues: list[str] = []
            if str(row["audit_status"]) != "dual_accepted":
                issues.append("audit_status is not dual_accepted")
            if str(row["claim_level"]) not in {
                "direct_fact",
                "supported_inference",
                "historical_context",
            }:
                issues.append("claim_level is not evidence-safe")
            if len(evidence) < 2:
                issues.append("evidence_json does not contain both endpoints")
            if not audits:
                issues.append("audit_json is empty")
            for audit in audits:
                if not isinstance(audit, dict):
                    issues.append("audit_json contains a non-object item")
                    continue
                if audit.get("primary_accept") is not True:
                    issues.append("primary audit did not accept")
                if audit.get("adversarial_accept") is not True:
                    issues.append("adversarial audit did not accept")
            edge_audits.append(
                {
                    "association_id": association_id,
                    "passed": not issues,
                    "issues": list(dict.fromkeys(issues)),
                    "claim_level": row["claim_level"],
                    "audit_status": row["audit_status"],
                    "relation_key": row["relation_key"],
                    "relation_text": row["relation_text"],
                }
            )

    completed_questions = [
        item for item in question_audits if item["status"] == "completed"
    ]
    all_questions_complete = bool(question_audits) and len(completed_questions) == len(
        question_audits
    )
    return {
        "evaluation_report": str(report_file),
        "database": str(database_path),
        "report_status": report.get("status"),
        "question_order": report.get("question_order", "original"),
        "passed": (
            report.get("status") == "completed"
            and all_questions_complete
            and all(item["changed_associations_used"] for item in question_audits)
            and all(item["passed"] for item in edge_audits)
        ),
        "question_audits": question_audits,
        "edge_audits": edge_audits,
        "reuse_diagnostic": {
            "questions_reusing_prior_changes": [
                item["question_id"]
                for item in question_audits
                if item["reused_prior_association_ids"]
            ],
            "note": "仅报告复用是否发生；新增、强化和复用数量都不是通过指标。",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit_growth(args.report)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
