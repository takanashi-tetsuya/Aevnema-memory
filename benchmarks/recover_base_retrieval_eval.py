from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from benchmarks.support.stage5 import load_json, score_evidence_retrieval, write_json


def _row(question_id: str, question: str, result: dict, criterion: dict) -> dict:
    return {
        "id": question_id,
        "question": question,
        "score": score_evidence_retrieval(result, criterion),
        "result": result,
    }


def _summary(rows: list[dict]) -> dict[str, Any]:
    candidate_values = [float(row["score"]["candidate_recall"]) for row in rows]
    selected_values = [float(row["score"]["recall_at_30"]) for row in rows]
    return {
        "questions": len(rows),
        "candidate_recall_at_100_mean": (
            sum(candidate_values) / len(candidate_values) if candidate_values else 0.0
        ),
        "candidate_recall_at_100_minimum": min(candidate_values, default=0.0),
        "selected_recall_at_20_mean": (
            sum(selected_values) / len(selected_values) if selected_values else 0.0
        ),
        "selected_recall_at_20_minimum": min(selected_values, default=0.0),
        "candidate_questions_above_99_percent": sum(
            value >= 0.99 for value in candidate_values
        ),
        "selected_questions_above_95_percent": sum(
            value > 0.95 for value in selected_values
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover a base-retrieval evaluation from completed JSONL logs."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("validation/evaluation-questions-stage4-network.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("validation/stage4-network-evidence-manifest.json"),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("validation/base-retrieval-online-logs"),
    )
    parser.add_argument("--log-pattern", default="query-*.jsonl")
    parser.add_argument("--part", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = load_json(args.questions)
    manifest = load_json(args.manifest)
    question_by_text = {str(item["question"]): item for item in questions}
    recovered: dict[str, dict] = {}
    provenance: dict[str, str] = {}

    for log_path in sorted(args.logs_dir.glob(args.log_pattern)):
        question = ""
        result: dict | None = None
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("event") == "query_started":
                    question = str(event.get("question", ""))
                elif event.get("event") == "retrieval_completed":
                    result = event.get("result")
        except (OSError, json.JSONDecodeError):
            continue
        item = question_by_text.get(question)
        if item is None or not isinstance(result, dict):
            continue
        question_id = str(item["id"])
        recovered[question_id] = _row(
            question_id,
            question,
            result,
            manifest["questions"][question_id],
        )
        provenance[question_id] = str(log_path)

    for part_path in args.part:
        payload = load_json(part_path)
        for row in payload.get("rows", []):
            question_id = str(row["id"])
            recovered[question_id] = row
            provenance[question_id] = str(part_path)

    missing = [str(item["id"]) for item in questions if str(item["id"]) not in recovered]
    if missing:
        raise RuntimeError(f"missing completed results: {', '.join(missing)}")

    rows = [recovered[str(item["id"])] for item in questions]
    write_json(
        args.output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "recovered_from_interrupted_run": True,
            "configuration": {
                "growth_max_rounds": 0,
                "graph_max_hops": 0,
                "paragraph_enabled": False,
                "sparse_enabled": True,
                "rerank_enabled": True,
                "rerank_audit_enabled": True,
                "candidate_top_k": 100,
                "rerank_shortlist_top_k": 48,
                "selected_top_k": 20,
                "answer_generation_skipped": True,
            },
            "summary": _summary(rows),
            "provenance": provenance,
            "rows": rows,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
