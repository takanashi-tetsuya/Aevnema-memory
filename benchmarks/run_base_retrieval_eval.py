from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from benchmarks.support.stage5 import load_json, score_evidence_retrieval, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the no-growth Dense+Sparse+LLM-rerank retrieval stage."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "validation/evaluation-stage7-generation/pilot/block-001/C/graph.db"
        ),
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
        "--output",
        type=Path,
        default=Path("validation/base-retrieval-online-eval.json"),
    )
    parser.add_argument(
        "--question-id",
        action="append",
        help="run one question; repeat the option to select several",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed question rows from an interrupted output file",
    )
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument(
        "--paragraph-mode",
        choices=("off", "seed", "context", "both"),
        default="off",
        help="Paragraph contribution to candidate recall and/or rerank context",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="stop after Candidate@100 and reranked Top-20",
    )
    parser.add_argument(
        "--rerank-review-mode",
        choices=("strict", "adaptive", "lean"),
        default="strict",
        help="number of evidence-review passes; strict preserves historical runs",
    )
    args = parser.parse_args()

    config = AppConfig.from_env()
    config.model.max_retries = max(0, int(args.max_retries))
    config.database_path = args.database
    config.log_dir = args.output.parent / "base-retrieval-online-logs"
    config.paragraph.enabled = args.paragraph_mode != "off"
    config.retrieval.paragraph_seed_enabled = args.paragraph_mode in {"seed", "both"}
    config.retrieval.paragraph_rerank_context_enabled = args.paragraph_mode in {
        "context",
        "both",
    }
    config.retrieval.sparse_enabled = True
    config.retrieval.rerank_enabled = True
    config.retrieval.rerank_coverage_audit_enabled = True
    config.retrieval.rerank_audit_enabled = True
    config.retrieval.rerank_review_mode = args.rerank_review_mode
    config.retrieval.graph_max_hops = 0
    config.retrieval.growth_max_rounds = 0
    config.retrieval.answer_episode_limit = 20
    app = MemoryApplication(config)
    app.rebuild_indexes()
    questions = load_json(args.questions)
    if args.question_id:
        requested_ids = set(args.question_id)
        questions = [item for item in questions if item["id"] in requested_ids]
        if not questions:
            raise ValueError(f"unknown question ids: {sorted(requested_ids)}")
    manifest = load_json(args.manifest)
    question_ids = [str(item["id"]) for item in questions]

    def run_one(item: dict) -> dict:
        result = app.query_engine().query(
            str(item["question"]),
            generate_answer=not args.retrieval_only,
        )
        score = score_evidence_retrieval(
            result,
            manifest["questions"][str(item["id"])],
        )
        return {
            "id": str(item["id"]),
            "question": str(item["question"]),
            "score": score,
            "result": result,
        }

    def payload_for(current_rows: list[dict], *, complete: bool) -> dict:
        candidate_values = [
            float(row["score"]["candidate_recall"]) for row in current_rows
        ]
        selected_values = [
            float(row["score"]["recall_at_30"]) for row in current_rows
        ]
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": not complete,
            "configuration": {
                "source_database": str(args.database.resolve()),
                "question_ids": question_ids,
                "growth_max_rounds": 0,
                "graph_max_hops": 0,
                "paragraph_enabled": config.paragraph.enabled,
                "paragraph_mode": args.paragraph_mode,
                "sparse_enabled": True,
                "rerank_enabled": True,
                "rerank_coverage_audit_enabled": True,
                "rerank_audit_enabled": True,
                "rerank_review_mode": args.rerank_review_mode,
                "candidate_top_k": 100,
                "rerank_shortlist_top_k": config.retrieval.rerank_shortlist_limit,
                "selected_top_k": 20,
                "answer_generation_skipped": args.retrieval_only,
            },
            "summary": {
                "questions": len(current_rows),
                "candidate_recall_at_100_mean": (
                    sum(candidate_values) / len(candidate_values)
                    if candidate_values
                    else 0.0
                ),
                "candidate_recall_at_100_minimum": min(
                    candidate_values, default=0.0
                ),
                "selected_recall_at_20_mean": (
                    sum(selected_values) / len(selected_values)
                    if selected_values
                    else 0.0
                ),
                "selected_recall_at_20_minimum": min(
                    selected_values, default=0.0
                ),
                "candidate_questions_above_99_percent": sum(
                    value >= 0.99 for value in candidate_values
                ),
                "selected_questions_above_95_percent": sum(
                    value > 0.95 for value in selected_values
                ),
            },
            "rows": current_rows,
        }

    rows: list[dict] = []
    if args.resume and args.output.exists():
        previous = load_json(args.output)
        previous_configuration = previous.get("configuration", {})
        previous_database = previous_configuration.get("source_database")
        if previous_database and Path(previous_database).resolve() != args.database.resolve():
            raise ValueError("resume output belongs to a different source database")
        previous_ids = previous_configuration.get("question_ids")
        if previous_ids and list(previous_ids) != question_ids:
            raise ValueError("resume output uses a different question list")
        allowed_ids = set(question_ids)
        rows = [
            row
            for row in previous.get("rows", [])
            if str(row.get("id")) in allowed_ids
        ]
        completed_ids = {str(row["id"]) for row in rows}
        questions = [
            item for item in questions if str(item["id"]) not in completed_ids
        ]
    if args.workers <= 1:
        for item in questions:
            rows.append(run_one(item))
            write_json(args.output, payload_for(rows, complete=False))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one, item): item for item in questions}
            for future in as_completed(futures):
                rows.append(future.result())
                write_json(args.output, payload_for(rows, complete=False))
        order = {value: index for index, value in enumerate(question_ids)}
        rows.sort(key=lambda item: order[item["id"]])
    if args.workers <= 1:
        order = {value: index for index, value in enumerate(question_ids)}
        rows.sort(key=lambda item: order[item["id"]])
    write_json(args.output, payload_for(rows, complete=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
