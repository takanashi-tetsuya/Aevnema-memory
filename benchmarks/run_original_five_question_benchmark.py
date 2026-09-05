from __future__ import annotations

"""Reproduce the 2026-09-04 five-question evidence-coverage check.

This is intentionally a diagnostic benchmark, not a truth-judging QA suite.
Its score is the number of predeclared evidence terms present in the first
eight selected Episodes.  The historical report used that exact definition
(nine matched terms out of fifteen).
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 5:
        raise ValueError("benchmark manifest must contain exactly five questions")
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("each benchmark row must be an object")
        terms = row.get("key_evidence_terms")
        if not isinstance(terms, list) or len(terms) != 3:
            raise ValueError("each question must declare exactly three evidence terms")
    return payload


def _configure_benchmark(config: AppConfig, mode: str, reranker: str) -> None:
    """Apply the historical standard retrieval budget without database writes."""

    config.paragraph.enabled = True
    config.retrieval.candidate_limit = 30
    config.retrieval.rerank_candidate_limit = 30
    config.retrieval.rerank_precompression_limit = 30
    config.retrieval.rerank_shortlist_limit = 8
    config.retrieval.graph_max_hops = 1
    config.retrieval.growth_max_rounds = 0
    config.retrieval.contextual_promotion_enabled = False
    config.retrieval.contextual_association_allow_network = False
    if reranker == "bge_enabled":
        config.retrieval.rerank_enabled = True
        config.retrieval.rerank_backend = "cross_encoder"
    elif reranker == "bge_disabled":
        config.retrieval.rerank_enabled = False
    else:
        raise ValueError(f"unsupported reranker mode: {reranker}")
    if mode == "baseline":
        config.retrieval.contextual_association_enabled = False
        config.retrieval.contextual_association_shadow = True
    elif mode == "contextual":
        config.retrieval.contextual_association_enabled = True
        config.retrieval.contextual_association_shadow = False
    else:
        raise ValueError(f"unsupported benchmark mode: {mode}")


def _excerpt(value: str, limit: int = 320) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _result_row(question: dict[str, Any], result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    evidence = list(result.get("evidence_episodes", []))[:8]
    haystack = "\n".join(str(item.get("text", "")) for item in evidence).casefold()
    terms = [str(term) for term in question["key_evidence_terms"]]
    hits = [term for term in terms if term.casefold() in haystack]
    contextual = dict(result.get("contextual_association", {}))
    return {
        "id": str(question["id"]),
        "label": str(question["label"]),
        "question": str(question["question"]),
        "key_evidence_terms": terms,
        "matched_terms": hits,
        "missing_terms": [term for term in terms if term not in hits],
        "coverage": {"matched": len(hits), "total": len(terms)},
        "elapsed_seconds": round(elapsed, 3),
        "candidate_episode_ids": list(result.get("candidate_episode_ids", [])),
        "selected_episode_ids": [int(item["id"]) for item in evidence],
        "selected_evidence": [
            {
                "id": int(item["id"]),
                "source_key": str(item.get("source_key", "")),
                "excerpt": _excerpt(str(item.get("text", ""))),
            }
            for item in evidence
        ],
        "contextual_association": {
            key: contextual.get(key)
            for key in (
                "enabled",
                "shadow",
                "candidate_count",
                "treatment_selected_count",
                "attached_edges",
                "attached_episode_ids",
                "new_slot_count",
                "harm_count",
                "external_calls",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the historical five-question evidence coverage benchmark"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifests/original_five_question_quality_benchmark.json"),
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--mode", choices=("baseline", "contextual"), required=True)
    parser.add_argument(
        "--reranker",
        choices=("bge_enabled", "bge_disabled"),
        required=True,
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=20.0,
        help="Per-question retrieval deadline; defaults to the historical 20-second budget.",
    )
    args = parser.parse_args()
    if args.deadline_seconds <= 0:
        raise ValueError("--deadline-seconds must be positive")

    questions = _load_manifest(args.manifest)
    config = AppConfig.from_env(args.env_file)
    config.database_path = args.database.resolve()
    config.log_dir = args.output.parent / "logs" / args.mode
    _configure_benchmark(config, args.mode, args.reranker)
    app = MemoryApplication(config)
    app.rebuild_indexes()
    engine = app.query_engine(app.new_logger(f"original-five-question-{args.mode}"))

    rows: list[dict[str, Any]] = []
    for question in questions:
        started = perf_counter()
        try:
            result = engine.query(
                str(question["question"]),
                generate_answer=False,
                deadline_seconds=args.deadline_seconds,
            )
        except Exception as error:  # keep an incomplete run inspectable
            rows.append(
                {
                    "id": str(question["id"]),
                    "label": str(question["label"]),
                    "question": str(question["question"]),
                    "key_evidence_terms": list(question["key_evidence_terms"]),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": round(perf_counter() - started, 3),
                }
            )
            continue
        rows.append(_result_row(question, result, perf_counter() - started))

    matched = sum(len(row.get("matched_terms", [])) for row in rows)
    total = sum(len(row.get("key_evidence_terms", [])) for row in rows)
    completed = [row for row in rows if "error" not in row]
    durations = [float(row["elapsed_seconds"]) for row in completed]
    report = {
        "benchmark": "original_five_question_evidence_coverage_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database": str(config.database_path),
        "mode": args.mode,
        "reranker": args.reranker,
        "configuration": {
            "paragraph_enabled": config.paragraph.enabled,
            "candidate_limit": config.retrieval.candidate_limit,
            "rerank_candidate_limit": config.retrieval.rerank_candidate_limit,
            "rerank_enabled": config.retrieval.rerank_enabled,
            "rerank_backend": config.retrieval.rerank_backend,
            "graph_max_hops": config.retrieval.graph_max_hops,
            "deadline_seconds": args.deadline_seconds,
            "contextual_association_enabled": config.retrieval.contextual_association_enabled,
            "contextual_association_shadow": config.retrieval.contextual_association_shadow,
            "contextual_promotion_enabled": config.retrieval.contextual_promotion_enabled,
            "model": {
                "base_url": config.model.base_url,
                "embedding_model": config.model.embedding_model,
                "reranker_model": config.model.reranker_model,
                "reasoning_model": config.model.reasoning_model,
                "fallback_model": config.model.fallback_model,
                "embedding_dimension": config.model.embedding_dimension,
                "timeout_seconds": config.model.timeout_seconds,
            },
        },
        "summary": {
            "questions": len(rows),
            "completed_questions": len(completed),
            "key_evidence_terms_matched": matched,
            "key_evidence_terms_total": total,
            "key_evidence_coverage": round(matched / total, 4) if total else 0.0,
            "average_elapsed_seconds": round(sum(durations) / len(durations), 3)
            if durations
            else None,
            "maximum_elapsed_seconds": round(max(durations), 3) if durations else None,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if len(completed) == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
