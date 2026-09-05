from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import numpy as np

from memory_demo.config import AppConfig
from memory_demo.embeddings import normalize_embedding
from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm import ModelClient


def resolve_expected_episode_ids(rows, item: dict) -> tuple[set[int], list[dict]]:
    """Resolve stable evidence matchers to run-local Episode IDs."""

    expected_ids = {
        int(value) for value in item.get("expected_episode_ids", [])
    }
    matcher_results: list[dict] = []
    matchers = item.get("expected_episode_matchers", [])
    if not isinstance(matchers, list):
        raise ValueError("expected_episode_matchers must be a list")
    for matcher_index, matcher in enumerate(matchers):
        if not isinstance(matcher, dict):
            raise ValueError("each expected_episode_matcher must be an object")
        all_terms = [
            str(value).casefold() for value in matcher.get("all_terms", [])
        ]
        any_terms = [
            str(value).casefold() for value in matcher.get("any_terms", [])
        ]
        source_key = str(matcher.get("source_key", "")).casefold()
        matched_ids: list[int] = []
        for row in rows:
            if source_key and str(row["source_key"]).casefold() != source_key:
                continue
            haystack = (
                str(row["text"])
                + "\n"
                + str(row["evidence_quotes_json"])
            ).casefold()
            if all_terms and not all(term in haystack for term in all_terms):
                continue
            if any_terms and not any(term in haystack for term in any_terms):
                continue
            matched_ids.append(int(row["id"]))
        expected_ids.update(matched_ids)
        matcher_results.append(
            {
                "matcher_index": matcher_index,
                "matched_episode_ids": matched_ids,
                "resolved": bool(matched_ids),
            }
        )
    if not expected_ids and not matchers:
        raise ValueError(
            f"question {item.get('id', '<unknown>')} has no expected evidence"
        )
    return expected_ids, matcher_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate float32 dense recall for a freshly imported database"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--require-resolved-evidence",
        action="store_true",
        help="fail when any frozen evidence matcher resolves to no Episode",
    )
    parser.add_argument(
        "--min-recall-at-20",
        type=float,
        default=0.0,
        help="minimum inclusive dense Recall@20 required for exit code 0",
    )
    args = parser.parse_args()
    if not 0.0 <= float(args.min_recall_at_20) <= 1.0:
        raise ValueError("--min-recall-at-20 must be between 0 and 1")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    questions = manifest.get("questions", [])
    if not isinstance(questions, list) or not questions:
        raise ValueError("manifest.questions must be a non-empty list")

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT id, source_key, text, evidence_quotes_json, embedding
        FROM episode ORDER BY id
        """
    ).fetchall()
    connection.close()
    if not rows:
        raise ValueError("database contains no Episodes")

    config = AppConfig.from_env(args.env_file)
    dimensions = {len(row["embedding"]) // 4 for row in rows}
    if dimensions != {config.model.embedding_dimension}:
        raise ValueError(
            f"database embedding dimensions {sorted(dimensions)} do not match "
            f"configured {config.model.embedding_dimension}"
        )
    episode_ids = np.asarray([int(row["id"]) for row in rows], dtype=np.int64)
    matrix = np.stack(
        [np.frombuffer(row["embedding"], dtype="<f4") for row in rows]
    ).astype(np.float32, copy=False)
    logger = JsonlEventLogger(args.output.with_suffix(".jsonl"))
    model = ModelClient(config.model, logger)
    query_matrix = model.embed([str(item["question"]) for item in questions])

    result_rows: list[dict] = []
    for item, vector in zip(questions, query_matrix, strict=True):
        query = normalize_embedding(vector, config.model.embedding_dimension)
        scores = matrix @ query
        order = np.argsort(-scores, kind="stable")
        expected_ids, matcher_results = resolve_expected_episode_ids(rows, item)
        ranked_ids = [int(episode_ids[index]) for index in order]
        required_groups: list[set[int]] = []
        explicit_ids = {
            int(value) for value in item.get("expected_episode_ids", [])
        }
        if explicit_ids:
            required_groups.append(explicit_ids)
        required_groups.extend(
            {
                int(value) for value in detail["matched_episode_ids"]
            }
            for detail in matcher_results
        )
        group_best_ranks = [
            min(
                (
                    ranked_ids.index(expected_id) + 1
                    for expected_id in group
                    if expected_id in ranked_ids
                ),
                default=None,
            )
            for group in required_groups
        ]
        evidence_resolved = bool(required_groups) and all(required_groups)
        best_rank = (
            max(int(rank) for rank in group_best_ranks if rank is not None)
            if evidence_resolved and all(rank is not None for rank in group_best_ranks)
            else None
        )
        result_rows.append(
            {
                "id": str(item["id"]),
                "question": str(item["question"]),
                "expected_episode_ids": sorted(expected_ids),
                "expected_matchers": matcher_results,
                "expected_evidence_groups": [
                    sorted(group) for group in required_groups
                ],
                "group_best_ranks": group_best_ranks,
                "expected_evidence_resolved": evidence_resolved,
                "best_rank": best_rank,
                "top_10": [
                    {
                        "episode_id": int(episode_ids[index]),
                        "score": round(float(scores[index]), 6),
                        "text": str(rows[index]["text"]),
                    }
                    for index in order[:10]
                ],
            }
        )

    ranks = [
        int(item["best_rank"])
        for item in result_rows
        if item["best_rank"] is not None
    ]
    total = len(result_rows)
    unresolved = sum(
        not item["expected_evidence_resolved"] for item in result_rows
    )
    recall_at_20 = sum(rank <= 20 for rank in ranks) / total
    acceptance_passed = (
        (not args.require_resolved_evidence or unresolved == 0)
        and recall_at_20 >= float(args.min_recall_at_20)
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database": str(args.database.resolve()),
        "manifest": str(args.manifest.resolve()),
        "configuration": {
            "embedding_model": config.model.embedding_model,
            "embedding_dimension": config.model.embedding_dimension,
            "dtype": "float32",
            "growth_hops": 0,
            "reranker": "disabled",
            "episode_count": len(rows),
        },
        "summary": {
            "questions": total,
            "unresolved_evidence_questions": unresolved,
            "recall_at_1": sum(rank <= 1 for rank in ranks) / total,
            "recall_at_5": sum(rank <= 5 for rank in ranks) / total,
            "recall_at_10": sum(rank <= 10 for rank in ranks) / total,
            "recall_at_20": recall_at_20,
            "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / total,
            "worst_rank": max(ranks, default=None),
        },
        "acceptance": {
            "require_resolved_evidence": args.require_resolved_evidence,
            "min_recall_at_20": float(args.min_recall_at_20),
            "passed": acceptance_passed,
        },
        "rows": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if acceptance_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
