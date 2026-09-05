from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from memory_demo.config import AppConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm import ModelClient
from memory_demo.retrieval_stability import validate_frozen_candidate_alignment
from benchmarks.support.stage5 import load_json, score_evidence_retrieval, write_json


_THREAD_LOCAL = threading.local()


def _model_for_worker(config: AppConfig, log_dir: Path) -> ModelClient:
    model = getattr(_THREAD_LOCAL, "model", None)
    if model is not None:
        return model
    thread_name = threading.current_thread().name.replace(" ", "-")
    logger = JsonlEventLogger(log_dir / f"atomic-bge-{thread_name}.jsonl")
    model = ModelClient(config.model, logger)
    _THREAD_LOCAL.model = model
    return model


def _load_candidate_rows(
    database: Path,
    source_rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    candidate_ids = sorted(
        {
            int(value)
            for row in source_rows
            for value in row["result"]["rerank_trace"]["candidate_episode_ids"]
        }
    )
    result: dict[int, dict[str, Any]] = {}
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for start in range(0, len(candidate_ids), 500):
            batch = candidate_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT id, text, participants_json, story_time_text, "
                "timeline_scope FROM episode WHERE id IN "
                f"({placeholders})"
            )
            for row in connection.execute(query, batch):
                try:
                    participants = json.loads(row["participants_json"] or "[]")
                except (TypeError, json.JSONDecodeError):
                    participants = []
                result[int(row["id"])] = {
                    "id": int(row["id"]),
                    "text": str(row["text"]),
                    "participants": (
                        participants if isinstance(participants, list) else []
                    ),
                    "story_time_text": str(row["story_time_text"] or ""),
                    "timeline_scope": str(row["timeline_scope"] or ""),
                }
    return result


def _candidate_document(row: dict[str, Any]) -> str:
    return "\n".join(
        value
        for value in (
            str(row["text"]),
            (
                "人物：" + ", ".join(str(value) for value in row["participants"])
                if row["participants"]
                else ""
            ),
            (
                "故事时间：" + str(row["story_time_text"])
                if row["story_time_text"]
                else ""
            ),
            (
                "时间线：" + str(row["timeline_scope"])
                if row["timeline_scope"]
                else ""
            ),
        )
        if value
    )


def _planned_queries(row: dict[str, Any], limit: int) -> list[str]:
    values = []
    question = str(row["question"]).strip()
    raw_queries = row["result"]["rerank_trace"].get("atomic_queries", [])
    if not raw_queries:
        raw_queries = row["result"]["intent"].get("search_queries", [])
    for raw in raw_queries:
        value = str(raw).strip()
        if (
            not value
            or value == question
            or value.startswith("__constraint_slot__ ")
        ):
            continue
        for prefix in ("__answer_slot__ ", "__coverage_slot__ "):
            if value.startswith(prefix):
                value = value[len(prefix) :].strip()
        if value and value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def _round_robin_select(
    rankings: list[list[int]],
    limit: int,
) -> list[int]:
    selected: list[int] = []
    maximum = max((len(values) for values in rankings), default=0)
    for rank in range(maximum):
        for values in rankings:
            if rank >= len(values):
                continue
            episode_id = int(values[rank])
            if episode_id not in selected:
                selected.append(episode_id)
            if len(selected) >= limit:
                return selected
    return selected


def _greedy_cover_select(
    rankings: list[list[int]],
    whole_question_ranking: list[int],
    limit: int,
    coverage_depth: int,
) -> list[int]:
    options = [values[:coverage_depth] for values in rankings if values]
    uncovered = set(range(len(options)))
    selected: list[int] = []
    while uncovered and len(selected) < limit:
        candidates = {
            int(episode_id)
            for query_index in uncovered
            for episode_id in options[query_index]
            if int(episode_id) not in selected
        }
        if not candidates:
            break
        best_id = max(
            candidates,
            key=lambda episode_id: (
                sum(
                    1.0 / (options[index].index(episode_id) + 1)
                    for index in uncovered
                    if episode_id in options[index]
                ),
                -whole_question_ranking.index(episode_id)
                if episode_id in whole_question_ranking
                else -len(whole_question_ranking),
            ),
        )
        selected.append(best_id)
        uncovered = {
            index for index in uncovered if best_id not in options[index]
        }
    for episode_id in whole_question_ranking:
        if len(selected) >= limit:
            break
        if episode_id not in selected:
            selected.append(int(episode_id))
    return selected


def _score_ids(
    source_row: dict[str, Any],
    criterion: dict[str, Any],
    selected_ids: list[int],
) -> dict[str, Any]:
    result = dict(source_row["result"])
    result["episode_ids"] = list(selected_ids)
    result["candidate_episode_ids"] = [
        int(value)
        for value in source_row["result"]["rerank_trace"][
            "candidate_episode_ids"
        ]
    ]
    return score_evidence_retrieval(result, criterion)


def _run_question(
    row: dict[str, Any],
    criterion: dict[str, Any],
    episode_rows: dict[int, dict[str, Any]],
    config: AppConfig,
    log_dir: Path,
    query_limit: int,
    selected_limit: int,
    coverage_depth: int,
) -> dict[str, Any]:
    candidate_ids = [
        int(value)
        for value in row["result"]["rerank_trace"]["candidate_episode_ids"]
    ]
    documents = [_candidate_document(episode_rows[value]) for value in candidate_ids]
    planned_queries = _planned_queries(row, query_limit)
    all_queries = [str(row["question"]), *planned_queries]
    model = _model_for_worker(config, log_dir)
    started = time.perf_counter()
    rankings: list[list[int]] = []
    scored_rankings: list[dict[str, Any]] = []
    for query in all_queries:
        ranked = model.rerank(query, documents, top_n=len(documents))
        ids = [candidate_ids[int(item["index"])] for item in ranked]
        rankings.append(ids)
        scored_rankings.append(
            {
                "query": query,
                "episode_ids": ids,
                "scores": [float(item["relevance_score"]) for item in ranked],
            }
        )
    whole_question = rankings[0]
    slot_rankings = rankings[1:]
    arms = {
        "whole_question": whole_question[:selected_limit],
        "slot_round_robin": _round_robin_select(
            slot_rankings,
            selected_limit,
        ),
        "slot_greedy_cover": _greedy_cover_select(
            slot_rankings,
            whole_question,
            selected_limit,
            coverage_depth,
        ),
    }
    for selected in arms.values():
        for episode_id in whole_question:
            if len(selected) >= selected_limit:
                break
            if episode_id not in selected:
                selected.append(int(episode_id))
    return {
        "id": str(row["id"]),
        "question": str(row["question"]),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "planned_queries": planned_queries,
        "candidate_episode_ids": candidate_ids,
        "rankings": scored_rankings,
        "arms": {
            name: {
                "selected_episode_ids": selected,
                "score": _score_ids(row, criterion, selected),
            }
            for name, selected in arms.items()
        },
    }


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = sorted({name for row in rows for name in row["arms"]})
    result = []
    for name in names:
        values = [float(row["arms"][name]["score"]["recall_at_30"]) for row in rows]
        result.append(
            {
                "arm": name,
                "questions": len(values),
                "mean_recall_at_20": sum(values) / len(values) if values else 0.0,
                "minimum_recall_at_20": min(values, default=0.0),
                "questions_above_95_percent": sum(value > 0.95 for value in values),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare whole-question and per-slot BGE reranking."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=24)
    parser.add_argument("--selected-limit", type=int, default=20)
    parser.add_argument("--coverage-depth", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = load_json(args.input)
    rows = list(source["rows"])
    manifest = load_json(args.manifest)["questions"]
    episode_rows = _load_candidate_rows(args.database, rows)
    alignment = validate_frozen_candidate_alignment(
        rows,
        {episode_id: str(value["text"]) for episode_id, value in episode_rows.items()},
    )
    if not alignment["passed"]:
        raise ValueError("candidate database failed semantic alignment")

    config = AppConfig.from_env()
    config.model.max_retries = max(0, int(args.max_retries))
    if not str(config.model.reranker_model).strip():
        raise ValueError("MEMORY_RERANKER_MODEL must not be empty")
    log_dir = args.output.parent / f"{args.output.stem}-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        completed = list(load_json(args.output).get("rows", []))
    completed_ids = {str(row["id"]) for row in completed}

    def payload(checkpoint: bool) -> dict[str, Any]:
        ordered = sorted(completed, key=lambda item: str(item["id"]))
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint,
            "configuration": {
                "input": str(args.input.resolve()),
                "database": str(args.database.resolve()),
                "manifest": str(args.manifest.resolve()),
                "reranker_model": config.model.reranker_model,
                "query_limit": args.query_limit,
                "selected_limit": args.selected_limit,
                "coverage_depth": args.coverage_depth,
                "database_alignment": alignment,
            },
            "summary": _summary(ordered),
            "rows": ordered,
        }

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                _run_question,
                row,
                manifest[str(row["id"])],
                episode_rows,
                config,
                log_dir,
                args.query_limit,
                args.selected_limit,
                args.coverage_depth,
            ): str(row["id"])
            for row in rows
            if str(row["id"]) not in completed_ids
        }
        for future in as_completed(futures):
            completed.append(future.result())
            write_json(args.output, payload(checkpoint=True))
            print(f"completed {futures[future]}", flush=True)
    write_json(args.output, payload(checkpoint=False))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
