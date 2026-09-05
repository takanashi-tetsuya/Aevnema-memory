from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from memory_demo.config import AppConfig
from memory_demo.database import Database
from memory_demo.repositories import EpisodeRepository
from memory_demo.retrieval.engine import QueryEngine
from memory_demo.retrieval.query_planning import structural_queries
from memory_demo.retrieval.sparse import SQLiteSparseIndex
from memory_demo.types import QueryIntent
from score_stage3_evaluation import score_result


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_payload(repository: EpisodeRepository, episode_id: int) -> dict:
    row = repository.get(episode_id)
    if row is None:
        raise ValueError(f"missing Episode {episode_id}")
    return {
        "id": int(row["id"]),
        "source_key": str(row["source_key"]),
        "text": str(row["text"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved plan's final evidence with deterministic "
            "cross-constraint sparse slots; no model or embedding calls."
        )
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("evaluation_report", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mode", default="graph_static")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = load(args.plan)
    report = load(args.evaluation_report)
    manifest = load(args.manifest)
    question = str(plan["question"])
    intent = QueryIntent.from_dict(dict(plan["intent"]))
    constraint_queries = [
        query
        for query in structural_queries(question, intent)
        if query.startswith(("__constraint_slot__ ", "__answer_slot__ "))
    ]

    database = Database(args.database)
    episodes = EpisodeRepository(database)
    sparse = SQLiteSparseIndex(database, "episode")
    config = AppConfig()
    per_query = max(0, config.retrieval.rerank_constraint_floor_per_query)
    total_limit = max(0, config.retrieval.rerank_constraint_floor_total_limit)
    slot_rows: list[dict] = []
    floor_ids: list[int] = []
    for query in constraint_queries:
        ranked = sparse.search(query, 20)
        admitted: list[int] = []
        for episode_id, _score in ranked[:per_query]:
            if episode_id not in floor_ids and len(floor_ids) < total_limit:
                floor_ids.append(episode_id)
                admitted.append(episode_id)
        slot_rows.append(
            {
                "query": query,
                "sparse_top20": [
                    {"episode_id": int(episode_id), "score": float(score)}
                    for episode_id, score in ranked
                ],
                "floor_episode_ids": admitted,
            }
        )

    mode_rows = report["modes"][args.mode]
    source_row = next(
        item for item in mode_rows if item["question"] == question
    )
    old_result = dict(source_row["result"])
    old_ids = [int(value) for value in old_result["episode_ids"]]
    allowed_ids = {
        *[int(value) for value in plan["rerank_trace"]["candidate_episode_ids"]],
        *floor_ids,
    }
    new_ids, replacements = QueryEngine._enforce_selection_floor(
        old_ids,
        floor_ids,
        allowed_ids,
        int(config.retrieval.answer_episode_limit),
    )

    criteria = manifest["questions"]
    question_id = str(source_row["id"])
    old_score = score_result(
        question_id,
        old_result,
        args.mode,
        str(source_row["status"]),
        criteria,
    )
    replay_result = {
        **old_result,
        "episode_ids": new_ids,
        "evidence_episodes": [
            episode_payload(episodes, episode_id) for episode_id in new_ids
        ],
    }
    new_score = score_result(
        question_id,
        replay_result,
        args.mode,
        str(source_row["status"]),
        criteria,
    )
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "offline_sparse_constraint_slot_replay_v1",
        "model_calls": 0,
        "embedding_calls": 0,
        "question_id": question_id,
        "plan_id": plan.get("plan_id"),
        "constraint_slots": slot_rows,
        "old_episode_ids": old_ids,
        "new_episode_ids": new_ids,
        "replacements": replacements,
        "old_score": old_score,
        "new_score": new_score,
        "interpretation": (
            "Diagnostic only: the corpus-grounded protected sparse slots close "
            "the frozen evidence gap; a full online rebuild is still required."
            if new_score["passed"]
            else "Diagnostic only: the protected sparse slots do not yet close "
            "the frozen evidence gap; do not proceed as if this passed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

