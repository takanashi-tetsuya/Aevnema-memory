from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from memory_demo.app import MemoryApplication
from memory_demo.config import AppConfig
from benchmarks.support.stage5 import load_json, write_json
from memory_demo.types import QueryIntent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iterate the LLM Top-20 reranker without repeating retrieval."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--question-id")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "validation/evaluation-stage7-generation/pilot/block-001/C/graph.db"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("validation/stage4-network-evidence-manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/base-rerank-saved-candidates.json"),
    )
    args = parser.parse_args()

    source = load_json(args.input)
    source_rows = source["rows"]
    if args.question_id:
        source_rows = [row for row in source_rows if row["id"] == args.question_id]
        if not source_rows:
            raise ValueError(f"unknown question id: {args.question_id}")
    source_row = source_rows[0]
    source_result = source_row["result"]
    question_id = str(source_row["id"])
    candidate_ids = [
        int(value)
        for value in source_result["rerank_trace"]["candidate_episode_ids"]
    ]
    config = AppConfig.from_env()
    config.database_path = args.database
    config.log_dir = args.output.parent / "base-rerank-iteration-logs"
    config.paragraph.enabled = False
    config.retrieval.graph_max_hops = 0
    config.retrieval.growth_max_rounds = 0
    app = MemoryApplication(config)
    app.rebuild_indexes()
    engine = app.query_engine()
    rows_by_id = {int(row["id"]): row for row in app.episodes.get_many(candidate_ids)}
    episodes = []
    for rank, node_id in enumerate(candidate_ids, start=1):
        row = rows_by_id[node_id]
        try:
            participants = json.loads(row["participants_json"])
        except (TypeError, json.JSONDecodeError):
            participants = []
        episodes.append(
            {
                "id": node_id,
                "score": 1.0 - (rank - 1) / max(1, len(candidate_ids)),
                "text": str(row["text"]),
                "participants": participants,
                "source_key": str(row["source_key"]),
                "story_time_text": str(row["story_time_text"]),
                "timeline_scope": str(row["timeline_scope"]),
            }
        )
    selected_ids, trace = engine._rerank_answer_episodes(
        str(source_row["question"]),
        QueryIntent.from_dict(source_result["intent"]),
        [str(value) for value in source_result["rerank_trace"]["atomic_queries"]],
        episodes,
        20,
    )
    criterion = load_json(args.manifest)["questions"][question_id]
    selected = set(selected_ids)
    groups = [
        {
            "alternatives": [int(value) for value in group],
            "matches": [int(value) for value in group if int(value) in selected],
        }
        for group in criterion["required_episode_groups"]
    ]
    hits = sum(bool(item["matches"]) for item in groups)
    write_json(
        args.output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "question_id": question_id,
            "selected_episode_ids": selected_ids,
            "required_group_hits": hits,
            "required_group_count": len(groups),
            "recall_at_20": hits / len(groups) if groups else 0.0,
            "groups": groups,
            "trace": trace,
        },
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
