from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np


def load_request(log_path: Path, request_id: str) -> tuple[list[str], np.ndarray]:
    inputs: list[str] | None = None
    vectors: np.ndarray | None = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("request_id") != request_id:
                continue
            if event.get("event") == "llm_request":
                raw = event.get("payload", {}).get("input", [])
                inputs = [str(item) for item in raw]
            elif event.get("event") == "llm_response":
                data = event.get("payload", {}).get("data", [])
                vectors = np.asarray(
                    [item["embedding"] for item in data], dtype=np.float32
                )
    if inputs is None or vectors is None:
        raise ValueError(f"request/response pair not found for {request_id}")
    return inputs, vectors


def load_episode_matrix(database: Path) -> tuple[np.ndarray, list[tuple[int, str, str]]]:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT id, source_key, text, embedding FROM episode ORDER BY id"
        ).fetchall()
    metadata = [(int(row[0]), str(row[1]), str(row[2])) for row in rows]
    matrix = np.stack(
        [np.frombuffer(row[3], dtype=np.float32) for row in rows]
    )
    return matrix, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect per-query Episode ranks using an embedding request saved in JSONL."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("request_id")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--episode-id",
        type=int,
        action="append",
        default=[],
        help="Also report the exact rank of this Episode ID (repeatable).",
    )
    parser.add_argument(
        "--targets-only",
        action="store_true",
        help="Suppress the ordinary top-k listing and print only requested Episode IDs.",
    )
    args = parser.parse_args()

    queries, query_matrix = load_request(args.log, args.request_id)
    episode_matrix, metadata = load_episode_matrix(args.database)
    for query, vector in zip(queries, query_matrix, strict=True):
        norm = np.linalg.norm(vector)
        normalized = vector / norm if norm else vector
        scores = episode_matrix @ normalized
        order = np.argsort(scores)[::-1]
        indices = order[: args.top_k]
        print(f"\nQUERY: {query}")
        if not args.targets_only:
            for rank, index in enumerate(indices, start=1):
                episode_id, source_key, text = metadata[int(index)]
                print(
                    f"{rank:>2}. {float(scores[index]):.6f} "
                    f"Episode {episode_id} {source_key}: {text[:180]}"
                )
        if args.episode_id:
            row_by_episode_id = {
                episode_id: row_index
                for row_index, (episode_id, _, _) in enumerate(metadata)
            }
            rank_by_row = np.empty_like(order)
            rank_by_row[order] = np.arange(1, len(order) + 1)
            for episode_id in args.episode_id:
                row_index = row_by_episode_id.get(episode_id)
                if row_index is None:
                    print(f"TARGET Episode {episode_id}: not found")
                    continue
                _, source_key, text = metadata[row_index]
                print(
                    f"TARGET rank {int(rank_by_row[row_index]):>3}, "
                    f"score {float(scores[row_index]):.6f}, "
                    f"Episode {episode_id} {source_key}: {text[:180]}"
                )


if __name__ == "__main__":
    main()
