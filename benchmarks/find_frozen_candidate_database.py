from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any

from memory_demo.retrieval_stability import validate_frozen_candidate_alignment
from benchmarks.support.stage5 import load_json, sha256_file, write_json


def _candidate_ids(source_rows: list[dict[str, Any]]) -> list[int]:
    return sorted(
        {
            int(value)
            for row in source_rows
            for value in (
                row.get("result", {})
                .get("rerank_trace", {})
                .get("candidate_episode_ids", [])
            )
        }
    )


def _load_texts(database: Path, candidate_ids: list[int]) -> dict[int, str]:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    rows: dict[int, str] = {}
    with sqlite3.connect(uri, uri=True) as connection:
        episode_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='episode'"
        ).fetchone()
        if not episode_table:
            raise ValueError("episode table is absent")
        for start in range(0, len(candidate_ids), 500):
            batch = candidate_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            for episode_id, text in connection.execute(
                f"SELECT id, text FROM episode WHERE id IN ({placeholders})",
                batch,
            ):
                rows[int(episode_id)] = str(text)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Locate the SQLite snapshot whose Episode ID namespace matches a "
            "saved retrieval report."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("search_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_rows = list(load_json(args.input)["rows"])
    candidate_ids = _candidate_ids(source_rows)
    results: list[dict[str, Any]] = []
    for database in sorted(args.search_root.rglob("*.db")):
        try:
            texts = _load_texts(database, candidate_ids)
            alignment = validate_frozen_candidate_alignment(source_rows, texts)
            results.append(
                {
                    "database": str(database.resolve()),
                    "bytes": database.stat().st_size,
                    "alignment": alignment,
                    "error": None,
                }
            )
        except (OSError, sqlite3.DatabaseError, ValueError) as exc:
            results.append(
                {
                    "database": str(database.resolve()),
                    "bytes": database.stat().st_size,
                    "alignment": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    results.sort(
        key=lambda item: (
            1 if item["alignment"] is None else 0,
            (
                10**9
                if item["alignment"] is None
                else int(item["alignment"]["evidence_text_mismatch_count"])
                + len(item["alignment"]["missing_candidate_ids"])
            ),
            str(item["database"]),
        )
    )
    exact = [
        item for item in results if item["alignment"] and item["alignment"]["passed"]
    ]
    for item in exact:
        item["sha256"] = sha256_file(item["database"])
    write_json(
        args.output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": str(args.input.resolve()),
            "input_sha256": sha256_file(args.input),
            "search_root": str(args.search_root.resolve()),
            "candidate_id_count": len(candidate_ids),
            "database_count": len(results),
            "exact_match_count": len(exact),
            "exact_matches": exact,
            "best_matches": results[:20],
        },
    )
    print(f"exact matches: {len(exact)}")
    for item in exact:
        print(item["database"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
