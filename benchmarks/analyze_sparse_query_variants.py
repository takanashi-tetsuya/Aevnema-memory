from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sqlite3

from memory_demo.retrieval.sparse import SQLiteSparseIndex


class ReadOnlyDatabase:
    """Minimal Database-compatible adapter that cannot mutate the corpus."""

    def __init__(self, path: Path):
        self.path = path.resolve()

    @contextmanager
    def connection(self):
        uri = f"file:{self.path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare read-only multilingual sparse rankings."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--target-id", type=int, action="append", default=[])
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    index = SQLiteSparseIndex(ReadOnlyDatabase(args.database), "episode")
    for query in args.query:
        ranking = index.search(query, args.top_k)
        ids = [node_id for node_id, _score in ranking]
        print(query)
        for target_id in args.target_id:
            rank = ids.index(target_id) + 1 if target_id in ids else None
            print(f"target {target_id}: rank={rank}")
        print(f"top_ids={ids[: max(0, args.show)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
