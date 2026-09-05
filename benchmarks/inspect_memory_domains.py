from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any


TABLES = ("source", "episode", "concept", "association")


def inspect(path: Path, sample_limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "tables": {},
        "episode_samples": [],
    }
    if not path.exists():
        return result
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        for table in TABLES:
            if table in names:
                count = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
                result["tables"][table] = int(count)
        if "episode" in names:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(episode)")
            }
            selected = [name for name in ("id", "text", "source_id") if name in columns]
            if selected:
                query = (
                    "SELECT "
                    + ", ".join(f'"{name}"' for name in selected)
                    + ' FROM "episode" ORDER BY id DESC LIMIT ?'
                )
                for row in connection.execute(query, (sample_limit,)):
                    item = dict(zip(selected, row, strict=True))
                    if "text" in item:
                        item["text"] = " ".join(str(item["text"]).split())[:500]
                    result["episode_samples"].append(item)
    finally:
        connection.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sample-limit", type=int, default=4)
    args = parser.parse_args()
    payload = [inspect(path.resolve(), max(0, args.sample_limit)) for path in args.paths]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
