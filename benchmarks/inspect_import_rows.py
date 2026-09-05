from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only inspection of imported Source/Episode rows"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--source-key")
    parser.add_argument(
        "--episode-id",
        action="append",
        type=int,
        default=[],
        help="return one Episode ID; repeatable",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="return Episode text containing any supplied literal; repeatable",
    )
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    uri = f"file:{args.database.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        parameters: list[object] = []
        predicates: list[str] = []
        if args.source_key:
            predicates.append("source_key = ?")
            parameters.append(args.source_key)
        if args.episode_id:
            episode_ids = list(dict.fromkeys(int(value) for value in args.episode_id))
            predicates.append(
                "id IN (" + ",".join("?" for _ in episode_ids) + ")"
            )
            parameters.extend(episode_ids)
        if args.contains:
            predicates.append(
                "(" + " OR ".join("instr(text, ?) > 0" for _ in args.contains) + ")"
            )
            parameters.extend(args.contains)
        where = "WHERE " + " AND ".join(predicates) if predicates else ""
        parameters.append(max(1, int(args.limit)))
        rows = connection.execute(
            f"""
            SELECT id, source_id, source_key, segment_index, text,
                   participants_json, event_type, story_time_text,
                   confidence, evidence_origin, epistemic_status, generation
            FROM episode
            {where}
            ORDER BY id
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        payload = [dict(row) for row in rows]
        for item in payload:
            try:
                item["participants"] = json.loads(item.pop("participants_json"))
            except (TypeError, json.JSONDecodeError):
                item["participants"] = []
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
