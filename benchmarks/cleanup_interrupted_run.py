from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def placeholders(values: list[int]) -> str:
    return ",".join("?" for _ in values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely remove an interrupted extraction run with no persisted Episodes"
    )
    parser.add_argument("database")
    parser.add_argument("run_id", type=int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    database = Path(args.database).resolve()
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        run = connection.execute(
            "SELECT id, status, started_at, finished_at FROM extraction_run WHERE id = ?",
            (args.run_id,),
        ).fetchone()
        if run is None:
            raise SystemExit(f"extraction run does not exist: {args.run_id}")

        tasks = connection.execute(
            """
            SELECT id, source_id, source_key, segment_index, status
            FROM extraction_task
            WHERE run_id = ?
            ORDER BY id
            """,
            (args.run_id,),
        ).fetchall()
        source_ids = sorted(
            {int(row["source_id"]) for row in tasks if row["source_id"] is not None}
        )
        episode_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM episode WHERE extraction_run_id = ?",
                (args.run_id,),
            ).fetchone()[0]
        )

        source_episode_ids: list[int] = []
        cross_run_source_ids: list[int] = []
        if source_ids:
            marks = placeholders(source_ids)
            source_episode_ids = [
                int(row[0])
                for row in connection.execute(
                    f"SELECT DISTINCT source_id FROM episode WHERE source_id IN ({marks})",
                    source_ids,
                )
            ]
            cross_run_source_ids = [
                int(row[0])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT source_id
                    FROM extraction_task
                    WHERE source_id IN ({marks}) AND run_id <> ?
                    """,
                    [*source_ids, args.run_id],
                )
            ]

        report = {
            "database": str(database),
            "run": dict(run),
            "task_count": len(tasks),
            "task_statuses": {
                status: sum(1 for row in tasks if row["status"] == status)
                for status in sorted({str(row["status"]) for row in tasks})
            },
            "source_ids": source_ids,
            "episode_count_for_run": episode_count,
            "source_ids_with_episodes": source_episode_ids,
            "source_ids_referenced_by_other_runs": cross_run_source_ids,
            "safe_to_delete": not episode_count
            and not source_episode_ids
            and not cross_run_source_ids,
            "applied": False,
        }
        if args.apply:
            if not report["safe_to_delete"]:
                raise SystemExit(
                    "refusing cleanup: run or its Sources already have persisted data"
                )
            with connection:
                connection.execute(
                    "DELETE FROM extraction_task WHERE run_id = ?", (args.run_id,)
                )
                if source_ids:
                    connection.execute(
                        f"DELETE FROM source WHERE id IN ({placeholders(source_ids)})",
                        source_ids,
                    )
                connection.execute(
                    "DELETE FROM extraction_run WHERE id = ?", (args.run_id,)
                )
            report["applied"] = True
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
