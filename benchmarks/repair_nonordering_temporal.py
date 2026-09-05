from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repair_nonordering_temporal(
    database: str | Path, *, apply: bool = False
) -> dict:
    """Remove temporal edges that cannot participate in an ordering graph.

    The durable temporal invariant is deliberately narrow: ``before`` and
    ``after`` are the only accepted keys.  Simultaneity or broad period
    similarity can still be represented as semantic/co-occurrence relations,
    but an LLM-created generic ``same_time`` edge is not silently reclassified.
    """

    path = Path(database).resolve()
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT id, from_type, from_id, to_type, to_id, relation_key,
                   relation_text, weight, confidence, generation,
                   claim_level, created_reason
            FROM association
            WHERE relation_type = 'temporal'
              AND relation_key NOT IN ('before', 'after')
            ORDER BY id
            """
        ).fetchall()
        edge_ids = [int(row["id"]) for row in rows]
        possible_dependents: list[int] = []
        for edge_id in edge_ids:
            matches = connection.execute(
                """
                SELECT id FROM association
                WHERE id <> ? AND (evidence_json LIKE ? OR audit_json LIKE ?)
                ORDER BY id
                """,
                (edge_id, f"%{edge_id}%", f"%{edge_id}%"),
            ).fetchall()
            possible_dependents.extend(int(row["id"]) for row in matches)
        possible_dependents = sorted(set(possible_dependents))
        report = {
            "database": str(path),
            "applied": bool(apply),
            "edge_count": len(edge_ids),
            "edge_ids": edge_ids,
            "possible_dependent_edge_ids": possible_dependents,
            "edges": [dict(row) for row in rows],
        }
        if not apply or not edge_ids:
            return report
        if possible_dependents:
            raise RuntimeError(
                "non-ordering temporal edges may be referenced by derived edges; "
                "manual review required"
            )
        placeholders = ",".join("?" for _ in edge_ids)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                f"DELETE FROM association WHERE id IN ({placeholders})",
                edge_ids,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        report["applied_at"] = utc_now()
        return report
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit or remove non-ordering temporal associations"
    )
    parser.add_argument("database")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", help="optional UTF-8 JSON report path")
    args = parser.parse_args()
    report = repair_nonordering_temporal(args.database, apply=args.apply)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
