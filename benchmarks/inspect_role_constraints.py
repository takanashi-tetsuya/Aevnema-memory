from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Inspect Concept and Association evidence for role-qualified queries"
    )
    parser.add_argument("database")
    parser.add_argument("terms", nargs="+")
    args = parser.parse_args()
    connection = sqlite3.connect(Path(args.database))
    connection.row_factory = sqlite3.Row
    concepts = connection.execute(
        "SELECT id, canonical_name, description FROM concept ORDER BY id"
    ).fetchall()
    matching = [
        row
        for row in concepts
        if any(
            term.casefold()
            in f"{row['canonical_name']} {row['description']}".casefold()
            for term in args.terms
        )
    ]
    matching_ids = {int(row["id"]) for row in matching}
    concept_by_id = {int(row["id"]): row for row in concepts}
    edges = []
    for row in connection.execute(
        """
        SELECT id, from_type, from_id, to_type, to_id, relation_type,
               relation_key, relation_text, weight, confidence, evidence_count
        FROM association ORDER BY id
        """
    ).fetchall():
        if not (
            (row["from_type"] == "concept" and int(row["from_id"]) in matching_ids)
            or (row["to_type"] == "concept" and int(row["to_id"]) in matching_ids)
        ):
            continue
        rendered = dict(row)
        for side in ("from", "to"):
            if row[f"{side}_type"] == "concept":
                concept = concept_by_id.get(int(row[f"{side}_id"]))
                rendered[f"{side}_label"] = (
                    str(concept["canonical_name"]) if concept else "<missing>"
                )
            else:
                episode = connection.execute(
                    "SELECT source_key, text FROM episode WHERE id = ?",
                    (int(row[f"{side}_id"]),),
                ).fetchone()
                rendered[f"{side}_label"] = (
                    f"{episode['source_key']}：{episode['text']}" if episode else "<missing>"
                )
        edges.append(rendered)
    print(
        json.dumps(
            {"concepts": [dict(row) for row in matching], "associations": edges},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
