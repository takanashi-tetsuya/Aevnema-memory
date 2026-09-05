from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from memory_demo.ingestion.extractor import MemoryExtractor
from memory_demo.llm.validation import parse_episode_fact_review_payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the latest fact-audit response through local gates"
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("source_key")
    parser.add_argument("segment_index", type=int)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    responses = []
    with args.log.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("event") == "llm_response":
                responses.append(item)
    if not responses:
        raise SystemExit("no llm_response event found")
    content = responses[-1]["payload"]["choices"][0]["message"]["content"]
    payload = json.loads(content[content.find("{") : content.rfind("}") + 1])
    reviews, parse_errors = parse_episode_fact_review_payload(
        payload, set(range(args.expected_count))
    )
    connection = sqlite3.connect(args.database)
    row = connection.execute(
        """
        SELECT s.raw_text
        FROM source s JOIN episode e ON e.source_id = s.id
        WHERE e.source_key = ? AND e.segment_index = ?
        ORDER BY s.id DESC LIMIT 1
        """,
        (args.source_key, args.segment_index),
    ).fetchone()
    connection.close()
    if row is None:
        raise SystemExit("source not found")
    evidence_errors = MemoryExtractor._validate_fact_review_evidence(
        str(row[0]), reviews
    )
    print(
        json.dumps(
            {
                "parse_errors": parse_errors,
                "evidence_errors": evidence_errors,
                "review_count": len(reviews),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not parse_errors and not evidence_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
