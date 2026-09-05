from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np


def _load(path: Path) -> dict[str, list[dict[str, object]]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT id, source_key, text, embedding FROM episode ORDER BY id"
    ).fetchall()
    connection.close()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        grouped.setdefault(str(row["source_key"]), []).append(
            {
                "id": int(row["id"]),
                "text": str(row["text"]),
                "vector": vector,
            }
        )
    return grouped


def _directional_matches(
    source: list[dict[str, object]],
    target: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not source:
        return []
    if not target:
        return [
            {
                "id": item["id"],
                "text": item["text"],
                "nearest_id": None,
                "nearest_text": "",
                "similarity": -1.0,
            }
            for item in source
        ]
    target_matrix = np.stack(
        [np.asarray(item["vector"], dtype=np.float32) for item in target]
    )
    result: list[dict[str, object]] = []
    for item in source:
        scores = target_matrix @ np.asarray(item["vector"], dtype=np.float32)
        nearest_index = int(np.argmax(scores))
        nearest = target[nearest_index]
        result.append(
            {
                "id": item["id"],
                "text": item["text"],
                "nearest_id": nearest["id"],
                "nearest_text": nearest["text"],
                "similarity": round(float(scores[nearest_index]), 6),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Episode coverage between two import databases"
    )
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output")
    args = parser.parse_args()

    baseline = _load(Path(args.baseline))
    candidate = _load(Path(args.candidate))
    source_keys = sorted(set(baseline) | set(candidate))
    sources: dict[str, object] = {}
    all_baseline_matches: list[dict[str, object]] = []
    all_candidate_matches: list[dict[str, object]] = []
    for source_key in source_keys:
        baseline_items = baseline.get(source_key, [])
        candidate_items = candidate.get(source_key, [])
        baseline_matches = _directional_matches(
            baseline_items, candidate_items
        )
        candidate_matches = _directional_matches(
            candidate_items, baseline_items
        )
        all_baseline_matches.extend(baseline_matches)
        all_candidate_matches.extend(candidate_matches)
        sources[source_key] = {
            "baseline_count": len(baseline_items),
            "candidate_count": len(candidate_items),
            "baseline_below_threshold": [
                item
                for item in baseline_matches
                if float(item["similarity"]) < args.threshold
            ],
            "candidate_below_threshold": [
                item
                for item in candidate_matches
                if float(item["similarity"]) < args.threshold
            ],
        }

    baseline_covered = sum(
        float(item["similarity"]) >= args.threshold
        for item in all_baseline_matches
    )
    candidate_covered = sum(
        float(item["similarity"]) >= args.threshold
        for item in all_candidate_matches
    )
    report = {
        "baseline": str(Path(args.baseline)),
        "candidate": str(Path(args.candidate)),
        "threshold": args.threshold,
        "summary": {
            "baseline_episodes": len(all_baseline_matches),
            "candidate_episodes": len(all_candidate_matches),
            "baseline_covered_by_candidate": baseline_covered,
            "baseline_coverage": round(
                baseline_covered / max(1, len(all_baseline_matches)), 6
            ),
            "candidate_covered_by_baseline": candidate_covered,
            "candidate_coverage": round(
                candidate_covered / max(1, len(all_candidate_matches)), 6
            ),
        },
        "sources": sources,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
