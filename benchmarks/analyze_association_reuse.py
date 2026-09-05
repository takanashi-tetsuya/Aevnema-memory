from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def load_mode(report_path: Path, mode: str) -> list[dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    modes = report.get("modes", {})
    if mode not in modes:
        raise ValueError(f"mode {mode!r} not found in {report_path}")
    return list(modes[mode])


def summarize_result(item: dict, learned_ids: set[int]) -> dict:
    result = item["result"]
    evidence_ids = [int(row["id"]) for row in result["evidence_episodes"]]
    path_ids = [int(row["association_id"]) for row in result["association_paths"]]
    return {
        "question_id": item["id"],
        "answer_characters": len(str(result.get("answer", ""))),
        "episode_ids": [int(value) for value in result.get("episode_ids", [])],
        "evidence_episode_ids": evidence_ids,
        "evidence_source_keys": sorted(
            {str(row["source_key"]) for row in result["evidence_episodes"]}
        ),
        "association_path_ids": path_ids,
        "learned_association_path_ids": [
            edge_id for edge_id in path_ids if edge_id in learned_ids
        ],
        "new_association_ids": [
            int(value) for value in result.get("new_association_ids", [])
        ],
    }


def edge_rows(database: Path, learned_ids: set[int]) -> list[dict]:
    placeholders = ",".join("?" for _ in learned_ids)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT * FROM association WHERE id IN ({placeholders}) ORDER BY id",
            sorted(learned_ids),
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            edge = dict(row)
            endpoint_ranks: list[dict] = []
            for side in ("from", "to"):
                node_type = str(row[f"{side}_type"])
                node_id = int(row[f"{side}_id"])
                neighbors = connection.execute(
                    """
                    SELECT id, weight, confidence
                    FROM association
                    WHERE (from_type = ? AND from_id = ?)
                       OR (to_type = ? AND to_id = ?)
                    ORDER BY weight DESC, confidence DESC
                    """,
                    (node_type, node_id, node_type, node_id),
                ).fetchall()
                rank = next(
                    (
                        index
                        for index, neighbor in enumerate(neighbors, start=1)
                        if int(neighbor["id"]) == int(row["id"])
                    ),
                    None,
                )
                endpoint_ranks.append(
                    {
                        "node": [node_type, node_id],
                        "degree": len(neighbors),
                        "neighbor_rank_by_weight": rank,
                        "inside_default_neighbor_limit_40": (
                            rank is not None and rank <= 40
                        ),
                    }
                )
            result.append(
                {
                    "id": int(row["id"]),
                    "from": [str(row["from_type"]), int(row["from_id"])],
                    "to": [str(row["to_type"]), int(row["to_id"])],
                    "relation_type": str(row["relation_type"]),
                    "relation_key": str(row["relation_key"]),
                    "relation_text": str(row["relation_text"]),
                    "weight": float(row["weight"]),
                    "confidence": float(row["confidence"]),
                    "use_count": int(row["use_count"]),
                    "endpoint_neighbor_ranks": endpoint_ranks,
                }
            )
    return result


def edge_use_counts(database: Path, learned_ids: set[int]) -> dict[int, int]:
    placeholders = ",".join("?" for _ in learned_ids)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            f"SELECT id, use_count FROM association WHERE id IN ({placeholders})",
            sorted(learned_ids),
        ).fetchall()
    return {int(edge_id): int(use_count) for edge_id, use_count in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and learned Association reuse evaluations"
    )
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("learned_report", type=Path)
    parser.add_argument("learned_source_database", type=Path)
    parser.add_argument("learned_result_database", type=Path)
    parser.add_argument("--learned-id-min", type=int, required=True)
    parser.add_argument("--learned-id-max", type=int, required=True)
    parser.add_argument("--mode", default="graph_static")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    learned_ids = set(range(args.learned_id_min, args.learned_id_max + 1))
    baseline = {
        item["id"]: summarize_result(item, learned_ids)
        for item in load_mode(args.baseline_report, args.mode)
    }
    learned = {
        item["id"]: summarize_result(item, learned_ids)
        for item in load_mode(args.learned_report, args.mode)
    }
    comparisons: list[dict] = []
    for question_id in baseline.keys() & learned.keys():
        before = baseline[question_id]
        after = learned[question_id]
        before_evidence = set(before["evidence_episode_ids"])
        after_evidence = set(after["evidence_episode_ids"])
        comparisons.append(
            {
                "question_id": question_id,
                "identical_episode_ids": (
                    before["episode_ids"] == after["episode_ids"]
                ),
                "identical_evidence_episode_ids": (
                    before["evidence_episode_ids"]
                    == after["evidence_episode_ids"]
                ),
                "evidence_added_by_learned_database": sorted(
                    after_evidence - before_evidence
                ),
                "evidence_removed_by_learned_database": sorted(
                    before_evidence - after_evidence
                ),
                "learned_association_path_ids": after[
                    "learned_association_path_ids"
                ],
                "baseline": before,
                "learned": after,
            }
        )

    before_use = edge_use_counts(args.learned_source_database, learned_ids)
    after_use = edge_use_counts(args.learned_result_database, learned_ids)
    use_deltas = {
        str(edge_id): after_use.get(edge_id, 0) - before_use.get(edge_id, 0)
        for edge_id in sorted(learned_ids)
    }
    report = {
        "configuration": {
            "mode": args.mode,
            "learned_association_ids": sorted(learned_ids),
            "baseline_report": str(args.baseline_report),
            "learned_report": str(args.learned_report),
            "learned_source_database": str(args.learned_source_database),
            "learned_result_database": str(args.learned_result_database),
        },
        "summary": {
            "question_count": len(comparisons),
            "questions_using_learned_paths": sum(
                bool(item["learned_association_path_ids"])
                for item in comparisons
            ),
            "questions_with_evidence_change": sum(
                not item["identical_evidence_episode_ids"]
                for item in comparisons
            ),
            "total_learned_edge_use_count_delta": sum(use_deltas.values()),
        },
        "learned_edge_use_count_delta": use_deltas,
        "learned_edges": edge_rows(args.learned_source_database, learned_ids),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
