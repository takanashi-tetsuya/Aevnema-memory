from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def edge_signature(row: dict, unordered: bool = False) -> tuple:
    left = (str(row["from_type"]), int(row["from_id"]))
    right = (str(row["to_type"]), int(row["to_id"]))
    if unordered and right < left:
        left, right = right, left
    return left, right, str(row["relation_type"])


def database_edges(path: Path) -> list[dict]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute("SELECT * FROM association ORDER BY id")
        ]


def report_mode(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["modes"]["graph_growing"])


def analyze_run(
    name: str,
    source_database: Path,
    result_database: Path,
    report_path: Path,
    log_path: Path,
) -> dict:
    source_edges = database_edges(source_database)
    result_edges = database_edges(result_database)
    source_by_id = {int(row["id"]): row for row in source_edges}
    result_by_id = {int(row["id"]): row for row in result_edges}
    source_max_id = max(source_by_id, default=0)
    new_edges = [row for row in result_edges if int(row["id"]) > source_max_id]
    source_directed = defaultdict(list)
    source_unordered = defaultdict(list)
    for row in source_edges:
        source_directed[edge_signature(row)].append(int(row["id"]))
        source_unordered[edge_signature(row, unordered=True)].append(int(row["id"]))

    events = read_jsonl(log_path)
    growth_results = [
        event for event in events if event.get("event") == "association_growth_result"
    ]
    growth_actions = [
        event for event in events if event.get("event") == "association_growth"
    ]
    actions_by_question: dict[str, list[dict]] = defaultdict(list)
    for event in growth_actions:
        actions_by_question[str(event.get("question", ""))].append(event)
    results_by_question: dict[str, list[dict]] = defaultdict(list)
    for event in growth_results:
        results_by_question[str(event.get("question", ""))].append(event)

    question_reports: list[dict] = []
    for item in report_mode(report_path):
        question = str(item["question"])
        result = item["result"]
        actions = actions_by_question.get(question, [])
        round_results = results_by_question.get(question, [])
        created_ids = [
            int(value)
            for event in round_results
            for value in event.get("created_association_ids", [])
        ]
        reinforced_ids = [
            int(value)
            for event in round_results
            for value in event.get("reinforced_association_ids", [])
        ]
        answer_path_ids = [
            int(path["association_id"])
            for path in result.get("association_paths", [])
        ]
        question_reports.append(
            {
                "question_id": item["id"],
                "question": question,
                "growth_round_count": len(round_results),
                "created_association_ids_from_log": created_ids,
                "reinforced_association_ids_from_log": reinforced_ids,
                "reinforced_preexisting_ids": sorted(
                    {edge_id for edge_id in reinforced_ids if edge_id <= source_max_id}
                ),
                "result_new_association_ids": [
                    int(value) for value in result.get("new_association_ids", [])
                ],
                "answer_path_ids": answer_path_ids,
                "created_ids_used_in_answer": sorted(
                    set(created_ids).intersection(answer_path_ids)
                ),
                "reinforced_ids_used_in_answer": sorted(
                    set(reinforced_ids).intersection(answer_path_ids)
                ),
                "evidence_episode_ids": [
                    int(row["id"]) for row in result.get("evidence_episodes", [])
                ],
                "growth_actions": [
                    {
                        "association_id": int(event["association_id"]),
                        "action": str(event["action"]),
                        "draft": event.get("draft", {}),
                    }
                    for event in actions
                ],
            }
        )

    new_edge_analysis: list[dict] = []
    for row in new_edges:
        directed_matches = source_directed.get(edge_signature(row), [])
        unordered_matches = source_unordered.get(
            edge_signature(row, unordered=True), []
        )
        new_edge_analysis.append(
            {
                "id": int(row["id"]),
                "from": [str(row["from_type"]), int(row["from_id"])],
                "to": [str(row["to_type"]), int(row["to_id"])],
                "relation_type": str(row["relation_type"]),
                "relation_key": str(row["relation_key"]),
                "relation_text": str(row["relation_text"]),
                "weight": float(row["weight"]),
                "confidence": float(row["confidence"]),
                "same_directed_pair_and_type_in_source": directed_matches,
                "same_unordered_pair_and_type_in_source": unordered_matches,
                "possible_semantic_duplicate_of_source": bool(
                    row["relation_type"] == "semantic" and unordered_matches
                ),
            }
        )

    original_use_deltas = []
    for edge_id, before in source_by_id.items():
        after = result_by_id.get(edge_id)
        if after is None:
            continue
        use_delta = int(after["use_count"]) - int(before["use_count"])
        evidence_delta = int(after["evidence_count"]) - int(before["evidence_count"])
        weight_delta = float(after["weight"]) - float(before["weight"])
        if use_delta or evidence_delta or abs(weight_delta) > 1e-12:
            original_use_deltas.append(
                {
                    "id": edge_id,
                    "use_count_delta": use_delta,
                    "evidence_count_delta": evidence_delta,
                    "weight_delta": weight_delta,
                    "relation_type": str(before["relation_type"]),
                    "relation_key": str(before["relation_key"]),
                    "relation_text_before": str(before["relation_text"]),
                    "relation_text_after": str(after["relation_text"]),
                }
            )

    return {
        "name": name,
        "source_database": str(source_database),
        "result_database": str(result_database),
        "source_association_count": len(source_edges),
        "result_association_count": len(result_edges),
        "new_association_count": len(new_edges),
        "growth_event_created_count": sum(
            len(item["created_association_ids_from_log"]) for item in question_reports
        ),
        "growth_event_reinforced_count": sum(
            len(item["reinforced_association_ids_from_log"])
            for item in question_reports
        ),
        "preexisting_edges_reinforced": sorted(
            {
                edge_id
                for item in question_reports
                for edge_id in item["reinforced_preexisting_ids"]
            }
        ),
        "possible_semantic_duplicates_of_source": sum(
            item["possible_semantic_duplicate_of_source"]
            for item in new_edge_analysis
        ),
        "original_edge_deltas": original_use_deltas,
        "new_edges": new_edge_analysis,
        "questions": question_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit created versus reinforced edges in graph-growing reuse runs"
    )
    parser.add_argument("--baseline-source-db", type=Path, required=True)
    parser.add_argument("--baseline-result-db", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--learned-source-db", type=Path, required=True)
    parser.add_argument("--learned-result-db", type=Path, required=True)
    parser.add_argument("--learned-report", type=Path, required=True)
    parser.add_argument("--learned-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "baseline": analyze_run(
            "baseline",
            args.baseline_source_db,
            args.baseline_result_db,
            args.baseline_report,
            args.baseline_log,
        ),
        "learned": analyze_run(
            "learned",
            args.learned_source_db,
            args.learned_result_db,
            args.learned_report,
            args.learned_log,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                name: {
                    key: value[key]
                    for key in (
                        "new_association_count",
                        "growth_event_created_count",
                        "growth_event_reinforced_count",
                        "preexisting_edges_reinforced",
                        "possible_semantic_duplicates_of_source",
                    )
                }
                for name, value in report.items()
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
