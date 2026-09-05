from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

import numpy as np


def extract_json_payload(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        positions = [
            position
            for position in (cleaned.find("{"), cleaned.find("["))
            if position >= 0
        ]
        for position in sorted(positions):
            try:
                value, _ = decoder.raw_decode(cleaned[position:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("model response does not contain valid JSON")


def normalize_alias(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_query_artifacts(log_path: Path) -> list[dict]:
    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries: list[dict] = []
    current: dict | None = None
    request_context: dict[str, tuple[dict, str]] = {}
    for event in events:
        event_type = event.get("event")
        if event_type == "query_started":
            current = {"question": str(event["question"])}
            queries.append(current)
            continue
        if current is None or event_type != "llm_request":
            if event_type == "llm_response" and event.get("request_id") in request_context:
                target, kind = request_context[str(event["request_id"])]
                payload = event.get("payload", {})
                if kind == "intent":
                    content = payload["choices"][0]["message"]["content"]
                    target["intent"] = extract_json_payload(str(content))
                elif kind == "embedding":
                    rows = sorted(payload["data"], key=lambda row: int(row.get("index", 0)))
                    target["embedding"] = rows[0]["embedding"]
            continue
        request_id = str(event.get("request_id", ""))
        endpoint = str(event.get("endpoint", ""))
        payload = event.get("payload", {})
        if endpoint == "embeddings":
            request_context[request_id] = (current, "embedding")
        elif endpoint == "chat/completions":
            messages = payload.get("messages", [])
            system = str(messages[0].get("content", "")) if messages else ""
            if "查询解析器" in system:
                request_context[request_id] = (current, "intent")
    missing = [
        item["question"]
        for item in queries
        if "embedding" not in item or "intent" not in item
    ]
    if missing:
        raise ValueError(f"missing logged query artifacts for: {missing}")
    return queries


def load_embeddings(connection: sqlite3.Connection, table: str) -> tuple[np.ndarray, np.ndarray]:
    where = " WHERE status = 'active'" if table == "concept" else ""
    rows = connection.execute(
        f"SELECT id, embedding FROM {table}{where} ORDER BY id"
    ).fetchall()
    ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    matrix = np.stack(
        [np.frombuffer(row[1], dtype="<f4") for row in rows]
    ).astype(np.float32, copy=False)
    return ids, matrix


def top_hits(ids: np.ndarray, matrix: np.ndarray, query: np.ndarray, limit: int):
    scores = matrix @ query
    k = min(limit, len(scores))
    if k == 0:
        return []
    selected = np.argpartition(scores, -k)[-k:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    return [(int(ids[index]), float(scores[index])) for index in selected]


def neighbors(connection: sqlite3.Connection, node: tuple[str, int], limit: int):
    return connection.execute(
        """
        SELECT * FROM association
        WHERE (from_type = ? AND from_id = ?)
           OR (to_type = ? AND to_id = ?)
        ORDER BY weight DESC, confidence DESC
        LIMIT ?
        """,
        (node[0], node[1], node[0], node[1], limit),
    ).fetchall()


def other_endpoint(row: sqlite3.Row, node: tuple[str, int]) -> tuple[str, int]:
    if str(row["from_type"]) == node[0] and int(row["from_id"]) == node[1]:
        return str(row["to_type"]), int(row["to_id"])
    return str(row["from_type"]), int(row["from_id"])


def expand(
    connection: sqlite3.Connection,
    seeds: list[tuple[str, int, float]],
    beam_width: int = 20,
    max_hops: int = 3,
) -> tuple[dict[tuple[str, int], dict], list[dict]]:
    best = {
        (node_type, node_id): {
            "node": (node_type, node_id),
            "score": max(0.0, score),
            "path": [],
        }
        for node_type, node_id, score in seeds
    }
    frontier = sorted(best.values(), key=lambda item: item["score"], reverse=True)[
        :beam_width
    ]
    path_records: list[dict] = []
    expanded_states: set[tuple[str, int, int]] = set()
    for hop in range(max_hops):
        next_frontier: list[dict] = []
        for current in frontier:
            current_node = current["node"]
            state = (current_node[0], current_node[1], hop)
            if state in expanded_states:
                continue
            expanded_states.add(state)
            for edge in neighbors(connection, current_node, beam_width * 2):
                other = other_endpoint(edge, current_node)
                polarity_factor = 1.0 if int(edge["polarity"]) >= 0 else 0.5
                edge_score = (
                    float(edge["weight"])
                    * float(edge["confidence"])
                    * polarity_factor
                )
                score = float(current["score"]) * edge_score * (0.85 ** (hop + 1))
                path = [*current["path"], int(edge["id"])]
                previous = best.get(other)
                if previous is None or score > float(previous["score"]):
                    item = {"node": other, "score": score, "path": path}
                    best[other] = item
                    next_frontier.append(item)
                path_records.append(
                    {
                        "association_id": int(edge["id"]),
                        "hop": hop + 1,
                        "from": list(current_node),
                        "to": list(other),
                        "relation_type": str(edge["relation_type"]),
                        "relation_key": str(edge["relation_key"]),
                        "weight": float(edge["weight"]),
                        "confidence": float(edge["confidence"]),
                        "path_score": score,
                    }
                )
        frontier = sorted(
            next_frontier, key=lambda item: item["score"], reverse=True
        )[:beam_width]
        if not frontier:
            break
    return best, path_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay logged query embeddings to trace learned edges in traversal"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--learned-id-min", type=int, required=True)
    parser.add_argument("--learned-id-max", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    learned_ids = set(range(args.learned_id_min, args.learned_id_max + 1))

    with sqlite3.connect(args.database) as connection:
        connection.row_factory = sqlite3.Row
        episode_ids, episode_matrix = load_embeddings(connection, "episode")
        concept_ids, concept_matrix = load_embeddings(connection, "concept")
        query_reports: list[dict] = []
        for query_data in load_query_artifacts(args.log):
            vector = np.asarray(query_data["embedding"], dtype=np.float32)
            vector /= np.linalg.norm(vector)
            seed_map: dict[tuple[str, int], float] = {}
            for node_id, score in top_hits(episode_ids, episode_matrix, vector, 40):
                seed_map[("episode", node_id)] = score
            for node_id, score in top_hits(concept_ids, concept_matrix, vector, 20):
                seed_map[("concept", node_id)] = score
            for entity in query_data["intent"].get("target_entities", []):
                normalized = normalize_alias(str(entity))
                rows = connection.execute(
                    """
                    SELECT c.id, c.canonical_concept_id
                    FROM concept_alias a
                    JOIN concept c ON c.id = a.concept_id
                    WHERE a.normalized_alias = ?
                    ORDER BY c.status = 'active' DESC, c.confidence DESC
                    """,
                    (normalized,),
                ).fetchall()
                for row in rows:
                    node_id = int(row["canonical_concept_id"] or row["id"])
                    seed_map[("concept", node_id)] = max(
                        1.0, seed_map.get(("concept", node_id), 0.0)
                    )
            seeds = sorted(
                [(*node, score) for node, score in seed_map.items()],
                key=lambda item: item[2],
                reverse=True,
            )
            seed_ranks = {
                (node_type, node_id): rank
                for rank, (node_type, node_id, _) in enumerate(seeds, start=1)
            }
            best, paths = expand(connection, seeds)
            ranked_paths = sorted(
                paths, key=lambda item: item["path_score"], reverse=True
            )
            path_rank_by_identity = {id(item): rank for rank, item in enumerate(ranked_paths, 1)}
            ranked_non_involves = [
                item for item in ranked_paths if item["relation_key"] != "involves"
            ]
            non_involves_rank = {
                id(item): rank for rank, item in enumerate(ranked_non_involves, 1)
            }
            ranked_episode_episode = [
                item
                for item in ranked_paths
                if item["from"][0] == "episode" and item["to"][0] == "episode"
            ]
            episode_episode_rank = {
                id(item): rank for rank, item in enumerate(ranked_episode_episode, 1)
            }
            learned_occurrences = []
            for item in paths:
                if item["association_id"] in learned_ids:
                    learned_occurrences.append(
                        {
                            **item,
                            "global_path_score_rank": path_rank_by_identity[id(item)],
                            "non_involves_path_score_rank": non_involves_rank.get(id(item)),
                            "episode_episode_path_score_rank": episode_episode_rank.get(id(item)),
                            "inside_answer_path_limit_24": (
                                path_rank_by_identity[id(item)] <= 24
                            ),
                        }
                    )
            ranked_nodes = sorted(
                best.values(), key=lambda item: item["score"], reverse=True
            )
            node_rank = {item["node"]: rank for rank, item in enumerate(ranked_nodes, 1)}
            learned_endpoints = {
                endpoint
                for item in learned_occurrences
                for endpoint in (tuple(item["from"]), tuple(item["to"]))
            }
            query_reports.append(
                {
                    "question": query_data["question"],
                    "intent": query_data["intent"],
                    "seed_count": len(seeds),
                    "initial_frontier": [list(item[:2]) for item in seeds[:20]],
                    "learned_edge_occurrences": learned_occurrences,
                    "learned_endpoint_diagnostics": [
                        {
                            "node": list(node),
                            "seed_rank": seed_ranks.get(node),
                            "seed_score": seed_map.get(node),
                            "traversed_rank": node_rank.get(node),
                            "traversed_score": (
                                float(best[node]["score"]) if node in best else None
                            ),
                            "inside_candidate_limit_120": (
                                node_rank.get(node) is not None
                                and node_rank[node] <= 120
                            ),
                        }
                        for node in sorted(learned_endpoints)
                    ],
                    "raw_path_count": len(paths),
                    "traversed_node_count": len(best),
                }
            )

    report = {
        "configuration": {
            "database": str(args.database),
            "log": str(args.log),
            "learned_association_ids": sorted(learned_ids),
            "episode_top_k": 40,
            "concept_top_k": 20,
            "graph_beam_width": 20,
            "graph_max_hops": 3,
            "candidate_limit": 120,
            "answer_path_limit": 24,
        },
        "queries": query_reports,
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
                "question_count": len(query_reports),
                "learned_occurrences": [
                    len(item["learned_edge_occurrences"]) for item in query_reports
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
