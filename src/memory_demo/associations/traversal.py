from __future__ import annotations

from dataclasses import dataclass, field

from memory_demo.repositories import AssociationRepository
from memory_demo.types import NodeType, SearchHit


@dataclass(slots=True)
class TraversedNode:
    node_type: NodeType
    node_id: int
    score: float
    path: list[int] = field(default_factory=list)


class GraphTraverser:
    def __init__(self, repository: AssociationRepository):
        self.repository = repository

    @staticmethod
    def _other_endpoint(row, node_type: NodeType, node_id: int) -> tuple[NodeType, int]:
        if row["from_type"] == node_type and int(row["from_id"]) == node_id:
            return row["to_type"], int(row["to_id"])
        return row["from_type"], int(row["from_id"])

    def expand(
        self,
        seeds: list[SearchHit],
        beam_width: int,
        max_hops: int,
    ) -> tuple[list[TraversedNode], list[dict]]:
        best: dict[tuple[NodeType, int], TraversedNode] = {
            (seed.node_type, seed.node_id): TraversedNode(
                seed.node_type, seed.node_id, max(0.0, seed.score), []
            )
            for seed in seeds
        }
        frontier = sorted(best.values(), key=lambda item: item.score, reverse=True)[
            :beam_width
        ]
        path_records: list[dict] = []
        expanded_states: set[tuple[NodeType, int, int]] = set()
        for hop in range(max_hops):
            next_frontier: list[TraversedNode] = []
            active_frontier: list[TraversedNode] = []
            for current in frontier:
                state = (current.node_type, current.node_id, hop)
                if state in expanded_states:
                    continue
                expanded_states.add(state)
                active_frontier.append(current)

            # Fetch every frontier node of one type with a single SQLite
            # connection.  The previous implementation opened a connection
            # per node and per hop; that is disproportionately expensive when
            # a Linux process reads the database from a mounted Windows disk.
            # ``neighbors_many`` preserves the same per-node ordering and
            # limit, so this changes I/O shape without changing graph scores.
            neighbors_by_node: dict[tuple[NodeType, int], list] = {}
            for node_type in ("episode", "concept"):
                node_ids = [
                    current.node_id
                    for current in active_frontier
                    if current.node_type == node_type
                ]
                if not node_ids:
                    continue
                grouped = self.repository.neighbors_many(
                    node_type,
                    node_ids,
                    limit=beam_width * 2,
                )
                neighbors_by_node.update(
                    {
                        (node_type, node_id): rows
                        for node_id, rows in grouped.items()
                    }
                )

            for current in active_frontier:
                for edge in neighbors_by_node.get(
                    (current.node_type, current.node_id), []
                ):
                    other_type, other_id = self._other_endpoint(
                        edge, current.node_type, current.node_id
                    )
                    polarity_factor = 1.0 if int(edge["polarity"]) >= 0 else 0.5
                    edge_score = (
                        float(edge["weight"])
                        * float(edge["confidence"])
                        * polarity_factor
                    )
                    score = current.score * edge_score * (0.85 ** (hop + 1))
                    key = (other_type, other_id)
                    path = [*current.path, int(edge["id"])]
                    previous = best.get(key)
                    if previous is None or score > previous.score:
                        node = TraversedNode(other_type, other_id, score, path)
                        best[key] = node
                        next_frontier.append(node)
                    path_records.append(
                        {
                            "association_id": int(edge["id"]),
                            "from": [current.node_type, current.node_id],
                            "to": [other_type, other_id],
                            "relation_type": edge["relation_type"],
                            "relation_key": edge["relation_key"],
                            "relation_text": edge["relation_text"],
                            "association_mode": (
                                str(edge["association_mode"])
                                if "association_mode" in edge.keys()
                                else "semantic"
                            ),
                            "polarity": int(edge["polarity"]),
                            "weight": float(edge["weight"]),
                            "confidence": float(edge["confidence"]),
                            "generation": (
                                int(edge["generation"])
                                if "generation" in edge.keys()
                                else 0
                            ),
                            "claim_level": (
                                str(edge["claim_level"])
                                if "claim_level" in edge.keys()
                                else "direct_fact"
                            ),
                            "audit_status": (
                                str(edge["audit_status"])
                                if "audit_status" in edge.keys()
                                else "not_required"
                            ),
                            "created_reason": (
                                str(edge["created_reason"])
                                if "created_reason" in edge.keys()
                                else ""
                            ),
                            "path_score": score,
                        }
                    )
            frontier = sorted(
                next_frontier, key=lambda item: item.score, reverse=True
            )[:beam_width]
            if not frontier:
                break
        return (
            sorted(best.values(), key=lambda item: item.score, reverse=True),
            path_records,
        )
