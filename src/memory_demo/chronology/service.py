from __future__ import annotations

from dataclasses import dataclass
import heapq

from memory_demo.event_log import JsonlEventLogger
from memory_demo.repositories import AssociationRepository, EpisodeRepository
from memory_demo.types import AssociationDraft


@dataclass(slots=True)
class ChronologyResult:
    ordered_ids: list[int]
    notes: list[str]
    has_cycle: bool = False


class ChronologyService:
    def __init__(
        self,
        episodes: EpisodeRepository,
        associations: AssociationRepository,
        logger: JsonlEventLogger | None = None,
    ):
        self.episodes = episodes
        self.associations = associations
        self.logger = logger

    def order(self, episode_ids: list[int]) -> ChronologyResult:
        input_rank = {
            int(node_id): rank
            for rank, node_id in enumerate(dict.fromkeys(episode_ids))
        }
        rows = {int(row["id"]): row for row in self.episodes.get_many(episode_ids)}
        nodes = set(rows)
        adjacency: dict[int, set[int]] = {node: set() for node in nodes}
        indegree: dict[int, int] = {node: 0 for node in nodes}
        notes: list[str] = []
        seen_edges: set[int] = set()
        neighbors_many = getattr(self.associations, "neighbors_many", None)
        if callable(neighbors_many):
            neighbor_map = neighbors_many("episode", nodes, limit=200)
        else:
            neighbor_map = {
                node: self.associations.neighbors("episode", node, limit=200)
                for node in nodes
            }
        for node in nodes:
            for edge in neighbor_map.get(node, []):
                edge_id = int(edge["id"])
                if (
                    edge_id in seen_edges
                    or edge["relation_type"] != "temporal"
                    or int(edge["polarity"]) <= 0
                ):
                    continue
                if edge["from_type"] != "episode" or edge["to_type"] != "episode":
                    continue
                seen_edges.add(edge_id)
                left, right = int(edge["from_id"]), int(edge["to_id"])
                if left not in nodes or right not in nodes:
                    continue
                key = str(edge["relation_key"]).casefold()
                if key == "after" or key.endswith("_after"):
                    left, right = right, left
                elif key not in {"before", "precedes"} and not key.endswith("_before"):
                    notes.append(f"未用于排序的时间关系 #{edge_id}: {edge['relation_text']}")
                    continue
                if right not in adjacency[left]:
                    adjacency[left].add(right)
                    indegree[right] += 1

        def priority(node_id: int):
            row = rows[node_id]
            story_order = row["story_order"]
            return (
                story_order is None,
                float(story_order) if story_order is not None else float("inf"),
                input_rank.get(node_id, len(input_rank)),
                int(row["segment_index"]),
                node_id,
            )

        heap = [(priority(node), node) for node in nodes if indegree[node] == 0]
        heapq.heapify(heap)
        ordered: list[int] = []
        while heap:
            _, node = heapq.heappop(heap)
            ordered.append(node)
            for neighbor in adjacency[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    heapq.heappush(heap, (priority(neighbor), neighbor))
        has_cycle = len(ordered) != len(nodes)
        if has_cycle:
            remaining = sorted(nodes.difference(ordered), key=priority)
            ordered.extend(remaining)
            notes.append(f"时间关系存在环，涉及 Episode：{remaining}")
            if self.logger:
                self.logger.emit("timeline_conflict", episode_ids=remaining)
        unknown = [node for node in ordered if rows[node]["story_order"] is None]
        if unknown:
            notes.append(f"以下 Episode 没有已确认 story_order：{unknown}")
        return ChronologyResult(ordered, notes, has_cycle)

    def set_order(self, episode_id: int, story_order: float, timeline_scope: str) -> None:
        self.episodes.set_story_order(episode_id, story_order, timeline_scope)
        if self.logger:
            self.logger.emit(
                "human_revision",
                action="set_story_order",
                episode_id=episode_id,
                story_order=story_order,
                timeline_scope=timeline_scope,
            )

    def add_temporal_relation(
        self,
        earlier_episode_id: int,
        later_episode_id: int,
        relation_text: str,
        confidence: float = 1.0,
    ) -> int:
        association_id = self.associations.upsert(
            AssociationDraft(
                from_type="episode",
                from_id=earlier_episode_id,
                to_type="episode",
                to_id=later_episode_id,
                relation_type="temporal",
                relation_key="before",
                relation_text=relation_text,
                weight=confidence,
                confidence=confidence,
                created_reason="人工时间线校正",
            )
        )
        if self.logger:
            self.logger.emit(
                "human_revision",
                action="add_temporal_relation",
                association_id=association_id,
            )
        return association_id
