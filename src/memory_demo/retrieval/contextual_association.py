from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

from memory_demo.embeddings import EmbeddingIndex, normalize_embedding
from memory_demo.types import ContextualSlotHit, QueryVectorBundle


def gate(similarity: float, threshold: float) -> float:
    """Convert cosine similarity into a fail-closed thresholded score."""
    value = float(similarity)
    threshold = float(threshold)
    if value <= threshold:
        return 0.0
    return max(0.0, min(1.0, (value - threshold) / max(1e-6, 1.0 - threshold)))


def combine_scores(context_score: float, need_score: float, mode: str) -> float:
    left = max(0.0, float(context_score))
    right = max(0.0, float(need_score))
    normalized = str(mode or "product").casefold()
    if normalized == "minimum":
        return min(left, right)
    if normalized == "geomean":
        return math.sqrt(left * right)
    if normalized != "product":
        raise ValueError("combine mode must be product, minimum, or geomean")
    return left * right


class ContextualAssociationMatcher:
    """Pure-local matcher for context + need prototype pairs.

    The object deliberately accepts a repository-like provider only for rows;
    it has no model, HTTP, prompt, or configuration-secret dependency.  This
    keeps the zero per-edge external-call invariant mechanically testable.
    """

    def __init__(
        self,
        context_index: EmbeddingIndex,
        need_index: EmbeddingIndex,
        associations,
        *,
        context_threshold: float = 0.55,
        need_threshold: float = 0.55,
        combine_mode: str = "product",
        context_top_k: int = 8,
        need_top_k: int = 8,
        edge_top_k: int = 16,
        endpoint_limit: int = 2,
    ):
        self.context_index = context_index
        self.need_index = need_index
        self.associations = associations
        self.context_threshold = float(context_threshold)
        self.need_threshold = float(need_threshold)
        self.combine_mode = str(combine_mode)
        self.context_top_k = max(1, int(context_top_k))
        self.need_top_k = max(1, int(need_top_k))
        self.edge_top_k = max(1, int(edge_top_k))
        self.endpoint_limit = max(1, int(endpoint_limit))

    def match(
        self,
        context_vector,
        need_vectors,
        *,
        domain: str | None = None,
        query_ids: Iterable[str] = (),
        slot_ids: Iterable[str] = (),
        active_anchor_ids: Iterable[int] | Mapping[int, float] | None = None,
        unresolved_slot_ids: Iterable[str] | None = None,
        target_support_scores: Mapping[tuple[str, int], float] | None = None,
        target_support_floor: float = 0.0,
    ) -> dict:
        """Match only from a base-activated anchor into an unresolved slot.

        ``active_anchor_ids`` is intentionally mandatory.  An association is
        a conditional recall from something the base lanes independently
        surfaced, not a second global semantic cache.  ``target_support`` is
        supplied by the caller's local slot-support map; this matcher never
        embeds or fetches an endpoint on its own.
        """
        if isinstance(active_anchor_ids, Mapping):
            raw_anchors = {
                int(node_id): max(0.0, float(score))
                for node_id, score in active_anchor_ids.items()
                if int(node_id) > 0 and float(score) > 0
            }
        else:
            raw_anchors = {
                int(node_id): 1.0
                for node_id in (active_anchor_ids or ())
                if int(node_id) > 0
            }
        if not raw_anchors:
            return {
                "hits": [],
                "context_hits": [],
                "need_hits": [],
                "external_calls": 0,
                "attached_episode_ids": [],
                "attached_edges": [],
                "reason": "no_active_anchor",
            }
        context = normalize_embedding(context_vector, self.context_index.dimension)
        needs = np.asarray(need_vectors, dtype=np.float32)
        if needs.ndim == 1:
            needs = needs.reshape(1, -1)
        if needs.ndim != 2 or needs.shape[1] != self.need_index.dimension:
            raise ValueError("need vector matrix does not match index dimension")
        context_hits = self.context_index.search(context, self.context_top_k)
        need_rankings = self.need_index.search_many(needs, self.need_top_k)
        context_scores = {
            int(node_id): float(score)
            for node_id, score in context_hits
            if gate(score, self.context_threshold) > 0
        }
        need_scores: dict[int, tuple[float, float, str, str]] = {}
        query_id_list = list(query_ids)
        slot_id_list = list(slot_ids)
        unresolved = {
            str(value).strip()
            for value in (unresolved_slot_ids or ())
            if str(value).strip()
        }
        for query_index, ranking in enumerate(need_rankings):
            query_id = query_id_list[query_index] if query_index < len(query_id_list) else ""
            slot_id = (
                slot_id_list[query_index]
                if query_index < len(slot_id_list)
                else query_id
            )
            if unresolved and slot_id not in unresolved:
                continue
            for node_id, score in ranking:
                value = gate(score, self.need_threshold)
                if value <= 0:
                    continue
                previous = need_scores.get(int(node_id))
                if previous is None or value > previous[1]:
                    need_scores[int(node_id)] = (
                        float(score),
                        value,
                        query_id,
                        slot_id,
                    )
        if not context_scores or not need_scores:
            return {
                "hits": [],
                "context_hits": context_hits,
                "need_hits": need_rankings,
                "external_calls": 0,
                "attached_episode_ids": [],
                "attached_edges": [],
            }
        rows = self.associations.get_contextual_for_prototypes(
            context_scores.keys(), need_scores.keys(), domain=domain,
            limit=max(self.edge_top_k * 4, self.edge_top_k),
        )
        ranked: list[ContextualSlotHit] = []
        for row in rows:
            if str(row["from_type"]) != "episode":
                continue
            anchor_id = int(row["from_id"])
            anchor_raw = raw_anchors.get(anchor_id, 0.0)
            if anchor_raw <= 0:
                continue
            context_id = int(row["context_cue_id"])
            need_id = int(row["need_cue_id"])
            context_gate = gate(context_scores.get(context_id, 0.0), self.context_threshold)
            need_raw, need_gate, query_id, slot_id = need_scores.get(
                need_id, (0.0, 0.0, "", "")
            )
            gate_score = combine_scores(context_gate, need_gate, self.combine_mode)
            if gate_score <= 0 or not slot_id:
                continue
            target_id = int(row["to_id"])
            support_key = (slot_id, target_id)
            # When the caller supplies a local target/slot score, an absent
            # pair is *not* evidence of support.  In particular, do not turn
            # a missing score into the floor: that would let the association
            # edge itself manufacture its endpoint's relevance.
            target_support = (
                float(target_support_scores.get(support_key, 0.0))
                if target_support_scores is not None
                else 1.0
            )
            if target_support < float(target_support_floor):
                continue
            lifecycle = str(row["lifecycle_state"])
            lifecycle_factor = {
                "probation": 0.5,
                "active": 1.0,
            }.get(lifecycle, 0.0)
            utility = max(0.0, min(1.0, float(row["utility_weight"])))
            if lifecycle_factor <= 0 or utility <= 0:
                continue
            normalizer = max(raw_anchors.values())
            anchor_activation = anchor_raw / normalizer if normalizer else 0.0
            total_score = (
                gate_score
                * anchor_activation
                * lifecycle_factor
                * utility
                * target_support
            )
            if total_score <= 0:
                continue
            hit = ContextualSlotHit(
                association_id=int(row["id"]),
                anchor_episode_id=anchor_id,
                target_episode_id=target_id,
                matched_slot_id=slot_id,
                matched_query_id=query_id,
                context_similarity=float(context_scores[context_id]),
                need_similarity=float(need_raw),
                anchor_activation=anchor_activation,
                utility_weight=utility,
                target_support_score=target_support,
                lifecycle_state=lifecycle,
                total_score=total_score,
            )
            # Endpoint cap keeps one noisy anchor from flooding the candidate
            # pool; ordering is deterministic and utility is only a tie-break.
            ranked.append(hit)
        ranked.sort(
            key=lambda item: (
                -item.score,
                -item.utility_weight,
                item.association_id,
            )
        )
        selected: list[ContextualSlotHit] = []
        seen_endpoint_slots: set[tuple[int, str]] = set()
        for hit in ranked:
            endpoint_slot = (hit.target_episode_id, hit.matched_slot_id)
            if endpoint_slot in seen_endpoint_slots:
                continue
            selected.append(hit)
            seen_endpoint_slots.add(endpoint_slot)
            if len(selected) >= min(self.edge_top_k, self.endpoint_limit):
                break
        return {
            "hits": selected,
            "context_hits": context_hits,
            "need_hits": need_rankings,
            "external_calls": 0,
            "attached_episode_ids": [int(item.target_episode_id) for item in selected],
            "attached_edges": [int(item.association_id) for item in selected],
        }

    def match_bundle(
        self,
        bundle: QueryVectorBundle,
        *,
        domain: str | None = None,
        endpoint_limit: int | None = None,
        active_anchor_ids: Iterable[int] | Mapping[int, float] | None = None,
        unresolved_slot_ids: Iterable[str] | None = None,
        target_support_scores: Mapping[tuple[str, int], float] | None = None,
        target_support_floor: float = 0.0,
    ) -> dict:
        needs = [item for item in bundle.queries if item.role != "whole"]
        if not needs:
            needs = [item for item in bundle.queries if item.role == "whole"]
        unresolved = {
            str(value).strip()
            for value in (unresolved_slot_ids or ())
            if str(value).strip()
        }
        if unresolved:
            needs = [
                item
                for item in needs
                if (item.slot_id or item.query_id) in unresolved
            ]
        if not needs:
            return {
                "hits": [],
                "context_hits": [],
                "need_hits": [],
                "external_calls": 0,
                "attached_episode_ids": [],
                "attached_edges": [],
                "reason": "no_unresolved_slot_vectors",
            }
        previous_limit = self.endpoint_limit
        if endpoint_limit is not None:
            self.endpoint_limit = max(1, int(endpoint_limit))
        try:
            return self.match(
                bundle.whole,
                np.stack([item.vector for item in needs], axis=0),
                domain=domain,
                query_ids=[item.query_id for item in needs],
                slot_ids=[item.slot_id or item.query_id for item in needs],
                active_anchor_ids=active_anchor_ids,
                unresolved_slot_ids=unresolved,
                target_support_scores=target_support_scores,
                target_support_floor=target_support_floor,
            )
        finally:
            self.endpoint_limit = previous_limit
