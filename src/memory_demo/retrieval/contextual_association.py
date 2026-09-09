from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any

import numpy as np

from memory_demo.embeddings import EmbeddingIndex, normalize_embedding
from memory_demo.repositories.association import normalize_evaluation_as_of
from memory_demo.types import ContextualSlotHit, QueryVectorBundle


def gate(similarity: float, threshold: float) -> float:
    """Convert cosine similarity into a fail-closed thresholded score."""

    value = float(similarity)
    threshold = float(threshold)
    if not math.isfinite(value) or not math.isfinite(threshold) or value <= threshold:
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


@dataclass(frozen=True, slots=True)
class ContextualPreTargetProposal:
    """One anchor-conditioned edge × slot × logical-binding proposal.

    This record is deliberately *before* endpoint/source checking and before
    an endpoint cap. T09 can validate targets without allowing a high-ranked
    invalid endpoint to crowd out a later valid one. It contains neither raw
    query text nor vector payloads.
    """

    association_id: int
    anchor_episode_id: int
    target_episode_id: int
    context_cue_id: int
    need_cue_id: int
    matched_slot_id: str
    matched_query_id: str
    matched_query_role: str
    matched_physical_id: str
    embedding_space_id: str
    context_similarity: float
    need_similarity: float
    context_gate_score: float
    need_gate_score: float
    anchor_activation: float
    utility_weight: float
    lifecycle_state: str
    pre_target_score: float
    proposal_key: str
    # One-based, deterministic rank over every eligible edge × slot × logical
    # binding, before target validation or any endpoint cap is applied.
    rank_before_endpoint_cap: int = 0

    @property
    def score(self) -> float:
        return self.pre_target_score


@dataclass(frozen=True, slots=True)
class _NeedBinding:
    """Local vector plus the logical identity that must not be collapsed."""

    query_id: str
    slot_id: str
    role: str
    physical_id: str
    embedding_space_id: str
    vector: Any

    @property
    def ordering_key(self) -> tuple[str, str, str, str]:
        return (self.slot_id, self.query_id, self.role, self.physical_id)


class ContextualAssociationMatcher:
    """Pure-local, anchor-first matcher for contextual associations.

    Candidate edges are loaded only after the caller supplies active Episode
    anchors. Context and need scores are calculated only for cue IDs attached
    to those edge rows, never from a global prototype top-k that unrelated
    anchors can fill. This object has no model, HTTP, or mutation path.
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
        embedding_space_id: str | None = None,
    ):
        self.context_index = context_index
        self.need_index = need_index
        self.associations = associations
        self.context_threshold = float(context_threshold)
        self.need_threshold = float(need_threshold)
        self.combine_mode = str(combine_mode)
        # Historical knobs remain immutable compatibility metadata. The v3
        # path intentionally does not apply global prototype top-k pruning.
        self.context_top_k = max(1, int(context_top_k))
        self.need_top_k = max(1, int(need_top_k))
        self.edge_top_k = max(1, int(edge_top_k))
        self.endpoint_limit = max(1, int(endpoint_limit))
        self.embedding_space_id = str(embedding_space_id or "").strip()

    @staticmethod
    def _empty_result(reason: str, **extra: Any) -> dict:
        gate_trace = {
            "reason": str(reason),
            "repository_anchor_filter_applied": False,
            "candidate_edge_count": 0,
            "pre_target_proposal_count": 0,
            "rejected": {},
        }
        gate_trace.update(extra.pop("gate_trace", {}))
        result = {
            "hits": [],
            "pre_target_proposals": [],
            "context_hits": [],
            "need_hits": [],
            "external_calls": 0,
            "attached_episode_ids": [],
            "attached_edges": [],
            "reason": str(reason),
            "gate_trace": gate_trace,
        }
        result.update(extra)
        return result

    @staticmethod
    def _normalise_unresolved_slot_ids(
        unresolved_slot_ids: Iterable[str] | None,
    ) -> tuple[str, ...] | None:
        if unresolved_slot_ids is None:
            return None
        return tuple(
            sorted(
                {
                    str(value).strip()
                    for value in unresolved_slot_ids
                    if str(value).strip()
                }
            )
        )

    @staticmethod
    def _normalise_active_anchors(
        active_anchor_ids: Iterable[int] | Mapping[int, float] | None,
    ) -> dict[int, float]:
        if isinstance(active_anchor_ids, Mapping):
            result: dict[int, float] = {}
            for raw_id, raw_score in active_anchor_ids.items():
                try:
                    node_id = int(raw_id)
                    score = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if node_id > 0 and math.isfinite(score) and score > 0:
                    result[node_id] = max(result.get(node_id, 0.0), score)
            return result
        result = {}
        for raw_id in active_anchor_ids or ():
            try:
                node_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if node_id > 0:
                result[node_id] = 1.0
        return result

    def _bundle_space_reason(self, bundle: QueryVectorBundle) -> str | None:
        """Fail closed when a declared index/model space disagrees with bundle."""

        try:
            bundle_space_id = str(bundle.embedding_space_id)
        except (AttributeError, TypeError, ValueError):
            return "invalid_query_vector_bundle"
        if not bundle_space_id:
            return "missing_embedding_space"
        if int(bundle.dimension) != int(self.context_index.dimension):
            return "context_embedding_dimension_mismatch"
        if int(bundle.dimension) != int(self.need_index.dimension):
            return "need_embedding_dimension_mismatch"
        declared_spaces = {
            value
            for value in (
                self.embedding_space_id,
                str(getattr(self.context_index, "embedding_space_id", "") or "").strip(),
                str(getattr(self.need_index, "embedding_space_id", "") or "").strip(),
            )
            if value
        }
        if len(declared_spaces) > 1 or (
            declared_spaces and bundle_space_id not in declared_spaces
        ):
            return "embedding_space_mismatch"
        return None

    @staticmethod
    def _score_candidate_prototypes(
        index: EmbeddingIndex,
        query_vector: Any,
        prototype_ids: Iterable[int],
    ) -> tuple[dict[int, float], set[int]]:
        """Score a caller-bounded set without global top-k pruning."""

        wanted = tuple(sorted({int(value) for value in prototype_ids if int(value) > 0}))
        if not wanted:
            return {}, set()
        query = normalize_embedding(query_vector, index.dimension)
        vectors = index.get_many(wanted)
        missing = set(wanted).difference(vectors)
        scores: dict[int, float] = {}
        for prototype_id in wanted:
            vector = vectors.get(prototype_id)
            if vector is None:
                continue
            try:
                normalized = normalize_embedding(vector, index.dimension)
                score = float(np.dot(query, normalized))
            except (TypeError, ValueError, FloatingPointError):
                missing.add(prototype_id)
                continue
            if math.isfinite(score):
                scores[prototype_id] = score
            else:
                missing.add(prototype_id)
        return scores, missing

    @staticmethod
    def _ranked_scores(scores: Mapping[int, float]) -> list[tuple[int, float]]:
        return sorted(
            ((int(node_id), float(score)) for node_id, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )

    @staticmethod
    def _proposal_sort_key(item: ContextualPreTargetProposal) -> tuple:
        return (
            -float(item.pre_target_score),
            -float(item.context_gate_score),
            -float(item.need_gate_score),
            -float(item.anchor_activation),
            -float(item.utility_weight),
            int(item.association_id),
            int(item.target_episode_id),
            str(item.matched_slot_id),
            str(item.matched_query_id),
            str(item.matched_query_role),
            str(item.matched_physical_id),
        )

    @staticmethod
    def _proposal_key(
        *,
        association_id: int,
        anchor_episode_id: int,
        target_episode_id: int,
        slot_id: str,
        query_id: str,
        role: str,
        physical_id: str,
        embedding_space_id: str,
    ) -> str:
        """Return a trace-safe, stable identity for one logical proposal."""

        payload = "\x1f".join(
            (
                str(association_id),
                str(anchor_episode_id),
                str(target_episode_id),
                str(slot_id),
                str(query_id),
                str(role),
                str(physical_id),
                str(embedding_space_id),
            )
        ).encode("utf-8")
        return "contextual-proposal:sha256:" + hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _hit_sort_key(item: ContextualSlotHit) -> tuple:
        return (
            -float(item.score),
            -float(item.utility_weight),
            int(item.association_id),
            int(item.target_episode_id),
            str(item.matched_slot_id),
            str(item.matched_query_id),
        )

    def _repository_rows(
        self,
        *,
        raw_anchors: Mapping[int, float],
        domain: str | None,
        evaluation_as_of: str,
    ) -> list:
        """Use the explicit v3 repository path; never fall back to utc_now."""

        return self.associations.get_contextual_for_prototypes(
            None,
            None,
            domain=domain,
            anchor_episode_ids=tuple(sorted(raw_anchors)),
            evaluation_as_of=evaluation_as_of,
            # A final endpoint cap is not valid before target checking. No SQL
            # utility limit is applied ahead of the local joint-score ranking.
            limit=None,
        )

    def _match_bindings(
        self,
        context_vector: Any,
        need_bindings: list[_NeedBinding],
        *,
        domain: str | None,
        active_anchor_ids: Iterable[int] | Mapping[int, float] | None,
        unresolved_slot_ids: Iterable[str] | None,
        target_support_scores: Mapping[tuple[str, int], float] | None,
        target_support_floor: float,
        endpoint_limit: int | None,
        evaluation_as_of: str | None,
        scoring_mode: str = "cn",
    ) -> dict:
        mode = str(scoring_mode or "cn").strip().casefold()
        if mode not in {"w", "c", "n", "cn"}:
            return self._empty_result(
                "invalid_scoring_mode",
                invalid_request_contract=True,
            )
        unresolved = self._normalise_unresolved_slot_ids(unresolved_slot_ids)
        if unresolved is None:
            return self._empty_result(
                "invalid_unresolved_slot_contract",
                invalid_request_contract=True,
            )
        if not unresolved:
            return self._empty_result("no_unresolved_slots", residual_mode=True)
        try:
            as_of = normalize_evaluation_as_of(evaluation_as_of)
        except (TypeError, ValueError):
            return self._empty_result(
                "invalid_evaluation_as_of",
                invalid_request_contract=True,
            )
        raw_anchors = self._normalise_active_anchors(active_anchor_ids)
        if not raw_anchors:
            return self._empty_result("no_active_anchor", evaluation_as_of=as_of)

        bindings_by_slot: dict[str, list[_NeedBinding]] = {
            slot_id: [] for slot_id in unresolved
        }
        for binding in need_bindings:
            if binding.slot_id in bindings_by_slot:
                bindings_by_slot[binding.slot_id].append(binding)
        missing_slots = sorted(
            slot_id for slot_id, bindings in bindings_by_slot.items() if not bindings
        )
        if missing_slots:
            return self._empty_result(
                "missing_slot_vector",
                invalid_request_contract=True,
                missing_slot_ids=missing_slots,
                evaluation_as_of=as_of,
            )
        bindings = [
            binding
            for slot_id in unresolved
            for binding in sorted(
                bindings_by_slot[slot_id], key=lambda item: item.ordering_key
            )
        ]
        try:
            # Validate all request-local vectors before consulting the database.
            normalize_embedding(context_vector, self.context_index.dimension)
            for binding in bindings:
                normalize_embedding(binding.vector, self.need_index.dimension)
        except (TypeError, ValueError, FloatingPointError):
            return self._empty_result(
                "invalid_slot_vector",
                invalid_request_contract=True,
                evaluation_as_of=as_of,
            )

        rows = list(
            self._repository_rows(
                raw_anchors=raw_anchors,
                domain=domain,
                evaluation_as_of=as_of,
            )
        )
        if not rows:
            return self._empty_result(
                "no_anchor_conditioned_edges",
                evaluation_as_of=as_of,
                gate_trace={
                    "repository_anchor_filter_applied": True,
                    "active_anchor_count": len(raw_anchors),
                },
            )

        ordered_rows = sorted(
            rows,
            key=lambda row: (int(row["id"]), int(row["to_id"])),
        )
        context_ids = {
            int(row["context_cue_id"])
            for row in ordered_rows
            if row["context_cue_id"] is not None
        }
        need_ids = {
            int(row["need_cue_id"])
            for row in ordered_rows
            if row["need_cue_id"] is not None
        }
        context_scores, missing_context_ids = self._score_candidate_prototypes(
            self.context_index, context_vector, context_ids
        )
        context_hits = self._ranked_scores(context_scores)
        need_scores_by_binding: dict[tuple[str, str, str, str], dict[int, float]] = {}
        need_hits: list[list[tuple[int, float]]] = []
        missing_need_ids: set[int] = set()
        for binding in bindings:
            scores, missing = self._score_candidate_prototypes(
                self.need_index, binding.vector, need_ids
            )
            need_scores_by_binding[binding.ordering_key] = scores
            need_hits.append(self._ranked_scores(scores))
            missing_need_ids.update(missing)

        context_gate_scores = {
            cue_id: gate(score, self.context_threshold)
            for cue_id, score in context_scores.items()
        }
        need_gate_scores = {
            binding_key: {
                cue_id: gate(score, self.need_threshold)
                for cue_id, score in scores.items()
            }
            for binding_key, scores in need_scores_by_binding.items()
        }
        rejected: dict[str, int] = {
            "missing_context_prototype_vector": 0,
            "missing_need_prototype_vector": 0,
            "context_gate": 0,
            "need_gate": 0,
            "inactive_lifecycle": 0,
            "nonpositive_utility": 0,
            "unexpected_anchor": 0,
        }
        proposals: list[ContextualPreTargetProposal] = []
        normalizer = max(raw_anchors.values())
        for row in ordered_rows:
            if str(row["from_type"]) != "episode":
                rejected["unexpected_anchor"] += 1
                continue
            anchor_id = int(row["from_id"])
            anchor_raw = raw_anchors.get(anchor_id, 0.0)
            if anchor_raw <= 0:
                rejected["unexpected_anchor"] += 1
                continue
            context_id = int(row["context_cue_id"])
            need_id = int(row["need_cue_id"])
            if mode in {"c", "cn"} and context_id in missing_context_ids:
                rejected["missing_context_prototype_vector"] += 1
                continue
            if mode in {"n", "cn"} and need_id in missing_need_ids:
                rejected["missing_need_prototype_vector"] += 1
                continue
            context_gate = context_gate_scores.get(context_id, 0.0)
            if mode in {"c", "cn"} and context_gate <= 0:
                rejected["context_gate"] += 1
                continue
            lifecycle = str(row["lifecycle_state"])
            lifecycle_factor = {"probation": 0.5, "active": 1.0}.get(lifecycle, 0.0)
            if lifecycle_factor <= 0:
                rejected["inactive_lifecycle"] += 1
                continue
            utility = max(0.0, min(1.0, float(row["utility_weight"])))
            if utility <= 0:
                rejected["nonpositive_utility"] += 1
                continue
            anchor_activation = anchor_raw / normalizer if normalizer else 0.0
            for binding in bindings:
                binding_key = binding.ordering_key
                need_gate = need_gate_scores[binding_key].get(need_id, 0.0)
                if mode in {"n", "cn"} and need_gate <= 0:
                    rejected["need_gate"] += 1
                    continue
                key_score = (
                    1.0
                    if mode == "w"
                    else context_gate
                    if mode == "c"
                    else need_gate
                    if mode == "n"
                    else combine_scores(
                        context_gate, need_gate, self.combine_mode
                    )
                )
                pre_target_score = (
                    key_score
                    * anchor_activation
                    * lifecycle_factor
                    * utility
                )
                if pre_target_score <= 0:
                    rejected["key_score"] = rejected.get("key_score", 0) + 1
                    continue
                proposals.append(
                    ContextualPreTargetProposal(
                        association_id=int(row["id"]),
                        anchor_episode_id=anchor_id,
                        target_episode_id=int(row["to_id"]),
                        context_cue_id=context_id,
                        need_cue_id=need_id,
                        matched_slot_id=binding.slot_id,
                        matched_query_id=binding.query_id,
                        matched_query_role=binding.role,
                        matched_physical_id=binding.physical_id,
                        embedding_space_id=binding.embedding_space_id,
                        context_similarity=float(context_scores[context_id]),
                        need_similarity=float(
                            need_scores_by_binding[binding_key][need_id]
                        ),
                        context_gate_score=context_gate,
                        need_gate_score=need_gate,
                        anchor_activation=anchor_activation,
                        utility_weight=utility,
                        lifecycle_state=lifecycle,
                        pre_target_score=pre_target_score,
                        proposal_key=self._proposal_key(
                            association_id=int(row["id"]),
                            anchor_episode_id=anchor_id,
                            target_episode_id=int(row["to_id"]),
                            slot_id=binding.slot_id,
                            query_id=binding.query_id,
                            role=binding.role,
                            physical_id=binding.physical_id,
                            embedding_space_id=binding.embedding_space_id,
                        ),
                    )
                )
        proposals.sort(key=self._proposal_sort_key)
        proposals = [
            replace(item, rank_before_endpoint_cap=index)
            for index, item in enumerate(proposals, start=1)
        ]

        target_rejected = 0
        pre_cap_hits: list[ContextualSlotHit] = []
        for proposal in proposals:
            support_key = (proposal.matched_slot_id, proposal.target_episode_id)
            target_support = (
                float(target_support_scores.get(support_key, 0.0))
                if target_support_scores is not None
                else 1.0
            )
            if not math.isfinite(target_support) or target_support < float(target_support_floor):
                target_rejected += 1
                continue
            total_score = proposal.pre_target_score * max(0.0, target_support)
            if total_score <= 0:
                target_rejected += 1
                continue
            pre_cap_hits.append(
                ContextualSlotHit(
                    association_id=proposal.association_id,
                    anchor_episode_id=proposal.anchor_episode_id,
                    target_episode_id=proposal.target_episode_id,
                    matched_slot_id=proposal.matched_slot_id,
                    matched_query_id=proposal.matched_query_id,
                    context_similarity=proposal.context_similarity,
                    need_similarity=proposal.need_similarity,
                    anchor_activation=proposal.anchor_activation,
                    utility_weight=proposal.utility_weight,
                    target_support_score=target_support,
                    lifecycle_state=proposal.lifecycle_state,
                    total_score=total_score,
                )
            )
        pre_cap_hits.sort(key=self._hit_sort_key)
        local_endpoint_limit = (
            self.endpoint_limit
            if endpoint_limit is None
            else max(1, int(endpoint_limit))
        )
        selected: list[ContextualSlotHit] = []
        seen_endpoint_slots: set[tuple[int, str]] = set()
        # This compatibility projection preserves historical post-target cap
        # behavior. T09 must consume pre_target_proposals, which remain full.
        for hit in pre_cap_hits:
            endpoint_slot = (hit.target_episode_id, hit.matched_slot_id)
            if endpoint_slot in seen_endpoint_slots:
                continue
            selected.append(hit)
            seen_endpoint_slots.add(endpoint_slot)
            if len(selected) >= min(self.edge_top_k, local_endpoint_limit):
                break
        gate_trace = {
            "repository_anchor_filter_applied": True,
            "evaluation_as_of": as_of,
            "active_anchor_count": len(raw_anchors),
            "candidate_edge_count": len(ordered_rows),
            "candidate_context_prototype_count": len(context_ids),
            "candidate_need_prototype_count": len(need_ids),
            "logical_binding_count": len(bindings),
            "missing_context_prototype_ids": sorted(missing_context_ids),
            "missing_need_prototype_ids": sorted(missing_need_ids),
            "pre_target_proposal_count": len(proposals),
            "post_target_pre_cap_count": len(pre_cap_hits),
            "target_rejected_count": target_rejected,
            "compatibility_endpoint_limit": local_endpoint_limit,
            "compatibility_edge_top_k": self.edge_top_k,
            "scoring_mode": mode,
            "rejected": rejected,
        }
        return {
            "hits": selected,
            "pre_target_proposals": proposals,
            "context_hits": context_hits,
            "need_hits": need_hits,
            "external_calls": 0,
            "attached_episode_ids": [int(item.target_episode_id) for item in selected],
            "attached_edges": [int(item.association_id) for item in selected],
            "evaluation_as_of": as_of,
            "gate_trace": gate_trace,
        }

    def match(
        self,
        context_vector: Any,
        need_vectors: Any,
        *,
        domain: str | None = None,
        query_ids: Iterable[str] = (),
        slot_ids: Iterable[str] = (),
        active_anchor_ids: Iterable[int] | Mapping[int, float] | None = None,
        unresolved_slot_ids: Iterable[str] | None = None,
        target_support_scores: Mapping[tuple[str, int], float] | None = None,
        target_support_floor: float = 0.0,
        endpoint_limit: int | None = None,
        evaluation_as_of: str | None = None,
    ) -> dict:
        """Anchor-first API for already-prepared local vectors.

        Bare arrays cannot prove an embedding space, so v3 callers should use
        match_bundle. This compatibility API is still strict about explicit
        residual slots and the fixed evaluation timestamp.
        """

        unresolved = self._normalise_unresolved_slot_ids(unresolved_slot_ids)
        if unresolved is None:
            return self._empty_result(
                "invalid_unresolved_slot_contract",
                invalid_request_contract=True,
            )
        if not unresolved:
            return self._empty_result("no_unresolved_slots", residual_mode=True)
        needs = np.asarray(need_vectors, dtype=np.float32)
        if needs.ndim == 1:
            needs = needs.reshape(1, -1)
        if needs.ndim != 2 or needs.shape[1] != self.need_index.dimension:
            return self._empty_result(
                "invalid_slot_vector",
                invalid_request_contract=True,
            )
        query_id_list = [str(value) for value in query_ids]
        slot_id_list = [str(value) for value in slot_ids]
        bindings: list[_NeedBinding] = []
        for index, vector in enumerate(needs):
            query_id = query_id_list[index] if index < len(query_id_list) else ""
            slot_id = slot_id_list[index] if index < len(slot_id_list) else query_id
            bindings.append(
                _NeedBinding(
                    query_id=query_id,
                    slot_id=slot_id,
                    role="legacy",
                    physical_id=f"legacy-query-vector:{index}",
                    embedding_space_id="legacy",
                    vector=vector,
                )
            )
        return self._match_bindings(
            context_vector,
            bindings,
            domain=domain,
            active_anchor_ids=active_anchor_ids,
            unresolved_slot_ids=unresolved,
            target_support_scores=target_support_scores,
            target_support_floor=target_support_floor,
            endpoint_limit=endpoint_limit,
            evaluation_as_of=evaluation_as_of,
            scoring_mode="cn",
        )

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
        evaluation_as_of: str | None = None,
        scoring_mode: str = "cn",
    ) -> dict:
        """Match every required logical slot against active-anchor edges.

        None for unresolved_slot_ids is an invalid request contract; an explicit
        empty collection is the ordinary residual no-op. Both return before an
        index or repository is touched.
        """

        unresolved = self._normalise_unresolved_slot_ids(unresolved_slot_ids)
        if unresolved is None:
            return self._empty_result(
                "invalid_unresolved_slot_contract",
                invalid_request_contract=True,
            )
        if not unresolved:
            return self._empty_result("no_unresolved_slots", residual_mode=True)
        if not isinstance(bundle, QueryVectorBundle):
            return self._empty_result(
                "invalid_query_vector_bundle",
                invalid_request_contract=True,
            )
        space_reason = self._bundle_space_reason(bundle)
        if space_reason is not None:
            return self._empty_result(space_reason, invalid_request_contract=True)
        try:
            whole_ref = bundle.whole_ref
            if whole_ref.embedding_space_id != bundle.embedding_space_id:
                raise ValueError("whole query vector space mismatch")
            context_vector = bundle.vector_for(whole_ref)
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._empty_result(
                "missing_context_vector",
                invalid_request_contract=True,
            )
        bindings: list[_NeedBinding] = []
        missing_slots: list[str] = []
        for slot_id in unresolved:
            logical_bindings = bundle.bindings_for_slot(slot_id)
            # Older callers identified a request obligation by query_id alone.
            # Preserve that narrow alias only when the binding has no explicit
            # slot_id; an explicit but different slot never silently matches.
            if not logical_bindings:
                logical_bindings = tuple(
                    item
                    for item in bundle.bindings_for_query_id(slot_id)
                    if not str(item.slot_id).strip()
                )
            if not logical_bindings:
                missing_slots.append(slot_id)
                continue
            for binding in logical_bindings:
                try:
                    if binding.embedding_space_id != bundle.embedding_space_id:
                        raise ValueError("logical binding embedding space mismatch")
                    vector = bundle.vector_for(binding)
                except (AttributeError, KeyError, TypeError, ValueError):
                    missing_slots.append(slot_id)
                    continue
                bindings.append(
                    _NeedBinding(
                        query_id=str(binding.query_id),
                        # The only permitted compatibility alias above has an
                        # empty binding slot and query_id == requested slot.
                        # Materialize that requested slot here so the shared
                        # matching core still has an explicit obligation.
                        slot_id=str(binding.slot_id).strip() or slot_id,
                        role=str(binding.role),
                        physical_id=str(binding.physical_id),
                        embedding_space_id=str(binding.embedding_space_id),
                        vector=vector,
                    )
                )
        if missing_slots:
            return self._empty_result(
                "missing_slot_vector",
                invalid_request_contract=True,
                missing_slot_ids=sorted(set(missing_slots)),
            )
        result = self._match_bindings(
            context_vector,
            bindings,
            domain=domain,
            active_anchor_ids=active_anchor_ids,
            unresolved_slot_ids=unresolved,
            target_support_scores=target_support_scores,
            target_support_floor=target_support_floor,
            endpoint_limit=endpoint_limit,
            evaluation_as_of=evaluation_as_of,
            scoring_mode=scoring_mode,
        )
        result.setdefault("query_vector_bundle", bundle.metadata())
        return result
