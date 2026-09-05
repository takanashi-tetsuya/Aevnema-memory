from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import re
from time import perf_counter
import unicodedata

import numpy as np

from memory_demo.association_overlay import (
    AssociationDelta,
    AssociationOverlay,
    StagedAssociationOverlay,
)
from memory_demo.associations.growth import AssociationGrowthEngine
from memory_demo.associations.traversal import GraphTraverser, TraversedNode
from memory_demo.chronology import ChronologyService
from memory_demo.config import AppConfig
from memory_demo.embeddings import EmbeddingCoordinator, normalize_embedding
from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm.prompts import (
    ANSWER_AUDIT_SYSTEM,
    ANSWER_SYSTEM,
    ASSOCIATION_CUE_GATE_SYSTEM,
    EVENT_CONTINUITY_AUDIT_SYSTEM,
    EVIDENCE_COVERAGE_AUDIT_SYSTEM,
    EVIDENCE_RERANK_AUDIT_SYSTEM,
    EVIDENCE_RERANK_SYSTEM,
    HOP_QUERY_SYSTEM,
    QUERY_SYSTEM,
    answer_audit_prompt,
    answer_correction_prompt,
    answer_prompt,
    event_continuity_audit_prompt,
    evidence_coverage_audit_prompt,
    evidence_rerank_audit_prompt,
    evidence_rerank_prompt,
    hop_query_prompt,
    query_intent_prompt,
)
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    ParagraphRepository,
    SourceRepository,
)
from memory_demo.retrieval.context import source_excerpt
from memory_demo.retrieval.cue_index import LexicalAssociationCueIndex
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.retrieval.coverage import (
    select_evidence,
    strict_contextual_attribution,
    treatment_masked_delta,
)
from memory_demo.retrieval.query_planning import (
    expand_rerank_atomic_queries,
    limit_rerank_atomic_queries,
    requires_entity_resolved_followup,
    structural_queries,
)
from memory_demo.types import EvidenceSlot, QueryIntent, QueryVectorBundle, SearchHit, SlotCandidate


_PATH_WORD_RE = re.compile(r"[a-z0-9]{2,}")
_PATH_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")


class QueryEngine:
    def __init__(
        self,
        config: AppConfig,
        model,
        episode_index,
        concept_index,
        episodes: EpisodeRepository,
        concepts: ConceptRepository,
        sources: SourceRepository,
        associations: AssociationRepository,
        logger: JsonlEventLogger | None = None,
        *,
        association_index=None,
        paragraph_index=None,
        paragraphs: ParagraphRepository | None = None,
        episode_sparse_index=None,
        source_sparse_index=None,
        contextual_matcher: ContextualAssociationMatcher | None = None,
    ):
        self.config = config
        self.model = model
        self.episode_index = episode_index
        self.concept_index = concept_index
        self.paragraph_index = paragraph_index
        self.association_index = association_index
        self.episodes = episodes
        self.concepts = concepts
        self.sources = sources
        self.paragraphs = paragraphs
        self.episode_sparse_index = episode_sparse_index
        self.source_sparse_index = source_sparse_index
        self.contextual_matcher = contextual_matcher
        self.associations = associations
        self._concept_reach_cache: dict[int, int] = {}
        self.last_query_embeddings: dict[str, np.ndarray] = {}
        self.last_query_embedding_cache_trace: dict[str, list[str]] = {
            "hits": [],
            "misses": [],
        }
        self.logger = logger
        if hasattr(self.model, "logger"):
            self.model.logger = logger
        self.traverser = GraphTraverser(associations)
        self.growth = AssociationGrowthEngine(
            model, associations, episodes, config.weights, logger
        )
        self.chronology = ChronologyService(episodes, associations, logger)
        self._lexical_association_cues: LexicalAssociationCueIndex | None = None

    def contextual_recall(
        self,
        bundle,
        *,
        domain: str | None = None,
        endpoint_limit: int | None = None,
        active_anchor_ids=None,
        unresolved_slot_ids=None,
        target_support_scores=None,
        target_support_floor: float = 0.0,
    ) -> dict:
        """Run the optional local double-key lane for a prepared query bundle.

        This method is intentionally additive: the normal ``query`` path is
        unchanged while the experiment flag is disabled.  Callers can attach
        returned target Episode IDs to their candidate pool and feed them
        through the existing evidence contract.
        """
        if self.contextual_matcher is None or not self.config.retrieval.contextual_association_enabled:
            return {
                "enabled": False,
                "hits": [],
                "context_hits": [],
                "need_hits": [],
                "external_calls": 0,
            }
        result = self.contextual_matcher.match_bundle(
            bundle,
            domain=domain,
            endpoint_limit=endpoint_limit,
            active_anchor_ids=active_anchor_ids,
            unresolved_slot_ids=unresolved_slot_ids,
            target_support_scores=target_support_scores,
            target_support_floor=target_support_floor,
        )
        result["enabled"] = True
        result["backend"] = "contextual_double_key"
        result["external_calls"] = 0
        result["attached_episode_ids"] = [
            int(hit.target_episode_id) for hit in result.get("hits", [])
        ]
        result["attached_edges"] = [
            int(hit.association_id) for hit in result.get("hits", [])
        ]
        if self.logger:
            self.logger.emit(
                "contextual_association_retrieval",
                enabled=True,
                backend="contextual_double_key",
                context_hits=len(result.get("context_hits", [])),
                need_hits=sum(len(item) for item in result.get("need_hits", [])),
                attached_edges=result["attached_edges"],
                attached_episode_ids=result["attached_episode_ids"],
                external_calls=0,
            )
        return result

    def embed_query_bundle(self, texts):
        """Build a request-level vector bundle through one provider batch."""
        coordinator = EmbeddingCoordinator(
            self.model,
            model_id=getattr(self.config.model, "embedding_model", ""),
            dimension=self.config.model.embedding_dimension,
        )
        return coordinator.embed_query_bundle_sync(texts)

    @staticmethod
    def _slot_query_vector(
        bundle: QueryVectorBundle | None,
        slot: EvidenceSlot,
    ):
        """Resolve a slot to an already-prepared request vector, if any."""
        if bundle is None:
            return None
        wanted = {
            str(slot.slot_id).strip(),
            str(slot.query_id).strip(),
            str(slot.question).strip(),
        }
        wanted.discard("")
        for item in bundle.queries:
            if (
                str(item.slot_id).strip() in wanted
                or str(item.query_id).strip() in wanted
                or str(getattr(item, "text", "")).strip() in wanted
            ):
                return item
        return None

    @staticmethod
    def _request_evidence_slots(
        rerank_trace: dict,
        bundle: QueryVectorBundle | None = None,
    ) -> tuple[list[EvidenceSlot], dict[int, set[str]]]:
        """Translate existing coverage/floor traces into local slot support.

        The conversion intentionally uses only already-produced deterministic
        or reranker metadata.  It never asks a model to label candidates just
        for contextual recall.
        """
        slots: list[EvidenceSlot] = []
        support: dict[int, set[str]] = {}
        seen_queries: set[str] = set()
        used_slot_ids: set[str] = set()

        def normalized(value: object) -> str:
            return " ".join(str(value or "").casefold().split())

        def add_slot(query: object, episode_ids: object, prefix: str) -> None:
            if not isinstance(episode_ids, list):
                return
            question = str(query or "").strip()
            key = normalized(question)
            if not question or key in seen_queries:
                return
            ids = [
                int(value)
                for value in episode_ids
                if isinstance(value, (int, float, str)) and str(value).strip().lstrip("-").isdigit()
            ]
            if not ids:
                return
            query_vector = next(
                (
                    item
                    for item in (bundle.queries if bundle is not None else ())
                    if normalized(getattr(item, "text", "")) == key
                ),
                None,
            )
            slot_id = (
                str(query_vector.slot_id or query_vector.query_id).strip()
                if query_vector is not None
                else f"{prefix}:{len(slots)}"
            )
            if not slot_id or slot_id in used_slot_ids:
                slot_id = f"{prefix}:{len(slots)}"
            slots.append(
                EvidenceSlot(
                    slot_id=slot_id,
                    question=question,
                    query_id=(
                        str(query_vector.query_id)
                        if query_vector is not None
                        else question
                    ),
                    required=True,
                )
            )
            seen_queries.add(key)
            used_slot_ids.add(slot_id)
            for episode_id in ids:
                support.setdefault(episode_id, set()).add(slot_id)

        merged = rerank_trace.get("merged_coverage", {})
        coverage = merged.get("coverage", []) if isinstance(merged, dict) else []
        for item in coverage if isinstance(coverage, list) else []:
            if isinstance(item, dict):
                add_slot(item.get("query"), item.get("episode_ids"), "coverage")
        deterministic = rerank_trace.get("deterministic_evidence_floor", {})
        if isinstance(deterministic, dict):
            for lane in ("constraint_slots", "atomic_slots"):
                for item in deterministic.get(lane, []):
                    if isinstance(item, dict):
                        add_slot(item.get("query"), item.get("floor_episode_ids"), lane)
        return slots, support

    def _contextual_target_support_scores(
        self,
        bundle: QueryVectorBundle | None,
        slots: list[EvidenceSlot],
        target_episode_ids: list[int],
    ) -> dict[tuple[str, int], float]:
        """Score target Episode ↔ unresolved slot locally from RAM vectors."""
        if bundle is None or not target_episode_ids:
            return {}
        vectors = self.episode_index.get_many(target_episode_ids)
        if not vectors:
            return {}
        scores: dict[tuple[str, int], float] = {}
        for slot in slots:
            query = self._slot_query_vector(bundle, slot)
            if query is None:
                continue
            query_vector = normalize_embedding(
                query.vector,
                self.config.model.embedding_dimension,
            )
            for episode_id, target_vector in vectors.items():
                value = float(np.dot(query_vector, target_vector))
                scores[(slot.slot_id, int(episode_id))] = max(0.0, value)
        return scores

    @staticmethod
    def _base_anchor_activations(seeds: list[SearchHit]) -> dict[int, float]:
        """Keep only independently retrieved Episode anchors for v2 recall."""
        raw = {
            int(item.node_id): max(0.0, float(item.score))
            for item in seeds
            if item.node_type == "episode" and float(item.score) > 0
        }
        maximum = max(raw.values(), default=0.0)
        return (
            {episode_id: score / maximum for episode_id, score in raw.items()}
            if maximum > 0
            else {}
        )

    def _slot_candidates(
        self,
        episodes: list[dict],
        slot_support: dict[int, set[str]],
        reranked_episode_ids: list[int],
        contextual_hits=(),
        target_support_scores: dict[tuple[str, int], float] | None = None,
    ) -> list[SlotCandidate]:
        rank = {
            int(episode_id): index
            for index, episode_id in enumerate(reranked_episode_ids, start=1)
        }
        span = max(1, len(rank))
        contextual_by_episode: dict[int, list] = {}
        for hit in contextual_hits:
            contextual_by_episode.setdefault(int(hit.target_episode_id), []).append(hit)
        result: list[SlotCandidate] = []
        for episode in episodes:
            episode_id = int(episode["id"])
            hits = contextual_by_episode.get(episode_id, [])
            contextual_slots = {str(hit.matched_slot_id) for hit in hits if hit.matched_slot_id}
            contextual_score = max((float(hit.total_score) for hit in hits), default=0.0)
            edge_id = (
                max(hits, key=lambda item: (item.total_score, -item.association_id)).association_id
                if hits
                else None
            )
            direct_rank = (span - rank[episode_id] + 1) / span if episode_id in rank else 0.0
            local_support = max(
                (
                    float(value)
                    for (slot_id, target_id), value in (target_support_scores or {}).items()
                    if target_id == episode_id and slot_id in contextual_slots
                ),
                default=0.0,
            )
            result.append(
                SlotCandidate(
                    episode_id=episode_id,
                    slot_ids=frozenset(slot_support.get(episode_id, set()).union(contextual_slots)),
                    source_lane="contextual" if hits else "base",
                    direct_score=max(float(episode.get("score", 0.0)), direct_rank, local_support),
                    specificity_score=local_support,
                    redundancy_group=str(episode.get("source_key", "")),
                    contextual_edge_id=int(edge_id) if edge_id is not None else None,
                    contextual_score=contextual_score,
                    source_quality=(
                        0.10
                        if int(episode.get("generation", 0) or 0) == 0
                        and str(episode.get("evidence_origin", "source")) in {"source", "mixed"}
                        else 0.0
                    ),
                )
            )
        return result

    def _select_contextual_slots(
        self,
        *,
        episodes: list[dict],
        baseline_selected: list[dict],
        rerank_trace: dict,
        reranked_episode_ids: list[int],
        bundle: QueryVectorBundle | None,
        domain: str | None,
        endpoint_limit: int | None,
        anchor_activations: dict[int, float],
    ) -> tuple[list[dict], dict]:
        """Run masked/treatment set-cover without changing factual evidence rules."""
        slots, slot_support = self._request_evidence_slots(rerank_trace, bundle)
        if not slots:
            return baseline_selected, {
                "enabled": False,
                "reason": "no_evidence_slots",
                "hits": [],
                "external_calls": 0,
            }
        base_candidates = self._slot_candidates(
            episodes, slot_support, reranked_episode_ids
        )
        budget = min(self.config.retrieval.answer_episode_limit, len(episodes))
        masked = select_evidence(base_candidates, slots, budget)
        masked_by_id = {int(item["id"]): item for item in episodes}
        masked_episodes = [
            masked_by_id[item.episode_id]
            for item in masked.selected
            if item.episode_id in masked_by_id
        ] or baseline_selected
        trace: dict = {
            "enabled": bool(self.config.retrieval.contextual_association_enabled),
            "backend": "contextual_double_key_slot_selector_v2",
            "slots": [
                {"slot_id": item.slot_id, "question": item.question, "required": item.required}
                for item in slots
            ],
            "masked_selected_episode_ids": [int(item["id"]) for item in masked_episodes],
            "masked_missing_slots": sorted(masked.missing_required),
            "hits": [],
            "context_hits": [],
            "need_hits": [],
            "attached_edges": [],
            "attached_episode_ids": [],
            "external_calls": 0,
        }
        if (
            not self.config.retrieval.contextual_association_enabled
            or self.contextual_matcher is None
            or bundle is None
            or not masked.missing_required
        ):
            trace["reason"] = (
                "no_unresolved_slots" if not masked.missing_required else "contextual_disabled_or_no_bundle"
            )
            return masked_episodes, trace
        unresolved_slots = [
            item for item in slots if item.slot_id in masked.missing_required
        ]
        # Match first to discover target IDs.  The matcher remains local and
        # has no fallback network path; support is then scored from the RAM
        # Episode index and the same request bundle.
        preliminary = self.contextual_recall(
            bundle,
            domain=domain,
            endpoint_limit=endpoint_limit,
            active_anchor_ids=anchor_activations,
            unresolved_slot_ids=[item.slot_id for item in unresolved_slots],
        )
        target_ids = [int(item.target_episode_id) for item in preliminary.get("hits", [])]
        target_support = self._contextual_target_support_scores(
            bundle, unresolved_slots, target_ids
        )
        trace = self.contextual_recall(
            bundle,
            domain=domain,
            endpoint_limit=endpoint_limit,
            active_anchor_ids=anchor_activations,
            unresolved_slot_ids=[item.slot_id for item in unresolved_slots],
            target_support_scores=target_support,
            target_support_floor=0.05,
        )
        trace["slots"] = [
            {"slot_id": item.slot_id, "question": item.question, "required": item.required}
            for item in slots
        ]
        trace["masked_selected_episode_ids"] = [int(item["id"]) for item in masked_episodes]
        trace["masked_missing_slots"] = sorted(masked.missing_required)
        trace["context_query_id"] = next(
            (str(item.query_id) for item in bundle.queries if item.role == "whole"),
            "",
        )
        trace["query_vectors"] = [
            {
                "query_id": str(item.query_id),
                "text_hash": str(item.text_hash),
                "role": str(item.role),
                "slot_id": str(item.slot_id),
                "text": str(getattr(item, "text", "")),
            }
            for item in bundle.queries
        ]
        if not trace.get("hits"):
            trace.setdefault("reason", "no_contextual_slot_hit")
            return masked_episodes, trace
        existing = {int(item["id"]) for item in episodes}
        new_nodes = [
            TraversedNode("episode", int(hit.target_episode_id), float(hit.target_support_score))
            for hit in trace["hits"]
            if int(hit.target_episode_id) not in existing
        ]
        if new_nodes:
            recovered, _ = self._materialize_nodes(new_nodes, include_sources=True)
            episodes = [*episodes, *recovered]
        treatment_candidates = self._slot_candidates(
            episodes,
            slot_support,
            reranked_episode_ids,
            trace["hits"],
            target_support,
        )
        attribution = strict_contextual_attribution(
            treatment_candidates, slots, budget
        )
        treatment = attribution["treatment"]
        by_id = {int(item["id"]): item for item in episodes}
        treatment_episodes = [
            by_id[item.episode_id]
            for item in treatment.selected
            if item.episode_id in by_id
        ] or masked_episodes
        delta = treatment_masked_delta(treatment, masked)
        trace.update(delta)
        trace["strict_attribution"] = attribution["edges"]
        trace["treatment_selected_episode_ids"] = [int(item["id"]) for item in treatment_episodes]
        trace["selected_count"] = len(treatment_episodes)
        trace["new_slot_count"] = int(delta["new_slot_count"])
        trace["harm_count"] = int(bool(delta["harm"]))
        if self.config.retrieval.contextual_association_shadow:
            trace["shadow"] = True
            return masked_episodes, trace
        trace["shadow"] = False
        return treatment_episodes, trace

    @staticmethod
    def _record_phase(
        phase_seconds: dict[str, float],
        name: str,
        started_at: float,
    ) -> None:
        phase_seconds[name] = round(
            phase_seconds.get(name, 0.0) + perf_counter() - started_at,
            6,
        )

    @property
    def paragraph_retrieval_enabled(self) -> bool:
        return bool(
            self.config.paragraph.enabled
            and getattr(self, "paragraph_index", None) is not None
            and getattr(self, "paragraphs", None) is not None
            and self.paragraph_index.count > 0
            and self.config.retrieval.paragraph_top_k > 0
            and self.config.retrieval.paragraph_rrf_weight > 0.0
        )

    @property
    def sparse_retrieval_enabled(self) -> bool:
        return bool(
            self.config.retrieval.sparse_enabled
            and getattr(self, "episode_sparse_index", None) is not None
            and getattr(self, "source_sparse_index", None) is not None
            and self.episode_sparse_index.count > 0
            and self.source_sparse_index.count > 0
            and (
                self.config.retrieval.sparse_episode_top_k > 0
                or self.config.retrieval.sparse_source_top_k > 0
            )
        )

    @property
    def association_cue_retrieval_enabled(self) -> bool:
        return bool(
            self.config.retrieval.association_cue_enabled
            and getattr(self, "association_index", None) is not None
            and self.association_index.count > 0
            and self.config.retrieval.association_cue_top_k > 0
        )

    @staticmethod
    def _association_is_cue_eligible(row) -> bool:
        return bool(
            row is not None
            and str(row["audit_status"]) == "dual_accepted"
            and str(row["relation_key"]) != "involves"
            and str(row["relation_text"]).strip()
            and "查询中自主增长：" in str(row["created_reason"])
        )

    @classmethod
    def _association_has_fast_path_confirmation(cls, row) -> bool:
        """Keep strong factual edges conservative without stalling cache edges.

        ``evidence_count`` counts independent insert/reinforcement events; it
        does not count the two endpoint Episodes or the two audit passes.  A
        bounded evidence bridge is explicitly a retrieval hint rather than a
        world-model fact, so one dual-audited observation is sufficient for
        immediate reuse.  Stronger semantic/causal edges still require a
        second independent reinforcement before they can bypass reranking.
        """

        relation_key = str(
            cls._association_row_value(row, "relation_key", "") or ""
        ).casefold()
        evidence_count = int(
            cls._association_row_value(row, "evidence_count", 0) or 0
        )
        minimum = 1 if relation_key == "evidence_bridge" else 2
        return evidence_count >= minimum

    def _association_cue_entries_from_matrix(
        self,
        queries: list[str],
        matrix: np.ndarray,
    ) -> list[dict]:
        if not self.association_cue_retrieval_enabled or not queries:
            return []
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (
            len(queries),
            self.config.model.embedding_dimension,
        ):
            raise ValueError("association cue matrix shape does not match queries")
        best_by_id: dict[int, dict] = {}
        for query_index, vector in enumerate(matrix):
            ranked = self.association_index.search(
                vector,
                self.config.retrieval.association_cue_top_k,
            )
            for rank, (association_id, cosine) in enumerate(ranked, start=1):
                if cosine < self.config.retrieval.association_cue_min_similarity:
                    continue
                row = self.associations.get(int(association_id))
                if not self._association_is_cue_eligible(row):
                    continue
                endpoint_score = (
                    self.config.retrieval.association_cue_rrf_weight
                    / (60 + rank)
                )
                item = {
                    "association_id": int(association_id),
                    "query_index": query_index,
                    "query": queries[query_index],
                    "rank": rank,
                    "cosine": float(cosine),
                    "endpoint_score": float(endpoint_score),
                    "endpoints": [
                        [str(row["from_type"]), int(row["from_id"])],
                        [str(row["to_type"]), int(row["to_id"])],
                    ],
                }
                previous = best_by_id.get(int(association_id))
                if previous is None or (
                    float(item["cosine"]), -int(item["rank"])
                ) > (
                    float(previous["cosine"]), -int(previous["rank"])
                ):
                    best_by_id[int(association_id)] = item
        entries = sorted(
            best_by_id.values(),
            key=lambda item: (
                float(item["cosine"]),
                -int(item["rank"]),
                -int(item["association_id"]),
            ),
            reverse=True,
        )
        if self.logger and entries:
            self.logger.emit("association_cue_retrieval", entries=entries)
        return entries

    def _association_cue_entries_from_text(self, queries: list[str]) -> list[dict]:
        """Find an unambiguous audited edge without a cloud embedding call."""

        retrieval = self.config.retrieval
        if (
            not self.association_cue_retrieval_enabled
            or not retrieval.association_cue_fast_path_enabled
            or not retrieval.association_cue_fast_path_local_enabled
            or not queries
        ):
            return []
        if self._lexical_association_cues is None:
            self._lexical_association_cues = LexicalAssociationCueIndex(
                self.associations.list_cue_candidates()
            )
        best_by_id: dict[int, dict] = {}
        for query_index, query in enumerate(queries):
            ranked = self._lexical_association_cues.search(
                query,
                max(2, int(retrieval.association_cue_top_k)),
            )
            if not ranked:
                continue
            top_score = float(ranked[0][1])
            runner_up = float(ranked[1][1]) if len(ranked) > 1 else 0.0
            if top_score < float(
                retrieval.association_cue_fast_path_local_min_coverage
            ) or top_score - runner_up < float(
                retrieval.association_cue_fast_path_local_min_margin
            ):
                continue
            row = ranked[0][0]
            if not self._association_is_cue_eligible(row):
                continue
            association_id = int(row["id"])
            item = {
                "association_id": association_id,
                "query_index": query_index,
                "query": query,
                "rank": 1,
                "cosine": top_score,
                "local_coverage": top_score,
                "local_margin": top_score - runner_up,
                "score_kind": "local_field_idf",
                "endpoint_score": float(
                    retrieval.association_cue_rrf_weight / 61
                ),
                "endpoints": [
                    [str(row["from_type"]), int(row["from_id"])],
                    [str(row["to_type"]), int(row["to_id"])],
                ],
            }
            previous = best_by_id.get(association_id)
            if previous is None or top_score > float(previous["local_coverage"]):
                best_by_id[association_id] = item
        return sorted(
            best_by_id.values(),
            key=lambda item: (
                float(item["local_coverage"]),
                float(item["local_margin"]),
            ),
            reverse=True,
        )

    def match_association_cue_text(self, text: str) -> dict | None:
        """Expose the same fail-closed local cue used by the retrieval lane."""

        entries = self._association_cue_entries_from_text([text])
        if not self._association_cue_fast_endpoint_keys(entries):
            return None
        entry = entries[0]
        return {
            "association_id": int(entry["association_id"]),
            "coverage": float(entry["local_coverage"]),
            "margin": float(entry["local_margin"]),
            "score_kind": str(entry["score_kind"]),
        }

    def _association_cue_endpoint_summary(
        self, node_type: str, node_id: int
    ) -> dict:
        if node_type == "episode":
            row = self.episodes.get(node_id)
            if row is None:
                return {"type": node_type, "id": node_id, "missing": True}
            return {
                "type": node_type,
                "id": node_id,
                "source_key": str(row["source_key"]),
                "text": str(row["text"])[:600],
            }
        if node_type == "concept":
            row = self.concepts.get(node_id)
            if row is None:
                return {"type": node_type, "id": node_id, "missing": True}
            return {
                "type": node_type,
                "id": node_id,
                "text": (
                    f"{row['canonical_name']}：{row['description']}"
                )[:600],
            }
        return {"type": node_type, "id": node_id, "missing": True}

    def _association_cue_fast_endpoint_keys(
        self, entries: list[dict]
    ) -> set[tuple[str, int]]:
        """Return only strong cue endpoints that must survive pool truncation."""

        retrieval = self.config.retrieval
        if not retrieval.association_cue_fast_path_enabled:
            return set()
        keys: set[tuple[str, int]] = set()
        accepted_edges = 0
        ordered_entries = sorted(
            entries,
            key=lambda item: float(item.get("cosine", 0.0)),
            reverse=True,
        )
        if len(ordered_entries) >= 2 and (
            float(ordered_entries[0].get("cosine", 0.0))
            - float(ordered_entries[1].get("cosine", 0.0))
            < float(retrieval.association_cue_fast_path_min_margin)
        ):
            return set()
        for entry in ordered_entries:
            if accepted_edges >= max(
                1, int(retrieval.association_cue_fast_path_max_edges)
            ):
                break
            if str(entry.get("score_kind", "")).startswith("local_"):
                if float(entry.get("local_coverage", 0.0)) < float(
                    retrieval.association_cue_fast_path_local_min_coverage
                ) or float(entry.get("local_margin", 0.0)) < float(
                    retrieval.association_cue_fast_path_local_min_margin
                ):
                    continue
            elif float(entry.get("cosine", 0.0)) < float(
                retrieval.association_cue_fast_path_min_similarity
            ):
                continue
            row = self.associations.get(int(entry.get("association_id", -1)))
            if not self._association_is_cue_eligible(row):
                continue
            if float(
                self._association_row_value(row, "confidence", 0.0) or 0.0
            ) < float(retrieval.association_cue_fast_path_min_confidence):
                continue
            if not self._association_has_fast_path_confirmation(row):
                continue
            endpoints = [
                (str(node_type), int(node_id))
                for node_type, node_id in entry.get("endpoints", [])
            ]
            if len(endpoints) != 2 or any(
                node_type != "episode" for node_type, _ in endpoints
            ):
                continue
            keys.update(endpoints)
            accepted_edges += 1
        return keys

    @staticmethod
    def _truncate_traversed_nodes(
        nodes: list[TraversedNode],
        limit: int,
        protected: set[tuple[str, int]] | None = None,
    ) -> list[TraversedNode]:
        """Apply a hard pool budget without dropping a protected recall lane."""

        if limit <= 0:
            return []
        if len(nodes) <= limit or not protected:
            return nodes[:limit]
        protected_nodes = [
            item
            for item in nodes
            if (item.node_type, int(item.node_id)) in protected
        ][:limit]
        protected_keys = {
            (item.node_type, int(item.node_id)) for item in protected_nodes
        }
        ordinary = [
            item
            for item in nodes
            if (item.node_type, int(item.node_id)) not in protected_keys
        ][: max(0, limit - len(protected_nodes))]
        return [*ordinary, *protected_nodes]

    def _semantic_gate_association_cues(
        self,
        question: str,
        entries: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Fail closed unless both relation endpoints fit the query scope."""
        if (
            not self.config.retrieval.association_cue_semantic_gate_enabled
            or not entries
        ):
            return entries, []
        candidates: list[dict] = []
        for entry in entries:
            association_id = int(entry["association_id"])
            row = self.associations.get(association_id)
            if not self._association_is_cue_eligible(row):
                continue
            endpoints = [
                self._association_cue_endpoint_summary(
                    str(row["from_type"]), int(row["from_id"])
                ),
                self._association_cue_endpoint_summary(
                    str(row["to_type"]), int(row["to_id"])
                ),
            ]
            candidates.append(
                {
                    "association_id": association_id,
                    "cosine": round(float(entry.get("cosine", 0.0)), 6),
                    "relation_key": str(row["relation_key"]),
                    "relation_text": str(row["relation_text"]),
                    "association_mode": (
                        str(row["association_mode"])
                        if "association_mode" in row.keys()
                        else "semantic"
                    ),
                    "generation": int(row["generation"]),
                    "claim_level": str(row["claim_level"]),
                    "endpoints": endpoints,
                }
            )
        payload = self.model.chat_json(
            ASSOCIATION_CUE_GATE_SYSTEM,
            json.dumps(
                {"question": question, "candidates": candidates},
                ensure_ascii=False,
            ),
        )
        raw_decisions = (
            payload.get("decisions", []) if isinstance(payload, dict) else []
        )
        allowed_ids = {int(item["association_id"]) for item in candidates}
        decisions_by_id: dict[int, dict] = {}
        if isinstance(raw_decisions, list):
            for raw in raw_decisions:
                if not isinstance(raw, dict):
                    continue
                try:
                    association_id = int(raw["association_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                if association_id not in allowed_ids:
                    continue
                decisions_by_id[association_id] = {
                    "association_id": association_id,
                    "accept": raw.get("accept") is True,
                    "reason": str(raw.get("reason", ""))[:500],
                }
        decisions = [
            decisions_by_id.get(
                int(candidate["association_id"]),
                {
                    "association_id": int(candidate["association_id"]),
                    "accept": False,
                    "reason": "模型未返回有效决定，按 fail-closed 拒绝。",
                },
            )
            for candidate in candidates
        ]
        accepted_ids = {
            int(item["association_id"])
            for item in decisions
            if item["accept"]
        }
        maximum = max(
            0,
            int(
                self.config.retrieval.association_cue_semantic_gate_max_selected
            ),
        )
        accepted = [
            dict(entry)
            for entry in entries
            if int(entry["association_id"]) in accepted_ids
        ][:maximum]
        for entry in accepted:
            decision = decisions_by_id[int(entry["association_id"])]
            entry["semantic_gate_accept"] = True
            entry["semantic_gate_reason"] = decision["reason"]
        if self.logger:
            self.logger.emit(
                "association_cue_semantic_gate",
                question=question,
                decisions=decisions,
                accepted_association_ids=[
                    int(item["association_id"]) for item in accepted
                ],
            )
        return accepted, decisions

    def _active_association_cue_hits(
        self,
        entries: list[dict],
    ) -> tuple[list[SearchHit], list[int], list[dict]]:
        hits: list[SearchHit] = []
        association_ids: list[int] = []
        active_entries: list[dict] = []
        for raw in entries:
            association_id = int(raw["association_id"])
            row = self.associations.get(association_id)
            if not self._association_is_cue_eligible(row):
                continue
            association_ids.append(association_id)
            item = dict(raw)
            # The durable row is authoritative if an overlay restored it.
            item["endpoints"] = [
                [str(row["from_type"]), int(row["from_id"])],
                [str(row["to_type"]), int(row["to_id"])],
            ]
            active_entries.append(item)
            score = float(item.get("endpoint_score", 0.0))
            for node_type, node_id in item["endpoints"]:
                hits.append(SearchHit(str(node_type), int(node_id), score))
        return self._merge_hits(hits), list(dict.fromkeys(association_ids)), active_entries

    def _source_episode_expansions(
        self,
        query: np.ndarray,
        source_ranked: list[tuple[int, float]],
    ) -> list[dict[str, float | int]]:
        """Map Source lexical hits to the locally closest persisted Episodes."""
        if not source_ranked:
            return []
        episode_rows = self.episodes.list_ids_by_source_ids(
            source_id for source_id, _score in source_ranked
        )
        ids_by_source: dict[int, list[int]] = {}
        for row in episode_rows:
            ids_by_source.setdefault(int(row["source_id"]), []).append(int(row["id"]))
        vectors = self.episode_index.get_many(
            episode_id
            for episode_ids in ids_by_source.values()
            for episode_id in episode_ids
        )
        limit = max(
            1, self.config.retrieval.sparse_source_episode_expansion_limit
        )
        expansions: list[dict[str, float | int]] = []
        for source_rank, (source_id, sparse_score) in enumerate(
            source_ranked, start=1
        ):
            scored = sorted(
                (
                    (episode_id, float(vectors[episode_id] @ query))
                    for episode_id in ids_by_source.get(int(source_id), [])
                    if episode_id in vectors
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
            expansions.extend(
                {
                    "source_id": int(source_id),
                    "source_rank": source_rank,
                    "source_sparse_score": float(sparse_score),
                    "episode_local_rank": local_rank,
                    "episode_id": episode_id,
                    "episode_cosine": episode_cosine,
                }
                for local_rank, (episode_id, episode_cosine) in enumerate(
                    scored, start=1
                )
            )
        # Do not let all local Episodes from Source rank 1 consume the atomic
        # anchor budget before Source rank 2 is inspected.  Round-robin by
        # local rank preserves the sparse Source ranking while representing
        # several independently matched Source rows early.
        expansions.sort(
            key=lambda item: (
                int(item["episode_local_rank"]),
                int(item["source_rank"]),
                int(item["episode_id"]),
            )
        )
        return expansions

    def _source_key_cohort_hits(
        self,
        episode_anchor_ids: list[int],
        traversed: list[TraversedNode],
    ) -> tuple[list[SearchHit], dict]:
        """Recover bounded Episodes split across one logical input file.

        Source is intentionally a short evidence fragment, while source_key is
        the stable logical file identity.  Two independent Episode anchors in
        the same source_key are enough evidence that the file is locally
        relevant; all of its Episodes may then enter the rerank candidate pool
        when the cohort is small.  The added rows remain ordinary direct
        Episode evidence and do not become Associations.
        """
        retrieval = self.config.retrieval
        trace = {
            "enabled": bool(retrieval.source_key_cohort_enabled),
            "supported_source_keys": [],
            "added_episode_ids": [],
            "boosted_episode_ids": [],
            "skipped_source_keys": [],
        }
        if not retrieval.source_key_cohort_enabled or not episode_anchor_ids:
            return [], trace

        ordered_anchor_ids = list(dict.fromkeys(int(value) for value in episode_anchor_ids))
        anchor_rows = {
            int(row["id"]): row
            for row in self.episodes.get_many(ordered_anchor_ids)
        }
        ids_by_key: dict[str, list[int]] = {}
        first_rank: dict[str, int] = {}
        for rank, episode_id in enumerate(ordered_anchor_ids):
            row = anchor_rows.get(episode_id)
            if row is None:
                continue
            source_key = str(row["source_key"] or "")
            if not source_key:
                continue
            ids_by_key.setdefault(source_key, []).append(episode_id)
            first_rank.setdefault(source_key, rank)

        minimum_hits = max(1, int(retrieval.source_key_cohort_min_anchor_hits))
        supported_keys = sorted(
            (
                source_key
                for source_key, ids in ids_by_key.items()
                if len(set(ids)) >= minimum_hits
            ),
            key=lambda source_key: (
                first_rank[source_key],
                -len(set(ids_by_key[source_key])),
                source_key,
            ),
        )
        trace["supported_source_keys"] = supported_keys
        if not supported_keys:
            return [], trace

        existing_ids = {
            int(item.node_id)
            for item in traversed
            if item.node_type == "episode"
        }
        traversed_scores = {
            int(item.node_id): float(item.score)
            for item in traversed
            if item.node_type == "episode"
        }
        per_key_limit = max(
            minimum_hits,
            int(retrieval.source_key_cohort_max_episodes_per_key),
        )
        remaining = max(0, int(retrieval.source_key_cohort_total_limit))
        score_ratio = max(0.0, min(1.0, float(retrieval.source_key_cohort_score_ratio)))
        hits: list[SearchHit] = []
        maximum_keys = max(0, int(retrieval.source_key_cohort_max_keys))
        for source_key in supported_keys[:maximum_keys]:
            rows = list(self.episodes.list_by_source_key(source_key))
            if len(rows) > per_key_limit:
                trace["skipped_source_keys"].append(
                    {
                        "source_key": source_key,
                        "episode_count": len(rows),
                        "reason": "cohort_exceeds_per_key_limit",
                    }
                )
                continue
            if len(rows) > remaining:
                trace["skipped_source_keys"].append(
                    {
                        "source_key": source_key,
                        "episode_count": len(rows),
                        "reason": "cohort_exceeds_total_limit",
                    }
                )
                continue
            anchor_score = max(
                (
                    traversed_scores.get(episode_id, 0.0)
                    for episode_id in ids_by_key[source_key]
                ),
                default=0.0,
            )
            cohort_score = max(0.01, anchor_score * score_ratio)
            for row in rows:
                episode_id = int(row["id"])
                hits.append(SearchHit("episode", episode_id, cohort_score))
                if episode_id in existing_ids:
                    if traversed_scores.get(episode_id, 0.0) < cohort_score:
                        trace["boosted_episode_ids"].append(episode_id)
                else:
                    existing_ids.add(episode_id)
                    trace["added_episode_ids"].append(episode_id)
            remaining -= len(rows)
            if remaining <= 0:
                break
        return hits, trace

    def _paragraph_episode_expansions(
        self,
        query: np.ndarray,
        paragraph_ranked: list[tuple[int, float]],
    ) -> list[dict[str, float | int]]:
        """Map a raw-text hit to the most query-relevant Episodes in its Source."""
        if not paragraph_ranked or self.paragraphs is None:
            return []
        rows = {
            int(row["id"]): row
            for row in self.paragraphs.get_many(
                paragraph_id for paragraph_id, _score in paragraph_ranked
            )
        }
        best_by_source: dict[int, tuple[int, int, float]] = {}
        for rank, (paragraph_id, score) in enumerate(paragraph_ranked, start=1):
            row = rows.get(int(paragraph_id))
            if row is None:
                continue
            source_id = int(row["source_id"])
            best_by_source.setdefault(
                source_id, (rank, int(paragraph_id), float(score))
            )
        episode_rows = self.episodes.list_ids_by_source_ids(best_by_source)
        ids_by_source: dict[int, list[int]] = {}
        for row in episode_rows:
            ids_by_source.setdefault(int(row["source_id"]), []).append(int(row["id"]))
        all_episode_ids = [
            episode_id
            for episode_ids in ids_by_source.values()
            for episode_id in episode_ids
        ]
        vectors = self.episode_index.get_many(all_episode_ids)
        limit = max(1, self.config.retrieval.paragraph_episode_expansion_limit)
        expansions: list[dict[str, float | int]] = []
        for source_id, (rank, paragraph_id, paragraph_cosine) in best_by_source.items():
            scored = sorted(
                (
                    (episode_id, float(vectors[episode_id] @ query))
                    for episode_id in ids_by_source.get(source_id, [])
                    if episode_id in vectors
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
            expansions.extend(
                {
                    "paragraph_id": paragraph_id,
                    "paragraph_rank": rank,
                    "paragraph_cosine": paragraph_cosine,
                    "source_id": source_id,
                    "episode_id": episode_id,
                    "episode_cosine": episode_cosine,
                }
                for episode_id, episode_cosine in scored
            )
        return expansions

    def _paragraph_context_by_source(
        self,
        paragraph_rankings: list[list[dict]],
    ) -> dict[int, list[dict]]:
        """Keep query-matched raw Paragraphs as Source-level rerank evidence.

        The snippets deliberately do not become Episode facts.  Every candidate
        from the same Source receives the same labelled raw context so the LLM
        can use omitted dialogue while preserving the provenance boundary.
        """
        retrieval = self.config.retrieval
        if (
            not self.paragraph_retrieval_enabled
            or not retrieval.paragraph_rerank_context_enabled
            or self.paragraphs is None
        ):
            return {}
        best: dict[int, dict] = {}
        for query_index, ranking in enumerate(paragraph_rankings):
            for rank, item in enumerate(ranking, start=1):
                paragraph_id = int(item["id"])
                score = float(item.get("score", 0.0))
                current = best.get(paragraph_id)
                candidate = {
                    "paragraph_id": paragraph_id,
                    "query_index": query_index,
                    "rank": rank,
                    "score": score,
                }
                if current is None or (score, -rank, -query_index) > (
                    float(current["score"]),
                    -int(current["rank"]),
                    -int(current["query_index"]),
                ):
                    best[paragraph_id] = candidate
        rows = {
            int(row["id"]): row
            for row in self.paragraphs.get_many(best)
        }
        by_source: dict[int, list[dict]] = {}
        maximum_chars = max(1, retrieval.paragraph_rerank_context_chars)
        for paragraph_id, match in best.items():
            row = rows.get(paragraph_id)
            if row is None:
                continue
            source_id = int(row["source_id"])
            by_source.setdefault(source_id, []).append(
                {
                    **match,
                    "source_id": source_id,
                    "text": str(row["text"])[:maximum_chars],
                    "provenance": "source_level_raw_paragraph",
                }
            )
        per_source = max(1, retrieval.paragraph_rerank_context_per_source)
        for source_id, items in by_source.items():
            items.sort(
                key=lambda item: (
                    float(item["score"]),
                    -int(item["rank"]),
                    -int(item["paragraph_id"]),
                ),
                reverse=True,
            )
            by_source[source_id] = items[:per_source]
        return by_source

    def _parse_intent(self, question: str) -> QueryIntent:
        payload = self.model.chat_json(QUERY_SYSTEM, query_intent_prompt(question))
        if not isinstance(payload, dict):
            raise ValueError("query intent response must be an object")
        intent = QueryIntent.from_dict(payload)
        intent.search_queries = list(
            dict.fromkeys(
                [
                    *intent.search_queries,
                    *structural_queries(question, intent),
                ]
            )
        )
        return intent

    def _vector_seed_hits(
        self,
        queries: list[str],
        episode_anchor_ids: list[int] | None = None,
        first_query_anchor_limit: int | None = None,
    ) -> list[SearchHit]:
        if not queries:
            return []
        matrix = np.asarray(self.model.embed(queries), dtype=np.float32)
        hits, _ = self._vector_seed_hits_from_matrix(
            queries,
            matrix,
            episode_anchor_ids,
            first_query_anchor_limit,
        )
        return hits

    def _vector_seed_hits_with_cues(
        self,
        queries: list[str],
        episode_anchor_ids: list[int] | None = None,
        first_query_anchor_limit: int | None = None,
        cue_scope_question: str | None = None,
        *,
        phase_seconds: dict[str, float] | None = None,
        timing_prefix: str = "",
        query_embeddings_override: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[SearchHit], list[int], list[dict], dict[str, list]]:
        if not queries:
            return [], [], [], {
                "episode": [],
                "concept": [],
                "paragraph": [],
                "paragraph_episode_expansion": [],
            }
        # A strongly separated local match against an already dual-audited
        # learned edge is a true cache hit.  Its two direct Episode endpoints
        # are sufficient seed evidence, so neither query embedding nor broad
        # vector/sparse recall is needed on this lane.
        local_entries = self._association_cue_entries_from_text(queries)
        local_cue_hits, local_cue_ids, local_active_entries = (
            self._active_association_cue_hits(local_entries)
        )
        if self._association_cue_fast_endpoint_keys(local_active_entries):
            rankings = {
                "episode": [],
                "concept": [],
                "paragraph": [],
                "paragraph_episode_expansion": [],
                "association_capsule_fast_lane": True,
                "association_capsule_lookup": "local_field_idf",
            }
            if phase_seconds is not None and timing_prefix:
                phase_seconds[f"{timing_prefix}_embedding"] = 0.0
                phase_seconds[f"{timing_prefix}_retrieval"] = 0.0
            return (
                local_cue_hits,
                local_cue_ids,
                local_active_entries,
                rankings,
            )
        embed_started = perf_counter()
        overrides = query_embeddings_override or {}
        missing_queries = [query for query in queries if query not in overrides]
        embedded_by_query: dict[str, np.ndarray] = {}
        if missing_queries:
            missing_matrix = np.asarray(
                self.model.embed(missing_queries), dtype=np.float32
            )
            if missing_matrix.shape != (
                len(missing_queries),
                self.config.model.embedding_dimension,
            ):
                raise ValueError("query embedding matrix shape does not match queries")
            embedded_by_query = {
                query: vector
                for query, vector in zip(
                    missing_queries, missing_matrix, strict=True
                )
            }
        normalized_by_query = {
            query: normalize_embedding(
                np.asarray(
                    overrides[query]
                    if query in overrides
                    else embedded_by_query[query],
                    dtype=np.float32,
                ),
                self.config.model.embedding_dimension,
            ).copy()
            for query in queries
        }
        matrix = np.asarray(
            [normalized_by_query[query] for query in queries],
            dtype=np.float32,
        )
        self.last_query_embeddings.update(
            {str(query): vector.copy() for query, vector in normalized_by_query.items()}
        )
        self.last_query_embedding_cache_trace["hits"].extend(
            query for query in queries if query in overrides
        )
        self.last_query_embedding_cache_trace["misses"].extend(missing_queries)
        if phase_seconds is not None and timing_prefix:
            self._record_phase(
                phase_seconds,
                f"{timing_prefix}_embedding",
                embed_started,
            )
        retrieval_started = perf_counter()
        entries = self._association_cue_entries_from_matrix(queries, matrix)
        entries, _decisions = self._semantic_gate_association_cues(
            cue_scope_question or queries[0], entries
        )
        cue_hits, cue_ids, active_entries = self._active_association_cue_hits(entries)
        fast_lane = bool(self._association_cue_fast_endpoint_keys(active_entries))
        vector_hits, rankings = self._vector_seed_hits_from_matrix(
            queries,
            matrix,
            episode_anchor_ids,
            first_query_anchor_limit,
            dense_only=fast_lane,
        )
        rankings["association_capsule_fast_lane"] = fast_lane
        if phase_seconds is not None and timing_prefix:
            self._record_phase(
                phase_seconds,
                f"{timing_prefix}_retrieval",
                retrieval_started,
            )
        return (
            self._merge_hits(vector_hits, cue_hits),
            cue_ids,
            active_entries,
            rankings,
        )

    def embed_query_text(self, text: str) -> np.ndarray:
        """Return one normalized query vector for request-cache matching.

        This uses the embedding endpoint only.  Embeddings never fall back to
        a reasoning model, matching the retrieval and import contract.
        """

        matrix = np.asarray(self.model.embed([text]), dtype=np.float32)
        if matrix.shape != (1, self.config.model.embedding_dimension):
            raise ValueError("query embedding shape does not match configured dimension")
        return normalize_embedding(
            matrix[0], self.config.model.embedding_dimension
        ).copy()

    def _vector_seed_hits_from_matrix(
        self,
        queries: list[str],
        matrix: np.ndarray,
        episode_anchor_ids: list[int] | None = None,
        first_query_anchor_limit: int | None = None,
        *,
        dense_only: bool = False,
    ) -> tuple[list[SearchHit], dict[str, list[list[dict[str, float | int]]]]]:
        """Rank fixed float32 query vectors and expose rankings for replay."""
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (
            len(queries),
            self.config.model.embedding_dimension,
        ):
            raise ValueError(
                "query embedding matrix shape does not match queries and dimension"
            )
        fused_scores: dict[tuple[str, int], float] = {}
        best_cosines: dict[tuple[str, int], float] = {}
        episode_rankings: list[list[tuple[int, float]]] = []
        concept_rankings: list[list[tuple[int, float]]] = []
        concept_raw_rankings: list[list[tuple[int, float]]] = []
        concept_gate_audits: list[list[dict[str, int | float | bool]]] = []
        paragraph_rankings: list[list[tuple[int, float]]] = []
        paragraph_expansion_rankings: list[list[dict[str, float | int]]] = []
        sparse_episode_rankings: list[list[tuple[int, float]]] = []
        sparse_source_rankings: list[list[tuple[int, float]]] = []
        sparse_source_expansion_rankings: list[
            list[dict[str, float | int]]
        ] = []
        fused_episode_rankings: list[list[tuple[int, float]]] = []
        atomic_episode_rankings: list[list[tuple[int, float]]] = []
        paragraph_only_scores: dict[tuple[str, int], float] = {}
        paragraph_only_cosines: dict[tuple[str, int], float] = {}
        for query_index, vector in enumerate(matrix):
            normalized = normalize_embedding(
                vector, self.config.model.embedding_dimension
            )
            episode_ranked = self.episode_index.search(
                normalized, self.config.retrieval.episode_top_k
            )
            concept_ranked_raw = self.concept_index.search(
                normalized, self.config.retrieval.concept_top_k
            )
            minimum_concept_reach = max(
                1,
                self.config.retrieval.concept_seed_min_reachable_episodes,
            )
            if minimum_concept_reach <= 1:
                concept_degrees = {
                    int(node_id): 1 for node_id, _score in concept_ranked_raw
                }
                concept_ranked = list(concept_ranked_raw)
            else:
                raw_ids = [int(node_id) for node_id, _score in concept_ranked_raw]
                missing_ids = [
                    node_id
                    for node_id in raw_ids
                    if node_id not in self._concept_reach_cache
                ]
                if missing_ids:
                    self._concept_reach_cache.update(
                        self.associations.concept_reachable_episode_counts(
                            missing_ids
                        )
                    )
                concept_degrees = {
                    node_id: self._concept_reach_cache.get(node_id, 0)
                    for node_id in raw_ids
                }
                concept_ranked = [
                    (node_id, score)
                    for node_id, score in concept_ranked_raw
                    if concept_degrees.get(int(node_id), 0)
                    >= minimum_concept_reach
                ]
            paragraph_ranked = (
                self.paragraph_index.search(
                    normalized, self.config.retrieval.paragraph_top_k
                )
                if self.paragraph_retrieval_enabled and not dense_only
                else []
            )
            episode_rankings.append(episode_ranked)
            concept_raw_rankings.append(concept_ranked_raw)
            concept_rankings.append(concept_ranked)
            admitted_concept_ids = {int(node_id) for node_id, _ in concept_ranked}
            concept_gate_audits.append(
                [
                    {
                        "id": int(node_id),
                        "score": float(score),
                        "reachable_episode_count": int(
                            concept_degrees.get(int(node_id), 0)
                        ),
                        "admitted": int(node_id) in admitted_concept_ids,
                    }
                    for node_id, score in concept_ranked_raw
                ]
            )
            paragraph_rankings.append(paragraph_ranked)
            sparse_episode_ranked = (
                self.episode_sparse_index.search(
                    queries[query_index],
                    self.config.retrieval.sparse_episode_top_k,
                )
                if self.sparse_retrieval_enabled and not dense_only
                else []
            )
            sparse_source_ranked = (
                self.source_sparse_index.search(
                    queries[query_index],
                    self.config.retrieval.sparse_source_top_k,
                )
                if self.sparse_retrieval_enabled and not dense_only
                else []
            )
            sparse_episode_rankings.append(sparse_episode_ranked)
            sparse_source_rankings.append(sparse_source_ranked)
            query_episode_scores: dict[int, float] = {}
            ranked_groups = (
                (
                    "episode",
                    episode_ranked,
                ),
                (
                    "concept",
                    concept_ranked,
                ),
            )
            for node_type, ranked in ranked_groups:
                for rank, (node_id, cosine) in enumerate(ranked, start=1):
                    key = (node_type, node_id)
                    fused_scores[key] = (
                        fused_scores.get(key, 0.0) + 1.0 / (60 + rank)
                    )
                    best_cosines[key] = max(
                        best_cosines.get(key, -1.0), cosine
                    )
                    if node_type == "episode":
                        query_episode_scores[node_id] = (
                            query_episode_scores.get(node_id, 0.0)
                            + 1.0 / (60 + rank)
                        )
            sparse_vectors = (
                self.episode_index.get_many(
                    node_id for node_id, _score in sparse_episode_ranked
                )
                if sparse_episode_ranked
                else {}
            )
            for rank, (node_id, _sparse_score) in enumerate(
                sparse_episode_ranked, start=1
            ):
                key = ("episode", node_id)
                contribution = (
                    self.config.retrieval.sparse_episode_rrf_weight
                    / (60 + rank)
                )
                fused_scores[key] = fused_scores.get(key, 0.0) + contribution
                query_episode_scores[node_id] = (
                    query_episode_scores.get(node_id, 0.0) + contribution
                )
                vector_row = sparse_vectors.get(node_id)
                if vector_row is not None:
                    best_cosines[key] = max(
                        best_cosines.get(key, -1.0),
                        float(vector_row @ normalized),
                    )
            source_expansions = self._source_episode_expansions(
                normalized, sparse_source_ranked
            )
            sparse_source_expansion_rankings.append(source_expansions)
            for expansion in source_expansions:
                node_id = int(expansion["episode_id"])
                key = ("episode", node_id)
                combined_rank = (
                    int(expansion["source_rank"])
                    + int(expansion["episode_local_rank"])
                    - 1
                )
                contribution = (
                    self.config.retrieval.sparse_source_rrf_weight
                    / (60 + combined_rank)
                )
                fused_scores[key] = fused_scores.get(key, 0.0) + contribution
                query_episode_scores[node_id] = (
                    query_episode_scores.get(node_id, 0.0) + contribution
                )
                best_cosines[key] = max(
                    best_cosines.get(key, -1.0),
                    float(expansion["episode_cosine"]),
                )
            paragraph_expansions = self._paragraph_episode_expansions(
                normalized, paragraph_ranked
            )
            paragraph_expansion_rankings.append(paragraph_expansions)
            for expansion in paragraph_expansions:
                if not self.config.retrieval.paragraph_seed_enabled:
                    continue
                key = ("episode", int(expansion["episode_id"]))
                contribution = (
                    self.config.retrieval.paragraph_rrf_weight
                    / (60 + int(expansion["paragraph_rank"]))
                )
                if self.config.retrieval.paragraph_recall_only:
                    paragraph_only_scores[key] = (
                        paragraph_only_scores.get(key, 0.0) + contribution
                    )
                    paragraph_only_cosines[key] = max(
                        paragraph_only_cosines.get(key, -1.0),
                        float(expansion["episode_cosine"]),
                    )
                else:
                    fused_scores[key] = fused_scores.get(key, 0.0) + contribution
                    best_cosines[key] = max(
                        best_cosines.get(key, -1.0),
                        float(expansion["episode_cosine"]),
                    )
                    query_episode_scores[int(expansion["episode_id"])] = (
                        query_episode_scores.get(int(expansion["episode_id"]), 0.0)
                        + contribution
                    )
            fused_episode_ranking = sorted(
                query_episode_scores.items(),
                key=lambda item: (
                    item[1],
                    best_cosines.get(("episode", item[0]), -1.0),
                ),
                reverse=True,
            )
            fused_episode_rankings.append(fused_episode_ranking)
            source_episode_ranking = [
                (
                    int(expansion["episode_id"]),
                    float(expansion["episode_cosine"]),
                )
                for expansion in source_expansions
            ]
            paragraph_episode_ranking = [
                (
                    int(expansion["episode_id"]),
                    float(expansion["episode_cosine"]),
                )
                for expansion in paragraph_expansions
            ]
            atomic_ranking: list[tuple[int, float]] = []
            atomic_seen: set[int] = set()
            channels = [
                episode_ranked,
                sparse_episode_ranked,
                source_episode_ranking,
                fused_episode_ranking,
            ]
            if (
                self.config.retrieval.paragraph_seed_enabled
                and not self.config.retrieval.paragraph_recall_only
            ):
                channels.insert(3, paragraph_episode_ranking)
            maximum_channel_length = max((len(channel) for channel in channels), default=0)
            for rank_index in range(maximum_channel_length):
                for channel in channels:
                    if rank_index >= len(channel):
                        continue
                    node_id, score = channel[rank_index]
                    if node_id not in atomic_seen:
                        atomic_seen.add(node_id)
                        atomic_ranking.append((node_id, score))
            atomic_episode_rankings.append(atomic_ranking)
        if episode_anchor_ids is not None and episode_rankings:
            remaining_rankings = atomic_episode_rankings
            if first_query_anchor_limit is not None:
                for node_id, _ in atomic_episode_rankings[0][:first_query_anchor_limit]:
                    if node_id not in episode_anchor_ids:
                        episode_anchor_ids.append(node_id)
                remaining_rankings = atomic_episode_rankings[1:]
            # Round-robin keeps every atomic evidence slot represented before a
            # single broad subquery can consume the whole answer budget.
            for rank_index in range(
                self.config.retrieval.answer_anchor_episodes_per_query
            ):
                for ranking in remaining_rankings:
                    if rank_index >= len(ranking):
                        continue
                    node_id = ranking[rank_index][0]
                    if node_id not in episode_anchor_ids:
                        episode_anchor_ids.append(node_id)
        if (
            self.config.retrieval.paragraph_seed_enabled
            and self.config.retrieval.paragraph_recall_only
        ):
            # Add only genuinely missing Episode seeds.  Existing baseline
            # scores and atomic anchor order remain byte-for-byte unaffected.
            for key, score in paragraph_only_scores.items():
                if key in fused_scores:
                    continue
                fused_scores[key] = score
                best_cosines[key] = paragraph_only_cosines[key]
        maximum = max(fused_scores.values(), default=1.0)
        hits = [
            SearchHit(
                node_type,
                node_id,
                0.99 * (score / maximum)
                + 0.01
                * max(0.0, min(1.0, (best_cosines[key] + 1.0) / 2.0)),
            )
            for key, score in fused_scores.items()
            for node_type, node_id in (key,)
        ]
        rankings = {
            "episode": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in episode_rankings
            ],
            "concept": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in concept_rankings
            ],
            "concept_raw": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in concept_raw_rankings
            ],
            "concept_gate": concept_gate_audits,
            "paragraph": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in paragraph_rankings
            ],
            "paragraph_episode_expansion": paragraph_expansion_rankings,
            "sparse_episode": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in sparse_episode_rankings
            ],
            "sparse_source": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in sparse_source_rankings
            ],
            "sparse_source_episode_expansion": sparse_source_expansion_rankings,
            "fused_episode": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in fused_episode_rankings
            ],
            "atomic_episode": [
                [
                    {"id": int(node_id), "score": float(score)}
                    for node_id, score in ranking
                ]
                for ranking in atomic_episode_rankings
            ],
        }
        if getattr(self, "logger", None) and any(paragraph_rankings):
            self.logger.emit(
                "paragraph_retrieval",
                queries=queries,
                paragraph_rankings=rankings["paragraph"],
                episode_expansions=paragraph_expansion_rankings,
            )
        if getattr(self, "logger", None) and any(sparse_episode_rankings):
            self.logger.emit(
                "sparse_retrieval",
                queries=queries,
                episode_rankings=rankings["sparse_episode"],
                source_rankings=rankings["sparse_source"],
                source_episode_expansions=sparse_source_expansion_rankings,
                fused_episode_rankings=rankings["fused_episode"],
            )
        return hits, rankings

    @staticmethod
    def _merge_hits(*groups: list[SearchHit]) -> list[SearchHit]:
        best: dict[tuple[str, int], SearchHit] = {}
        for hit in (item for group in groups for item in group):
            key = (hit.node_type, hit.node_id)
            if key not in best or hit.score > best[key].score:
                best[key] = hit
        return sorted(best.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _interleave_anchor_ids(*groups: list[int]) -> list[int]:
        """Interleave retrieval phases so later-hop anchors are not tail-dropped."""
        merged: list[int] = []
        seen: set[int] = set()
        maximum = max((len(group) for group in groups), default=0)
        for index in range(maximum):
            for group in groups:
                if index >= len(group):
                    continue
                node_id = int(group[index])
                if node_id not in seen:
                    seen.add(node_id)
                    merged.append(node_id)
        return merged

    def _seed_hits(
        self,
        question: str,
        intent: QueryIntent,
        episode_anchor_ids: list[int] | None = None,
    ) -> list[SearchHit]:
        queries = list(dict.fromkeys([question, *intent.search_queries]))
        hits, _cue_ids, _cue_entries, _rankings = self._vector_seed_hits_with_cues(
            queries,
            episode_anchor_ids,
            self.config.retrieval.answer_whole_question_anchor_episodes,
            question,
        )
        for entity in intent.target_entities:
            for row in self.concepts.find_by_alias(entity):
                concept_id = int(row["canonical_concept_id"] or row["id"])
                hits.append(SearchHit("concept", concept_id, 1.0))
        return self._merge_hits(hits)

    def _replay_configuration(self) -> dict[str, int | float | bool]:
        retrieval = self.config.retrieval
        configuration: dict[str, int | float | bool] = {
            "embedding_dimension": self.config.model.embedding_dimension,
            "episode_top_k": retrieval.episode_top_k,
            "concept_top_k": retrieval.concept_top_k,
            "graph_beam_width": retrieval.graph_beam_width,
            "graph_max_hops": retrieval.graph_max_hops,
            "candidate_limit": retrieval.candidate_limit,
            "answer_episode_limit": retrieval.answer_episode_limit,
            "answer_concept_limit": retrieval.answer_concept_limit,
            "answer_path_limit": retrieval.answer_path_limit,
            "growth_persist_only_used": retrieval.growth_persist_only_used,
            "growth_counterfactual_utility_enabled": (
                retrieval.growth_counterfactual_utility_enabled
            ),
            "growth_staging_enabled": retrieval.growth_staging_enabled,
            "source_key_cohort_enabled": retrieval.source_key_cohort_enabled,
            "source_key_cohort_min_anchor_hits": (
                retrieval.source_key_cohort_min_anchor_hits
            ),
            "source_key_cohort_max_keys": retrieval.source_key_cohort_max_keys,
            "source_key_cohort_max_episodes_per_key": (
                retrieval.source_key_cohort_max_episodes_per_key
            ),
            "source_key_cohort_total_limit": (
                retrieval.source_key_cohort_total_limit
            ),
            "source_key_cohort_score_ratio": (
                retrieval.source_key_cohort_score_ratio
            ),
            "learned_bridge_slots": retrieval.learned_bridge_slots,
            "learned_bridge_min_query_relevance": (
                retrieval.learned_bridge_min_query_relevance
            ),
            "learned_bridge_duplicate_threshold": (
                retrieval.learned_bridge_duplicate_threshold
            ),
            "answer_whole_question_anchor_episodes": (
                retrieval.answer_whole_question_anchor_episodes
            ),
            "answer_anchor_episodes_per_query": (
                retrieval.answer_anchor_episodes_per_query
            ),
            "rerank_atomic_query_limit": retrieval.rerank_atomic_query_limit,
            "rerank_atomic_floor_enabled": retrieval.rerank_atomic_floor_enabled,
            "rerank_atomic_floor_query_limit": (
                retrieval.rerank_atomic_floor_query_limit
            ),
            "rerank_atomic_floor_per_query": (
                retrieval.rerank_atomic_floor_per_query
            ),
            "rerank_atomic_floor_total_limit": (
                retrieval.rerank_atomic_floor_total_limit
            ),
            "rerank_constraint_floor_per_query": (
                retrieval.rerank_constraint_floor_per_query
            ),
            "rerank_constraint_floor_total_limit": (
                retrieval.rerank_constraint_floor_total_limit
            ),
            "rerank_answer_slot_neighbor_radius": (
                retrieval.rerank_answer_slot_neighbor_radius
            ),
            "rerank_answer_slot_neighbor_total_limit": (
                retrieval.rerank_answer_slot_neighbor_total_limit
            ),
            "rerank_constraint_candidate_per_query": (
                retrieval.rerank_constraint_candidate_per_query
            ),
            "rerank_constraint_candidate_total_limit": (
                retrieval.rerank_constraint_candidate_total_limit
            ),
            "rerank_question_sparse_floor_limit": (
                retrieval.rerank_question_sparse_floor_limit
            ),
            "rerank_combined_floor_limit": (
                retrieval.rerank_combined_floor_limit
            ),
        }
        if self.paragraph_retrieval_enabled:
            configuration.update(
                {
                    "paragraph_enabled": True,
                    "paragraph_top_k": retrieval.paragraph_top_k,
                    "paragraph_episode_expansion_limit": (
                        retrieval.paragraph_episode_expansion_limit
                    ),
                    "paragraph_rrf_weight": retrieval.paragraph_rrf_weight,
                    "paragraph_seed_enabled": retrieval.paragraph_seed_enabled,
                    "paragraph_rerank_context_enabled": (
                        retrieval.paragraph_rerank_context_enabled
                    ),
                    "paragraph_rerank_context_per_source": (
                        retrieval.paragraph_rerank_context_per_source
                    ),
                    "paragraph_rerank_context_chars": (
                        retrieval.paragraph_rerank_context_chars
                    ),
                }
            )
        if self.sparse_retrieval_enabled:
            configuration.update(
                {
                    "sparse_enabled": True,
                    "sparse_episode_top_k": retrieval.sparse_episode_top_k,
                    "sparse_source_top_k": retrieval.sparse_source_top_k,
                    "sparse_source_episode_expansion_limit": (
                        retrieval.sparse_source_episode_expansion_limit
                    ),
                    "sparse_episode_rrf_weight": retrieval.sparse_episode_rrf_weight,
                    "sparse_source_rrf_weight": retrieval.sparse_source_rrf_weight,
                }
            )
        if retrieval.rerank_enabled:
            configuration.update(
                {
                    "rerank_enabled": True,
                    "rerank_candidate_limit": retrieval.rerank_candidate_limit,
                    "rerank_precompression_limit": (
                        retrieval.rerank_precompression_limit
                    ),
                    "rerank_shortlist_limit": retrieval.rerank_shortlist_limit,
                    "rerank_coverage_audit_enabled": (
                        retrieval.rerank_coverage_audit_enabled
                    ),
                    "rerank_audit_enabled": retrieval.rerank_audit_enabled,
                }
            )
        if retrieval.association_cue_enabled:
            configuration.update(
                {
                    "association_cue_enabled": True,
                    "association_cue_top_k": retrieval.association_cue_top_k,
                    "association_cue_min_similarity": (
                        retrieval.association_cue_min_similarity
                    ),
                    "association_cue_rrf_weight": (
                        retrieval.association_cue_rrf_weight
                    ),
                }
            )
            if retrieval.association_cue_semantic_gate_enabled:
                configuration.update(
                    {
                        "association_cue_semantic_gate_enabled": True,
                        "association_cue_semantic_gate_max_selected": (
                            retrieval.association_cue_semantic_gate_max_selected
                        ),
                    }
                )
            if retrieval.association_cue_fast_path_enabled:
                configuration.update(
                    {
                        "association_cue_fast_path_enabled": True,
                        "association_cue_fast_path_min_similarity": (
                            retrieval.association_cue_fast_path_min_similarity
                        ),
                        "association_cue_fast_path_min_confidence": (
                            retrieval.association_cue_fast_path_min_confidence
                        ),
                        "association_cue_fast_path_min_margin": (
                            retrieval.association_cue_fast_path_min_margin
                        ),
                        "association_cue_fast_path_max_edges": (
                            retrieval.association_cue_fast_path_max_edges
                        ),
                    }
                )
        if retrieval.contextual_association_enabled:
            configuration.update(
                {
                    "contextual_association_enabled": True,
                    "contextual_association_shadow": retrieval.contextual_association_shadow,
                    "contextual_promotion_enabled": retrieval.contextual_promotion_enabled,
                    "contextual_context_top_k": retrieval.contextual_context_top_k,
                    "contextual_need_top_k": retrieval.contextual_need_top_k,
                    "contextual_edge_top_k": retrieval.contextual_edge_top_k,
                    "contextual_context_threshold": retrieval.contextual_context_threshold,
                    "contextual_need_threshold": retrieval.contextual_need_threshold,
                    "contextual_combine_mode": retrieval.contextual_combine_mode,
                    "contextual_endpoint_limit_light": retrieval.contextual_endpoint_limit_light,
                    "contextual_endpoint_limit_standard": retrieval.contextual_endpoint_limit_standard,
                    "contextual_endpoint_limit_deep": retrieval.contextual_endpoint_limit_deep,
                }
            )
        return configuration

    @staticmethod
    def _serialize_hits(hits: list[SearchHit]) -> list[dict]:
        return [
            {
                "node_type": hit.node_type,
                "node_id": int(hit.node_id),
                "score": float(hit.score),
            }
            for hit in hits
        ]

    def attach_association_cues(self, bundle: dict) -> dict:
        """Attach deterministic cue matches to an already frozen base plan.

        The reranker plan stays the no-cue baseline. Replay can then add or
        remove only relation-derived seed endpoints without another model call.
        """
        if not self.association_cue_retrieval_enabled:
            raise ValueError("association cue retrieval is not active")
        result = deepcopy(bundle)
        initial_queries = [str(value) for value in result["initial_queries"]]
        initial_matrix = np.asarray(
            result["initial_query_embeddings_float32"], dtype=np.float32
        )
        entries = self._association_cue_entries_from_matrix(
            initial_queries, initial_matrix
        )
        followup_queries = [
            str(value) for value in result.get("followup_queries", [])
        ]
        if followup_queries:
            followup_matrix = np.asarray(
                result["followup_query_embeddings_float32"], dtype=np.float32
            )
            entries.extend(
                self._association_cue_entries_from_matrix(
                    followup_queries, followup_matrix
                )
            )
        best_by_id: dict[int, dict] = {}
        for entry in entries:
            association_id = int(entry["association_id"])
            previous = best_by_id.get(association_id)
            if previous is None or float(entry["cosine"]) > float(
                previous["cosine"]
            ):
                best_by_id[association_id] = dict(entry)
        entries = sorted(
            best_by_id.values(),
            key=lambda item: float(item["cosine"]),
            reverse=True,
        )
        entries, gate_decisions = self._semantic_gate_association_cues(
            str(result["question"]), entries
        )
        base_seed_payload = list(
            result.get("base_final_seed_hits", result["final_seed_hits"])
        )
        base_hits = [
            SearchHit(
                str(item["node_type"]),
                int(item["node_id"]),
                float(item["score"]),
            )
            for item in base_seed_payload
        ]
        cue_hits, cue_ids, active_entries = self._active_association_cue_hits(
            entries
        )
        result["version"] = 4
        result["base_final_seed_hits"] = base_seed_payload
        result["association_cue_entries"] = active_entries
        result["association_cue_gate_decisions"] = gate_decisions
        result["association_cue_association_ids"] = cue_ids
        result["final_seed_hits"] = self._serialize_hits(
            self._merge_hits(base_hits, cue_hits)
        )
        result["configuration"] = self._replay_configuration()
        return result

    def build_replay_bundle(self, question: str) -> dict:
        """Plan Q2 once and freeze every stochastic/vector input for replay."""
        intent = self._parse_intent(question)
        initial_queries = list(dict.fromkeys([question, *intent.search_queries]))
        initial_matrix = np.asarray(
            self.model.embed(initial_queries), dtype=np.float32
        )
        initial_anchor_ids: list[int] = []
        initial_hits, initial_rankings = self._vector_seed_hits_from_matrix(
            initial_queries,
            initial_matrix,
            initial_anchor_ids,
            self.config.retrieval.answer_whole_question_anchor_episodes,
        )
        alias_hits: list[SearchHit] = []
        for entity in intent.target_entities:
            for row in self.concepts.find_by_alias(entity):
                alias_hits.append(
                    SearchHit(
                        "concept",
                        int(row["canonical_concept_id"] or row["id"]),
                        1.0,
                    )
                )
        seeds = self._merge_hits(initial_hits, alias_hits)
        traversed, _ = self.traverser.expand(
            seeds,
            self.config.retrieval.graph_beam_width,
            self.config.retrieval.graph_max_hops,
        )
        traversed = traversed[: self.config.retrieval.candidate_limit]
        should_plan_followup, _reason = self._followup_planning_decision(
            question,
            intent,
        )
        followup_queries = (
            self._plan_followup_queries(question, intent, traversed)
            if should_plan_followup
            else []
        )
        followup_matrix = np.empty(
            (0, self.config.model.embedding_dimension), dtype=np.float32
        )
        followup_rankings: dict[str, list] = {"episode": [], "concept": []}
        followup_hits: list[SearchHit] = []
        followup_anchor_ids: list[int] = []
        if followup_queries:
            followup_matrix = np.asarray(
                self.model.embed(followup_queries), dtype=np.float32
            )
            followup_hits, followup_rankings = self._vector_seed_hits_from_matrix(
                followup_queries,
                followup_matrix,
                followup_anchor_ids,
            )
            seeds = self._merge_hits(seeds, followup_hits)
        episode_anchor_ids = self._interleave_anchor_ids(
            followup_anchor_ids,
            initial_anchor_ids,
        ) if followup_queries else list(initial_anchor_ids)
        final_traversed, _final_paths = self.traverser.expand(
            seeds,
            self.config.retrieval.graph_beam_width,
            self.config.retrieval.graph_max_hops,
        )
        final_traversed = final_traversed[: self.config.retrieval.candidate_limit]
        cohort_hits, source_key_cohort_trace = self._source_key_cohort_hits(
            initial_anchor_ids,
            final_traversed,
        )
        if cohort_hits:
            seeds = self._merge_hits(seeds, cohort_hits)
            final_traversed, _final_paths = self.traverser.expand(
                seeds,
                self.config.retrieval.graph_beam_width,
                self.config.retrieval.graph_max_hops,
            )
            final_traversed = final_traversed[
                : self.config.retrieval.candidate_limit
            ]
        candidate_episodes, _candidate_concepts = self._materialize_nodes(
            final_traversed
        )
        paragraph_context_by_source = self._paragraph_context_by_source(
            [
                *initial_rankings.get("paragraph", []),
                *followup_rankings.get("paragraph", []),
            ]
        )
        evidence_floor_trace = self._hybrid_evidence_floor_trace(
            question,
            initial_rankings,
            followup_rankings,
            initial_queries,
            followup_queries,
        )
        constraint_candidate_ids = self._constraint_candidate_ids(
            question,
            initial_queries,
            initial_rankings,
            followup_queries,
            followup_rankings,
        )
        reranked_episode_ids, rerank_trace = self._rerank_answer_episodes(
            question,
            intent,
            list(dict.fromkeys([*initial_queries, *followup_queries])),
            candidate_episodes,
            min(self.config.retrieval.answer_episode_limit, len(candidate_episodes)),
            [*constraint_candidate_ids, *episode_anchor_ids],
            paragraph_context_by_source,
            required_candidate_ids=evidence_floor_trace["selected_episode_ids"],
        )
        evidence_floor_trace = self._admit_evidence_floor_trace(
            evidence_floor_trace,
            rerank_trace.get("required_evidence_floor_ids", []),
        )
        rerank_trace["deterministic_evidence_floor"] = evidence_floor_trace
        rerank_trace["constraint_candidate_ids"] = constraint_candidate_ids
        bundle = {
            "version": (
                3
                if self.sparse_retrieval_enabled
                else (2 if self.paragraph_retrieval_enabled else 1)
            ),
            "question": question,
            "intent": asdict(intent),
            "initial_queries": initial_queries,
            "followup_queries": followup_queries,
            "initial_query_embeddings_float32": initial_matrix.tolist(),
            "followup_query_embeddings_float32": followup_matrix.tolist(),
            "initial_rankings": initial_rankings,
            "followup_rankings": followup_rankings,
            "initial_seed_hits": self._serialize_hits(
                self._merge_hits(initial_hits, alias_hits)
            ),
            "followup_seed_hits": self._serialize_hits(followup_hits),
            "final_seed_hits": self._serialize_hits(seeds),
            "initial_episode_anchor_ids": initial_anchor_ids,
            "followup_episode_anchor_ids": followup_anchor_ids,
            "episode_anchor_ids": episode_anchor_ids,
            "source_key_cohort": source_key_cohort_trace,
            "reranked_episode_ids": reranked_episode_ids,
            "rerank_trace": rerank_trace,
            "configuration": self._replay_configuration(),
        }
        if self.association_cue_retrieval_enabled:
            return self.attach_association_cues(bundle)
        return bundle

    def build_query_plan(self, question: str) -> dict:
        """Build a serializable plan shared by all comparable query arms.

        The historical replay bundle already freezes intent parsing, follow-up
        planning, embeddings, dense/sparse rankings and the LLM rerank.  A
        query plan gives that artifact a stable identity so the normal answer
        and growth pipeline can consume it, rather than limiting it to offline
        retrieval probes.
        """
        plan = self.build_replay_bundle(question)
        plan["plan_schema"] = "frozen_query_plan_v1"
        canonical = json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        plan["plan_id"] = hashlib.sha256(canonical).hexdigest()
        return plan

    def _validate_query_plan(self, question: str, plan: dict) -> None:
        if plan.get("plan_schema") != "frozen_query_plan_v1":
            raise ValueError("unsupported frozen query plan schema")
        if str(plan.get("question", "")) != question:
            raise ValueError("frozen query plan question does not match query")
        expected_id = str(plan.get("plan_id", ""))
        payload = deepcopy(plan)
        payload.pop("plan_id", None)
        actual_id = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if not expected_id or expected_id != actual_id:
            raise ValueError("frozen query plan fingerprint is invalid")
        if plan.get("configuration") != self._replay_configuration():
            raise ValueError("frozen query plan configuration does not match engine")

    def replay_retrieval(self, bundle: dict) -> dict:
        """Replay graph expansion/evidence selection without model calls or writes."""
        if int(bundle.get("version", 0)) not in {1, 2, 3, 4}:
            raise ValueError("unsupported replay bundle version")
        if bundle.get("configuration") != self._replay_configuration():
            raise ValueError("replay bundle configuration does not match engine")
        question = str(bundle["question"])
        base_seed_payload = bundle.get(
            "base_final_seed_hits", bundle["final_seed_hits"]
        )
        seeds = [
            SearchHit(
                str(item["node_type"]),
                int(item["node_id"]),
                float(item["score"]),
            )
            for item in base_seed_payload
        ]
        active_cue_ids: list[int] = []
        active_cue_entries: list[dict] = []
        if self.config.retrieval.association_cue_enabled:
            cue_hits, active_cue_ids, active_cue_entries = (
                self._active_association_cue_hits(
                    list(bundle.get("association_cue_entries", []))
                )
            )
            seeds = self._merge_hits(seeds, cue_hits)
        traversed, paths = self.traverser.expand(
            seeds,
            self.config.retrieval.graph_beam_width,
            self.config.retrieval.graph_max_hops,
        )
        cue_scores = {
            int(item["association_id"]): float(item.get("cosine", 0.0))
            for item in active_cue_entries
        }
        paths.extend(
            self._explicit_association_paths(active_cue_ids, cue_scores)
        )
        traversed = self._truncate_traversed_nodes(
            traversed,
            self.config.retrieval.candidate_limit,
            self._association_cue_fast_endpoint_keys(active_cue_entries),
        )
        preferred_episode_ids = [
            int(value)
            for value in (
                bundle.get("reranked_episode_ids")
                or bundle.get("episode_anchor_ids", [])
            )
        ]
        late_nodes, late_paths = self._late_learned_bridge_closure(
            question,
            preferred_episode_ids,
            traversed,
        )
        if late_nodes:
            traversed = [*traversed, *late_nodes]
            paths.extend(late_paths)
        candidate_episode_ids = [
            item.node_id for item in traversed if item.node_type == "episode"
        ]
        episodes, concepts = self._materialize_nodes(traversed, include_sources=True)
        episode_limit = min(self.config.retrieval.answer_episode_limit, len(episodes))
        selected_episodes, answer_paths = self._select_answer_evidence(
            episodes,
            paths,
            episode_limit,
            self.config.retrieval.answer_path_limit,
            set(active_cue_ids),
            question=question,
            preferred_episode_ids=[
                *preferred_episode_ids
            ],
            learned_bridge_slots=self.config.retrieval.learned_bridge_slots,
            learned_bridge_min_query_relevance=(
                self.config.retrieval.learned_bridge_min_query_relevance
            ),
            learned_bridge_duplicate_threshold=(
                self.config.retrieval.learned_bridge_duplicate_threshold
            ),
            coverage_groups=(
                bundle.get("rerank_trace", {})
                .get("merged_coverage", {})
                .get("coverage", [])
            ),
            preferred_association_scores=cue_scores,
        )
        chronology = self.chronology.order(
            [int(item["id"]) for item in selected_episodes]
        )
        episode_map = {int(item["id"]): item for item in selected_episodes}
        ordered_episodes = [
            episode_map[node_id]
            for node_id in chronology.ordered_ids
            if node_id in episode_map
        ]
        seed_episode_ids = {
            int(item["node_id"])
            for item in base_seed_payload
            if item["node_type"] == "episode"
        }
        cue_episode_ids = {
            int(endpoint[1])
            for entry in active_cue_entries
            for endpoint in entry.get("endpoints", [])
            if isinstance(endpoint, (list, tuple))
            and len(endpoint) == 2
            and endpoint[0] == "episode"
        }
        return {
            "question": question,
            "intent": dict(bundle["intent"]),
            "followup_search_queries": list(bundle.get("followup_queries", [])),
            "atomic_anchor_episode_ids": [
                int(value) for value in bundle.get("episode_anchor_ids", [])
            ],
            "reranked_episode_ids": [
                int(value) for value in bundle.get("reranked_episode_ids", [])
            ],
            "evidence_slot_trace": self._final_evidence_slot_trace(
                dict(bundle.get("rerank_trace", {})),
                [int(item["id"]) for item in ordered_episodes],
            ),
            "seed_episode_ids": sorted(seed_episode_ids),
            "candidate_episode_ids": candidate_episode_ids,
            "graph_added_episode_ids": [
                value for value in candidate_episode_ids if value not in seed_episode_ids
            ],
            "association_cue_ids": active_cue_ids,
            "association_cue_entries": active_cue_entries,
            "association_cue_added_episode_ids": sorted(
                cue_episode_ids.difference(seed_episode_ids)
            ),
            "episode_ids": [int(item["id"]) for item in ordered_episodes],
            "concept_ids": [
                int(item["id"])
                for item in concepts[: self.config.retrieval.answer_concept_limit]
            ],
            "association_ids": list(
                dict.fromkeys(
                    int(item["association_id"])
                    for item in answer_paths
                    if "association_id" in item
                )
            ),
            "chronology_notes": chronology.notes,
            "evidence_episodes": [
                {
                    key: item[key]
                    for key in (
                        "id",
                        "score",
                        "text",
                        "participants",
                        "source_key",
                        "segment_index",
                        "story_time_text",
                        "story_order",
                        "timeline_scope",
                        "evidence_origin",
                        "epistemic_status",
                        "generation",
                        "epistemic_note",
                    )
                }
                for item in ordered_episodes
            ],
            "evidence_concepts": [
                {
                    key: item[key]
                    for key in ("id", "score", "canonical_name", "description")
                }
                for item in concepts[: self.config.retrieval.answer_concept_limit]
            ],
            "association_paths": answer_paths,
            "paragraph_retrieval_enabled": self.paragraph_retrieval_enabled,
            "sparse_retrieval_enabled": self.sparse_retrieval_enabled,
        }

    def _plan_followup_queries(
        self,
        question: str,
        intent: QueryIntent,
        traversed,
    ) -> list[str]:
        episodes, concepts = self._materialize_nodes(traversed)
        payload = self.model.chat_json(
            HOP_QUERY_SYSTEM,
            hop_query_prompt(
                question,
                asdict(intent),
                episodes[:30],
                concepts[:20],
            ),
        )
        if not isinstance(payload, dict):
            return []
        raw_queries = payload.get("followup_queries", [])
        if not isinstance(raw_queries, list):
            return []
        existing = {question, *intent.search_queries}
        followups: list[str] = []
        for raw_query in raw_queries:
            query = str(raw_query).strip()
            if query and query not in existing and query not in followups:
                followups.append(query)
        return followups[:8]

    def _followup_planning_decision(
        self,
        question: str,
        intent: QueryIntent,
    ) -> tuple[bool, str]:
        mode = str(
            self.config.retrieval.followup_planning_mode
        ).strip().casefold()
        if mode == "always":
            return True, "configured_always"
        if mode == "off":
            return False, "configured_off"
        if mode != "entity_resolved":
            raise ValueError(f"unknown followup planning mode: {mode}")
        if len(intent.search_queries) >= 12:
            return True, "high_slot_count_safety_net"
        if requires_entity_resolved_followup(question, intent):
            return True, "answer_dependent_reference_detected"
        return False, "intent_queries_already_name_each_relation"

    @staticmethod
    def _validated_rerank_ids(
        payload: dict | None,
        field: str,
        allowed_ids: set[int],
        limit: int,
    ) -> list[int]:
        if not isinstance(payload, dict):
            return []
        raw_ids = payload.get(field, [])
        if not isinstance(raw_ids, list):
            return []
        result: list[int] = []
        for raw_id in raw_ids:
            try:
                node_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if node_id in allowed_ids and node_id not in result:
                result.append(node_id)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _coverage_rerank_ids(
        payload: dict | None,
        allowed_ids: set[int],
        limit: int,
    ) -> list[int]:
        groups = QueryEngine._coverage_groups(payload, allowed_ids)
        result: list[int] = []
        maximum = max((len(group) for group in groups), default=0)
        for rank_index in range(maximum):
            for group in groups:
                if rank_index < len(group) and group[rank_index] not in result:
                    result.append(group[rank_index])
                    if len(result) >= limit:
                        return result
        return result

    @staticmethod
    def _coverage_groups(
        payload: dict | None,
        allowed_ids: set[int],
    ) -> list[list[int]]:
        if not isinstance(payload, dict):
            return []
        coverage = payload.get("coverage", [])
        if not isinstance(coverage, list):
            return []
        groups: list[list[int]] = []
        for item in coverage:
            if not isinstance(item, dict) or not isinstance(
                item.get("episode_ids"), list
            ):
                continue
            group: list[int] = []
            for raw_id in item["episode_ids"]:
                try:
                    node_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if node_id in allowed_ids and node_id not in group:
                    group.append(node_id)
            if group:
                if str(item.get("mode", "alternatives")).lower() == "joint":
                    groups.extend([[node_id] for node_id in group[:5]])
                else:
                    groups.append(group[:5])
        return groups

    @staticmethod
    def _merge_coverage_payloads(
        primary: dict | None,
        supplemental: dict | None,
    ) -> dict:
        """Interleave two independent passes and union evidence for equal slots."""
        entries: dict[str, dict] = {}
        positions: dict[str, int] = {}
        for pass_index, payload in enumerate((primary, supplemental)):
            if not isinstance(payload, dict) or not isinstance(
                payload.get("coverage"), list
            ):
                continue
            for item_index, item in enumerate(payload["coverage"]):
                if not isinstance(item, dict):
                    continue
                raw_ids = item.get("episode_ids", [])
                if not isinstance(raw_ids, list):
                    continue
                ids: list[int] = []
                for raw_id in raw_ids:
                    try:
                        node_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    if node_id not in ids:
                        ids.append(node_id)
                normalized = {
                    "query": str(item.get("query", "")).strip(),
                    "mode": (
                        "joint"
                        if str(item.get("mode", "alternatives")).lower()
                        == "joint"
                        else "alternatives"
                    ),
                    "episode_ids": ids,
                    "reason": str(item.get("reason", "")),
                }
                key = re.sub(
                    r"[（(][^）)]*[）)]",
                    "",
                    normalized["query"],
                )
                key = re.sub(r"\s+", "", key).casefold()
                position = item_index * 2 + pass_index
                if key not in entries:
                    entries[key] = normalized
                    positions[key] = position
                    continue
                existing = entries[key]
                if (
                    existing["mode"] != "joint"
                    or normalized["mode"] != "joint"
                ):
                    existing["mode"] = "alternatives"
                for node_id in normalized["episode_ids"]:
                    if node_id not in existing["episode_ids"]:
                        existing["episode_ids"].append(node_id)
                if (
                    normalized["reason"]
                    and normalized["reason"] not in existing["reason"]
                ):
                    existing["reason"] = (
                        f"{existing['reason']} | 独立侦察：{normalized['reason']}"
                    ).strip(" |")
                positions[key] = min(positions[key], position)
        coverage = [
            entries[key]
            for key in sorted(entries, key=lambda item: positions[item])
        ]
        missing: list[str] = []
        for payload in (primary, supplemental):
            if not isinstance(payload, dict) or not isinstance(
                payload.get("missing_aspects"), list
            ):
                continue
            for value in payload["missing_aspects"]:
                item = str(value).strip()
                if item and item not in missing:
                    missing.append(item)
        return {"coverage": coverage, "missing_aspects": missing}

    @staticmethod
    def _enforce_coverage_selection(
        payload: dict | None,
        selected_ids: list[int],
        allowed_ids: set[int],
        limit: int,
    ) -> tuple[list[int], list[dict[str, int]]]:
        """Keep one candidate for each declared evidence slot when possible."""
        if limit <= 0:
            return [], []
        result = [
            node_id
            for node_id in dict.fromkeys(int(value) for value in selected_ids)
            if node_id in allowed_ids
        ][:limit]
        groups = QueryEngine._coverage_groups(payload, allowed_ids)

        # A Top-K list cannot represent more than K mutually disjoint slots.
        # Coverage order follows the user's question, so the first K slots win.
        protected_groups = groups[:limit]
        changes: list[dict[str, int]] = []
        for group_index, group in enumerate(protected_groups):
            if any(node_id in result for node_id in group):
                continue
            add_id = group[0]
            if len(result) < limit:
                result.append(add_id)
                changes.append({"remove_id": -1, "add_id": add_id})
                continue
            previous_groups = protected_groups[:group_index]
            replacement_index = None
            for index in range(len(result) - 1, -1, -1):
                trial = result[:index] + result[index + 1 :] + [add_id]
                if all(any(node_id in trial for node_id in prior) for prior in previous_groups):
                    replacement_index = index
                    break
            if replacement_index is None:
                continue
            remove_id = result[replacement_index]
            result[replacement_index] = add_id
            changes.append({"remove_id": remove_id, "add_id": add_id})
        return list(dict.fromkeys(result))[:limit], changes

    @staticmethod
    def _enforce_final_selection_constraints(
        selected_ids: list[int],
        required_ids: list[int],
        coverage_payload: dict | None,
        allowed_ids: set[int],
        coverage_allowed_ids: set[int],
        limit: int,
    ) -> tuple[list[int], list[dict[str, int]], list[dict[str, int]]]:
        """Apply cheap recall floors before the stronger coverage contract.

        Atomic floors deliberately trade precision for recall.  They must not
        get the final opportunity to evict a unique Episode chosen by the
        independent evidence coverage passes.
        """
        with_floor, floor_changes = QueryEngine._enforce_selection_floor(
            selected_ids,
            required_ids,
            allowed_ids,
            limit,
        )
        final_ids, coverage_changes = QueryEngine._enforce_coverage_selection(
            coverage_payload,
            with_floor,
            coverage_allowed_ids,
            limit,
        )
        return final_ids, floor_changes, coverage_changes

    @staticmethod
    def _apply_rerank_replacements(
        payload: dict | None,
        selected_ids: list[int],
        allowed_ids: set[int],
        limit: int,
    ) -> list[int]:
        """Execute valid replacement directives even if the model's final list drifts."""
        result = list(selected_ids)
        if not isinstance(payload, dict):
            return result
        replacements = payload.get("replacements", [])
        if not isinstance(replacements, list):
            return result
        for replacement in replacements:
            if not isinstance(replacement, dict):
                continue
            try:
                add_id = int(replacement.get("add_id"))
            except (TypeError, ValueError):
                continue
            if add_id not in allowed_ids or add_id in result:
                continue
            try:
                remove_id = int(replacement.get("remove_id"))
            except (TypeError, ValueError):
                remove_id = -1
            if remove_id in result:
                result[result.index(remove_id)] = add_id
            elif len(result) < limit:
                result.append(add_id)
            elif result:
                result[-1] = add_id
        return list(dict.fromkeys(result))[:limit]

    def _rerank_review_level(
        self,
        atomic_queries: list[str],
        initial: dict | None,
        allowed_ids: set[int],
        question: str = "",
    ) -> tuple[str, list[str]]:
        """Choose the cheapest evidence review path that fits visible risk."""
        mode = str(self.config.retrieval.rerank_review_mode).casefold()
        if mode == "strict":
            return "strict", ["configured_strict"]
        if mode == "lean":
            return "none", ["configured_lean"]
        if mode not in {"adaptive", "fast_adaptive"}:
            raise ValueError(f"unknown rerank review mode: {mode}")
        reasons: list[str] = []
        if not isinstance(initial, dict):
            return "strict", ["invalid_initial_payload"]
        # The reply path needs a bounded tail.  It trusts a complete initial
        # coverage map, but still spends one compressor call when the first
        # pass reports a visible gap, has too many atomic slots, or is too
        # thin.  Background/deep retrieval keeps the stronger independent
        # audit policy below.
        if mode == "fast_adaptive":
            missing = initial.get("missing_aspects", [])
            reported_missing = isinstance(missing, list) and any(
                str(item).strip() for item in missing
            )
            if reported_missing:
                reasons.append("initial_reported_missing_aspects")
            if len(atomic_queries) >= max(
                1,
                int(
                    self.config.retrieval.rerank_strict_atomic_query_threshold
                ),
            ):
                reasons.append("many_atomic_queries")
            coverage_group_count = len(
                self._coverage_groups(initial, allowed_ids)
            )
            if coverage_group_count <= max(
                0,
                int(
                    self.config.retrieval.rerank_compress_coverage_group_threshold
                ),
            ):
                reasons.append("thin_initial_coverage")
            # A compressor receives the same evidence shortlist.  A declared
            # source gap alone cannot be repaired by asking that model to
            # compress the list again; preserve it as answer-time uncertainty.
            # Compression is still useful for a structurally thin first pass
            # or an unusually large set of genuinely independent slots.
            if any(
                reason in {"many_atomic_queries", "thin_initial_coverage"}
                for reason in reasons
            ):
                return "compress", reasons
            if reported_missing:
                return "none", [
                    *reasons,
                    "reported_gap_preserved_as_uncertainty",
                ]
            return "none", ["initial_coverage_sufficient"]
        if self._requires_strict_evidence_review(question):
            reasons.append("high_risk_contrast_causality_or_projection")
        missing = initial.get("missing_aspects", [])
        if isinstance(missing, list) and any(str(item).strip() for item in missing):
            reasons.append("initial_reported_missing_aspects")
        if len(atomic_queries) >= max(
            1, int(self.config.retrieval.rerank_strict_atomic_query_threshold)
        ):
            reasons.append("many_atomic_queries")
        if reasons:
            return "strict", reasons
        coverage_group_count = len(self._coverage_groups(initial, allowed_ids))
        if coverage_group_count <= max(
            0,
            int(
                self.config.retrieval.rerank_compress_coverage_group_threshold
            ),
        ):
            return "compress", ["thin_initial_coverage"]
        return "none", ["initial_coverage_sufficient"]

    @staticmethod
    def _requires_strict_evidence_review(question: str) -> bool:
        explicit_risk = bool(
            re.search(
                r"为什么|为何|原因|因果|身份|属于|关系|"
                r"比较|对比|先后|第一次|最后|是否|有没有|"
                r"推断|证明|反驳|否定",
                question,
            )
        )
        # Two or more independently requested facts require coverage-aware
        # selection even when none of the old high-risk keywords is present.
        # This changes evidence protection, not workload routing.
        claim_markers = re.findall(
            r"谁|哪(?:位|个|些|一)|什么|为何|为什么|怎么|如何|是否|有没有",
            question,
        )
        return explicit_risk or len(claim_markers) >= 2

    def _atomic_evidence_floor_ids(
        self,
        question: str,
        atomic_rankings: list[list[dict]],
    ) -> list[int]:
        retrieval = self.config.retrieval
        if (
            not retrieval.rerank_atomic_floor_enabled
            or not self._requires_strict_evidence_review(question)
        ):
            return []
        query_limit = max(0, int(retrieval.rerank_atomic_floor_query_limit))
        per_query = max(0, int(retrieval.rerank_atomic_floor_per_query))
        total_limit = max(0, int(retrieval.rerank_atomic_floor_total_limit))
        rankings = atomic_rankings[:query_limit]
        selected: list[int] = []
        for rank_index in range(per_query):
            for ranking in rankings:
                if rank_index >= len(ranking):
                    continue
                episode_id = int(ranking[rank_index]["id"])
                if episode_id not in selected:
                    selected.append(episode_id)
                if len(selected) >= total_limit:
                    return selected
        return selected

    @staticmethod
    def _balanced_atomic_floor_rankings(
        initial_rankings: list[list[dict]],
        followup_rankings: list[list[dict]],
        query_limit: int,
    ) -> list[list[dict]]:
        """Reserve part of the floor for entity-resolved follow-up hops."""
        limit = max(0, int(query_limit))
        if not limit:
            return []
        initial = list(initial_rankings)
        followup = list(followup_rankings)
        if not initial or not followup:
            return (initial or followup)[:limit]

        followup_budget = min(len(followup), max(1, limit // 3))
        initial_budget = min(len(initial), limit - followup_budget)
        remaining = limit - initial_budget - followup_budget
        if remaining:
            extra_initial = min(len(initial) - initial_budget, remaining)
            initial_budget += extra_initial
            remaining -= extra_initial
        if remaining:
            followup_budget += min(
                len(followup) - followup_budget,
                remaining,
            )
        return [
            *initial[:initial_budget],
            *followup[:followup_budget],
        ]

    def _hybrid_evidence_floor_ids(
        self,
        question: str,
        initial_rankings: dict[str, list],
        followup_rankings: dict[str, list],
        initial_queries: list[str] | None = None,
        followup_queries: list[str] | None = None,
    ) -> list[int]:
        return self._hybrid_evidence_floor_trace(
            question,
            initial_rankings,
            followup_rankings,
            initial_queries,
            followup_queries,
        )["selected_episode_ids"]

    @staticmethod
    def _is_constraint_candidate_query(query: str) -> bool:
        normalized = str(query).strip()
        return normalized.startswith(
            ("__constraint_slot__ ", "__answer_slot__ ")
        )

    def _constraint_candidate_ids(
        self,
        question: str,
        initial_queries: list[str],
        initial_rankings: dict[str, list],
        followup_queries: list[str],
        followup_rankings: dict[str, list],
    ) -> list[int]:
        """Reserve rerank admission for late causal/identity subproblems.

        These IDs are candidate-only: the evidence reranker must still decide
        whether they support the question. Follow-up queries are inspected
        first because they are produced after first-hop evidence has made the
        missing relation more specific.
        """

        has_explicit_slot = any(
            str(query).startswith(("__constraint_slot__ ", "__answer_slot__ "))
            for query in [
                *(initial_queries or []),
                *(followup_queries or []),
            ]
        )
        if (
            not self._requires_strict_evidence_review(question)
            and not has_explicit_slot
        ):
            return []
        per_query = max(
            0,
            int(self.config.retrieval.rerank_constraint_candidate_per_query),
        )
        total_limit = max(
            0,
            int(self.config.retrieval.rerank_constraint_candidate_total_limit),
        )
        if not per_query or not total_limit:
            return []

        selected: list[int] = []
        groups = (
            (followup_queries, followup_rankings),
            (initial_queries, initial_rankings),
        )
        for queries, rankings in groups:
            fused = rankings.get("fused_episode", [])
            atomic = rankings.get("atomic_episode", [])
            for index, query in enumerate(queries):
                if not self._is_constraint_candidate_query(query):
                    continue
                rows = (
                    fused[index]
                    if index < len(fused)
                    else atomic[index]
                    if index < len(atomic)
                    else []
                )
                for item in rows[:per_query]:
                    episode_id = int(item["id"])
                    if episode_id not in selected:
                        selected.append(episode_id)
                    if len(selected) >= total_limit:
                        return selected
        return selected

    def _hybrid_evidence_floor_trace(
        self,
        question: str,
        initial_rankings: dict[str, list],
        followup_rankings: dict[str, list],
        initial_queries: list[str] | None = None,
        followup_queries: list[str] | None = None,
    ) -> dict:
        retrieval = self.config.retrieval
        has_explicit_slot = any(
            str(query).startswith(("__constraint_slot__ ", "__answer_slot__ "))
            for query in [
                *(initial_queries or []),
                *(followup_queries or []),
            ]
        )
        if (
            not self._requires_strict_evidence_review(question)
            and not has_explicit_slot
        ):
            return {
                "enabled": False,
                "constraint_slots": [],
                "atomic_slots": [],
                "whole_question_sparse_ids": [],
                "atomic_floor_ids": [],
                "selected_episode_ids": [],
            }
        sparse_limit = max(
            0, int(retrieval.rerank_question_sparse_floor_limit)
        )
        sparse_rankings = initial_rankings.get("sparse_episode", [])
        whole_question_sparse = sparse_rankings[0] if sparse_rankings else []
        sparse_ids = [
            int(item["id"])
            for item in whole_question_sparse[:sparse_limit]
        ]
        atomic_floor_rankings = self._balanced_atomic_floor_rankings(
            initial_rankings.get("atomic_episode", []),
            followup_rankings.get("atomic_episode", []),
            int(retrieval.rerank_atomic_floor_query_limit),
        )
        atomic_ids = self._atomic_evidence_floor_ids(
            question,
            atomic_floor_rankings,
        )
        query_limit = max(0, int(retrieval.rerank_atomic_floor_query_limit))
        per_atomic_query = max(
            0, int(retrieval.rerank_atomic_floor_per_query)
        )
        initial_pairs = list(
            zip(
                list(initial_queries or []),
                initial_rankings.get("atomic_episode", []),
            )
        )
        followup_pairs = list(
            zip(
                list(followup_queries or []),
                followup_rankings.get("atomic_episode", []),
            )
        )
        if initial_pairs and followup_pairs and query_limit:
            followup_budget = min(
                len(followup_pairs), max(1, query_limit // 3)
            )
            initial_budget = min(
                len(initial_pairs), query_limit - followup_budget
            )
            remaining = query_limit - initial_budget - followup_budget
            if remaining:
                extra_initial = min(
                    len(initial_pairs) - initial_budget, remaining
                )
                initial_budget += extra_initial
                remaining -= extra_initial
            if remaining:
                followup_budget += min(
                    len(followup_pairs) - followup_budget, remaining
                )
            atomic_pairs = [
                *initial_pairs[:initial_budget],
                *followup_pairs[:followup_budget],
            ]
        else:
            atomic_pairs = (initial_pairs or followup_pairs)[:query_limit]
        atomic_id_set = set(atomic_ids)
        atomic_slots = [
            {
                "query": str(query),
                "candidate_episode_ids": [
                    int(item["id"])
                    for item in ranking[:per_atomic_query]
                ],
                "floor_episode_ids": [
                    int(item["id"])
                    for item in ranking[:per_atomic_query]
                    if int(item["id"]) in atomic_id_set
                ],
            }
            for query, ranking in atomic_pairs
        ]
        constraint_slots: list[dict] = []
        constraint_ids: list[int] = []
        per_query = max(
            0, int(retrieval.rerank_constraint_floor_per_query)
        )
        constraint_total = max(
            0, int(retrieval.rerank_constraint_floor_total_limit)
        )
        query_ranking_groups = (
            (
                list(initial_queries or []),
                initial_rankings,
            ),
            (
                list(followup_queries or []),
                followup_rankings,
            ),
        )
        active_groups = (
            query_ranking_groups if per_query and constraint_total else ()
        )
        slot_candidates: list[tuple[int, int, str, list[dict]]] = []
        for group_index, (queries, rankings) in enumerate(active_groups):
            fused = rankings.get("fused_episode", [])
            atomic = rankings.get("atomic_episode", [])
            sparse = rankings.get("sparse_episode", [])
            for index, query in enumerate(queries):
                if (
                    not query.startswith(
                        ("__constraint_slot__ ", "__answer_slot__ ")
                    )
                ):
                    continue
                rows = (
                    fused[index]
                    if index < len(fused) and fused[index]
                    else atomic[index]
                    if index < len(atomic) and atomic[index]
                    else sparse[index]
                    if index < len(sparse)
                    else []
                )
                slot_candidates.append(
                    (
                        0 if query.startswith("__answer_slot__ ") else 1,
                        group_index,
                        query,
                        rows,
                    )
                )
        for _priority, _group_index, query, rows in sorted(
            slot_candidates, key=lambda item: (item[0], item[1])
        ):
                candidates = [int(item["id"]) for item in rows[:per_query]]
                admitted: list[int] = []
                for node_id in candidates:
                    if node_id not in constraint_ids:
                        constraint_ids.append(node_id)
                        admitted.append(node_id)
                    if len(constraint_ids) >= constraint_total:
                        break
                constraint_slots.append(
                    {
                        "query": query,
                        "candidate_episode_ids": candidates,
                        "floor_episode_ids": admitted,
                    }
                )
                if len(constraint_ids) >= constraint_total:
                    break
        combined_limit = max(0, int(retrieval.rerank_combined_floor_limit))
        # Whole-question sparse results are high-recall candidates, not proof
        # that every top lexical match belongs in the final answer.  Hard-floor
        # only the bounded typed/atomic slots and leave the remaining budget to
        # the evidence selector.
        selected = list(
            dict.fromkeys([*constraint_ids, *atomic_ids])
        )[:combined_limit]
        return {
            "enabled": True,
            "constraint_slots": constraint_slots,
            "atomic_slots": atomic_slots,
            "whole_question_sparse_ids": sparse_ids,
            "atomic_floor_ids": atomic_ids,
            "selected_episode_ids": selected,
        }

    @staticmethod
    def _final_evidence_slot_trace(
        rerank_trace: dict,
        final_episode_ids: list[int],
    ) -> dict:
        """Explain which deterministic/LLM evidence slots reached the answer."""
        final_ids = {int(value) for value in final_episode_ids}
        deterministic = deepcopy(
            rerank_trace.get("deterministic_evidence_floor", {})
        )
        for slot in deterministic.get("constraint_slots", []):
            floor_ids = [
                int(value) for value in slot.get("floor_episode_ids", [])
            ]
            slot["final_matched_episode_ids"] = [
                value for value in floor_ids if value in final_ids
            ]
            slot["satisfied"] = bool(slot["final_matched_episode_ids"])
        for slot in deterministic.get("atomic_slots", []):
            floor_ids = [
                int(value) for value in slot.get("floor_episode_ids", [])
            ]
            slot["final_matched_episode_ids"] = [
                value for value in floor_ids if value in final_ids
            ]
            slot["satisfied"] = bool(slot["final_matched_episode_ids"])
        selected_floor_ids = [
            int(value)
            for value in deterministic.get("selected_episode_ids", [])
        ]
        deterministic["final_matched_episode_ids"] = [
            value for value in selected_floor_ids if value in final_ids
        ]
        deterministic["missing_floor_episode_ids"] = [
            value for value in selected_floor_ids if value not in final_ids
        ]
        coverage_slots: list[dict] = []
        coverage = (
            rerank_trace.get("merged_coverage", {}).get("coverage", [])
        )
        coverage_items = coverage if isinstance(coverage, list) else []
        for item in coverage_items:
            if not isinstance(item, dict):
                continue
            candidate_ids = [
                int(value) for value in item.get("episode_ids", [])
            ]
            matched = [value for value in candidate_ids if value in final_ids]
            coverage_slots.append(
                {
                    "query": str(item.get("query", "")),
                    "mode": str(item.get("mode", "alternatives")),
                    "candidate_episode_ids": candidate_ids,
                    "final_matched_episode_ids": matched,
                    "satisfied": bool(matched),
                }
            )
        return {
            "final_episode_ids": sorted(final_ids),
            "deterministic": deterministic,
            "coverage_slots": coverage_slots,
        }

    @staticmethod
    def _admit_evidence_floor_trace(
        floor_trace: dict,
        admitted_episode_ids: list[int],
    ) -> dict:
        """Separate planned floor hits from rows admitted to rerank candidates."""
        result = deepcopy(floor_trace)
        admitted = {int(value) for value in admitted_episode_ids}
        planned = [
            int(value) for value in result.get("selected_episode_ids", [])
        ]
        result["planned_episode_ids"] = planned
        result["selected_episode_ids"] = [
            value for value in planned if value in admitted
        ]
        result["candidate_missing_episode_ids"] = [
            value for value in planned if value not in admitted
        ]
        for slot in result.get("constraint_slots", []):
            floor_ids = [
                int(value) for value in slot.get("floor_episode_ids", [])
            ]
            slot["planned_floor_episode_ids"] = floor_ids
            slot["floor_episode_ids"] = [
                value for value in floor_ids if value in admitted
            ]
            slot["candidate_missing_episode_ids"] = [
                value for value in floor_ids if value not in admitted
            ]
        for slot in result.get("atomic_slots", []):
            floor_ids = [
                int(value) for value in slot.get("floor_episode_ids", [])
            ]
            slot["planned_floor_episode_ids"] = floor_ids
            slot["floor_episode_ids"] = [
                value for value in floor_ids if value in admitted
            ]
            slot["candidate_missing_episode_ids"] = [
                value for value in floor_ids if value not in admitted
            ]
        return result

    @staticmethod
    def _enforce_selection_floor(
        selected_ids: list[int],
        required_ids: list[int],
        allowed_ids: set[int],
        limit: int,
    ) -> tuple[list[int], list[dict[str, int]]]:
        required = [
            node_id
            for node_id in dict.fromkeys(int(value) for value in required_ids)
            if node_id in allowed_ids
        ][:limit]
        required_set = set(required)
        result = list(dict.fromkeys(int(value) for value in selected_ids))[:limit]
        changes: list[dict[str, int]] = []
        for add_id in required:
            if add_id in result:
                continue
            if len(result) < limit:
                result.append(add_id)
                changes.append({"add_id": add_id, "remove_id": -1})
                continue
            remove_id = next(
                (
                    node_id
                    for node_id in reversed(result)
                    if node_id not in required_set
                ),
                None,
            )
            if remove_id is None:
                break
            result[result.index(remove_id)] = add_id
            changes.append({"add_id": add_id, "remove_id": remove_id})
        return list(dict.fromkeys(result))[:limit], changes

    @staticmethod
    def _answer_slot_neighbor_floor_ids(
        anchor_ids: list[int],
        episodes: list[dict],
        radius: int,
        total_limit: int,
    ) -> list[int]:
        """Protect bounded narrative neighbors around explicit answer slots."""
        if radius <= 0 or total_limit <= 0 or not anchor_ids:
            return []
        episode_by_id = {int(item["id"]): item for item in episodes}
        by_source: dict[str, list[int]] = {}
        for item in episodes:
            source_key = str(item.get("source_key", "")).strip()
            if not source_key:
                continue
            by_source.setdefault(source_key, []).append(int(item["id"]))
        for ids in by_source.values():
            ids.sort()

        selected: list[int] = []
        for anchor_id in dict.fromkeys(int(value) for value in anchor_ids):
            anchor = episode_by_id.get(anchor_id)
            if anchor is None:
                continue
            siblings = by_source.get(str(anchor.get("source_key", "")).strip(), [])
            try:
                position = siblings.index(anchor_id)
            except ValueError:
                continue
            for distance in range(1, radius + 1):
                for neighbor_index in (position - distance, position + distance):
                    if not 0 <= neighbor_index < len(siblings):
                        continue
                    neighbor_id = siblings[neighbor_index]
                    if neighbor_id not in selected and neighbor_id not in anchor_ids:
                        selected.append(neighbor_id)
                    if len(selected) >= total_limit:
                        return selected
        return selected

    @staticmethod
    def _precompress_rerank_candidate_ids(
        candidate_pool_ids: list[int],
        preferred_ids: list[int],
        required_ids: list[int],
        limit: int,
    ) -> tuple[list[int], list[dict]]:
        """Deterministically reduce Candidate@N without erasing answer slots.

        ``candidate_pool_ids`` is already a one-global/two-atomic interleave,
        while ``preferred_ids`` is round-robin across the atomic retrieval
        queries.  Required floor rows are admitted first; the remainder keeps
        the established pool order.  Consequently compression adds no model
        judgement and is exactly replayable from the trace.
        """
        pool = list(dict.fromkeys(int(value) for value in candidate_pool_ids))
        maximum = min(len(pool), max(0, int(limit)))
        preferred = set(int(value) for value in preferred_ids)
        required = [
            node_id
            for node_id in dict.fromkeys(int(value) for value in required_ids)
            if node_id in pool
        ]
        if maximum >= len(pool):
            selected = list(pool)
        else:
            selected = required[:maximum]
            for node_id in pool:
                if len(selected) >= maximum:
                    break
                if node_id not in selected:
                    selected.append(node_id)

        selected_set = set(selected)
        required_set = set(required)
        decisions = []
        for pool_rank, node_id in enumerate(pool, start=1):
            if node_id not in selected_set:
                reason = "dropped_budget"
            elif node_id in required_set:
                reason = "required_evidence_floor"
            elif node_id in preferred:
                reason = "atomic_anchor"
            else:
                reason = "global_rank"
            decisions.append(
                {
                    "episode_id": node_id,
                    "pool_rank": pool_rank,
                    "kept": node_id in selected_set,
                    "reason": reason,
                }
            )
        return selected, decisions

    @staticmethod
    def _association_row_value(row, key: str, default=None):
        """Read sqlite3.Row and dict fixtures through one narrow boundary."""

        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            return default

    def _association_cue_fast_rerank(
        self,
        *,
        episodes: list[dict],
        limit: int,
        required_ids: list[int],
        preferred_ids: list[int],
        association_cue_entries: list[dict],
    ) -> tuple[list[int], dict] | None:
        """Reuse a learned relation as a local, evidence-bound rerank cache.

        Relation text is only the lookup key.  It is never promoted to source
        evidence: both endpoints must be direct/source-grounded Episodes, and
        both are passed to the answer model.  Any failed guard falls back to
        the configured cross-encoder or LLM reranker.
        """

        retrieval = self.config.retrieval
        if (
            not retrieval.association_cue_fast_path_enabled
            or not association_cue_entries
            or limit < 2
        ):
            return None
        episode_by_id = {int(item["id"]): item for item in episodes}
        accepted: list[dict] = []
        endpoint_ids: list[int] = []
        ordered_entries = sorted(
            association_cue_entries,
            key=lambda item: float(item.get("cosine", 0.0)),
            reverse=True,
        )
        if len(ordered_entries) >= 2 and (
            float(ordered_entries[0].get("cosine", 0.0))
            - float(ordered_entries[1].get("cosine", 0.0))
            < float(retrieval.association_cue_fast_path_min_margin)
        ):
            return None
        for entry in ordered_entries:
            if len(accepted) >= max(
                1, int(retrieval.association_cue_fast_path_max_edges)
            ):
                break
            cosine = float(entry.get("cosine", 0.0))
            if str(entry.get("score_kind", "")).startswith("local_"):
                if float(entry.get("local_coverage", 0.0)) < float(
                    retrieval.association_cue_fast_path_local_min_coverage
                ) or float(entry.get("local_margin", 0.0)) < float(
                    retrieval.association_cue_fast_path_local_min_margin
                ):
                    continue
            elif cosine < float(
                retrieval.association_cue_fast_path_min_similarity
            ):
                continue
            association_id = int(entry.get("association_id", -1))
            row = self.associations.get(association_id)
            if not self._association_is_cue_eligible(row):
                continue
            if float(
                self._association_row_value(row, "confidence", 0.0) or 0.0
            ) < float(retrieval.association_cue_fast_path_min_confidence):
                continue
            if not self._association_has_fast_path_confirmation(row):
                continue
            endpoints = [
                (
                    str(self._association_row_value(row, "from_type", "")),
                    int(self._association_row_value(row, "from_id", -1)),
                ),
                (
                    str(self._association_row_value(row, "to_type", "")),
                    int(self._association_row_value(row, "to_id", -1)),
                ),
            ]
            if any(node_type != "episode" for node_type, _ in endpoints):
                continue
            candidate_endpoint_ids = [node_id for _, node_id in endpoints]
            endpoint_rows = [
                episode_by_id.get(node_id) for node_id in candidate_endpoint_ids
            ]
            if any(item is None for item in endpoint_rows):
                continue
            if any(
                int(item.get("generation", 0) or 0) != 0
                or str(item.get("evidence_origin", "")).casefold()
                not in {"source", "direct", "imported"}
                for item in endpoint_rows
                if item is not None
            ):
                continue
            accepted.append(
                {
                    "association_id": association_id,
                    "cosine": cosine,
                    "score_kind": str(entry.get("score_kind", "embedding")),
                    "relation_type": str(
                        self._association_row_value(row, "relation_type", "")
                    ),
                    "relation_key": str(
                        self._association_row_value(row, "relation_key", "")
                    ),
                    "relation_text": str(
                        self._association_row_value(row, "relation_text", "")
                    )[:900],
                    "endpoint_episode_ids": candidate_endpoint_ids,
                    "generation": int(
                        self._association_row_value(row, "generation", 0) or 0
                    ),
                    "confidence": float(
                        self._association_row_value(row, "confidence", 0.0)
                        or 0.0
                    ),
                }
            )
            endpoint_ids.extend(candidate_endpoint_ids)
        endpoint_ids = list(dict.fromkeys(endpoint_ids))
        if len(endpoint_ids) < 2:
            return None
        protected_ids = list(dict.fromkeys([*endpoint_ids, *required_ids]))
        if len(protected_ids) > limit:
            return None
        global_ids = [int(item["id"]) for item in episodes]
        selected_ids = list(
            dict.fromkeys(
                [*endpoint_ids, *required_ids, *preferred_ids, *global_ids]
            )
        )[:limit]
        trace = {
            "enabled": True,
            "backend": "association_capsule",
            "model": "local_audited_edge_selector",
            "review_mode": "local",
            "review_level": "association_cache",
            "coverage_audit_performed": False,
            "compressor_performed": False,
            "cache_hit": True,
            "cache_kind": "audited_association_cue",
            "cloud_requests_avoided": 1,
            "candidate_episode_ids": global_ids,
            "rerank_input_episode_ids": selected_ids,
            "required_evidence_floor_ids": list(required_ids),
            "atomic_evidence_floor_ids": list(required_ids),
            "required_evidence_floor_replacements": [],
            "atomic_evidence_floor_replacements": [],
            "association_capsules": accepted,
            "final_episode_ids": selected_ids,
        }
        if self.logger:
            self.logger.emit(
                "association_cue_fast_rerank",
                trace=trace,
            )
        return selected_ids, trace

    def _rerank_answer_episodes(
        self,
        question: str,
        intent: QueryIntent,
        atomic_queries: list[str],
        episodes: list[dict],
        limit: int,
        preferred_candidate_ids: list[int] | None = None,
        paragraph_context_by_source: dict[int, list[dict]] | None = None,
        result_cache: dict[str, tuple[list[int], dict]] | None = None,
        required_candidate_ids: list[int] | None = None,
        answer_slot_anchor_ids: list[int] | None = None,
        association_cue_entries: list[dict] | None = None,
    ) -> tuple[list[int], dict]:
        """Use an evidence-bound LLM pass to compress Candidate@100 to Top-20."""
        if (
            not self.config.retrieval.rerank_enabled
            or self.model is None
            or limit <= 0
        ):
            return [], {"enabled": False}
        expanded_atomic_queries = expand_rerank_atomic_queries(
            atomic_queries,
            intent,
            question,
        )
        atomic_queries = limit_rerank_atomic_queries(
            expanded_atomic_queries,
            int(self.config.retrieval.rerank_atomic_query_limit),
        )
        candidate_limit = max(limit, self.config.retrieval.rerank_candidate_limit)
        episode_by_id = {int(item["id"]): item for item in episodes}
        global_ids = [int(item["id"]) for item in episodes]
        required_ids = [
            int(value)
            for value in dict.fromkeys(required_candidate_ids or [])
            if int(value) in episode_by_id
        ]
        answer_slot_neighbor_ids = self._answer_slot_neighbor_floor_ids(
            list(answer_slot_anchor_ids or []),
            episodes,
            max(
                0,
                int(
                    self.config.retrieval.rerank_answer_slot_neighbor_radius
                ),
            ),
            max(
                0,
                int(
                    self.config.retrieval.rerank_answer_slot_neighbor_total_limit
                ),
            ),
        )
        required_ids = list(
            dict.fromkeys([*required_ids, *answer_slot_neighbor_ids])
        )
        preferred_ids = [
            int(value)
            for value in dict.fromkeys(
                [*required_ids, *(preferred_candidate_ids or [])]
            )
            if int(value) in episode_by_id
        ]
        cue_fast_result = self._association_cue_fast_rerank(
            episodes=episodes,
            limit=limit,
            required_ids=required_ids,
            preferred_ids=preferred_ids,
            association_cue_entries=association_cue_entries or [],
        )
        if cue_fast_result is not None:
            return cue_fast_result
        # Atomic slots are the scarce resource in multi-hop questions. One
        # global candidate followed by two atomic-channel candidates preserves
        # the broad lane while representing more independent subproblems.
        ordered_candidate_ids: list[int] = []
        global_index = 0
        preferred_index = 0
        while len(ordered_candidate_ids) < candidate_limit:
            added = False
            for _ in range(1):
                while (
                    global_index < len(global_ids)
                    and global_ids[global_index] in ordered_candidate_ids
                ):
                    global_index += 1
                if global_index < len(global_ids):
                    ordered_candidate_ids.append(global_ids[global_index])
                    global_index += 1
                    added = True
            for _ in range(2):
                while (
                    preferred_index < len(preferred_ids)
                    and preferred_ids[preferred_index] in ordered_candidate_ids
                ):
                    preferred_index += 1
                if preferred_index < len(preferred_ids):
                    ordered_candidate_ids.append(preferred_ids[preferred_index])
                    preferred_index += 1
                    added = True
            if not added:
                break
        candidate_pool_ids = ordered_candidate_ids[:candidate_limit]
        rerank_backend = str(
            self.config.retrieval.rerank_backend
        ).strip().casefold()
        if rerank_backend not in {"llm", "cross_encoder"}:
            raise ValueError(f"unknown rerank backend: {rerank_backend}")
        rerank_input_limit = min(
            len(candidate_pool_ids),
            max(
                limit,
                int(self.config.retrieval.rerank_precompression_limit),
            ),
        )
        precompression_pool_ids = list(candidate_pool_ids)
        bge_prefilter: dict = {"performed": False}
        reranker_model = str(
            getattr(self.config.model, "reranker_model", "") or ""
        ).strip()
        if (
            rerank_backend == "llm"
            and reranker_model
            and len(candidate_pool_ids) > rerank_input_limit
        ):
            try:
                prefilter_documents = [
                    "\n".join(
                        value
                        for value in (
                            str(episode_by_id[node_id].get("text", "")),
                            "人物：" + ", ".join(
                                episode_by_id[node_id].get("participants", [])
                            )
                            if episode_by_id[node_id].get("participants")
                            else "",
                            "故事时间：" + str(
                                episode_by_id[node_id].get("story_time_text", "")
                            )
                            if episode_by_id[node_id].get("story_time_text")
                            else "",
                        )
                        if value
                    )
                    for node_id in candidate_pool_ids
                ]
                ranked = self.model.rerank(
                    question,
                    prefilter_documents,
                    top_n=len(prefilter_documents),
                )
                ranked_pool_ids = [
                    candidate_pool_ids[int(item["index"])] for item in ranked
                ]
                if ranked_pool_ids:
                    precompression_pool_ids = list(
                        dict.fromkeys(ranked_pool_ids)
                    )
                bge_prefilter = {
                    "performed": True,
                    "model": reranker_model,
                    "input_count": len(candidate_pool_ids),
                    "ranked_episode_ids": precompression_pool_ids,
                    "scores": [
                        round(float(item["relevance_score"]), 8)
                        for item in ranked
                    ],
                }
            except Exception as exc:
                bge_prefilter = {
                    "performed": True,
                    "model": reranker_model,
                    "input_count": len(candidate_pool_ids),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        ordered_candidate_ids, precompression_decisions = (
            self._precompress_rerank_candidate_ids(
                precompression_pool_ids,
                preferred_ids,
                required_ids,
                rerank_input_limit,
            )
        )
        candidate_rows = [episode_by_id[node_id] for node_id in ordered_candidate_ids]
        preferred_rank = {
            int(node_id): rank
            for rank, node_id in enumerate(
                dict.fromkeys(preferred_candidate_ids or []), start=1
            )
        }
        paragraph_context_catalog = sorted(
            (
                context
                for contexts in (paragraph_context_by_source or {}).values()
                for context in contexts
            ),
            key=lambda item: int(item["paragraph_id"]),
        )
        candidates = [
            {
                "id": int(item["id"]),
                "score": round(float(item.get("score", 0.0)), 6),
                "text": str(item.get("text", ""))[:700],
                "participants": item.get("participants", []),
                "source_key": str(item.get("source_key", "")),
                "story_time_text": str(item.get("story_time_text", "")),
                "timeline_scope": str(item.get("timeline_scope", "")),
                "evidence_origin": str(item.get("evidence_origin", "unknown")),
                "epistemic_status": str(item.get("epistemic_status", "unknown")),
                "generation": int(item.get("generation", 0) or 0),
                "epistemic_note": str(item.get("epistemic_note", "")),
                "atomic_anchor_rank": preferred_rank.get(int(item["id"])),
                "source_context_refs": [
                    int(context["paragraph_id"])
                    for context in (paragraph_context_by_source or {}).get(
                        int(item.get("source_id", -1)), []
                    )
                ],
            }
            for item in candidate_rows
        ]
        if rerank_backend == "llm":
            initial_prompt = evidence_rerank_prompt(
                question,
                asdict(intent),
                atomic_queries,
                candidates,
                limit,
                paragraph_context_catalog,
            )
            coverage_prompt = evidence_coverage_audit_prompt(
                question,
                atomic_queries,
                candidates,
                paragraph_context_catalog,
            )
        else:
            initial_prompt = ""
            coverage_prompt = ""
        cache_payload = {
            "version": 1,
            "model": (
                self.config.model.reranker_model
                if rerank_backend == "cross_encoder"
                else self.config.model.reasoning_model
            ),
            "rerank_backend": rerank_backend,
            "prompt_version": self.config.prompt_version,
            "question": question,
            "intent": asdict(intent),
            "atomic_queries": atomic_queries,
            "candidates": candidates,
            "candidate_pool_ids": candidate_pool_ids,
            "bge_prefilter": bge_prefilter,
            "rerank_precompression_limit": rerank_input_limit,
            "source_contexts": paragraph_context_catalog,
            "required_candidate_ids": required_ids,
            "answer_slot_anchor_ids": list(answer_slot_anchor_ids or []),
            "answer_slot_neighbor_ids": answer_slot_neighbor_ids,
            "limit": limit,
            "shortlist_limit": self.config.retrieval.rerank_shortlist_limit,
            "coverage_audit_enabled": (
                self.config.retrieval.rerank_coverage_audit_enabled
            ),
            "rerank_audit_enabled": self.config.retrieval.rerank_audit_enabled,
            "rerank_review_mode": self.config.retrieval.rerank_review_mode,
            "rerank_atomic_query_limit": (
                self.config.retrieval.rerank_atomic_query_limit
            ),
            "rerank_strict_atomic_query_threshold": (
                self.config.retrieval.rerank_strict_atomic_query_threshold
            ),
            "rerank_compress_coverage_group_threshold": (
                self.config.retrieval.rerank_compress_coverage_group_threshold
            ),
            "rerank_answer_slot_neighbor_radius": (
                self.config.retrieval.rerank_answer_slot_neighbor_radius
            ),
            "rerank_answer_slot_neighbor_total_limit": (
                self.config.retrieval.rerank_answer_slot_neighbor_total_limit
            ),
            "systems": (
                []
                if rerank_backend == "cross_encoder"
                else [
                    EVIDENCE_RERANK_SYSTEM,
                    EVIDENCE_COVERAGE_AUDIT_SYSTEM,
                    EVIDENCE_RERANK_AUDIT_SYSTEM,
                ]
            ),
            "initial_prompt": initial_prompt,
            "coverage_prompt": coverage_prompt,
        }
        input_hash = hashlib.sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        allowed_ids = {int(item["id"]) for item in candidates}
        trace: dict = {
            "enabled": True,
            "backend": rerank_backend,
            "model": (
                self.config.model.reranker_model
                if rerank_backend == "cross_encoder"
                else self.config.model.reasoning_model
            ),
            # Preserve Candidate@100 semantics for recall measurement even
            # when fewer rows are sent to the model.
            "candidate_episode_ids": candidate_pool_ids,
            "rerank_input_episode_ids": [
                int(item["id"]) for item in candidates
            ],
            "precompression": {
                "enabled": len(candidate_pool_ids) > len(candidates),
                "pool_limit": candidate_limit,
                "configured_limit": int(
                    self.config.retrieval.rerank_precompression_limit
                ),
                "effective_limit": rerank_input_limit,
                "pool_count": len(candidate_pool_ids),
                "rerank_input_count": len(candidates),
                "dropped_count": len(candidate_pool_ids) - len(candidates),
                "decisions": precompression_decisions,
            },
            "bge_prefilter": bge_prefilter,
            "atomic_queries": atomic_queries,
            "expanded_atomic_query_count": len(expanded_atomic_queries),
            "atomic_queries_truncated": (
                len(atomic_queries) < len(expanded_atomic_queries)
            ),
            "input_hash": input_hash,
            "cache_hit": False,
            "paragraph_context_episode_count": sum(
                bool(item["source_context_refs"]) for item in candidates
            ),
            "paragraph_context_ids": sorted(
                {
                    int(paragraph_id)
                    for item in candidates
                    for paragraph_id in item["source_context_refs"]
                }
            ),
            "required_evidence_floor_ids": required_ids,
            "atomic_evidence_floor_ids": required_ids,
            "answer_slot_anchor_ids": [
                int(value) for value in (answer_slot_anchor_ids or [])
            ],
            "answer_slot_neighbor_floor_ids": answer_slot_neighbor_ids,
            "rerank_candidate_text_chars": sum(
                len(str(item.get("text", ""))) for item in candidates
            ),
            "initial_prompt_chars": len(initial_prompt),
            "coverage_prompt_chars": len(coverage_prompt),
        }
        if result_cache is not None and input_hash in result_cache:
            cached_ids, cached_trace = result_cache[input_hash]
            reused_trace = deepcopy(cached_trace)
            reused_trace["cache_hit"] = True
            reused_trace["cache_source_input_hash"] = input_hash
            if self.logger:
                self.logger.emit(
                    "evidence_rerank_reused",
                    question=question,
                    input_hash=input_hash,
                )
            return list(cached_ids), reused_trace
        if rerank_backend == "cross_encoder":
            try:
                documents = [
                    "\n".join(
                        value
                        for value in (
                            str(item.get("text", "")),
                            (
                                "人物：" + ", ".join(item["participants"])
                                if item.get("participants")
                                else ""
                            ),
                            (
                                "故事时间：" + str(item["story_time_text"])
                                if item.get("story_time_text")
                                else ""
                            ),
                            (
                                "时间线：" + str(item["timeline_scope"])
                                if item.get("timeline_scope")
                                else ""
                            ),
                        )
                        if value
                    )
                    for item in candidates
                ]
                ranked = self.model.rerank(
                    question,
                    documents,
                    top_n=len(documents),
                )
                ranked_ids = [
                    int(candidates[int(item["index"])]["id"])
                    for item in ranked
                ]
                selected_ids = list(dict.fromkeys(ranked_ids))[:limit]
                selected_ids, floor_changes = self._enforce_selection_floor(
                    selected_ids,
                    required_ids,
                    allowed_ids,
                    limit,
                )
                for item in candidate_rows:
                    node_id = int(item["id"])
                    if len(selected_ids) >= limit:
                        break
                    if node_id not in selected_ids:
                        selected_ids.append(node_id)
                trace.update(
                    {
                        "review_mode": "cross_encoder",
                        "review_level": "cross_encoder",
                        "coverage_audit_performed": False,
                        "compressor_performed": False,
                        "cross_encoder_ranked_episode_ids": ranked_ids,
                        "cross_encoder_scores": [
                            round(float(item["relevance_score"]), 8)
                            for item in ranked
                        ],
                        "required_evidence_floor_replacements": floor_changes,
                        "atomic_evidence_floor_replacements": floor_changes,
                        "final_episode_ids": selected_ids,
                    }
                )
                if result_cache is not None:
                    result_cache[input_hash] = (
                        list(selected_ids),
                        deepcopy(trace),
                    )
                if self.logger:
                    self.logger.emit(
                        "evidence_reranked", question=question, trace=trace
                    )
                return selected_ids, trace
            except Exception as exc:
                trace["error"] = f"{type(exc).__name__}: {exc}"
                if self.logger:
                    self.logger.emit(
                        "evidence_rerank_failed",
                        question=question,
                        error=trace["error"],
                    )
                return [], trace
        try:
            initial = self.model.chat_json(
                EVIDENCE_RERANK_SYSTEM,
                initial_prompt,
                allow_fallback=False,
                max_retries=0,
            )
            review_level, review_reasons = self._rerank_review_level(
                atomic_queries,
                initial,
                allowed_ids,
                question,
            )
            run_coverage_audit = bool(
                review_level == "strict"
                and self.config.retrieval.rerank_coverage_audit_enabled
            )
            run_compressor = bool(
                review_level in {"strict", "compress"}
                and self.config.retrieval.rerank_audit_enabled
            )
            trace["review_mode"] = self.config.retrieval.rerank_review_mode
            trace["review_level"] = review_level
            trace["review_reasons"] = review_reasons
            trace["coverage_audit_performed"] = run_coverage_audit
            trace["compressor_performed"] = run_compressor
            coverage_audit = None
            coverage_payload = initial
            if run_coverage_audit:
                try:
                    coverage_audit = self.model.chat_json(
                        EVIDENCE_COVERAGE_AUDIT_SYSTEM,
                        coverage_prompt,
                        allow_fallback=False,
                        max_retries=0,
                    )
                    coverage_payload = self._merge_coverage_payloads(
                        initial,
                        coverage_audit,
                    )
                except Exception as exc:
                    trace["coverage_audit_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            shortlist_limit = max(
                limit, self.config.retrieval.rerank_shortlist_limit
            )
            initial_ids = self._coverage_rerank_ids(
                coverage_payload, allowed_ids, shortlist_limit
            )
            if not initial_ids:
                initial_ids = self._validated_rerank_ids(
                    initial, "selected_episode_ids", allowed_ids, shortlist_limit
                )
            shortlist_ids = list(initial_ids)
            for item in candidate_rows:
                node_id = int(item["id"])
                if len(shortlist_ids) >= shortlist_limit:
                    break
                if node_id not in shortlist_ids:
                    shortlist_ids.append(node_id)
            candidate_by_id = {int(item["id"]): item for item in candidates}
            shortlist_candidates = [
                candidate_by_id[node_id]
                for node_id in shortlist_ids
                if node_id in candidate_by_id
            ]
            trace["initial"] = initial
            trace["coverage_audit"] = coverage_audit
            trace["independent_coverage"] = coverage_audit
            trace["merged_coverage"] = coverage_payload
            trace["shortlist_episode_ids"] = shortlist_ids
            selected_ids = initial_ids[:limit]
            trace["audit"] = None
            if run_compressor and shortlist_ids:
                audit = self.model.chat_json(
                    EVIDENCE_RERANK_AUDIT_SYSTEM,
                    evidence_rerank_audit_prompt(
                        question,
                        atomic_queries,
                        shortlist_candidates,
                        coverage_payload,
                        limit,
                        paragraph_context_catalog,
                    ),
                    allow_fallback=False,
                    max_retries=0,
                )
                audited_ids = self._validated_rerank_ids(
                    audit, "final_episode_ids", allowed_ids, limit
                )
                if audited_ids:
                    selected_ids = self._apply_rerank_replacements(
                        audit,
                        audited_ids,
                        set(shortlist_ids),
                        limit,
                    )
                trace["audit"] = audit
            selected_ids, floor_changes, coverage_changes = (
                self._enforce_final_selection_constraints(
                    selected_ids,
                    required_ids,
                    coverage_payload,
                    allowed_ids,
                    set(shortlist_ids),
                    limit,
                )
            )
            trace["required_evidence_floor_replacements"] = floor_changes
            trace["atomic_evidence_floor_replacements"] = floor_changes
            trace["coverage_enforced_replacements"] = coverage_changes
            # Do not waste unused answer slots when the model returns a short list.
            for item in candidate_rows:
                node_id = int(item["id"])
                if len(selected_ids) >= limit:
                    break
                if node_id not in selected_ids:
                    selected_ids.append(node_id)
            trace["final_episode_ids"] = selected_ids
            if result_cache is not None and not any(
                trace.get(key)
                for key in (
                    "error",
                    "coverage_audit_error",
                    "independent_coverage_error",
                )
            ):
                result_cache[input_hash] = (
                    list(selected_ids),
                    deepcopy(trace),
                )
            if self.logger:
                self.logger.emit("evidence_reranked", question=question, trace=trace)
            return selected_ids, trace
        except Exception as exc:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            if self.logger:
                self.logger.emit(
                    "evidence_rerank_failed",
                    question=question,
                    error=trace["error"],
                )
            return [], trace

    def _generate_audited_answer(
        self,
        question: str,
        intent: QueryIntent,
        episodes: list[dict],
        concepts: list[dict],
        paths: list[dict],
        chronology_notes: list[str],
    ) -> tuple[str, list[dict], int]:
        """Generate, verify, and at most three times revise an evidence-bound answer."""
        intent_dict = asdict(intent)
        base_prompt = answer_prompt(
            question,
            intent_dict,
            episodes,
            concepts,
            paths,
            chronology_notes,
        )
        answer = self.model.chat_text(ANSWER_SYSTEM, base_prompt)
        audits: list[dict] = []
        revision_count = 0
        audit_event_continuity = self._should_audit_event_continuity(
            question, intent, episodes
        )
        for audit_index in range(4):
            try:
                payload = self.model.chat_json(
                    ANSWER_AUDIT_SYSTEM,
                    answer_audit_prompt(question, intent_dict, answer, episodes),
                )
                if audit_event_continuity:
                    continuity_payload = self.model.chat_json(
                        EVENT_CONTINUITY_AUDIT_SYSTEM,
                        event_continuity_audit_prompt(question, answer, episodes),
                    )
                    if isinstance(payload, dict) and isinstance(
                        continuity_payload, dict
                    ):
                        continuity_reviews = continuity_payload.get("reviews", [])
                        if not isinstance(continuity_reviews, list):
                            continuity_reviews = [
                                {
                                    "claim": "<跨事件连续性审计响应>",
                                    "verdict": "unsupported",
                                    "requires_revision": True,
                                    "reason": "reviews 必须是数组",
                                }
                            ]
                        continuity_reviews = [
                            {
                                **review,
                                "audit_scope": "event_continuity",
                            }
                            if isinstance(review, dict)
                            else review
                            for review in continuity_reviews
                        ]
                        payload_reviews = payload.get("reviews")
                        if isinstance(payload_reviews, list):
                            payload["reviews"] = [
                                *payload_reviews,
                                *continuity_reviews,
                            ]
                        elif continuity_reviews:
                            payload["reviews"] = continuity_reviews
                        payload["event_continuity_reviews"] = continuity_reviews
                        corrections = continuity_payload.get(
                            "correction_instructions", []
                        )
                        if isinstance(corrections, list):
                            existing_corrections = payload.get(
                                "correction_instructions", []
                            )
                            if not isinstance(existing_corrections, list):
                                existing_corrections = []
                            payload["correction_instructions"] = [
                                *existing_corrections,
                                *corrections,
                            ]
            except Exception as exc:
                audit = {
                    "valid": None,
                    "audit_error": f"{type(exc).__name__}: {exc}",
                }
                audits.append(audit)
                if self.logger:
                    self.logger.emit(
                        "answer_audit_failed",
                        question=question,
                        audit_index=audit_index,
                        error=audit["audit_error"],
                    )
                break
            if not isinstance(payload, dict):
                payload = {
                    "valid": False,
                    "issues": [
                        {
                            "claim": "<audit response>",
                            "reason": "answer audit did not return a JSON object",
                        }
                    ],
                    "correction_instructions": ["重新核对所有直接结论"],
                }
            audit = self._normalize_answer_audit(payload, answer)
            audits.append(audit)
            if self.logger:
                self.logger.emit(
                    "answer_evidence_audit",
                    question=question,
                    audit_index=audit_index,
                    audit=audit,
                )
            if audit.get("valid") is True:
                break
            if revision_count >= 3:
                issues = audit.get("issues", [])
                answer += (
                    "\n\n[系统提示：答案在三次修订后仍未通过全部证据审计，"
                    f"未解决项：{json.dumps(issues, ensure_ascii=False)}]"
                )
                break
            answer = self.model.chat_text(
                ANSWER_SYSTEM,
                base_prompt + answer_correction_prompt(answer, audit),
            )
            revision_count += 1
        return answer, audits, revision_count

    @staticmethod
    def _should_audit_event_continuity(
        question: str,
        intent: QueryIntent,
        episodes: list[dict],
    ) -> bool:
        source_keys = {
            str(episode.get("source_key", ""))
            for episode in episodes
            if episode.get("source_key")
        }
        if len(source_keys) < 2:
            return False
        causal_text = f"{question} {intent.causal_constraint}".casefold()
        return any(
            marker in causal_text
            for marker in (
                "为什么",
                "为何",
                "因为",
                "导致",
                "因果",
                "原因",
                "trigger",
                "cause",
                "enable",
            )
        )

    @staticmethod
    def _normalize_answer_audit(payload: dict, answer: str = "") -> dict:
        """Derive validity from per-claim verdicts instead of free-form prose."""
        audit = dict(payload)
        reviews = audit.get("reviews")
        if not isinstance(reviews, list):
            return audit
        normalized_reviews: list[dict] = []
        issues: list[dict] = []
        allowed = {
            "supported_fact",
            "supported_inference",
            "unsupported",
            "contradicted",
        }
        for raw_review in reviews:
            if not isinstance(raw_review, dict):
                review = {
                    "claim": "<malformed review>",
                    "verdict": "unsupported",
                    "requires_revision": True,
                    "reason": "answer audit review must be an object",
                }
            else:
                review = dict(raw_review)
                verdict = str(review.get("verdict", "")).strip()
                if verdict not in allowed:
                    verdict = "unsupported"
                    review["verdict"] = verdict
                    review["reason"] = (
                        str(review.get("reason", ""))
                        + " [invalid or missing verdict]"
                    ).strip()
                requires_revision = verdict in {"unsupported", "contradicted"}
                review["requires_revision"] = requires_revision
            normalized_reviews.append(review)
            if review.get("requires_revision") is True:
                issues.append(review)
        if not normalized_reviews:
            issues.append(
                {
                    "claim": "<missing reviews>",
                    "verdict": "unsupported",
                    "requires_revision": True,
                    "reason": "answer audit returned no claim reviews",
                }
            )
        audit["reviews"] = normalized_reviews
        audit["issues"] = issues
        audit["valid"] = not issues
        return audit

    def _explicit_association_paths(
        self,
        association_ids: list[int],
        cue_scores: dict[int, float] | None = None,
    ) -> list[dict]:
        """Make newly changed edges eligible even when neighbor fan-out is full."""
        cue_scores = cue_scores or {}
        paths: list[dict] = []
        for association_id in dict.fromkeys(association_ids):
            row = self.associations.get(int(association_id))
            if row is None:
                continue
            paths.append(
                {
                    "association_id": int(row["id"]),
                    "from": [str(row["from_type"]), int(row["from_id"])],
                    "to": [str(row["to_type"]), int(row["to_id"])],
                    "relation_type": str(row["relation_type"]),
                    "relation_key": str(row["relation_key"]),
                    "relation_text": str(row["relation_text"]),
                    "association_mode": (
                        str(row["association_mode"])
                        if "association_mode" in row.keys()
                        else "semantic"
                    ),
                    "polarity": int(row["polarity"]),
                    "weight": float(row["weight"]),
                    "confidence": float(row["confidence"]),
                    "generation": (
                        int(row["generation"])
                        if "generation" in row.keys()
                        else 0
                    ),
                    "claim_level": (
                        str(row["claim_level"])
                        if "claim_level" in row.keys()
                        else "direct_fact"
                    ),
                    "audit_status": (
                        str(row["audit_status"])
                        if "audit_status" in row.keys()
                        else "not_required"
                    ),
                    "created_reason": (
                        str(row["created_reason"])
                        if "created_reason" in row.keys()
                        else ""
                    ),
                    "path_score": float(row["weight"])
                    * float(row["confidence"]),
                    "association_cue_similarity": cue_scores.get(
                        int(row["id"])
                    ),
                }
            )
        return paths

    def _late_learned_bridge_closure(
        self,
        question: str,
        preferred_episode_ids: list[int],
        traversed: list[TraversedNode],
    ) -> tuple[list[TraversedNode], list[dict]]:
        """Admit one-hop endpoints of strong learned edges after reranking."""

        remaining = max(0, int(self.config.retrieval.learned_bridge_slots))
        if remaining <= 0:
            return [], []
        existing = {
            (str(item.node_type), int(item.node_id)) for item in traversed
        }
        score_by_episode = {
            int(item.node_id): float(item.score)
            for item in traversed
            if item.node_type == "episode"
        }
        preferred = list(
            dict.fromkeys(
                [
                    *(int(value) for value in preferred_episode_ids),
                    *(
                        int(item.node_id)
                        for item in traversed
                        if item.node_type == "episode"
                    ),
                ]
            )
        )
        additions: list[TraversedNode] = []
        paths: list[dict] = []
        used_edges: set[int] = set()
        neighbors_many = getattr(self.associations, "neighbors_many", None)
        if callable(neighbors_many):
            neighbor_map = neighbors_many("episode", preferred, limit=24)
        else:
            neighbor_map = {
                node_id: self.associations.neighbors(
                    "episode", node_id, limit=24
                )
                for node_id in preferred
            }
        for rank, anchor_id in enumerate(preferred, start=1):
            if remaining <= 0:
                break
            anchor_score = score_by_episode.get(
                anchor_id,
                0.55 / (1.0 + (rank - 1) / 20.0),
            )
            for row in neighbor_map.get(anchor_id, []):
                association_id = int(row["id"])
                if association_id in used_edges:
                    continue
                if (
                    str(row["audit_status"]) != "dual_accepted"
                    or int(row["generation"] or 0) > 1
                    or float(row["confidence"]) < 0.75
                    or int(row["polarity"]) < 0
                    or str(row["relation_key"]) == "involves"
                    or str(row["from_type"]) != "episode"
                    or str(row["to_type"]) != "episode"
                ):
                    continue
                explicit_paths = self._explicit_association_paths(
                    [association_id]
                )
                if not explicit_paths:
                    continue
                path = explicit_paths[0]
                relevance = self._path_query_relevance(question, path)
                if relevance < float(
                    self.config.retrieval.learned_bridge_min_query_relevance
                ):
                    continue
                other_id = (
                    int(row["to_id"])
                    if int(row["from_id"]) == anchor_id
                    else int(row["from_id"])
                )
                path_score = (
                    anchor_score
                    * float(row["weight"])
                    * float(row["confidence"])
                    * 0.85
                )
                if ("episode", other_id) not in existing:
                    additions.append(
                        TraversedNode(
                            "episode",
                            other_id,
                            path_score,
                            [association_id],
                        )
                    )
                path.update(
                    {
                        "from": ["episode", anchor_id],
                        "to": ["episode", other_id],
                        "path_score": path_score,
                        "late_bridge_closure": True,
                    }
                )
                paths.append(path)
                existing.add(("episode", other_id))
                used_edges.add(association_id)
                remaining -= 1
                if remaining <= 0:
                    break
        return additions, paths

    def _materialize_nodes(
        self, traversed, include_sources: bool = False
    ) -> tuple[list[dict], list[dict]]:
        episode_scores = {
            item.node_id: item.score for item in traversed if item.node_type == "episode"
        }
        concept_scores = {
            item.node_id: item.score for item in traversed if item.node_type == "concept"
        }
        episode_rows = self.episodes.get_many(episode_scores)
        concept_rows = self.concepts.get_many(concept_scores)
        source_map = {}
        if include_sources:
            source_map = {
                int(row["id"]): row
                for row in self.sources.get_many(
                    int(episode["source_id"]) for episode in episode_rows
                )
            }
        episodes: list[dict] = []
        for row in episode_rows:
            try:
                participants = json.loads(row["participants_json"])
            except (TypeError, json.JSONDecodeError):
                participants = []
            source = source_map.get(int(row["source_id"]))
            episodes.append(
                {
                    "type": "episode",
                    "id": int(row["id"]),
                    "source_id": int(row["source_id"]),
                    "score": episode_scores[int(row["id"])],
                    "text": row["text"],
                    "participants": participants,
                    "source_key": row["source_key"],
                    "segment_index": int(row["segment_index"]),
                    "story_time_text": row["story_time_text"],
                    "story_order": row["story_order"],
                    "timeline_scope": row["timeline_scope"],
                    "evidence_origin": row["evidence_origin"],
                    "epistemic_status": row["epistemic_status"],
                    "generation": int(row["generation"] or 0),
                    "epistemic_note": row["epistemic_note"],
                    "source_text": source_excerpt(
                        str(source["raw_text"]),
                        str(row["text"]),
                        participants,
                        self.config.retrieval.source_excerpt_chars,
                    ) if source else "",
                }
            )
        concepts = [
            {
                "type": "concept",
                "id": int(row["id"]),
                "score": concept_scores[int(row["id"])],
                "canonical_name": row["canonical_name"],
                "description": row["description"],
            }
            for row in concept_rows
        ]
        episodes.sort(key=lambda item: item["score"], reverse=True)
        concepts.sort(key=lambda item: item["score"], reverse=True)
        return episodes, concepts

    @staticmethod
    def _path_text_features(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        features = set(_PATH_WORD_RE.findall(normalized.replace("_", " ")))
        for run in _PATH_CJK_RUN_RE.findall(normalized):
            for size in (2, 3):
                features.update(
                    run[index : index + size]
                    for index in range(max(0, len(run) - size + 1))
                )
        return features

    @classmethod
    def _path_query_relevance(cls, question: str, path: dict) -> float:
        if not question:
            return 0.0
        question_features = cls._path_text_features(question)
        relation_features = cls._path_text_features(
            f"{path.get('relation_key', '')} {path.get('relation_text', '')}"
        )
        if not question_features or not relation_features:
            return 0.0
        overlap = len(question_features.intersection(relation_features))
        return overlap / math.sqrt(len(question_features) * len(relation_features))

    @classmethod
    def _rank_answer_paths(
        cls,
        paths: list[dict],
        question: str,
        path_limit: int,
        preferred_association_ids: set[int] | None = None,
        preferred_association_scores: dict[int, float] | None = None,
    ) -> list[dict]:
        """Deduplicate and diversify paths before they consume answer budget.

        Raw graph score remains useful inside a relation class, but high-confidence
        Episode→Concept ``involves`` edges must not crowd every cross-Episode bridge
        out of the answer. Query text overlap is deliberately lightweight and only
        reranks already traversed, evidence-backed Association rows.
        """
        preferred = preferred_association_ids or set()
        preferred_scores = preferred_association_scores or {}
        best_by_association: dict[int, dict] = {}
        for raw_path in paths:
            association_id = int(raw_path.get("association_id", -1))
            previous = best_by_association.get(association_id)
            use_raw = previous is None or float(
                raw_path.get("path_score", 0.0)
            ) > float((previous or {}).get("path_score", 0.0))
            winner = dict(raw_path if use_raw else previous)
            if raw_path.get("late_bridge_closure") or (
                previous and previous.get("late_bridge_closure")
            ):
                # The same edge can arrive through ordinary graph traversal
                # and the stricter late-closure lane. Keep the best score but
                # do not discard the latter's admission contract.
                winner["late_bridge_closure"] = True
            best_by_association[association_id] = winner

        candidates: list[dict] = []
        for raw_path in best_by_association.values():
            # Contextual double-key edges are retrieval hints, not claims.
            # Their Episode endpoints may still be selected as evidence, but
            # the edge itself must never be presented as answer provenance or
            # fed into growth as if it were a factual relation.
            if (
                str(raw_path.get("association_mode", "")) == "contextual_recall"
                or str(raw_path.get("claim_level", "")) == "retrieval_only"
                or str(raw_path.get("relation_key", "")) == "contextual_recall"
            ):
                continue
            item = dict(raw_path)
            association_id = int(item.get("association_id", -1))
            if association_id in preferred_scores:
                item["association_cue_similarity"] = float(
                    preferred_scores[association_id]
                )
            item["query_relevance"] = round(
                cls._path_query_relevance(question, item), 6
            )
            item["path_class"] = (
                "structural_involves"
                if str(item.get("relation_key", "")) == "involves"
                else "informative"
            )
            candidates.append(item)

        def score_key(item: dict) -> tuple[float, float]:
            return (
                float(item.get("query_relevance", 0.0)),
                float(item.get("path_score", 0.0)),
            )

        def preferred_score_key(item: dict) -> tuple[int, float, float, float]:
            cue_similarity = item.get("association_cue_similarity")
            return (
                1 if cue_similarity is not None else 0,
                float(cue_similarity or 0.0),
                float(item.get("query_relevance", 0.0)),
                float(item.get("path_score", 0.0)),
            )

        preferred_paths = sorted(
            [
                item
                for item in candidates
                if int(item.get("association_id", -1)) in preferred
            ],
            key=preferred_score_key,
            reverse=True,
        )
        remaining = [
            item
            for item in candidates
            if int(item.get("association_id", -1)) not in preferred
        ]
        informative = sorted(
            [item for item in remaining if item["path_class"] == "informative"],
            key=score_key,
            reverse=True,
        )
        structural = sorted(
            [
                item
                for item in remaining
                if item["path_class"] == "structural_involves"
            ],
            key=score_key,
            reverse=True,
        )
        informative_quota = max(1, math.ceil(max(1, path_limit) * 0.75))
        structural_quota = max(0, max(1, path_limit) - informative_quota)
        ranked = [
            *preferred_paths,
            *informative[:informative_quota],
            *structural[:structural_quota],
        ]
        used = {int(item["association_id"]) for item in ranked}
        ranked.extend(
            sorted(
                [
                    item
                    for item in remaining
                    if int(item["association_id"]) not in used
                ],
                key=score_key,
                reverse=True,
            )
        )
        return ranked

    @classmethod
    def _select_diverse_episode_ids(
        cls,
        episodes: list[dict],
        quota: int,
        question: str,
        preferred_episode_ids: list[int] | None,
    ) -> set[int]:
        """Compress a broad candidate set without selecting paraphrase clusters.

        Dense/Sparse retrieval is deliberately high-recall.  This selector uses
        lightweight lexical coverage and redundancy penalties only after recall,
        so no evidence is invented and no corpus-specific rule is required.
        """
        if quota <= 0:
            return set()
        episode_by_id = {int(item["id"]): item for item in episodes}
        preferred = [
            int(value)
            for value in dict.fromkeys(preferred_episode_ids or [])
            if int(value) in episode_by_id
        ]
        # Preserve the explicit contract for small atomic-query anchor sets.
        if len(preferred) <= quota:
            selected = set(preferred)
        else:
            selected = set()

        question_features = cls._path_text_features(question)
        feature_map = {
            node_id: cls._path_text_features(
                " ".join(
                    (
                        str(item.get("text", "")),
                        " ".join(str(value) for value in item.get("participants", [])),
                        str(item.get("source_key", "")),
                        str(item.get("story_time_text", "")),
                    )
                )
            )
            for node_id, item in episode_by_id.items()
        }
        preferred_rank = {
            node_id: rank for rank, node_id in enumerate(preferred, start=1)
        }
        maximum_score = max(
            (float(item.get("score", 0.0)) for item in episodes), default=1.0
        ) or 1.0
        covered: set[str] = set()
        source_counts: dict[str, int] = {}
        for node_id in selected:
            covered.update(feature_map[node_id].intersection(question_features))
            source_key = str(episode_by_id[node_id].get("source_key", ""))
            source_counts[source_key] = source_counts.get(source_key, 0) + 1

        pool = list(episode_by_id)
        while len(selected) < min(quota, len(pool)):
            best_id: int | None = None
            best_score = -math.inf
            for node_id in pool:
                if node_id in selected:
                    continue
                item = episode_by_id[node_id]
                features = feature_map[node_id]
                overlap = features.intersection(question_features)
                relevance = (
                    len(overlap)
                    / math.sqrt(len(features) * len(question_features))
                    if features and question_features
                    else 0.0
                )
                novel = overlap.difference(covered)
                novelty = (
                    len(novel) / math.sqrt(len(features) * len(question_features))
                    if features and question_features
                    else 0.0
                )
                redundancy = 0.0
                for selected_id in selected:
                    other = feature_map[selected_id]
                    union = features.union(other)
                    if union:
                        redundancy = max(
                            redundancy,
                            len(features.intersection(other)) / len(union),
                        )
                rank = preferred_rank.get(node_id)
                preferred_bonus = (
                    0.12 / (1.0 + (rank - 1) / 20.0)
                    if rank is not None
                    else 0.0
                )
                source_key = str(item.get("source_key", ""))
                source_count = source_counts.get(source_key, 0)
                score = (
                    0.42 * relevance
                    + 0.30 * novelty
                    + 0.20 * (float(item.get("score", 0.0)) / maximum_score)
                    + preferred_bonus
                    + (0.04 if source_count == 0 else 0.0)
                    - 0.25 * redundancy
                    - 0.025 * source_count
                )
                if score > best_score:
                    best_id = node_id
                    best_score = score
            if best_id is None:
                break
            selected.add(best_id)
            covered.update(feature_map[best_id].intersection(question_features))
            source_key = str(episode_by_id[best_id].get("source_key", ""))
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
        return selected

    @classmethod
    def _select_answer_evidence(
        cls,
        episodes: list[dict],
        paths: list[dict],
        episode_limit: int,
        path_limit: int,
        preferred_association_ids: set[int] | None = None,
        question: str = "",
        preferred_episode_ids: list[int] | None = None,
        learned_bridge_slots: int = 2,
        learned_bridge_min_query_relevance: float = 0.03,
        learned_bridge_duplicate_threshold: float = 0.05,
        coverage_groups: list[dict] | None = None,
        preferred_association_scores: dict[int, float] | None = None,
        protected_episode_ids: set[int] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Keep the base Top-K and admit only a few audited learned bridges.

        Legacy graph density must not automatically evict a quarter of the
        reranked evidence. A durable query-grown edge may replace a tail item
        only when it is relevant to this question and bridges from an Episode
        already selected. Edges created by the current query may retain both
        endpoints so their provenance stays immediately auditable.
        """
        if episode_limit <= 0:
            return [], []
        episode_by_id = {int(item["id"]): item for item in episodes}
        preferred = [
            int(value)
            for value in dict.fromkeys(preferred_episode_ids or [])
            if int(value) in episode_by_id
        ]
        preferred_rank = {
            node_id: rank for rank, node_id in enumerate(preferred, start=1)
        }
        ranked_paths = cls._rank_answer_paths(
            paths,
            question,
            path_limit,
            preferred_association_ids,
            preferred_association_scores,
        )
        # Once the reranker has produced an ordered shortlist, generic graph
        # nodes must not enter a smaller budget merely because they expand the
        # lexical pool. Learned bridges get a separate, explicit lane below.
        base_pool = episodes
        if preferred:
            preferred_set = set(preferred)
            preferred_pool = [
                item for item in episodes if int(item["id"]) in preferred_set
            ]
            if preferred_pool:
                base_pool = preferred_pool
        selected_ids = cls._select_diverse_episode_ids(
            base_pool,
            min(len(episodes), episode_limit),
            question,
            preferred,
        )
        externally_protected = {
            int(value)
            for value in (protected_episode_ids or set())
            if int(value) in episode_by_id
        }
        for node_id in externally_protected:
            if node_id in selected_ids:
                continue
            if len(selected_ids) >= episode_limit:
                removable = [
                    value
                    for value in selected_ids
                    if value not in externally_protected
                ]
                if not removable:
                    break
                selected_ids.remove(
                    max(
                        removable,
                        key=lambda value: preferred_rank.get(
                            value, len(preferred) + 1
                        ),
                    )
                )
            selected_ids.add(node_id)

        current_growth_ids = preferred_association_ids or set()
        retrieval_cue_ids = set(preferred_association_scores or {})
        bridge_paths_used: set[int] = set()
        bridge_added_ids: set[int] = set()
        remaining_bridge_slots = max(0, int(learned_bridge_slots))

        def coverage_protected_ids() -> set[int]:
            protected_ids: set[int] = set()
            for group in coverage_groups or []:
                raw_ids = group.get("episode_ids", [])
                if not isinstance(raw_ids, list):
                    continue
                present = {
                    int(value) for value in raw_ids
                    if int(value) in selected_ids
                }
                if str(group.get("mode", "alternatives")) == "joint":
                    protected_ids.update(present)
                elif len(present) == 1:
                    protected_ids.update(present)
            return protected_ids

        feature_map = {
            node_id: cls._path_text_features(
                " ".join(
                    (
                        str(item.get("text", "")),
                        " ".join(
                            str(value) for value in item.get("participants", [])
                        ),
                        str(item.get("source_key", "")),
                    )
                )
            )
            for node_id, item in episode_by_id.items()
        }
        coverage_membership: dict[int, int] = {}
        for group in coverage_groups or []:
            raw_ids = group.get("episode_ids", [])
            if not isinstance(raw_ids, list):
                continue
            for value in raw_ids:
                node_id = int(value)
                coverage_membership[node_id] = (
                    coverage_membership.get(node_id, 0) + 1
                )
        for path in ranked_paths:
            if remaining_bridge_slots <= 0:
                break
            association_id = int(path.get("association_id", -1))
            is_current_growth = association_id in current_growth_ids
            is_retrieval_cue = association_id in retrieval_cue_ids
            created_reason = str(path.get("created_reason", ""))
            is_oracle = created_reason.startswith("Stage 8 oracle probe")
            is_durable_learned = (
                "查询中自主增长：" in created_reason
                and str(path.get("audit_status", "")) == "dual_accepted"
            )
            if not (is_current_growth or is_oracle or is_durable_learned):
                continue
            if int(path.get("polarity", 1)) < 0:
                continue
            if (
                not is_current_growth
                and not is_retrieval_cue
                and float(path.get("query_relevance", 0.0))
                < float(learned_bridge_min_query_relevance)
            ):
                continue
            endpoint_ids = [
                int(endpoint[1])
                for endpoint in (path.get("from"), path.get("to"))
                if isinstance(endpoint, (list, tuple))
                and len(endpoint) == 2
                and endpoint[0] == "episode"
                and int(endpoint[1]) in episode_by_id
            ]
            missing = [node_id for node_id in endpoint_ids if node_id not in selected_ids]
            if not is_current_growth and missing and not any(
                node_id in selected_ids or node_id in preferred_rank
                for node_id in endpoint_ids
            ):
                continue
            additions = missing[:remaining_bridge_slots]
            if not additions:
                continue
            protected = set(endpoint_ids).union(externally_protected)
            for node_id in additions:
                candidate_features = feature_map.get(node_id, set())
                maximum_redundancy = 0.0
                for selected_id in selected_ids:
                    selected_features = feature_map.get(selected_id, set())
                    union = candidate_features.union(selected_features)
                    if union:
                        maximum_redundancy = max(
                            maximum_redundancy,
                            len(candidate_features.intersection(selected_features))
                            / len(union),
                        )
                adds_uncovered_slot = any(
                    node_id in {
                        int(value) for value in group.get("episode_ids", [])
                    }
                    and not selected_ids.intersection(
                        int(value) for value in group.get("episode_ids", [])
                    )
                    for group in coverage_groups or []
                    if isinstance(group.get("episode_ids", []), list)
                )
                # A late closure is already restricted to positive,
                # dual-audited generation-1 Episode bridges. Its endpoints are
                # expected to share entity vocabulary, so the generic 0.05
                # lexical duplicate threshold is too aggressive here. Exact
                # or near-exact duplicate Episodes still fail the 0.20 guard.
                effective_duplicate_threshold = float(
                    learned_bridge_duplicate_threshold
                )
                if path.get("late_bridge_closure"):
                    effective_duplicate_threshold = max(
                        effective_duplicate_threshold,
                        0.20,
                    )
                if (
                    maximum_redundancy
                    >= effective_duplicate_threshold
                    and not adds_uncovered_slot
                    and not is_retrieval_cue
                ):
                    continue
                if len(selected_ids) >= episode_limit:
                    slot_protected = coverage_protected_ids()
                    removable = [
                        selected_id
                        for selected_id in selected_ids
                        if selected_id not in protected
                        and selected_id not in bridge_added_ids
                        and selected_id not in slot_protected
                    ]
                    if not removable:
                        # Preserve the entire protected base Top-K. A late
                        # bridge has its own bounded slot, so it may extend the
                        # answer set instead of deleting required evidence.
                        if not path.get("late_bridge_closure"):
                            break
                    else:
                        def replacement_key(
                            selected_id: int,
                        ) -> tuple[int, float, int, float]:
                            selected_features = feature_map.get(selected_id, set())
                            union = candidate_features.union(selected_features)
                            redundancy = (
                                len(candidate_features.intersection(selected_features))
                                / len(union)
                                if union
                                else 0.0
                            )
                            return (
                                -coverage_membership.get(selected_id, 0),
                                redundancy,
                                preferred_rank.get(
                                    selected_id, len(preferred) + 1
                                ),
                                -float(
                                    episode_by_id[selected_id].get("score", 0.0)
                                ),
                            )

                        victim = max(
                            removable,
                            key=replacement_key,
                        )
                        selected_ids.remove(victim)
                selected_ids.add(node_id)
                bridge_added_ids.add(node_id)
                remaining_bridge_slots -= 1
                bridge_paths_used.add(association_id)
                if remaining_bridge_slots <= 0:
                    break
        for item in episodes:
            if len(selected_ids) >= episode_limit:
                break
            selected_ids.add(int(item["id"]))
        selected_episodes = [
            item for item in episodes if int(item["id"]) in selected_ids
        ]
        # Membership is selected with sets for efficient diversity/bridge
        # decisions, but presentation order must retain the rerank contract.
        # Otherwise repository/graph score order can move the strongest direct
        # evidence behind a large source cohort before the answer model sees it.
        selected_episodes.sort(
            key=lambda item: (
                preferred_rank.get(int(item["id"]), len(preferred) + 1),
                -float(item.get("score", 0.0)),
            )
        )
        auditable_paths: list[dict] = []
        for path in ranked_paths:
            if int(path.get("association_id", -1)) in bridge_paths_used:
                path["learned_bridge_slot_used"] = True
            endpoints_are_available = all(
                not (
                    isinstance(endpoint, (list, tuple))
                    and len(endpoint) == 2
                    and endpoint[0] == "episode"
                )
                or int(endpoint[1]) in selected_ids
                for endpoint in (path.get("from"), path.get("to"))
            )
            if endpoints_are_available:
                auditable_paths.append(path)
            if len(auditable_paths) >= path_limit:
                break
        return selected_episodes, auditable_paths

    @staticmethod
    def _growth_premise_ids(row: dict) -> set[int]:
        """Return Association premises recorded in a durable growth row."""
        try:
            evidence = json.loads(str(row.get("evidence_json", "[]")))
        except (TypeError, json.JSONDecodeError):
            return set()
        if not isinstance(evidence, list):
            return set()
        premise_ids: set[int] = set()
        for item in evidence:
            if not isinstance(item, dict) or item.get("type") != "association":
                continue
            try:
                premise_ids.add(int(item["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        return premise_ids

    @staticmethod
    def _counterfactual_utility_from_results(
        treatment: dict,
        masked: dict,
        changed_ids: set[int],
    ) -> dict:
        """Measure query-batch utility from a shared deterministic plan.

        No evidence manifest is consulted here. A durable growth batch must be
        visible in the treatment answer paths *and* expose at least one direct
        Episode that disappears when every changed row is masked.
        """
        treatment_episode_ids = [
            int(value) for value in treatment.get("episode_ids", [])
        ]
        masked_episode_ids = [int(value) for value in masked.get("episode_ids", [])]
        treatment_set = set(treatment_episode_ids)
        masked_set = set(masked_episode_ids)
        changed_path_ids = sorted(
            {
                int(path["association_id"])
                for path in treatment.get("association_paths", [])
                if int(path.get("association_id", -1)) in changed_ids
            }
        )
        additional_episode_ids = sorted(treatment_set - masked_set)
        displaced_episode_ids = sorted(masked_set - treatment_set)
        # With a shared plan the only treatment/mask difference is the changed
        # Association batch. A newly selected Episode is therefore causal even
        # when the bounded answer-path list omits the traversed bridge itself.
        utility_observed = bool(additional_episode_ids)
        treatment_sources = {
            str(item.get("source_key", ""))
            for item in treatment.get("evidence_episodes", [])
            if str(item.get("source_key", ""))
        }
        masked_sources = {
            str(item.get("source_key", ""))
            for item in masked.get("evidence_episodes", [])
            if str(item.get("source_key", ""))
        }
        return {
            "enabled": True,
            "scope": "query_batch",
            "changed_association_ids": sorted(changed_ids),
            "changed_path_ids": changed_path_ids,
            "treatment_episode_ids": treatment_episode_ids,
            "masked_episode_ids": masked_episode_ids,
            "additional_episode_ids": additional_episode_ids,
            "displaced_episode_ids": displaced_episode_ids,
            "additional_source_keys": sorted(treatment_sources - masked_sources),
            "displaced_source_keys": sorted(masked_sources - treatment_sources),
            "evidence_gain_count": len(additional_episode_ids),
            "causal_utility_observed": utility_observed,
            "persistable_changed_ids": (
                (changed_path_ids or sorted(changed_ids))
                if utility_observed
                else []
            ),
        }

    def _counterfactual_growth_utility(
        self,
        replay_plan: dict,
        created_ids: list[int],
        reinforced_ids: list[int],
        reinforced_before: dict[int, dict],
    ) -> dict:
        changed = set(
            int(value) for value in [*created_ids, *reinforced_ids]
        )
        if not changed:
            return {
                "enabled": bool(
                    self.config.retrieval.growth_counterfactual_utility_enabled
                ),
                "scope": "query_batch",
                "changed_association_ids": [],
                "changed_path_ids": [],
                "additional_episode_ids": [],
                "displaced_episode_ids": [],
                "additional_source_keys": [],
                "displaced_source_keys": [],
                "evidence_gain_count": 0,
                "causal_utility_observed": False,
                "persistable_changed_ids": [],
            }
        if not self.config.retrieval.growth_counterfactual_utility_enabled:
            return {
                "enabled": False,
                "scope": "query_batch",
                "changed_association_ids": sorted(changed),
                "persistable_changed_ids": sorted(changed),
            }
        after = self.associations.snapshot(list(changed))
        delta = AssociationDelta(
            created=[
                {"id": value, "before": None, "after": after[value]}
                for value in dict.fromkeys(int(item) for item in created_ids)
                if value in after
            ],
            reinforced=[
                {
                    "id": value,
                    "before": reinforced_before[value],
                    "after": after[value],
                }
                for value in dict.fromkeys(int(item) for item in reinforced_ids)
                if value not in set(int(item) for item in created_ids)
                and value in reinforced_before
                and value in after
            ],
        )
        treatment = self.replay_retrieval(replay_plan)
        overlay = AssociationOverlay.from_delta(self.associations, delta)
        masked_engine = QueryEngine(
            self.config,
            self.model,
            self.episode_index,
            self.concept_index,
            self.episodes,
            self.concepts,
            self.sources,
            overlay,
            self.logger,
            association_index=self.association_index,
            paragraph_index=self.paragraph_index,
            paragraphs=self.paragraphs,
            episode_sparse_index=self.episode_sparse_index,
            source_sparse_index=self.source_sparse_index,
        )
        masked = masked_engine.replay_retrieval(replay_plan)
        result = self._counterfactual_utility_from_results(
            treatment, masked, changed
        )
        if self.logger:
            self.logger.emit("growth_counterfactual_utility", **result)
        return result

    def _prune_unused_growth(
        self,
        created_ids: list[int],
        reinforced_ids: list[int],
        reinforced_before: dict[int, dict],
        used_edge_ids: list[int],
    ) -> dict:
        """Keep only query growth that is traceable from final answer paths.

        The existing repository remains the provisional execution substrate,
        so no second graph implementation is needed. Before returning, unused
        creations are deleted and unused reinforcements are restored. Premise
        edges required by a retained higher-generation edge are retained too.
        """
        created = list(dict.fromkeys(int(value) for value in created_ids))
        reinforced = list(dict.fromkeys(int(value) for value in reinforced_ids))
        changed = set([*created, *reinforced])
        retained = changed.intersection(int(value) for value in used_edge_ids)
        rows = self.associations.snapshot(list(changed)) if changed else {}
        pending = list(retained)
        while pending:
            association_id = pending.pop()
            row = rows.get(association_id)
            if row is None:
                continue
            for premise_id in self._growth_premise_ids(row):
                if premise_id in changed and premise_id not in retained:
                    retained.add(premise_id)
                    pending.append(premise_id)

        removed_created: list[int] = []
        for association_id in reversed(created):
            if association_id in retained:
                continue
            if self.associations.delete(association_id):
                removed_created.append(association_id)

        restore_payload = {
            association_id: reinforced_before[association_id]
            for association_id in reinforced
            if association_id not in set(created)
            and association_id not in retained
            and association_id in reinforced_before
        }
        restored_reinforced = self.associations.restore_rows(restore_payload)
        retained_created = [value for value in created if value in retained]
        retained_reinforced = [
            value for value in reinforced
            if value in retained and value not in set(retained_created)
        ]
        result = {
            "enabled": True,
            "retained_created_ids": retained_created,
            "retained_reinforced_ids": retained_reinforced,
            "removed_created_ids": sorted(removed_created),
            "restored_reinforced_ids": sorted(restored_reinforced),
            "premise_closure_ids": sorted(
                retained.difference(int(value) for value in used_edge_ids)
            ),
        }
        if self.logger:
            self.logger.emit("growth_utility_gate", **result)
        return result

    @staticmethod
    def _remap_staged_result(result: dict, mapping: dict[int, int]) -> dict:
        if not mapping:
            result["growth_staging"] = {
                "enabled": True,
                "committed": False,
                "temporary_to_durable_ids": {},
            }
            return result

        def remap_list(values) -> list[int]:
            return [mapping.get(int(value), int(value)) for value in values]

        for key in (
            "association_ids",
            "new_association_ids",
            "reinforced_association_ids",
        ):
            result[key] = remap_list(result.get(key, []))
        for path in result.get("association_paths", []):
            if "association_id" in path:
                value = int(path["association_id"])
                path["association_id"] = mapping.get(value, value)
        gate = result.get("growth_utility_gate", {})
        for key in (
            "retained_created_ids",
            "retained_reinforced_ids",
            "premise_closure_ids",
        ):
            gate[key] = remap_list(gate.get(key, []))
        utility = result.get("growth_counterfactual_utility", {})
        for key in (
            "changed_association_ids",
            "changed_path_ids",
            "persistable_changed_ids",
        ):
            utility[key] = remap_list(utility.get(key, []))
        answer = str(result.get("answer", ""))
        chronology_notes = [
            str(value) for value in result.get("chronology_notes", [])
        ]
        for temporary_id, durable_id in mapping.items():
            answer = answer.replace(f"#{temporary_id}", f"#{durable_id}")
            answer = answer.replace(
                f"Association {temporary_id}", f"Association {durable_id}"
            )
            chronology_notes = [
                value.replace(f"#{temporary_id}", f"#{durable_id}")
                for value in chronology_notes
            ]
        result["answer"] = answer
        result["chronology_notes"] = chronology_notes
        result["growth_staging"] = {
            "enabled": True,
            "committed": True,
            "temporary_to_durable_ids": {
                str(key): value for key, value in mapping.items() if key < 0
            },
        }
        return result

    def query(
        self,
        question: str,
        *,
        generate_answer: bool = True,
        intent_override: QueryIntent | dict | None = None,
        followup_queries_override: list[str] | None = None,
        frozen_plan: dict | None = None,
        deadline_seconds: float | None = None,
        query_embeddings_override: dict[str, np.ndarray] | None = None,
        query_vector_bundle: QueryVectorBundle | None = None,
        contextual_domain: str | None = None,
        contextual_endpoint_limit: int | None = None,
    ) -> dict:
        if (
            self.config.retrieval.growth_max_rounds <= 0
            or not self.config.retrieval.growth_staging_enabled
            or isinstance(self.associations, StagedAssociationOverlay)
        ):
            result = self._query_impl(
                question,
                generate_answer=generate_answer,
                intent_override=intent_override,
                followup_queries_override=followup_queries_override,
                frozen_plan=frozen_plan,
                deadline_seconds=deadline_seconds,
                query_embeddings_override=query_embeddings_override,
                query_vector_bundle=query_vector_bundle,
                contextual_domain=contextual_domain,
                contextual_endpoint_limit=contextual_endpoint_limit,
            )
            result.setdefault(
                "growth_staging",
                {
                    "enabled": False,
                    "committed": False,
                    "temporary_to_durable_ids": {},
                },
            )
            return result

        durable_associations = self.associations
        durable_traverser = self.traverser
        durable_growth = self.growth
        durable_chronology = self.chronology
        staged = StagedAssociationOverlay(durable_associations)
        self.associations = staged
        self.traverser = GraphTraverser(staged)
        self.growth = AssociationGrowthEngine(
            self.model, staged, self.episodes, self.config.weights, self.logger
        )
        self.chronology = ChronologyService(self.episodes, staged, self.logger)
        try:
            result = self._query_impl(
                question,
                generate_answer=generate_answer,
                intent_override=intent_override,
                followup_queries_override=followup_queries_override,
                frozen_plan=frozen_plan,
                deadline_seconds=deadline_seconds,
                query_embeddings_override=query_embeddings_override,
                query_vector_bundle=query_vector_bundle,
                contextual_domain=contextual_domain,
                contextual_endpoint_limit=contextual_endpoint_limit,
            )
            mapping = staged.commit()
            return self._remap_staged_result(result, mapping)
        finally:
            self.associations = durable_associations
            self.traverser = durable_traverser
            self.growth = durable_growth
            self.chronology = durable_chronology

    def _query_impl(
        self,
        question: str,
        *,
        generate_answer: bool = True,
        intent_override: QueryIntent | dict | None = None,
        followup_queries_override: list[str] | None = None,
        frozen_plan: dict | None = None,
        deadline_seconds: float | None = None,
        query_embeddings_override: dict[str, np.ndarray] | None = None,
        query_vector_bundle: QueryVectorBundle | None = None,
        contextual_domain: str | None = None,
        contextual_endpoint_limit: int | None = None,
    ) -> dict:
        self.last_query_embeddings.clear()
        self.last_query_embedding_cache_trace = {"hits": [], "misses": []}
        query_started_at = perf_counter()
        deadline_at = (
            query_started_at + max(0.001, float(deadline_seconds))
            if deadline_seconds is not None
            else None
        )

        def ensure_deadline(stage: str) -> None:
            if deadline_at is not None and perf_counter() >= deadline_at:
                raise TimeoutError(f"query deadline exceeded before {stage}")
        phase_seconds = {
            name: 0.0
            for name in (
                "intent_parse",
                "initial_embedding",
                "initial_retrieval",
                "initial_graph_expansion",
                "contextual_association",
                "followup_planning",
                "followup_embedding",
                "followup_retrieval",
                "followup_graph_expansion",
                "source_cohort",
                "evidence_preparation",
                "evidence_rerank",
                "association_growth",
                "final_selection",
                "growth_utility",
                "answer_generation",
            )
        }
        if frozen_plan is not None:
            if intent_override is not None or followup_queries_override is not None:
                raise ValueError(
                    "frozen_plan cannot be combined with intent/follow-up overrides"
                )
            self._validate_query_plan(question, frozen_plan)
        if self.logger:
            self.logger.emit(
                "query_started",
                question=question,
                intent_overridden=intent_override is not None,
                followup_queries_overridden=followup_queries_override is not None,
                frozen_query_plan_id=(
                    str(frozen_plan["plan_id"]) if frozen_plan is not None else None
                ),
            )
        phase_started = perf_counter()
        ensure_deadline("intent_parse")
        if frozen_plan is not None:
            intent = QueryIntent.from_dict(dict(frozen_plan["intent"]))
        elif isinstance(intent_override, QueryIntent):
            intent = intent_override
        elif isinstance(intent_override, dict):
            intent = QueryIntent.from_dict(intent_override)
        else:
            intent = self._parse_intent(question)
        # An intent override freezes stochastic model planning, not the current
        # deterministic retrieval policy.  Re-apply structural answer slots so
        # controlled A/B runs and production overrides do not silently bypass
        # newer evidence protections.  A full frozen_plan remains byte-stable.
        if frozen_plan is None:
            intent.search_queries = list(
                dict.fromkeys(
                    [
                        *intent.search_queries,
                        *structural_queries(question, intent),
                    ]
                )
            )
        self._record_phase(phase_seconds, "intent_parse", phase_started)
        if frozen_plan is not None:
            initial_queries = [str(value) for value in frozen_plan["initial_queries"]]
            initial_episode_anchor_ids = [
                int(value)
                for value in frozen_plan.get("initial_episode_anchor_ids", [])
            ]
            seeds = [
                SearchHit(
                    str(item["node_type"]),
                    int(item["node_id"]),
                    float(item["score"]),
                )
                for item in frozen_plan["final_seed_hits"]
            ]
            active_cue_ids = [
                int(value)
                for value in frozen_plan.get("association_cue_association_ids", [])
            ]
            active_cue_entries = [
                dict(item)
                for item in frozen_plan.get("association_cue_entries", [])
            ]
            initial_rankings = deepcopy(frozen_plan.get("initial_rankings", {}))
            episode_anchor_ids = [
                int(value) for value in frozen_plan.get("episode_anchor_ids", [])
            ]
        else:
            initial_episode_anchor_ids = []
            initial_queries = list(dict.fromkeys([question, *intent.search_queries]))
            seeds, active_cue_ids, active_cue_entries, initial_rankings = (
                self._vector_seed_hits_with_cues(
                    initial_queries,
                    initial_episode_anchor_ids,
                    self.config.retrieval.answer_whole_question_anchor_episodes,
                    question,
                    phase_seconds=phase_seconds,
                    timing_prefix="initial",
                    query_embeddings_override=query_embeddings_override,
                )
            )
            initial_finalize_started = perf_counter()
            for entity in intent.target_entities:
                for row in self.concepts.find_by_alias(entity):
                    seeds.append(
                        SearchHit(
                            "concept",
                            int(row["canonical_concept_id"] or row["id"]),
                            1.0,
                        )
                    )
            seeds = self._merge_hits(seeds)
            episode_anchor_ids = list(initial_episode_anchor_ids)
            self._record_phase(
                phase_seconds,
                "initial_retrieval",
                initial_finalize_started,
            )
        contextual_trace = {
            "enabled": False,
            "backend": "contextual_double_key",
            "context_hits": [],
            "need_hits": [],
            "attached_edges": [],
            "attached_episode_ids": [],
            "external_calls": 0,
        }
        # Keep a request-local provenance split.  It is deliberately plain
        # metadata (no vectors) so it is safe to place in the trace and to
        # hand to the background plasticity worker.
        base_episode_ids_before_contextual = sorted(
            {
                int(item.node_id)
                for item in seeds
                if item.node_type == "episode"
            }
        )
        base_anchor_activations = self._base_anchor_activations(seeds)
        if query_vector_bundle is not None:
            whole_query = next(
                (item for item in query_vector_bundle.queries if item.role == "whole"),
                None,
            )
            contextual_trace["context_query_id"] = (
                str(whole_query.query_id)
                if whole_query is not None
                else ""
            )
            contextual_trace["query_vectors"] = [
                {
                    "query_id": str(item.query_id),
                    "text_hash": str(item.text_hash),
                    "role": str(item.role),
                    "slot_id": str(item.slot_id),
                    "text": str(getattr(item, "text", "")),
                }
                for item in query_vector_bundle.queries
            ]
            # v2 deliberately waits for the masked base slot selection below.
            # A contextual target must never enter graph traversal as a zero
            # score seed, where a global ranker can silently discard it.
            contextual_trace["deferred_until_residual_slots"] = True
        all_new_associations: list[int] = []
        all_reinforced_associations: list[int] = []
        reinforced_before: dict[int, dict] = {}
        seen_growth_fingerprints: set[tuple] = set()
        phase_started = perf_counter()
        ensure_deadline("initial_graph_expansion")
        traversed, paths = self.traverser.expand(
            seeds,
            self.config.retrieval.graph_beam_width,
            self.config.retrieval.graph_max_hops,
        )
        cue_scores = {
            int(item["association_id"]): float(item.get("cosine", 0.0))
            for item in active_cue_entries
        }
        paths.extend(
            self._explicit_association_paths(active_cue_ids, cue_scores)
        )
        cue_fast_endpoint_keys = self._association_cue_fast_endpoint_keys(
            active_cue_entries
        )
        traversed = self._truncate_traversed_nodes(
            traversed,
            self.config.retrieval.candidate_limit,
            cue_fast_endpoint_keys,
        )
        self._record_phase(
            phase_seconds,
            "initial_graph_expansion",
            phase_started,
        )
        phase_started = perf_counter()
        ensure_deadline("followup_planning")
        followup_planner_invoked = False
        if frozen_plan is not None:
            followup_queries = [
                str(value) for value in frozen_plan.get("followup_queries", [])
            ]
            followup_planning_reason = "frozen_plan"
        elif followup_queries_override is not None:
            followup_queries = list(dict.fromkeys(followup_queries_override))
            followup_planning_reason = "explicit_override"
        else:
            should_plan_followup, followup_planning_reason = (
                self._followup_planning_decision(question, intent)
            )
            followup_planner_invoked = should_plan_followup
            followup_queries = (
                self._plan_followup_queries(question, intent, traversed)
                if should_plan_followup
                else []
            )
        self._record_phase(phase_seconds, "followup_planning", phase_started)
        followup_rankings: dict[str, list] = {
            "episode": [],
            "concept": [],
            "paragraph": [],
            "paragraph_episode_expansion": [],
        }
        if frozen_plan is not None:
            followup_rankings = deepcopy(
                frozen_plan.get("followup_rankings", followup_rankings)
            )
        elif followup_queries:
            followup_episode_anchor_ids: list[int] = []
            (
                followup_hits,
                followup_cue_ids,
                followup_cue_entries,
                followup_rankings,
            ) = (
                self._vector_seed_hits_with_cues(
                    followup_queries,
                    followup_episode_anchor_ids,
                    cue_scope_question=question,
                    phase_seconds=phase_seconds,
                    timing_prefix="followup",
                    query_embeddings_override=query_embeddings_override,
                )
            )
            phase_started = perf_counter()
            ensure_deadline("followup_retrieval")
            active_cue_ids = list(
                dict.fromkeys([*active_cue_ids, *followup_cue_ids])
            )
            active_cue_entries = [
                *active_cue_entries,
                *[
                    item
                    for item in followup_cue_entries
                    if int(item["association_id"])
                    not in {
                        int(existing["association_id"])
                        for existing in active_cue_entries
                    }
                ],
            ]
            cue_scores = {
                int(item["association_id"]): float(item.get("cosine", 0.0))
                for item in active_cue_entries
            }
            seeds = self._merge_hits(seeds, followup_hits)
            episode_anchor_ids = self._interleave_anchor_ids(
                followup_episode_anchor_ids,
                initial_episode_anchor_ids,
            )
            traversed, paths = self.traverser.expand(
                seeds,
                self.config.retrieval.graph_beam_width,
                self.config.retrieval.graph_max_hops,
            )
            paths.extend(
                self._explicit_association_paths(active_cue_ids, cue_scores)
            )
            cue_fast_endpoint_keys = self._association_cue_fast_endpoint_keys(
                active_cue_entries
            )
            traversed = self._truncate_traversed_nodes(
                traversed,
                self.config.retrieval.candidate_limit,
                cue_fast_endpoint_keys,
            )
            self._record_phase(
                phase_seconds,
                "followup_graph_expansion",
                phase_started,
            )
        if self.logger:
            self.logger.emit(
                "followup_queries_planned",
                question=question,
                followup_queries=followup_queries,
                planner_invoked=followup_planner_invoked,
                planning_mode=self.config.retrieval.followup_planning_mode,
                planning_reason=followup_planning_reason,
            )

        phase_started = perf_counter()
        ensure_deadline("source_cohort")
        association_capsule_fast_lane = bool(
            initial_rankings.get("association_capsule_fast_lane")
            or followup_rankings.get("association_capsule_fast_lane")
        )
        if association_capsule_fast_lane:
            source_key_cohort_trace = {
                "enabled": False,
                "reason": "association_capsule_fast_lane",
                "supported_source_keys": [],
                "added_episode_ids": [],
                "boosted_episode_ids": [],
                "skipped_source_keys": [],
            }
        elif frozen_plan is not None:
            source_key_cohort_trace = deepcopy(
                frozen_plan.get("source_key_cohort", {})
            )
        else:
            cohort_hits, source_key_cohort_trace = self._source_key_cohort_hits(
                initial_episode_anchor_ids,
                traversed,
            )
            if cohort_hits:
                seeds = self._merge_hits(seeds, cohort_hits)
                traversed, paths = self.traverser.expand(
                    seeds,
                    self.config.retrieval.graph_beam_width,
                    self.config.retrieval.graph_max_hops,
                )
                paths.extend(
                    self._explicit_association_paths(active_cue_ids, cue_scores)
                )
                traversed = self._truncate_traversed_nodes(
                    traversed,
                    self.config.retrieval.candidate_limit,
                    cue_fast_endpoint_keys,
                )
        self._record_phase(phase_seconds, "source_cohort", phase_started)
        if self.logger:
            self.logger.emit(
                "source_key_cohort_retrieval",
                question=question,
                **source_key_cohort_trace,
            )

        # Freeze the direct/static evidence lane before query-time growth. The
        # graph may add a bounded bridge later, but it cannot trigger a second
        # stochastic rerank that silently replaces already selected evidence.
        phase_started = perf_counter()
        base_episodes, _base_concepts = self._materialize_nodes(
            traversed, include_sources=True
        )
        base_episode_limit = min(
            self.config.retrieval.answer_episode_limit,
            len(base_episodes),
        )
        paragraph_context_by_source = self._paragraph_context_by_source(
            [
                *initial_rankings.get("paragraph", []),
                *followup_rankings.get("paragraph", []),
            ]
        )
        self._record_phase(
            phase_seconds,
            "evidence_preparation",
            phase_started,
        )
        evidence_floor_trace: dict = {}
        if frozen_plan is not None:
            reranked_episode_ids = [
                int(value) for value in frozen_plan.get("reranked_episode_ids", [])
            ]
            rerank_trace = deepcopy(frozen_plan.get("rerank_trace", {}))
        else:
            phase_started = perf_counter()
            evidence_floor_trace = self._hybrid_evidence_floor_trace(
                question,
                initial_rankings,
                followup_rankings,
                initial_queries,
                followup_queries,
            )
            constraint_candidate_ids = self._constraint_candidate_ids(
                question,
                initial_queries,
                initial_rankings,
                followup_queries,
                followup_rankings,
            )
            self._record_phase(
                phase_seconds,
                "evidence_preparation",
                phase_started,
            )
            phase_started = perf_counter()
            ensure_deadline("evidence_rerank")
            reranked_episode_ids, rerank_trace = self._rerank_answer_episodes(
                question,
                intent,
                list(
                    dict.fromkeys(
                        [question, *intent.search_queries, *followup_queries]
                    )
                ),
                base_episodes,
                base_episode_limit,
                [*constraint_candidate_ids, *episode_anchor_ids],
                paragraph_context_by_source,
                required_candidate_ids=(
                    evidence_floor_trace["selected_episode_ids"]
                ),
                answer_slot_anchor_ids=[
                    int(value)
                    for slot in evidence_floor_trace.get(
                        "constraint_slots", []
                    )
                    if str(slot.get("query", "")).startswith(
                        "__answer_slot__ "
                    )
                    for value in slot.get("floor_episode_ids", [])
                ],
                association_cue_entries=active_cue_entries,
            )
            self._record_phase(
                phase_seconds,
                "evidence_rerank",
                phase_started,
            )
            phase_started = perf_counter()
            evidence_floor_trace = self._admit_evidence_floor_trace(
                evidence_floor_trace,
                rerank_trace.get("required_evidence_floor_ids", []),
            )
            rerank_trace["deterministic_evidence_floor"] = evidence_floor_trace
            rerank_trace["constraint_candidate_ids"] = constraint_candidate_ids
            self._record_phase(
                phase_seconds,
                "evidence_preparation",
                phase_started,
            )
        frozen_coverage_groups = (
            rerank_trace.get("merged_coverage", {})
            .get("coverage", [])
        )
        counterfactual_replay_plan = (
            frozen_plan
            if frozen_plan is not None
            else {
                "version": 4,
                "question": question,
                "intent": asdict(intent),
                "initial_queries": initial_queries,
                "followup_queries": followup_queries,
                "final_seed_hits": self._serialize_hits(seeds),
                "episode_anchor_ids": episode_anchor_ids,
                "association_cue_entries": active_cue_entries,
                "association_cue_association_ids": active_cue_ids,
                "reranked_episode_ids": reranked_episode_ids,
                "rerank_trace": rerank_trace,
                "configuration": self._replay_configuration(),
            }
        )
        phase_started = perf_counter()
        ensure_deadline("association_growth")
        for round_index in range(self.config.retrieval.growth_max_rounds):
            episodes, concepts = self._materialize_nodes(traversed)
            growth_episode_limit = min(
                self.config.retrieval.growth_episode_limit,
                len(episodes),
            )
            growth_episodes, growth_paths = self._select_answer_evidence(
                episodes,
                paths,
                growth_episode_limit,
                30,
                set([*all_new_associations, *all_reinforced_associations]),
                question,
                reranked_episode_ids or episode_anchor_ids,
                self.config.retrieval.learned_bridge_slots,
                self.config.retrieval.learned_bridge_min_query_relevance,
                self.config.retrieval.learned_bridge_duplicate_threshold,
                frozen_coverage_groups,
            )
            if self.logger:
                self.logger.emit(
                    "growth_evidence_selected",
                    question=question,
                    round=round_index + 1,
                    episode_limit=growth_episode_limit,
                    episode_ids=[item["id"] for item in growth_episodes],
                    path_association_ids=[
                        item.get("association_id") for item in growth_paths
                    ],
                )
            growth_nodes = [
                *[
                    {
                        "type": "episode",
                        "id": item["id"],
                        "text": item["text"],
                        "participants": item.get("participants", []),
                        "source_key": item.get("source_key", ""),
                    }
                    for item in growth_episodes
                ],
                *[
                    {
                        "type": "concept",
                        "id": item["id"],
                        "text": f"{item['canonical_name']}：{item['description']}",
                    }
                    for item in concepts[:20]
                ],
            ]
            outcome = self.growth.grow(
                question,
                growth_nodes,
                growth_paths,
                seen_growth_fingerprints,
            )
            all_new_associations.extend(outcome.created_ids)
            all_reinforced_associations.extend(outcome.reinforced_ids)
            for association_id, row in outcome.reinforced_before.items():
                reinforced_before.setdefault(int(association_id), dict(row))
            if not outcome.changed:
                break
            if self.logger:
                self.logger.emit(
                    "graph_expansion",
                    question=question,
                    round=round_index + 1,
                    created_association_ids=outcome.created_ids,
                    reinforced_association_ids=outcome.reinforced_ids,
                )
            traversed, paths = self.traverser.expand(
                seeds,
                self.config.retrieval.graph_beam_width,
                self.config.retrieval.graph_max_hops,
            )
            paths.extend(
                self._explicit_association_paths(active_cue_ids, cue_scores)
            )
            traversed = self._truncate_traversed_nodes(
                traversed,
                self.config.retrieval.candidate_limit,
                cue_fast_endpoint_keys,
            )
        self._record_phase(
            phase_seconds,
            "association_growth",
            phase_started,
        )
        phase_started = perf_counter()
        ensure_deadline("final_selection")
        late_nodes, late_paths = self._late_learned_bridge_closure(
            question,
            reranked_episode_ids or episode_anchor_ids,
            traversed,
        )
        if late_nodes:
            traversed = [*traversed, *late_nodes]
        if late_paths:
            paths.extend(late_paths)
        episodes, concepts = self._materialize_nodes(traversed, include_sources=True)
        changed_association_ids = [
            *all_new_associations,
            *all_reinforced_associations,
        ]
        paths.extend(
            self._explicit_association_paths(
                [*active_cue_ids, *changed_association_ids],
                cue_scores,
            )
        )
        episode_limit = min(self.config.retrieval.answer_episode_limit, len(episodes))
        capsule_episode_ids = {
            int(value)
            for capsule in rerank_trace.get("association_capsules", [])
            for value in capsule.get("endpoint_episode_ids", [])
        }
        if association_capsule_fast_lane:
            # An audited capsule already names the small premise closure that
            # answered an earlier query.  Sending the ordinary broad Top-K to
            # the chat model defeats its purpose as a latency/token cache.
            # Keep a few slots for evidence floors, but cap the prompt-facing
            # evidence independently of the broad candidate pool.
            episode_limit = min(
                episode_limit,
                max(
                    len(capsule_episode_ids),
                    int(
                        self.config.retrieval.association_cue_fast_path_evidence_limit
                    ),
                ),
            )
        selected_episodes, answer_paths = self._select_answer_evidence(
            episodes,
            paths,
            episode_limit,
            self.config.retrieval.answer_path_limit,
            set([*active_cue_ids, *changed_association_ids]),
            question,
            reranked_episode_ids or episode_anchor_ids,
            self.config.retrieval.learned_bridge_slots,
            self.config.retrieval.learned_bridge_min_query_relevance,
            self.config.retrieval.learned_bridge_duplicate_threshold,
            frozen_coverage_groups,
            cue_scores,
            capsule_episode_ids,
        )
        # M2/M3: apply deterministic slot coverage after the ordinary
        # candidate/rerank work is frozen.  This gives the masked selector and
        # contextual treatment exactly the same base candidate universe.
        contextual_started = perf_counter()
        if frozen_plan is None:
            selected_episodes, contextual_trace = self._select_contextual_slots(
                episodes=episodes,
                baseline_selected=selected_episodes,
                rerank_trace=rerank_trace,
                reranked_episode_ids=reranked_episode_ids,
                bundle=query_vector_bundle,
                domain=contextual_domain,
                endpoint_limit=contextual_endpoint_limit,
                anchor_activations=base_anchor_activations,
            )
        self._record_phase(
            phase_seconds,
            "contextual_association",
            contextual_started,
        )
        if self.config.retrieval.graph_max_hops > 0:
            chronology = self.chronology.order(
                [item["id"] for item in selected_episodes]
            )
            episode_map = {item["id"]: item for item in selected_episodes}
            ordered_episodes = [
                episode_map[node_id]
                for node_id in chronology.ordered_ids
                if node_id in episode_map
            ]
            chronology_notes = chronology.notes
        else:
            ordered_episodes = selected_episodes
            chronology_notes = [
                "纯向量基线：未读取 Association，Episode 保持向量相似度顺序。"
            ]
        self._record_phase(
            phase_seconds,
            "final_selection",
            phase_started,
        )
        provisional_used_edge_ids = list(
            dict.fromkeys(
                int(item["association_id"])
                for item in answer_paths
                if "association_id" in item
            )
        )
        phase_started = perf_counter()
        ensure_deadline("answer_generation")
        growth_counterfactual_utility = self._counterfactual_growth_utility(
            counterfactual_replay_plan,
            all_new_associations,
            all_reinforced_associations,
            reinforced_before,
        )
        changed_set = {
            int(value)
            for value in [*all_new_associations, *all_reinforced_associations]
        }
        persistable_changed = {
            int(value)
            for value in growth_counterfactual_utility.get(
                "persistable_changed_ids", []
            )
        }
        persistence_edge_ids = [
            value
            for value in provisional_used_edge_ids
            if value not in changed_set or value in persistable_changed
        ]
        if self.config.retrieval.growth_persist_only_used:
            growth_utility_gate = self._prune_unused_growth(
                all_new_associations,
                all_reinforced_associations,
                reinforced_before,
                persistence_edge_ids,
            )
            all_new_associations = list(
                growth_utility_gate["retained_created_ids"]
            )
            all_reinforced_associations = list(
                growth_utility_gate["retained_reinforced_ids"]
            )
        else:
            growth_utility_gate = {
                "enabled": False,
                "retained_created_ids": list(
                    dict.fromkeys(all_new_associations)
                ),
                "retained_reinforced_ids": list(
                    dict.fromkeys(all_reinforced_associations)
                ),
                "removed_created_ids": [],
                "restored_reinforced_ids": [],
                "premise_closure_ids": [],
            }
        durable_changed_ids = {
            int(value)
            for value in [*all_new_associations, *all_reinforced_associations]
        }
        removed_changed_ids = changed_set.difference(durable_changed_ids)
        if removed_changed_ids:
            answer_paths = [
                item
                for item in answer_paths
                if int(item.get("association_id", -1)) not in removed_changed_ids
            ]
            if self.config.retrieval.graph_max_hops > 0:
                chronology = self.chronology.order(
                    [item["id"] for item in selected_episodes]
                )
                episode_map = {item["id"]: item for item in selected_episodes}
                ordered_episodes = [
                    episode_map[node_id]
                    for node_id in chronology.ordered_ids
                    if node_id in episode_map
                ]
                chronology_notes = chronology.notes
        used_edge_ids = list(
            dict.fromkeys(
                int(item["association_id"])
                for item in answer_paths
                if "association_id" in item
            )
        )
        self.associations.mark_used(used_edge_ids)
        self._record_phase(
            phase_seconds,
            "growth_utility",
            phase_started,
        )
        phase_started = perf_counter()
        if generate_answer:
            answer, answer_audits, answer_revision_count = (
                self._generate_audited_answer(
                    question,
                    intent,
                    ordered_episodes,
                    concepts[: self.config.retrieval.answer_concept_limit],
                    answer_paths,
                    chronology_notes,
                )
            )
        else:
            answer, answer_audits, answer_revision_count = "", [], 0
        self._record_phase(
            phase_seconds,
            "answer_generation",
            phase_started,
        )
        evidence_slot_trace = self._final_evidence_slot_trace(
            rerank_trace,
            [int(item["id"]) for item in ordered_episodes],
        )
        evidence_slot_trace["slot_selector_v2"] = {
            key: deepcopy(value)
            for key, value in contextual_trace.items()
            if key in {
                "slots",
                "masked_selected_episode_ids",
                "masked_missing_slots",
                "treatment_selected_episode_ids",
                "new_slots",
                "lost_slots",
                "new_slot_count",
                "harm",
                "strict_attribution",
                "shadow",
                "reason",
            }
        }
        result = {
            "question": question,
            "query_plan_frozen": frozen_plan is not None,
            "query_plan_id": (
                str(frozen_plan["plan_id"]) if frozen_plan is not None else None
            ),
            "intent": asdict(intent),
            "followup_search_queries": followup_queries,
            "followup_planner_invoked": followup_planner_invoked,
            "followup_planning_mode": (
                self.config.retrieval.followup_planning_mode
            ),
            "followup_planning_reason": followup_planning_reason,
            "atomic_anchor_episode_ids": episode_anchor_ids,
            "candidate_episode_ids": [
                int(value)
                for value in (
                    rerank_trace.get("candidate_episode_ids")
                    or [
                        item.node_id
                        for item in traversed
                        if item.node_type == "episode"
                    ][: self.config.retrieval.rerank_candidate_limit]
                )
            ],
            "reranked_episode_ids": reranked_episode_ids,
            "rerank_trace": rerank_trace,
            "evidence_slot_trace": evidence_slot_trace,
            "rerank_frozen_before_growth": True,
            "source_key_cohort": source_key_cohort_trace,
            "answer": answer,
            "answer_audits": answer_audits,
            "answer_revision_count": answer_revision_count,
            "answer_generation_skipped": not generate_answer,
            "episode_ids": [item["id"] for item in ordered_episodes],
            "concept_ids": [
                item["id"]
                for item in concepts[: self.config.retrieval.answer_concept_limit]
            ],
            "association_ids": used_edge_ids,
            "new_association_ids": list(dict.fromkeys(all_new_associations)),
            "reinforced_association_ids": list(
                dict.fromkeys(all_reinforced_associations)
            ),
            "growth_utility_gate": growth_utility_gate,
            "growth_counterfactual_utility": growth_counterfactual_utility,
            "association_cue_ids": active_cue_ids,
            "association_cue_entries": active_cue_entries,
            "association_capsule_fast_lane": association_capsule_fast_lane,
            "contextual_association": {
                "enabled": bool(contextual_trace.get("enabled", False)),
                "backend": str(contextual_trace.get("backend", "contextual_double_key")),
                "context_prototype_hits": len(contextual_trace.get("context_hits", [])),
                "need_prototype_hits": sum(
                    len(item) for item in contextual_trace.get("need_hits", [])
                ),
                "candidate_count": len(contextual_trace.get("hits", [])),
                "selected_count": len(
                    contextual_trace.get("strict_attribution", [])
                ),
                "treatment_selected_count": int(
                    contextual_trace.get("selected_count", 0)
                ),
                "shadow": bool(contextual_trace.get("shadow", False)),
                "new_slot_count": int(contextual_trace.get("new_slot_count", 0)),
                "harm_count": int(contextual_trace.get("harm_count", 0)),
                "attached_edges": [
                    int(value) for value in contextual_trace.get("attached_edges", [])
                ],
                "attached_episode_ids": [
                    int(value) for value in contextual_trace.get("attached_episode_ids", [])
                ],
                "masked_episode_ids": [
                    int(value)
                    for value in contextual_trace.get(
                        "masked_selected_episode_ids", []
                    )
                ],
                "treatment_episode_ids": [
                    int(value)
                    for value in contextual_trace.get(
                        "treatment_selected_episode_ids", []
                    )
                ],
                "external_calls": 0,
                "context_query_id": str(
                    contextual_trace.get("context_query_id", "")
                ),
                "query_vectors": list(
                    contextual_trace.get("query_vectors", [])
                ),
                "base_episode_ids": base_episode_ids_before_contextual,
                "contextual_episode_ids": [
                    int(value)
                    for value in contextual_trace.get(
                        "attached_episode_ids", []
                    )
                ],
                # Utility learning consumes only this deterministic
                # single-edge/leave-one-out receipt; it never infers success
                # from the final answer prose or an unmasked batch result.
                "strict_attribution": list(
                    contextual_trace.get("strict_attribution", [])
                ),
            },
            "chronology_notes": chronology_notes,
            "evidence_episodes": [
                {
                    key: item[key]
                    for key in (
                        "id",
                        "score",
                        "text",
                        "participants",
                        "source_key",
                        "segment_index",
                        "story_time_text",
                        "story_order",
                        "timeline_scope",
                        "evidence_origin",
                        "epistemic_status",
                        "generation",
                        "epistemic_note",
                    )
                }
                for item in ordered_episodes
            ],
            "evidence_concepts": [
                {
                    key: item[key]
                    for key in ("id", "score", "canonical_name", "description")
                }
                for item in concepts[: self.config.retrieval.answer_concept_limit]
            ],
            "association_paths": answer_paths,
            "paragraph_retrieval_enabled": self.paragraph_retrieval_enabled,
            "sparse_retrieval_enabled": self.sparse_retrieval_enabled,
            "query_embedding_cache": {
                "hit_count": len(
                    set(self.last_query_embedding_cache_trace["hits"])
                ),
                "miss_count": len(
                    set(self.last_query_embedding_cache_trace["misses"])
                ),
            },
        }
        total_seconds = round(perf_counter() - query_started_at, 6)
        measured_seconds = round(sum(phase_seconds.values()), 6)
        result["timings"] = {
            "version": "query-stage-timing-v1",
            "phases_seconds": phase_seconds,
            "measured_seconds": measured_seconds,
            "unattributed_seconds": round(
                max(0.0, total_seconds - measured_seconds),
                6,
            ),
            "total_seconds": total_seconds,
            "deadline_seconds": deadline_seconds,
        }
        if self.logger:
            self.logger.emit(
                "answer_generated" if generate_answer else "retrieval_completed",
                result=result,
            )
        return result
