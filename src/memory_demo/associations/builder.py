from __future__ import annotations

import re
from typing import Any

from memory_demo.config import WeightConfig
from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm.prompts import (
    RELATION_SYSTEM,
    concept_relation_batch_prompt,
    concept_relation_prompt,
    episode_relation_batch_prompt,
    episode_relation_prompt,
)
from memory_demo.llm.client import ModelClientError, ModelTransportUnavailable
from memory_demo.llm.validation import (
    parse_concept_relation_batch_payload,
    parse_episode_relation_batch_payload,
    parse_relationships,
)
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    TemporalCycleError,
)
from memory_demo.types import AssociationDraft, ConceptDraft


class AssociationBuilder:
    _EPISODE_IDENTITY_PERSON_MARKERS = (
        "同一人物",
        "同一个人",
        "同一角色",
        "指向同一人物",
        "身份相同",
    )
    _EPISODE_IDENTITY_EVENT_MARKERS = (
        "同一事件",
        "同一场景",
        "同一情节",
        "同一段",
        "更详细描述",
    )
    _ORDERED_MAIN_SOURCE = re.compile(r"(?:^|/)main/(\d+)\.json$")
    _EXPLICIT_RECALL_MARKERS = (
        "想起",
        "回想",
        "回忆起",
        "勾起",
        "唤起",
        "记起",
        "忆起",
        "떠올",
        "생각났",
        "기억났",
        "思い出",
        "思い起",
    )

    def __init__(
        self,
        model,
        association_repository: AssociationRepository,
        episode_repository: EpisodeRepository,
        concept_repository: ConceptRepository,
        weights: WeightConfig,
        logger: JsonlEventLogger | None = None,
    ):
        self.model = model
        self.associations = association_repository
        self.episodes = episode_repository
        self.concepts = concept_repository
        self.weights = weights
        self.logger = logger

    def _weight(self, llm_score: float, embedding_score: float, structural: float = 0.0) -> float:
        normalized_similarity = max(0.0, min(1.0, (embedding_score + 1.0) / 2.0))
        return max(
            0.0,
            min(
                1.0,
                self.weights.llm * llm_score
                + self.weights.embedding * normalized_similarity
                + self.weights.structural * structural
                + self.weights.evidence,
            ),
        )

    @staticmethod
    def _episode_evidence_payload(row) -> dict[str, Any]:
        return {
            "evidence_origin": str(row["evidence_origin"] or "unknown"),
            "epistemic_status": str(row["epistemic_status"] or "unknown"),
            "generation": int(row["generation"] or 0),
            "epistemic_note": str(row["epistemic_note"] or ""),
        }

    def link_episode_concept(
        self,
        episode_id: int,
        concept_id: int,
        confidence: float,
        concept_name: str,
        *,
        reinforce_existing: bool = True,
    ) -> int:
        episode_row = self.episodes.get(episode_id)
        episode_generation = (
            int(episode_row["generation"] or 0) if episode_row is not None else 0
        )
        draft = self._episode_concept_draft(
            episode_id,
            concept_id,
            confidence,
            concept_name,
            episode_generation,
        )
        existing_id = (
            self.associations.find_exact_id(draft)
            if not reinforce_existing
            else None
        )
        association_id = (
            existing_id
            if existing_id is not None and not reinforce_existing
            else self.associations.upsert(draft)
        )
        if self.logger:
            self.logger.emit(
                "association_reused"
                if existing_id is not None and not reinforce_existing
                else "association_created",
                association_id=association_id,
                draft=draft,
            )
        return association_id

    def _episode_concept_draft(
        self,
        episode_id: int,
        concept_id: int,
        confidence: float,
        concept_name: str,
        episode_generation: int,
    ) -> AssociationDraft:
        return AssociationDraft(
            from_type="episode",
            from_id=episode_id,
            to_type="concept",
            to_id=concept_id,
            relation_type="semantic",
            relation_key="involves",
            relation_text=f"该 Episode 涉及概念“{concept_name}”",
            weight=self._weight(confidence, 0.0, structural=1.0),
            confidence=confidence,
            generation=episode_generation,
            created_reason="由 Episode 的 Concept 提取结果建立",
        )

    def link_new_episode_concepts(
        self,
        links: list[tuple[int, int, float, str, int]],
    ) -> list[int]:
        """Persist direct edges for newly allocated Episodes in one batch."""

        unique: dict[tuple[int, int], tuple[int, int, float, str, int]] = {}
        for link in links:
            key = (int(link[0]), int(link[1]))
            current = unique.get(key)
            if current is None or float(link[2]) > float(current[2]):
                unique[key] = link
        drafts = [
            self._episode_concept_draft(
                episode_id,
                concept_id,
                confidence,
                concept_name,
                generation,
            )
            for episode_id, concept_id, confidence, concept_name, generation in unique.values()
        ]
        association_ids = self.associations.insert_new_non_temporal_many(drafts)
        if self.logger:
            for association_id, draft in zip(
                association_ids, drafts, strict=True
            ):
                self.logger.emit(
                    "association_created",
                    association_id=association_id,
                    draft=draft,
                )
        return association_ids

    def judge_episode_candidates(
        self,
        episode_id: int,
        candidates: list[tuple[int, float]],
    ) -> list[AssociationDraft]:
        current_row = self.episodes.get(episode_id)
        filtered = [(node_id, score) for node_id, score in candidates if node_id != episode_id]
        rows = {int(row["id"]): row for row in self.episodes.get_many(node for node, _ in filtered)}
        if current_row is None or not rows:
            return []
        score_map = dict(filtered)
        candidate_payload = [
            {
                "id": node_id,
                "text": rows[node_id]["text"],
                "story_time_text": rows[node_id]["story_time_text"],
                "similarity": score,
                **self._episode_evidence_payload(rows[node_id]),
            }
            for node_id, score in filtered
            if node_id in rows
        ]
        payload = self.model.chat_json(
            RELATION_SYSTEM,
            episode_relation_prompt(
                {
                    "id": episode_id,
                    "text": current_row["text"],
                    "story_time_text": current_row["story_time_text"],
                    **self._episode_evidence_payload(current_row),
                },
                candidate_payload,
            ),
        )
        relationships, errors = parse_relationships(payload)
        if errors and self.logger:
            self.logger.emit("validation_failed", stage="episode_relations", errors=errors)
        return self._draft_episode_relationships(
            episode_id, relationships, rows, score_map
        )

    def relate_episode_candidates(
        self,
        episode_id: int,
        candidates: list[tuple[int, float]],
    ) -> list[int]:
        return self.store_relation_drafts(
            self.judge_episode_candidates(episode_id, candidates)
        )

    def _draft_episode_relationships(
        self,
        episode_id: int,
        relationships: list[dict[str, Any]],
        rows: dict[int, Any],
        score_map: dict[int, float],
    ) -> list[AssociationDraft]:
        drafts: list[AssociationDraft] = []
        current_row = self.episodes.get(episode_id)
        for relation in relationships:
            candidate_id = relation["candidate_id"]
            if candidate_id not in rows:
                continue
            rejection_reason = self.episode_relationship_rejection_reason(
                relation, current_row, rows[candidate_id]
            )
            if rejection_reason:
                if self.logger:
                    self.logger.emit(
                        "association_rejected",
                        stage="episode_relations",
                        episode_id=episode_id,
                        candidate_id=candidate_id,
                        reason=rejection_reason,
                        relation=relation,
                    )
                continue
            from_id, to_id, relation_key = self.normalize_episode_relationship(
                episode_id, candidate_id, relation
            )
            draft = AssociationDraft(
                from_type="episode",
                from_id=from_id,
                to_type="episode",
                to_id=to_id,
                relation_type=relation["relation_type"],
                relation_key=relation_key,
                relation_text=relation["relation_text"],
                polarity=relation["polarity"],
                weight=self._weight(
                    relation["llm_score"], score_map.get(candidate_id, 0.0)
                ),
                confidence=relation["confidence"],
                generation=max(
                    int(current_row["generation"] or 0),
                    int(rows[candidate_id]["generation"] or 0),
                )
                + 1,
                claim_level="supported_inference",
                created_reason="导入阶段 Episode 候选关系判断",
            )
            drafts.append(draft)
        return drafts

    @classmethod
    def episode_relationship_rejection_reason(
        cls,
        relation: dict[str, Any],
        current_row: Any | None = None,
        candidate_row: Any | None = None,
    ) -> str | None:
        """Keep Episode identity edges about duplicate events, not shared personas."""
        relation_type = relation.get("relation_type")
        polarity = int(relation.get("polarity", 1))
        if relation_type == "recall_trigger" and polarity > 0:
            endpoint_text = " ".join(
                cls._episode_text(row) for row in (current_row, candidate_row)
            )
            if not any(marker in endpoint_text for marker in cls._EXPLICIT_RECALL_MARKERS):
                return (
                    "recall_trigger requires an Episode that explicitly describes "
                    "remembering or a memory being evoked"
                )
        if relation_type == "temporal" and polarity > 0:
            contradiction = cls._temporal_provenance_contradiction(
                relation, current_row, candidate_row
            )
            if contradiction:
                return contradiction
        if relation_type != "identity" or polarity < 0:
            return None
        relation_text = str(relation.get("relation_text", ""))
        if any(marker in relation_text for marker in cls._EPISODE_IDENTITY_PERSON_MARKERS):
            return "positive Episode identity may not assert that two speakers are one person"
        if not any(marker in relation_text for marker in cls._EPISODE_IDENTITY_EVENT_MARKERS):
            return "positive Episode identity must explicitly describe the same event or scene"
        return None

    @staticmethod
    def _episode_text(row: Any | None) -> str:
        if row is None:
            return ""
        try:
            return str(row["text"] or "")
        except (KeyError, IndexError, TypeError):
            return ""

    @classmethod
    def _temporal_provenance_contradiction(
        cls,
        relation: dict[str, Any],
        current_row: Any | None,
        candidate_row: Any | None,
    ) -> str | None:
        if current_row is None or candidate_row is None:
            return None
        if str(current_row["timeline_scope"] or "") != str(
            candidate_row["timeline_scope"] or ""
        ):
            return None
        if str(current_row["story_time_text"] or "").strip() or str(
            candidate_row["story_time_text"] or ""
        ).strip():
            return None
        current_key = str(current_row["source_key"] or "")
        candidate_key = str(candidate_row["source_key"] or "")
        if current_key == candidate_key:
            current_rank = (
                int(current_row["segment_index"]),
                int(current_row["id"]),
            )
            candidate_rank = (
                int(candidate_row["segment_index"]),
                int(candidate_row["id"]),
            )
        else:
            current_match = cls._ORDERED_MAIN_SOURCE.search(current_key)
            candidate_match = cls._ORDERED_MAIN_SOURCE.search(candidate_key)
            if not current_match or not candidate_match:
                return None
            current_rank = (int(current_match.group(1)), int(current_row["id"]))
            candidate_rank = (int(candidate_match.group(1)), int(candidate_row["id"]))
        key = str(relation.get("relation_key", "")).casefold()
        claims_after = key == "after" or key.endswith("_after")
        claims_before = key in {"before", "precedes"} or key.endswith("_before")
        if claims_after and current_rank < candidate_rank:
            return "temporal direction contradicts current-event source provenance order"
        if claims_before and current_rank > candidate_rank:
            return "temporal direction contradicts current-event source provenance order"
        return None

    @staticmethod
    def normalize_episode_relationship(
        episode_id: int, candidate_id: int, relation: dict[str, Any]
    ) -> tuple[int, int, str]:
        """Canonicalize positive temporal edges as earlier --before--> later."""
        relation_type = str(relation.get("relation_type", ""))
        relation_key = str(relation.get("relation_key", ""))
        if relation_type != "temporal" or int(relation.get("polarity", 1)) < 0:
            return episode_id, candidate_id, relation_key
        key = relation_key.casefold()
        if key == "after" or key.endswith("_after"):
            return candidate_id, episode_id, "before"
        if key in {"before", "precedes"} or key.endswith("_before"):
            return episode_id, candidate_id, "before"
        return episode_id, candidate_id, relation_key

    def judge_episode_batches(
        self,
        candidate_groups: dict[int, list[tuple[int, float]]],
    ) -> list[AssociationDraft]:
        active_groups = {
            episode_id: [
                (candidate_id, score)
                for candidate_id, score in candidates
                if candidate_id != episode_id
            ]
            for episode_id, candidates in candidate_groups.items()
        }
        active_groups = {
            episode_id: candidates
            for episode_id, candidates in active_groups.items()
            if candidates
        }
        if not active_groups:
            return []
        current_rows = {
            int(row["id"]): row
            for row in self.episodes.get_many(active_groups.keys())
        }
        all_candidate_ids = {
            candidate_id
            for candidates in active_groups.values()
            for candidate_id, _score in candidates
        }
        candidate_rows = {
            int(row["id"]): row
            for row in self.episodes.get_many(all_candidate_ids)
        }
        prompt_items: list[dict[str, Any]] = []
        valid_groups: dict[int, list[tuple[int, float]]] = {}
        for episode_id, candidates in active_groups.items():
            current = current_rows.get(episode_id)
            if current is None:
                continue
            candidate_payload = [
                {
                    "id": candidate_id,
                    "text": candidate_rows[candidate_id]["text"],
                    "story_time_text": candidate_rows[candidate_id]["story_time_text"],
                    "similarity": score,
                    **self._episode_evidence_payload(candidate_rows[candidate_id]),
                }
                for candidate_id, score in candidates
                if candidate_id in candidate_rows
            ]
            if not candidate_payload:
                continue
            valid_groups[episode_id] = candidates
            prompt_items.append(
                {
                    "current": {
                        "id": episode_id,
                        "text": current["text"],
                        "story_time_text": current["story_time_text"],
                        **self._episode_evidence_payload(current),
                    },
                    "candidates": candidate_payload,
                }
            )
        if not prompt_items:
            return []
        try:
            batch_request_options = (
                {"allow_fallback": False, "max_retries": 0}
                if len(prompt_items) > 1
                else {}
            )
            payload = self.model.chat_json(
                RELATION_SYSTEM,
                episode_relation_batch_prompt(prompt_items),
                **batch_request_options,
            )
        except ModelTransportUnavailable:
            raise
        except ModelClientError as exc:
            if len(active_groups) <= 1:
                raise
            items = list(active_groups.items())
            midpoint = len(items) // 2
            if self.logger:
                self.logger.emit(
                    "relation_batch_split",
                    stage="episode_relations_batch",
                    groups=len(items),
                    left_groups=midpoint,
                    right_groups=len(items) - midpoint,
                    error=str(exc),
                )
            return self.judge_episode_batches(dict(items[:midpoint])) + self.judge_episode_batches(
                dict(items[midpoint:])
            )
        relationships_by_id, group_errors, global_errors = (
            parse_episode_relation_batch_payload(payload, set(valid_groups))
        )
        if (group_errors or global_errors) and self.logger:
            self.logger.emit(
                "validation_failed",
                stage="episode_relations_batch",
                group_errors=group_errors,
                errors=global_errors,
            )
        drafts: list[AssociationDraft] = []
        for episode_id, candidates in valid_groups.items():
            if episode_id in group_errors:
                drafts.extend(self.judge_episode_candidates(episode_id, candidates))
                continue
            rows = {
                candidate_id: candidate_rows[candidate_id]
                for candidate_id, _score in candidates
                if candidate_id in candidate_rows
            }
            drafts.extend(
                self._draft_episode_relationships(
                    episode_id,
                    relationships_by_id.get(episode_id, []),
                    rows,
                    dict(candidates),
                )
            )
        return drafts

    def relate_episode_batches(
        self,
        candidate_groups: dict[int, list[tuple[int, float]]],
    ) -> list[int]:
        return self.store_relation_drafts(
            self.judge_episode_batches(candidate_groups)
        )

    def judge_new_concept(
        self,
        concept_id: int,
        draft: ConceptDraft,
        candidates: list[tuple[int, float]],
    ) -> list[AssociationDraft]:
        filtered = [(node_id, score) for node_id, score in candidates if node_id != concept_id]
        rows = {int(row["id"]): row for row in self.concepts.get_many(node for node, _ in filtered)}
        if not rows:
            return []
        score_map = dict(filtered)
        payload = self.model.chat_json(
            RELATION_SYSTEM,
            concept_relation_prompt(
                {
                    "id": concept_id,
                    "canonical_name": draft.canonical_name,
                    "description": draft.description,
                },
                [
                    {
                        "id": node_id,
                        "canonical_name": rows[node_id]["canonical_name"],
                        "description": rows[node_id]["description"],
                        "similarity": score,
                    }
                    for node_id, score in filtered
                    if node_id in rows
                ],
            ),
        )
        relationships, errors = parse_relationships(payload)
        if errors and self.logger:
            self.logger.emit("validation_failed", stage="concept_relations", errors=errors)
        return self._draft_concept_relationships(
            concept_id, relationships, rows, score_map
        )

    def relate_new_concept(
        self,
        concept_id: int,
        draft: ConceptDraft,
        candidates: list[tuple[int, float]],
    ) -> list[int]:
        return self.store_relation_drafts(
            self.judge_new_concept(concept_id, draft, candidates)
        )

    def _draft_concept_relationships(
        self,
        concept_id: int,
        relationships: list[dict[str, Any]],
        rows: dict[int, Any],
        score_map: dict[int, float],
    ) -> list[AssociationDraft]:
        drafts: list[AssociationDraft] = []
        for relation in relationships:
            candidate_id = relation["candidate_id"]
            if candidate_id not in rows:
                continue
            draft = AssociationDraft(
                from_type="concept",
                from_id=concept_id,
                to_type="concept",
                to_id=candidate_id,
                relation_type=relation["relation_type"],
                relation_key=relation["relation_key"],
                relation_text=relation["relation_text"],
                polarity=relation["polarity"],
                weight=self._weight(
                    relation["llm_score"], score_map.get(candidate_id, 0.0)
                ),
                confidence=relation["confidence"],
                generation=1,
                claim_level="supported_inference",
                created_reason="新 Concept 与已有 Concept 的关系判断",
            )
            drafts.append(draft)
        return drafts

    def judge_new_concept_batches(
        self,
        jobs: list[tuple[int, ConceptDraft, list[tuple[int, float]]]],
    ) -> list[AssociationDraft]:
        if not jobs:
            return []
        all_candidate_ids = {
            candidate_id
            for _concept_id, _draft, candidates in jobs
            for candidate_id, _score in candidates
        }
        candidate_rows = {
            int(row["id"]): row
            for row in self.concepts.get_many(all_candidate_ids)
        }
        valid_jobs: dict[
            int, tuple[ConceptDraft, list[tuple[int, float]]]
        ] = {}
        prompt_items: list[dict[str, Any]] = []
        for concept_id, draft, candidates in jobs:
            filtered = [
                (candidate_id, score)
                for candidate_id, score in candidates
                if candidate_id != concept_id and candidate_id in candidate_rows
            ]
            if not filtered:
                continue
            valid_jobs[concept_id] = (draft, filtered)
            prompt_items.append(
                {
                    "current": {
                        "id": concept_id,
                        "canonical_name": draft.canonical_name,
                        "description": draft.description,
                    },
                    "candidates": [
                        {
                            "id": candidate_id,
                            "canonical_name": candidate_rows[candidate_id]["canonical_name"],
                            "description": candidate_rows[candidate_id]["description"],
                            "similarity": score,
                        }
                        for candidate_id, score in filtered
                    ],
                }
            )
        if not prompt_items:
            return []
        try:
            batch_request_options = (
                {"allow_fallback": False, "max_retries": 0}
                if len(prompt_items) > 1
                else {}
            )
            payload = self.model.chat_json(
                RELATION_SYSTEM,
                concept_relation_batch_prompt(prompt_items),
                **batch_request_options,
            )
        except ModelTransportUnavailable:
            raise
        except ModelClientError as exc:
            if len(jobs) <= 1:
                raise
            midpoint = len(jobs) // 2
            if self.logger:
                self.logger.emit(
                    "relation_batch_split",
                    stage="concept_relations_batch",
                    groups=len(jobs),
                    left_groups=midpoint,
                    right_groups=len(jobs) - midpoint,
                    error=str(exc),
                )
            return self.judge_new_concept_batches(
                jobs[:midpoint]
            ) + self.judge_new_concept_batches(jobs[midpoint:])
        relationships_by_id, group_errors, global_errors = (
            parse_concept_relation_batch_payload(payload, set(valid_jobs))
        )
        if (group_errors or global_errors) and self.logger:
            self.logger.emit(
                "validation_failed",
                stage="concept_relations_batch",
                group_errors=group_errors,
                errors=global_errors,
            )
        drafts: list[AssociationDraft] = []
        for concept_id, (draft, candidates) in valid_jobs.items():
            if concept_id in group_errors:
                drafts.extend(self.judge_new_concept(concept_id, draft, candidates))
                continue
            rows = {
                candidate_id: candidate_rows[candidate_id]
                for candidate_id, _score in candidates
            }
            drafts.extend(
                self._draft_concept_relationships(
                    concept_id,
                    relationships_by_id.get(concept_id, []),
                    rows,
                    dict(candidates),
                )
            )
        return drafts

    @staticmethod
    def relation_draft_sort_key(draft: AssociationDraft) -> tuple:
        """Order a decision batch independently of worker completion timing."""
        claim_priority = {
            "direct_fact": 0,
            "supported_inference": 1,
            "historical_context": 2,
        }.get(str(draft.claim_level), 3)
        relation_priority = {
            "temporal": 0,
            "causal": 1,
            "identity": 2,
            "recall_trigger": 3,
            "interpersonal": 4,
            "semantic": 5,
            "co_occurrence": 6,
        }.get(str(draft.relation_type), 7)
        return (
            int(draft.generation),
            claim_priority,
            -float(draft.confidence),
            -float(draft.weight),
            relation_priority,
            str(draft.from_type),
            int(draft.from_id),
            str(draft.to_type),
            int(draft.to_id),
            str(draft.relation_key),
            int(draft.polarity),
            str(draft.relation_text),
        )

    def store_relation_drafts(
        self, drafts: list[AssociationDraft]
    ) -> list[int]:
        """Persist already judged drafts on the caller's single writer thread."""
        created: list[int] = []
        for draft in drafts:
            try:
                association_id = self.associations.upsert(draft)
            except TemporalCycleError as exc:
                if self.logger:
                    self.logger.emit(
                        "association_rejected",
                        stage="temporal_dag_guard",
                        reason=str(exc),
                        earlier_episode_id=exc.earlier_id,
                        later_episode_id=exc.later_id,
                        draft=draft,
                    )
                continue
            created.append(association_id)
            if self.logger:
                self.logger.emit(
                    "association_upserted",
                    association_id=association_id,
                    draft=draft,
                )
        return created

    def relate_new_concept_batches(
        self,
        jobs: list[tuple[int, ConceptDraft, list[tuple[int, float]]]],
    ) -> list[int]:
        return self.store_relation_drafts(
            self.judge_new_concept_batches(jobs)
        )
