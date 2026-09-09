from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter
import traceback
from typing import Any

from memory_demo.adapters import BlueArchiveJsonAdapter, InputAdapter, TextAdapter
from memory_demo.adapters.base import logical_source_key
from memory_demo.associations import AssociationBuilder
from memory_demo.config import AppConfig
from memory_demo.concepts import ConceptResolver
from memory_demo.database import Database, DatabaseBusyError
from memory_demo.embeddings import EmbeddingIndex, encode_embedding, normalize_embedding
from memory_demo.event_log import JsonlEventLogger
from memory_demo.event_log import safe_config_snapshot
from memory_demo.ingestion.extractor import (
    EmptyEpisodeAudit,
    EmptyEpisodeExtraction,
    ExtractionValidationError,
    MemoryExtractor,
)
from memory_demo.ingestion.ordering import infer_timeline_scope, natural_path_sort_key
from memory_demo.ingestion.paragraphs import ParagraphSegmenter
from memory_demo.ingestion.segmenter import NaturalSegmenter
from memory_demo.llm import ModelTransportUnavailable
from memory_demo.llm.prompts import (
    CONCEPT_ADMISSION_SYSTEM,
    SECOND_PASS_SYSTEM,
    concept_admission_batch_prompt,
    second_pass_batch_prompt,
    second_pass_prompt,
    second_pass_retry_prompt,
)
from memory_demo.llm.validation import parse_concept_admission_payload
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    ExtractionRepository,
    ParagraphRepository,
    SourceRepository,
)
from memory_demo.repositories.concept import normalize_alias
from memory_demo.types import (
    AssociationDraft,
    ConceptDraft,
    EpisodeDraft,
    ParagraphDraft,
    SourceSegment,
)


class EmptyEpisodeSkipped(RuntimeError):
    """A source-only independent reviewer proved a clean empty result safe."""

    def __init__(self, audit: EmptyEpisodeAudit):
        super().__init__("independent empty Episode audit confirmed safe skip")
        self.audit = audit


class ImportPipeline:
    def __init__(
        self,
        config: AppConfig,
        db: Database,
        model,
        logger: JsonlEventLogger,
        episode_index: EmbeddingIndex | None = None,
        concept_index: EmbeddingIndex | None = None,
        paragraph_index: EmbeddingIndex | None = None,
        *,
        prepare_workers: int = 1,
        relation_workers: int = 1,
        relation_batch_size: int = 1,
        build_inference_relations: bool = True,
    ):
        self.config = config
        self.db = db
        self.model = model
        self.logger = logger
        if hasattr(self.model, "logger"):
            self.model.logger = logger
        self.sources = SourceRepository(db)
        self.episodes = EpisodeRepository(db)
        self.concepts = ConceptRepository(db)
        self.paragraphs = ParagraphRepository(db)
        self.associations = AssociationRepository(db, config.weights)
        self.extractions = ExtractionRepository(db)
        self.episode_index = episode_index or EmbeddingIndex(
            config.model.embedding_dimension
        )
        self.concept_index = concept_index or EmbeddingIndex(
            config.model.embedding_dimension
        )
        self.paragraph_index = paragraph_index or EmbeddingIndex(
            config.model.embedding_dimension
        )
        self.builder = AssociationBuilder(
            model,
            self.associations,
            self.episodes,
            self.concepts,
            config.weights,
            logger,
        )
        self.resolver = ConceptResolver(
            model,
            self.concepts,
            self.concept_index,
            self.builder,
            config.model.embedding_dimension,
            config.retrieval.concept_relation_min_similarity,
            logger,
        )
        self.extractor = MemoryExtractor(
            model,
            logger,
            concept_profile=config.concept_extraction.profile,
            concept_target_min=config.concept_extraction.target_min,
            concept_target_max=config.concept_extraction.target_max,
            episode_audit_mode=config.ingestion.episode_audit_mode,
            episode_audit_always=config.ingestion.episode_audit_always,
            episode_factual_audit_mode=(config.ingestion.episode_factual_audit_mode),
            episode_factual_audit_model=(config.ingestion.episode_factual_audit_model),
            episode_factual_audit_batch_size=(
                config.ingestion.episode_factual_audit_batch_size
            ),
            empty_episode_audit_mode=config.ingestion.empty_episode_audit_mode,
            empty_episode_audit_model=config.ingestion.empty_episode_audit_model,
            episode_extraction_profile=(config.ingestion.episode_extraction_profile),
        )
        self.segmenter = NaturalSegmenter(config.segment)
        self.paragraph_segmenter = ParagraphSegmenter(config.paragraph)
        self.adapters: list[InputAdapter] = [BlueArchiveJsonAdapter(), TextAdapter()]
        self.prepare_workers = max(1, int(prepare_workers))
        self.relation_workers = max(1, int(relation_workers))
        self.relation_batch_size = max(1, int(relation_batch_size))
        self.build_inference_relations = bool(build_inference_relations)
        self.task_max_attempts = max(1, int(config.ingestion.task_max_attempts))
        self._active_relation_executor: ThreadPoolExecutor | None = None

    def _background_provider_scope(self):
        """Mark import-originated model work as background when supported.

        ``ContextVar`` bindings do not cross the preparation/relation worker
        threads. Their worker entrypoints bind this same scope again, while
        ordinary lightweight test models simply receive a no-op context.
        """

        bind_workload = getattr(
            getattr(self, "model", None), "provider_workload", None
        )
        if not callable(bind_workload):
            return nullcontext()
        return bind_workload("background")

    def _shutdown_relation_executor(self, *, wait: bool) -> None:
        executor = self._active_relation_executor
        self._active_relation_executor = None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=not wait)

    def rebuild_indexes(self) -> None:
        self.episode_index.rebuild(
            self.episodes.count(), self.episodes.iter_embeddings()
        )
        self.concept_index.rebuild(
            self.concepts.count(), self.concepts.iter_embeddings()
        )
        self.paragraph_index.rebuild(
            self.paragraphs.count(), self.paragraphs.iter_embeddings()
        )

    def _insert_paragraphs(
        self,
        source_id: int,
        source_key: str,
        segment_index: int,
        source_text: str,
        *,
        prepared_drafts: list[ParagraphDraft] | None = None,
        prepared_vectors: list[Any] | None = None,
    ) -> list[int]:
        drafts = (
            self.paragraph_segmenter.segment(source_text)
            if prepared_drafts is None
            else prepared_drafts
        )
        if not drafts:
            return []
        embedding_texts = [
            self.paragraph_segmenter.embedding_text(draft.text) for draft in drafts
        ]
        if prepared_vectors is not None and len(prepared_vectors) != len(drafts):
            raise ValueError("paragraph drafts and vectors must have equal length")
        matrix = (
            self.model.embed(embedding_texts)
            if prepared_vectors is None
            else prepared_vectors
        )
        vectors = [
            normalize_embedding(row, self.config.model.embedding_dimension)
            for row in matrix
        ]
        ids = self.paragraphs.insert_many(
            source_id,
            source_key,
            segment_index,
            drafts,
            [
                encode_embedding(vector, self.config.model.embedding_dimension)
                for vector in vectors
            ],
        )
        for paragraph_id, vector in zip(ids, vectors, strict=True):
            self.paragraph_index.upsert(paragraph_id, vector)
        self.logger.emit(
            "paragraphs_created",
            source_id=source_id,
            source_key=source_key,
            segment_index=segment_index,
            paragraph_ids=ids,
            paragraph_lengths=[len(draft.text) for draft in drafts],
            paragraph_embedding_lengths=[len(text) for text in embedding_texts],
        )
        return ids

    def backfill_paragraphs(self) -> dict[str, int]:
        """Populate only missing Sources, making the feature safe to retry."""
        summary = {"sources": 0, "paragraphs": 0, "failed_sources": 0}
        if not self.config.paragraph.enabled:
            return summary
        for row in self.paragraphs.sources_without_paragraphs():
            try:
                ids = self._insert_paragraphs(
                    int(row["id"]),
                    str(row["source_key"]),
                    int(row["segment_index"]),
                    str(row["raw_text"]),
                )
                summary["sources"] += 1
                summary["paragraphs"] += len(ids)
            except Exception as exc:
                summary["failed_sources"] += 1
                self.logger.emit(
                    "paragraph_backfill_failed",
                    source_id=int(row["id"]),
                    source_key=str(row["source_key"]),
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
        self.logger.emit("paragraph_backfill_finished", summary=summary)
        return summary

    def backfill_paragraphs_batched(self, batch_size: int = 64) -> dict[str, int]:
        """Backfill Paragraphs with batched embedding calls for large assets."""
        summary = {"sources": 0, "paragraphs": 0, "failed_sources": 0}
        if not self.config.paragraph.enabled:
            return summary
        prepared: list[tuple[object, list, list[str]]] = []
        for row in self.paragraphs.sources_without_paragraphs():
            try:
                drafts = self.paragraph_segmenter.segment(str(row["raw_text"]))
                texts = [
                    self.paragraph_segmenter.embedding_text(draft.text)
                    for draft in drafts
                ]
                prepared.append((row, drafts, texts))
            except Exception as exc:
                summary["failed_sources"] += 1
                self.logger.emit(
                    "paragraph_backfill_failed",
                    source_id=int(row["id"]),
                    source_key=str(row["source_key"]),
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )

        flat_texts = [text for _row, _drafts, texts in prepared for text in texts]
        vectors: list[object | None] = [None] * len(flat_texts)
        size = max(1, int(batch_size))
        for start in range(0, len(flat_texts), size):
            batch = flat_texts[start : start + size]
            try:
                matrix = self.model.embed(batch)
                for offset, raw_vector in enumerate(matrix):
                    vectors[start + offset] = normalize_embedding(
                        raw_vector, self.config.model.embedding_dimension
                    )
            except Exception as exc:
                self.logger.emit(
                    "paragraph_embedding_batch_failed",
                    start=start,
                    size=len(batch),
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )

        cursor = 0
        for row, drafts, texts in prepared:
            source_vectors = vectors[cursor : cursor + len(texts)]
            cursor += len(texts)
            if any(vector is None for vector in source_vectors):
                summary["failed_sources"] += 1
                self.logger.emit(
                    "paragraph_backfill_failed",
                    source_id=int(row["id"]),
                    source_key=str(row["source_key"]),
                    error="one or more Paragraph embedding batches failed",
                )
                continue
            ids = self.paragraphs.insert_many(
                int(row["id"]),
                str(row["source_key"]),
                int(row["segment_index"]),
                drafts,
                [
                    encode_embedding(vector, self.config.model.embedding_dimension)
                    for vector in source_vectors
                ],
            )
            for paragraph_id, vector in zip(ids, source_vectors, strict=True):
                self.paragraph_index.upsert(paragraph_id, vector)
            summary["sources"] += 1
            summary["paragraphs"] += len(ids)
            self.logger.emit(
                "paragraphs_created",
                source_id=int(row["id"]),
                source_key=str(row["source_key"]),
                segment_index=int(row["segment_index"]),
                paragraph_ids=ids,
                paragraph_lengths=[len(draft.text) for draft in drafts],
                paragraph_embedding_lengths=[len(text) for text in texts],
                batched=True,
            )
        self.logger.emit("paragraph_backfill_finished", summary=summary, batched=True)
        return summary

    def augment_concepts(self) -> dict[str, int | str]:
        """Run the selected Concept profile over persisted Episodes once per DB copy."""
        if self.config.concept_extraction.promotion_gate_enabled:
            return self._augment_concepts_with_promotion_gate()
        before_count = self.concepts.count()
        summary: dict[str, int | str] = {
            "profile": self.config.concept_extraction.profile,
            "sources": 0,
            "episodes": 0,
            "concept_candidates": 0,
            "concepts_before": before_count,
            "concepts_after": before_count,
            "failed_sources": 0,
        }
        source_jobs: list[tuple[int, object, list[object], list[EpisodeDraft]]] = []
        for source_id in self.episodes.list_source_ids():
            source = self.sources.get(source_id)
            rows = list(self.episodes.list_by_source_id(source_id))
            if source is None or not rows:
                continue
            drafts = [
                EpisodeDraft.from_dict(self._episode_row_payload(row)) for row in rows
            ]
            source_jobs.append((source_id, source, rows, drafts))

        executor = (
            ThreadPoolExecutor(
                max_workers=self.prepare_workers,
                thread_name_prefix="concept-augmentation",
            )
            if self.prepare_workers > 1
            else None
        )
        futures: dict[int, Future] = {}
        if executor is not None:
            for source_id, source, _rows, drafts in source_jobs:
                futures[source_id] = executor.submit(
                    self.extractor.extract_concepts_batch,
                    drafts,
                    str(source["raw_text"]),
                )

        pending_relation_jobs: list[
            tuple[int, ConceptDraft, list[tuple[int, float]]]
        ] = []
        for source_id, source, rows, drafts in source_jobs:
            try:
                groups, errors = (
                    futures[source_id].result()
                    if executor is not None
                    else self.extractor.extract_concepts_batch(
                        drafts, str(source["raw_text"])
                    )
                )
                if errors:
                    self.logger.emit(
                        "validation_failed",
                        stage="concept_augmentation",
                        source_id=source_id,
                        errors=errors,
                    )
                pending_links = [
                    (int(rows[position]["id"]), concept_draft)
                    for position, concept_drafts in enumerate(groups)
                    for concept_draft in concept_drafts
                ]
                concept_ids, relation_jobs = self.resolver.resolve_many_deferred(
                    [draft for _episode_id, draft in pending_links]
                )
                for (episode_id, concept_draft), concept_id in zip(
                    pending_links, concept_ids, strict=True
                ):
                    self.builder.link_episode_concept(
                        episode_id,
                        concept_id,
                        concept_draft.confidence,
                        concept_draft.canonical_name,
                        reinforce_existing=False,
                    )
                pending_relation_jobs.extend(relation_jobs)
                summary["sources"] = int(summary["sources"]) + 1
                summary["episodes"] = int(summary["episodes"]) + len(rows)
                summary["concept_candidates"] = int(
                    summary["concept_candidates"]
                ) + len(pending_links)
            except Exception as exc:
                summary["failed_sources"] = int(summary["failed_sources"]) + 1
                self.logger.emit(
                    "concept_augmentation_failed",
                    source_id=source_id,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

        relation_batch_size = max(16, self.relation_batch_size)
        for start in range(0, len(pending_relation_jobs), relation_batch_size):
            self.builder.relate_new_concept_batches(
                pending_relation_jobs[start : start + relation_batch_size]
            )
        summary["concept_relation_jobs"] = len(pending_relation_jobs)
        summary["concepts_after"] = self.concepts.count()
        self.logger.emit("concept_augmentation_finished", summary=summary)
        return summary

    @staticmethod
    def _merge_concept_occurrences(occurrences: list[dict]) -> ConceptDraft:
        drafts = [item["draft"] for item in occurrences]
        representative = max(
            drafts,
            key=lambda draft: (
                draft.confidence,
                len(draft.description),
                len(draft.embedding_text),
            ),
        )
        aliases: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for draft in drafts:
            for alias, language in draft.aliases:
                key = (normalize_alias(alias), str(language).casefold())
                if alias and key not in seen:
                    seen.add(key)
                    aliases.append((alias, language))
        return ConceptDraft(
            canonical_name=representative.canonical_name,
            description=representative.description,
            embedding_text=representative.embedding_text,
            aliases=aliases,
            confidence=max(draft.confidence for draft in drafts),
        )

    def _existing_concept_id_for_draft(self, draft: ConceptDraft) -> int | None:
        for alias in [draft.canonical_name, *(value for value, _ in draft.aliases)]:
            rows = self.concepts.find_by_alias(alias)
            if not rows:
                continue
            active = next((row for row in rows if row["status"] == "active"), rows[0])
            return int(active["canonical_concept_id"] or active["id"])
        return None

    def _concept_admission_prompt_item(self, item: dict) -> dict:
        if not self.config.concept_extraction.promotion_compact_prompt:
            return item
        concept = item["concept"]
        evidence_limit = max(
            1,
            int(self.config.concept_extraction.promotion_evidence_text_limit),
        )
        evidence_chars = max(
            80,
            int(self.config.concept_extraction.promotion_evidence_text_chars),
        )
        description_chars = max(
            40,
            int(self.config.concept_extraction.promotion_similar_description_chars),
        )
        return {
            "candidate_index": int(item["candidate_index"]),
            "concept": {
                "canonical_name": str(concept.get("canonical_name", "")),
                "description": str(concept.get("description", ""))[:evidence_chars],
                "aliases": list(concept.get("aliases", []))[:8],
                "confidence": float(concept.get("confidence", 0.0)),
            },
            "episode_occurrence_count": int(item["episode_occurrence_count"]),
            "source_occurrence_count": int(item["source_occurrence_count"]),
            "evidence_episode_texts": [
                str(text)[:evidence_chars]
                for text in item.get("evidence_episode_texts", [])[:evidence_limit]
            ],
            "similar_concepts": [
                {
                    **similar,
                    "description": str(similar.get("description", ""))[
                        :description_chars
                    ],
                }
                for similar in item.get("similar_concepts", [])
            ],
        }

    def _augment_concepts_with_promotion_gate(self) -> dict[str, int | str]:
        """Discover broadly, then admit only durable Concepts to the graph."""
        before_count = self.concepts.count()
        summary: dict[str, int | str] = {
            "profile": self.config.concept_extraction.profile,
            "promotion_gate": "prefiltered_llm_durable_v2",
            "sources": 0,
            "episodes": 0,
            "concept_candidates": 0,
            "candidate_groups": 0,
            "existing_groups_reused": 0,
            "promoted_groups": 0,
            "transient_groups": 0,
            "llm_reused_groups": 0,
            "llm_admission_groups": 0,
            "prefiltered_transient_groups": 0,
            "admission_batches": 0,
            "admission_validation_errors": 0,
            "concepts_before": before_count,
            "concepts_after": before_count,
            "failed_sources": 0,
        }
        source_jobs: list[tuple[int, object, list[object], list[EpisodeDraft]]] = []
        for source_id in self.episodes.list_source_ids():
            source = self.sources.get(source_id)
            rows = list(self.episodes.list_by_source_id(source_id))
            if source is None or not rows:
                continue
            drafts = [
                EpisodeDraft.from_dict(self._episode_row_payload(row)) for row in rows
            ]
            source_jobs.append((source_id, source, rows, drafts))

        executor = (
            ThreadPoolExecutor(
                max_workers=self.prepare_workers,
                thread_name_prefix="concept-discovery",
            )
            if self.prepare_workers > 1
            else None
        )
        futures: dict[int, Future] = {}
        if executor is not None:
            for source_id, source, _rows, drafts in source_jobs:
                futures[source_id] = executor.submit(
                    self.extractor.extract_concepts_batch,
                    drafts,
                    str(source["raw_text"]),
                )

        occurrences_by_name: dict[str, list[dict]] = {}
        for source_id, source, rows, drafts in source_jobs:
            try:
                groups, errors = (
                    futures[source_id].result()
                    if executor is not None
                    else self.extractor.extract_concepts_batch(
                        drafts, str(source["raw_text"])
                    )
                )
                if errors:
                    self.logger.emit(
                        "validation_failed",
                        stage="concept_discovery",
                        source_id=source_id,
                        errors=errors,
                    )
                for position, concept_drafts in enumerate(groups):
                    episode_id = int(rows[position]["id"])
                    for concept_draft in concept_drafts:
                        key = normalize_alias(concept_draft.canonical_name)
                        occurrences_by_name.setdefault(key, []).append(
                            {
                                "source_id": source_id,
                                "episode_id": episode_id,
                                "episode_text": str(rows[position]["text"]),
                                "draft": concept_draft,
                            }
                        )
                        summary["concept_candidates"] = (
                            int(summary["concept_candidates"]) + 1
                        )
                summary["sources"] = int(summary["sources"]) + 1
                summary["episodes"] = int(summary["episodes"]) + len(rows)
            except Exception as exc:
                summary["failed_sources"] = int(summary["failed_sources"]) + 1
                self.logger.emit(
                    "concept_discovery_failed",
                    source_id=source_id,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

        grouped = [
            {
                "key": key,
                "occurrences": occurrences,
                "draft": self._merge_concept_occurrences(occurrences),
            }
            for key, occurrences in sorted(occurrences_by_name.items())
        ]
        summary["candidate_groups"] = len(grouped)
        new_groups: list[dict] = []
        for group in grouped:
            draft = group["draft"]
            existing_id = self._existing_concept_id_for_draft(draft)
            if existing_id is None:
                new_groups.append(group)
                continue
            self.concepts.add_aliases(
                existing_id,
                [(draft.canonical_name, "unknown"), *draft.aliases],
                draft.confidence,
            )
            for occurrence in group["occurrences"]:
                self.builder.link_episode_concept(
                    int(occurrence["episode_id"]),
                    existing_id,
                    occurrence["draft"].confidence,
                    draft.canonical_name,
                    reinforce_existing=False,
                )
            summary["existing_groups_reused"] = (
                int(summary["existing_groups_reused"]) + 1
            )
            self.logger.emit(
                "concept_admission_existing_reuse",
                concept_id=existing_id,
                draft=draft,
                episode_ids=[item["episode_id"] for item in group["occurrences"]],
            )

        vectors: list = []
        if new_groups:
            for start in range(0, len(new_groups), 64):
                matrix = self.model.embed(
                    [
                        group["draft"].embedding_text
                        for group in new_groups[start : start + 64]
                    ]
                )
                vectors.extend(
                    normalize_embedding(row, self.config.model.embedding_dimension)
                    for row in matrix
                )

        admission_items: list[dict] = []
        allowed_reuse_ids: dict[int, set[int]] = {}
        for candidate_index, (group, vector) in enumerate(
            zip(new_groups, vectors, strict=True)
        ):
            ranked = self.concept_index.search(vector, 5)
            rows = {
                int(row["id"]): row
                for row in self.concepts.get_many(node_id for node_id, _ in ranked)
            }
            similar = [
                {
                    "id": int(node_id),
                    "canonical_name": str(rows[node_id]["canonical_name"]),
                    "description": str(rows[node_id]["description"]),
                    "similarity": float(score),
                }
                for node_id, score in ranked
                if node_id in rows
            ]
            allowed_reuse_ids[candidate_index] = {int(item["id"]) for item in similar}
            occurrences = group["occurrences"]
            admission_items.append(
                {
                    "candidate_index": candidate_index,
                    "concept": asdict(group["draft"]),
                    "episode_occurrence_count": len(
                        {int(item["episode_id"]) for item in occurrences}
                    ),
                    "source_occurrence_count": len(
                        {int(item["source_id"]) for item in occurrences}
                    ),
                    "evidence_episode_texts": list(
                        dict.fromkeys(str(item["episode_text"]) for item in occurrences)
                    )[:3],
                    "similar_concepts": similar,
                }
            )

        minimum_episode_count = max(
            1,
            int(self.config.concept_extraction.promotion_min_distinct_episodes),
        )
        decisions: dict[int, dict] = {}
        llm_admission_items: list[dict] = []
        reuse_floor = float(
            self.config.concept_extraction.promotion_singleton_reuse_similarity_floor
        )
        for item in admission_items:
            recurring = int(item["episode_occurrence_count"]) >= minimum_episode_count
            top_similarity = max(
                (
                    float(similar.get("similarity", -1.0))
                    for similar in item.get("similar_concepts", [])
                ),
                default=-1.0,
            )
            if (
                self.config.concept_extraction.promotion_prefilter_enabled
                and not recurring
                and top_similarity < reuse_floor
            ):
                candidate_index = int(item["candidate_index"])
                decisions[candidate_index] = {
                    "candidate_index": candidate_index,
                    "action": "transient",
                    "existing_concept_id": None,
                    "reason": (
                        "确定性预筛选：候选未达到跨 Episode 图效用门，且最高已有 Concept "
                        f"相似度 {top_similarity:.6f} 低于安全复用门 {reuse_floor:.6f}"
                    ),
                    "confidence": 1.0,
                    "decision_source": "deterministic_prefilter",
                }
                summary["prefiltered_transient_groups"] = (
                    int(summary["prefiltered_transient_groups"]) + 1
                )
            else:
                llm_admission_items.append(item)

        summary["llm_admission_groups"] = len(llm_admission_items)
        batch_size = max(1, int(self.config.concept_extraction.promotion_batch_size))
        admission_batches = [
            llm_admission_items[start : start + batch_size]
            for start in range(0, len(llm_admission_items), batch_size)
        ]
        summary["admission_batches"] = len(admission_batches)

        def evaluate_admission_batch(
            batch: list[dict],
        ) -> tuple[dict, list[str], set[int]]:
            expected = {int(item["candidate_index"]) for item in batch}
            prompt_batch = [self._concept_admission_prompt_item(item) for item in batch]
            payload = self.model.chat_json(
                CONCEPT_ADMISSION_SYSTEM,
                concept_admission_batch_prompt(prompt_batch),
            )
            parsed, errors = parse_concept_admission_payload(payload, expected)
            return parsed, errors, expected

        if self.prepare_workers > 1 and len(admission_batches) > 1:
            with ThreadPoolExecutor(
                max_workers=self.prepare_workers,
                thread_name_prefix="concept-admission",
            ) as admission_executor:
                admission_futures = [
                    admission_executor.submit(evaluate_admission_batch, batch)
                    for batch in admission_batches
                ]
                admission_results = [future.result() for future in admission_futures]
        else:
            admission_results = [
                evaluate_admission_batch(batch) for batch in admission_batches
            ]

        for parsed, errors, expected in admission_results:
            decisions.update(parsed)
            if errors:
                summary["admission_validation_errors"] = int(
                    summary["admission_validation_errors"]
                ) + len(errors)
                self.logger.emit(
                    "validation_failed",
                    stage="concept_admission",
                    candidate_indexes=sorted(expected),
                    errors=errors,
                )

        relation_jobs: list[tuple[int, ConceptDraft, list[tuple[int, float]]]] = []
        recurrence_fallback = max(
            1,
            int(self.config.concept_extraction.promotion_recurrence_fallback),
        )
        for candidate_index, (group, vector, item) in enumerate(
            zip(new_groups, vectors, admission_items, strict=True)
        ):
            decision = decisions.get(candidate_index)
            if decision is None:
                recurring = int(item["episode_occurrence_count"]) >= recurrence_fallback
                decision = {
                    "candidate_index": candidate_index,
                    "action": "promote" if recurring else "transient",
                    "existing_concept_id": None,
                    "reason": "准入响应缺失，使用预注册的重复出现保守回退规则",
                    "confidence": 0.0,
                }
            action = str(decision["action"])
            if (
                action == "promote"
                and int(item["episode_occurrence_count"]) < minimum_episode_count
            ):
                action = "transient"
                decision = {
                    **decision,
                    "action": action,
                    "reason": (
                        str(decision.get("reason", ""))
                        + f"；仅覆盖 {item['episode_occurrence_count']} 个独立 Episode，"
                        f"低于图效用门 {minimum_episode_count}"
                    ).strip("；"),
                }
            concept_id: int | None = None
            candidates: list[tuple[int, float]] = []
            if action == "reuse":
                proposed_id = int(decision["existing_concept_id"])
                if proposed_id not in allowed_reuse_ids[candidate_index]:
                    action = "transient"
                    decision = {
                        **decision,
                        "action": action,
                        "reason": (
                            str(decision.get("reason", ""))
                            + "；复用 ID 不在相似候选中，安全降级为 transient"
                        ).strip("；"),
                    }
                else:
                    concept_id = proposed_id
                    draft = group["draft"]
                    self.concepts.add_aliases(
                        concept_id,
                        [(draft.canonical_name, "unknown"), *draft.aliases],
                        draft.confidence,
                    )
                    summary["llm_reused_groups"] = int(summary["llm_reused_groups"]) + 1
            elif action == "promote":
                concept_id, candidates = self.resolver.resolve_preembedded_deferred(
                    group["draft"], vector
                )
                if candidates:
                    relation_jobs.append((concept_id, group["draft"], candidates))
                summary["promoted_groups"] = int(summary["promoted_groups"]) + 1

            if concept_id is None:
                summary["transient_groups"] = int(summary["transient_groups"]) + 1
            else:
                for occurrence in group["occurrences"]:
                    self.builder.link_episode_concept(
                        int(occurrence["episode_id"]),
                        concept_id,
                        occurrence["draft"].confidence,
                        group["draft"].canonical_name,
                        reinforce_existing=False,
                    )
            self.logger.emit(
                "concept_admission_decision",
                decision=decision,
                draft=group["draft"],
                episode_ids=[item["episode_id"] for item in group["occurrences"]],
                admitted_concept_id=concept_id,
                similar_concepts=item["similar_concepts"],
            )

        relation_batch_size = max(16, self.relation_batch_size)
        if self.build_inference_relations:
            for start in range(0, len(relation_jobs), relation_batch_size):
                self.builder.relate_new_concept_batches(
                    relation_jobs[start : start + relation_batch_size]
                )
        elif relation_jobs:
            self.logger.emit(
                "inference_relations_deferred",
                stage="concept_augmentation",
                concept_candidate_groups=len(relation_jobs),
                episode_candidate_groups=0,
            )
        summary["concept_relation_jobs"] = len(relation_jobs)
        summary["concepts_after"] = self.concepts.count()
        self.logger.emit("concept_augmentation_finished", summary=summary)
        return summary

    def _adapter_for(self, path: Path) -> InputAdapter | None:
        return next(
            (adapter for adapter in self.adapters if adapter.supports(path)), None
        )

    def _input_files(self, path: Path) -> tuple[Path, list[Path], list[Path]]:
        if path.is_file():
            if self._adapter_for(path) is None:
                return path.parent, [], [path]
            return path.parent, [path], []
        all_files = sorted(
            (item for item in path.rglob("*") if item.is_file()),
            key=natural_path_sort_key,
        )
        files = [item for item in all_files if self._adapter_for(item) is not None]
        unsupported = [item for item in all_files if self._adapter_for(item) is None]
        return path, files, unsupported

    @staticmethod
    def _failure_detail(
        stage: str,
        source_key: str,
        error: BaseException | str,
        *,
        segment_index: int | None = None,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "stage": stage,
            "source_key": source_key,
            "error_type": (
                type(error).__name__
                if isinstance(error, BaseException)
                else "ImportError"
            ),
            "error": str(error),
        }
        if segment_index is not None:
            detail["segment_index"] = int(segment_index)
        if task_id is not None:
            detail["task_id"] = int(task_id)
        return detail

    def _prepare_source_embeddings(
        self,
        source_text: str,
        episode_drafts: list[EpisodeDraft],
        concept_groups: list[list[ConceptDraft]],
    ) -> tuple[
        list[ParagraphDraft],
        list[Any],
        list[Any],
        list[Any],
    ]:
        """Embed every persisted retrieval view of one Source in one request.

        The program owns batching and slicing.  The embedding provider only
        receives a flat list of ordinary texts and is never asked to infer a
        schema or change models on failure.
        """

        started_at = perf_counter()
        paragraph_drafts = self.paragraph_segmenter.segment(source_text)
        paragraph_texts = [
            self.paragraph_segmenter.embedding_text(draft.text)
            for draft in paragraph_drafts
        ]
        concept_drafts = [draft for group in concept_groups for draft in group]
        episode_texts = [draft.text for draft in episode_drafts]
        concept_texts = [draft.embedding_text for draft in concept_drafts]
        texts = [*paragraph_texts, *episode_texts, *concept_texts]
        if not texts:
            raise ValueError("source produced no text requiring an embedding")

        matrix = self.model.embed(texts)
        if len(matrix) != len(texts):
            raise ValueError(
                "embedding provider returned "
                f"{len(matrix)} vectors for {len(texts)} texts"
            )
        vectors = [
            normalize_embedding(row, self.config.model.embedding_dimension)
            for row in matrix
        ]
        paragraph_end = len(paragraph_drafts)
        episode_end = paragraph_end + len(episode_drafts)
        result = (
            paragraph_drafts,
            vectors[:paragraph_end],
            vectors[paragraph_end:episode_end],
            vectors[episode_end:],
        )
        self.logger.emit(
            "source_embedding_batch_prepared",
            paragraph_count=len(paragraph_drafts),
            episode_count=len(episode_drafts),
            concept_count=len(concept_drafts),
            input_count=len(texts),
            seconds=round(perf_counter() - started_at, 6),
        )
        return result

    def _insert_episode_batch(
        self,
        run_id: int,
        source_id: int,
        source_key: str,
        segment_index: int,
        drafts: list[EpisodeDraft],
        prepared_concept_groups: list[list[ConceptDraft]] | None = None,
        prepared_concept_errors: list[str] | None = None,
        prepared_episode_vectors: list[Any] | None = None,
        prepared_concept_vectors: list[Any] | None = None,
    ) -> tuple[
        list[int],
        list[tuple[int, ConceptDraft, list[tuple[int, float]]]],
        dict[int, list[tuple[int, float]]],
    ]:
        if not drafts:
            return [], [], {}
        batch_started_at = perf_counter()
        if prepared_episode_vectors is not None and len(
            prepared_episode_vectors
        ) != len(drafts):
            raise ValueError("episode drafts and vectors must have equal length")
        matrix = (
            self.model.embed([draft.text for draft in drafts])
            if prepared_episode_vectors is None
            else prepared_episode_vectors
        )
        vectors = [
            normalize_embedding(row, self.config.model.embedding_dimension)
            for row in matrix
        ]
        episode_ids: list[int] = []
        for draft, vector in zip(drafts, vectors, strict=True):
            episode_id = self.episodes.insert(
                source_id,
                source_key,
                segment_index,
                draft,
                encode_embedding(vector, self.config.model.embedding_dimension),
                extraction_run_id=run_id,
            )
            self.episode_index.upsert(episode_id, vector)
            episode_ids.append(episode_id)
            self.logger.emit(
                "episode_created",
                episode_id=episode_id,
                source_id=source_id,
                draft=draft,
            )
        episodes_finished_at = perf_counter()

        if prepared_concept_groups is None:
            source_row = self.sources.get(source_id)
            source_text = str(source_row["raw_text"]) if source_row is not None else ""
            concept_groups, concept_errors = self.extractor.extract_concepts_batch(
                drafts, source_text
            )
        else:
            concept_groups = prepared_concept_groups
            concept_errors = list(prepared_concept_errors or [])
        if concept_errors:
            self.logger.emit(
                "validation_failed",
                stage="concepts_batch",
                errors=concept_errors,
            )
        pending_links = [
            (episode_ids[position], concept_draft)
            for position, concept_drafts in enumerate(concept_groups)
            for concept_draft in concept_drafts
        ]
        concept_drafts = [
            concept_draft for _episode_id, concept_draft in pending_links
        ]
        if prepared_concept_vectors is None:
            concept_ids, concept_relation_jobs = self.resolver.resolve_many_deferred(
                concept_drafts
            )
        else:
            if len(prepared_concept_vectors) != len(concept_drafts):
                raise ValueError("concept drafts and vectors must have equal length")
            concept_ids, concept_relation_jobs = (
                self.resolver.resolve_many_preembedded_deferred(
                    concept_drafts,
                    prepared_concept_vectors,
                )
            )
        concepts_resolved_at = perf_counter()
        episode_generations = {
            episode_id: int(draft.generation)
            for episode_id, draft in zip(episode_ids, drafts, strict=True)
        }
        self.builder.link_new_episode_concepts(
            [
                (
                    episode_id,
                    concept_id,
                    concept_draft.confidence,
                    concept_draft.canonical_name,
                    episode_generations[episode_id],
                )
                for (episode_id, concept_draft), concept_id in zip(
                    pending_links, concept_ids, strict=True
                )
            ]
        )
        concepts_linked_at = perf_counter()
        new_id_set = set(episode_ids)
        candidate_groups: dict[int, list[tuple[int, float]]] = {}
        for episode_id, vector in zip(episode_ids, vectors, strict=True):
            candidates = self.episode_index.search(
                vector,
                top_k=self.config.retrieval.episode_relation_candidate_k,
            )
            # Match streaming semantics: compare a new Episode with older
            # memories and earlier Episodes from this batch, not both pair
            # directions inside the same Source.
            candidate_groups[episode_id] = [
                (candidate_id, score)
                for candidate_id, score in candidates
                if candidate_id not in new_id_set or candidate_id < episode_id
            ]
        candidates_finished_at = perf_counter()
        self.logger.emit(
            "episode_batch_persisted",
            source_id=source_id,
            episode_count=len(episode_ids),
            concept_link_count=len(pending_links),
            concept_relation_job_count=len(concept_relation_jobs),
            seconds={
                "episode_insert": round(
                    episodes_finished_at - batch_started_at, 6
                ),
                "concept_resolve": round(
                    concepts_resolved_at - episodes_finished_at, 6
                ),
                "concept_link": round(
                    concepts_linked_at - concepts_resolved_at, 6
                ),
                "episode_candidate_search": round(
                    candidates_finished_at - concepts_linked_at, 6
                ),
                "total": round(candidates_finished_at - batch_started_at, 6),
            },
        )
        return episode_ids, concept_relation_jobs, candidate_groups

    def _flush_relation_batches(
        self,
        concept_jobs: list[tuple[int, ConceptDraft, list[tuple[int, float]]]],
        episode_groups: list[tuple[int, list[tuple[int, float]]]],
        *,
        force: bool,
        executor: ThreadPoolExecutor | None = None,
        pending_futures: list[Future[list[AssociationDraft]]] | None = None,
    ) -> None:
        if not self.build_inference_relations:
            deferred_concepts = len(concept_jobs)
            deferred_episodes = len(episode_groups)
            concept_jobs.clear()
            episode_groups.clear()
            if deferred_concepts or deferred_episodes:
                self.logger.emit(
                    "inference_relations_deferred",
                    concept_candidate_groups=deferred_concepts,
                    episode_candidate_groups=deferred_episodes,
                )
            return
        batch_size = self.relation_batch_size
        ready: list[tuple[str, object]] = []
        while len(concept_jobs) >= batch_size or (force and concept_jobs):
            take = min(batch_size, len(concept_jobs))
            chunk = concept_jobs[:take]
            del concept_jobs[:take]
            ready.append(("concept", chunk))
        while len(episode_groups) >= batch_size or (force and episode_groups):
            take = min(batch_size, len(episode_groups))
            chunk = episode_groups[:take]
            del episode_groups[:take]
            ready.append(("episode", dict(chunk)))
        if not ready and not pending_futures:
            return

        def execute(item: tuple[str, object]) -> list[AssociationDraft]:
            with self._background_provider_scope():
                kind, payload = item
                if kind == "concept":
                    return self.builder.judge_new_concept_batches(payload)
                return self.builder.judge_episode_batches(payload)

        judged: list[AssociationDraft] = []
        if executor is not None:
            if pending_futures is None:
                raise ValueError("pending_futures is required with a relation executor")
            pending_futures.extend(executor.submit(execute, item) for item in ready)
            errors: list[BaseException] = []
            max_pending = max(1, self.relation_workers * 2)
            while pending_futures and (
                force
                or pending_futures[0].done()
                or len(pending_futures) >= max_pending
            ):
                future = pending_futures.pop(0)
                try:
                    drafts = future.result()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    drafts.sort(key=self.builder.relation_draft_sort_key)
                    self.builder.store_relation_drafts(drafts)
                if errors and not force:
                    break
            if errors:
                details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
                raise RuntimeError(
                    f"{len(errors)} relation judgement batch(es) failed: {details}"
                ) from errors[0]
            return

        if self.relation_workers <= 1 or len(ready) == 1:
            for item in ready:
                judged.extend(execute(item))
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.relation_workers, len(ready)),
                thread_name_prefix="relation-judge",
            ) as executor:
                futures = [executor.submit(execute, item) for item in ready]
                for future in as_completed(futures):
                    judged.extend(future.result())

        # Model judgment is parallel, but all durable graph mutations happen
        # here on one writer thread. Stable ordering makes conflict resolution
        # reproducible even when requests finish in a different order.
        judged.sort(key=self.builder.relation_draft_sort_key)
        self.builder.store_relation_drafts(judged)

    def _extract_segment_drafts(
        self,
        source_text: str,
        scope: str,
        document_context: str = "",
    ) -> tuple[
        list[EpisodeDraft],
        list[str],
        list[list[ConceptDraft]],
        list[str],
    ]:
        drafts, episode_errors = self.extractor.extract_episodes(
            source_text,
            scope,
            document_context,
        )
        if not drafts and not episode_errors:
            # A valid empty list is neither a transport failure nor a malformed
            # candidate.  It has to pass the independent source-only audit
            # below; never let a caller interpret it as a harmless no-op.
            raise EmptyEpisodeExtraction("primary extraction returned a clean empty Episode list")
        for draft in drafts:
            # The corpus path owns chronology scope.  A model-produced value such as
            # "past" describes story_time_text, not a separate comparable timeline.
            draft.timeline_scope = scope
        concept_groups, concept_errors = self.extractor.extract_concepts_batch(
            drafts, source_text
        )
        return drafts, episode_errors, concept_groups, concept_errors

    def _extract_segment_drafts_with_retry(
        self,
        task_id: int,
        source_text: str,
        scope: str,
        document_context: str = "",
    ) -> tuple[
        list[EpisodeDraft],
        list[str],
        list[list[ConceptDraft]],
        list[str],
    ]:
        for attempt in range(1, self.task_max_attempts + 1):
            try:
                return self._extract_segment_drafts(
                    source_text,
                    scope,
                    document_context,
                )
            except ExtractionValidationError:
                # Single-pass extraction already performed primary, focused
                # repair and optional reasoning-fallback attempts. Replaying
                # that entire sequence is expensive and does not change a
                # deterministic evidence-contract failure.
                raise
            except ModelTransportUnavailable:
                # A local socket circuit is process-wide.  Do not convert it
                # into a per-segment failure or spend the outer task retry.
                raise
            except Exception as exc:
                if attempt >= self.task_max_attempts:
                    raise
                self.extractions.retry(task_id, str(exc))
                self.logger.emit(
                    "task_retry",
                    task_id=task_id,
                    stage="pass1",
                    attempt=attempt + 1,
                    max_attempts=self.task_max_attempts,
                    error=str(exc),
                )
        raise AssertionError("unreachable import retry state")

    def _prepare_segment_for_persistence(
        self,
        task_id: int,
        source_text: str,
        scope: str,
        document_context: str = "",
    ) -> tuple[
        list[EpisodeDraft],
        list[str],
        list[list[ConceptDraft]],
        list[str],
        list[ParagraphDraft],
        list[Any],
        list[Any],
        list[Any],
    ]:
        # This method commonly runs in a ThreadPoolExecutor, whose worker
        # context starts empty. Rebind the import workload before any model
        # extraction/embedding call can enter the shared provider gate.
        with self._background_provider_scope():
            return self._prepare_segment_for_persistence_unscoped(
                task_id,
                source_text,
                scope,
                document_context,
            )

    def _prepare_segment_for_persistence_unscoped(
        self,
        task_id: int,
        source_text: str,
        scope: str,
        document_context: str = "",
    ) -> tuple[
        list[EpisodeDraft],
        list[str],
        list[list[ConceptDraft]],
        list[str],
        list[ParagraphDraft],
        list[Any],
        list[Any],
        list[Any],
    ]:
        """Finish remote extraction and embedding before ordered persistence."""

        try:
            drafts, errors, concept_groups, concept_errors = (
                self._extract_segment_drafts_with_retry(
                    task_id,
                    source_text,
                    scope,
                    document_context,
                )
            )
        except EmptyEpisodeExtraction as empty_result:
            audit = self.extractor.review_empty_episode(
                source_text,
                attempted_models=empty_result.attempted_models,
            )
            include_quotes = (
                os.getenv("MEMORY_LOG_MODEL_PAYLOADS", "false").casefold()
                == "true"
            )
            self.logger.emit(
                "empty_episode_adversarial_audit_completed",
                task_id=task_id,
                **audit.event_payload(include_quotes=include_quotes),
            )
            if audit.verdict == "safe_skip":
                raise EmptyEpisodeSkipped(audit)
            if audit.verdict == "episode_required":
                drafts, errors = self.extractor.extract_after_empty_episode_review(
                    source_text,
                    scope,
                    audit,
                )
                for draft in drafts:
                    draft.timeline_scope = scope
                concept_groups, concept_errors = self.extractor.extract_concepts_batch(
                    drafts, source_text
                )
            else:
                raise ExtractionValidationError(
                    "empty Episode adversarial audit did not confirm a safe skip; "
                    + audit.compact_receipt()
                )
        if not drafts:
            raise ExtractionValidationError(
                "no valid Episode was extracted after candidate validation"
            )
        (
            paragraph_drafts,
            paragraph_vectors,
            episode_vectors,
            concept_vectors,
        ) = self._prepare_source_embeddings(
            source_text,
            drafts,
            concept_groups,
        )
        return (
            drafts,
            errors,
            concept_groups,
            concept_errors,
            paragraph_drafts,
            paragraph_vectors,
            episode_vectors,
            concept_vectors,
        )

    def _second_pass_draft(self, prompt: str) -> EpisodeDraft:
        errors: list[str] = []
        fallback = getattr(getattr(self.model, "config", None), "fallback_model", None)
        attempts: list[str | None] = [None, None]
        if fallback:
            attempts.append(str(fallback))
        for attempt, model_name in enumerate(attempts):
            active_prompt = prompt
            if errors:
                active_prompt = second_pass_retry_prompt(prompt, errors)
            kwargs = {}
            if model_name:
                kwargs = {"model": model_name, "allow_fallback": False}
            try:
                payload = self.model.chat_json(
                    SECOND_PASS_SYSTEM, active_prompt, **kwargs
                )
                raw_draft = (
                    payload.get("episode", {}) if isinstance(payload, dict) else {}
                )
                return self.extractor.sanitize_episode_draft(
                    EpisodeDraft.from_dict(raw_draft)
                )
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
                self.logger.emit(
                    "validation_failed",
                    stage="pass2",
                    attempt=attempt + 1,
                    errors=errors,
                )
        raise ValueError("; ".join(errors))

    def _second_pass_batch_drafts(
        self,
        items: list[dict],
        source_text: str,
        shared_context: list[dict] | None = None,
    ) -> tuple[dict[int, EpisodeDraft], list[str]]:
        expected_ids = {int(item["episode_id"]) for item in items}
        try:
            payload = self.model.chat_json(
                SECOND_PASS_SYSTEM,
                second_pass_batch_prompt(items, source_text, shared_context),
            )
        except Exception as exc:
            return {}, [str(exc)]
        raw_items = payload.get("episodes", []) if isinstance(payload, dict) else []
        if not isinstance(raw_items, list):
            return {}, ["episodes must be an array"]
        drafts: dict[int, EpisodeDraft] = {}
        errors: list[str] = []
        for position, raw_item in enumerate(raw_items):
            try:
                if not isinstance(raw_item, dict):
                    raise TypeError("item must be an object")
                episode_id = int(raw_item["episode_id"])
                if episode_id not in expected_ids:
                    raise ValueError(f"unexpected episode_id {episode_id}")
                if episode_id in drafts:
                    raise ValueError(f"duplicate episode_id {episode_id}")
                drafts[episode_id] = self.extractor.sanitize_episode_draft(
                    EpisodeDraft.from_dict(raw_item["episode"])
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"item {position}: {exc}")
        missing = sorted(expected_ids - drafts.keys())
        if missing:
            errors.append(f"missing episode_id values: {missing}")
        if errors:
            self.logger.emit("validation_failed", stage="pass2_batch", errors=errors)
        return drafts, errors

    @staticmethod
    def _episode_row_payload(row) -> dict:
        try:
            participants = json.loads(row["participants_json"])
        except (TypeError, json.JSONDecodeError):
            participants = []
        return {
            "text": row["text"],
            "participants": participants,
            "event_type": row["event_type"],
            "location_text": row["location_text"],
            "story_time_text": row["story_time_text"],
            "timeline_scope": row["timeline_scope"],
            "confidence": row["confidence"],
            "evidence_origin": row["evidence_origin"],
            "epistemic_status": row["epistemic_status"],
            "generation": row["generation"],
            "epistemic_note": row["epistemic_note"],
        }

    def _prepare_second_pass_group(
        self,
        targets: list[tuple[int, object, int]],
        summaries: list[dict],
        source_text: str,
    ) -> tuple[
        list[tuple[object, int, EpisodeDraft]],
        list[tuple[int, str]],
        list,
        list[list[ConceptDraft]],
        list[str],
    ]:
        # As with pass one, pass-two preparation may execute in a worker.
        with self._background_provider_scope():
            return self._prepare_second_pass_group_unscoped(
                targets,
                summaries,
                source_text,
            )

    def _prepare_second_pass_group_unscoped(
        self,
        targets: list[tuple[int, object, int]],
        summaries: list[dict],
        source_text: str,
    ) -> tuple[
        list[tuple[object, int, EpisodeDraft]],
        list[tuple[int, str]],
        list,
        list[list[ConceptDraft]],
        list[str],
    ]:
        compact_context = self.config.ingestion.second_pass_context_mode == "compact"
        reasoning_source = (
            self.extractor.compact_source_for_reasoning(source_text)
            if compact_context
            else source_text
        )
        context_positions = sorted(
            {
                index
                for position, _row, _task_id in targets
                for index in range(
                    max(0, position - 8), min(len(summaries), position + 9)
                )
            }
        )
        shared_context = (
            [summaries[index] for index in context_positions] if compact_context else []
        )
        if compact_context:
            self.logger.emit(
                "second_pass_context_compacted",
                target_episode_count=len(targets),
                raw_source_chars=len(source_text),
                reasoning_source_chars=len(reasoning_source),
                shared_context_episode_count=len(shared_context),
            )
        batch_items: list[dict] = []
        for position, row, _task_id in targets:
            item = {
                "episode_id": int(row["id"]),
                "episode": self._episode_row_payload(row),
            }
            if not compact_context:
                item["nearby"] = summaries[max(0, position - 8) : position + 9]
            batch_items.append(item)
        if len(batch_items) > 1:
            drafts_by_id, _batch_errors = self._second_pass_batch_drafts(
                batch_items,
                reasoning_source,
                shared_context,
            )
        else:
            drafts_by_id = {}
        prepared: list[tuple[object, int, EpisodeDraft]] = []
        failures: list[tuple[int, str]] = []
        for (_position, row, task_id), item in zip(targets, batch_items, strict=True):
            episode_id = int(row["id"])
            draft = drafts_by_id.get(episode_id)
            if draft is None:
                try:
                    draft = self._second_pass_draft(
                        second_pass_prompt(
                            episode_id,
                            item["episode"],
                            (shared_context if compact_context else item["nearby"]),
                            reasoning_source,
                        )
                    )
                except Exception as exc:
                    if isinstance(exc, ModelTransportUnavailable):
                        raise
                    failures.append((task_id, str(exc)))
                    continue
            # Pass 2 may improve semantic fields, but it must not move an Episode to
            # a model-invented chronology scope.
            draft.timeline_scope = str(row["timeline_scope"] or "")
            # Pass 2 rewrites semantics but never changes the provenance or
            # promotes a reported/speculative claim into direct experience.
            draft.evidence_origin = str(row["evidence_origin"] or "unknown")  # type: ignore[assignment]
            draft.epistemic_status = str(row["epistemic_status"] or "unknown")  # type: ignore[assignment]
            draft.generation = int(row["generation"] or 0)
            draft.epistemic_note = str(row["epistemic_note"] or "")
            finalization_errors = self.extractor.finalize_episode_drafts(
                source_text, [draft], self.logger
            )
            if finalization_errors:
                # Pass 2 is allowed to clarify an unknown only with literal
                # evidence from the same Source. If it invents an identity,
                # keep the already-grounded pass-1 row instead of turning a
                # safe quarantine into a durable hallucination.
                self.logger.emit(
                    "second_pass_grounding_rejected",
                    episode_id=episode_id,
                    errors=finalization_errors,
                )
                draft = EpisodeDraft.from_dict(item["episode"])
            prepared.append((row, task_id, draft))
        if not prepared:
            return [], failures, [], [], []
        matrix = self.model.embed([draft.text for _row, _task, draft in prepared])
        vectors = [
            normalize_embedding(row, self.config.model.embedding_dimension)
            for row in matrix
        ]
        concept_groups, concept_errors = self.extractor.extract_concepts_batch(
            [draft for _row, _task, draft in prepared], source_text
        )
        return prepared, failures, vectors, concept_groups, concept_errors

    def _run_second_pass(self, run_id: int, source_key: str) -> tuple[int, int]:
        rows = list(
            self.episodes.list_by_source_key(source_key, extraction_run_id=run_id)
        )
        if not rows:
            return 0, 0
        summaries = [
            {
                "id": int(row["id"]),
                "segment_index": int(row["segment_index"]),
                "text": row["text"],
                "story_time_text": row["story_time_text"],
            }
            for row in rows
        ]
        targets_by_source: dict[int, list[tuple[int, object, int]]] = {}
        for position, row in enumerate(rows):
            text = str(row["text"])
            if (
                not any(
                    marker in text
                    for marker in (
                        "???",
                        "[USERNAME]",
                        "未标注发言者",
                        "未知发言者",
                    )
                )
                and float(row["confidence"]) >= 0.6
            ):
                continue
            task_id = self.extractions.start_task(
                run_id,
                source_key,
                int(row["segment_index"]),
                "pass2",
                self.config.model.reasoning_model,
                self.config.prompt_version,
                source_id=int(row["source_id"]),
            )
            targets_by_source.setdefault(int(row["source_id"]), []).append(
                (position, row, task_id)
            )

        revised = 0
        failed = 0
        executor = (
            ThreadPoolExecutor(
                max_workers=self.prepare_workers,
                thread_name_prefix="pass2-prepare",
            )
            if self.prepare_workers > 1 and len(targets_by_source) > 1
            else None
        )
        staged_groups: list[
            tuple[list[tuple[int, object, int]], Future | None, object]
        ] = []
        for source_id, targets in targets_by_source.items():
            source_row = self.sources.get(source_id)
            source_text = str(source_row["raw_text"]) if source_row is not None else ""
            future = (
                executor.submit(
                    self._prepare_second_pass_group,
                    targets,
                    summaries,
                    source_text,
                )
                if executor is not None
                else None
            )
            immediate = (
                None
                if future is not None
                else self._prepare_second_pass_group(targets, summaries, source_text)
            )
            staged_groups.append((targets, future, immediate))

        for targets, future, immediate in staged_groups:
            try:
                (
                    prepared,
                    preparation_failures,
                    vectors,
                    concept_groups,
                    concept_errors,
                ) = future.result() if future is not None else immediate
            except Exception as exc:
                if isinstance(exc, ModelTransportUnavailable):
                    raise
                failed += len(targets)
                for _position, _row, task_id in targets:
                    self.extractions.finish_task(task_id, "failed", str(exc))
                    self.logger.emit(
                        "task_failed",
                        task_id=task_id,
                        stage="pass2_prepare",
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                continue
            for task_id, error in preparation_failures:
                failed += 1
                self.extractions.finish_task(task_id, "failed", error)
                self.logger.emit(
                    "task_failed",
                    task_id=task_id,
                    stage="pass2_prepare",
                    error=error,
                )
            if not prepared:
                continue
            try:
                for (row, _task_id, draft), vector in zip(
                    prepared, vectors, strict=True
                ):
                    episode_id = int(row["id"])
                    self.episodes.update_draft(
                        episode_id,
                        draft,
                        encode_embedding(vector, self.config.model.embedding_dimension),
                    )
                    self.episode_index.upsert(episode_id, vector)
                    self.logger.emit(
                        "episode_revised",
                        episode_id=episode_id,
                        previous_text=row["text"],
                        draft=draft,
                    )

                if concept_errors:
                    self.logger.emit(
                        "validation_failed",
                        stage="pass2_concepts_batch",
                        errors=concept_errors,
                    )
                pending_links = [
                    (int(prepared[position][0]["id"]), concept_draft)
                    for position, concept_drafts in enumerate(concept_groups)
                    for concept_draft in concept_drafts
                ]
                concept_ids, concept_relation_jobs = (
                    self.resolver.resolve_many_deferred(
                        [draft for _episode_id, draft in pending_links]
                    )
                )
                for (episode_id, concept_draft), concept_id in zip(
                    pending_links, concept_ids, strict=True
                ):
                    self.builder.link_episode_concept(
                        episode_id,
                        concept_id,
                        concept_draft.confidence,
                        concept_draft.canonical_name,
                    )
                if self.build_inference_relations:
                    self.builder.relate_new_concept_batches(concept_relation_jobs)
                candidate_groups = {
                    int(row["id"]): [
                        (candidate_id, score)
                        for candidate_id, score in self.episode_index.search(
                            vector,
                            top_k=self.config.retrieval.episode_relation_candidate_k,
                        )
                        if candidate_id != int(row["id"])
                    ]
                    for (row, _task_id, _draft), vector in zip(
                        prepared, vectors, strict=True
                    )
                }
                if self.build_inference_relations:
                    self.builder.relate_episode_batches(candidate_groups)
                else:
                    self.logger.emit(
                        "inference_relations_deferred",
                        stage="pass2",
                        concept_candidate_groups=len(concept_relation_jobs),
                        episode_candidate_groups=len(candidate_groups),
                    )
                for _row, task_id, _draft in prepared:
                    self.extractions.finish_task(task_id, "completed")
                revised += len(prepared)
            except Exception as exc:
                if isinstance(exc, ModelTransportUnavailable):
                    raise
                failed += len(prepared)
                for _row, task_id, _draft in prepared:
                    self.extractions.finish_task(task_id, "failed", str(exc))
                    self.logger.emit(
                        "task_failed",
                        task_id=task_id,
                        stage="pass2",
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        return revised, failed

    def _import_files(
        self,
        run_id: int,
        root: Path,
        files: list[Path],
        summary: dict[str, Any],
    ) -> None:
        # Keep relation queues across file boundaries.  Flushing a tiny tail at
        # the end of every file turns corpora containing many short documents
        # into one or two under-filled LLM calls per file.  Full batches are
        # still persisted as soon as they are ready; only the bounded tail is
        # carried into the next file.
        pending_concept_jobs: list[
            tuple[int, ConceptDraft, list[tuple[int, float]]]
        ] = []
        pending_episode_groups: list[tuple[int, list[tuple[int, float]]]] = []
        relation_executor = (
            ThreadPoolExecutor(
                max_workers=self.relation_workers,
                thread_name_prefix="relation-judge",
            )
            if self.build_inference_relations and self.relation_workers > 1
            else None
        )
        self._active_relation_executor = relation_executor
        pending_relation_futures: list[Future[list[AssociationDraft]]] = []
        for file_path in files:
            adapter = self._adapter_for(file_path)
            source_key = logical_source_key(file_path, root)
            file_result: dict[str, Any] = {
                "source_key": source_key,
                "status": "running",
                "segments_planned": 0,
                "segments_completed": 0,
                "segments_partial": 0,
                "segments_skipped": 0,
                "segments_failed": 0,
                "sources": 0,
                "episodes": 0,
                "paragraphs": 0,
                "revised_episodes": 0,
            }
            summary["file_results"].append(file_result)
            self.logger.emit(
                "file_started",
                run_id=run_id,
                source_key=source_key,
                path=str(file_path),
            )
            if adapter is None:
                error = ValueError("no adapter supports this file")
                summary["failed_tasks"] += 1
                summary["failure_details"].append(
                    self._failure_detail("adapter", source_key, error)
                )
                file_result["status"] = "failed"
                summary["files_failed"] += 1
                continue
            try:
                blocks = adapter.read_blocks(file_path)
                if not blocks:
                    raise ValueError("file contains no usable text blocks")
                segments = self.segmenter.segment(source_key, blocks)
                if not segments:
                    raise ValueError("file produced no importable source segments")
            except Exception as exc:
                summary["failed_tasks"] += 1
                detail = self._failure_detail("adapter", source_key, exc)
                summary["failure_details"].append(detail)
                self.logger.emit(
                    "task_failed",
                    stage="adapter",
                    source_key=source_key,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                file_result["status"] = "failed"
                file_result["error"] = str(exc)
                summary["files_failed"] += 1
                continue

            summary["files_read"] += 1
            file_result["segments_planned"] = len(segments)
            scope = infer_timeline_scope(source_key)
            document_contexts: dict[int, str] = {}
            episode_profile = self.config.ingestion.episode_extraction_profile
            if episode_profile in {
                "document_map_assisted",
                "document_map_contextual",
            }:
                try:
                    document_contexts = self.extractor.build_document_map(
                        source_key,
                        [
                            (segment.segment_index, segment.raw_text)
                            for segment in segments
                        ],
                    )
                except Exception as exc:
                    if isinstance(exc, ModelTransportUnavailable):
                        raise
                    summary["failed_tasks"] += 1
                    detail = self._failure_detail("document_map", source_key, exc)
                    summary["failure_details"].append(detail)
                    self.logger.emit(
                        "task_failed",
                        stage="document_map",
                        source_key=source_key,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                    file_result["status"] = "failed"
                    file_result["error"] = str(exc)
                    summary["files_failed"] += 1
                    continue
            elif episode_profile == "adaptive_anchor_map" and len(segments) > 1:
                try:
                    self.logger.emit(
                        "document_anchor_map_triggered",
                        source_key=source_key,
                        reason="logical_file_split_across_sources",
                        segment_count=len(segments),
                    )
                    document_contexts = self.extractor.build_document_anchor_map(
                        source_key,
                        [
                            (segment.segment_index, segment.raw_text)
                            for segment in segments
                        ],
                    )
                except Exception as exc:
                    if isinstance(exc, ModelTransportUnavailable):
                        raise
                    # The adaptive map is a fallible navigation aid, never
                    # factual evidence.  Import the independently auditable
                    # Source segments without it when the map is unavailable.
                    document_contexts = {}
                    self.logger.emit(
                        "document_anchor_map_skipped",
                        stage="document_anchor_map",
                        source_key=source_key,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
            executor = (
                ThreadPoolExecutor(
                    max_workers=self.prepare_workers,
                    thread_name_prefix="source-prepare",
                )
                if self.prepare_workers > 1
                else None
            )
            staged: list[
                tuple[
                    SourceSegment,
                    int,
                    Future[
                        tuple[
                            list[EpisodeDraft],
                            list[str],
                            list[list[ConceptDraft]],
                            list[str],
                            list[ParagraphDraft],
                            list[Any],
                            list[Any],
                            list[Any],
                        ]
                    ]
                    | None,
                ]
            ] = []
            try:
                for segment in segments:
                    task_id: int | None = None
                    try:
                        task_id = self.extractions.start_task(
                            run_id,
                            source_key,
                            segment.segment_index,
                            "pass1",
                            self.config.model.reasoning_model,
                            self.config.prompt_version,
                        )
                        future = (
                            executor.submit(
                                self._prepare_segment_for_persistence,
                                task_id,
                                segment.raw_text,
                                scope,
                                document_contexts.get(segment.segment_index, ""),
                            )
                            if executor is not None
                            else None
                        )
                        staged.append((segment, task_id, future))
                    except Exception as exc:
                        summary["failed_tasks"] += 1
                        file_result["segments_failed"] += 1
                        if task_id is not None:
                            self.extractions.finish_task(task_id, "failed", str(exc))
                        detail = self._failure_detail(
                            "pass1_setup",
                            source_key,
                            exc,
                            segment_index=segment.segment_index,
                            task_id=task_id,
                        )
                        summary["failure_details"].append(detail)
                        self.logger.emit(
                            "task_failed",
                            stage="pass1_setup",
                            source_key=source_key,
                            segment_index=segment.segment_index,
                            error=str(exc),
                            traceback=traceback.format_exc(),
                        )

                for segment, task_id, future in staged:
                    try:
                        (
                            drafts,
                            errors,
                            concept_groups,
                            concept_errors,
                            paragraph_drafts,
                            paragraph_vectors,
                            episode_vectors,
                            concept_vectors,
                        ) = (
                            future.result()
                            if future is not None
                            else self._prepare_segment_for_persistence(
                                task_id,
                                segment.raw_text,
                                scope,
                                document_contexts.get(segment.segment_index, ""),
                            )
                        )

                        # Persist only after extraction and the one fixed-model
                        # embedding request have both succeeded. A failed remote
                        # call therefore leaves no empty Source row behind.
                        source_id = self.sources.insert(segment.raw_text)
                        self.extractions.set_source(task_id, source_id)
                        summary["sources"] += 1
                        file_result["sources"] += 1

                        try:
                            paragraph_ids = self._insert_paragraphs(
                                source_id,
                                source_key,
                                segment.segment_index,
                                segment.raw_text,
                                prepared_drafts=paragraph_drafts,
                                prepared_vectors=paragraph_vectors,
                            )
                            summary["paragraphs"] += len(paragraph_ids)
                            file_result["paragraphs"] += len(paragraph_ids)
                        except Exception as paragraph_exc:
                            summary["failed_paragraph_sources"] += 1
                            detail = self._failure_detail(
                                "paragraph",
                                source_key,
                                paragraph_exc,
                                segment_index=segment.segment_index,
                                task_id=task_id,
                            )
                            summary["failure_details"].append(detail)
                            self.logger.emit(
                                "paragraph_creation_failed",
                                task_id=task_id,
                                source_id=source_id,
                                source_key=source_key,
                                error=str(paragraph_exc),
                                traceback=traceback.format_exc(),
                            )

                        (
                            _episode_ids,
                            concept_relation_jobs,
                            episode_candidate_groups,
                        ) = self._insert_episode_batch(
                            run_id,
                            source_id,
                            source_key,
                            segment.segment_index,
                            drafts,
                            prepared_concept_groups=concept_groups,
                            prepared_concept_errors=concept_errors,
                            prepared_episode_vectors=episode_vectors,
                            prepared_concept_vectors=concept_vectors,
                        )
                        pending_concept_jobs.extend(concept_relation_jobs)
                        pending_episode_groups.extend(episode_candidate_groups.items())
                        summary["episodes"] += len(drafts)
                        file_result["episodes"] += len(drafts)
                        validation_errors = [*errors, *concept_errors]
                        if validation_errors:
                            summary["partial_tasks"] += 1
                            file_result["segments_partial"] += 1
                            self.extractions.finish_task(
                                task_id,
                                "partial",
                                "; ".join(validation_errors),
                            )
                        else:
                            file_result["segments_completed"] += 1
                            self.extractions.finish_task(task_id, "completed")

                        try:
                            self._flush_relation_batches(
                                pending_concept_jobs,
                                pending_episode_groups,
                                force=False,
                                executor=relation_executor,
                                pending_futures=pending_relation_futures,
                            )
                        except Exception as relation_exc:
                            if isinstance(relation_exc, ModelTransportUnavailable):
                                raise
                            summary["failed_relation_batches"] += 1
                            summary["failure_details"].append(
                                self._failure_detail(
                                    "relations",
                                    source_key,
                                    relation_exc,
                                    segment_index=segment.segment_index,
                                    task_id=task_id,
                                )
                            )
                            self.logger.emit(
                                "relation_batch_failed",
                                task_id=task_id,
                                source_key=source_key,
                                error=str(relation_exc),
                                traceback=traceback.format_exc(),
                            )
                    except EmptyEpisodeSkipped as skipped:
                        summary["skipped_tasks"] += 1
                        file_result["segments_skipped"] += 1
                        self.extractions.finish_task(
                            task_id,
                            "skipped",
                            skipped.audit.compact_receipt(),
                        )
                        self.logger.emit(
                            "empty_episode_skip_confirmed",
                            task_id=task_id,
                            source_key=source_key,
                            segment_index=segment.segment_index,
                            **skipped.audit.event_payload(
                                include_quotes=(
                                    os.getenv(
                                        "MEMORY_LOG_MODEL_PAYLOADS", "false"
                                    ).casefold()
                                    == "true"
                                )
                            ),
                        )
                    except Exception as exc:
                        if isinstance(exc, ModelTransportUnavailable):
                            raise
                        summary["failed_tasks"] += 1
                        file_result["segments_failed"] += 1
                        self.extractions.finish_task(task_id, "failed", str(exc))
                        detail = self._failure_detail(
                            "pass1",
                            source_key,
                            exc,
                            segment_index=segment.segment_index,
                            task_id=task_id,
                        )
                        summary["failure_details"].append(detail)
                        self.logger.emit(
                            "task_failed",
                            task_id=task_id,
                            stage="pass1",
                            source_key=source_key,
                            error=str(exc),
                            traceback=traceback.format_exc(),
                        )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=False)

            try:
                self._flush_relation_batches(
                    pending_concept_jobs,
                    pending_episode_groups,
                    force=False,
                    executor=relation_executor,
                    pending_futures=pending_relation_futures,
                )
            except Exception as exc:
                if isinstance(exc, ModelTransportUnavailable):
                    raise
                summary["failed_relation_batches"] += 1
                summary["failure_details"].append(
                    self._failure_detail("relations", source_key, exc)
                )
                self.logger.emit(
                    "relation_batch_failed",
                    source_key=source_key,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )

            if self.config.ingestion.episode_extraction_profile in {
                "single_pass_evidence",
                "single_pass_audited",
                "document_map_assisted",
                "document_map_contextual",
                "adaptive_anchor_map",
                "source_scoped_plain",
            }:
                revised, pass2_failed = 0, 0
                self.logger.emit(
                    "second_pass_skipped",
                    source_key=source_key,
                    reason=(
                        self.config.ingestion.episode_extraction_profile + "_profile"
                    ),
                )
            else:
                try:
                    revised, pass2_failed = self._run_second_pass(run_id, source_key)
                except Exception as exc:
                    if isinstance(exc, ModelTransportUnavailable):
                        raise
                    revised, pass2_failed = 0, 1
                    summary["failure_details"].append(
                        self._failure_detail("pass2", source_key, exc)
                    )
                    self.logger.emit(
                        "task_failed",
                        stage="pass2",
                        source_key=source_key,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
            summary["revised_episodes"] += revised
            summary["failed_tasks"] += pass2_failed
            file_result["revised_episodes"] = revised
            file_result["pass2_failed"] = pass2_failed
            if pass2_failed and not any(
                detail["source_key"] == source_key and detail["stage"] == "pass2"
                for detail in summary["failure_details"]
            ):
                summary["failure_details"].append(
                    self._failure_detail(
                        "pass2",
                        source_key,
                        f"{pass2_failed} second-pass task(s) failed; see the JSONL log for task details",
                    )
                )

            successful_segments = (
                file_result["segments_completed"]
                + file_result["segments_partial"]
                + file_result["segments_skipped"]
            )
            file_has_partial = bool(
                file_result["segments_partial"]
                or file_result["segments_failed"]
                or pass2_failed
                or any(
                    detail["source_key"] == source_key
                    and detail["stage"] in {"paragraph", "relations"}
                    for detail in summary["failure_details"]
                )
            )
            if not successful_segments:
                file_result["status"] = "failed"
                summary["files_failed"] += 1
            elif file_has_partial:
                file_result["status"] = "partial"
                summary["files_partial"] += 1
            else:
                file_result["status"] = "completed"
                summary["files_completed"] += 1
            # A bounded relation tail may still contain jobs from this file.
            # Do not publish a completed file before that tail has passed the
            # single-writer graph checks and reached SQLite.
            file_result["provisional_status"] = file_result["status"]
            file_result["status"] = "relations_pending"
            file_result["relations_finalized"] = False

        relation_tail_succeeded = True
        try:
            self._flush_relation_batches(
                pending_concept_jobs,
                pending_episode_groups,
                force=True,
                executor=relation_executor,
                pending_futures=pending_relation_futures,
            )
        except Exception as exc:
            if isinstance(exc, ModelTransportUnavailable):
                raise
            relation_tail_succeeded = False
            summary["failed_relation_batches"] += 1
            summary["failure_details"].append(
                self._failure_detail("relations", "__corpus_tail__", exc)
            )
            self.logger.emit(
                "relation_batch_failed",
                source_key="__corpus_tail__",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        self._shutdown_relation_executor(wait=True)

        for file_result in summary["file_results"]:
            if file_result["status"] == "relations_pending":
                final_file_status = file_result.pop("provisional_status")
                if not relation_tail_succeeded and final_file_status == "completed":
                    final_file_status = "partial"
                    summary["files_completed"] -= 1
                    summary["files_partial"] += 1
                file_result["status"] = final_file_status
                file_result["relations_finalized"] = relation_tail_succeeded
            self.logger.emit("file_finished", result=file_result)

    def import_path(
        self, path: str | Path, source_root: str | Path | None = None
    ) -> dict[str, Any]:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        root, files, unsupported_files = self._input_files(input_path)
        if source_root is not None:
            root = Path(source_root).resolve()
            if not root.is_dir():
                raise NotADirectoryError(root)
            try:
                input_path.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"input path {input_path} is outside source root {root}"
                ) from exc
        if not files:
            if input_path.is_file():
                raise ValueError(
                    f"unsupported input file type: {input_path.suffix or '<none>'}"
                )
            raise ValueError(
                f"directory contains no supported .txt, .md, or .json files: {input_path}"
            )
        config_snapshot = safe_config_snapshot(self.config)
        run_id = self.extractions.start_run(
            config_snapshot,
            {"all": self.config.prompt_version},
            self.logger.path,
        )
        unsupported_display = [
            logical_source_key(file_path, root) for file_path in unsupported_files
        ]
        summary: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "files": len(files),
            "files_read": 0,
            "files_completed": 0,
            "files_partial": 0,
            "files_failed": 0,
            "unsupported_files": unsupported_display,
            "sources": 0,
            "episodes": 0,
            "paragraphs": 0,
            "failed_tasks": 0,
            "partial_tasks": 0,
            "skipped_tasks": 0,
            "failed_paragraph_sources": 0,
            "failed_relation_batches": 0,
            "revised_episodes": 0,
            "failure_details": [],
            "file_results": [],
        }
        self.logger.emit(
            "import_started",
            run_id=run_id,
            path=str(input_path),
            source_root=str(root),
            supported_files=len(files),
            unsupported_files=unsupported_display,
        )
        try:
            with self._background_provider_scope():
                self._import_files(run_id, root, files, summary)
        except BaseException as exc:
            self._shutdown_relation_executor(wait=False)
            retryable_infrastructure_abort = isinstance(
                exc, (ModelTransportUnavailable, DatabaseBusyError)
            )
            final_status = (
                "interrupted"
                if isinstance(
                    exc,
                    (
                        KeyboardInterrupt,
                        SystemExit,
                        ModelTransportUnavailable,
                        DatabaseBusyError,
                    ),
                )
                else "failed"
            )
            summary["status"] = final_status
            summary["fatal_error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            interrupted_tasks = self.extractions.interrupt_running_tasks(
                run_id, f"{type(exc).__name__}: {exc}"
            )
            summary["interrupted_tasks"] = interrupted_tasks
            self.extractions.finish_run(run_id, final_status, summary)
            if retryable_infrastructure_abort:
                summary["retryable_abort_cleanup"] = (
                    self.extractions.discard_incomplete_run(run_id)
                )
            self.logger.emit(
                "run_finished",
                status=final_status,
                summary=summary,
                traceback=traceback.format_exc(),
            )
            raise

        successful_files = summary["files_completed"] + summary["files_partial"]
        if not successful_files and summary["files_failed"]:
            final_status = "failed"
        elif (
            summary["unsupported_files"]
            or summary["failed_tasks"]
            or summary["partial_tasks"]
            or summary["failed_paragraph_sources"]
            or summary["failed_relation_batches"]
        ):
            final_status = "partial"
        else:
            final_status = "completed"
        summary["status"] = final_status
        self.extractions.finish_run(run_id, final_status, summary)
        self.logger.emit("run_finished", status=final_status, summary=summary)
        return summary
