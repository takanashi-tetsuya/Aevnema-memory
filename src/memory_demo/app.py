from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from memory_demo.config import AppConfig
from memory_demo.database import Database
from memory_demo.embeddings import (
    EmbeddingIndex,
    decode_embedding,
    encode_embedding,
)
from memory_demo.event_log import JsonlEventLogger
from memory_demo.ingestion.pipeline import ImportPipeline
from memory_demo.llm import ModelClient
from memory_demo.repositories import (
    AssociationRepository,
    ConceptRepository,
    EpisodeRepository,
    ParagraphRepository,
    SourceRepository,
)
from memory_demo.retrieval import QueryEngine, SQLiteSparseIndex
from memory_demo.retrieval.contextual_association import ContextualAssociationMatcher
from memory_demo.types import ContextualRecallCandidate
from memory_demo.retrieval.cue_index import association_cue_text


class MemoryApplication:
    def __init__(self, config: AppConfig):
        self.config = config
        config.ensure_directories()
        self.db = Database(config.database_path)
        self.db.initialize()
        self.sources = SourceRepository(self.db)
        self.episodes = EpisodeRepository(self.db)
        self.concepts = ConceptRepository(self.db)
        self.paragraphs = ParagraphRepository(self.db)
        self.associations = AssociationRepository(
            self.db,
            config.weights,
            contextual_noop_decay=config.retrieval.contextual_noop_decay,
            contextual_harm_multiplier=config.retrieval.contextual_harm_multiplier,
            contextual_min_distinct_successes=(
                config.retrieval.contextual_min_distinct_successes
            ),
            contextual_promotion_enabled=(
                config.retrieval.contextual_promotion_enabled
            ),
        )
        self.episode_index = EmbeddingIndex(config.model.embedding_dimension)
        self.concept_index = EmbeddingIndex(config.model.embedding_dimension)
        self.paragraph_index = EmbeddingIndex(config.model.embedding_dimension)
        self.association_index = EmbeddingIndex(config.model.embedding_dimension)
        self.context_cue_index = EmbeddingIndex(config.model.embedding_dimension)
        self.need_cue_index = EmbeddingIndex(config.model.embedding_dimension)
        self.episode_sparse_index = SQLiteSparseIndex(self.db, "episode")
        self.source_sparse_index = SQLiteSparseIndex(self.db, "source")

    def new_logger(self, operation: str) -> JsonlEventLogger:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        unique = uuid4().hex[:8]
        return JsonlEventLogger(
            self.config.log_dir / f"{operation}-{timestamp}-{unique}.jsonl"
        )

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
        self.rebuild_contextual_indexes()

    def rebuild_contextual_indexes(self) -> dict[str, int | str]:
        """Restore persisted context/need prototypes without network access."""
        prototypes = self.associations.list_cue_prototypes()
        context_rows = [row for row in prototypes if str(row["cue_kind"]) == "context"]
        need_rows = [row for row in prototypes if str(row["cue_kind"]) == "need"]
        self.context_cue_index = EmbeddingIndex(
            self.config.model.embedding_dimension,
            initial_capacity=max(16, len(context_rows) + 1),
        )
        self.need_cue_index = EmbeddingIndex(
            self.config.model.embedding_dimension,
            initial_capacity=max(16, len(need_rows) + 1),
        )
        skipped = 0
        for row in prototypes:
            try:
                vector = decode_embedding(
                    row["vector_blob"], self.config.model.embedding_dimension
                )
                target = (
                    self.context_cue_index
                    if str(row["cue_kind"]) == "context"
                    else self.need_cue_index
                )
                target.add(int(row["id"]), vector)
            except (TypeError, ValueError):
                skipped += 1
        result = {
            "context_prototypes": self.context_cue_index.count,
            "need_prototypes": self.need_cue_index.count,
            "skipped": skipped,
            "external_calls": 0,
        }
        return result

    def create_contextual_association(
        self,
        candidate: ContextualRecallCandidate,
        *,
        domain: str,
        model_id: str,
        context_vector,
        need_vector,
        context_text_hash: str,
        need_text_hash: str,
        context_display_text: str = "",
        need_display_text: str = "",
    ) -> int:
        """Persist cues and an edge, then update RAM after both commits succeed."""
        context_id = self.associations.get_or_create_cue_prototype(
            domain=domain,
            cue_kind="context",
            model_id=model_id,
            dimension=self.config.model.embedding_dimension,
            vector=context_vector,
            text_hash=context_text_hash,
            display_text=context_display_text,
            source_request_hash=candidate.source_request_hash,
        )
        need_id = self.associations.get_or_create_cue_prototype(
            domain=domain,
            cue_kind="need",
            model_id=model_id,
            dimension=self.config.model.embedding_dimension,
            vector=need_vector,
            text_hash=need_text_hash,
            display_text=need_display_text,
            source_request_hash=candidate.source_request_hash,
        )
        association_id = self.associations.create_contextual(
            candidate,
            context_cue_id=context_id,
            need_cue_id=need_id,
            utility_weight=0.20,
            probation_ttl=self.config.retrieval.contextual_probation_ttl,
        )
        # Database is authoritative.  A crash before these two adds is safe;
        # startup rebuild restores them from persisted prototype rows.
        self.context_cue_index.add(context_id, context_vector)
        self.need_cue_index.add(need_id, need_vector)
        return association_id

    def rebuild_association_cue_index(
        self,
        *,
        batch_size: int = 64,
        logger: JsonlEventLogger | None = None,
    ) -> dict[str, int | str]:
        """Load persisted cue vectors and embed only new/changed relations."""
        active_logger = logger or self.new_logger("association-cue-index")
        rows = list(self.associations.list_cue_candidates())
        self.association_index = EmbeddingIndex(
            self.config.model.embedding_dimension,
            initial_capacity=max(16, len(rows) + 1),
        )
        size = max(1, int(batch_size))
        dimension = self.config.model.embedding_dimension
        missing: list[tuple[object, str]] = []
        reused = 0
        for row in rows:
            blob = row["cue_embedding"]
            relation_text = str(row["relation_text"])
            cue_text = association_cue_text(row)
            if blob is not None and str(row["cue_embedding_text"]) == cue_text:
                try:
                    vector = decode_embedding(blob, dimension)
                except (TypeError, ValueError):
                    missing.append((row, cue_text))
                else:
                    self.association_index.upsert(int(row["id"]), vector)
                    reused += 1
            else:
                missing.append((row, cue_text))
        model = ModelClient(self.config.model, active_logger) if missing else None
        embedded = 0
        for start in range(0, len(missing), size):
            batch = missing[start : start + size]
            assert model is not None
            matrix = model.embed([cue_text for _row, cue_text in batch])
            for (row, cue_text), vector in zip(batch, matrix, strict=True):
                self.association_index.upsert(int(row["id"]), vector)
                self.associations.store_cue_embedding(
                    int(row["id"]),
                    str(row["relation_text"]),
                    cue_text,
                    encode_embedding(vector, dimension),
                )
                embedded += 1
        result = {
            "dtype": "float32",
            "dimension": dimension,
            "eligible_rows": len(rows),
            "indexed_rows": self.association_index.count,
            "reused_persisted_rows": reused,
            "embedded_rows": embedded,
            "memory_bytes": self.association_index.memory_bytes,
        }
        active_logger.emit("association_cue_index_rebuilt", result=result)
        return result

    def import_path(
        self,
        path: str | Path,
        source_root: str | Path | None = None,
        *,
        rebuild_indexes_before_import: bool = True,
    ) -> dict:
        logger = self.new_logger("import")
        model = ModelClient(self.config.model, logger)
        pipeline = ImportPipeline(
            self.config,
            self.db,
            model,
            logger,
            self.episode_index,
            self.concept_index,
            self.paragraph_index,
            prepare_workers=self.config.ingestion.prepare_workers,
            relation_workers=self.config.ingestion.relation_workers,
            relation_batch_size=self.config.ingestion.relation_batch_size,
            build_inference_relations=(
                self.config.ingestion.build_inference_relations
            ),
        )
        # Standalone callers need to load persisted vectors. Long-lived
        # services already own current indexes and update them in place after
        # every insert, so rebuilding before every file is pure repeated work.
        if rebuild_indexes_before_import:
            pipeline.rebuild_indexes()
        return pipeline.import_path(path, source_root=source_root)

    def backfill_paragraphs(self) -> dict[str, int]:
        logger = self.new_logger("paragraph-backfill")
        model = ModelClient(self.config.model, logger)
        pipeline = ImportPipeline(
            self.config,
            self.db,
            model,
            logger,
            self.episode_index,
            self.concept_index,
            self.paragraph_index,
            prepare_workers=self.config.ingestion.prepare_workers,
            relation_workers=self.config.ingestion.relation_workers,
            relation_batch_size=self.config.ingestion.relation_batch_size,
            build_inference_relations=(
                self.config.ingestion.build_inference_relations
            ),
        )
        pipeline.rebuild_indexes()
        return pipeline.backfill_paragraphs()

    def augment_concepts(self) -> dict[str, int | str]:
        logger = self.new_logger("concept-augmentation")
        model = ModelClient(self.config.model, logger)
        pipeline = ImportPipeline(
            self.config,
            self.db,
            model,
            logger,
            self.episode_index,
            self.concept_index,
            self.paragraph_index,
            prepare_workers=self.config.ingestion.prepare_workers,
            relation_workers=self.config.ingestion.relation_workers,
            relation_batch_size=self.config.ingestion.relation_batch_size,
            build_inference_relations=(
                self.config.ingestion.build_inference_relations
            ),
        )
        pipeline.rebuild_indexes()
        return pipeline.augment_concepts()

    def query_engine(
        self,
        logger: JsonlEventLogger | None = None,
        *,
        config: AppConfig | None = None,
    ) -> QueryEngine:
        """Create a query-local engine over the application's shared indexes.

        ``config`` may be a request-scoped snapshot.  Repositories and RAM
        indexes remain shared, while QueryEngine's mutable traversal/growth
        helpers and ModelClient are isolated per request.
        """

        active_logger = logger or self.new_logger("query")
        active_config = config or self.config
        model = ModelClient(active_config.model, active_logger)
        return QueryEngine(
            active_config,
            model,
            self.episode_index,
            self.concept_index,
            self.episodes,
            self.concepts,
            self.sources,
            self.associations,
            active_logger,
            association_index=self.association_index,
            paragraph_index=self.paragraph_index,
            paragraphs=self.paragraphs,
            episode_sparse_index=self.episode_sparse_index,
            source_sparse_index=self.source_sparse_index,
            contextual_matcher=(
                ContextualAssociationMatcher(
                    self.context_cue_index,
                    self.need_cue_index,
                    self.associations,
                    context_threshold=active_config.retrieval.contextual_context_threshold,
                    need_threshold=active_config.retrieval.contextual_need_threshold,
                    combine_mode=active_config.retrieval.contextual_combine_mode,
                    context_top_k=active_config.retrieval.contextual_context_top_k,
                    need_top_k=active_config.retrieval.contextual_need_top_k,
                    edge_top_k=active_config.retrieval.contextual_edge_top_k,
                )
                if active_config.retrieval.contextual_association_enabled
                else None
            ),
        )

    def stats(self) -> dict:
        return {
            "database": str(self.config.database_path),
            "sources": self.sources.count(),
            "episodes": self.episodes.count(),
            "concepts": self.concepts.count(),
            "paragraphs": self.paragraphs.count(),
            "associations": self.associations.stats(),
            "episode_index_count": self.episode_index.count,
            "concept_index_count": self.concept_index.count,
            "paragraph_index_count": self.paragraph_index.count,
            "association_index_count": self.association_index.count,
            "episode_index_memory_bytes": self.episode_index.memory_bytes,
            "concept_index_memory_bytes": self.concept_index.memory_bytes,
            "paragraph_index_memory_bytes": self.paragraph_index.memory_bytes,
            "association_index_memory_bytes": self.association_index.memory_bytes,
            "context_cue_index_count": self.context_cue_index.count,
            "need_cue_index_count": self.need_cue_index.count,
            "contextual_association_enabled": (
                self.config.retrieval.contextual_association_enabled
            ),
            "association_cue_enabled": (
                self.config.retrieval.association_cue_enabled
            ),
            "paragraph_retrieval_enabled": self.config.paragraph.enabled,
            "sparse_retrieval_enabled": self.config.retrieval.sparse_enabled,
            "episode_sparse_index_count": self.episode_sparse_index.count,
            "source_sparse_index_count": self.source_sparse_index.count,
            "concept_extraction_profile": self.config.concept_extraction.profile,
            "optimization_profile": self.config.optimization_profile,
            "episode_audit_mode": self.config.ingestion.episode_audit_mode,
            "episode_audit_always": self.config.ingestion.episode_audit_always,
            "episode_factual_audit_mode": (
                self.config.ingestion.episode_factual_audit_mode
            ),
            "episode_factual_audit_model": (
                self.config.ingestion.episode_factual_audit_model
                or self.config.model.reasoning_model
            ),
            "rerank_review_mode": self.config.retrieval.rerank_review_mode,
            "rerank_backend": self.config.retrieval.rerank_backend,
            "reranker_model": self.config.model.reranker_model,
            "embedding_dtype": "float32",
            "embedding_dimension": self.config.model.embedding_dimension,
        }
