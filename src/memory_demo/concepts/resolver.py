from __future__ import annotations

from memory_demo.embeddings.codec import encode_embedding, normalize_embedding
from dataclasses import asdict
import numpy as np
from threading import RLock

from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm.prompts import (
    CONCEPT_UPDATE_SYSTEM,
    concept_update_prompt,
)
from memory_demo.repositories import ConceptRepository
from memory_demo.repositories.concept import normalize_alias
from memory_demo.types import ConceptDraft


class ConceptResolver:
    def __init__(
        self,
        model,
        repository: ConceptRepository,
        index,
        association_builder,
        dimension: int,
        relation_min_similarity: float = 0.72,
        logger: JsonlEventLogger | None = None,
    ):
        self.model = model
        self.repository = repository
        self.index = index
        self.association_builder = association_builder
        self.dimension = dimension
        self.relation_min_similarity = relation_min_similarity
        self.logger = logger
        # Exact-alias lookup followed by insert must be atomic across concurrent
        # file imports, otherwise two files can create the same Concept at once.
        self._resolve_lock = RLock()
        self._concept_rows: dict[int, dict[str, object]] = {}
        self._alias_rows: dict[str, list[dict[str, object]]] = {}
        for raw in self.repository.resolution_snapshot():
            concept_id = int(raw["id"])
            row = self._concept_rows.setdefault(
                concept_id,
                {
                    "id": concept_id,
                    "canonical_name": str(raw["canonical_name"]),
                    "description": str(raw["description"]),
                    "embedding_text": str(raw["embedding_text"]),
                    "status": str(raw["status"]),
                    "canonical_concept_id": raw["canonical_concept_id"],
                },
            )
            normalized = str(raw["normalized_alias"] or "")
            if normalized:
                self._alias_rows.setdefault(normalized, []).append(row)

    @staticmethod
    def _row_canonical_id(row) -> int:
        return int(row["canonical_concept_id"] or row["id"])

    def _find_exact_concept_id(self, draft: ConceptDraft) -> int | None:
        """Reuse an unambiguous canonical name or alias before vector search.

        Alias collisions are legal because natural-language names can be
        homonyms.  We therefore reuse only when all matching incoming names
        resolve to one canonical Concept, except that one exact canonical-name
        match is strong enough to disambiguate unrelated alias collisions.
        """

        canonical_rows = self._alias_rows.get(
            normalize_alias(draft.canonical_name), []
        )
        exact_canonical_ids = {
            self._row_canonical_id(row)
            for row in canonical_rows
            if row["status"] == "active"
            and normalize_alias(str(row["canonical_name"]))
            == normalize_alias(draft.canonical_name)
        }
        if len(exact_canonical_ids) == 1:
            return next(iter(exact_canonical_ids))
        matched_ids: set[int] = set()
        uniquely_matched_ids: set[int] = set()
        for name in [draft.canonical_name, *(alias for alias, _ in draft.aliases)]:
            name_matches = {
                self._row_canonical_id(row)
                for row in self._alias_rows.get(normalize_alias(name), [])
            }
            matched_ids.update(name_matches)
            if len(name_matches) == 1:
                uniquely_matched_ids.update(name_matches)
        # A broad display label can also be attached to a named variant while
        # two language-specific aliases
        # still uniquely identify the base character.  Unique alias consensus
        # is stronger than the ambiguous label and avoids creating a third row.
        if len(uniquely_matched_ids) == 1:
            return next(iter(uniquely_matched_ids))
        return next(iter(matched_ids)) if len(matched_ids) == 1 else None

    def _missing_aliases_for_concept(
        self, concept_id: int, draft: ConceptDraft
    ) -> list[tuple[str, str]]:
        missing: list[tuple[str, str]] = []
        seen: set[str] = set()
        for alias, language in [
            (draft.canonical_name, "unknown"),
            *draft.aliases,
        ]:
            normalized = normalize_alias(alias)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            rows = self._alias_rows.get(normalized, [])
            if not rows:
                missing.append((alias, language))
                continue
            # An alias already owned by this Concept needs no duplicate row.
            # A collision owned by another Concept is deliberately not stolen.
            if any(self._row_canonical_id(row) == concept_id for row in rows):
                continue
        return missing

    def _remember_aliases(
        self,
        concept_id: int,
        aliases: list[tuple[str, str]],
    ) -> None:
        row = self._concept_rows[concept_id]
        for alias, _language in aliases:
            normalized = normalize_alias(alias)
            if not normalized:
                continue
            owners = self._alias_rows.setdefault(normalized, [])
            if not any(int(owner["id"]) == concept_id for owner in owners):
                owners.append(row)

    def _remember_new_concept(self, concept_id: int, draft: ConceptDraft) -> None:
        self._concept_rows[concept_id] = {
            "id": concept_id,
            "canonical_name": draft.canonical_name,
            "description": draft.description,
            "embedding_text": draft.embedding_text,
            "status": "active",
            "canonical_concept_id": None,
        }
        self._remember_aliases(
            concept_id,
            [(draft.canonical_name, "canonical"), *draft.aliases],
        )

    def _resolve(
        self,
        draft: ConceptDraft,
        vector: np.ndarray | None = None,
        *,
        defer_relations: bool = False,
    ) -> tuple[int, list[tuple[int, float]]]:
        concept_id = self._find_exact_concept_id(draft)
        if concept_id is not None:
            canonical = self._concept_rows.get(concept_id)
            description_is_placeholder = canonical is not None and (
                str(canonical["description"]).strip()
                == str(canonical["canonical_name"]).strip()
            )
            if canonical is not None and description_is_placeholder:
                payload = self.model.chat_json(
                    CONCEPT_UPDATE_SYSTEM,
                    concept_update_prompt(
                        {
                            "canonical_name": canonical["canonical_name"],
                            "description": canonical["description"],
                            "embedding_text": canonical["embedding_text"],
                        },
                        asdict(draft),
                    ),
                )
                raw = payload.get("concept", {}) if isinstance(payload, dict) else {}
                updated = ConceptDraft.from_dict(raw)
                vector = normalize_embedding(
                    self.model.embed([updated.embedding_text])[0], self.dimension
                )
                self.repository.update(
                    concept_id, updated, encode_embedding(vector, self.dimension)
                )
                self.index.upsert(concept_id, vector)
                canonical.update(
                    {
                        "canonical_name": updated.canonical_name,
                        "description": updated.description,
                        "embedding_text": updated.embedding_text,
                    }
                )
                self._remember_aliases(
                    concept_id,
                    [(updated.canonical_name, "canonical"), *updated.aliases],
                )
                if self.logger:
                    self.logger.emit(
                        "concept_updated", concept_id=concept_id, draft=updated
                    )
            new_aliases = self._missing_aliases_for_concept(concept_id, draft)
            if new_aliases:
                inserted = self.repository.add_aliases(
                    concept_id, new_aliases, draft.confidence
                )
                self._remember_aliases(concept_id, new_aliases)
                if self.logger:
                    self.logger.emit(
                        "concept_aliases_added",
                        concept_id=concept_id,
                        aliases=new_aliases,
                        inserted=inserted,
                    )
            return concept_id, []
        if vector is None:
            vector = normalize_embedding(
                self.model.embed([draft.embedding_text])[0], self.dimension
            )
        candidates = [
            (node_id, score)
            for node_id, score in self.index.search(vector, 8)
            if score >= self.relation_min_similarity
        ]
        concept_id = self.repository.insert(
            draft, encode_embedding(vector, self.dimension)
        )
        self.index.upsert(concept_id, vector)
        self._remember_new_concept(concept_id, draft)
        if self.logger:
            self.logger.emit(
                "concept_created", concept_id=concept_id, canonical_name=draft.canonical_name
            )
        if candidates and not defer_relations:
            self.association_builder.relate_new_concept(concept_id, draft, candidates)
        return concept_id, candidates

    def resolve(self, draft: ConceptDraft, vector: np.ndarray | None = None) -> int:
        with self._resolve_lock:
            concept_id, _candidates = self._resolve(draft, vector)
        return concept_id

    def resolve_preembedded_deferred(
        self, draft: ConceptDraft, vector: np.ndarray
    ) -> tuple[int, list[tuple[int, float]]]:
        """Resolve one audited draft without paying for a duplicate embedding."""
        normalized = normalize_embedding(vector, self.dimension)
        with self._resolve_lock:
            return self._resolve(draft, normalized, defer_relations=True)

    def resolve_many_deferred(
        self, drafts: list[ConceptDraft]
    ) -> tuple[
        list[int], list[tuple[int, ConceptDraft, list[tuple[int, float]]]]
    ]:
        if not drafts:
            return [], []
        # Exact aliases do not need a fresh embedding.  Stage-14 contained many
        # repeated participants, so embedding every draft before checking the
        # repository paid for vectors that _resolve immediately discarded.
        concept_ids: list[int | None] = [None] * len(drafts)
        relation_jobs: list[
            tuple[int, ConceptDraft, list[tuple[int, float]]]
        ] = []
        with self._resolve_lock:
            unresolved_indexes: list[int] = []
            for index, draft in enumerate(drafts):
                if self._find_exact_concept_id(draft) is not None:
                    concept_id, _candidates = self._resolve(
                        draft, None, defer_relations=True
                    )
                    concept_ids[index] = concept_id
                else:
                    unresolved_indexes.append(index)
            if unresolved_indexes:
                matrix = self.model.embed(
                    [drafts[index].embedding_text for index in unresolved_indexes]
                )
                vectors = [
                    normalize_embedding(row, self.dimension) for row in matrix
                ]
            else:
                vectors = []
            # Resolve sequentially so duplicate aliases inside this Source see a
            # Concept inserted by an earlier draft instead of creating duplicates.
            for index, vector in zip(unresolved_indexes, vectors, strict=True):
                draft = drafts[index]
                concept_id, candidates = self._resolve(
                    draft, vector, defer_relations=True
                )
                concept_ids[index] = concept_id
                if candidates:
                    relation_jobs.append((concept_id, draft, candidates))
        if any(value is None for value in concept_ids):
            raise RuntimeError("concept batch resolution left an unresolved slot")
        return [int(value) for value in concept_ids], relation_jobs

    def resolve_many_preembedded_deferred(
        self,
        drafts: list[ConceptDraft],
        vectors: list[np.ndarray],
    ) -> tuple[
        list[int], list[tuple[int, ConceptDraft, list[tuple[int, float]]]]
    ]:
        """Resolve one ordered batch using vectors produced by its file worker.

        Exact aliases may ignore their supplied vector.  Keeping them in the
        combined embedding request is intentional: one larger request has much
        lower latency than separate Episode, Paragraph and Concept calls.
        """

        if len(drafts) != len(vectors):
            raise ValueError("concept drafts and vectors must have equal length")
        concept_ids: list[int] = []
        relation_jobs: list[
            tuple[int, ConceptDraft, list[tuple[int, float]]]
        ] = []
        with self._resolve_lock:
            for draft, vector in zip(drafts, vectors, strict=True):
                concept_id, candidates = self._resolve(
                    draft,
                    normalize_embedding(vector, self.dimension),
                    defer_relations=True,
                )
                concept_ids.append(concept_id)
                if candidates:
                    relation_jobs.append((concept_id, draft, candidates))
        return concept_ids, relation_jobs

    def resolve_many(self, drafts: list[ConceptDraft]) -> list[int]:
        concept_ids, relation_jobs = self.resolve_many_deferred(drafts)
        self.association_builder.relate_new_concept_batches(relation_jobs)
        return concept_ids
