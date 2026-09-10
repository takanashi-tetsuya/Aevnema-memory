from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Literal

NodeType = Literal["episode", "concept"]
EvidenceOrigin = Literal["source", "importer", "system", "mixed", "unknown"]
EpistemicStatus = Literal[
    "observed", "asserted", "reported", "speculative", "mixed", "unknown"
]

EVIDENCE_ORIGINS = {"source", "importer", "system", "mixed", "unknown"}
EPISTEMIC_STATUSES = {
    "observed", "asserted", "reported", "speculative", "mixed", "unknown"
}


QueryVectorRole = Literal["whole", "atomic", "followup"]


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """The immutable compatibility contract for one embedding vector space.

    Vector equality is meaningful only inside this contract.  The canonical
    identifier intentionally contains metadata hashes rather than input text,
    so it is safe to put in a local diagnostic or a redacted trace receipt.
    """

    provider: str = "unknown"
    model_id: str = "unknown"
    revision: str = "unspecified"
    dimension: int = 0
    dtype: str = "float32"
    preprocessing: str = "nfkc_space_v1"
    normalization: str = "l2_float32_v1"
    version: str = "embedding_space_v1"

    def __post_init__(self) -> None:
        if int(self.dimension) <= 0:
            raise ValueError("embedding space dimension must be positive")
        for field_name in (
            "provider",
            "model_id",
            "revision",
            "dtype",
            "preprocessing",
            "normalization",
            "version",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"embedding space {field_name} is required")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "version": str(self.version),
            "provider": str(self.provider),
            "model_id": str(self.model_id),
            "revision": str(self.revision),
            "dimension": int(self.dimension),
            "dtype": str(self.dtype),
            "preprocessing": str(self.preprocessing),
            "normalization": str(self.normalization),
        }

    @property
    def canonical_id(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "embedding-space:sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PhysicalQueryVector:
    """One provider-produced vector, stored once per request and text hash."""

    physical_id: str
    text_hash: str
    vector: Any
    embedding_space_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.physical_id).strip():
            raise ValueError("physical query vector id is required")
        if not str(self.text_hash).strip():
            raise ValueError("physical query vector text_hash is required")


@dataclass(frozen=True, slots=True)
class QueryVectorRef:
    """Opaque reference from a logical request role to a physical vector."""

    physical_id: str
    embedding_space_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.physical_id).strip():
            raise ValueError("query vector reference physical_id is required")


@dataclass(frozen=True, slots=True)
class LogicalQueryBinding:
    """One logical query role/slot bound to a deduplicated physical vector."""

    query_id: str
    text_hash: str
    role: QueryVectorRole
    vector_ref: QueryVectorRef
    slot_id: str = ""
    # Text remains process-local compatibility metadata only.  It is never a
    # trace field; trace code must use ``text_hash`` and vector references.
    text: str = ""

    @property
    def physical_id(self) -> str:
        return self.vector_ref.physical_id

    @property
    def embedding_space_id(self) -> str:
        return self.vector_ref.embedding_space_id


@dataclass(frozen=True, slots=True)
class QueryVectorRequest:
    """A logical request submitted to the request-scoped coordinator.

    Duplicate text is allowed and expected: e.g. a whole-query role and two
    independently required slots can point to one physical provider vector.
    ``query_id`` and ``slot_id`` retain those distinct logical obligations.
    """

    role: QueryVectorRole
    text: str
    query_id: str = ""
    slot_id: str = ""


@dataclass(frozen=True, slots=True)
class QueryVector:
    """Compatibility view of one logical binding and its in-memory vector.

    New code should use the bundle's physical vector store plus logical
    bindings.  The vector field is retained for older contextual callers and
    is a shared in-memory reference, not a second provider result.
    """

    query_id: str
    text_hash: str
    role: QueryVectorRole
    vector: Any
    slot_id: str = ""
    # Kept outside raw_result by callers unless explicitly needed for a local
    # learning receipt.  This lets the coordinator retain the exact source
    # text for diagnostics without requiring another embedding request.
    text: str = ""
    physical_id: str = ""
    embedding_space_id: str = ""


@dataclass(frozen=True, slots=True)
class QueryVectorBundle:
    """Logical query bindings over a request-local deduplicated vector store.

    ``whole`` and ``queries`` are backward-compatible views.  ``physical_vectors``
    is the authoritative storage: one physical item may have many logical
    bindings with distinct roles, query IDs, and evidence-slot IDs.
    """

    model_id: str
    dimension: int
    whole: Any
    queries: tuple[QueryVector, ...] = ()
    embedding_space: EmbeddingSpace | None = None
    physical_vectors: tuple[PhysicalQueryVector, ...] = ()
    logical_bindings: tuple[LogicalQueryBinding, ...] = ()
    source_request_hash: str = ""
    schema_version: int = 2
    whole_physical_id: str = ""

    @staticmethod
    def _legacy_physical_id(
        space_id: str,
        text_hash: str,
        fallback: str,
    ) -> str:
        digest = hashlib.sha256(
            f"{space_id}\0{text_hash}\0{fallback}".encode("utf-8")
        ).hexdigest()
        return "query-vector:sha256:" + digest

    def __post_init__(self) -> None:
        dimension = int(self.dimension)
        if dimension <= 0:
            raise ValueError("query vector bundle dimension must be positive")
        space = self.embedding_space or EmbeddingSpace(
            provider="legacy",
            model_id=str(self.model_id or "unknown"),
            dimension=dimension,
        )
        if int(space.dimension) != dimension:
            raise ValueError("query vector bundle dimension conflicts with embedding space")
        space_id = space.canonical_id
        physical_by_id: dict[str, PhysicalQueryVector] = {}
        physical_by_hash: dict[str, str] = {}
        for item in self.physical_vectors:
            item_space_id = str(item.embedding_space_id or space_id)
            if item_space_id != space_id:
                raise ValueError("physical query vector uses a different embedding space")
            physical_id = str(item.physical_id)
            previous = physical_by_id.get(physical_id)
            if previous is not None and previous.text_hash != item.text_hash:
                raise ValueError("physical query vector id is ambiguous")
            normalized = (
                item
                if item.embedding_space_id == space_id
                else replace(item, embedding_space_id=space_id)
            )
            physical_by_id[physical_id] = normalized
            physical_by_hash.setdefault(str(item.text_hash), physical_id)

        normalized_queries: list[QueryVector] = []
        for index, query in enumerate(self.queries):
            if query.role not in {"whole", "atomic", "followup"}:
                raise ValueError("query vector role is invalid")
            text_hash = str(query.text_hash or "").strip() or hashlib.sha256(
                str(query.text or query.query_id or index).encode("utf-8")
            ).hexdigest()
            physical_id = str(query.physical_id or "").strip()
            if not physical_id:
                physical_id = physical_by_hash.get(text_hash, "")
            if not physical_id:
                physical_id = self._legacy_physical_id(
                    space_id, text_hash, str(query.query_id or index)
                )
            existing = physical_by_id.get(physical_id)
            if existing is None:
                physical_by_id[physical_id] = PhysicalQueryVector(
                    physical_id=physical_id,
                    text_hash=text_hash,
                    vector=query.vector,
                    embedding_space_id=space_id,
                )
                physical_by_hash.setdefault(text_hash, physical_id)
            elif existing.text_hash != text_hash:
                raise ValueError("logical query binding conflicts with physical vector")
            normalized_queries.append(
                query
                if (
                    query.text_hash == text_hash
                    and query.physical_id == physical_id
                    and query.embedding_space_id == space_id
                )
                else replace(
                    query,
                    text_hash=text_hash,
                    physical_id=physical_id,
                    embedding_space_id=space_id,
                )
            )

        bindings = tuple(self.logical_bindings)
        if bindings:
            for binding in bindings:
                if binding.role not in {"whole", "atomic", "followup"}:
                    raise ValueError("logical query binding role is invalid")
                if binding.vector_ref.embedding_space_id not in {"", space_id}:
                    raise ValueError("logical query binding uses a different embedding space")
                if binding.physical_id not in physical_by_id:
                    raise ValueError("logical query binding references a missing physical vector")
            if not normalized_queries:
                normalized_queries = [
                    QueryVector(
                        query_id=item.query_id,
                        text_hash=item.text_hash,
                        role=item.role,
                        vector=physical_by_id[item.physical_id].vector,
                        slot_id=item.slot_id,
                        text=item.text,
                        physical_id=item.physical_id,
                        embedding_space_id=space_id,
                    )
                    for item in bindings
                ]
        else:
            bindings = tuple(
                LogicalQueryBinding(
                    query_id=item.query_id,
                    text_hash=item.text_hash,
                    role=item.role,
                    vector_ref=QueryVectorRef(
                        physical_id=item.physical_id,
                        embedding_space_id=space_id,
                    ),
                    slot_id=item.slot_id,
                    text=item.text,
                )
                for item in normalized_queries
            )

        whole_physical_id = str(self.whole_physical_id or "").strip()
        whole_query = next(
            (item for item in normalized_queries if item.role == "whole"),
            None,
        )
        if whole_query is not None:
            whole_physical_id = whole_query.physical_id
        elif not whole_physical_id:
            whole_hash = hashlib.sha256(
                f"whole\0{self.model_id}\0{dimension}".encode("utf-8")
            ).hexdigest()
            whole_physical_id = self._legacy_physical_id(
                space_id, whole_hash, "whole"
            )
        if whole_physical_id not in physical_by_id:
            physical_by_id[whole_physical_id] = PhysicalQueryVector(
                physical_id=whole_physical_id,
                text_hash=hashlib.sha256(
                    f"whole\0{self.model_id}\0{dimension}".encode("utf-8")
                ).hexdigest(),
                vector=self.whole,
                embedding_space_id=space_id,
            )

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "embedding_space", space)
        object.__setattr__(self, "queries", tuple(normalized_queries))
        object.__setattr__(self, "logical_bindings", bindings)
        object.__setattr__(self, "physical_vectors", tuple(physical_by_id.values()))
        object.__setattr__(self, "whole_physical_id", whole_physical_id)

    @property
    def embedding_space_id(self) -> str:
        return self.embedding_space.canonical_id

    @property
    def physical_count(self) -> int:
        return len(self.physical_vectors)

    @property
    def logical_count(self) -> int:
        return len(self.logical_bindings)

    @property
    def atomic(self) -> tuple[QueryVector, ...]:
        return tuple(item for item in self.queries if item.role == "atomic")

    @property
    def followups(self) -> tuple[QueryVector, ...]:
        return tuple(item for item in self.queries if item.role == "followup")

    @property
    def whole_ref(self) -> QueryVectorRef:
        """Stable physical reference for the whole-question context vector."""

        return QueryVectorRef(
            physical_id=self.whole_physical_id,
            embedding_space_id=self.embedding_space_id,
        )

    def bindings_for_slot(self, slot_id: str) -> tuple[LogicalQueryBinding, ...]:
        wanted = str(slot_id).strip()
        return tuple(
            item for item in self.logical_bindings if item.slot_id == wanted
        )

    def query_refs_for_slot(self, slot_id: str) -> tuple[QueryVectorRef, ...]:
        """Return every logical slot binding without collapsing shared vectors."""

        return tuple(item.vector_ref for item in self.bindings_for_slot(slot_id))

    def bindings_for_query_id(self, query_id: str) -> tuple[LogicalQueryBinding, ...]:
        wanted = str(query_id).strip()
        return tuple(
            item for item in self.logical_bindings if item.query_id == wanted
        )

    def vector_for_physical(self, physical_id: str) -> Any:
        wanted = str(physical_id).strip()
        for item in self.physical_vectors:
            if item.physical_id == wanted:
                return item.vector
        raise KeyError(f"unknown physical query vector: {wanted}")

    def vector_for(self, value: QueryVector | LogicalQueryBinding | QueryVectorRef | str) -> Any:
        if isinstance(value, QueryVector):
            return self.vector_for_physical(value.physical_id)
        if isinstance(value, LogicalQueryBinding):
            return self.vector_for_physical(value.physical_id)
        if isinstance(value, QueryVectorRef):
            return self.vector_for_physical(value.physical_id)
        return self.vector_for_physical(str(value))

    def vector_for_slot(self, slot_id: str) -> Any:
        """Resolve exactly one slot vector; ambiguity is an explicit error."""

        bindings = self.bindings_for_slot(slot_id)
        if not bindings:
            raise KeyError(f"unknown query vector slot: {slot_id}")
        physical_ids = {item.physical_id for item in bindings}
        if len(physical_ids) != 1:
            raise ValueError("slot maps to more than one physical query vector")
        return self.vector_for(bindings[0])

    def metadata(self) -> dict[str, object]:
        """Safe structural metadata; deliberately excludes raw text and vectors."""

        return {
            "schema_version": int(self.schema_version),
            "embedding_space_id": self.embedding_space_id,
            "embedding_space": self.embedding_space.canonical_payload(),
            "model_id": str(self.model_id),
            "dimension": int(self.dimension),
            "whole_physical_id": str(self.whole_physical_id),
            "physical_vector_count": self.physical_count,
            "logical_binding_count": self.logical_count,
            "source_request_hash": str(self.source_request_hash),
        }


@dataclass(frozen=True, slots=True)
class EvidenceSlot:
    """One request-scoped obligation that selected evidence must cover.

    Slots are execution metadata, not durable knowledge.  Keeping the query
    identifier alongside the natural-language obligation lets a caller reuse
    the already-batched vector for a missing-slot repair without embedding a
    second copy of the question.
    """

    slot_id: str
    question: str = ""
    required: bool = True
    query_id: str = ""
    subject_terms: tuple[str, ...] = ()
    relation_hint: str = ""
    temporal_hint: str = ""
    epistemic_hint: str = ""
    # The fields below are appended deliberately: older callers construct
    # ``EvidenceSlot`` positionally, so the original positional layout must
    # remain valid while v3 can carry a request-frozen requirement contract.
    # They are request-scoped metadata only; they are never a replacement for
    # independent offline gold during evaluation.
    query_refs: tuple[str, ...] = ()
    origin: Literal["user_explicit", "planner", "reused_template"] = "planner"
    support_mode: Literal["alternative", "joint"] = "alternative"
    clause_ids: tuple[str, ...] = ()
    object_terms: tuple[str, ...] = ()
    negation_hint: str = ""
    modality_hint: str = ""

    def __post_init__(self) -> None:
        if not str(self.slot_id).strip():
            raise ValueError("evidence slot_id is required")
        if self.origin not in {"user_explicit", "planner", "reused_template"}:
            raise ValueError("evidence slot origin is invalid")
        if self.support_mode not in {"alternative", "joint"}:
            raise ValueError("evidence slot support_mode is invalid")

        # A frozen dataclass does not protect a caller that supplied a mutable
        # list.  Normalize all collection metadata to immutable, de-duplicated
        # tuples at construction so a resolved request cannot later have its
        # coverage semantics changed in place.
        for field_name in (
            "subject_terms",
            "query_refs",
            "clause_ids",
            "object_terms",
        ):
            raw_values = getattr(self, field_name)
            if isinstance(raw_values, str):
                raw_values = (raw_values,)
            values = tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in (raw_values or ())
                    if str(value).strip()
                )
            )
            object.__setattr__(self, field_name, values)


# These v3 provenance objects are deliberately request-local runtime values.
# They describe why an Episode is a candidate; they are not a new durable
# knowledge layer and they never make a relevance score into a fact claim.
ClauseVerificationStatus = Literal[
    "verified",
    "source_bound",
    "relevance_only",
    "unverified",
]


@dataclass(frozen=True, slots=True)
class SourceFactRef:
    """Immutable identity for one source-bound fact span.

    A logical ``source_key`` names a file-shaped import path, not a fact.  It
    is retained only as a display diagnostic and is intentionally excluded
    from equality/hash identity.  The identity instead closes over the source
    revision, the record/span locator and the raw-span digest.  Consequently,
    two independent facts in one long source are not made redundant merely
    because they share a source key, while a revised source cannot silently
    reuse an old support mapping.
    """

    source_revision_id: str
    record_span: tuple[str, ...] = ()
    span_hash: str = ""
    raw_span_hash: str = ""
    source_key: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        revision = str(self.source_revision_id or "").strip()
        if not revision:
            raise ValueError("source fact source_revision_id is required")
        raw_record_span = self.record_span
        if isinstance(raw_record_span, str):
            raw_record_span = (raw_record_span,)
        record_span = tuple(
            str(value).strip()
            for value in (raw_record_span or ())
            if str(value).strip()
        )
        span_hash = str(self.span_hash or "").strip()
        raw_span_hash = str(self.raw_span_hash or "").strip()
        if not record_span and not span_hash:
            raise ValueError(
                "source fact needs a record_span or source-span hash"
            )
        if not raw_span_hash:
            raise ValueError("source fact raw_span_hash is required")
        object.__setattr__(self, "source_revision_id", revision)
        object.__setattr__(self, "record_span", record_span)
        object.__setattr__(self, "span_hash", span_hash)
        object.__setattr__(self, "raw_span_hash", raw_span_hash)
        object.__setattr__(self, "source_key", str(self.source_key or "").strip())

    @property
    def source_revision(self) -> str:
        """Compatibility-friendly spelling for diagnostics."""

        return self.source_revision_id

    @property
    def identity_key(self) -> tuple[str, tuple[str, ...], str, str]:
        """The only key permitted for source-fact de-duplication."""

        return (
            self.source_revision_id,
            self.record_span,
            self.span_hash,
            self.raw_span_hash,
        )

    @property
    def fact_id(self) -> str:
        """Stable opaque identifier suitable for an internal trace."""

        payload = json.dumps(
            {
                "source_revision_id": self.source_revision_id,
                "record_span": self.record_span,
                "span_hash": self.span_hash,
                "raw_span_hash": self.raw_span_hash,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "source-fact:sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ClauseSupport:
    """One source-bound mapping from a candidate to a requirement clause.

    ``relevance_only`` is intentionally represented rather than discarded: it
    remains useful for ranking/diagnostics, but ``is_required_coverage`` is
    false so a model score can never satisfy a required fact clause by itself.
    """

    slot_id: str
    clause_id: str
    source_fact: SourceFactRef | None = None
    support_mode: Literal["alternative", "joint"] = "alternative"
    verification_status: ClauseVerificationStatus = "relevance_only"
    mapping_ref: str = ""

    def __post_init__(self) -> None:
        slot_id = str(self.slot_id or "").strip()
        clause_id = str(self.clause_id or "").strip()
        if not slot_id or not clause_id:
            raise ValueError("clause support slot_id and clause_id are required")
        if self.support_mode not in {"alternative", "joint"}:
            raise ValueError("clause support mode is invalid")
        if self.verification_status not in {
            "verified",
            "source_bound",
            "relevance_only",
            "unverified",
        }:
            raise ValueError("clause support verification_status is invalid")
        if self.source_fact is not None and not isinstance(
            self.source_fact, SourceFactRef
        ):
            raise TypeError("clause support source_fact must be a SourceFactRef")
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "clause_id", clause_id)
        object.__setattr__(self, "mapping_ref", str(self.mapping_ref or "").strip())

    @property
    def requirement_id(self) -> str:
        return self.slot_id

    @property
    def is_required_coverage(self) -> bool:
        """Only verified/source-bound mappings with a fact locator can cover."""

        return bool(
            self.source_fact is not None
            and self.verification_status in {"verified", "source_bound"}
        )


def _unique_text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    return tuple(
        dict.fromkeys(
            str(item).strip() for item in (value or ()) if str(item).strip()
        )
    )


def _unique_source_facts(
    values: Any,
) -> tuple[SourceFactRef, ...]:
    if values is None:
        return ()
    result: dict[tuple[str, tuple[str, ...], str, str], SourceFactRef] = {}
    for item in values:
        if not isinstance(item, SourceFactRef):
            raise TypeError("source facts must be SourceFactRef instances")
        result.setdefault(item.identity_key, item)
    return tuple(
        result[key]
        for key in sorted(
            result,
            key=lambda key: (key[0], key[1], key[2], key[3]),
        )
    )


@dataclass(frozen=True, slots=True)
class CandidateContribution:
    """One provenance-preserving route by which an Episode became a candidate.

    An Episode can have many instances of this class: e.g. dense and sparse
    base lanes, several slot mappings, and several contextual edges.  They
    must remain distinct until a request-level aggregate is formed, because a
    later contribution mask may remove one edge without removing an independent
    base route to the exact same Episode.
    """

    contribution_id: str
    episode_id: int
    slot_id: str
    lane: str
    edge_id: int | None = None
    parent_contribution_ids: tuple[str, ...] = ()
    query_ref: str = ""
    rank_features: Any = ()
    source_facts: tuple[SourceFactRef, ...] = ()
    clause_supports: tuple[ClauseSupport, ...] = ()
    source_span_refs: tuple[str, ...] = ()
    evidence_ref: str = ""
    delivery_token_cost: int = 0

    def __post_init__(self) -> None:
        contribution_id = str(self.contribution_id or "").strip()
        slot_id = str(self.slot_id or "").strip()
        lane = str(self.lane or "").strip()
        if not contribution_id or not slot_id or not lane:
            raise ValueError(
                "candidate contribution id, episode slot_id and lane are required"
            )
        if int(self.episode_id) <= 0:
            raise ValueError("candidate contribution episode_id must be positive")
        edge_id = self.edge_id
        if edge_id is not None and int(edge_id) <= 0:
            raise ValueError("candidate contribution edge_id must be positive")
        if int(self.delivery_token_cost) < 0:
            raise ValueError("candidate contribution delivery_token_cost cannot be negative")
        parents = _unique_text_tuple(self.parent_contribution_ids)
        if contribution_id in parents:
            raise ValueError("candidate contribution cannot parent itself")

        raw_features = self.rank_features
        feature_items = (
            raw_features.items()
            if isinstance(raw_features, Mapping)
            else (raw_features or ())
        )
        features: dict[str, float | None] = {}
        for pair in feature_items:
            try:
                name, raw_value = pair
            except (TypeError, ValueError) as exc:
                raise TypeError("rank_features must contain name/value pairs") from exc
            feature_name = str(name or "").strip()
            if not feature_name:
                raise ValueError("rank feature name is required")
            if raw_value is None:
                value: float | None = None
            else:
                value = float(raw_value)
                if not math.isfinite(value):
                    raise ValueError("rank feature values must be finite")
            features[feature_name] = value

        raw_supports = self.clause_supports
        supports: dict[ClauseSupport, ClauseSupport] = {}
        for support in raw_supports or ():
            if not isinstance(support, ClauseSupport):
                raise TypeError("clause_supports must contain ClauseSupport instances")
            supports.setdefault(support, support)
        explicit_facts = _unique_source_facts(self.source_facts)
        support_facts = _unique_source_facts(
            item.source_fact for item in supports if item.source_fact is not None
        )
        facts = _unique_source_facts((*explicit_facts, *support_facts))

        object.__setattr__(self, "contribution_id", contribution_id)
        object.__setattr__(self, "episode_id", int(self.episode_id))
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "lane", lane)
        object.__setattr__(self, "edge_id", int(edge_id) if edge_id is not None else None)
        object.__setattr__(self, "parent_contribution_ids", parents)
        object.__setattr__(self, "query_ref", str(self.query_ref or "").strip())
        object.__setattr__(self, "rank_features", tuple(sorted(features.items())))
        object.__setattr__(
            self,
            "clause_supports",
            tuple(
                sorted(
                    supports,
                    key=lambda item: (
                        item.slot_id,
                        item.clause_id,
                        item.support_mode,
                        item.verification_status,
                        item.source_fact.identity_key
                        if item.source_fact is not None
                        else ("", (), "", ""),
                        item.mapping_ref,
                    ),
                )
            ),
        )
        object.__setattr__(self, "source_facts", facts)
        object.__setattr__(self, "source_span_refs", _unique_text_tuple(self.source_span_refs))
        object.__setattr__(self, "evidence_ref", str(self.evidence_ref or "").strip())
        object.__setattr__(self, "delivery_token_cost", int(self.delivery_token_cost))

    @property
    def feature_map(self) -> dict[str, float | None]:
        return dict(self.rank_features)

    @property
    def relevance_score(self) -> float:
        """A deterministic ranking-only value; never a coverage assertion."""

        values = self.feature_map
        for name in (
            "current_relevance_estimate",
            "cross_encoder_score",
            "fusion_rank_score",
            "vector_cosine",
            "sparse_score",
        ):
            value = values.get(name)
            if value is not None:
                return float(value)
        return 0.0

    @property
    def is_contextual(self) -> bool:
        return self.edge_id is not None or self.lane == "contextual"


@dataclass(frozen=True, slots=True)
class CandidateAggregate:
    """All candidate contributions attached to one distinct Episode."""

    episode_id: int
    contributions: tuple[CandidateContribution, ...]
    source_facts: tuple[SourceFactRef, ...] = ()
    source_span_refs: tuple[str, ...] = ()
    delivery_token_cost: int = 0

    def __post_init__(self) -> None:
        episode_id = int(self.episode_id)
        if episode_id <= 0:
            raise ValueError("candidate aggregate episode_id must be positive")
        if int(self.delivery_token_cost) < 0:
            raise ValueError("candidate aggregate delivery_token_cost cannot be negative")
        by_id: dict[str, CandidateContribution] = {}
        for contribution in self.contributions or ():
            if not isinstance(contribution, CandidateContribution):
                raise TypeError("aggregate contributions must be CandidateContribution instances")
            if contribution.episode_id != episode_id:
                raise ValueError("aggregate cannot merge different episode ids")
            existing = by_id.get(contribution.contribution_id)
            if existing is not None and existing != contribution:
                raise ValueError("contribution_id is ambiguous within an aggregate")
            by_id.setdefault(contribution.contribution_id, contribution)
        if not by_id:
            raise ValueError("candidate aggregate needs at least one contribution")
        contributions = tuple(by_id[key] for key in sorted(by_id))
        contribution_facts = _unique_source_facts(
            fact for item in contributions for fact in item.source_facts
        )
        facts = _unique_source_facts((*self.source_facts, *contribution_facts))
        spans = _unique_text_tuple(
            (*self.source_span_refs, *(
                span for item in contributions for span in item.source_span_refs
            ))
        )
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "source_facts", facts)
        object.__setattr__(self, "source_span_refs", spans)
        object.__setattr__(
            self,
            "delivery_token_cost",
            max(
                int(self.delivery_token_cost),
                max(
                    (item.delivery_token_cost for item in contributions),
                    default=0,
                ),
            ),
        )

    @property
    def contribution_ids(self) -> tuple[str, ...]:
        return tuple(item.contribution_id for item in self.contributions)

    @property
    def clause_supports(self) -> tuple[ClauseSupport, ...]:
        return tuple(
            support
            for item in self.contributions
            for support in item.clause_supports
        )

    @property
    def relevance_score(self) -> float:
        """Combine independent ranking routes without multiplying evidence.

        A contextual route is a bounded inspection-priority signal, not a new
        source fact and not a requirement mapping.  When the same Episode was
        also found by ordinary retrieval, keep both route contributions and
        combine their already-normalised priorities with the simple
        ``1 - (1-base) * (1-contextual)`` union.  Masking an edge removes only
        its contribution, returning the independent base priority unchanged.
        """

        base_scores = [
            float(item.relevance_score)
            for item in self.contributions
            if not item.is_contextual
        ]
        base_score = max(base_scores, default=0.0)
        contextual_scores: list[float] = []
        for item in self.contributions:
            if not item.is_contextual:
                continue
            raw_priority = item.feature_map.get("contextual_priority_score")
            if raw_priority is None:
                continue
            try:
                priority = float(raw_priority)
            except (TypeError, ValueError):
                continue
            if math.isfinite(priority):
                contextual_scores.append(min(1.0, max(0.0, priority)))
        if not contextual_scores:
            return base_score
        contextual_score = max(contextual_scores)
        normalized_base = min(1.0, max(0.0, base_score))
        return 1.0 - (1.0 - normalized_base) * (1.0 - contextual_score)

    @property
    def contextual_contribution_ids(self) -> tuple[str, ...]:
        return tuple(
            item.contribution_id for item in self.contributions if item.is_contextual
        )

    @property
    def base_contribution_ids(self) -> tuple[str, ...]:
        return tuple(
            item.contribution_id for item in self.contributions if not item.is_contextual
        )


@dataclass(frozen=True, slots=True)
class ClauseRequirement:
    """An authoritative coverage clause with alternative/joint semantics."""

    slot_id: str
    clause_ids: tuple[str, ...] = ()
    support_mode: Literal["alternative", "joint"] = "alternative"
    required: bool = True

    def __post_init__(self) -> None:
        slot_id = str(self.slot_id or "").strip()
        if not slot_id:
            raise ValueError("clause requirement slot_id is required")
        clause_ids = _unique_text_tuple(self.clause_ids) or (slot_id,)
        if self.support_mode not in {"alternative", "joint"}:
            raise ValueError("clause requirement support_mode is invalid")
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "clause_ids", clause_ids)

    @classmethod
    def from_evidence_slot(cls, slot: EvidenceSlot) -> "ClauseRequirement":
        return cls(
            slot_id=slot.slot_id,
            clause_ids=slot.clause_ids,
            support_mode=slot.support_mode,
            required=slot.required,
        )

    @property
    def requirement_id(self) -> str:
        return self.slot_id


@dataclass(frozen=True, slots=True)
class EvidenceSelectionBudget:
    """Explicit request budget for evidence selection and delivery planning."""

    episode_limit: int
    source_fact_limit: int | None = None
    delivery_token_limit: int | None = None

    def __post_init__(self) -> None:
        if int(self.episode_limit) < 0:
            raise ValueError("evidence selection episode_limit cannot be negative")
        for field_name in ("source_fact_limit", "delivery_token_limit"):
            raw_value = getattr(self, field_name)
            if raw_value is not None and int(raw_value) < 0:
                raise ValueError(f"evidence selection {field_name} cannot be negative")
        object.__setattr__(self, "episode_limit", int(self.episode_limit))
        if self.source_fact_limit is not None:
            object.__setattr__(self, "source_fact_limit", int(self.source_fact_limit))
        if self.delivery_token_limit is not None:
            object.__setattr__(
                self,
                "delivery_token_limit",
                int(self.delivery_token_limit),
            )

    @property
    def max_episodes(self) -> int:
        return self.episode_limit


# Naming alias for callers that prefer the request-oriented vocabulary from the
# v3 design.  It is an alias, not a second mutable budget representation.
RequestEvidenceBudget = EvidenceSelectionBudget


@dataclass(frozen=True, slots=True)
class SlotCandidate:
    """A local Episode-to-slot support record used by set-cover selection."""

    episode_id: int
    slot_ids: frozenset[str] = frozenset()
    source_lane: str = "base"
    direct_score: float = 0.0
    specificity_score: float = 0.0
    redundancy_group: str = ""
    contextual_edge_id: int | None = None
    contextual_score: float = 0.0
    source_quality: float = 0.0


@dataclass(frozen=True, slots=True)
class ContextualSlotHit:
    """A double-key recall proposal tied to one unresolved evidence slot."""

    association_id: int
    anchor_episode_id: int
    target_episode_id: int
    matched_slot_id: str = ""
    matched_query_id: str = ""
    context_similarity: float = 0.0
    need_similarity: float = 0.0
    anchor_activation: float = 0.0
    utility_weight: float = 0.0
    target_support_score: float = 0.0
    lifecycle_state: str = "probation"
    total_score: float = 0.0

    @property
    def score(self) -> float:
        """Compatibility name for the v1 trace shape."""

        return self.total_score


@dataclass(frozen=True, slots=True)
class CuePrototype:
    id: int
    domain: str
    cue_kind: Literal["context", "need"]
    model_id: str
    dimension: int
    dtype: str
    vector: Any
    text_hash: str
    display_text: str = ""
    source_request_hash: str = ""
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class ContextualRecallCandidate:
    """A retrieval-only edge candidate; it contains no factual claim."""

    anchor_type: NodeType
    anchor_id: int
    target_episode_id: int
    context_query_id: str
    need_query_id: str
    slot_id: str = ""
    source_request_hash: str = ""
    reason: str = "recovered_missing_evidence"


LearningCandidateReason = Literal[
    "candidate_missing",
    "candidate_present_dropped",
    "source_only",
    "clarification_recovery",
    "costly_success_reuse",
]


def _unique_positive_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    """Normalize an ID collection without turning an invalid ID into a fact.

    Learning records deliberately retain only stable database IDs and opaque
    references.  A zero/negative value would make a later durable receipt
    ambiguous, so reject it at the boundary instead of silently filtering it.
    """

    if value is None:
        return ()
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        value = (value,)
    result: list[int] = []
    seen: set[int] = set()
    for raw_value in value:
        if isinstance(raw_value, bool):
            raise ValueError(f"{field_name} must contain positive integer IDs")
        try:
            normalized = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must contain positive integer IDs"
            ) from exc
        if normalized <= 0:
            raise ValueError(f"{field_name} must contain positive integer IDs")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LearningAnchor:
    """One independently found, source-bound entrance to a learning edge.

    This is request-local provenance, not a new graph node.  ``lane`` is kept
    explicit so a contextual expansion can never be re-labelled as a base
    anchor after the fact.  Missing proof is representable for a rejected
    event, but such an anchor is not eligible for candidate derivation.
    """

    anchor_type: NodeType
    anchor_id: int
    contribution_id: str
    source_facts: tuple[SourceFactRef, ...] = ()
    vector_ref: str = ""
    provenance_refs: tuple[str, ...] = ()
    activation: float = 0.0
    lane: str = "base"
    independent: bool = True

    def __post_init__(self) -> None:
        if self.anchor_type not in {"episode", "concept"}:
            raise ValueError("learning anchor type is invalid")
        anchor_id = int(self.anchor_id)
        if anchor_id <= 0:
            raise ValueError("learning anchor id must be positive")
        contribution_id = str(self.contribution_id or "").strip()
        if not contribution_id:
            raise ValueError("learning anchor contribution_id is required")
        activation = float(self.activation)
        if not math.isfinite(activation):
            raise ValueError("learning anchor activation must be finite")
        object.__setattr__(self, "anchor_id", anchor_id)
        object.__setattr__(self, "contribution_id", contribution_id)
        object.__setattr__(self, "source_facts", _unique_source_facts(self.source_facts))
        object.__setattr__(self, "vector_ref", str(self.vector_ref or "").strip())
        object.__setattr__(self, "provenance_refs", _unique_text_tuple(self.provenance_refs))
        object.__setattr__(self, "activation", activation)
        object.__setattr__(self, "lane", str(self.lane or "").strip().casefold())
        object.__setattr__(self, "independent", bool(self.independent))

    @property
    def is_contextual(self) -> bool:
        return self.lane in {"contextual", "contextual_recall", "derived"}

    @property
    def is_eligible_base_anchor(self) -> bool:
        return bool(
            self.independent
            and not self.is_contextual
            and self.source_facts
            and self.vector_ref
            and self.provenance_refs
        )


@dataclass(frozen=True, slots=True)
class RecallLearningEvent:
    """Immutable evidence from one completed retrieval round.

    The event records the before/after candidate and delivery sets separately.
    This makes ``candidate_missing`` distinguishable from
    ``candidate_present_dropped`` and makes a successful but expensive first
    pass eligible for a *reuse* proposal.  It contains no durable IDs and does
    not itself create an association or quality observation.
    """

    request_id: str
    request_hash: str
    domain: str
    target_episode_ids: tuple[int, ...]
    initial_candidate_episode_ids: tuple[int, ...] = ()
    initial_delivered_episode_ids: tuple[int, ...] = ()
    final_selected_episode_ids: tuple[int, ...] = ()
    final_delivered_episode_ids: tuple[int, ...] = ()
    contextual_expansion_episode_ids: tuple[int, ...] = ()
    source_request_hash: str = ""
    slot_id: str = ""
    context_query_id: str = ""
    need_query_id: str = ""
    context_vector_ref: str = ""
    need_vector_ref: str = ""
    source_facts: tuple[SourceFactRef, ...] = ()
    verification_refs: tuple[str, ...] = ()
    verification_status: ClauseVerificationStatus = "unverified"
    target_provenance_refs: tuple[str, ...] = ()
    costly_stage_refs: tuple[str, ...] = ()
    answer_guard_passed: bool = True
    cancelled: bool = False

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        request_hash = str(self.request_hash or "").strip()
        domain = str(self.domain or "").strip()
        if not request_id or not request_hash or not domain:
            raise ValueError(
                "learning event request_id, request_hash and domain are required"
            )
        target_ids = _unique_positive_int_tuple(
            self.target_episode_ids, "learning event target_episode_ids"
        )
        if not target_ids:
            raise ValueError("learning event needs at least one target episode")
        initial_candidates = _unique_positive_int_tuple(
            self.initial_candidate_episode_ids,
            "learning event initial_candidate_episode_ids",
        )
        initial_delivered = _unique_positive_int_tuple(
            self.initial_delivered_episode_ids,
            "learning event initial_delivered_episode_ids",
        )
        final_selected = _unique_positive_int_tuple(
            self.final_selected_episode_ids,
            "learning event final_selected_episode_ids",
        )
        final_delivered = _unique_positive_int_tuple(
            self.final_delivered_episode_ids,
            "learning event final_delivered_episode_ids",
        )
        contextual = _unique_positive_int_tuple(
            self.contextual_expansion_episode_ids,
            "learning event contextual_expansion_episode_ids",
        )
        if not set(initial_delivered).issubset(initial_candidates):
            raise ValueError(
                "learning event initial delivered episodes must be initial candidates"
            )
        if not set(final_delivered).issubset(final_selected):
            raise ValueError(
                "learning event final delivered episodes must be final selections"
            )
        verification_status = str(self.verification_status or "").strip()
        if verification_status not in {
            "verified",
            "source_bound",
            "relevance_only",
            "unverified",
        }:
            raise ValueError("learning event verification_status is invalid")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "request_hash", request_hash)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "target_episode_ids", target_ids)
        object.__setattr__(self, "initial_candidate_episode_ids", initial_candidates)
        object.__setattr__(self, "initial_delivered_episode_ids", initial_delivered)
        object.__setattr__(self, "final_selected_episode_ids", final_selected)
        object.__setattr__(self, "final_delivered_episode_ids", final_delivered)
        object.__setattr__(self, "contextual_expansion_episode_ids", contextual)
        object.__setattr__(
            self,
            "source_request_hash",
            str(self.source_request_hash or request_hash).strip(),
        )
        for field_name in (
            "context_query_id",
            "need_query_id",
            "context_vector_ref",
            "need_vector_ref",
        ):
            object.__setattr__(self, field_name, str(getattr(self, field_name) or "").strip())
        object.__setattr__(self, "slot_id", str(self.slot_id or "").strip())
        object.__setattr__(self, "source_facts", _unique_source_facts(self.source_facts))
        object.__setattr__(self, "verification_refs", _unique_text_tuple(self.verification_refs))
        object.__setattr__(self, "verification_status", verification_status)
        object.__setattr__(
            self,
            "target_provenance_refs",
            _unique_text_tuple(self.target_provenance_refs),
        )
        object.__setattr__(self, "costly_stage_refs", _unique_text_tuple(self.costly_stage_refs))
        object.__setattr__(self, "answer_guard_passed", bool(self.answer_guard_passed))
        object.__setattr__(self, "cancelled", bool(self.cancelled))

    @property
    def has_verified_source_evidence(self) -> bool:
        return bool(
            self.source_facts
            and self.verification_refs
            and self.verification_status in {"verified", "source_bound"}
        )

    @property
    def is_creation_eligible(self) -> bool:
        return bool(
            self.answer_guard_passed
            and not self.cancelled
            and self.slot_id
            and self.context_query_id
            and self.need_query_id
            and self.context_vector_ref
            and self.need_vector_ref
            and self.target_provenance_refs
            and self.has_verified_source_evidence
        )

    def reason_for_target(self, target_episode_id: int) -> LearningCandidateReason | None:
        """Return a proven creation reason, never an inferred success claim."""

        target = int(target_episode_id)
        if (
            target not in self.target_episode_ids
            or target not in self.final_delivered_episode_ids
            or target in self.contextual_expansion_episode_ids
            or not self.is_creation_eligible
        ):
            return None
        if target not in self.initial_candidate_episode_ids:
            return "candidate_missing"
        if target not in self.initial_delivered_episode_ids:
            return "candidate_present_dropped"
        if self.costly_stage_refs:
            return "costly_success_reuse"
        return None


# Short public spelling for callers that do not need to distinguish a recall
# event from future learning-event variants.  It remains the same immutable
# record, not a parallel mutable representation.
LearningEvent = RecallLearningEvent


@dataclass(frozen=True, slots=True)
class LearningCandidate:
    """A fully evidenced, non-persistent proposal to create one recall edge."""

    candidate_id: str
    anchor_type: NodeType
    anchor_id: int
    target_episode_id: int
    reason: LearningCandidateReason
    request_id: str
    request_hash: str
    source_request_hash: str
    context_query_id: str
    need_query_id: str
    slot_id: str
    context_vector_ref: str
    need_vector_ref: str
    anchor_vector_ref: str
    source_facts: tuple[SourceFactRef, ...]
    verification_refs: tuple[str, ...]
    verification_status: ClauseVerificationStatus
    anchor_contribution_id: str
    anchor_provenance_refs: tuple[str, ...]
    target_provenance_refs: tuple[str, ...]
    costly_stage_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.anchor_type not in {"episode", "concept"}:
            raise ValueError("learning candidate anchor type is invalid")
        anchor_id = int(self.anchor_id)
        target_episode_id = int(self.target_episode_id)
        if anchor_id <= 0 or target_episode_id <= 0:
            raise ValueError("learning candidate endpoint IDs must be positive")
        if self.anchor_type == "episode" and anchor_id == target_episode_id:
            raise ValueError("learning candidate cannot create an episode self-edge")
        if self.reason not in {
            "candidate_missing",
            "candidate_present_dropped",
            "source_only",
            "clarification_recovery",
            "costly_success_reuse",
        }:
            raise ValueError("learning candidate reason is invalid")
        required_text = (
            "candidate_id",
            "request_id",
            "request_hash",
            "source_request_hash",
            "context_query_id",
            "need_query_id",
            "slot_id",
            "context_vector_ref",
            "need_vector_ref",
            "anchor_vector_ref",
            "anchor_contribution_id",
        )
        normalized: dict[str, str] = {}
        for field_name in required_text:
            text = str(getattr(self, field_name) or "").strip()
            if not text:
                raise ValueError(f"learning candidate {field_name} is required")
            normalized[field_name] = text
        source_facts = _unique_source_facts(self.source_facts)
        verification_refs = _unique_text_tuple(self.verification_refs)
        anchor_provenance_refs = _unique_text_tuple(self.anchor_provenance_refs)
        target_provenance_refs = _unique_text_tuple(self.target_provenance_refs)
        costly_stage_refs = _unique_text_tuple(self.costly_stage_refs)
        if not source_facts or not verification_refs:
            raise ValueError("learning candidate needs source and verification references")
        if self.verification_status not in {"verified", "source_bound"}:
            raise ValueError("learning candidate must have verified source evidence")
        if not anchor_provenance_refs or not target_provenance_refs:
            raise ValueError("learning candidate needs anchor and target provenance")
        if self.reason == "costly_success_reuse" and not costly_stage_refs:
            raise ValueError("costly reuse candidate needs costly stage references")
        for field_name, value in normalized.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "anchor_id", anchor_id)
        object.__setattr__(self, "target_episode_id", target_episode_id)
        object.__setattr__(self, "source_facts", source_facts)
        object.__setattr__(self, "verification_refs", verification_refs)
        object.__setattr__(self, "anchor_provenance_refs", anchor_provenance_refs)
        object.__setattr__(self, "target_provenance_refs", target_provenance_refs)
        object.__setattr__(self, "costly_stage_refs", costly_stage_refs)

    @property
    def creation_is_quality_observation(self) -> bool:
        """A creation receipt is never evidence of a future successful reuse."""

        return False

    def as_contextual_candidate(self) -> ContextualRecallCandidate:
        """Lossless endpoint conversion for the legacy persistence interface."""

        return ContextualRecallCandidate(
            anchor_type=self.anchor_type,
            anchor_id=self.anchor_id,
            target_episode_id=self.target_episode_id,
            context_query_id=self.context_query_id,
            need_query_id=self.need_query_id,
            slot_id=self.slot_id,
            source_request_hash=self.source_request_hash,
            reason=self.reason,
        )


@dataclass(frozen=True, slots=True)
class LearningCandidateRejection:
    """A deterministic, local reason a potential learning edge was rejected."""

    code: str
    target_episode_id: int = 0
    anchor_id: int = 0

    def __post_init__(self) -> None:
        code = str(self.code or "").strip()
        if not code:
            raise ValueError("learning candidate rejection code is required")
        target = int(self.target_episode_id)
        anchor = int(self.anchor_id)
        if target < 0 or anchor < 0:
            raise ValueError("learning candidate rejection IDs cannot be negative")
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "target_episode_id", target)
        object.__setattr__(self, "anchor_id", anchor)


@dataclass(frozen=True, slots=True)
class LearningCandidatePlan:
    """Pure derivation output; T13 will persist it atomically if approved."""

    request_id: str
    selected_anchors: tuple[LearningAnchor, ...] = ()
    candidates: tuple[LearningCandidate, ...] = ()
    rejections: tuple[LearningCandidateRejection, ...] = ()

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise ValueError("learning candidate plan request_id is required")
        anchors = tuple(self.selected_anchors)
        candidates = tuple(self.candidates)
        rejections = tuple(self.rejections)
        if any(not isinstance(item, LearningAnchor) for item in anchors):
            raise TypeError("learning candidate plan anchors must be LearningAnchor instances")
        if any(not isinstance(item, LearningCandidate) for item in candidates):
            raise TypeError("learning candidate plan candidates must be LearningCandidate instances")
        if any(not isinstance(item, LearningCandidateRejection) for item in rejections):
            raise TypeError("learning candidate plan rejections must be LearningCandidateRejection instances")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "selected_anchors", anchors)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "rejections", rejections)


# v1 callers exposed this name in traces.  The richer v2 object remains fully
# compatible for attribute reads while making the matched evidence slot and
# anchor activation explicit.
AssociationCueHit = ContextualSlotHit


@dataclass(frozen=True, slots=True)
class ContextualUtilityObservation:
    association_id: int
    query_hash: str
    outcome: Literal["sufficient", "necessary", "redundant", "no_op", "harmful"]
    delta_slots: int = 0
    treatment_episode_ids: tuple[int, ...] = ()
    masked_episode_ids: tuple[int, ...] = ()
    attribution: Literal["batch", "single_edge", "leave_one_out"] = "batch"


_UTILITY_OPAQUE_DIGEST_RE = re.compile(
    r"^(?:[a-z][a-z0-9_.-]*:)?sha256:[0-9a-f]{64}$"
)


def _utility_opaque_digest(value: object, field_name: str) -> str:
    """Accept only versioned opaque digests in durable utility records."""

    normalized = str(value or "").strip()
    if not _UTILITY_OPAQUE_DIGEST_RE.fullmatch(normalized):
        raise ValueError(f"utility ledger {field_name} must be an opaque sha256 ID")
    return normalized


def _utility_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"utility ledger {field_name} must be a non-negative integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"utility ledger {field_name} must be a non-negative integer"
        ) from error
    if normalized < 0:
        raise ValueError(f"utility ledger {field_name} cannot be negative")
    return normalized


@dataclass(frozen=True, slots=True)
class ContextualUtilityLedgerObservation:
    """One source-bound, immutable V3 contextual-utility observation.

    This is deliberately not an extension of the historical mutable
    ``ContextualUtilityObservation``.  It names an already-ready creation
    receipt and carries opaque V3 selector fingerprints rather than question,
    source key, source text, or a mutable Association summary.
    """

    observation_id: str
    family_id: str
    association_id: int
    creation_receipt_id: int
    evaluation_as_of: str
    candidate_universe_fingerprint: str
    requirements_fingerprint: str
    budget_fingerprint: str
    input_fingerprint: str
    treatment_fingerprint: str
    masked_fingerprint: str
    single_edge_fingerprint: str
    leave_one_out_fingerprint: str
    factual_support_verified: bool
    is_shadow: bool
    treatment_episode_count: int = 0
    masked_episode_count: int = 0
    treatment_required_count: int = 0
    masked_required_count: int = 0
    treatment_gain_count: int = 0
    treatment_loss_count: int = 0
    single_edge_gain_count: int = 0
    single_edge_loss_count: int = 0
    leave_one_out_gain_count: int = 0
    leave_one_out_loss_count: int = 0
    recall_gain: int = 0
    work_metric: Literal["", "provider_receipt_delta"] = ""
    treatment_work: int | None = None
    masked_work: int | None = None
    work_saved: int | None = None
    provider_receipt_refs: tuple[str, ...] = ()
    harm: bool = False
    outcome: Literal[
        "recall_gain", "equal_quality_faster", "no_op", "harmful"
    ] = "no_op"

    def __post_init__(self) -> None:
        for field_name in (
            "observation_id",
            "family_id",
            "candidate_universe_fingerprint",
            "requirements_fingerprint",
            "budget_fingerprint",
            "input_fingerprint",
            "treatment_fingerprint",
            "masked_fingerprint",
            "single_edge_fingerprint",
            "leave_one_out_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _utility_opaque_digest(getattr(self, field_name), field_name),
            )
        for field_name in ("association_id", "creation_receipt_id"):
            value = _utility_nonnegative_int(getattr(self, field_name), field_name)
            if value <= 0:
                raise ValueError(f"utility ledger {field_name} must be positive")
            object.__setattr__(self, field_name, value)
        evaluation_as_of = str(self.evaluation_as_of or "").strip()
        if not evaluation_as_of or "\n" in evaluation_as_of or "\r" in evaluation_as_of:
            raise ValueError("utility ledger evaluation_as_of is required")
        object.__setattr__(self, "evaluation_as_of", evaluation_as_of)
        if not isinstance(self.factual_support_verified, bool):
            raise TypeError("utility ledger factual_support_verified must be bool")
        if not isinstance(self.is_shadow, bool):
            raise TypeError("utility ledger is_shadow must be bool")
        if not isinstance(self.harm, bool):
            raise TypeError("utility ledger harm must be bool")
        for field_name in (
            "treatment_episode_count",
            "masked_episode_count",
            "treatment_required_count",
            "masked_required_count",
            "treatment_gain_count",
            "treatment_loss_count",
            "single_edge_gain_count",
            "single_edge_loss_count",
            "leave_one_out_gain_count",
            "leave_one_out_loss_count",
            "recall_gain",
        ):
            object.__setattr__(
                self,
                field_name,
                _utility_nonnegative_int(getattr(self, field_name), field_name),
            )
        loss_detected = bool(
            self.treatment_loss_count
            or self.single_edge_loss_count
            or self.leave_one_out_loss_count
        )
        if self.harm != loss_detected:
            raise ValueError(
                "utility ledger harm must exactly reflect required-coverage loss"
            )
        max_observed_gain = max(
            self.treatment_gain_count,
            self.single_edge_gain_count,
            self.leave_one_out_gain_count,
        )
        if self.recall_gain > max_observed_gain:
            raise ValueError("utility ledger recall_gain exceeds observed gain")
        if self.recall_gain and not self.factual_support_verified:
            raise ValueError(
                "relevance-only contextual support cannot claim recall_gain"
            )
        if not self.factual_support_verified and max_observed_gain:
            raise ValueError(
                "relevance-only contextual support cannot claim required coverage"
            )
        if self.harm and self.recall_gain:
            raise ValueError("harmful utility observations cannot receive recall_gain")

        work_metric = str(self.work_metric or "")
        if work_metric not in {"", "provider_receipt_delta"}:
            raise ValueError("utility ledger work_metric is invalid")
        raw_receipts = tuple(self.provider_receipt_refs or ())
        receipt_refs = tuple(
            sorted(
                {
                    _utility_opaque_digest(value, "provider_receipt_ref")
                    for value in raw_receipts
                }
            )
        )
        work_values = (
            self.treatment_work,
            self.masked_work,
            self.work_saved,
        )
        if work_metric == "":
            if any(value is not None for value in work_values) or receipt_refs:
                raise ValueError("unmeasured work must remain unknown")
        else:
            if not receipt_refs or any(value is None for value in work_values):
                raise ValueError("measured work requires provider receipt references")
            treatment_work = _utility_nonnegative_int(
                self.treatment_work, "treatment_work"
            )
            masked_work = _utility_nonnegative_int(self.masked_work, "masked_work")
            work_saved = _utility_nonnegative_int(self.work_saved, "work_saved")
            if work_saved != masked_work - treatment_work:
                raise ValueError("utility ledger work_saved must be the measured delta")
            object.__setattr__(self, "treatment_work", treatment_work)
            object.__setattr__(self, "masked_work", masked_work)
            object.__setattr__(self, "work_saved", work_saved)
        object.__setattr__(self, "work_metric", work_metric)
        object.__setattr__(self, "provider_receipt_refs", receipt_refs)

        expected_outcome: str
        if self.harm:
            expected_outcome = "harmful"
        elif self.recall_gain:
            expected_outcome = "recall_gain"
        elif self.work_saved is not None and self.work_saved > 0:
            expected_outcome = "equal_quality_faster"
        else:
            expected_outcome = "no_op"
        if self.outcome != expected_outcome:
            raise ValueError(
                "utility ledger outcome must use harm-first measured semantics"
            )


@dataclass(frozen=True, slots=True)
class ContextualUtilityLedgerReceipt:
    """Result of one append-only ledger write or an exact idempotent retry."""

    ledger_id: int
    observation_id: str
    family_id: str
    association_id: int
    creation_receipt_id: int
    idempotent: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        ledger_id = _utility_nonnegative_int(self.ledger_id, "ledger_id")
        if ledger_id <= 0:
            raise ValueError("utility ledger receipt ledger_id must be positive")
        object.__setattr__(self, "ledger_id", ledger_id)
        object.__setattr__(
            self, "observation_id", _utility_opaque_digest(self.observation_id, "observation_id")
        )
        object.__setattr__(
            self, "family_id", _utility_opaque_digest(self.family_id, "family_id")
        )
        for field_name in ("association_id", "creation_receipt_id"):
            value = _utility_nonnegative_int(getattr(self, field_name), field_name)
            if value <= 0:
                raise ValueError(f"utility ledger receipt {field_name} must be positive")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "idempotent", bool(self.idempotent))
        object.__setattr__(self, "promotion_eligible", bool(self.promotion_eligible))


_REVISIT_SAFE_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}$"
)
_REVISIT_CONTRACT_VERSION_RE = re.compile(r"^contextual-revisit-v[1-9][0-9]*$")


def _revisit_opaque_digest(value: object, field_name: str) -> str:
    """Accept only opaque IDs/hashes in a durable revisit contract."""

    normalized = str(value or "").strip()
    if not _UTILITY_OPAQUE_DIGEST_RE.fullmatch(normalized):
        raise ValueError(
            f"contextual revisit contract {field_name} must be an opaque sha256 ID"
        )
    return normalized


def _revisit_positive_int(value: object, field_name: str) -> int:
    normalized = _utility_nonnegative_int(value, field_name)
    if normalized <= 0:
        raise ValueError(f"contextual revisit contract {field_name} must be positive")
    return normalized


def _revisit_safe_identifier(value: object, field_name: str) -> str:
    """Keep metadata identifiers bounded and incapable of carrying prose."""

    normalized = str(value or "").strip()
    if not _REVISIT_SAFE_IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(
            f"contextual revisit contract {field_name} must be a safe identifier"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ContextualRevisitSlotNeedBinding:
    """One opaque selector-slot to need-query binding.

    These fields are deliberately all opaque digests.  Durable contracts must
    not preserve a question, need text, source key, answer, or vector merely
    to make a later revisit possible.
    """

    slot_id: str
    need_query_id: str
    need_hash: str

    def __post_init__(self) -> None:
        for field_name in ("slot_id", "need_query_id", "need_hash"):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(getattr(self, field_name), field_name),
            )

    def canonical_payload(self) -> dict[str, str]:
        return {
            "need_hash": self.need_hash,
            "need_query_id": self.need_query_id,
            "slot_id": self.slot_id,
        }


@dataclass(frozen=True, slots=True)
class ContextualRevisitContract:
    """Immutable, redacted v16 input contract for a future contextual revisit.

    It is intentionally a creation-origin contract, not a query record.  The
    repository binds it to the earliest ready source-bound creation receipt,
    and it serializes only opaque IDs/fingerprints plus embedding metadata.
    Query/source/answer prose and vector bytes have no field in this type.
    """

    creation_receipt_id: int
    association_id: int
    context_cue_id: int
    need_cue_id: int
    domain: str
    model_id: str
    embedding_space_id: str
    dimension: int
    dtype: str
    context_hash: str
    slot_need_bindings: tuple[ContextualRevisitSlotNeedBinding, ...]
    requirements_fingerprint: str
    source_closure_fingerprint: str
    retrieval_policy_fingerprint: str
    budget_fingerprint: str
    anchor_manifest_fingerprint: str
    source_fact_roles_fingerprint: str
    source_fact_refs_fingerprint: str
    ready_index_epoch: int
    ready_publication_fingerprint: str
    contract_version: str = "contextual-revisit-v1"

    def __post_init__(self) -> None:
        for field_name in (
            "creation_receipt_id",
            "association_id",
            "context_cue_id",
            "need_cue_id",
            "dimension",
            "ready_index_epoch",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_positive_int(getattr(self, field_name), field_name),
            )
        for field_name in ("domain", "model_id", "embedding_space_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_safe_identifier(getattr(self, field_name), field_name),
            )
        dtype = str(self.dtype or "").strip()
        if dtype != "float32":
            raise ValueError("contextual revisit contract dtype must be float32")
        object.__setattr__(self, "dtype", dtype)
        contract_version = str(self.contract_version or "").strip()
        if not _REVISIT_CONTRACT_VERSION_RE.fullmatch(contract_version):
            raise ValueError("contextual revisit contract version is invalid")
        object.__setattr__(self, "contract_version", contract_version)
        for field_name in (
            "context_hash",
            "requirements_fingerprint",
            "source_closure_fingerprint",
            "retrieval_policy_fingerprint",
            "budget_fingerprint",
            "anchor_manifest_fingerprint",
            "source_fact_roles_fingerprint",
            "source_fact_refs_fingerprint",
            "ready_publication_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(getattr(self, field_name), field_name),
            )

        raw_bindings = tuple(self.slot_need_bindings or ())
        if not raw_bindings:
            raise ValueError(
                "contextual revisit contract needs at least one slot/need binding"
            )
        if any(
            not isinstance(binding, ContextualRevisitSlotNeedBinding)
            for binding in raw_bindings
        ):
            raise TypeError(
                "contextual revisit contract bindings must use the typed opaque record"
            )
        bindings = tuple(
            sorted(
                raw_bindings,
                key=lambda binding: (
                    binding.slot_id,
                    binding.need_query_id,
                    binding.need_hash,
                ),
            )
        )
        if len({binding.slot_id for binding in bindings}) != len(bindings):
            raise ValueError(
                "contextual revisit contract cannot bind one slot to multiple needs"
            )
        object.__setattr__(self, "slot_need_bindings", bindings)

    @property
    def slot_need_bindings_json(self) -> str:
        return json.dumps(
            [binding.canonical_payload() for binding in self.slot_need_bindings],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def canonical_payload(self) -> dict[str, object]:
        """Return the exact redacted material covered by the contract hash."""

        return {
            "anchor_manifest_fingerprint": self.anchor_manifest_fingerprint,
            "association_id": self.association_id,
            "budget_fingerprint": self.budget_fingerprint,
            "context_cue_id": self.context_cue_id,
            "context_hash": self.context_hash,
            "contract_version": self.contract_version,
            "creation_receipt_id": self.creation_receipt_id,
            "dimension": self.dimension,
            "domain": self.domain,
            "dtype": self.dtype,
            "embedding_space_id": self.embedding_space_id,
            "model_id": self.model_id,
            "need_cue_id": self.need_cue_id,
            "ready_index_epoch": self.ready_index_epoch,
            "ready_publication_fingerprint": self.ready_publication_fingerprint,
            "requirements_fingerprint": self.requirements_fingerprint,
            "retrieval_policy_fingerprint": self.retrieval_policy_fingerprint,
            "slot_need_bindings": [
                binding.canonical_payload() for binding in self.slot_need_bindings
            ],
            "source_closure_fingerprint": self.source_closure_fingerprint,
            "source_fact_refs_fingerprint": self.source_fact_refs_fingerprint,
            "source_fact_roles_fingerprint": self.source_fact_roles_fingerprint,
        }

    @property
    def contract_fingerprint(self) -> str:
        canonical = json.dumps(
            self.canonical_payload(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "revisit-contract:sha256:" + hashlib.sha256(canonical).hexdigest()

    def storage_payload(self) -> dict[str, object]:
        payload = self.canonical_payload()
        payload["slot_need_bindings_json"] = self.slot_need_bindings_json
        payload.pop("slot_need_bindings")
        payload["contract_fingerprint"] = self.contract_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class ContextualRevisitContractLookup:
    """Typed, exact, redacted key for a future candidate-contract lookup."""

    domain: str
    model_id: str
    embedding_space_id: str
    dimension: int
    dtype: str
    context_hash: str
    slot_need_bindings: tuple[ContextualRevisitSlotNeedBinding, ...]
    requirements_fingerprint: str
    source_closure_fingerprint: str
    retrieval_policy_fingerprint: str
    budget_fingerprint: str
    anchor_manifest_fingerprint: str
    contract_version: str = "contextual-revisit-v1"

    def __post_init__(self) -> None:
        for field_name in ("dimension",):
            object.__setattr__(
                self,
                field_name,
                _revisit_positive_int(getattr(self, field_name), field_name),
            )
        for field_name in ("domain", "model_id", "embedding_space_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_safe_identifier(getattr(self, field_name), field_name),
            )
        dtype = str(self.dtype or "").strip()
        if dtype != "float32":
            raise ValueError("contextual revisit contract dtype must be float32")
        object.__setattr__(self, "dtype", dtype)
        contract_version = str(self.contract_version or "").strip()
        if not _REVISIT_CONTRACT_VERSION_RE.fullmatch(contract_version):
            raise ValueError("contextual revisit contract version is invalid")
        object.__setattr__(self, "contract_version", contract_version)
        for field_name in (
            "context_hash",
            "requirements_fingerprint",
            "source_closure_fingerprint",
            "retrieval_policy_fingerprint",
            "budget_fingerprint",
            "anchor_manifest_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(getattr(self, field_name), field_name),
            )
        raw_bindings = tuple(self.slot_need_bindings or ())
        if not raw_bindings:
            raise ValueError(
                "contextual revisit contract needs at least one slot/need binding"
            )
        if any(
            not isinstance(binding, ContextualRevisitSlotNeedBinding)
            for binding in raw_bindings
        ):
            raise TypeError(
                "contextual revisit contract bindings must use the typed opaque record"
            )
        bindings = tuple(
            sorted(
                raw_bindings,
                key=lambda binding: (
                    binding.slot_id,
                    binding.need_query_id,
                    binding.need_hash,
                ),
            )
        )
        if len({binding.slot_id for binding in bindings}) != len(bindings):
            raise ValueError(
                "contextual revisit contract cannot bind one slot to multiple needs"
            )
        object.__setattr__(self, "slot_need_bindings", bindings)

    @property
    def slot_need_bindings_json(self) -> str:
        return json.dumps(
            [binding.canonical_payload() for binding in self.slot_need_bindings],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class ContextualRevisitContractReceipt:
    """Result of a v16 contract write or exact idempotent retry."""

    contract_id: int
    creation_receipt_id: int
    association_id: int
    contract_fingerprint: str
    idempotent: bool = False

    def __post_init__(self) -> None:
        for field_name in ("contract_id", "creation_receipt_id", "association_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_positive_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "contract_fingerprint",
            _revisit_opaque_digest(self.contract_fingerprint, "contract_fingerprint"),
        )
        object.__setattr__(self, "idempotent", bool(self.idempotent))


_REVISIT_RUNTIME_MANIFEST_VERSION_RE = re.compile(
    r"^contextual-runtime-manifest-v[1-9][0-9]*$"
)
_REVISIT_RESTRICTED_REWRITE_GUARD_VERSION_RE = re.compile(
    r"^contextual-restricted-rewrite-guard-v[1-9][0-9]*$"
)
_REVISIT_RESTRICTED_REWRITE_GRAMMAR_VERSION_RE = re.compile(
    r"^restricted-rewrite-grammar-v[1-9][0-9]*$"
)
_REVISIT_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _revisit_sha256_hex(value: object, field_name: str) -> str:
    """Accept a bare SHA-256 text hash without retaining its source text."""

    normalized = str(value or "").strip().casefold()
    if not _REVISIT_SHA256_HEX_RE.fullmatch(normalized):
        raise ValueError(
            f"contextual runtime manifest {field_name} must be a SHA-256 text hash"
        )
    return normalized


def _revisit_utc_instant(value: object, field_name: str) -> str:
    """Normalize one durable timestamp to a canonical UTC ISO-8601 spelling."""

    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError as error:
        raise ValueError(
            f"contextual runtime manifest {field_name} must be a UTC timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(
            f"contextual runtime manifest {field_name} must be a UTC timestamp"
        )
    return parsed.astimezone(timezone.utc).isoformat()


def _revisit_optional_nonnegative_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    normalized = _utility_nonnegative_int(value, field_name)
    return normalized


def _revisit_canonical_digest(namespace: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{namespace}:sha256:" + hashlib.sha256(encoded).hexdigest()


def _revisit_runtime_identifier(prefix: str, opaque_ref: str) -> str:
    """Derive a process-local safe ID without serialising question prose."""

    digest = str(opaque_ref).rsplit(":", 1)[-1]
    return f"{prefix}:{digest}"


@dataclass(frozen=True, slots=True)
class ContextualRevisitRuntimeSeed:
    """One redacted, single-slot Q1 seed for a restart-safe automatic revisit.

    This is deliberately a *pre-receipt* object: its caller supplies the
    creation request ID, but never a receipt, association, or cue database ID.
    The repository binds those values inside the same transaction that creates
    the cue/edge/receipt.  It has no question, source/answer prose, raw slot
    or clause label, or vector field.
    """

    creation_request_id: str
    domain: str
    context_scope_hash: str
    context_hash: str
    source_request_hash: str
    context_cue_text_hash: str
    need_cue_text_hash: str
    slot_need_bindings: tuple[ContextualRevisitSlotNeedBinding, ...]
    requirements_fingerprint: str
    source_closure_fingerprint: str
    retrieval_policy_fingerprint: str
    budget_fingerprint: str
    anchor_manifest_fingerprint: str
    source_fact_refs_fingerprint: str
    anchor_episode_id: int
    anchor_activation: float
    anchor_source_fact_id: str
    target_episode_id: int
    target_source_fact_id: str
    target_mapping_ref: str
    runtime_slot_ref: str
    runtime_query_ref: str
    runtime_clause_ref: str
    endpoint_limit: int
    episode_limit: int
    source_fact_limit: int | None
    delivery_token_limit: int | None
    support_mode: str = "alternative"
    contract_version: str = "contextual-revisit-v1"
    manifest_version: str = "contextual-runtime-manifest-v1"

    def __post_init__(self) -> None:
        creation_request_id = _revisit_safe_identifier(
            self.creation_request_id, "runtime_seed.creation_request_id"
        )
        domain = _revisit_safe_identifier(self.domain, "runtime_seed.domain")
        for field_name in (
            "context_scope_hash",
            "context_hash",
            "source_request_hash",
            "requirements_fingerprint",
            "source_closure_fingerprint",
            "retrieval_policy_fingerprint",
            "budget_fingerprint",
            "anchor_manifest_fingerprint",
            "source_fact_refs_fingerprint",
            "anchor_source_fact_id",
            "target_source_fact_id",
            "target_mapping_ref",
            "runtime_slot_ref",
            "runtime_query_ref",
            "runtime_clause_ref",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "creation_request_id", creation_request_id)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(
            self,
            "context_cue_text_hash",
            _revisit_sha256_hex(
                self.context_cue_text_hash, "context_cue_text_hash"
            ),
        )
        object.__setattr__(
            self,
            "need_cue_text_hash",
            _revisit_sha256_hex(self.need_cue_text_hash, "need_cue_text_hash"),
        )

        raw_bindings = tuple(self.slot_need_bindings or ())
        if len(raw_bindings) != 1 or not isinstance(
            raw_bindings[0], ContextualRevisitSlotNeedBinding
        ):
            raise ValueError(
                "contextual runtime manifest needs exactly one typed slot binding"
            )
        binding = raw_bindings[0]
        expected_need_hash = _revisit_canonical_digest(
            "revisit-need", {"text_hash": self.need_cue_text_hash}
        )
        expected_slot_id = _revisit_canonical_digest(
            "revisit-slot", self.runtime_slot_id
        )
        expected_need_query_id = _revisit_canonical_digest(
            "revisit-need-query", self.runtime_query_id
        )
        if (
            binding.need_hash != expected_need_hash
            or binding.slot_id != expected_slot_id
            or binding.need_query_id != expected_need_query_id
        ):
            raise ValueError(
                "contextual runtime manifest binding does not match runtime IDs/cue hash"
            )
        object.__setattr__(self, "slot_need_bindings", (binding,))

        for field_name in ("anchor_episode_id", "target_episode_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_positive_int(getattr(self, field_name), field_name),
            )
        if int(self.anchor_episode_id) == int(self.target_episode_id):
            raise ValueError("contextual runtime manifest cannot use a self target")
        try:
            activation = float(self.anchor_activation)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "contextual runtime manifest anchor_activation must be finite"
            ) from error
        if not math.isfinite(activation) or activation <= 0.0:
            raise ValueError(
                "contextual runtime manifest anchor_activation must be positive finite"
            )
        object.__setattr__(self, "anchor_activation", activation)

        endpoint_limit = _revisit_positive_int(
            self.endpoint_limit, "endpoint_limit"
        )
        episode_limit = _revisit_positive_int(self.episode_limit, "episode_limit")
        source_fact_limit = _revisit_optional_nonnegative_int(
            self.source_fact_limit, "source_fact_limit"
        )
        delivery_token_limit = _revisit_optional_nonnegative_int(
            self.delivery_token_limit, "delivery_token_limit"
        )
        object.__setattr__(self, "endpoint_limit", endpoint_limit)
        object.__setattr__(self, "episode_limit", episode_limit)
        object.__setattr__(self, "source_fact_limit", source_fact_limit)
        object.__setattr__(self, "delivery_token_limit", delivery_token_limit)
        expected_budget = _revisit_canonical_digest(
            "revisit-budget",
            {
                "episode_limit": episode_limit,
                "source_fact_limit": source_fact_limit,
                "delivery_token_limit": delivery_token_limit,
            },
        )
        if self.budget_fingerprint != expected_budget:
            raise ValueError(
                "contextual runtime manifest budget fingerprint does not match limits"
            )
        expected_anchor = _revisit_canonical_digest(
            "revisit-anchor-manifest",
            [
                {
                    "episode_id": int(self.anchor_episode_id),
                    "activation": activation,
                }
            ],
        )
        if self.anchor_manifest_fingerprint != expected_anchor:
            raise ValueError(
                "contextual runtime manifest anchor fingerprint does not match anchor"
            )
        if self.support_mode != "alternative":
            raise ValueError(
                "contextual runtime manifest supports only alternative single-slot evidence"
            )
        contract_version = str(self.contract_version or "").strip()
        if not _REVISIT_CONTRACT_VERSION_RE.fullmatch(contract_version):
            raise ValueError("contextual runtime manifest contract version is invalid")
        manifest_version = str(self.manifest_version or "").strip()
        if not _REVISIT_RUNTIME_MANIFEST_VERSION_RE.fullmatch(manifest_version):
            raise ValueError("contextual runtime manifest version is invalid")
        object.__setattr__(self, "contract_version", contract_version)
        object.__setattr__(self, "manifest_version", manifest_version)

    @property
    def runtime_slot_id(self) -> str:
        return _revisit_runtime_identifier("runtime-slot", self.runtime_slot_ref)

    @property
    def runtime_query_id(self) -> str:
        return _revisit_runtime_identifier("runtime-query", self.runtime_query_ref)

    @property
    def runtime_clause_id(self) -> str:
        return _revisit_runtime_identifier("runtime-clause", self.runtime_clause_ref)

    @property
    def slot_need_bindings_json(self) -> str:
        return json.dumps(
            [binding.canonical_payload() for binding in self.slot_need_bindings],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def canonical_payload(self) -> dict[str, object]:
        """Return only redacted, typed material covered by the seed digest."""

        return {
            "anchor_activation": self.anchor_activation,
            "anchor_episode_id": self.anchor_episode_id,
            "anchor_manifest_fingerprint": self.anchor_manifest_fingerprint,
            "anchor_source_fact_id": self.anchor_source_fact_id,
            "budget_fingerprint": self.budget_fingerprint,
            "context_cue_text_hash": self.context_cue_text_hash,
            "context_hash": self.context_hash,
            "context_scope_hash": self.context_scope_hash,
            "contract_version": self.contract_version,
            "creation_request_id": self.creation_request_id,
            "delivery_token_limit": self.delivery_token_limit,
            "domain": self.domain,
            "endpoint_limit": self.endpoint_limit,
            "episode_limit": self.episode_limit,
            "manifest_version": self.manifest_version,
            "need_cue_text_hash": self.need_cue_text_hash,
            "requirements_fingerprint": self.requirements_fingerprint,
            "retrieval_policy_fingerprint": self.retrieval_policy_fingerprint,
            "runtime_clause_ref": self.runtime_clause_ref,
            "runtime_query_ref": self.runtime_query_ref,
            "runtime_slot_ref": self.runtime_slot_ref,
            "slot_need_bindings": [
                binding.canonical_payload() for binding in self.slot_need_bindings
            ],
            "source_closure_fingerprint": self.source_closure_fingerprint,
            "source_fact_limit": self.source_fact_limit,
            "source_fact_refs_fingerprint": self.source_fact_refs_fingerprint,
            "source_request_hash": self.source_request_hash,
            "support_mode": self.support_mode,
            "target_episode_id": self.target_episode_id,
            "target_mapping_ref": self.target_mapping_ref,
            "target_source_fact_id": self.target_source_fact_id,
        }

    @property
    def seed_fingerprint(self) -> str:
        return _revisit_canonical_digest(
            "revisit-runtime-seed", self.canonical_payload()
        )

    def durable_payload(self) -> dict[str, object]:
        """Flatten the safe subset that belongs in the runtime-manifest table."""

        payload = self.canonical_payload()
        payload.pop("creation_request_id")
        payload["slot_need_bindings_json"] = self.slot_need_bindings_json
        payload.pop("slot_need_bindings")
        payload["seed_fingerprint"] = self.seed_fingerprint
        return payload

    def source_fact_roles_fingerprint(self, association_id: int) -> str:
        """Bind the automatic runtime proof to its final edge ID.

        The shape intentionally matches the existing V16 role-fingerprint
        algorithm, but uses deterministic safe runtime IDs instead of the
        original slot/clause labels.  This lets a restart reconstruct a single
        transient slot without persisting either label.
        """

        edge_id = _revisit_positive_int(association_id, "association_id")
        rows = [
            {
                "role": "contextual_source_mapping",
                "association_id": edge_id,
                "episode_id": int(self.target_episode_id),
                "slot_id": self.runtime_slot_id,
                "clause_ids": [self.runtime_clause_id],
                "expected_source_fact_id": self.target_source_fact_id,
                "current_source_fact_id": self.target_source_fact_id,
                "mapping_ref": self.target_mapping_ref,
            }
        ]
        return _revisit_canonical_digest("revisit-source-fact-roles", rows)


@dataclass(frozen=True, slots=True)
class ContextualRevisitRuntimeManifest:
    """A bound v17 seed, pending index publication or safely ready.

    The type has only identifiers, hashes, numeric limits and timestamps.  It
    intentionally does not offer a field that could hold question/source/answer
    prose, a raw slot/clause name, or any vector bytes.
    """

    manifest_id: int
    creation_receipt_id: int
    association_id: int
    context_cue_id: int
    need_cue_id: int
    model_id: str
    embedding_space_id: str
    dimension: int
    dtype: str
    seed: ContextualRevisitRuntimeSeed
    context_cue_vector_fingerprint: str
    need_cue_vector_fingerprint: str
    source_fact_roles_fingerprint: str
    not_before_at: str
    expires_at: str
    binding_fingerprint: str
    state: str = "pending_index"
    ready_index_epoch: int | None = None
    ready_at: str = ""
    ready_publication_fingerprint: str = ""
    contract_id: int | None = None
    contract_fingerprint: str = ""
    manifest_fingerprint: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "manifest_id",
            "creation_receipt_id",
            "association_id",
            "context_cue_id",
            "need_cue_id",
            "dimension",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_positive_int(getattr(self, field_name), field_name),
            )
        if not isinstance(self.seed, ContextualRevisitRuntimeSeed):
            raise TypeError("contextual runtime manifest must use a typed seed")
        for field_name in ("model_id", "embedding_space_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_safe_identifier(getattr(self, field_name), field_name),
            )
        dtype = str(self.dtype or "").strip()
        if dtype != "float32":
            raise ValueError("contextual runtime manifest dtype must be float32")
        object.__setattr__(self, "dtype", dtype)
        for field_name in (
            "context_cue_vector_fingerprint",
            "need_cue_vector_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(getattr(self, field_name), field_name),
            )
        source_fact_roles_fingerprint = _revisit_opaque_digest(
            self.source_fact_roles_fingerprint, "source_fact_roles_fingerprint"
        )
        expected_roles = self.seed.source_fact_roles_fingerprint(self.association_id)
        if source_fact_roles_fingerprint != expected_roles:
            raise ValueError(
                "contextual runtime manifest source role fingerprint does not match seed"
            )
        object.__setattr__(
            self, "source_fact_roles_fingerprint", source_fact_roles_fingerprint
        )
        not_before_at = _revisit_utc_instant(self.not_before_at, "not_before_at")
        expires_at = _revisit_utc_instant(self.expires_at, "expires_at")
        if expires_at <= not_before_at:
            raise ValueError("contextual runtime manifest must expire after not_before")
        object.__setattr__(self, "not_before_at", not_before_at)
        object.__setattr__(self, "expires_at", expires_at)
        expected_binding = self.expected_binding_fingerprint(
            creation_receipt_id=self.creation_receipt_id,
            association_id=self.association_id,
            context_cue_id=self.context_cue_id,
            need_cue_id=self.need_cue_id,
            model_id=self.model_id,
            embedding_space_id=self.embedding_space_id,
            dimension=self.dimension,
            dtype=self.dtype,
            seed=self.seed,
            context_cue_vector_fingerprint=self.context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=self.need_cue_vector_fingerprint,
            source_fact_roles_fingerprint=source_fact_roles_fingerprint,
            not_before_at=not_before_at,
            expires_at=expires_at,
        )
        if _revisit_opaque_digest(
            self.binding_fingerprint, "binding_fingerprint"
        ) != expected_binding:
            raise ValueError(
                "contextual runtime manifest binding fingerprint mismatch"
            )
        object.__setattr__(self, "binding_fingerprint", expected_binding)

        state = str(self.state or "").strip()
        if state == "pending_index":
            if any(
                value not in {None, ""}
                for value in (
                    self.ready_index_epoch,
                    self.ready_at,
                    self.ready_publication_fingerprint,
                    self.contract_id,
                    self.contract_fingerprint,
                    self.manifest_fingerprint,
                )
            ):
                raise ValueError(
                    "pending contextual runtime manifest cannot advertise readiness"
                )
            object.__setattr__(self, "ready_index_epoch", None)
            object.__setattr__(self, "ready_at", "")
            object.__setattr__(self, "ready_publication_fingerprint", "")
            object.__setattr__(self, "contract_id", None)
            object.__setattr__(self, "contract_fingerprint", "")
            object.__setattr__(self, "manifest_fingerprint", "")
        elif state == "ready":
            ready_epoch = _revisit_positive_int(
                self.ready_index_epoch, "ready_index_epoch"
            )
            ready_at = _revisit_utc_instant(self.ready_at, "ready_at")
            if ready_at < not_before_at:
                raise ValueError(
                    "contextual runtime manifest ready_at precedes not_before"
                )
            contract_id = _revisit_positive_int(self.contract_id, "contract_id")
            ready_publication = _revisit_opaque_digest(
                self.ready_publication_fingerprint,
                "ready_publication_fingerprint",
            )
            contract_fingerprint = _revisit_opaque_digest(
                self.contract_fingerprint, "contract_fingerprint"
            )
            object.__setattr__(self, "ready_index_epoch", ready_epoch)
            object.__setattr__(self, "ready_at", ready_at)
            object.__setattr__(self, "contract_id", contract_id)
            object.__setattr__(
                self, "ready_publication_fingerprint", ready_publication
            )
            object.__setattr__(self, "contract_fingerprint", contract_fingerprint)
            expected_manifest = self.expected_manifest_fingerprint(
                binding_fingerprint=self.binding_fingerprint,
                contract_id=contract_id,
                contract_fingerprint=contract_fingerprint,
                ready_at=ready_at,
                ready_index_epoch=ready_epoch,
                ready_publication_fingerprint=ready_publication,
            )
            if _revisit_opaque_digest(
                self.manifest_fingerprint, "manifest_fingerprint"
            ) != expected_manifest:
                raise ValueError(
                    "contextual runtime manifest fingerprint mismatch"
                )
            object.__setattr__(self, "manifest_fingerprint", expected_manifest)
        else:
            raise ValueError("contextual runtime manifest state is invalid")
        object.__setattr__(self, "state", state)

    def binding_payload(self) -> dict[str, object]:
        return {
            "association_id": int(self.association_id),
            "context_cue_vector_fingerprint": self.context_cue_vector_fingerprint,
            "context_cue_id": int(self.context_cue_id),
            "creation_receipt_id": int(self.creation_receipt_id),
            "dimension": int(self.dimension),
            "domain": self.seed.domain,
            "dtype": self.dtype,
            "embedding_space_id": self.embedding_space_id,
            "expires_at": self.expires_at,
            "model_id": self.model_id,
            "need_cue_vector_fingerprint": self.need_cue_vector_fingerprint,
            "need_cue_id": int(self.need_cue_id),
            "not_before_at": self.not_before_at,
            "seed_fingerprint": self.seed.seed_fingerprint,
            "source_fact_roles_fingerprint": self.source_fact_roles_fingerprint,
        }

    @staticmethod
    def expected_binding_fingerprint(
        *,
        creation_receipt_id: int,
        association_id: int,
        context_cue_id: int,
        need_cue_id: int,
        model_id: str,
        embedding_space_id: str,
        dimension: int,
        dtype: str,
        seed: ContextualRevisitRuntimeSeed,
        context_cue_vector_fingerprint: str,
        need_cue_vector_fingerprint: str,
        source_fact_roles_fingerprint: str,
        not_before_at: str,
        expires_at: str,
    ) -> str:
        """Calculate a bound seed digest before SQLite assigns manifest ID."""

        payload = {
            "association_id": int(association_id),
            "context_cue_vector_fingerprint": str(context_cue_vector_fingerprint),
            "context_cue_id": int(context_cue_id),
            "creation_receipt_id": int(creation_receipt_id),
            "dimension": int(dimension),
            "domain": str(seed.domain),
            "dtype": str(dtype),
            "embedding_space_id": str(embedding_space_id),
            "expires_at": str(expires_at),
            "model_id": str(model_id),
            "need_cue_vector_fingerprint": str(need_cue_vector_fingerprint),
            "need_cue_id": int(need_cue_id),
            "not_before_at": str(not_before_at),
            "seed_fingerprint": seed.seed_fingerprint,
            "source_fact_roles_fingerprint": str(source_fact_roles_fingerprint),
        }
        return _revisit_canonical_digest("revisit-runtime-binding", payload)

    def ready_payload(self) -> dict[str, object]:
        return {
            "binding_fingerprint": self.binding_fingerprint,
            "contract_fingerprint": self.contract_fingerprint,
            "contract_id": self.contract_id,
            "ready_at": self.ready_at,
            "ready_index_epoch": self.ready_index_epoch,
            "ready_publication_fingerprint": self.ready_publication_fingerprint,
            "state": "ready",
        }

    @staticmethod
    def expected_manifest_fingerprint(
        *,
        binding_fingerprint: str,
        contract_id: int,
        contract_fingerprint: str,
        ready_at: str,
        ready_index_epoch: int,
        ready_publication_fingerprint: str,
    ) -> str:
        return _revisit_canonical_digest(
            "revisit-runtime-manifest",
            {
                "binding_fingerprint": str(binding_fingerprint),
                "contract_fingerprint": str(contract_fingerprint),
                "contract_id": int(contract_id),
                "ready_at": str(ready_at),
                "ready_index_epoch": int(ready_index_epoch),
                "ready_publication_fingerprint": str(
                    ready_publication_fingerprint
                ),
                "state": "ready",
            },
        )

    def to_revisit_contract(self) -> ContextualRevisitContract:
        """Build the only V16 contract a ready V17 manifest can advertise."""

        if self.state != "ready":
            raise ValueError("pending contextual runtime manifest has no revisit contract")
        return ContextualRevisitContract(
            creation_receipt_id=self.creation_receipt_id,
            association_id=self.association_id,
            context_cue_id=self.context_cue_id,
            need_cue_id=self.need_cue_id,
            domain=self.seed.domain,
            model_id=self.model_id,
            embedding_space_id=self.embedding_space_id,
            dimension=self.dimension,
            dtype=self.dtype,
            context_hash=self.seed.context_hash,
            slot_need_bindings=self.seed.slot_need_bindings,
            requirements_fingerprint=self.seed.requirements_fingerprint,
            source_closure_fingerprint=self.seed.source_closure_fingerprint,
            retrieval_policy_fingerprint=self.seed.retrieval_policy_fingerprint,
            budget_fingerprint=self.seed.budget_fingerprint,
            anchor_manifest_fingerprint=self.seed.anchor_manifest_fingerprint,
            source_fact_roles_fingerprint=self.source_fact_roles_fingerprint,
            source_fact_refs_fingerprint=self.seed.source_fact_refs_fingerprint,
            ready_index_epoch=int(self.ready_index_epoch or 0),
            ready_publication_fingerprint=self.ready_publication_fingerprint,
            contract_version=self.seed.contract_version,
        )

    def storage_payload(self) -> dict[str, object]:
        payload = self.seed.durable_payload()
        payload.update(
            {
                "association_id": self.association_id,
                "binding_fingerprint": self.binding_fingerprint,
                "context_cue_id": self.context_cue_id,
                "context_cue_vector_fingerprint": self.context_cue_vector_fingerprint,
                "contract_fingerprint": self.contract_fingerprint or None,
                "contract_id": self.contract_id,
                "creation_receipt_id": self.creation_receipt_id,
                "dimension": self.dimension,
                "dtype": self.dtype,
                "embedding_space_id": self.embedding_space_id,
                "expires_at": self.expires_at,
                "manifest_fingerprint": self.manifest_fingerprint or None,
                "model_id": self.model_id,
                "need_cue_id": self.need_cue_id,
                "need_cue_vector_fingerprint": self.need_cue_vector_fingerprint,
                "not_before_at": self.not_before_at,
                "ready_at": self.ready_at or None,
                "ready_index_epoch": self.ready_index_epoch,
                "ready_publication_fingerprint": (
                    self.ready_publication_fingerprint or None
                ),
                "source_fact_roles_fingerprint": self.source_fact_roles_fingerprint,
                "state": self.state,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class ContextualRevisitRuntimeManifestLookup:
    """Exact, scope-bound key for a public automatic Q2 manifest lookup."""

    domain: str
    context_scope_hash: str
    context_hash: str
    source_request_hash: str
    model_id: str
    embedding_space_id: str
    dimension: int
    dtype: str = "float32"
    manifest_version: str = "contextual-runtime-manifest-v1"

    def __post_init__(self) -> None:
        for field_name in ("domain", "model_id", "embedding_space_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_safe_identifier(getattr(self, field_name), field_name),
            )
        for field_name in (
            "context_scope_hash",
            "context_hash",
            "source_request_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self, "dimension", _revisit_positive_int(self.dimension, "dimension")
        )
        if str(self.dtype or "").strip() != "float32":
            raise ValueError("contextual runtime manifest lookup dtype must be float32")
        object.__setattr__(self, "dtype", "float32")
        version = str(self.manifest_version or "").strip()
        if not _REVISIT_RUNTIME_MANIFEST_VERSION_RE.fullmatch(version):
            raise ValueError("contextual runtime manifest lookup version is invalid")
        object.__setattr__(self, "manifest_version", version)


@dataclass(frozen=True, slots=True)
class ContextualRevisitRuntimeManifestReceipt:
    """Outcome of a v17 seed write or ready promotion, without query prose."""

    manifest_id: int
    creation_receipt_id: int
    association_id: int
    state: str
    seed_fingerprint: str
    manifest_fingerprint: str = ""
    contract_id: int | None = None
    contract_fingerprint: str = ""
    idempotent: bool = False
    promoted: bool = False

    def __post_init__(self) -> None:
        for field_name in ("manifest_id", "creation_receipt_id", "association_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_positive_int(getattr(self, field_name), field_name),
            )
        state = str(self.state or "").strip()
        if state not in {"pending_index", "ready"}:
            raise ValueError("contextual runtime manifest receipt state is invalid")
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "seed_fingerprint",
            _revisit_opaque_digest(self.seed_fingerprint, "seed_fingerprint"),
        )
        if state == "ready":
            object.__setattr__(
                self,
                "manifest_fingerprint",
                _revisit_opaque_digest(
                    self.manifest_fingerprint, "manifest_fingerprint"
                ),
            )
            object.__setattr__(
                self,
                "contract_id",
                _revisit_positive_int(self.contract_id, "contract_id"),
            )
            object.__setattr__(
                self,
                "contract_fingerprint",
                _revisit_opaque_digest(
                    self.contract_fingerprint, "contract_fingerprint"
                ),
            )
        elif any(
            value not in {None, ""}
            for value in (
                self.manifest_fingerprint,
                self.contract_id,
                self.contract_fingerprint,
            )
        ):
            raise ValueError(
                "pending contextual runtime manifest receipt cannot advertise readiness"
            )
        object.__setattr__(self, "idempotent", bool(self.idempotent))
        object.__setattr__(self, "promoted", bool(self.promoted))


@dataclass(frozen=True, slots=True)
class ContextualRestrictedRewriteGuardDraft:
    """Opaque Q1 authorization for one deliberately tiny rewrite grammar.

    ``rewrite_commitment`` is an HMAC-root made from a fully consumed,
    request-local grammar IR.  ``binding_commitment`` binds that root to the
    request/seed and cue provenance.  At Q1 materialization, the private
    signer derives the V5 ``manifest_binding_commitment`` over the canonical
    pending manifest lifecycle binding.  The later V5 ready signer completes
    the same guard with a commitment over the ready manifest/contract state.
    Together they make a copied root or altered lifecycle/publication binding
    fail closed under database-only tampering.  This object must never retain
    the question,
    entities, answer, source text, planner output, or vector bytes.  The two
    vector fields are non-reversible origin digests only.  The repository can
    bind it only while creating the matching V17 pending manifest.
    """

    creation_request_id: str
    domain: str
    context_scope_hash: str
    rewrite_commitment: str
    binding_commitment: str
    context_cue_vector_fingerprint: str
    need_cue_vector_fingerprint: str
    model_id: str
    embedding_space_id: str
    dimension: int
    dtype: str
    commitment_key_id: str
    grammar_version: str = "restricted-rewrite-grammar-v1"
    signature_version: str = "contextual-restricted-rewrite-guard-v5"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "creation_request_id",
            _revisit_safe_identifier(
                self.creation_request_id,
                "restricted_rewrite_guard.creation_request_id",
            ),
        )
        for field_name in (
            "domain",
            "model_id",
            "embedding_space_id",
            "commitment_key_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_safe_identifier(
                    getattr(self, field_name),
                    f"restricted_rewrite_guard.{field_name}",
                ),
            )
        for field_name in (
            "context_scope_hash",
            "rewrite_commitment",
            "binding_commitment",
            "context_cue_vector_fingerprint",
            "need_cue_vector_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(
                    getattr(self, field_name),
                    f"restricted_rewrite_guard.{field_name}",
                ),
            )
        grammar_version = str(self.grammar_version or "").strip()
        if not _REVISIT_RESTRICTED_REWRITE_GRAMMAR_VERSION_RE.fullmatch(
            grammar_version
        ):
            raise ValueError("restricted rewrite grammar version is invalid")
        object.__setattr__(self, "grammar_version", grammar_version)
        object.__setattr__(
            self,
            "dimension",
            _revisit_positive_int(
                self.dimension, "restricted_rewrite_guard.dimension"
            ),
        )
        if str(self.dtype or "").strip() != "float32":
            raise ValueError("restricted rewrite guard dtype must be float32")
        object.__setattr__(self, "dtype", "float32")
        signature_version = str(self.signature_version or "").strip()
        if not _REVISIT_RESTRICTED_REWRITE_GUARD_VERSION_RE.fullmatch(
            signature_version
        ) or signature_version != "contextual-restricted-rewrite-guard-v5":
            raise ValueError("restricted rewrite guard version is invalid")
        object.__setattr__(self, "signature_version", signature_version)

    def bind(
        self,
        *,
        creation_receipt_id: int,
        association_id: int,
        seed_fingerprint: str,
        created_at: str,
        manifest_binding_commitment: str,
    ) -> "ContextualRestrictedRewriteGuard":
        """Attach only repository-owned receipt, edge, and seed identities."""

        return ContextualRestrictedRewriteGuard(
            creation_receipt_id=creation_receipt_id,
            association_id=association_id,
            domain=self.domain,
            context_scope_hash=self.context_scope_hash,
            rewrite_commitment=self.rewrite_commitment,
            binding_commitment=self.binding_commitment,
            manifest_binding_commitment=manifest_binding_commitment,
            ready_manifest_commitment="",
            context_cue_vector_fingerprint=self.context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=self.need_cue_vector_fingerprint,
            model_id=self.model_id,
            embedding_space_id=self.embedding_space_id,
            dimension=self.dimension,
            dtype=self.dtype,
            commitment_key_id=self.commitment_key_id,
            grammar_version=self.grammar_version,
            seed_fingerprint=seed_fingerprint,
            signature_version=self.signature_version,
            created_at=created_at,
            guard_fingerprint=ContextualRestrictedRewriteGuard.expected_fingerprint(
                creation_receipt_id=creation_receipt_id,
                association_id=association_id,
                domain=self.domain,
                context_scope_hash=self.context_scope_hash,
                rewrite_commitment=self.rewrite_commitment,
                binding_commitment=self.binding_commitment,
                manifest_binding_commitment=manifest_binding_commitment,
                ready_manifest_commitment="",
                context_cue_vector_fingerprint=self.context_cue_vector_fingerprint,
                need_cue_vector_fingerprint=self.need_cue_vector_fingerprint,
                model_id=self.model_id,
                embedding_space_id=self.embedding_space_id,
                dimension=self.dimension,
                dtype=self.dtype,
                commitment_key_id=self.commitment_key_id,
                grammar_version=self.grammar_version,
                seed_fingerprint=seed_fingerprint,
                signature_version=self.signature_version,
                created_at=created_at,
            ),
        )


@dataclass(frozen=True, slots=True)
class ContextualRestrictedRewriteGuard:
    """Immutable receipt-bound HMAC commitment for a T16 rewrite probe."""

    creation_receipt_id: int
    association_id: int
    domain: str
    context_scope_hash: str
    rewrite_commitment: str
    binding_commitment: str
    manifest_binding_commitment: str
    ready_manifest_commitment: str
    context_cue_vector_fingerprint: str
    need_cue_vector_fingerprint: str
    model_id: str
    embedding_space_id: str
    dimension: int
    dtype: str
    commitment_key_id: str
    grammar_version: str
    seed_fingerprint: str
    signature_version: str
    created_at: str
    guard_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in ("creation_receipt_id", "association_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_positive_int(
                    getattr(self, field_name),
                    f"restricted_rewrite_guard.{field_name}",
                ),
            )
        for field_name in (
            "domain",
            "model_id",
            "embedding_space_id",
            "commitment_key_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_safe_identifier(
                    getattr(self, field_name),
                    f"restricted_rewrite_guard.{field_name}",
                ),
            )
        for field_name in (
            "context_scope_hash",
            "rewrite_commitment",
            "binding_commitment",
            "manifest_binding_commitment",
            "context_cue_vector_fingerprint",
            "need_cue_vector_fingerprint",
            "seed_fingerprint",
        ):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(
                    getattr(self, field_name),
                    f"restricted_rewrite_guard.{field_name}",
                ),
            )
        ready_manifest_commitment = str(self.ready_manifest_commitment or "").strip()
        if ready_manifest_commitment:
            ready_manifest_commitment = _revisit_opaque_digest(
                ready_manifest_commitment,
                "restricted_rewrite_guard.ready_manifest_commitment",
            )
        object.__setattr__(
            self, "ready_manifest_commitment", ready_manifest_commitment
        )
        grammar_version = str(self.grammar_version or "").strip()
        if not _REVISIT_RESTRICTED_REWRITE_GRAMMAR_VERSION_RE.fullmatch(
            grammar_version
        ):
            raise ValueError("restricted rewrite grammar version is invalid")
        object.__setattr__(self, "grammar_version", grammar_version)
        object.__setattr__(
            self,
            "dimension",
            _revisit_positive_int(
                self.dimension, "restricted_rewrite_guard.dimension"
            ),
        )
        if str(self.dtype or "").strip() != "float32":
            raise ValueError("restricted rewrite guard dtype must be float32")
        object.__setattr__(self, "dtype", "float32")
        signature_version = str(self.signature_version or "").strip()
        if not _REVISIT_RESTRICTED_REWRITE_GUARD_VERSION_RE.fullmatch(
            signature_version
        ) or signature_version != "contextual-restricted-rewrite-guard-v5":
            raise ValueError("restricted rewrite guard version is invalid")
        object.__setattr__(self, "signature_version", signature_version)
        object.__setattr__(
            self,
            "created_at",
            _revisit_utc_instant(
                self.created_at, "restricted_rewrite_guard.created_at"
            ),
        )
        expected = self.expected_fingerprint(
            creation_receipt_id=self.creation_receipt_id,
            association_id=self.association_id,
            domain=self.domain,
            context_scope_hash=self.context_scope_hash,
            rewrite_commitment=self.rewrite_commitment,
            binding_commitment=self.binding_commitment,
            manifest_binding_commitment=self.manifest_binding_commitment,
            ready_manifest_commitment=self.ready_manifest_commitment,
            context_cue_vector_fingerprint=self.context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=self.need_cue_vector_fingerprint,
            model_id=self.model_id,
            embedding_space_id=self.embedding_space_id,
            dimension=self.dimension,
            dtype=self.dtype,
            commitment_key_id=self.commitment_key_id,
            grammar_version=self.grammar_version,
            seed_fingerprint=self.seed_fingerprint,
            signature_version=self.signature_version,
            created_at=self.created_at,
        )
        if (
            _revisit_opaque_digest(
                self.guard_fingerprint,
                "restricted_rewrite_guard.guard_fingerprint",
            )
            != expected
        ):
            raise ValueError("restricted rewrite guard fingerprint mismatch")
        object.__setattr__(self, "guard_fingerprint", expected)

    def with_ready_manifest_commitment(
        self,
        ready_manifest_commitment: str,
    ) -> "ContextualRestrictedRewriteGuard":
        """Complete the sole permitted pending-to-ready guard transition.

        The caller obtains this opaque HMAC only from the trusted publication
        signer after it has constructed one canonical ready manifest.  A
        completed guard can never be re-signed or repointed.
        """

        if self.ready_manifest_commitment:
            raise ValueError("restricted rewrite guard is already ready-bound")
        commitment = _revisit_opaque_digest(
            ready_manifest_commitment,
            "restricted_rewrite_guard.ready_manifest_commitment",
        )
        return ContextualRestrictedRewriteGuard(
            creation_receipt_id=self.creation_receipt_id,
            association_id=self.association_id,
            domain=self.domain,
            context_scope_hash=self.context_scope_hash,
            rewrite_commitment=self.rewrite_commitment,
            binding_commitment=self.binding_commitment,
            manifest_binding_commitment=self.manifest_binding_commitment,
            ready_manifest_commitment=commitment,
            context_cue_vector_fingerprint=self.context_cue_vector_fingerprint,
            need_cue_vector_fingerprint=self.need_cue_vector_fingerprint,
            model_id=self.model_id,
            embedding_space_id=self.embedding_space_id,
            dimension=self.dimension,
            dtype=self.dtype,
            commitment_key_id=self.commitment_key_id,
            grammar_version=self.grammar_version,
            seed_fingerprint=self.seed_fingerprint,
            signature_version=self.signature_version,
            created_at=self.created_at,
            guard_fingerprint=self.expected_fingerprint(
                creation_receipt_id=self.creation_receipt_id,
                association_id=self.association_id,
                domain=self.domain,
                context_scope_hash=self.context_scope_hash,
                rewrite_commitment=self.rewrite_commitment,
                binding_commitment=self.binding_commitment,
                manifest_binding_commitment=self.manifest_binding_commitment,
                ready_manifest_commitment=commitment,
                context_cue_vector_fingerprint=(
                    self.context_cue_vector_fingerprint
                ),
                need_cue_vector_fingerprint=self.need_cue_vector_fingerprint,
                model_id=self.model_id,
                embedding_space_id=self.embedding_space_id,
                dimension=self.dimension,
                dtype=self.dtype,
                commitment_key_id=self.commitment_key_id,
                grammar_version=self.grammar_version,
                seed_fingerprint=self.seed_fingerprint,
                signature_version=self.signature_version,
                created_at=self.created_at,
            ),
        )

    @staticmethod
    def expected_fingerprint(
        *,
        creation_receipt_id: int,
        association_id: int,
        domain: str,
        context_scope_hash: str,
        rewrite_commitment: str,
        binding_commitment: str,
        manifest_binding_commitment: str,
        ready_manifest_commitment: str,
        context_cue_vector_fingerprint: str,
        need_cue_vector_fingerprint: str,
        model_id: str,
        embedding_space_id: str,
        dimension: int,
        dtype: str,
        commitment_key_id: str,
        grammar_version: str,
        seed_fingerprint: str,
        signature_version: str,
        created_at: str,
    ) -> str:
        return _revisit_canonical_digest(
            "contextual-restricted-rewrite-guard",
            {
                "association_id": int(association_id),
                "binding_commitment": str(binding_commitment),
                "manifest_binding_commitment": str(manifest_binding_commitment),
                "ready_manifest_commitment": str(ready_manifest_commitment),
                "commitment_key_id": str(commitment_key_id),
                "context_cue_vector_fingerprint": str(
                    context_cue_vector_fingerprint
                ),
                "context_scope_hash": str(context_scope_hash),
                "creation_receipt_id": int(creation_receipt_id),
                "created_at": str(created_at),
                "dimension": int(dimension),
                "domain": str(domain),
                "dtype": str(dtype),
                "embedding_space_id": str(embedding_space_id),
                "grammar_version": str(grammar_version),
                "model_id": str(model_id),
                "need_cue_vector_fingerprint": str(need_cue_vector_fingerprint),
                "rewrite_commitment": str(rewrite_commitment),
                "seed_fingerprint": str(seed_fingerprint),
                "signature_version": str(signature_version),
            },
        )

    def storage_payload(self) -> dict[str, object]:
        return {
            "association_id": self.association_id,
            "binding_commitment": self.binding_commitment,
            "manifest_binding_commitment": self.manifest_binding_commitment,
            "ready_manifest_commitment": self.ready_manifest_commitment,
            "commitment_key_id": self.commitment_key_id,
            "context_cue_vector_fingerprint": self.context_cue_vector_fingerprint,
            "context_scope_hash": self.context_scope_hash,
            "creation_receipt_id": self.creation_receipt_id,
            "created_at": self.created_at,
            "dimension": self.dimension,
            "domain": self.domain,
            "dtype": self.dtype,
            "embedding_space_id": self.embedding_space_id,
            "grammar_version": self.grammar_version,
            "guard_fingerprint": self.guard_fingerprint,
            "model_id": self.model_id,
            "need_cue_vector_fingerprint": self.need_cue_vector_fingerprint,
            "rewrite_commitment": self.rewrite_commitment,
            "seed_fingerprint": self.seed_fingerprint,
            "signature_version": self.signature_version,
        }


@dataclass(frozen=True, slots=True)
class ContextualRestrictedRewriteGuardLookup:
    """Exact, scope-bound lookup for a fully parsed restricted rewrite."""

    association_id: int | None
    domain: str
    context_scope_hash: str
    rewrite_commitment: str
    commitment_key_id: str
    grammar_version: str = "restricted-rewrite-grammar-v1"
    signature_version: str = "contextual-restricted-rewrite-guard-v5"

    def __post_init__(self) -> None:
        if self.association_id is None:
            object.__setattr__(self, "association_id", None)
        else:
            object.__setattr__(
                self,
                "association_id",
                _revisit_positive_int(
                    self.association_id,
                    "restricted_rewrite_guard_lookup.association_id",
                ),
            )
        for field_name in ("domain", "commitment_key_id"):
            object.__setattr__(
                self,
                field_name,
                _revisit_safe_identifier(
                    getattr(self, field_name),
                    f"restricted_rewrite_guard_lookup.{field_name}",
                ),
            )
        for field_name in ("context_scope_hash", "rewrite_commitment"):
            object.__setattr__(
                self,
                field_name,
                _revisit_opaque_digest(
                    getattr(self, field_name),
                    f"restricted_rewrite_guard_lookup.{field_name}",
                ),
            )
        grammar_version = str(self.grammar_version or "").strip()
        if not _REVISIT_RESTRICTED_REWRITE_GRAMMAR_VERSION_RE.fullmatch(
            grammar_version
        ):
            raise ValueError("restricted rewrite lookup grammar version is invalid")
        object.__setattr__(self, "grammar_version", grammar_version)
        signature_version = str(self.signature_version or "").strip()
        if not _REVISIT_RESTRICTED_REWRITE_GUARD_VERSION_RE.fullmatch(
            signature_version
        ) or signature_version != "contextual-restricted-rewrite-guard-v5":
            raise ValueError("restricted rewrite lookup version is invalid")
        object.__setattr__(self, "signature_version", signature_version)


@dataclass(frozen=True, slots=True)
class PlasticityEvent:
    """Local, post-answer event used to update contextual edge utility."""

    request_hash: str
    domain: str
    candidates: tuple[ContextualRecallCandidate, ...] = ()
    observations: tuple[ContextualUtilityObservation, ...] = ()
    ledger_observations: tuple[ContextualUtilityLedgerObservation, ...] = ()
    answer_guard_passed: bool = True

_CONCEPT_ALIAS_CHAIN_SEPARATOR = re.compile(r"\s+[/／]\s+")
_CONCEPT_META_COMMENTARY_MARKERS = (
    "此处修正",
    "应提取",
    "未提及",
    "候选条目",
    "更核心",
)


def _preferred_concept_name(parts: list[str]) -> str:
    """Choose one display name while retaining the other parts as aliases."""

    def score(value: str) -> tuple[int, int, int]:
        has_han = any("\u3400" <= char <= "\u9fff" for char in value)
        has_kana = any("\u3040" <= char <= "\u30ff" for char in value)
        has_hangul = any("\uac00" <= char <= "\ud7af" for char in value)
        has_latin = any("a" <= char.casefold() <= "z" for char in value)
        # A Han-only label is normally the Chinese display name. Japanese,
        # Korean and Latin spellings remain searchable aliases.
        return (
            3 if has_han and not has_kana and not has_hangul else
            2 if has_han else
            1 if has_latin else
            0,
            -int(has_kana or has_hangul),
            -len(value),
        )

    return max(parts, key=score)


def normalize_evidence_origin(value: Any, default: str = "source") -> str:
    normalized = str(value or "").strip().casefold()
    return normalized if normalized in EVIDENCE_ORIGINS else default


def normalize_epistemic_status(value: Any, default: str = "asserted") -> str:
    normalized = str(value or "").strip().casefold()
    return normalized if normalized in EPISTEMIC_STATUSES else default


@dataclass(slots=True)
class NormalizedBlock:
    record_index: int
    speaker_raw: str = ""
    languages: dict[str, str] = field(default_factory=dict)
    boundary_score: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        return any(value.strip() for value in self.languages.values())


@dataclass(slots=True)
class SourceSegment:
    source_key: str
    segment_index: int
    raw_text: str
    first_record: int
    last_record: int


@dataclass(slots=True)
class ParagraphDraft:
    """A deterministic, record-aligned sub-chunk of one Source."""

    paragraph_index: int
    text: str


@dataclass(slots=True)
class EpisodeDraft:
    text: str
    participants: list[str] = field(default_factory=list)
    event_type: str = ""
    location_text: str = ""
    story_time_text: str = ""
    timeline_scope: str = ""
    confidence: float = 0.5
    evidence_origin: EvidenceOrigin = "source"
    epistemic_status: EpistemicStatus = "asserted"
    generation: int = 0
    epistemic_note: str = ""
    # Exact import-time proof. Source remains authoritative; these fields are
    # also persisted so acceptance audits can reproduce the extraction claim.
    evidence_quotes: list[str] = field(default_factory=list)
    evidence_spans: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeDraft":
        text = str(value.get("text", "")).strip()
        if not text:
            raise ValueError("episode text is required")
        participants = value.get("participants", [])
        if isinstance(participants, str):
            try:
                decoded = json.loads(participants)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                participants = decoded
        if not isinstance(participants, list):
            raise ValueError("participants must be a list")
        evidence_quotes = value.get("evidence_quotes", [])
        if not isinstance(evidence_quotes, list) or not all(
            isinstance(item, str) for item in evidence_quotes
        ):
            raise ValueError("evidence_quotes must be a list of strings")
        evidence_spans = value.get("evidence_spans", [])
        if not isinstance(evidence_spans, list):
            raise ValueError("evidence_spans must be a list of [start, end]")
        normalized_spans: list[Any] = []
        singleton_spans: list[bool] = []
        for span in evidence_spans:
            if isinstance(span, (list, tuple)):
                normalized_span: list[Any] = []
                for item in span:
                    matched = (
                        re.fullmatch(r"[Ll]0*(\d+)", item.strip())
                        if isinstance(item, str)
                        else None
                    )
                    normalized_span.append(
                        int(matched.group(1)) if matched else item
                    )
                was_singleton = len(normalized_span) == 1
                if was_singleton:
                    # Some JSON models serialize a one-line range as [line]
                    # instead of [line, line].  It is losslessly equivalent.
                    normalized_span.append(normalized_span[0])
                normalized_spans.append(normalized_span)
                singleton_spans.append(was_singleton)
            else:
                normalized_spans.append(span)
                singleton_spans.append(False)
        evidence_spans = normalized_spans
        # A single range is occasionally emitted as [start, end] rather than
        # [[start, end]].  The meaning is unambiguous, so normalize this shape
        # without asking the model to regenerate an otherwise valid batch.
        if (
            len(evidence_spans) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in evidence_spans
            )
        ):
            evidence_spans = [evidence_spans]
        parsed_spans: list[tuple[int, int]] = []
        for span in evidence_spans:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in span
                )
            ):
                raise ValueError(
                    "evidence_spans must be a list of integer [start, end] pairs"
                )
            parsed_spans.append((int(span[0]), int(span[1])))
        # Models also commonly emit every cited line as its own singleton.
        # Compact an all-singleton representation so a contiguous quotation is
        # judged by its actual ranges, not rejected for JSON verbosity.  Keep
        # explicit [n, n] ranges unchanged: those may intentionally describe
        # separate snippets and remain subject to the range-count guard.
        if parsed_spans and len(singleton_spans) == len(parsed_spans) and all(
            singleton_spans
        ):
            compacted: list[tuple[int, int]] = []
            for start, end in sorted(set(parsed_spans)):
                if compacted and start <= compacted[-1][1] + 1:
                    compacted[-1] = (
                        compacted[-1][0],
                        max(compacted[-1][1], end),
                    )
                else:
                    compacted.append((start, end))
            parsed_spans = compacted
        confidence = float(value.get("confidence", 0.5))
        evidence_origin = normalize_evidence_origin(value.get("evidence_origin"))
        epistemic_status = normalize_epistemic_status(
            value.get("epistemic_status")
        )
        generation = max(0, int(value.get("generation", 0) or 0))
        # An importer/system-authored speculation is already one inferential
        # step away from the document's direct evidence.  A character's guess
        # in the source remains generation 0 because "the character guessed X"
        # itself is directly recorded; epistemic_status still prevents X from
        # being promoted to fact.
        if (
            evidence_origin in {"importer", "system", "mixed"}
            and epistemic_status in {"speculative", "mixed"}
        ):
            generation = max(1, generation)
        return cls(
            text=text,
            participants=[str(item).strip() for item in participants if str(item).strip()],
            event_type=str(value.get("event_type", "")).strip(),
            location_text=str(value.get("location_text", "")).strip(),
            story_time_text=str(value.get("story_time_text", "")).strip(),
            timeline_scope=str(value.get("timeline_scope", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            evidence_origin=evidence_origin,  # type: ignore[arg-type]
            epistemic_status=epistemic_status,  # type: ignore[arg-type]
            generation=generation,
            epistemic_note=str(value.get("epistemic_note", "")).strip(),
            evidence_quotes=[
                item.strip() for item in evidence_quotes if item.strip()
            ],
            evidence_spans=parsed_spans,
        )


@dataclass(slots=True)
class ConceptDraft:
    canonical_name: str
    description: str
    embedding_text: str
    aliases: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.5

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConceptDraft":
        raw_name = str(value.get("canonical_name", "")).strip()
        if not raw_name:
            raise ValueError("canonical_name is required")
        name_parts = [
            part.strip()
            for part in _CONCEPT_ALIAS_CHAIN_SEPARATOR.split(raw_name)
            if part.strip()
        ]
        name = _preferred_concept_name(name_parts)
        description = str(value.get("description", "")).strip() or name
        embedding_text = str(value.get("embedding_text", "")).strip() or description
        combined_text = f"{description}\n{embedding_text}".casefold()
        marker = next(
            (
                item
                for item in _CONCEPT_META_COMMENTARY_MARKERS
                if item.casefold() in combined_text
            ),
            None,
        )
        if marker is not None:
            raise ValueError(
                f"concept contains model meta-commentary marker: {marker}"
            )
        aliases: list[tuple[str, str]] = [
            (part, "unknown") for part in name_parts if part != name
        ]
        for item in value.get("aliases", []):
            if isinstance(item, str):
                aliases.append((item.strip(), "unknown"))
            elif isinstance(item, dict):
                alias = str(item.get("alias", "")).strip()
                if alias:
                    aliases.append((alias, str(item.get("language", "unknown"))))
            elif isinstance(item, (list, tuple)) and item:
                alias = str(item[0]).strip()
                if alias:
                    language = str(item[1]) if len(item) > 1 else "unknown"
                    aliases.append((alias, language))
        deduplicated_aliases: list[tuple[str, str]] = []
        seen_aliases = {name.casefold()}
        for alias, language in aliases:
            normalized = alias.casefold()
            if not alias or normalized in seen_aliases:
                continue
            seen_aliases.add(normalized)
            deduplicated_aliases.append((alias, language))
        confidence = float(value.get("confidence", 0.5))
        return cls(
            canonical_name=name,
            description=description,
            embedding_text=embedding_text,
            aliases=deduplicated_aliases,
            confidence=max(0.0, min(1.0, confidence)),
        )


@dataclass(slots=True)
class AssociationDraft:
    from_type: NodeType
    from_id: int
    to_type: NodeType
    to_id: int
    relation_type: str
    relation_key: str
    relation_text: str
    polarity: int = 1
    weight: float = 0.5
    confidence: float = 0.5
    generation: int = 0
    claim_level: Literal[
        "direct_fact", "supported_inference", "historical_context", "retrieval_only"
    ] = "direct_fact"
    audit_status: Literal["not_required", "dual_accepted"] = "not_required"
    evidence_json: str = "[]"
    audit_json: str = "[]"
    created_reason: str = ""


@dataclass(slots=True)
class SearchHit:
    node_type: NodeType
    node_id: int
    score: float
    text: str = ""


@dataclass(slots=True)
class QueryIntent:
    language: str = "zh"
    target_entities: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    requested_relation: str = ""
    temporal_constraint: str = ""
    causal_constraint: str = ""
    answer_shape: str = "complete_evidence_answer"
    uncertainty_required: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryIntent":
        def optional_constraint(key: str) -> str:
            raw = value.get(key, "")
            if raw is None:
                return ""
            text = str(raw).strip()
            normalized = text.casefold().replace(" ", "")
            if normalized in {
                "",
                "none",
                "null",
                "n/a",
                "na",
                "无",
                "没有",
                "不适用",
                "无明确",
                "无约束",
                "无明确约束",
                "无明确因果约束",
                "无明确时间约束",
                "无明确关系约束",
            }:
                return ""
            return text

        entities = value.get("target_entities", [])
        if not isinstance(entities, list):
            entities = []
        raw_queries = value.get("search_queries", [])
        if not isinstance(raw_queries, list):
            raw_queries = []
        search_queries: list[str] = []
        for item in raw_queries:
            query = str(item).strip()
            if query and query not in search_queries:
                search_queries.append(query)
        return cls(
            language=str(value.get("language", "zh")),
            target_entities=[str(item) for item in entities],
            search_queries=search_queries[:12],
            requested_relation=optional_constraint("requested_relation"),
            temporal_constraint=optional_constraint("temporal_constraint"),
            causal_constraint=optional_constraint("causal_constraint"),
            answer_shape=str(value.get("answer_shape", "complete_evidence_answer")),
            uncertainty_required=bool(value.get("uncertainty_required", True)),
        )
