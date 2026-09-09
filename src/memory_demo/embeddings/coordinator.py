from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from memory_demo.embeddings.codec import normalize_embedding
from memory_demo.types import (
    EmbeddingSpace,
    PhysicalQueryVector,
    QueryVector,
    QueryVectorBundle,
    QueryVectorRequest,
)


_SPACE_RE = re.compile(r"\s+")
_QUERY_ROLES = frozenset({"whole", "atomic", "followup"})


def normalize_query_text(text: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(text or ""))).strip()


class EmbeddingCoordinator:
    """The sole request-level boundary for query embedding work.

    The coordinator deliberately separates logical requests from physical
    provider inputs. A duplicate text can remain present as a whole-query,
    atomic requirement, and follow-up requirement while it is sent to the
    provider exactly once. The returned bundle preserves every logical role
    and slot identity, so downstream contextual selection never has to guess
    which requirement a deduplicated vector belonged to.
    """

    def __init__(self, model, *, model_id: str = "", dimension: int | None = None):
        self.model = model
        self.model_id = model_id or str(
            getattr(getattr(model, "config", None), "embedding_model", "unknown")
        )
        self.dimension = int(
            dimension
            or getattr(getattr(model, "config", None), "embedding_dimension", 0)
            or getattr(model, "dimension", 0)
        )
        self.last_batch_size = 0
        self.last_logical_query_count = 0
        self.last_provider_calls = 0
        self.cache_hits = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embedding_space(self, *, dimension: int | None = None) -> EmbeddingSpace:
        """Return the versioned space contract used by this coordinator."""

        resolved_dimension = int(dimension or self.dimension)
        if resolved_dimension <= 0:
            raise ValueError("embedding dimension is required before building a space")
        config = getattr(self.model, "config", None)
        provider = str(
            getattr(self.model, "embedding_provider", "")
            or getattr(config, "embedding_provider", "")
            or type(self.model).__name__.casefold()
        )
        revision = str(
            getattr(self.model, "embedding_revision", "")
            or getattr(config, "embedding_revision", "")
            or "unspecified"
        )
        return EmbeddingSpace(
            provider=provider,
            model_id=self.model_id or "unknown",
            revision=revision,
            dimension=resolved_dimension,
            dtype="float32",
            preprocessing="nfkc_space_v1",
            normalization="l2_float32_v1",
        )

    @staticmethod
    def _coerce_request(item: object) -> QueryVectorRequest:
        if isinstance(item, QueryVectorRequest):
            return item
        if isinstance(item, Mapping):
            role = str(item.get("role", "atomic"))
            text = str(item.get("text", ""))
            return QueryVectorRequest(
                role=role if role in _QUERY_ROLES else "atomic",  # type: ignore[arg-type]
                text=text,
                query_id=str(item.get("query_id", "")),
                slot_id=str(item.get("slot_id", "")),
            )
        if isinstance(item, tuple):
            if len(item) == 2:
                role, text = item
                query_id, slot_id = "", ""
            elif len(item) == 3:
                role, text, slot_id = item
                query_id = ""
            elif len(item) == 4:
                role, text, query_id, slot_id = item
            else:
                raise ValueError("query vector tuple must have 2, 3, or 4 items")
            role_value = str(role)
            return QueryVectorRequest(
                role=role_value if role_value in _QUERY_ROLES else "atomic",  # type: ignore[arg-type]
                text=str(text),
                query_id=str(query_id),
                slot_id=str(slot_id),
            )
        return QueryVectorRequest(role="atomic", text=str(item))

    def _normalize_requests(
        self,
        requests: Sequence[QueryVectorRequest | tuple | str | Mapping[str, object]],
    ) -> list[tuple[QueryVectorRequest, str, str, str, str]]:
        """Normalize text while retaining every non-empty logical request."""

        normalized: list[tuple[QueryVectorRequest, str, str, str, str]] = []
        query_id_counts: dict[str, int] = {}
        for raw_item in requests:
            request = self._coerce_request(raw_item)
            text = normalize_query_text(request.text)
            if not text:
                continue
            role = str(request.role)
            if role not in _QUERY_ROLES:
                role = "atomic"
            text_hash = self._hash(text)
            base_query_id = str(request.query_id or "").strip() or f"q-{text_hash[:16]}"
            occurrence = query_id_counts.get(base_query_id, 0) + 1
            query_id_counts[base_query_id] = occurrence
            query_id = base_query_id if occurrence == 1 else f"{base_query_id}:{occurrence}"
            normalized.append(
                (
                    request,
                    text,
                    role,
                    query_id,
                    str(request.slot_id or "").strip(),
                )
            )
        if not normalized:
            raise ValueError("at least one non-empty query text is required")
        return normalized

    @staticmethod
    def _physical_id(space_id: str, text_hash: str) -> str:
        digest = hashlib.sha256(
            f"{space_id}\0{text_hash}".encode("utf-8")
        ).hexdigest()
        return "query-vector:sha256:" + digest

    def _bundle_from_normalized(
        self,
        requests: list[tuple[QueryVectorRequest, str, str, str, str]],
        physical_vectors: Mapping[str, Any],
        *,
        embedding_space: EmbeddingSpace,
        source_request_hash: str = "",
    ) -> QueryVectorBundle:
        """Materialize one logical bundle from unique normalized vectors."""

        if int(embedding_space.dimension) <= 0:
            raise ValueError("embedding space dimension must be positive")
        dimension = int(embedding_space.dimension)
        space_id = embedding_space.canonical_id
        physical_items: list[PhysicalQueryVector] = []
        physical_by_text: dict[str, PhysicalQueryVector] = {}
        for _request, text, _role, _query_id, _slot_id in requests:
            if text in physical_by_text:
                continue
            if text not in physical_vectors:
                raise ValueError("missing precomputed vector for normalized query text")
            vector = normalize_embedding(physical_vectors[text], dimension)
            # A read-only flag makes accidental in-place mutation fail early;
            # all logical compatibility views intentionally share this one RAM
            # allocation rather than making hidden copies.
            vector.setflags(write=False)
            text_hash = self._hash(text)
            physical = PhysicalQueryVector(
                physical_id=self._physical_id(space_id, text_hash),
                text_hash=text_hash,
                vector=vector,
                embedding_space_id=space_id,
            )
            physical_by_text[text] = physical
            physical_items.append(physical)
        logical_queries: list[QueryVector] = []
        for _request, text, role, query_id, slot_id in requests:
            physical = physical_by_text[text]
            logical_queries.append(
                QueryVector(
                    query_id=query_id,
                    text_hash=physical.text_hash,
                    role=role,  # type: ignore[arg-type]
                    vector=physical.vector,
                    slot_id=slot_id,
                    text=text,
                    physical_id=physical.physical_id,
                    embedding_space_id=space_id,
                )
            )
        whole = next(
            (item.vector for item in logical_queries if item.role == "whole"),
            logical_queries[0].vector,
        )
        whole_physical_id = next(
            (item.physical_id for item in logical_queries if item.role == "whole"),
            logical_queries[0].physical_id,
        )
        return QueryVectorBundle(
            model_id=self.model_id,
            dimension=dimension,
            whole=whole,
            queries=tuple(logical_queries),
            embedding_space=embedding_space,
            physical_vectors=tuple(physical_items),
            source_request_hash=str(source_request_hash or ""),
            schema_version=2,
            whole_physical_id=whole_physical_id,
        )

    def bundle_from_precomputed_vectors(
        self,
        requests: Sequence[QueryVectorRequest | tuple | str | Mapping[str, object]],
        vectors: Sequence[Any] | Mapping[str, Any],
        *,
        embedding_space: EmbeddingSpace | None = None,
        source_request_hash: str = "",
    ) -> QueryVectorBundle:
        """Create a v3 bundle from frozen/local vectors without a provider call.

        This is the bridge for replay and legacy exact-vector overrides. It
        normalizes and validates vectors using the same code path as live
        provider results, but never invokes ``model.embed``.
        """

        normalized = self._normalize_requests(requests)
        space = embedding_space or self.embedding_space()
        physical_texts = list(dict.fromkeys(text for _r, text, *_rest in normalized))
        by_text: dict[str, Any] = {}
        if isinstance(vectors, Mapping):
            for raw_text, vector in vectors.items():
                text = normalize_query_text(str(raw_text))
                if text:
                    by_text[text] = vector
        else:
            values = list(vectors)
            if len(values) == len(physical_texts):
                by_text = dict(zip(physical_texts, values, strict=True))
            elif len(values) == len(normalized):
                for (_request, text, *_rest), vector in zip(
                    normalized, values, strict=True
                ):
                    by_text.setdefault(text, vector)
            else:
                raise ValueError(
                    "precomputed vector count must match physical or logical query count"
                )
        return self._bundle_from_normalized(
            normalized,
            by_text,
            embedding_space=space,
            source_request_hash=source_request_hash,
        )

    def embed_request_bundle_sync(
        self,
        requests: Sequence[QueryVectorRequest | tuple | str | Mapping[str, object]],
        *,
        source_request_hash: str = "",
    ) -> QueryVectorBundle:
        """Embed one request's unique texts once while preserving all bindings."""

        normalized = self._normalize_requests(requests)
        physical_texts = list(dict.fromkeys(text for _r, text, *_rest in normalized))
        self.last_logical_query_count = len(normalized)
        self.cache_hits += len(normalized) - len(physical_texts)
        matrix = np.asarray(self.model.embed(physical_texts), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(physical_texts):
            raise ValueError("embedding provider returned an invalid batch shape")
        dimension = self.dimension or int(matrix.shape[1])
        if matrix.shape[1] != dimension:
            raise ValueError("embedding dimension does not match coordinator")
        self.dimension = dimension
        self.last_batch_size = len(physical_texts)
        self.last_provider_calls += 1
        return self._bundle_from_normalized(
            normalized,
            {
                text: matrix[index]
                for index, text in enumerate(physical_texts)
            },
            embedding_space=self.embedding_space(dimension=dimension),
            source_request_hash=source_request_hash,
        )

    # The historical name remains an exact compatibility alias. Its behavior
    # now retains duplicate logical requests instead of discarding them.
    def embed_query_bundle_sync(
        self,
        texts: Sequence[QueryVectorRequest | tuple | str | Mapping[str, object]],
        *,
        source_request_hash: str = "",
    ) -> QueryVectorBundle:
        return self.embed_request_bundle_sync(
            texts,
            source_request_hash=source_request_hash,
        )

    async def embed_request_bundle(
        self,
        requests: Sequence[QueryVectorRequest | tuple | str | Mapping[str, object]],
        *,
        source_request_hash: str = "",
    ) -> QueryVectorBundle:
        return await asyncio.to_thread(
            self.embed_request_bundle_sync,
            requests,
            source_request_hash=source_request_hash,
        )

    async def embed_query_bundle(
        self,
        texts: Sequence[QueryVectorRequest | tuple | str | Mapping[str, object]],
        *,
        source_request_hash: str = "",
    ) -> QueryVectorBundle:
        return await self.embed_request_bundle(
            texts,
            source_request_hash=source_request_hash,
        )
