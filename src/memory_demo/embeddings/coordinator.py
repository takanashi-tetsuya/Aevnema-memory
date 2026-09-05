from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from collections.abc import Sequence

import numpy as np

from memory_demo.embeddings.codec import normalize_embedding
from memory_demo.types import QueryVector, QueryVectorBundle


_SPACE_RE = re.compile(r"\s+")


def normalize_query_text(text: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(text or ""))).strip()


class EmbeddingCoordinator:
    """The single request-level embedding boundary.

    A coordinator is intentionally a very small adapter around the existing
    model client.  It deduplicates normalized texts and sends one batch per
    request; contextual association code only receives the resulting vectors
    and therefore cannot fan out provider calls.
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
        self.last_provider_calls = 0
        self.cache_hits = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_query_bundle_sync(
        self,
        texts: Sequence[tuple[str, str] | str],
        *,
        source_request_hash: str = "",
    ) -> QueryVectorBundle:
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in texts:
            if isinstance(item, tuple):
                role, raw_text = item
            else:
                role, raw_text = "atomic", item
            text = normalize_query_text(str(raw_text))
            if not text or text in seen:
                if text:
                    self.cache_hits += 1
                continue
            seen.add(text)
            normalized.append((str(role), text))
        if not normalized:
            raise ValueError("at least one non-empty query text is required")
        matrix = np.asarray(self.model.embed([text for _role, text in normalized]), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(normalized):
            raise ValueError("embedding provider returned an invalid batch shape")
        dimension = self.dimension or int(matrix.shape[1])
        if matrix.shape[1] != dimension:
            raise ValueError("embedding dimension does not match coordinator")
        matrix = np.stack(
            [normalize_embedding(row, dimension) for row in matrix], axis=0
        ).astype(np.float32, copy=False)
        self.last_batch_size = len(normalized)
        self.last_provider_calls += 1
        whole_index = next(
            (index for index, (role, _text) in enumerate(normalized) if role == "whole"),
            0,
        )
        queries: list[QueryVector] = []
        for index, (role, text) in enumerate(normalized):
            role_value = role if role in {"whole", "atomic", "followup"} else "atomic"
            queries.append(
                QueryVector(
                    query_id=f"q-{self._hash(text)[:16]}",
                    text_hash=self._hash(text),
                    role=role_value,
                    vector=np.ascontiguousarray(matrix[index], dtype=np.float32),
                    slot_id="",
                    text=text,
                )
            )
        return QueryVectorBundle(
            model_id=self.model_id,
            dimension=dimension,
            whole=np.ascontiguousarray(matrix[whole_index], dtype=np.float32),
            queries=tuple(queries),
        )

    async def embed_query_bundle(
        self,
        texts: Sequence[tuple[str, str] | str],
        *,
        source_request_hash: str = "",
    ) -> QueryVectorBundle:
        return await asyncio.to_thread(
            self.embed_query_bundle_sync,
            texts,
            source_request_hash=source_request_hash,
        )
