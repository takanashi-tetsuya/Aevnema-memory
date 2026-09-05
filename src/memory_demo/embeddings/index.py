from __future__ import annotations

from contextlib import contextmanager
from threading import Condition, Lock
from typing import Iterable

import numpy as np

from memory_demo.embeddings.codec import decode_embedding, normalize_embedding


class EmbeddingIndex:
    """Contiguous float32 brute-force index with concurrent readers."""

    def __init__(self, dimension: int, initial_capacity: int = 16):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.capacity = max(1, initial_capacity)
        self.count = 0
        self.ids = np.empty(self.capacity, dtype=np.int64)
        self.embeddings = np.empty((self.capacity, dimension), dtype=np.float32)
        self._positions: dict[int, int] = {}
        self._condition = Condition(Lock())
        self._active_readers = 0
        self._writer_active = False
        self._waiting_writers = 0

    @contextmanager
    def _read_access(self):
        with self._condition:
            while self._writer_active or self._waiting_writers:
                self._condition.wait()
            self._active_readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_readers -= 1
                if self._active_readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def _write_access(self):
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer_active or self._active_readers:
                    self._condition.wait()
                self._writer_active = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._condition:
                self._writer_active = False
                self._condition.notify_all()

    def _expand_locked(self, required: int) -> None:
        if required <= self.capacity:
            return
        new_capacity = self.capacity
        while new_capacity < required:
            new_capacity *= 2
        new_ids = np.empty(new_capacity, dtype=np.int64)
        new_embeddings = np.empty((new_capacity, self.dimension), dtype=np.float32)
        new_ids[: self.count] = self.ids[: self.count]
        new_embeddings[: self.count] = self.embeddings[: self.count]
        self.ids = new_ids
        self.embeddings = new_embeddings
        self.capacity = new_capacity

    def upsert(self, node_id: int, embedding) -> None:
        vector = normalize_embedding(embedding, self.dimension)
        with self._write_access():
            position = self._positions.get(int(node_id))
            if position is not None:
                self.embeddings[position] = vector
                return
            self._expand_locked(self.count + 1)
            position = self.count
            self.embeddings[position] = vector
            self.ids[position] = int(node_id)
            self._positions[int(node_id)] = position
            self.count = position + 1

    def remove(self, node_id: int) -> bool:
        """Remove in O(dimension) by moving the last active row into the gap."""
        with self._write_access():
            position = self._positions.pop(int(node_id), None)
            if position is None:
                return False
            last = self.count - 1
            if position != last:
                moved_id = int(self.ids[last])
                self.ids[position] = moved_id
                self.embeddings[position] = self.embeddings[last]
                self._positions[moved_id] = position
            self.count = last
            return True

    def rebuild(
        self,
        count: int,
        batches: Iterable,
    ) -> None:
        with self._write_access():
            capacity = max(16, int(count * 1.1) + 1)
            ids = np.empty(capacity, dtype=np.int64)
            embeddings = np.empty((capacity, self.dimension), dtype=np.float32)
            positions: dict[int, int] = {}
            loaded = 0
            for batch in batches:
                for row in batch:
                    if loaded >= capacity:
                        new_capacity = capacity * 2
                        expanded_ids = np.empty(new_capacity, dtype=np.int64)
                        expanded_embeddings = np.empty(
                            (new_capacity, self.dimension), dtype=np.float32
                        )
                        expanded_ids[:loaded] = ids[:loaded]
                        expanded_embeddings[:loaded] = embeddings[:loaded]
                        ids, embeddings, capacity = (
                            expanded_ids,
                            expanded_embeddings,
                            new_capacity,
                        )
                    node_id = int(row["id"])
                    ids[loaded] = node_id
                    embeddings[loaded] = decode_embedding(
                        row["embedding"], self.dimension, renormalize=True
                    )
                    positions[node_id] = loaded
                    loaded += 1
            self.ids = ids
            self.embeddings = embeddings
            self.capacity = capacity
            self._positions = positions
            self.count = loaded

    def search(self, query, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        normalized = normalize_embedding(query, self.dimension)
        with self._read_access():
            count = self.count
            if count == 0:
                return []
            scores = self.embeddings[:count] @ normalized
            k = min(top_k, count)
            if k == count:
                candidate_positions = np.arange(count)
            else:
                candidate_positions = np.argpartition(scores, -k)[-k:]
            ordered = candidate_positions[
                np.argsort(scores[candidate_positions])[::-1]
            ]
            return [
                (int(self.ids[position]), float(scores[position]))
                for position in ordered
            ]

    def search_many(
        self,
        query_matrix: np.ndarray,
        top_k: int,
        block_rows: int = 32_768,
    ) -> list[list[tuple[int, float]]]:
        """Search several normalized queries in one blockwise exact scan.

        The implementation keeps the active matrix read-locked only while
        scoring.  It never touches the uninitialized tail of a preallocated
        array and returns the same cosine scores as calling :meth:`search`
        independently.  Blockwise scoring bounds the temporary ``rows ×
        queries`` matrix for large indexes.
        """
        matrix = np.asarray(query_matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("query_matrix shape does not match index dimension")
        if top_k <= 0:
            return [[] for _ in range(matrix.shape[0])]
        normalized = np.asarray(matrix, dtype=np.float32).copy()
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError("query vectors must be finite and non-zero")
        normalized /= norms.astype(np.float32)
        block_rows = max(1, int(block_rows))
        with self._read_access():
            count = int(self.count)
            if count == 0:
                return [[] for _ in range(matrix.shape[0])]
            k = min(int(top_k), count)
            # Keep a small candidate set per query while scanning blocks; the
            # final stable ordering below makes ties deterministic by id.
            candidates: list[list[tuple[int, float]]] = [
                [] for _ in range(matrix.shape[0])
            ]
            for start in range(0, count, block_rows):
                stop = min(count, start + block_rows)
                block_scores = (
                    self.embeddings[start:stop] @ normalized.T
                )
                block_ids = self.ids[start:stop]
                for query_index in range(matrix.shape[0]):
                    scores = block_scores[:, query_index]
                    take = min(k, stop - start)
                    positions = np.argpartition(scores, -take)[-take:]
                    candidates[query_index].extend(
                        (int(block_ids[position]), float(scores[position]))
                        for position in positions
                    )
            return [
                [
                    (node_id, score)
                    for node_id, score in sorted(
                        values,
                        key=lambda item: (-item[1], item[0]),
                    )[:k]
                ]
                for values in candidates
            ]

    def add(self, node_id: int, embedding) -> None:
        """Alias used by the contextual indexes for incremental insertion."""
        self.upsert(node_id, embedding)

    @property
    def id_to_index(self) -> dict[int, int]:
        """Return a detached id→row mapping for diagnostics and replay."""
        with self._read_access():
            return dict(self._positions)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return a consistent copy of active ids and vectors."""
        with self._read_access():
            return (
                self.ids[: self.count].copy(),
                self.embeddings[: self.count].copy(),
                int(self.count),
            )

    def get_many(self, node_ids: Iterable[int]) -> dict[int, np.ndarray]:
        """Return copies so callers can score a small filtered subset safely."""
        requested = list(dict.fromkeys(int(value) for value in node_ids))
        with self._read_access():
            return {
                node_id: self.embeddings[position].copy()
                for node_id in requested
                if (position := self._positions.get(node_id)) is not None
            }

    @property
    def memory_bytes(self) -> int:
        with self._read_access():
            return int(self.ids.nbytes + self.embeddings.nbytes)
