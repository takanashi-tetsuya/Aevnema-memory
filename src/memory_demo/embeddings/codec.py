from __future__ import annotations

import numpy as np

FLOAT32_LE = np.dtype("<f4")


def normalize_embedding(value, dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (dimension,):
        raise ValueError(f"embedding shape {array.shape} != ({dimension},)")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding contains NaN or infinity")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("embedding is a zero vector")
    normalized = array / np.float32(norm)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def encode_embedding(value, dimension: int) -> bytes:
    normalized = normalize_embedding(value, dimension)
    return normalized.astype(FLOAT32_LE, copy=False).tobytes(order="C")


def decode_embedding(blob: bytes, dimension: int, renormalize: bool = True) -> np.ndarray:
    expected_bytes = dimension * FLOAT32_LE.itemsize
    if len(blob) != expected_bytes:
        raise ValueError(f"embedding BLOB has {len(blob)} bytes, expected {expected_bytes}")
    array = np.frombuffer(blob, dtype=FLOAT32_LE, count=dimension).astype(
        np.float32, copy=True
    )
    return normalize_embedding(array, dimension) if renormalize else array

