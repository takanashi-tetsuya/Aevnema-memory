from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CPU brute-force cosine search over float32 embeddings."
    )
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--dimension", type=int, default=1_024)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.count, args.dimension, args.queries, args.top_k) <= 0:
        raise SystemExit("all numeric arguments must be positive")

    rng = np.random.default_rng(args.seed)
    allocation_started = time.perf_counter()
    embeddings = rng.standard_normal(
        (args.count, args.dimension), dtype=np.float32
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    allocation_seconds = time.perf_counter() - allocation_started
    latencies_ms: list[float] = []
    checksum = 0
    for _ in range(args.queries):
        query = rng.standard_normal(args.dimension, dtype=np.float32)
        query /= np.linalg.norm(query)
        started = time.perf_counter()
        scores = embeddings @ query
        k = min(args.top_k, args.count)
        positions = np.argpartition(scores, -k)[-k:]
        positions = positions[np.argsort(scores[positions])[::-1]]
        latencies_ms.append((time.perf_counter() - started) * 1_000)
        checksum ^= int(positions[0])

    report = {
        "dtype": str(embeddings.dtype),
        "shape": list(embeddings.shape),
        "matrix_bytes": int(embeddings.nbytes),
        "matrix_gib": embeddings.nbytes / 1024**3,
        "allocation_and_normalization_seconds": allocation_seconds,
        "queries": args.queries,
        "top_k": min(args.top_k, args.count),
        "latency_ms": {
            "min": min(latencies_ms),
            "mean": sum(latencies_ms) / len(latencies_ms),
            "max": max(latencies_ms),
        },
        "checksum": checksum,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
