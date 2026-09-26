"""Source-bound follow-up cues and pure checkpoint transitions.

This module performs no model calls or writes. The caller embeds only
``pending_cue_texts`` and atomically saves the returned queue, seeds and graph
epoch in the same RecallSessionStore write. Cues schedule searches; they never
certify an answer, coverage or an association.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
import unicodedata

import numpy as np

from memory_demo.associations.spreading import SpreadingSeed
from memory_demo.embeddings.codec import normalize_embedding


@dataclass(frozen=True)
class CueLimits:
    max_total: int = 12
    max_per_wave: int = 3
    max_cue_chars: int = 80
    max_quote_chars: int = 800

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CueEnqueue:
    queue: dict
    stats: dict


@dataclass(frozen=True)
class CueExpansion:
    queue: dict
    seeds: list[SpreadingSeed]
    ranked_episode_ids: list[int]
    changed: bool
    processed_count: int
    scored_count: int
    episode_scores: dict[int, float]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _copy_queue(queue: dict | None) -> dict:
    if queue is None:
        return {"version": 1, "pending": [], "processed": [], "excluded_cue_keys": []}
    if not isinstance(queue, dict) or queue.get("version") != 1:
        raise ValueError("unsupported follow-up cue queue")
    result = {"version": 1}
    seen = set()
    for field in ("pending", "processed"):
        entries = queue.get(field)
        if not isinstance(entries, list):
            raise ValueError("invalid follow-up cue queue entries")
        result[field] = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("cue"), str):
                raise ValueError("invalid queued cue")
            key = _key(entry["cue"])
            if not key or key in seen or entry.get("cue_id") != _sha(key):
                raise ValueError("duplicate or invalid queued cue identity")
            seen.add(key)
            result[field].append(dict(entry))
    excluded = queue.get("excluded_cue_keys", [])
    if not isinstance(excluded, list) or any(not isinstance(v, str) for v in excluded):
        raise ValueError("invalid initial cue identities")
    result["excluded_cue_keys"] = list(excluded)
    return result


def enqueue_followup_cues(
    queue: dict | None,
    proposed: object,
    *,
    visible: Sequence[Mapping],
    sources: Mapping[int, str],
    episodes: Mapping[int, Mapping],
    initial_cues: Sequence[str] = (),
    limits: CueLimits = CueLimits(),
) -> CueEnqueue:
    """Return a detached queue containing a bounded set of literal new cues.

    A cue must itself occur verbatim inside its quote: local checks cannot
    establish that a freely generated paraphrase follows from that quote.
    The quote must have exactly one absolute location across this round's
    visible windows for the specified Source/Episode pair. Overlapping windows
    containing the same absolute span do not create duplicate evidence.
    """
    result = _copy_queue(queue)
    stats = {"accepted": 0, "rejected": 0, "reasons": {}}

    def reject(reason):
        stats["rejected"] += 1
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
    excluded = set(result["excluded_cue_keys"])
    excluded.update(_key(c) for c in initial_cues if isinstance(c, str) and c.strip())
    result["excluded_cue_keys"] = sorted(excluded)
    seen = excluded | {_key(c["cue"]) for field in ("pending", "processed") for c in result[field]}
    capacity = min(limits.max_per_wave, max(0, limits.max_total - len(result["pending"]) - len(result["processed"])))
    if not isinstance(proposed, list):
        reject("invalid_container")
        return CueEnqueue(result, stats)
    for entry in proposed:
        if capacity == 0:
            reject("capacity_reached")
            continue
        if not isinstance(entry, dict):
            reject("invalid_entry")
            continue
        cue, quote = entry.get("cue"), entry.get("quote")
        sid, eid = entry.get("source_id"), entry.get("episode_id")
        if not isinstance(cue, str) or not isinstance(quote, str):
            reject("invalid_text_fields")
            continue
        cue = cue.strip()
        if not 2 <= len(cue) <= limits.max_cue_chars or not quote.strip() or len(quote) > limits.max_quote_chars:
            reject("text_length")
            continue
        if cue not in quote:
            reject("cue_not_literal_quote_substring")
            continue
        key = _key(cue)
        if key in seen:
            reject("duplicate")
            continue
        if type(sid) is not int or type(eid) is not int:
            reject("invalid_source_episode_id")
            continue
        if sid not in sources or eid not in episodes or episodes[eid]["source_id"] != sid:
            reject("source_episode_mismatch")
            continue
        raw = sources[sid]
        spans = set()
        for window in visible:
            allowed = window.get("episode_ids", [window.get("episode_id")])
            if window.get("source_id") != sid or not isinstance(allowed, (list, tuple)) or not any(type(value) is int and value == eid for value in allowed):
                continue
            start, end, text = window.get("start"), window.get("end"), window.get("text")
            if type(start) is not int or type(end) is not int or not isinstance(text, str):
                continue
            if not 0 <= start <= end <= len(raw) or raw[start:end] != text:
                continue
            position = text.find(quote)
            while position >= 0:
                spans.add((start + position, start + position + len(quote)))
                position = text.find(quote, position + 1)
        if len(spans) != 1:
            reject("quote_not_uniquely_visible")
            continue
        start, end = spans.pop()
        result["pending"].append({"cue_id": _sha(key), "cue": cue, "source_id": sid,
                                  "episode_id": eid, "source_sha256": _sha(raw),
                                  "start": start, "end": end, "quote": quote})
        seen.add(key)
        capacity -= 1
        stats["accepted"] += 1
    return CueEnqueue(result, stats)


def pending_cue_texts(queue: dict | None, *, max_batch: int = 3) -> list[str]:
    if type(max_batch) is not int or max_batch <= 0:
        raise ValueError("max_batch must be a positive integer")
    return [entry["cue"] for entry in _copy_queue(queue)["pending"][:max_batch]]


def _canonical_seeds(seeds: Sequence[SpreadingSeed | Mapping]) -> list[SpreadingSeed]:
    combined = {}
    for value in seeds:
        seed = value if isinstance(value, SpreadingSeed) else SpreadingSeed(**value)
        if seed.node_type not in ("episode", "concept") or type(seed.node_id) is not int or seed.node_id < 0:
            raise ValueError("invalid seed node")
        if not math.isfinite(seed.activation) or not 0 <= seed.activation <= 1:
            raise ValueError("invalid seed activation")
        root = seed.root_id if seed.root_id is not None else f"{seed.node_type}:{seed.node_id}"
        if not isinstance(root, str) or not root:
            raise ValueError("invalid seed root")
        key = (seed.node_type, seed.node_id, root)
        if seed.activation > combined.get(key, 0):
            combined[key] = seed.activation
    return [SpreadingSeed(kind, eid, score, root) for (kind, eid, root), score in sorted(combined.items())]


def expand_pending_cues(
    queue: dict | None,
    vectors: Sequence,
    *,
    episode_ids: Sequence[int],
    episode_matrix: np.ndarray,
    episodes: Mapping[int, Mapping],
    sources: Mapping[int, str],
    existing_seeds: Sequence[SpreadingSeed | Mapping],
    seeds_per_cue: int,
    max_batch: int = 3,
) -> CueExpansion:
    """Rank newly embedded cues against an existing normalized episode matrix.

    Nothing is consumed if validation or ranking raises. The caller must keep
    the same queue between pending_cue_texts and this operation, then atomically
    persist the returned queue with its seeds. An interrupted HTTP call may be
    retried after resume; a successfully persisted processed cue is not embedded
    again. Exactly-once external billing across a process crash is not promised.
    """
    result = _copy_queue(queue)
    texts = pending_cue_texts(result, max_batch=max_batch)
    batch = result["pending"][:len(texts)]
    vectors = list(vectors)
    if len(vectors) != len(batch):
        raise ValueError("embedding count does not match pending cue batch")
    if type(seeds_per_cue) is not int or seeds_per_cue <= 0:
        raise ValueError("seeds_per_cue must be a positive integer")
    prior = _canonical_seeds(existing_seeds)
    if not batch:
        return CueExpansion(result, prior, [], False, 0, 0, {})
    ids = list(episode_ids)
    if len(set(ids)) != len(ids) or any(type(eid) is not int or eid not in episodes for eid in ids):
        raise ValueError("episode matrix IDs must be unique existing episodes")
    matrix = np.asarray(episode_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("invalid episode embedding matrix")
    if len(ids) and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("episode matrix must already be normalized")
    for cue in batch:
        sid, eid = cue["source_id"], cue["episode_id"]
        raw = sources.get(sid)
        if raw is None or eid not in episodes or episodes[eid]["source_id"] != sid or _sha(raw) != cue["source_sha256"] or raw[cue["start"]:cue["end"]] != cue["quote"] or cue["cue"] not in cue["quote"]:
            raise ValueError("queued cue source evidence changed")
    additions = []
    all_scores = np.zeros(len(ids), dtype=np.float32)
    for vector in vectors:
        scores = matrix @ normalize_embedding(vector, matrix.shape[1])
        all_scores = np.maximum(all_scores, np.clip(scores, 0.0, 1.0))
        # Episode ID is the deterministic secondary key, independent of row order.
        ranking = sorted(range(len(ids)), key=lambda position: (-float(scores[position]), ids[position]))
        for position in ranking[:seeds_per_cue]:
            score = min(1.0, max(0.0, float(scores[position])))
            if score > 0:
                eid = ids[position]
                root = "source:" + _sha(sources[episodes[eid]["source_id"]])
                additions.append(SpreadingSeed("episode", eid, score, root))
    # Canonicalize old Episode roots to the same Source rule as new candidates:
    # finding the same Source with several cues does not create independent roots.
    source_prior = [SpreadingSeed(seed.node_type, seed.node_id, seed.activation,
                    "source:" + _sha(sources[episodes[seed.node_id]["source_id"]]))
                    if seed.node_type == "episode" else seed for seed in prior]
    merged = _canonical_seeds([*source_prior, *additions])
    result["pending"] = result["pending"][len(batch):]
    result["processed"].extend(batch)
    ranked = [ids[p] for p in sorted(range(len(ids)), key=lambda p: (-float(all_scores[p]), ids[p]))]
    # These scores cover only this new batch. The caller retains historical
    # maxima so a weak later cue cannot overwrite earlier fallback priorities.
    episode_scores = {eid: float(all_scores[position]) for position, eid in enumerate(ids)}
    return CueExpansion(result, merged, ranked, merged != prior, len(batch), len(ids) * len(batch), episode_scores)


def restart_spreading_epoch(state: dict, new_seeds: Sequence[SpreadingSeed | Mapping]) -> dict:
    """Return one atomic session transition when the seed fingerprint changes.

    SpreadingRecall rejects adding seeds to an existing checkpoint. A changed
    seed set starts a fresh graph epoch while Source offsets, evidence and used
    edges remain intact. The caller reports graph work as
    ``spreading_expansion_base + current_result.expansions`` after every search.
    Restarts repeat some graph work, which remains counted rather than hidden.
    """
    merged = _canonical_seeds(new_seeds)
    prior = _canonical_seeds(state.get("seeds", []))
    result = dict(state)
    if merged == prior:
        return result
    checkpoint = state.get("spreading")
    completed = checkpoint.get("expansions", 0) if checkpoint else 0
    base = state.get("spreading_expansion_base", 0)
    if any(type(v) is not int or v < 0 for v in (base, completed)):
        raise ValueError("invalid accumulated spreading work")
    result["spreading_expansion_base"] = base + completed
    result["spreading"] = None
    result["seeds"] = [asdict(seed) for seed in merged]
    metrics = dict(state.get("metrics", {}))
    metrics["edge_expansions"] = base + completed
    metrics["spreading_restarts"] = metrics.get("spreading_restarts", 0) + 1
    result["metrics"] = metrics
    return result
