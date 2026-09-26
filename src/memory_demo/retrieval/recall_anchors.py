"""Question-blind Episode navigation into immutable Source record windows.

Scores are lexical location proposals, never semantic or evidentiary approval.
The strongest location is retained even when unavailable: callers own fallback.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import re
import unicodedata

from memory_demo.retrieval.recall_records import project_source


METHOD = "episode-record-bigram-idf-cosine-v1"
_RUN = re.compile(r"[a-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _features(text: str) -> frozenset[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return frozenset(run[i:i + 2] for run in _RUN.findall(text) for i in range(len(run) - 1))


def _positive_id(value: object) -> bool:
    return type(value) is int and value > 0


class EpisodeAnchorIndex:
    """Bind Source text and Episode summaries before constructing lazy anchors.

    Inputs remain caller-owned. Mutation of a bound item is an error, preventing
    cached source offsets from silently surviving changed text or ownership.
    """

    def __init__(self, sources: dict[int, str], episodes: dict[int, dict]):
        if not isinstance(sources, Mapping) or not isinstance(episodes, Mapping):
            raise ValueError("sources and episodes must be mappings")
        self._sources = sources
        self._episodes = episodes
        self._source_hashes: dict[int, str] = {}
        self._bindings: dict[int, tuple[int, str]] = {}
        self._source_episodes: dict[int, list[int]] = {}
        self._records: dict[int, list[dict]] = {}
        self._documents: dict[int, list[frozenset[str]]] = {}
        self._dfs: dict[int, Counter] = {}
        self._anchors: dict[int, list[dict]] = {}
        for sid, raw in sources.items():
            if not _positive_id(sid) or not isinstance(raw, str):
                raise ValueError("invalid Source identity or text")
            self._source_hashes[sid] = _sha(raw)
            self._source_episodes[sid] = []
        for eid, episode in episodes.items():
            if not _positive_id(eid) or not isinstance(episode, Mapping):
                raise ValueError("invalid Episode identity")
            sid, text = episode.get("source_id"), episode.get("text")
            if not _positive_id(sid) or sid not in sources or not isinstance(text, str):
                raise ValueError("invalid Episode Source binding or text")
            expected = episode.get("source_sha256")
            if expected is not None and expected != self._source_hashes[sid]:
                raise ValueError("Episode Source hash does not match initialization snapshot")
            self._bindings[eid] = sid, _sha(text)
            self._source_episodes[sid].append(eid)
        for ids in self._source_episodes.values():
            ids.sort()
        self._fingerprint = _sha(json.dumps({
            "method": METHOD,
            "sources": sorted(self._source_hashes.items()),
            "episodes": sorted((eid, *binding) for eid, binding in self._bindings.items()),
        }, ensure_ascii=False, separators=(",", ":")))

    def _source(self, sid: int) -> str:
        if not _positive_id(sid) or sid not in self._source_hashes:
            raise ValueError("unknown Source")
        raw = self._sources.get(sid)
        if not isinstance(raw, str) or _sha(raw) != self._source_hashes[sid]:
            raise ValueError("Source changed after anchor initialization")
        return raw

    def _episode(self, eid: int) -> tuple[int, str]:
        if not _positive_id(eid) or eid not in self._bindings:
            raise ValueError("unknown Episode")
        sid, digest = self._bindings[eid]
        episode = self._episodes.get(eid)
        if (not isinstance(episode, Mapping) or episode.get("source_id") != sid
                or not isinstance(episode.get("text"), str) or _sha(episode["text"]) != digest):
            raise ValueError("Episode changed after anchor initialization")
        if episode.get("source_sha256") not in (None, self._source_hashes[sid]):
            raise ValueError("Episode Source hash changed after anchor initialization")
        self._source(sid)
        return sid, episode["text"]

    @property
    def fingerprint(self) -> str:
        if set(self._sources) != set(self._source_hashes) or set(self._episodes) != set(self._bindings):
            raise ValueError("anchor snapshot identities changed")
        for sid in self._source_hashes:
            self._source(sid)
        for eid in self._bindings:
            self._episode(eid)
        return self._fingerprint

    def records(self, sid: int) -> list[dict]:
        raw = self._source(sid)
        for eid in self._source_episodes[sid]:
            self._episode(eid)
        if sid not in self._records:
            ids = self._source_episodes[sid]
            if not ids:
                raise ValueError("Source has no Episode binding for record projection")
            self._records[sid] = project_source(sid, raw, ids)
            documents = []
            for record in self._records[sid]:
                aliases = "\n".join(str(value) for value in record["speaker_aliases"].values())
                documents.append(_features(record["text"] + "\n" + aliases))
            self._documents[sid] = documents
            self._dfs[sid] = Counter(term for document in documents for term in document)
        return deepcopy(self._records[sid])

    def anchors(self, eid: int) -> list[dict]:
        sid, text = self._episode(eid)
        if eid not in self._anchors:
            records = self.records(sid)
            query = _features(text)
            count, df = len(records), self._dfs[sid]
            weights = {term: (1.0 + math.log((count + 1) / (df[term] + 1))) ** 2
                       for term in query | frozenset(df)}
            query_norm = math.sqrt(math.fsum(weights[term] for term in query))
            candidates = []
            if query_norm:
                for record, document in zip(records, self._documents[sid]):
                    shared = query & document
                    if not shared:
                        continue
                    doc_norm = math.sqrt(math.fsum(weights[term] for term in document))
                    score = math.fsum(weights[term] for term in shared) / (query_norm * doc_norm)
                    candidates.append({**record, "score": min(1.0, score), "method": METHOD,
                                       "episode_id": eid, "episode_text_sha256": self._bindings[eid][1]})
            candidates.sort(key=lambda item: (-item["score"], item["context_start"], item["record_id"]))
            self._anchors[eid] = candidates[:1]
        return deepcopy(self._anchors[eid])

    def window(self, eid: int, *, unavailable: list[tuple[int, int]], max_chars: int) -> dict | None:
        sid, _ = self._episode(eid)
        raw = self._source(sid)
        if type(max_chars) is not int or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        if not isinstance(unavailable, (list, tuple)):
            raise ValueError("unavailable must contain Source intervals")
        blocked = []
        for span in unavailable:
            if (not isinstance(span, (list, tuple)) or len(span) != 2
                    or any(type(value) is not int for value in span)
                    or not 0 <= span[0] <= span[1] <= len(raw)):
                raise ValueError("invalid unavailable Source interval")
            if span[0] < span[1]:
                blocked.append(span)

        def available(start: int, end: int) -> bool:
            return all(end <= lo or start >= hi for lo, hi in blocked)

        anchors = self.anchors(eid)
        if not anchors:
            return None
        anchor = anchors[0]
        start, end = anchor["context_start"], anchor["context_end"]
        if not available(start, end):
            return None
        records = self.records(sid)
        left = right = next(i for i, record in enumerate(records) if record["record_id"] == anchor["record_id"])
        while end - start <= max_chars:
            choices = []
            if left > 0:
                lo = records[left - 1]["context_start"]
                if end - lo <= max_chars and available(lo, end):
                    choices.append((start - lo, 0, lo, end))
            if right + 1 < len(records):
                hi = records[right + 1]["context_end"]
                if hi - start <= max_chars and available(start, hi):
                    choices.append((hi - end, 1, start, hi))
            if not choices:
                break
            _, side, start, end = min(choices)
            if side == 0:
                left -= 1
            else:
                right += 1
        return {
            "start": start, "end": end,
            "anchor_record_id": anchor["record_id"], "anchor_score": anchor["score"],
            "method": METHOD, "source_sha256": self._source_hashes[sid],
            "episode_text_sha256": self._bindings[eid][1],
            "overflow": end - start > max_chars,
            "overflow_chars": max(0, end - start - max_chars),
        }
