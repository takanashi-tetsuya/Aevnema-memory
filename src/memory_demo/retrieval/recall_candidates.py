"""Local, exhaustive candidate scheduling from literal and existing rank signals.

The index uses only supplied Episode/Source text. It has no model, database,
benchmark, plot-specific vocabulary or answer rules. Lexical scores are search
signals, never factual certification or additional independent graph roots.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
import unicodedata


_TOKENS = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[\u3400-\u4dbf\u4e00-\u9fff]+")


def _terms(text: str) -> Counter[str]:
    terms = Counter()
    for token in _TOKENS.findall(unicodedata.normalize("NFKC", text).casefold()):
        if token[0].isascii():
            terms["w:" + token] += 1
        else:
            # Single characters keep short clues searchable; their low weight
            # and independent-Source IDF limit ubiquitous grammatical text.
            for size in (1, 2, 3):
                terms.update(f"c{size}:" + token[start:start + size]
                             for start in range(len(token) - size + 1))
    return terms


class _TextField:
    def __init__(self, documents: Mapping[int, str], source_ids: Mapping[int, int]):
        postings = defaultdict(list)
        lengths = {}
        for identifier, text in documents.items():
            terms = _terms(text)
            lengths[identifier] = sum(terms.values())
            for term, frequency in terms.items():
                postings[term].append((identifier, frequency))
        average = sum(lengths.values()) / max(1, len(lengths))
        self.length_factors = {identifier: 1.2 * (0.25 + 0.75 * length / max(1.0, average))
                               for identifier, length in lengths.items()}
        source_count = len(set(source_ids.values()))
        self.idf = {}
        for term, entries in postings.items():
            # Ten Episodes from the same Source count as one document for DF.
            frequency = len({source_ids[identifier] for identifier, _ in entries})
            self.idf[term] = math.log1p((source_count - frequency + 0.5) / (frequency + 0.5))
        self.postings = dict(postings)

    def score(self, terms: Sequence[str]) -> dict[int, float]:
        scores = defaultdict(float)
        for term in terms:
            weight = 0.15 if term.startswith("c1:") else 1.0
            idf = self.idf.get(term, 0.0) * weight
            for identifier, frequency in self.postings.get(term, ()):
                scores[identifier] += idf * frequency * 2.2 / (frequency + self.length_factors[identifier])
        return dict(scores)


@dataclass(frozen=True)
class CandidateScore:
    source_id: int
    episode_bm25: float
    source_bm25: float
    lexical_rrf: float
    dense_rrf: float
    graph_rrf: float
    fused_score: float
    matched_queries: int


@dataclass(frozen=True)
class CandidateRanking:
    episode_ids: list[int]
    source_episode_ids: list[int]
    scores: dict[int, CandidateScore]


class RecallCandidateIndex:
    """Cache a mixed Chinese/English lexical index for one immutable snapshot.

    Each distinct query/cue/need supplies a lexical ranking. Its Episode BM25
    and Source BM25 are separately normalized, with Source text weighted 0.35.
    RRF uses Source rank, then the Episode's relative score within that Source,
    so a Source cannot win extra lexical votes by having many Episodes.
    Lexical votes are averaged across the distinct prompts; dense and graph
    each supply one additional RRF vote (constant 60). Zero lexical matches
    supply no vote. All known Episodes survive, with deterministic ID ties.

    ``source_episode_ids`` selects the highest-scoring representative of each
    Source; callers can use it directly for diverse Source reading windows.
    This class never creates roots: graph seeding must retain the caller's
    existing Source-content-hash deduplication rule.
    """
    def __init__(self, episodes: Mapping[int, Mapping], sources: Mapping[int, str]):
        self.episode_ids = tuple(sorted(episodes))
        self.source_by_episode = {}
        texts = {}
        for eid in self.episode_ids:
            row = episodes[eid]
            sid, text = row.get("source_id"), row.get("text")
            if type(eid) is not int or type(sid) is not int or sid not in sources or not isinstance(text, str):
                raise ValueError("Episode must have an integer ID, a present Source, and text")
            if not isinstance(sources[sid], str):
                raise ValueError("Source text must be a string")
            self.source_by_episode[eid] = sid
            texts[eid] = text
        used_sources = sorted(set(self.source_by_episode.values()))
        self.episodes_by_source = defaultdict(list)
        for eid, sid in self.source_by_episode.items():
            self.episodes_by_source[sid].append(eid)
        self._episodes = _TextField(texts, self.source_by_episode)
        self._sources = _TextField({sid: sources[sid] for sid in used_sources}, {sid: sid for sid in used_sources})
        self._query_cache = {}
        self.stats = {
            "episode_count": len(self.episode_ids), "source_count": len(used_sources),
            "episode_terms": len(self._episodes.postings), "source_terms": len(self._sources.postings),
            "episode_postings": sum(map(len, self._episodes.postings.values())),
            "source_postings": sum(map(len, self._sources.postings.values())),
        }

    @staticmethod
    def _prompts(query: str, cues: Sequence[str], needs: Sequence[str]) -> list[str]:
        unique = {}
        for text in (query, *cues, *needs):
            if not isinstance(text, str):
                raise ValueError("question, cues and needs must be strings")
            key = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
            if key:
                unique.setdefault(key, text)
        return list(unique)

    def _lexical(self, text: str):
        if text in self._query_cache:
            return self._query_cache[text]
        terms = tuple(_terms(text))
        episode_scores, source_scores = self._episodes.score(terms), self._sources.score(terms)
        episode_scale = max(episode_scores.values(), default=1.0) or 1.0
        source_scale = max(source_scores.values(), default=1.0) or 1.0
        combined, best_by_source = {}, {}
        for eid in self.episode_ids:
            sid = self.source_by_episode[eid]
            score = episode_scores.get(eid, 0.0) / episode_scale + 0.35 * source_scores.get(sid, 0.0) / source_scale
            if score > 0:
                combined[eid] = score
                best_by_source[sid] = max(best_by_source.get(sid, 0.0), score)
        source_ranks = {sid: rank for rank, sid in enumerate(
            sorted(best_by_source, key=lambda sid: (-best_by_source[sid], sid)), 1)}
        votes = {eid: score / best_by_source[self.source_by_episode[eid]]
                 / (60 + source_ranks[self.source_by_episode[eid]]) for eid, score in combined.items()}
        value = votes, episode_scores, source_scores
        # Keep construction cached across waves without retaining an unlimited
        # history when a service processes many unrelated user questions.
        if len(self._query_cache) >= 32:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[text] = value
        return value

    def _rank_votes(self, order: Sequence[int]) -> dict[int, float]:
        unique = dict.fromkeys(eid for eid in order if type(eid) is int and eid in self.source_by_episode)
        return {eid: 1.0 / (60 + rank) for rank, eid in enumerate(unique, 1)}

    def rank_details(self, query: str, *, cues: Sequence[str] = (), needs: Sequence[str] = (),
                     dense_order: Sequence[int] = (), graph_order: Sequence[int] = ()) -> CandidateRanking:
        prompts = self._prompts(query, cues, needs)
        lexical, episode_bm25, source_bm25, matches = defaultdict(float), {}, {}, defaultdict(int)
        for text in prompts:
            votes, episode_scores, source_scores = self._lexical(text)
            for eid, vote in votes.items():
                lexical[eid] += vote / len(prompts)
                matches[eid] += 1
            for eid, score in episode_scores.items():
                episode_bm25[eid] = max(episode_bm25.get(eid, 0.0), score)
            for sid, score in source_scores.items():
                source_bm25[sid] = max(source_bm25.get(sid, 0.0), score)
        dense, graph = self._rank_votes(dense_order), self._rank_votes(graph_order)
        scores = {}
        for eid in self.episode_ids:
            sid = self.source_by_episode[eid]
            lex, dv, gv = lexical.get(eid, 0.0), dense.get(eid, 0.0), graph.get(eid, 0.0)
            scores[eid] = CandidateScore(sid, episode_bm25.get(eid, 0.0), source_bm25.get(sid, 0.0),
                                         lex, dv, gv, lex + dv + gv, matches.get(eid, 0))
        ranked = sorted(self.episode_ids, key=lambda eid: (-scores[eid].fused_score, eid))
        representatives, seen = [], set()
        for eid in ranked:
            sid = self.source_by_episode[eid]
            if sid not in seen:
                representatives.append(eid)
                seen.add(sid)
        return CandidateRanking(ranked, representatives, scores)

    def rank(self, query: str, *, cues: Sequence[str] = (), needs: Sequence[str] = (),
             dense_order: Sequence[int] = (), graph_order: Sequence[int] = ()) -> list[int]:
        return self.rank_details(query, cues=cues, needs=needs, dense_order=dense_order,
                                 graph_order=graph_order).episode_ids
