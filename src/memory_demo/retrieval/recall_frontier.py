"""Question-local fair Source selection; candidates never certify evidence."""
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SourceSelection:
    need_index: int
    episode_id: int
    source_id: int
    need_rank: int


def round_robin_sources(rankings: Sequence[Sequence[int]], source_by_episode: Mapping[int, int],
                        *, limit: int, excluded_sources=(), start_need: int = 0) -> list[SourceSelection]:
    """Give each need one new Source per turn, deduplicating across all needs.

    The caller determines active needs and candidate rankings. No Source IDs,
    vocabulary, graph weights, evidence roots or semantic decisions are added.
    ``need_rank`` retains the candidate's original one-based Episode rank.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError('limit must be a nonnegative integer')
    if type(start_need) is not int or start_need < 0:
        raise ValueError('start_need must be a nonnegative integer')
    if not rankings or limit == 0:
        return []
    cursors = [0] * len(rankings)
    seen = set(excluded_sources)
    selected = []
    order = [(start_need + i) % len(rankings) for i in range(len(rankings))]
    while len(selected) < limit:
        progressed = False
        for need in order:
            ranking = rankings[need]
            while cursors[need] < len(ranking):
                eid = ranking[cursors[need]]
                cursors[need] += 1
                sid = source_by_episode.get(eid)
                if sid is None or sid in seen:
                    continue
                selected.append(SourceSelection(need, eid, sid, cursors[need]))
                seen.add(sid)
                progressed = True
                break
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected
