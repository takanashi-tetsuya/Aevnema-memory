from __future__ import annotations

import re
import unicodedata

from memory_demo.database import Database


_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_WORD_RE = re.compile(r"[a-z0-9_]+")
_QUOTED_RE = re.compile(r"[\"'“”‘’「」『』《》【】]([^\"'“”‘’「」『』《》【】]+)[\"'“”‘’「」『』《》【】]")


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _quote_fts(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _search_units(value: str, limit: int = 96) -> list[str]:
    """Create language-neutral units understood by SQLite's trigram tokenizer."""
    normalized = _normalize(value)
    units: list[str] = []
    for word in _WORD_RE.findall(normalized):
        if len(word) >= 3:
            units.append(word)
    for run in _CJK_RUN_RE.findall(normalized):
        if len(run) < 3:
            continue
        units.extend(run[index : index + 3] for index in range(len(run) - 2))
    return list(dict.fromkeys(units))[:limit]


def _bigram_units(value: str, limit: int = 96) -> list[str]:
    normalized = _normalize(value)
    units = [word for word in _WORD_RE.findall(normalized) if len(word) >= 2]
    for run in _CJK_RUN_RE.findall(normalized):
        if len(run) == 1:
            units.append(run)
        else:
            units.extend(run[index : index + 2] for index in range(len(run) - 1))
    return list(dict.fromkeys(units))[:limit]


def _anchor_units(value: str, limit: int = 24) -> list[str]:
    anchors: list[str] = []
    for match in _QUOTED_RE.finditer(_normalize(value)):
        candidate = match.group(1).strip()
        if len(candidate.replace(" ", "")) >= 3:
            anchors.append(candidate)
    return list(dict.fromkeys(anchors))[:limit]


class SQLiteSparseIndex:
    """Persistent multilingual lexical index backed by SQLite FTS5.

    It exposes the same small ``search`` surface as the dense index while the
    base Episode/Source tables remain the source of truth.  Exact quoted spans
    and broad character/word overlap are fused with reciprocal-rank fusion.
    """

    _TABLES = {
        "episode": (
            "episode_fts",
            "bm25(episode_fts, 1.0, 0.15)",
            "episode_bigram_fts",
            "bm25(episode_bigram_fts)",
        ),
        "source": (
            "source_fts",
            "bm25(source_fts)",
            "source_bigram_fts",
            "bm25(source_bigram_fts)",
        ),
    }

    def __init__(self, db: Database, kind: str):
        if kind not in self._TABLES:
            raise ValueError(f"unsupported sparse index kind: {kind}")
        self.db = db
        self.kind = kind
        (
            self.table,
            self.rank_expression,
            self.bigram_table,
            self.bigram_rank_expression,
        ) = self._TABLES[kind]

    def _run(
        self,
        table: str,
        rank_expression: str,
        expression: str,
        top_k: int,
    ) -> list[tuple[int, float]]:
        if not expression or top_k <= 0:
            return []
        sql = (
            f"SELECT rowid AS id, {rank_expression} AS rank "
            f"FROM {table} WHERE {table} MATCH ? "
            "ORDER BY rank LIMIT ?"
        )
        with self.db.connection() as connection:
            rows = connection.execute(sql, (expression, int(top_k))).fetchall()
        return [(int(row["id"]), -float(row["rank"])) for row in rows]

    def search(self, text: str, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        broad_units = _search_units(text)
        bigram_units = _bigram_units(text)
        anchor_units = _anchor_units(text)
        searches: list[tuple[float, str, str, str]] = []
        if anchor_units:
            searches.append(
                (
                    1.40,
                    self.table,
                    self.rank_expression,
                    " OR ".join(_quote_fts(value) for value in anchor_units),
                )
            )
        if broad_units:
            searches.append(
                (
                    1.00,
                    self.table,
                    self.rank_expression,
                    " OR ".join(_quote_fts(value) for value in broad_units),
                )
            )
        if bigram_units:
            searches.append(
                (
                    1.10,
                    self.bigram_table,
                    self.bigram_rank_expression,
                    " OR ".join(_quote_fts(value) for value in bigram_units),
                )
            )
        fused: dict[int, float] = {}
        best_raw: dict[int, float] = {}
        for weight, table, rank_expression, expression in searches:
            for rank, (node_id, raw_score) in enumerate(
                self._run(
                    table,
                    rank_expression,
                    expression,
                    max(top_k, top_k * 2),
                ),
                start=1,
            ):
                # Trigram and bigram are alternate tokenizations of the same
                # lexical evidence, not independent votes.  Taking the best
                # channel avoids rewarding generic text merely for appearing
                # in both tokenizations.
                fused[node_id] = max(
                    fused.get(node_id, 0.0), weight / (60 + rank)
                )
                best_raw[node_id] = max(best_raw.get(node_id, 0.0), raw_score)
        ranked = sorted(
            fused,
            key=lambda node_id: (fused[node_id], best_raw[node_id]),
            reverse=True,
        )[:top_k]
        return [(node_id, fused[node_id]) for node_id in ranked]

    @property
    def count(self) -> int:
        with self.db.connection() as connection:
            return int(
                connection.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]
            )
