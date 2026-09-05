from __future__ import annotations

from collections import Counter
import json
import math
import re
import unicodedata


_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_BRIDGE_QUERY_RE = re.compile(r"问题[“\"](?P<value>.*?)[”\"]")
_BRIDGE_SLOT_RE = re.compile(r"槽[“\"](?P<value>.*?)[”\"]")


def _row_value(row, key: str, default: str = ""):
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _participant_names(row) -> list[str]:
    """Read endpoint labels without coupling cue indexing to Episode objects."""

    result: list[str] = []
    seen: set[str] = set()
    for key in ("from_participants_json", "to_participants_json"):
        raw = _row_value(row, key, "")
        if not raw:
            continue
        try:
            values = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(values, list):
            continue
        for value in values:
            name = str(value).strip() if isinstance(value, str) else ""
            folded = name.casefold()
            if not name or folded in seen:
                continue
            seen.add(folded)
            result.append(name)
            if len(result) >= 10:
                return result
    return result


def _evidence_bridge_anchor_text(relation_text: str) -> str:
    query = _BRIDGE_QUERY_RE.search(relation_text)
    slots = [
        match.group("value") for match in _BRIDGE_SLOT_RE.finditer(relation_text)
    ]
    return " ".join(
        value.strip()
        for value in [query.group("value") if query else "", *slots]
        if value.strip()
    )


def association_cue_text(row) -> str:
    """Return the compact lookup representation of an audited relation.

    A normal learned relation is its own semantic cue. An evidence bridge also
    contains long endpoint observations for auditing; indexing those passages
    dilutes the successful query and slot labels that should drive reuse.
    """

    relation_text = str(_row_value(row, "relation_text", "") or "")
    relation_key = str(_row_value(row, "relation_key", "") or "")
    if relation_key.casefold() != "evidence_bridge":
        return relation_text
    anchor_text = _evidence_bridge_anchor_text(relation_text)
    compact = " ".join(
        value.strip()
        for value in [
            anchor_text,
            *_participant_names(row),
        ]
        if value.strip()
    )
    return compact or relation_text


def cue_features(value: str) -> set[str]:
    """Return language-neutral lexical features for a short relation claim.

    CJK characters preserve paraphrase recall where word boundaries are
    unavailable, while adjacent bigrams prevent a handful of common characters
    from activating an unrelated learned edge. Latin text uses words, avoiding
    noisy single-letter matches. Corpus IDF supplies specificity; no story
    vocabulary is encoded here.
    """

    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    result = {f"w:{word}" for word in _WORD_RE.findall(normalized)}
    for run in _CJK_RUN_RE.findall(normalized):
        result.update(f"c:{char}" for char in run)
        result.update(
            f"n2:{run[index:index + 2]}"
            for index in range(max(0, len(run) - 1))
        )
    return result


class LexicalAssociationCueIndex:
    """Tiny in-RAM IDF index over audited Association relation text."""

    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.features = [cue_features(association_cue_text(row)) for row in self.rows]
        self.anchor_features: list[set[str]] = []
        self.participant_names: list[list[str]] = []
        for row in self.rows:
            relation_text = str(_row_value(row, "relation_text", "") or "")
            if str(_row_value(row, "relation_key", "") or "").casefold() == (
                "evidence_bridge"
            ):
                anchor_text = _evidence_bridge_anchor_text(relation_text)
                names = _participant_names(row)
            else:
                anchor_text = relation_text
                names = []
            self.anchor_features.append(cue_features(anchor_text))
            self.participant_names.append(names)
        document_frequency: Counter[str] = Counter()
        for features in self.features:
            document_frequency.update(features)
        count = len(self.rows)
        self.idf = {
            feature: math.log((count + 1.0) / (frequency + 0.5))
            for feature, frequency in document_frequency.items()
        }
        self.unseen_idf = math.log((count + 1.0) / 0.5) if count else 0.0

    def search(self, text: str, limit: int = 4) -> list[tuple[dict, float]]:
        query = cue_features(text)
        if not query or limit <= 0:
            return []
        ranked: list[tuple[dict, float]] = []
        normalized_query = unicodedata.normalize("NFKC", text or "").casefold()
        for row, features, anchor_features, participant_names in zip(
            self.rows,
            self.features,
            self.anchor_features,
            self.participant_names,
            strict=True,
        ):
            document_weight = sum(
                self.idf.get(feature, 0.0) for feature in features
            )
            query_weight = sum(
                self.idf.get(feature, self.unseen_idf) for feature in query
            )
            if document_weight <= 0.0 or query_weight <= 0.0:
                continue
            overlap = sum(
                self.idf.get(feature, 0.0)
                for feature in features.intersection(query)
            )
            # Document coverage keeps entity-only questions from activating a
            # large evidence capsule. Query coverage prevents useful endpoint
            # labels from being drowned by the rest of the compact cue. Their
            # geometric mean is symmetric and remains fail-closed when either
            # direction has little support.
            document_coverage = overlap / document_weight
            query_coverage = overlap / query_weight
            score = math.sqrt(document_coverage * query_coverage)

            # Evidence bridges have two distinct fields. A named endpoint plus
            # a matching event/relation anchor is a strong reusable route, but
            # an endpoint name alone must not turn unrelated character chat
            # into a cache hit.
            matched_name_features: set[str] = set()
            for name in participant_names:
                normalized_name = unicodedata.normalize(
                    "NFKC", name
                ).casefold()
                if normalized_name and normalized_name in normalized_query:
                    matched_name_features.update(cue_features(name))
            anchor_query = query.difference(matched_name_features)
            if matched_name_features and anchor_query and anchor_features:
                anchor_weight = sum(
                    self.idf.get(feature, 0.0)
                    for feature in anchor_features
                )
                anchor_query_weight = sum(
                    self.idf.get(feature, self.unseen_idf)
                    for feature in anchor_query
                )
                anchor_overlap = sum(
                    self.idf.get(feature, 0.0)
                    for feature in anchor_features.intersection(anchor_query)
                )
                if anchor_weight > 0.0 and anchor_query_weight > 0.0:
                    anchor_score = math.sqrt(
                        (anchor_overlap / anchor_weight)
                        * (anchor_overlap / anchor_query_weight)
                    )
                    score = max(score, math.sqrt(anchor_score))
            ranked.append((row, score))
        ranked.sort(key=lambda item: (item[1], -int(item[0]["id"])), reverse=True)
        return ranked[:limit]
