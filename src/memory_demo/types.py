from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Literal

NodeType = Literal["episode", "concept"]
EvidenceOrigin = Literal["source", "importer", "system", "mixed", "unknown"]
EpistemicStatus = Literal[
    "observed", "asserted", "reported", "speculative", "mixed", "unknown"
]

EVIDENCE_ORIGINS = {"source", "importer", "system", "mixed", "unknown"}
EPISTEMIC_STATUSES = {
    "observed", "asserted", "reported", "speculative", "mixed", "unknown"
}


@dataclass(frozen=True, slots=True)
class QueryVector:
    """A normalized vector produced for one request-level query.

    ``vector`` deliberately remains an in-memory value and is never serialized
    into prompts or ordinary event logs.  ``query_id`` and ``text_hash`` are
    stable diagnostics that let replay code trace which atomic query supplied
    a successful retrieval without retaining the embedding itself.
    """

    query_id: str
    text_hash: str
    role: Literal["whole", "atomic", "followup"]
    vector: Any
    slot_id: str = ""
    # Kept outside raw_result by callers unless explicitly needed for a local
    # learning receipt.  This lets the coordinator retain the exact source
    # text for diagnostics without requiring another embedding request.
    text: str = ""


@dataclass(frozen=True, slots=True)
class QueryVectorBundle:
    """All vectors needed by one retrieval request."""

    model_id: str
    dimension: int
    whole: Any
    queries: tuple[QueryVector, ...] = ()

    @property
    def atomic(self) -> tuple[QueryVector, ...]:
        return tuple(item for item in self.queries if item.role == "atomic")

    @property
    def followups(self) -> tuple[QueryVector, ...]:
        return tuple(item for item in self.queries if item.role == "followup")


@dataclass(frozen=True, slots=True)
class EvidenceSlot:
    """One request-scoped obligation that selected evidence must cover.

    Slots are execution metadata, not durable knowledge.  Keeping the query
    identifier alongside the natural-language obligation lets a caller reuse
    the already-batched vector for a missing-slot repair without embedding a
    second copy of the question.
    """

    slot_id: str
    question: str = ""
    required: bool = True
    query_id: str = ""
    subject_terms: tuple[str, ...] = ()
    relation_hint: str = ""
    temporal_hint: str = ""
    epistemic_hint: str = ""

    def __post_init__(self) -> None:
        if not str(self.slot_id).strip():
            raise ValueError("evidence slot_id is required")


@dataclass(frozen=True, slots=True)
class SlotCandidate:
    """A local Episode-to-slot support record used by set-cover selection."""

    episode_id: int
    slot_ids: frozenset[str] = frozenset()
    source_lane: str = "base"
    direct_score: float = 0.0
    specificity_score: float = 0.0
    redundancy_group: str = ""
    contextual_edge_id: int | None = None
    contextual_score: float = 0.0
    source_quality: float = 0.0


@dataclass(frozen=True, slots=True)
class ContextualSlotHit:
    """A double-key recall proposal tied to one unresolved evidence slot."""

    association_id: int
    anchor_episode_id: int
    target_episode_id: int
    matched_slot_id: str = ""
    matched_query_id: str = ""
    context_similarity: float = 0.0
    need_similarity: float = 0.0
    anchor_activation: float = 0.0
    utility_weight: float = 0.0
    target_support_score: float = 0.0
    lifecycle_state: str = "probation"
    total_score: float = 0.0

    @property
    def score(self) -> float:
        """Compatibility name for the v1 trace shape."""

        return self.total_score


@dataclass(frozen=True, slots=True)
class CuePrototype:
    id: int
    domain: str
    cue_kind: Literal["context", "need"]
    model_id: str
    dimension: int
    dtype: str
    vector: Any
    text_hash: str
    display_text: str = ""
    source_request_hash: str = ""
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class ContextualRecallCandidate:
    """A retrieval-only edge candidate; it contains no factual claim."""

    anchor_type: NodeType
    anchor_id: int
    target_episode_id: int
    context_query_id: str
    need_query_id: str
    slot_id: str = ""
    source_request_hash: str = ""
    reason: str = "recovered_missing_evidence"


# v1 callers exposed this name in traces.  The richer v2 object remains fully
# compatible for attribute reads while making the matched evidence slot and
# anchor activation explicit.
AssociationCueHit = ContextualSlotHit


@dataclass(frozen=True, slots=True)
class ContextualUtilityObservation:
    association_id: int
    query_hash: str
    outcome: Literal["sufficient", "necessary", "redundant", "no_op", "harmful"]
    delta_slots: int = 0
    treatment_episode_ids: tuple[int, ...] = ()
    masked_episode_ids: tuple[int, ...] = ()
    attribution: Literal["batch", "single_edge", "leave_one_out"] = "batch"


@dataclass(frozen=True, slots=True)
class PlasticityEvent:
    """Local, post-answer event used to update contextual edge utility."""

    request_hash: str
    domain: str
    candidates: tuple[ContextualRecallCandidate, ...] = ()
    observations: tuple[ContextualUtilityObservation, ...] = ()
    answer_guard_passed: bool = True

_CONCEPT_ALIAS_CHAIN_SEPARATOR = re.compile(r"\s+[/／]\s+")
_CONCEPT_META_COMMENTARY_MARKERS = (
    "此处修正",
    "应提取",
    "未提及",
    "候选条目",
    "更核心",
)


def _preferred_concept_name(parts: list[str]) -> str:
    """Choose one display name while retaining the other parts as aliases."""

    def score(value: str) -> tuple[int, int, int]:
        has_han = any("\u3400" <= char <= "\u9fff" for char in value)
        has_kana = any("\u3040" <= char <= "\u30ff" for char in value)
        has_hangul = any("\uac00" <= char <= "\ud7af" for char in value)
        has_latin = any("a" <= char.casefold() <= "z" for char in value)
        # A Han-only label is normally the Chinese display name. Japanese,
        # Korean and Latin spellings remain searchable aliases.
        return (
            3 if has_han and not has_kana and not has_hangul else
            2 if has_han else
            1 if has_latin else
            0,
            -int(has_kana or has_hangul),
            -len(value),
        )

    return max(parts, key=score)


def normalize_evidence_origin(value: Any, default: str = "source") -> str:
    normalized = str(value or "").strip().casefold()
    return normalized if normalized in EVIDENCE_ORIGINS else default


def normalize_epistemic_status(value: Any, default: str = "asserted") -> str:
    normalized = str(value or "").strip().casefold()
    return normalized if normalized in EPISTEMIC_STATUSES else default


@dataclass(slots=True)
class NormalizedBlock:
    record_index: int
    speaker_raw: str = ""
    languages: dict[str, str] = field(default_factory=dict)
    boundary_score: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        return any(value.strip() for value in self.languages.values())


@dataclass(slots=True)
class SourceSegment:
    source_key: str
    segment_index: int
    raw_text: str
    first_record: int
    last_record: int


@dataclass(slots=True)
class ParagraphDraft:
    """A deterministic, record-aligned sub-chunk of one Source."""

    paragraph_index: int
    text: str


@dataclass(slots=True)
class EpisodeDraft:
    text: str
    participants: list[str] = field(default_factory=list)
    event_type: str = ""
    location_text: str = ""
    story_time_text: str = ""
    timeline_scope: str = ""
    confidence: float = 0.5
    evidence_origin: EvidenceOrigin = "source"
    epistemic_status: EpistemicStatus = "asserted"
    generation: int = 0
    epistemic_note: str = ""
    # Exact import-time proof. Source remains authoritative; these fields are
    # also persisted so acceptance audits can reproduce the extraction claim.
    evidence_quotes: list[str] = field(default_factory=list)
    evidence_spans: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeDraft":
        text = str(value.get("text", "")).strip()
        if not text:
            raise ValueError("episode text is required")
        participants = value.get("participants", [])
        if isinstance(participants, str):
            try:
                decoded = json.loads(participants)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list):
                participants = decoded
        if not isinstance(participants, list):
            raise ValueError("participants must be a list")
        evidence_quotes = value.get("evidence_quotes", [])
        if not isinstance(evidence_quotes, list) or not all(
            isinstance(item, str) for item in evidence_quotes
        ):
            raise ValueError("evidence_quotes must be a list of strings")
        evidence_spans = value.get("evidence_spans", [])
        if not isinstance(evidence_spans, list):
            raise ValueError("evidence_spans must be a list of [start, end]")
        normalized_spans: list[Any] = []
        singleton_spans: list[bool] = []
        for span in evidence_spans:
            if isinstance(span, (list, tuple)):
                normalized_span: list[Any] = []
                for item in span:
                    matched = (
                        re.fullmatch(r"[Ll]0*(\d+)", item.strip())
                        if isinstance(item, str)
                        else None
                    )
                    normalized_span.append(
                        int(matched.group(1)) if matched else item
                    )
                was_singleton = len(normalized_span) == 1
                if was_singleton:
                    # Some JSON models serialize a one-line range as [line]
                    # instead of [line, line].  It is losslessly equivalent.
                    normalized_span.append(normalized_span[0])
                normalized_spans.append(normalized_span)
                singleton_spans.append(was_singleton)
            else:
                normalized_spans.append(span)
                singleton_spans.append(False)
        evidence_spans = normalized_spans
        # A single range is occasionally emitted as [start, end] rather than
        # [[start, end]].  The meaning is unambiguous, so normalize this shape
        # without asking the model to regenerate an otherwise valid batch.
        if (
            len(evidence_spans) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in evidence_spans
            )
        ):
            evidence_spans = [evidence_spans]
        parsed_spans: list[tuple[int, int]] = []
        for span in evidence_spans:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in span
                )
            ):
                raise ValueError(
                    "evidence_spans must be a list of integer [start, end] pairs"
                )
            parsed_spans.append((int(span[0]), int(span[1])))
        # Models also commonly emit every cited line as its own singleton.
        # Compact an all-singleton representation so a contiguous quotation is
        # judged by its actual ranges, not rejected for JSON verbosity.  Keep
        # explicit [n, n] ranges unchanged: those may intentionally describe
        # separate snippets and remain subject to the range-count guard.
        if parsed_spans and len(singleton_spans) == len(parsed_spans) and all(
            singleton_spans
        ):
            compacted: list[tuple[int, int]] = []
            for start, end in sorted(set(parsed_spans)):
                if compacted and start <= compacted[-1][1] + 1:
                    compacted[-1] = (
                        compacted[-1][0],
                        max(compacted[-1][1], end),
                    )
                else:
                    compacted.append((start, end))
            parsed_spans = compacted
        confidence = float(value.get("confidence", 0.5))
        evidence_origin = normalize_evidence_origin(value.get("evidence_origin"))
        epistemic_status = normalize_epistemic_status(
            value.get("epistemic_status")
        )
        generation = max(0, int(value.get("generation", 0) or 0))
        # An importer/system-authored speculation is already one inferential
        # step away from the document's direct evidence.  A character's guess
        # in the source remains generation 0 because "the character guessed X"
        # itself is directly recorded; epistemic_status still prevents X from
        # being promoted to fact.
        if (
            evidence_origin in {"importer", "system", "mixed"}
            and epistemic_status in {"speculative", "mixed"}
        ):
            generation = max(1, generation)
        return cls(
            text=text,
            participants=[str(item).strip() for item in participants if str(item).strip()],
            event_type=str(value.get("event_type", "")).strip(),
            location_text=str(value.get("location_text", "")).strip(),
            story_time_text=str(value.get("story_time_text", "")).strip(),
            timeline_scope=str(value.get("timeline_scope", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            evidence_origin=evidence_origin,  # type: ignore[arg-type]
            epistemic_status=epistemic_status,  # type: ignore[arg-type]
            generation=generation,
            epistemic_note=str(value.get("epistemic_note", "")).strip(),
            evidence_quotes=[
                item.strip() for item in evidence_quotes if item.strip()
            ],
            evidence_spans=parsed_spans,
        )


@dataclass(slots=True)
class ConceptDraft:
    canonical_name: str
    description: str
    embedding_text: str
    aliases: list[tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.5

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConceptDraft":
        raw_name = str(value.get("canonical_name", "")).strip()
        if not raw_name:
            raise ValueError("canonical_name is required")
        name_parts = [
            part.strip()
            for part in _CONCEPT_ALIAS_CHAIN_SEPARATOR.split(raw_name)
            if part.strip()
        ]
        name = _preferred_concept_name(name_parts)
        description = str(value.get("description", "")).strip() or name
        embedding_text = str(value.get("embedding_text", "")).strip() or description
        combined_text = f"{description}\n{embedding_text}".casefold()
        marker = next(
            (
                item
                for item in _CONCEPT_META_COMMENTARY_MARKERS
                if item.casefold() in combined_text
            ),
            None,
        )
        if marker is not None:
            raise ValueError(
                f"concept contains model meta-commentary marker: {marker}"
            )
        aliases: list[tuple[str, str]] = [
            (part, "unknown") for part in name_parts if part != name
        ]
        for item in value.get("aliases", []):
            if isinstance(item, str):
                aliases.append((item.strip(), "unknown"))
            elif isinstance(item, dict):
                alias = str(item.get("alias", "")).strip()
                if alias:
                    aliases.append((alias, str(item.get("language", "unknown"))))
            elif isinstance(item, (list, tuple)) and item:
                alias = str(item[0]).strip()
                if alias:
                    language = str(item[1]) if len(item) > 1 else "unknown"
                    aliases.append((alias, language))
        deduplicated_aliases: list[tuple[str, str]] = []
        seen_aliases = {name.casefold()}
        for alias, language in aliases:
            normalized = alias.casefold()
            if not alias or normalized in seen_aliases:
                continue
            seen_aliases.add(normalized)
            deduplicated_aliases.append((alias, language))
        confidence = float(value.get("confidence", 0.5))
        return cls(
            canonical_name=name,
            description=description,
            embedding_text=embedding_text,
            aliases=deduplicated_aliases,
            confidence=max(0.0, min(1.0, confidence)),
        )


@dataclass(slots=True)
class AssociationDraft:
    from_type: NodeType
    from_id: int
    to_type: NodeType
    to_id: int
    relation_type: str
    relation_key: str
    relation_text: str
    polarity: int = 1
    weight: float = 0.5
    confidence: float = 0.5
    generation: int = 0
    claim_level: Literal[
        "direct_fact", "supported_inference", "historical_context", "retrieval_only"
    ] = "direct_fact"
    audit_status: Literal["not_required", "dual_accepted"] = "not_required"
    evidence_json: str = "[]"
    audit_json: str = "[]"
    created_reason: str = ""


@dataclass(slots=True)
class SearchHit:
    node_type: NodeType
    node_id: int
    score: float
    text: str = ""


@dataclass(slots=True)
class QueryIntent:
    language: str = "zh"
    target_entities: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    requested_relation: str = ""
    temporal_constraint: str = ""
    causal_constraint: str = ""
    answer_shape: str = "complete_evidence_answer"
    uncertainty_required: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QueryIntent":
        def optional_constraint(key: str) -> str:
            raw = value.get(key, "")
            if raw is None:
                return ""
            text = str(raw).strip()
            normalized = text.casefold().replace(" ", "")
            if normalized in {
                "",
                "none",
                "null",
                "n/a",
                "na",
                "无",
                "没有",
                "不适用",
                "无明确",
                "无约束",
                "无明确约束",
                "无明确因果约束",
                "无明确时间约束",
                "无明确关系约束",
            }:
                return ""
            return text

        entities = value.get("target_entities", [])
        if not isinstance(entities, list):
            entities = []
        raw_queries = value.get("search_queries", [])
        if not isinstance(raw_queries, list):
            raw_queries = []
        search_queries: list[str] = []
        for item in raw_queries:
            query = str(item).strip()
            if query and query not in search_queries:
                search_queries.append(query)
        return cls(
            language=str(value.get("language", "zh")),
            target_entities=[str(item) for item in entities],
            search_queries=search_queries[:12],
            requested_relation=optional_constraint("requested_relation"),
            temporal_constraint=optional_constraint("temporal_constraint"),
            causal_constraint=optional_constraint("causal_constraint"),
            answer_shape=str(value.get("answer_shape", "complete_evidence_answer")),
            uncertainty_required=bool(value.get("uncertainty_required", True)),
        )
