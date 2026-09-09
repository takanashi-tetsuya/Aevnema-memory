"""Deterministic transformations of model-produced retrieval intent."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Literal, Mapping, Sequence

from memory_demo.types import EvidenceSlot, QueryIntent


RequestMode = Literal["factual", "conversational", "exploratory"]
RequirementStatus = Literal["resolved", "unknown", "not_applicable"]
PlannerOrigin = Literal[
    "explicit",
    "cloud",
    "local",
    "persisted",
    "not_applicable",
]

_REQUEST_MODES = frozenset({"factual", "conversational", "exploratory"})
_PLANNER_ORIGINS = frozenset(
    {"explicit", "cloud", "local", "persisted", "not_applicable"}
)
_SLOT_ORIGINS = frozenset({"user_explicit", "planner", "reused_template"})
_SUPPORT_MODES = frozenset({"alternative", "joint"})
_STRUCTURAL_SLOT_PREFIXES = (
    "__constraint_slot__ ",
    "__answer_slot__ ",
)
_SAFE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CONVERSATIONAL_ANSWER_SHAPES = frozenset(
    {"conversation", "conversational", "chat", "chitchat", "greeting"}
)
_EXPLORATORY_ANSWER_SHAPES = frozenset(
    {"exploratory", "explore", "brainstorm", "discovery"}
)
_SOURCE_FACT_ID_RE = re.compile(r"source-fact:sha256:[0-9a-f]{64}\Z")
# A residual repair is a single bounded local pass, not a second whole-query
# pipeline.  Keep the request fan-out finite even if a malformed/old selector
# reports a large missing set; the dropped IDs are retained in the trace plan.
MAX_LOCAL_RESIDUAL_REPAIR_SLOTS = 4


def _normalized(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _opaque_reference(value: object) -> str:
    """Return a trace-safe reference without carrying user/source text."""

    return "sha256:" + sha256(str(value or "").encode("utf-8")).hexdigest()


def _safe_identifier(value: object, *, field_name: str) -> str:
    identifier = str(value or "").strip()
    if not _SAFE_IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"{field_name} must be a safe opaque identifier")
    return identifier


def _stable_identifier(prefix: str, value: object) -> str:
    return f"{prefix}:{sha256(str(value).encode('utf-8')).hexdigest()[:20]}"


def _as_terms(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence):
        return ()
    return tuple(
        dict.fromkeys(
            str(item).strip() for item in value if str(item).strip()
        )
    )


def _metadata_origin(planner_origin: PlannerOrigin) -> str:
    return "user_explicit" if planner_origin == "explicit" else "planner"


def _derived_hints(question: str) -> tuple[str, str]:
    """Keep structural modality/negation flags without inventing facts."""

    normalized = str(question).casefold()
    negation = (
        "explicit_negation"
        if any(token in normalized for token in ("不", "未", "没有", "并非", "not ", "never"))
        else ""
    )
    modality = (
        "modal_or_conditional"
        if any(
            token in normalized
            for token in ("是否", "可能", "能否", "如果", "应当", "must", "may", "might")
        )
        else ""
    )
    return negation, modality


@dataclass(frozen=True, slots=True)
class RequirementResolution:
    """Request-frozen production evidence requirements.

    This object deliberately has no candidate-pool field.  Candidate support
    is attached only later, after retrieval, so an empty retrieval can never
    shrink the request denominator or turn an unknown factual request into a
    non-requirement.
    """

    request_mode: RequestMode
    status: RequirementStatus
    requirements: tuple[EvidenceSlot, ...]
    planner_origin: PlannerOrigin
    planner_call_ids: tuple[str, ...] = ()

    @property
    def unresolved_slot_ids(self) -> tuple[str, ...]:
        return tuple(
            item.slot_id for item in self.requirements if item.required
        )

    def as_result_dict(self) -> dict[str, Any]:
        """Return local result metadata, retaining normal query diagnostics."""

        return {
            "request_mode": self.request_mode,
            "status": self.status,
            "planner_origin": self.planner_origin,
            "planner_call_ids": list(self.planner_call_ids),
            "requirements": [
                {
                    "slot_id": item.slot_id,
                    "question": item.question,
                    "required": item.required,
                    "query_id": item.query_id,
                    "query_refs": list(item.query_refs),
                    "origin": item.origin,
                    "support_mode": item.support_mode,
                    "clause_ids": list(item.clause_ids),
                    "subject_terms": list(item.subject_terms),
                    "object_terms": list(item.object_terms),
                    "relation_hint": item.relation_hint,
                    "temporal_hint": item.temporal_hint,
                    "negation_hint": item.negation_hint,
                    "modality_hint": item.modality_hint,
                    "epistemic_hint": item.epistemic_hint,
                }
                for item in self.requirements
            ],
            "unresolved_slot_ids": list(self.unresolved_slot_ids),
        }

    def as_trace_payload(self) -> dict[str, Any]:
        """Return the exact redacted payload shape for a v3 trace stage.

        The trace contract permits requirement metadata but not raw request or
        source text.  Every potentially natural-language field is therefore
        represented by an opaque content hash here; callers wanting prose keep
        it in a ``full_local`` artifact outside trace JSONL.
        """

        def opaque_if_present(value: object) -> str:
            return _opaque_reference(value) if str(value or "").strip() else ""

        return {
            "requirements": [
                {
                    "slot_id": item.slot_id,
                    "question": _opaque_reference(item.question),
                    "required": bool(item.required),
                    "query_refs": [
                        _opaque_reference(value) for value in item.query_refs
                    ],
                    "origin": item.origin,
                    "support_mode": item.support_mode,
                    "clause_ids": [
                        _opaque_reference(value) for value in item.clause_ids
                    ],
                    "subject_terms": [
                        _opaque_reference(value) for value in item.subject_terms
                    ],
                    "object_terms": [
                        _opaque_reference(value) for value in item.object_terms
                    ],
                    "relation_hint": opaque_if_present(item.relation_hint),
                    "temporal_hint": opaque_if_present(item.temporal_hint),
                    "epistemic_hint": opaque_if_present(item.epistemic_hint),
                }
                for item in self.requirements
            ],
            "unresolved_slot_ids": list(self.unresolved_slot_ids),
            "uncertain_slot_ids": [],
            "planner_origin": self.planner_origin,
            "planner_call_ids": [
                _opaque_reference(value) for value in self.planner_call_ids
            ],
        }


@dataclass(frozen=True, slots=True)
class ResidualRepairPlan:
    """One bounded, local-only attempt to close already-observed gaps.

    This is intentionally *not* another planner.  ``slots`` can contain only
    required slots which the current selector has already reported as missing;
    it never expands a request back to its whole question or to every planned
    query.  The engine later requires an exact request-vector binding for each
    slot before it reads a local retrieval index.
    """

    slots: tuple[EvidenceSlot, ...]
    reason: str
    invalid_missing_slot_ids: tuple[str, ...] = ()
    dropped_slot_ids: tuple[str, ...] = ()
    round_limit: int = 1
    query_limit: int = MAX_LOCAL_RESIDUAL_REPAIR_SLOTS

    @property
    def slot_ids(self) -> tuple[str, ...]:
        return tuple(item.slot_id for item in self.slots)

    @property
    def enabled(self) -> bool:
        return bool(self.slots) and self.reason == "dynamic_missing_slots"

    def as_trace_payload(self) -> dict[str, object]:
        """Return the redacted plan receipt used by the local repair trace."""

        return {
            "version": "residual-local-repair-plan-v1",
            "mode": "local_only",
            "round_limit": int(self.round_limit),
            "slot_ids": list(self.slot_ids),
            "dropped_slot_ids": list(self.dropped_slot_ids),
            "query_limit": int(self.query_limit),
            "reason": self.reason,
            "invalid_missing_slot_ids": list(self.invalid_missing_slot_ids),
            "whole_query_fallback": False,
            "full_rerank_skipped": True,
        }


@dataclass(frozen=True, slots=True)
class ResidualSourceMappingReceipt:
    """A typed frozen mapping receipt for local residual diagnostics.

    It binds an Episode to one exact logical slot/query identity, its complete
    clause set, and a prior :class:`~memory_demo.types.SourceFactRef` identity.
    Receipt shape plus a current fact-identity comparison are useful debugging
    signals, but they are *not* a semantic-verifier capability.  In the
    current trace-only residual component, this type can never establish slot
    coverage or make an Episode eligible for delivery.
    """

    episode_id: int
    slot_id: str
    query_id: str
    clause_ids: tuple[str, ...]
    source_fact_id: str

    def __post_init__(self) -> None:
        try:
            episode_id = int(self.episode_id)
        except (TypeError, ValueError) as error:
            raise TypeError("residual mapping receipt episode_id must be positive") from error
        if episode_id <= 0:
            raise ValueError("residual mapping receipt episode_id must be positive")
        slot_id = _safe_identifier(self.slot_id, field_name="receipt.slot_id")
        query_id = _safe_identifier(self.query_id, field_name="receipt.query_id")
        raw_clause_ids = self.clause_ids
        if isinstance(raw_clause_ids, str):
            raw_clause_ids = (raw_clause_ids,)
        clause_ids = tuple(
            sorted(
                {
                    _safe_identifier(value, field_name="receipt.clause_id")
                    for value in (raw_clause_ids or ())
                    if str(value).strip()
                }
            )
        )
        if not clause_ids:
            raise ValueError("residual mapping receipt clause_ids are required")
        source_fact_id = str(self.source_fact_id or "").strip()
        if not _SOURCE_FACT_ID_RE.fullmatch(source_fact_id):
            raise ValueError("residual mapping receipt source_fact_id is invalid")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "slot_id", slot_id)
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "clause_ids", clause_ids)
        object.__setattr__(self, "source_fact_id", source_fact_id)

    @property
    def mapping_ref(self) -> str:
        """Derive a stable opaque receipt reference without prose payloads."""

        material = "\x1f".join(
            (
                str(self.episode_id),
                self.slot_id,
                self.query_id,
                *self.clause_ids,
                self.source_fact_id,
            )
        )
        return "residual-source-mapping:sha256:" + sha256(
            material.encode("utf-8")
        ).hexdigest()

    def as_trace_payload(self, *, status: str) -> dict[str, object]:
        return {
            "episode_id": int(self.episode_id),
            "slot_id": self.slot_id,
            "query_id": self.query_id,
            "clause_ids": list(self.clause_ids),
            "source_fact_id": self.source_fact_id,
            "mapping_ref": self.mapping_ref,
            "status": str(status),
        }


def plan_local_residual_repair(
    resolution: RequirementResolution,
    missing_slot_ids: Sequence[str],
    *,
    max_slots: int = MAX_LOCAL_RESIDUAL_REPAIR_SLOTS,
) -> ResidualRepairPlan:
    """Freeze the exact dynamic gaps eligible for one local repair round.

    The caller supplies a coverage receipt from the *current delivered
    evidence*.  An empty receipt means there is nothing to repair; it never
    means "try every requirement again".  Unknown IDs fail closed rather than
    allowing a stale trace to steer a new retrieval request.
    """

    if not isinstance(resolution, RequirementResolution):
        raise TypeError("residual repair requires a RequirementResolution")
    try:
        normalized_max_slots = int(max_slots)
    except (TypeError, ValueError) as error:
        raise TypeError("residual repair max_slots must be positive") from error
    if normalized_max_slots <= 0:
        raise ValueError("residual repair max_slots must be positive")
    if resolution.status != "resolved":
        return ResidualRepairPlan(
            (),
            "requirements_not_resolved",
            query_limit=normalized_max_slots,
        )

    known_required = {
        str(item.slot_id): item
        for item in resolution.requirements
        if bool(item.required)
    }
    ordered_missing = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in missing_slot_ids
            if str(value).strip()
        )
    )
    if not ordered_missing:
        return ResidualRepairPlan(
            (),
            "no_dynamic_missing_slots",
            query_limit=normalized_max_slots,
        )
    invalid = tuple(
        value for value in ordered_missing if value not in known_required
    )
    if invalid:
        return ResidualRepairPlan(
            (),
            "invalid_dynamic_missing_slots",
            invalid_missing_slot_ids=invalid,
            query_limit=normalized_max_slots,
        )
    # Keep request-plan order, not the order of an incidental selector map,
    # so repeated local replay remains deterministic.
    eligible_slots = tuple(
        item
        for item in resolution.requirements
        if bool(item.required) and item.slot_id in set(ordered_missing)
    )
    slots = eligible_slots[:normalized_max_slots]
    return ResidualRepairPlan(
        slots,
        "dynamic_missing_slots",
        dropped_slot_ids=tuple(item.slot_id for item in eligible_slots[normalized_max_slots:]),
        query_limit=normalized_max_slots,
    )


def infer_request_mode(question: str, intent: QueryIntent) -> RequestMode:
    """Classify obvious non-factual requests without fabricating slots.

    The public query API does not yet carry a typed ``request_mode``.  This is
    deliberately a narrow, deterministic compatibility inference: callers of
    the resolver can pass an explicit mode, while ordinary unknown requests
    remain factual by default and therefore resolve to ``unknown`` if planning
    yielded no authoritative requirement.
    """

    answer_shape = str(intent.answer_shape or "").strip().casefold()
    if answer_shape in _CONVERSATIONAL_ANSWER_SHAPES:
        return "conversational"
    if answer_shape in _EXPLORATORY_ANSWER_SHAPES:
        return "exploratory"
    compact = "".join(str(question or "").strip().casefold().split())
    greetings = {
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "hello",
        "hi",
        "早上好",
        "晚安",
        "谢谢",
        "感谢",
    }
    return "conversational" if compact in greetings else "factual"


def resolve_authoritative_requirements(
    question: str,
    intent: QueryIntent,
    *,
    request_mode: RequestMode = "factual",
    planner_origin: PlannerOrigin = "cloud",
    requirement_specs: Sequence[Mapping[str, Any]] | None = None,
    discovery_hints: Sequence[str] = (),
    planner_call_ids: Sequence[str] = (),
) -> RequirementResolution:
    """Freeze evidence obligations before any vector/candidate operation.

    ``QueryIntent.search_queries`` are the production planner's atomic
    evidence questions.  The whole user question, later follow-up planning,
    rerank expansion and contextual matching may still be useful discovery
    inputs, but they intentionally do not become required coverage slots.
    ``discovery_hints`` is accepted only to make that boundary explicit and is
    never inspected for requirements.
    """

    del discovery_hints
    mode = str(request_mode).strip().casefold()
    if mode not in _REQUEST_MODES:
        raise ValueError("request_mode is invalid")
    resolved_mode: RequestMode = mode  # type: ignore[assignment]
    raw_planner_origin = str(planner_origin).strip().casefold()
    if raw_planner_origin not in _PLANNER_ORIGINS:
        raise ValueError("planner_origin is invalid")
    resolved_planner_origin: PlannerOrigin = raw_planner_origin  # type: ignore[assignment]
    safe_call_ids = tuple(
        _safe_identifier(value, field_name="planner_call_id")
        for value in planner_call_ids
    )

    raw_specs: list[dict[str, Any]] = []
    if requirement_specs is not None:
        for raw_spec in requirement_specs:
            if not isinstance(raw_spec, Mapping):
                raise ValueError("requirement_specs must contain mappings")
            raw_specs.append({str(key): value for key, value in raw_spec.items()})
    else:
        for query in intent.search_queries:
            normalized_query = str(query).strip()
            if not normalized_query:
                continue
            structural = normalized_query.startswith(_STRUCTURAL_SLOT_PREFIXES)
            query_origin = (
                "user_explicit"
                if structural and normalized_query.startswith("__answer_slot__ ")
                else _metadata_origin(resolved_planner_origin)
            )
            raw_specs.append(
                {
                    "question": normalized_query,
                    "origin": query_origin,
                    "query_refs": [
                        _stable_identifier("query", normalized_query)
                    ],
                    "clause_ids": [
                        _stable_identifier("clause", normalized_query)
                    ],
                }
            )

    requirements: list[EvidenceSlot] = []
    seen_slot_ids: set[str] = set()
    seen_questions: set[str] = set()
    for index, spec in enumerate(raw_specs):
        requirement_question = str(spec.get("question", "")).strip()
        if not requirement_question:
            continue
        normalized_question = _normalized(requirement_question)
        if normalized_question in seen_questions:
            continue
        default_slot_id = _stable_identifier("slot", requirement_question)
        slot_id = _safe_identifier(
            spec.get("slot_id", default_slot_id), field_name="slot_id"
        )
        if slot_id in seen_slot_ids:
            raise ValueError("authoritative requirement slot_id is duplicated")
        raw_origin = str(
            spec.get("origin", _metadata_origin(resolved_planner_origin))
        ).strip()
        if raw_origin not in _SLOT_ORIGINS:
            raise ValueError("authoritative requirement origin is invalid")
        raw_support_mode = str(spec.get("support_mode", "alternative")).strip()
        if raw_support_mode not in _SUPPORT_MODES:
            raise ValueError("authoritative requirement support_mode is invalid")
        query_refs = _as_terms(spec.get("query_refs", ())) or (
            _stable_identifier("query", requirement_question),
        )
        clause_ids = _as_terms(spec.get("clause_ids", ())) or (
            _stable_identifier("clause", requirement_question),
        )
        for clause_id in clause_ids:
            _safe_identifier(clause_id, field_name="clause_id")
        if raw_support_mode == "joint" and len(clause_ids) < 2:
            raise ValueError("joint requirements need at least two clause_ids")
        negation_hint, modality_hint = _derived_hints(requirement_question)
        requirements.append(
            EvidenceSlot(
                slot_id=slot_id,
                question=requirement_question,
                required=bool(spec.get("required", True)),
                query_id=str(spec.get("query_id", query_refs[0])).strip(),
                subject_terms=_as_terms(
                    spec.get("subject_terms", intent.target_entities)
                ),
                relation_hint=str(
                    spec.get("relation_hint", intent.requested_relation)
                ).strip(),
                temporal_hint=str(
                    spec.get("temporal_hint", intent.temporal_constraint)
                ).strip(),
                epistemic_hint=str(
                    spec.get(
                        "epistemic_hint",
                        "uncertainty_required"
                        if intent.uncertainty_required
                        else "",
                    )
                ).strip(),
                query_refs=query_refs,
                origin=raw_origin,  # type: ignore[arg-type]
                support_mode=raw_support_mode,  # type: ignore[arg-type]
                clause_ids=clause_ids,
                object_terms=_as_terms(spec.get("object_terms", ())),
                negation_hint=str(
                    spec.get("negation_hint", negation_hint)
                ).strip(),
                modality_hint=str(
                    spec.get("modality_hint", modality_hint)
                ).strip(),
            )
        )
        seen_slot_ids.add(slot_id)
        seen_questions.add(normalized_question)

    if requirements:
        status: RequirementStatus = "resolved"
    elif resolved_mode == "factual":
        status = "unknown"
    else:
        status = "not_applicable"
    if status == "not_applicable":
        resolved_planner_origin = "not_applicable"
    return RequirementResolution(
        request_mode=resolved_mode,
        status=status,
        requirements=tuple(requirements),
        planner_origin=resolved_planner_origin,
        planner_call_ids=safe_call_ids,
    )


def requirement_support_from_records(
    requirements: Sequence[EvidenceSlot],
    records: Sequence[Mapping[str, Any]],
) -> dict[int, set[str]]:
    """Attach candidate observations to an already-frozen requirement set.

    A record can add support, never create or remove an obligation.  Joint
    requirements fail closed unless one observation explicitly attests every
    clause in that joint group; taking a union of episode IDs across separate
    records would incorrectly claim a joint proposition is covered.
    """

    slots_by_question: dict[str, list[EvidenceSlot]] = {}
    for slot in requirements:
        slots_by_question.setdefault(_normalized(slot.question), []).append(slot)
    support: dict[int, set[str]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        matched_slots = slots_by_question.get(
            _normalized(record.get("query", "")), []
        )
        raw_episode_ids = record.get("episode_ids", [])
        if not isinstance(raw_episode_ids, Sequence) or isinstance(
            raw_episode_ids, (str, bytes)
        ):
            continue
        episode_ids: list[int] = []
        for value in raw_episode_ids:
            if isinstance(value, bool):
                continue
            try:
                episode_id = int(value)
            except (TypeError, ValueError):
                continue
            episode_ids.append(episode_id)
        if not episode_ids:
            continue
        observed_clause_ids = set(_as_terms(record.get("clause_ids", ())))
        for slot in matched_slots:
            if (
                slot.support_mode == "joint"
                and not set(slot.clause_ids).issubset(observed_clause_ids)
            ):
                continue
            for episode_id in episode_ids:
                support.setdefault(episode_id, set()).add(slot.slot_id)
    return support


def structural_queries(question: str, intent: QueryIntent) -> list[str]:
    """Translate typed intent fields into explicit evidence-budget slots."""

    queries: list[str] = []
    fields = (
        ("关系约束", intent.requested_relation),
        ("时间约束", intent.temporal_constraint),
        ("因果约束", intent.causal_constraint),
    )
    for label, raw_value in fields:
        value = str(raw_value).strip()
        if value:
            queries.append(f"__constraint_slot__ {label}：{value}")

    quoted: list[str] = []
    for match in re.finditer(r"[‘“\"]([^’”\"]{2,80})[’”\"]", question):
        value = match.group(1).strip()
        before = question[max(0, match.start() - 18) : match.start()]
        after = question[match.end() : match.end() + 18]
        prohibited = bool(
            re.search(
                r"(?:不要|无需|禁止)(?:直接)?"
                r"(?:回答|认定|写成|证明)?[^。；]{0,8}$",
                before,
            )
            or re.search(
                r"^[^‘’“”\"。；]{0,8}"
                r"(?:未经证明|不能确定|不代表|并非事实)",
                after,
            )
        )
        if value and not prohibited and value not in quoted:
            quoted.append(value)
    for value in quoted:
        queries.append(f"__answer_slot__ 原文中“{value}”对应的事实是什么")
        if len(queries) >= 6:
            break
    return list(dict.fromkeys(queries))


def expand_rerank_atomic_queries(
    atomic_queries: list[str],
    intent: QueryIntent,
    whole_question: str | None = None,
) -> list[str]:
    """Split residual Chinese conjuncts for evidence-slot accounting only."""

    base_queries = list(
        dict.fromkeys(str(query).strip() for query in atomic_queries)
    )
    result: list[str] = []
    context = "、".join(intent.target_entities[:4])
    for query in base_queries:
        if query and query not in result:
            result.append(query)
        if (
            (whole_question is not None and query == whole_question)
            or query.startswith(("__constraint_slot__ ", "__answer_slot__ "))
            or "还是" in query
        ):
            continue
        paired_subjects = re.match(
            r"^([^，。；：！？]{2,12})与([^，。；：！？的]{2,16})的(.{3,})$",
            query,
        )
        if paired_subjects:
            first, second, predicate = paired_subjects.groups()
            for expanded in (
                f"{first}本人直接陈述或表现的{predicate}",
                f"{second}直接陈述或被转述的{predicate}",
            ):
                if expanded not in result:
                    result.append(expanded)
        for segment in re.split(r"[。；\n]", query):
            if re.search(r"[‘“\"][^’”\"]*、[^’”\"]*[’”\"]", segment):
                continue
            parts = [part.strip() for part in segment.split("、")]
            eligible_parts = [part for part in parts if len(part) >= 4]
            if not 2 <= len(parts) <= 8 or len(eligible_parts) < 2:
                continue
            for part in eligible_parts:
                expanded = (
                    f"{context}：{part}（独立证据槽，不能由同列另一条件替代）"
                    if context
                    else f"{part}（独立证据槽，不能由同列另一条件替代）"
                )
                if expanded not in result:
                    result.append(expanded)
    return result[:40]


def limit_rerank_atomic_queries(queries: list[str], limit: int) -> list[str]:
    """Bound slots while preserving constraints and late resolved queries."""

    unique = list(
        dict.fromkeys(
            str(value).strip() for value in queries if str(value).strip()
        )
    )
    if limit <= 0 or len(unique) <= limit:
        return unique
    selected_indices = {0}
    selected_indices.update(
        index
        for index, value in enumerate(unique)
        if value.startswith(("__constraint_slot__ ", "__answer_slot__ "))
    )
    if len(selected_indices) >= limit:
        return [unique[index] for index in sorted(selected_indices)[:limit]]
    regular = [
        index
        for index in range(1, len(unique))
        if index not in selected_indices
    ]
    available = limit - len(selected_indices)
    front_count = min(len(regular), (available * 2 + 2) // 3)
    back_count = min(
        len(regular) - front_count,
        available - front_count,
    )
    selected_indices.update(regular[:front_count])
    if back_count:
        selected_indices.update(regular[-back_count:])
    return [unique[index] for index in sorted(selected_indices)][:limit]


def requires_entity_resolved_followup(
    question: str,
    intent: QueryIntent,
) -> bool:
    """Detect a later clause that refers to an earlier unknown answer."""

    text = " ".join([question, *intent.search_queries, intent.requested_relation])
    dependent_patterns = (
        r"(?:谁|哪位|哪个|什么人).{0,100}"
        r"(?:这位|这名|该(?:对象|实体|主体)|其|此人).{0,80}"
        r"(?:谁|哪位|哪个|什么|又|身份|名称|别名|化名|代号)",
        r"(?:这位|这名|该(?:对象|实体|主体)|他们|她们|它们|其|此人)"
        r".{0,80}(?:谁|哪位|哪个|什么|名称|别名|化名|代号|又)",
        r"(?:已识别|上一跳|前一跳|该候选|此人|上述(?:对象|实体|主体))",
    )
    return any(re.search(pattern, text) for pattern in dependent_patterns)
