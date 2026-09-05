from __future__ import annotations

import re
from typing import Any, Callable, TypeVar

from memory_demo.types import ConceptDraft, EpisodeDraft

T = TypeVar("T")


# Source renderers reserve only these wrappers for structural metadata.  Every
# other nonblank physical line is content: this includes continuation lines of
# a multi-line language value, which deliberately have no repeated language
# prefix.  ``script_raw`` is intentionally *not* structural because it can
# carry the only proof that a card is an engine command.
_STRUCTURAL_SOURCE_LINE = re.compile(
    r"^\[(?:source_key|segment_index|record|speaker_raw|speaker_aliases|"
    r"speaker_alias_legend|evidence_origin|epistemic_status|"
    r"evidence_generation|epistemic_note|document_role|document_style)"
    r"(?:\s*:[^\]]*)?\]$",
    re.IGNORECASE,
)


def _parse_items(
    payload: Any,
    key: str,
    parser: Callable[[dict[str, Any]], T],
) -> tuple[list[T], list[str]]:
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict):
        raw_items = payload.get(key, [])
    else:
        return [], ["response must be an object or list"]
    if not isinstance(raw_items, list):
        return [], [f"{key} must be a list"]
    valid: list[T] = []
    errors: list[str] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            errors.append(f"item {index}: must be an object")
            continue
        try:
            valid.append(parser(item))
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"item {index}: {exc}")
    return valid, errors


def parse_episode_payload(payload: Any) -> tuple[list[EpisodeDraft], list[str]]:
    return _parse_items(payload, "episodes", EpisodeDraft.from_dict)


def parse_empty_episode_audit_payload(
    payload: Any,
    source_lines: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Parse a source-only adversarial review without trusting its labels.

    The reviewer must prove its decision with exact line spans and quotes.  A
    malformed or incomplete ``safe_skip`` response is rejected rather than
    silently converted into a non-story segment.
    """

    if not isinstance(payload, dict):
        return {}, ["empty Episode audit must be an object"]
    errors: list[str] = []
    if payload.get("contract_version") != "empty_episode_adversarial_audit_v1":
        errors.append("empty Episode audit has an unsupported contract_version")
    verdict = str(payload.get("verdict", "")).strip().casefold()
    if verdict not in {"safe_skip", "episode_required", "uncertain"}:
        errors.append("empty Episode audit has an invalid verdict")
    source_kind = str(payload.get("source_kind", "")).strip().casefold()
    if source_kind not in {
        "control_only",
        "title_preview_only",
        "non_event",
        "eventful",
        "mixed_or_ambiguous",
    }:
        errors.append("empty Episode audit has an invalid source_kind")

    raw_reviews = payload.get("line_reviews")
    if not isinstance(raw_reviews, list):
        errors.append("empty Episode audit line_reviews must be a list")
        raw_reviews = []
    allowed_kinds = {"control", "title_preview", "non_event", "event", "ambiguous"}
    reviews: list[dict[str, Any]] = []
    reviewed_lines: set[int] = set()
    for position, item in enumerate(raw_reviews):
        if not isinstance(item, dict):
            errors.append(f"empty Episode audit review {position}: must be an object")
            continue
        try:
            start_line = int(item["start_line"])
            end_line = int(item["end_line"])
        except (KeyError, TypeError, ValueError):
            errors.append(
                f"empty Episode audit review {position}: start_line/end_line are required integers"
            )
            continue
        if start_line < 1 or end_line < start_line or end_line > len(source_lines):
            errors.append(f"empty Episode audit review {position}: line range is out of bounds")
            continue
        kind = str(item.get("kind", "")).strip().casefold()
        if kind not in allowed_kinds:
            errors.append(f"empty Episode audit review {position}: invalid kind")
            continue
        quote = item.get("quote")
        if not isinstance(quote, str):
            errors.append(f"empty Episode audit review {position}: quote must be a string")
            continue
        expected_quote = "\n".join(source_lines[start_line - 1 : end_line])
        if quote != expected_quote:
            errors.append(
                f"empty Episode audit review {position}: quote does not exactly match Source"
            )
            continue
        span_lines = set(range(start_line, end_line + 1))
        if reviewed_lines.intersection(span_lines):
            errors.append(f"empty Episode audit review {position}: overlaps another review")
            continue
        reviewed_lines.update(span_lines)
        reviews.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "kind": kind,
                "quote": quote,
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    raw_ranges = payload.get("required_ranges", [])
    if not isinstance(raw_ranges, list):
        errors.append("empty Episode audit required_ranges must be a list")
        raw_ranges = []
    required_ranges: list[tuple[int, int]] = []
    for position, raw_range in enumerate(raw_ranges):
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            errors.append(
                f"empty Episode audit required_range {position}: must contain two line numbers"
            )
            continue
        try:
            start_line, end_line = int(raw_range[0]), int(raw_range[1])
        except (TypeError, ValueError):
            errors.append(
                f"empty Episode audit required_range {position}: values must be integers"
            )
            continue
        if start_line < 1 or end_line < start_line or end_line > len(source_lines):
            errors.append(
                f"empty Episode audit required_range {position}: line range is out of bounds"
            )
            continue
        if (start_line, end_line) not in required_ranges:
            required_ranges.append((start_line, end_line))

    substantive_lines = {
        number
        for number, line in enumerate(source_lines, 1)
        if line.strip() and _STRUCTURAL_SOURCE_LINE.fullmatch(line.strip()) is None
    }
    if verdict == "safe_skip":
        if source_kind not in {"control_only", "title_preview_only", "non_event"}:
            errors.append("safe_skip must use a non-event source_kind")
        if required_ranges:
            errors.append("safe_skip must not declare required_ranges")
        if not reviews:
            errors.append("safe_skip requires line_reviews")
        if any(
            review["kind"] not in {"control", "title_preview", "non_event"}
            for review in reviews
        ):
            errors.append("safe_skip may not include event or ambiguous reviews")
        if any(
            review["start_line"] != review["end_line"] for review in reviews
        ):
            errors.append(
                "safe_skip requires one single-line review for each substantive Source line"
            )
        missing = sorted(substantive_lines - reviewed_lines)
        if missing:
            errors.append(
                "safe_skip does not cover every substantive Source line: "
                + ", ".join(str(number) for number in missing)
            )
    elif verdict == "episode_required":
        if source_kind not in {"eventful", "mixed_or_ambiguous"}:
            errors.append("episode_required must use eventful or mixed_or_ambiguous source_kind")
        event_reviews = [review for review in reviews if review["kind"] == "event"]
        if not event_reviews:
            errors.append("episode_required requires at least one event review")
        if not required_ranges:
            errors.append("episode_required requires at least one required_range")
        for start_line, end_line in required_ranges:
            if not any(
                review["start_line"] <= end_line and review["end_line"] >= start_line
                for review in event_reviews
            ):
                errors.append(
                    "episode_required range does not intersect an event review: "
                    f"{start_line}-{end_line}"
                )

    if errors:
        return {}, errors
    return {
        "contract_version": "empty_episode_adversarial_audit_v1",
        "verdict": verdict,
        "source_kind": source_kind,
        "reason": str(payload.get("reason", "")).strip(),
        "line_reviews": reviews,
        "required_ranges": required_ranges,
        "substantive_lines": sorted(substantive_lines),
    }, []


_EPISODE_TEXT_CITATION = re.compile(
    r"(?:\[|【)\s*(?:证据|evidence)\s*[:：]\s*"
    r"(?P<ranges>[^\]】]+)\s*(?:\]|】)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_EVIDENCE_RANGE = re.compile(
    r"[Ll]?(?P<start>\d+)\s*"
    r"(?:-|–|—|~|～|至)\s*"
    r"[Ll]?(?P<end>\d+)",
    re.IGNORECASE,
)
_EPISODE_PREFIX = re.compile(
    r"^\s*(?:(?:Episode|事件|记忆)\s*\d+\s*[:：.-]?|\d+\s*[.)、．]\s*)",
    re.IGNORECASE,
)


def parse_episode_text(
    response: str,
    *,
    timeline_scope: str,
) -> tuple[list[EpisodeDraft], list[str]]:
    """Turn human-readable Episode paragraphs into program-owned drafts.

    The model is not asked to serialize application objects.  Its only
    machine-readable annotation is a normal source citation at the end of
    each paragraph.  All database fields and defaults are assembled here.
    """

    text = str(response or "").strip()
    if not text:
        return [], ["response is empty"]
    if text.casefold() in {
        "无可提取事件",
        "没有可提取的事件",
        "no extractable episode",
        "no extractable episodes",
    }:
        return [], []

    blocks = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    # Models occasionally omit blank lines while still placing one complete
    # cited Episode on each line.  This remains ordinary readable text and is
    # losslessly separable by the citations themselves.
    if len(blocks) == 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1 and all(
            _EPISODE_TEXT_CITATION.search(line) for line in lines
        ):
            blocks = lines

    episodes: list[EpisodeDraft] = []
    errors: list[str] = []
    for index, block in enumerate(blocks):
        citation = _EPISODE_TEXT_CITATION.search(block)
        if citation is None:
            errors.append(f"paragraph {index}: missing source citation")
            continue
        body = _EPISODE_PREFIX.sub("", block[: citation.start()].strip()).strip()
        if not body:
            errors.append(f"paragraph {index}: episode text is empty")
            continue
        spans: list[tuple[int, int]] = []
        for matched in _EVIDENCE_RANGE.finditer(citation.group("ranges")):
            start = int(matched.group("start"))
            end = int(matched.group("end"))
            if start > end:
                start, end = end, start
            spans.append((start, end))
        if not spans:
            errors.append(f"paragraph {index}: citation has no valid line range")
            continue
        if len(spans) > 4:
            errors.append(f"paragraph {index}: citation exceeds four line ranges")
            continue
        episodes.append(
            EpisodeDraft(
                text=body,
                participants=[],
                event_type="",
                location_text="",
                story_time_text="",
                timeline_scope=timeline_scope,
                confidence=0.9,
                evidence_origin="source",
                epistemic_status="asserted",
                generation=0,
                evidence_spans=spans,
            )
        )
    return episodes, errors


def parse_source_scoped_episode_text(
    response: str,
    *,
    timeline_scope: str,
) -> tuple[list[EpisodeDraft], list[str]]:
    """Parse ordinary Episode prose without asking the model for metadata.

    Source ownership is established by the import task, so citations,
    participants and persistence fields are deliberately not model outputs.
    """

    text = str(response or "").strip()
    if not text:
        return [], ["response is empty"]
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    if text.casefold() in {
        "无可提取事件",
        "没有可提取的事件",
        "no extractable episode",
        "no extractable episodes",
    }:
        return [], []

    blocks = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    if len(blocks) == 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        numbered = [line for line in lines if _EPISODE_PREFIX.match(line)]
        if len(numbered) >= 2 and len(numbered) == len(lines):
            blocks = lines

    episodes: list[EpisodeDraft] = []
    errors: list[str] = []
    for index, block in enumerate(blocks):
        body = _EPISODE_PREFIX.sub("", block).strip()
        if re.fullmatch(r"[-=*~_━─—]{3,}", body):
            continue
        if not body:
            errors.append(f"paragraph {index}: episode text is empty")
            continue
        if len(body) > 2_000:
            errors.append(f"paragraph {index}: episode text exceeds 2000 characters")
            continue
        episodes.append(
            EpisodeDraft(
                text=body,
                timeline_scope=timeline_scope,
                confidence=0.8,
                evidence_origin="source",
                epistemic_status="asserted",
                generation=0,
            )
        )
    if len(episodes) > 24:
        return [], ["response exceeds 24 Episodes for one Source"]
    return episodes, errors


_ENTAILMENT_HEADING = re.compile(
    r"^\s*(?:Episode|事件)\s*(?P<index>\d+)\s*[:：]\s*"
    r"(?P<verdict>通过|支持|需修改|修改|supported|revise)\s*$",
    re.IGNORECASE,
)


def parse_entailment_text(
    response: str,
    *,
    expected_indexes: set[int],
) -> tuple[dict[int, dict[str, str]], list[str]]:
    """Parse a compact, human-readable evidence review."""

    reviews: dict[int, dict[str, str]] = {}
    errors: list[str] = []
    current_index: int | None = None
    current_lines: list[str] = []

    def finish() -> None:
        nonlocal current_index, current_lines
        if current_index is None:
            return
        heading = _ENTAILMENT_HEADING.match(current_lines[0])
        assert heading is not None
        raw_verdict = heading.group("verdict").casefold()
        verdict = (
            "supported" if raw_verdict in {"通过", "支持", "supported"} else "revise"
        )
        details = "\n".join(current_lines[1:]).strip()
        revised = ""
        unsupported = ""
        for line in current_lines[1:]:
            stripped = line.strip()
            if re.match(r"^(?:修改为|修订为|revised)\s*[:：]", stripped, re.IGNORECASE):
                revised = re.split(r"[:：]", stripped, maxsplit=1)[1].strip()
            elif re.match(
                r"^(?:问题|不支持|unsupported)\s*[:：]", stripped, re.IGNORECASE
            ):
                unsupported = re.split(r"[:：]", stripped, maxsplit=1)[1].strip()
        if verdict == "revise" and not revised:
            errors.append(f"review {current_index}: revision text is missing")
        reviews[current_index] = {
            "verdict": verdict,
            "revised_text": revised,
            "unsupported_claims": unsupported or details,
        }
        current_index = None
        current_lines = []

    for line in str(response or "").splitlines():
        heading = _ENTAILMENT_HEADING.match(line)
        if heading:
            finish()
            current_index = int(heading.group("index"))
            current_lines = [line]
        elif current_index is not None and line.strip():
            current_lines.append(line)
    finish()

    unexpected = set(reviews) - expected_indexes
    missing = expected_indexes - set(reviews)
    for index in sorted(unexpected):
        errors.append(f"unexpected episode_index {index}")
    for index in sorted(missing):
        errors.append(f"missing episode_index {index}")
    return reviews, errors


def parse_episode_fact_review_payload(
    payload: Any, expected_indexes: set[int]
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Parse evidence-first Episode reviews without trusting rewritten prose."""

    if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
        return {}, ["reviews must be a list"]
    allowed_statuses = {"supported", "corrected", "ambiguous", "rejected"}
    allowed_issues = {
        "role_reversal",
        "predicate_collapse",
        "argument_loss",
        "attribution_loss",
        "modality_loss",
        "negation_loss",
        "unsupported_identity",
        "name_invention",
        "other",
    }
    reviews: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for position, item in enumerate(payload["reviews"]):
        if not isinstance(item, dict):
            errors.append(f"review {position}: must be an object")
            continue
        try:
            episode_index = int(item["episode_index"])
            if episode_index not in expected_indexes:
                raise ValueError("unexpected episode_index")
            if episode_index in reviews:
                raise ValueError("duplicate episode_index")
            status = str(item["status"]).strip().casefold()
            if status not in allowed_statuses:
                raise ValueError("invalid status")
            raw_issues = item.get("issue_types", [])
            if not isinstance(raw_issues, list):
                raise ValueError("issue_types must be a list")
            issue_types = [str(value).strip().casefold() for value in raw_issues]
            if any(value not in allowed_issues for value in issue_types):
                raise ValueError("unsupported issue_type")
            raw_frames = item.get("evidence_frames", [])
            if not isinstance(raw_frames, list) or not raw_frames:
                raise ValueError("evidence_frames must be a non-empty list")
            frames: list[dict[str, Any]] = []
            for frame_position, frame in enumerate(raw_frames):
                if not isinstance(frame, dict):
                    raise ValueError(
                        f"evidence_frame {frame_position} must be an object"
                    )
                quote = str(frame.get("quote", "")).strip()
                predicate = str(frame.get("predicate_span", "")).strip()
                if not quote or not predicate:
                    raise ValueError(
                        f"evidence_frame {frame_position} requires quote and predicate_span"
                    )
                raw_qualifiers = frame.get("qualifiers", [])
                if not isinstance(raw_qualifiers, list):
                    raise ValueError(
                        f"evidence_frame {frame_position} qualifiers must be a list"
                    )
                frames.append(
                    {
                        "quote": quote,
                        "subject_span": str(frame.get("subject_span", "")).strip(),
                        "predicate_span": predicate,
                        "object_span": str(frame.get("object_span", "")).strip(),
                        "semantic_predicate": str(
                            frame.get("semantic_predicate", "")
                        ).strip(),
                        "attribution": str(frame.get("attribution", "")).strip(),
                        "qualifiers": [
                            str(value).strip()
                            for value in raw_qualifiers
                            if str(value).strip()
                        ],
                    }
                )
            raw_corrected = item.get("corrected_episode")
            corrected = None
            if status == "corrected":
                if not isinstance(raw_corrected, dict):
                    raise ValueError("corrected status requires corrected_episode")
                corrected = EpisodeDraft.from_dict(raw_corrected)
            elif raw_corrected not in (None, {}):
                raise ValueError("only corrected status may provide corrected_episode")
            reviews[episode_index] = {
                "episode_index": episode_index,
                "status": status,
                "issue_types": issue_types,
                "evidence_frames": frames,
                "reason": str(item.get("reason", "")).strip(),
                "corrected_episode": corrected,
            }
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"review {position}: {exc}")
    for episode_index in sorted(expected_indexes - reviews.keys()):
        errors.append(f"missing episode_index {episode_index}")
    return reviews, errors


def parse_concept_payload(payload: Any) -> tuple[list[ConceptDraft], list[str]]:
    return _parse_items(payload, "concepts", ConceptDraft.from_dict)


_CONCEPT_PREFIX = re.compile(r"^\s*(?:[-*•]\s*|\d+\s*[.)、．]\s*)")
_CONCEPT_BATCH_HEADING = re.compile(
    r"^\s*Episode\s+(?P<index>\d+)\s*[:：]?\s*$",
    re.IGNORECASE,
)


def parse_concept_text(
    response: str,
    *,
    evidence_text: str = "",
) -> tuple[list[ConceptDraft], list[str]]:
    """Parse one readable Concept name per line.

    When Episode evidence is available, the program—not the model—builds the
    durable description from literal supporting sentences.  A model-provided
    explanation after a colon is accepted for backwards compatibility but is
    deliberately not persisted.
    """

    text = str(response or "").strip()
    if not text:
        return [], ["response is empty"]
    if text.casefold() in {
        "无可提取概念",
        "没有可提取的概念",
        "无",
        "none",
        "no reusable concepts",
    }:
        return [], []
    concepts: list[ConceptDraft] = []
    errors: list[str] = []
    seen: set[str] = set()
    for position, raw_line in enumerate(text.splitlines()):
        line = _CONCEPT_PREFIX.sub("", raw_line.strip()).strip()
        if not line:
            continue
        if _CONCEPT_BATCH_HEADING.match(line):
            continue
        parts = re.split(r"[:：]", line, maxsplit=1)
        name = parts[0].strip()
        model_description = parts[1].strip() if len(parts) == 2 else name
        if not name or len(name) > 100 or not model_description:
            errors.append(f"line {position}: invalid concept line")
            continue
        folded = name.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        description = model_description
        if evidence_text.strip():
            sentences = [
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[。！？.!?])\s*|\n+", evidence_text.strip()
                )
                if sentence.strip()
            ]
            supporting = [sentence for sentence in sentences if name in sentence]
            description = " ".join(supporting)[:800] or evidence_text.strip()[:800]
        try:
            concepts.append(
                ConceptDraft.from_dict(
                    {
                        "canonical_name": name,
                        "description": description,
                        "embedding_text": f"{name}。{description}",
                        "aliases": [],
                        "confidence": 0.85,
                    }
                )
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"line {position}: {exc}")
    return concepts, errors


def parse_concept_batch_text(
    response: str,
    expected_count: int,
    *,
    evidence_texts: list[str] | None = None,
) -> tuple[dict[int, list[ConceptDraft]], dict[int, list[str]], list[str]]:
    groups: dict[int, list[str]] = {}
    global_errors: list[str] = []
    current: int | None = None
    for position, raw_line in enumerate(str(response or "").splitlines()):
        heading = _CONCEPT_BATCH_HEADING.match(raw_line)
        if heading:
            current = int(heading.group("index"))
            if current in groups:
                global_errors.append(f"line {position}: duplicate Episode {current}")
            groups.setdefault(current, [])
            continue
        if raw_line.strip():
            if current is None:
                global_errors.append(
                    f"line {position}: concept appears before an Episode heading"
                )
            else:
                groups[current].append(raw_line)
    parsed: dict[int, list[ConceptDraft]] = {}
    group_errors: dict[int, list[str]] = {}
    for episode_index in range(expected_count):
        if episode_index not in groups:
            group_errors[episode_index] = ["missing Episode group"]
            continue
        body = "\n".join(groups[episode_index]).strip() or "无"
        evidence = (
            evidence_texts[episode_index]
            if evidence_texts is not None and episode_index < len(evidence_texts)
            else ""
        )
        concepts, errors = parse_concept_text(body, evidence_text=evidence)
        parsed[episode_index] = concepts
        if errors:
            group_errors[episode_index] = errors
    for unexpected in sorted(set(groups) - set(range(expected_count))):
        global_errors.append(f"unexpected Episode {unexpected}")
    return parsed, group_errors, global_errors


def parse_concept_batch_payload(
    payload: Any, expected_count: int
) -> tuple[dict[int, list[ConceptDraft]], dict[int, list[str]], list[str]]:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("episode_concepts"), list
    ):
        return {}, {}, ["episode_concepts must be a list"]
    groups: dict[int, list[ConceptDraft]] = {}
    group_errors: dict[int, list[str]] = {}
    global_errors: list[str] = []
    for position, group in enumerate(payload["episode_concepts"]):
        if not isinstance(group, dict):
            global_errors.append(f"group {position}: must be an object")
            continue
        try:
            episode_index = int(group["episode_index"])
        except (KeyError, TypeError, ValueError):
            global_errors.append(f"group {position}: invalid episode_index")
            continue
        if not 0 <= episode_index < expected_count:
            global_errors.append(f"group {position}: episode_index out of range")
            continue
        if episode_index in groups:
            global_errors.append(
                f"group {position}: duplicate episode_index {episode_index}"
            )
            continue
        concepts, errors = parse_concept_payload(
            {"concepts": group.get("concepts", [])}
        )
        groups[episode_index] = concepts
        if errors:
            group_errors[episode_index] = errors
    for episode_index in range(expected_count):
        if episode_index not in groups:
            group_errors.setdefault(episode_index, []).append("missing episode group")
    return groups, group_errors, global_errors


def parse_concept_admission_payload(
    payload: Any, expected_indexes: set[int]
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        return {}, ["decisions must be a list"]
    decisions: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for position, item in enumerate(payload["decisions"]):
        if not isinstance(item, dict):
            errors.append(f"decision {position}: must be an object")
            continue
        try:
            candidate_index = int(item["candidate_index"])
            if candidate_index not in expected_indexes:
                raise ValueError("unexpected candidate_index")
            if candidate_index in decisions:
                raise ValueError("duplicate candidate_index")
            action = str(item["action"]).strip().casefold()
            if action not in {"promote", "transient", "reuse"}:
                raise ValueError("invalid action")
            raw_existing_id = item.get("existing_concept_id")
            existing_id = (
                int(raw_existing_id) if raw_existing_id not in (None, "") else None
            )
            if action == "reuse" and existing_id is None:
                raise ValueError("reuse requires existing_concept_id")
            if action != "reuse" and existing_id is not None:
                raise ValueError("only reuse may set existing_concept_id")
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
            decisions[candidate_index] = {
                "candidate_index": candidate_index,
                "action": action,
                "existing_concept_id": existing_id,
                "reason": str(item.get("reason", "")).strip(),
                "confidence": confidence,
            }
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"decision {position}: {exc}")
    for candidate_index in sorted(expected_indexes - decisions.keys()):
        errors.append(f"missing candidate_index {candidate_index}")
    return decisions, errors


def parse_episode_relation_batch_payload(
    payload: Any, expected_ids: set[int]
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[str]], list[str]]:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("episode_relationships"), list
    ):
        return {}, {}, ["episode_relationships must be a list"]
    groups: dict[int, list[dict[str, Any]]] = {}
    group_errors: dict[int, list[str]] = {}
    global_errors: list[str] = []
    for position, group in enumerate(payload["episode_relationships"]):
        if not isinstance(group, dict):
            global_errors.append(f"group {position}: must be an object")
            continue
        try:
            current_id = int(group["current_id"])
        except (KeyError, TypeError, ValueError):
            global_errors.append(f"group {position}: invalid current_id")
            continue
        if current_id not in expected_ids:
            global_errors.append(
                f"group {position}: unexpected current_id {current_id}"
            )
            continue
        if current_id in groups:
            global_errors.append(f"group {position}: duplicate current_id {current_id}")
            continue
        relationships, errors = parse_relationships(
            {"relationships": group.get("relationships", [])}
        )
        groups[current_id] = relationships
        if errors:
            group_errors[current_id] = errors
    for current_id in expected_ids:
        if current_id not in groups:
            group_errors.setdefault(current_id, []).append("missing current group")
    return groups, group_errors, global_errors


def parse_concept_relation_batch_payload(
    payload: Any, expected_ids: set[int]
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[str]], list[str]]:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("concept_relationships"), list
    ):
        return {}, {}, ["concept_relationships must be a list"]
    groups: dict[int, list[dict[str, Any]]] = {}
    group_errors: dict[int, list[str]] = {}
    global_errors: list[str] = []
    for position, group in enumerate(payload["concept_relationships"]):
        if not isinstance(group, dict):
            global_errors.append(f"group {position}: must be an object")
            continue
        try:
            current_id = int(group["current_id"])
        except (KeyError, TypeError, ValueError):
            global_errors.append(f"group {position}: invalid current_id")
            continue
        if current_id not in expected_ids:
            global_errors.append(
                f"group {position}: unexpected current_id {current_id}"
            )
            continue
        if current_id in groups:
            global_errors.append(f"group {position}: duplicate current_id {current_id}")
            continue
        relationships, errors = parse_relationships(
            {"relationships": group.get("relationships", [])}
        )
        groups[current_id] = relationships
        if errors:
            group_errors[current_id] = errors
    for current_id in expected_ids:
        if current_id not in groups:
            group_errors.setdefault(current_id, []).append("missing current group")
    return groups, group_errors, global_errors


def parse_relationships(payload: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(payload, list):
        raw_relationships = payload
    elif isinstance(payload, dict) and isinstance(
        payload.get("relationships", []), list
    ):
        raw_relationships = payload["relationships"]
    else:
        return [], ["relationships must be a list"]
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    allowed = {
        "temporal",
        "causal",
        "identity",
        "semantic",
        "co_occurrence",
        "recall_trigger",
        "interpersonal",
    }
    for index, item in enumerate(raw_relationships):
        try:
            candidate_id = int(item["candidate_id"])
            relation_key = str(item["relation_key"]).strip()
            relation_type = str(item.get("relation_type", "")).strip()
            if not relation_type:
                if relation_key in allowed:
                    relation_type, relation_key = relation_key, "related_to"
                elif "/" in relation_key and relation_key.split("/", 1)[0] in allowed:
                    relation_type, relation_key = relation_key.split("/", 1)
            relation_text = str(item["relation_text"]).strip()
            if relation_type not in allowed:
                raise ValueError(f"unsupported relation_type {relation_type}")
            if relation_type == "temporal":
                normalized_key = relation_key.casefold()
                if normalized_key == "after" or normalized_key.endswith("_after"):
                    relation_key = "after"
                elif normalized_key in {
                    "before",
                    "precedes",
                } or normalized_key.endswith("_before"):
                    relation_key = "before"
                else:
                    raise ValueError(
                        "temporal relation_key must express before or after"
                    )
            if not relation_key or not relation_text:
                raise ValueError("relation_key and relation_text are required")
            valid.append(
                {
                    "candidate_id": candidate_id,
                    "relation_type": relation_type,
                    "relation_key": relation_key,
                    "relation_text": relation_text,
                    "polarity": max(-1, min(1, int(item.get("polarity", 1)))),
                    "llm_score": max(0.0, min(1.0, float(item.get("llm_score", 0.5)))),
                    "confidence": max(
                        0.0, min(1.0, float(item.get("confidence", 0.5)))
                    ),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"relationship {index}: {exc}")
    return valid, errors


def parse_growth_relationships(payload: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("relationships", []), list
    ):
        return [], ["relationships must be a list"]
    allowed_types = {"episode", "concept"}
    allowed_relations = {
        "temporal",
        "causal",
        "identity",
        "semantic",
        "co_occurrence",
        "recall_trigger",
        "interpersonal",
    }
    valid: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(payload["relationships"]):
        try:
            from_type = str(item["from_type"])
            to_type = str(item["to_type"])
            relation_type = str(item["relation_type"])
            if from_type not in allowed_types or to_type not in allowed_types:
                raise ValueError("unsupported node type")
            if relation_type not in allowed_relations:
                raise ValueError("unsupported relation type")
            relation_key = str(item["relation_key"]).strip()
            if relation_type == "temporal":
                normalized_key = relation_key.casefold()
                if normalized_key == "after" or normalized_key.endswith("_after"):
                    relation_key = "after"
                elif normalized_key in {
                    "before",
                    "precedes",
                } or normalized_key.endswith("_before"):
                    relation_key = "before"
                else:
                    raise ValueError(
                        "temporal relation_key must express before or after"
                    )
            raw_premises = item.get("premise_association_ids", [])
            if not isinstance(raw_premises, list):
                raise ValueError("premise_association_ids must be a list")
            premise_association_ids: list[int] = []
            for raw_id in raw_premises:
                if isinstance(raw_id, bool):
                    raise ValueError(
                        "premise association IDs must be positive integers"
                    )
                premise_id = int(raw_id)
                if premise_id <= 0 or premise_id != raw_id:
                    raise ValueError(
                        "premise association IDs must be positive integers"
                    )
                if premise_id not in premise_association_ids:
                    premise_association_ids.append(premise_id)
            parsed = {
                "from_type": from_type,
                "from_id": int(item["from_id"]),
                "to_type": to_type,
                "to_id": int(item["to_id"]),
                "relation_type": relation_type,
                "relation_key": relation_key,
                "relation_text": str(item["relation_text"]).strip(),
                "polarity": max(-1, min(1, int(item.get("polarity", 1)))),
                "llm_score": max(0.0, min(1.0, float(item.get("llm_score", 0.5)))),
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                "premise_association_ids": premise_association_ids,
            }
            if not parsed["relation_key"] or not parsed["relation_text"]:
                raise ValueError("relation_key and relation_text are required")
            valid.append(parsed)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"relationship {index}: {exc}")
    return valid, errors
