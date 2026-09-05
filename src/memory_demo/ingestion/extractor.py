from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import re
from typing import Iterable

from memory_demo.adapters.blue_archive import parse_korean_text

from memory_demo.event_log import JsonlEventLogger
from memory_demo.llm.prompts import (
    CONCEPT_FINE_GRAINED_SYSTEM,
    CONCEPT_SYSTEM,
    DOCUMENT_ANCHOR_MAP_SYSTEM,
    DOCUMENT_MAP_SYSTEM,
    EMPTY_EPISODE_ADVERSARIAL_AUDIT_SYSTEM,
    EPISODE_ENTAILMENT_AUDIT_SYSTEM,
    EPISODE_FACTUAL_AUDIT_SYSTEM,
    EPISODE_SYSTEM,
    SINGLE_PASS_EPISODE_SYSTEM,
    SOURCE_SCOPED_EPISODE_SYSTEM,
    EPISODE_QUALITY_AUDIT_SYSTEM,
    GRANULARITY_AUDIT_SYSTEM,
    TEMPORAL_AUDIT_SYSTEM,
    concept_batch_prompt,
    concept_prompt,
    audit_retry_prompt,
    document_anchor_map_prompt,
    document_anchor_retry_prompt,
    document_map_prompt,
    document_map_retry_prompt,
    empty_episode_adversarial_audit_prompt,
    entailment_retry_prompt,
    episode_entailment_audit_prompt,
    episode_factual_audit_prompt,
    episode_quality_audit_prompt,
    episode_prompt,
    factual_audit_retry_prompt,
    partial_item_retry_prompt,
    reference_coverage_fallback_prompt,
    reference_coverage_prompt,
    single_pass_episode_prompt,
    single_pass_retry_prompt,
    source_scoped_episode_prompt,
    granularity_audit_prompt,
    temporal_audit_prompt,
)
from memory_demo.llm.validation import (
    parse_concept_batch_payload,
    parse_concept_batch_text,
    parse_concept_payload,
    parse_concept_text,
    parse_entailment_text,
    parse_empty_episode_audit_payload,
    parse_episode_fact_review_payload,
    parse_episode_payload,
    parse_episode_text,
    parse_source_scoped_episode_text,
)
from memory_demo.llm import ModelTransportUnavailable
from memory_demo.llm.client import extract_json_payload
from memory_demo.types import ConceptDraft, EpisodeDraft


class ExtractionValidationError(ValueError):
    """A model response exhausted its own repair path and will not improve by replay."""


class EmptyEpisodeExtraction(ExtractionValidationError):
    """A primary extractor returned a clean empty result, not a bad result.

    This marker prevents the pipeline from mistaking an absent candidate for a
    validation failure or automatically treating it as harmless.  The caller
    must either obtain an independently verified safe-skip decision or fail.
    """

    def __init__(
        self,
        message: str,
        *,
        attempted_models: Iterable[str] = (),
    ):
        super().__init__(message)
        seen: set[str] = set()
        normalized: list[str] = []
        for model in attempted_models:
            value = str(model or "").strip()
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                normalized.append(value)
        # These are actual models which completed an extraction attempt.  The
        # reviewer must never be selected from this set.
        self.attempted_models = tuple(normalized)


@dataclass(slots=True)
class EmptyEpisodeAudit:
    """Validated receipt from an independent source-only empty-result review."""

    verdict: str
    source_kind: str
    reason: str
    line_reviews: list[dict[str, object]]
    required_ranges: list[tuple[int, int]]
    primary_model: str
    reviewer_model: str
    source_sha256: str
    response_sha256: str = ""
    validation_errors: list[str] | None = None
    attempted_models: list[str] = field(default_factory=list)

    def compact_receipt(self) -> str:
        payload = {
            "contract_version": "empty_episode_adversarial_audit_v1",
            "verdict": self.verdict,
            "source_kind": self.source_kind,
            "primary_model": self.primary_model,
            "attempted_models": self.attempted_models,
            "reviewer_model": self.reviewer_model,
            "source_sha256": self.source_sha256,
            "response_sha256": self.response_sha256,
            "required_ranges": self.required_ranges,
            "validation_errors": self.validation_errors or [],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def event_payload(self, *, include_quotes: bool) -> dict[str, object]:
        reviews: list[dict[str, object]] = []
        for review in self.line_reviews:
            item = {
                "start_line": review["start_line"],
                "end_line": review["end_line"],
                "kind": review["kind"],
                "reason_sha256": hashlib.sha256(
                    str(review["reason"]).encode("utf-8")
                ).hexdigest(),
                "quote_sha256": hashlib.sha256(
                    str(review["quote"]).encode("utf-8")
                ).hexdigest(),
            }
            if include_quotes:
                item["quote"] = review["quote"]
                item["reason"] = review["reason"]
            reviews.append(item)
        return {
            "audit_contract_version": "empty_episode_adversarial_audit_v1",
            "audit_verdict": self.verdict,
            "audit_source_kind": self.source_kind,
            "audit_reason_sha256": hashlib.sha256(
                self.reason.encode("utf-8")
            ).hexdigest(),
            "audit_primary_model": self.primary_model,
            "audit_attempted_models": self.attempted_models,
            "audit_reviewer_model": self.reviewer_model,
            "source_sha256": self.source_sha256,
            "audit_response_sha256": self.response_sha256,
            "audit_required_ranges": self.required_ranges,
            "audit_line_reviews": reviews,
            "audit_validation_errors": self.validation_errors or [],
        }


class MemoryExtractor:
    _AUDIT_MAX_EXPANSION_RATIO = 2.5
    _AUDIT_MAX_EXPANSION_ABSOLUTE = 8
    _TEMPORAL_RISK_MARKERS = (
        "上次",
        "过去",
        "曾经",
        "此前",
        "之前",
        "当年",
        "小时候",
        "童年",
        "回忆",
        "last time",
        "previously",
        "in the past",
        "used to",
        "かつて",
        "以前",
        "前回",
        "지난번",
        "과거",
        "예전에",
    )
    _SPECULATIVE_IDENTITY_MARKERS = (
        "推测为",
        "可能是",
        "疑似",
        "似乎是",
        "根据上下文",
        "应为",
        "probably",
        "possibly",
    )
    _TRIVIAL_EVENT_MARKERS = (
        "沉默",
        "点头",
        "呼唤",
        "应答",
        "回应",
        "反应",
        "询问",
        "提议",
        "感谢",
        "抱怨",
        "称赞",
        "章节开始",
        "标题",
    )
    _REFERENCE_META_EVENT_MARKERS = (
        "文档发布",
        "资料发布",
        "文档标题",
        "资料标题",
        "百科标题",
        "文档创建",
        "作品发布",
        "作品标题确认",
    )
    _GENERIC_STORY_TIME_VALUES = {
        "当前",
        "当前时间",
        "当前故事时间",
        "当前场景",
        "现在",
        "現在",
        "현재",
    }
    _SOURCE_NAME_SEPARATOR = re.compile(r"\s*[/／]\s*|[()（）]")
    _FOREIGN_NAME_CHAR = re.compile(r"[A-Za-z\u3040-\u30ff\uac00-\ud7af]")
    _HAN_CHAR = re.compile(r"[\u3400-\u9fff]")
    _SOURCE_DIALOGUE_LABEL = re.compile(r"^([^:\n]{2,80}):(?:\s|$)")
    _FOREIGN_ALIAS_PAREN = re.compile(
        r"(?P<name>[A-Za-z\u3040-\u30ff\uac00-\ud7af]"
        r"[A-Za-z0-9._·・\-\u3040-\u30ff\uac00-\ud7af\u3400-\u9fff]{1,31})"
        r"[（(](?P<aliases>[^()（）]{1,80})[）)]"
    )
    _NON_NAME_LABELS = {
        "unknown",
        "zh-cn",
        "zh-tw",
        "ja",
        "en",
        "ko",
        "th",
    }

    def _uses_natural_text_output(self) -> bool:
        """Real model clients expose chat_text; older test doubles stay compatible."""

        return getattr(
            self.model, "semantic_output_format", "json"
        ) == "natural_text" and callable(getattr(self.model, "chat_text", None))

    def _request_natural_text(
        self,
        system: str,
        prompt: str,
        **kwargs: object,
    ) -> str:
        method = getattr(self.model, "chat_text", None)
        if not callable(method):
            raise TypeError("model does not provide chat_text")
        return str(method(system, prompt, **kwargs))

    @classmethod
    def _remove_reference_meta_episodes(
        cls,
        episodes: list[EpisodeDraft],
        reference_record_count: int,
        logger: JsonlEventLogger | None = None,
    ) -> list[EpisodeDraft]:
        """Drop title-only hallucinations without suppressing real release facts.

        Reference Sources commonly contain one unlabelled document title before
        their labelled fact records.  A model can turn that title into an extra
        "the document was published/created" Episode.  Only remove a strong
        metadata-shaped candidate while the model returned *more* Episodes than
        labelled records; this preserves legitimate labelled publication facts.
        The check is repeated after replacement and quality-audit passes because
        either pass can reintroduce the title candidate.
        """

        if reference_record_count < 2:
            return episodes

        retained = list(episodes)
        removed: list[EpisodeDraft] = []
        while len(retained) > reference_record_count:
            candidate_index = next(
                (
                    index
                    for index, draft in enumerate(retained)
                    if (
                        draft.evidence_quotes
                        and not any(
                            "[资料类型" in quote for quote in draft.evidence_quotes
                        )
                    )
                    or any(
                        marker in draft.event_type
                        for marker in cls._REFERENCE_META_EVENT_MARKERS
                    )
                    or (
                        "文档作者" in " ".join(draft.participants)
                        and "文档" in draft.text
                    )
                    or (
                        ("文档" in draft.text or "作品标题" in draft.text)
                        and any(
                            phrase in draft.text
                            for phrase in (
                                "被创建",
                                "作为作品标题被正式",
                                "作为官方发布的作品标题",
                            )
                        )
                    )
                ),
                None,
            )
            if candidate_index is None:
                break
            removed.append(retained.pop(candidate_index))
        if removed and logger:
            logger.emit(
                "reference_meta_episodes_removed",
                removed=[asdict(draft) for draft in removed],
            )
        return retained

    def __init__(
        self,
        model,
        logger: JsonlEventLogger | None = None,
        *,
        concept_profile: str = "conservative",
        concept_target_min: int = 2,
        concept_target_max: int = 6,
        episode_audit_mode: str = "combined",
        episode_audit_always: bool = False,
        episode_factual_audit_mode: str = "off",
        episode_factual_audit_model: str = "",
        episode_factual_audit_batch_size: int = 3,
        empty_episode_audit_mode: str = "adversarial",
        empty_episode_audit_model: str = "",
        episode_extraction_profile: str = "legacy",
    ):
        self.model = model
        self.logger = logger
        if concept_profile not in {"conservative", "fine_grained"}:
            raise ValueError("unknown concept extraction profile")
        self.concept_profile = concept_profile
        self.concept_target_min = max(0, int(concept_target_min))
        self.concept_target_max = max(self.concept_target_min, int(concept_target_max))
        if episode_audit_mode not in {"combined", "split", "off"}:
            raise ValueError("unknown episode audit mode")
        self.episode_audit_mode = episode_audit_mode
        self.episode_audit_always = bool(episode_audit_always)
        if episode_factual_audit_mode not in {"off", "adaptive", "always"}:
            raise ValueError("unknown episode factual audit mode")
        self.episode_factual_audit_mode = episode_factual_audit_mode
        self.episode_factual_audit_model = episode_factual_audit_model.strip()
        self.episode_factual_audit_batch_size = max(
            1, int(episode_factual_audit_batch_size)
        )
        if empty_episode_audit_mode not in {"off", "adversarial"}:
            raise ValueError("unknown empty Episode audit mode")
        self.empty_episode_audit_mode = empty_episode_audit_mode
        self.empty_episode_audit_model = empty_episode_audit_model.strip()
        if episode_extraction_profile not in {
            "legacy",
            "single_pass_evidence",
            "single_pass_audited",
            "document_map_assisted",
            "document_map_contextual",
            "adaptive_anchor_map",
            "source_scoped_plain",
        }:
            raise ValueError("unknown Episode extraction profile")
        self.episode_extraction_profile = episode_extraction_profile

    @staticmethod
    def _single_pass_source_lines(source_text: str) -> tuple[list[str], str]:
        lines = [line for line in source_text.splitlines() if line.strip()]
        indexed = "\n".join(
            f"[L{index:04d}] {line}" for index, line in enumerate(lines, 1)
        )
        return lines, indexed

    @staticmethod
    def _source_scoped_model_text(source_text: str) -> str:
        """Reduce normalized records to the prose the model must summarize.

        Import metadata remains in SQLite and is interpreted by the program.
        Front matter is removed here instead of asking the model to remember a
        growing list of structural exceptions.
        """

        lines = source_text.splitlines()
        records: list[list[str]] = []
        current: list[str] | None = None
        for line in lines:
            if line.startswith("[record:"):
                if current is not None:
                    records.append(current)
                current = [line]
            elif current is not None:
                current.append(line)
        if current is not None:
            records.append(current)
        if not records:
            return source_text.strip()

        rendered: list[str] = []
        language_prefix = re.compile(
            r"^(?:zh-CN|zh-TW|ja|en|ko|th|unknown):\s*",
            re.IGNORECASE,
        )
        for record in records:
            if "[document_role: front_matter]" in record:
                continue
            speaker = next(
                (
                    line[len("[speaker_raw:") : -1].strip()
                    for line in record
                    if line.startswith("[speaker_raw:") and line.endswith("]")
                ),
                "",
            )
            content = [
                language_prefix.sub("", line, count=1).strip()
                for line in record
                if line.strip() and not line.lstrip().startswith("[")
            ]
            content = [line for line in content if line]
            if not content:
                continue
            body = "\n".join(content)
            if speaker and speaker.casefold() not in {
                "???",
                "unknown speaker",
                "未知发言者",
                "未标注发言者",
            }:
                body = f"{speaker}: {body}"
            rendered.append(body)
        return "\n\n".join(rendered).strip()

    @staticmethod
    def _split_source_scoped_episode_blocks(
        episodes: list[EpisodeDraft],
        *,
        max_sentences: int = 3,
        max_chars: int = 520,
    ) -> list[EpisodeDraft]:
        """Bound model paragraphs using only visible sentence structure.

        A change in the leading opaque speaker token is an explicit boundary,
        not a semantic guess. Length limits prevent one paragraph from becoming
        an entire scene while retaining short same-speaker continuations.
        """

        result: list[EpisodeDraft] = []
        sentence_pattern = re.compile(r".+?(?:[。！？!?]|$)", re.DOTALL)
        leading_token = re.compile(
            r"^\s*(SPEAKER_(?:\d{3}|[A-F0-9]{8}))"
        )
        for episode in episodes:
            sentences = [
                match.group(0).strip()
                for match in sentence_pattern.finditer(episode.text)
                if match.group(0).strip()
            ]
            if len(sentences) <= 1:
                result.append(episode)
                continue
            groups: list[list[str]] = []
            current: list[str] = []
            current_actor = ""
            current_chars = 0
            for sentence in sentences:
                matched = leading_token.match(sentence)
                actor = matched.group(1) if matched else ""
                boundary = bool(
                    current
                    and (
                        (actor and current_actor and actor != current_actor)
                        or len(current) >= max_sentences
                        or current_chars + len(sentence) > max_chars
                    )
                )
                if boundary:
                    groups.append(current)
                    current = []
                    current_actor = ""
                    current_chars = 0
                current.append(sentence)
                current_chars += len(sentence)
                if actor:
                    current_actor = actor
            if current:
                groups.append(current)
            result.extend(
                replace(
                    episode,
                    text="".join(group),
                    participants=list(episode.participants),
                    evidence_quotes=list(episode.evidence_quotes),
                    evidence_spans=list(episode.evidence_spans),
                )
                for group in groups
                if group
            )
        return result

    @classmethod
    def _speaker_token_map(cls, source_lines: list[str]) -> dict[str, str]:
        """Give Source speaker labels opaque, request-local identifiers.

        The semantic model only needs stable actors, not translated names.
        Masking those labels before extraction prevents cross-language entity
        substitution; the program restores the literal Source values before
        evidence validation and persistence.
        """

        labels: list[str] = []
        for raw_line in source_lines:
            stripped = raw_line.strip()
            metadata = re.fullmatch(r"\[speaker_raw:\s*([^\]]+)\]", stripped)
            dialogue = (
                None
                if stripped.startswith("[")
                else cls._SOURCE_DIALOGUE_LABEL.match(stripped)
            )
            label = (
                metadata.group(1).strip()
                if metadata is not None
                else dialogue.group(1).strip()
                if dialogue is not None
                else ""
            )
            if (
                not label
                or label.casefold() in cls._NON_NAME_LABELS
                or label.casefold()
                in {
                    "???",
                    "[username]",
                    "unknown speaker",
                    "未知发言者",
                    "未标注发言者",
                }
                or label in labels
            ):
                continue
            labels.append(label)
        return {f"SPEAKER_{index:03d}": label for index, label in enumerate(labels, 1)}

    @staticmethod
    def _mask_speaker_tokens(value: str, tokens: dict[str, str]) -> str:
        result = str(value)
        for token, label in sorted(
            tokens.items(), key=lambda item: len(item[1]), reverse=True
        ):
            result = result.replace(label, token)
        return result

    @staticmethod
    def _restore_speaker_tokens(
        episodes: list[EpisodeDraft], tokens: dict[str, str]
    ) -> None:
        for episode in episodes:
            for token, label in tokens.items():
                episode.text = episode.text.replace(token, label)
                episode.participants = [
                    participant.replace(token, label)
                    for participant in episode.participants
                ]
                episode.location_text = episode.location_text.replace(token, label)
                episode.story_time_text = episode.story_time_text.replace(token, label)
                episode.epistemic_note = episode.epistemic_note.replace(token, label)

    @staticmethod
    def _complete_participants_from_speaker_tokens(
        episodes: list[EpisodeDraft], tokens: dict[str, str]
    ) -> None:
        token_pattern = re.compile(r"SPEAKER_(?:\d{3}|[A-F0-9]{8})")
        for episode in episodes:
            seen = {participant.casefold() for participant in episode.participants}
            for matched in token_pattern.finditer(episode.text):
                token = matched.group(0)
                if token not in tokens or token.casefold() in seen:
                    continue
                episode.participants.append(token)
                seen.add(token.casefold())

    @staticmethod
    def _complete_participants_from_source_speakers(
        episodes: list[EpisodeDraft], speaker_labels: list[str]
    ) -> None:
        """Recover participants from literal Source labels after summarisation.

        The plain-text profile gives the model ordinary multilingual labels,
        not a machine codebook.  Parenthesised aliases are structural adapter
        output, so matching any literal alias back to its full Source label is
        deterministic and does not require another model call.
        """

        parenthetical = re.compile(
            r"^\s*(?P<name>.+?)\s*[（(](?P<aliases>[^）)]+)[）)]\s*$"
        )
        aliases_by_label: list[tuple[str, list[str]]] = []
        for label in speaker_labels:
            aliases = [label]
            matched = parenthetical.fullmatch(label)
            if matched is not None:
                aliases.append(matched.group("name").strip())
                aliases.extend(
                    value.strip()
                    for value in re.split(
                        r"\s*[/／]\s*", matched.group("aliases")
                    )
                    if value.strip()
                )
            aliases_by_label.append((label, list(dict.fromkeys(aliases))))

        for episode in episodes:
            folded = episode.text.casefold()
            matched_labels: list[tuple[int, str]] = []
            for label, aliases in aliases_by_label:
                positions = [
                    folded.find(alias.casefold())
                    for alias in aliases
                    if alias and alias.casefold() in folded
                ]
                if positions:
                    matched_labels.append((min(positions), label))
            seen = {value.casefold() for value in episode.participants}
            for _position, label in sorted(matched_labels):
                if label.casefold() not in seen:
                    episode.participants.append(label)
                    seen.add(label.casefold())

    @staticmethod
    def _source_scoped_self_containment_errors(
        episodes: list[EpisodeDraft],
    ) -> list[str]:
        """Reject only high-confidence fragments in otherwise plain prose."""

        closing_fragment = re.compile(r"^[”’」』】）》〉）)\]]")
        unbound_pronoun = re.compile(
            r"^(?:(?:她|他|它)(?:们|們)?(?:认为|認為|表示|希望|知道|决定|決定|"
            r"觉得|覺得|发现|發現|注意|承认|承認|询问|詢問|回答|解释|解釋|"
            r"建议|建議|要求|担心|擔心|意识|意識|计划|計劃|试图|試圖|想|说|說|"
            r"拒绝|拒絕|同意)|其(?:认为|認為|表示|希望|决定|決定)|"
            r"彼女は|彼は|彼らは|(?:she|he|they|it)\b|그녀는|그는|그들은)",
            re.IGNORECASE,
        )
        errors: list[str] = []
        for index, episode in enumerate(episodes):
            text = episode.text.strip()
            if closing_fragment.match(text):
                errors.append(
                    f"episode {index}: text starts with a closing fragment"
                )
            elif not episode.participants and unbound_pronoun.match(text):
                errors.append(
                    f"episode {index}: unbound leading pronoun is not self-contained"
                )
        return errors

    @staticmethod
    def _apply_speaker_attribution_provenance(
        episodes: list[EpisodeDraft],
    ) -> None:
        """Downgrade content-led subjects without guessing a bad speaker tag.

        A dialogue can directly prove that the labelled actor spoke.  A claim
        led by some other named person is still useful source content, but is
        persisted as reported rather than silently upgraded to direct fact.
        """

        note = "正文首要人物不是证据记录的声明发言者"
        for episode in episodes:
            if not episode.participants:
                continue
            declared = {
                value.strip().casefold()
                for value in re.findall(
                    r"\[speaker_raw:\s*([^\]]+)\]",
                    "\n".join(episode.evidence_quotes),
                )
                if value.strip()
                and value.strip().casefold()
                not in {"???", "unknown speaker", "未知发言者", "未标注发言者"}
            }
            if (
                not declared
                or episode.participants[0].strip().casefold() in declared
                or episode.epistemic_status not in {"observed", "asserted"}
            ):
                continue
            episode.epistemic_status = "reported"
            episode.confidence = min(episode.confidence, 0.8)
            existing = episode.epistemic_note.strip()
            if note not in existing:
                episode.epistemic_note = (f"{existing}；{note}" if existing else note)[
                    :1_000
                ]

    @classmethod
    def _literal_reference_episodes(
        cls,
        source_lines: list[str],
        timeline_scope: str,
    ) -> list[EpisodeDraft] | None:
        """Use already curated reference records without summarising them.

        The adapter marks non-dialogue, repeated decorated records as a
        reference document.  Their prose is already the desired memory text;
        asking a model to paraphrase it can only omit negation or uncertainty.
        This parser consumes only that structural contract.  Unmarked prose,
        dialogue and partially structured Sources return ``None`` and keep the
        normal semantic extraction path.
        """

        record_starts = [
            index
            for index, line in enumerate(source_lines)
            if line.startswith("[record:")
        ]
        if not record_starts:
            return None
        records: list[tuple[int, int, list[str]]] = []
        for position, start in enumerate(record_starts):
            end = (
                record_starts[position + 1]
                if position + 1 < len(record_starts)
                else len(source_lines)
            )
            records.append((start, end, source_lines[start:end]))
        episodes: list[EpisodeDraft] = []
        language_prefix = re.compile(
            r"^(?:zh-CN|zh-TW|ja|en|ko|th|unknown):\s*",
            re.IGNORECASE,
        )
        decorated = re.compile(
            r"^【(?P<label>[^】\n]{1,180})】\s*(?P<body>.*)$",
            re.DOTALL,
        )
        for start, end, record_lines in records:
            if "[document_style: reference]" not in record_lines:
                return None
            payload_lines = [
                line for line in record_lines[1:] if not line.lstrip().startswith("[")
            ]
            if not payload_lines:
                continue
            payload_lines[0] = language_prefix.sub("", payload_lines[0], count=1)
            payload = "\n".join(payload_lines).strip()
            matched = decorated.fullmatch(payload)
            if not matched:
                return None
            body = matched.group("body").strip()
            if not body:
                continue
            episodes.append(
                EpisodeDraft(
                    text=body,
                    timeline_scope=timeline_scope,
                    confidence=0.98,
                    evidence_quotes=["\n".join(record_lines)],
                    evidence_spans=[(start + 1, end)],
                )
            )
        return episodes or None

    @staticmethod
    def _single_pass_evidence_errors(
        source_lines: list[str], episodes: list[EpisodeDraft]
    ) -> list[str]:
        errors: list[str] = []
        record_starts = [
            number
            for number, line in enumerate(source_lines, 1)
            if line.startswith("[record:")
        ]
        record_ranges = [
            (
                start,
                record_starts[index + 1] - 1
                if index + 1 < len(record_starts)
                else len(source_lines),
            )
            for index, start in enumerate(record_starts)
        ]

        def align_to_records(start: int, end: int) -> tuple[int, int]:
            aligned_start, aligned_end = start, end
            for record_start, record_end in record_ranges:
                if record_start <= start <= record_end:
                    aligned_start = record_start
                if record_start <= end <= record_end:
                    aligned_end = record_end
                    break
            return aligned_start, aligned_end

        for index, episode in enumerate(episodes):
            if not episode.evidence_spans:
                errors.append(f"episode {index}: evidence_spans is required")
                continue
            if len(episode.evidence_spans) > 8:
                errors.append(f"episode {index}: evidence_spans exceeds 8 ranges")
            reconstructed: list[str] = []
            total_lines = 0
            normalized_spans: list[tuple[int, int]] = []
            for span_index, (start, end) in enumerate(episode.evidence_spans):
                if (
                    1 <= start <= len(source_lines)
                    and len(source_lines) < end <= len(source_lines) + 2
                ):
                    end = len(source_lines)
                if 1 <= start <= end <= len(source_lines) and record_ranges:
                    start, end = align_to_records(start, end)
                episode.evidence_spans[span_index] = (start, end)
                if start < 1 or end < start or end > len(source_lines):
                    errors.append(
                        f"episode {index}: evidence span {span_index} is out of range"
                    )
                    continue
                if (start, end) in normalized_spans:
                    continue
                normalized_spans.append((start, end))
                span_lines = end - start + 1
                total_lines += span_lines
                if span_lines > 64:
                    errors.append(
                        f"episode {index}: evidence span {span_index} exceeds 64 lines"
                    )
                selected = source_lines[start - 1 : end]
                informative = [
                    line
                    for line in selected
                    if not line.lstrip().startswith("[")
                    and not re.fullmatch(r"[-—_=\s【】]+", line)
                ]
                if not informative or len(re.sub(r"\s+", "", "".join(informative))) < 4:
                    errors.append(
                        f"episode {index}: evidence span {span_index} has no informative Source line"
                    )
                reconstructed.append("\n".join(selected))
            episode.evidence_spans = normalized_spans
            if total_lines > 128:
                errors.append(f"episode {index}: evidence_spans exceed 128 total lines")
            episode.evidence_quotes = reconstructed
        return errors

    @staticmethod
    def _single_pass_coverage_errors(
        source_lines: list[str],
        episodes: list[EpisodeDraft],
        *,
        minimum_run: int = 6,
    ) -> list[str]:
        """Reject long, meaningful holes between accepted evidence spans.

        The extractor may legitimately omit titles, ellipses and short reaction
        lines.  It may not silently skip a sustained dialogue turn merely
        because the surrounding scenes were summarized.  This deterministic
        gate checks coverage only; it never invents an Episode boundary or a
        fact.
        """

        return [
            "episode evidence coverage: uncovered meaningful Source lines "
            f"{start}-{end} ({count} informative lines)"
            for start, end, count in MemoryExtractor._single_pass_uncovered_ranges(
                source_lines,
                episodes,
                minimum_run=minimum_run,
            )
        ]

    @staticmethod
    def _single_pass_uncovered_ranges(
        source_lines: list[str],
        episodes: list[EpisodeDraft],
        *,
        minimum_run: int = 6,
        minimum_substantive_chars: int = 32,
    ) -> list[tuple[int, int, int]]:
        covered: set[int] = set()
        for episode in episodes:
            for start, end in episode.evidence_spans:
                if 1 <= start <= end <= len(source_lines):
                    covered.update(range(start, end + 1))

        def payload_size(line: str) -> int:
            stripped = line.strip()
            if not stripped or stripped.startswith("["):
                return 0
            matched = re.match(r"^[^:\n]{1,80}:\s*(.*)$", stripped)
            payload = matched.group(1) if matched else stripped
            return sum(character.isalnum() for character in payload)

        def informative(line: str) -> bool:
            return payload_size(line) >= 3

        # Normalized Sources expose stable record boundaries.  For those
        # inputs, coverage is a record-level contract: one long dialogue turn
        # is one unit, not a dozen unrelated "lines", and a cited part of that
        # turn proves that the extractor considered the record.  Plain text
        # passed directly to this helper keeps the historical line behavior.
        record_starts = [
            number
            for number, line in enumerate(source_lines, 1)
            if line.startswith("[record:")
        ]
        if record_starts:
            records: list[tuple[int, int, int, bool]] = []
            for index, start in enumerate(record_starts):
                end = (
                    record_starts[index + 1] - 1
                    if index + 1 < len(record_starts)
                    else len(source_lines)
                )
                lines = source_lines[start - 1 : end]
                size = sum(payload_size(line) for line in lines)
                has_speaker = any(
                    line.startswith("[speaker_raw:")
                    and not line.startswith("[speaker_raw: ???")
                    for line in lines
                )
                records.append((start, end, size, has_speaker))

            # In a transcript, unlabelled records before the first actor are a
            # document preamble (title, provenance, branch notes), not events.
            # This is inferred only from normalized structure and does not
            # depend on corpus names or story vocabulary.
            speaker_record_count = sum(record[3] for record in records)
            front_matter_indexes = {
                index
                for index, (start, end, _size, _has_speaker) in enumerate(records)
                if any(
                    line.startswith("[document_role: front_matter]")
                    for line in source_lines[start - 1 : end]
                )
            }
            reference_record_indexes = {
                index
                for index, (start, end, _size, _has_speaker) in enumerate(records)
                if any("[资料类型" in line for line in source_lines[start - 1 : end])
            }
            first_speaker = next(
                (index for index, record in enumerate(records) if record[3]),
                0,
            )
            if len(reference_record_indexes) >= 2:
                ignored_prefix = set(range(len(records))) - reference_record_indexes
            else:
                ignored_prefix = (
                    set(range(first_speaker)) if speaker_record_count >= 3 else set()
                )
            ignored_prefix.update(front_matter_indexes)

            uncovered: list[tuple[int, int, int, int]] = []
            for index, (start, end, size, _has_speaker) in enumerate(records):
                if index in ignored_prefix or size < 3:
                    continue
                if any(line in covered for line in range(start, end + 1)):
                    uncovered.append((-1, -1, -1, -1))
                    continue
                uncovered.append((start, end, size, index))

            ranges: list[tuple[int, int, int]] = []
            current: list[tuple[int, int, int, int]] = []

            def flush_records() -> None:
                if len(current) >= max(1, int(minimum_run)):
                    ranges.append((current[0][0], current[-1][1], len(current)))
                current.clear()

            for record in uncovered:
                if record[0] < 0:
                    flush_records()
                else:
                    current.append(record)
            flush_records()

            already_flagged_records = {
                record_index
                for start, end, _count in ranges
                for record_start, record_end, _size, record_index in uncovered
                if record_index >= 0 and record_start >= start and record_end <= end
            }
            for start, end, size, record_index in uncovered:
                if (
                    record_index >= 0
                    and record_index not in already_flagged_records
                    and size >= max(1, int(minimum_substantive_chars))
                ):
                    ranges.append((start, end, 1))
            ranges.sort(key=lambda value: (value[0], value[1]))
            return ranges

        ranges: list[tuple[int, int, int]] = []
        current: list[int] = []

        def flush() -> None:
            if len(current) < max(1, int(minimum_run)):
                current.clear()
                return
            ranges.append((current[0], current[-1], len(current)))
            current.clear()

        for line_number, line in enumerate(source_lines, 1):
            if line_number in covered:
                if informative(line):
                    flush()
                continue
            if informative(line):
                current.append(line_number)
        flush()
        already_flagged = {
            line_number
            for start, end, _count in ranges
            for line_number in range(start, end + 1)
        }
        for line_number, line in enumerate(source_lines, 1):
            if line_number in covered or line_number in already_flagged:
                continue
            if payload_size(line) >= max(1, int(minimum_substantive_chars)):
                ranges.append((line_number, line_number, 1))
        ranges.sort(key=lambda value: (value[0], value[1]))
        return ranges

    @staticmethod
    def _indexed_coverage_excerpt(
        indexed_source: str,
        required_ranges: list[tuple[int, int]],
        *,
        context_lines: int = 5,
    ) -> str:
        """Focus a repair call on uncovered evidence while retaining line IDs."""

        lines = indexed_source.splitlines()
        selected: set[int] = set()
        for start, end in required_ranges:
            selected.update(
                range(
                    max(1, start - max(0, int(context_lines))),
                    min(len(lines), end + max(0, int(context_lines))) + 1,
                )
            )
        if not selected or len(selected) >= max(1, int(len(lines) * 0.75)):
            return indexed_source
        return "\n".join(lines[number - 1] for number in sorted(selected))

    @classmethod
    def _remove_unreferenced_unsupported_participants(
        cls,
        episodes: list[EpisodeDraft],
        logger: JsonlEventLogger | None = None,
    ) -> None:
        """Drop harmless over-inclusive participants, never named claims.

        If a name is absent from both the Episode prose and its evidence, it is
        metadata noise and can be removed deterministically.  A name present in
        the prose remains a factual claim and must still be rejected when its
        evidence does not support it.
        """

        unknown_markers = {
            "???",
            "[username]",
            "未标注发言者",
            "未知发言者",
            "unknown speaker",
        }
        for episode_index, episode in enumerate(episodes):
            evidence_folded = "\n".join(episode.evidence_quotes).casefold()
            text_folded = episode.text.casefold()
            kept: list[str] = []
            removed: list[str] = []
            for participant in episode.participants:
                normalized = participant.strip()
                parts = [
                    part.strip()
                    for part in cls._SOURCE_NAME_SEPARATOR.split(normalized)
                    if part.strip()
                ]
                grounded = (
                    normalized.casefold() in unknown_markers
                    or normalized.casefold() in evidence_folded
                    or (
                        bool(parts)
                        and all(
                            part.casefold() in unknown_markers
                            or part.casefold() in evidence_folded
                            for part in parts
                        )
                    )
                )
                mentioned = normalized.casefold() in text_folded or any(
                    part.casefold() in text_folded for part in parts
                )
                if not grounded and not mentioned:
                    removed.append(participant)
                else:
                    kept.append(participant)
            if removed:
                episode.participants = kept
                if logger:
                    logger.emit(
                        "unreferenced_participants_removed",
                        episode_index=episode_index,
                        participants=removed,
                    )

    def _supplement_single_pass_coverage(
        self,
        *,
        source_lines: list[str],
        indexed_source: str,
        timeline_scope: str,
        episodes: list[EpisodeDraft],
        model_name: str | None,
        speaker_tokens: dict[str, str] | None = None,
        masked_source_lines: list[str] | None = None,
    ) -> tuple[list[EpisodeDraft], list[str]]:
        gaps = self._single_pass_uncovered_ranges(source_lines, episodes)
        if not gaps:
            return episodes, []
        required_ranges = [(start, end) for start, end, _count in gaps]
        focused_source = self._indexed_coverage_excerpt(indexed_source, required_ranges)
        kwargs = {"allow_fallback": False, "max_retries": 0}
        if model_name:
            kwargs["model"] = model_name
        try:
            active_prompt = single_pass_episode_prompt(
                focused_source,
                timeline_scope,
                required_ranges=required_ranges,
            )
            if self._uses_natural_text_output():
                response = self._request_natural_text(
                    SINGLE_PASS_EPISODE_SYSTEM,
                    active_prompt,
                    **kwargs,
                )
                supplements, parse_errors = parse_episode_text(
                    response,
                    timeline_scope=timeline_scope,
                )
            else:
                payload = self.model.chat_json(
                    SINGLE_PASS_EPISODE_SYSTEM,
                    active_prompt,
                    **kwargs,
                )
                supplements, parse_errors = parse_episode_payload(payload)
        except Exception as exc:
            return episodes, [f"coverage supplement request failed: {exc}"]
        supplements = [
            supplement
            for supplement in supplements
            if any(
                span_start <= gap_end and span_end >= gap_start
                for span_start, span_end in supplement.evidence_spans
                for gap_start, gap_end in required_ranges
            )
        ]
        errors: list[str] = []
        if not supplements:
            errors.extend(parse_errors)
            errors.append(
                "coverage supplement returned no Episode intersecting a required gap"
            )
        evidence_errors = self._single_pass_evidence_errors(source_lines, supplements)
        if not evidence_errors:
            expansion_count = self._expand_adjacent_speaker_evidence(
                masked_source_lines or source_lines,
                supplements,
                speaker_tokens or {},
            )
            if expansion_count:
                evidence_errors = self._single_pass_evidence_errors(
                    source_lines, supplements
                )
                if self.logger:
                    self.logger.emit(
                        "adjacent_speaker_evidence_expanded",
                        record_count=expansion_count,
                        mode="coverage_supplement",
                    )
        token_errors = self._single_pass_speaker_token_errors(
            masked_source_lines or source_lines,
            supplements,
            speaker_tokens or {},
        )
        errors.extend(token_errors)
        self._complete_participants_from_speaker_tokens(
            supplements, speaker_tokens or {}
        )
        self._restore_speaker_tokens(supplements, speaker_tokens or {})
        if not evidence_errors:
            self._derive_episode_fields_from_evidence(supplements)
            self._apply_speaker_attribution_provenance(supplements)
            if self._uses_natural_text_output():
                supplements = self._remove_structural_front_matter(supplements)
            self._repair_single_pass_evidence_name_typos(supplements, self.logger)
            self._remove_unreferenced_unsupported_participants(supplements, self.logger)
            if self._uses_natural_text_output():
                self.finalize_episode_drafts(
                    "\n".join(source_lines), supplements, self.logger
                )
        errors.extend(evidence_errors)
        audited_profile = self.episode_extraction_profile in {
            "single_pass_audited",
            "document_map_assisted",
            "document_map_contextual",
            "adaptive_anchor_map",
        }
        # A supplement is merged into the same entailment-audited batch as the
        # primary Episodes.  Let that audit repair a participant named in the
        # prose but absent from the local evidence, just as the primary path
        # does.  Evidence-only profiles must continue to fail closed.
        if not audited_profile:
            errors.extend(self._single_pass_participant_evidence_errors(supplements))
        if errors or not supplements:
            return episodes, errors or ["coverage supplement returned no Episode"]
        merged = sorted(
            [*episodes, *supplements],
            key=lambda draft: min(
                (start for start, _end in draft.evidence_spans),
                default=len(source_lines) + 1,
            ),
        )
        remaining = self._single_pass_coverage_errors(source_lines, merged)
        if remaining:
            return episodes, remaining
        if self.logger:
            self.logger.emit(
                "single_pass_coverage_supplemented",
                required_ranges=required_ranges,
                added_episode_count=len(supplements),
            )
        return merged, []

    @classmethod
    def _single_pass_participant_evidence_errors(
        cls,
        episodes: list[EpisodeDraft],
    ) -> list[str]:
        """Require every declared identity to occur in its own evidence.

        Source-wide grounding is too weak for long transcripts: two different
        people can both occur in one Source while only one participates in the
        selected event.  This check deliberately runs before any rewrite and
        rejects the model response instead of silently turning a wrong person
        into durable direct evidence.
        """

        unknown_markers = {
            "???",
            "[username]",
            "未标注发言者",
            "未知发言者",
            "unknown speaker",
        }
        errors: list[str] = []
        for episode_index, episode in enumerate(episodes):
            evidence_folded = "\n".join(episode.evidence_quotes).casefold()
            for participant in episode.participants:
                normalized = participant.strip()
                if not normalized:
                    continue
                parts = [
                    part.strip()
                    for part in cls._SOURCE_NAME_SEPARATOR.split(normalized)
                    if part.strip()
                ]
                grounded = (
                    normalized.casefold() in unknown_markers
                    or normalized.casefold() in evidence_folded
                    or (
                        bool(parts)
                        and all(
                            part.casefold() in unknown_markers
                            or part.casefold() in evidence_folded
                            for part in parts
                        )
                    )
                )
                if not grounded:
                    errors.append(
                        "episode participant evidence grounding: candidate "
                        f"{episode_index} contains a name absent from its "
                        f"evidence_spans: {normalized}"
                    )
        return errors

    @staticmethod
    def _single_pass_speaker_token_errors(
        source_lines: list[str],
        episodes: list[EpisodeDraft],
        speaker_tokens: dict[str, str],
    ) -> list[str]:
        """Bind model-used speaker markers to each Episode's cited lines."""

        if not speaker_tokens:
            return []
        token_pattern = re.compile(r"SPEAKER_\d{3}")
        known = set(speaker_tokens)
        errors: list[str] = []
        for episode_index, episode in enumerate(episodes):
            cited: set[str] = set()
            for start, end in episode.evidence_spans:
                if 1 <= start <= end <= len(source_lines):
                    cited.update(
                        token
                        for line in source_lines[start - 1 : end]
                        for token in token_pattern.findall(line)
                        if token in known
                    )
            used = set(token_pattern.findall(episode.text))
            unknown = used - known
            outside = used - cited
            if unknown:
                errors.append(
                    "episode speaker grounding: candidate "
                    f"{episode_index} uses unknown markers {sorted(unknown)}"
                )
            if outside:
                errors.append(
                    "episode speaker grounding: candidate "
                    f"{episode_index} uses markers outside evidence "
                    f"{sorted(outside)}"
                )
            if cited and not (used & cited):
                errors.append(
                    "episode speaker grounding: candidate "
                    f"{episode_index} must name at least one cited speaker marker "
                    f"from {sorted(cited)}"
                )
        return errors

    @staticmethod
    def _expand_adjacent_speaker_evidence(
        source_lines: list[str],
        episodes: list[EpisodeDraft],
        speaker_tokens: dict[str, str],
    ) -> int:
        """Include one adjacent record when a cited turn resolves its actor.

        This only widens evidence.  It does not infer that two labels are the
        same person or change Episode prose.  A marker farther than one record
        away remains an error.
        """

        if not speaker_tokens:
            return 0
        starts = [
            index
            for index, line in enumerate(source_lines, 1)
            if line.startswith("[record:")
        ]
        if not starts:
            return 0
        records = [
            (
                start,
                starts[position + 1] - 1
                if position + 1 < len(starts)
                else len(source_lines),
            )
            for position, start in enumerate(starts)
        ]
        line_record = {
            line: record_index
            for record_index, (start, end) in enumerate(records)
            for line in range(start, end + 1)
        }
        token_pattern = re.compile(r"SPEAKER_\d{3}")
        token_records: dict[str, set[int]] = {token: set() for token in speaker_tokens}
        for line_number, line in enumerate(source_lines, 1):
            record_index = line_record.get(line_number)
            if record_index is None:
                continue
            for token in token_pattern.findall(line):
                if token in token_records:
                    token_records[token].add(record_index)

        expanded = 0
        for episode in episodes:
            used = set(token_pattern.findall(episode.text)) & set(speaker_tokens)
            cited_records = {
                record_index
                for start, end in episode.evidence_spans
                for line in range(start, end + 1)
                if (record_index := line_record.get(line)) is not None
            }
            cited_tokens = {
                token
                for record_index in cited_records
                for line in source_lines[
                    records[record_index][0] - 1 : records[record_index][1]
                ]
                for token in token_pattern.findall(line)
            }
            additions: set[int] = set()
            for token in used - cited_tokens:
                candidates = token_records.get(token, set())
                nearest = min(
                    (
                        (abs(candidate - cited), candidate)
                        for candidate in candidates
                        for cited in cited_records
                    ),
                    default=None,
                )
                if nearest is not None and nearest[0] == 1:
                    additions.add(nearest[1])
            if not additions:
                continue
            ranges = [*episode.evidence_spans]
            ranges.extend(records[index] for index in sorted(additions))
            merged: list[tuple[int, int]] = []
            for start, end in sorted(ranges):
                if merged and start <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            if len(merged) <= 4:
                episode.evidence_spans = merged
                expanded += len(additions)
        return expanded

    @classmethod
    def _repair_single_pass_evidence_name_typos(
        cls,
        episodes: list[EpisodeDraft],
        logger: JsonlEventLogger | None = None,
    ) -> None:
        """Repair only uniquely Source-grounded copy errors before rejection.

        Each Episode is repaired against its own accepted evidence, never the
        whole Source.  This handles a final-character script substitution in a
        copied speaker label without allowing a nearby character elsewhere in
        the document to legitimize a wrong participant.
        """

        for episode in episodes:
            evidence_scope = "\n".join(episode.evidence_quotes).strip()
            if evidence_scope:
                cls._repair_source_name_typos(evidence_scope, [episode], logger)

    @staticmethod
    def _validated_document_map_payload(
        payload: object,
        expected_line_counts: dict[int, int],
    ) -> tuple[str, dict[int, dict[str, object]]]:
        if not isinstance(payload, dict):
            raise ValueError("document map must be an object")
        overview = str(payload.get("overview", "")).strip()
        if not overview:
            raise ValueError("document map overview is required")
        raw_contexts = payload.get("segment_contexts")
        if not isinstance(raw_contexts, list):
            raise ValueError("segment_contexts must be a list")
        contexts: dict[int, dict[str, object]] = {}
        list_fields = (
            "participants",
            "timeline_notes",
            "unresolved",
        )
        for position, item in enumerate(raw_contexts):
            if not isinstance(item, dict):
                raise ValueError(f"segment_contexts[{position}] must be an object")
            raw_index = item.get("segment_index")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError(
                    f"segment_contexts[{position}].segment_index must be an integer"
                )
            segment_index = int(raw_index)
            if segment_index in contexts:
                raise ValueError(
                    f"duplicate document map segment_index {segment_index}"
                )
            role = str(item.get("role_in_document", "")).strip()
            if not role:
                raise ValueError(
                    f"segment_contexts[{position}].role_in_document is required"
                )
            normalized: dict[str, object] = {
                "segment_index": segment_index,
                "role_in_document": role[:1_200],
            }
            stages = item.get("event_stages")
            if not isinstance(stages, list) or not stages:
                raise ValueError(
                    f"segment_contexts[{position}].event_stages must be a non-empty list"
                )
            normalized_stages: list[dict[str, object]] = []
            line_count = expected_line_counts.get(segment_index, 0)
            for stage_position, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    raise ValueError(
                        f"segment_contexts[{position}].event_stages[{stage_position}] must be an object"
                    )
                raw_stage_index = stage.get("stage_index")
                if raw_stage_index != stage_position:
                    raise ValueError(
                        f"segment {segment_index} stage_index must be contiguous from 0"
                    )
                start = stage.get("start_line")
                end = stage.get("end_line")
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (start, end)
                ):
                    raise ValueError(
                        f"segment {segment_index} stage {stage_position} line bounds must be integers"
                    )
                start, end = int(start), int(end)
                if start < 1 or end < start or end > line_count:
                    raise ValueError(
                        f"segment {segment_index} stage {stage_position} line bounds are out of range"
                    )
                if end - start + 1 > 64:
                    raise ValueError(
                        f"segment {segment_index} stage {stage_position} exceeds 64 lines"
                    )
                hint = str(stage.get("hint", "")).strip()
                if not hint:
                    raise ValueError(
                        f"segment {segment_index} stage {stage_position} hint is required"
                    )
                time_mode = str(stage.get("time_mode", "unknown")).strip().casefold()
                if time_mode not in {
                    "current",
                    "past",
                    "memory",
                    "reported",
                    "mixed",
                    "unknown",
                }:
                    raise ValueError(
                        f"segment {segment_index} stage {stage_position} has invalid time_mode"
                    )
                stage_unresolved = stage.get("unresolved", [])
                if not isinstance(stage_unresolved, list) or not all(
                    isinstance(value, str) for value in stage_unresolved
                ):
                    raise ValueError(
                        f"segment {segment_index} stage {stage_position} unresolved must be a string list"
                    )
                normalized_stages.append(
                    {
                        "stage_index": stage_position,
                        "start_line": start,
                        "end_line": end,
                        "hint": hint[:800],
                        "time_mode": time_mode,
                        "unresolved": [
                            value.strip()[:800]
                            for value in stage_unresolved[:12]
                            if value.strip()
                        ],
                    }
                )
            normalized["event_stages"] = normalized_stages
            for field_name in list_fields:
                values = item.get(field_name, [])
                if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise ValueError(
                        f"segment_contexts[{position}].{field_name} must be a string list"
                    )
                normalized[field_name] = [
                    value.strip()[:800] for value in values[:16] if value.strip()
                ]
            contexts[segment_index] = normalized
        expected_indexes = set(expected_line_counts)
        returned_indexes = set(contexts)
        if returned_indexes != expected_indexes:
            missing = sorted(expected_indexes - returned_indexes)
            extra = sorted(returned_indexes - expected_indexes)
            raise ValueError(
                f"document map indexes mismatch; missing={missing}, extra={extra}"
            )
        return overview[:2_000], contexts

    @staticmethod
    def _validated_document_anchor_payload(
        payload: object,
        segment_texts: dict[int, str],
    ) -> dict[int, dict[str, object]]:
        if not isinstance(payload, dict):
            raise ValueError("document anchor map must be an object")
        raw_anchors = payload.get("segment_anchors")
        if not isinstance(raw_anchors, list):
            raise ValueError("segment_anchors must be a list")
        valid_roles = {
            "setup",
            "continuation",
            "turn",
            "resolution",
            "epilogue",
            "unknown",
        }
        valid_time_modes = {
            "current",
            "past",
            "memory",
            "reported",
            "mixed",
            "unknown",
        }
        contexts: dict[int, dict[str, object]] = {}
        for position, raw_item in enumerate(raw_anchors):
            if not isinstance(raw_item, dict):
                raise ValueError(f"segment_anchors[{position}] must be an object")
            raw_index = raw_item.get("segment_index")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError(
                    f"segment_anchors[{position}].segment_index must be an integer"
                )
            segment_index = int(raw_index)
            if segment_index in contexts:
                raise ValueError(
                    f"duplicate document anchor segment_index {segment_index}"
                )
            if segment_index not in segment_texts:
                raise ValueError(
                    f"unexpected document anchor segment_index {segment_index}"
                )
            role = str(raw_item.get("role_in_document", "unknown")).strip().casefold()
            if role not in valid_roles:
                role = "unknown"
            time_mode = str(raw_item.get("time_mode", "unknown")).strip().casefold()
            if time_mode not in valid_time_modes:
                time_mode = "unknown"
            segment_text = segment_texts[segment_index]
            normalized_lists: dict[str, list[str]] = {}
            for field_name, maximum in (
                ("anchor_terms", 8),
                ("participants", 16),
            ):
                values = raw_item.get(field_name, [])
                if not isinstance(values, list) or not all(
                    isinstance(value, str) for value in values
                ):
                    raise ValueError(
                        f"segment_anchors[{position}].{field_name} must be a string list"
                    )
                normalized: list[str] = []
                for value in values[:maximum]:
                    literal = value.strip()
                    if not literal:
                        continue
                    if literal not in segment_text:
                        # Navigation is optional and never evidence.  A model
                        # that drops punctuation or translates one name should
                        # lose only that anchor, not the whole import.
                        continue
                    if literal not in normalized:
                        normalized.append(literal[:240])
                normalized_lists[field_name] = normalized
            contexts[segment_index] = {
                "segment_index": segment_index,
                "role_in_document": role,
                "time_mode": time_mode,
                **normalized_lists,
            }
        expected = set(segment_texts)
        returned = set(contexts)
        if returned != expected:
            raise ValueError(
                "document anchor indexes mismatch; "
                f"missing={sorted(expected - returned)}, "
                f"extra={sorted(returned - expected)}"
            )
        return contexts

    def _document_anchor_group(
        self,
        source_key: str,
        segments: list[dict[str, object]],
    ) -> dict[int, dict[str, object]]:
        segment_texts = {
            int(item["segment_index"]): str(item["text"]) for item in segments
        }
        prompt = document_anchor_map_prompt(source_key, segments)
        fallback = self._fallback_model()
        attempts: list[tuple[str | None, str]] = [
            (None, "primary"),
            (None, "primary_correction"),
        ]
        if fallback:
            attempts.append((fallback, "reasoning_fallback"))
        errors: list[str] = []
        for model_name, label in attempts:
            active_prompt = prompt
            if errors:
                active_prompt = document_anchor_retry_prompt(prompt, errors)
            kwargs: dict[str, object] = {
                "allow_fallback": False,
                "max_retries": 0,
            }
            if model_name:
                kwargs["model"] = model_name
            try:
                payload = self.model.chat_json(
                    DOCUMENT_ANCHOR_MAP_SYSTEM,
                    active_prompt,
                    **kwargs,
                )
                return self._validated_document_anchor_payload(
                    payload,
                    segment_texts,
                )
            except Exception as exc:
                errors = [f"{label}: {exc}"]
                if self.logger:
                    self.logger.emit(
                        "document_anchor_map_attempt_rejected",
                        attempt=label,
                        source_key=source_key,
                        segment_indexes=sorted(segment_texts),
                        errors=errors,
                    )
        raise ValueError("document anchor map failed: " + "; ".join(errors))

    def build_document_anchor_map(
        self,
        source_key: str,
        segments: list[tuple[int, str]],
    ) -> dict[int, str]:
        """Build a literal-only map when one file crossed Source boundaries."""

        if len(segments) < 2:
            return {}
        if self._uses_natural_text_output():
            return self._deterministic_document_contexts(source_key, segments)
        groups: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        current_chars = 0
        max_group_chars = 32_000
        for segment_index, raw_text in segments:
            source_lines, indexed_source = self._single_pass_source_lines(
                self.compact_source_for_reasoning(raw_text)
            )
            item = {
                "segment_index": int(segment_index),
                "line_count": len(source_lines),
                "text": indexed_source,
            }
            item_chars = len(indexed_source)
            if current and current_chars + item_chars > max_group_chars:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += item_chars
        if current:
            groups.append(current)

        contexts: dict[int, dict[str, object]] = {}
        for group in groups:
            contexts.update(self._document_anchor_group(source_key, group))
        literal_anchor_count = sum(
            len(context.get("anchor_terms", [])) for context in contexts.values()
        )
        if literal_anchor_count == 0:
            if self.logger:
                self.logger.emit(
                    "document_anchor_map_empty",
                    source_key=source_key,
                    segment_count=len(segments),
                )
            return {}
        ordered_indexes = sorted(contexts)
        rendered: dict[int, str] = {}
        for segment_index in ordered_indexes:
            if len(ordered_indexes) <= 64:
                visible_indexes = ordered_indexes
            else:
                position = ordered_indexes.index(segment_index)
                visible_indexes = sorted(
                    set(
                        ordered_indexes[:4]
                        + ordered_indexes[max(0, position - 16) : position + 17]
                        + ordered_indexes[-4:]
                    )
                )
            rendered[segment_index] = json.dumps(
                {
                    "map_kind": "extractive_anchor_map",
                    "document_segment_count": len(ordered_indexes),
                    "current_segment_index": segment_index,
                    "document_sequence": [contexts[index] for index in visible_indexes],
                },
                ensure_ascii=False,
            )
        if self.logger:
            self.logger.emit(
                "document_anchor_map_built",
                source_key=source_key,
                segment_count=len(segments),
                group_count=len(groups),
                context_chars={
                    str(index): len(value) for index, value in rendered.items()
                },
            )
        return rendered

    def _deterministic_document_contexts(
        self,
        source_key: str,
        segments: list[tuple[int, str]],
    ) -> dict[int, str]:
        """Expose bounded neighbouring excerpts without another model call.

        This is navigation only.  It deliberately performs no summarisation,
        identity resolution or causal inference, so it cannot become hidden
        evidence.  The Episode model still has to cite the current Source.
        """

        ordered = sorted(
            ((int(index), str(text)) for index, text in segments),
            key=lambda item: item[0],
        )

        def meaningful_lines(value: str) -> list[str]:
            return [
                line.strip()
                for line in self.compact_source_for_reasoning(value).splitlines()
                if line.strip()
                and not line.startswith("[source_key:")
                and not line.startswith("[segment_index:")
                and line != "[speaker_alias_legend]"
            ]

        compact = {index: meaningful_lines(text) for index, text in ordered}
        result: dict[int, str] = {}
        for position, (segment_index, _text) in enumerate(ordered):
            parts = [
                f"同一文档：{source_key}",
                f"当前位置：第 {position + 1}/{len(ordered)} 个片段",
            ]
            if position > 0:
                previous_index = ordered[position - 1][0]
                previous_tail = "\n".join(compact[previous_index][-8:])[-1_200:]
                if previous_tail:
                    parts.append(
                        "上一片段末尾（仅导航，不可作为证据）：\n" + previous_tail
                    )
            if position + 1 < len(ordered):
                next_index = ordered[position + 1][0]
                next_head = "\n".join(compact[next_index][:8])[:1_200]
                if next_head:
                    parts.append("下一片段开头（仅导航，不可作为证据）：\n" + next_head)
            result[segment_index] = "\n\n".join(parts)
        if self.logger:
            self.logger.emit(
                "document_navigation_built",
                mode="deterministic_adjacent_excerpts",
                source_key=source_key,
                segment_count=len(ordered),
                model_calls=0,
                context_chars={
                    str(index): len(value) for index, value in result.items()
                },
            )
        return result

    def _document_map_group(
        self,
        source_key: str,
        segments: list[dict[str, object]],
    ) -> tuple[str, dict[int, dict[str, object]]]:
        expected_line_counts = {
            int(item["segment_index"]): int(item["line_count"]) for item in segments
        }
        expected = set(expected_line_counts)
        prompt = document_map_prompt(source_key, segments)
        fallback = self._fallback_model()
        attempts: list[tuple[str | None, str]] = [
            (None, "primary"),
            (None, "primary_correction"),
        ]
        if fallback:
            attempts.append((fallback, "reasoning_fallback"))
        errors: list[str] = []
        for model_name, label in attempts:
            active_prompt = prompt
            if errors:
                active_prompt = document_map_retry_prompt(prompt, errors)
            kwargs: dict[str, object] = {
                "allow_fallback": False,
                "max_retries": 0,
            }
            if model_name:
                kwargs["model"] = model_name
            try:
                payload = self.model.chat_json(
                    DOCUMENT_MAP_SYSTEM,
                    active_prompt,
                    **kwargs,
                )
                return self._validated_document_map_payload(
                    payload, expected_line_counts
                )
            except Exception as exc:
                errors = [f"{label}: {exc}"]
                if self.logger:
                    self.logger.emit(
                        "document_map_attempt_rejected",
                        attempt=label,
                        source_key=source_key,
                        segment_indexes=sorted(expected),
                        errors=errors,
                    )
        raise ValueError("document map failed: " + "; ".join(errors))

    def build_document_map(
        self,
        source_key: str,
        segments: list[tuple[int, str]],
    ) -> dict[int, str]:
        """Build a transient, hierarchical navigation map for one file."""

        if not segments:
            return {}
        groups: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        current_chars = 0
        max_group_chars = 32_000
        for segment_index, raw_text in segments:
            source_lines, indexed_source = self._single_pass_source_lines(
                self.compact_source_for_reasoning(raw_text)
            )
            item = {
                "segment_index": int(segment_index),
                "line_count": len(source_lines),
                "text": indexed_source,
            }
            item_chars = len(str(item["text"]))
            if current and current_chars + item_chars > max_group_chars:
                groups.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += item_chars
        if current:
            groups.append(current)

        overviews: list[str] = []
        contexts: dict[int, dict[str, object]] = {}
        for group in groups:
            overview, group_contexts = self._document_map_group(source_key, group)
            overviews.append(overview)
            contexts.update(group_contexts)
        full_overview = "\n".join(
            f"Part {index}: {overview}" for index, overview in enumerate(overviews, 1)
        )[:6_000]
        rendered = {
            segment_index: json.dumps(
                {
                    "document_overview": full_overview,
                    "current_segment": context,
                },
                ensure_ascii=False,
            )
            for segment_index, context in contexts.items()
        }
        if self.logger:
            self.logger.emit(
                "document_map_built",
                source_key=source_key,
                segment_count=len(segments),
                group_count=len(groups),
                overview_chars=len(full_overview),
                context_chars={
                    str(index): len(value) for index, value in rendered.items()
                },
            )
        return rendered

    @staticmethod
    def _document_map_stage_spans(
        document_context: str,
    ) -> list[tuple[int, int]]:
        if not document_context.strip():
            return []
        try:
            payload = json.loads(document_context)
            current = payload["current_segment"]
            stages = current["event_stages"]
            return [
                (int(stage["start_line"]), int(stage["end_line"])) for stage in stages
            ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid document map navigation context") from exc

    @staticmethod
    def _apply_entailment_reviews(
        payload: object,
        episodes: list[EpisodeDraft],
    ) -> tuple[int, list[dict[str, object]]]:
        if not isinstance(payload, dict):
            raise ValueError("entailment audit must be an object")
        raw_reviews = payload.get("reviews")
        if not isinstance(raw_reviews, list):
            raise ValueError("entailment reviews must be a list")
        reviews: dict[int, dict[str, object]] = {}
        for position, raw_review in enumerate(raw_reviews):
            if not isinstance(raw_review, dict):
                raise ValueError(f"reviews[{position}] must be an object")
            raw_index = raw_review.get("episode_index")
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ValueError(
                    f"reviews[{position}].episode_index must be an integer"
                )
            episode_index = int(raw_index)
            if episode_index in reviews:
                raise ValueError(f"duplicate entailment episode_index {episode_index}")
            verdict = str(raw_review.get("verdict", "")).strip().casefold()
            if verdict not in {"supported", "revise"}:
                raise ValueError(
                    f"reviews[{position}].verdict must be supported or revise"
                )
            unsupported = raw_review.get("unsupported_claims", [])
            if not isinstance(unsupported, list) or not all(
                isinstance(value, str) for value in unsupported
            ):
                raise ValueError(
                    f"reviews[{position}].unsupported_claims must be a string list"
                )
            revised_text = str(raw_review.get("revised_text", "")).strip()
            if verdict == "revise" and not revised_text:
                raise ValueError(
                    f"reviews[{position}].revised_text is required for revise"
                )
            reviews[episode_index] = {
                "episode_index": episode_index,
                "verdict": verdict,
                "unsupported_claims": [
                    value.strip()[:800] for value in unsupported[:16] if value.strip()
                ],
                "revised_text": revised_text[:4_000],
            }
        expected = set(range(len(episodes)))
        if set(reviews) != expected:
            raise ValueError(
                "entailment review indexes mismatch; "
                f"missing={sorted(expected - set(reviews))}, "
                f"extra={sorted(set(reviews) - expected)}"
            )
        revised_count = 0
        ordered: list[dict[str, object]] = []
        for episode_index in range(len(episodes)):
            review = reviews[episode_index]
            ordered.append(review)
            if review["verdict"] == "revise":
                episodes[episode_index].text = str(review["revised_text"])
                episodes[episode_index].confidence = min(
                    episodes[episode_index].confidence,
                    0.85,
                )
                revised_count += 1
        return revised_count, ordered

    def _audit_single_pass_entailment(
        self,
        episodes: list[EpisodeDraft],
        *,
        model_override: str = "",
        audit_stage: str = "primary_entailment",
    ) -> None:
        items = [
            {
                "episode_index": index,
                "episode_text": episode.text,
                "participants": episode.participants,
                "evidence_origin": episode.evidence_origin,
                "epistemic_status": episode.epistemic_status,
                "evidence": episode.evidence_quotes,
            }
            for index, episode in enumerate(episodes)
        ]
        prompt = episode_entailment_audit_prompt(items)
        fallback = None if model_override else self._fallback_model()
        attempts: list[tuple[str | None, str]] = [
            (model_override or None, "primary"),
            (model_override or None, "primary_correction"),
        ]
        if fallback:
            attempts.append((fallback, "reasoning_fallback"))
        errors: list[str] = []
        for model_name, label in attempts:
            active_prompt = prompt
            if errors:
                active_prompt = entailment_retry_prompt(prompt, errors)
            kwargs: dict[str, object] = {
                "allow_fallback": False,
                "max_retries": 0,
            }
            if model_name:
                kwargs["model"] = model_name
            try:
                if self._uses_natural_text_output():
                    response = self._request_natural_text(
                        EPISODE_ENTAILMENT_AUDIT_SYSTEM,
                        active_prompt,
                        **kwargs,
                    )
                    text_reviews, text_errors = parse_entailment_text(
                        response,
                        expected_indexes=set(range(len(episodes))),
                    )
                    if text_errors:
                        raise ValueError("; ".join(text_errors))
                    payload = {
                        "reviews": [
                            {
                                "episode_index": index,
                                "verdict": review["verdict"],
                                "unsupported_claims": (
                                    [review["unsupported_claims"]]
                                    if review["unsupported_claims"]
                                    else []
                                ),
                                "revised_text": review["revised_text"],
                            }
                            for index, review in sorted(text_reviews.items())
                        ]
                    }
                else:
                    payload = self.model.chat_json(
                        EPISODE_ENTAILMENT_AUDIT_SYSTEM,
                        active_prompt,
                        **kwargs,
                    )
                revised_count, reviews = self._apply_entailment_reviews(
                    payload, episodes
                )
                if self.logger:
                    self.logger.emit(
                        "single_pass_entailment_audit_completed",
                        audit_stage=audit_stage,
                        attempt=label,
                        audit_model=model_override or "primary",
                        episode_count=len(episodes),
                        revised_count=revised_count,
                        reviews=reviews,
                    )
                return
            except Exception as exc:
                errors = [f"{label}: {exc}"]
                if self.logger:
                    self.logger.emit(
                        "single_pass_entailment_audit_rejected",
                        audit_stage=audit_stage,
                        attempt=label,
                        audit_model=model_override or "primary",
                        errors=errors,
                    )
        raise ValueError("single-pass entailment audit failed: " + "; ".join(errors))

    def _extract_episodes_source_scoped_plain(
        self,
        source_text: str,
        timeline_scope: str,
    ) -> tuple[list[EpisodeDraft], list[str]]:
        """Extract prose while the program owns every persistence decision."""

        source_lines, _indexed_source = self._single_pass_source_lines(source_text)
        literal_reference = self._literal_reference_episodes(
            source_lines,
            timeline_scope,
        )
        if literal_reference is not None:
            self._derive_episode_fields_from_evidence(literal_reference)
            self.finalize_episode_drafts(
                source_text,
                literal_reference,
                self.logger,
            )
            self._apply_explicit_segment_evidence_floor(
                source_text,
                literal_reference,
            )
            if self.logger:
                self.logger.emit(
                    "literal_reference_episodes_created",
                    episode_count=len(literal_reference),
                    model_calls=0,
                    exact_text=True,
                )
            return literal_reference, []

        speaker_labels = list(self._speaker_token_map(source_lines).values())
        model_source = self._source_scoped_model_text(source_text)
        if not model_source:
            raise EmptyEpisodeExtraction(
                "source-scoped extraction has no model-visible content"
            )
        prompt = source_scoped_episode_prompt(model_source, timeline_scope)
        system_prompt = SOURCE_SCOPED_EPISODE_SYSTEM
        attempts: list[tuple[str | None, str]] = [(None, "primary")]
        fallback = self._fallback_model()
        if fallback:
            attempts.append((fallback, "reasoning_fallback"))

        errors: list[str] = []
        attempted_models: list[str] = []
        clean_empty_seen = False
        had_nonempty_rejection = False
        for model_name, label in attempts:
            effective_model = model_name or self._primary_reasoning_model()
            if effective_model:
                attempted_models.append(effective_model)
            kwargs = {"allow_fallback": False}
            if model_name:
                kwargs["model"] = model_name
            try:
                response = self._request_natural_text(
                    system_prompt,
                    prompt,
                    max_retries=0,
                    **kwargs,
                )
                episodes, parse_errors = parse_source_scoped_episode_text(
                    response,
                    timeline_scope=timeline_scope,
                )
                episodes = self._split_source_scoped_episode_blocks(episodes)
            except ModelTransportUnavailable:
                raise
            except Exception as exc:
                had_nonempty_rejection = True
                errors = [f"{label} request failed: {exc}"]
                if self.logger:
                    self.logger.emit(
                        "source_scoped_episode_attempt_rejected",
                        attempt=label,
                        errors=errors,
                    )
                continue

            errors = list(parse_errors)
            if errors:
                had_nonempty_rejection = True
            elif not episodes:
                clean_empty_seen = True
                errors = ["no Episode was returned"]
            if errors:
                if self.logger:
                    self.logger.emit(
                        "source_scoped_episode_attempt_rejected",
                        attempt=label,
                        errors=errors,
                    )
                continue

            self._complete_participants_from_source_speakers(
                episodes, speaker_labels
            )
            errors = self._source_scoped_self_containment_errors(episodes)
            if errors:
                had_nonempty_rejection = True
                if self.logger:
                    self.logger.emit(
                        "source_scoped_episode_attempt_rejected",
                        attempt=label,
                        errors=errors,
                    )
                continue

            # Provenance tags are deterministic import metadata.  Reuse the
            # existing interpreter transiently, then avoid duplicating the
            # whole Source in every Episode row.
            for episode in episodes:
                episode.evidence_quotes = [source_text]
            self._derive_episode_fields_from_evidence(episodes)
            for episode in episodes:
                episode.evidence_quotes = []
                episode.evidence_spans = []
                if speaker_labels:
                    episode.epistemic_status = "reported"
                    episode.confidence = min(episode.confidence, 0.8)
                    episode.epistemic_note = "由带发言者标记的对白归纳"

            self.finalize_episode_drafts(source_text, episodes, self.logger)
            self._apply_explicit_segment_evidence_floor(source_text, episodes)
            if self.logger:
                self.logger.emit(
                    "source_scoped_episodes_created",
                    attempt=label,
                    episode_count=len(episodes),
                    model_calls=1,
                    evidence_basis="source_id",
                )
            return episodes, []

        if clean_empty_seen and not had_nonempty_rejection:
            raise EmptyEpisodeExtraction(
                "source-scoped extraction returned only clean empty Episode lists",
                attempted_models=attempted_models,
            )
        raise ExtractionValidationError(
            "source-scoped Episode extraction failed: " + "; ".join(errors)
        )

    def _extract_episodes_single_pass(
        self,
        source_text: str,
        timeline_scope: str,
        document_context: str = "",
    ) -> tuple[list[EpisodeDraft], list[str]]:
        source_lines, _indexed_source = self._single_pass_source_lines(source_text)
        literal_reference = self._literal_reference_episodes(
            source_lines,
            timeline_scope,
        )
        if literal_reference is not None:
            evidence_errors = self._single_pass_evidence_errors(
                source_lines,
                literal_reference,
            )
            if evidence_errors:
                raise ExtractionValidationError(
                    "literal reference extraction failed: " + "; ".join(evidence_errors)
                )
            self._derive_episode_fields_from_evidence(literal_reference)
            self.finalize_episode_drafts(
                source_text,
                literal_reference,
                self.logger,
            )
            self._apply_explicit_segment_evidence_floor(
                source_text,
                literal_reference,
            )
            if self.logger:
                self.logger.emit(
                    "literal_reference_episodes_created",
                    episode_count=len(literal_reference),
                    model_calls=0,
                    exact_text=True,
                )
            return literal_reference, []
        speaker_tokens = (
            self._speaker_token_map(source_lines)
            if self._uses_natural_text_output()
            else {}
        )
        masked_source_lines = [
            self._mask_speaker_tokens(line, speaker_tokens) for line in source_lines
        ]
        indexed_source = "\n".join(
            f"[L{index:04d}] {line}"
            for index, line in enumerate(masked_source_lines, 1)
        )
        masked_document_context = self._mask_speaker_tokens(
            document_context, speaker_tokens
        )
        if speaker_tokens and self.logger:
            self.logger.emit(
                "source_speakers_tokenized",
                speaker_count=len(speaker_tokens),
            )
        prompt = single_pass_episode_prompt(
            indexed_source,
            timeline_scope,
            masked_document_context,
            enforce_document_stages=(
                self.episode_extraction_profile == "document_map_assisted"
            ),
        )
        planned_stage_spans = (
            self._document_map_stage_spans(document_context)
            if self.episode_extraction_profile == "document_map_assisted"
            else []
        )
        fallback = self._fallback_model()
        attempts: list[tuple[str | None, str]] = [
            (None, "primary"),
            (None, "primary_correction"),
        ]
        if fallback:
            attempts.append((fallback, "reasoning_fallback"))
        previous_errors: list[str] = []
        attempted_models: list[str] = []
        clean_empty_seen = False
        had_nonempty_rejection = False
        for model_name, label in attempts:
            effective_model = model_name or self._primary_reasoning_model()
            if effective_model:
                attempted_models.append(effective_model)
            active_prompt = prompt
            if previous_errors:
                active_prompt = single_pass_retry_prompt(prompt, previous_errors)
            kwargs = {"allow_fallback": False}
            if model_name:
                kwargs["model"] = model_name
            try:
                if self._uses_natural_text_output():
                    response = self._request_natural_text(
                        SINGLE_PASS_EPISODE_SYSTEM,
                        active_prompt,
                        max_retries=0,
                        **kwargs,
                    )
                    episodes, parse_errors = parse_episode_text(
                        response,
                        timeline_scope=timeline_scope,
                    )
                else:
                    payload = self.model.chat_json(
                        SINGLE_PASS_EPISODE_SYSTEM,
                        active_prompt,
                        max_retries=0,
                        **kwargs,
                    )
                    episodes, parse_errors = parse_episode_payload(payload)
            except ModelTransportUnavailable:
                raise
            except Exception as exc:
                had_nonempty_rejection = True
                previous_errors = [f"{label} request failed: {exc}"]
                if self.logger:
                    self.logger.emit(
                        "single_pass_episode_attempt_rejected",
                        attempt=label,
                        errors=previous_errors,
                    )
                continue
            if planned_stage_spans:
                if len(episodes) != len(planned_stage_spans):
                    parse_errors.append(
                        "document map stage coverage: "
                        f"{len(episodes)} Episodes for "
                        f"{len(planned_stage_spans)} stages"
                    )
                else:
                    for episode, planned_span in zip(
                        episodes, planned_stage_spans, strict=True
                    ):
                        episode.evidence_spans = [planned_span]
            evidence_errors = self._single_pass_evidence_errors(source_lines, episodes)
            if not evidence_errors:
                expansion_count = self._expand_adjacent_speaker_evidence(
                    masked_source_lines,
                    episodes,
                    speaker_tokens,
                )
                if expansion_count:
                    evidence_errors = self._single_pass_evidence_errors(
                        source_lines, episodes
                    )
                    if self.logger:
                        self.logger.emit(
                            "adjacent_speaker_evidence_expanded",
                            record_count=expansion_count,
                            mode="primary",
                        )
            speaker_token_errors = self._single_pass_speaker_token_errors(
                masked_source_lines,
                episodes,
                speaker_tokens,
            )
            self._complete_participants_from_speaker_tokens(episodes, speaker_tokens)
            self._restore_speaker_tokens(episodes, speaker_tokens)
            if not evidence_errors:
                self._derive_episode_fields_from_evidence(episodes)
                self._apply_speaker_attribution_provenance(episodes)
                if self._uses_natural_text_output():
                    episodes = self._remove_structural_front_matter(episodes)
                self._repair_single_pass_evidence_name_typos(episodes, self.logger)
                self._remove_unreferenced_unsupported_participants(
                    episodes, self.logger
                )
                if self._uses_natural_text_output():
                    self.finalize_episode_drafts(source_text, episodes, self.logger)
            participant_errors = self._single_pass_participant_evidence_errors(episodes)
            audited_profile = self.episode_extraction_profile in {
                "single_pass_audited",
                "document_map_assisted",
                "document_map_contextual",
                "adaptive_anchor_map",
            }
            coverage_errors = self._single_pass_coverage_errors(source_lines, episodes)
            noncoverage_errors = [
                *parse_errors,
                *evidence_errors,
                *speaker_token_errors,
                *(participant_errors if not audited_profile else []),
            ]
            supplement_errors: list[str] = []
            if not noncoverage_errors and coverage_errors:
                episodes, supplement_errors = self._supplement_single_pass_coverage(
                    source_lines=source_lines,
                    indexed_source=indexed_source,
                    timeline_scope=timeline_scope,
                    episodes=episodes,
                    model_name=model_name,
                    speaker_tokens=speaker_tokens,
                    masked_source_lines=masked_source_lines,
                )
                if not supplement_errors:
                    coverage_errors = []
            validation_errors = [
                *noncoverage_errors,
                *coverage_errors,
                *supplement_errors,
            ]
            reference_record_count = len(re.findall(r"\[资料类型[：:]", source_text))
            if reference_record_count >= 2 and len(episodes) < reference_record_count:
                validation_errors.append(
                    "reference record coverage: "
                    f"{len(episodes)} Episodes for {reference_record_count} records"
                )
            if validation_errors:
                had_nonempty_rejection = True
            elif not episodes:
                clean_empty_seen = True
            if validation_errors or not episodes:
                previous_errors = validation_errors or ["no Episode was returned"]
                if self.logger:
                    self.logger.emit(
                        "single_pass_episode_attempt_rejected",
                        attempt=label,
                        errors=previous_errors,
                    )
                continue
            if audited_profile:
                self._audit_single_pass_entailment(episodes)
                self._remove_unreferenced_unsupported_participants(
                    episodes, self.logger
                )
            if audited_profile and self._should_audit_episode_facts(
                source_text, episodes
            ):
                episodes, factual_errors = self._audit_episode_facts(
                    source_text,
                    timeline_scope,
                    episodes,
                )
                if factual_errors:
                    had_nonempty_rejection = True
                    previous_errors = factual_errors
                    if self.logger:
                        self.logger.emit(
                            "single_pass_episode_attempt_rejected",
                            attempt=f"{label}_independent_factual_audit",
                            errors=previous_errors,
                        )
                    continue
                self._remove_unreferenced_unsupported_participants(
                    episodes, self.logger
                )
            self.finalize_episode_drafts(source_text, episodes, self.logger)
            self._apply_speaker_attribution_provenance(episodes)
            episodes = self._remove_reference_meta_episodes(
                episodes,
                reference_record_count,
                self.logger,
            )
            post_finalize_errors = [
                *self._single_pass_evidence_errors(source_lines, episodes),
                *self._single_pass_coverage_errors(source_lines, episodes),
                *self._single_pass_participant_evidence_errors(episodes),
            ]
            if post_finalize_errors:
                had_nonempty_rejection = True
                previous_errors = [
                    "post-finalization evidence validation: " + error
                    for error in post_finalize_errors
                ]
                if self.logger:
                    self.logger.emit(
                        "single_pass_episode_attempt_rejected",
                        attempt=f"{label}_post_finalize",
                        errors=previous_errors,
                    )
                continue
            self._apply_explicit_segment_evidence_floor(source_text, episodes)
            if self.logger:
                self.logger.emit(
                    "single_pass_episode_evidence_validated",
                    attempt=label,
                    episode_count=len(episodes),
                    evidence_quote_count=sum(
                        len(episode.evidence_quotes) for episode in episodes
                    ),
                    evidence_span_count=sum(
                        len(episode.evidence_spans) for episode in episodes
                    ),
                )
            return episodes, []
        if clean_empty_seen and not had_nonempty_rejection:
            raise EmptyEpisodeExtraction(
                "single-pass extraction returned only clean empty Episode lists",
                attempted_models=attempted_models,
            )
        raise ExtractionValidationError(
            "single-pass Episode extraction failed: " + "; ".join(previous_errors)
        )

    @property
    def concept_system(self) -> str:
        return (
            CONCEPT_FINE_GRAINED_SYSTEM
            if self.concept_profile == "fine_grained"
            else CONCEPT_SYSTEM
        )

    def _concept_prompt(self, episode: EpisodeDraft, alias_context: str) -> str:
        return concept_prompt(
            episode.text,
            episode.participants,
            alias_context,
            self.concept_profile,
            self.concept_target_min,
            self.concept_target_max,
            {
                "evidence_origin": episode.evidence_origin,
                "epistemic_status": episode.epistemic_status,
                "generation": episode.generation,
                "epistemic_note": episode.epistemic_note,
            },
        )

    @staticmethod
    def _attach_literal_source_aliases(
        concepts: list[ConceptDraft],
        alias_context: str,
    ) -> None:
        """Attach only aliases explicitly present in the adapter legend."""

        groups: list[list[tuple[str, str]]] = []
        for raw_line in alias_context.splitlines():
            if ":" not in raw_line:
                continue
            raw_name, rendered = raw_line.split(":", 1)
            aliases: list[tuple[str, str]] = [(raw_name.strip(), "unknown")]
            for item in rendered.split("|"):
                if "=" not in item:
                    continue
                language, value = item.split("=", 1)
                if value.strip():
                    aliases.append((value.strip(), language.strip() or "unknown"))
            aliases = [(value, language) for value, language in aliases if value]
            if aliases:
                groups.append(aliases)
        for concept in concepts:
            matching = next(
                (
                    group
                    for group in groups
                    if any(
                        value.casefold() == concept.canonical_name.casefold()
                        for value, _language in group
                    )
                ),
                None,
            )
            if matching is None:
                continue
            seen = {
                concept.canonical_name.casefold(),
                *(value.casefold() for value, _language in concept.aliases),
            }
            for value, language in matching:
                if value.casefold() in seen:
                    continue
                seen.add(value.casefold())
                concept.aliases.append((value, language))

    _UNKNOWN_IDENTITY_QUALIFIER = re.compile(
        r"((?:一个)?(?:未标注|未知)发言者|\?\?\?)"
        r"[（(][^）)]*(?:根据上下文|推测|可能|疑似|似乎|应为|应该是)"
        r"[^）)]*[）)]"
    )
    _GUESSED_IDENTITY_BEFORE_UNKNOWN = re.compile(
        r"[A-Za-z\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af·・]{1,32}"
        r"[（(](\?\?\?|\[USERNAME\]|(?:未标注|未知)发言者)[）)]",
        re.IGNORECASE,
    )
    _GUESSED_PARTICIPANT_BEFORE_UNKNOWN = re.compile(
        r"[A-Za-z\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af·・]{1,32}"
        r"[（(](\?\?\?|\[USERNAME\]|(?:未标注|未知)发言者)[）)]"
    )

    @classmethod
    def sanitize_episode_draft(cls, draft: EpisodeDraft) -> EpisodeDraft:
        """Preserve an unknown speaker marker instead of an LLM identity guess."""
        draft.text = cls._UNKNOWN_IDENTITY_QUALIFIER.sub(r"\1", draft.text)
        participants: list[str] = []
        for participant in draft.participants:
            cleaned = cls._UNKNOWN_IDENTITY_QUALIFIER.sub(r"\1", participant)
            guessed = cls._GUESSED_PARTICIPANT_BEFORE_UNKNOWN.fullmatch(cleaned)
            if guessed:
                marker = guessed.group(1)
                # Exact participant replacement is safe: unlike a broad text
                # regex, it cannot consume preceding verbs such as “向/询问”.
                draft.text = draft.text.replace(cleaned, marker)
                cleaned = marker
            participants.append(cleaned)
        draft.text = cls._GUESSED_IDENTITY_BEFORE_UNKNOWN.sub(r"\1", draft.text)
        draft.participants = participants
        if draft.story_time_text.strip().casefold() in {
            value.casefold() for value in cls._GENERIC_STORY_TIME_VALUES
        }:
            draft.story_time_text = ""
        return draft

    @classmethod
    def _ground_episode_participants(
        cls,
        source_text: str,
        episodes: list[EpisodeDraft],
        logger: JsonlEventLogger | None = None,
    ) -> list[str]:
        """Quarantine names that the model cannot point to in this Source.

        This is intentionally an evidence rule, not a name dictionary.  It is
        equally applicable to scripts, meeting transcripts and encyclopedias:
        a participant label may be copied from the document, but a remembered
        translation or guessed identity may not be persisted as direct fact.
        """

        source_folded = source_text.casefold()
        unknown_markers = {
            "???",
            "[username]",
            "未标注发言者",
            "未知发言者",
            "unknown speaker",
        }
        quarantined: list[dict[str, object]] = []
        errors: list[str] = []
        for episode_index, episode in enumerate(episodes):
            retained: list[str] = []
            for participant in episode.participants:
                normalized = participant.strip()
                if not normalized:
                    continue
                parts = [
                    part.strip()
                    for part in cls._SOURCE_NAME_SEPARATOR.split(normalized)
                    if part.strip()
                ]
                grounded = (
                    normalized.casefold() in source_folded
                    or (
                        bool(parts)
                        and all(
                            part.casefold() in source_folded
                            or part.casefold() in unknown_markers
                            for part in parts
                        )
                    )
                    or normalized.casefold() in unknown_markers
                )
                if grounded:
                    if normalized not in retained:
                        retained.append(normalized)
                    continue

                replacement = "???"
                if normalized in episode.text:
                    episode.text = episode.text.replace(normalized, replacement)
                for part in parts:
                    if (
                        len(part) >= 2
                        and part.casefold() not in source_folded
                        and part in episode.text
                    ):
                        episode.text = episode.text.replace(part, replacement)
                if replacement not in retained:
                    retained.append(replacement)
                quarantined.append(
                    {
                        "episode_index": episode_index,
                        "participant": normalized,
                        "replacement": replacement,
                    }
                )
                errors.append(
                    "episode participant grounding: candidate "
                    f"{episode_index} contains a name absent from Source: {normalized}"
                )
            episode.participants = retained
        if quarantined and logger:
            logger.emit(
                "unsupported_participants_quarantined",
                items=quarantined,
            )
        return errors

    @classmethod
    def _complete_episode_participants_from_source(
        cls,
        source_text: str,
        episodes: list[EpisodeDraft],
        logger: JsonlEventLogger | None = None,
    ) -> None:
        """Add Source speaker identities that are named in Episode prose.

        This is the inverse of the grounding guard above.  It never performs
        entity recognition or alias guessing: a label must come from a literal
        dialogue/speaker field and one of its literal forms must occur in the
        Episode text before it can be appended.
        """

        raw_labels = [
            *re.findall(r"\[speaker_raw:\s*([^\]]+)\]", source_text),
            *(
                matched.group(1)
                for line in source_text.splitlines()
                if (
                    matched := cls._SOURCE_DIALOGUE_LABEL.match(
                        re.sub(r"^\[L\d+\]\s*", "", line.strip())
                    )
                )
            ),
        ]
        groups: list[tuple[str, set[str]]] = []
        seen_groups: set[tuple[str, ...]] = set()
        for raw_label in raw_labels:
            label = raw_label.strip()
            if not label or label.casefold() in cls._NON_NAME_LABELS:
                continue
            aliases = {
                value.strip()
                for value in cls._SOURCE_NAME_SEPARATOR.split(label)
                if len(value.strip()) >= 2
                and value.strip().casefold() not in cls._NON_NAME_LABELS
            }
            if len(label) >= 2:
                aliases.add(label)
            key = tuple(sorted(value.casefold() for value in aliases))
            if not aliases or key in seen_groups:
                continue
            seen_groups.add(key)
            groups.append((label, aliases))

        additions: list[dict[str, object]] = []

        def literal_name_mentioned(text: str, alias: str) -> bool:
            start = 0
            while True:
                index = text.find(alias, start)
                if index < 0:
                    return False
                before = text[index - 1] if index > 0 else ""
                after_index = index + len(alias)
                after = text[after_index] if after_index < len(text) else ""
                if any("\u30a0" <= char <= "\u30ff" for char in alias):
                    if not (before and "\u30a0" <= before <= "\u30ff") and not (
                        after and "\u30a0" <= after <= "\u30ff"
                    ):
                        return True
                elif any("\u3040" <= char <= "\u309f" for char in alias):
                    if not (before and "\u3040" <= before <= "\u309f") and not (
                        after and "\u3040" <= after <= "\u309f"
                    ):
                        return True
                elif any("\uac00" <= char <= "\ud7af" for char in alias):
                    if not (before and "\uac00" <= before <= "\ud7af") and not (
                        after and "\uac00" <= after <= "\ud7af"
                    ):
                        return True
                elif any(char.isascii() and char.isalpha() for char in alias):

                    def joins_ascii_identifier(character: str) -> bool:
                        return bool(character) and (
                            character.isascii()
                            and (character.isalnum() or character == "_")
                        )

                    if not joins_ascii_identifier(
                        before
                    ) and not joins_ascii_identifier(after):
                        return True
                else:
                    return True
                start = index + 1

        for episode_index, episode in enumerate(episodes):
            participant_parts = {
                value.strip().casefold()
                for participant in episode.participants
                for value in cls._SOURCE_NAME_SEPARATOR.split(participant)
                if value.strip()
            }
            for label, aliases in groups:
                mentioned = {
                    alias
                    for alias in aliases
                    if literal_name_mentioned(episode.text, alias)
                }
                if not mentioned:
                    continue
                if any(alias.casefold() in participant_parts for alias in aliases):
                    continue
                episode.participants.append(label)
                participant_parts.update(alias.casefold() for alias in aliases)
                additions.append(
                    {
                        "episode_index": episode_index,
                        "participant": label,
                        "matched_text": sorted(mentioned, key=len, reverse=True),
                    }
                )
        if additions and logger:
            logger.emit(
                "source_participants_completed",
                items=additions,
            )

    @classmethod
    def finalize_episode_drafts(
        cls,
        source_text: str,
        episodes: list[EpisodeDraft],
        logger: JsonlEventLogger | None = None,
    ) -> list[str]:
        """Apply the Source-evidence boundary to any model-written Episode.

        Pass 1, quality rewrites and pass 2 must all cross the same boundary.
        Keeping this sequence in one method prevents a later rewrite from
        reintroducing a translated name or identity that an earlier pass had
        already quarantined.
        """

        grounding_errors: list[str] = []
        for index, draft in enumerate(episodes):
            evidence_scope = "\n".join(draft.evidence_quotes).strip()
            active_source = evidence_scope or source_text
            scoped = [draft]
            cls._remove_unsupported_name_aliases(active_source, scoped, logger)
            cls._repair_source_name_typos(active_source, scoped, logger)
            episodes[index] = cls.sanitize_episode_draft(scoped[0])
            grounding_errors.extend(
                cls._ground_episode_participants(
                    active_source, [episodes[index]], logger
                )
            )
            cls._complete_episode_participants_from_source(
                active_source, [episodes[index]], logger
            )
        return grounding_errors

    @classmethod
    def _source_foreign_names(cls, source_text: str) -> list[str]:
        """Return Source-grounded foreign-script tokens safe for typo repair.

        A model occasionally copies a Japanese or Korean proper name with one
        substituted Han character (for example ``フランシ斯``).  Speaker
        labels are read from both transcript syntax and normalized adapter
        metadata.  Foreign-script runs in the accepted evidence are also safe:
        the repair below still requires an otherwise literal, unique copy with
        exactly one cross-script final-character substitution.  It cannot add
        a token that is absent from the Episode's own evidence.
        """

        names: set[str] = set()
        raw_labels = [
            *re.findall(r"\[speaker_raw:\s*([^\]]+)\]", source_text),
            *(
                matched.group(1)
                for raw_line in source_text.splitlines()
                if (
                    matched := cls._SOURCE_DIALOGUE_LABEL.match(
                        re.sub(r"^\[L\d+\]\s*", "", raw_line.strip())
                    )
                )
            ),
        ]
        for raw_label in raw_labels:
            label = raw_label.strip()
            if not label or label.casefold() in cls._NON_NAME_LABELS:
                continue
            parts = [
                part.strip()
                for part in cls._SOURCE_NAME_SEPARATOR.split(label)
                if part.strip()
            ]
            for part in (label, *parts):
                if len(part) >= 3 and cls._FOREIGN_NAME_CHAR.search(part):
                    names.add(part)
                # Japanese organisation/role labels often contain a reusable
                # proper-name prefix before the possessive marker.
                if "の" in part:
                    prefix = part.split("の", 1)[0].strip()
                    if len(prefix) >= 3 and cls._FOREIGN_NAME_CHAR.search(prefix):
                        names.add(prefix)

        # Keep script runs separate so a Japanese honorific does not become
        # part of the token being repaired (``モモカちゃん`` -> ``モモカ``).
        for matched in re.finditer(
            r"[\u30a0-\u30ffー]{3,}|[\u3040-\u309fー]{3,}|[\uac00-\ud7af]{3,}",
            source_text,
        ):
            names.add(matched.group(0))

        return sorted(names, key=len, reverse=True)

    @classmethod
    def _repair_source_name_typos(
        cls,
        source_text: str,
        episodes: list[EpisodeDraft],
        logger: JsonlEventLogger | None = None,
    ) -> None:
        """Repair only one-substitution copies of Source-grounded names.

        Pure Han names are deliberately excluded: mapping an invented Chinese
        translation back to a foreign name requires semantic judgement.  The
        narrow rule below is deterministic and cannot introduce a name that is
        absent from Source.
        """

        names = cls._source_foreign_names(source_text)
        names_by_width: dict[int, list[str]] = {}
        for name in names:
            names_by_width.setdefault(len(name), []).append(name)
        replacements: list[dict[str, str]] = []

        def script_bucket(value: str) -> str:
            if cls._HAN_CHAR.fullmatch(value):
                return "han"
            if "\u3040" <= value <= "\u309f":
                return "hiragana"
            if "\u30a0" <= value <= "\u30ff":
                return "katakana"
            if "\uac00" <= value <= "\ud7af":
                return "hangul"
            if "a" <= value.casefold() <= "z":
                return "latin"
            return "other"

        def is_unique_cross_script_substitution(candidate: str, target: str) -> bool:
            mismatched = [
                (left, right)
                for left, right in zip(candidate, target, strict=True)
                if left != right
            ]
            if len(mismatched) != 1:
                return False
            # Only repair the observed multilingual copy error: a model copied
            # the whole Source token and substituted its final character.  A
            # sliding window that differs at the beginning or in the middle can
            # straddle ordinary prose next to a short foreign name and
            # must never be treated as a name token.
            mismatch_index = next(
                index
                for index, (left, right) in enumerate(
                    zip(candidate, target, strict=True)
                )
                if left != right
            )
            if mismatch_index != len(target) - 1 or candidate[:-1] != target[:-1]:
                return False
            left, right = mismatched[0]
            # The observed failure mode substitutes one character with a
            # visually plausible character from another script.  Do not fuzzy
            # correct ordinary same-script words such as アロハ -> アロナ.
            left_script = script_bucket(left)
            right_script = script_bucket(right)
            return (
                left_script != right_script
                and left_script != "other"
                and right_script != "other"
            )

        def repair(value: str) -> str:
            result = value
            for name in names:
                width = len(name)
                if width > len(result):
                    continue
                index = 0
                while index <= len(result) - width:
                    candidate = result[index : index + width]
                    if candidate == name:
                        index += width
                        continue
                    before = result[index - 1] if index > 0 else ""
                    target_lead_script = script_bucket(name[0])
                    if (
                        is_unique_cross_script_substitution(candidate, name)
                        and cls._FOREIGN_NAME_CHAR.search(candidate)
                        and (not before or script_bucket(before) != target_lead_script)
                        and sum(
                            is_unique_cross_script_substitution(candidate, other)
                            for other in names_by_width[width]
                        )
                        == 1
                    ):
                        result = result[:index] + name + result[index + width :]
                        replacements.append({"from": candidate, "to": name})
                        index += width
                    else:
                        index += 1
            # A common multilingual-summary typo appends one Katakana to a
            # Source name before a Chinese predicate (e.g. a name followed by
            # a malformed translation of “explains”).  Only strip one terminal
            # Katakana when the evidence contains exactly one matching base
            # name and the following character is not Katakana.  Hiragana
            # particles and ordinary suffixes are deliberately excluded.
            for name in names:
                if not any("\u30a0" <= char <= "\u30ff" for char in name):
                    continue
                index = 0
                while (index := result.find(name, index)) >= 0:
                    extra_index = index + len(name)
                    if extra_index >= len(result):
                        break
                    extra = result[extra_index]
                    after_index = extra_index + 1
                    after = result[after_index] if after_index < len(result) else ""
                    candidate = result[index : extra_index + 1]
                    unique_base = (
                        sum(
                            candidate.startswith(other)
                            and len(candidate) == len(other) + 1
                            for other in names
                        )
                        == 1
                    )
                    if (
                        "\u30a0" <= extra <= "\u30ff"
                        and not (after and "\u30a0" <= after <= "\u30ff")
                        and candidate not in source_text
                        and unique_base
                    ):
                        result = result[:extra_index] + result[extra_index + 1 :]
                        replacements.append({"from": candidate, "to": name})
                        index += len(name)
                    else:
                        index += len(name)
            return result

        for episode in episodes:
            episode.text = repair(episode.text)
            episode.participants = [repair(item) for item in episode.participants]
            episode.location_text = repair(episode.location_text)
            episode.epistemic_note = repair(episode.epistemic_note)
        if replacements and logger:
            unique = list({(item["from"], item["to"]) for item in replacements})
            logger.emit(
                "source_name_typos_repaired",
                replacements=[
                    {"from": source, "to": target} for source, target in sorted(unique)
                ],
            )

    @classmethod
    def _remove_unsupported_name_aliases(
        cls,
        source_text: str,
        episodes: list[EpisodeDraft],
        logger: JsonlEventLogger | None = None,
    ) -> None:
        """Remove model-invented translations while retaining Source names."""

        source_folded = source_text.casefold()
        replacements: set[tuple[str, str]] = set()

        def grounded(value: str) -> bool:
            normalized = value.strip()
            return bool(normalized) and normalized.casefold() in source_folded

        for episode in episodes:
            participant_mappings: dict[str, str] = {}
            normalized_participants: list[str] = []
            for participant in episode.participants:
                parts = [
                    part.strip()
                    for part in cls._SOURCE_NAME_SEPARATOR.split(participant)
                    if part.strip()
                ]
                grounded_parts: list[str] = []
                for part in parts:
                    if grounded(part) and part not in grounded_parts:
                        grounded_parts.append(part)
                if not grounded_parts:
                    normalized_participants.append(participant)
                    continue
                replacement = " / ".join(grounded_parts)
                normalized_participants.append(replacement)
                preferred = grounded_parts[0]
                for part in parts:
                    if not grounded(part) and len(part) >= 2:
                        participant_mappings[part] = preferred
            episode.participants = normalized_participants
            for unsupported, supported in participant_mappings.items():
                if unsupported in episode.text:
                    episode.text = episode.text.replace(unsupported, supported)
                    replacements.add((unsupported, supported))

            def repair_foreign_parenthetical(match: re.Match[str]) -> str:
                name = match.group("name")
                raw_aliases = match.group("aliases")
                aliases = [
                    item.strip()
                    for item in re.split(r"\s*[/／]\s*", raw_aliases)
                    if item.strip()
                ]
                supported_aliases = [
                    item
                    for item in aliases
                    if grounded(item) and item.casefold() != name.casefold()
                ]
                if grounded(name):
                    if not supported_aliases:
                        replacements.add((match.group(0), name))
                        return name
                    replacement = f"{name}（{' / '.join(supported_aliases)}）"
                    if replacement != match.group(0):
                        replacements.add((match.group(0), replacement))
                    return replacement
                if supported_aliases:
                    replacement = supported_aliases[0]
                    replacements.add((match.group(0), replacement))
                    return replacement
                return match.group(0)

            episode.text = cls._FOREIGN_ALIAS_PAREN.sub(
                repair_foreign_parenthetical, episode.text
            )

        if replacements and logger:
            logger.emit(
                "unsupported_name_aliases_removed",
                replacements=[
                    {"from": source, "to": target}
                    for source, target in sorted(replacements)
                ],
            )

    def _fallback_model(self) -> str | None:
        config = getattr(self.model, "config", None)
        value = getattr(config, "fallback_model", None)
        return str(value) if value else None

    def _primary_reasoning_model(self) -> str:
        config = getattr(self.model, "config", None)
        value = getattr(config, "reasoning_model", None)
        return str(value).strip() if value else ""

    def _empty_episode_reviewer_model(
        self,
        attempted_models: Iterable[str] = (),
    ) -> str | None:
        """Select a reviewer outside every completed extraction attempt.

        The ordinary fallback can be an independent reviewer only when it was
        not already asked to extract this exact Source.  A dedicated third
        model can be configured through ``empty_episode_audit_model`` when
        both ordinary extraction models have been attempted.
        """

        excluded = {
            str(model).strip().casefold()
            for model in attempted_models
            if str(model).strip()
        }
        primary = self._primary_reasoning_model()
        if primary:
            excluded.add(primary.casefold())
        candidate = self.empty_episode_audit_model or self._fallback_model() or ""
        candidate = str(candidate).strip()
        if not candidate or candidate.casefold() in excluded:
            return None
        return candidate

    @staticmethod
    def _audit_response_sha256(payload: object) -> str:
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def review_empty_episode(
        self,
        source_text: str,
        *,
        attempted_models: Iterable[str] = (),
    ) -> EmptyEpisodeAudit:
        """Independently decide whether a clean empty extraction can be skipped.

        The primary response is intentionally absent from the reviewer prompt.
        Any unavailable model, malformed receipt, or non-independent model
        identity produces ``uncertain`` so the caller fails closed.
        """

        source_lines, indexed_source = self._single_pass_source_lines(source_text)
        source_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        primary_model = self._primary_reasoning_model()
        seen_models: set[str] = set()
        normalized_attempts: list[str] = []
        for model in attempted_models:
            value = str(model or "").strip()
            key = value.casefold()
            if value and key not in seen_models:
                seen_models.add(key)
                normalized_attempts.append(value)
        reviewer_model = self._empty_episode_reviewer_model(normalized_attempts)
        if self.empty_episode_audit_mode != "adversarial":
            return EmptyEpisodeAudit(
                verdict="uncertain",
                source_kind="mixed_or_ambiguous",
                reason="empty Episode adversarial audit is disabled",
                line_reviews=[],
                required_ranges=[],
                primary_model=primary_model,
                reviewer_model="",
                source_sha256=source_sha256,
                validation_errors=["empty Episode adversarial audit is disabled"],
                attempted_models=normalized_attempts,
            )
        if reviewer_model is None:
            return EmptyEpisodeAudit(
                verdict="uncertain",
                source_kind="mixed_or_ambiguous",
                reason="no independently identifiable empty Episode reviewer is configured",
                line_reviews=[],
                required_ranges=[],
                primary_model=primary_model,
                reviewer_model="",
                source_sha256=source_sha256,
                validation_errors=[
                    "an explicit reviewer distinct from every attempted extraction model is required"
                ],
                attempted_models=normalized_attempts,
            )
        if not source_lines:
            return EmptyEpisodeAudit(
                verdict="uncertain",
                source_kind="mixed_or_ambiguous",
                reason="empty Source cannot be independently reviewed",
                line_reviews=[],
                required_ranges=[],
                primary_model=primary_model,
                reviewer_model=reviewer_model,
                source_sha256=source_sha256,
                validation_errors=["empty Source cannot be independently reviewed"],
                attempted_models=normalized_attempts,
            )
        try:
            audit_prompt = empty_episode_adversarial_audit_prompt(indexed_source)
            if self._uses_natural_text_output():
                # ``chat_json`` can ask the same model to repair malformed
                # JSON.  A safety decision gets exactly one reviewer request;
                # malformed output must fail closed instead of being repaired.
                response = self._request_natural_text(
                    EMPTY_EPISODE_ADVERSARIAL_AUDIT_SYSTEM,
                    audit_prompt,
                    model=reviewer_model,
                    allow_fallback=False,
                    max_retries=0,
                )
                payload = extract_json_payload(response)
            else:
                payload = self.model.chat_json(
                    EMPTY_EPISODE_ADVERSARIAL_AUDIT_SYSTEM,
                    audit_prompt,
                    model=reviewer_model,
                    allow_fallback=False,
                    max_retries=0,
                )
        except ModelTransportUnavailable:
            # Preserve the pipeline's process-wide transport circuit rather
            # than turning it into an ordinary per-segment validation error.
            raise
        except Exception as exc:
            return EmptyEpisodeAudit(
                verdict="uncertain",
                source_kind="mixed_or_ambiguous",
                reason="independent empty Episode review request failed",
                line_reviews=[],
                required_ranges=[],
                primary_model=primary_model,
                reviewer_model=reviewer_model,
                source_sha256=source_sha256,
                validation_errors=[f"independent reviewer request failed: {exc}"],
                attempted_models=normalized_attempts,
            )
        response_sha256 = self._audit_response_sha256(payload)
        parsed, errors = parse_empty_episode_audit_payload(payload, source_lines)
        if errors:
            return EmptyEpisodeAudit(
                verdict="uncertain",
                source_kind="mixed_or_ambiguous",
                reason="independent empty Episode review failed deterministic validation",
                line_reviews=[],
                required_ranges=[],
                primary_model=primary_model,
                reviewer_model=reviewer_model,
                source_sha256=source_sha256,
                response_sha256=response_sha256,
                validation_errors=errors,
                attempted_models=normalized_attempts,
            )
        return EmptyEpisodeAudit(
            verdict=str(parsed["verdict"]),
            source_kind=str(parsed["source_kind"]),
            reason=str(parsed["reason"]),
            line_reviews=list(parsed["line_reviews"]),
            required_ranges=list(parsed["required_ranges"]),
            primary_model=primary_model,
            reviewer_model=reviewer_model,
            source_sha256=source_sha256,
            response_sha256=response_sha256,
            validation_errors=[],
            attempted_models=normalized_attempts,
        )

    def extract_after_empty_episode_review(
        self,
        source_text: str,
        timeline_scope: str,
        audit: EmptyEpisodeAudit,
    ) -> tuple[list[EpisodeDraft], list[str]]:
        """Run one reviewer-owned, evidence-bound rescue extraction.

        This is deliberately not a replay of the primary extraction path.  The
        independent reviewer has identified literal line ranges that require
        an Episode, and it receives exactly one constrained extraction call.
        """

        if audit.verdict != "episode_required" or not audit.required_ranges:
            raise ExtractionValidationError(
                "empty Episode review did not provide a constrained rescue range"
            )
        if not audit.reviewer_model:
            raise ExtractionValidationError(
                "empty Episode review has no independent reviewer model"
            )
        source_lines, indexed_source = self._single_pass_source_lines(source_text)
        prompt = single_pass_episode_prompt(
            indexed_source,
            timeline_scope,
            required_ranges=audit.required_ranges,
        )
        kwargs = {
            "model": audit.reviewer_model,
            "allow_fallback": False,
            "max_retries": 0,
        }
        try:
            if self._uses_natural_text_output():
                response = self._request_natural_text(
                    SINGLE_PASS_EPISODE_SYSTEM,
                    prompt,
                    **kwargs,
                )
                episodes, parse_errors = parse_episode_text(
                    response,
                    timeline_scope=timeline_scope,
                )
            else:
                payload = self.model.chat_json(
                    SINGLE_PASS_EPISODE_SYSTEM,
                    prompt,
                    **kwargs,
                )
                episodes, parse_errors = parse_episode_payload(payload)
        except ModelTransportUnavailable:
            raise
        except Exception as exc:
            raise ExtractionValidationError(
                "independent empty Episode rescue request failed: " + str(exc)
            ) from exc

        validation_errors = list(parse_errors)
        if not episodes:
            validation_errors.append("independent empty Episode rescue returned no Episode")
        validation_errors.extend(
            self._single_pass_evidence_errors(source_lines, episodes)
        )
        for start_line, end_line in audit.required_ranges:
            if not any(
                span_start <= end_line and span_end >= start_line
                for episode in episodes
                for span_start, span_end in episode.evidence_spans
            ):
                validation_errors.append(
                    "independent empty Episode rescue did not cite required range "
                    f"{start_line}-{end_line}"
                )
        validation_errors.extend(
            self._single_pass_participant_evidence_errors(episodes)
        )
        if validation_errors:
            raise ExtractionValidationError(
                "independent empty Episode rescue failed: "
                + "; ".join(validation_errors)
            )

        self._derive_episode_fields_from_evidence(episodes)
        self._apply_speaker_attribution_provenance(episodes)
        self._repair_single_pass_evidence_name_typos(episodes, self.logger)
        self._remove_unreferenced_unsupported_participants(episodes, self.logger)
        self.finalize_episode_drafts(source_text, episodes, self.logger)
        self._apply_explicit_segment_evidence_floor(source_text, episodes)
        if self.logger:
            self.logger.emit(
                "empty_episode_rescue_completed",
                reviewer_model=audit.reviewer_model,
                required_ranges=audit.required_ranges,
                episode_count=len(episodes),
                source_sha256=audit.source_sha256,
            )
        return episodes, []

    @staticmethod
    def compact_source_for_reasoning(source_text: str) -> str:
        """Keep the full Source in SQLite but send one preferred translation."""
        lines = source_text.splitlines()
        prefix: list[str] = []
        records: list[list[str]] = []
        current: list[str] | None = None
        for line in lines:
            if line.startswith("[record:"):
                if current is not None:
                    records.append(current)
                current = [line]
            elif current is None:
                prefix.append(line)
            else:
                current.append(line)
        if current is not None:
            records.append(current)
        if not records:
            return source_text

        primary_languages = ("zh-CN", "ja", "en")
        fallback_languages = ("zh-TW", "th", "unknown")
        language_order = (*primary_languages, *fallback_languages)
        compact_records: list[str] = []
        for record in records:
            metadata = [
                line
                for line in record
                if line.startswith("[record:")
                or line.startswith("[speaker_raw:")
                or line.startswith("[evidence_origin:")
                or line.startswith("[epistemic_status:")
                or line.startswith("[evidence_generation:")
                or line.startswith("[epistemic_note:")
                or line.startswith("[document_role:")
                or line.startswith("[document_style:")
            ]
            is_reference = any(
                line.strip().casefold() == "[document_style: reference]"
                for line in metadata
            )
            if not is_reference and not any(
                line.startswith("[speaker_raw:") for line in metadata
            ):
                metadata.append("[speaker_raw: ???]")
            translations: dict[str, list[str]] = {}
            current_language = ""
            for line in record:
                matched_language = ""
                for language in language_order:
                    marker = f"{language}:"
                    if line.startswith(marker):
                        translations[language] = [line[len(marker) :].lstrip()]
                        current_language = language
                        matched_language = language
                        break
                if matched_language:
                    continue
                if line.startswith("["):
                    current_language = ""
                elif current_language and line.strip():
                    # Text records can span multiple lines. Keep continuation
                    # lines in the compact reasoning view instead of silently
                    # reducing a paragraph to its first line.
                    translations[current_language].append(line)
            script_line = next(
                (line for line in record if line.startswith("[script_raw:")), ""
            )
            korean_text = ""
            if script_line:
                korean_text = parse_korean_text(
                    script_line[len("[script_raw:") :].removesuffix("]").strip()
                )
            selected_lines = [
                f"{language}: " + "\n".join(translations[language])
                for language in primary_languages
                if language in translations
            ]
            if not selected_lines:
                fallback = next(
                    (
                        translations[language]
                        for language in fallback_languages
                        if language in translations
                    ),
                    "",
                )
                if fallback:
                    fallback_language = next(
                        language
                        for language in fallback_languages
                        if language in translations
                    )
                    selected_lines.append(
                        f"{fallback_language}: "
                        + "\n".join(translations[fallback_language])
                    )
            if korean_text:
                selected_lines.append(f"ko: {korean_text}")
            compact_records.append("\n".join([*metadata, *selected_lines]))
        return "\n".join(prefix).rstrip() + "\n\n" + "\n\n".join(compact_records)

    @staticmethod
    def _speaker_alias_context(source_text: str) -> str:
        marker = "[speaker_alias_legend]\n"
        start = source_text.find(marker)
        lines: list[str] = []
        if start >= 0:
            start += len(marker)
            end = source_text.find("\n\n", start)
            lines.extend(
                (
                    source_text[start:] if end < 0 else source_text[start:end]
                ).splitlines()
            )

        # Some human-readable corpora place multilingual aliases directly in
        # the speaker label instead of emitting a separate adapter legend.
        # Convert only that explicit syntax into the same program-owned legend;
        # no model or external name dictionary is involved.
        parenthetical = re.compile(
            r"^\s*(?P<name>.+?)\s*[（(](?P<aliases>[^）)]+)[）)]\s*$"
        )
        for raw_label in re.findall(r"\[speaker_raw:\s*([^\]]+)\]", source_text):
            matched = parenthetical.fullmatch(raw_label)
            if matched is None:
                continue
            name = matched.group("name").strip()
            aliases = [
                value.strip()
                for value in re.split(r"\s*[/／]\s*", matched.group("aliases"))
                if value.strip()
            ]
            if not name or not aliases:
                continue
            lines.append(
                name + ": " + " | ".join(f"unknown={value}" for value in aliases)
            )
        return "\n".join(dict.fromkeys(line.strip() for line in lines if line.strip()))

    @staticmethod
    def _derive_episode_fields_from_evidence(
        episodes: list[EpisodeDraft],
    ) -> None:
        """Populate persistence metadata from the cited Source, not the model.

        Natural-language extraction only supplies the memory sentence and its
        citations.  Evidence provenance is structural input metadata and can
        therefore be assembled deterministically after the cited text has been
        reconstructed.
        """

        allowed_origins = {"source", "importer", "system", "mixed", "unknown"}
        allowed_statuses = {
            "observed",
            "asserted",
            "reported",
            "speculative",
            "mixed",
            "unknown",
        }
        for episode in episodes:
            evidence = "\n".join(episode.evidence_quotes)
            origins = {
                value.strip().casefold()
                for value in re.findall(r"\[evidence_origin:\s*([^\]]+)\]", evidence)
                if value.strip().casefold() in allowed_origins
            }
            statuses = {
                value.strip().casefold()
                for value in re.findall(r"\[epistemic_status:\s*([^\]]+)\]", evidence)
                if value.strip().casefold() in allowed_statuses
            }
            generations = [
                max(0, int(value))
                for value in re.findall(r"\[evidence_generation:\s*(\d+)\]", evidence)
            ]
            notes = list(
                dict.fromkeys(
                    value.strip()
                    for value in re.findall(r"\[epistemic_note:\s*([^\]]+)\]", evidence)
                    if value.strip()
                )
            )
            episode.evidence_origin = (  # type: ignore[assignment]
                next(iter(origins))
                if len(origins) == 1
                else "mixed"
                if origins
                else "source"
            )
            episode.epistemic_status = (  # type: ignore[assignment]
                next(iter(statuses))
                if len(statuses) == 1
                else "mixed"
                if statuses
                else "asserted"
            )
            episode.generation = max(generations, default=0)
            if episode.evidence_origin in {
                "importer",
                "system",
                "mixed",
            } and episode.epistemic_status in {"speculative", "mixed"}:
                episode.generation = max(1, episode.generation)
            episode.epistemic_note = "；".join(notes)[:1_000]
            if episode.epistemic_status in {"speculative", "mixed", "unknown"}:
                episode.confidence = min(episode.confidence, 0.7)
            elif episode.epistemic_status == "reported":
                episode.confidence = min(episode.confidence, 0.8)

    def _remove_structural_front_matter(
        self,
        episodes: list[EpisodeDraft],
    ) -> list[EpisodeDraft]:
        retained: list[EpisodeDraft] = []
        removed: list[dict[str, object]] = []
        for index, episode in enumerate(episodes):
            evidence = "\n".join(episode.evidence_quotes)
            record_count = evidence.count("[record:")
            front_count = evidence.count("[document_role: front_matter]")
            if record_count > 0 and front_count >= record_count:
                removed.append(
                    {
                        "episode_index": index,
                        "text": episode.text[:500],
                        "evidence_spans": episode.evidence_spans,
                    }
                )
                continue
            retained.append(episode)
        if removed and self.logger:
            self.logger.emit(
                "structural_front_matter_episodes_removed",
                items=removed,
            )
        return retained

    @staticmethod
    def _source_declares_unresolved_alternatives(source_text: str) -> bool:
        """Recognize an explicit document-level alternative boundary.

        This is provenance parsing, not story interpretation: both the mutual
        exclusion and the absence of a recorded selection must be stated by
        the imported text.  Supporting several common languages keeps the
        rule tied to document semantics rather than one corpus vocabulary.
        """

        value = source_text.casefold()
        has_exclusion = bool(
            re.search(
                r"(?:彼此互斥|相互互斥|互斥(?:选项|選項)|同一(?:选择|選擇)组|"
                r"mutually\s+exclusive|exclusive\s+alternatives?|"
                r"排他的(?:な)?選択肢|選択肢.{0,30}同時には選べない|"
                r"상호\s*배타|동시에.{0,20}선택할\s*수\s*없)",
                value,
                re.DOTALL,
            )
        )
        has_unrecorded_selection = bool(
            re.search(
                r"(?:(?:实际|實際)(?:选择|選擇).{0,20}(?:未记录|未記錄|未知|不明|"
                r"无法确定|無法確定)|"
                r"actual\s+(?:choice|selection).{0,40}(?:not\s+recorded|unrecorded|"
                r"unknown|cannot\s+be\s+determined)|"
                r"実際.{0,30}(?:記録されていない|不明|確認できない)|"
                r"실제.{0,30}(?:기록되지|알\s*수\s*없|확인할\s*수\s*없))",
                value,
                re.DOTALL,
            )
        )
        return has_exclusion and has_unrecorded_selection

    @staticmethod
    def _apply_explicit_segment_evidence_floor(
        source_text: str, episodes: list[EpisodeDraft]
    ) -> None:
        """Prevent a homogeneous explicitly speculative Source from upgrading.

        Per-record evidence labels remain visible to the model for mixed
        reference segments. When every factual record is explicitly marked as
        importer speculation, the boundary is deterministic and can be enforced
        after all LLM repair/audit passes.
        """

        if MemoryExtractor._source_declares_unresolved_alternatives(source_text):
            for draft in episodes:
                draft.evidence_origin = "source"
                draft.epistemic_status = "speculative"
                # The alternatives are directly present in the Source. Their
                # selection is unknown, so this is not an importer inference.
                draft.generation = max(0, int(draft.generation))
                draft.confidence = min(draft.confidence, 0.7)
                if not draft.epistemic_note:
                    draft.epistemic_note = (
                        "原文明确标记为互斥备选，实际选择未记录"
                    )

        inferential_markers = (
            "推测",
            "推论",
            "考据",
            "社区观点",
            "剧情解读",
        )
        reference_types = re.findall(r"\[资料类型[：:]\s*([^\]|]+)", source_text)
        if reference_types and len(reference_types) == len(episodes):
            for reference_type, draft in zip(reference_types, episodes, strict=True):
                if not any(marker in reference_type for marker in inferential_markers):
                    continue
                draft.evidence_origin = "importer"
                draft.epistemic_status = "speculative"
                draft.generation = max(1, int(draft.generation))
                if not draft.epistemic_note:
                    draft.epistemic_note = f"导入文档标记为{reference_type.strip()}"

        fact_records = len(reference_types)
        record_count = source_text.count("[record:")
        relevant_records = fact_records or record_count
        speculative_labels = source_text.count("[epistemic_status: speculative]")
        importer_labels = source_text.count("[evidence_origin: importer]")
        if (
            relevant_records < 1
            or speculative_labels < relevant_records
            or importer_labels < relevant_records
        ):
            return
        note_match = re.search(r"\[epistemic_note:\s*([^\]]+)\]", source_text)
        note = note_match.group(1).strip() if note_match else "导入文档明确标记为推测"
        for draft in episodes:
            draft.evidence_origin = "importer"
            draft.epistemic_status = "speculative"
            draft.generation = max(1, int(draft.generation))
            if not draft.epistemic_note:
                draft.epistemic_note = note

    def extract_episodes(
        self,
        source_text: str,
        timeline_scope: str,
        document_context: str = "",
    ) -> tuple[list[EpisodeDraft], list[str]]:
        reasoning_source = self.compact_source_for_reasoning(source_text)
        if self.logger:
            self.logger.emit(
                "source_reasoning_view",
                raw_chars=len(source_text),
                reasoning_chars=len(reasoning_source),
            )
        if self.episode_extraction_profile == "source_scoped_plain":
            return self._extract_episodes_source_scoped_plain(
                reasoning_source,
                timeline_scope,
            )
        if self.episode_extraction_profile in {
            "single_pass_evidence",
            "single_pass_audited",
            "document_map_assisted",
            "document_map_contextual",
            "adaptive_anchor_map",
        }:
            return self._extract_episodes_single_pass(
                reasoning_source,
                timeline_scope,
                document_context,
            )
        primary_model = self._primary_reasoning_model()
        payload = self.model.chat_json(
            EPISODE_SYSTEM,
            episode_prompt(reasoning_source, timeline_scope),
            allow_fallback=False,
        )
        valid, errors = parse_episode_payload(payload)
        saw_candidate = bool(valid)
        had_invalid_response = bool(errors)
        if errors and self.logger:
            self.logger.emit("validation_failed", stage="episodes", errors=errors)
        if errors:
            correction = self.model.chat_json(
                EPISODE_SYSTEM,
                partial_item_retry_prompt(
                    episode_prompt(reasoning_source, timeline_scope), errors
                ),
                allow_fallback=False,
            )
            corrected, correction_errors = parse_episode_payload(correction)
            valid.extend(corrected)
            saw_candidate = saw_candidate or bool(corrected)
            errors = correction_errors
        if errors and self._fallback_model():
            fallback = self.model.chat_json(
                EPISODE_SYSTEM,
                partial_item_retry_prompt(
                    episode_prompt(reasoning_source, timeline_scope),
                    errors,
                    prior_attempts=2,
                ),
                model=self._fallback_model(),
                allow_fallback=False,
            )
            corrected, errors = parse_episode_payload(fallback)
            valid.extend(corrected)
            saw_candidate = saw_candidate or bool(corrected)
        reference_record_count = len(re.findall(r"\[资料类型[：:]", reasoning_source))
        if reference_record_count >= 2:
            valid = self._remove_reference_meta_episodes(
                valid, reference_record_count, self.logger
            )
        if reference_record_count >= 2 and (
            errors or len(valid) < reference_record_count
        ):
            coverage_prompt = reference_coverage_prompt(
                episode_prompt(reasoning_source, timeline_scope),
                record_count=reference_record_count,
                valid_count=len(valid),
            )
            replacement_payload = self.model.chat_json(
                EPISODE_SYSTEM,
                coverage_prompt,
                allow_fallback=False,
            )
            replacement, replacement_errors = parse_episode_payload(replacement_payload)
            replacement = self._remove_reference_meta_episodes(
                replacement, reference_record_count, self.logger
            )
            if not replacement_errors and len(replacement) >= reference_record_count:
                valid, errors = replacement, []
                saw_candidate = saw_candidate or bool(replacement)
            elif self._fallback_model():
                fallback_payload = self.model.chat_json(
                    EPISODE_SYSTEM,
                    reference_coverage_fallback_prompt(coverage_prompt),
                    model=self._fallback_model(),
                    allow_fallback=False,
                )
                fallback_valid, fallback_errors = parse_episode_payload(
                    fallback_payload
                )
                fallback_valid = self._remove_reference_meta_episodes(
                    fallback_valid, reference_record_count, self.logger
                )
                if (
                    not fallback_errors
                    and len(fallback_valid) >= reference_record_count
                ):
                    valid, errors = fallback_valid, []
                    saw_candidate = saw_candidate or bool(fallback_valid)
                else:
                    valid, errors = fallback_valid, fallback_errors
            if errors or len(valid) < reference_record_count:
                if self.logger:
                    self.logger.emit(
                        "validation_failed",
                        stage="reference_record_coverage",
                        expected_records=reference_record_count,
                        valid_episodes=len(valid),
                        errors=errors,
                    )
                raise ValueError(
                    "reference-record coverage failed: "
                    f"{len(valid)} valid Episodes for "
                    f"{reference_record_count} fact records"
                )
        temporal_risk = bool(valid and self._needs_temporal_audit(valid))
        granularity_risk = bool(valid and self._needs_granularity_audit(valid))
        if self.episode_audit_mode == "combined" and (
            self.episode_audit_always or temporal_risk or granularity_risk
        ):
            valid, audit_errors = self._audit_episode_quality(
                reasoning_source,
                timeline_scope,
                valid,
                temporal_risk=temporal_risk,
                granularity_risk=granularity_risk,
            )
            errors.extend(audit_errors)
        elif self.episode_audit_mode == "split":
            if temporal_risk:
                valid, audit_errors = self._audit_episode_times(
                    reasoning_source, timeline_scope, valid
                )
                errors.extend(audit_errors)
            if valid and (
                self.episode_audit_always or self._needs_granularity_audit(valid)
            ):
                valid, audit_errors = self._audit_episode_granularity(
                    reasoning_source, timeline_scope, valid
                )
                errors.extend(audit_errors)
        if reference_record_count >= 2:
            valid = self._remove_reference_meta_episodes(
                valid, reference_record_count, self.logger
            )
        if valid and self._should_audit_episode_facts(reasoning_source, valid):
            valid, audit_errors = self._audit_episode_facts(
                reasoning_source, timeline_scope, valid
            )
            errors.extend(audit_errors)
        # A safely quarantined unsupported name is a recovered model error, not
        # an unresolved import failure. The complete incident remains in JSONL
        # and the resulting ??? can be revisited by pass 2. Only unresolved
        # validation failures make the extraction task partial.
        self.finalize_episode_drafts(reasoning_source, valid, self.logger)
        self._apply_explicit_segment_evidence_floor(reasoning_source, valid)
        if not valid and not errors:
            if not saw_candidate and not had_invalid_response:
                raise EmptyEpisodeExtraction(
                    "legacy extraction returned a clean empty Episode list",
                    attempted_models=([primary_model] if primary_model else []),
                )
            raise ExtractionValidationError(
                "no Episode remained after candidate validation"
            )
        return valid, errors

    def _should_audit_episode_facts(
        self, source_text: str, episodes: list[EpisodeDraft]
    ) -> bool:
        if self.episode_factual_audit_mode == "off":
            return False
        if self.episode_factual_audit_mode == "always":
            return True
        searchable = (
            source_text + "\n" + "\n".join(episode.text for episode in episodes)
        )
        return self._has_factual_audit_risk(searchable)

    @classmethod
    def _has_factual_audit_risk(cls, value: str) -> bool:
        # Dialogue and attributed prose are structurally risky regardless of
        # the corpus, language, character names or job titles involved.  Keep
        # the adaptive trigger about syntax shape rather than lore keywords.
        dialogue_labels = re.findall(r"(?m)^([^:\n]{1,80}):(?:\s|$)", value)
        tagged_speakers = re.findall(r"\[speaker_raw:\s*([^\]]+)\]", value)
        distinct_speakers = {
            re.sub(r"\s+", " ", label).strip().casefold()
            for label in [*dialogue_labels, *tagged_speakers]
            if label.strip()
        }
        attributed_clause = bool(
            re.search(
                r"[\"'“”‘’「」『』]|\b(?:said|reported|claimed|according to)\b",
                value,
                re.IGNORECASE,
            )
        )
        return len(distinct_speakers) >= 2 or attributed_clause

    @classmethod
    def _factual_audit_candidate_indexes(
        cls, source_text: str, episodes: list[EpisodeDraft]
    ) -> list[int]:
        # Once an attributed Source is selected, review every Episode in that
        # Source.  Filtering by story words can omit the very candidate whose
        # predicate or actor was lost during summarisation.
        return list(range(len(episodes)))

    def _audit_episode_facts(
        self,
        source_text: str,
        timeline_scope: str,
        episodes: list[EpisodeDraft],
    ) -> tuple[list[EpisodeDraft], list[str]]:
        selected_indexes = self._factual_audit_candidate_indexes(source_text, episodes)
        selected = [episodes[index] for index in selected_indexes]
        audit_model = self.episode_factual_audit_model
        if self._uses_natural_text_output():
            errors: list[str] = []
            batch_size = self.episode_factual_audit_batch_size
            if self.logger:
                self.logger.emit(
                    "episode_factual_audit_started",
                    candidate_episode_count=len(episodes),
                    selected_episode_count=len(selected),
                    selected_episode_indexes=selected_indexes,
                    batch_size=batch_size,
                    audit_model=audit_model or "primary",
                    output_format="natural_text",
                )
            for batch_start in range(0, len(selected), batch_size):
                batch = selected[batch_start : batch_start + batch_size]
                try:
                    self._audit_single_pass_entailment(
                        batch,
                        model_override=audit_model,
                        audit_stage="independent_factual_audit",
                    )
                except Exception as exc:
                    errors.append(f"batch {batch_start}: {exc}")
                    continue
                if self.logger:
                    self.logger.emit(
                        "episode_factual_audit_batch_completed",
                        batch_start=batch_start,
                        batch_count=len(batch),
                        output_format="natural_text",
                    )
            if errors:
                final_errors = [
                    f"episode factual audit: {message}" for message in errors
                ]
                if self.logger:
                    self.logger.emit(
                        "episode_factual_audit_rejected",
                        errors=final_errors,
                        output_format="natural_text",
                    )
                return episodes, final_errors
            if self.logger:
                self.logger.emit(
                    "episode_factual_audit_completed",
                    episode_count=len(episodes),
                    selected_episode_count=len(selected),
                    audit_model=audit_model or "primary",
                    output_format="natural_text",
                )
            return episodes, []
        kwargs = {"allow_fallback": False}
        if audit_model:
            kwargs["model"] = audit_model
        if self.logger:
            self.logger.emit(
                "episode_factual_audit_started",
                candidate_episode_count=len(episodes),
                selected_episode_count=len(selected),
                selected_episode_indexes=selected_indexes,
                batch_size=self.episode_factual_audit_batch_size,
                audit_model=audit_model or "primary",
            )
        reviews: dict[int, dict[str, object]] = {}
        errors: list[str] = []
        batch_size = self.episode_factual_audit_batch_size
        for batch_start in range(0, len(selected), batch_size):
            batch = selected[batch_start : batch_start + batch_size]
            audit_source = self._fact_audit_source_excerpt(source_text, batch)
            prompt = episode_factual_audit_prompt(
                audit_source,
                timeline_scope,
                [asdict(episode) for episode in batch],
            )
            expected_indexes = set(range(len(batch)))
            if self.logger:
                self.logger.emit(
                    "episode_factual_audit_batch_started",
                    batch_start=batch_start,
                    batch_count=len(batch),
                    source_chars=len(audit_source),
                    full_source_chars=len(source_text),
                    audit_model=audit_model or "primary",
                )
            payload = self.model.chat_json(
                EPISODE_FACTUAL_AUDIT_SYSTEM, prompt, **kwargs
            )
            batch_reviews, batch_errors = parse_episode_fact_review_payload(
                payload, expected_indexes
            )
            batch_errors.extend(
                self._validate_fact_review_evidence(source_text, batch_reviews)
            )
            if batch_errors:
                correction = self.model.chat_json(
                    EPISODE_FACTUAL_AUDIT_SYSTEM,
                    factual_audit_retry_prompt(prompt, batch_errors),
                    **kwargs,
                )
                batch_reviews, batch_errors = parse_episode_fact_review_payload(
                    correction, expected_indexes
                )
                batch_errors.extend(
                    self._validate_fact_review_evidence(source_text, batch_reviews)
                )
            if batch_errors:
                errors.extend(
                    f"batch {batch_start}: {message}" for message in batch_errors
                )
                continue
            for local_index, review in batch_reviews.items():
                reviews[batch_start + local_index] = review
            if self.logger:
                self.logger.emit(
                    "episode_factual_audit_batch_completed",
                    batch_start=batch_start,
                    batch_count=len(batch),
                )
        if errors:
            final_errors = [f"episode factual audit: {message}" for message in errors]
            if self.logger:
                self.logger.emit(
                    "episode_factual_audit_rejected",
                    errors=final_errors,
                )
            return episodes, final_errors
        merged = list(episodes)
        accepted_indexes: list[int] = []
        supported_indexes: list[int] = []
        unresolved_indexes: list[int] = []
        unresolved_errors: list[str] = []
        review_log: list[dict[str, object]] = []
        for local_index, index in enumerate(selected_indexes):
            review = reviews[local_index]
            status = str(review["status"])
            corrected = review.get("corrected_episode")
            if status == "corrected" and isinstance(corrected, EpisodeDraft):
                if not corrected.timeline_scope:
                    corrected.timeline_scope = timeline_scope
                merged[index] = self.sanitize_episode_draft(corrected)
                accepted_indexes.append(index)
            elif status == "supported":
                supported_indexes.append(index)
            else:
                unresolved_indexes.append(index)
                unresolved_errors.append(
                    "episode factual audit: candidate "
                    f"{index} is {status}: {review.get('reason', '')}"
                )
            review_log.append(
                {
                    "episode_index": index,
                    "status": status,
                    "issue_types": review.get("issue_types", []),
                    "evidence_frames": review.get("evidence_frames", []),
                    "reason": review.get("reason", ""),
                    "corrected_episode": (
                        asdict(corrected)
                        if isinstance(corrected, EpisodeDraft)
                        else None
                    ),
                }
            )
        if self.logger:
            self.logger.emit(
                "episode_factual_audit_completed",
                episode_count=len(merged),
                selected_episode_count=len(selected),
                accepted_episode_indexes=accepted_indexes,
                supported_episode_indexes=supported_indexes,
                unresolved_episode_indexes=unresolved_indexes,
                reviews=review_log,
                audit_model=audit_model or "primary",
            )
        return merged, unresolved_errors

    @classmethod
    def _fact_audit_source_excerpt(
        cls, source_text: str, episodes: list[EpisodeDraft]
    ) -> str:
        """Select participant-centred literal lines for one evidence batch.

        This is a latency optimisation, not a semantic filter: attributed
        dialogue keeps every line spoken by a named participant plus adjacent
        context.  If the batch cannot be located reliably, the complete Source
        is retained.  Returned evidence lines are never rewritten.
        """

        if len(source_text) <= 3_500:
            return source_text
        tokens: set[str] = set()
        for episode in episodes:
            for participant in episode.participants:
                for value in re.split(r"\s*[/／|]\s*|[()（）]", participant):
                    normalized = value.strip()
                    if len(normalized) >= 2:
                        tokens.add(normalized.casefold())
        if not tokens:
            return source_text
        lines = source_text.splitlines()
        max_token_lines = max(6, int(len(lines) * 0.35))
        tokens = {
            token
            for token in tokens
            if sum(token in line.casefold() for line in lines) <= max_token_lines
        }
        if not tokens:
            return source_text
        selected: set[int] = set()
        for index, line in enumerate(lines):
            folded = line.casefold()
            if any(token in folded for token in tokens):
                selected.update(range(max(0, index - 2), min(len(lines), index + 3)))
        if not selected:
            return source_text
        # Preserve source metadata required by prompt/version diagnostics.
        selected.update(range(min(4, len(lines))))
        ordered = sorted(selected)
        chunks: list[str] = []
        start = previous = ordered[0]
        for index in ordered[1:]:
            if index == previous + 1:
                previous = index
                continue
            chunks.append("\n".join(lines[start : previous + 1]))
            start = previous = index
        chunks.append("\n".join(lines[start : previous + 1]))
        excerpt = "\n\n[...source excerpt gap...]\n\n".join(chunks)
        # Tiny excerpts often mean an alias collision rather than reliable
        # evidence coverage; use the complete Source in that case.
        return source_text if len(excerpt) < 300 else excerpt

    @staticmethod
    def _normalize_evidence_surface(value: str) -> str:
        # Newlines and layout spaces are presentation only.  Removing all
        # whitespace lets a verbatim multi-line quote match the Source without
        # accepting translated or paraphrased words.
        normalized = re.sub(r"\s+", "", value).casefold()
        # Japanese source text often carries a short hiragana reading after a
        # kanji while the model copies the same surface without the ruby.  This
        # is a presentation difference, not a different argument.  Do not drop
        # arbitrary parentheses: only an inline Han + hiragana reading pair.
        return re.sub(
            r"(?<=[\u3400-\u9fff])（[ぁ-ゖー]{1,24}）",
            "",
            normalized,
        )

    @classmethod
    def _evidence_surface_parts(cls, value: str) -> list[str]:
        normalized = cls._normalize_evidence_surface(value)
        return [
            part.strip()
            for part in re.split(r"[,，、;；/／]+", normalized)
            if part.strip()
        ]

    @classmethod
    def _validate_fact_review_evidence(
        cls, source_text: str, reviews: dict[int, dict[str, object]]
    ) -> list[str]:
        """Require every semantic frame to be traceable to literal Source text.

        The validator deliberately knows nothing about any story, role title or
        language-specific predicate.  It only checks the evidence contract:
        the quoted clause and its source-side arguments must be visible in the
        Source before a model correction is allowed to replace an Episode.
        """

        source_surface = cls._normalize_evidence_surface(source_text)
        errors: list[str] = []
        for episode_index, review in sorted(reviews.items()):
            status = str(review.get("status", "")).strip().casefold()
            raw_frames = review.get("evidence_frames", [])
            if not isinstance(raw_frames, list) or not raw_frames:
                errors.append(f"review {episode_index}: missing evidence_frames")
                continue
            valid_frame_count = 0
            for frame_index, frame in enumerate(raw_frames):
                if not isinstance(frame, dict):
                    continue
                quote = cls._normalize_evidence_surface(str(frame.get("quote", "")))
                if not quote or quote not in source_surface:
                    continue
                # A supported verdict does not write anything, so literal
                # quote + predicate grounding is sufficient.  A correction
                # can replace durable data and therefore must additionally
                # ground both argument spans before admission.
                grounded_fields = (
                    ("subject_span", "predicate_span", "object_span")
                    if status == "corrected"
                    else ("predicate_span",)
                )
                frame_valid = True
                for field in grounded_fields:
                    parts = cls._evidence_surface_parts(str(frame.get(field, "")))
                    if any(part not in quote for part in parts):
                        frame_valid = False
                        break
                if frame_valid:
                    valid_frame_count += 1
            if valid_frame_count == 0:
                strictness = (
                    "literal quote, predicate and arguments"
                    if status == "corrected"
                    else "literal quote and predicate"
                )
                errors.append(f"review {episode_index}: no frame grounds {strictness}")
        return errors

    def _needs_temporal_audit(self, episodes: list[EpisodeDraft]) -> bool:
        for episode in episodes:
            searchable = f"{episode.text}\n{episode.story_time_text}".casefold()
            if any(
                marker.casefold() in searchable
                for marker in self._TEMPORAL_RISK_MARKERS
            ):
                return True
        return False

    def _audit_episode_quality(
        self,
        source_text: str,
        timeline_scope: str,
        episodes: list[EpisodeDraft],
        *,
        temporal_risk: bool,
        granularity_risk: bool,
    ) -> tuple[list[EpisodeDraft], list[str]]:
        prompt = episode_quality_audit_prompt(
            source_text, timeline_scope, [asdict(episode) for episode in episodes]
        )
        if self.logger:
            self.logger.emit(
                "episode_quality_audit_started",
                candidate_episode_count=len(episodes),
                temporal_risk=temporal_risk,
                granularity_risk=granularity_risk,
            )
        payload = self.model.chat_json(EPISODE_QUALITY_AUDIT_SYSTEM, prompt)
        audited, errors = parse_episode_payload(payload)
        if errors or not audited:
            correction = self.model.chat_json(
                EPISODE_QUALITY_AUDIT_SYSTEM,
                audit_retry_prompt(prompt, errors),
            )
            audited, errors = parse_episode_payload(correction)
        if errors or not audited:
            final_errors = [f"episode quality audit: {message}" for message in errors]
            if not audited:
                final_errors.append("episode quality audit: no valid Episode returned")
            if self.logger:
                self.logger.emit(
                    "validation_failed",
                    stage="episode_quality_audit",
                    errors=final_errors,
                )
            return episodes, final_errors
        if self.episode_audit_expanded_too_far(len(episodes), len(audited)):
            if self.logger:
                self.logger.emit(
                    "episode_quality_audit_rejected",
                    reason="excessive_episode_expansion",
                    previous_episode_count=len(episodes),
                    audited_episode_count=len(audited),
                    max_expansion_ratio=self._AUDIT_MAX_EXPANSION_RATIO,
                    max_expansion_absolute=self._AUDIT_MAX_EXPANSION_ABSOLUTE,
                )
            # The combined audit is a replaceable quality layer.  A sudden
            # explosion usually means it split dialogue lines instead of event
            # stages; the pre-audit candidates remain valid and evidence-safe.
            return episodes, []
        if self.logger:
            self.logger.emit(
                "episode_quality_audit_completed",
                previous_episode_count=len(episodes),
                audited_episode_count=len(audited),
                temporal_risk=temporal_risk,
                granularity_risk=granularity_risk,
            )
        return audited, []

    @classmethod
    def episode_audit_expanded_too_far(
        cls, previous_count: int, audited_count: int
    ) -> bool:
        previous = max(1, int(previous_count))
        audited = max(0, int(audited_count))
        return (
            audited > previous * cls._AUDIT_MAX_EXPANSION_RATIO
            and audited - previous >= cls._AUDIT_MAX_EXPANSION_ABSOLUTE
        )

    def _audit_episode_times(
        self,
        source_text: str,
        timeline_scope: str,
        episodes: list[EpisodeDraft],
    ) -> tuple[list[EpisodeDraft], list[str]]:
        prompt = temporal_audit_prompt(
            source_text, timeline_scope, [asdict(episode) for episode in episodes]
        )
        if self.logger:
            self.logger.emit(
                "temporal_audit_started", candidate_episode_count=len(episodes)
            )
        payload = self.model.chat_json(TEMPORAL_AUDIT_SYSTEM, prompt)
        audited, errors = parse_episode_payload(payload)
        if errors or not audited:
            correction = self.model.chat_json(
                TEMPORAL_AUDIT_SYSTEM,
                audit_retry_prompt(prompt, errors),
            )
            audited, errors = parse_episode_payload(correction)
        if errors or not audited:
            final_errors = [f"temporal audit: {message}" for message in errors]
            if not audited:
                final_errors.append("temporal audit: no valid Episode returned")
            if self.logger:
                self.logger.emit(
                    "validation_failed", stage="temporal_audit", errors=final_errors
                )
            return episodes, final_errors
        if self.logger:
            self.logger.emit(
                "temporal_audit_completed",
                previous_episode_count=len(episodes),
                audited_episode_count=len(audited),
            )
        return audited, []

    def _needs_granularity_audit(self, episodes: list[EpisodeDraft]) -> bool:
        if any(
            marker.casefold() in episode.text.casefold()
            for episode in episodes
            for marker in self._SPECULATIVE_IDENTITY_MARKERS
        ):
            return True
        if any(
            marker in episode.event_type
            for episode in episodes
            for marker in ("章节开始", "标题")
        ):
            return True
        if len(episodes) < 10:
            return False
        short_or_trivial = sum(
            len(episode.text) < 50
            or any(
                marker in episode.event_type for marker in self._TRIVIAL_EVENT_MARKERS
            )
            for episode in episodes
        )
        return short_or_trivial / len(episodes) >= 0.4

    def _audit_episode_granularity(
        self,
        source_text: str,
        timeline_scope: str,
        episodes: list[EpisodeDraft],
    ) -> tuple[list[EpisodeDraft], list[str]]:
        prompt = granularity_audit_prompt(
            source_text, timeline_scope, [asdict(episode) for episode in episodes]
        )
        if self.logger:
            self.logger.emit(
                "granularity_audit_started", candidate_episode_count=len(episodes)
            )
        payload = self.model.chat_json(GRANULARITY_AUDIT_SYSTEM, prompt)
        audited, errors = parse_episode_payload(payload)
        if errors or not audited:
            correction = self.model.chat_json(
                GRANULARITY_AUDIT_SYSTEM,
                audit_retry_prompt(prompt, errors, granularity=True),
            )
            audited, errors = parse_episode_payload(correction)
        if errors or not audited:
            final_errors = [f"granularity audit: {message}" for message in errors]
            if not audited:
                final_errors.append("granularity audit: no valid Episode returned")
            if self.logger:
                self.logger.emit(
                    "validation_failed", stage="granularity_audit", errors=final_errors
                )
            return episodes, final_errors
        if self.logger:
            self.logger.emit(
                "granularity_audit_completed",
                previous_episode_count=len(episodes),
                audited_episode_count=len(audited),
            )
        return audited, []

    def extract_concepts(
        self, episode: EpisodeDraft, source_text: str = ""
    ) -> tuple[list[ConceptDraft], list[str]]:
        alias_context = self._speaker_alias_context(source_text)
        base_prompt = self._concept_prompt(episode, alias_context)
        if self._uses_natural_text_output():
            response = self._request_natural_text(
                self.concept_system,
                base_prompt,
            )
            valid, errors = parse_concept_text(
                response,
                evidence_text=episode.text,
            )
        else:
            payload = self.model.chat_json(
                self.concept_system,
                base_prompt,
            )
            valid, errors = parse_concept_payload(payload)
        if errors and self.logger:
            self.logger.emit("validation_failed", stage="concepts", errors=errors)
        if errors:
            retry_prompt = partial_item_retry_prompt(base_prompt, errors)
            if self._uses_natural_text_output():
                correction = self._request_natural_text(
                    self.concept_system,
                    retry_prompt,
                )
                corrected, errors = parse_concept_text(
                    correction,
                    evidence_text=episode.text,
                )
            else:
                correction = self.model.chat_json(
                    self.concept_system,
                    retry_prompt,
                )
                corrected, errors = parse_concept_payload(correction)
            valid.extend(corrected)
        if errors and self._fallback_model():
            fallback_prompt = partial_item_retry_prompt(
                base_prompt,
                errors,
                prior_attempts=2,
            )
            if self._uses_natural_text_output():
                fallback = self._request_natural_text(
                    self.concept_system,
                    fallback_prompt,
                    model=self._fallback_model(),
                    allow_fallback=False,
                )
                corrected, errors = parse_concept_text(
                    fallback,
                    evidence_text=episode.text,
                )
            else:
                fallback = self.model.chat_json(
                    self.concept_system,
                    fallback_prompt,
                    model=self._fallback_model(),
                    allow_fallback=False,
                )
                corrected, errors = parse_concept_payload(fallback)
            valid.extend(corrected)
        self._attach_literal_source_aliases(valid, alias_context)
        return valid, errors

    def extract_concepts_batch(
        self, episodes: list[EpisodeDraft], source_text: str = ""
    ) -> tuple[list[list[ConceptDraft]], list[str]]:
        if not episodes:
            return [], []
        if len(episodes) == 1:
            concepts, errors = self.extract_concepts(episodes[0], source_text)
            return [concepts], errors
        episode_payloads = [
            {
                "episode_index": index,
                "text": episode.text,
                "participants": episode.participants,
                "evidence_origin": episode.evidence_origin,
                "epistemic_status": episode.epistemic_status,
                "generation": episode.generation,
                "epistemic_note": episode.epistemic_note,
            }
            for index, episode in enumerate(episodes)
        ]
        prompt = concept_batch_prompt(
            episode_payloads,
            self._speaker_alias_context(source_text),
            self.concept_profile,
            self.concept_target_min,
            self.concept_target_max,
        )
        if self._uses_natural_text_output():
            response = self._request_natural_text(
                self.concept_system,
                prompt,
            )
            groups, group_errors, global_errors = parse_concept_batch_text(
                response,
                len(episodes),
                evidence_texts=[episode.text for episode in episodes],
            )
        else:
            payload = self.model.chat_json(
                self.concept_system,
                prompt,
            )
            groups, group_errors, global_errors = parse_concept_batch_payload(
                payload, len(episodes)
            )
        errors = list(global_errors)
        result: list[list[ConceptDraft]] = []
        for episode_index, episode in enumerate(episodes):
            if episode_index in group_errors:
                if self.logger:
                    self.logger.emit(
                        "validation_failed",
                        stage="concepts_batch",
                        episode_index=episode_index,
                        errors=group_errors[episode_index],
                    )
                # Retry only the affected Episode through the existing
                # validated/fallback path.  Valid batch groups are retained.
                concepts, retry_errors = self.extract_concepts(episode, source_text)
                result.append(concepts)
                errors.extend(
                    f"episode {episode_index}: {message}" for message in retry_errors
                )
            else:
                result.append(groups.get(episode_index, []))
        alias_context = self._speaker_alias_context(source_text)
        for concepts in result:
            self._attach_literal_source_aliases(concepts, alias_context)
        return result, errors
