from __future__ import annotations

from pathlib import Path
import json
import re

from memory_demo.adapters.base import (
    InputAdapter,
    normalize_memory_evidence,
    read_text_document,
    split_text_chunks,
)
from memory_demo.types import NormalizedBlock


class TextAdapter(InputAdapter):
    _MEMORY_DIRECTIVE = re.compile(
        r"^\s*\[\[memory\s+(\{.*\})\]\]\s*(?:\r?\n|$)",
        re.IGNORECASE,
    )
    _REFERENCE_TYPE = re.compile(
        r"^\s*\[资料类型[：:]\s*([^\]|]+)", re.MULTILINE
    )
    _DECORATED_REFERENCE_LABEL = re.compile(
        r"^\s*【(?P<label>[^】\n]{1,180})】"
    )
    _INFERENTIAL_REFERENCE_TYPES = (
        "推测",
        "推论",
        "考据",
        "社区观点",
        "剧情解读",
        "尚未确认",
        "资料边界",
        "事实边界",
    )
    _REPORTED_REFERENCE_TYPES = (
        "来源网页",
        "网络补充",
        "资料性质",
    )
    # A plain-text file often contains several scenes without blank lines.
    # Treat only strong, format-level headings as hard boundaries; the rules
    # deliberately know nothing about a corpus, character or story language.
    _MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")
    _DECORATED_HEADING = re.compile(
        r"^\s*(?:[-=*~_]{3,}\s*)?"
        r"(?:【[^】\n]{1,180}】|\[[^\]\n]{1,180}\])"
        r"(?:\s*[-=*~_]{3,})?\s*$"
    )
    _SURROUNDED_HEADING = re.compile(
        r"^\s*[-=*~_]{3,}\s+\S.{0,180}?\s+[-=*~_]{3,}\s*$"
    )
    _HORIZONTAL_SEPARATOR = re.compile(r"^\s*[-=*~_━─—]{3,}\s*$")
    _NORMALIZED_METADATA = re.compile(
        r"^\s*\[(?:source_key|segment_index|record|speaker_raw|"
        r"speaker_aliases|speaker_alias_legend|script_raw|evidence_origin|"
        r"epistemic_status|evidence_generation|epistemic_note|document_role|"
        r"document_style)\s*(?::|\])",
        re.IGNORECASE,
    )
    _STANDALONE_LABEL = re.compile(
        r"^\s*([^\s:：][^\r\n:：]{0,79}?)\s*[:：]\s*$"
    )

    def __init__(self, language: str = "unknown"):
        self.language = language

    def supports(self, path: Path) -> bool:
        return path.suffix.casefold() in {".txt", ".md"}

    @classmethod
    def _dialogue_label(cls, paragraph: str) -> tuple[str, str] | None:
        """Return a leading ``label:`` and its body without guessing names.

        Human-readable transcripts commonly put the actor on a line of its
        own and the utterance on the following lines.  Preserving that label
        as ``speaker_raw`` is structural parsing, not entity resolution.  The
        document-level gate in ``_is_dialogue_document`` prevents ordinary
        prose headings from being reinterpreted as speakers.
        """

        lines = str(paragraph).splitlines()
        if len(lines) < 2:
            return None
        matched = cls._STANDALONE_LABEL.fullmatch(lines[0])
        if not matched:
            return None
        body = "\n".join(lines[1:]).strip()
        if not body:
            return None
        return matched.group(1).strip(), body

    @classmethod
    def _is_dialogue_document(cls, paragraphs: list[str]) -> bool:
        """Detect repeated turn-shaped records using document structure.

        This deliberately avoids corpus names and language-specific role
        lists.  Three or more standalone labels, with either recurrence or a
        meaningful share of the document, are enough to preserve turn actors.
        """

        labels = [
            parsed[0]
            for paragraph in paragraphs
            if (parsed := cls._dialogue_label(paragraph)) is not None
        ]
        if len(labels) < 3:
            return False
        recurring = len(set(labels)) < len(labels)
        density = len(labels) / max(1, len(paragraphs))
        return recurring or density >= 0.4

    @classmethod
    def _split_structural_sections(cls, text: str) -> list[tuple[str, bool]]:
        """Split format-declared scenes before character-count chunking.

        The boolean records whether the section begins at a strong structural
        boundary.  Content-free horizontal separators are discarded while the
        following content inherits their boundary.  This prevents a later
        segmenter from putting unrelated scenes back into one model request.
        """

        sections: list[tuple[str, bool]] = []
        current: list[str] = []
        current_is_boundary = False
        pending_boundary = False

        def flush() -> None:
            nonlocal current, current_is_boundary
            value = "\n".join(current).strip()
            if value:
                sections.append((value, current_is_boundary))
            current = []
            current_is_boundary = False

        for line in str(text).splitlines():
            stripped = line.strip()
            if cls._HORIZONTAL_SEPARATOR.fullmatch(stripped):
                flush()
                pending_boundary = True
                continue
            is_heading = not cls._NORMALIZED_METADATA.match(stripped) and bool(
                cls._MARKDOWN_HEADING.match(line)
                or cls._DECORATED_HEADING.fullmatch(stripped)
                or cls._SURROUNDED_HEADING.fullmatch(stripped)
            )
            if is_heading:
                flush()
                current_is_boundary = True
                current.append(line)
                pending_boundary = False
                continue
            if not current:
                current_is_boundary = pending_boundary
                pending_boundary = False
            current.append(line)
        flush()
        return sections

    def read_blocks(self, path: Path) -> list[NormalizedBlock]:
        text = read_text_document(path)
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
        dialogue_document = self._is_dialogue_document(paragraphs)
        reference_indexes = [
            index
            for index, paragraph in enumerate(paragraphs)
            if self._REFERENCE_TYPE.search(paragraph)
            or self._DECORATED_REFERENCE_LABEL.match(paragraph)
        ]
        reference_style = (
            sum(
                bool(self._REFERENCE_TYPE.search(paragraph))
                for paragraph in paragraphs
            )
            >= 2
            or (len(reference_indexes) >= 2 and not dialogue_document)
        )
        has_explicit_section = any(
            hard_boundary
            for paragraph in paragraphs
            for _section, hard_boundary in self._split_structural_sections(paragraph)
        )
        has_dialogue_turn = any(
            self._dialogue_label(paragraph) is not None
            for paragraph in paragraphs
        )
        dialogue_style = not reference_style and (
            dialogue_document
            or (has_explicit_section and has_dialogue_turn)
        )
        chunks: list[tuple[str, dict, int, str]] = []
        first_reference_index = min(reference_indexes) if reference_indexes else None
        for paragraph_index, paragraph in enumerate(paragraphs):
            metadata: dict = (
                {"document_style": "reference"} if reference_style else {}
            )
            reference_type = self._REFERENCE_TYPE.search(paragraph)
            decorated_reference = self._DECORATED_REFERENCE_LABEL.match(paragraph)
            reference_label = (
                reference_type.group(1).strip()
                if reference_type
                else decorated_reference.group("label").strip()
                if decorated_reference
                else ""
            )
            if (
                reference_style
                and first_reference_index is not None
                and paragraph_index < first_reference_index
            ):
                metadata["document_role"] = "front_matter"
            if reference_label and any(
                marker in reference_label
                for marker in self._INFERENTIAL_REFERENCE_TYPES
            ):
                metadata.update(
                    {
                        "evidence_origin": "importer",
                        "epistemic_status": "speculative",
                        "evidence_generation": 1,
                        "epistemic_note": f"导入文档标记为{reference_label}",
                    }
                )
            elif reference_label and any(
                marker in reference_label
                for marker in self._REPORTED_REFERENCE_TYPES
            ):
                metadata.update(
                    {
                        "evidence_origin": "importer",
                        "epistemic_status": "reported",
                        "evidence_generation": 0,
                        "epistemic_note": f"导入文档标记为{reference_label}",
                    }
                )
            directive = self._MEMORY_DIRECTIVE.match(paragraph)
            if directive:
                try:
                    metadata.update(
                        normalize_memory_evidence(
                            json.loads(directive.group(1))
                        )
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid [[memory ...]] directive in {path}: {exc}"
                    ) from exc
                paragraph = paragraph[directive.end() :].strip()
                if not paragraph:
                    continue
            speaker_raw = ""
            if dialogue_style:
                dialogue = self._dialogue_label(paragraph)
                if dialogue is not None:
                    speaker_raw, paragraph = dialogue
            # Reference records already have their own bounded, non-overlapping
            # batching policy.  Turning every bracketed fact header into a hard
            # scene would defeat reference_max_blocks and multiply model calls.
            sections = (
                [(paragraph, False)]
                if reference_style
                else self._split_structural_sections(paragraph)
            )
            for section_position, (section, hard_boundary) in enumerate(sections):
                section_chunks = split_text_chunks(section)
                for chunk_position, chunk in enumerate(section_chunks):
                    if chunk_position == 0 and hard_boundary:
                        boundary_score = 3
                    elif chunk_position == 0 and section_position > 0:
                        boundary_score = 3
                    else:
                        boundary_score = 2
                    chunks.append(
                        (chunk, dict(metadata), boundary_score, speaker_raw)
                    )
        # In a dialogue document, a hard-boundary single line followed by one
        # or more speaker-free records and then a dialogue turn is structural
        # scene navigation. Keep it out of semantic extraction and transfer the
        # boundary to the first real turn. Paragraph/Source storage still keeps
        # the original file, so this does not destroy evidence.
        for index, (text, metadata, score, speaker) in enumerate(chunks):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if (
                score < 3
                or speaker
                or len(lines) != 1
                or len(lines[0]) > 240
                or metadata.get("document_style") == "reference"
            ):
                continue
            first_turn = next(
                (
                    position
                    for position in range(index + 1, min(len(chunks), index + 5))
                    if chunks[position][3]
                ),
                None,
            )
            if first_turn is None:
                continue
            for position in range(index, first_turn):
                if chunks[position][3]:
                    break
                chunks[position][1].setdefault("document_role", "front_matter")
            turn_text, turn_metadata, turn_score, turn_speaker = chunks[first_turn]
            chunks[first_turn] = (
                turn_text,
                turn_metadata,
                max(3, turn_score),
                turn_speaker,
            )
        hard_boundaries = [
            index
            for index, (_text, _metadata, score, _speaker) in enumerate(chunks)
            if score >= 3
        ]
        # A short, speaker-free prefix before several explicit sections is
        # document front matter.  Record that structural role without trying
        # to understand its vocabulary or the corpus domain.
        first_boundary = hard_boundaries[0] if hard_boundaries else 0
        prefix_has_dialogue = any(
            self._dialogue_label(text) is not None or bool(speaker)
            for text, _metadata, _score, speaker in chunks[:first_boundary]
        )
        body_has_dialogue = any(
            self._dialogue_label(text) is not None or bool(speaker)
            for text, _metadata, _score, speaker in chunks[first_boundary:]
        )
        repeated_reference_sections = len(hard_boundaries) >= 3
        titled_transcript = bool(hard_boundaries) and body_has_dialogue
        if (
            (repeated_reference_sections or titled_transcript)
            and 0 < first_boundary <= 4
            and not prefix_has_dialogue
        ):
            for index in range(first_boundary):
                chunks[index][1]["document_role"] = "front_matter"
        return [
            NormalizedBlock(
                record_index=index,
                speaker_raw=speaker_raw,
                languages={self.language: "\n".join(line.rstrip() for line in paragraph.splitlines())},
                boundary_score=0 if index == 0 else boundary_score,
                metadata=metadata,
            )
            for index, (paragraph, metadata, boundary_score, speaker_raw) in enumerate(chunks)
        ]
