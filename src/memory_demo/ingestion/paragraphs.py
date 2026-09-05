from __future__ import annotations

import re

from memory_demo.config import ParagraphConfig
from memory_demo.types import ParagraphDraft


_RECORD_START = re.compile(r"(?m)^\[record:\s*\d+\]")
_SPEAKER_RAW = re.compile(r"^\[speaker_raw:\s*(.*?)\]$")


class ParagraphSegmenter:
    """Split a Source into overlapping, record-aligned retrieval paragraphs.

    Paragraphs are deliberately deterministic.  They preserve wording that may
    disappear from an Episode summary without becoming graph nodes themselves.
    """

    def __init__(self, config: ParagraphConfig):
        self.config = config

    @staticmethod
    def _prefix_and_records(source_text: str) -> tuple[str, list[str]]:
        matches = list(_RECORD_START.finditer(source_text))
        if not matches:
            blocks = [item.strip() for item in source_text.split("\n\n") if item.strip()]
            return "", blocks
        prefix = source_text[: matches[0].start()].strip()
        records = [
            source_text[match.start() : matches[index + 1].start()].strip()
            if index + 1 < len(matches)
            else source_text[match.start() :].strip()
            for index, match in enumerate(matches)
        ]
        return prefix, [record for record in records if record]

    def segment(self, source_text: str) -> list[ParagraphDraft]:
        if not self.config.enabled or not source_text.strip():
            return []
        prefix, records = self._prefix_and_records(source_text)
        if not records:
            return []

        drafts: list[ParagraphDraft] = []
        start = 0
        while start < len(records):
            end = start
            body_chars = 0
            while end < len(records):
                addition = len(records[end]) + (2 if end > start else 0)
                if end > start and body_chars + addition > self.config.max_chars:
                    break
                body_chars += addition
                end += 1
                if body_chars >= self.config.target_chars:
                    break
            if end <= start:
                end = start + 1

            # Avoid a tiny final paragraph when it can be kept on a natural
            # record boundary.  max_chars remains a soft limit, just like the
            # Source segmenter, because splitting one dialogue record is worse.
            tail_chars = sum(len(record) + 2 for record in records[end:])
            if end < len(records) and tail_chars < self.config.minimum_chars:
                end = len(records)

            body = "\n\n".join(records[start:end])
            text = f"{prefix}\n\n{body}" if prefix else body
            drafts.append(ParagraphDraft(len(drafts), text.strip()))
            if end >= len(records):
                break

            overlap = 0
            next_start = end
            while next_start > start:
                candidate = len(records[next_start - 1]) + 2
                if overlap > 0 and overlap + candidate > self.config.overlap_chars:
                    break
                next_start -= 1
                overlap += candidate
            start = max(start + 1, next_start)
        return drafts

    @staticmethod
    def embedding_text(paragraph_text: str) -> str:
        """Remove storage/provenance scaffolding from Paragraph embeddings.

        The complete Paragraph remains stored in SQLite.  Only the model input
        is cleaned so repeated alias legends and raw script directives cannot
        dominate semantic similarity.  Human-readable multilingual lines are
        intentionally retained because multilingual retrieval is a demo goal.
        """
        cleaned: list[str] = []
        in_alias_legend = False
        in_script_raw = False
        for raw_line in paragraph_text.splitlines():
            line = raw_line.strip()
            if in_script_raw:
                if line.endswith("]"):
                    in_script_raw = False
                continue
            if not line:
                if cleaned and cleaned[-1] != "":
                    cleaned.append("")
                continue
            if line == "[speaker_alias_legend]":
                in_alias_legend = True
                continue
            if line.startswith("[record:"):
                in_alias_legend = False
                continue
            if in_alias_legend:
                continue
            if line.startswith("[script_raw:"):
                if not line.endswith("]"):
                    in_script_raw = True
                continue
            if line.startswith(("[source_key:", "[segment_index:")):
                continue
            speaker = _SPEAKER_RAW.match(line)
            if speaker:
                value = speaker.group(1).strip()
                if value:
                    cleaned.append(f"speaker: {value}")
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            cleaned.append(line)
        value = "\n".join(cleaned).strip()
        return value or paragraph_text.strip()
