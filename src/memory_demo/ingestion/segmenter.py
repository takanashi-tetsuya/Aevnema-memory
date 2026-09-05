from __future__ import annotations

from memory_demo.config import SegmentConfig
from memory_demo.types import NormalizedBlock, SourceSegment


LANGUAGE_ORDER = ("zh-CN", "ja", "en", "zh-TW", "th", "unknown")


def render_block(
    block: NormalizedBlock, *, include_speaker_aliases: bool = True
) -> str:
    lines = [f"[record: {block.record_index}]"]
    if block.speaker_raw:
        lines.append(f"[speaker_raw: {block.speaker_raw}]")
    for key in (
        "evidence_origin",
        "epistemic_status",
        "evidence_generation",
        "epistemic_note",
        "document_role",
        "document_style",
    ):
        value = block.metadata.get(key)
        if value is not None and str(value).strip():
            lines.append(f"[{key}: {str(value).strip()}]")
    speaker_aliases = block.metadata.get("speaker_aliases", {})
    if include_speaker_aliases and isinstance(speaker_aliases, dict) and speaker_aliases:
        rendered_aliases = " | ".join(
            f"{language}={value}"
            for language, value in speaker_aliases.items()
            if str(value).strip()
        )
        if rendered_aliases:
            lines.append(f"[speaker_aliases: {rendered_aliases}]")
    script_raw = str(block.metadata.get("script_raw", "")).strip()
    if script_raw:
        lines.append(f"[script_raw: {script_raw}]")
    emitted: set[str] = set()
    for language in LANGUAGE_ORDER:
        value = block.languages.get(language, "").strip()
        if value:
            lines.append(f"{language}: {value}")
            emitted.add(language)
    for language, value in block.languages.items():
        if language not in emitted and value.strip():
            lines.append(f"{language}: {value.strip()}")
    return "\n".join(lines)


def render_speaker_alias(block: NormalizedBlock) -> str:
    aliases = block.metadata.get("speaker_aliases", {})
    if not block.speaker_raw or not isinstance(aliases, dict) or not aliases:
        return ""
    rendered = " | ".join(
        f"{language}={value}"
        for language, value in aliases.items()
        if str(value).strip()
    )
    return f"{block.speaker_raw}: {rendered}" if rendered else ""


def render_segment_text(
    source_key: str,
    segment_index: int,
    blocks: list[NormalizedBlock],
    rendered_blocks: list[str],
) -> str:
    legend_lines = list(
        dict.fromkeys(
            alias
            for alias in (render_speaker_alias(block) for block in blocks)
            if alias
        )
    )
    legend = ""
    if legend_lines:
        legend = "[speaker_alias_legend]\n" + "\n".join(legend_lines) + "\n\n"
    body = legend + "\n\n".join(rendered_blocks)
    return (
        f"[source_key: {source_key}]\n"
        f"[segment_index: {segment_index}]\n\n{body}"
    )


class NaturalSegmenter:
    def __init__(self, config: SegmentConfig):
        self.config = config

    def segment(
        self, source_key: str, blocks: list[NormalizedBlock]
    ) -> list[SourceSegment]:
        blocks = [
            block
            for block in blocks
            if block.metadata.get("document_role") != "front_matter"
        ]
        if not blocks:
            return []
        # Speaker aliases are emitted once in a compact segment-level legend,
        # rather than repeated on every dialogue record.
        rendered = [render_block(block, include_speaker_aliases=False) for block in blocks]
        segments: list[SourceSegment] = []
        start = 0
        previous_end = -1
        while start < len(blocks):
            reference_style = (
                blocks[start].metadata.get("document_style") == "reference"
            )
            reference_limit = max(
                1, int(self.config.reference_max_blocks)
            )
            end = start
            total = 0
            seen_aliases: set[str] = set()
            best_boundary: int | None = None
            while end < len(blocks):
                # A format-declared chapter/scene boundary is stronger than a
                # character target.  Never recombine two such sections into a
                # single Source merely because both happen to be short.
                if end > start and blocks[end].boundary_score >= 3:
                    break
                if (
                    reference_style
                    and end > start
                    and end - start >= reference_limit
                ):
                    break
                block_size = len(rendered[end]) + 2
                alias = render_speaker_alias(blocks[end])
                alias_size = 0
                if alias and alias not in seen_aliases:
                    alias_size = len(alias) + 1
                if end > start and total + block_size + alias_size > self.config.max_chars:
                    break
                candidate_text = render_segment_text(
                    source_key,
                    len(segments),
                    blocks[start : end + 1],
                    rendered[start : end + 1],
                )
                # max_chars is a hard limit for the complete model input, not
                # merely for the dialogue body.  Source/segment headers and
                # the alias legend must count toward the same budget.
                if end > start and len(candidate_text) > self.config.max_chars:
                    break
                total += block_size + alias_size
                if alias:
                    seen_aliases.add(alias)
                end += 1
                if (
                    end - start >= self.config.minimum_blocks
                    and end < len(blocks)
                    and blocks[end].boundary_score >= 2
                    and total >= self.config.target_chars // 2
                ):
                    best_boundary = end
                if total >= self.config.target_chars and best_boundary is not None:
                    end = best_boundary
                    break
                if total >= self.config.max_chars:
                    break
            if end <= start:
                end = start + 1
            raw_text = render_segment_text(
                source_key,
                len(segments),
                blocks[start:end],
                rendered[start:end],
            )
            segments.append(
                SourceSegment(
                    source_key=source_key,
                    segment_index=len(segments),
                    raw_text=raw_text,
                    first_record=blocks[start].record_index,
                    last_record=blocks[end - 1].record_index,
                )
            )
            if end >= len(blocks):
                break
            if reference_style:
                # Every reference record is independent evidence. Repeating it
                # in an overlap would create duplicate Episodes on import.
                previous_end = end - 1
                start = end
                continue
            if blocks[end].boundary_score >= 3:
                # Overlap is useful inside one scene, but copying the previous
                # scene across an explicit boundary defeats the boundary and
                # reintroduces participant/name contamination.
                previous_end = end - 1
                start = end
                continue
            overlap_size = 0
            next_start = end
            while next_start > start:
                candidate_size = len(rendered[next_start - 1]) + 2
                # Overlap only complete records that fit the declared budget.
                # Text/Markdown adapters can emit one multi-line record that
                # is thousands of characters long. Copying that whole record
                # merely to guarantee non-zero overlap duplicates entire
                # scenes and creates duplicate Episodes downstream.
                # Allow one moderately oversized dialogue record so ordinary
                # multilingual turns still retain context.  The bounded slack
                # is intentionally too small for a multi-thousand-character
                # transcript record.
                allowed_overlap = (
                    max(
                        self.config.overlap_chars * 2,
                        self.config.overlap_chars + 256,
                    )
                    if overlap_size == 0
                    else self.config.overlap_chars
                )
                if overlap_size + candidate_size > allowed_overlap:
                    break
                next_start -= 1
                overlap_size += candidate_size
            if next_start <= previous_end:
                next_start = previous_end + 1
            previous_end = end - 1
            start = max(next_start, start + 1)
        return segments
