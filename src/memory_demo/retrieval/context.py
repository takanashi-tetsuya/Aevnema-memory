"""Bounded Source evidence views used by answer verification."""

from __future__ import annotations

import re


_EVIDENCE_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]{2,}"
)
_RECORD_HEADER_RE = re.compile(r"(?m)^\[record:\s*(\d+)\]\s*$")
_SPEAKER_RAW_RE = re.compile(r"(?m)^\[speaker_raw:\s*([^\]]+)\]\s*$")
_TRANSLATION_LINE_RE = re.compile(
    r"(?m)^([A-Za-z]{2}(?:-[A-Za-z]{2})?):\s*(.+)$"
)
_TRANSLATION_PREFIX_RE = re.compile(r"^[A-Za-z]{2}(?:-[A-Za-z]{2})?:")


def _source_record_blocks(raw_text: str) -> dict[str, str]:
    """Return unique, complete raw Source blocks keyed by record id.

    Evidence extraction can use a normalized "reasoning view" which omits
    rendering-only ``script_raw`` or language lines.  The answer layer must
    nevertheless deliver the original Source block, never that normalized
    view.  Duplicate record ids are treated as unusable rather than guessed.
    """

    matches = list(_RECORD_HEADER_RE.finditer(raw_text))
    blocks: dict[str, str] = {}
    duplicates: set[str] = set()
    for index, match in enumerate(matches):
        record_id = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        block = raw_text[match.start():end].strip()
        if record_id in blocks:
            duplicates.add(record_id)
            continue
        blocks[record_id] = block
    for record_id in duplicates:
        blocks.pop(record_id, None)
    return blocks


def _translation_lines(value: str) -> tuple[dict[str, str] | None, str | None]:
    """Parse single-line translation fields, rejecting ambiguous projections.

    Evidence projections are a constrained record format, not free-form prose.
    The persisted view may omit raw-only metadata, but a repeated language
    field or an unlabelled continuation would make it impossible to prove that
    *each* shared translation agrees with the Source.  Until this format grows
    an explicitly specified multiline representation, fail closed instead of
    silently retaining the last duplicate or first line.
    """

    fields: dict[str, str] = {}
    saw_translation = False
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _TRANSLATION_LINE_RE.fullmatch(stripped)
        if match is not None:
            language, text = match.groups()
            normalized = text.strip()
            if not normalized:
                return None, "empty_translation_field"
            if language in fields:
                return None, "duplicate_translation_field"
            fields[language] = normalized
            saw_translation = True
            continue
        if _TRANSLATION_PREFIX_RE.match(stripped):
            return None, "empty_or_malformed_translation_field"
        if saw_translation:
            return None, "multiline_or_unlabelled_translation_field"
    return fields, None


def _projected_record_matches_raw(projected: str, raw_block: str) -> tuple[bool, str]:
    """Verify a normalized evidence record against its immutable raw block.

    A record id alone is not proof.  The speaker (when present) must match and
    every translation present in both representations must agree.  At least
    one actual language line must agree; projection-only languages such as a
    generated ``ko`` line do not make an otherwise matching raw record fail.
    """

    projected_speakers = _SPEAKER_RAW_RE.findall(projected)
    raw_speakers = _SPEAKER_RAW_RE.findall(raw_block)
    if projected_speakers and projected_speakers != raw_speakers:
        return False, "speaker_mismatch"
    projected_lines, projected_error = _translation_lines(projected)
    raw_lines, raw_error = _translation_lines(raw_block)
    if projected_error is not None:
        return False, "projected_" + projected_error
    if raw_error is not None:
        return False, "raw_" + raw_error
    assert projected_lines is not None and raw_lines is not None
    shared_languages = set(projected_lines).intersection(raw_lines)
    if not shared_languages:
        return False, "no_shared_translation_field"
    if not all(
        projected_lines[language] == raw_lines[language]
        for language in shared_languages
    ):
        return False, "shared_translation_mismatch"
    return True, "projected_record_matches_raw"


def _projected_quote_record_ids(value: str) -> list[str]:
    return [match.group(1) for match in _RECORD_HEADER_RE.finditer(value)]


def verified_evidence_view_outcomes(
    raw_text: str, evidence_quotes: list[str] | None
) -> list[dict[str, str | None]]:
    """Return one explicit local verification outcome for each evidence quote.

    This preserves why a quote did not become Source-bound evidence, so the
    delivery layer can distinguish failed validation from an otherwise valid
    quote that simply could not fit in the excerpt budget.
    """

    blocks = _source_record_blocks(raw_text)
    outcomes: list[dict[str, str | None]] = []
    for raw_quote in evidence_quotes or []:
        if not isinstance(raw_quote, str) or not raw_quote.strip():
            outcomes.append({"view": None, "reason": "missing_or_empty_quote"})
            continue
        quote = raw_quote.strip()
        if quote in raw_text:
            outcomes.append({"view": quote, "reason": "exact_quote"})
            continue
        record_ids = _projected_quote_record_ids(quote)
        if not record_ids:
            outcomes.append({"view": None, "reason": "quote_not_structured_by_record"})
            continue
        if len(record_ids) != len(set(record_ids)):
            outcomes.append({"view": None, "reason": "duplicate_projected_record_id"})
            continue
        quote_sections = list(_RECORD_HEADER_RE.finditer(quote))
        projected_blocks: list[str] = []
        for index, match in enumerate(quote_sections):
            end = (
                quote_sections[index + 1].start()
                if index + 1 < len(quote_sections)
                else len(quote)
            )
            projected_blocks.append(quote[match.start():end].strip())
        raw_blocks: list[str] = []
        failure_reason: str | None = None
        for record_id, projected_block in zip(record_ids, projected_blocks):
            raw_block = blocks.get(record_id)
            if raw_block is None:
                failure_reason = "source_record_missing_or_ambiguous"
                break
            matched, reason = _projected_record_matches_raw(projected_block, raw_block)
            if not matched:
                failure_reason = reason
                break
            raw_blocks.append(raw_block)
        if failure_reason is not None:
            outcomes.append({"view": None, "reason": failure_reason})
        else:
            outcomes.append({
                "view": "\n\n".join(raw_blocks),
                "reason": "projected_quote_mapped_to_raw_records",
            })
    return outcomes


def verified_evidence_views(raw_text: str, evidence_quotes: list[str] | None) -> list[str]:
    """Map persisted evidence quotations to verbatim, verified Source views.

    Exact quotation remains the fast path.  For a structured reasoning-view
    quotation that is no longer a byte-for-byte substring because it omitted
    ``script_raw`` or optional language fields, every cited Source record is
    matched independently and the *raw record block* becomes the delivered
    view.  A malformed, ambiguous, or partially mismatching quotation returns
    no view and is never substituted with a lexical neighbour.
    """

    return [
        str(outcome["view"])
        for outcome in verified_evidence_view_outcomes(raw_text, evidence_quotes)
        if outcome["view"] is not None
    ]


def source_excerpt(
    raw_text: str,
    episode_text: str,
    participants: list[str],
    max_chars: int,
    *,
    evidence_quotes: list[str] | None = None,
) -> str:
    """Select bounded, verbatim Source evidence without changing stored Source.

    ``episode_text`` is an extracted summary, so it is not a reliable locator for
    the underlying dialogue by itself.  When the importer has persisted exact
    evidence quotations, they take precedence over lexical heuristics.  This
    keeps the answer/audit view tied to the Source records that grounded the
    Episode rather than to whichever similarly named records happen to fit the
    character budget.
    """

    if max_chars <= 0 or len(raw_text) <= max_chars:
        return raw_text
    verified_quotes = list(dict.fromkeys(verified_evidence_views(
        raw_text, evidence_quotes
    )))
    if verified_quotes:
        blocks = raw_text.split("\n\n")
        header = (
            blocks[0]
            if blocks and blocks[0].lstrip().startswith("[source_key:")
            else ""
        )
        output: list[str] = [header] if header else []
        used = len(header)
        delivered = 0
        for quote in verified_quotes:
            separator = "\n\n" if output else ""
            addition = separator + quote
            if used + len(addition) > max_chars:
                continue
            output.append(addition)
            used += len(addition)
            delivered += 1
        if delivered:
            if delivered < len(verified_quotes):
                marker = "\n\n[...部分已验证 Source 证据因摘要长度限制未交付...]"
                if used + len(marker) <= max_chars:
                    output.append(marker)
            return "".join(output)
        # A known evidence quote that cannot fit is a delivery failure, not a
        # license to substitute lexical neighbours as if they were its proof.
        return (
            (header + "\n\n" if header else "")
            + "[...已验证 Source 证据超过摘要长度限制，未交付...]"
        )[:max_chars]
    keywords = {
        value.strip()
        for value in participants
        if len(value.strip()) >= 2 and value.strip() not in {"???", "[USERNAME]"}
    }
    keywords.update(_EVIDENCE_TOKEN_RE.findall(episode_text))
    blocks = raw_text.split("\n\n")
    scored = [
        (sum(1 for keyword in keywords if keyword in block), index)
        for index, block in enumerate(blocks)
    ]
    relevant = [index for score, index in scored if score > 0]
    if not relevant:
        half = max(1, (max_chars - 40) // 2)
        return (
            raw_text[:half]
            + "\n\n[...Source 中段省略...]\n\n"
            + raw_text[-half:]
        )
    ranked = sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)
    selected: set[int] = {0}
    for score, index in ranked:
        if score <= 0 or len(selected) >= 12:
            break
        selected.update(
            neighbor
            for neighbor in (index - 1, index, index + 1)
            if 0 <= neighbor < len(blocks)
        )
    output: list[str] = []
    used = 0
    previous: int | None = None
    for index in sorted(selected):
        block = blocks[index]
        separator = (
            "\n\n[...省略若干记录...]\n\n"
            if previous is not None and index > previous + 1
            else "\n\n"
        )
        addition = (separator if output else "") + block
        if output and used + len(addition) > max_chars:
            continue
        output.append(addition)
        used += len(addition)
        previous = index
    return "".join(output)[:max_chars]
