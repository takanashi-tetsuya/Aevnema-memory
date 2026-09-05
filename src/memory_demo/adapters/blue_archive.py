from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from memory_demo.adapters.base import (
    InputAdapter,
    normalize_memory_evidence,
    read_text_document,
    split_text_chunks,
)
from memory_demo.types import NormalizedBlock


LANGUAGE_FIELDS = {
    "zh-CN": "TextCn",
    "ja": "TextJp",
    "en": "TextEn",
    "zh-TW": "TextTw",
    "th": "TextTh",
}

_PROTECTED_USERNAME = "\u0000USERNAME\u0000"
_TAG_RE = re.compile(r"\[(?:/?ruby(?:=[^\]]+)?)\]", re.IGNORECASE)
_DISPLAY_TAG_RE = re.compile(r"\[(?:[0-9A-Fa-f]{6}|-)\]")
_GENERIC_CONTAINER_FIELDS = (
    "items",
    "records",
    "documents",
    "entries",
    "data",
    "content",
)
_GENERIC_TEXT_FIELDS = (
    "text",
    "raw_text",
    "body",
    "summary",
    "description",
    "content",
)
_GENERIC_TITLE_FIELDS = ("title", "name", "heading")
_NEXT_EPISODE_CARD = re.compile(
    r"^\s*#nextepisode(?:;[^\r\n]*)?\s*$", re.IGNORECASE
)
_LOCALIZED_CONTROL_CARD = re.compile(
    r"^\s*#(?:st(?:;[^\r\n]*)?|clearst)\s*$", re.IGNORECASE
)


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not supported: {value}")


def clean_game_text(value: Any) -> str:
    text = str(value or "").replace("[USERNAME]", _PROTECTED_USERNAME)
    text = _TAG_RE.sub("", text)
    text = _DISPLAY_TAG_RE.sub("", text)
    text = text.replace(_PROTECTED_USERNAME, "[USERNAME]")
    return " ".join(text.replace("\r", "\n").split())


def parse_speaker(script: str) -> str:
    """Return the speaker attached to dialogue, not the first visible actor.

    Some records first update multiple portrait slots and only then contain a
    fourth-field dialogue command, for example ``1;A;01`` followed by
    ``3;Mika;02;text``.  The localized text belongs to Mika.  The first actor is
    retained only as a fallback for unusual records with no Korean dialogue.
    """
    fallback = ""
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if line.startswith("#na;"):
            parts = line.split(";", 2)
            if len(parts) >= 3 and clean_game_text(parts[2]):
                return clean_game_text(parts[1])
        parts = line.split(";", 3)
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].strip():
            fallback = fallback or clean_game_text(parts[1])
            if len(parts) >= 4 and clean_game_text(parts[3]):
                return clean_game_text(parts[1])
    return fallback


def parse_korean_text(script: str) -> str:
    """Extract Korean text from the actual dialogue command in ScriptKr."""
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if line.startswith("#na;"):
            parts = line.split(";", 2)
            if len(parts) >= 3 and clean_game_text(parts[2]):
                return clean_game_text(parts[2])
        parts = line.split(";", 3)
        if len(parts) >= 4 and parts[0].isdigit() and clean_game_text(parts[3]):
            return clean_game_text(parts[3])
        if line.startswith("#title;") or line.startswith("#place;"):
            return clean_game_text(line.split(";", 1)[1])
    return ""


def _unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


@lru_cache(maxsize=4)
def load_student_speaker_aliases(path: Path) -> dict[str, dict[str, str]]:
    """Read only the multilingual family/name fields from students.yaml.

    The story format exposes Korean speaker markers while the dialogue itself is
    localized.  The bundled catalog is input metadata, so the adapter can expose
    its aliases without adding a YAML dependency or changing persistent schema.
    """
    if not path.is_file():
        return {}
    entries: list[dict[str, dict[str, str]]] = []
    current: dict[str, dict[str, str]] | None = None
    section = ""
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        if raw_line.startswith("- id:"):
            if current is not None:
                entries.append(current)
            current = {"familyName": {}, "name": {}}
            section = ""
            continue
        if current is None:
            continue
        if raw_line.startswith("  familyName:"):
            section = "familyName"
            continue
        if raw_line.startswith("  name:"):
            section = "name"
            continue
        if raw_line.startswith("  ") and not raw_line.startswith("    "):
            section = ""
            continue
        if section and raw_line.startswith("    ") and ":" in raw_line:
            language, value = raw_line.strip().split(":", 1)
            value = _unquote_yaml_scalar(value)
            if language in {"cn", "jp", "en", "kr", "tw", "th"} and value:
                current[section][language] = value
    if current is not None:
        entries.append(current)

    aliases: dict[str, dict[str, str]] = {}
    language_names = {
        "cn": "zh-CN",
        "jp": "ja",
        "en": "en",
        "kr": "ko",
        "tw": "zh-TW",
        "th": "th",
    }
    for entry in entries:
        names = entry["name"]
        family_names = entry["familyName"]
        korean_name = names.get("kr", "")
        if korean_name:
            aliases[korean_name] = {
                language_names[key]: value for key, value in names.items() if value
            }
        korean_full = (family_names.get("kr", "") + korean_name).strip()
        if korean_full:
            aliases[korean_full] = {
                language_names[key]: (family_names.get(key, "") + value).strip()
                for key, value in names.items()
                if value
            }
    return aliases


def find_student_speaker_aliases(story_path: Path) -> dict[str, dict[str, str]]:
    for parent in story_path.parents:
        candidate = parent / "config" / "yaml" / "students.yaml"
        if candidate.is_file():
            return load_student_speaker_aliases(candidate)
    return {}


_SPEAKER_ROLE_PREFIXES = ("통신복면", "복면", "통신")
_SPEAKER_ROLE_SUFFIXES = (" 마스크", "마스크")


def resolve_speaker_aliases(
    speaker_raw: str, aliases: dict[str, dict[str, str]]
) -> dict[str, str]:
    """Resolve decorated story markers to the underlying person identity.

    Markers such as ``통신아야네`` (communications Ayane) and
    ``복면호시노`` (masked Hoshino) are presentation roles, not different
    people.  The raw marker is still persisted verbatim; this lookup only adds
    the base person's multilingual aliases for Concept identity resolution.
    """
    exact = aliases.get(speaker_raw)
    if exact:
        return exact
    candidates: list[str] = []
    for prefix in _SPEAKER_ROLE_PREFIXES:
        if speaker_raw.startswith(prefix):
            candidates.append(speaker_raw[len(prefix) :].strip())
    for suffix in _SPEAKER_ROLE_SUFFIXES:
        if speaker_raw.endswith(suffix):
            candidates.append(speaker_raw[: -len(suffix)].strip())
    for candidate in candidates:
        resolved = aliases.get(candidate)
        if resolved:
            return resolved
    return {}


def boundary_score(record: dict[str, Any]) -> int:
    script = str(record.get("ScriptKr", ""))
    if record.get("Transition") or "#all;hide" in script:
        return 3
    if record.get("BGName") or "#showmenu" in script or "#hidemenu" in script:
        return 2
    if script.startswith("#wait;"):
        try:
            milliseconds = int(script.split(";", 1)[1])
            return 2 if milliseconds >= 2_000 else 1
        except (ValueError, IndexError):
            return 1
    return 0


def is_non_story_preview_card(script: str) -> bool:
    """Return whether a localized record is an end-of-file episode preview.

    ``#nextepisode`` cards carry translated display text, but that text is a
    navigation/preview label rather than an in-story fact.  Keeping it as a
    normal block can make a final one-record Source which has no extractable
    Episode and incorrectly turns an otherwise successful story into a partial
    import.
    """

    return bool(_NEXT_EPISODE_CARD.fullmatch(script))


def is_non_story_control_card(script: str, languages: dict[str, str]) -> bool:
    """Return whether a record is only a rendered engine-control command.

    A handful of exports localize their UI state commands (``#st`` and
    ``#clearST``) into the display-text fields.  The script itself contains no
    dialogue/narration, and every available translation remains a command.
    This deliberately requires both facts: many ordinary Blue Archive
    ``#``-prefixed scripts do carry story text and must remain importable.
    """

    return bool(
        languages
        and not parse_korean_text(script)
        and all(_LOCALIZED_CONTROL_CARD.fullmatch(text) for text in languages.values())
    )


def _looks_like_blue_archive_content(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(item, dict)
        and any(
            key in item
            for key in (
                "ScriptKr",
                "TextCn",
                "TextJp",
                "TextEn",
                "TextTw",
                "TextTh",
            )
        )
        for item in content
    )


def generic_json_blocks(payload: Any, path: Path) -> list[NormalizedBlock]:
    """Normalize common JSON knowledge-document shapes without inventing text."""

    document_evidence = (
        normalize_memory_evidence(
            payload.get("_memory", payload.get("MemoryEvidence"))
        )
        if isinstance(payload, dict)
        else {}
    )
    blocks: list[NormalizedBlock] = []

    def append_text(
        text: str, metadata: dict[str, Any], section_path: tuple[str, ...]
    ) -> None:
        normalized = "\n".join(
            line.rstrip()
            for line in str(text).replace("\r", "\n").splitlines()
        ).strip()
        if not normalized:
            return
        if section_path:
            normalized = f"section: {' / '.join(section_path)}\n{normalized}"
        for chunk in split_text_chunks(normalized):
            blocks.append(
                NormalizedBlock(
                    record_index=len(blocks),
                    languages={"unknown": chunk},
                    boundary_score=2 if blocks else 0,
                    metadata=dict(metadata),
                )
            )

    def walk(
        value: Any,
        inherited_evidence: dict[str, Any],
        section_path: tuple[str, ...] = (),
    ) -> None:
        if isinstance(value, str):
            append_text(value, inherited_evidence, section_path)
            return
        if isinstance(value, list):
            for item in value:
                walk(item, inherited_evidence, section_path)
            return
        if not isinstance(value, dict):
            return

        evidence = {
            **inherited_evidence,
            **normalize_memory_evidence(
                value.get("_memory", value.get("MemoryEvidence"))
            ),
        }
        title_parts = [
            f"{key}: {value[key]}"
            for key in _GENERIC_TITLE_FIELDS
            if isinstance(value.get(key), str) and value[key].strip()
        ]
        body_parts = [
            str(value[key]).strip()
            for key in _GENERIC_TEXT_FIELDS
            if isinstance(value.get(key), str) and value[key].strip()
        ]
        consumed = {
            "_memory",
            "MemoryEvidence",
            *_GENERIC_TITLE_FIELDS,
            *_GENERIC_TEXT_FIELDS,
        }
        scalar_parts = [
            f"{key}: {item}"
            for key, item in value.items()
            if key not in consumed
            and isinstance(item, (str, int, float, bool))
            and (not isinstance(item, str) or item.strip())
        ]
        if body_parts:
            append_text(
                "\n".join([*title_parts, *body_parts, *scalar_parts]),
                evidence,
                section_path,
            )
        elif title_parts or scalar_parts:
            append_text(
                "\n".join([*title_parts, *scalar_parts]),
                evidence,
                section_path,
            )

        # Known containers cover common export formats. Walking every other
        # nested object/list as well keeps custom JSON schemas importable instead
        # of silently discarding facts merely because their field is named, for
        # example, ``characters`` or ``chapters``.
        ordered_nested_keys = [
            key
            for key, item in value.items()
            if key not in {"_memory", "MemoryEvidence"}
            and isinstance(item, (list, dict))
        ]
        for key in ordered_nested_keys:
            child_path = (
                section_path
                if key in _GENERIC_CONTAINER_FIELDS
                else (*section_path, key)
            )
            walk(value[key], evidence, child_path)

    walk(payload, document_evidence)
    if not blocks:
        raise ValueError(f"{path} does not contain usable textual JSON content")
    return blocks


class BlueArchiveJsonAdapter(InputAdapter):
    def supports(self, path: Path) -> bool:
        return path.suffix.casefold() == ".json"

    def read_blocks(self, path: Path) -> list[NormalizedBlock]:
        payload = json.loads(
            read_text_document(path),
            object_pairs_hook=_json_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
        content = payload.get("content") if isinstance(payload, dict) else None
        if not _looks_like_blue_archive_content(content):
            return generic_json_blocks(payload, path)
        blocks: list[NormalizedBlock] = []
        speaker_aliases = find_student_speaker_aliases(path)
        document_evidence = normalize_memory_evidence(
            payload.get("_memory", payload.get("MemoryEvidence"))
        )
        pending_boundary = 0
        for index, raw_record in enumerate(content):
            if not isinstance(raw_record, dict):
                continue
            pending_boundary = max(pending_boundary, boundary_score(raw_record))
            script = str(raw_record.get("ScriptKr", ""))
            if is_non_story_preview_card(script):
                # Preserve a preceding scene boundary for the next real record,
                # but never treat a localized "next episode" label as story
                # evidence.
                pending_boundary = max(pending_boundary, 3)
                continue
            languages = {
                language: clean_game_text(raw_record.get(field, ""))
                for language, field in LANGUAGE_FIELDS.items()
                if clean_game_text(raw_record.get(field, ""))
            }
            if not languages:
                continue
            if is_non_story_control_card(script, languages):
                # Keep a boundary already supplied by a preceding visual
                # transition, but do not create a retrieval Source from a
                # control-only localized UI command.
                continue
            speaker_raw = parse_speaker(script)
            record_evidence = {
                **document_evidence,
                **normalize_memory_evidence(
                    raw_record.get("_memory", raw_record.get("MemoryEvidence"))
                ),
            }
            blocks.append(
                NormalizedBlock(
                    record_index=index,
                    speaker_raw=speaker_raw,
                    languages=languages,
                    boundary_score=pending_boundary,
                    metadata={
                        "group_id": raw_record.get("GroupId", payload.get("GroupId")),
                        "script_raw": script,
                        "voice_jp": raw_record.get("VoiceJp", ""),
                        "speaker_aliases": resolve_speaker_aliases(
                            speaker_raw, speaker_aliases
                        ),
                        **record_evidence,
                    },
                )
            )
            pending_boundary = 0
        return blocks
