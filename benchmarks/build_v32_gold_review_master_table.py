"""Build a local, human-reviewable master table for the 19 narrowed atoms.

The source records are copied only from the existing local narrowed review
packet.  This is presentation work: it never promotes a literal lead to a
semantic gold answer, and it never supplies a candidate to runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "aevnema.v3_2.narrowed_source_review_master.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object, fallback: str = "not_observed") -> str:
    value = str(value or "").strip()
    return value or fallback


def _record_markdown(record: Mapping[str, Any]) -> list[str]:
    if not record:
        return ["`not_observed`"]
    return [
        f"- `ScriptKr`: {_text(record.get('ScriptKr'))}",
        f"- `TextJp`: {_text(record.get('TextJp'))}",
        f"- `TextCn`: {_text(record.get('TextCn'))}",
    ]


def build(*, narrowed_packet: Path, output_dir: Path) -> dict[str, Any]:
    narrowed_packet = narrowed_packet.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be new: {output_dir}")
    root = json.loads(narrowed_packet.read_text(encoding="utf-8"))
    if not isinstance(root, Mapping):
        raise ValueError("narrowed packet is not an object")
    # The E32-07 packet calls this collection ``items``; keep the historical
    # shape rather than silently rebuilding it under a new schema.
    atoms = root.get("items")
    if not isinstance(atoms, list):
        raise ValueError("narrowed packet lacks items")
    normalized: list[dict[str, Any]] = []
    lines = [
        "# v3.2 narrowed Source claim review master",
        "",
        "This local packet converts existing literal Source leads into a reviewer worksheet. Every item remains **unapproved**, **not scored**, and **forbidden at runtime** until a human records a decision. The text below is not a model-generated answer and is not an evidence migration.",
        "",
        f"- Input packet SHA-256: `{_sha256_file(narrowed_packet)}`",
        f"- Candidate atoms: `{len(atoms)}`",
        "- Local work completed: original record, immediate context, locator, source hash, historical summary and proposed literal claim assembled.",
        "- Human work required: accept, split, request more context, or reject each semantic claim; do not sign merely because a locator exists.",
        "",
        "## Review index",
        "",
        "| # | Atom | Source | Record | Recommended local action | Human decision |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    for index, raw_atom in enumerate(atoms, start=1):
        atom = _mapping(raw_atom)
        primary = _mapping(atom.get("primary_proposed_literal_claim"))
        source = _mapping(atom.get("source"))
        locator = _mapping(primary.get("record_locator"))
        normalized.append(
            {
                "ordinal": index,
                "atom_id": _text(atom.get("atom_id")),
                "question_id": _text(atom.get("question_id")),
                "source_key": _text(source.get("source_key")),
                "source_file_sha256": _text(source.get("source_file_sha256")),
                "record_index": primary.get("record_index", "not_observed"),
                "record_pointer": _text(locator.get("pointer")),
                "literal_record_claim": _text(primary.get("literal_record_claim")),
                "historical_episode_id": _mapping(atom.get("legacy_candidate")).get("episode_id", "not_observed"),
                "historical_episode_text": _text(_mapping(atom.get("legacy_candidate")).get("episode_text")),
                "recommended_reviewer_action": _text(atom.get("recommended_reviewer_action")),
                "recommendation_rationale": _text(atom.get("recommendation_rationale")),
                "speaker_binding_status": _text(atom.get("speaker_binding_status")),
                "local_decision": "locator_and_context_assembled_no_semantic_decision",
                "human_decision": "pending_human_review",
                "runtime_eligibility": "forbidden",
                "usable_for_scoring": False,
            }
        )
        lines.append(
            "| {index} | `{atom}` | `{source}` | `{record}` | `{action}` | `pending_human_review` |".format(
                index=index,
                atom=_text(atom.get("atom_id")),
                source=_text(source.get("source_key")),
                record=primary.get("record_index", "not_observed"),
                action=_text(atom.get("recommended_reviewer_action")),
            )
        )
    for index, raw_atom in enumerate(atoms, start=1):
        atom = _mapping(raw_atom)
        source = _mapping(atom.get("source"))
        primary = _mapping(atom.get("primary_proposed_literal_claim"))
        primary_locator = _mapping(primary.get("record_locator"))
        legacy = _mapping(atom.get("legacy_candidate"))
        context = atom.get("adjacent_raw_context")
        context = context if isinstance(context, list) else []
        lines.extend(
            [
                "",
                f"## {index}. `{_text(atom.get('atom_id'))}`",
                "",
                "### Binding and historical lead",
                "",
                f"- Source: `{_text(source.get('source_key'))}`",
                f"- Source file SHA-256: `{_text(source.get('source_file_sha256'))}`",
                f"- Proposed locator: `{_text(primary_locator.get('pointer'))}` (record `{primary.get('record_index', 'not_observed')}`)",
                f"- Historical Episode: `{legacy.get('episode_id', 'not_observed')}`",
                f"- Historical summary (lead only): {_text(legacy.get('episode_text'))}",
                f"- Proposed literal claim (not semantically accepted): {_text(primary.get('literal_record_claim'))}",
                f"- Source-span hash: `{_text(_mapping(primary.get('raw_span')).get('sha256'))}`",
                "",
                "### Proposed original record",
                "",
                *_record_markdown(_mapping(primary.get("original_record"))),
                "",
                "### Immediate raw context",
                "",
            ]
        )
        if not context:
            lines.append("`not_observed`")
        for entry in context:
            item = _mapping(entry)
            lines.extend(
                [
                    f"#### Record `{item.get('record_index', 'not_observed')}` — `{_text(_mapping(item.get('record_locator')).get('pointer'))}`",
                    "",
                    *_record_markdown(item),
                    "",
                ]
            )
        lines.extend(
            [
                "### Boundary and decision",
                "",
                f"- Speaker/context binding: `{_text(atom.get('speaker_binding_status'))}`",
                f"- Local recommendation: `{_text(atom.get('recommended_reviewer_action'))}` — {_text(atom.get('recommendation_rationale'))}",
                f"- Local completion: `locator_and_context_assembled_no_semantic_decision`",
                "- Human decision: `pending_human_review` (accept / split / context request / reject)",
                "- Semantic scoring: `not_accepted_not_scored`",
                "- Runtime: `forbidden`",
                "",
            ]
        )
    payload = {
        "schema": SCHEMA,
        "status": "review_material_assembled_human_decisions_pending",
        "created_at": _utc_now(),
        "paid_provider_calls_issued": 0,
        "input": {"path": str(narrowed_packet), "sha256": _sha256_file(narrowed_packet)},
        "atom_count": len(normalized),
        "local_completion": "original_records_context_and_claim_boundaries_assembled",
        "human_review": "required_for_every_atom",
        "runtime_eligibility": "forbidden",
        "atoms": normalized,
    }
    output_dir.mkdir(parents=True)
    json_path = output_dir / "narrowed_source_claim_review_master.json"
    md_path = output_dir / "NARROWED_SOURCE_CLAIM_REVIEW_MASTER.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "atoms": len(normalized)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--narrowed-packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
