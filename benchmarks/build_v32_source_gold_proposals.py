"""Prepare reviewable, non-approved direct-Source claim proposals for v3.2.

The prior Source packet deliberately preserved legacy retrieval slots without
choosing a semantic gold answer.  This tool does not change that packet.  It
turns every supplied raw-record candidate into a literal, reviewable Source
statement and recommends the narrow/split review action required before a
human can accept or reject it.  It never loads the proposal at runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping


SCHEMA = "aevnema.v3_2.source_gold_claim_proposals.v1"
FULL_LOCAL_NAME = "source_gold_claim_proposals.full_local.json"
REPORT_NAME = "source_gold_claim_proposals.md"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _record_proposal(
    *, atom: Mapping[str, object], candidate: Mapping[str, object], ordinal: int
) -> dict[str, object]:
    source = atom.get("source")
    source = source if isinstance(source, Mapping) else {}
    source_key = str(source.get("source_key", "not_observed"))
    original = candidate.get("source_original")
    original = original if isinstance(original, Mapping) else {}
    span = candidate.get("full_record_span")
    span = span if isinstance(span, Mapping) else {}
    text_cn = str(original.get("TextCn", "")).strip()
    literal_text = text_cn or str(span.get("text", "")).strip()
    locator = candidate.get("record_locator")
    locator = locator if isinstance(locator, Mapping) else {}
    claim_group = atom.get("claim_boundary")
    claim_group = claim_group if isinstance(claim_group, Mapping) else {}
    members = claim_group.get("claim_group_coverage_clause")
    members = members if isinstance(members, Mapping) else {}
    # Legacy any-of records cannot become a one-step semantic answer merely
    # because one raw record is readable.  A literal record statement is the
    # most narrow claim that can be proposed without inventing entailment.
    recommended_action = (
        "split_required_before_semantic_gold_review"
        if len(members.get("members", [])) > 1
        else "narrow_then_human_accept_or_reject"
    )
    return {
        "proposal_id": f"{atom.get('atom_id', 'unknown')}/record-{ordinal}",
        "atom_id": atom.get("atom_id"),
        "source_key": source_key,
        "record_locator": dict(locator),
        "raw_span": {
            "span_start": span.get("span_start"),
            "span_end": span.get("span_end"),
            "sha256": span.get("raw_span_sha256"),
        },
        "original_record": {
            "ScriptKr": original.get("ScriptKr"),
            "TextJp": original.get("TextJp"),
            "TextCn": original.get("TextCn"),
        },
        "proposed_direct_source_claim": (
            f"原始 Source 的 {source_key} 中该定位记录包含的 TextCn 原文为：{literal_text}"
            if literal_text
            else "not_observed: candidate record lacks a TextCn or full-record text value"
        ),
        "semantic_answer_status": "not_proposed_from_legacy_candidate",
        "recommended_reviewer_action": recommended_action,
        "human_decision": "pending_human_review",
        "legacy_review_status": atom.get("review_status"),
        "legacy_reviewer_decision": atom.get("reviewer_decision"),
        "usable_for_scoring": False,
        "promotion_prohibited": True,
    }


def build_proposals(*, review_packet: Path, output_dir: Path) -> dict[str, object]:
    review_packet = review_packet.resolve()
    output_dir = output_dir.resolve()
    if not review_packet.is_file():
        raise FileNotFoundError(review_packet)
    if output_dir.exists():
        raise FileExistsError("output directory must be new")
    packet = json.loads(review_packet.read_text(encoding="utf-8"))
    if not isinstance(packet, dict) or not isinstance(packet.get("atoms"), list):
        raise ValueError("review packet has no atom list")
    if packet.get("scoring_eligible") is not False:
        raise ValueError("input packet must remain non-scorable")
    if packet.get("promotion_prohibited") is not True:
        raise ValueError("input packet must retain promotion prohibition")
    source_root = packet.get("source_root")
    source_root = source_root if isinstance(source_root, Mapping) else {}
    root = Path(str(source_root.get("path", "")))
    source_hashes: dict[str, dict[str, object]] = {}
    proposals: list[dict[str, object]] = []
    for atom in packet["atoms"]:
        if not isinstance(atom, Mapping):
            raise ValueError("atom must be an object")
        source = atom.get("source")
        source = source if isinstance(source, Mapping) else {}
        source_key = str(source.get("source_key", ""))
        if source_key and source_key not in source_hashes:
            path = root / source_key
            expected = str(source.get("source_file_sha256", ""))
            observed = _sha256(path) if path.is_file() else "not_observed"
            source_hashes[source_key] = {
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": bool(expected) and expected == observed,
            }
        candidates = atom.get("candidate_records")
        if not isinstance(candidates, list) or not candidates:
            proposals.append(
                {
                    "proposal_id": f"{atom.get('atom_id', 'unknown')}/no-record",
                    "atom_id": atom.get("atom_id"),
                    "proposed_direct_source_claim": "not_observed: no raw candidate record",
                    "semantic_answer_status": "not_proposed",
                    "recommended_reviewer_action": "reject_or_supply_new_raw_candidate",
                    "human_decision": "pending_human_review",
                    "usable_for_scoring": False,
                    "promotion_prohibited": True,
                }
            )
            continue
        for ordinal, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate record must be an object")
            proposals.append(_record_proposal(atom=atom, candidate=candidate, ordinal=ordinal))
    output_dir.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "pending_human_review",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "input_packet": {
            "path": str(review_packet),
            "sha256": _sha256(review_packet),
            "atom_count": len(packet["atoms"]),
        },
        "source_root": dict(source_root),
        "source_file_revalidation": source_hashes,
        "proposal_count": len(proposals),
        "proposals": proposals,
        "formal_scoring": {"gold_loaded": False, "status": "not_scored"},
        "promotion_prohibited": True,
        "runtime_eligibility": "forbidden",
        "decision_boundary": (
            "These are literal raw-Source claim proposals only. A human must explicitly "
            "accept, split, or reject each proposal before any semantic gold claim exists."
        ),
    }
    _write_json(output_dir / FULL_LOCAL_NAME, payload)
    rows = [
        "# v3.2 Source-gold claim proposals",
        "",
        "This packet proposes literal raw-Source statements for review; it does not accept a claim, set gold, or permit scoring/promotion.",
        "",
        f"- Input atoms: `{len(packet['atoms'])}`",
        f"- Record proposals: `{len(proposals)}`",
        f"- Source files revalidated: `{sum(item['matches'] is True for item in source_hashes.values())}/{len(source_hashes)}`",
        "",
        "| Proposed action | Count |",
        "| --- | ---: |",
    ]
    for action in sorted({str(item["recommended_reviewer_action"]) for item in proposals}):
        rows.append(
            f"| `{action}` | `{sum(item['recommended_reviewer_action'] == action for item in proposals)}` |"
        )
    rows.extend(
        [
            "",
            "Every row remains `pending_human_review`; the full-local JSON contains the exact original-language fields, locator, and raw-span hash for audit.",
            "",
        ]
    )
    (output_dir / REPORT_NAME).write_text("\n".join(rows), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "full_local": str(output_dir / FULL_LOCAL_NAME),
        "report": str(output_dir / REPORT_NAME),
        "proposal_count": len(proposals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_proposals(review_packet=args.review_packet, output_dir=args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
