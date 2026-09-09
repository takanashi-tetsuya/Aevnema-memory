"""Build one narrow, human-reviewable Source proposal for each legacy atom.

The input packet is intentionally not gold.  Its legacy candidate episodes
cover broad segments, often with several possible Source statements.  This
tool retains that uncertainty: it selects a small lexical lead set from the
already-authorized raw records, preserves exact locators/hashes/context, and
proposes only literal-record statements.  It cannot accept a semantic answer,
choose a reviewer, or make any material available to runtime/scoring.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping, Sequence


SCHEMA = "aevnema.v3_2.narrowed_source_atom_review.v1"
FULL_LOCAL_NAME = "narrowed_source_atom_review.full_local.json"
REPORT_NAME = "narrowed_source_atom_review.md"
_TOKEN = re.compile(r"[A-Za-z0-9_]{2,}|[\u3400-\u9fff]{2,}")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _tokens(value: str) -> Counter[str]:
    tokens: list[str] = []
    for run in _TOKEN.findall(value.casefold()):
        if re.fullmatch(r"[\u3400-\u9fff]+", run):
            tokens.extend(run[index : index + size] for size in (2, 3) for index in range(len(run) - size + 1))
        else:
            tokens.append(run)
    return Counter(tokens)


def _score(reference: Counter[str], candidate: str) -> tuple[float, list[str]]:
    observed = _tokens(candidate)
    overlap = sorted(set(reference).intersection(observed))
    matched = sum(min(reference[token], observed[token]) for token in overlap)
    scale = max(1, sum(reference.values()))
    return round(matched / scale, 6), overlap[:24]


def _raw_candidate(candidate: Mapping[str, object], score: float, terms: Sequence[str]) -> dict[str, object]:
    original = candidate.get("source_original")
    original = original if isinstance(original, Mapping) else {}
    span = candidate.get("full_record_span")
    span = span if isinstance(span, Mapping) else {}
    cn = str(original.get("TextCn", ""))
    return {
        "record_locator": candidate.get("record_locator"),
        "record_index": candidate.get("record_index"),
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
        "lexical_lead_score": score,
        "lexical_overlap_terms": list(terms),
        "literal_record_claim": (
            f"在该定位处，原始 Source 的 TextCn 原文为：{cn}"
            if cn else "not_observed: no raw TextCn at this candidate"
        ),
    }


def _context(records: Sequence[object], record_index: int, *, radius: int = 1) -> list[dict[str, object]]:
    by_index = {
        int(item.get("record_index")): item
        for item in records
        if isinstance(item, Mapping) and item.get("full_record_span") is not None
    }
    rows: list[dict[str, object]] = []
    for index in range(record_index - radius, record_index + radius + 1):
        item = by_index.get(index)
        if item is None:
            continue
        original = item.get("source_original")
        original = original if isinstance(original, Mapping) else {}
        span = item.get("full_record_span")
        span = span if isinstance(span, Mapping) else {}
        rows.append(
            {
                "record_index": index,
                "record_locator": item.get("record_locator"),
                "raw_span_sha256": span.get("raw_span_sha256"),
                "ScriptKr": original.get("ScriptKr"),
                "TextJp": original.get("TextJp"),
                "TextCn": original.get("TextCn"),
            }
        )
    return rows


def _review_action(text: str) -> tuple[str, str]:
    lower = text.casefold()
    if not text or text.strip("…。！？!? ") == "":
        return "reject_as_nonpropositional_record", "The candidate is non-propositional punctuation or empty text."
    if any(marker in text for marker in ("她", "他", "这", "那", "这里", "那里", "也许", "可能", "？")) or "maybe" in lower:
        return "needs_context_before_semantic_review", "The literal record has a possible referent, speaker, or modality boundary that must be reviewed with adjacent raw records."
    return "split_literal_record_then_human_accept_or_reject", "The literal record is readable, but it is not automatically an answer to the legacy multi-fact slot."


def build_review(*, review_packet: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError("output directory must be new")
    packet = json.loads(review_packet.read_text(encoding="utf-8"))
    if not isinstance(packet, dict) or not isinstance(packet.get("atoms"), list):
        raise ValueError("review packet has no atom list")
    if packet.get("scoring_eligible") is not False or packet.get("promotion_prohibited") is not True:
        raise ValueError("input packet must remain non-scorable and promotion-prohibited")
    source_root = packet.get("source_root")
    source_root = source_root if isinstance(source_root, Mapping) else {}
    root = Path(str(source_root.get("path", "")))
    source_files: dict[str, dict[str, object]] = {}
    items: list[dict[str, object]] = []
    for atom in packet["atoms"]:
        if not isinstance(atom, Mapping):
            raise ValueError("atom must be an object")
        source = atom.get("source")
        source = source if isinstance(source, Mapping) else {}
        source_key = str(source.get("source_key", ""))
        if source_key and source_key not in source_files:
            path = root / source_key
            expected = str(source.get("source_file_sha256", ""))
            observed = _sha(path) if path.is_file() else "not_observed"
            source_files[source_key] = {
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matches": bool(expected) and observed == expected,
            }
        legacy = atom.get("legacy_candidate")
        legacy = legacy if isinstance(legacy, Mapping) else {}
        boundary = atom.get("claim_boundary")
        boundary = boundary if isinstance(boundary, Mapping) else {}
        facts = boundary.get("legacy_required_facts")
        facts = [str(item) for item in facts] if isinstance(facts, list) else []
        reference_text = "\n".join([str(legacy.get("episode_text", "")), *facts])
        reference_tokens = _tokens(reference_text)
        candidates = atom.get("candidate_records")
        candidates = candidates if isinstance(candidates, list) else []
        ranked: list[tuple[float, list[str], Mapping[str, object]]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or not candidate.get("in_legacy_window"):
                continue
            original = candidate.get("source_original")
            original = original if isinstance(original, Mapping) else {}
            text = str(original.get("TextCn", ""))
            if not text.strip():
                continue
            score, terms = _score(reference_tokens, text)
            ranked.append((score, terms, candidate))
        ranked.sort(key=lambda item: (item[0], len(item[1]), -int(item[2].get("record_index", 0))), reverse=True)
        leads = [_raw_candidate(candidate, score, terms) for score, terms, candidate in ranked[:3]]
        primary = leads[0] if leads else None
        primary_cn = str(primary.get("original_record", {}).get("TextCn", "")) if isinstance(primary, Mapping) and isinstance(primary.get("original_record"), Mapping) else ""
        action, rationale = _review_action(primary_cn)
        primary_index = int(primary.get("record_index", -10_000)) if isinstance(primary, Mapping) else -10_000
        item = {
            "atom_id": atom.get("atom_id"),
            "question_id": boundary.get("question_id"),
            "legacy_candidate": {
                "episode_id": legacy.get("episode_id"),
                "episode_text": legacy.get("episode_text"),
                "source_segment_index": legacy.get("source_segment_index"),
                "source_segment_sha256": legacy.get("source_segment_sha256"),
            },
            "legacy_required_facts": facts,
            "source": dict(source),
            "candidate_count": len(candidates),
            "lexical_lead_selection": "local lexical triage only; not semantic entailment or runtime retrieval",
            "primary_proposed_literal_claim": primary,
            "alternative_literal_leads": leads[1:],
            "adjacent_raw_context": _context(candidates, primary_index) if primary is not None else [],
            "speaker_binding_status": "requires_human_review_of_ScriptKr_and_neighbouring_records",
            "semantic_answer_status": "not_accepted_not_scored",
            "recommended_reviewer_action": action,
            "recommendation_rationale": rationale,
            "dispute_notes": atom.get("dispute_notes", []),
            "human_decision": "pending_human_review",
            "runtime_eligibility": "forbidden",
            "promotion_prohibited": True,
            "usable_for_scoring": False,
        }
        items.append(item)
    output_dir.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "pending_human_review",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "input_packet": {"path": str(review_packet.resolve()), "sha256": _sha(review_packet), "atom_count": len(packet["atoms"])},
        "source_root": dict(source_root),
        "source_file_revalidation": source_files,
        "atom_count": len(items),
        "items": items,
        "formal_scoring": {"gold_loaded": False, "status": "not_scored"},
        "runtime_eligibility": "forbidden",
        "promotion_prohibited": True,
        "decision_boundary": "Every proposed claim is a literal raw-record statement. A human must accept, split, request context, or reject it before any semantic gold exists.",
    }
    (output_dir / FULL_LOCAL_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(str(item["recommended_reviewer_action"]) for item in items)
    lines = [
        "# Narrowed v3.2 Source atom review",
        "",
        "This is a 19-item human-review packet. It proposes only literal Source-record claims, not answer gold or an approval decision.",
        "",
        f"- Atoms: `{len(items)}`",
        f"- Source files revalidated: `{sum(row['matches'] is True for row in source_files.values())}/{len(source_files)}`",
        f"- Runtime/scoring eligibility: `forbidden` / `not_scored`",
        "",
        "| Recommended action | Atoms |",
        "| --- | ---: |",
    ]
    for action, count in sorted(counts.items()):
        lines.append(f"| `{action}` | `{count}` |")
    lines.extend(["", "The full-local JSON contains each selected original-language record, locator, span hash, and neighbouring context. No human decision is supplied or inferred.", ""])
    (output_dir / REPORT_NAME).write_text("\n".join(lines), encoding="utf-8")
    return {"output_dir": str(output_dir), "full_local": str(output_dir / FULL_LOCAL_NAME), "report": str(output_dir / REPORT_NAME), "atom_count": len(items)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_review(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
