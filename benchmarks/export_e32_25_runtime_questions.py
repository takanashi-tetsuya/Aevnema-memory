"""Create the answer-blind E32-25 runtime question manifest.

This is a preparation-only bridge.  It deliberately reads the evaluator file
offline and copies only frozen question wording plus run labels.  Runtime code
must consume the exported manifest, never the evaluator-side Source locations,
support records, or provisional answer criteria.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNTIME_SCHEMA = "aevnema.e32_25.runtime_questions.v1"
_VARIANTS = (
    "same_text",
    "paraphrase",
    "partial_clue_with_neutral_context",
    "near_neighbor_counterexample",
)
_VARIANT_LABELS = {
    "same_text": ("same_need_exact_text", "edge_and_masked_comparison"),
    "paraphrase": ("same_need_reworded", "edge_and_masked_comparison"),
    "partial_clue_with_neutral_context": (
        "related_or_partial_clue_variant",
        "edge_and_masked_comparison",
    ),
    "near_neighbor_counterexample": (
        "near_neighbor_distinct_need",
        "counterexample_delivery_check",
    ),
}
_RUNTIME_CASE_KEYS = {"case_id", "q1", "q2"}
_RUNTIME_Q2_KEYS = {
    "variant",
    "question",
    "visible_context",
    "scope_label",
    "run_condition_label",
}
_FORBIDDEN_RUNTIME_FIELD_PARTS = (
    "source",
    "support",
    "answer",
    "expected",
    "episode",
    "need",
    "claim",
    "record",
)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _runtime_q2(variant: str, question: str) -> dict[str, object]:
    scope_label, condition_label = _VARIANT_LABELS[variant]
    return {
        "variant": variant,
        # Keep the original wording whole.  Splitting its wording into a
        # synthetic prompt would quietly alter the frozen experiment question.
        "question": question,
        "visible_context": (
            {"placement": "embedded_in_question", "text": question}
            if variant == "partial_clue_with_neutral_context"
            else {"placement": "none", "text": ""}
        ),
        "scope_label": scope_label,
        "run_condition_label": condition_label,
    }


def export_runtime_questions(evaluator_payload: dict[str, Any]) -> dict[str, object]:
    """Whitelist frozen wording without looking at evaluator answer fields."""

    raw_cases = evaluator_payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluator cases must be a non-empty list")
    cases: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for position, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {position} must be an object")
        case_id = _required_string(raw_case.get("case_id"), f"case {position} id")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_case_ids.add(case_id)
        q1 = _required_string(raw_case.get("q1"), f"case {case_id} q1")
        raw_q2 = raw_case.get("q2")
        if not isinstance(raw_q2, dict):
            raise ValueError(f"case {case_id} q2 must be an object")
        if set(raw_q2) != set(_VARIANTS):
            raise ValueError(f"case {case_id} q2 variants do not match the frozen set")
        q2 = [
            _runtime_q2(
                variant,
                _required_string(raw_q2[variant], f"case {case_id} {variant}"),
            )
            for variant in _VARIANTS
        ]
        # This explicit construction is the whitelist: evaluator-only keys are
        # intentionally neither inspected nor carried into the return value.
        cases.append({"case_id": case_id, "q1": q1, "q2": q2})
    payload: dict[str, object] = {
        "schema": RUNTIME_SCHEMA,
        "status": "offline_prepared_answer_blind",
        "runtime_exclusion": (
            "No source key, support record, answer, expected episode, claim, or need is "
            "present. This manifest is question input only."
        ),
        "cases": cases,
    }
    validate_runtime_questions(payload)
    return payload


def _validate_no_forbidden_key(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).casefold()
            if any(part in lowered for part in _FORBIDDEN_RUNTIME_FIELD_PARTS):
                raise ValueError(f"runtime manifest contains forbidden field: {key}")
            _validate_no_forbidden_key(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_no_forbidden_key(nested)


def validate_runtime_questions(payload: object) -> None:
    """Reject extra runtime fields before a runner is allowed to consume it."""

    if not isinstance(payload, dict):
        raise ValueError("runtime manifest must be an object")
    if set(payload) != {"schema", "status", "runtime_exclusion", "cases"}:
        raise ValueError("runtime manifest has unexpected top-level fields")
    if payload.get("schema") != RUNTIME_SCHEMA:
        raise ValueError("runtime manifest schema mismatch")
    if not isinstance(payload.get("cases"), list):
        raise ValueError("runtime manifest cases must be a list")
    _validate_no_forbidden_key(payload)
    for position, case in enumerate(payload["cases"], start=1):
        if not isinstance(case, dict) or set(case) != _RUNTIME_CASE_KEYS:
            raise ValueError(f"runtime case {position} has unexpected fields")
        _required_string(case.get("case_id"), f"runtime case {position} id")
        _required_string(case.get("q1"), f"runtime case {position} q1")
        q2 = case.get("q2")
        if not isinstance(q2, list) or len(q2) != len(_VARIANTS):
            raise ValueError(f"runtime case {position} q2 count is invalid")
        seen_variants: set[str] = set()
        for entry in q2:
            if not isinstance(entry, dict) or set(entry) != _RUNTIME_Q2_KEYS:
                raise ValueError(f"runtime case {position} q2 has unexpected fields")
            variant = _required_string(entry.get("variant"), "runtime q2 variant")
            if variant not in _VARIANTS or variant in seen_variants:
                raise ValueError("runtime q2 variants are invalid")
            seen_variants.add(variant)
            _required_string(entry.get("question"), "runtime q2 question")
            visible_context = entry.get("visible_context")
            if not isinstance(visible_context, dict) or set(visible_context) != {
                "placement",
                "text",
            }:
                raise ValueError("runtime visible_context shape is invalid")
            _required_string(visible_context.get("placement"), "visible context placement")
            if not isinstance(visible_context.get("text"), str):
                raise ValueError("visible context text must be a string")
            _required_string(entry.get("scope_label"), "runtime scope label")
            _required_string(entry.get("run_condition_label"), "runtime condition label")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.evaluator_cases.read_text(encoding="utf-8"))
    payload = export_runtime_questions(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
