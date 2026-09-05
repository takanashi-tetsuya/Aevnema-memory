from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for value in (SCRIPT_DIR, PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_cross_domain_answer_contract_ab as ab
from config.prompt_config.answer_persistence_repair_prompts import (
    ANSWER_PERSISTENCE_REPAIR_VERSION,
    PERSISTENCE_REPAIR_SYSTEM,
    REPAIR_QUALITY_SYSTEM,
    persistence_repair_prompt,
    repair_quality_prompt,
)
from run_answer_persistence_repair_eval import (
    _call_batch,
    _latency,
    _persona,
    _quality_packet,
    _repair_packet,
    _write,
)
from run_atomic_answer_scope_eval import run as run_atomic_scope


def _atomic_audit(row: dict[str, Any]) -> dict[str, Any]:
    claims = {
        claim["id"]: claim
        for claim in (row.get("extraction", {}).get("result") or {}).get("claims")
        or []
    }
    additions: list[dict[str, str]] = []
    for decision in (row.get("scope", {}).get("result") or {}).get("decisions") or []:
        if decision.get("scope") != "unsupported":
            continue
        claim = claims.get(str(decision.get("claim_id") or "")) or {}
        additions.append(
            {
                "text": str(claim.get("text") or ""),
                "type": str(claim.get("kind") or "state"),
                "reason": "not licensed by the Answer Contract",
            }
        )
    return {
        "verdict": "unsafe" if additions else "safe",
        "persistent_additions": additions,
        "ephemeral_fragments": [],
        "reason": "atomic claim-to-contract mapping",
    }


def _is_complete_scope(row: dict[str, Any]) -> bool:
    scope = row.get("scope") or {}
    return bool(not scope.get("error") and (scope.get("result") or {}).get("overall_verdict"))


def _scope_is_safe(row: dict[str, Any]) -> bool:
    return bool(
        _is_complete_scope(row)
        and (row.get("scope", {}).get("result") or {}).get("overall_verdict")
        == "safe"
    )


def _repaired_source_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a self-contained post-repair audit manifest.

    The post-repair target is always a persistence-safe answer.  Preserve each
    case's deterministic expectations, but replace the original human verdict
    (which describes the pre-repair answer) with the target verdict for the
    repaired answer.
    """
    return {
        "cases": [
            {
                "id": row["id"],
                "name": row["name"],
                "question": row["question"],
                "answer": row["final_answer"],
                "contract": row["contract"],
                "local_guard": row["local_guard"],
                "expected": deepcopy(row.get("expected") or {}),
                "human_persistence_verdict": "safe",
            }
            for row in rows
        ]
    }


def _deterministic_quality(
    expected: dict[str, Any], answer: str, contract: dict[str, Any]
) -> dict[str, Any]:
    """Run legacy text checks only when a case declares their full schema."""
    if not expected:
        return {
            "passed": True,
            "skipped": True,
            "reason": "case declares no deterministic text constraints",
        }
    return ab._evaluate_case(expected, answer, contract)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    chatbot_root = args.chatbot_root.resolve()
    if str(chatbot_root) not in sys.path:
        sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv

    load_dotenv(chatbot_root / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    source = json.loads(args.atomic_source.resolve().read_text(encoding="utf-8"))
    rows = deepcopy(source.get("cases") or [])
    if not rows:
        raise ValueError("atomic source contains no cases")
    selected_ids = {value.strip() for value in args.ids.split(",") if value.strip()}
    if selected_ids:
        known_ids = {str(row.get("id") or "") for row in rows}
        unknown_ids = selected_ids - known_ids
        if unknown_ids:
            raise ValueError(f"unknown case ids: {sorted(unknown_ids)}")
        rows = [row for row in rows if str(row.get("id") or "") in selected_ids]
    persona = _persona(chatbot_root)
    output = args.output.resolve()
    payload: dict[str, Any] = {
        "status": "prepared",
        "version": ANSWER_PERSISTENCE_REPAIR_VERSION + "+atomic-repair-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.atomic_source.resolve()),
        "cases": rows,
    }
    for row in rows:
        row["atomic_audit"] = _atomic_audit(row)
        row["repair_eligible"] = bool(
            _is_complete_scope(row)
            and row["atomic_audit"]["verdict"] == "unsafe"
            and str((row.get("local_guard") or {}).get("verdict") or "safe")
            == "safe"
        )
    _write(output, payload)

    repair_inputs = [row for row in rows if row["repair_eligible"]]
    repair_batch = await _call_batch(
        label="atomic_repair",
        model=args.repair_model,
        items=repair_inputs,
        chatbot_root=chatbot_root,
        concurrency=args.concurrency,
        request_timeout=args.repair_request_timeout,
        total_deadline=args.repair_deadline,
        max_tokens=args.repair_max_tokens,
        system_prompt=PERSISTENCE_REPAIR_SYSTEM,
        prompt_builder=lambda row: persistence_repair_prompt(
            question=row["question"],
            persona=persona,
            contract=row["contract"],
            answer=row["original_answer"],
            audit=row["atomic_audit"],
        ),
        parser=_repair_packet,
    )
    repairs_by_id = {
        row["id"]: result
        for row, result in zip(repair_inputs, repair_batch["results"], strict=True)
    }
    for row in rows:
        row["repair"] = repairs_by_id.get(row["id"], {})
        repaired = str((row["repair"].get("result") or {}).get("answer") or "")
        row["repair_succeeded"] = bool(repaired)
        row["final_answer"] = repaired or row["original_answer"]

    quality_inputs = [row for row in rows if row["repair_succeeded"]]
    quality_batch = await _call_batch(
        label="atomic_repair_quality",
        model=args.quality_model,
        items=quality_inputs,
        chatbot_root=chatbot_root,
        concurrency=args.concurrency,
        request_timeout=args.quality_request_timeout,
        total_deadline=args.quality_deadline,
        max_tokens=args.quality_max_tokens,
        system_prompt=REPAIR_QUALITY_SYSTEM,
        prompt_builder=lambda row: repair_quality_prompt(
            question=row["question"],
            persona=persona,
            contract=row["contract"],
            original=row["original_answer"],
            repaired=row["final_answer"],
        ),
        parser=_quality_packet,
    )
    quality_by_id = {
        row["id"]: result
        for row, result in zip(quality_inputs, quality_batch["results"], strict=True)
    }
    for row in rows:
        row["quality_judge"] = quality_by_id.get(row["id"], {})
        contract = deepcopy(row["contract"])
        contract["answer"] = row["final_answer"]
        row["deterministic_quality"] = _deterministic_quality(
            row["expected"], row["final_answer"], contract
        )

    repaired_source = args.repaired_source.resolve()
    repaired_source.parent.mkdir(parents=True, exist_ok=True)
    repaired_source.write_text(
        json.dumps(
            _repaired_source_payload(rows),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    post_scope_args = argparse.Namespace(
        chatbot_root=chatbot_root,
        source=repaired_source,
        output=args.post_scope_output.resolve(),
        ids=",".join(row["id"] for row in rows),
        model=args.extraction_model,
        scope_model=args.scope_model,
        extraction_fallback_model=args.extraction_fallback_model,
        concurrency=args.concurrency,
        request_timeout=args.extraction_request_timeout,
        deadline=args.extraction_deadline,
        fallback_request_timeout=args.fallback_request_timeout,
        fallback_deadline=args.fallback_deadline,
        fallback_chunk_chars=args.fallback_chunk_chars,
        extraction_max_tokens=args.extraction_max_tokens,
        scope_max_tokens=args.scope_max_tokens,
        chunk_chars=args.chunk_chars,
        scope_request_timeout=args.scope_request_timeout,
        scope_deadline=args.scope_deadline,
        scope_claim_limit=args.scope_claim_limit,
        review_unsupported_model=args.review_unsupported_model,
        review_request_timeout=args.review_request_timeout,
        review_deadline=args.review_deadline,
    )
    post_scope_payload = await run_atomic_scope(post_scope_args)
    post_scope_by_id = {row["id"]: row for row in post_scope_payload["cases"]}
    for row in rows:
        row["post_scope"] = post_scope_by_id[row["id"]]
        quality = row.get("quality_judge") or {}
        row["repair_accepted"] = bool(
            row["repair_succeeded"]
            and _scope_is_safe(row["post_scope"])
            and bool((quality.get("result") or {}).get("passed"))
            and bool(row["deterministic_quality"].get("passed"))
        )

    eligible = [row for row in rows if row["repair_eligible"]]
    summary = {
        "cases": len(rows),
        "repair_eligible": len(eligible),
        "repair_succeeded": sum(row["repair_succeeded"] for row in eligible),
        "repair_accepted": sum(row["repair_accepted"] for row in eligible),
        "unaccepted": [row["id"] for row in eligible if not row["repair_accepted"]],
        "unnecessary_repairs": [
            row["id"]
            for row in rows
            if row["human_persistence_verdict"] == "safe" and row["repair_eligible"]
        ],
        "post_scope_errors": sum(
            bool(row["post_scope"].get("scope", {}).get("error")) for row in rows
        ),
        "post_extraction_errors": sum(
            bool(row["post_scope"].get("extraction", {}).get("error")) for row in rows
        ),
        "latency": {
            "repair": _latency(repair_batch["results"]),
            "quality": _latency(quality_batch["results"]),
            "post_scope_wall_seconds": post_scope_payload.get("scope_batch", {}).get(
                "wall_seconds", 0.0
            ),
        },
    }
    payload.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "repair_batch": repair_batch,
            "quality_batch": quality_batch,
            "post_scope_output": str(args.post_scope_output.resolve()),
            "summary": summary,
        }
    )
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--atomic-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repaired-source", type=Path, required=True)
    parser.add_argument("--post-scope-output", type=Path, required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--repair-model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--quality-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--extraction-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--extraction-fallback-model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--scope-model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repair-request-timeout", type=float, default=28.0)
    parser.add_argument("--repair-deadline", type=float, default=30.0)
    parser.add_argument("--repair-max-tokens", type=int, default=1000)
    parser.add_argument("--quality-request-timeout", type=float, default=18.0)
    parser.add_argument("--quality-deadline", type=float, default=20.0)
    parser.add_argument("--quality-max-tokens", type=int, default=450)
    parser.add_argument("--extraction-request-timeout", type=float, default=15.0)
    parser.add_argument("--extraction-deadline", type=float, default=17.0)
    parser.add_argument("--fallback-request-timeout", type=float, default=25.0)
    parser.add_argument("--fallback-deadline", type=float, default=27.0)
    parser.add_argument("--fallback-chunk-chars", type=int, default=160)
    parser.add_argument("--extraction-max-tokens", type=int, default=1200)
    parser.add_argument("--scope-request-timeout", type=float, default=28.0)
    parser.add_argument("--scope-deadline", type=float, default=30.0)
    parser.add_argument("--scope-max-tokens", type=int, default=700)
    parser.add_argument("--scope-claim-limit", type=int, default=8)
    parser.add_argument(
        "--review-unsupported-model", default="deepseek-ai/DeepSeek-V3.2"
    )
    parser.add_argument("--review-request-timeout", type=float, default=28.0)
    parser.add_argument("--review-deadline", type=float, default=30.0)
    parser.add_argument("--chunk-chars", type=int, default=240)
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
