from __future__ import annotations

import argparse
import asyncio
from collections import Counter
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

from config.prompt_config.request_scoped_answer_prompts import (
    CONTROL_ANSWER_SYSTEM,
    REQUEST_SCOPED_AB_JUDGE_SYSTEM,
    REQUEST_SCOPED_ANSWER_SYSTEM,
    REQUEST_SCOPED_ANSWER_VERSION,
    control_answer_prompt,
    request_scoped_ab_judge_batch_prompt,
    request_scoped_answer_prompt,
)
from memory_demo.contracts import (
    Limitation,
    QuestionPremise,
    RequestAnswerContract,
    RequestedAction,
    RuntimeReceipt,
)
from run_answer_persistence_repair_eval import (
    _call_batch,
    _latency,
    _persona,
    _write,
)


def _answer_packet(value: dict[str, Any]) -> dict[str, Any]:
    answer = str(value.get("answer") or "").strip()
    if not answer:
        raise ValueError("answer is required")
    return {"answer": answer}


def _judge_side(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judge side must be an object")
    fields = (
        "action_truth",
        "factual_grounding",
        "premise_handling",
        "task_preservation",
        "natural_roleplay",
    )
    scores: dict[str, int] = {}
    for field in fields:
        score = int(value.get(field, -1))
        if score not in {0, 1, 2}:
            raise ValueError(f"invalid {field} score: {score}")
        scores[field] = score
    return {
        "passed": bool(value.get("passed")),
        **scores,
        "violations": [str(item) for item in value.get("violations") or []],
    }


def _judge_packet(value: dict[str, Any]) -> dict[str, Any]:
    winner = str(value.get("winner") or "")
    if winner not in {"control", "request_scoped", "tie"}:
        raise ValueError(f"invalid winner: {winner!r}")
    return {
        "control": _judge_side(value.get("control")),
        "request_scoped": _judge_side(value.get("request_scoped")),
        "winner": winner,
        "reason": str(value.get("reason") or ""),
    }


def _judge_batch_packet(value: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in value.get("results") or []:
        case_id = str(item.get("id") or "").strip()
        if not case_id:
            raise ValueError("judge result id is required")
        results.append({"id": case_id, **_judge_packet(item)})
    if not results:
        raise ValueError("judge batch returned no results")
    return {"results": results}


def _receipt(value: dict[str, Any], default_request_id: str) -> RuntimeReceipt:
    return RuntimeReceipt(
        request_id=str(value.get("request_id") or default_request_id),
        receipt_id=str(value.get("receipt_id") or ""),
        action=str(value.get("action") or ""),
        target=str(value.get("target") or ""),
        status=str(value.get("status") or ""),  # type: ignore[arg-type]
        result_summary=str(value.get("result_summary") or ""),
        error_summary=str(value.get("error_summary") or ""),
    )


def _build_snapshot(row: dict[str, Any]) -> tuple[RequestAnswerContract, dict[str, Any]]:
    value = row["contract"]
    request_id = str(value["request_id"])
    contract = RequestAnswerContract(
        request_id=request_id,
        question=str(row["question"]),
        intent_mode=str(value.get("intent_mode") or "conversation"),
        answer_language=str(value.get("answer_language") or "other"),
        answerability=str(value.get("answerability") or "not_applicable"),
        requested_actions=tuple(
            RequestedAction(
                str(item["action"]),
                str(item["target"]),
                str(item["public_action_summary"]),
            )
            for item in value.get("requested_actions") or []
        ),
        supported_facts=tuple(str(item) for item in value.get("supported_facts") or []),
        limitations=tuple(
            Limitation(
                limitation_id=str(item["limitation_id"]),
                text=str(item["text"]),
                kind=str(item["kind"]),  # type: ignore[arg-type]
                action=str(item.get("action") or ""),
                target=str(item.get("target") or ""),
                claim_ref=str(item.get("claim_ref") or ""),
            )
            for item in value.get("limitations") or []
        ),
        private_write_candidates=tuple(
            str(item) for item in value.get("private_write_candidates") or []
        ),
        knowledge_write_candidates=tuple(
            str(item) for item in value.get("knowledge_write_candidates") or []
        ),
        question_premises=tuple(
            QuestionPremise(
                premise=str(item["premise"]),
                status=str(item["status"]),  # type: ignore[arg-type]
                evidence_refs=tuple(str(ref) for ref in item.get("evidence_refs") or []),
            )
            for item in value.get("question_premises") or []
        ),
    )
    diagnostics: dict[str, Any] = {
        "rejected_receipts": [],
        "late_receipt_ids": [],
        "snapshot_phase": str(row.get("snapshot_phase") or "before_late"),
    }
    late: list[RuntimeReceipt] = []
    for item in value.get("receipts") or []:
        receipt = _receipt(item, request_id)
        if str(item.get("timing") or "before_answer") == "late":
            late.append(receipt)
            diagnostics["late_receipt_ids"].append(receipt.receipt_id)
            continue
        try:
            contract = contract.with_receipt(receipt)
        except ValueError as exc:
            diagnostics["rejected_receipts"].append(
                {"receipt_id": receipt.receipt_id, "reason": str(exc)}
            )
    before_late = contract
    after_late = contract
    for receipt in late:
        try:
            after_late = after_late.with_receipt(receipt)
        except ValueError as exc:
            diagnostics["rejected_receipts"].append(
                {"receipt_id": receipt.receipt_id, "reason": str(exc)}
            )
    snapshot = after_late if diagnostics["snapshot_phase"] == "after_late" else before_late
    diagnostics["visible_receipt_ids"] = [
        receipt.receipt_id for receipt in snapshot.matching_receipts()
    ]
    diagnostics["ignored_receipt_ids"] = list(snapshot.ignored_receipt_ids())
    diagnostics["next_snapshot_receipt_ids"] = [
        receipt.receipt_id for receipt in after_late.matching_receipts()
    ]
    return snapshot, diagnostics


def _atomic_source(rows: list[dict[str, Any]], answer_field: str, prefix: str) -> dict[str, Any]:
    return {
        "version": REQUEST_SCOPED_ANSWER_VERSION + "+" + prefix,
        "cases": [
            {
                "id": f"{prefix}:{row['id']}",
                "name": str(row["id"]).split(":", 1)[-1],
                "question": row["question"],
                "answer": str((row[answer_field].get("result") or {}).get("answer") or ""),
                "contract": row["snapshot"].atomic_audit_contract(),
                "local_guard": {"verdict": "safe", "reasons": []},
                "expected": {},
                "human_persistence_verdict": "safe",
            }
            for row in rows
            if not row[answer_field].get("error")
        ],
    }


def _renderer_persona(persona: dict[str, str]) -> dict[str, str]:
    """Keep identity and surface voice without injecting relationship lore."""
    return {
        "self_name": str(persona.get("name") or ""),
        "user_address": "",
        "style": "保持该角色的称呼和轻量语气；回答简洁，不新增事实、关系或承诺。",
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    chatbot_root = args.chatbot_root.resolve()
    if str(chatbot_root) not in sys.path:
        sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv

    load_dotenv(chatbot_root / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    if not os.getenv("SILICONFLOW_API_KEY"):
        raise RuntimeError("SILICONFLOW_API_KEY is not configured")
    source = json.loads(args.source.resolve().read_text(encoding="utf-8"))
    selected = {item.strip() for item in args.ids.split(",") if item.strip()}
    rows = [deepcopy(item) for item in source.get("cases") or []]
    if selected:
        rows = [row for row in rows if row["id"] in selected]
    if not rows:
        raise ValueError("no request-scoped cases selected")
    persona = _persona(chatbot_root)
    renderer_persona = _renderer_persona(persona)
    for row in rows:
        snapshot, diagnostics = _build_snapshot(row)
        row["snapshot"] = snapshot
        row["generation_view"] = snapshot.generation_view()
        row["diagnostics"] = diagnostics

    control_batch = await _call_batch(
        label="request_scoped_control",
        model=args.generation_model,
        items=rows,
        chatbot_root=chatbot_root,
        concurrency=args.concurrency,
        request_timeout=args.generation_request_timeout,
        total_deadline=args.generation_deadline,
        max_tokens=args.generation_max_tokens,
        system_prompt=CONTROL_ANSWER_SYSTEM,
        prompt_builder=lambda row: control_answer_prompt(
            question=row["question"],
            persona=renderer_persona,
            view=row["generation_view"],
        ),
        parser=_answer_packet,
    )
    scoped_batch = await _call_batch(
        label="request_scoped_treatment",
        model=args.generation_model,
        items=rows,
        chatbot_root=chatbot_root,
        concurrency=args.concurrency,
        request_timeout=args.generation_request_timeout,
        total_deadline=args.generation_deadline,
        max_tokens=args.generation_max_tokens,
        system_prompt=REQUEST_SCOPED_ANSWER_SYSTEM,
        prompt_builder=lambda row: request_scoped_answer_prompt(
            question=row["question"],
            persona=renderer_persona,
            view=row["generation_view"],
        ),
        parser=_answer_packet,
    )
    for row, control, scoped in zip(
        rows, control_batch["results"], scoped_batch["results"], strict=True
    ):
        row["control"] = control
        row["request_scoped"] = scoped

    judge_rows = [
        row
        for row in rows
        if not row["control"].get("error") and not row["request_scoped"].get("error")
    ]
    judge_cases = [
        {
            "id": row["id"],
            "question": row["question"],
            "persona": renderer_persona,
            "request_contract": row["generation_view"],
            "expected_behavior": row["expected_behavior"],
            "answers": {
                "control": row["control"]["result"]["answer"],
                "request_scoped": row["request_scoped"]["result"]["answer"],
            },
        }
        for row in judge_rows
    ]
    judge_batch_size = max(1, args.judge_batch_size)
    judge_items = [
        {
            "id": f"judge-batch-{index // judge_batch_size + 1}",
            "cases": judge_cases[index : index + judge_batch_size],
        }
        for index in range(0, len(judge_cases), judge_batch_size)
    ]
    judge_batch = await _call_batch(
        label="request_scoped_ab_judge",
        model=args.judge_model,
        items=judge_items,
        chatbot_root=chatbot_root,
        concurrency=args.concurrency,
        request_timeout=args.judge_request_timeout,
        total_deadline=args.judge_deadline,
        max_tokens=args.judge_max_tokens,
        system_prompt=REQUEST_SCOPED_AB_JUDGE_SYSTEM,
        prompt_builder=lambda item: request_scoped_ab_judge_batch_prompt(
            item["cases"]
        ),
        parser=_judge_batch_packet,
    )
    judge_by_id: dict[str, dict[str, Any]] = {}
    for call in judge_batch["results"]:
        if call.get("error"):
            continue
        for result in (call.get("result") or {}).get("results") or []:
            judge_by_id[result["id"]] = {
                "model": call.get("model"),
                "result": {key: value for key, value in result.items() if key != "id"},
                "raw_response": call.get("raw_response", ""),
                "error": "",
                "timed_out": False,
                "queue_seconds": call.get("queue_seconds", 0.0),
                "request_seconds": call.get("request_seconds", 0.0),
            }
    for row in rows:
        row["judge"] = judge_by_id.get(row["id"], {})

    control_audit_source = args.control_audit_source.resolve()
    scoped_audit_source = args.scoped_audit_source.resolve()
    _write(control_audit_source, _atomic_source(rows, "control", "request_control"))
    _write(scoped_audit_source, _atomic_source(rows, "request_scoped", "request_scoped"))

    completed_judges = [
        row["judge"]["result"]
        for row in rows
        if (row.get("judge") or {}).get("result")
    ]
    winner_counts = Counter(item["winner"] for item in completed_judges)
    summary = {
        "cases": len(rows),
        "control_generation": _latency(control_batch["results"]),
        "request_scoped_generation": _latency(scoped_batch["results"]),
        "judge": _latency(judge_batch["results"]),
        "control_passed": sum(item["control"]["passed"] for item in completed_judges),
        "request_scoped_passed": sum(
            item["request_scoped"]["passed"] for item in completed_judges
        ),
        "completed_judges": len(completed_judges),
        "judge_calls": len(judge_batch["results"]),
        "winner_counts": dict(winner_counts),
        "rejected_receipts": sum(
            len(row["diagnostics"]["rejected_receipts"]) for row in rows
        ),
        "ignored_receipts": sum(
            len(row["diagnostics"]["ignored_receipt_ids"]) for row in rows
        ),
        "batch_wall_seconds": {
            "control": control_batch["wall_seconds"],
            "request_scoped": scoped_batch["wall_seconds"],
            "judge": judge_batch["wall_seconds"],
        },
    }
    serializable_rows = []
    for row in rows:
        item = deepcopy(row)
        item["snapshot"] = row["snapshot"].atomic_audit_contract()
        serializable_rows.append(item)
    payload = {
        "status": "complete",
        "version": REQUEST_SCOPED_ANSWER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "generation_model": args.generation_model,
        "judge_model": args.judge_model,
        "cases": serializable_rows,
        "control_batch": control_batch,
        "request_scoped_batch": scoped_batch,
        "judge_batch": judge_batch,
        "control_audit_source": str(control_audit_source),
        "request_scoped_audit_source": str(scoped_audit_source),
        "summary": summary,
    }
    _write(args.output.resolve(), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-audit-source", type=Path, required=True)
    parser.add_argument("--scoped-audit-source", type=Path, required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--generation-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--judge-model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--generation-request-timeout", type=float, default=18.0)
    parser.add_argument("--generation-deadline", type=float, default=20.0)
    parser.add_argument("--generation-max-tokens", type=int, default=600)
    parser.add_argument("--judge-request-timeout", type=float, default=28.0)
    parser.add_argument("--judge-deadline", type=float, default=30.0)
    parser.add_argument("--judge-max-tokens", type=int, default=2400)
    parser.add_argument("--judge-batch-size", type=int, default=4)
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
