from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_cross_domain_answer_contract_ab as ab


def _valid_refs(candidates: dict[str, Any]) -> set[str]:
    return {
        str(item["ref"])
        for domain in candidates.values()
        for group in ("episodes", "concepts")
        for item in domain.get(group) or []
        if item.get("ref")
    }


def _reparse_qwen(row: dict[str, Any]) -> dict[str, Any]:
    raw = str(row["contract"].get("raw_response") or "")
    refs = _valid_refs(row["contract"]["candidates"])
    try:
        packet, answer = ab._contract_packet(raw)
        contract = ab._validate_contract(packet, refs, answer=answer)
        error = ""
    except Exception as exc:
        contract = {
            "intent_mode": "conversation",
            "answer_language": "other",
            "answerability": "not_applicable",
            "domain_decisions": [],
            "claims": [],
            "grounded_answer_outline": "",
            "missing_requirements": [],
            "upgrade_recommended": False,
            "write_candidates": {"private": [], "knowledge": []},
            "answer": "",
            "validation": {
                "invalid_refs": [],
                "supported_without_refs": [],
                "parse_error": f"{type(exc).__name__}: {exc}",
            },
        }
        error = contract["validation"]["parse_error"]
    return {
        "answer": contract["answer"],
        "contract": contract,
        "error": error,
        "evaluation": ab._evaluate_case(row["expected"], contract["answer"], contract),
    }


async def run(
    *,
    chatbot_root: Path,
    source_path: Path,
    output_path: Path,
    concurrency: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv
    from src.llm.engine import Message

    load_dotenv(chatbot_root / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = list(source.get("cases") or [])
    engine = ab._engine(
        model="deepseek-ai/DeepSeek-V3.2",
        json_mode=False,
        max_tokens=1_000,
        temperature=0.0,
    )
    judge_engine = ab._engine(
        model="Qwen/Qwen3.5-9B",
        json_mode=True,
        max_tokens=1_800,
        temperature=0.0,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    completed: dict[str, dict[str, Any]] = {}
    lock = asyncio.Lock()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def evaluate(row: dict[str, Any]) -> None:
        async with semaphore:
            candidates = row["contract"]["candidates"]
            refs = _valid_refs(candidates)
            started = perf_counter()
            raw = ""
            error = ""
            try:
                raw = await engine.generate_response(
                    [
                        Message(
                            role="user",
                            content=ab.answer_evidence_contract_prompt(
                                question=row["question"],
                                candidates=candidates,
                                current_user_label=(
                                    "Teacher B"
                                    if row["name"] == "cross_platform_isolation"
                                    else "Teacher A"
                                ),
                                request_plan={},
                            ),
                        )
                    ],
                    system_prompt=ab.ANSWER_EVIDENCE_CONTRACT_SYSTEM,
                    task_context=f"saved-contract-deepseek:{row['name']}",
                )
                packet, answer = ab._contract_packet(raw)
                contract = ab._validate_contract(packet, refs, answer=answer)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                contract = {
                    "intent_mode": "conversation",
                    "answer_language": "other",
                    "answerability": "not_applicable",
                    "domain_decisions": [],
                    "claims": [],
                    "grounded_answer_outline": "",
                    "missing_requirements": [],
                    "upgrade_recommended": False,
                    "write_candidates": {"private": [], "knowledge": []},
                    "answer": "",
                    "validation": {
                        "invalid_refs": [],
                        "supported_without_refs": [],
                        "parse_error": error,
                    },
                }
            seconds = perf_counter() - started
            result = {
                "name": row["name"],
                "question": row["question"],
                "expected": row["expected"],
                "candidates": candidates,
                "qwen_original": {
                    "answer": row["contract"]["answer"],
                    "evaluation": row["contract"]["evaluation"],
                },
                "qwen_reparsed": _reparse_qwen(row),
                "deepseek": {
                    "answer": contract["answer"],
                    "contract": contract,
                    "raw_response": raw,
                    "error": error,
                    "generation_seconds": round(seconds, 6),
                    "projected_total_seconds": round(
                        seconds + float(row["contract"].get("probe_seconds") or 0.0),
                        6,
                    ),
                    "evaluation": ab._evaluate_case(
                        row["expected"], contract["answer"], contract
                    ),
                },
            }
            async with lock:
                completed[row["name"]] = result
                ab._write_json_atomic(
                    output_path,
                    {
                        "status": "in_progress",
                        "completed": len(completed),
                        "total": len(rows),
                        "cases": list(completed.values()),
                    },
                )

    wall_started = perf_counter()
    await asyncio.gather(*(evaluate(row) for row in rows))
    generation_wall_seconds = perf_counter() - wall_started
    ordered = [completed[row["name"]] for row in rows]

    judge_input = [
        {
            "name": row["name"],
            "question": row["question"],
            "minimum_answer": row["expected"]["minimum_answer"],
            "expected_intent": row["expected"]["expected_intent"],
            "private_write_expected": row["expected"]["private_write"],
            "baseline_answer": row["qwen_reparsed"]["answer"],
            "contract_answer": row["deepseek"]["answer"],
            "contract_metadata": {
                "intent_mode": row["deepseek"]["contract"]["intent_mode"],
                "claims": row["deepseek"]["contract"]["claims"],
                "write_candidates": row["deepseek"]["contract"]["write_candidates"],
            },
        }
        for row in ordered
    ]
    judged = await ab._judge_batches(judge_engine, judge_input)
    judge_by_name = {
        str(item.get("name")): item
        for item in judged["results"]
        if item.get("name")
    }
    for row in ordered:
        row["judge"] = judge_by_name.get(row["name"], {})

    generation_seconds = [
        float(row["deepseek"]["generation_seconds"]) for row in ordered
    ]
    total_seconds = [
        float(row["deepseek"]["projected_total_seconds"]) for row in ordered
    ]
    payload = {
        "status": "complete",
        "version": ab.ANSWER_EVIDENCE_CONTRACT_VERSION,
        "source": str(source_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "cases": len(ordered),
            "qwen_original_deterministic_pass": sum(
                bool(row["qwen_original"]["evaluation"]["passed"]) for row in ordered
            ),
            "qwen_reparsed_deterministic_pass": sum(
                bool(row["qwen_reparsed"]["evaluation"]["passed"]) for row in ordered
            ),
            "deepseek_deterministic_pass": sum(
                bool(row["deepseek"]["evaluation"]["passed"]) for row in ordered
            ),
            "qwen_reparsed_judge_pass": sum(
                bool((row.get("judge") or {}).get("baseline", {}).get("passed"))
                for row in ordered
            ),
            "deepseek_judge_pass": sum(
                bool((row.get("judge") or {}).get("contract", {}).get("passed"))
                for row in ordered
            ),
            "deepseek_parse_failures": sum(bool(row["deepseek"]["error"]) for row in ordered),
            "generation_wall_seconds": round(generation_wall_seconds, 6),
            "generation_mean_seconds": round(mean(generation_seconds), 6),
            "generation_median_seconds": round(median(generation_seconds), 6),
            "projected_total_mean_seconds": round(mean(total_seconds), 6),
            "projected_total_median_seconds": round(median(total_seconds), 6),
            "judge_seconds": judged["seconds"],
        },
        "cases": ordered,
    }
    ab._write_json_atomic(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    payload = asyncio.run(
        run(
            chatbot_root=args.chatbot_root.resolve(),
            source_path=args.source.resolve(),
            output_path=args.output.resolve(),
            concurrency=max(1, args.concurrency),
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
