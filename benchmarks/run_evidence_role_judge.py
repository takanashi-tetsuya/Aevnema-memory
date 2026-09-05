from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.prompt_config.evidence_role_prompts import (
    EVIDENCE_ROLE_PROMPT_VERSION,
    EVIDENCE_ROLE_SYSTEM,
    evidence_role_prompt,
)


VALID_SUPPORT = {"complete", "partial_or_related", "none", "not_applicable"}
VALID_POLICY = {"answer", "uncertainty", "creative_context", "discard"}


def _gold(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("knowledge_expected"):
        return "complete", "answer"
    category = str(row.get("category", ""))
    if category == "same_entity_missing":
        return "partial_or_related", "uncertainty"
    if category == "creative":
        return "not_applicable", "creative_context"
    return "none", "discard"


def _parse_json(text: str) -> dict[str, Any]:
    rendered = str(text).strip()
    if rendered.startswith("```"):
        rendered = rendered.removeprefix("```json").removeprefix("```")
        rendered = rendered.removesuffix("```").strip()
    try:
        value = json.loads(rendered)
    except json.JSONDecodeError:
        start = rendered.find("{")
        end = rendered.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(rendered[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("evidence role response must be an object")
    support = str(value.get("factual_support", "")).strip()
    policy = str(value.get("use_policy", "")).strip()
    if support not in VALID_SUPPORT:
        raise ValueError(f"invalid factual_support: {support}")
    if policy not in VALID_POLICY:
        raise ValueError(f"invalid use_policy: {policy}")
    ids = value.get("supporting_ids", [])
    missing = value.get("missing_requirements", [])
    if not isinstance(ids, list) or not isinstance(missing, list):
        raise ValueError("supporting_ids and missing_requirements must be lists")
    return {
        "factual_support": support,
        "use_policy": policy,
        "supporting_ids": ids,
        "grounded_answer_outline": str(
            value.get("grounded_answer_outline", "")
        ).strip(),
        "missing_requirements": [str(item) for item in missing],
        "reason": str(value.get("reason", "")).strip(),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


async def run(
    *,
    chatbot_root: Path,
    input_path: Path,
    output_path: Path,
    model: str,
    concurrency: int,
    evidence_limit: int,
    checkpoint_every: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv
    from src.llm.engine import LLMConfig, LLMEngine, Message

    load_dotenv(chatbot_root / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    source = json.loads(input_path.read_text(encoding="utf-8"))
    rows = list(source.get("cases") or [])
    completed_by_name: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        try:
            previous = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if (
            previous.get("status") == "in_progress"
            and previous.get("model") == model
            and previous.get("input") == str(input_path)
            and previous.get("prompt_version") == EVIDENCE_ROLE_PROMPT_VERSION
        ):
            completed_by_name = {
                str(item["name"]): item
                for item in previous.get("cases") or []
                if isinstance(item, dict) and item.get("name")
            }
    config = LLMConfig(
        provider="openai",
        model_name=model,
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        timeout_seconds=90,
        temperature=0.0,
        max_tokens=500,
        enable_thinking=False,
        response_format={"type": "json_object"},
    )
    engine = LLMEngine([config], max_retries=2, retry_delay=0.25)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def judge(row: dict[str, Any]) -> dict[str, Any]:
        evidence = list(row.get("top_evidence") or [])[:evidence_limit]
        concepts = list(row.get("top_concepts") or [])[:8]
        gold_support, gold_policy = _gold(row)
        try:
            async with semaphore:
                started = perf_counter()
                response = await engine.generate_response(
                    [
                        Message(
                            role="user",
                            content=evidence_role_prompt(
                                row["question"], evidence, concepts
                            ),
                        )
                    ],
                    system_prompt=EVIDENCE_ROLE_SYSTEM,
                    task_context=f"evidence-role:{row['name']}",
                )
                elapsed = perf_counter() - started
            parsed = _parse_json(response)
            error = ""
        except Exception as exc:
            elapsed = perf_counter() - started if "started" in locals() else 0.0
            parsed = {
                "factual_support": "none",
                "use_policy": "discard",
                "supporting_ids": [],
                "grounded_answer_outline": "",
                "missing_requirements": [],
                "reason": "",
            }
            error = f"{type(exc).__name__}: {exc}"
        predicted_answerable = parsed["factual_support"] == "complete"
        expected_answerable = gold_support == "complete"
        return {
            "name": row["name"],
            "family": row["family"],
            "category": row["category"],
            "language": row["language"],
            "question": row["question"],
            "bge_top_score": row["top_score"],
            "gold_factual_support": gold_support,
            "gold_use_policy": gold_policy,
            "predicted": parsed,
            "support_exact": parsed["factual_support"] == gold_support,
            "policy_exact": parsed["use_policy"] == gold_policy,
            "answerability_correct": predicted_answerable == expected_answerable,
            "false_sufficient": predicted_answerable and not expected_answerable,
            "false_insufficient": expected_answerable and not predicted_answerable,
            "seconds": round(elapsed, 6),
            "error": error,
            "top_evidence": evidence,
            "top_concepts": concepts,
        }

    started = perf_counter()
    pending = [row for row in rows if row["name"] not in completed_by_name]
    tasks = [asyncio.create_task(judge(row)) for row in pending]
    completed_since_checkpoint = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        completed_by_name[result["name"]] = result
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_every:
            _write_json_atomic(
                output_path,
                {
                    "status": "in_progress",
                    "experiment": "evidence-role-judge-v1",
                    "model": model,
                    "input": str(input_path),
                    "prompt_version": EVIDENCE_ROLE_PROMPT_VERSION,
                    "completed": len(completed_by_name),
                    "total": len(rows),
                    "cases": list(completed_by_name.values()),
                },
            )
            completed_since_checkpoint = 0
    judged = [completed_by_name[row["name"]] for row in rows]
    wall_seconds = perf_counter() - started
    errors = [row for row in judged if row["error"]]
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in judged:
        categories.setdefault(row["category"], []).append(row)
    payload = {
        "status": "complete",
        "experiment": "evidence-role-judge-v1",
        "model": model,
        "input": str(input_path),
        "prompt_version": EVIDENCE_ROLE_PROMPT_VERSION,
        "summary": {
            "cases": len(judged),
            "wall_seconds": round(wall_seconds, 6),
            "mean_seconds": round(mean(row["seconds"] for row in judged), 6),
            "p95_seconds": round(_percentile([row["seconds"] for row in judged], 0.95), 6),
            "error_count": len(errors),
            "answerability_accuracy": round(
                sum(row["answerability_correct"] for row in judged) / len(judged), 6
            ),
            "support_exact_accuracy": round(
                sum(row["support_exact"] for row in judged) / len(judged), 6
            ),
            "policy_exact_accuracy": round(
                sum(row["policy_exact"] for row in judged) / len(judged), 6
            ),
            "false_sufficient": sum(row["false_sufficient"] for row in judged),
            "false_insufficient": sum(row["false_insufficient"] for row in judged),
        },
        "category_summary": {
            key: {
                "cases": len(items),
                "answerability_accuracy": round(
                    sum(row["answerability_correct"] for row in items) / len(items), 6
                ),
                "policy_accuracy": round(
                    sum(row["policy_exact"] for row in items) / len(items), 6
                ),
                "false_sufficient": sum(row["false_sufficient"] for row in items),
                "false_insufficient": sum(row["false_insufficient"] for row in items),
            }
            for key, items in sorted(categories.items())
        },
        "errors": errors,
        "false_sufficient": [row for row in judged if row["false_sufficient"]],
        "false_insufficient": [row for row in judged if row["false_insufficient"]],
        "policy_mismatches": [row for row in judged if not row["policy_exact"]],
        "cases": judged,
    }
    _write_json_atomic(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--evidence-limit", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    args = parser.parse_args()
    payload = asyncio.run(
        run(
            chatbot_root=args.chatbot_root.resolve(),
            input_path=args.input.resolve(),
            output_path=args.output.resolve(),
            model=args.model,
            concurrency=args.concurrency,
            evidence_limit=max(1, args.evidence_limit),
            checkpoint_every=max(1, args.checkpoint_every),
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
