from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
import tomllib
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for value in (SCRIPT_DIR, PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_cross_domain_answer_contract_ab as ab
from config.prompt_config.answer_persistence_repair_prompts import (
    ANSWER_PERSISTENCE_REPAIR_VERSION,
    PERSISTENCE_AUDIT_SYSTEM,
    PERSISTENCE_REPAIR_SYSTEM,
    REPAIR_QUALITY_SYSTEM,
    persistence_audit_prompt,
    persistence_repair_prompt,
    repair_quality_prompt,
)


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "validation/answer-claim-consistency-nonprivate-v3-20260901/results.json"
)
EXPECTED_SOURCES = {
    "full_v2": PROJECT_ROOT
    / "validation/layered-candidate-contract-v2-20260901/results.json",
    "pilot_v8": PROJECT_ROOT
    / "validation/layered-candidate-contract-pilot-v8-20260901/results.json",
    "renderer_v5": PROJECT_ROOT
    / "validation/layered-candidate-contract-pilot-v5-20260901/results.json",
}
PERSISTENT_UNSAFE_IDS = {
    "full_v2:knowledge_hina_en",
    "full_v2:knowledge_saori_chain",
    "full_v2:recall_shell_lore",
    "full_v2:creative_messages_explicit",
    "full_v2:creative_tasks_ja",
    "full_v2:trip_context_advice",
    "full_v2:external_weather",
    "full_v2:casual_greeting",
    "pilot_v8:trip_context_advice",
    "renderer_v5:knowledge_hina_en",
    "renderer_v5:creative_tasks_implicit",
}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_object(text: str) -> dict[str, Any]:
    rendered = str(text or "").strip()
    if rendered.startswith("```"):
        lines = rendered.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        rendered = "\n".join(lines).strip()
    start = rendered.find("{")
    end = rendered.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response does not contain a JSON object")
    value = json.loads(rendered[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response root is not an object")
    return value


def _audit_packet(value: dict[str, Any]) -> dict[str, Any]:
    verdict = str(value.get("verdict") or "").lower()
    if verdict not in {"safe", "unsafe", "uncertain"}:
        raise ValueError(f"invalid audit verdict: {verdict!r}")
    return {
        "verdict": verdict,
        "persistent_additions": list(value.get("persistent_additions") or []),
        "ephemeral_fragments": list(value.get("ephemeral_fragments") or []),
        "reason": str(value.get("reason") or ""),
    }


def _repair_packet(value: dict[str, Any]) -> dict[str, Any]:
    answer = str(value.get("answer") or "").strip()
    if not answer:
        raise ValueError("repair returned an empty answer")
    return {
        "answer": answer,
        "removed_or_reframed": list(value.get("removed_or_reframed") or []),
        "retained_ephemeral": list(value.get("retained_ephemeral") or []),
    }


def _quality_packet(value: dict[str, Any]) -> dict[str, Any]:
    coverage = str(value.get("contract_coverage") or "")
    if coverage not in {"complete", "partial", "lost"}:
        raise ValueError(f"invalid contract coverage: {coverage!r}")
    packet = {
        "passed": bool(value.get("passed")),
        "task_preserved": bool(value.get("task_preserved")),
        "contract_coverage": coverage,
        "language_correct": bool(value.get("language_correct")),
        "natural_roleplay": bool(value.get("natural_roleplay")),
        "internal_mechanism_exposed": bool(
            value.get("internal_mechanism_exposed")
        ),
        "problems": list(value.get("problems") or []),
        "reason": str(value.get("reason") or ""),
    }
    packet["passed"] = bool(
        packet["passed"]
        and packet["task_preserved"]
        and packet["contract_coverage"] == "complete"
        and packet["language_correct"]
        and packet["natural_roleplay"]
        and not packet["internal_mechanism_exposed"]
    )
    return packet


def _persona(chatbot_root: Path) -> dict[str, str]:
    with (chatbot_root / "config/persona.toml").open("rb") as handle:
        value = tomllib.load(handle)
    return {
        "name": str((value.get("persona") or {}).get("name") or ""),
        "description": str(
            (value.get("persona") or {}).get("description") or ""
        ),
        "style": str((value.get("rules") or {}).get("style") or ""),
    }


def _expected_by_id() -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for tag, path in EXPECTED_SOURCES.items():
        source = json.loads(path.read_text(encoding="utf-8"))
        for row in source.get("cases") or []:
            value = deepcopy(row.get("expected") or {})
            if row.get("name") == "knowledge_hina_en":
                value["required_groups"] = [
                    ["Prefect", "Disciplinary Committee", "风纪", "風紀"]
                ]
            expected[f"{tag}:{row['name']}"] = value
    return expected


def _load_rows(source_path: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected = _expected_by_id()
    rows: list[dict[str, Any]] = []
    for item in source.get("cases") or []:
        case_id = str(item["id"])
        if selected_ids and case_id not in selected_ids:
            continue
        item_expected = deepcopy(item.get("expected") or expected.get(case_id) or {})
        if case_id not in expected and "expected" not in item:
            raise ValueError(f"missing expected definition for {case_id}")
        human = str(
            item.get("human_persistence_verdict")
            or ("unsafe" if case_id in PERSISTENT_UNSAFE_IDS else "safe")
        )
        if human not in {"safe", "unsafe"}:
            raise ValueError(f"invalid human persistence verdict for {case_id}: {human}")
        rows.append(
            {
                "id": case_id,
                "name": item["name"],
                "question": item["question"],
                "original_answer": item["answer"],
                "contract": item["contract"],
                "local_guard": item["local_guard"],
                "expected": item_expected,
                "human_persistence_verdict": human,
            }
        )
    unknown = selected_ids - {row["id"] for row in rows}
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")
    return rows


def _engine(
    *, chatbot_root: Path, model: str, timeout_seconds: float, max_tokens: int
) -> Any:
    if str(chatbot_root) not in sys.path:
        sys.path.insert(0, str(chatbot_root))
    from src.llm.engine import LLMConfig, LLMEngine

    config = LLMConfig(
        provider="openai",
        model_name=model,
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        timeout_seconds=timeout_seconds,
        temperature=0.0,
        max_tokens=max_tokens,
        enable_thinking=False,
        response_format={"type": "json_object"},
    )
    return LLMEngine([config], max_retries=1, retry_delay=0.0)


async def _call_batch(
    *,
    label: str,
    model: str,
    items: list[dict[str, Any]],
    chatbot_root: Path,
    concurrency: int,
    request_timeout: float,
    total_deadline: float,
    max_tokens: int,
    system_prompt: str,
    prompt_builder: Callable[[dict[str, Any]], str],
    parser: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    from src.llm.engine import Message

    engine = _engine(
        chatbot_root=chatbot_root,
        model=model,
        timeout_seconds=request_timeout,
        max_tokens=max_tokens,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def call(item: dict[str, Any]) -> dict[str, Any]:
        queued_at = perf_counter()
        queue_seconds = 0.0
        request_seconds = 0.0
        raw = ""
        parsed: dict[str, Any] = {}
        error = ""
        timed_out = False
        try:
            async with semaphore:
                queue_seconds = perf_counter() - queued_at
                started = perf_counter()
                try:
                    raw = await asyncio.wait_for(
                        engine.generate_response(
                            [Message(role="user", content=prompt_builder(item))],
                            system_prompt=system_prompt,
                            max_retries=1,
                            retry_delay=0.0,
                            task_context=f"persistence-repair:{label}:{item['id']}",
                        ),
                        timeout=total_deadline,
                    )
                    parsed = parser(_json_object(raw))
                except TimeoutError:
                    timed_out = True
                    error = f"total deadline exceeded: {total_deadline:.3f}s"
                finally:
                    request_seconds = perf_counter() - started
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "model": model,
            "result": parsed,
            "raw_response": raw,
            "error": error,
            "timed_out": timed_out,
            "queue_seconds": round(queue_seconds, 6),
            "request_seconds": round(request_seconds, 6),
        }

    started = perf_counter()
    results = await asyncio.gather(*(call(item) for item in items))
    return {
        "label": label,
        "model": model,
        "request_timeout": request_timeout,
        "total_deadline": total_deadline,
        "wall_seconds": round(perf_counter() - started, 6),
        "results": results,
    }


def _audit_is_unsafe(call: dict[str, Any]) -> bool:
    return str((call.get("result") or {}).get("verdict") or "") != "safe"


def _audit_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    correct = 0
    false_accepts: list[str] = []
    false_blocks: list[str] = []
    for row in rows:
        predicted_unsafe = _audit_is_unsafe(row.get(field) or {})
        actual_unsafe = row["human_persistence_verdict"] == "unsafe"
        if predicted_unsafe == actual_unsafe:
            correct += 1
        elif predicted_unsafe:
            false_blocks.append(row["id"])
        else:
            false_accepts.append(row["id"])
    return {
        "correct": correct,
        "cases": len(rows),
        "accuracy": round(correct / len(rows), 6) if rows else None,
        "false_accepts": false_accepts,
        "false_blocks": false_blocks,
    }


def _latency(calls: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(item.get("request_seconds") or 0.0) for item in calls]
    return {
        "count": len(values),
        "mean": round(mean(values), 6) if values else 0.0,
        "median": round(median(values), 6) if values else 0.0,
        "max": round(max(values), 6) if values else 0.0,
        "timeouts": sum(bool(item.get("timed_out")) for item in calls),
        "errors": sum(bool(item.get("error")) for item in calls),
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
    selected = {value.strip() for value in args.ids.split(",") if value.strip()}
    rows = _load_rows(args.source.resolve(), selected)
    persona = _persona(chatbot_root)
    output = args.output.resolve()
    payload: dict[str, Any] = {
        "status": "prepared",
        "version": ANSWER_PERSISTENCE_REPAIR_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "persistence_policy": {
            "persistent": [
                "events and states",
                "stable traits and relationships",
                "intent and causes",
                "provenance and commitments",
                "exact literals",
                "creative details outside private writes",
            ],
            "ephemeral": [
                "persona tone and gestures",
                "light subjective reaction anchored to current allowed content",
                "non-committal question, suggestion or offer",
            ],
        },
        "cases": rows,
    }
    _write(output, payload)

    eligible = [
        row for row in rows if row["local_guard"].get("verdict") == "safe"
    ]
    original_audit_batch = await _call_batch(
        label="original_audit",
        model=args.audit_model,
        items=eligible,
        chatbot_root=chatbot_root,
        concurrency=args.concurrency,
        request_timeout=args.audit_request_timeout,
        total_deadline=args.audit_deadline,
        max_tokens=args.audit_max_tokens,
        system_prompt=PERSISTENCE_AUDIT_SYSTEM,
        prompt_builder=lambda row: persistence_audit_prompt(
            question=row["question"],
            persona=persona,
            contract=row["contract"],
            answer=row["original_answer"],
        ),
        parser=_audit_packet,
    )
    audit_by_id = {
        row["id"]: result
        for row, result in zip(
            eligible, original_audit_batch["results"], strict=True
        )
    }
    for row in rows:
        if row["id"] in audit_by_id:
            row["original_audit"] = audit_by_id[row["id"]]
            row["structural_blocked"] = False
        else:
            row["original_audit"] = {
                "result": {"verdict": "unsafe", "reason": "local structural guard"},
                "error": "",
                "timed_out": False,
                "request_seconds": 0.0,
            }
            row["structural_blocked"] = True
    payload.update(
        {
            "status": "original_audit_complete",
            "original_audit_batch": original_audit_batch,
        }
    )
    _write(output, payload)

    repair_inputs = [
        row
        for row in rows
        if not row["structural_blocked"] and _audit_is_unsafe(row["original_audit"])
    ]
    repair_batch = await _call_batch(
        label="repair",
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
            audit=row["original_audit"].get("result") or {},
        ),
        parser=_repair_packet,
    )
    repair_by_id = {
        row["id"]: result
        for row, result in zip(repair_inputs, repair_batch["results"], strict=True)
    }
    repaired_rows: list[dict[str, Any]] = []
    for row in rows:
        call = repair_by_id.get(row["id"])
        row["repair_triggered"] = call is not None
        row["repair"] = call or {}
        repaired = str(((call or {}).get("result") or {}).get("answer") or "")
        row["final_answer"] = repaired or row["original_answer"]
        row["repair_succeeded"] = bool(repaired)
        if repaired:
            repaired_rows.append(row)
    payload.update({"status": "repair_complete", "repair_batch": repair_batch})
    _write(output, payload)

    async def post_audit(label: str, model: str, timeout: float, deadline: float):
        return await _call_batch(
            label=label,
            model=model,
            items=repaired_rows,
            chatbot_root=chatbot_root,
            concurrency=args.concurrency,
            request_timeout=timeout,
            total_deadline=deadline,
            max_tokens=args.audit_max_tokens,
            system_prompt=PERSISTENCE_AUDIT_SYSTEM,
            prompt_builder=lambda row: persistence_audit_prompt(
                question=row["question"],
                persona=persona,
                contract=row["contract"],
                answer=row["final_answer"],
            ),
            parser=_audit_packet,
        )

    qwen_post_batch = await post_audit(
        "qwen_post_audit",
        args.qwen_model,
        args.qwen_request_timeout,
        args.qwen_deadline,
    )
    deep_post_batch = await post_audit(
        "deep_post_audit",
        args.audit_model,
        args.audit_request_timeout,
        args.audit_deadline,
    )
    quality_batch = await _call_batch(
        label="quality",
        model=args.qwen_model,
        items=repaired_rows,
        chatbot_root=chatbot_root,
        concurrency=args.concurrency,
        request_timeout=args.qwen_request_timeout,
        total_deadline=args.qwen_deadline,
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
    for index, row in enumerate(repaired_rows):
        row["qwen_post_audit"] = qwen_post_batch["results"][index]
        row["deep_post_audit"] = deep_post_batch["results"][index]
        row["quality_judge"] = quality_batch["results"][index]
        contract = deepcopy(row["contract"])
        contract["answer"] = row["final_answer"]
        row["deterministic_quality"] = ab._evaluate_case(
            row["expected"], row["final_answer"], contract
        )
        row["repair_accepted"] = bool(
            not _audit_is_unsafe(row["qwen_post_audit"])
            and not _audit_is_unsafe(row["deep_post_audit"])
            and bool((row["quality_judge"].get("result") or {}).get("passed"))
            and bool(row["deterministic_quality"].get("passed"))
        )

    human_unsafe = [
        row for row in rows if row["human_persistence_verdict"] == "unsafe"
    ]
    semantic_unsafe = [row for row in human_unsafe if not row["structural_blocked"]]
    safe_rows = [
        row for row in rows if row["human_persistence_verdict"] == "safe"
    ]
    repaired_human_unsafe = [row for row in semantic_unsafe if row["repair_succeeded"]]
    summary = {
        "cases": len(rows),
        "human_safe": len(safe_rows),
        "human_unsafe": len(human_unsafe),
        "structural_blocks": sum(row["structural_blocked"] for row in rows),
        "original_semantic_audit": _audit_metrics(eligible, "original_audit"),
        "repair": {
            "triggered": len(repair_inputs),
            "succeeded": sum(row["repair_succeeded"] for row in rows),
            "unnecessary_safe_repairs": [
                row["id"] for row in safe_rows if row["repair_triggered"]
            ],
            "human_unsafe_semantic_cases": len(semantic_unsafe),
            "human_unsafe_repaired": len(repaired_human_unsafe),
            "accepted": sum(row.get("repair_accepted", False) for row in rows),
            "accepted_human_unsafe": sum(
                row.get("repair_accepted", False) for row in human_unsafe
            ),
            "remaining_unaccepted_human_unsafe": [
                row["id"]
                for row in human_unsafe
                if not row["structural_blocked"]
                and not row.get("repair_accepted", False)
            ],
        },
        "latency": {
            "original_audit": _latency(original_audit_batch["results"]),
            "repair": _latency(repair_batch["results"]),
            "qwen_post_audit": _latency(qwen_post_batch["results"]),
            "deep_post_audit": _latency(deep_post_batch["results"]),
            "quality": _latency(quality_batch["results"]),
        },
    }
    payload.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "qwen_post_audit_batch": qwen_post_batch,
            "deep_post_audit_batch": deep_post_batch,
            "quality_batch": quality_batch,
            "summary": summary,
        }
    )
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--audit-model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--repair-model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--audit-request-timeout", type=float, default=28.0)
    parser.add_argument("--audit-deadline", type=float, default=30.0)
    parser.add_argument("--qwen-request-timeout", type=float, default=18.0)
    parser.add_argument("--qwen-deadline", type=float, default=20.0)
    parser.add_argument("--repair-request-timeout", type=float, default=33.0)
    parser.add_argument("--repair-deadline", type=float, default=35.0)
    parser.add_argument("--audit-max-tokens", type=int, default=650)
    parser.add_argument("--repair-max-tokens", type=int, default=950)
    parser.add_argument("--quality-max-tokens", type=int, default=500)
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
