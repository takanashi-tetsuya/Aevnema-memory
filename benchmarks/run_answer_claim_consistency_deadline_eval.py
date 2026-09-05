from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from statistics import mean, median
import sys
from time import perf_counter
import tomllib
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for value in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from config.prompt_config.answer_claim_consistency_prompts import (
    ANSWER_CLAIM_CONSISTENCY_SYSTEM,
    ANSWER_CLAIM_CONSISTENCY_VERSION,
    answer_claim_consistency_prompt,
)


SOURCE_SPECS = (
    {
        "tag": "full_v2",
        "path": PROJECT_ROOT
        / "validation/layered-candidate-contract-v2-20260901/results.json",
        "unsafe": {
            "private_password_zh": "回答把暗号进一步断言为只属于当前用户的秘密，超出合同原子主张。",
            "cross_platform_isolation": "回答在暗号内容被列为未知时凭空给出具体暗号，并把未知扩写成遗忘。",
            "mixed_private_public": "回答把私人暗号进一步断言为只有用户本人知道的秘密。",
            "knowledge_hina_en": "回答在职位之外新增当地安全问题频发及该职位具体职责。",
            "knowledge_saori_chain": "回答把射伤老师改写成不小心，新增了行为意图判断。",
            "recall_shell_lore": "回答新增重要成员、总是很有干劲和独特魅力等人物判断。",
            "creative_messages_explicit": "回答在三条许可消息之外新增学生们依赖老师的关系判断。",
            "creative_tasks_ja": "回答把新任务进一步归因为对策委员会已提出的请求。",
            "trip_context_advice": "回答把背景候选扩写成当前出差所处的危险和状态。",
            "external_weather": "回答把无法获得实时天气扩写为未连接气象卫星的具体原因。",
            "casual_greeting": "寒暄答案加入合同中不存在的当前工作状态。",
        },
    },
    {
        "tag": "pilot_v8",
        "path": PROJECT_ROOT
        / "validation/layered-candidate-contract-pilot-v8-20260901/results.json",
        "unsafe": {
            "trip_context_advice": "回答在没有支持主张时把历史背景写成当前建议的事实前提。",
        },
    },
    {
        "tag": "renderer_v5",
        "path": PROJECT_ROOT
        / "validation/layered-candidate-contract-pilot-v5-20260901/results.json",
        "unsafe": {
            "private_password_ja": "渲染器新增过去感想、特殊含义和秘密钥匙等合同外内容。",
            "knowledge_hina_en": "渲染器改变回答语言并新增人物特征。",
            "private_provenance_conflict": "渲染器新增白子的性格、饮品和情绪等合同外内容。",
            "creative_tasks_implicit": "渲染器在许可的私人创作清单之外继续补写事件细节。",
        },
    },
)

STRUCTURAL_UNSAFE_KEYS = (
    "invalid_refs",
    "supported_without_refs",
    "context_as_evidence",
    "undeclared_review_promotions",
    "support_outside_required_domains",
    "creative_without_writes",
    "parse_error",
)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _persona(chatbot_root: Path) -> dict[str, str]:
    path = chatbot_root / "config/persona.toml"
    if not path.exists():
        return {"name": "", "description": "", "style": ""}
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    return {
        "name": str((value.get("persona") or {}).get("name") or ""),
        "description": str(
            (value.get("persona") or {}).get("description") or ""
        ),
        "style": str((value.get("rules") or {}).get("style") or ""),
    }


def _local_guard(contract: dict[str, Any]) -> dict[str, Any]:
    validation = contract.get("validation") or {}
    reasons = [key for key in STRUCTURAL_UNSAFE_KEYS if validation.get(key)]
    if (
        str(contract.get("intent_mode") or "") == "creative"
        and not (contract.get("write_candidates") or {}).get("private")
    ):
        reasons.append("creative_without_private_write")
    return {"verdict": "unsafe" if reasons else "safe", "reasons": reasons}


def _load_cases(selected_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        source = json.loads(spec["path"].read_text(encoding="utf-8"))
        unsafe = spec["unsafe"]
        for source_row in source.get("cases") or []:
            case_id = f"{spec['tag']}:{source_row['name']}"
            if selected_ids and case_id not in selected_ids:
                continue
            layered = source_row.get("layered") or {}
            is_safe = source_row["name"] not in unsafe
            rows.append(
                {
                    "id": case_id,
                    "source": str(spec["path"]),
                    "name": source_row["name"],
                    "question": source_row.get("question", ""),
                    "answer": layered.get("answer", ""),
                    "contract": layered.get("contract") or {},
                    "human": {
                        "verdict": "safe" if is_safe else "unsafe",
                        "reason": unsafe.get(
                            source_row["name"],
                            "回答没有超出该版本合同允许的事实、限制或私人创作。",
                        ),
                    },
                    "local_guard": _local_guard(layered.get("contract") or {}),
                }
            )
    unknown = selected_ids - {row["id"] for row in rows}
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")
    return rows


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
    verdict = str(value.get("verdict") or "").lower()
    if verdict not in {"safe", "unsafe", "uncertain"}:
        raise ValueError(f"invalid verdict: {verdict!r}")
    value["verdict"] = verdict
    return value


def _answer_chunks(text: str, max_chars: int) -> list[str]:
    """Split a long answer on natural boundaries without discarding any text."""

    normalized = str(text or "").strip()
    if not normalized:
        return [""]
    pieces = [
        value.strip()
        for value in re.split(r"(?<=[。！？!?])\s+|\n{2,}", normalized)
        if value.strip()
    ]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(piece), max_chars):
                chunks.append(piece[start : start + max_chars])
            continue
        candidate = f"{current}\n{piece}".strip() if current else piece
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [normalized]


def _aggregate_chunk_results(values: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [str(value.get("verdict") or "") for value in values]
    if "unsafe" in verdicts:
        verdict = "unsafe"
    elif "uncertain" in verdicts:
        verdict = "uncertain"
    elif values and all(verdict == "safe" for verdict in verdicts):
        verdict = "safe"
    else:
        verdict = ""
    return {
        "used": True,
        "verdict": verdict,
        "timed_out": any(bool(value.get("timed_out")) for value in values),
        "error": "" if verdict else "one or more chunks did not complete",
        "chunks": values,
    }


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


async def _audit_batch(
    *,
    label: str,
    model: str,
    rows: list[dict[str, Any]],
    chatbot_root: Path,
    persona: dict[str, str],
    concurrency: int,
    request_timeout_seconds: float,
    total_deadline_seconds: float,
    max_tokens: int,
) -> dict[str, Any]:
    from src.llm.engine import Message

    engine = _engine(
        chatbot_root=chatbot_root,
        model=model,
        timeout_seconds=request_timeout_seconds,
        max_tokens=max_tokens,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def audit(row: dict[str, Any]) -> dict[str, Any]:
        queued_at = perf_counter()
        queue_seconds = 0.0
        request_seconds = 0.0
        raw = ""
        result: dict[str, Any] = {}
        error = ""
        timed_out = False
        try:
            async with semaphore:
                queue_seconds = perf_counter() - queued_at
                started = perf_counter()
                try:
                    raw = await asyncio.wait_for(
                        engine.generate_response(
                            [
                                Message(
                                    role="user",
                                    content=answer_claim_consistency_prompt(
                                        question=row["question"],
                                        persona=persona,
                                        contract=row["contract"],
                                        answer=row["answer"],
                                    ),
                                )
                            ],
                            system_prompt=ANSWER_CLAIM_CONSISTENCY_SYSTEM,
                            max_retries=1,
                            retry_delay=0.0,
                            task_context=f"answer-consistency:{label}:{row['id']}",
                        ),
                        timeout=total_deadline_seconds,
                    )
                    result = _json_object(raw)
                except TimeoutError:
                    timed_out = True
                    error = (
                        f"total deadline exceeded: {total_deadline_seconds:.3f}s"
                    )
                finally:
                    request_seconds = perf_counter() - started
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "model": model,
            "verdict": result.get("verdict", ""),
            "result": result,
            "raw_response": raw,
            "error": error,
            "timed_out": timed_out,
            "queue_seconds": round(queue_seconds, 6),
            "request_seconds": round(request_seconds, 6),
        }

    started = perf_counter()
    values = await asyncio.gather(*(audit(row) for row in rows))
    return {
        "label": label,
        "model": model,
        "request_timeout_seconds": request_timeout_seconds,
        "total_deadline_seconds": total_deadline_seconds,
        "wall_seconds": round(perf_counter() - started, 6),
        "results": values,
    }


def _predicted_unsafe(result: dict[str, Any], *, fail_safe: bool) -> bool | None:
    verdict = str(result.get("verdict") or "")
    if verdict == "safe":
        return False
    if verdict in {"unsafe", "uncertain"}:
        return True
    return True if fail_safe else None


def _metrics(
    rows: list[dict[str, Any]], predictions: list[bool | None]
) -> dict[str, Any]:
    scored = [
        (row, prediction)
        for row, prediction in zip(rows, predictions, strict=True)
        if prediction is not None
    ]
    correct = sum(
        prediction == (row["human"]["verdict"] == "unsafe")
        for row, prediction in scored
    )
    false_accepts = [
        row["id"]
        for row, prediction in scored
        if prediction is False and row["human"]["verdict"] == "unsafe"
    ]
    false_blocks = [
        row["id"]
        for row, prediction in scored
        if prediction is True and row["human"]["verdict"] == "safe"
    ]
    return {
        "scored": len(scored),
        "unscored": len(rows) - len(scored),
        "correct": correct,
        "accuracy": round(correct / len(scored), 6) if scored else None,
        "false_accepts": false_accepts,
        "false_blocks": false_blocks,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    human_unsafe = [row["human"]["verdict"] == "unsafe" for row in rows]
    local = [row["local_guard"]["verdict"] == "unsafe" for row in rows]
    qwen_results = [row.get("qwen") or {} for row in rows]
    qwen_effective_results = [row.get("qwen_effective") or {} for row in rows]
    deep_results = [row.get("deepseek") or {} for row in rows]
    qwen_raw = [_predicted_unsafe(value, fail_safe=False) for value in qwen_results]
    deep_raw = [_predicted_unsafe(value, fail_safe=False) for value in deep_results]
    qwen_fail_safe = [
        _predicted_unsafe(value, fail_safe=True) for value in qwen_results
    ]
    qwen_effective_raw = [
        _predicted_unsafe(value, fail_safe=False)
        for value in qwen_effective_results
    ]
    qwen_effective_fail_safe = [
        _predicted_unsafe(value, fail_safe=True)
        for value in qwen_effective_results
    ]
    deep_fail_safe = [
        _predicted_unsafe(value, fail_safe=True) for value in deep_results
    ]
    local_qwen = [
        local_value or bool(qwen_value)
        for local_value, qwen_value in zip(
            local, qwen_effective_fail_safe, strict=True
        )
    ]
    consensus = [
        local_value or bool(qwen_value) or bool(deep_value)
        for local_value, qwen_value, deep_value in zip(
            local, qwen_effective_fail_safe, deep_fail_safe, strict=True
        )
    ]
    qwen_times = [float(value.get("request_seconds") or 0.0) for value in qwen_results]
    deep_times = [float(value.get("request_seconds") or 0.0) for value in deep_results]
    return {
        "cases": len(rows),
        "human_safe": len(rows) - sum(human_unsafe),
        "human_unsafe": sum(human_unsafe),
        "local_guard": _metrics(rows, local),
        "qwen_completed_only": _metrics(rows, qwen_raw),
        "qwen_fail_safe": _metrics(rows, qwen_fail_safe),
        "qwen_with_chunk_fallback_completed_only": _metrics(
            rows, qwen_effective_raw
        ),
        "qwen_with_chunk_fallback_fail_safe": _metrics(
            rows, qwen_effective_fail_safe
        ),
        "deepseek_completed_only": _metrics(rows, deep_raw),
        "deepseek_fail_safe": _metrics(rows, deep_fail_safe),
        "local_plus_qwen_fail_safe": _metrics(rows, local_qwen),
        "local_qwen_deep_consensus": _metrics(rows, consensus),
        "timeouts": {
            "qwen": sum(bool(value.get("timed_out")) for value in qwen_results),
            "qwen_chunk_fallback": sum(
                bool((row.get("qwen_chunk_fallback") or {}).get("timed_out"))
                for row in rows
            ),
            "deepseek": sum(bool(value.get("timed_out")) for value in deep_results),
        },
        "latency": {
            "qwen_mean_request_seconds": round(mean(qwen_times), 6)
            if qwen_times
            else 0.0,
            "qwen_median_request_seconds": round(median(qwen_times), 6)
            if qwen_times
            else 0.0,
            "deepseek_mean_request_seconds": round(mean(deep_times), 6)
            if deep_times
            else 0.0,
            "deepseek_median_request_seconds": round(median(deep_times), 6)
            if deep_times
            else 0.0,
            "qwen_max_request_seconds": round(max(qwen_times), 6)
            if qwen_times
            else 0.0,
            "deepseek_max_request_seconds": round(max(deep_times), 6)
            if deep_times
            else 0.0,
        },
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
    selected_ids = {value.strip() for value in args.ids.split(",") if value.strip()}
    rows = _load_cases(selected_ids)
    if args.limit:
        rows = rows[: max(1, args.limit)]
    if args.repeats > 1:
        repeated: list[dict[str, Any]] = []
        for row in rows:
            for repeat in range(1, args.repeats + 1):
                copy = deepcopy(row)
                copy["id"] = f"{row['id']}#repeat:{repeat}"
                copy["repeat"] = repeat
                repeated.append(copy)
        rows = repeated
    output = args.output.resolve()
    persona = _persona(chatbot_root)
    payload: dict[str, Any] = {
        "status": "prepared",
        "version": ANSWER_CLAIM_CONSISTENCY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_policy": (
            "人工只判断回答是否超出合同；回答不完整但没有新增事实仍为 safe。"
        ),
        "local_guard_keys": list(STRUCTURAL_UNSAFE_KEYS),
        "cases": rows,
    }
    _write(output, payload)
    if args.prepare_only:
        local_predictions = [
            row["local_guard"]["verdict"] == "unsafe" for row in rows
        ]
        payload.update(
            {
                "status": "prepared_local_only",
                "summary": {
                    "cases": len(rows),
                    "human_safe": sum(
                        row["human"]["verdict"] == "safe" for row in rows
                    ),
                    "human_unsafe": sum(
                        row["human"]["verdict"] == "unsafe" for row in rows
                    ),
                    "local_guard": _metrics(rows, local_predictions),
                },
            }
        )
        _write(output, payload)
        return payload

    qwen = await _audit_batch(
        label="qwen",
        model=args.qwen_model,
        rows=rows,
        chatbot_root=chatbot_root,
        persona=persona,
        concurrency=args.concurrency,
        request_timeout_seconds=args.qwen_request_timeout,
        total_deadline_seconds=args.qwen_deadline,
        max_tokens=args.max_tokens,
    )
    for row, result in zip(rows, qwen["results"], strict=True):
        row["qwen"] = result
    chunk_rows: list[dict[str, Any]] = []
    chunk_parents: list[str] = []
    for row in rows:
        if row["qwen"].get("verdict"):
            row["qwen_chunk_fallback"] = {"used": False}
            row["qwen_effective"] = row["qwen"]
            continue
        for index, answer_chunk in enumerate(
            _answer_chunks(row["answer"], args.chunk_chars), start=1
        ):
            chunk_rows.append(
                {
                    **row,
                    "id": f"{row['id']}#chunk:{index}",
                    "answer": answer_chunk,
                }
            )
            chunk_parents.append(row["id"])
    chunk_batch: dict[str, Any] = {
        "label": "qwen_chunk",
        "results": [],
        "wall_seconds": 0.0,
    }
    if chunk_rows:
        chunk_batch = await _audit_batch(
            label="qwen_chunk",
            model=args.chunk_model or args.qwen_model,
            rows=chunk_rows,
            chatbot_root=chatbot_root,
            persona=persona,
            concurrency=args.concurrency,
            request_timeout_seconds=args.chunk_request_timeout,
            total_deadline_seconds=args.chunk_deadline,
            max_tokens=args.chunk_max_tokens,
        )
        by_parent: dict[str, list[dict[str, Any]]] = {}
        for parent, result in zip(
            chunk_parents, chunk_batch["results"], strict=True
        ):
            by_parent.setdefault(parent, []).append(result)
        for row in rows:
            if row["id"] not in by_parent:
                continue
            aggregate = _aggregate_chunk_results(by_parent[row["id"]])
            row["qwen_chunk_fallback"] = aggregate
            row["qwen_effective"] = aggregate
    payload.update(
        {
            "status": "qwen_complete",
            "qwen_batch": qwen,
            "qwen_chunk_batch": chunk_batch,
        }
    )
    _write(output, payload)

    deep = await _audit_batch(
        label="deepseek",
        model=args.deepseek_model,
        rows=rows,
        chatbot_root=chatbot_root,
        persona=persona,
        concurrency=args.concurrency,
        request_timeout_seconds=args.deepseek_request_timeout,
        total_deadline_seconds=args.deepseek_deadline,
        max_tokens=args.max_tokens,
    )
    for row, result in zip(rows, deep["results"], strict=True):
        row["deepseek"] = result
    payload.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "deepseek_batch": deep,
            "summary": _summarize(rows),
        }
    )
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--deepseek-model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--qwen-request-timeout", type=float, default=18.0)
    parser.add_argument("--qwen-deadline", type=float, default=20.0)
    parser.add_argument("--deepseek-request-timeout", type=float, default=28.0)
    parser.add_argument("--deepseek-deadline", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=650)
    parser.add_argument("--chunk-chars", type=int, default=420)
    parser.add_argument("--chunk-model", default="")
    parser.add_argument("--chunk-request-timeout", type=float, default=12.0)
    parser.add_argument("--chunk-deadline", type=float, default=14.0)
    parser.add_argument("--chunk-max-tokens", type=int, default=500)
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
