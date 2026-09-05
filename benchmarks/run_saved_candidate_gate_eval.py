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
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOURCE_ROOT = PROJECT_ROOT / "src"
for value in (SCRIPT_DIR, PROJECT_ROOT, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_cross_domain_answer_contract_ab as ab
from memory_demo.config import AppConfig
from memory_demo.llm.client import ModelClient


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _document_text(group: str, item: dict[str, Any]) -> str:
    if group == "episodes":
        return str(item.get("text") or "")
    return "\n".join(
        value
        for value in (
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            " ".join(str(value) for value in item.get("aliases") or []),
        )
        if value
    )


async def run(
    *,
    chatbot_root: Path,
    source_path: Path,
    output_path: Path,
    threshold: float,
    concurrency: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv
    from src.llm.engine import Message

    load_dotenv(chatbot_root / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    config = AppConfig.from_env(chatbot_root / ".env")
    reranker = ModelClient(config.model)
    qwen = ab._engine(
        model="Qwen/Qwen3.5-9B", json_mode=False, max_tokens=900, temperature=0.0
    )
    deepseek = ab._engine(
        model="deepseek-ai/DeepSeek-V3.2",
        json_mode=False,
        max_tokens=1_000,
        temperature=0.0,
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = list(source.get("cases") or [])
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def gate(row: dict[str, Any]) -> dict[str, Any]:
        candidates = deepcopy(row["contract"]["candidates"])
        catalog: list[tuple[str, str, dict[str, Any], str]] = []
        for domain, domain_value in candidates.items():
            for group in ("episodes", "concepts"):
                for item in domain_value.get(group) or []:
                    text = _document_text(group, item)
                    if text:
                        catalog.append((domain, group, item, text))
        queued_at = perf_counter()
        queue_seconds = 0.0
        model_seconds = 0.0
        if catalog:
            async with semaphore:
                queue_seconds = perf_counter() - queued_at
                started = perf_counter()
                ranked = await asyncio.to_thread(
                    reranker.rerank,
                    row["question"],
                    [entry[3] for entry in catalog],
                    top_n=len(catalog),
                )
                model_seconds = perf_counter() - started
        else:
            ranked = []
        score_by_ref = {
            str(catalog[int(item["index"])][2]["ref"]): float(
                item["relevance_score"]
            )
            for item in ranked
        }
        selected_refs = {
            ref for ref, score in score_by_ref.items() if score >= threshold
        }
        domain_trace: dict[str, Any] = {}
        for domain, domain_value in candidates.items():
            before = sum(
                len(domain_value.get(group) or [])
                for group in ("episodes", "concepts")
            )
            for group in ("episodes", "concepts"):
                selected = []
                for item in domain_value.get(group) or []:
                    ref = str(item["ref"])
                    if ref in selected_refs:
                        enriched = dict(item)
                        enriched["reranker_score"] = round(score_by_ref[ref], 8)
                        selected.append(enriched)
                domain_value[group] = selected
            domain_scores = [
                score_by_ref[ref]
                for ref in selected_refs
                if ref.startswith(domain + ":")
            ]
            domain_value["top_score"] = max(domain_scores, default=0.0)
            domain_trace[domain] = {
                "before": before,
                "after": sum(
                    len(domain_value.get(group) or [])
                    for group in ("episodes", "concepts")
                ),
                "raw_top_score": max(
                    (
                        score
                        for ref, score in score_by_ref.items()
                        if ref.startswith(domain + ":")
                    ),
                    default=0.0,
                ),
            }
        return {
            "candidates": candidates,
            "valid_refs": sorted(selected_refs),
            "scores": score_by_ref,
            "domains": domain_trace,
            "seconds": round(model_seconds, 6),
            "queue_seconds": round(queue_seconds, 6),
        }

    gate_started = perf_counter()
    gated = await asyncio.gather(*(gate(row) for row in source_rows))
    gate_wall_seconds = perf_counter() - gate_started
    _write(
        output_path,
        {
            "status": "gated",
            "source": str(source_path),
            "threshold": threshold,
            "gate_wall_seconds": round(gate_wall_seconds, 6),
            "gates": [
                {"name": row["name"], "gate": gate_value}
                for row, gate_value in zip(source_rows, gated, strict=True)
            ],
        },
    )

    async def answer(
        row: dict[str, Any], gate_value: dict[str, Any], model_name: str, engine: Any
    ) -> dict[str, Any]:
        queued_at = perf_counter()
        queue_seconds = 0.0
        model_seconds = 0.0
        raw = ""
        error = ""
        try:
            async with semaphore:
                queue_seconds = perf_counter() - queued_at
                started = perf_counter()
                raw = await engine.generate_response(
                    [
                        Message(
                            role="user",
                            content=ab.answer_evidence_contract_prompt(
                                question=row["question"],
                                candidates=gate_value["candidates"],
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
                    task_context=f"candidate-gate:{model_name}:{row['name']}",
                )
                model_seconds = perf_counter() - started
            packet, answer_text = ab._contract_packet(raw)
            contract = ab._validate_contract(
                packet, gate_value["valid_refs"], answer=answer_text
            )
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
        return {
            "answer": contract["answer"],
            "contract": contract,
            "raw_response": raw,
            "error": error,
            "generation_seconds": round(model_seconds, 6),
            "queue_seconds": round(queue_seconds, 6),
            "evaluation": ab._evaluate_case(
                row["expected"], contract["answer"], contract
            ),
        }

    answer_started = perf_counter()
    task_specs = [
        (row, gate_value, model_name, engine)
        for row, gate_value in zip(source_rows, gated, strict=True)
        for model_name, engine in (("qwen", qwen), ("deepseek", deepseek))
    ]
    outputs = await asyncio.gather(
        *(answer(row, gate_value, model_name, engine) for row, gate_value, model_name, engine in task_specs)
    )
    answer_wall_seconds = perf_counter() - answer_started
    result_rows = []
    for index, (row, gate_value) in enumerate(zip(source_rows, gated, strict=True)):
        result_rows.append(
            {
                "name": row["name"],
                "question": row["question"],
                "expected": row["expected"],
                "gate": gate_value,
                "qwen": outputs[index * 2],
                "deepseek": outputs[index * 2 + 1],
            }
        )

    qwen_times = [float(row["qwen"]["generation_seconds"]) for row in result_rows]
    deep_times = [float(row["deepseek"]["generation_seconds"]) for row in result_rows]
    payload = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "threshold": threshold,
        "reranker_model": config.model.reranker_model,
        "summary": {
            "cases": len(result_rows),
            "gate_wall_seconds": round(gate_wall_seconds, 6),
            "gate_mean_seconds": round(mean(float(value["seconds"]) for value in gated), 6),
            "gate_zero_candidate_cases": sum(
                not value["valid_refs"] for value in gated
            ),
            "qwen_pass": sum(row["qwen"]["evaluation"]["passed"] for row in result_rows),
            "deepseek_pass": sum(row["deepseek"]["evaluation"]["passed"] for row in result_rows),
            "qwen_parse_failures": sum(bool(row["qwen"]["error"]) for row in result_rows),
            "deepseek_parse_failures": sum(bool(row["deepseek"]["error"]) for row in result_rows),
            "qwen_generation_mean_seconds": round(mean(qwen_times), 6),
            "qwen_generation_median_seconds": round(median(qwen_times), 6),
            "deepseek_generation_mean_seconds": round(mean(deep_times), 6),
            "deepseek_generation_median_seconds": round(median(deep_times), 6),
            "answer_wall_seconds": round(answer_wall_seconds, 6),
        },
        "cases": result_rows,
    }
    _write(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    payload = asyncio.run(
        run(
            chatbot_root=args.chatbot_root.resolve(),
            source_path=args.source.resolve(),
            output_path=args.output.resolve(),
            threshold=max(0.0, min(1.0, args.threshold)),
            concurrency=max(1, args.concurrency),
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
