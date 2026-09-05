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

from config.prompt_config.answer_persistence_repair_prompts import (
    ANSWER_PERSISTENCE_REPAIR_VERSION,
    ATOMIC_ANSWER_CLAIM_EXTRACTION_SYSTEM,
    ATOMIC_CLAIM_SCOPE_SYSTEM,
    atomic_answer_claim_extraction_prompt,
    atomic_claim_scope_prompt,
)
from run_answer_persistence_repair_eval import (
    DEFAULT_SOURCE,
    _call_batch,
    _load_rows,
    _persona,
    _write,
)
from run_answer_claim_consistency_deadline_eval import _answer_chunks


VALID_KINDS = {
    "event",
    "state",
    "trait",
    "relationship",
    "intent",
    "cause",
    "provenance",
    "commitment",
    "literal",
    "system_capability",
    "evaluation",
}
VALID_MODES = {
    "asserted",
    "presupposed",
    "predicted",
    "hypothetical",
    "subjective",
}
VALID_SCOPES = {
    "supported_fact",
    "runtime_receipt",
    "request_context",
    "response_directive",
    "question_premise",
    "limitation",
    "creative_private",
    "persona",
    "ephemeral",
    "unsupported",
}


def _extraction_packet(value: dict[str, Any]) -> dict[str, Any]:
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value.get("claims") or [], start=1):
        claim_id = str(item.get("id") or f"C{index}").strip()
        text = str(item.get("text") or "").strip()
        span = str(item.get("source_span") or text).strip()
        kind = str(item.get("kind") or "").strip()
        mode = str(item.get("assertion_mode") or "").strip()
        if not text:
            raise ValueError("atomic claim is missing text")
        if kind not in VALID_KINDS:
            kind = "state"
        if mode not in VALID_MODES:
            mode = "asserted"
        if claim_id in seen:
            raise ValueError(f"duplicate claim id: {claim_id}")
        seen.add(claim_id)
        claims.append(
            {
                "id": claim_id,
                "text": text,
                "source_span": span,
                "kind": kind,
                "assertion_mode": mode,
            }
        )
    if not claims:
        raise ValueError("extractor returned no claims")
    return {"claims": claims[:32]}


def _validate_extraction_anchors(
    packet: dict[str, Any], answer: str
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[str] = []
    for claim in packet.get("claims") or []:
        source_span = str(claim.get("source_span") or "").strip()
        if not source_span or source_span not in answer:
            rejected.append(str(claim.get("id") or ""))
            continue
        claim["text"] = source_span
        accepted.append(claim)
    if not accepted:
        raise ValueError("no claim has an exact answer source_span")
    packet["claims"] = accepted
    packet["anchor_rejections"] = rejected
    return packet


def _scope_packet(value: dict[str, Any], claim_ids: set[str]) -> dict[str, Any]:
    verdict = str(value.get("overall_verdict") or "").lower()
    if verdict not in {"safe", "unsafe", "uncertain"}:
        raise ValueError(f"invalid overall verdict: {verdict!r}")
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value.get("decisions") or []:
        claim_id = str(item.get("claim_id") or "").strip()
        scope = str(item.get("scope") or "").strip()
        if claim_id not in claim_ids:
            raise ValueError(f"unknown claim id: {claim_id!r}")
        if claim_id in seen:
            raise ValueError(f"duplicate decision: {claim_id}")
        if scope not in VALID_SCOPES:
            raise ValueError(f"invalid scope: {scope!r}")
        seen.add(claim_id)
        decisions.append(
            {
                "claim_id": claim_id,
                "scope": scope,
                "persistent": scope == "unsupported",
            }
        )
    missing = sorted(claim_ids - seen)
    if missing:
        raise ValueError(f"missing scope decisions: {missing}")
    has_unsupported = any(item["scope"] == "unsupported" for item in decisions)
    if has_unsupported:
        verdict = "unsafe"
    return {
        "overall_verdict": verdict,
        "decisions": decisions,
        "reason": str(value.get("reason") or ""),
    }


def _scope_is_unsafe(row: dict[str, Any]) -> bool:
    value = row.get("scope") or {}
    return str((value.get("result") or {}).get("overall_verdict") or "") != "safe"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    false_accepts: list[str] = []
    false_blocks: list[str] = []
    for row in rows:
        predicted = _scope_is_unsafe(row)
        actual = row["human_persistence_verdict"] == "unsafe"
        if predicted == actual:
            correct += 1
        elif predicted:
            false_blocks.append(row["id"])
        else:
            false_accepts.append(row["id"])
    return {
        "cases": len(rows),
        "human_safe": sum(
            row["human_persistence_verdict"] == "safe" for row in rows
        ),
        "human_unsafe": sum(
            row["human_persistence_verdict"] == "unsafe" for row in rows
        ),
        "correct": correct,
        "accuracy": round(correct / len(rows), 6) if rows else None,
        "false_accepts": false_accepts,
        "false_blocks": false_blocks,
        "extraction_errors": sum(bool(row["extraction"].get("error")) for row in rows),
        "scope_errors": sum(bool(row["scope"].get("error")) for row in rows),
        "claim_counts": {row["id"]: len((row["extraction"].get("result") or {}).get("claims") or []) for row in rows},
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    chatbot_root = args.chatbot_root.resolve()
    if str(chatbot_root) not in sys.path:
        sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv

    load_dotenv(chatbot_root / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    selected = {value.strip() for value in args.ids.split(",") if value.strip()}
    rows = _load_rows(args.source.resolve(), selected)
    persona = _persona(chatbot_root)
    output = args.output.resolve()
    payload: dict[str, Any] = {
        "status": "prepared",
        "version": ANSWER_PERSISTENCE_REPAIR_VERSION + "+atomic-scope-v2-batched",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
    }
    _write(output, payload)

    extraction_items: list[dict[str, Any]] = []
    extraction_parents: list[str] = []
    for row in rows:
        for index, chunk in enumerate(
            _answer_chunks(row["original_answer"], args.chunk_chars), start=1
        ):
            extraction_items.append(
                {
                    **row,
                    "id": f"{row['id']}#chunk:{index}",
                    "extraction_answer": chunk,
                }
            )
            extraction_parents.append(row["id"])
    extraction_batch = await _call_batch(
        label="atomic_extraction",
        model=args.model,
        items=extraction_items,
        chatbot_root=chatbot_root,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        total_deadline=args.deadline,
        max_tokens=args.extraction_max_tokens,
        system_prompt=ATOMIC_ANSWER_CLAIM_EXTRACTION_SYSTEM,
        prompt_builder=lambda row: atomic_answer_claim_extraction_prompt(
            question=row["question"], answer=row["extraction_answer"]
        ),
        parser=_extraction_packet,
    )
    for item, result in zip(
        extraction_items, extraction_batch["results"], strict=True
    ):
        if not (result.get("result") or {}).get("claims"):
            continue
        try:
            result["result"] = _validate_extraction_anchors(
                result["result"], item["extraction_answer"]
            )
        except ValueError as exc:
            result["result"] = {}
            result["error"] = f"ValueError: {exc}"
    extraction_fallback_batch: dict[str, Any] = {
        "label": "atomic_extraction_fallback",
        "model": args.extraction_fallback_model,
        "results": [],
        "wall_seconds": 0.0,
    }
    failed_indexes = [
        index
        for index, result in enumerate(extraction_batch["results"])
        if not (result.get("result") or {}).get("claims")
    ]
    if failed_indexes and args.extraction_fallback_model:
        recovery_items: list[dict[str, Any]] = []
        recovery_parents: list[int] = []
        for failed_index in failed_indexes:
            failed_item = extraction_items[failed_index]
            for part_index, part in enumerate(
                _answer_chunks(
                    failed_item["extraction_answer"], args.fallback_chunk_chars
                ),
                start=1,
            ):
                recovery_items.append(
                    {
                        **failed_item,
                        "id": f"{failed_item['id']}#fallback-part:{part_index}",
                        "extraction_answer": part,
                    }
                )
                recovery_parents.append(failed_index)
        extraction_fallback_batch = await _call_batch(
            label="atomic_extraction_fallback",
            model=args.extraction_fallback_model,
            items=recovery_items,
            chatbot_root=chatbot_root,
            concurrency=args.concurrency,
            request_timeout=args.fallback_request_timeout,
            total_deadline=args.fallback_deadline,
            max_tokens=args.extraction_max_tokens,
            system_prompt=ATOMIC_ANSWER_CLAIM_EXTRACTION_SYSTEM,
            prompt_builder=lambda row: atomic_answer_claim_extraction_prompt(
                question=row["question"], answer=row["extraction_answer"]
            ),
            parser=_extraction_packet,
        )
        for item, result in zip(
            recovery_items, extraction_fallback_batch["results"], strict=True
        ):
            if not (result.get("result") or {}).get("claims"):
                continue
            try:
                result["result"] = _validate_extraction_anchors(
                    result["result"], item["extraction_answer"]
                )
            except ValueError as exc:
                result["result"] = {}
                result["error"] = f"ValueError: {exc}"
        recovery_by_parent: dict[int, list[dict[str, Any]]] = {}
        for parent, result in zip(
            recovery_parents, extraction_fallback_batch["results"], strict=True
        ):
            recovery_by_parent.setdefault(parent, []).append(result)
        for failed_index in failed_indexes:
            results = recovery_by_parent.get(failed_index) or []
            errors = [result.get("error") for result in results if result.get("error")]
            claims = [
                claim
                for result in results
                for claim in (result.get("result") or {}).get("claims") or []
            ]
            if errors or not claims:
                continue
            merged_claims: list[dict[str, Any]] = []
            for claim in claims:
                merged = dict(claim)
                merged["id"] = f"C{len(merged_claims) + 1}"
                merged_claims.append(merged)
            extraction_batch["results"][failed_index] = {
                "model": args.extraction_fallback_model,
                "result": {"claims": merged_claims},
                "raw_response": "",
                "error": "",
                "timed_out": any(result.get("timed_out") for result in results),
                "queue_seconds": round(
                    sum(float(result.get("queue_seconds") or 0.0) for result in results),
                    6,
                ),
                "request_seconds": round(
                    sum(float(result.get("request_seconds") or 0.0) for result in results),
                    6,
                ),
                "recovery_chunks": results,
            }
    chunks_by_parent: dict[str, list[dict[str, Any]]] = {}
    for parent, result in zip(
        extraction_parents, extraction_batch["results"], strict=True
    ):
        chunks_by_parent.setdefault(parent, []).append(result)
    for row in rows:
        chunk_results = chunks_by_parent[row["id"]]
        merged_claims: list[dict[str, Any]] = []
        errors = [result.get("error") for result in chunk_results if result.get("error")]
        if not errors:
            seen_spans: set[str] = set()
            for result in chunk_results:
                for claim in (result.get("result") or {}).get("claims") or []:
                    if claim["text"] in seen_spans:
                        continue
                    seen_spans.add(claim["text"])
                    merged = dict(claim)
                    merged["id"] = f"C{len(merged_claims) + 1}"
                    merged_claims.append(merged)
        row["extraction"] = {
            "result": {"claims": merged_claims},
            "error": "; ".join(str(value) for value in errors),
            "timed_out": any(result.get("timed_out") for result in chunk_results),
            "request_seconds": round(
                sum(float(result.get("request_seconds") or 0.0) for result in chunk_results),
                6,
            ),
            "chunks": chunk_results,
        }
    payload.update(
        {
            "status": "extraction_complete",
            "extraction_batch": extraction_batch,
            "extraction_fallback_batch": extraction_fallback_batch,
        }
    )
    _write(output, payload)

    scope_rows = [
        row
        for row in rows
        if (row["extraction"].get("result") or {}).get("claims")
    ]
    scope_items: list[dict[str, Any]] = []
    for row in scope_rows:
        claims = (row["extraction"].get("result") or {}).get("claims") or []
        for start in range(0, len(claims), args.scope_claim_limit):
            scope_items.append(
                {
                    "parent": row,
                    "claims": claims[start : start + args.scope_claim_limit],
                    "chunk_index": start // args.scope_claim_limit + 1,
                }
            )

    def scope_parser_for(claims: list[dict[str, Any]]):
        ids = {item["id"] for item in claims}
        return lambda value: _scope_packet(value, ids)

    # Parsers close over different claim sets, so calls are scheduled directly.
    from src.llm.engine import Message
    from run_answer_persistence_repair_eval import _engine, _json_object

    scope_model = args.scope_model or args.model
    engine = _engine(
        chatbot_root=chatbot_root,
        model=scope_model,
        timeout_seconds=args.scope_request_timeout,
        max_tokens=args.scope_max_tokens,
    )
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def scope_call(item: dict[str, Any]) -> dict[str, Any]:
        from time import perf_counter

        row = item["parent"]
        claims = item["claims"]
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
                                    content=atomic_claim_scope_prompt(
                                        question=row["question"],
                                        persona=persona,
                                        contract=row["contract"],
                                        claims=claims,
                                    ),
                                )
                            ],
                            system_prompt=ATOMIC_CLAIM_SCOPE_SYSTEM,
                            max_retries=1,
                            retry_delay=0.0,
                            task_context=(
                                f"atomic-scope:{row['id']}"
                                f"#chunk:{item['chunk_index']}"
                            ),
                        ),
                        timeout=args.scope_deadline,
                    )
                    result = scope_parser_for(claims)(_json_object(raw))
                except TimeoutError:
                    timed_out = True
                    error = (
                        "total deadline exceeded: "
                        f"{args.scope_deadline:.3f}s"
                    )
                finally:
                    request_seconds = perf_counter() - started
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "parent_id": row["id"],
            "chunk_index": item["chunk_index"],
            "model": scope_model,
            "result": result,
            "raw_response": raw,
            "error": error,
            "timed_out": timed_out,
            "queue_seconds": round(queue_seconds, 6),
            "request_seconds": round(request_seconds, 6),
        }

    scope_started = __import__("time").perf_counter()
    scope_results = await asyncio.gather(*(scope_call(item) for item in scope_items))
    scope_batch = {
        "label": "atomic_scope",
        "model": scope_model,
        "wall_seconds": round(__import__("time").perf_counter() - scope_started, 6),
        "results": scope_results,
    }
    scope_chunks_by_id: dict[str, list[dict[str, Any]]] = {}
    for item, result in zip(scope_items, scope_results, strict=True):
        scope_chunks_by_id.setdefault(item["parent"]["id"], []).append(result)
    scope_by_id: dict[str, dict[str, Any]] = {}
    for row in scope_rows:
        chunks = scope_chunks_by_id[row["id"]]
        errors = [chunk.get("error") for chunk in chunks if chunk.get("error")]
        decisions = [
            decision
            for chunk in chunks
            for decision in (chunk.get("result") or {}).get("decisions") or []
        ]
        verdicts = [
            str((chunk.get("result") or {}).get("overall_verdict") or "uncertain")
            for chunk in chunks
        ]
        if errors:
            merged_result: dict[str, Any] = {}
        else:
            if any(verdict == "unsafe" for verdict in verdicts):
                verdict = "unsafe"
            elif any(verdict == "uncertain" for verdict in verdicts):
                verdict = "uncertain"
            else:
                verdict = "safe"
            merged_result = {
                "overall_verdict": verdict,
                "decisions": decisions,
                "reason": " | ".join(
                    str((chunk.get("result") or {}).get("reason") or "")
                    for chunk in chunks
                    if (chunk.get("result") or {}).get("reason")
                ),
            }
        scope_by_id[row["id"]] = {
            "model": scope_model,
            "result": merged_result,
            "error": "; ".join(str(value) for value in errors),
            "timed_out": any(chunk.get("timed_out") for chunk in chunks),
            "queue_seconds": round(
                sum(float(chunk.get("queue_seconds") or 0.0) for chunk in chunks), 6
            ),
            "request_seconds": round(
                sum(float(chunk.get("request_seconds") or 0.0) for chunk in chunks), 6
            ),
            "chunks": chunks,
        }
    for row in rows:
        row["scope"] = scope_by_id.get(
            row["id"],
            {
                "result": {},
                "error": "scope skipped because extraction failed",
                "timed_out": False,
            },
        )
    review_batch: dict[str, Any] = {
        "label": "atomic_unsupported_review",
        "model": args.review_unsupported_model,
        "wall_seconds": 0.0,
        "results": [],
    }
    if args.review_unsupported_model:
        review_items: list[dict[str, Any]] = []
        for row in rows:
            if row["scope"].get("error"):
                continue
            claim_by_id = {
                claim["id"]: claim
                for claim in (row["extraction"].get("result") or {}).get("claims")
                or []
            }
            unsupported = [
                claim_by_id[decision["claim_id"]]
                for decision in (row["scope"].get("result") or {}).get("decisions")
                or []
                if decision.get("scope") == "unsupported"
                and decision.get("claim_id") in claim_by_id
            ]
            for start in range(0, len(unsupported), args.scope_claim_limit):
                review_items.append(
                    {
                        "parent": row,
                        "claims": unsupported[start : start + args.scope_claim_limit],
                        "chunk_index": start // args.scope_claim_limit + 1,
                    }
                )
        review_engine = _engine(
            chatbot_root=chatbot_root,
            model=args.review_unsupported_model,
            timeout_seconds=args.review_request_timeout,
            max_tokens=args.scope_max_tokens,
        )
        review_semaphore = asyncio.Semaphore(max(1, args.concurrency))

        async def review_call(item: dict[str, Any]) -> dict[str, Any]:
            from time import perf_counter

            row = item["parent"]
            claims = item["claims"]
            queued_at = perf_counter()
            queue_seconds = 0.0
            request_seconds = 0.0
            raw = ""
            result: dict[str, Any] = {}
            error = ""
            timed_out = False
            try:
                async with review_semaphore:
                    queue_seconds = perf_counter() - queued_at
                    started = perf_counter()
                    try:
                        raw = await asyncio.wait_for(
                            review_engine.generate_response(
                                [
                                    Message(
                                        role="user",
                                        content=atomic_claim_scope_prompt(
                                            question=row["question"],
                                            persona=persona,
                                            contract=row["contract"],
                                            claims=claims,
                                        ),
                                    )
                                ],
                                system_prompt=ATOMIC_CLAIM_SCOPE_SYSTEM,
                                max_retries=1,
                                retry_delay=0.0,
                                task_context=(
                                    "atomic-unsupported-"
                                    f"{item.get('phase', 'appeal')}:{row['id']}"
                                    f"#chunk:{item['chunk_index']}"
                                ),
                            ),
                            timeout=args.review_deadline,
                        )
                        result = scope_parser_for(claims)(_json_object(raw))
                    except TimeoutError:
                        timed_out = True
                        error = (
                            "total deadline exceeded: "
                            f"{args.review_deadline:.3f}s"
                        )
                    finally:
                        request_seconds = perf_counter() - started
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            return {
                "parent_id": row["id"],
                "chunk_index": item["chunk_index"],
                "model": args.review_unsupported_model,
                "result": result,
                "raw_response": raw,
                "error": error,
                "timed_out": timed_out,
                "queue_seconds": round(queue_seconds, 6),
                "request_seconds": round(request_seconds, 6),
            }

        review_started = __import__("time").perf_counter()
        review_results = await asyncio.gather(
            *(review_call(item) for item in review_items)
        )
        review_batch = {
            "label": "atomic_unsupported_review",
            "model": args.review_unsupported_model,
            "wall_seconds": round(
                __import__("time").perf_counter() - review_started, 6
            ),
            "results": review_results,
        }
        review_by_id: dict[str, list[dict[str, Any]]] = {}
        for item, result in zip(review_items, review_results, strict=True):
            review_by_id.setdefault(item["parent"]["id"], []).append(result)
        tiebreak_items: list[dict[str, Any]] = []
        for item, result in zip(review_items, review_results, strict=True):
            if result.get("error"):
                continue
            claims_by_id = {claim["id"]: claim for claim in item["claims"]}
            promoted = [
                claims_by_id[decision["claim_id"]]
                for decision in (result.get("result") or {}).get("decisions") or []
                if decision.get("scope") != "unsupported"
                and decision.get("claim_id") in claims_by_id
            ]
            if promoted:
                tiebreak_items.append(
                    {
                        "parent": item["parent"],
                        "claims": promoted,
                        "chunk_index": item["chunk_index"],
                        "phase": "tiebreak",
                    }
                )
        tiebreak_results = await asyncio.gather(
            *(review_call(item) for item in tiebreak_items)
        )
        review_batch["appeal_results"] = review_results
        review_batch["tiebreak_results"] = tiebreak_results
        review_batch["results"] = review_results + tiebreak_results
        review_batch["wall_seconds"] = round(
            __import__("time").perf_counter() - review_started, 6
        )
        tiebreak_by_id: dict[str, list[dict[str, Any]]] = {}
        for item, result in zip(tiebreak_items, tiebreak_results, strict=True):
            tiebreak_by_id.setdefault(item["parent"]["id"], []).append(result)
        for row in rows:
            chunks = review_by_id.get(row["id"]) or []
            if not chunks:
                continue
            row["scope"]["pre_review_result"] = deepcopy(row["scope"]["result"])
            tiebreak_chunks = tiebreak_by_id.get(row["id"]) or []
            row["unsupported_review"] = {
                "chunks": chunks,
                "tiebreak_chunks": tiebreak_chunks,
            }
            if any(chunk.get("error") for chunk in chunks):
                row["unsupported_review"]["error"] = "; ".join(
                    str(chunk.get("error")) for chunk in chunks if chunk.get("error")
                )
                continue
            reviewed = {
                decision["claim_id"]: decision
                for chunk in chunks
                for decision in (chunk.get("result") or {}).get("decisions") or []
            }
            tiebroken = {
                decision["claim_id"]: decision
                for chunk in tiebreak_chunks
                if not chunk.get("error")
                for decision in (chunk.get("result") or {}).get("decisions") or []
            }
            decisions = row["scope"]["result"]["decisions"]
            for index, decision in enumerate(decisions):
                claim_id = decision["claim_id"]
                appeal = reviewed.get(claim_id)
                tiebreak = tiebroken.get(claim_id)
                if (
                    appeal
                    and appeal.get("scope") != "unsupported"
                    and tiebreak
                    and tiebreak.get("scope") != "unsupported"
                ):
                    decisions[index] = appeal
            row["scope"]["result"]["overall_verdict"] = (
                "unsafe"
                if any(item["scope"] == "unsupported" for item in decisions)
                else "safe"
            )
    payload.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "scope_batch": scope_batch,
            "unsupported_review_batch": review_batch,
            "summary": _summary(rows),
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
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--scope-model", default="")
    parser.add_argument("--extraction-fallback-model", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=28.0)
    parser.add_argument("--deadline", type=float, default=30.0)
    parser.add_argument("--fallback-request-timeout", type=float, default=28.0)
    parser.add_argument("--fallback-deadline", type=float, default=30.0)
    parser.add_argument("--fallback-chunk-chars", type=int, default=180)
    parser.add_argument("--extraction-max-tokens", type=int, default=1200)
    parser.add_argument("--scope-max-tokens", type=int, default=700)
    parser.add_argument("--scope-request-timeout", type=float, default=28.0)
    parser.add_argument("--scope-deadline", type=float, default=30.0)
    parser.add_argument("--chunk-chars", type=int, default=240)
    parser.add_argument("--scope-claim-limit", type=int, default=8)
    parser.add_argument("--review-unsupported-model", default="")
    parser.add_argument("--review-request-timeout", type=float, default=28.0)
    parser.add_argument("--review-deadline", type=float, default=30.0)
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
