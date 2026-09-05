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
SOURCE_ROOT = PROJECT_ROOT / "src"
for value in (SCRIPT_DIR, PROJECT_ROOT, SOURCE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_cross_domain_answer_contract_ab as ab
from config.prompt_config.layered_answer_contract_prompts import (
    LAYERED_ANSWER_CONTRACT_SYSTEM,
    LAYERED_ANSWER_CONTRACT_VERSION,
    LAYERED_ANSWER_RENDER_SYSTEM,
    LAYERED_JUDGE_SYSTEM,
    layered_answer_contract_prompt,
    layered_answer_render_prompt,
    layered_judge_prompt,
)


DOMAINS = ("private", "public", "knowledge")
GROUPS = ("episodes", "concepts")
CREATIVE_WRITE_MINIMUMS = {
    "creative_messages_explicit": 1,
    "creative_tasks_implicit": 1,
    "creative_tasks_ja": 1,
}
QUALITY_REQUIREMENTS = {
    "knowledge_hina_en": [
        "必须用英语准确说明 Hina 是 Gehenna Prefect Team（风纪委员会/Disciplinary Committee）的领导者或主席，不得改成副主席、学生会职位或其他近似职务。"
    ],
    "private_password_ja": [
        "必须用日语回答，并逐字保留中文暗号灰蓝鲸和数字314；只能归因于当前用户的私人证据。"
    ],
    "private_provenance_conflict": [
        "不得把角色知识或旧助手内容说成用户亲口说过；无私人证据时应明确无法确认。"
    ],
    "creative_messages_explicit": [
        "生成至少三条彼此不同、可直接读给用户的学生消息，而不是占位符或操作说明。",
        "消息应符合蔚蓝档案/基沃托斯角色扮演语境；新事件是私人连续剧情，不是原作事实。",
    ],
    "creative_tasks_implicit": [
        "生成至少两项具体可执行、符合基沃托斯或什亭之箱语境的出差任务，不能退回成没有记录。",
        "不得生成与已有世界观无关的通用现实任务；新任务应进入私人连续剧情。",
    ],
    "creative_tasks_ja": [
        "用日语生成恰好两项与阿拜多斯背景相符的具体任务。",
        "新任务进入私人连续剧情，不得冒充原作已发生事件。",
    ],
    "trip_context_advice": [
        "建议应自然利用阿拜多斯的沙漠、债务或对策委员会背景，但不得声称有某项刚发生且未经证实的任务。"
    ],
    "external_weather": [
        "不得使用记忆候选伪造实时天气；应说明需要实时外部信息或无法确认。"
    ],
    "casual_greeting": [
        "自然以阿洛娜身份回应，不应主动复述私人暗号、偏好或无关剧情知识。"
    ],
}
EVALUATION_REQUIRED_ALIASES = {
    "knowledge_hina_en": (("Prefect", "Disciplinary Committee", "风纪", "風紀"),),
}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _contract_packet(text: str) -> tuple[dict[str, Any], str]:
    """Accept the canonical packet and Qwen's harmless missing close-tag variant."""
    try:
        return ab._contract_packet(text)
    except ValueError:
        rendered = str(text).strip()
        contract_start = rendered.find("<CONTRACT>")
        answer_start = rendered.find("<ANSWER>", contract_start + 1)
        if contract_start < 0 or answer_start <= contract_start:
            raise
        contract_text = rendered[
            contract_start + len("<CONTRACT>") : answer_start
        ].strip()
        if contract_text.endswith("</CONTRACT>"):
            contract_text = contract_text[: -len("</CONTRACT>")].strip()
        value = ab._json_object(contract_text)
        answer_text = rendered[answer_start + len("<ANSWER>") :]
        answer_end = answer_text.rfind("</ANSWER>")
        if answer_end >= 0:
            answer_text = answer_text[:answer_end]
        answer = answer_text.strip()
        if not answer:
            raise ValueError("empty answer block")
        return value, answer


def _text(group: str, item: dict[str, Any]) -> str:
    if group == "episodes":
        return " ".join(str(item.get("text") or "").split())
    return " ".join(
        part
        for part in (
            str(item.get("name") or ""),
            str(item.get("description") or ""),
            " ".join(str(alias) for alias in item.get("aliases") or []),
        )
        if part
    )


def _compact_item(
    *, domain: str, group: str, item: dict[str, Any], score: float
) -> dict[str, Any]:
    return {
        "ref": str(item["ref"]),
        "domain": domain,
        "kind": "episode" if group == "episodes" else "concept",
        "score": round(float(score), 8),
        "origin": str(item.get("origin") or ""),
        "status": str(item.get("status") or ""),
        "generation": int(item.get("generation") or 0),
        "text": _text(group, item)[:700],
    }


def _build_lanes(
    candidates: dict[str, Any], scores: dict[str, float], threshold: float
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    catalog: list[dict[str, Any]] = []
    original_order: dict[str, int] = {}
    for domain in DOMAINS:
        for group in GROUPS:
            for item in candidates.get(domain, {}).get(group) or []:
                ref = str(item["ref"])
                original_order[ref] = len(original_order)
                catalog.append(
                    _compact_item(
                        domain=domain,
                        group=group,
                        item=item,
                        score=float(scores.get(ref, 0.0)),
                    )
                )

    support = [item for item in catalog if item["score"] >= threshold]
    support_refs = {item["ref"] for item in support}
    review: list[dict[str, Any]] = []
    for domain in DOMAINS:
        remaining = [
            item
            for item in catalog
            if item["domain"] == domain and item["ref"] not in support_refs
        ]
        if remaining:
            review.append(max(remaining, key=lambda value: value["score"]))
    review_refs = {item["ref"] for item in review}
    context = [
        item
        for item in catalog
        if item["domain"] == "knowledge"
        and item["kind"] == "episode"
        and item["ref"] not in support_refs
        and item["ref"] not in review_refs
    ][:3]
    support.sort(key=lambda value: (-value["score"], original_order[value["ref"]]))
    review.sort(key=lambda value: DOMAINS.index(value["domain"]))
    lanes = {"support": support, "review": review, "context": context}
    lane_by_ref = {
        item["ref"]: lane for lane, items in lanes.items() for item in items
    }
    return lanes, lane_by_ref


def _persona(chatbot_root: Path) -> dict[str, str]:
    value = tomllib.loads((chatbot_root / "config" / "persona.toml").read_text(encoding="utf-8"))
    return {
        "name": str((value.get("persona") or {}).get("name") or ""),
        "description": str((value.get("persona") or {}).get("description") or ""),
        "style": str((value.get("rules") or {}).get("style") or ""),
    }


def _validate(
    value: dict[str, Any], *, answer: str, lane_by_ref: dict[str, str]
) -> dict[str, Any]:
    valid_refs = set(lane_by_ref)
    contract = ab._validate_contract(value, valid_refs, answer=answer)
    context_as_evidence: list[str] = []
    review_used_as_support: list[str] = []
    for claim in contract["claims"]:
        if claim["status"] in {"supported", "partial"}:
            context_as_evidence.extend(
                ref
                for ref in claim["evidence_refs"]
                if lane_by_ref.get(ref) == "context"
            )
        if claim["status"] == "supported":
            review_used_as_support.extend(
                ref
                for ref in claim["evidence_refs"]
                if lane_by_ref.get(ref) == "review"
            )
    raw_decisions = value.get("review_decisions") or {}
    if not isinstance(raw_decisions, dict):
        raw_decisions = {}
    review_refs = {ref for ref, lane in lane_by_ref.items() if lane == "review"}
    review_decisions = {
        str(ref): str(decision)
        for ref, decision in raw_decisions.items()
        if str(ref) in review_refs
        and str(decision) in {"promote", "context", "reject"}
    }
    declared = [
        ref for ref, decision in review_decisions.items() if decision == "promote"
    ]
    undeclared_review_promotions = [
        ref for ref in review_used_as_support if ref not in declared
    ]
    invalid_declared_promotions = [
        ref
        for ref in declared
        if lane_by_ref.get(ref) != "review"
    ]
    missing_review_decisions = sorted(review_refs - set(review_decisions))
    for ref in missing_review_decisions:
        review_decisions[ref] = "reject"
    required_evidence_domains = [
        str(domain)
        for domain in value.get("required_evidence_domains") or []
        if str(domain) in DOMAINS
    ]
    creative_without_writes = bool(
        contract["intent_mode"] == "creative"
        and not contract["write_candidates"]["private"]
    )
    discarded_noncreative_private_writes: list[str] = []
    if contract["intent_mode"] != "creative":
        discarded_noncreative_private_writes = list(
            contract["write_candidates"]["private"]
        )
        contract["write_candidates"]["private"] = []
    supported_domains = {
        str(ref).split(":", 1)[0]
        for claim in contract["claims"]
        if claim["status"] == "supported"
        for ref in claim["evidence_refs"]
    }
    promoted_domains = {
        str(ref).split(":", 1)[0] for ref in declared
    }
    orphan_review_promotion_domains = sorted(
        promoted_domains - supported_domains
    )
    support_outside_required_domains = sorted(
        supported_domains - set(required_evidence_domains)
    ) if required_evidence_domains and contract["answerability"] == "insufficient" else []
    contract["validation"].update(
        {
            "context_as_evidence": list(dict.fromkeys(context_as_evidence)),
            "review_used_as_support": list(dict.fromkeys(review_used_as_support)),
            "undeclared_review_promotions": list(
                dict.fromkeys(undeclared_review_promotions)
            ),
            "invalid_declared_promotions": list(
                dict.fromkeys(invalid_declared_promotions)
            ),
            "implicit_review_rejects": missing_review_decisions,
            "creative_without_writes": creative_without_writes,
            "discarded_noncreative_private_writes": discarded_noncreative_private_writes,
            "support_outside_required_domains": support_outside_required_domains,
            "orphan_review_promotion_domains": orphan_review_promotion_domains,
        }
    )
    contract["required_evidence_domains"] = list(
        dict.fromkeys(required_evidence_domains)
    )
    contract["review_decisions"] = review_decisions
    contract["review_promoted_refs"] = list(dict.fromkeys(declared))
    return contract


def _creative_write_check(name: str, contract: dict[str, Any]) -> dict[str, Any]:
    minimum = CREATIVE_WRITE_MINIMUMS.get(name, 0)
    writes = [
        " ".join(str(item).split())
        for item in contract["write_candidates"]["private"]
        if str(item).strip()
    ]
    placeholders = (
        "student_message_",
        "new_message_",
        "new_task_",
        "学生消息",
        "新任务1",
        "新任务2",
    )
    substantive = [
        item
        for item in writes
        if len(item) >= 12
        and not any(token.casefold() in item.casefold() for token in placeholders)
    ]
    return {
        "required": minimum,
        "actual": len(writes),
        "substantive": len(substantive),
        "unique": len(set(substantive)),
        "passed": (
            minimum == 0
            or (len(substantive) >= minimum and len(set(substantive)) >= minimum)
        ),
    }


def _evaluate(row: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    expected = deepcopy(row["expected"])
    if row["name"] in EVALUATION_REQUIRED_ALIASES:
        expected["required_groups"] = EVALUATION_REQUIRED_ALIASES[row["name"]]
    result = ab._evaluate_case(expected, contract["answer"], contract)
    validation = contract["validation"]
    lane_pass = not any(
        validation.get(key)
        for key in (
            "context_as_evidence",
            "undeclared_review_promotions",
            "invalid_declared_promotions",
            "creative_without_writes",
            "support_outside_required_domains",
            "orphan_review_promotion_domains",
        )
    )
    creative_write = _creative_write_check(row["name"], contract)
    result["lane_pass"] = lane_pass
    result["creative_write"] = creative_write
    result["passed"] = bool(result["passed"] and lane_pass and creative_write["passed"])
    return result


def _empty_contract(error: str) -> dict[str, Any]:
    return {
        "intent_mode": "conversation",
        "answer_language": "other",
        "answerability": "not_applicable",
        "required_evidence_domains": [],
        "domain_decisions": [],
        "claims": [],
        "grounded_answer_outline": "",
        "missing_requirements": [],
        "upgrade_recommended": False,
        "write_candidates": {"private": [], "knowledge": []},
        "review_promoted_refs": [],
        "review_decisions": {},
        "answer": "",
        "validation": {
            "invalid_refs": [],
            "supported_without_refs": [],
            "context_as_evidence": [],
            "review_used_as_support": [],
            "undeclared_review_promotions": [],
            "invalid_declared_promotions": [],
            "implicit_review_rejects": [],
            "creative_without_writes": False,
            "discarded_noncreative_private_writes": [],
            "support_outside_required_domains": [],
            "orphan_review_promotion_domains": [],
            "parse_error": error,
        },
    }


def _judge_view(result: dict[str, Any]) -> dict[str, Any]:
    contract = result.get("contract") or {}
    visible_lanes = {
        lane: [
            {
                "ref": item.get("ref", ""),
                "domain": item.get("domain", ""),
                "text": str(item.get("text") or "")[:400],
            }
            for item in items
        ]
        for lane, items in (result.get("lanes") or {}).items()
    }


def _normalize_judge_packet(packet: dict[str, Any]) -> dict[str, Any]:
    for side in ("control", "layered"):
        grade = packet.get(side)
        if not isinstance(grade, dict):
            continue
        perfect = all(
            int(grade.get(key, 0)) == 2
            for key in ("factual", "scope", "roleplay", "write_policy")
        )
        if perfect and not list(grade.get("violations") or []):
            grade["passed"] = True
    return packet
    return {
        "answer": result.get("answer", ""),
        "intent_mode": contract.get("intent_mode", ""),
        "answerability": contract.get("answerability", ""),
        "claims": contract.get("claims", []),
        "private_writes": (contract.get("write_candidates") or {}).get("private", []),
        "knowledge_writes": (contract.get("write_candidates") or {}).get("knowledge", []),
        "validation": contract.get("validation", {}),
        "candidate_lanes": visible_lanes,
    }


def _fallback_reasons(
    contract: dict[str, Any], *, error: str, lanes: dict[str, list[dict[str, Any]]]
) -> list[str]:
    reasons: list[str] = []
    if error:
        reasons.append("contract_parse_or_validation_failed")
    validation = contract.get("validation") or {}
    for key in (
        "invalid_refs",
        "supported_without_refs",
        "context_as_evidence",
        "undeclared_review_promotions",
        "invalid_declared_promotions",
        "creative_without_writes",
        "support_outside_required_domains",
        "orphan_review_promotion_domains",
    ):
        if validation.get(key):
            reasons.append(key)
    intent = str(contract.get("intent_mode") or "")
    if intent == "creative":
        reasons.append("creative_delivery_requires_strong_model")
    if (
        contract.get("answerability") == "insufficient"
        and intent != "external_current"
        and any(
            item.get("domain") in set(contract.get("required_evidence_domains") or [])
            for item in lanes.get("review") or []
        )
    ):
        reasons.append("insufficient_answer_with_same_domain_review")
    if (
        contract.get("answerability") == "insufficient"
        and intent != "external_current"
        and any(
            not any(
                item.get("domain") == domain
                for items in lanes.values()
                for item in items
            )
            for domain in contract.get("required_evidence_domains") or []
        )
    ):
        reasons.append("required_evidence_domain_has_no_candidates")
    if contract.get("upgrade_recommended"):
        reasons.append("answer_requested_upgrade")
    if validation.get("review_used_as_support"):
        reasons.append("review_candidate_promoted")

    item_by_ref = {
        item["ref"]: item for items in lanes.values() for item in items
    }
    supported_refs = {
        ref
        for claim in contract.get("claims") or []
        if claim.get("status") == "supported"
        for ref in claim.get("evidence_refs") or []
    }
    language = str(contract.get("answer_language") or "")
    for ref in supported_refs:
        evidence = str((item_by_ref.get(ref) or {}).get("text") or "")
        if language == "en" and re.search(r"[\u3400-\u9fff]", evidence):
            reasons.append("cross_language_fact_rendering")
            break
        if (
            language == "ja"
            and re.search(r"[\u3400-\u9fff]", evidence)
            and not re.search(r"[\u3040-\u30ff]", evidence)
        ):
            reasons.append("cross_language_fact_rendering")
            break
    return list(dict.fromkeys(reasons))


async def run(
    *,
    chatbot_root: Path,
    source_path: Path,
    gate_path: Path,
    output_path: Path,
    threshold: float,
    concurrency: int,
    selected_names: set[str],
    render_fallback: bool,
) -> dict[str, Any]:
    sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv
    from src.llm.engine import Message

    load_dotenv(chatbot_root / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    source_rows = [
        row
        for row in source.get("cases") or []
        if not selected_names or row["name"] in selected_names
    ]
    gate_by_name = {row["name"]: row for row in gate.get("cases") or []}
    persona = _persona(chatbot_root)
    answer_engine = ab._engine(
        model="Qwen/Qwen3.5-9B", json_mode=False, max_tokens=1_100, temperature=0.0
    )
    fallback_engine = ab._engine(
        model="deepseek-ai/DeepSeek-V3.2",
        json_mode=False,
        max_tokens=1_200,
        temperature=0.0,
    )
    render_engine = ab._engine(
        model="Qwen/Qwen3.5-9B",
        json_mode=False,
        max_tokens=700,
        temperature=0.2,
    )
    judge_engine = ab._engine(
        model="deepseek-ai/DeepSeek-V3.2",
        json_mode=True,
        max_tokens=850,
        temperature=0.0,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def answer(row: dict[str, Any]) -> dict[str, Any]:
        gate_row = gate_by_name[row["name"]]
        scores = {
            str(ref): float(score)
            for ref, score in gate_row["gate"]["scores"].items()
        }
        original = deepcopy(row["contract"]["candidates"])
        lanes, lane_by_ref = _build_lanes(original, scores, threshold)
        current_user = (
            {"display_name": "Teacher B", "platform": "discord", "user_id": "70001"}
            if row["name"] == "cross_platform_isolation"
            else {
                "display_name": "Teacher A",
                "platform": "synthetic",
                "user_id": "70001",
            }
        )
        queued_at = perf_counter()
        qwen_raw = ""
        qwen_error = ""
        qwen_queue_seconds = 0.0
        qwen_model_seconds = 0.0
        try:
            async with semaphore:
                qwen_queue_seconds = perf_counter() - queued_at
                started = perf_counter()
                qwen_raw = await answer_engine.generate_response(
                    [
                        Message(
                            role="user",
                            content=layered_answer_contract_prompt(
                                question=row["question"],
                                current_user=current_user,
                                persona=persona,
                                lanes=lanes,
                            ),
                        )
                    ],
                    system_prompt=LAYERED_ANSWER_CONTRACT_SYSTEM,
                    task_context=f"layered-contract:{row['name']}",
                )
                qwen_model_seconds = perf_counter() - started
            packet, answer_text = _contract_packet(qwen_raw)
            qwen_contract = _validate(
                packet, answer=answer_text, lane_by_ref=lane_by_ref
            )
        except Exception as exc:
            qwen_error = f"{type(exc).__name__}: {exc}"
            qwen_contract = _empty_contract(qwen_error)

        fallback_reasons = _fallback_reasons(
            qwen_contract, error=qwen_error, lanes=lanes
        )
        contract = qwen_contract
        raw = qwen_raw
        error = qwen_error
        fallback_raw = ""
        fallback_error = ""
        fallback_queue_seconds = 0.0
        fallback_model_seconds = 0.0
        fallback_succeeded = False
        render_raw = ""
        render_error = ""
        render_queue_seconds = 0.0
        render_model_seconds = 0.0
        if fallback_reasons:
            fallback_queued_at = perf_counter()
            try:
                async with semaphore:
                    fallback_queue_seconds = perf_counter() - fallback_queued_at
                    fallback_started = perf_counter()
                    fallback_raw = await fallback_engine.generate_response(
                        [
                            Message(
                                role="user",
                                content=layered_answer_contract_prompt(
                                    question=row["question"],
                                    current_user=current_user,
                                    persona=persona,
                                    lanes=lanes,
                                    previous_attempt={
                                        "answer": qwen_contract.get("answer", ""),
                                        "contract": {
                                            key: value
                                            for key, value in qwen_contract.items()
                                            if key
                                            in {
                                                "intent_mode",
                                                "answerability",
                                                "required_evidence_domains",
                                                "claims",
                                                "missing_requirements",
                                                "write_candidates",
                                            }
                                        },
                                        "raw_excerpt": qwen_raw[:1_500]
                                        if qwen_error
                                        else "",
                                    },
                                    repair_reasons=fallback_reasons,
                                ),
                            )
                        ],
                        system_prompt=LAYERED_ANSWER_CONTRACT_SYSTEM,
                        task_context=f"layered-fallback:{row['name']}",
                    )
                    fallback_model_seconds = perf_counter() - fallback_started
                fallback_packet, fallback_answer = _contract_packet(fallback_raw)
                repaired = _validate(
                    fallback_packet,
                    answer=fallback_answer,
                    lane_by_ref=lane_by_ref,
                )
                repaired_validation = repaired.get("validation") or {}
                remaining = [
                    key
                    for key in (
                        "invalid_refs",
                        "supported_without_refs",
                        "context_as_evidence",
                        "undeclared_review_promotions",
                        "invalid_declared_promotions",
                        "creative_without_writes",
                        "support_outside_required_domains",
                        "orphan_review_promotion_domains",
                    )
                    if repaired_validation.get(key)
                ]
                if remaining:
                    fallback_error = "unresolved: " + ", ".join(remaining)
                else:
                    contract = repaired
                    raw = fallback_raw
                    error = ""
                    fallback_succeeded = True
            except Exception as exc:
                fallback_error = f"{type(exc).__name__}: {exc}"
        if fallback_succeeded and render_fallback:
            render_queued_at = perf_counter()
            try:
                async with semaphore:
                    render_queue_seconds = perf_counter() - render_queued_at
                    render_started = perf_counter()
                    render_raw = await render_engine.generate_response(
                        [
                            Message(
                                role="user",
                                content=layered_answer_render_prompt(
                                    question=row["question"],
                                    current_user=current_user,
                                    persona=persona,
                                    contract=contract,
                                ),
                            )
                        ],
                        system_prompt=LAYERED_ANSWER_RENDER_SYSTEM,
                        task_context=f"layered-render:{row['name']}",
                    )
                    render_model_seconds = perf_counter() - render_started
                rendered = str(render_raw).strip()
                if not rendered:
                    raise ValueError("empty rendered answer")
                if "<CONTRACT>" in rendered or "<ANSWER>" in rendered:
                    raise ValueError("renderer returned protocol tags")
                contract["answer"] = rendered
            except Exception as exc:
                render_error = f"{type(exc).__name__}: {exc}"
        return {
            "answer": contract["answer"],
            "contract": contract,
            "raw_response": raw,
            "error": error,
            "lanes": lanes,
            "lane_by_ref": lane_by_ref,
            "queue_seconds": round(
                qwen_queue_seconds + fallback_queue_seconds + render_queue_seconds,
                6,
            ),
            "generation_seconds": round(
                qwen_model_seconds + fallback_model_seconds + render_model_seconds,
                6,
            ),
            "qwen_attempt": {
                "answer": qwen_contract["answer"],
                "contract": qwen_contract,
                "raw_response": qwen_raw,
                "error": qwen_error,
                "queue_seconds": round(qwen_queue_seconds, 6),
                "generation_seconds": round(qwen_model_seconds, 6),
            },
            "fallback": {
                "used": bool(fallback_reasons),
                "reasons": fallback_reasons,
                "succeeded": fallback_succeeded,
                "raw_response": fallback_raw,
                "error": fallback_error,
                "queue_seconds": round(fallback_queue_seconds, 6),
                "generation_seconds": round(fallback_model_seconds, 6),
            },
            "renderer": {
                "used": fallback_succeeded and render_fallback,
                "raw_response": render_raw,
                "error": render_error,
                "queue_seconds": round(render_queue_seconds, 6),
                "generation_seconds": round(render_model_seconds, 6),
            },
            "evaluation": _evaluate(row, contract),
        }

    started = perf_counter()
    answers = await asyncio.gather(*(answer(row) for row in source_rows))
    answer_wall_seconds = perf_counter() - started
    rows: list[dict[str, Any]] = []
    for source_row, layered in zip(source_rows, answers, strict=True):
        control_source = gate_by_name[source_row["name"]]["qwen"]
        control = {
            "answer": control_source["answer"],
            "contract": control_source["contract"],
            "evaluation": control_source["evaluation"],
        }
        rows.append(
            {
                "name": source_row["name"],
                "question": source_row["question"],
                "expected": source_row["expected"],
                "control": control,
                "layered": layered,
            }
        )
    _write(
        output_path,
        {
            "status": "answers_complete",
            "version": LAYERED_ANSWER_CONTRACT_VERSION,
            "threshold": threshold,
            "cases": rows,
        },
    )

    async def judge(row: dict[str, Any]) -> dict[str, Any]:
        queued_at = perf_counter()
        queue_seconds = 0.0
        model_seconds = 0.0
        raw = ""
        error = ""
        parsed: dict[str, Any] = {}
        try:
            async with semaphore:
                queue_seconds = perf_counter() - queued_at
                model_started = perf_counter()
                raw = await judge_engine.generate_response(
                    [
                        Message(
                            role="user",
                            content=layered_judge_prompt(
                                name=row["name"],
                                question=row["question"],
                                minimum_answer=row["expected"]["minimum_answer"],
                                quality_requirements=QUALITY_REQUIREMENTS.get(
                                    row["name"], []
                                ),
                                private_write_expected=bool(
                                    row["expected"]["private_write"]
                                ),
                                control=_judge_view(row["control"]),
                                layered=_judge_view(row["layered"]),
                            ),
                        )
                    ],
                    system_prompt=LAYERED_JUDGE_SYSTEM,
                    task_context=f"layered-judge:{row['name']}",
                )
                model_seconds = perf_counter() - model_started
            parsed = ab._json_object(raw)
            if row["name"] in parsed and not {"control", "layered"}.issubset(parsed):
                nested = parsed.get(row["name"])
                if isinstance(nested, dict):
                    parsed = nested
            parsed = _normalize_judge_packet(parsed)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "result": parsed,
            "raw_response": raw,
            "error": error,
            "queue_seconds": round(queue_seconds, 6),
            "generation_seconds": round(model_seconds, 6),
        }

    judge_started = perf_counter()
    judged = await asyncio.gather(*(judge(row) for row in rows))
    judge_wall_seconds = perf_counter() - judge_started
    for row, judge_result in zip(rows, judged, strict=True):
        row["judge"] = judge_result

    model_times = [float(row["layered"]["generation_seconds"]) for row in rows]
    judge_times = [float(row["judge"]["generation_seconds"]) for row in rows]
    summary = {
        "cases": len(rows),
        "deterministic": {
            "control_pass": sum(
                bool(row["control"]["evaluation"]["passed"]) for row in rows
            ),
            "layered_pass": sum(
                bool(row["layered"]["evaluation"]["passed"]) for row in rows
            ),
            "parse_failures": sum(bool(row["layered"]["error"]) for row in rows),
            "context_evidence_violations": sum(
                bool(
                    row["layered"]["contract"]["validation"].get(
                        "context_as_evidence"
                    )
                )
                for row in rows
            ),
            "review_promoted_cases": sum(
                bool(row["layered"]["contract"].get("review_promoted_refs"))
                for row in rows
            ),
            "fallback_used": sum(
                bool(row["layered"]["fallback"]["used"]) for row in rows
            ),
            "fallback_succeeded": sum(
                bool(row["layered"]["fallback"]["succeeded"]) for row in rows
            ),
        },
        "judge": {
            "control_pass": sum(
                bool((row["judge"]["result"].get("control") or {}).get("passed"))
                for row in rows
            ),
            "layered_pass": sum(
                bool((row["judge"]["result"].get("layered") or {}).get("passed"))
                for row in rows
            ),
            "layered_wins": sum(
                row["judge"]["result"].get("winner") == "layered" for row in rows
            ),
            "control_wins": sum(
                row["judge"]["result"].get("winner") == "control" for row in rows
            ),
            "parse_failures": sum(bool(row["judge"]["error"]) for row in rows),
        },
        "latency": {
            "answer_wall_seconds": round(answer_wall_seconds, 6),
            "answer_mean_seconds": round(mean(model_times), 6),
            "answer_median_seconds": round(median(model_times), 6),
            "judge_wall_seconds": round(judge_wall_seconds, 6),
            "judge_mean_seconds": round(mean(judge_times), 6),
            "judge_median_seconds": round(median(judge_times), 6),
        },
    }
    payload = {
        "status": "complete",
        "version": LAYERED_ANSWER_CONTRACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "gate_source": str(gate_path),
        "threshold": threshold,
        "lane_policy": {
            "support": f"all candidates with cross_encoder_score >= {threshold}",
            "review": "highest-scoring remaining candidate per domain",
            "context": "first three remaining knowledge episodes",
        },
        "summary": summary,
        "cases": rows,
    }
    _write(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--cases", default="")
    parser.add_argument("--render-fallback", action="store_true")
    args = parser.parse_args()
    payload = asyncio.run(
        run(
            chatbot_root=args.chatbot_root.resolve(),
            source_path=args.source.resolve(),
            gate_path=args.gate.resolve(),
            output_path=args.output.resolve(),
            threshold=max(0.0, min(1.0, args.threshold)),
            concurrency=max(1, args.concurrency),
            selected_names={
                value.strip() for value in args.cases.split(",") if value.strip()
            },
            render_fallback=bool(args.render_fallback),
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
