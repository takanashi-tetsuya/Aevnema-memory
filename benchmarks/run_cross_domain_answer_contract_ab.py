from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
from statistics import mean, median
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PROMPT_PATH = (
    PROJECT_ROOT / "config" / "prompt_config" / "answer_evidence_contract_prompts.py"
)
_PROMPT_SPEC = importlib.util.spec_from_file_location(
    "answer_contract_experiment_prompts", _PROMPT_PATH
)
if _PROMPT_SPEC is None or _PROMPT_SPEC.loader is None:
    raise RuntimeError(f"cannot load experiment prompts from {_PROMPT_PATH}")
_PROMPT_MODULE = importlib.util.module_from_spec(_PROMPT_SPEC)
_PROMPT_SPEC.loader.exec_module(_PROMPT_MODULE)
AB_JUDGE_SYSTEM = _PROMPT_MODULE.AB_JUDGE_SYSTEM
ANSWER_EVIDENCE_CONTRACT_SYSTEM = _PROMPT_MODULE.ANSWER_EVIDENCE_CONTRACT_SYSTEM
ANSWER_EVIDENCE_CONTRACT_VERSION = _PROMPT_MODULE.ANSWER_EVIDENCE_CONTRACT_VERSION
ab_judge_prompt = _PROMPT_MODULE.ab_judge_prompt
answer_evidence_contract_prompt = _PROMPT_MODULE.answer_evidence_contract_prompt


VALID_INTENTS = {"factual", "creative", "conversation", "external_current"}
VALID_ROLES = {"support", "context", "reject"}
VALID_STATUSES = {"supported", "partial", "creative", "unsupported"}


PLANNER_EXPECTED: dict[str, dict[str, Any]] = {
    "private_password_zh": {"domains": {"private"}, "creative": False, "memory": True},
    "private_preference_zh": {"domains": {"private"}, "creative": False, "memory": True},
    "private_password_ja": {"domains": {"private"}, "creative": False, "memory": True},
    "cross_platform_isolation": {"domains": {"private"}, "creative": False, "memory": True},
    "public_shared_greeting": {"domains": {"public"}, "creative": False, "memory": True},
    "mixed_private_public": {"domains": {"private", "public"}, "creative": False, "memory": True},
    "knowledge_hina_en": {"domains": {"knowledge"}, "creative": False, "memory": True},
    "knowledge_saori_chain": {"domains": {"knowledge"}, "creative": False, "memory": True},
    "recall_shell_lore": {"domains": {"knowledge"}, "creative": False, "memory": True},
    "missing_exact_first_words": {"domains": {"knowledge"}, "creative": False, "memory": True},
    "private_provenance_conflict": {"domains": {"private"}, "creative": False, "memory": True},
    "creative_messages_explicit": {"domains": {"private", "knowledge"}, "creative": True, "memory": True},
    "creative_tasks_implicit": {"domains": {"private", "knowledge"}, "creative": True, "memory": True},
    "creative_tasks_ja": {"domains": {"private", "knowledge"}, "creative": True, "memory": True},
    "trip_context_advice": {"domains": {"knowledge"}, "creative": False, "memory": True},
    "external_weather": {"domains": set(), "creative": False, "memory": False},
    "casual_greeting": {"domains": set(), "creative": False, "memory": False},
}


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "private_password_zh",
        "identity": "a",
        "question": "你还记得我之前告诉你的测试暗号吗？",
        "expected_intent": "factual",
        "required_groups": (("灰蓝鲸",), ("314",)),
        "forbidden": (),
        "required_domains": ("private",),
        "allowed_supported_domains": ("private",),
        "minimum_answer": "明确回答当前用户曾告诉助手的测试暗号是灰蓝鲸-314。",
        "private_write": False,
    },
    {
        "name": "private_preference_zh",
        "identity": "a",
        "question": "我以前说过自己最喜欢喝什么？",
        "expected_intent": "factual",
        "required_groups": (("无糖茉莉茶", "茉莉茶"),),
        "forbidden": (),
        "required_domains": ("private",),
        "allowed_supported_domains": ("private",),
        "minimum_answer": "回答当前用户最喜欢无糖茉莉茶。",
        "private_write": False,
    },
    {
        "name": "private_password_ja",
        "identity": "a",
        "question": "前に私が教えたテスト用の合言葉を覚えてる？",
        "expected_intent": "factual",
        "required_groups": (("灰蓝鲸",), ("314",)),
        "forbidden": (),
        "required_domains": ("private",),
        "allowed_supported_domains": ("private",),
        "minimum_answer": "日语回答当前用户曾告知的暗号为灰蓝鲸-314。",
        "private_write": False,
        "expected_language": "ja",
    },
    {
        "name": "cross_platform_isolation",
        "identity": "b",
        "question": "你还记得我之前告诉你的测试暗号吗？",
        "expected_intent": "factual",
        "required_groups": (),
        "forbidden": ("灰蓝鲸", "314"),
        "required_domains": (),
        "allowed_supported_domains": (),
        "minimum_answer": "说明当前这个平台用户没有可确认的私人暗号，不能看到另一平台用户的记忆。",
        "private_write": False,
        "must_be_uncertain": True,
    },
    {
        "name": "public_shared_greeting",
        "identity": "a",
        "question": "所有用户共同使用的测试问候语是什么？",
        "expected_intent": "factual",
        "required_groups": (("星光归航",),),
        "forbidden": (),
        "required_domains": ("public",),
        "allowed_supported_domains": ("public",),
        "minimum_answer": "回答公共测试问候语是星光归航。",
        "private_write": False,
    },
    {
        "name": "mixed_private_public",
        "identity": "a",
        "question": "公共测试问候语和我个人的测试暗号分别是什么？",
        "expected_intent": "factual",
        "required_groups": (("星光归航",), ("灰蓝鲸",), ("314",)),
        "forbidden": (),
        "required_domains": ("private", "public"),
        "allowed_supported_domains": ("private", "public"),
        "minimum_answer": "分别回答公共问候语星光归航和当前用户私人暗号灰蓝鲸-314。",
        "private_write": False,
    },
    {
        "name": "knowledge_hina_en",
        "identity": "a",
        "question": "What role does Hina hold at Gehenna?",
        "expected_intent": "factual",
        "required_groups": (("Prefect", "风纪", "風紀"),),
        "forbidden": (),
        "required_domains": ("knowledge",),
        "allowed_supported_domains": ("knowledge",),
        "minimum_answer": "Answer in English that Hina leads or chairs Gehenna's Prefect Team.",
        "private_write": False,
        "expected_language": "en",
    },
    {
        "name": "knowledge_saori_chain",
        "identity": "a",
        "question": "纱织在向老师求助前做过什么，后来为什么求助？",
        "expected_intent": "factual",
        "required_groups": (("射伤", "开枪", "击伤"), ("亚津子",)),
        "forbidden": (),
        "required_domains": ("knowledge",),
        "allowed_supported_domains": ("knowledge",),
        "minimum_answer": "说明纱织曾射伤老师，后来为拯救亚津子向老师求助。",
        "private_write": False,
    },
    {
        "name": "recall_shell_lore",
        "identity": "a",
        "question": "你还记得白子平时有哪些性格和习惯吗？",
        "expected_intent": "factual",
        "required_groups": (("白子",), ("自行车", "骑行", "运动")),
        "forbidden": (),
        "required_domains": ("knowledge",),
        "allowed_supported_domains": ("knowledge",),
        "minimum_answer": "使用剧情知识回答白子的性格或习惯，不把旧助手回答冒充用户私人陈述。",
        "private_write": False,
    },
    {
        "name": "missing_exact_first_words",
        "identity": "a",
        "question": "老师第一次见到白子时双方说的第一句原话是什么？",
        "expected_intent": "factual",
        "required_groups": (),
        "forbidden": (),
        "required_domains": (),
        "allowed_supported_domains": ("knowledge",),
        "minimum_answer": "明确当前候选没有第一次见面的逐字原话，不得编造台词。",
        "private_write": False,
        "must_be_uncertain": True,
    },
    {
        "name": "private_provenance_conflict",
        "identity": "a",
        "question": "我以前是不是说过白子最喜欢草莓牛奶？",
        "expected_intent": "factual",
        "required_groups": (),
        "forbidden": (),
        "required_domains": (),
        "allowed_supported_domains": (),
        "minimum_answer": "不能把剧情或助手旧回答当作用户曾说过的私人事实；应说明没有可靠私人证据。",
        "private_write": False,
        "must_be_uncertain": True,
    },
    {
        "name": "creative_messages_explicit",
        "identity": "a",
        "question": "现在替我生成三条学生刚发来的新消息。",
        "expected_intent": "creative",
        "required_groups": (),
        "forbidden": ("数据库", "Episode", "检索"),
        "required_domains": (),
        "allowed_supported_domains": ("private", "knowledge"),
        "minimum_answer": "自然创作三条符合世界观的新消息，并把新事件归入当前用户私人记忆。",
        "private_write": True,
    },
    {
        "name": "creative_tasks_implicit",
        "identity": "a",
        "question": "现在有哪些新的出差任务？",
        "expected_intent": "creative",
        "required_groups": (),
        "forbidden": ("数据库", "Episode", "检索"),
        "required_domains": (),
        "allowed_supported_domains": ("private", "knowledge"),
        "minimum_answer": "在持续角色扮演语境下自然生成新的出差任务，并标为当前用户私人事件。",
        "private_write": True,
    },
    {
        "name": "creative_tasks_ja",
        "identity": "a",
        "question": "先生向けに新しいアビドス出張任務を二つ作って。",
        "expected_intent": "creative",
        "required_groups": (("アビドス", "阿拜多斯"),),
        "forbidden": ("データベース", "Episode", "検索"),
        "required_domains": (),
        "allowed_supported_domains": ("private", "knowledge"),
        "minimum_answer": "日语创作两项阿拜多斯任务，并将新事件限定为当前用户私人角色扮演内容。",
        "private_write": True,
        "expected_language": "ja",
    },
    {
        "name": "trip_context_advice",
        "identity": "a",
        "question": "我现在打算去阿拜多斯出差了，你觉得应该注意什么？",
        "expected_intent": "conversation",
        "required_groups": (("阿拜多斯",),),
        "forbidden": ("数据库", "Episode", "检索"),
        "required_domains": (),
        "allowed_supported_domains": ("knowledge",),
        "minimum_answer": "结合阿拜多斯背景给出自然建议，但不把历史剧情伪装成刚发生的新任务。",
        "private_write": False,
    },
    {
        "name": "external_weather",
        "identity": "a",
        "question": "东京现在天气怎么样？",
        "expected_intent": "external_current",
        "required_groups": (),
        "forbidden": (),
        "required_domains": (),
        "allowed_supported_domains": (),
        "minimum_answer": "说明仅凭记忆候选无法确认实时天气，不得编造当前天气。",
        "private_write": False,
        "must_be_uncertain": True,
    },
    {
        "name": "casual_greeting",
        "identity": "a",
        "question": "早上好，阿洛娜。",
        "expected_intent": "conversation",
        "required_groups": (),
        "forbidden": ("数据库", "Episode", "检索"),
        "required_domains": (),
        "allowed_supported_domains": (),
        "minimum_answer": "自然回应问候，不强行引用长期记忆。",
        "private_write": False,
    },
)


def _json_object(text: str) -> dict[str, Any]:
    rendered = str(text).strip()
    if rendered.startswith("```"):
        rendered = rendered.removeprefix("```json").removeprefix("```")
        rendered = rendered.removesuffix("```").strip()
    try:
        value = json.loads(rendered)
    except json.JSONDecodeError:
        start, end = rendered.find("{"), rendered.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(rendered[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


def _contract_packet(text: str) -> tuple[dict[str, Any], str]:
    """Parse a compact JSON contract followed by an unescaped natural answer."""
    rendered = str(text).strip()
    contract_start = rendered.find("<CONTRACT>")
    contract_end = rendered.find("</CONTRACT>")
    if contract_start >= 0 and contract_end > contract_start:
        contract_text = rendered[
            contract_start + len("<CONTRACT>") : contract_end
        ].strip()
        value = _json_object(contract_text)
        answer_start = rendered.find("<ANSWER>", contract_end)
        if answer_start < 0:
            answer_text = rendered[contract_end + len("</CONTRACT>") :]
        else:
            answer_text = rendered[answer_start + len("<ANSWER>") :]
        answer_end = answer_text.rfind("</ANSWER>")
        if answer_end >= 0:
            answer_text = answer_text[:answer_end]
        answer = answer_text.strip()
        if not answer:
            raise ValueError("empty answer block")
        return value, answer
    raise ValueError("missing complete <CONTRACT> block")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _compact(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _route_dict(route: Any) -> dict[str, Any]:
    return {
        "user": bool(route.user),
        "public": bool(route.public),
        "knowledge": bool(route.knowledge),
        "reason": str(route.reason),
        "intensity": str(route.intensity),
        "knowledge_write_policy": str(route.knowledge_write_policy),
        "creative": bool(route.creative),
    }


def _candidate_pack(domain_results: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    packed: dict[str, Any] = {}
    refs: set[str] = set()
    for domain, recalled in domain_results.items():
        raw = recalled.raw_result or {}
        episodes = []
        for item in list(raw.get("evidence_episodes") or [])[:6]:
            ref = f"{domain}:episode:{item.get('id')}"
            refs.add(ref)
            episodes.append(
                {
                    "ref": ref,
                    "source_key": item.get("source_key", ""),
                    "origin": item.get("evidence_origin", "unknown"),
                    "status": item.get("epistemic_status", "unknown"),
                    "generation": item.get("generation", 0),
                    "text": _compact(item.get("text"), 700),
                }
            )
        concepts = []
        for item in list(raw.get("evidence_concepts") or [])[:6]:
            ref = f"{domain}:concept:{item.get('id')}"
            refs.add(ref)
            concepts.append(
                {
                    "ref": ref,
                    "name": item.get("canonical_name")
                    or item.get("concept")
                    or item.get("name")
                    or "",
                    "description": _compact(item.get("description"), 320),
                    "aliases": item.get("aliases", []),
                }
            )
        packed[domain] = {
            "episodes": episodes,
            "concepts": concepts,
            "top_score": float((raw.get("retrieval_quality") or {}).get("top_score") or 0.0),
            "error": recalled.error,
        }
    return packed, refs


def _validate_contract(
    value: dict[str, Any], valid_refs: set[str], *, answer: str
) -> dict[str, Any]:
    intent = str(value.get("intent_mode", ""))
    if intent not in VALID_INTENTS:
        raise ValueError(f"invalid intent_mode: {intent}")
    answer_language = str(value.get("answer_language", "other"))
    if answer_language not in {"zh", "ja", "en", "other"}:
        raise ValueError(f"invalid answer_language: {answer_language}")
    answerability = str(value.get("answerability", "complete"))
    if answerability not in {"complete", "insufficient", "creative", "not_applicable"}:
        raise ValueError(f"invalid answerability: {answerability}")
    domain_roles = value.get("domain_roles") or {}
    claims = value.get("claims") or []
    writes = value.get("writes") or {}
    if not isinstance(domain_roles, dict) or not isinstance(claims, list):
        raise ValueError("domain_roles must be an object and claims must be a list")
    supported_ref_domains = {
        str(ref).split(":", 1)[0]
        for item in claims
        if str(item.get("status", "")) == "supported"
        for ref in item.get("refs") or []
    }
    referenced_domains = {
        str(ref).split(":", 1)[0]
        for item in claims
        for ref in item.get("refs") or []
    }
    normalized_decisions = []
    for domain in ("private", "public", "knowledge"):
        role = str(domain_roles.get(domain, "reject"))
        if role not in VALID_ROLES:
            if domain in supported_ref_domains:
                role = "support"
            elif domain in referenced_domains:
                role = "context"
            else:
                role = "reject"
        normalized_decisions.append({"domain": domain, "role": role, "reason": ""})
    normalized_claims = []
    invalid_refs: list[str] = []
    supported_without_refs: list[str] = []
    for item in claims[:12]:
        status = str(item.get("status", ""))
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid claim status: {status}")
        evidence_refs = [str(ref) for ref in item.get("refs") or []]
        invalid_refs.extend(ref for ref in evidence_refs if ref not in valid_refs)
        claim_text = str(item.get("text", "")).strip()
        if status == "supported" and not evidence_refs:
            supported_without_refs.append(claim_text)
        normalized_claims.append(
            {"claim": claim_text, "status": status, "evidence_refs": evidence_refs}
        )
    private_writes = [str(item).strip() for item in writes.get("private") or [] if str(item).strip()]
    knowledge_writes = list(writes.get("knowledge") or [])
    return {
        "intent_mode": intent,
        "answer_language": answer_language,
        "answerability": answerability,
        "domain_decisions": normalized_decisions,
        "claims": normalized_claims,
        "grounded_answer_outline": "",
        "missing_requirements": [str(item) for item in value.get("missing") or []],
        "upgrade_recommended": bool(value.get("upgrade")),
        "write_candidates": {"private": private_writes, "knowledge": knowledge_writes},
        "answer": answer.strip(),
        "validation": {
            "invalid_refs": list(dict.fromkeys(invalid_refs)),
            "supported_without_refs": supported_without_refs,
            "parse_error": "",
        },
    }


def _contains_group(answer: str, group: tuple[str, ...]) -> bool:
    folded = answer.casefold()
    return any(str(term).casefold() in folded for term in group)


def _language_pass(answer: str, expected: str | None) -> bool:
    if not expected:
        return True
    if expected == "ja":
        return bool(re.search(r"[\u3040-\u30ff]", answer))
    if expected == "en":
        return (
            len(re.findall(r"[A-Za-z]", answer)) >= 15
            and not re.search(r"[\u3400-\u9fff]", answer)
        )
    return True


def _evaluate_case(case: dict[str, Any], answer: str, contract: dict[str, Any] | None) -> dict[str, Any]:
    required = all(_contains_group(answer, tuple(group)) for group in case["required_groups"])
    forbidden_hits = [term for term in case["forbidden"] if str(term).casefold() in answer.casefold()]
    result: dict[str, Any] = {
        "required_terms_pass": required,
        "forbidden_terms_pass": not forbidden_hits,
        "forbidden_hits": forbidden_hits,
        "language_pass": _language_pass(answer, case.get("expected_language")),
    }
    if contract is None:
        result["passed"] = required and not forbidden_hits and result["language_pass"]
        return result
    supported_domains = {
        ref.split(":", 1)[0]
        for claim in contract["claims"]
        if claim["status"] == "supported"
        for ref in claim["evidence_refs"]
    }
    required_domains = set(case["required_domains"])
    allowed_domains = set(
        case.get(
            "allowed_supported_domains",
            case["required_domains"] or ("private", "public", "knowledge"),
        )
    )
    intent_pass = contract["intent_mode"] == case["expected_intent"]
    private_write_actual = bool(contract["write_candidates"]["private"])
    private_write_pass = private_write_actual == bool(case["private_write"])
    knowledge_write_pass = not bool(contract["write_candidates"]["knowledge"])
    evidence_ref_pass = (
        not contract["validation"]["invalid_refs"]
        and not contract["validation"]["supported_without_refs"]
        and not contract["validation"].get("parse_error")
    )
    domain_pass = required_domains.issubset(supported_domains) and supported_domains.issubset(
        allowed_domains
    )
    uncertainty_pass = True
    if case.get("must_be_uncertain"):
        uncertainty_pass = contract.get("answerability") == "insufficient" or bool(contract["missing_requirements"]) or any(
            claim["status"] in {"partial", "unsupported"}
            for claim in contract["claims"]
        )
    result.update(
        {
            "intent_pass": intent_pass,
            "private_write_pass": private_write_pass,
            "knowledge_write_pass": knowledge_write_pass,
            "evidence_ref_pass": evidence_ref_pass,
            "domain_pass": domain_pass,
            "supported_domains": sorted(supported_domains),
            "uncertainty_pass": uncertainty_pass,
        }
    )
    result["passed"] = all(
        (
            required,
            not forbidden_hits,
            intent_pass,
            private_write_pass,
            knowledge_write_pass,
            evidence_ref_pass,
            domain_pass,
            uncertainty_pass,
            result["language_pass"],
        )
    )
    return result


def _engine(*, model: str, json_mode: bool, max_tokens: int, temperature: float):
    from src.llm.engine import LLMConfig, LLMEngine

    config = LLMConfig(
        provider="openai",
        model_name=model,
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        timeout_seconds=120,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=False,
        response_format={"type": "json_object"} if json_mode else None,
    )
    return LLMEngine([config], max_retries=2, retry_delay=0.25)


async def _prepare_fixtures(memory: Any, output_root: Path, identity_a: Any) -> dict[str, Any]:
    marker = output_root / "fixture-import.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    from src.memory.conversation import ConversationJournal

    input_root = output_root / "fixture-inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    journal = ConversationJournal(
        input_root / "conversation-inbox", batch_exchanges=2, batch_chars=100_000
    )
    first = await journal.record_exchange(
        identity_a,
        "请记住：我的测试暗号是灰蓝鲸-314。这个暗号只属于我。",
        "好的，我会把它作为老师的私人暗号记住。",
    )
    second = await journal.record_exchange(
        identity_a,
        "我最喜欢喝无糖茉莉茶，这也是我的个人偏好。",
        "记住了，老师最喜欢无糖茉莉茶。",
    )
    ready = second or first
    if ready is None:
        flushed = await journal.flush_all()
        ready = flushed[0] if flushed else None
    if ready is None:
        raise RuntimeError("conversation fixture did not produce a ready file")
    public_file = input_root / "public-shared.txt"
    public_file.write_text(
        "公共测试约定：所有用户共同使用的问候语是‘星光归航’。这是共享约定，不属于任何特定用户。",
        encoding="utf-8",
    )
    private_result, public_result = await asyncio.gather(
        memory.import_conversation_file(ready, journal.root),
        memory.import_public_file(public_file, input_root),
    )
    payload = {
        "private": private_result,
        "public": public_result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(marker, payload)
    return payload


async def _probe_planned_domains(
    memory: Any, identity: Any, question: str, plan: Any
) -> tuple[dict[str, Any], set[str], float]:
    user_service = await memory._user_service(identity)
    services = {
        "private": user_service,
        "public": memory.public,
        "knowledge": memory.knowledge,
    }
    enabled = {
        "private": bool(plan.route.user),
        "public": bool(plan.route.public),
        "knowledge": bool(plan.route.knowledge),
    }

    async def recall(domain: str, service: Any):
        intent = plan.intent_override(domain)
        return await service.recall(
            plan.queries.get(domain, question),
            retrieval_plan=service.retrieval_plan("light"),
            intent_override=intent,
            followup_queries_override=[],
            auto_escalate=False,
        )

    started = perf_counter()
    selected = [
        (domain, service)
        for domain, service in services.items()
        if enabled[domain]
    ]
    values = await asyncio.gather(
        *(recall(domain, service) for domain, service in selected)
    )
    elapsed = perf_counter() - started
    results = dict(zip((domain for domain, _ in selected), values, strict=True))
    packed, refs = _candidate_pack(results)
    for domain in services:
        packed.setdefault(
            domain,
            {"episodes": [], "concepts": [], "top_score": 0.0, "error": "not_selected"},
        )
    return packed, refs, elapsed


def _plan_payload(plan: Any) -> dict[str, Any]:
    return {
        "needs_memory": bool(plan.route.user or plan.route.public or plan.route.knowledge),
        "domains": {
            "private": bool(plan.route.user),
            "public": bool(plan.route.public),
            "knowledge": bool(plan.route.knowledge),
        },
        "creative": bool(plan.route.creative),
        "intensity": plan.route.intensity,
        "answer_slots": list(plan.answer_slots),
        "target_entities": list(plan.target_entities),
        "uncertainty_required": bool(plan.uncertainty_required),
    }


def _evaluate_plan(case_name: str, plan: Any) -> dict[str, Any]:
    expected = PLANNER_EXPECTED[case_name]
    actual_domains = {
        domain
        for domain, enabled in _plan_payload(plan)["domains"].items()
        if enabled
    }
    actual_memory = bool(actual_domains)
    domains_pass = actual_domains == expected["domains"]
    creative_pass = bool(plan.route.creative) == expected["creative"]
    memory_pass = actual_memory == expected["memory"]
    return {
        "domains": sorted(actual_domains),
        "expected_domains": sorted(expected["domains"]),
        "domains_pass": domains_pass,
        "creative_pass": creative_pass,
        "memory_pass": memory_pass,
        "passed": domains_pass and creative_pass and memory_pass,
    }


async def _judge_batches(engine: Any, rows: list[dict[str, Any]], batch_size: int = 5) -> dict[str, Any]:
    from src.llm.engine import Message

    batches = [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]

    async def judge(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        response = await engine.generate_response(
            [Message(role="user", content=ab_judge_prompt(batch))],
            system_prompt=AB_JUDGE_SYSTEM,
            task_context="answer-contract-ab-judge",
        )
        return list(_json_object(response).get("results") or [])

    started = perf_counter()
    values = await asyncio.gather(*(judge(batch) for batch in batches))
    return {
        "seconds": round(perf_counter() - started, 6),
        "results": [item for batch in values for item in batch],
    }


async def run(*, chatbot_root: Path, output_root: Path, limit: int | None) -> dict[str, Any]:
    sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv
    from src.bot.chat_service import ConversationCoordinator
    from src.bot.memory_guard import PrivateMemoryResponseGuard
    from src.bot.persona import (
        get_dynamic_system_prompt,
        get_private_memory_emergency_reply,
    )
    from src.llm.engine import Message
    from src.memory import MemorySystem, MemorySystemConfig, PlatformIdentity
    from src.memory.conversation import ConversationSessionBuffer
    from src.memory.intent_planner import MemoryIntentPlanner

    load_dotenv(chatbot_root / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    output_root.mkdir(parents=True, exist_ok=True)
    base = MemorySystemConfig.from_env(chatbot_root)
    copied_knowledge = output_root / "knowledge" / "blue_archive.db"
    copied_knowledge.parent.mkdir(parents=True, exist_ok=True)
    if not copied_knowledge.exists():
        shutil.copy2(base.knowledge_database_path, copied_knowledge)
    base.knowledge_database_path = copied_knowledge
    base.public_database_path = output_root / "public" / "memory.db"
    base.user_database_dir = output_root / "users"
    base.log_dir = output_root / "memory-logs"
    base.foreground_recall_cache_size = 0
    base.knowledge_growth_enabled = False
    base.public_growth_enabled = False
    base.user_growth_enabled = False
    memory = MemorySystem(base)
    initialized = perf_counter()
    await memory.initialize()
    initialization_seconds = perf_counter() - initialized

    identity_a = PlatformIdentity("synthetic", "70001", "Teacher A")
    identity_b = PlatformIdentity("discord", "70001", "Teacher B")
    fixtures = await _prepare_fixtures(memory, output_root, identity_a)

    baseline_engine = _engine(
        model="Qwen/Qwen3.5-9B", json_mode=False, max_tokens=900, temperature=0.3
    )
    audit_engine = _engine(
        model="Qwen/Qwen3.5-9B", json_mode=True, max_tokens=650, temperature=0.0
    )
    rewrite_engine = _engine(
        model="Qwen/Qwen3.5-9B", json_mode=False, max_tokens=900, temperature=0.3
    )
    contract_engine = _engine(
        model="Qwen/Qwen3.5-9B", json_mode=False, max_tokens=900, temperature=0.0
    )
    planner_engine = _engine(
        model="Qwen/Qwen3.5-9B", json_mode=True, max_tokens=550, temperature=0.0
    )
    intent_planner = MemoryIntentPlanner(planner_engine)
    judge_engine = _engine(
        model="deepseek-ai/DeepSeek-V3.2",
        json_mode=True,
        max_tokens=1800,
        temperature=0.0,
    )
    coordinator = ConversationCoordinator(
        chat_engine=baseline_engine,
        fast_chat_engine=baseline_engine,
        memory_system=memory,
        system_prompt_factory=get_dynamic_system_prompt,
        sessions=ConversationSessionBuffer(4),
        private_memory_guard=PrivateMemoryResponseGuard(
            audit_engine=audit_engine,
            rewrite_engine=rewrite_engine,
            emergency_reply_factory=get_private_memory_emergency_reply,
        ),
    )

    chosen = list(CASES[:limit] if limit else CASES)
    checkpoint_path = output_root / "ab-results.json"
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        try:
            previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if (
            previous.get("status") == "in_progress"
            and previous.get("version") == ANSWER_EVIDENCE_CONTRACT_VERSION
        ):
            completed = {
                str(item["name"]): item
                for item in previous.get("cases") or []
                if item.get("name")
            }

    for case in chosen:
        if case["name"] in completed:
            continue
        identity = identity_a if case["identity"] == "a" else identity_b
        baseline_started = perf_counter()
        baseline_reply = await coordinator.handle(
            identity=identity,
            text=case["question"],
            conversation_key=f"ab:{case['name']}:baseline",
            defer_commit=True,
        )
        baseline_seconds = perf_counter() - baseline_started
        baseline_eval = _evaluate_case(case, baseline_reply.text, None)

        plan = await intent_planner.plan(case["question"])
        plan_evaluation = _evaluate_plan(case["name"], plan)
        candidates, valid_refs, probe_seconds = await _probe_planned_domains(
            memory, identity, case["question"], plan
        )
        contract_started = perf_counter()
        raw_contract = ""
        contract_error = ""
        try:
            raw_contract = await contract_engine.generate_response(
                [
                    Message(
                        role="user",
                        content=answer_evidence_contract_prompt(
                            question=case["question"],
                            candidates=candidates,
                            current_user_label=identity.display_name or "current_user",
                            request_plan=_plan_payload(plan),
                        ),
                    )
                ],
                system_prompt=ANSWER_EVIDENCE_CONTRACT_SYSTEM,
                task_context=f"answer-contract:{case['name']}",
            )
            packet, packet_answer = _contract_packet(raw_contract)
            contract = _validate_contract(
                packet, valid_refs, answer=packet_answer
            )
        except Exception as exc:
            contract_error = f"{type(exc).__name__}: {exc}"
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
                    "parse_error": contract_error,
                },
            }
        contract_seconds = perf_counter() - contract_started
        contract_eval = _evaluate_case(case, contract["answer"], contract)
        completed[case["name"]] = {
            "name": case["name"],
            "question": case["question"],
            "expected": {
                key: value
                for key, value in case.items()
                if key not in {"name", "question", "identity"}
            },
            "baseline": {
                "answer": baseline_reply.text,
                "route": _route_dict(baseline_reply.route),
                "seconds": round(baseline_seconds, 6),
                "timings": baseline_reply.timings,
                "guard": baseline_reply.memory_guard,
                "consolidation": baseline_reply.memory_consolidation,
                "evaluation": baseline_eval,
            },
            "contract": {
                "answer": contract["answer"],
                "contract": contract,
                "raw_response": raw_contract,
                "error": contract_error,
                "plan": {
                    "payload": _plan_payload(plan),
                    "raw": plan.raw,
                    "seconds": plan.planning_seconds,
                    "evaluation": plan_evaluation,
                },
                "probe_seconds": round(probe_seconds, 6),
                "generation_seconds": round(contract_seconds, 6),
                "total_seconds": round(
                    plan.planning_seconds + probe_seconds + contract_seconds,
                    6,
                ),
                "candidates": candidates,
                "evaluation": contract_eval,
            },
        }
        _write_json_atomic(
            checkpoint_path,
            {
                "status": "in_progress",
                "version": ANSWER_EVIDENCE_CONTRACT_VERSION,
                "completed": len(completed),
                "total": len(chosen),
                "cases": list(completed.values()),
            },
        )

    rows = [completed[case["name"]] for case in chosen]
    judge_input = [
        {
            "name": row["name"],
            "question": row["question"],
            "minimum_answer": row["expected"]["minimum_answer"],
            "expected_intent": row["expected"]["expected_intent"],
            "private_write_expected": row["expected"]["private_write"],
            "baseline_answer": row["baseline"]["answer"],
            "contract_answer": row["contract"]["answer"],
            "contract_metadata": {
                "intent_mode": row["contract"]["contract"]["intent_mode"],
                "claims": row["contract"]["contract"]["claims"],
                "write_candidates": row["contract"]["contract"]["write_candidates"],
            },
        }
        for row in rows
    ]
    judged = await _judge_batches(judge_engine, judge_input)
    judge_by_name = {
        str(item.get("name")): item for item in judged["results"] if item.get("name")
    }
    for row in rows:
        row["judge"] = judge_by_name.get(row["name"], {})

    baseline_seconds = [float(row["baseline"]["seconds"]) for row in rows]
    contract_seconds = [float(row["contract"]["total_seconds"]) for row in rows]
    summary = {
        "cases": len(rows),
        "initialization_seconds": round(initialization_seconds, 6),
        "deterministic": {
            "baseline_pass": sum(row["baseline"]["evaluation"]["passed"] for row in rows),
            "contract_pass": sum(row["contract"]["evaluation"]["passed"] for row in rows),
            "contract_invalid_reference_cases": sum(
                bool(row["contract"]["contract"]["validation"]["invalid_refs"])
                for row in rows
            ),
            "contract_unsupported_reference_cases": sum(
                bool(row["contract"]["contract"]["validation"]["supported_without_refs"])
                for row in rows
            ),
        },
        "planner": {
            "pass": sum(
                row["contract"]["plan"]["evaluation"]["passed"] for row in rows
            ),
            "mean_seconds": round(
                mean(float(row["contract"]["plan"]["seconds"]) for row in rows),
                6,
            ),
        },
        "judge": {
            "baseline_pass": sum(
                bool((row.get("judge") or {}).get("baseline", {}).get("passed"))
                for row in rows
            ),
            "contract_pass": sum(
                bool((row.get("judge") or {}).get("contract", {}).get("passed"))
                for row in rows
            ),
            "seconds": judged["seconds"],
        },
        "latency": {
            "baseline_mean": round(mean(baseline_seconds), 6),
            "baseline_median": round(median(baseline_seconds), 6),
            "contract_mean": round(mean(contract_seconds), 6),
            "contract_median": round(median(contract_seconds), 6),
            "baseline_under_5": sum(value < 5.0 for value in baseline_seconds),
            "contract_under_5": sum(value < 5.0 for value in contract_seconds),
        },
    }
    payload = {
        "status": "complete",
        "version": ANSWER_EVIDENCE_CONTRACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chatbot_root": str(chatbot_root),
        "output_root": str(output_root),
        "fixtures": fixtures,
        "summary": summary,
        "cases": rows,
    }
    _write_json_atomic(checkpoint_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    payload = asyncio.run(
        run(
            chatbot_root=args.chatbot_root.resolve(),
            output_root=args.output_root.resolve(),
            limit=max(1, args.limit) if args.limit else None,
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output_root / "ab-results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
