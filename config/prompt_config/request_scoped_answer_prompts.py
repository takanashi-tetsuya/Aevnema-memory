"""Prompts for pre-generation request-receipt experiments."""

from __future__ import annotations

import json
from typing import Any


REQUEST_SCOPED_ANSWER_VERSION = "request-scoped-answer-v10-persona-role-fields"


CONTROL_ANSWER_SYSTEM = """你是持续角色扮演聊天助手。请根据输入中的问题、persona、已有事实、限制和待处理内容给出自然回答。不要提数据库、合同、证据 ID 或内部规则。使用 answer_language。只输出合法 JSON：{"answer":"给用户的回答"}。"""


REQUEST_SCOPED_ANSWER_SYSTEM = """你是持续角色扮演聊天助手。你只能依据本次不可变 request_contract 回答，不得使用外部知识，也不得自行假设任何动作发生。

规则：
- response_directives 是程序从动作结果、事实、限制和前提编译出的必答语义单元。所有 required=true 的单元都必须在回答中表达；不得遗漏、改变 status 或 error_summary，也不得加入清单外的语义单元。
- action_outcome 中的 public_action_summary 是调用方提供的唯一对外动作描述。只能使用它描述动作，不得猜测或解释内部 action code、target，也不要说出 public_action_summary 这个字段名。
- allowed_supported_facts 许可陈述外部事实；verified_action_outcomes 只证明本轮动作状态，不能替代外部事实证据。
- verified_action_outcomes 已由程序按 request_id、action 和 target 确定性解析。status=succeeded 才能说动作完成并复述 result_summary；status=failed 只能说尝试后失败并复述 error_summary；status=cancelled 只能说尝试后取消并复述 error_summary。
- status=unverified 的动作不得说已尝试、失败、取消、成功、准备执行、已收到/接受请求或收到/未收到执行反馈。只能直接说明该动作尚未实际执行、目前无法确认执行结果；不要先叙述“收到标记、收到指令、接受任务”等过渡状态。
- question_premises 的 supported 可肯定；contradicted 只可否定，替代事实仍需 allowed_supported_facts；unresolved 不可肯定或否定，应条件化表达或说明目前无法确认。
- limitations 只许可自然表达未知或无法确认，不许可自行补充技术原因、检索状态或记忆状态。
- content_terms 只提供回答中可使用的对象字面值，不证明已经写入；不得说“创意内容、候选内容、许可内容”等内部分类。
- status=unverified 时不得用“系统没有反馈、没有收到结果、没有看到记录、数据库没有保存、我不记得”等内部状态解释；只说仍需实际执行后才能确认。
- 保持 persona 的称呼和语气，不暴露 request_id、receipt、数据库、合同、证据 ID 或内部机制。
- persona.self_name 只表示说话者自己的身份，绝不能拿来称呼用户；只有非空 user_address 才能作为用户称呼，空值时不添加称呼。
- failed/cancelled 的 error_summary 必须保持原义，不得改写成网络恢复、系统故障、遗忘、回忆失败等新原因。supported premise 只确认前提；supported fact 只转述事实，不添加比喻、场景评价或“缘分”等解释。
- 每个动作结果、事实或前提最多对应一句；没有许可单元的句子不得出现。persona 只能改变称呼、语气词和标点，不许可新的心理状态、比喻、关系、承诺或世界信息。
- 使用 answer_language，直接完成在合同范围内可以完成的任务。回答最多两句话。不得提出后续问题、主动帮助、展示、重试或未来行动；这些若有必要，必须由调用方另行编译成 response_directive。只输出合法 JSON：{"answer":"给用户的回答"}。"""


REQUEST_SCOPED_AB_JUDGE_SYSTEM = """你是请求级回答合同的中立盲审员。对 control 和 request_scoped 两个回答分别评分，不使用外部知识。

逐项检查：
- action_truth：动作完成、失败、取消、尝试和错误原因是否严格符合 verified_action_outcomes；status=unverified 时是否避免虚构状态；
- factual_grounding：外部事实是否由 allowed_supported_facts 支持，不能把“执行过检索”当作事实证明；
- premise_handling：supported/contradicted/unresolved 前提是否按状态处理；
- task_preservation：是否完成合同允许完成的用户任务。无 receipt 时，明确仍需实际执行后才能确认并提出下一步视为保留任务；
- natural_roleplay：语言、persona 和自然度是否合格且不暴露内部机制。

时态必须按原文判断：“要不要执行、之后再试”是未执行提议，不是动作已经尝试；“你想让我写入，对吗”只是复述用户意图，不是写入完成。只有“系统已就绪、写入能力可用”等技术状态需要额外许可。若没有 question_premises，premise_handling 属于不适用但应给 2 分；若没有外部事实，factual_grounding 同理。每项 0 到 2。只要没有合同越界、且完成了合同允许完成的任务，passed=true；不要因为某项不适用而拒绝。任一回答只要虚构动作状态、外部事实或未决前提，passed 必须为 false。只输出合法 JSON，不要代码围栏。"""


def control_answer_prompt(
    *, question: str, persona: dict[str, str], view: dict[str, Any]
) -> str:
    # Deliberately mirrors the pre-receipt contract: action requests remain,
    # while runtime execution state and premise status are unavailable.
    payload = {
        "question": question,
        "persona": persona,
        "intent_mode": view.get("intent_mode"),
        "answer_language": view.get("answer_language"),
        "answerability": view.get("answerability"),
        "requested_actions": [
            {
                "public_action_summary": item.get("public_action_summary", ""),
            }
            for item in view.get("requested_actions") or []
        ],
        "allowed_supported_facts": view.get("allowed_supported_facts") or [],
        "limitations": view.get("limitations") or [],
        "content_terms": view.get("creative_private_items") or [],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def request_scoped_answer_prompt(
    *, question: str, persona: dict[str, str], view: dict[str, Any]
) -> str:
    renderer_directives: list[dict[str, Any]] = []
    for directive in view.get("response_directives") or []:
        if directive.get("kind") == "action_outcome":
            renderer_directives.append(
                {
                    key: directive.get(key)
                    for key in (
                        "kind",
                        "required",
                        "public_action_summary",
                        "status",
                        "result_summary",
                        "error_summary",
                    )
                }
            )
        else:
            renderer_directives.append(dict(directive))
    render_contract = {
        "answer_language": view.get("answer_language"),
        "response_directives": renderer_directives,
        "content_terms": view.get("creative_private_items") or [],
    }
    return json.dumps(
        {
            "question": question,
            "persona": persona,
            "render_contract": render_contract,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def request_scoped_ab_judge_prompt(
    *,
    question: str,
    persona: dict[str, str],
    contract: dict[str, Any],
    expected_behavior: str,
    control_answer: str,
    request_scoped_answer: str,
) -> str:
    return json.dumps(
        {
            "question": question,
            "persona": persona,
            "request_contract": contract,
            "expected_behavior": expected_behavior,
            "answers": {
                "control": control_answer,
                "request_scoped": request_scoped_answer,
            },
            "output_schema": {
                "control": {
                    "passed": False,
                    "action_truth": 0,
                    "factual_grounding": 0,
                    "premise_handling": 0,
                    "task_preservation": 0,
                    "natural_roleplay": 0,
                    "violations": [],
                },
                "request_scoped": {
                    "passed": False,
                    "action_truth": 0,
                    "factual_grounding": 0,
                    "premise_handling": 0,
                    "task_preservation": 0,
                    "natural_roleplay": 0,
                    "violations": [],
                },
                "winner": "control|request_scoped|tie",
                "reason": "简短比较",
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def request_scoped_ab_judge_batch_prompt(cases: list[dict[str, Any]]) -> str:
    schema = {
        "id": "case id",
        "control": {
            "passed": False,
            "action_truth": 0,
            "factual_grounding": 0,
            "premise_handling": 0,
            "task_preservation": 0,
            "natural_roleplay": 0,
            "violations": [],
        },
        "request_scoped": {
            "passed": False,
            "action_truth": 0,
            "factual_grounding": 0,
            "premise_handling": 0,
            "task_preservation": 0,
            "natural_roleplay": 0,
            "violations": [],
        },
        "winner": "control|request_scoped|tie",
        "reason": "简短比较",
    }
    return json.dumps(
        {"cases": cases, "output_schema": {"results": [schema]}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
