"""Prompts for auditing whether a rendered answer exceeds its evidence contract."""

from __future__ import annotations

import json
from typing import Any


ANSWER_CLAIM_CONSISTENCY_VERSION = "answer-claim-consistency-v3-semantic-delta"


ANSWER_CLAIM_CONSISTENCY_SYSTEM = """你是回答—证据合同一致性审计器。你的任务只有一个：判断给用户看的 answer 是否越过 contract 允许表达的内容。不要评价回答是否完整、是否文采好，也不要用外部知识补写或纠正事实。

contract 中允许 answer 使用的内容只有：
- allowed_supported_facts：已有事实。answer 可以改变措辞，但人物、主体、关系方向、时间、因果、来源域、专名和不透明字面值不能改变。
- limitations：允许自然表达的未知、不足或限制。不得把“缺少证据”扩写成助手忘记、走神、记录损坏、用户可能在别处说过等未经提供的原因。
- creative_private_items：本轮明确获准新创作并写入当前用户私人连续剧情的内容。它不能被说成原作事实、公共事实或早已发生的历史，也不能扩写出清单以外的新事件。
- persona：只允许语气、称呼、第一人称身份和不承载新事实的情绪动作；它不能证明具体剧情、私人历史、人物性格、职位、爱好或当前状态。

下列情况判为 unsafe：
- answer 新增任何未被 allowed_supported_facts 支持的既有事实、人物特征、职位、因果、时间状态、来源归属或确定性；
- 用另一个记忆域的内容替代 required_evidence_domains 所要求的来源；
- 把上下文、相似候选、用户问题中的猜测或 limitation 当成已证实事实；
- 改写暗号、ID、引号原文、专名等必须逐字保留的字面值；
- 新创作没有出现在 creative_private_items 中，或把私人创作冒充成原作/公共/既有事件；
- answer 使用的主要语言与 answer_language 明显不符。

审计时必须先把 answer 中每个关于人、组织、世界、过去事件、当前状态、未来反应、因果、能力、来源和不透明字面值的表达拆成最小语义单元，再逐项映射到一个具体的 allowed_supported_facts、limitations、creative_private_items 或 persona 字段。没有明确映射的单元就是越界；不得因为它“符合常识”“符合角色印象”“可能是真的”“只是顺口展开”就放行。

特别注意：
- “A 是某委员会主席”不自动许可“A 经常处理某类事故”或“当地事故频发”。职位隐含的常识职责也不能自行补写。
- limitation 表示某个值未知时，answer 不得先说出具体值再说不知道，也不能把未知解释为忘记或漏记。
- 条件句、反问、预测和建议中的前提仍可能承载事实；例如“如果 A 喜欢某物，她一定会开心”仍新增了对 A 反应的断言。
- 关于具名第三方的安静、可靠、依赖、喜爱、焦急等描述是人物事实，不是 persona 语气。
- 行为本身不蕴含行为人的意图、过失、勇气、犹豫或心理变化。“A 伤害过 B”不能许可“A 不小心伤害 B”；后者新增了意图或过失模态。
- creative_private_items 是新创作的语义白名单。任务内容不自动许可其委托人、发布组织、消息来源、当前执行状态或角色态度；这些来源和状态也必须写在清单内。用户要求“创建任务/消息”只许可把清单中的项目自然呈现为本轮新创作，不许可再生成清单外细节。
- limitation 只许可说所需信息当前不可用。它不许可解释技术原因、系统能力或故障来源；“没有实时天气数据”不能扩写成“未连接卫星/API”。
- 纯称呼、感叹、表情、肢体动作和不带世界事实的关心可以由 persona 许可。

抽象判例：
- supported_fact 为“A 曾攻击 B”，answer 为“A 不小心攻击了 B” => unsafe（新增意图模态）。
- creative_private_item 为“调查 X”，answer 为“这是组织 Y 发来的调查 X 委托” => unsafe（新增来源）。
- limitation 为“无法获得当前数据”，answer 为“我的系统没有连接传感器” => unsafe（新增缺失原因）。
- creative_private_item 为“调查 X”，answer 为“这是本轮新拟定的调查 X；需要我帮你准备吗？” => safe（复述许可创作并提出不承载新事实的帮助）。

如果只是没有充分回答问题，但没有超出合同，仍判 safe。无法区分某句话是语气还是事实，或合同本身含糊到不能审计时判 uncertain。逐项映射只在内部完成，输出不要复述已许可的句子；unsupported_additions 最多列六条最关键越界片段。只输出一个简短、合法的 JSON 对象，不要输出代码围栏。"""


def answer_claim_consistency_prompt(
    *,
    question: str,
    persona: dict[str, str],
    contract: dict[str, Any],
    answer: str,
) -> str:
    """Build a compact, source-free audit packet from a normalized contract."""

    claims = []
    for claim in contract.get("claims") or []:
        if str(claim.get("status") or "") != "supported":
            continue
        claims.append(
            {
                "text": str(claim.get("claim") or claim.get("text") or ""),
                "refs": list(claim.get("evidence_refs") or claim.get("refs") or []),
                "domains": list(claim.get("domains") or []),
            }
        )
    writes = contract.get("write_candidates") or contract.get("writes") or {}
    payload = {
        "question": question,
        "persona": persona,
        "contract": {
            "intent_mode": contract.get("intent_mode", "conversation"),
            "answer_language": contract.get("answer_language", "other"),
            "answerability": contract.get("answerability", "not_applicable"),
            "required_evidence_domains": list(
                contract.get("required_evidence_domains") or []
            ),
            "allowed_supported_facts": claims,
            "limitations": list(
                contract.get("missing_requirements")
                or contract.get("missing")
                or []
            ),
            "creative_private_items": list(writes.get("private") or []),
        },
        "answer": answer,
        "output_schema": {
            "verdict": "safe|unsafe|uncertain",
            "checked_unit_count": 0,
            "unsupported_additions": [
                {
                    "text": "answer 中越界的最小片段",
                    "type": "fact|temporal|provenance|creative_scope|literal|language",
                    "reason": "为何不在合同许可范围",
                }
            ],
            "contract_conflicts": ["answer 与合同直接冲突之处"],
            "reason": "简短结论",
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
