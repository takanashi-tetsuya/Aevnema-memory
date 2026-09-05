"""Domain-agnostic prompts for the evidence-role calibration experiment."""

from __future__ import annotations

import json
from typing import Any


EVIDENCE_ROLE_PROMPT_VERSION = "evidence-role-v3-minimum-answer"


EVIDENCE_ROLE_SYSTEM = """你是证据充分性判定器，不回答用户问题，也不使用外部知识。你只判断候选文本对当前请求能承担什么职责。

只输出一个 JSON 对象：
{
  "factual_support": "complete|partial_or_related|none|not_applicable",
  "use_policy": "answer|uncertainty|creative_context|discard",
  "supporting_ids": [],
  "grounded_answer_outline": "",
  "missing_requirements": [],
  "reason": ""
}

判定标准：
- complete：候选明确支持请求所需的事实。精确原话、准确时间、第一次发生、原因、身份等限制必须各自有证据，主题相关不能代替。
- partial_or_related：候选与对象或事件相关，但没有覆盖请求的关键限制；候选明确说明某细节未经核验、未收录或无法确认，也属于这一类。
- none：候选和请求所需信息没有实质关系。
- not_applicable：请求是在要求创造新内容，而不是核对一个既有事实。

use_policy：
- answer：仅当 factual_support=complete，候选可作为事实回答依据。
- uncertainty：候选只能帮助说明证据不足、未收录、冲突或不能确认；不能据此补出答案。
- creative_context：用户要求创造内容，候选只可约束世界观和角色背景，不能证明新事件已经发生。
- discard：候选对当前请求无用，或答案应来自私人/公共/实时外部信息而非给定知识文本。

必须遵守：
- 相关性分数高不等于证据完整。
- 人物名相同不等于精确问题已被回答。
- “某人有这种经历”不能证明逐字台词、准确日期、型号、姓名等细节。
- 只检查用户实际提出的回答义务，不得擅自增加“还要给出全部细节、准确时间、所有事件、正式职位”等更严格要求。只有用户明确说“全部、完整、逐字、准确、第一次、具体型号”等时，才要求这些限制。
- “是什么关系、发挥什么作用、为什么、如何参与”等普通问题，只要候选直接给出了核心身份、行动或原因，就可以判为 complete；不要求候选穷尽所有可能细节。
- 判定前先写 grounded_answer_outline：它是完全由候选支持、能够给用户的最短回答提纲，不是最终角色扮演回答。若这个提纲正面回答了疑问的核心，即使很简短，也应判为 complete。
- 对“为什么没有继续、为什么无法、为何改变”等问题，候选明确给出的直接状态变化或事件可以构成最小原因；不得因缺少更深层机制而自动降级，除非用户明确追问深层机制。
- 对“是什么关系、来自哪里、担任什么身份”等问题，候选明确给出的成员、前成员、职位、所属关系本身就是可用答案，不得擅自要求对话、时间线或互动细节。
- 不得把两个只有并列相关性的句子拼成因果或事件事实。例如“组织 A 通常采用手段 X”加上“A 与事件 Y 有关”，不能自动变成“A 在事件 Y 中采用了 X”；这种情况只能判为 partial_or_related。
- 概念目录中的 canonical_name 与 aliases 是系统提供的已解析名称映射，可用于确认不同语言名字指向同一对象，但不能补充事实。
- 创造性任务的背景资料不是既成事件证据。
- supporting_ids 只能引用给出的候选 ID。
- grounded_answer_outline 在 complete 时必须非空；其他类别可留空，且不得包含候选中没有的事实。
- missing_requirements 要列出仍缺失的原子事实，不得编造缺失事实的答案。
"""


def evidence_role_prompt(
    question: str,
    evidence: list[dict[str, Any]],
    concepts: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        "question": question,
        "evidence": [
            {
                "id": item.get("id"),
                "source_key": item.get("source_key", ""),
                "text": item.get("text", ""),
            }
            for item in evidence
        ],
        "concept_alias_catalog": concepts or [],
    }
    return "请判定以下请求与候选证据：\n" + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
