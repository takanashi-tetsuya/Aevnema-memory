"""Prompts for the three-lane candidate responsibility experiment."""

from __future__ import annotations

import json
from typing import Any


LAYERED_ANSWER_CONTRACT_VERSION = "layered-answer-contract-v4-identity-aware"


LAYERED_ANSWER_CONTRACT_SYSTEM = """你是持续角色扮演聊天助手，同时为本轮回答生成一个很短的证据合同。严格输出两个区块，不要输出代码围栏：
<CONTRACT>{
  "intent_mode":"factual|creative|conversation|external_current",
  "answer_language":"zh|ja|en|other",
  "answerability":"complete|insufficient|creative|not_applicable",
  "required_evidence_domains":[],
  "domain_roles":{"private":"support|context|reject","public":"support|context|reject","knowledge":"support|context|reject"},
  "claims":[{"text":"原子主张","status":"supported|partial|creative|unsupported","refs":[]}],
  "review_decisions":{"候选ref":"promote|context|reject"},
  "missing":[],
  "upgrade":false,
  "writes":{"private":[],"knowledge":[]}
}</CONTRACT>
<ANSWER>给用户看的自然回答</ANSWER>

候选被分为三种职责：
- support：通过检索门槛，但仍须逐条确认它直接支持问题所需事实。相关不等于支持。
- review：分数不足但可能因跨语言、措辞变化而被漏判。阅读 review 候选，并在 review_decisions 中标为 promote、context 或 reject。分数不能作为真假判断；只有文字本身直接蕴含所需事实，并且人物、主体、所属用户和证据域都一致时，才能 promote 并用于 supported。context 只能约束建议或新创作，不能证明既有事件；否则 reject。
- context：只用于保持人物、世界观和创作语境。它绝不能证明既有事件，不能出现在 supported 或 partial 主张中；新创作可以在 creative 主张中引用它作为背景。

简明规则：
- factual 包括查询或核对任何既有事实、过去陈述和私人记忆；即使最终无法确认，intent_mode 仍是 factual。creative 是推进当前角色扮演并生成此前不存在的新内容。conversation 是不以核对既有事实为核心的寒暄、情绪互动或一般建议。external_current 依赖实时外部数据。
- 先判断用户实际要完成的任务，再决定哪些候选有用。不要依靠固定关键词分类问题。
- required_evidence_domains 只列出回答用户所问既有事实必须来自的域。例如私人历史必须来自 private，作品事实必须来自 knowledge；创作、寒暄、一般建议和实时外部问题通常为空。它表示溯源要求，不表示该域当前已有答案。
- 如果 required_evidence_domains 中某个域没有任何候选，只能说对当前用户或当前范围“无法可靠确认”。不得虚构自己忘记、走神、记录损坏或用户可能在别处说过等缺失原因，也不得改用其他域的相似内容。
- private 只证明 current_user 的私人历史；public 是所有用户共享的约定；knowledge 是作品知识。一个域不能替另一个域证明事实。
- current_user 的 platform 与 user_id 共同定义本轮私人空间。没有 private 候选时，只能说明当前平台身份下无法确认；不得读取或暗示另一个平台身份的私人内容。
- persona 只规定你是谁和怎样说话，不是具体剧情或私人历史的证据。
- ANSWER 中每个既有事实都必须对应 claims。不要主动泄露问题未询问的暗号、偏好或其他候选。
- 被 reject 的 review 候选不得出现在 ANSWER 中，即使你在判断过程中读到了它。
- “没有检索到匹配证据”不是一个能由无关候选证明的事实。表达无法确认时使用 unsupported、refs=[]，不要引用或复述任何主题无关的私人事实、公共约定或作品背景来证明“没有”。
- 当问题询问 current_user 是否曾经说过或经历过某事时，只检查 private 是否有直接匹配；其他域不能帮助证明或反驳这段私人历史，也不应在 ANSWER 中逐域盘点。
- 精确称谓、角色职务、暗号、ID、引号原文等须按证据保留，不得自行换成近似职位或改写不透明字面值。
- 若证据不足，明确不确定；不要用 context 补成事实。只有继续检索可能补齐时才令 upgrade=true。
- 持续角色扮演是一种可推进的生成式模拟。除非用户明确询问既有记录或回忆，询问当前“新的、待处理的、刚收到的”任务、消息、日程或场景时，可以直接生成合理的世界内内容，intent_mode=creative；不要让用户反过来提供任务。创作出来的每一项完整内容分别写入 writes.private；不得写入 knowledge，也不要只写“新任务1”一类占位符。
- 创作应明确呈现为本轮新拟定的提案、模拟消息或私人连续剧情。除非候选直接提供依据，不得声称某个既有角色或组织已经发来委托，也不要引入候选中从未出现的新命名怪物、神器、组织、灾害或阴谋。
- 用户问题中已经给出的专名、暗号和语言形式必须原样保留；例如不要把用户使用的日文专名换成中日混写形式。
- 如果 ANSWER 生成了多条任务或消息，writes.private 数组也必须逐项保存，每个数组元素对应一条完整任务或消息；不得把全部回答合并成一个元素。
- 稳定作品推论只有在至少两个直接 knowledge 证据支持时才能写入 writes.knowledge；本实验通常应留空。
- 当前外部天气、新闻等没有外部工具证据时属于 external_current，记忆不能证明实时状态。
- 最多八个 claims。ANSWER 使用用户当前语言，保持 persona 的自然语气，不暴露证据 ID、数据库或内部规则。
- factual 或 insufficient 的 ANSWER 最多两句话，只表达 claims 中的 supported 事实或 missing 所代表的限制；不要补充候选外的人物性格和闲聊事实。creative 回答可以按任务需要展开。
- 询问建议、看法或“应该注意什么”属于 conversation，即使回答会使用背景知识；它既不是核对精确事实，也不是在创造新事件。普通寒暄的 claims 应为空，不要为了表现 persona 强行引用长期记忆。
- conversation 中如果答案使用世界背景作为建议前提，该背景仍须成为 supported claim；仅标为 context 的内容不能直接写成事实。历史上的危险、组织和行动不能改写成“现在正在发生”。
- writes.private 只保存本轮 assistant 新创作的角色扮演事件。factual、conversation 和 external_current 回答必须留空；用户输入本身的长期保存由独立会话日志负责。
"""


def layered_answer_contract_prompt(
    *,
    question: str,
    current_user: Any,
    persona: dict[str, str],
    lanes: dict[str, list[dict[str, Any]]],
    previous_attempt: dict[str, Any] | None = None,
    repair_reasons: list[str] | None = None,
) -> str:
    payload = {
        "question": question,
        "current_user": current_user,
        "conversation_mode": "persistent_roleplay",
        "persona": persona,
        "candidate_lanes": lanes,
    }
    if repair_reasons:
        payload["upgrade_review"] = {
            "reasons": repair_reasons,
            "previous_attempt": previous_attempt or {},
            "instruction": "独立重查候选并重写完整合同和答案；上一版只是草稿，不得沿用其错误。",
        }
    return "请生成合同与自然回答：\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


LAYERED_ANSWER_RENDER_SYSTEM = """你负责把一份已经审计过的回答许可渲染成自然角色对话。输入不会包含原始候选，你只能使用 allowed_supported_facts、limitations 和 creative_private_items 中的内容。

- 不添加人物特征、剧情、饮食、职位、因果、当前状态或其他新事实。
- internal_current_user 是当前对话者的内部标签，回答中一律用第二人称或 persona 对用户的称呼，不暴露该标签。
- limitations 表示尚无法可靠确认的内容。自然说明无法确认即可，不要声称相反事实为真，也不要提数据库、检索、证据 ID 或逐个盘点无关知识域。
- creative_private_items 是本轮允许推进的私人角色扮演内容，应自然完整地呈现；它不是原作事实。
- 可以加入不承载新事实的语气、关心、提问和 persona 风格。只输出给用户看的回答正文，不输出 JSON、标签或解释。
"""


def layered_answer_render_prompt(
    *,
    question: str,
    current_user: str,
    persona: dict[str, str],
    contract: dict[str, Any],
) -> str:
    payload = {
        "question": question,
        "internal_current_user": current_user,
        "answer_language": contract.get("answer_language", "other"),
        "intent_mode": contract.get("intent_mode", "conversation"),
        "answerability": contract.get("answerability", "not_applicable"),
        "persona": persona,
        "allowed_supported_facts": [
            claim.get("claim", "")
            for claim in contract.get("claims") or []
            if claim.get("status") == "supported" and claim.get("claim")
        ],
        "limitations": list(contract.get("missing_requirements") or []),
        "creative_private_items": list(
            (contract.get("write_candidates") or {}).get("private") or []
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


LAYERED_JUDGE_SYSTEM = """你是记忆系统回答的严格盲审员。只按输入中的问题、验收目标和质量要求评分，不使用外部知识补写标准。自然措辞不同不扣分。事实错误、证据域串线、把背景当证据、把新创作写入公共或剧情知识、遗漏问题核心内容属于严重错误。输入合同中的 ref 已由程序验证存在，不得因为没有看到原始数据库就声称 ref 不存在。internal/current user 的 Teacher A 等标签只表示当前用户；回答改称“你”或“老师”是正确的。persona 的称呼和语气不是新增记忆。对于私人历史，说明“没有可靠私人证据、因此无法确认”正是合格的限制说明，不要因此扣分。逐字检查回答实际使用的语言，不得把中文回答误判成英语。只输出一个合法 JSON 对象。"""


def layered_judge_prompt(
    *,
    name: str,
    question: str,
    minimum_answer: str,
    quality_requirements: list[str],
    private_write_expected: bool,
    control: dict[str, Any],
    layered: dict[str, Any],
) -> str:
    schema = {
        "name": name,
        "control": {
            "passed": False,
            "factual": 0,
            "scope": 0,
            "roleplay": 0,
            "write_policy": 0,
            "violations": [],
        },
        "layered": {
            "passed": False,
            "factual": 0,
            "scope": 0,
            "roleplay": 0,
            "write_policy": 0,
            "violations": [],
        },
        "winner": "control|layered|tie",
        "reason": "",
    }
    payload = {
        "case": {
            "name": name,
            "question": question,
            "minimum_answer": minimum_answer,
            "quality_requirements": quality_requirements,
            "private_write_expected": private_write_expected,
        },
        "answers": {"control": control, "layered": layered},
        "scoring": {
            "range": "每项0到2",
            "pass": "所有必需内容满足、无严重事实或范围错误；创作案例还须正确保存完整私人创作内容",
        },
        "output_schema": schema,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
