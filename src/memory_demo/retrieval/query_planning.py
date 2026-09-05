"""Deterministic transformations of model-produced retrieval intent."""

from __future__ import annotations

import re

from memory_demo.types import QueryIntent


def structural_queries(question: str, intent: QueryIntent) -> list[str]:
    """Translate typed intent fields into explicit evidence-budget slots."""

    queries: list[str] = []
    fields = (
        ("关系约束", intent.requested_relation),
        ("时间约束", intent.temporal_constraint),
        ("因果约束", intent.causal_constraint),
    )
    for label, raw_value in fields:
        value = str(raw_value).strip()
        if value:
            queries.append(f"__constraint_slot__ {label}：{value}")

    quoted: list[str] = []
    for match in re.finditer(r"[‘“\"]([^’”\"]{2,80})[’”\"]", question):
        value = match.group(1).strip()
        before = question[max(0, match.start() - 18) : match.start()]
        after = question[match.end() : match.end() + 18]
        prohibited = bool(
            re.search(
                r"(?:不要|无需|禁止)(?:直接)?"
                r"(?:回答|认定|写成|证明)?[^。；]{0,8}$",
                before,
            )
            or re.search(
                r"^[^‘’“”\"。；]{0,8}"
                r"(?:未经证明|不能确定|不代表|并非事实)",
                after,
            )
        )
        if value and not prohibited and value not in quoted:
            quoted.append(value)
    for value in quoted:
        queries.append(f"__answer_slot__ 原文中“{value}”对应的事实是什么")
        if len(queries) >= 6:
            break
    return list(dict.fromkeys(queries))


def expand_rerank_atomic_queries(
    atomic_queries: list[str],
    intent: QueryIntent,
    whole_question: str | None = None,
) -> list[str]:
    """Split residual Chinese conjuncts for evidence-slot accounting only."""

    base_queries = list(
        dict.fromkeys(str(query).strip() for query in atomic_queries)
    )
    result: list[str] = []
    context = "、".join(intent.target_entities[:4])
    for query in base_queries:
        if query and query not in result:
            result.append(query)
        if (
            (whole_question is not None and query == whole_question)
            or query.startswith(("__constraint_slot__ ", "__answer_slot__ "))
            or "还是" in query
        ):
            continue
        paired_subjects = re.match(
            r"^([^，。；：！？]{2,12})与([^，。；：！？的]{2,16})的(.{3,})$",
            query,
        )
        if paired_subjects:
            first, second, predicate = paired_subjects.groups()
            for expanded in (
                f"{first}本人直接陈述或表现的{predicate}",
                f"{second}直接陈述或被转述的{predicate}",
            ):
                if expanded not in result:
                    result.append(expanded)
        for segment in re.split(r"[。；\n]", query):
            if re.search(r"[‘“\"][^’”\"]*、[^’”\"]*[’”\"]", segment):
                continue
            parts = [part.strip() for part in segment.split("、")]
            eligible_parts = [part for part in parts if len(part) >= 4]
            if not 2 <= len(parts) <= 8 or len(eligible_parts) < 2:
                continue
            for part in eligible_parts:
                expanded = (
                    f"{context}：{part}（独立证据槽，不能由同列另一条件替代）"
                    if context
                    else f"{part}（独立证据槽，不能由同列另一条件替代）"
                )
                if expanded not in result:
                    result.append(expanded)
    return result[:40]


def limit_rerank_atomic_queries(queries: list[str], limit: int) -> list[str]:
    """Bound slots while preserving constraints and late resolved queries."""

    unique = list(
        dict.fromkeys(
            str(value).strip() for value in queries if str(value).strip()
        )
    )
    if limit <= 0 or len(unique) <= limit:
        return unique
    selected_indices = {0}
    selected_indices.update(
        index
        for index, value in enumerate(unique)
        if value.startswith(("__constraint_slot__ ", "__answer_slot__ "))
    )
    if len(selected_indices) >= limit:
        return [unique[index] for index in sorted(selected_indices)[:limit]]
    regular = [
        index
        for index in range(1, len(unique))
        if index not in selected_indices
    ]
    available = limit - len(selected_indices)
    front_count = min(len(regular), (available * 2 + 2) // 3)
    back_count = min(
        len(regular) - front_count,
        available - front_count,
    )
    selected_indices.update(regular[:front_count])
    if back_count:
        selected_indices.update(regular[-back_count:])
    return [unique[index] for index in sorted(selected_indices)][:limit]


def requires_entity_resolved_followup(
    question: str,
    intent: QueryIntent,
) -> bool:
    """Detect a later clause that refers to an earlier unknown answer."""

    text = " ".join([question, *intent.search_queries, intent.requested_relation])
    dependent_patterns = (
        r"(?:谁|哪位|哪个|什么人).{0,100}"
        r"(?:这位|这名|该(?:对象|实体|主体)|其|此人).{0,80}"
        r"(?:谁|哪位|哪个|什么|又|身份|名称|别名|化名|代号)",
        r"(?:这位|这名|该(?:对象|实体|主体)|他们|她们|它们|其|此人)"
        r".{0,80}(?:谁|哪位|哪个|什么|名称|别名|化名|代号|又)",
        r"(?:已识别|上一跳|前一跳|该候选|此人|上述(?:对象|实体|主体))",
    )
    return any(re.search(pattern, text) for pattern in dependent_patterns)
