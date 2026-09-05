from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable

from memory_demo.llm.client import extract_json_payload
from memory_demo.llm.prompts import (
    CONCEPT_ADMISSION_SYSTEM,
    EPISODE_QUALITY_AUDIT_SYSTEM,
    concept_admission_batch_prompt,
    episode_quality_audit_prompt,
    second_pass_batch_prompt,
)
from memory_demo.ingestion.extractor import MemoryExtractor
from benchmarks.support.stage5 import load_json, write_json


def _jsonl_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
    return sorted({path.resolve() for path in files if path.is_file()})


def _events(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in _jsonl_files(paths):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return result


def _messages(event: dict[str, Any]) -> tuple[str, str]:
    messages = event.get("payload", {}).get("messages", [])
    if not isinstance(messages, list):
        return "", ""
    system = str(messages[0].get("content", "")) if messages else ""
    user = str(messages[1].get("content", "")) if len(messages) > 1 else ""
    return system, user


def _usage(event: dict[str, Any] | None) -> dict[str, int]:
    raw = (event or {}).get("payload", {}).get("usage", {})
    return {
        key: int(raw.get(key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _request_category(event: dict[str, Any]) -> str:
    endpoint = str(event.get("endpoint", ""))
    if endpoint == "embeddings":
        return "embedding"
    system, _ = _messages(event)
    markers = (
        ("事实提取器", "episode_extraction"),
        ("边界与证据审计器", "episode_quality_audit"),
        ("时间边界审计器", "temporal_audit"),
        ("粒度与证据审计器", "granularity_audit"),
        ("Concept 提取器", "concept_extraction"),
        ("Concept 准入审计器", "concept_admission"),
        ("多跳检索规划器", "hop_planning"),
        ("查询解析器", "query_intent"),
        ("第二名独立证据覆盖侦察器", "coverage_audit"),
        ("证据覆盖侦察器", "rerank_coverage"),
        ("证据短名单的槽位压缩器", "rerank_compression"),
        ("证据约束的长期记忆回答器", "answer"),
        ("答案证据审计器", "answer_audit"),
        ("自主生长", "association_growth"),
        ("关系判断器", "relationship_judgment"),
        ("二次理解器", "second_pass"),
    )
    if "证据约束" in system and "重排" in system:
        return "rerank_coverage"
    return next((name for marker, name in markers if marker in system), "other_chat")


def _token_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    requests = {
        str(event["request_id"]): event
        for event in events
        if event.get("event") == "llm_request" and event.get("request_id")
    }
    responses = {
        str(event["request_id"]): event
        for event in events
        if event.get("event") == "llm_response" and event.get("request_id")
    }
    totals: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = {}
    for request_id, request_event in requests.items():
        category = _request_category(request_event)
        category_totals = by_category.setdefault(category, Counter())
        category_totals["requests"] += 1
        system, user = _messages(request_event)
        category_totals["prompt_characters"] += len(system) + len(user)
        usage = _usage(responses.get(request_id))
        for key, value in usage.items():
            category_totals[key] += value
            totals[key] += value
        totals["requests"] += 1
        totals["prompt_characters"] += len(system) + len(user)
    return {
        "totals": dict(totals),
        "by_category": {
            key: dict(value) for key, value in sorted(by_category.items())
        },
        "second_pass_outcomes": _second_pass_outcomes(events),
        "episode_audit_overlap": _episode_audit_overlap(events),
        "second_pass_prompt_compaction": _second_pass_prompt_compaction(events),
    }


def _second_pass_outcomes(events: list[dict[str, Any]]) -> dict[str, int]:
    markers = ("???", "[USERNAME]", "未标注发言者", "未知发言者")
    counters: Counter[str] = Counter()
    for event in events:
        if event.get("event") != "episode_revised":
            continue
        counters["revised_events"] += 1
        previous = str(event.get("previous_text", ""))
        raw_draft = event.get("draft", {})
        current = str(raw_draft.get("text", "")) if isinstance(raw_draft, dict) else ""
        if previous != current:
            counters["text_changed"] += 1
        before = {marker for marker in markers if marker in previous}
        after = {marker for marker in markers if marker in current}
        if before:
            counters["started_with_marker"] += 1
        if before - after:
            counters["removed_any_marker"] += 1
        if before and not after:
            counters["removed_all_markers"] += 1
        if after - before:
            counters["introduced_marker"] += 1
    return dict(counters)


def _episode_audit_overlap(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, dict[str, tuple[list[dict[str, Any]], str]]] = {
        "temporal": {},
        "granularity": {},
    }
    request_counts: Counter[str] = Counter()
    for event in events:
        if event.get("event") != "llm_request":
            continue
        system, user = _messages(event)
        if "时间边界审计器" in system:
            kind, marker = "temporal", "候选 Episode："
        elif "粒度与证据审计器" in system:
            kind, marker = "granularity", "候选 EPISODES："
        else:
            continue
        request_counts[kind] += 1
        if marker not in user or "SOURCE：\n" not in user:
            continue
        candidate_text = user.split(marker, 1)[1]
        separator = "\n\nSOURCE：\n" if "\n\nSOURCE：\n" in candidate_text else "\nSOURCE：\n"
        if separator not in candidate_text:
            continue
        raw_candidates, source_text = candidate_text.split(separator, 1)
        try:
            candidates = json.loads(raw_candidates.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(candidates, list):
            continue
        source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        by_kind[kind].setdefault(source_hash, (candidates, source_text))

    temporal_keys = set(by_kind["temporal"])
    granularity_keys = set(by_kind["granularity"])
    combined_inputs = {
        **by_kind["granularity"],
        **by_kind["temporal"],
    }
    combined_prompt_chars = 0
    for candidates, source_text in combined_inputs.values():
        prompt = episode_quality_audit_prompt(source_text, "main", candidates)
        combined_prompt_chars += len(EPISODE_QUALITY_AUDIT_SYSTEM) + len(prompt)
    return {
        "historical_requests": dict(request_counts),
        "unique_temporal_sources": len(temporal_keys),
        "unique_granularity_sources": len(granularity_keys),
        "sources_requiring_both": len(temporal_keys & granularity_keys),
        "unique_sources_requiring_any": len(temporal_keys | granularity_keys),
        "estimated_combined_first_pass_requests": len(combined_inputs),
        "estimated_combined_prompt_characters": combined_prompt_chars,
        "limitation": "correction retries and completion tokens are not projected",
    }


def _second_pass_prompt_compaction(events: list[dict[str, Any]]) -> dict[str, Any]:
    historical_chars = 0
    estimated_chars = 0
    requests = 0
    estimated_requests = 0
    batch_requests = 0
    compacted_batch_requests = 0
    for event in events:
        if event.get("event") != "llm_request":
            continue
        system, user = _messages(event)
        if "二次理解器" not in system:
            continue
        requests += 1
        historical_chars += len(system) + len(user)
        source_marker = "SOURCE：\n"
        if source_marker not in user:
            estimated_chars += len(system) + len(user)
            continue
        before_source, source_text = user.rsplit(source_marker, 1)
        compact_source = MemoryExtractor.compact_source_for_reasoning(source_text)
        if "待修订项：" not in before_source:
            estimated_chars += len(system) + len(before_source) + len(source_marker) + len(compact_source)
            estimated_requests += 1
            continue
        batch_requests += 1
        raw_items = before_source.split("待修订项：", 1)[1].strip()
        try:
            items = json.loads(raw_items)
        except json.JSONDecodeError:
            estimated_chars += len(system) + len(before_source) + len(source_marker) + len(compact_source)
            estimated_requests += 1
            continue
        if not isinstance(items, list):
            continue
        shared_by_id: dict[int, dict[str, Any]] = {}
        compact_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for context in item.get("nearby", []):
                if isinstance(context, dict) and "id" in context:
                    shared_by_id[int(context["id"])] = context
            compact_items.append(
                {key: value for key, value in item.items() if key != "nearby"}
            )
        prompt = second_pass_batch_prompt(
            compact_items,
            compact_source,
            [shared_by_id[key] for key in sorted(shared_by_id)],
        )
        estimated_chars += len(system) + len(prompt)
        estimated_requests += 1
        compacted_batch_requests += 1
    return {
        "historical_requests": requests,
        "historical_prompt_characters": historical_chars,
        "batch_requests": batch_requests,
        "compacted_batch_requests": compacted_batch_requests,
        "estimated_requests": estimated_requests,
        "estimated_prompt_characters": estimated_chars,
        "estimated_prompt_character_reduction": (
            1.0 - estimated_chars / historical_chars if historical_chars else 0.0
        ),
    }


def _concept_admission_analysis(
    events: list[dict[str, Any]],
    *,
    minimum_episodes: int,
    batch_size: int,
) -> dict[str, Any]:
    def compact_item(item: dict[str, Any]) -> dict[str, Any]:
        concept = item.get("concept", {})
        return {
            "candidate_index": int(item["candidate_index"]),
            "concept": {
                "canonical_name": str(concept.get("canonical_name", "")),
                "description": str(concept.get("description", ""))[:480],
                "aliases": list(concept.get("aliases", []))[:8],
                "confidence": float(concept.get("confidence", 0.0)),
            },
            "episode_occurrence_count": int(item["episode_occurrence_count"]),
            "source_occurrence_count": int(item["source_occurrence_count"]),
            "evidence_episode_texts": [
                str(text)[:480]
                for text in item.get("evidence_episode_texts", [])[:2]
            ],
            "similar_concepts": [
                {
                    **similar,
                    "description": str(similar.get("description", ""))[:220],
                }
                for similar in item.get("similar_concepts", [])
            ],
        }

    requests = {
        str(event["request_id"]): event
        for event in events
        if event.get("event") == "llm_request" and event.get("request_id")
    }
    responses = {
        str(event["request_id"]): event
        for event in events
        if event.get("event") == "llm_response" and event.get("request_id")
    }
    items: list[dict[str, Any]] = []
    decisions: dict[int, dict[str, Any]] = {}
    original_request_count = 0
    original_prompt_characters = 0
    original_usage: Counter[str] = Counter()
    marker = "候选："
    for request_id, request_event in requests.items():
        system, user = _messages(request_event)
        if "Concept 准入审计器" not in system or marker not in user:
            continue
        original_request_count += 1
        original_prompt_characters += len(system) + len(user)
        response_event = responses.get(request_id)
        original_usage.update(_usage(response_event))
        try:
            batch = json.loads(user.split(marker, 1)[1])
        except (json.JSONDecodeError, IndexError):
            continue
        if isinstance(batch, list):
            items.extend(item for item in batch if isinstance(item, dict))
        try:
            content = str(
                response_event["payload"]["choices"][0]["message"]["content"]
            )
            payload = extract_json_payload(content)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("decisions"), list):
            for decision in payload["decisions"]:
                if isinstance(decision, dict):
                    try:
                        decisions[int(decision["candidate_index"])] = decision
                    except (KeyError, TypeError, ValueError):
                        continue

    action_counts = Counter(
        str(decision.get("action", "missing")) for decision in decisions.values()
    )
    singleton_reuse_scores = []
    for item in items:
        index = int(item["candidate_index"])
        decision = decisions.get(index, {})
        if (
            int(item.get("episode_occurrence_count", 0)) < minimum_episodes
            and decision.get("action") == "reuse"
        ):
            similar = item.get("similar_concepts", [])
            singleton_reuse_scores.append(
                max(
                    (float(value.get("similarity", -1.0)) for value in similar),
                    default=-1.0,
                )
            )

    policies: list[dict[str, Any]] = []
    for threshold in (0.58, 0.59, 0.595, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85):
        kept: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for item in items:
            recurring = (
                int(item.get("episode_occurrence_count", 0)) >= minimum_episodes
            )
            top_similarity = max(
                (
                    float(value.get("similarity", -1.0))
                    for value in item.get("similar_concepts", [])
                ),
                default=-1.0,
            )
            (kept if recurring or top_similarity >= threshold else skipped).append(item)
        skipped_actions = Counter(
            str(decisions.get(int(item["candidate_index"]), {}).get("action", "missing"))
            for item in skipped
        )
        estimated_prompts = [
            [compact_item(item) for item in kept[start : start + batch_size]]
            for start in range(0, len(kept), batch_size)
        ]
        estimated_chars = sum(
            len(CONCEPT_ADMISSION_SYSTEM) + len(concept_admission_batch_prompt(batch))
            for batch in estimated_prompts
        )
        policies.append(
            {
                "similarity_threshold": threshold,
                "sent_to_llm": len(kept),
                "prefiltered_transient": len(skipped),
                "estimated_requests": len(estimated_prompts),
                "estimated_prompt_characters": estimated_chars,
                "prompt_character_reduction": (
                    1.0 - estimated_chars / original_prompt_characters
                    if original_prompt_characters
                    else 0.0
                ),
                "historical_model_actions_among_skipped": dict(skipped_actions),
            }
        )
    return {
        "historical": {
            "requests": original_request_count,
            "items": len(items),
            "decisions": len(decisions),
            "prompt_characters": original_prompt_characters,
            "usage": dict(original_usage),
            "actions": dict(action_counts),
            "singleton_reuse_top_similarity_min": (
                min(singleton_reuse_scores) if singleton_reuse_scores else None
            ),
            "singleton_reuse_top_similarity_mean": (
                mean(singleton_reuse_scores) if singleton_reuse_scores else None
            ),
        },
        "minimum_distinct_episodes": minimum_episodes,
        "proposed_batch_size": batch_size,
        "estimated_prompt_uses_balanced_compaction": True,
        "policies": policies,
    }


def _coverage_groups(payload: dict[str, Any] | None) -> list[list[int]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("coverage"), list):
        return []
    groups: list[list[int]] = []
    for item in payload["coverage"]:
        if not isinstance(item, dict) or not isinstance(item.get("episode_ids"), list):
            continue
        ids: list[int] = []
        for value in item["episode_ids"]:
            try:
                node_id = int(value)
            except (TypeError, ValueError):
                continue
            if node_id not in ids:
                ids.append(node_id)
        if not ids:
            continue
        if str(item.get("mode", "alternatives")).casefold() == "joint":
            groups.extend([[node_id] for node_id in ids[:5]])
        else:
            groups.append(ids[:5])
    return groups


def _coverage_ids(payload: dict[str, Any] | None, limit: int) -> list[int]:
    groups = _coverage_groups(payload)
    result: list[int] = []
    for rank in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if rank < len(group) and group[rank] not in result:
                result.append(group[rank])
                if len(result) >= limit:
                    return result
    return result


def _fill(ids: list[int], candidates: list[int], limit: int) -> list[int]:
    result = list(dict.fromkeys(ids))[:limit]
    for node_id in candidates:
        if len(result) >= limit:
            break
        if node_id not in result:
            result.append(node_id)
    return result


def _recall(ids: list[int], groups: list[dict[str, Any]]) -> float:
    selected = set(ids)
    return (
        sum(bool(selected.intersection(int(value) for value in group["alternatives"])) for group in groups)
        / len(groups)
        if groups
        else 0.0
    )


def _retrieval_ablation(report: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    variants: dict[str, list[float]] = {
        "initial_coverage_only": [],
        "dual_coverage_without_compressor": [],
        "strict_current": [],
    }
    adaptive_values: list[float] = []
    review_levels: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        trace = row.get("result", {}).get("rerank_trace", {})
        candidates = [int(value) for value in trace.get("candidate_episode_ids", [])]
        initial_ids = _fill(_coverage_ids(trace.get("initial"), limit), candidates, limit)
        merged_ids = _fill(
            _coverage_ids(trace.get("merged_coverage"), limit), candidates, limit
        )
        strict_ids = [int(value) for value in row.get("result", {}).get("episode_ids", [])]
        groups = row.get("score", {}).get("episode_groups", [])
        values = {
            "initial_coverage_only": _recall(initial_ids, groups),
            "dual_coverage_without_compressor": _recall(merged_ids, groups),
            "strict_current": _recall(strict_ids, groups),
        }
        missing = trace.get("initial", {}).get("missing_aspects", [])
        if (
            len(trace.get("atomic_queries", [])) >= 24
            or (isinstance(missing, list) and any(str(item).strip() for item in missing))
        ):
            review_level = "strict"
            adaptive_recall = values["strict_current"]
        elif len(_coverage_groups(trace.get("initial"))) <= 1:
            review_level = "compress"
            # Historical compressor output is used as a proxy.  It saw merged
            # coverage, so the online balanced run still needs confirmation.
            adaptive_recall = values["strict_current"]
        else:
            review_level = "none"
            adaptive_recall = values["initial_coverage_only"]
        review_levels[review_level] += 1
        adaptive_values.append(adaptive_recall)
        for key, value in values.items():
            variants[key].append(value)
        rows.append(
            {
                "question_id": row.get("id"),
                **values,
                "adaptive_review_level": review_level,
                "adaptive_recall_proxy": adaptive_recall,
            }
        )
    return {
        "summary": {
            key: {
                "mean_recall": mean(values) if values else 0.0,
                "minimum_recall": min(values, default=0.0),
                "perfect_questions": sum(value >= 0.999 for value in values),
            }
            for key, values in variants.items()
        },
        "adaptive_policy": {
            "strict_atomic_query_threshold": 24,
            "compress_coverage_group_threshold": 1,
            "review_levels": dict(review_levels),
            "model_review_calls": (
                len(rows)
                + 2 * review_levels["strict"]
                + review_levels["compress"]
            ),
            "historical_strict_model_review_calls": 3 * len(rows),
            "mean_recall_proxy": mean(adaptive_values) if adaptive_values else 0.0,
            "minimum_recall_proxy": min(adaptive_values, default=0.0),
            "proxy_limitation": (
                "compress-only rows reuse the historical final compressor output, "
                "which was conditioned on merged coverage"
            ),
        },
        "rows": rows,
    }


def _adaptive_token_estimate(
    token_summary: dict[str, Any], retrieval_ablation: dict[str, Any]
) -> dict[str, Any]:
    baseline = token_summary["totals"]
    categories = token_summary["by_category"]
    levels = retrieval_ablation["adaptive_policy"]["review_levels"]
    strict = int(levels.get("strict", 0))
    compress = int(levels.get("compress", 0))
    coverage = categories.get("coverage_audit", {})
    compressor = categories.get("rerank_compression", {})
    estimated: dict[str, float] = {}
    for metric in ("prompt_tokens", "completion_tokens", "total_tokens"):
        coverage_average = float(coverage.get(metric, 0)) / max(
            1, int(coverage.get("requests", 0))
        )
        compressor_average = float(compressor.get(metric, 0)) / max(
            1, int(compressor.get("requests", 0))
        )
        removed = float(coverage.get(metric, 0)) + float(
            compressor.get(metric, 0)
        )
        value = (
            float(baseline.get(metric, 0))
            - removed
            + coverage_average * strict
            + compressor_average * (strict + compress)
        )
        estimated[metric] = round(value, 1)
        estimated[f"{metric}_reduction"] = (
            1.0 - value / float(baseline.get(metric, 1))
        )
    return {
        "method": "historical per-stage mean multiplied by adaptive call counts",
        "estimated": estimated,
    }


def _balanced_import_token_estimate(token_summary: dict[str, Any]) -> dict[str, Any]:
    totals = token_summary["totals"]
    categories = token_summary["by_category"]
    overlap = token_summary["episode_audit_overlap"]
    second_pass = token_summary["second_pass_prompt_compaction"]
    temporal = categories.get("temporal_audit", {})
    granularity = categories.get("granularity_audit", {})
    audit_prompt_tokens = float(temporal.get("prompt_tokens", 0)) + float(
        granularity.get("prompt_tokens", 0)
    )
    audit_completion_tokens = float(temporal.get("completion_tokens", 0)) + float(
        granularity.get("completion_tokens", 0)
    )
    audit_historical_chars = float(temporal.get("prompt_characters", 0)) + float(
        granularity.get("prompt_characters", 0)
    )
    audit_prompt_ratio = float(
        overlap.get("estimated_combined_prompt_characters", 0)
    ) / max(1.0, audit_historical_chars)
    historical_audit_requests = int(temporal.get("requests", 0)) + int(
        granularity.get("requests", 0)
    )
    audit_request_ratio = float(
        overlap.get("estimated_combined_first_pass_requests", 0)
    ) / max(1, historical_audit_requests)
    pass2_prompt_ratio = float(
        second_pass.get("estimated_prompt_characters", 0)
    ) / max(1.0, float(second_pass.get("historical_prompt_characters", 0)))
    pass2 = categories.get("second_pass", {})
    estimated_prompt = (
        float(totals.get("prompt_tokens", 0))
        - audit_prompt_tokens
        - float(pass2.get("prompt_tokens", 0))
        + audit_prompt_tokens * audit_prompt_ratio
        + float(pass2.get("prompt_tokens", 0)) * pass2_prompt_ratio
    )
    estimated_completion = (
        float(totals.get("completion_tokens", 0))
        - audit_completion_tokens
        + audit_completion_tokens * audit_request_ratio
    )
    estimated_total = estimated_prompt + estimated_completion
    return {
        "method": (
            "historical token/character ratios; exact-alias embedding savings are "
            "excluded, so this is conservative"
        ),
        "estimated_prompt_tokens": round(estimated_prompt, 1),
        "estimated_completion_tokens": round(estimated_completion, 1),
        "estimated_total_tokens": round(estimated_total, 1),
        "estimated_total_token_reduction": (
            1.0 - estimated_total / max(1.0, float(totals.get("total_tokens", 0)))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit token-reduction policies against completed experiment logs."
    )
    parser.add_argument("--concept-logs", type=Path, required=True)
    parser.add_argument("--query-logs", type=Path, required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    args = parser.parse_args()

    concept_events = _events([args.concept_logs])
    query_events = _events([args.query_logs])
    query_token_baseline = _token_summary(query_events)
    retrieval_ablation = _retrieval_ablation(load_json(args.retrieval_report))
    report = {
        "version": "architecture-optimization-audit-v1",
        "concept_admission": _concept_admission_analysis(
            concept_events,
            minimum_episodes=max(1, args.minimum_episodes),
            batch_size=max(1, args.batch_size),
        ),
        "query_token_baseline": query_token_baseline,
        "retrieval_ablation": retrieval_ablation,
        "adaptive_query_token_estimate": _adaptive_token_estimate(
            query_token_baseline, retrieval_ablation
        ),
        "balanced_import_token_estimate": _balanced_import_token_estimate(
            query_token_baseline
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
