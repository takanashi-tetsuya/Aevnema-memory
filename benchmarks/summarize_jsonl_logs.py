from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import statistics


HTTP_CODE = re.compile(r"HTTP\s+(\d{3})")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float], digits: int = 3) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "p50": None, "p95": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": round(min(values), digits),
        "p50": round(statistics.median(values), digits),
        "p95": round(float(percentile(values, 0.95)), digits),
        "maximum": round(max(values), digits),
    }


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def request_category(payload: dict) -> str:
    endpoint = str(payload.get("endpoint", "unknown"))
    if endpoint == "embeddings":
        return "embedding"
    messages = payload.get("payload", {}).get("messages", [])
    if not messages:
        return endpoint
    system = str(messages[0].get("content", ""))
    markers = (
        ("事实提取器", "episode_extraction"),
        ("边界与证据审计器", "episode_quality_audit"),
        ("时间边界审计器", "temporal_audit"),
        ("粒度与证据审计器", "granularity_audit"),
        ("Concept 提取器", "concept_extraction"),
        ("Concept 准入审计器", "concept_admission"),
        ("关系判断器", "relationship_judgment"),
        ("二次理解器", "second_pass"),
        ("查询解析器", "query_intent"),
        ("自主生长", "association_growth"),
        ("长期记忆回答器", "answer"),
        ("JSON 格式修复器", "json_repair"),
    )
    return next((name for marker, name in markers if marker in system), "other_chat")


def empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def add_usage(target: dict[str, int], usage: object) -> None:
    """Accumulate OpenAI-compatible usage while tolerating missing fields."""
    if not isinstance(usage, dict):
        return
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    try:
        prompt_value = int(prompt or 0)
    except (TypeError, ValueError):
        prompt_value = 0
    try:
        completion_value = int(completion or 0)
    except (TypeError, ValueError):
        completion_value = 0
    try:
        total_value = int(usage.get("total_tokens") or 0)
    except (TypeError, ValueError):
        total_value = 0
    target["prompt_tokens"] += prompt_value
    target["completion_tokens"] += completion_value
    target["total_tokens"] += total_value or prompt_value + completion_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize memory-demo JSONL logs")
    parser.add_argument("inputs", nargs="+", help="JSONL file or directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.inputs:
        path = Path(raw)
        files.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
    files = sorted(dict.fromkeys(path.resolve() for path in files if path.is_file()))

    events: list[dict] = []
    parse_errors: list[dict] = []
    per_file: list[dict] = []
    for path in files:
        file_events: list[dict] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    parse_errors.append(
                        {"file": str(path), "line": line_number, "error": str(exc)}
                    )
                    continue
                event["_file"] = str(path)
                file_events.append(event)
                events.append(event)
        timestamps = [
            parse_time(str(event["timestamp"]))
            for event in file_events
            if event.get("timestamp")
        ]
        per_file.append(
            {
                "file": str(path),
                "bytes": path.stat().st_size,
                "events": len(file_events),
                "started_at": min(timestamps).isoformat() if timestamps else None,
                "finished_at": max(timestamps).isoformat() if timestamps else None,
                "duration_seconds": (
                    round((max(timestamps) - min(timestamps)).total_seconds(), 3)
                    if timestamps else None
                ),
            }
        )

    event_counts = Counter(str(event.get("event", "unknown")) for event in events)
    requests = {
        str(event["request_id"]): event
        for event in events
        if event.get("event") == "llm_request" and event.get("request_id")
    }
    outcomes = {
        str(event["request_id"]): event
        for event in events
        if event.get("event") in {"llm_response", "llm_error"}
        and event.get("request_id")
    }
    durations_by_category: dict[str, list[float]] = defaultdict(list)
    prompt_chars_by_category: dict[str, list[float]] = defaultdict(list)
    usage_total = empty_usage()
    usage_by_category: dict[str, dict[str, int]] = defaultdict(empty_usage)
    usage_by_model: dict[str, dict[str, int]] = defaultdict(empty_usage)
    responses_with_usage = 0
    model_counts: Counter[str] = Counter()
    for request_id, event in requests.items():
        category = request_category(event)
        payload = event.get("payload", {})
        model = payload.get("model")
        if model:
            model_counts[str(model)] += 1
        messages = payload.get("messages", [])
        if messages:
            prompt_chars_by_category[category].append(
                float(sum(len(str(message.get("content", ""))) for message in messages))
            )
        outcome = outcomes.get(request_id)
        if outcome and outcome.get("event") == "llm_response":
            response_payload = outcome.get("payload", {})
            usage = (
                response_payload.get("usage")
                if isinstance(response_payload, dict)
                else None
            )
            if isinstance(usage, dict):
                responses_with_usage += 1
                add_usage(usage_total, usage)
                add_usage(usage_by_category[category], usage)
                add_usage(usage_by_model[str(model or "unknown")], usage)
        if outcome and event.get("timestamp") and outcome.get("timestamp"):
            duration = (
                parse_time(str(outcome["timestamp"]))
                - parse_time(str(event["timestamp"]))
            ).total_seconds()
            durations_by_category[category].append(max(0.0, duration))

    error_codes: Counter[str] = Counter()
    error_messages: Counter[str] = Counter()
    for event in events:
        if event.get("event") != "llm_error":
            continue
        message = str(event.get("error", ""))
        match = HTTP_CODE.search(message)
        error_codes[match.group(1) if match else "non_http"] += 1
        error_messages[message[:500]] += 1

    validation_stages = Counter(
        str(event.get("stage", "unknown"))
        for event in events
        if event.get("event") == "validation_failed"
    )
    retry_delays = [
        float(event["retry_after_seconds"])
        for event in events
        if event.get("event") == "retry"
        and event.get("retry_after_seconds") is not None
    ]
    timestamps = [
        parse_time(str(event["timestamp"])) for event in events if event.get("timestamp")
    ]
    report = {
        "inputs": [str(path) for path in files],
        "files": per_file,
        "totals": {
            "bytes": sum(item["bytes"] for item in per_file),
            "events": len(events),
            "parse_errors": len(parse_errors),
            "started_at": min(timestamps).isoformat() if timestamps else None,
            "finished_at": max(timestamps).isoformat() if timestamps else None,
            "duration_seconds": (
                round((max(timestamps) - min(timestamps)).total_seconds(), 3)
                if timestamps else None
            ),
        },
        "event_counts": dict(event_counts.most_common()),
        "models": dict(model_counts.most_common()),
        "request_outcome_counts": {
            "requests": len(requests),
            "responses_or_errors": len(outcomes),
            "requests_without_logged_outcome": len(set(requests).difference(outcomes)),
            "responses_with_usage": responses_with_usage,
        },
        "token_usage": {
            "total": usage_total,
            "by_category": {
                category: values
                for category, values in sorted(usage_by_category.items())
            },
            "by_model": {
                model: values for model, values in sorted(usage_by_model.items())
            },
        },
        "request_duration_seconds": {
            category: describe(values)
            for category, values in sorted(durations_by_category.items())
        },
        "prompt_characters": {
            category: describe(values, digits=1)
            for category, values in sorted(prompt_chars_by_category.items())
        },
        "retry_delay_seconds": describe(retry_delays),
        "http_error_codes": dict(error_codes.most_common()),
        "validation_failed_by_stage": dict(validation_stages.most_common()),
        "fallbacks": int(event_counts.get("fallback", 0)),
        "top_error_messages": [
            {"count": count, "message": message}
            for message, count in error_messages.most_common(10)
        ],
        "parse_error_details": parse_errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), **report["totals"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
