from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from typing import Any


FOLLOWUP_PROTOCOL_VERSION = "evidence-gap-v1-graph2-candidate80"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def _compact(text: Any, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


async def run(
    *,
    chatbot_root: Path,
    probe_path: Path,
    judge_path: Path,
    output_path: Path,
    concurrency: int,
    checkpoint_every: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv
    from src.memory import MemorySystem, MemorySystemConfig, RetrievalPlan

    load_dotenv(chatbot_root / ".env")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    judged = json.loads(judge_path.read_text(encoding="utf-8"))
    probe_by_name = {row["name"]: row for row in probe.get("cases") or []}

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for role_row in judged.get("cases") or []:
        category = str(role_row.get("category", ""))
        support = str((role_row.get("predicted") or {}).get("factual_support", ""))
        if category == "knowledge_supported" and support != "complete":
            selected.append((probe_by_name[role_row["name"]], role_row))
        elif category == "same_entity_missing":
            selected.append((probe_by_name[role_row["name"]], role_row))

    completed_by_name: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        try:
            previous = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if (
            previous.get("status") == "in_progress"
            and previous.get("probe") == str(probe_path)
            and previous.get("judge") == str(judge_path)
            and previous.get("protocol_version") == FOLLOWUP_PROTOCOL_VERSION
        ):
            completed_by_name = {
                str(item["name"]): item
                for item in previous.get("cases") or []
                if isinstance(item, dict) and item.get("name")
            }

    memory = MemorySystem(MemorySystemConfig.from_env(chatbot_root))
    await memory.initialize()
    plan = RetrievalPlan(
        preset="standard",
        query_planner="heuristic",
        graph_hops=2,
        candidate_limit=80,
        reranker="configured",
        evidence_slots=True,
        followup_policy="never",
        verification="local",
        deadline_seconds=12.0,
        answer_episode_limit=20,
        answer_concept_limit=12,
        answer_path_limit=10,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def deepen(
        pair: tuple[dict[str, Any], dict[str, Any]]
    ) -> dict[str, Any]:
        base, role = pair
        predicted = role.get("predicted") or {}
        missing = [
            _compact(item, 240)
            for item in predicted.get("missing_requirements") or []
            if _compact(item, 240)
        ][:4]
        followups = list(dict.fromkeys(missing or [base["question"]]))
        intent = {
            "language": "auto",
            "target_entities": [],
            "search_queries": [base["question"]],
            "requested_relation": "",
            "temporal_constraint": "",
            "causal_constraint": "",
            "answer_shape": "evidence_gap_followup",
            "uncertainty_required": True,
        }
        async with semaphore:
            started = perf_counter()
            recalled = await memory.knowledge.recall(
                base["question"],
                intent_override=intent,
                followup_queries_override=followups,
                retrieval_plan=plan,
                auto_escalate=False,
            )
            elapsed = perf_counter() - started

        raw = recalled.raw_result or {}
        quality = raw.get("retrieval_quality") or {}
        evidence = list(raw.get("evidence_episodes") or [])
        concepts = list(raw.get("evidence_concepts") or [])
        old_ids = {int(value) for value in base.get("episode_ids") or []}
        episode_ids = [int(value) for value in (raw.get("episode_ids") or [])]
        return {
            **{
                key: base[key]
                for key in (
                    "name",
                    "family",
                    "category",
                    "language",
                    "question",
                    "knowledge_expected",
                    "support_groups",
                )
                if key in base
            },
            "seconds": round(elapsed, 6),
            "top_score": float(quality.get("top_score") or 0.0),
            "initial_factual_support": predicted.get("factual_support", ""),
            "initial_use_policy": predicted.get("use_policy", ""),
            "followup_queries": followups,
            "initial_episode_ids": sorted(old_ids),
            "episode_ids": episode_ids,
            "new_episode_ids": [value for value in episode_ids if value not in old_ids],
            "quality": quality,
            "timings": raw.get("timings") or {},
            "top_evidence": [
                {
                    "id": item.get("id"),
                    "source_key": item.get("source_key"),
                    "text": _compact(item.get("text"), 600),
                }
                for item in evidence[:12]
            ],
            "top_concepts": [
                {
                    "id": item.get("id"),
                    "canonical_name": item.get("canonical_name", ""),
                    "description": _compact(item.get("description"), 320),
                    "aliases": item.get("aliases", []),
                }
                for item in concepts[:12]
            ],
            "error": recalled.error,
        }

    started = perf_counter()
    pending = [pair for pair in selected if pair[0]["name"] not in completed_by_name]
    tasks = [asyncio.create_task(deepen(pair)) for pair in pending]
    completed_since_checkpoint = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        completed_by_name[result["name"]] = result
        completed_since_checkpoint += 1
        if completed_since_checkpoint >= checkpoint_every:
            _write_json_atomic(
                output_path,
                {
                    "status": "in_progress",
                    "experiment": "evidence-driven-followup-retrieval-v1",
                    "probe": str(probe_path),
                    "judge": str(judge_path),
                    "protocol_version": FOLLOWUP_PROTOCOL_VERSION,
                    "completed": len(completed_by_name),
                    "total": len(selected),
                    "cases": list(completed_by_name.values()),
                },
            )
            completed_since_checkpoint = 0
    rows = [completed_by_name[pair[0]["name"]] for pair in selected]
    wall_seconds = perf_counter() - started
    positive = [row for row in rows if row["category"] == "knowledge_supported"]
    safety = [row for row in rows if row["category"] == "same_entity_missing"]
    payload = {
        "status": "complete",
        "experiment": "evidence-driven-followup-retrieval-v1",
        "probe": str(probe_path),
        "judge": str(judge_path),
        "protocol_version": FOLLOWUP_PROTOCOL_VERSION,
        "protocol": {
            "selection": "all initially non-complete supported cases plus all same-entity missing-detail controls",
            "followup_source": "judge missing_requirements; original question only when empty",
            "retrieval": "standard preset, graph_hops=2, candidate_limit=80, BGE configured",
        },
        "summary": {
            "cases": len(rows),
            "knowledge_gap_cases": len(positive),
            "missing_detail_controls": len(safety),
            "wall_seconds": round(wall_seconds, 6),
            "mean_seconds": round(mean(row["seconds"] for row in rows), 6),
            "p95_seconds": round(
                _percentile([row["seconds"] for row in rows], 0.95), 6
            ),
            "error_count": sum(bool(row["error"]) for row in rows),
            "mean_new_episode_count": round(
                mean(len(row["new_episode_ids"]) for row in rows)
                if rows
                else 0.0,
                3,
            ),
        },
        "cases": rows,
    }
    _write_json_atomic(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    args = parser.parse_args()
    payload = asyncio.run(
        run(
            chatbot_root=args.chatbot_root.resolve(),
            probe_path=args.probe.resolve(),
            judge_path=args.judge.resolve(),
            output_path=args.output.resolve(),
            concurrency=max(1, args.concurrency),
            checkpoint_every=max(1, args.checkpoint_every),
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
