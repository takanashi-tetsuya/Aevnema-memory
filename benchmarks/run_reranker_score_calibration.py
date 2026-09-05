from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import numpy as np
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from memory_demo.config import AppConfig
from memory_demo.llm.client import ModelClient


DOCUMENTS: tuple[dict[str, str], ...] = (
    {
        "id": "private_password",
        "text": "Teacher A 在对话中陈述其测试暗号为“灰蓝鲸 -314”，并强调该暗号仅属于其本人。",
    },
    {
        "id": "private_drink",
        "text": "Teacher A 在对话中陈述其个人偏好为最喜欢喝无糖茉莉茶。",
    },
    {
        "id": "public_greeting",
        "text": "所有用户共同使用的公共测试问候语是“星光归航”，不属于任何特定用户。",
    },
    {
        "id": "hina_role",
        "text": "空崎日奈（Hina）担任格黑娜风纪委员会委员长，负责应对格黑娜频繁发生的治安问题。",
    },
    {
        "id": "shiroko_profile",
        "text": "砂狼白子是阿拜多斯对策委员会成员，表达简短直接，喜爱运动、公路自行车和城市观察。",
    },
    {
        "id": "saori_chain",
        "text": "锭前纱织曾射伤老师，后来为了拯救亚津子而向老师求助。",
    },
)


QUERIES: tuple[dict[str, Any], ...] = (
    {"name": "password_exact", "text": "我以前告诉你的测试暗号是什么？", "relevant": ["private_password"]},
    {"name": "drink_exact", "text": "我以前说自己最喜欢喝什么？", "relevant": ["private_drink"]},
    {"name": "public_exact", "text": "公共测试问候语是什么？", "relevant": ["public_greeting"]},
    {"name": "hina_exact_en", "text": "What role does Hina hold at Gehenna?", "relevant": ["hina_role"]},
    {"name": "shiroko_profile", "text": "白子平时有哪些性格和习惯？", "relevant": ["shiroko_profile"]},
    {"name": "saori_causal", "text": "纱织向老师求助前做过什么，后来为什么求助？", "relevant": ["saori_chain"]},
    {"name": "private_false_claim", "text": "我以前是不是说过白子最喜欢草莓牛奶？", "relevant": []},
    {"name": "weather_unrelated", "text": "东京现在天气怎么样？", "relevant": []},
    {"name": "greeting_unrelated", "text": "早上好，阿洛娜。", "relevant": []},
    {"name": "creative_unrelated", "text": "现在有哪些新的出差任务？", "relevant": []},
    {"name": "password_exact_ja", "text": "前に私が教えたテスト用の合言葉を覚えてる？", "relevant": ["private_password"]},
    {"name": "abydos_task_ja", "text": "先生向けに新しいアビドス出張任務を二つ作って。", "relevant": ["shiroko_profile"]},
    {
        "name": "password_ja_private_wrapped",
        "text": "检索当前用户私人记忆中能直接回答以下问题的证据：前に私が教えたテスト用の合言葉を覚えてる？",
        "relevant": ["private_password"],
    },
    {
        "name": "password_ja_public_wrapped",
        "text": "检索所有用户共享记忆中能直接回答以下问题的证据：前に私が教えたテスト用の合言葉を覚えてる？",
        "relevant": [],
    },
    {
        "name": "false_private_wrapped",
        "text": "检索当前用户私人记忆中能直接证明以下说法的证据：我以前是不是说过白子最喜欢草莓牛奶？",
        "relevant": [],
    },
    {
        "name": "trip_knowledge_wrapped",
        "text": "检索外部作品知识中能直接支持以下建议的世界背景：我现在打算去阿拜多斯出差了，应该注意什么？",
        "relevant": ["shiroko_profile"],
    },
    {
        "name": "creative_context_wrapped",
        "text": "寻找可约束以下角色扮演创作的作品世界观、人物或事件背景：现在有哪些新的出差任务？",
        "relevant": ["hina_role", "shiroko_profile", "saori_chain"],
    },
)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


async def run(*, env_file: Path, output: Path, concurrency: int) -> dict[str, Any]:
    config = AppConfig.from_env(env_file)
    client = ModelClient(config.model)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    document_embeddings = await asyncio.to_thread(
        client.embed, [item["text"] for item in DOCUMENTS]
    )
    document_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
    normalized_documents = document_embeddings / np.maximum(document_norms, 1e-12)

    async def score(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            started = perf_counter()
            ranked, query_embedding = await asyncio.gather(
                asyncio.to_thread(
                    client.rerank,
                    case["text"],
                    [item["text"] for item in DOCUMENTS],
                    top_n=len(DOCUMENTS),
                ),
                asyncio.to_thread(client.embed, [case["text"]]),
            )
            elapsed = perf_counter() - started
        values = [
            {
                "rank": rank,
                "document_id": DOCUMENTS[int(item["index"])]["id"],
                "score": round(float(item["relevance_score"]), 8),
            }
            for rank, item in enumerate(ranked, start=1)
        ]
        relevant = set(case["relevant"])
        normalized_query = query_embedding[0] / max(
            float(np.linalg.norm(query_embedding[0])), 1e-12
        )
        cosine_by_id = {
            item["id"]: float(normalized_documents[index] @ normalized_query)
            for index, item in enumerate(DOCUMENTS)
        }
        for item in values:
            item["cosine"] = round(cosine_by_id[item["document_id"]], 8)
        relevant_scores = [
            item["score"] for item in values if item["document_id"] in relevant
        ]
        irrelevant_scores = [
            item["score"] for item in values if item["document_id"] not in relevant
        ]
        relevant_cosines = [cosine_by_id[value] for value in relevant]
        irrelevant_cosines = [
            cosine
            for document_id, cosine in cosine_by_id.items()
            if document_id not in relevant
        ]
        return {
            "name": case["name"],
            "query": case["text"],
            "expected_relevant": sorted(relevant),
            "seconds": round(elapsed, 6),
            "ranking": values,
            "top_is_relevant": bool(values and values[0]["document_id"] in relevant),
            "best_relevant_score": max(relevant_scores) if relevant_scores else None,
            "best_irrelevant_score": max(irrelevant_scores) if irrelevant_scores else None,
            "best_relevant_cosine": max(relevant_cosines) if relevant_cosines else None,
            "best_irrelevant_cosine": max(irrelevant_cosines) if irrelevant_cosines else None,
        }

    started = perf_counter()
    cases = await asyncio.gather(*(score(case) for case in QUERIES))
    wall_seconds = perf_counter() - started
    positive = [case for case in cases if case["expected_relevant"]]
    negative = [case for case in cases if not case["expected_relevant"]]
    positive_scores = [float(case["best_relevant_score"]) for case in positive]
    negative_scores = [float(case["best_irrelevant_score"]) for case in negative]
    positive_cosines = [float(case["best_relevant_cosine"]) for case in positive]
    negative_cosines = [float(case["best_irrelevant_cosine"]) for case in negative]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model.reranker_model,
        "summary": {
            "queries": len(cases),
            "positive_queries": len(positive),
            "negative_queries": len(negative),
            "positive_top1_accuracy": (
                sum(case["top_is_relevant"] for case in positive) / len(positive)
            ),
            "positive_best_score_min": min(positive_scores),
            "positive_best_score_median": median(positive_scores),
            "negative_top_score_max": max(negative_scores),
            "negative_top_score_median": median(negative_scores),
            "score_overlap": min(positive_scores) <= max(negative_scores),
            "positive_best_cosine_min": min(positive_cosines),
            "positive_best_cosine_median": median(positive_cosines),
            "negative_top_cosine_max": max(negative_cosines),
            "negative_top_cosine_median": median(negative_cosines),
            "cosine_overlap": min(positive_cosines) <= max(negative_cosines),
            "mean_request_seconds": mean(float(case["seconds"]) for case in cases),
            "wall_seconds": round(wall_seconds, 6),
        },
        "documents": list(DOCUMENTS),
        "cases": cases,
    }
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    payload = asyncio.run(
        run(
            env_file=args.env_file.resolve(),
            output=args.output.resolve(),
            concurrency=max(1, args.concurrency),
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
