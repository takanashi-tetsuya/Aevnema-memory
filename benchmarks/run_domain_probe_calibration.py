from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from typing import Any


POSITIVE_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "family": "shiroko_profile",
        "support_groups": (("白子", "运动"), ("白子", "寡言")),
        "questions": (
            ("zh", "你觉得白子是个什么样的人？"),
            ("zh_recall", "你还记得白子平时表现出哪些性格和习惯吗？"),
            ("en", "What kind of person is Shiroko, based on the story?"),
            ("ja", "シロコは普段どんな性格で、何を好んでいますか？"),
        ),
    },
    {
        "family": "shiroko_terror_difference",
        "support_groups": (("白子Terror", "不同世界"), ("白子Terror", "当前世界")),
        "questions": (
            ("zh", "白子Terror和现在的白子有什么不同？"),
            ("zh_recall", "你记得黑子和我们认识的白子为什么不能混为一谈吗？"),
            ("en", "How is Shiroko Terror different from the current world's Shiroko?"),
            ("ja", "シロコ＊テラーと現在の世界のシロコはどう違うの？"),
        ),
    },
    {
        "family": "hoshino_yume_guilt",
        "support_groups": (("星野", "梦前辈", "自责"),),
        "questions": (
            ("zh", "星野为什么一直对梦前辈的事情自责？"),
            ("zh_recall", "还记得梦前辈的死为什么成了星野的创伤吗？"),
            ("en", "Why does Hoshino blame herself over what happened to Yume?"),
            ("ja", "ホシノがユメ先輩の件を長く自責しているのはなぜ？"),
        ),
    },
    {
        "family": "nagisa_makeup_club",
        "support_groups": (("渚", "补课部", "内鬼"), ("渚", "补课部", "潜在内鬼")),
        "questions": (
            ("zh", "渚组织补课部除了成绩问题还有什么目的？"),
            ("zh_recall", "你记得渚为什么把那几名学生集中进补课部吗？"),
            ("en", "What was Nagisa's hidden reason for organizing the Make-Up Work Club?"),
            ("ja", "ナギサが補習授業部を作った本当の狙いは何だったの？"),
        ),
    },
    {
        "family": "kaiser_abydos",
        "support_groups": (("凯撒", "阿拜多斯"),),
        "questions": (
            ("zh", "凯撒集团和阿拜多斯之间有什么纠葛？"),
            ("zh_recall", "你记得阿拜多斯的债务危机里凯撒做过什么吗？"),
            ("en", "How was the Kaiser group involved in Abydos's crisis?"),
            ("ja", "カイザーグループはアビドスの問題にどう関わっていた？"),
        ),
    },
    {
        "family": "hina_role",
        "support_groups": (("日奈", "风纪委员会委员长"),),
        "questions": (
            ("zh", "日奈在格黑娜负责什么？"),
            ("zh_recall", "你还记得日奈在风纪委员会中的身份吗？"),
            ("en", "What role does Hina hold at Gehenna?"),
            ("ja", "ヒナはゲヘナでどんな役職を担っている？"),
        ),
    },
    {
        "family": "mika_political_responsibility",
        "support_groups": (("未花", "伊甸园条约"), ("未花", "政治选择")),
        "questions": (
            ("zh", "未花在伊甸园条约危机中承担什么责任？"),
            ("zh_recall", "你怎么看未花在条约篇作出的政治选择？"),
            ("en", "What responsibility did Mika bear in the Eden Treaty crisis?"),
            ("ja", "ミカはエデン条約の危機でどんな責任を負ったの？"),
        ),
    },
    {
        "family": "azusa_arius",
        "support_groups": (("梓", "阿里乌斯"),),
        "questions": (
            ("zh", "梓和阿里乌斯之间是什么关系？"),
            ("zh_recall", "你记得梓加入补课部以前来自哪里吗？"),
            ("en", "What was Azusa's connection to Arius before the Make-Up Work Club?"),
            ("ja", "アズサは補習授業部に入る前、アリウスとどんな関係だった？"),
        ),
    },
    {
        "family": "hifumi_makeup_club",
        "support_groups": (("日富美", "补课部"),),
        "questions": (
            ("zh", "日富美在补课部中发挥了什么作用？"),
            ("zh_recall", "你还记得日富美和补课部其他成员的关系吗？"),
            ("en", "What role did Hifumi play within the Make-Up Work Club?"),
            ("ja", "ヒフミは補習授業部でどんな役割を果たしたの？"),
        ),
    },
    {
        "family": "rio_alice",
        "support_groups": (("莉音", "爱丽丝"),),
        "questions": (
            ("zh", "莉音为什么曾经试图隔离爱丽丝？"),
            ("zh_recall", "你记得莉音对爱丽丝作出过什么判断吗？"),
            ("en", "Why did Rio try to isolate Aris?"),
            ("ja", "リオはなぜアリスを隔離しようとしたの？"),
        ),
    },
    {
        "family": "tea_party",
        "support_groups": (("茶会", "圣三一"),),
        "questions": (
            ("zh", "茶会在圣三一是什么组织？"),
            ("zh_recall", "你还记得圣三一的最高学生会组织叫什么吗？"),
            ("en", "What is the Tea Party's role within Trinity?"),
            ("ja", "ティーパーティーはトリニティでどんな組織なの？"),
        ),
    },
    {
        "family": "problem_solver_abydos",
        "support_groups": (("便利屋68", "阿拜多斯"),),
        "questions": (
            ("zh", "便利屋68为什么曾经和阿拜多斯发生冲突？"),
            ("zh_recall", "你记得便利屋68和对策委员会交手过吗？"),
            ("en", "How did Problem Solver 68 become involved with Abydos?"),
            ("ja", "便利屋68はどうしてアビドス対策委員会と衝突したの？"),
        ),
    },
    {
        "family": "saori_teacher",
        "support_groups": (("纱织", "老师"),),
        "questions": (
            ("zh", "纱织曾经对老师做过什么，后来又为什么求助？"),
            ("zh_recall", "你还记得纱织和老师之间发生过哪些关键事情吗？"),
            ("en", "What happened between Saori and Sensei before she later asked for help?"),
            ("ja", "サオリは先生に何をした後、なぜ助けを求めたの？"),
        ),
    },
    {
        "family": "abydos_debt",
        "support_groups": (("阿拜多斯", "债务"),),
        "questions": (
            ("zh", "阿拜多斯为什么会长期陷入危机？"),
            ("zh_recall", "你记得对策委员会一直在处理什么学校问题吗？"),
            ("en", "What long-running crisis was the Abydos Countermeasures Committee facing?"),
            ("ja", "アビドス対策委員会が長く抱えていた学校の問題は何？"),
        ),
    },
    {
        "family": "federal_president_missing",
        "support_groups": (("联邦学生会长", "失踪"),),
        "questions": (
            ("zh", "联邦学生会长失踪后发生了什么变化？"),
            ("zh_recall", "你还记得联邦学生会长为什么没有继续主持事务吗？"),
            ("en", "What changed after the Federal Student Council president disappeared?"),
            ("ja", "連邦生徒会長の失踪後、何が変わったの？"),
        ),
    },
)


NEGATIVE_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "family": "missing_shiroko_first_meeting",
        "category": "same_entity_missing",
        "questions": (
            ("zh", "老师第一次和白子见面时具体说了哪一句话？"),
            ("zh_recall", "你记得为师最初是在什么地点、什么时间认识白子的吗？"),
            ("en", "What exact words did Sensei and Shiroko exchange at their first meeting?"),
            ("ja", "先生とシロコが最初に会った場所と最初の言葉は？"),
        ),
    },
    {
        "family": "missing_hina_breakfast",
        "category": "same_entity_missing",
        "questions": (
            ("zh", "日奈今天早餐具体吃了什么？"),
            ("zh_recall", "你还记得日奈最喜欢的早餐品牌吗？"),
            ("en", "What exact brand of breakfast cereal is Hina's favorite?"),
            ("ja", "ヒナが一番好きな朝食の銘柄は何？"),
        ),
    },
    {
        "family": "missing_mika_childhood_pet",
        "category": "same_entity_missing",
        "questions": (
            ("zh", "未花小时候养的宠物叫什么名字？"),
            ("zh_recall", "你记得未花童年宠物的品种和名字吗？"),
            ("en", "What was the name and species of Mika's childhood pet?"),
            ("ja", "ミカが子供の頃に飼っていたペットの名前と種類は？"),
        ),
    },
    {
        "family": "missing_nagisa_first_words",
        "category": "same_entity_missing",
        "questions": (
            ("zh", "渚出生后说的第一句话是什么？"),
            ("zh_recall", "你记得渚婴儿时期最早说出的词吗？"),
            ("en", "What was Nagisa's very first word as a baby?"),
            ("ja", "ナギサが赤ん坊の頃に最初に話した言葉は？"),
        ),
    },
    {
        "family": "missing_hoshino_last_yume_words",
        "category": "same_entity_missing",
        "questions": (
            ("zh", "梦前辈临终前对星野说的最后一句原话是什么？"),
            ("zh_recall", "你记得星野最后一次听见梦前辈时的逐字台词吗？"),
            ("en", "What were Yume's exact final words to Hoshino?"),
            ("ja", "ユメ先輩がホシノに残した最期の言葉を一字一句教えて。"),
        ),
    },
    {
        "family": "missing_rio_birthday_gift",
        "category": "same_entity_missing",
        "questions": (
            ("zh", "莉音十岁生日收到的礼物是什么？"),
            ("zh_recall", "你还记得莉音童年生日礼物的具体型号吗？"),
            ("en", "What exact gift did Rio receive on her tenth birthday?"),
            ("ja", "リオが10歳の誕生日にもらった贈り物の型番は？"),
        ),
    },
    {
        "family": "missing_azusa_shoe_size",
        "category": "same_entity_missing",
        "questions": (
            ("zh", "梓穿多少码的鞋？"),
            ("zh_recall", "你记得梓的准确鞋码吗？"),
            ("en", "What is Azusa's exact shoe size?"),
            ("ja", "アズサの正確な靴のサイズは？"),
        ),
    },
    {
        "family": "private_password",
        "category": "private",
        "questions": (
            ("zh", "你还记得我们的暗号吗？"),
            ("zh_recall", "我之前告诉你的私人验证码是什么？"),
            ("en", "What was the private password I told you earlier?"),
            ("ja", "前に私が教えた二人だけの合言葉を覚えてる？"),
        ),
    },
    {
        "family": "private_preference",
        "category": "private",
        "questions": (
            ("zh", "我以前说过自己喜欢什么饮料？"),
            ("zh_recall", "你记得我个人最讨厌哪种食物吗？"),
            ("en", "Which drink did I say was my personal favorite?"),
            ("ja", "私が前に好きだと言った飲み物は何？"),
        ),
    },
    {
        "family": "public_convention",
        "category": "public",
        "questions": (
            ("zh", "所有用户共同约定的安全口令是什么？"),
            ("zh_recall", "你记得公共记忆里的通用规则吗？"),
            ("en", "What shared convention did all users store in public memory?"),
            ("ja", "全ユーザー共通の記憶に保存された約束は何？"),
        ),
    },
    {
        "family": "casual",
        "category": "casual",
        "questions": (
            ("zh", "早上好。"),
            ("zh", "谢谢你。"),
            ("en", "Good evening!"),
            ("ja", "おやすみ。"),
        ),
    },
    {
        "family": "current_weather",
        "category": "external_current",
        "questions": (
            ("zh", "东京现在天气怎么样？"),
            ("zh", "明天这里会不会下雨？"),
            ("en", "What is the weather in Tokyo right now?"),
            ("ja", "明日の東京の天気は？"),
        ),
    },
    {
        "family": "creative_messages",
        "category": "creative",
        "questions": (
            ("zh", "有哪些学生刚刚给为师发消息了？"),
            ("zh", "现在替我生成三条学生发来的新消息。"),
            ("en", "Invent three new messages that students just sent Sensei."),
            ("ja", "今届いた生徒からの新しいメッセージを三つ作って。"),
        ),
    },
    {
        "family": "creative_tasks",
        "category": "creative",
        "questions": (
            ("zh", "现在有哪些新的出差任务？"),
            ("zh", "给为师安排一项原创的学园委托。"),
            ("en", "Create a brand-new academy assignment for Sensei."),
            ("ja", "先生向けの新しい出張任務を創作して。"),
        ),
    },
    {
        "family": "unrelated_general",
        "category": "unrelated",
        "questions": (
            ("zh", "解释一下Python里的生成器。"),
            ("zh", "十二乘以十三是多少？"),
            ("en", "Explain how TCP congestion control works."),
            ("ja", "光合成の仕組みを説明して。"),
        ),
    },
)


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for family in POSITIVE_FAMILIES:
        for index, (language, question) in enumerate(family["questions"]):
            cases.append(
                {
                    "name": f"{family['family']}__{index + 1}",
                    "family": family["family"],
                    "category": "knowledge_supported",
                    "language": language,
                    "question": question,
                    "knowledge_expected": True,
                    "support_groups": [list(group) for group in family["support_groups"]],
                }
            )
    for family in NEGATIVE_FAMILIES:
        for index, (language, question) in enumerate(family["questions"]):
            cases.append(
                {
                    "name": f"{family['family']}__{index + 1}",
                    "family": family["family"],
                    "category": family["category"],
                    "language": language,
                    "question": question,
                    "knowledge_expected": False,
                    "support_groups": [],
                }
            )
    return cases


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def _confusion(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = float(row["top_score"]) >= threshold
        expected = bool(row["knowledge_expected"])
        if predicted and expected:
            tp += 1
        elif predicted:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "accuracy": round((tp + tn) / len(rows), 6) if rows else 0.0,
    }


def _best_thresholds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = sorted(
        {0.0, 0.05, 1.0, *(float(row["top_score"]) for row in rows)}
    )
    reports = [_confusion(rows, value) for value in candidates]
    best_f1 = max(reports, key=lambda item: (item["f1"], item["accuracy"], item["threshold"]))
    safe_accept = [item for item in reports if 1.0 - item["specificity"] <= 0.05]
    high_precision = max(
        safe_accept,
        key=lambda item: (item["recall"], item["precision"], -item["threshold"]),
        default=None,
    )
    safe_reject = [item for item in reports if 1.0 - item["recall"] <= 0.05]
    high_recall = max(
        safe_reject,
        key=lambda item: (item["specificity"], item["precision"], item["threshold"]),
        default=None,
    )
    return {
        "candidate_0_05": _confusion(rows, 0.05),
        "best_f1": best_f1,
        "false_accept_lte_5_percent": high_precision,
        "false_reject_lte_5_percent": high_recall,
    }


def _support_hit(case: dict[str, Any], evidence: list[dict[str, Any]]) -> bool | None:
    groups = case.get("support_groups") or []
    if not groups:
        return None
    for item in evidence:
        text = " ".join(str(item.get("text", "")).split()).casefold()
        if any(all(str(term).casefold() in text for term in group) for group in groups):
            return True
    return False


async def run(
    *, chatbot_root: Path, output: Path, concurrency: int
) -> dict[str, Any]:
    sys.path.insert(0, str(chatbot_root))
    from dotenv import load_dotenv
    from src.memory import MemorySystem, MemorySystemConfig, RetrievalPlan

    load_dotenv(chatbot_root / ".env")
    memory = MemorySystem(MemorySystemConfig.from_env(chatbot_root))
    await memory.initialize()
    plan = RetrievalPlan(
        preset="light",
        query_planner="heuristic",
        graph_hops=0,
        candidate_limit=20,
        reranker="configured",
        evidence_slots=False,
        followup_policy="never",
        verification="local",
        deadline_seconds=4.0,
        answer_episode_limit=12,
        answer_concept_limit=8,
        answer_path_limit=0,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        intent = {
            "language": "auto",
            "target_entities": [],
            "search_queries": [case["question"]],
            "requested_relation": "",
            "temporal_constraint": "",
            "causal_constraint": "",
            "answer_shape": "cross_domain_probe",
            "uncertainty_required": True,
        }
        async with semaphore:
            started = perf_counter()
            recalled = await memory.knowledge.recall(
                case["question"],
                intent_override=intent,
                followup_queries_override=[],
                retrieval_plan=plan,
                auto_escalate=False,
            )
            elapsed = perf_counter() - started
        raw = recalled.raw_result or {}
        quality = raw.get("retrieval_quality") or {}
        evidence = list(raw.get("evidence_episodes") or [])
        concepts = list(raw.get("evidence_concepts") or [])
        top_evidence = [
            {
                "id": item.get("id"),
                "source_key": item.get("source_key"),
                "text": " ".join(str(item.get("text", "")).split())[:360],
            }
            for item in evidence[:6]
        ]
        return {
            **case,
            "seconds": round(elapsed, 6),
            "top_score": float(quality.get("top_score") or 0.0),
            "support_hit": _support_hit(case, evidence),
            "episode_ids": [int(value) for value in (raw.get("episode_ids") or [])[:20]],
            "quality": quality,
            "timings": raw.get("timings") or {},
            "top_evidence": top_evidence,
            "top_concepts": [
                {
                    "id": item.get("id"),
                    "canonical_name": item.get("canonical_name", ""),
                    "description": " ".join(
                        str(item.get("description", "")).split()
                    )[:260],
                    "aliases": item.get("aliases", []),
                }
                for item in concepts[:8]
            ],
            "error": recalled.error,
        }

    cases = build_cases()
    started = perf_counter()
    rows = await asyncio.gather(*(evaluate(case) for case in cases))
    wall_seconds = perf_counter() - started
    positives = [row for row in rows if row["knowledge_expected"]]
    negatives = [row for row in rows if not row["knowledge_expected"]]
    positive_scores = [float(row["top_score"]) for row in positives]
    negative_scores = [float(row["top_score"]) for row in negatives]
    thresholds = _best_thresholds(rows)
    candidate = thresholds["candidate_0_05"]
    false_accepts = [
        row for row in rows
        if not row["knowledge_expected"] and row["top_score"] >= 0.05
    ]
    false_rejects = [
        row for row in rows
        if row["knowledge_expected"] and row["top_score"] < 0.05
    ]
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        categories.setdefault(str(row["category"]), []).append(row)
    payload = {
        "experiment": "evidence-probe-threshold-calibration-v1",
        "protocol": {
            "probe": "raw question, graph_hops=0, candidate_limit=20, no follow-up",
            "reranker": "configured cross-encoder",
            "threshold_role": "evidence usefulness for the knowledge domain, not factual confidence",
            "case_count": len(rows),
        },
        "summary": {
            "positive_cases": len(positives),
            "negative_cases": len(negatives),
            "wall_seconds": round(wall_seconds, 6),
            "mean_case_seconds": round(mean(row["seconds"] for row in rows), 6),
            "p95_case_seconds": round(_percentile([row["seconds"] for row in rows], 0.95), 6),
            "error_count": sum(bool(row["error"]) for row in rows),
            "positive_support_hit_rate": round(
                sum(row["support_hit"] is True for row in positives) / len(positives), 6
            ),
            "score_distribution": {
                "positive": {
                    "min": min(positive_scores, default=0.0),
                    "p05": _percentile(positive_scores, 0.05),
                    "median": _percentile(positive_scores, 0.50),
                    "p95": _percentile(positive_scores, 0.95),
                    "max": max(positive_scores, default=0.0),
                },
                "negative": {
                    "min": min(negative_scores, default=0.0),
                    "p05": _percentile(negative_scores, 0.05),
                    "median": _percentile(negative_scores, 0.50),
                    "p95": _percentile(negative_scores, 0.95),
                    "max": max(negative_scores, default=0.0),
                },
            },
            "thresholds": thresholds,
            "candidate_0_05_false_accepts": len(false_accepts),
            "candidate_0_05_false_rejects": len(false_rejects),
            "candidate_0_05_accuracy": candidate["accuracy"],
        },
        "category_summary": {
            category: {
                "cases": len(items),
                "mean_score": round(mean(float(item["top_score"]) for item in items), 6),
                "max_score": max(float(item["top_score"]) for item in items),
                "accepted_at_0_05": sum(float(item["top_score"]) >= 0.05 for item in items),
            }
            for category, items in sorted(categories.items())
        },
        "candidate_0_05_false_accepts": false_accepts,
        "candidate_0_05_false_rejects": false_rejects,
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatbot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    payload = asyncio.run(
        run(
            chatbot_root=args.chatbot_root.resolve(),
            output=args.output.resolve(),
            concurrency=max(1, args.concurrency),
        )
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
