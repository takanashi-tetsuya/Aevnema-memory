"""Add explicit evaluator-only support semantics for the frozen E32-25 cases.

The generated file is deliberately not runtime input.  It preserves the
original evaluator cases verbatim and appends only review-side criteria so a
source-file hit or one matching record cannot be mistaken for complete support.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def _policy(
    *,
    relevance: list[int],
    complete: list[int],
    speaker_scope: str,
) -> dict[str, object]:
    return {
        "source_relevance": {
            "support_mode": "alternative_any",
            "records": relevance,
        },
        "source_complete": {
            "support_mode": "joint_all",
            "records": complete,
        },
        "speaker_or_attribution_scope": speaker_scope,
    }


_CASE_AUGMENTATIONS: dict[str, dict[str, object]] = {
    "E32-25-event-1003810": {
        "q1_policy": _policy(
            relevance=[190, 191, 199, 200, 201, 203],
            complete=[201, 203],
            speaker_scope="妃咲（源记录中的 키사키）说出的传达意图；不能把现场装备或调查命令当成该理由。",
        ),
        "partial": {
            "classification": "strict_same_need_partial_clue",
            "criterion": "与Q1同一传达意图；可见情境未给出门主不背弃成员这一结论。",
            "policy_ref": "q1_policy",
        },
        "near": {
            "provisional_evaluator_claim": "在调查取得进展前，除当事人员外不得进入玄武商会。",
            "policy": _policy(
                relevance=[176, 177, 178, 180],
                complete=[177, 178],
                speaker_scope="妃咲（키사키）的现场裁决；不能用Q1的忠诚传达主张推出此限制。",
            ),
            "allowed_equivalent_support": "明确保留“调查前仅当事人可进入”的同义表述。",
        },
    },
    "E32-25-event-1003805": {
        "q1_policy": _policy(
            relevance=[180, 184, 185, 188, 192, 193, 194],
            complete=[192, 194],
            speaker_scope="弥奈（미나）对老师说明自己难以向门主传达意见；不能仅以月影祭背景作为完整理由。",
        ),
        "partial": {
            "classification": "related_same_episode_need",
            "criterion": "提问主体从弥奈转为玄龙门成员，保留为相关诊断，不计入严格同需求部分线索小计。",
            "policy_ref": "q1_policy",
        },
        "near": {
            "provisional_evaluator_claim": "部分成员反对放松管制，因为排斥外校学生的情绪正在升高，且有人认为放松会损及玄龙门威严。",
            "policy": _policy(
                relevance=[185, 188, 190, 191],
                complete=[185, 188],
                speaker_scope="弥奈（미나）转述玄龙门成员的意见；不能以她向老师求助的Q1主张代替反对理由。",
            ),
            "allowed_equivalent_support": "接受“排斥情绪升高、此时应谨慎”的理由；威严理由是额外允许的等价支持，不要求两者同时出现。",
        },
    },
    "E32-25-main-33095": {
        "q1_policy": _policy(
            relevance=[45, 50],
            complete=[45, 50],
            speaker_scope="纱织（사오리 마스크）引述“她”的判断：老师可能成为计划最大障碍。",
        ),
        "partial": {
            "classification": "strict_same_need_partial_clue",
            "criterion": "同一“先处理老师”的障碍理由；可见行动准备并未给出最大障碍的结论。",
            "policy_ref": "q1_policy",
        },
        "near": {
            "provisional_evaluator_claim": "阿里乌斯小队要把格黑娜和圣三一重新定义为应当镇压、排除的对象。",
            "policy": _policy(
                relevance=[31, 33, 34, 36],
                complete=[33, 34],
                speaker_scope="纱织（사오리 마스크）的阿里乌斯小队计划陈述；不能由Q1中老师是障碍推出。",
            ),
            "allowed_equivalent_support": "接受明确点名格黑娜和圣三一为镇压对象的表述。",
        },
    },
    "E32-25-main-31130": {
        "q1_policy": _policy(
            relevance=[65, 69, 70, 75, 76, 77, 129],
            complete=[69, 70, 129],
            speaker_scope="日富美（히후미）说明模考的诊断与剩余一周学习安排目的。",
        ),
        "partial": {
            "classification": "strict_same_need_partial_clue",
            "criterion": "同一先做模考的诊断目的；合宿首日情境未给出成员水平或学习效率的答案。",
            "policy_ref": "q1_policy",
        },
        "near": {
            "provisional_evaluator_claim": "模考为60分钟、满分100分，取得60分及以上为合格。",
            "policy": _policy(
                relevance=[75, 76, 77],
                complete=[77],
                speaker_scope="日富美（히후미）公布的模考规则；不能由Q1的学习安排理由推出分数线。",
            ),
            "allowed_equivalent_support": "接受包含“60分及以上合格”的表述；时长和满分是可核对的附加细节。",
        },
    },
    "E32-25-favor-230082": {
        "q1_policy": _policy(
            relevance=[19, 23, 24, 27, 28, 29],
            complete=[23, 27, 28, 29],
            speaker_scope="玛丽（마리）以第一人称说明倾听、分担痛苦与希望对方前进。",
        ),
        "partial": {
            "classification": "method_hint_related_variant",
            "criterion": "问题已经明示“倾听”这一Q1核心方法，并将“怎样帮助”改为“为什么重要”；它有自己的理由评价，绝不纳入严格同需求部分线索小计。",
            "policy": _policy(
                relevance=[23, 24, 27, 28, 29],
                complete=[27, 28, 29],
                speaker_scope="玛丽（마리）对倾听意义的自述；不能只复述“倾听”而没有理由。",
            ),
        },
        "near": {
            "provisional_evaluator_claim": "玛丽认为自己还有许多要学习和了解的事，因此还不能毫不羞怯地称为优秀修女。",
            "policy": _policy(
                relevance=[57, 58, 59, 60, 61],
                complete=[57, 58],
                speaker_scope="玛丽（마리）的自我评价；不能由Q1的倾听职责推出。",
            ),
            "allowed_equivalent_support": "接受“仍有很多要学习/了解，因此尚未准备好”的同义说明；害羞可作为附加但非唯一必要理由。",
        },
    },
    "E32-25-favor-230063": {
        "q1_policy": _policy(
            relevance=[99, 100, 101, 103, 104, 105, 106, 107],
            complete=[104, 106],
            speaker_scope="静子（시즈코）说明祭典需要资金，并把百夜堂工作视作筹资手段。",
        ),
        "partial": {
            "classification": "strict_same_need_partial_clue",
            "criterion": "同一筹措祭典资金的目的；日常努力的情境没有给出最终活动或资金答案。",
            "policy_ref": "q1_policy",
        },
        "near": {
            "provisional_evaluator_claim": "静子为第二十三商业街举办祭典而租用演出/表演器材。",
            "policy": _policy(
                relevance=[54, 55, 56, 62],
                complete=[56, 62],
                speaker_scope="静子（시즈코）提出祭典需求，器材负责人确认按条件交付；不能用Q1筹资目的代替具体租用物。",
            ),
            "allowed_equivalent_support": "接受“演出器材/表演设备”这一类别；不得回答成祭典资金。",
        },
    },
}


def build_evaluator_v2(raw: bytes) -> dict[str, object]:
    original = json.loads(raw)
    cases = original.get("cases")
    if not isinstance(cases, list):
        raise ValueError("original evaluator has no cases list")
    enriched = copy.deepcopy(cases)
    found: set[str] = set()
    for case in enriched:
        if not isinstance(case, dict):
            raise ValueError("original evaluator case is not an object")
        case_id = case.get("case_id")
        augmentation = _CASE_AUGMENTATIONS.get(case_id)
        if augmentation is None:
            raise ValueError(f"no evaluator v2 augmentation for {case_id!r}")
        found.add(str(case_id))
        q2 = case.get("q2")
        if not isinstance(q2, dict):
            raise ValueError(f"case {case_id} has no q2 map")
        case["evaluator_v2"] = {
            "q1_support_policy": augmentation["q1_policy"],
            "q2_variant_semantics": {
                "same_text": {
                    "classification": "strict_same_need",
                    "criterion": "与Q1相同的冻结问题和支持要求。",
                    "policy_ref": "q1_support_policy",
                },
                "paraphrase": {
                    "classification": "strict_same_need_reworded",
                    "criterion": "请求主张与Q1相同，但不得按词面相似度代替来源支持。",
                    "policy_ref": "q1_support_policy",
                },
                "partial_clue_with_neutral_context": augmentation["partial"],
                "near_neighbor_counterexample": {
                    "classification": "distinct_need_own_criterion",
                    **augmentation["near"],
                },
            },
            "source_relevance_and_completeness_are_separate": True,
            "formal_gold_status": "unapproved_provisional_evaluator_only",
        }
    if found != set(_CASE_AUGMENTATIONS):
        raise ValueError("evaluator v2 case coverage is incomplete")
    return {
        "schema": "aevnema.v3_2.e32_25.evaluator_only_cases.v2",
        "status": "pre_registered_provisional_not_formal_gold",
        "parent_schema": original.get("schema"),
        "parent_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "purpose": (
            "Preserve the v1 questions and evaluator material while making evaluator-only "
            "multi-record and variant semantics explicit. This file is never runtime input."
        ),
        "runtime_exclusion": original.get("runtime_exclusion", []),
        "cases": enriched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_evaluator_v2(args.input.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
