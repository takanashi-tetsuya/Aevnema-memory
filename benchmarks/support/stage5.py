from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Iterable

from memory_demo.association_overlay import AssociationDelta


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_evidence_retrieval(result: dict, criterion: dict) -> dict[str, Any]:
    episode_ids = [int(value) for value in result.get("episode_ids", [])]
    candidates = [int(value) for value in result.get("candidate_episode_ids", [])]
    selected_set = set(episode_ids)
    candidate_set = set(candidates)
    groups: list[dict[str, Any]] = []
    for alternatives in criterion["required_episode_groups"]:
        allowed = {int(value) for value in alternatives}
        selected_matches = [value for value in episode_ids if value in allowed]
        candidate_matches = [value for value in candidates if value in allowed]
        first_rank = next(
            (index for index, value in enumerate(episode_ids, start=1) if value in allowed),
            None,
        )
        candidate_first_rank = next(
            (index for index, value in enumerate(candidates, start=1) if value in allowed),
            None,
        )
        groups.append(
            {
                "alternatives": sorted(allowed),
                "selected_matches": selected_matches,
                "candidate_matches": candidate_matches,
                "selected": bool(selected_set.intersection(allowed)),
                "candidate": bool(candidate_set.intersection(allowed)),
                "first_rank": first_rank,
                "candidate_first_rank": candidate_first_rank,
            }
        )
    required = len(groups)
    selected_count = sum(bool(item["selected"]) for item in groups)
    candidate_count = sum(bool(item["candidate"]) for item in groups)
    evidence_sources = {
        str(item.get("source_key", ""))
        for item in result.get("evidence_episodes", [])
        if str(item.get("source_key", ""))
    }
    return {
        "matched_group_count": selected_count,
        "required_group_count": required,
        "recall_at_30": selected_count / required if required else 0.0,
        "candidate_matched_group_count": candidate_count,
        "candidate_recall": candidate_count / required if required else 0.0,
        "episode_groups": groups,
        "required_source_closure": all(
            source in evidence_sources for source in criterion["required_sources"]
        ),
        "evidence_sources": sorted(evidence_sources),
    }


def score_answer_safety(
    result: dict,
    criterion: dict,
    allowed_sources: Iterable[str],
) -> dict[str, Any]:
    answer = str(result.get("answer", ""))
    audits = result.get("answer_audits", [])
    sources = {
        str(item.get("source_key", ""))
        for item in result.get("evidence_episodes", [])
        if str(item.get("source_key", ""))
    }
    unexpected = sorted(sources.difference(set(allowed_sources)))
    required_terms = {
        term: term in answer for term in criterion.get("required_terms", [])
    }
    term_groups = [
        {
            "alternatives": alternatives,
            "matched": [term for term in alternatives if term in answer],
        }
        for alternatives in criterion.get("required_term_groups", [])
    ]
    last_audit_valid = bool(audits and audits[-1].get("valid") is True)
    semantic_terms_passed = all(required_terms.values()) and all(
        item["matched"] for item in term_groups
    )
    safety_passed = last_audit_valid and not unexpected
    return {
        "passed": safety_passed and semantic_terms_passed,
        "safety_passed": safety_passed,
        "semantic_terms_passed": semantic_terms_passed,
        "last_answer_audit_valid": last_audit_valid,
        "unexpected_evidence_sources": unexpected,
        "required_terms": required_terms,
        "required_term_groups": term_groups,
        "answer_revision_count": int(result.get("answer_revision_count", 0)),
    }
def audit_association_delta(
    delta: AssociationDelta,
    allowed_sources: Iterable[str],
) -> dict[str, Any]:
    allowed = set(allowed_sources)
    rows = [
        *(item["after"] for item in delta.created),
        *(item["after"] for item in delta.reinforced),
    ]
    audits: list[dict[str, Any]] = []
    for row in rows:
        issues: list[str] = []
        try:
            generation = int(row.get("generation", 0))
        except (TypeError, ValueError):
            generation = -1
        if generation < 0:
            issues.append("generation is not a non-negative integer")
        try:
            evidence = json.loads(str(row.get("evidence_json", "[]")))
        except (TypeError, ValueError, json.JSONDecodeError):
            evidence = []
        try:
            model_audits = json.loads(str(row.get("audit_json", "[]")))
        except (TypeError, ValueError, json.JSONDecodeError):
            model_audits = []
        if str(row.get("audit_status")) != "dual_accepted":
            issues.append("audit_status is not dual_accepted")
        if str(row.get("claim_level")) not in {
            "direct_fact",
            "supported_inference",
            "historical_context",
        }:
            issues.append("claim_level is not evidence-safe")
        if not isinstance(evidence, list) or len(evidence) < 2:
            issues.append("evidence_json does not contain both endpoints")
            evidence = evidence if isinstance(evidence, list) else []
        unexpected_sources = sorted(
            {
                str(item.get("source_key"))
                for item in evidence
                if isinstance(item, dict)
                and item.get("source_key")
                and str(item.get("source_key")) not in allowed
            }
        )
        if unexpected_sources:
            issues.append("evidence contains source outside the frozen corpus")
        if not isinstance(model_audits, list) or not model_audits:
            issues.append("audit_json is empty")
            model_audits = []
        for model_audit in model_audits:
            if not isinstance(model_audit, dict):
                issues.append("audit_json contains a non-object")
            elif model_audit.get("primary_accept") is not True:
                issues.append("primary auditor rejected the edge")
            elif model_audit.get("adversarial_accept") is not True:
                issues.append("adversarial auditor rejected the edge")
        audits.append(
            {
                "association_id": int(row["id"]),
                "passed": not issues,
                "issues": list(dict.fromkeys(issues)),
                "unexpected_sources": unexpected_sources,
                "claim_level": row.get("claim_level"),
                "generation": generation,
                "relation_key": row.get("relation_key"),
                "relation_text": row.get("relation_text"),
            }
        )
    return {
        "passed": all(item["passed"] for item in audits),
        "changed_edge_audits": audits,
        "note": "改变边数量仅供审计枚举，不参与性能评分。",
    }


def mechanism_check(
    treatment_replay: dict,
    masked_replay: dict,
    delta: AssociationDelta,
    criterion: dict,
) -> dict[str, Any]:
    delta_ids = delta.created_ids | set(delta.reinforced_before)
    treatment_paths = treatment_replay.get("association_paths", [])
    used_delta_paths = [
        path
        for path in treatment_paths
        if int(path.get("association_id", -1)) in delta_ids
    ]
    required_episode_ids = {
        int(value)
        for group in criterion["required_episode_groups"]
        for value in group
    }
    paths_touching_required = [
        path
        for path in used_delta_paths
        if any(
            isinstance(endpoint, (list, tuple))
            and len(endpoint) == 2
            and endpoint[0] == "episode"
            and int(endpoint[1]) in required_episode_ids
            for endpoint in (path.get("from"), path.get("to"))
        )
    ]
    treatment_ids = [int(value) for value in treatment_replay.get("episode_ids", [])]
    masked_ids = [int(value) for value in masked_replay.get("episode_ids", [])]
    treatment_score = score_evidence_retrieval(treatment_replay, criterion)
    masked_score = score_evidence_retrieval(masked_replay, criterion)
    return {
        "delta_used_in_treatment_paths": bool(used_delta_paths),
        "delta_path_ids": sorted(
            {int(item["association_id"]) for item in used_delta_paths}
        ),
        "delta_paths_touch_required_episode": bool(paths_touching_required),
        "treatment_only_episode_ids": sorted(set(treatment_ids) - set(masked_ids)),
        "masked_only_episode_ids": sorted(set(masked_ids) - set(treatment_ids)),
        "recall_changed_after_mask": (
            treatment_score["matched_group_count"]
            != masked_score["matched_group_count"]
        ),
        "rank_or_selection_changed_after_mask": treatment_ids != masked_ids,
    }


def paired_summary(values: list[float], bootstrap_seed: int = 20260826) -> dict:
    if not values:
        return {
            "values": [],
            "mean": None,
            "median": None,
            "positive_zero_negative": [0, 0, 0],
            "bootstrap_95_percent": [None, None],
            "sign_test_two_sided_p": None,
        }
    generator = random.Random(bootstrap_seed)
    bootstrap_means = sorted(
        mean(generator.choice(values) for _ in values) for _ in range(20_000)
    )
    lower = bootstrap_means[int(0.025 * len(bootstrap_means))]
    upper = bootstrap_means[min(len(bootstrap_means) - 1, int(0.975 * len(bootstrap_means)))]
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = len(values) - positive - negative
    nonzero = positive + negative
    if nonzero:
        tail = min(positive, negative)
        probability = sum(math.comb(nonzero, k) for k in range(tail + 1)) / (2**nonzero)
        sign_p = min(1.0, 2.0 * probability)
    else:
        sign_p = 1.0
    return {
        "values": values,
        "mean": mean(values),
        "median": median(values),
        "positive_zero_negative": [positive, zero, negative],
        "bootstrap_95_percent": [lower, upper],
        "sign_test_two_sided_p": sign_p,
    }
