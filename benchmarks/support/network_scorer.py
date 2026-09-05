from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("validation/stage4-network-evidence-manifest.json")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _term_group_check(answer: str, alternatives: list[str]) -> dict[str, Any]:
    matched = [term for term in alternatives if term in answer]
    return {
        "alternatives": alternatives,
        "matched": matched,
        "passed": bool(matched),
    }


def score_result(
    question_id: str,
    result: dict[str, Any],
    mode: str,
    status: str,
    criterion: dict[str, Any],
    allowed_sources: set[str],
) -> dict[str, Any]:
    episode_ids = {int(value) for value in result.get("episode_ids", [])}
    evidence_sources = {
        str(row.get("source_key", ""))
        for row in result.get("evidence_episodes", [])
        if str(row.get("source_key", ""))
    }
    answer = str(result.get("answer", ""))
    audits = result.get("answer_audits", [])
    last_audit_valid = bool(audits and audits[-1].get("valid") is True)
    changed_ids = {
        *[int(value) for value in result.get("new_association_ids", [])],
        *[int(value) for value in result.get("reinforced_association_ids", [])],
    }
    used_ids = {int(value) for value in result.get("association_ids", [])}
    if mode == "vector_only":
        mode_behavior_passed = not used_ids
    elif mode == "graph_static":
        mode_behavior_passed = bool(used_ids)
    elif mode == "graph_growing":
        # Growth quantity is a diagnostic target, not a hard correctness gate.
        # The graph must participate, and every changed edge must be used when
        # changes do occur, but a safe zero-write query may still pass.
        mode_behavior_passed = bool(used_ids) and changed_ids.issubset(used_ids)
    else:
        raise ValueError(f"unknown mode: {mode}")

    episode_groups = [
        {
            "alternatives": group,
            "matched": sorted(episode_ids.intersection(group)),
            "passed": bool(episode_ids.intersection(group)),
        }
        for group in criterion["required_episode_groups"]
    ]
    source_checks = {
        source: source in evidence_sources
        for source in criterion["required_sources"]
    }
    source_names_in_answer = {
        source: source in answer for source in criterion["required_sources"]
    }
    cited_evidence_sources = sorted(
        source for source in evidence_sources if source in answer
    )
    traceable_source_citation = bool(cited_evidence_sources)
    has_evidence_sources = bool(evidence_sources)
    term_checks = {
        term: term in answer for term in criterion["required_terms"]
    }
    term_group_checks = [
        _term_group_check(answer, group)
        for group in criterion.get("required_term_groups", [])
    ]
    unexpected_sources = sorted(evidence_sources.difference(allowed_sources))
    changed_edges_used = changed_ids.issubset(used_ids)
    episode_group_closure = all(item["passed"] for item in episode_groups)
    required_source_closure = all(source_checks.values())
    term_group_closure = all(item["passed"] for item in term_group_checks)
    passed = (
        status == "completed"
        and has_evidence_sources
        and all(term_checks.values())
        and not unexpected_sources
        and last_audit_valid
        and changed_edges_used
        and mode_behavior_passed
    )
    return {
        "passed": passed,
        "last_answer_audit_valid": last_audit_valid,
        "answer_revision_count": int(result.get("answer_revision_count", 0)),
        "episode_groups": episode_groups,
        "episode_group_closure": episode_group_closure,
        "matched_group_count": sum(item["passed"] for item in episode_groups),
        "required_group_count": len(episode_groups),
        "required_sources": source_checks,
        "required_source_closure": required_source_closure,
        "required_source_names_in_answer": source_names_in_answer,
        "cited_evidence_sources": cited_evidence_sources,
        "traceable_source_citation": traceable_source_citation,
        "has_evidence_sources": has_evidence_sources,
        "evidence_sources": sorted(evidence_sources),
        "unexpected_evidence_sources": unexpected_sources,
        "required_terms": term_checks,
        "required_term_groups": term_group_checks,
        "term_group_closure": term_group_closure,
        "atomic_anchor_episode_ids": result.get("atomic_anchor_episode_ids", []),
        "new_association_ids": result.get("new_association_ids", []),
        "reinforced_association_ids": result.get("reinforced_association_ids", []),
        "changed_associations_used_in_answer_paths": changed_edges_used,
        "mode_behavior_passed": mode_behavior_passed,
    }


def score_report(
    report: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    criteria = manifest["questions"]
    allowed_sources = set(manifest["corpus_boundary"]["allowed_source_keys"])
    mode_scores: dict[str, list[dict[str, Any]]] = {}
    for mode in report["mode_order"]:
        rows: list[dict[str, Any]] = []
        for item in report["modes"].get(mode, []):
            question_id = str(item["id"])
            if question_id not in criteria:
                raise KeyError(f"question missing from manifest: {question_id}")
            rows.append(
                {
                    "question_id": question_id,
                    "status": item["status"],
                    **score_result(
                        question_id,
                        item.get("result", {}),
                        mode,
                        item["status"],
                        criteria[question_id],
                        allowed_sources,
                    ),
                }
            )
        mode_scores[mode] = rows
    all_rows = [row for rows in mode_scores.values() for row in rows]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": report.get("configuration", {}).get("prompt_version"),
        "manifest_version": manifest.get("version"),
        "difficulty_target": manifest.get("difficulty_target"),
        "passed": bool(all_rows) and all(row["passed"] for row in all_rows),
        "passed_count": sum(bool(row["passed"]) for row in all_rows),
        "total_count": len(all_rows),
        "modes": mode_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score the Stage 4 network-level evidence evaluation."
    )
    parser.add_argument("evaluation_report", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = _load_json(args.evaluation_report)
    manifest = _load_json(args.manifest)
    output = {
        "evaluation_report": str(args.evaluation_report),
        "manifest": str(args.manifest),
        **score_report(report, manifest),
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
