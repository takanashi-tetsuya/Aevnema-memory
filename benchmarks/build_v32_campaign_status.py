"""Create the non-scoring v3.2 campaign baseline and evidence-status report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Mapping


CAMPAIGN_SCHEMA = "aevnema.v3_2.campaign_state.v1"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected object: {path}")
    return loaded


def _write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _provider_counts(record: object) -> dict[str, object]:
    if not isinstance(record, Mapping):
        return {"http_attempts": "not_observed"}
    provider = record.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    counts = provider.get("counts")
    return dict(counts) if isinstance(counts, Mapping) else {"http_attempts": "not_observed"}


def _matrix_summary(path: Path) -> dict[str, object]:
    payload = _load(path)
    arms = payload.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    compact_arms: list[dict[str, object]] = []
    for arm_id, raw_arm in arms.items():
        if not isinstance(raw_arm, Mapping):
            continue
        summary = raw_arm.get("summary")
        summary = summary if isinstance(summary, Mapping) else {}
        record = raw_arm.get("record")
        record = record if isinstance(record, Mapping) else {}
        result = record.get("result")
        result = result if isinstance(result, Mapping) else {}
        evidence = result.get("evidence_result")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        participation = evidence.get("participation")
        participation = participation if isinstance(participation, Mapping) else {}
        selector = result.get("evidence_slot_trace")
        selector = selector if isinstance(selector, Mapping) else {}
        selector_v3 = selector.get("slot_selector_v3")
        selector_v3 = selector_v3 if isinstance(selector_v3, Mapping) else {}
        compact_arms.append(
            {
                "arm": str(arm_id),
                "run_status": summary.get("run_status"),
                "execution_profile": summary.get("execution_profile"),
                "edge_state": summary.get("this_run_edge_participated"),
                "edge_entry_reason": participation.get("entry_reason", "not_recorded_in_this_version"),
                "selector_reason": selector_v3.get("reason", "not_recorded_in_this_version"),
                "delivered_episode_ids": summary.get("delivered_episode_ids", []),
                "evidence_state": summary.get("evidence_state"),
                "q2_http_attempts": (
                    summary.get("provider_stages", {})
                    .get("evidence_acquisition", {})
                    .get("http_attempts", "not_observed")
                    if isinstance(summary.get("provider_stages"), Mapping)
                    else "not_observed"
                ),
                "evidence_acquisition_ms": summary.get("evidence_acquisition_ms"),
                "answer_status": summary.get("ordinary_answer_status"),
                "actual_edge_path": summary.get("actual_edge_path"),
                "full_local_arm": f"{path.parent.name}/q2_{arm_id}.full_local.json",
            }
        )
    preparation = payload.get("input_preparation")
    preparation = preparation if isinstance(preparation, Mapping) else {}
    preparation_records = preparation.get("records")
    preparation_records = preparation_records if isinstance(preparation_records, Mapping) else {}
    return {
        "artifact": str(path),
        "sha256": _sha(path),
        "status": payload.get("status"),
        "snapshot_unchanged": payload.get("snapshot_unchanged_after_all_arms"),
        "input_preparation_mode": preparation.get(
            "mode",
            "live_input_per_condition"
            if payload.get("execution_profile") == "public_query_stop_after_evidence"
            else "not_observed",
        ),
        "preparation": {
            str(key): {
                "status": value.get("status") if isinstance(value, Mapping) else "not_observed",
                "provider_counts": _provider_counts(value),
                "plan_sha256": value.get("plan_sha256") if isinstance(value, Mapping) else None,
                "vector_mode": (
                    value.get("vector_material", {}).get("mode")
                    if isinstance(value, Mapping) and isinstance(value.get("vector_material"), Mapping)
                    else "not_observed"
                ),
            }
            for key, value in preparation_records.items()
        },
        "arms": compact_arms,
    }


def _cache_control_summary(path: Path) -> dict[str, object]:
    payload = _load(path)
    controls = payload.get("controls")
    controls = controls if isinstance(controls, Mapping) else {}
    invalidation = payload.get("source_change_invalidation")
    invalidation = invalidation if isinstance(invalidation, Mapping) else {}
    return {
        "artifact": str(path),
        "sha256": _sha(path),
        "status": payload.get("status", "not_observed"),
        "controls": {
            str(name): {
                "status": item.get("status", "not_observed"),
                "delivered_episode_ids": item.get("delivered_episode_ids", []),
                "evidence_acquisition_ms": item.get("evidence_acquisition_ms"),
                "provider_counts": item.get("provider_counts", {}),
                "comparison_status": item.get("comparison_status", "not_observed"),
            }
            for name, item in controls.items()
            if isinstance(item, Mapping)
        },
        "source_change_invalidation": dict(invalidation),
        "fairness_boundary": payload.get("fairness_boundary", {}),
    }


def _q1_trial_summary(path: Path) -> dict[str, object]:
    """Compact a preserved fresh-Q1 trial without inferring missing stages.

    These trials are deliberately separate from the older same-chapter Q2
    matrices.  In particular, a failed Q1 has no eligible Q2 arms; reporting
    its in-progress checkpoint as a zero-result arm would erase the useful
    distinction between ``not_run`` and ``failed``.
    """

    payload = _load(path)
    q1 = payload.get("q1")
    q1 = q1 if isinstance(q1, Mapping) else {}
    record = q1.get("record")
    record = record if isinstance(record, Mapping) else {}
    result = record.get("result")
    result = result if isinstance(result, Mapping) else {}
    error = record.get("error")
    error = error if isinstance(error, Mapping) else {}
    finalizer = q1.get("finalizer_observer")
    finalizer = finalizer if isinstance(finalizer, Mapping) else {}
    material = q1.get("new_material")
    material = material if isinstance(material, Mapping) else {}
    terminal_state = result.get("answer_terminal_state")
    terminal_state = terminal_state if isinstance(terminal_state, Mapping) else {}
    modes: list[str] = []
    reasons: list[str] = []
    answer_input_checkpoint = "not_observed"
    evidence = result.get("evidence_episodes")
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            mode = item.get("source_evidence_delivery")
            reason = item.get("source_evidence_delivery_reason")
            if isinstance(mode, str) and mode not in modes:
                modes.append(mode)
            if isinstance(reason, str) and reason not in reasons:
                reasons.append(reason)
    # Newer answer-boundary checkpoints carry the actual materialized Source
    # views.  Older result objects retain only Episode summaries, so consult
    # the local-only event file when it exists; the compact campaign state
    # stores modes/reasons only, never the evidence body or answer.
    checkpoint_logs = sorted((path.parent / "logs" / "q1").glob("*.answer-evidence.jsonl"))
    for checkpoint_log in checkpoint_logs:
        for raw_line in checkpoint_log.read_text(encoding="utf-8").splitlines():
            try:
                checkpoint = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(checkpoint, Mapping) or checkpoint.get("event") != "answer_input_checkpoint":
                continue
            answer_input_checkpoint = "observed"
            selected = checkpoint.get("selected_evidence")
            if not isinstance(selected, list):
                continue
            for item in selected:
                if not isinstance(item, Mapping):
                    continue
                mode = item.get("source_evidence_delivery")
                reason = item.get("source_evidence_delivery_reason")
                if isinstance(mode, str) and mode not in modes:
                    modes.append(mode)
                if isinstance(reason, str) and reason not in reasons:
                    reasons.append(reason)
    arms = payload.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    arm_states = sorted(
        {
            str(summary.get("run_status"))
            for arm in arms.values()
            if isinstance(arm, Mapping)
            and isinstance((summary := arm.get("summary")), Mapping)
            and summary.get("run_status") is not None
        }
    )
    source = payload.get("source_database")
    source = source if isinstance(source, Mapping) else {}
    case = payload.get("case")
    case = case if isinstance(case, Mapping) else {}
    return {
        "artifact": str(path),
        "sha256": _sha(path),
        "status": payload.get("status"),
        "q1_status": record.get("status", "not_observed"),
        "q1_error": {
            "type": error.get("type", "not_observed"),
            "message": error.get("message", "not_observed"),
        },
        "q1_provider_counts": _provider_counts(record),
        "learning_status": q1.get("learning_status", "not_observed"),
        "learning_reason": q1.get("learning_reason", "not_observed"),
        "public_finalizer_calls": finalizer.get("finalizer_calls", "not_observed"),
        "new_receipt_ids": material.get("new_receipt_ids", "not_observed"),
        "new_association_ids": material.get("new_association_ids", "not_observed"),
        "runtime_manifest_status": (
            "not_created" if not material.get("new_receipt_ids") else "inspect_receipt"
        ),
        "answer_terminal_state": dict(terminal_state),
        "answer_input_checkpoint": answer_input_checkpoint,
        "source_evidence_delivery_modes": modes or ["not_observed"],
        "source_evidence_delivery_reasons": reasons or ["not_observed"],
        "q2_arm_states": arm_states or ["not_observed"],
        "source_database": {
            "path": source.get("path", "not_observed"),
            "sha256": source.get("sha256", "not_observed"),
        },
        "q1_text": case.get("q1_text", "not_observed"),
        "formal_scoring": "not_scored_unapproved_gold",
    }


def build_campaign(*, root: Path, campaign_dir: Path) -> dict[str, object]:
    root = root.resolve()
    campaign_dir = campaign_dir.resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    relevant_files = [
        root / "src/memory_demo/retrieval/engine.py",
        root / "benchmarks/run_frozen_evidence_matrix.py",
        root / "benchmarks/run_same_chapter_learning_matrix.py",
        root / "benchmarks/run_q1_q2_diagnostic_pilot.py",
        root / "benchmarks/control_cache.py",
        root / "benchmarks/run_source_validated_cache_control.py",
        root / "benchmarks/build_v32_narrowed_source_review.py",
        root / "benchmarks/build_v32_full_corpus_disposition.py",
        root / "benchmarks/build_v32_cross_topic_report.py",
        root / "benchmarks/build_v32_final_reports.py",
        root / "benchmarks/run_v32_local_regression.py",
        root / "benchmarks/package_v32_campaign.py",
        root / "benchmarks/build_v32_closure_status.py",
        root / "config/prompt_config/memory_prompts.py",
        root / "benchmarks/manifests/v3_same_chapter_different_requirement_matrix.json",
    ]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    ).stdout.strip() or "not_observed"
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=root, text=True, capture_output=True, check=False
    ).stdout.splitlines()
    experiment_root = campaign_dir / "experiments"
    experiment_paths = {
        "E32-02_live_input_observation": experiment_root / "E32-02_frozen_evidence_matrix" / "same_chapter_learning_matrix.full_local.json",
        "E32-02_frozen_plan_without_runtime_vectors": experiment_root / "E32-03_frozen_input_evidence_matrix" / "same_chapter_learning_matrix.full_local.json",
        "E32-02_frozen_plan_with_unbound_slots": experiment_root / "E32-04_frozen_input_vectors_matrix" / "same_chapter_learning_matrix.full_local.json",
        "E32-02_restored_slot_vector_observation": experiment_root / "E32-05_restored_frozen_vectors_matrix" / "same_chapter_learning_matrix.full_local.json",
    }
    experiments = {
        label: _matrix_summary(path) if path.is_file() else {"status": "not_observed", "artifact": str(path)}
        for label, path in experiment_paths.items()
    }
    cache_control_paths = {
        "E32-04c_first_source_change_attempt": experiment_root / "E32-04c_source_validated_cache_control" / "source_validated_cache_control.full_local.json",
        "E32-04d_source_validated_cache_control": experiment_root / "E32-04d_source_validated_cache_control" / "source_validated_cache_control.full_local.json",
    }
    cache_controls = {
        label: _cache_control_summary(path) if path.is_file() else {"status": "not_observed", "artifact": str(path)}
        for label, path in cache_control_paths.items()
    }
    q1_trial_paths = {
        "event_moon_festival": experiment_root / "E32-05a_event_q1" / "same_chapter_learning_matrix.full_local.json",
        "main_rest": experiment_root / "E32-05b_main_q1" / "same_chapter_learning_matrix.full_local.json",
        "favor_competition": experiment_root / "E32-05c_favor_q1" / "same_chapter_learning_matrix.full_local.json",
    }
    q1_trials = {
        label: _q1_trial_summary(path) if path.is_file() else {
            "status": "not_started",
            "artifact": str(path),
            "reason": "paid_provider_branch_paused_after_two_terminal_Q1_technical_failures",
        }
        for label, path in q1_trial_paths.items()
    }
    n12c_path = root / "validation/v3_1_execution_20260907T184000JST/n12c_same_chapter_matrix_q2_continuation_20260908T103000JST/same_chapter_learning_matrix.full_local.json"
    n12c = _load(n12c_path)
    n12c_arms = n12c.get("arms")
    n12c_arms = n12c_arms if isinstance(n12c_arms, Mapping) else {}
    n12c_summary = {
        "artifact": str(n12c_path),
        "sha256": _sha(n12c_path),
        "status": n12c.get("status"),
        "q1_provider_counts": _provider_counts(
            n12c.get("q1", {}).get("record") if isinstance(n12c.get("q1"), Mapping) else None
        ),
        "q2_arms": [
            {
                "arm": key,
                "run_status": value.get("summary", {}).get("run_status") if isinstance(value, Mapping) and isinstance(value.get("summary"), Mapping) else "not_observed",
                "edge_state": value.get("summary", {}).get("this_run_edge_participated") if isinstance(value, Mapping) and isinstance(value.get("summary"), Mapping) else "not_observed",
                "delivered_episode_ids": value.get("summary", {}).get("delivered_episode_ids", []) if isinstance(value, Mapping) and isinstance(value.get("summary"), Mapping) else [],
            }
            for key, value in n12c_arms.items()
        ],
        "rule": "Historical terminal outcomes are preserved; no missing event is inferred.",
    }
    proposals_path = campaign_dir / "source_gold/E32-04_claim_proposals/source_gold_claim_proposals.full_local.json"
    proposals = _load(proposals_path) if proposals_path.is_file() else {}
    narrowed_source_path = campaign_dir / "source_gold/E32-07_narrowed_atom_review/narrowed_source_atom_review.full_local.json"
    narrowed_source = _load(narrowed_source_path) if narrowed_source_path.is_file() else {}
    corpus_path = root / "validation/v3_1_execution_20260907T184000JST/full_corpus_preflight_20260908_current.json"
    corpus = _load(corpus_path)
    corpus_disposition_path = campaign_dir / "corpus/full_corpus_disposition.full_local.json"
    corpus_disposition = _load(corpus_disposition_path) if corpus_disposition_path.is_file() else {}
    test_receipt_path = campaign_dir / "tests/E32-00_local_regression.json"
    test_receipt = _load(test_receipt_path) if test_receipt_path.is_file() else {}
    corpus_categories = corpus.get("categories")
    corpus_categories = corpus_categories if isinstance(corpus_categories, Mapping) else {}
    in_scope_categories = [
        value for key, value in corpus_categories.items()
        if key in {"event", "main", "favor"} and isinstance(value, Mapping)
    ]
    baseline = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "working_tree_dirty": bool(git_status),
        "working_tree_status_count": len(git_status),
        "relevant_file_sha256": {str(path.relative_to(root)): _sha(path) for path in relevant_files},
        "formal_scoring": "disabled_unapproved_gold",
        "promotion": "disabled",
        "canary": "disabled",
    }
    _write(campaign_dir / "E32-00_baseline.json", baseline)
    _write(campaign_dir / "E32-01_n12c_stage_reconstruction.json", n12c_summary)
    state: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "active_diagnostic_non_scorable",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "historical_N11_N11d_N12b_N12c": {"preserved": True, "n12c": n12c_summary},
        "E32_02": experiments,
        "E32_04": cache_controls,
        "E32_05": {
            "q1_trials": q1_trials,
            "batch_provider_http_attempts": 39,
            "suggested_batch_cap": 40,
            "paid_provider_branch": {
                "status": "paused",
                "reason": (
                    "two consecutive fresh-Q1 trials ended without a legal "
                    "receipt/edge/manifest; neither trial will be retried"
                ),
                "unstarted_trial": "favor_competition",
            },
            "scope_boundary": (
                "The two Q1 trials used the preserved diagnostic SQLite snapshot "
                "and questions anchored to authorized event/main/favor Source files. "
                "They are non-scoring diagnostics, not a K_full import or a "
                "current-scope quality comparison."
            ),
        },
        "source_gold": {
            "artifact": str(proposals_path),
            "proposal_count": proposals.get("proposal_count", "not_observed"),
            "status": proposals.get("status", "not_observed"),
            "runtime_eligibility": proposals.get("runtime_eligibility", "not_observed"),
            "scoring": proposals.get("formal_scoring", "not_observed"),
            "narrowed_atom_review": {
                "artifact": str(narrowed_source_path),
                "sha256": _sha(narrowed_source_path) if narrowed_source_path.is_file() else "not_observed",
                "atom_count": narrowed_source.get("atom_count", "not_observed"),
                "status": narrowed_source.get("status", "not_observed"),
                "runtime_eligibility": narrowed_source.get("runtime_eligibility", "not_observed"),
            },
        },
        "full_corpus_static": {
            "artifact": str(corpus_path),
            "scope": ["event", "main", "favor"],
            "files": sum(int(value.get("files", 0) or 0) for value in in_scope_categories),
            "segments": sum(int(value.get("segments", 0) or 0) for value in in_scope_categories),
            "empty_files": 6,
            "parse_failures": 0,
            "formal_old_vs_full_comparison": "not_run_unapproved_gold",
            "disposition": {
                "artifact": str(corpus_disposition_path),
                "sha256": _sha(corpus_disposition_path) if corpus_disposition_path.is_file() else "not_observed",
                "status": (
                    "read_only_disposition_complete"
                    if corpus_disposition_path.is_file()
                    else "not_observed"
                ),
                "K_full": (
                    corpus_disposition.get("snapshots", {}).get("K_full", "not_observed")
                    if isinstance(corpus_disposition.get("snapshots"), Mapping)
                    else "not_observed"
                ),
            },
        },
        "delivery": {
            "local_regression": {
                "artifact": str(test_receipt_path),
                "status": test_receipt.get("status", "not_observed"),
                "test_count": test_receipt.get("test_count", "not_observed"),
                "network": test_receipt.get("network", "not_observed"),
            },
            "reports": {
                "final_capability": str(campaign_dir / "reports/FINAL_CAPABILITY_REPORT.md"),
                "failure_uncertainty": str(campaign_dir / "reports/FAILURE_AND_UNCERTAINTY.md"),
                "decisions": str(campaign_dir / "reports/DECISIONS_REQUIRED.md"),
                "cross_topic_full_local": str(campaign_dir / "reports/cross_topic_results.full_local.json"),
            },
            "redacted_archive": {
                "path": str(campaign_dir / "deliverable.redacted.zip"),
                "status": (
                    "present_excludes_answer_evidence_logs_and_full_local"
                    if (campaign_dir / "deliverable.redacted.zip").is_file()
                    else "not_observed"
                ),
                "manifest": str(campaign_dir / "ARTIFACT_MANIFEST.redacted.json"),
                "checksums": str(campaign_dir / "artifacts.sha256"),
            },
        },
        "next": [
            "E32-03 position/key ablation is not run: this Q2 family has a recorded base-coverage no-op, not an edge benefit.",
            "E32-04 now has a local B3 Source-validation cache read and its source-change rejection; B2 and a fair B4/B4M ordinary-fallback latency pair remain not observed.",
            "E32-05 fresh cross-topic learning and E32-06 limited full-answer pairs require pre-registered independent topic inputs and remain separate from this same-chapter diagnostic.",
        ],
    }
    _write(campaign_dir / "campaign_state.json", state)
    lines = [
        "# Aevnema v3.2 campaign status",
        "",
        "Diagnostic only: formal gold is not loaded; promotion and canary remain disabled.",
        "",
        f"- Baseline Git commit: `{git_head}`; relevant working files are recorded by SHA-256 in `E32-00_baseline.json`.",
        f"- N12c historical trace preserved: `{n12c_summary['status']}`; its failures were not rerun or rewritten.",
        f"- Source-gold proposals: `{proposals.get('proposal_count', 'not_observed')}` literal-record candidates; narrowed reviewer packet: `{narrowed_source.get('atom_count', 'not_observed')}` atoms. All remain pending human review and runtime-forbidden.",
        f"- Static corpus inventory: `{state['full_corpus_static']['files']}` files / `{state['full_corpus_static']['segments']}` segments; the read-only residual disposition is `{state['full_corpus_static']['disposition']['status']}` and no formal old-vs-full score exists.",
        f"- Local regression: `{state['delivery']['local_regression']['status']}` / `{state['delivery']['local_regression']['test_count']}` tests, with network `{state['delivery']['local_regression']['network']}`.",
        f"- Redacted archive: `{state['delivery']['redacted_archive']['status']}`; full-local case trails remain separate.",
        "",
        "## E32-02 observations",
        "",
        "| Observation | Input mode | Snapshot unchanged | Edge states | Q2 HTTP |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for label, item in experiments.items():
        arm_values = item.get("arms", []) if isinstance(item, Mapping) else []
        edge_states = sorted({str(arm.get("edge_state")) for arm in arm_values if isinstance(arm, Mapping)})
        attempts = sum(
            int(arm.get("q2_http_attempts", 0) or 0)
            for arm in arm_values if isinstance(arm, Mapping)
        )
        lines.append(
            f"| {label} | `{item.get('input_preparation_mode', 'not_observed')}` | `{item.get('snapshot_unchanged', 'not_observed')}` | `{', '.join(edge_states)}` | {attempts} |"
        )
    lines.extend([
        "",
        "The restored-vector observation records `not_entered`. Its historical compact trace did not retain the selector's first non-entry reason; the base-coverage interpretation is an inference from the recorded selected Source and satisfied deterministic/coverage entries, not an invented observed event. It is not evidence that the edge is harmful or that non-exact association has no utility in other topics.",
        "",
    ])
    lines.extend([
        "## E32-05 fresh cross-topic Q1 trials",
        "",
        "The paid provider branch is paused after two consecutive terminal Q1 technical failures. These trials are preserved once; the unstarted favor case is not a zero-result or a retry candidate.",
        "",
        "| Topic | Q1 terminal state | Learning / finalizer | Q2 state | HTTP | Source delivery observed |",
        "| --- | --- | --- | --- | ---: | --- |",
    ])
    for label, item in q1_trials.items():
        counts = item.get("q1_provider_counts", {}) if isinstance(item, Mapping) else {}
        counts = counts if isinstance(counts, Mapping) else {}
        finalizer = item.get("public_finalizer_calls", "not_observed") if isinstance(item, Mapping) else "not_observed"
        learning = item.get("learning_status", "not_observed") if isinstance(item, Mapping) else "not_observed"
        q2_states = item.get("q2_arm_states", ["not_observed"]) if isinstance(item, Mapping) else ["not_observed"]
        modes = item.get("source_evidence_delivery_modes", ["not_observed"]) if isinstance(item, Mapping) else ["not_observed"]
        lines.append(
            f"| {label} | `{item.get('q1_status', item.get('status', 'not_observed'))}` | `{learning}` / `{finalizer}` | `{', '.join(map(str, q2_states))}` | {counts.get('http_attempts', 'not_observed')} | `{', '.join(map(str, modes))}` |"
        )
    lines.extend([
        "",
        "The event Q1 failed at the public query deadline before a post-query result was returned. The main Q1 returned an answer attempt but has `selector_delivery_incomplete`; its selected Episodes recorded `legacy_heuristic_excerpt` / `no_persisted_evidence_quote`, and its answer audits timed out. Those are recorded facts, not a claim that the association mechanism lacks utility. No receipt, edge, exact contract, or runtime manifest was created in either trial.",
        "",
    ])
    lines.extend([
        "## E32-04 cache controls",
        "",
        "| Observation | B3 | Source change | B2 | B4M fallback |",
        "| --- | --- | --- | --- | --- |",
    ])
    for label, item in cache_controls.items():
        controls = item.get("controls", {}) if isinstance(item, Mapping) else {}
        controls = controls if isinstance(controls, Mapping) else {}
        b3 = controls.get("B3_source_validated_evidence_cache", {})
        b3 = b3 if isinstance(b3, Mapping) else {}
        b2 = controls.get("B2_historical_plan_cache", {})
        b2 = b2 if isinstance(b2, Mapping) else {}
        b4m = controls.get("B4M_exact_edge_mask", {})
        b4m = b4m if isinstance(b4m, Mapping) else {}
        invalidation = item.get("source_change_invalidation", {}) if isinstance(item, Mapping) else {}
        invalidation = invalidation if isinstance(invalidation, Mapping) else {}
        lines.append(
            f"| {label} | `{b3.get('status', 'not_observed')}` | `{invalidation.get('status', 'not_observed')}` | `{b2.get('status', 'not_observed')}` | `{b4m.get('comparison_status', 'not_observed')}` |"
        )
    lines.extend([
        "",
        "E32-04c is preserved as a control-implementation failure: it reached a B3 hit but mutated a non-dependent Source row, so its requested invalidation check was an `unexpected_hit`. E32-04d changes the actual cached EvidenceRef Source on a clone and records the required rejection. Neither observation supplies a fair B4/B4M ordinary-fallback latency ratio.",
        "",
    ])
    (campaign_dir / "V3_2_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    state = build_campaign(root=args.root, campaign_dir=args.campaign_dir)
    print(json.dumps({"status": state["status"], "campaign_dir": str(args.campaign_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
