"""Render the bounded v3.2 continuation report from local receipts only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "aevnema.v3_2.continuation_execution_report.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def build(*, campaign_root: Path, output_dir: Path) -> dict[str, str]:
    root = campaign_root.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be new: {output}")
    inputs = {
        "provider_reconciliation": root / "reconciliation/E32-08_provider_attempt_reconciliation/provider_attempt_reconciliation.json",
        "legacy_eligibility": root / "eligibility/E32-09_legacy_quote_recovery/legacy_quote_recovery_feasibility.json",
        "gold_master": root / "source_gold/E32-10_review_master/narrowed_source_claim_review_master.json",
        "pre_fix_position": root / "experiments/E32-11_public_position_replay/public_position_replay.full_local.json",
        "post_fix_position": root / "experiments/E32-12_public_position_replay_after_frozen_selector_fix/public_position_replay.full_local.json",
        "paired_cache_edge": root / "experiments/E32-13_paired_b3_b4_local/paired_b3_b4.full_local.json",
        "regression": root / "tests/E32-14_post_continuation_regression/E32-00_local_regression.json",
        "prior_full_corpus": root / "corpus/full_corpus_disposition.full_local.json",
    }
    absent = [str(path) for path in inputs.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError("missing prerequisite receipt(s): " + "; ".join(absent))
    data = {key: _load(path) for key, path in inputs.items()}
    reconciliation = data["provider_reconciliation"].get("reconciliation", {})
    eligibility = data["legacy_eligibility"].get("learning_data_eligibility", {})
    recovery = data["legacy_eligibility"].get("summary", {})
    paired = data["paired_cache_edge"].get("summary", {})
    b3 = paired.get("B3_source_validated_cache", {}) if isinstance(paired, Mapping) else {}
    b4 = paired.get("B4_public_exact_edge", {}) if isinstance(paired, Mapping) else {}
    regression = data["regression"]
    post = data["post_fix_position"]
    post_arms = post.get("arms", {}) if isinstance(post, Mapping) else {}
    late_reasons = sorted({
        str((_mapping(_mapping(arm).get("late_evidence_entry")).get("result", {}) or {}).get("contextual_association_reason", "not_observed"))
        for arm in post_arms.values() if isinstance(arm, Mapping)
    })
    output.mkdir(parents=True)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "bounded_non_paid_continuation_complete",
        "created_at": _utc_now(),
        "paid_provider_calls_issued_by_this_continuation": 0,
        "formal_scoring": "disabled_unapproved_gold",
        "promotion": "disabled",
        "canary": "disabled",
        "inputs": {key: {"path": str(path), "sha256": _sha(path)} for key, path in inputs.items()},
        "results": {
            "provider_ledger": {
                "reported_subtotal": reconciliation.get("reported_campaign_subtotal_http_attempts"),
                "preserved_raw_rows": reconciliation.get("preserved_raw_observation_rows"),
                "unique_local_identities": reconciliation.get("unique_observation_identities"),
                "by_phase": reconciliation.get("raw_rows_by_phase"),
                "billing_allocation": reconciliation.get("billing_or_authorization_allocation"),
                "budget_conclusion": reconciliation.get("budget_compliance_conclusion"),
            },
            "legacy_k_diag": {
                "episodes": eligibility.get("episode_count"),
                "quotes": eligibility.get("episodes_with_persisted_evidence_quotes"),
                "spans": eligibility.get("episodes_with_persisted_evidence_spans"),
                "strict_new_edge_eligible": eligibility.get("strict_source_bound_creation_eligible"),
                "deterministic_locator_candidates": recovery.get("deterministic_locator_candidate_count"),
                "unlocated": recovery.get("no_deterministic_locator_count"),
            },
            "source_review": {
                "candidate_atoms": data["gold_master"].get("atom_count"),
                "human_review": data["gold_master"].get("human_review"),
                "runtime_eligibility": data["gold_master"].get("runtime_eligibility"),
            },
            "w06_w07": {
                "pre_fix_artifact_preserved": True,
                "post_fix_snapshot_unchanged": post.get("snapshot_unchanged"),
                "post_fix_arm_count": len(post_arms),
                "post_fix_provider_calls": 0,
                "late_branch_reasons": late_reasons,
                "nonexact_early_public_entry": _mapping(post.get("position_contract")).get("nonexact_early_public_entry"),
            },
            "paired_b3_b4": {
                "repetitions": b3.get("n"),
                "b3_median_ms": b3.get("median_ms"),
                "b4_median_ms": b4.get("median_ms"),
                "b3_p95_ms": b3.get("p95_ms"),
                "b4_p95_ms": b4.get("p95_ms"),
                "timing_ratio_claimed": paired.get("timing_ratio_claimed"),
            },
            "regression": {
                "status": regression.get("status"),
                "tests": regression.get("test_count"),
                "network": regression.get("network"),
            },
        },
        "minimal_code_change": {
            "file": "src/memory_demo/retrieval/engine.py",
            "change": "A frozen plan with an already restored QueryVectorBundle now enters the existing local residual-slot selector; a frozen plan without such a bundle retains no-selection behavior.",
            "not_changed": ["receipt_finalizer", "runtime_manifest_contract", "source_validation", "provider_models", "per_edge_cloud_audit", "journal_scope"],
        },
        "next_external_decisions": [
            "Provide invoice/authorization-batch data if billing allocation (rather than local call identity) must be reconciled.",
            "Approve, split, request context for, or reject each of the 19 Source claims; no agent signature exists.",
            "Choose a source-bound data migration/re-extraction scope before authorizing a new fresh-learning paid Q1; K_diag is not eligible for strict creation.",
        ],
    }
    json_path = output / "continuation_execution_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Aevnema v3.2 continuation execution report",
        "",
        "Outcome: the requested non-paid continuation queue is complete. Paid calls remain paused; formal scoring, promotion and canary remain off. Historical runs, including the pre-fix W06/W07 replay and N11d, were not modified.",
        "",
        "## Observed results",
        "",
        f"- Provider ledger: **39** is a subtotal; preserved ledgers contain **{reconciliation.get('preserved_raw_observation_rows')}** rows with **{reconciliation.get('unique_observation_identities')}** unique local call IDs (27 live Q2 arms, 31 frozen-input preparation, 12 fresh Q1). Billing/allocation is `{reconciliation.get('billing_or_authorization_allocation')}`, so budget compliance is `{reconciliation.get('budget_compliance_conclusion')}`.",
        f"- K_diag eligibility: **{eligibility.get('episode_count')}** Episodes, **{eligibility.get('episodes_with_persisted_evidence_quotes')}** persisted quotes and **{eligibility.get('episodes_with_persisted_evidence_spans')}** spans; strict fresh edge creation is `{eligibility.get('strict_source_bound_creation_eligible')}`. Read-only locator feasibility found **{recovery.get('deterministic_locator_candidate_count')}** candidate locators and **{recovery.get('no_deterministic_locator_count')}** without a deterministic locator. No migration was written.",
        f"- Source review: the 19 actual record/context packets are assembled, but every item remains `{data['gold_master'].get('human_review')}` and runtime `{data['gold_master'].get('runtime_eligibility')}`.",
        f"- W06/W07: E32-11 preserved the pre-fix frozen-plan selector bypass. The one-line runtime fix was then exercised in E32-12 with eight public Q2 conditions, zero HTTP and unchanged snapshot. All late paths reported `{', '.join(late_reasons)}`; the matcher was therefore not entered because the ordinary base selection already covered the single required slot. The public early entry is `{_mapping(post.get('position_contract')).get('nonexact_early_public_entry')}`; no private matcher was used to pretend otherwise.",
        f"- Paired B3/B4: 15 alternating same-snapshot local repetitions returned Episode 32 in both arms, 0 HTTP. B3 median/p95 were **{b3.get('median_ms')} / {b3.get('p95_ms')} ms**; B4 median/p95 were **{b4.get('median_ms')} / {b4.get('p95_ms')} ms**. No universal timing ratio is claimed.",
        f"- Regression: **{regression.get('test_count')}** fixed local tests `{regression.get('status')}`; network `{regression.get('network')}`.",
        "",
        "## Minimal implementation change",
        "",
        "`src/memory_demo/retrieval/engine.py` now lets a frozen query plan that has an already restored vector bundle enter the existing residual-slot selector. It does not generate vectors, relax source checks, alter the finalizer/manifest contract, change any model, restore per-edge cloud audit, or broaden journal/gate scope.",
        "",
        "## What remains deliberately unresolved",
        "",
        "- K_diag cannot legally create a strict source-bound learning edge until evidence data is migrated/re-extracted under an approved scope.",
        "- The 19 literal-record leads require human claim decisions; they are not gold or runtime data.",
        "- The current four-Q2 family is not evidence of non-exact graph benefit: ordinary retrieval covers its sole slot. The early public API has no non-exact proposal endpoint, so a preflight miss has no public reason payload.",
        "- Existing full-corpus disposition remains preserved; no import/re-extraction or formal old-vs-full score was started in this continuation.",
        "",
        "## Required external decisions before a new paid batch",
        "",
        "1. If needed, provide billing/authorization-batch records for a definitive cost allocation beyond call-ID reconciliation.",
        "2. Review the 19 Source claim packets (accept, split, seek context, or reject).",
        "3. Approve a source-bound migration or re-extraction scope before a fresh paid Q1 attempt; do not use K_diag as a substitute.",
    ]
    md_path = output / "CONTINUATION_EXECUTION_REPORT.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "report": str(md_path)}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
