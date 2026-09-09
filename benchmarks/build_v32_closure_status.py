"""Write the bounded v3.2 work-package closure state and No-Go recommendation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


WORK_PACKAGES = [
    ("W00", "done", "baseline, historical assets and authorized source root recorded"),
    ("W01", "verified_existing", "exact/source contract paths covered by current local regression"),
    ("W02", "done", "public evidence-only boundary and local answer checkpoints implemented"),
    ("W03", "done", "frozen plan/vector public replay and shared-input matrix path implemented"),
    ("W04", "limited", "prompt evidence boundary and narrow answer budget regression covered; no complete live audit succeeded"),
    ("W05", "done", "N12c read-only reconstruction and E32-02 evidence-only observations preserved"),
    ("W06", "blocked_dependency", "no fresh source-bound non-exact learned edge is available for a meaningful position trial"),
    ("W07", "blocked_dependency", "no fresh legal cross-topic edge exists for key/contribution ablation"),
    ("W08", "limited", "B3 source-validated cache and invalidation completed; B2 and fair B4/B4M fallback pair remain unobserved"),
    ("W09", "limited", "two complete Q1 provider ledgers expose answer/audit deadline failure; paid branch paused"),
    ("W10", "done", "388 literal proposals and 19 narrowed human-review atoms delivered"),
    ("W11", "limited", "all historical residuals statically disposed; source-bound K_full rebuild is externally blocked"),
    ("W12", "done", "three pre-registered independent source-topic manifests prepared without runtime answers/gold"),
    ("W13", "limited", "two fresh Q1 trials terminally preserved; favor not started and no dependent Q2 ran"),
    ("W14", "blocked_dependency", "requires a fresh legal learned edge and independent Q2 snapshot"),
    ("W15", "limited", "core answer-boundary checkpoints observed; no accepted paired Q2/chat end-to-end result"),
    ("W16", "blocked_external", "requires source-bound K_full and approved independent gold/split"),
    ("W17", "blocked_dependency", "requires legal cross-topic learning and K_full; synthetic scale is not substituted for semantic testing"),
    ("W18", "limited", "capability report gives a diagnostic No-Go; no production action was authorized or taken"),
    ("W19", "done", "state, reports, checksums and redacted archive produced"),
]


def build(*, campaign_dir: Path) -> dict[str, object]:
    campaign_dir = campaign_dir.resolve()
    rows = [
        {"id": identifier, "status": status, "reason": reason}
        for identifier, status, reason in WORK_PACKAGES
    ]
    payload = {
        "schema": "aevnema.v3_2.work_package_status.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "statuses": rows,
        "terminal_boundary": (
            "All work currently independent of human gold approval and a new paid "
            "provider batch has been advanced. Historical failures remain preserved."
        ),
    }
    (campaign_dir / "work_package_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status_lines = [
        "# v3.2 work-package status",
        "",
        "| Package | State | Evidence / boundary |",
        "| --- | --- | --- |",
        *[f"| {row['id']} | `{row['status']}` | {row['reason']} |" for row in rows],
        "",
        "`blocked_external` and `blocked_dependency` are not pass/fail values. They identify the earliest missing prerequisite without rewriting old trials or inventing a negative result.",
    ]
    (campaign_dir / "WORK_PACKAGE_STATUS.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    commands = [
        "# v3.2 actual commands",
        "",
        "Commands are normalized to omit credentials and only list invocations actually used in this campaign.",
        "",
        "```text",
        ".venv\\Scripts\\python.exe aevnema_v3_2_integrated_plan\\aevnema_v3_2_integrated_plan\\validate_plan_package.py",
        ".venv\\Scripts\\python.exe aevnema_v3_2_integrated_plan\\aevnema_v3_2_integrated_plan\\validate_plan_package.py --self-test",
        ".venv\\Scripts\\python.exe -m benchmarks.run_v32_local_regression --root . --output-dir validation\\aevnema-v3_2\\v3_2_20260908T133000JST\\tests",
        ".venv\\Scripts\\python.exe -m benchmarks.run_same_chapter_learning_matrix --source-database validation\\v3-freeze-20260906T233016Z\\testing_kb.sqlite --matrix-manifest benchmarks\\manifests\\v3_2_cross_topic_event_moon_festival.json --output-dir validation\\aevnema-v3_2\\v3_2_20260908T133000JST\\experiments\\E32-05a_event_q1 --env-file .env --deadline-seconds 120 --defer-q2",
        ".venv\\Scripts\\python.exe -m benchmarks.run_same_chapter_learning_matrix --source-database validation\\v3-freeze-20260906T233016Z\\testing_kb.sqlite --matrix-manifest benchmarks\\manifests\\v3_2_cross_topic_main_rest.json --output-dir validation\\aevnema-v3_2\\v3_2_20260908T133000JST\\experiments\\E32-05b_main_q1 --env-file .env --deadline-seconds 180 --defer-q2",
        ".venv\\Scripts\\python.exe -m benchmarks.build_v32_full_corpus_disposition ...",
        ".venv\\Scripts\\python.exe -m benchmarks.build_v32_cross_topic_report ...",
        ".venv\\Scripts\\python.exe -m benchmarks.build_v32_final_reports ...",
        ".venv\\Scripts\\python.exe -m benchmarks.package_v32_campaign ...",
        "```",
        "",
        "No command retried either fresh Q1, modified a historical trial, imported K_full, or sent a user-facing chat message.",
    ]
    (campaign_dir / "actual_commands.md").write_text("\n".join(commands) + "\n", encoding="utf-8")
    recommendation = [
        "# v3.2 release recommendation",
        "",
        "## Decision: No-Go for promotion or canary",
        "",
        "Keep formal promotion and canary disabled. The only end-to-end demonstrated reuse is the preserved single exact N11d case. The current paid batch did not establish a new legal source-bound edge, K_full is absent, Source gold is pending human review, and the two observed fresh full-answer chains ended in audit/deadline technical failures.",
        "",
        "The diagnostic evidence-only/exact code and local cache control may remain available for offline research. This is not authorization to route production traffic, learn from unreviewed answers, or claim cross-topic benefit.",
    ]
    (campaign_dir / "reports" / "release_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")
    rollback = {
        "schema": "aevnema.v3_2.rollback_check.v1",
        "production_changes_made": False,
        "promotion": "disabled",
        "canary": "disabled",
        "formal_kb_writes": "not_performed",
        "rollback_action": "not_applicable_no_production_change",
        "ordinary_path_preserved": True,
        "note": "This is a state observation, not a substituted feature-off live service test.",
    }
    (campaign_dir / "reports" / "rollback_check.json").write_text(
        json.dumps(rollback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = build(campaign_dir=args.campaign_dir)
    print(json.dumps({"status_count": len(payload["statuses"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
