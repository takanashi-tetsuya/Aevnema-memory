"""Build the evidence-backed v3.2 capability and uncertainty reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def build_reports(*, campaign_dir: Path) -> None:
    campaign_dir = campaign_dir.resolve()
    reports = campaign_dir / "reports"
    state = _load(campaign_dir / "campaign_state.json")
    tests = _load(campaign_dir / "tests" / "E32-00_local_regression.json")
    cross = _load(reports / "cross_topic_results.full_local.json")
    corpus = _load(campaign_dir / "corpus" / "full_corpus_disposition.full_local.json")
    proposals = _load(
        campaign_dir / "source_gold" / "E32-04_claim_proposals" / "source_gold_claim_proposals.full_local.json"
    )
    narrow = _load(
        campaign_dir / "source_gold" / "E32-07_narrowed_atom_review" / "narrowed_source_atom_review.full_local.json"
    )
    cache = _load(
        campaign_dir / "experiments" / "E32-04d_source_validated_cache_control" / "source_validated_cache_control.full_local.json"
    )
    controls = _mapping(cache.get("controls"))
    b3 = _mapping(controls.get("B3_source_validated_evidence_cache"))
    b4 = _mapping(controls.get("B4_exact_edge_reuse"))
    b4m = _mapping(controls.get("B4M_exact_edge_mask"))
    invalidation = _mapping(cache.get("source_change_invalidation"))
    trials = cross.get("trials")
    trials = trials if isinstance(trials, list) else []
    by_label = {
        str(item.get("label")): item
        for item in trials
        if isinstance(item, Mapping)
    }
    event = _mapping(by_label.get("event_moon_festival"))
    main = _mapping(by_label.get("main_rest"))
    q1_event = _mapping(event.get("q1"))
    q1_main = _mapping(main.get("q1"))
    main_learning = _mapping(main.get("learning"))
    static_scope = _mapping(corpus.get("static_scope"))
    snapshots = _mapping(corpus.get("snapshots"))
    diag = _mapping(snapshots.get("K_diag"))
    kfull = _mapping(snapshots.get("K_full"))
    provider = _mapping(state.get("E32_05"))
    provider_policy = _mapping(provider.get("paid_provider_branch"))

    _write(
        reports / "IMPLEMENTATION_AND_TESTS.md",
        [
            "# v3.2 implementation and local regression receipt",
            "",
            f"The current local suite passed **{tests.get('test_count', 'not_observed')} tests** with no network use (receipt: `{tests.get('stdout_artifact', 'not_observed')}`).",
            "",
            "## Implemented, bounded changes",
            "",
            "- The public query entry now supports evidence-only termination: it uses normal retrieval and Source materialization, but does not generate/audit an answer, learn, write utility, or mark an edge used.",
            "- Frozen query plans rebuild validated request vectors through the public engine path; they do not inject a hand-built matcher result or make a per-edge cloud call.",
            "- A projected reasoning quote maps back to its immutable raw Source record only when every shared single-line translation and optional speaker field match. The delivered view is the raw record, including script fields, rather than the compressed projection.",
            "- Repeated language fields are rejected as `duplicate_translation_field`; a continuation line after a language field is rejected as `multiline_or_unlabelled_translation_field`. These are fail-closed format checks, not semantic equivalence rules.",
            "- Quote verification failure and verified-quote budget overflow have distinct delivery reasons (`quote_verification_failed` and `verified_quote_exceeds_excerpt_budget`). Neither receives `source_bound`.",
            "- The answer-boundary companion log is experimental and local-only. It records actual evidence, answer and audit checkpoints with prompt/version binding while redacting credentials. `local_only` is a label, not a sharing control; the redacted delivery archive excludes these logs by default.",
            "- The audit prompt regression verifies that the full delivered Source excerpt—including a record beyond a 2,400-character prefix and its speaker marker—survives prompt assembly.",
            "",
            "The test set includes exact-source invalidation, public evidence-only, frozen input, source cache, matrix failure terminal states, duplicate/multiline projection rejection, and the 2,400-character audit-boundary regression. It is a local control-flow and source-binding receipt, not a live semantic score.",
        ],
    )

    _write(
        reports / "FAILURE_AND_UNCERTAINTY.md",
        [
            "# v3.2 failures and uncertainty",
            "",
            "## Fresh cross-topic Q1 observations",
            "",
            f"- **event_moon_festival:** Q1 terminal state `{q1_event.get('status', 'not_observed')}`. The answer boundary did record Source excerpts and an answer revision, then its audit checkpoint recorded a deadline exception; the public query deadline elapsed before `post_query_result`. Result-object retrieval/selector/learning fields are therefore not observed, rather than inferred as misses.",
            f"- **main_rest:** Q1 terminal state `{q1_main.get('status', 'not_observed')}`. Source excerpts reached the answer boundary, but they came from `legacy_heuristic_excerpt` / `no_persisted_evidence_quote`; the answer audit had a technical exception. Learning is `{main_learning.get('status', 'not_observed')}` / `{main_learning.get('reason', 'not_observed')}`, with `{main_learning.get('public_finalizer_calls', 'not_observed')}` public-finalizer calls.",
            "- **favor_competition:** not started after the frozen paid-provider stop rule. It is not a Q1 failure, a retrieval miss, or a negative semantic sample.",
            "",
            "## What remains unknown",
            "",
            "- These two trials did not create a receipt, edge, exact contract, or ready runtime manifest. Their dependent Q2 arms are not-run, so they do not measure cross-topic edge utility, paraphrase handling, partial-clue handling, restart behavior, or masking.",
            "- The current K_diag snapshot has no persisted evidence quotes. A legacy heuristic excerpt can show what reached an attempted answer boundary, but cannot be reclassified as source-bound learning material.",
            "- No approved independent gold was loaded. Candidate, selected, materialized, answer-input, audit and final-answer quality are not formal recall/precision measurements here.",
            "- The same-chapter restored-vector no-entry observation remains a base-coverage diagnostic; it does not show that association is harmful or generally without benefit.",
        ],
    )

    _write(
        reports / "DECISIONS_REQUIRED.md",
        [
            "# v3.2 external decisions required",
            "",
            "No decision is needed to preserve the completed artifacts. The following are required only to advance blocked branches.",
            "",
            f"1. **Human Source review:** review the `{narrow.get('atom_count', 'not_observed')}` narrowed atoms and `{proposals.get('proposal_count', 'not_observed')}` literal-record proposals. They remain evaluator-only, runtime-forbidden and unapproved; no one has signed them as gold.",
            "2. **A new paid diagnostic batch:** the current batch is paused after two terminal Q1 technical failures and 39 observed provider attempts. A new authorization/budget is required before attempting favor or a new source-bound rebuild. Existing event/main trials will not be retried.",
            "3. **Source-bound K_full rebuild:** create a new static snapshot from the authorized event/main/favor root using the current quote-persistence path, then handle the one unresolved historical pass2 HTTP-400 task within that new rebuild. Do not patch legacy rows or rewrite the old ledger.",
            "4. **Formal comparison:** only after both an approved independent Source gold/split and a static K_full exist may old-vs-full and holdout quality scores be computed. Promotion and canary remain off.",
        ],
    )

    _write(
        reports / "FINAL_CAPABILITY_REPORT.md",
        [
            "# Aevnema v3.2 capability report",
            "",
            "## Supported now",
            "",
            "- **Single-case exact evidence reuse:** the preserved N11d result remains the only demonstrated end-to-end exact closure. It used the public lookup, returned bound Source with 0 new HTTP, and rejected masked/source-changed paths. This is not a recall or generalization score.",
            f"- **Source-validated evidence cache control:** B3 actually revalidated the exact question/scope/budget, source identity, raw bytes and quotes, delivered episode `{b3.get('delivered_episode_ids', [])}`, used 0 HTTP and took `{b3.get('evidence_acquisition_ms', 'not_observed')}` ms for the observed local replay. A source change was `{invalidation.get('status', 'not_observed')}` with `{invalidation.get('reason', 'not_observed')}`. It is a cache control, not an association edge.",
            "- **Evidence-delivery contract:** current code fails closed for duplicate or multiline projected translations, preserves raw Source on valid projection, and separates verified-quote failure from excerpt-budget failure. The current 47-test local receipt covers these paths.",
            "",
            "## Not supported or not demonstrated",
            "",
            "- No fresh cross-topic Q1 produced a legal learned edge in this batch; therefore no new Q2 edge/mask/cache/restart/source-change comparison exists for those topics.",
            "- No claim is made for paraphrase, partial clues, near-neighbor negatives, multi-slot transfer, formal recall, independent holdout, humanlike memory, or full-corpus quality.",
            "- K_full does not exist. The retained K_diag has `0` persisted evidence-quote Episodes and is not a current source-bound full benchmark database.",
            "- Full-answer service is not reliable in the two observed fresh Q1 trials: answer/audit deadlines prevented legal learning even where an answer-input checkpoint existed.",
            "",
            "## Cache and latency boundary",
            "",
            f"B4 exact-edge reuse is a preserved historical observation; B4M is `{b4m.get('comparison_status', 'not_observed')}` and did not observe ordinary fallback. B2 remains `{_mapping(controls.get('B2_historical_plan_cache')).get('status', 'not_observed')}`. Therefore no fair B4/B4M end-to-end latency ratio is claimed, and B3's local replay time is not compared to a Q1 full-answer duration.",
            "",
            "## Corpus and gold boundary",
            "",
            f"The authorized static scope is `{static_scope.get('files', 'not_observed')}` files / `{static_scope.get('segments', 'not_observed')}` segments. Historical ledger states were preserved, including three partial and one failed file; two partials are recorded audited safe-skips, one remains a historical pass2 HTTP 400, and six main JSON files contain no usable localized story blocks.",
            f"K_diag classification: `{diag.get('classification', 'not_observed')}`. K_full status: `{kfull.get('status', 'not_observed')}`. Source-gold state: `{narrow.get('status', 'not_observed')}` for `{narrow.get('atom_count', 'not_observed')}` narrowed atoms; formal scoring stays disabled.",
            "",
            "## Operating decision",
            "",
            f"The paid-provider branch is `{provider_policy.get('status', 'not_observed')}`. No promotion, canary, formal KB write, gold sign-off, model substitution, hand-built edge, or answer-cache substitution occurred in this campaign. The next valid paid work is a new authorized batch on a new source-bound snapshot—not a retry of the event/main trials.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    build_reports(campaign_dir=args.campaign_dir)
    print(json.dumps({"status": "written", "campaign_dir": str(args.campaign_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
