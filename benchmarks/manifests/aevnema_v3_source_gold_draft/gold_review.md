# Aevnema v3 Source-Gold Draft Review

Status: **draft_pending_source_span_review**. This is an evaluator-only re-anchoring worklist, not an accepted Source-level gold set.

## What was preserved

- The three historical Stage3 questions and their nine Episode alternative groups were retained as legacy candidate slots.
- The original five-question / fifteen-keyword benchmark remains separately frozen as a diagnostic and was not promoted to gold.
- Each candidate records its snapshot-local Episode ID, source key, segment index, Episode text hash, and source-segment hash.

## Why this draft cannot score systems

- Historical candidates have no reviewed source record locator, span start/end, source-file hash, or raw span hash.
- Each evidence atom is marked `pending_source_span_review` and `usable_for_scoring: false`.
- The historical questions are one source-connected component and are all assigned to `legacy_calibration`; none is a blind holdout.

## Required review before promotion

1. Anchor every retained claim to a source record/JSONPath and exact source span.
2. Store raw span and source-file hashes, then independently review the claim's epistemic strength.
3. Express any joint claim with an explicit `all_of` clause and any alternative with `any_of`.
4. Create source-disjoint reviewed families before declaring a calibration/holdout split.
5. Keep this material evaluator-only; do not feed it into `MemoryApplication`, `QueryEngine`, or model prompts.
