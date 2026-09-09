# V3 requirement → exact-revisit runtime repair

## Outcome

The identified hand-off from a rich Q1 requirement to the V17 exact-revisit
runtime path is repaired and covered by isolated, provider-free regression
tests.  The one permitted new real-Source run (`v11`) did not reach the public
Q1 finalizer: it exceeded the existing 120-second public deadline in the answer
audit path.  It created no receipt, edge, contract, or runtime manifest; its
dependent Q2 arms were therefore deliberately not run.  This report does not
claim an edge-reuse result for `v11`.

Scope was limited to the requirement → ordinary contract → V17 runtime-manifest
→ public Q2 lookup hand-off.  Journal behavior, general gates, promotion,
canary, scoring, and per-edge cloud review were not changed.

## Located breakpoints and repair

1. The V17 runtime projection rejected a requirement merely because its
   authoritative semantic fields were populated (`subject_terms`, relation,
   time, epistemic/negation, or modality).  That is incorrect for an exact
   revisit: Q1 must retain those fields, while the runtime template may be
   reconstructed only when the whole normalized Q1 question is the single
   required slot and every other exact guard is satisfied.
2. The diagnostic manifest held a human-readable scope label, but the Q1/Q2
   API expects an opaque `namespace:sha256:<64-hex>` scope key.  The old runner
   passed the label directly.  In the preserved v10 record this made ordinary
   contract construction fail with `exact revisit context_scope_hash must be
   an opaque sha256 digest`, so neither a V16 contract nor a V17 manifest could
   exist despite a ready receipt and persisted edge.

The runner now deterministically projects its readable label at the public
runner boundary:

```
pilot-scope:sha256:SHA256("aevnema/v3/q1-q2-diagnostic-scope/v1\\0" + UTF-8(scope_label))
```

The same derived key is supplied to Q1 and every Q2 arm.  The original manifest
is not manually edited, and the engine still rejects missing, different, or
non-opaque scope values.  This is a typed boundary conversion, not a source or
scope guard bypass.

The finalizer now also emits a separate, redacted `revisit_projections` result.
It distinguishes a persisted receipt/edge from exact reuse readiness:

| Condition | Reported projection result |
| --- | --- |
| Runtime manifest published | `ready` / `runtime_manifest_ready` |
| Publication deferred or failed | `deferred_not_ready` or `failed` |
| Exact reconstruction is impossible | `rejected_not_reconstructable` with structural reason |

No query text, source text, vectors, or manual manifest contents are exposed by
that diagnostic field.

## Regression evidence (isolated, zero provider calls)

The new rich-requirement fixture keeps the original question unchanged:

> 圣园未花说自己一直在暗中支援哪个组织？

It retains `subject_terms=[圣园未花]`, `relation_hint=暗中支援的组织`,
`temporal_hint=一直`, and `negation_hint=explicit_negation`.  It creates Q1
through the public `QueryEngine.query` and asks Q2 through the public
`try_contextual_revisit` entrance; it neither calls an internal matcher nor
inserts a manifest or edge by hand.

The regression verifies all of the following:

| Check | Observed result |
| --- | --- |
| Q1 receipt and edge | ready and persisted |
| V16 contract / V17 manifest | present / `ready` |
| Public exact Q2 | hit; returned the newly created edge |
| Model activity on exact preflight | zero; ordinary-query fallback is wrapped to fail the test |
| Actually skipped stages | planner, embedding, and reranker |
| Still executed guards | contextual matcher and source closure |
| Edge mask | public exact lookup misses |
| Source text mutation | public exact lookup misses |
| Policy condition change | public exact lookup misses |
| Rebuild/restart | public exact lookup hits again |
| Missing scope counterpart | receipt remains ready but projection is explicitly `rejected_not_reconstructable` |

The runner's readable-label-to-opaque-key projection has its own regression
assertion.  The relevant focused suite completed with **57 tests passing** in
14.982 seconds:

```
tests.test_q1_q2_diagnostic_pilot
tests.test_query_learning_finalization_v3
tests.test_contextual_auto_revisit_v17
tests.test_contextual_revisit_runtime_manifest_v17
tests.test_contextual_revisit_contract_v16
tests.test_contextual_exact_revisit_t15
```

## One real-Source v11 attempt — preserved terminal result

Output package: `validation/v3-q1-q2-real-source-diagnostic-v11-20260907T141500Z`.
It cloned the static source database read-only; the source hash is
`04fa505c91d7ce4c1a7c8619e68194eddaa0b76717a13c0836d3cd96340db1b0`.
All 45 eligible episodes carried literal source evidence.  The manifest hash
is `87d11f8b0c4b08a7ba29836c8e8eaf6acd9a299593bbd9b3300e0bc7434d3fa1`.

The Q1/Q2 question was the unchanged rich question above.  The derived,
recorded opaque scope key was
`pilot-scope:sha256:e2ffe0b45373e429fa507153e0fc0c62be2b37534784c3cba24655ee78e9bdbd`.

| Item | Actual v11 observation |
| --- | --- |
| Run status | `diagnostic_complete`; Q1 `failed` |
| Wall time | 121,734 ms; Q1 120,038.739 ms |
| Q1 terminal error | `TimeoutError: query deadline exceeded before post_query_result` |
| Finalizer | not entered (`finalizer_calls=0`) |
| Receipt / edge / contract / runtime manifest | not observed; none created |
| Per-edge source verification and delivered evidence | not observed, because no edge exists |
| Candidate and selection output | not observed as a completed Q1 result |
| Q2 arms | all `not_run`, with their explicit dependent-Q1 terminal reason |
| Exact-revisit skipped stages | not observed; no Q2 entered the exact path |
| Formal score / promotion / canary | false / prohibited / prohibited |

The Q1 provider ledger records nine HTTP attempts across seven logical batches:
six successful, two DeepSeek read timeouts followed by the configured GLM-4.5V
fallbacks, and one final audit response discarded after the public deadline.
No unapproved model was used; GLM-4.5-Air was not invoked.  The observed Q1
stage times (ms) were: intent 10,219; embedding 1,468; rerank 8,062; answer
generation DeepSeek timeout 25,016 then GLM-4.5V 10,000; evidence audit 13,890;
answer correction DeepSeek timeout 25,016 then GLM-4.5V 10,844; final audit
late-discarded 14,687.  These calls belong to the existing answer pipeline,
not to exact-revisit reuse.

No second real run was made after this terminal result.  The preserved v10 raw
package was not altered or reclassified; its SHA-256 remains
`b25bf6b810145f65429b88abdc954b39aed54915dbfac82c13769795ddc61556`.
The v11 full-local package SHA-256 is
`8d5948bf80a60a6bae358c11ebf6b882777bc4a37e92febe2dc1e7f36bb986dc`.

## Deliverables

- `validation/v3-q1-q2-real-source-diagnostic-v11-20260907T141500Z/q1_q2_case.full_local.json` — full local, observed v11 trace.
- `validation/v3-q1-q2-real-source-diagnostic-v11-20260907T141500Z/q1_q2_case.report.md` — readable diagnostic case report.
- `validation/v3-q1-q2-real-source-diagnostic-v10-20260907T131038Z/q1_q2_case.full_local.json` — untouched original v10 record.

The local regression demonstrates the repaired integration and fails closed on
mask, source, policy, and scope changes.  A real-Source exact-reuse outcome
remains unobserved, rather than inferred, until a separately authorized trial
can complete Q1 within its existing deadline.
