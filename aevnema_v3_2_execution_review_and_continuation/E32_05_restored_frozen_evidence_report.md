# Frozen Q2 evidence-delivery matrix

This is a non-scoring Q2-only observation. The Q1 state was read from a frozen, pre-existing snapshot; this run made no Q1, answer generation, answer audit, Q2-learning, association-growth, or edge-use write. A live-preparation run may still report an evidence-coverage audit as part of retrieval selection; it is not an answer audit.

| Variant | Condition | Run | Evidence | Edge state | Delivered episodes | HTTP in arm | Evidence ms |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| same_text | edge_available | completed | complete | not_entered | [18, 14] | 0 | 76.559 |
| same_text | this_run_edge_masked | completed | complete | not_entered | [18, 14] | 0 | 84.866 |
| paraphrase | edge_available | completed | complete | not_entered | [19, 18] | 0 | 77.452 |
| paraphrase | this_run_edge_masked | completed | complete | not_entered | [19, 18] | 0 | 88.432 |
| partial_clue_no_context | edge_available | completed | complete | not_entered | [18, 19] | 0 | 79.689 |
| partial_clue_no_context | this_run_edge_masked | completed | complete | not_entered | [18, 19] | 0 | 80.09 |
| near_neighbor_counterexample | edge_available | completed | complete | not_entered | [14, 32] | 0 | 88.694 |
| near_neighbor_counterexample | this_run_edge_masked | completed | complete | not_entered | [14, 32] | 0 | 129.47 |

A `source_bound` delivery state only identifies a provenance-bound Source excerpt. It is not a formal correctness or generalization score; gold was not loaded.

Preparation records, when present, are separate from both conditions. A `frozen_query_plan` is shared only when its plan hash is identical in both arm records; otherwise the result is a live-input observation, not a frozen-input comparison.
