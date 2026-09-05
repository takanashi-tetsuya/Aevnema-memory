# Prompt catalog

`memory_prompts.py` is the only canonical location for model-facing text in
the associative-memory engine. It contains:

- ingestion and document-map prompts;
- Episode and Concept extraction prompts;
- temporal, granularity, entailment, and factual audits;
- relation discovery and autonomous growth prompts;
- query planning, multi-hop retrieval, reranking, and coverage prompts;
- answer generation, answer audit, correction, retry, and JSON repair prompts.

`src/memory_demo/llm/prompts.py` is an import facade only. Runtime modules
must import prompt constants and builders through that facade and must not add
model instructions inline.

Experimental cross-project answer-contract prompts are isolated in this directory as
well, but are not imported by the associative-memory runtime:

- `answer_evidence_contract_prompts.py`: early answer evidence contract experiment;
- `evidence_role_prompts.py`: candidate evidence-role experiment;
- `layered_answer_contract_prompts.py`: Support/Review/Context answer experiment;
- `answer_claim_consistency_prompts.py`: rendered-answer semantic delta audit.
- `answer_persistence_repair_prompts.py`: anchored answer-span extraction,
  runtime receipt/question premise scope classification, constrained repair,
  and persona-aware repair quality review.
- `request_scoped_answer_prompts.py`: experimental immutable request-contract
  renderer, public action summaries, typed-limitation directive rendering, and
  offline A/B judge prompts.
