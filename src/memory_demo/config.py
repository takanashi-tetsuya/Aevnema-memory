from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    """Load a small KEY=VALUE env file without an external dependency."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(slots=True)
class ModelConfig:
    base_url: str = "https://api.siliconflow.cn/v1"
    api_key: str = ""
    embedding_model: str = "Pro/BAAI/bge-m3"
    reranker_model: str = "Pro/BAAI/bge-reranker-v2-m3"
    reasoning_model: str = "deepseek-ai/DeepSeek-V3.2"
    fallback_model: str = "zai-org/GLM-4.5V"
    reasoning_max_tokens: int | None = None
    reasoning_enable_thinking: bool | None = None
    embedding_dimension: int = 1024
    # A single stalled Source must not hold an ordered import batch for five
    # minutes. Query plans may lower this further on a per-request basis.
    timeout_seconds: float = 90.0
    max_retries: int = 2
    # File-level import concurrency can be intentionally high while each file
    # is waiting on database work.  Bound only the external HTTP calls so a
    # burst of file workers cannot exhaust local socket resources.
    max_concurrent_requests: int = 8


@dataclass(slots=True)
class SegmentConfig:
    target_chars: int = 6_000
    max_chars: int = 8_000
    overlap_chars: int = 800
    minimum_blocks: int = 2
    # Structured reference documents contain independent fact records rather
    # than one continuous scene. Bound records per Source and do not overlap
    # them, so the extractor cannot collapse an encyclopedia into one Episode.
    reference_max_blocks: int = 10


@dataclass(slots=True)
class ParagraphConfig:
    """Deterministic Source sub-chunks used as a reversible recall channel."""

    enabled: bool = False
    target_chars: int = 900
    max_chars: int = 1_400
    overlap_chars: int = 200
    minimum_chars: int = 200


@dataclass(slots=True)
class IngestionConfig:
    """Execution policy for model-heavy import stages.

    ``balanced`` keeps every evidence boundary but combines compatible review
    passes and removes repeated context.  ``strict`` restores the historical
    multi-pass prompts for controlled comparisons.
    """

    prepare_workers: int = 4
    # ``legacy`` preserves the multi-pass experimental pipeline.  The
    # reversible ``single_pass_evidence`` arm performs one Episode extraction,
    # requires literal Source evidence, and disables automatic rewrites.
    # ``document_map_assisted`` first creates a transient whole-file map and
    # binds Episode boundaries to its stages. ``document_map_contextual`` uses
    # the same map only as fallible context, leaving boundaries to Source.
    # ``adaptive_anchor_map`` creates literal-only navigation only when one
    # logical file was actually split across multiple Source rows.
    # ``single_pass_audited`` adds one compact claim-vs-evidence review without
    # reopening Episode boundaries.
    episode_extraction_profile: str = "legacy"
    relation_workers: int = 4
    relation_batch_size: int = 24
    build_inference_relations: bool = True
    task_max_attempts: int = 2
    episode_audit_mode: str = "combined"
    # High-quality corpus builds can review every Source segment.  The default
    # remains adaptive because this adds one reasoning call per segment.
    episode_audit_always: bool = False
    # A focused second-model pass checks speaker/actor/object and identity
    # claims without changing Episode boundaries. ``adaptive`` limits the
    # additional call to dialogue segments containing factual-risk markers.
    episode_factual_audit_mode: str = "off"
    # Empty means use the primary reasoning model.  A transport fallback is
    # not automatically a competent independent evidence auditor.
    episode_factual_audit_model: str = ""
    # Bound evidence-review output so one long Source cannot create a single
    # multi-minute request. Import workers provide outer concurrency.
    episode_factual_audit_batch_size: int = 3
    # A clean primary empty result is never silently treated as a non-story
    # segment.  An independently configured reviewer must prove that every
    # substantive line is control/title/non-event material before a task can
    # be marked skipped.  ``off`` is retained only for controlled rollback.
    empty_episode_audit_mode: str = "adversarial"
    # Empty selects the configured fallback only when it differs from the
    # primary reasoning model; otherwise the review fails closed.
    empty_episode_audit_model: str = ""
    second_pass_context_mode: str = "compact"


@dataclass(slots=True)
class ConceptExtractionConfig:
    """Concept extraction profile; conservative remains an explicit rollback."""

    profile: str = "fine_grained"
    target_min: int = 4
    target_max: int = 10
    # Stage 12: discovery may remain aggressive while durable graph admission
    # is independently audited.  Disabled preserves the original import path.
    promotion_gate_enabled: bool = False
    promotion_batch_size: int = 48
    promotion_recurrence_fallback: int = 2
    # Experimental graph-utility gate.  A new label seen in fewer independent
    # Episodes is kept transient even when the LLM calls it durable.  Existing
    # Concept reuse remains allowed because it can add a new bridge immediately.
    promotion_min_distinct_episodes: int = 1
    # A singleton below this similarity cannot be promoted by the graph-utility
    # gate and historical Stage-14 data contained no successful reuse below
    # 0.596.  It can therefore be classified transient without an LLM call.
    promotion_prefilter_enabled: bool = True
    promotion_singleton_reuse_similarity_floor: float = 0.595
    # The embedding text and long repeated contexts are not needed by the
    # admission judge after similarity candidates have already been computed.
    promotion_compact_prompt: bool = True
    promotion_evidence_text_limit: int = 2
    promotion_evidence_text_chars: int = 480
    promotion_similar_description_chars: int = 220


@dataclass(slots=True)
class RetrievalConfig:
    episode_top_k: int = 40
    concept_top_k: int = 20
    # Vector-matched Concepts below this one-hop graph reach are ignored as
    # seeds. Exact alias matches are injected later and deliberately bypass it.
    concept_seed_min_reachable_episodes: int = 1
    sparse_enabled: bool = True
    sparse_episode_top_k: int = 80
    sparse_source_top_k: int = 16
    sparse_source_episode_expansion_limit: int = 8
    sparse_episode_rrf_weight: float = 1.20
    sparse_source_rrf_weight: float = 0.70
    # A JSON/TXT file can be split into several Source rows.  When independent
    # atomic anchors repeatedly hit one logical source_key, expose the bounded
    # file cohort to reranking so a later segment is not lost at a Source-row
    # boundary.  This is a recall channel only and makes no factual claim.
    source_key_cohort_enabled: bool = True
    source_key_cohort_min_anchor_hits: int = 2
    source_key_cohort_max_keys: int = 1
    source_key_cohort_max_episodes_per_key: int = 40
    source_key_cohort_total_limit: int = 64
    source_key_cohort_score_ratio: float = 0.95
    # ``always`` preserves the deep/background path. ``entity_resolved``
    # spends a second LLM planning call only when a later question clause
    # depends on the unknown person/organization found by an earlier clause.
    # ``missing_slots`` waits for the local initial candidate pass and invokes
    # the planner only if an initial answer/constraint slot has no candidate.
    # ``off`` is reserved for controlled retrieval experiments.
    followup_planning_mode: str = "always"
    # V3.1 diagnostic answer completion profile.  These are admission
    # envelopes, not a promise that a provider will respond within the
    # interval.  A correction may start only when the shared request deadline
    # can still cover its primary attempt, one permitted transport fallback,
    # the required re-audit, and local finalization/delivery reserve.
    answer_correction_max_revisions: int = 1
    answer_correction_attempt_envelope_seconds: float = 25.0
    answer_correction_fallback_envelope_seconds: float = 25.0
    answer_reaudit_envelope_seconds: float = 25.0
    answer_finalization_reserve_seconds: float = 5.0
    rerank_enabled: bool = True
    # ``llm`` preserves the evidence-coverage reasoner. ``cross_encoder`` uses
    # the dedicated reranker and then reapplies deterministic evidence floors.
    rerank_backend: str = "llm"
    rerank_candidate_limit: int = 100
    # Candidate@100 remains the high-recall pool.  A smaller value performs a
    # deterministic, answer-slot-aware compression before any LLM rerank call.
    # Keeping the default equal to rerank_candidate_limit is a no-op.
    rerank_precompression_limit: int = 100
    rerank_shortlist_limit: int = 48
    rerank_coverage_audit_enabled: bool = True
    rerank_audit_enabled: bool = True
    # strict: historical three-pass rerank; adaptive: spend extra passes only
    # on complex or visibly thin coverage; lean: one coverage pass.
    rerank_review_mode: str = "adaptive"
    # Retrieval may use many paraphrases, but they are not all independent
    # answer facts. Bound only the LLM evidence-slot prompt; dense/sparse recall
    # and Candidate@N keep every planned query and candidate.
    rerank_atomic_query_limit: int = 24
    rerank_strict_atomic_query_threshold: int = 24
    rerank_compress_coverage_group_threshold: int = 1
    # High-risk multi-hop questions keep a small deterministic floor from the
    # first independent atomic channels.  The LLM ranks the remaining slots
    # but cannot erase an entire explicitly requested evidence subproblem.
    rerank_atomic_floor_enabled: bool = True
    rerank_atomic_floor_query_limit: int = 6
    rerank_atomic_floor_per_query: int = 1
    rerank_atomic_floor_total_limit: int = 6
    # Questions that explicitly combine a clue with a causal/relational
    # constraint receive a dedicated cross-constraint slot.  Its top
    # candidates are protected before the broad sparse/atomic floor so a late
    # but independently requested hop is not starved by earlier subqueries.
    rerank_constraint_floor_per_query: int = 1
    rerank_constraint_floor_total_limit: int = 3
    # Episode extraction may split the answer noun/action across adjacent
    # summaries in one source file. Preserve a tiny local window around an
    # explicit __answer_slot__ floor hit without enabling Paragraph retrieval.
    rerank_answer_slot_neighbor_radius: int = 1
    rerank_answer_slot_neighbor_total_limit: int = 4
    # Constraint candidates are admitted to the LLM reranker, but are not
    # forced into the final evidence set. This protects a late causal hop from
    # global multi-query fusion without turning a retrieval hint into a fact.
    rerank_constraint_candidate_per_query: int = 12
    rerank_constraint_candidate_total_limit: int = 24
    rerank_question_sparse_floor_limit: int = 16
    # Hard floors must leave most of Top-K to evidence coverage/compression.
    # Sparse whole-question hits remain a recall lane and are not mandatory.
    rerank_combined_floor_limit: int = 8
    paragraph_top_k: int = 16
    paragraph_episode_expansion_limit: int = 6
    paragraph_rrf_weight: float = 0.35
    # Stage 13 separates the two possible Paragraph effects.  Seed expansion
    # changes recall; rerank context lets the selector inspect matched raw text
    # without changing the graph candidate set.
    paragraph_seed_enabled: bool = True
    paragraph_rerank_context_enabled: bool = False
    paragraph_rerank_context_per_source: int = 2
    paragraph_rerank_context_chars: int = 1_000
    # Paragraph is an auxiliary recall channel.  It may add Episode seeds that
    # the mature baseline missed, but must not reorder baseline Episode anchors.
    paragraph_recall_only: bool = True
    episode_relation_candidate_k: int = 4
    graph_beam_width: int = 20
    graph_max_hops: int = 3
    growth_max_rounds: int = 2
    growth_episode_limit: int = 40
    # Provisional growth rows may participate in this query, but only rows
    # that reach the final answer path remain durable after the query returns.
    growth_persist_only_used: bool = True
    # Keep proposed/reinforced rows in RAM until the counterfactual utility gate
    # has decided what is allowed to become durable.
    growth_staging_enabled: bool = True
    # Compare the same frozen plan with query-grown rows visible and masked.
    # Path use alone is insufficient: persistence requires new direct Episode
    # evidence in the treatment result.
    growth_counterfactual_utility_enabled: bool = True
    candidate_limit: int = 600
    answer_episode_limit: int = 30
    answer_concept_limit: int = 20
    answer_path_limit: int = 24
    # Dense legacy graph paths must not automatically displace a strong base
    # Top-K. Only a small number of audited query-grown bridges receive slots.
    learned_bridge_slots: int = 2
    learned_bridge_min_query_relevance: float = 0.03
    learned_bridge_duplicate_threshold: float = 0.05
    # Reversible Stage 9 experiment: directly search audited Association text
    # and use the matched relation only as a cue for its endpoint nodes.
    association_cue_enabled: bool = False
    association_cue_top_k: int = 8
    association_cue_min_similarity: float = 0.45
    association_cue_rrf_weight: float = 0.80
    association_cue_semantic_gate_enabled: bool = False
    association_cue_semantic_gate_max_selected: int = 4
    # A high-confidence, dual-audited query-grown Episode↔Episode edge can
    # serve as a reusable evidence capsule.  The local selector keeps both
    # source-grounded endpoints and the deterministic evidence floor, avoiding
    # a repeated remote reranker call for a close paraphrase.  This is opt-in
    # because its precision threshold must be calibrated on each corpus.
    association_cue_fast_path_enabled: bool = False
    association_cue_fast_path_min_similarity: float = 0.62
    association_cue_fast_path_min_margin: float = 0.08
    association_cue_fast_path_min_confidence: float = 0.72
    association_cue_fast_path_max_edges: int = 1
    association_cue_fast_path_evidence_limit: int = 2
    association_cue_fast_path_local_enabled: bool = True
    association_cue_fast_path_local_min_coverage: float = 0.18
    association_cue_fast_path_local_min_margin: float = 0.04
    # Contextual association v1: a local AND gate over a whole-query context
    # prototype and one or more missing-evidence (need) prototypes.  Keeping
    # this opt-in preserves the baseline retrieval behaviour and makes rollback
    # a single configuration change.
    contextual_association_enabled: bool = False
    contextual_association_shadow: bool = True
    # W06 prepared-early is a separately controlled diagnostic lane.  It may
    # inspect a request's already-bound requirements and vector bundle before
    # broad traversal, but the initial implementation is shadow-only: it
    # never short-circuits ordinary retrieval or represents a proposal as
    # delivered evidence.
    contextual_prepared_early_enabled: bool = False
    contextual_prepared_early_shadow: bool = True
    # A separate local diagnostic may admit already source-validated prepared
    # candidates into the ordinary graph seed pool.  It never permits an
    # answer/early-stop path and remains off in normal operation.
    contextual_prepared_early_candidate_pool_enabled: bool = False
    # W07 diagnostic-only key ablation.  ``cn`` is the ordinary double-key
    # rule; the other values exist only for a frozen, local comparison.
    contextual_prepared_early_scoring_mode: str = "cn"
    contextual_context_top_k: int = 8
    contextual_need_top_k: int = 8
    contextual_edge_top_k: int = 16
    contextual_context_threshold: float = 0.55
    contextual_need_threshold: float = 0.55
    contextual_combine_mode: str = "product"
    contextual_endpoint_limit_light: int = 1
    contextual_endpoint_limit_standard: int = 2
    contextual_endpoint_limit_deep: int = 4
    contextual_max_candidates_per_turn: int = 8
    contextual_probation_limit: int = 10000
    contextual_probation_ttl: int = 2_592_000
    contextual_min_distinct_successes: int = 2
    # v2 does not allow observations to promote an edge until strict
    # Treatment/Masked + single-edge attribution is enabled by an operator.
    contextual_promotion_enabled: bool = False
    contextual_noop_decay: float = 0.98
    contextual_harm_multiplier: float = 0.5
    contextual_association_allow_network: bool = False
    # T16 is a deliberately narrow, HMAC-authorized rewrite lane.  It stays
    # disabled until an operator also supplies its process-local HMAC key via
    # MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_HMAC_KEY(_ID); the secret never
    # becomes part of this serialisable configuration object.
    contextual_restricted_rewrite_enabled: bool = False
    answer_whole_question_anchor_episodes: int = 10
    answer_anchor_episodes_per_query: int = 6
    source_excerpt_chars: int = 3_000
    # Raw Source/answer/audit checkpoints are an explicitly enabled,
    # experiment-local artifact.  Ordinary operational JSONL stays redacted.
    answer_evidence_checkpoint_enabled: bool = False
    concept_relation_min_similarity: float = 0.72

    def validate(self) -> None:
        if self.contextual_combine_mode not in {"product", "minimum", "geomean"}:
            raise ValueError(
                "contextual_combine_mode must be product, minimum, or geomean"
            )
        if self.contextual_prepared_early_scoring_mode not in {"w", "c", "n", "cn"}:
            raise ValueError(
                "contextual_prepared_early_scoring_mode must be w, c, n, or cn"
            )
        for name in (
            "contextual_context_threshold",
            "contextual_need_threshold",
            "contextual_noop_decay",
            "contextual_harm_multiplier",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "contextual_context_top_k",
            "contextual_need_top_k",
            "contextual_edge_top_k",
            "contextual_endpoint_limit_light",
            "contextual_endpoint_limit_standard",
            "contextual_endpoint_limit_deep",
            "contextual_max_candidates_per_turn",
            "contextual_probation_limit",
            "contextual_min_distinct_successes",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.contextual_probation_ttl) < 0:
            raise ValueError("contextual_probation_ttl must be non-negative")


@dataclass(slots=True)
class WeightConfig:
    llm: float = 0.80
    embedding: float = 0.10
    structural: float = 0.05
    evidence: float = 0.05
    learning_rate: float = 0.20


@dataclass(slots=True)
class AppConfig:
    database_path: Path = Path("database/memory_demo.db")
    log_dir: Path = Path("logs")
    prompt_version: str = "v3.45_structural_grounded_roles"
    optimization_profile: str = "balanced"
    model: ModelConfig = field(default_factory=ModelConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    paragraph: ParagraphConfig = field(default_factory=ParagraphConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    concept_extraction: ConceptExtractionConfig = field(
        default_factory=ConceptExtractionConfig
    )
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "AppConfig":
        load_env_file(env_file)
        config = cls(
            database_path=Path(os.getenv("MEMORY_DB_PATH", "database/memory_demo.db")),
            log_dir=Path(os.getenv("MEMORY_LOG_DIR", "logs")),
        )
        config.model.api_key = os.getenv("SILICONFLOW_API_KEY", "")
        reasoning_model = os.getenv("MEMORY_REASONING_MODEL")
        if reasoning_model is not None and reasoning_model.strip():
            config.model.reasoning_model = reasoning_model.strip()
        fallback_model = os.getenv("MEMORY_FALLBACK_MODEL")
        if fallback_model is not None and fallback_model.strip():
            config.model.fallback_model = fallback_model.strip()
        config.model.reranker_model = os.getenv(
            "MEMORY_RERANKER_MODEL", config.model.reranker_model
        ).strip()
        model_timeout = os.getenv("MEMORY_MODEL_TIMEOUT_SECONDS")
        if model_timeout is not None:
            config.model.timeout_seconds = max(1.0, float(model_timeout))
        model_request_limit = os.getenv("MEMORY_MODEL_MAX_CONCURRENT_REQUESTS")
        if model_request_limit is not None:
            config.model.max_concurrent_requests = max(
                1, int(model_request_limit)
            )
        reasoning_max_tokens = os.getenv("MEMORY_REASONING_MAX_TOKENS")
        if reasoning_max_tokens is not None:
            config.model.reasoning_max_tokens = max(1, int(reasoning_max_tokens))
        reasoning_thinking = os.getenv("MEMORY_REASONING_ENABLE_THINKING")
        if reasoning_thinking is not None:
            normalized_thinking = reasoning_thinking.strip().casefold()
            config.model.reasoning_enable_thinking = (
                None
                if normalized_thinking in {"", "auto", "default"}
                else normalized_thinking not in {"0", "false", "no", "off"}
            )
        optimization_profile = (
            os.getenv("MEMORY_OPTIMIZATION_PROFILE", config.optimization_profile)
            .strip()
            .casefold()
        )
        config.apply_optimization_profile(optimization_profile)
        segment_target = os.getenv("MEMORY_SEGMENT_TARGET_CHARS")
        segment_max = os.getenv("MEMORY_SEGMENT_MAX_CHARS")
        segment_overlap = os.getenv("MEMORY_SEGMENT_OVERLAP_CHARS")
        if segment_target is not None:
            config.segment.target_chars = max(256, int(segment_target))
        if segment_max is not None:
            config.segment.max_chars = max(256, int(segment_max))
        if segment_overlap is not None:
            config.segment.overlap_chars = max(0, int(segment_overlap))
        if config.segment.target_chars > config.segment.max_chars:
            raise ValueError(
                "MEMORY_SEGMENT_TARGET_CHARS must not exceed MEMORY_SEGMENT_MAX_CHARS"
            )
        if config.segment.overlap_chars >= config.segment.max_chars:
            raise ValueError(
                "MEMORY_SEGMENT_OVERLAP_CHARS must be smaller than "
                "MEMORY_SEGMENT_MAX_CHARS"
            )
        paragraph_enabled = os.getenv("MEMORY_PARAGRAPH_ENABLED")
        if paragraph_enabled is not None:
            config.paragraph.enabled = paragraph_enabled.strip().casefold() not in {
                "0",
                "false",
                "no",
                "off",
            }
        concept_profile = os.getenv("MEMORY_CONCEPT_PROFILE")
        if concept_profile:
            normalized_profile = concept_profile.strip().casefold()
            if normalized_profile not in {"conservative", "fine_grained"}:
                raise ValueError(
                    "MEMORY_CONCEPT_PROFILE must be conservative or fine_grained"
                )
            config.concept_extraction.profile = normalized_profile
        task_max_attempts = os.getenv("MEMORY_IMPORT_TASK_MAX_ATTEMPTS")
        if task_max_attempts is not None:
            config.ingestion.task_max_attempts = max(1, int(task_max_attempts))
        episode_profile = os.getenv("MEMORY_IMPORT_EPISODE_PROFILE")
        if episode_profile is not None:
            normalized_episode_profile = episode_profile.strip().casefold()
            if normalized_episode_profile not in {
                "legacy",
                "single_pass_evidence",
                "single_pass_audited",
                "document_map_assisted",
                "document_map_contextual",
                "adaptive_anchor_map",
                "source_scoped_plain",
            }:
                raise ValueError(
                    "MEMORY_IMPORT_EPISODE_PROFILE must be legacy, "
                    "single_pass_evidence, single_pass_audited, or "
                    "document_map_assisted, document_map_contextual, or "
                    "adaptive_anchor_map, or source_scoped_plain"
                )
            config.ingestion.episode_extraction_profile = normalized_episode_profile
            if normalized_episode_profile == "single_pass_evidence":
                config.prompt_version = "v4.1_single_pass_line_spans"
            elif normalized_episode_profile == "single_pass_audited":
                config.prompt_version = "v4.4_single_pass_entailment_roles"
            elif normalized_episode_profile == "document_map_assisted":
                config.prompt_version = "v4.5_document_map_planned_audited"
            elif normalized_episode_profile == "document_map_contextual":
                config.prompt_version = "v4.6_document_map_context_audited"
            elif normalized_episode_profile == "adaptive_anchor_map":
                config.prompt_version = "v4.7_adaptive_literal_anchor_audited"
            elif normalized_episode_profile == "source_scoped_plain":
                config.prompt_version = "v5.0_source_scoped_program_owned"
        prepare_workers = os.getenv("MEMORY_IMPORT_PREPARE_WORKERS")
        if prepare_workers is not None:
            config.ingestion.prepare_workers = max(1, int(prepare_workers))
        relation_workers = os.getenv("MEMORY_IMPORT_RELATION_WORKERS")
        if relation_workers is not None:
            config.ingestion.relation_workers = max(1, int(relation_workers))
        relation_batch_size = os.getenv("MEMORY_IMPORT_RELATION_BATCH_SIZE")
        if relation_batch_size is not None:
            config.ingestion.relation_batch_size = max(1, int(relation_batch_size))
        episode_audit_always = os.getenv("MEMORY_IMPORT_EPISODE_AUDIT_ALWAYS")
        if episode_audit_always is not None:
            config.ingestion.episode_audit_always = (
                episode_audit_always.strip().casefold()
                not in {"0", "false", "no", "off"}
            )
        factual_audit_mode = os.getenv("MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODE")
        if factual_audit_mode is not None:
            normalized_mode = factual_audit_mode.strip().casefold()
            if normalized_mode not in {"off", "adaptive", "always"}:
                raise ValueError(
                    "MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODE must be off, adaptive, or always"
                )
            config.ingestion.episode_factual_audit_mode = normalized_mode
        factual_audit_model = os.getenv("MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODEL")
        if factual_audit_model is not None:
            config.ingestion.episode_factual_audit_model = factual_audit_model.strip()
        factual_audit_batch_size = os.getenv(
            "MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_BATCH_SIZE"
        )
        if factual_audit_batch_size is not None:
            config.ingestion.episode_factual_audit_batch_size = max(
                1, int(factual_audit_batch_size)
            )
        empty_audit_mode = os.getenv("MEMORY_IMPORT_EMPTY_EPISODE_AUDIT_MODE")
        if empty_audit_mode is not None:
            normalized_mode = empty_audit_mode.strip().casefold()
            if normalized_mode not in {"off", "adversarial"}:
                raise ValueError(
                    "MEMORY_IMPORT_EMPTY_EPISODE_AUDIT_MODE must be off or adversarial"
                )
            config.ingestion.empty_episode_audit_mode = normalized_mode
        empty_audit_model = os.getenv("MEMORY_IMPORT_EMPTY_EPISODE_AUDIT_MODEL")
        if empty_audit_model is not None:
            config.ingestion.empty_episode_audit_model = empty_audit_model.strip()
        sparse_enabled = os.getenv("MEMORY_SPARSE_ENABLED")
        if sparse_enabled is not None:
            config.retrieval.sparse_enabled = sparse_enabled.strip().casefold() not in {
                "0",
                "false",
                "no",
                "off",
            }
        source_key_cohort_enabled = os.getenv("MEMORY_SOURCE_KEY_COHORT_ENABLED")
        if source_key_cohort_enabled is not None:
            config.retrieval.source_key_cohort_enabled = (
                source_key_cohort_enabled.strip().casefold()
                not in {"0", "false", "no", "off"}
            )
        rerank_enabled = os.getenv("MEMORY_RERANK_ENABLED")
        if rerank_enabled is not None:
            config.retrieval.rerank_enabled = rerank_enabled.strip().casefold() not in {
                "0",
                "false",
                "no",
                "off",
            }
        rerank_backend = os.getenv("MEMORY_RERANK_BACKEND")
        if rerank_backend:
            normalized_backend = rerank_backend.strip().casefold()
            if normalized_backend not in {"llm", "cross_encoder"}:
                raise ValueError("MEMORY_RERANK_BACKEND must be llm or cross_encoder")
            config.retrieval.rerank_backend = normalized_backend
        rerank_precompression_limit = os.getenv("MEMORY_RERANK_PRECOMPRESSION_LIMIT")
        if rerank_precompression_limit is not None:
            config.retrieval.rerank_precompression_limit = max(
                1, int(rerank_precompression_limit)
            )
        rerank_coverage_audit_enabled = os.getenv(
            "MEMORY_RERANK_COVERAGE_AUDIT_ENABLED"
        )
        if rerank_coverage_audit_enabled is not None:
            config.retrieval.rerank_coverage_audit_enabled = (
                rerank_coverage_audit_enabled.strip().casefold()
                not in {"0", "false", "no", "off"}
            )
        growth_persist_only_used = os.getenv("MEMORY_GROWTH_PERSIST_ONLY_USED")
        if growth_persist_only_used is not None:
            config.retrieval.growth_persist_only_used = (
                growth_persist_only_used.strip().casefold()
                not in {"0", "false", "no", "off"}
            )
        growth_counterfactual_utility = os.getenv(
            "MEMORY_GROWTH_COUNTERFACTUAL_UTILITY_ENABLED"
        )
        if growth_counterfactual_utility is not None:
            config.retrieval.growth_counterfactual_utility_enabled = (
                growth_counterfactual_utility.strip().casefold()
                not in {"0", "false", "no", "off"}
            )
        growth_staging_enabled = os.getenv("MEMORY_GROWTH_STAGING_ENABLED")
        if growth_staging_enabled is not None:
            config.retrieval.growth_staging_enabled = (
                growth_staging_enabled.strip().casefold()
                not in {"0", "false", "no", "off"}
            )
        def env_bool(name: str, current: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return current
            return value.strip().casefold() not in {"0", "false", "no", "off"}

        config.retrieval.contextual_association_enabled = env_bool(
            "MEMORY_CONTEXTUAL_ASSOCIATION_ENABLED",
            config.retrieval.contextual_association_enabled,
        )
        config.retrieval.contextual_association_shadow = env_bool(
            "MEMORY_CONTEXTUAL_ASSOCIATION_SHADOW",
            config.retrieval.contextual_association_shadow,
        )
        config.retrieval.contextual_prepared_early_enabled = env_bool(
            "MEMORY_CONTEXTUAL_PREPARED_EARLY_ENABLED",
            config.retrieval.contextual_prepared_early_enabled,
        )
        config.retrieval.contextual_prepared_early_shadow = env_bool(
            "MEMORY_CONTEXTUAL_PREPARED_EARLY_SHADOW",
            config.retrieval.contextual_prepared_early_shadow,
        )
        config.retrieval.contextual_prepared_early_candidate_pool_enabled = env_bool(
            "MEMORY_CONTEXTUAL_PREPARED_EARLY_CANDIDATE_POOL_ENABLED",
            config.retrieval.contextual_prepared_early_candidate_pool_enabled,
        )
        prepared_early_scoring_mode = os.getenv(
            "MEMORY_CONTEXTUAL_PREPARED_EARLY_SCORING_MODE"
        )
        if prepared_early_scoring_mode is not None:
            config.retrieval.contextual_prepared_early_scoring_mode = (
                prepared_early_scoring_mode.strip().casefold()
            )
        config.retrieval.contextual_promotion_enabled = env_bool(
            "MEMORY_CONTEXTUAL_ASSOCIATION_PROMOTION_ENABLED",
            config.retrieval.contextual_promotion_enabled,
        )
        config.retrieval.contextual_association_allow_network = env_bool(
            "MEMORY_CONTEXTUAL_ASSOCIATION_ALLOW_NETWORK",
            config.retrieval.contextual_association_allow_network,
        )
        config.retrieval.contextual_restricted_rewrite_enabled = env_bool(
            "MEMORY_CONTEXTUAL_RESTRICTED_REWRITE_ENABLED",
            config.retrieval.contextual_restricted_rewrite_enabled,
        )
        if config.retrieval.contextual_association_allow_network:
            raise ValueError(
                "contextual association is local-only; "
                "MEMORY_CONTEXTUAL_ASSOCIATION_ALLOW_NETWORK must be 0"
            )
        integer_contextual_keys = {
            "contextual_context_top_k": "MEMORY_CONTEXTUAL_ASSOCIATION_CONTEXT_TOP_K",
            "contextual_need_top_k": "MEMORY_CONTEXTUAL_ASSOCIATION_NEED_TOP_K",
            "contextual_edge_top_k": "MEMORY_CONTEXTUAL_ASSOCIATION_EDGE_TOP_K",
            "contextual_endpoint_limit_light": "MEMORY_CONTEXTUAL_ASSOCIATION_ENDPOINT_LIMIT_LIGHT",
            "contextual_endpoint_limit_standard": "MEMORY_CONTEXTUAL_ASSOCIATION_ENDPOINT_LIMIT_STANDARD",
            "contextual_endpoint_limit_deep": "MEMORY_CONTEXTUAL_ASSOCIATION_ENDPOINT_LIMIT_DEEP",
            "contextual_max_candidates_per_turn": "MEMORY_CONTEXTUAL_ASSOCIATION_MAX_CANDIDATES_PER_TURN",
            "contextual_probation_limit": "MEMORY_CONTEXTUAL_ASSOCIATION_PROBATION_LIMIT",
            "contextual_probation_ttl": "MEMORY_CONTEXTUAL_ASSOCIATION_PROBATION_TTL",
            "contextual_min_distinct_successes": "MEMORY_CONTEXTUAL_ASSOCIATION_MIN_DISTINCT_SUCCESSES",
        }
        for attribute, env_name in integer_contextual_keys.items():
            raw = os.getenv(env_name)
            if raw is not None:
                setattr(config.retrieval, attribute, max(0, int(raw)))
        float_contextual_keys = {
            "contextual_context_threshold": "MEMORY_CONTEXTUAL_ASSOCIATION_CONTEXT_THRESHOLD",
            "contextual_need_threshold": "MEMORY_CONTEXTUAL_ASSOCIATION_NEED_THRESHOLD",
            "contextual_noop_decay": "MEMORY_CONTEXTUAL_ASSOCIATION_NOOP_DECAY",
            "contextual_harm_multiplier": "MEMORY_CONTEXTUAL_ASSOCIATION_HARM_MULTIPLIER",
        }
        for attribute, env_name in float_contextual_keys.items():
            raw = os.getenv(env_name)
            if raw is not None:
                setattr(config.retrieval, attribute, float(raw))
        combine_mode = os.getenv("MEMORY_CONTEXTUAL_ASSOCIATION_COMBINE_MODE")
        if combine_mode:
            normalized_combine = combine_mode.strip().casefold()
            if normalized_combine not in {"product", "minimum", "geomean"}:
                raise ValueError(
                    "MEMORY_CONTEXTUAL_ASSOCIATION_COMBINE_MODE must be product, minimum, or geomean"
                )
            config.retrieval.contextual_combine_mode = normalized_combine
        config.retrieval.validate()
        return config

    def apply_optimization_profile(self, profile: str) -> None:
        normalized = profile.strip().casefold()
        if normalized not in {"strict", "balanced", "lean"}:
            raise ValueError(
                "MEMORY_OPTIMIZATION_PROFILE must be strict, balanced, or lean"
            )
        self.optimization_profile = normalized
        if normalized == "strict":
            self.ingestion.prepare_workers = 1
            self.ingestion.relation_workers = 1
            self.ingestion.relation_batch_size = 1
            self.ingestion.episode_audit_mode = "split"
            self.ingestion.second_pass_context_mode = "full"
            self.concept_extraction.promotion_batch_size = 20
            self.concept_extraction.promotion_prefilter_enabled = False
            self.concept_extraction.promotion_compact_prompt = False
            self.retrieval.rerank_review_mode = "strict"
        elif normalized == "lean":
            self.ingestion.prepare_workers = 4
            self.ingestion.relation_workers = 4
            self.ingestion.relation_batch_size = 32
            self.ingestion.episode_audit_mode = "combined"
            self.ingestion.second_pass_context_mode = "compact"
            self.concept_extraction.promotion_batch_size = 64
            self.concept_extraction.promotion_prefilter_enabled = True
            self.concept_extraction.promotion_compact_prompt = True
            self.retrieval.rerank_review_mode = "lean"
        else:
            self.ingestion.prepare_workers = 4
            self.ingestion.relation_workers = 4
            self.ingestion.relation_batch_size = 24
            self.ingestion.episode_audit_mode = "combined"
            self.ingestion.second_pass_context_mode = "compact"
            self.concept_extraction.promotion_batch_size = 48
            self.concept_extraction.promotion_prefilter_enabled = True
            self.concept_extraction.promotion_compact_prompt = True
            self.retrieval.rerank_review_mode = "adaptive"

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
