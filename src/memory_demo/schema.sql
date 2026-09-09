PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta(
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS paragraph(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    source_key TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    paragraph_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES source(id) ON DELETE CASCADE,
    UNIQUE(source_id, paragraph_index)
);

CREATE TABLE IF NOT EXISTS extraction_run(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,
    prompt_versions TEXT NOT NULL,
    log_path TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS extraction_task(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    source_id INTEGER,
    source_key TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(run_id) REFERENCES extraction_run(id),
    FOREIGN KEY(source_id) REFERENCES source(id)
);

CREATE TABLE IF NOT EXISTS episode(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    source_key TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    participants_json TEXT NOT NULL DEFAULT '[]',
    event_type TEXT NOT NULL DEFAULT '',
    location_text TEXT NOT NULL DEFAULT '',
    story_time_text TEXT NOT NULL DEFAULT '',
    story_order REAL,
    timeline_scope TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
    evidence_origin TEXT NOT NULL DEFAULT 'source'
        CHECK(evidence_origin IN ('source', 'importer', 'system', 'mixed', 'unknown')),
    epistemic_status TEXT NOT NULL DEFAULT 'asserted'
        CHECK(epistemic_status IN ('observed', 'asserted', 'reported', 'speculative', 'mixed', 'unknown')),
    generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
    epistemic_note TEXT NOT NULL DEFAULT '',
    evidence_quotes_json TEXT NOT NULL DEFAULT '[]',
    evidence_spans_json TEXT NOT NULL DEFAULT '[]',
    evidence_basis TEXT NOT NULL DEFAULT 'legacy_unavailable',
    embedding BLOB NOT NULL,
    extraction_run_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES source(id),
    FOREIGN KEY(extraction_run_id) REFERENCES extraction_run(id)
);

CREATE TABLE IF NOT EXISTS concept(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    description TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'merged')),
    canonical_concept_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(canonical_concept_id) REFERENCES concept(id)
);

CREATE TABLE IF NOT EXISTS concept_alias(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id INTEGER NOT NULL,
    alias TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'unknown',
    normalized_alias TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
    created_at TEXT NOT NULL,
    FOREIGN KEY(concept_id) REFERENCES concept(id),
    UNIQUE(concept_id, normalized_alias, language)
);

CREATE TABLE IF NOT EXISTS association(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_type TEXT NOT NULL CHECK(from_type IN ('episode', 'concept')),
    from_id INTEGER NOT NULL,
    to_type TEXT NOT NULL CHECK(to_type IN ('episode', 'concept')),
    to_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,
    relation_key TEXT NOT NULL,
    relation_text TEXT NOT NULL,
    cue_embedding BLOB,
    cue_embedding_text TEXT NOT NULL DEFAULT '',
    association_mode TEXT NOT NULL DEFAULT 'semantic',
    context_cue_id INTEGER,
    need_cue_id INTEGER,
    utility_weight REAL NOT NULL DEFAULT 0.0,
    utility_successes INTEGER NOT NULL DEFAULT 0,
    utility_noops INTEGER NOT NULL DEFAULT 0,
    utility_harms INTEGER NOT NULL DEFAULT 0,
    distinct_query_count INTEGER NOT NULL DEFAULT 0,
    lifecycle_state TEXT NOT NULL DEFAULT 'active',
    expires_at TEXT,
    last_evaluated_at TEXT,
    source_request_hash TEXT NOT NULL DEFAULT '',
    utility_query_hashes TEXT NOT NULL DEFAULT '[]',
    polarity INTEGER NOT NULL DEFAULT 1 CHECK(polarity IN (-1, 0, 1)),
    weight REAL NOT NULL DEFAULT 0.5 CHECK(weight >= 0 AND weight <= 1),
    confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
    generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
    evidence_count INTEGER NOT NULL DEFAULT 1,
    claim_level TEXT NOT NULL DEFAULT 'direct_fact'
        CHECK(claim_level IN ('direct_fact', 'supported_inference', 'historical_context', 'retrieval_only')),
    audit_status TEXT NOT NULL DEFAULT 'not_required'
        CHECK(audit_status IN ('not_required', 'dual_accepted')),
    evidence_json TEXT NOT NULL DEFAULT '[]',
    audit_json TEXT NOT NULL DEFAULT '[]',
    created_reason TEXT NOT NULL DEFAULT '',
    last_used TEXT,
    use_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(from_type, from_id, to_type, to_id, relation_type, relation_key, polarity)
);

CREATE INDEX IF NOT EXISTS idx_episode_source ON episode(source_id);
CREATE INDEX IF NOT EXISTS idx_episode_source_key ON episode(source_key, segment_index);
CREATE INDEX IF NOT EXISTS idx_episode_timeline ON episode(timeline_scope, story_order);
CREATE INDEX IF NOT EXISTS idx_paragraph_source ON paragraph(source_id, paragraph_index);
CREATE INDEX IF NOT EXISTS idx_paragraph_source_key ON paragraph(source_key, segment_index);
CREATE INDEX IF NOT EXISTS idx_concept_name ON concept(canonical_name);
CREATE INDEX IF NOT EXISTS idx_alias_normalized ON concept_alias(normalized_alias);
CREATE INDEX IF NOT EXISTS idx_association_from ON association(from_type, from_id);
CREATE INDEX IF NOT EXISTS idx_association_to ON association(to_type, to_id);
CREATE INDEX IF NOT EXISTS idx_association_relation ON association(relation_type);
CREATE INDEX IF NOT EXISTS idx_association_temporal_before
    ON association(from_id, to_id, relation_key)
    WHERE from_type = 'episode' AND to_type = 'episode'
      AND relation_type = 'temporal' AND polarity > 0;
CREATE INDEX IF NOT EXISTS idx_association_temporal_after
    ON association(to_id, from_id, relation_key)
    WHERE from_type = 'episode' AND to_type = 'episode'
      AND relation_type = 'temporal' AND polarity > 0;

CREATE TABLE IF NOT EXISTS association_cue_prototype(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,
    cue_kind TEXT NOT NULL CHECK(cue_kind IN ('context', 'need')),
    model_id TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK(dimension > 0),
    dtype TEXT NOT NULL DEFAULT 'float32' CHECK(dtype = 'float32'),
    vector_blob BLOB NOT NULL,
    text_hash TEXT NOT NULL,
    embedding_space_id TEXT NOT NULL DEFAULT '',
    display_text TEXT NOT NULL DEFAULT '',
    source_request_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(domain, cue_kind, model_id, text_hash)
);

-- Contextual recall creation is a two-phase local publication: an atomic
-- durable commit first, then a separately recorded RAM-index publication.
-- Receipt provenance contains only opaque source/vector/verification refs;
-- no Source text is copied into this table.
CREATE TABLE IF NOT EXISTS contextual_index_publication(
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    index_epoch INTEGER NOT NULL DEFAULT 0 CHECK(index_epoch >= 0),
    embedding_space_id TEXT NOT NULL DEFAULT '',
    context_cue_count INTEGER NOT NULL DEFAULT 0 CHECK(context_cue_count >= 0),
    need_cue_count INTEGER NOT NULL DEFAULT 0 CHECK(need_cue_count >= 0),
    published_at TEXT NOT NULL DEFAULT ''
);

INSERT OR IGNORE INTO contextual_index_publication(
    singleton, index_epoch, context_cue_count, need_cue_count, published_at
) VALUES(1, 0, 0, 0, '');

CREATE TABLE IF NOT EXISTS contextual_creation_receipt(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    creation_request_id TEXT NOT NULL UNIQUE,
    creation_request_hash TEXT NOT NULL,
    candidate_fingerprint TEXT NOT NULL,
    association_id INTEGER NOT NULL,
    context_cue_id INTEGER NOT NULL,
    need_cue_id INTEGER NOT NULL,
    domain TEXT NOT NULL,
    model_id TEXT NOT NULL,
    embedding_space_id TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK(dimension > 0),
    dtype TEXT NOT NULL CHECK(dtype = 'float32'),
    source_facts_json TEXT NOT NULL,
    verification_refs_json TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    anchor_provenance_json TEXT NOT NULL,
    target_provenance_json TEXT NOT NULL,
    source_request_hash TEXT NOT NULL DEFAULT '',
    context_vector_ref TEXT NOT NULL DEFAULT '',
    need_vector_ref TEXT NOT NULL DEFAULT '',
    anchor_vector_ref TEXT NOT NULL DEFAULT '',
    anchor_contribution_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN (
        'committed_pending_index',
        'ready',
        'legacy_pending_verification'
    )),
    ready_index_epoch INTEGER,
    durable_artifact_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    ready_at TEXT,
    FOREIGN KEY(association_id) REFERENCES association(id),
    FOREIGN KEY(context_cue_id) REFERENCES association_cue_prototype(id),
    FOREIGN KEY(need_cue_id) REFERENCES association_cue_prototype(id)
);

CREATE INDEX IF NOT EXISTS contextual_creation_receipt_edge_idx
    ON contextual_creation_receipt(association_id);
CREATE INDEX IF NOT EXISTS contextual_creation_receipt_status_idx
    ON contextual_creation_receipt(status, id);

DROP TRIGGER IF EXISTS contextual_creation_receipt_immutable;
CREATE TRIGGER contextual_creation_receipt_immutable
BEFORE UPDATE ON contextual_creation_receipt
WHEN
    NEW.creation_request_id <> OLD.creation_request_id
    OR NEW.creation_request_hash <> OLD.creation_request_hash
    OR NEW.candidate_fingerprint <> OLD.candidate_fingerprint
    OR NEW.association_id <> OLD.association_id
    OR NEW.context_cue_id <> OLD.context_cue_id
    OR NEW.need_cue_id <> OLD.need_cue_id
    OR NEW.domain <> OLD.domain
    OR NEW.model_id <> OLD.model_id
    OR NEW.embedding_space_id <> OLD.embedding_space_id
    OR NEW.dimension <> OLD.dimension
    OR NEW.dtype <> OLD.dtype
    OR NEW.source_facts_json <> OLD.source_facts_json
    OR NEW.verification_refs_json <> OLD.verification_refs_json
    OR NEW.verification_status <> OLD.verification_status
    OR NEW.anchor_provenance_json <> OLD.anchor_provenance_json
    OR NEW.target_provenance_json <> OLD.target_provenance_json
    OR NEW.source_request_hash <> OLD.source_request_hash
    OR NEW.context_vector_ref <> OLD.context_vector_ref
    OR NEW.need_vector_ref <> OLD.need_vector_ref
    OR NEW.anchor_vector_ref <> OLD.anchor_vector_ref
    OR NEW.anchor_contribution_id <> OLD.anchor_contribution_id
    OR NEW.durable_artifact_hash <> OLD.durable_artifact_hash
    OR NEW.created_at <> OLD.created_at
    OR (
        OLD.status = 'ready'
        AND (
            NEW.status <> 'ready'
            OR NEW.ready_index_epoch IS NOT OLD.ready_index_epoch
            OR NEW.ready_at IS NOT OLD.ready_at
        )
    )
    OR (
        OLD.status = 'legacy_pending_verification'
        AND NEW.status <> 'legacy_pending_verification'
    )
    OR (
        OLD.status = 'committed_pending_index'
        AND NEW.status NOT IN ('committed_pending_index', 'ready')
    )
    OR (
        OLD.status = 'committed_pending_index'
        AND NEW.status = 'committed_pending_index'
        AND (
            NEW.ready_index_epoch IS NOT OLD.ready_index_epoch
            OR NEW.ready_at IS NOT OLD.ready_at
        )
    )
    OR (
        NEW.status = 'ready'
        AND (NEW.ready_index_epoch IS NULL OR NEW.ready_index_epoch <= 0)
    )
BEGIN
    SELECT RAISE(ABORT, 'contextual creation receipt is immutable');
END;

-- V15 keeps post-creation utility as an immutable local ledger.  This is
-- deliberately separate from ``association``: a historical weight/counter is
-- not a source-backed observation, and cannot be reconstructed safely after
-- the old bounded query-hash cache has rolled over.  The ledger stores only
-- opaque IDs, content fingerprints, counts and provider receipt references.
CREATE TABLE IF NOT EXISTS contextual_utility_ledger(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT NOT NULL UNIQUE,
    family_id TEXT NOT NULL,
    association_id INTEGER NOT NULL,
    creation_receipt_id INTEGER NOT NULL,
    evaluation_as_of TEXT NOT NULL,
    candidate_universe_fingerprint TEXT NOT NULL,
    requirements_fingerprint TEXT NOT NULL,
    budget_fingerprint TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    treatment_fingerprint TEXT NOT NULL,
    masked_fingerprint TEXT NOT NULL,
    single_edge_fingerprint TEXT NOT NULL,
    leave_one_out_fingerprint TEXT NOT NULL,
    factual_support_verified INTEGER NOT NULL
        CHECK(factual_support_verified IN (0, 1)),
    treatment_episode_count INTEGER NOT NULL DEFAULT 0
        CHECK(treatment_episode_count >= 0),
    masked_episode_count INTEGER NOT NULL DEFAULT 0
        CHECK(masked_episode_count >= 0),
    treatment_required_count INTEGER NOT NULL DEFAULT 0
        CHECK(treatment_required_count >= 0),
    masked_required_count INTEGER NOT NULL DEFAULT 0
        CHECK(masked_required_count >= 0),
    treatment_gain_count INTEGER NOT NULL DEFAULT 0
        CHECK(treatment_gain_count >= 0),
    treatment_loss_count INTEGER NOT NULL DEFAULT 0
        CHECK(treatment_loss_count >= 0),
    single_edge_gain_count INTEGER NOT NULL DEFAULT 0
        CHECK(single_edge_gain_count >= 0),
    single_edge_loss_count INTEGER NOT NULL DEFAULT 0
        CHECK(single_edge_loss_count >= 0),
    leave_one_out_gain_count INTEGER NOT NULL DEFAULT 0
        CHECK(leave_one_out_gain_count >= 0),
    leave_one_out_loss_count INTEGER NOT NULL DEFAULT 0
        CHECK(leave_one_out_loss_count >= 0),
    recall_gain INTEGER NOT NULL DEFAULT 0 CHECK(recall_gain >= 0),
    work_metric TEXT NOT NULL DEFAULT '' CHECK(work_metric IN (
        '', 'provider_receipt_delta'
    )),
    treatment_work INTEGER,
    masked_work INTEGER,
    work_saved INTEGER,
    provider_receipt_refs_json TEXT NOT NULL DEFAULT '[]',
    harm INTEGER NOT NULL DEFAULT 0 CHECK(harm IN (0, 1)),
    is_shadow INTEGER NOT NULL DEFAULT 0 CHECK(is_shadow IN (0, 1)),
    outcome TEXT NOT NULL CHECK(outcome IN (
        'recall_gain', 'equal_quality_faster', 'no_op', 'harmful'
    )),
    created_at TEXT NOT NULL,
    FOREIGN KEY(association_id) REFERENCES association(id),
    FOREIGN KEY(creation_receipt_id) REFERENCES contextual_creation_receipt(id),
    UNIQUE(family_id, association_id),
    CHECK(
        (work_metric = '' AND treatment_work IS NULL
            AND masked_work IS NULL AND work_saved IS NULL)
        OR
        (work_metric = 'provider_receipt_delta'
            AND treatment_work IS NOT NULL AND masked_work IS NOT NULL
            AND work_saved IS NOT NULL
            AND treatment_work >= 0 AND masked_work >= 0
            AND work_saved >= 0
            AND work_saved = masked_work - treatment_work)
    ),
    CHECK((harm = 1 AND outcome = 'harmful')
          OR (harm = 0 AND outcome <> 'harmful')),
    CHECK(
        (harm = 1 AND (
            treatment_loss_count > 0 OR single_edge_loss_count > 0
            OR leave_one_out_loss_count > 0
        ))
        OR
        (harm = 0 AND treatment_loss_count = 0
            AND single_edge_loss_count = 0 AND leave_one_out_loss_count = 0)
    ),
    CHECK(recall_gain = 0 OR (harm = 0 AND outcome = 'recall_gain')),
    CHECK(
        factual_support_verified = 1
        OR (
            recall_gain = 0
            AND treatment_gain_count = 0
            AND single_edge_gain_count = 0
            AND leave_one_out_gain_count = 0
        )
    ),
    CHECK(recall_gain <= MAX(
        treatment_gain_count, single_edge_gain_count, leave_one_out_gain_count
    )),
    CHECK(outcome <> 'recall_gain' OR recall_gain > 0),
    CHECK(outcome <> 'equal_quality_faster' OR work_saved > 0)
);

CREATE INDEX IF NOT EXISTS contextual_utility_ledger_edge_idx
    ON contextual_utility_ledger(association_id, is_shadow, harm, recall_gain);
CREATE INDEX IF NOT EXISTS contextual_utility_ledger_receipt_idx
    ON contextual_utility_ledger(creation_receipt_id, id);

-- A utility observation may only use the stable, earliest ready strict
-- creation receipt for its edge.  This prevents a replay from relabelling a
-- later/reused receipt as a new source of evidence.
DROP TRIGGER IF EXISTS contextual_utility_ledger_source_guard;
CREATE TRIGGER contextual_utility_ledger_source_guard
BEFORE INSERT ON contextual_utility_ledger
WHEN NOT EXISTS (
    SELECT 1
    FROM contextual_creation_receipt AS receipt
    JOIN association AS edge ON edge.id = receipt.association_id
    WHERE receipt.id = NEW.creation_receipt_id
      AND receipt.association_id = NEW.association_id
      AND edge.association_mode = 'contextual_recall'
      AND receipt.status = 'ready'
      AND receipt.verification_status IN ('verified', 'source_bound')
      AND receipt.id = (
          SELECT MIN(candidate_receipt.id)
          FROM contextual_creation_receipt AS candidate_receipt
          WHERE candidate_receipt.association_id = NEW.association_id
            AND candidate_receipt.status = 'ready'
            AND candidate_receipt.verification_status IN ('verified', 'source_bound')
      )
)
BEGIN
    SELECT RAISE(ABORT, 'utility ledger needs canonical ready source-bound creation receipt');
END;

-- A conflict-mode REPLACE would otherwise delete an append-only observation
-- before inserting a new row.  Reject all identity/family collisions before
-- SQLite can choose its conflict strategy.
DROP TRIGGER IF EXISTS contextual_utility_ledger_no_replace;
CREATE TRIGGER contextual_utility_ledger_no_replace
BEFORE INSERT ON contextual_utility_ledger
WHEN EXISTS (
    SELECT 1
    FROM contextual_utility_ledger AS existing
    WHERE existing.id = NEW.id
       OR existing.observation_id = NEW.observation_id
       OR (
           existing.family_id = NEW.family_id
           AND existing.association_id = NEW.association_id
       )
)
BEGIN
    SELECT RAISE(ABORT, 'contextual utility ledger rows cannot be replaced');
END;

DROP TRIGGER IF EXISTS contextual_utility_ledger_factual_guard;
CREATE TRIGGER contextual_utility_ledger_factual_guard
BEFORE INSERT ON contextual_utility_ledger
WHEN NEW.factual_support_verified = 0
 AND (
     NEW.recall_gain <> 0
     OR NEW.treatment_gain_count <> 0
     OR NEW.single_edge_gain_count <> 0
     OR NEW.leave_one_out_gain_count <> 0
 )
BEGIN
    SELECT RAISE(ABORT, 'relevance-only utility cannot claim required coverage');
END;

DROP TRIGGER IF EXISTS contextual_utility_ledger_no_update;
CREATE TRIGGER contextual_utility_ledger_no_update
BEFORE UPDATE ON contextual_utility_ledger
BEGIN
    SELECT RAISE(ABORT, 'contextual utility ledger is append-only');
END;

DROP TRIGGER IF EXISTS contextual_utility_ledger_no_delete;
CREATE TRIGGER contextual_utility_ledger_no_delete
BEFORE DELETE ON contextual_utility_ledger
BEGIN
    SELECT RAISE(ABORT, 'contextual utility ledger is append-only');
END;


-- V16 stores the redacted, immutable input contract needed by a future
-- contextual revisit.  It intentionally has no query/source/answer prose or
-- vector column.  Runtime initialization adds stricter opaque-value guards
-- supplied by the local Database connection.
CREATE TABLE IF NOT EXISTS contextual_revisit_contract(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    creation_receipt_id INTEGER NOT NULL UNIQUE,
    association_id INTEGER NOT NULL,
    context_cue_id INTEGER NOT NULL,
    need_cue_id INTEGER NOT NULL,
    domain TEXT NOT NULL CHECK(length(domain) BETWEEN 1 AND 256),
    model_id TEXT NOT NULL CHECK(length(model_id) BETWEEN 1 AND 256),
    embedding_space_id TEXT NOT NULL CHECK(length(embedding_space_id) BETWEEN 1 AND 256),
    dimension INTEGER NOT NULL CHECK(dimension > 0),
    dtype TEXT NOT NULL CHECK(dtype = 'float32'),
    context_hash TEXT NOT NULL CHECK(length(context_hash) BETWEEN 72 AND 512),
    slot_need_bindings_json TEXT NOT NULL CHECK(length(slot_need_bindings_json) BETWEEN 2 AND 65536),
    requirements_fingerprint TEXT NOT NULL CHECK(length(requirements_fingerprint) BETWEEN 72 AND 512),
    source_closure_fingerprint TEXT NOT NULL CHECK(length(source_closure_fingerprint) BETWEEN 72 AND 512),
    retrieval_policy_fingerprint TEXT NOT NULL CHECK(length(retrieval_policy_fingerprint) BETWEEN 72 AND 512),
    budget_fingerprint TEXT NOT NULL CHECK(length(budget_fingerprint) BETWEEN 72 AND 512),
    anchor_manifest_fingerprint TEXT NOT NULL CHECK(length(anchor_manifest_fingerprint) BETWEEN 72 AND 512),
    source_fact_roles_fingerprint TEXT NOT NULL CHECK(length(source_fact_roles_fingerprint) BETWEEN 72 AND 512),
    source_fact_refs_fingerprint TEXT NOT NULL CHECK(length(source_fact_refs_fingerprint) BETWEEN 72 AND 512),
    ready_index_epoch INTEGER NOT NULL CHECK(ready_index_epoch > 0),
    ready_publication_fingerprint TEXT NOT NULL CHECK(length(ready_publication_fingerprint) BETWEEN 72 AND 512),
    contract_version TEXT NOT NULL CHECK(length(contract_version) BETWEEN 1 AND 256),
    contract_fingerprint TEXT NOT NULL UNIQUE CHECK(length(contract_fingerprint) BETWEEN 72 AND 512),
    created_at TEXT NOT NULL,
    FOREIGN KEY(creation_receipt_id) REFERENCES contextual_creation_receipt(id),
    FOREIGN KEY(association_id) REFERENCES association(id),
    FOREIGN KEY(context_cue_id) REFERENCES association_cue_prototype(id),
    FOREIGN KEY(need_cue_id) REFERENCES association_cue_prototype(id)
);

CREATE INDEX IF NOT EXISTS contextual_revisit_contract_edge_idx
    ON contextual_revisit_contract(association_id, id);

DROP TRIGGER IF EXISTS contextual_revisit_contract_no_replace;
CREATE TRIGGER contextual_revisit_contract_no_replace
BEFORE INSERT ON contextual_revisit_contract
WHEN EXISTS (
    SELECT 1
    FROM contextual_revisit_contract AS existing
    WHERE existing.id = NEW.id
       OR existing.creation_receipt_id = NEW.creation_receipt_id
       OR existing.contract_fingerprint = NEW.contract_fingerprint
)
BEGIN
    SELECT RAISE(ABORT, 'contextual revisit contracts cannot be replaced');
END;

DROP TRIGGER IF EXISTS contextual_revisit_contract_source_guard;
CREATE TRIGGER contextual_revisit_contract_source_guard
BEFORE INSERT ON contextual_revisit_contract
WHEN NOT EXISTS (
    SELECT 1
    FROM contextual_creation_receipt AS receipt
    JOIN association AS edge ON edge.id = receipt.association_id
    JOIN association_cue_prototype AS context_cue
        ON context_cue.id = receipt.context_cue_id
    JOIN association_cue_prototype AS need_cue
        ON need_cue.id = receipt.need_cue_id
    JOIN contextual_index_publication AS publication ON publication.singleton = 1
    WHERE receipt.id = NEW.creation_receipt_id
      AND receipt.association_id = NEW.association_id
      AND receipt.context_cue_id = NEW.context_cue_id
      AND receipt.need_cue_id = NEW.need_cue_id
      AND receipt.domain = NEW.domain
      AND receipt.model_id = NEW.model_id
      AND receipt.embedding_space_id = NEW.embedding_space_id
      AND receipt.dimension = NEW.dimension
      AND receipt.dtype = NEW.dtype
      AND receipt.ready_index_epoch = NEW.ready_index_epoch
      AND edge.association_mode = 'contextual_recall'
      AND edge.context_cue_id = receipt.context_cue_id
      AND edge.need_cue_id = receipt.need_cue_id
      AND context_cue.cue_kind = 'context'
      AND need_cue.cue_kind = 'need'
      AND context_cue.domain = receipt.domain
      AND need_cue.domain = receipt.domain
      AND context_cue.model_id = receipt.model_id
      AND need_cue.model_id = receipt.model_id
      AND context_cue.embedding_space_id = receipt.embedding_space_id
      AND need_cue.embedding_space_id = receipt.embedding_space_id
      AND context_cue.dimension = receipt.dimension
      AND need_cue.dimension = receipt.dimension
      AND context_cue.dtype = receipt.dtype
      AND need_cue.dtype = receipt.dtype
      AND receipt.status = 'ready'
      AND receipt.verification_status IN ('verified', 'source_bound')
      AND publication.embedding_space_id = receipt.embedding_space_id
      AND publication.index_epoch >= receipt.ready_index_epoch
      AND receipt.id = COALESCE(
          (
              SELECT MIN(existing_contract.creation_receipt_id)
              FROM contextual_revisit_contract AS existing_contract
              WHERE existing_contract.association_id = NEW.association_id
          ),
          (
              SELECT MIN(existing_ledger.creation_receipt_id)
              FROM contextual_utility_ledger AS existing_ledger
              WHERE existing_ledger.association_id = NEW.association_id
          ),
          (
              SELECT MIN(candidate_receipt.id)
              FROM contextual_creation_receipt AS candidate_receipt
              WHERE candidate_receipt.association_id = NEW.association_id
                AND candidate_receipt.status = 'ready'
                AND candidate_receipt.verification_status IN ('verified', 'source_bound')
          )
      )
)
BEGIN
    SELECT RAISE(ABORT, 'revisit contract needs canonical ready source-bound creation receipt');
END;

DROP TRIGGER IF EXISTS contextual_revisit_contract_no_update;
CREATE TRIGGER contextual_revisit_contract_no_update
BEFORE UPDATE ON contextual_revisit_contract
BEGIN
    SELECT RAISE(ABORT, 'contextual revisit contract is immutable');
END;

DROP TRIGGER IF EXISTS contextual_revisit_contract_no_delete;
CREATE TRIGGER contextual_revisit_contract_no_delete
BEFORE DELETE ON contextual_revisit_contract
BEGIN
    SELECT RAISE(ABORT, 'contextual revisit contract is immutable');
END;

-- The first durable contract/ledger consumer freezes a ready origin.  An
-- older receipt becoming ready later cannot invalidate an exact retry.
DROP TRIGGER IF EXISTS contextual_utility_ledger_source_guard;
CREATE TRIGGER contextual_utility_ledger_source_guard
BEFORE INSERT ON contextual_utility_ledger
WHEN NOT EXISTS (
    SELECT 1
    FROM contextual_creation_receipt AS receipt
    JOIN association AS edge ON edge.id = receipt.association_id
    WHERE receipt.id = NEW.creation_receipt_id
      AND receipt.association_id = NEW.association_id
      AND edge.association_mode = 'contextual_recall'
      AND receipt.status = 'ready'
      AND receipt.verification_status IN ('verified', 'source_bound')
      AND receipt.id = COALESCE(
          (
              SELECT MIN(existing_contract.creation_receipt_id)
              FROM contextual_revisit_contract AS existing_contract
              WHERE existing_contract.association_id = NEW.association_id
          ),
          (
              SELECT MIN(existing_ledger.creation_receipt_id)
              FROM contextual_utility_ledger AS existing_ledger
              WHERE existing_ledger.association_id = NEW.association_id
          ),
          (
              SELECT MIN(candidate_receipt.id)
              FROM contextual_creation_receipt AS candidate_receipt
              WHERE candidate_receipt.association_id = NEW.association_id
                AND candidate_receipt.status = 'ready'
                AND candidate_receipt.verification_status IN ('verified', 'source_bound')
          )
      )
)
BEGIN
    SELECT RAISE(ABORT, 'utility ledger needs canonical ready source-bound creation receipt');
END;

-- V17 adds a recovery seed for the public, scope-bound exact Q2 lane.  This
-- table is intentionally a fixed manifest rather than a query/answer cache:
-- it has no question/source/answer prose column and no vector blob.  Vectors
-- remain solely in association_cue_prototype.  Database.initialize() installs
-- stricter digest/JSON/source guards for ordinary application connections.
CREATE TABLE IF NOT EXISTS contextual_revisit_runtime_manifest(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    creation_receipt_id INTEGER NOT NULL UNIQUE,
    association_id INTEGER NOT NULL,
    context_cue_id INTEGER NOT NULL,
    need_cue_id INTEGER NOT NULL,
    domain TEXT NOT NULL CHECK(length(domain) BETWEEN 1 AND 256),
    model_id TEXT NOT NULL CHECK(length(model_id) BETWEEN 1 AND 256),
    embedding_space_id TEXT NOT NULL CHECK(length(embedding_space_id) BETWEEN 1 AND 256),
    dimension INTEGER NOT NULL CHECK(dimension > 0),
    dtype TEXT NOT NULL CHECK(dtype = 'float32'),
    context_scope_hash TEXT NOT NULL CHECK(length(context_scope_hash) BETWEEN 72 AND 512),
    context_hash TEXT NOT NULL CHECK(length(context_hash) BETWEEN 72 AND 512),
    source_request_hash TEXT NOT NULL CHECK(length(source_request_hash) BETWEEN 71 AND 512),
    context_cue_text_hash TEXT NOT NULL CHECK(length(context_cue_text_hash) = 64),
    need_cue_text_hash TEXT NOT NULL CHECK(length(need_cue_text_hash) = 64),
    context_cue_vector_fingerprint TEXT NOT NULL CHECK(length(context_cue_vector_fingerprint) BETWEEN 72 AND 512),
    need_cue_vector_fingerprint TEXT NOT NULL CHECK(length(need_cue_vector_fingerprint) BETWEEN 72 AND 512),
    slot_need_bindings_json TEXT NOT NULL CHECK(length(slot_need_bindings_json) BETWEEN 2 AND 65536),
    requirements_fingerprint TEXT NOT NULL CHECK(length(requirements_fingerprint) BETWEEN 72 AND 512),
    source_closure_fingerprint TEXT NOT NULL CHECK(length(source_closure_fingerprint) BETWEEN 72 AND 512),
    retrieval_policy_fingerprint TEXT NOT NULL CHECK(length(retrieval_policy_fingerprint) BETWEEN 72 AND 512),
    budget_fingerprint TEXT NOT NULL CHECK(length(budget_fingerprint) BETWEEN 72 AND 512),
    anchor_manifest_fingerprint TEXT NOT NULL CHECK(length(anchor_manifest_fingerprint) BETWEEN 72 AND 512),
    source_fact_refs_fingerprint TEXT NOT NULL CHECK(length(source_fact_refs_fingerprint) BETWEEN 72 AND 512),
    anchor_episode_id INTEGER NOT NULL CHECK(anchor_episode_id > 0),
    anchor_activation REAL NOT NULL CHECK(anchor_activation > 0.0),
    anchor_source_fact_id TEXT NOT NULL CHECK(length(anchor_source_fact_id) BETWEEN 72 AND 512),
    target_episode_id INTEGER NOT NULL CHECK(target_episode_id > 0),
    target_source_fact_id TEXT NOT NULL CHECK(length(target_source_fact_id) BETWEEN 72 AND 512),
    target_mapping_ref TEXT NOT NULL CHECK(length(target_mapping_ref) BETWEEN 72 AND 512),
    runtime_slot_ref TEXT NOT NULL CHECK(length(runtime_slot_ref) BETWEEN 72 AND 512),
    runtime_query_ref TEXT NOT NULL CHECK(length(runtime_query_ref) BETWEEN 72 AND 512),
    runtime_clause_ref TEXT NOT NULL CHECK(length(runtime_clause_ref) BETWEEN 72 AND 512),
    endpoint_limit INTEGER NOT NULL CHECK(endpoint_limit > 0),
    episode_limit INTEGER NOT NULL CHECK(episode_limit > 0),
    source_fact_limit INTEGER CHECK(source_fact_limit IS NULL OR source_fact_limit >= 0),
    delivery_token_limit INTEGER CHECK(delivery_token_limit IS NULL OR delivery_token_limit >= 0),
    support_mode TEXT NOT NULL CHECK(support_mode = 'alternative'),
    contract_version TEXT NOT NULL CHECK(length(contract_version) BETWEEN 1 AND 256),
    manifest_version TEXT NOT NULL CHECK(length(manifest_version) BETWEEN 1 AND 256),
    source_fact_roles_fingerprint TEXT NOT NULL CHECK(length(source_fact_roles_fingerprint) BETWEEN 72 AND 512),
    not_before_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    seed_fingerprint TEXT NOT NULL UNIQUE CHECK(length(seed_fingerprint) BETWEEN 72 AND 512),
    binding_fingerprint TEXT NOT NULL UNIQUE CHECK(length(binding_fingerprint) BETWEEN 72 AND 512),
    state TEXT NOT NULL CHECK(state IN ('pending_index', 'ready')),
    ready_index_epoch INTEGER,
    ready_at TEXT,
    ready_publication_fingerprint TEXT,
    contract_id INTEGER UNIQUE,
    contract_fingerprint TEXT UNIQUE,
    manifest_fingerprint TEXT UNIQUE,
    created_at TEXT NOT NULL,
    FOREIGN KEY(creation_receipt_id) REFERENCES contextual_creation_receipt(id),
    FOREIGN KEY(association_id) REFERENCES association(id),
    FOREIGN KEY(context_cue_id) REFERENCES association_cue_prototype(id),
    FOREIGN KEY(need_cue_id) REFERENCES association_cue_prototype(id),
    FOREIGN KEY(contract_id) REFERENCES contextual_revisit_contract(id),
    CHECK(anchor_episode_id <> target_episode_id),
    CHECK(expires_at > not_before_at),
    CHECK(
        (state = 'pending_index'
            AND ready_index_epoch IS NULL AND ready_at IS NULL
            AND ready_publication_fingerprint IS NULL
            AND contract_id IS NULL AND contract_fingerprint IS NULL
            AND manifest_fingerprint IS NULL)
        OR
        (state = 'ready'
            AND ready_index_epoch IS NOT NULL AND ready_index_epoch > 0
            AND ready_at IS NOT NULL
            AND ready_publication_fingerprint IS NOT NULL
            AND contract_id IS NOT NULL AND contract_fingerprint IS NOT NULL
            AND manifest_fingerprint IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS contextual_revisit_runtime_manifest_lookup_idx
    ON contextual_revisit_runtime_manifest(
        state, domain, context_scope_hash, context_hash,
        source_request_hash, model_id, embedding_space_id, dimension, dtype
    );
CREATE INDEX IF NOT EXISTS contextual_revisit_runtime_manifest_edge_idx
    ON contextual_revisit_runtime_manifest(association_id, id);

DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_no_replace;
CREATE TRIGGER contextual_revisit_runtime_manifest_no_replace
BEFORE INSERT ON contextual_revisit_runtime_manifest
WHEN EXISTS (
    SELECT 1 FROM contextual_revisit_runtime_manifest AS existing
    WHERE existing.id = NEW.id
       OR existing.creation_receipt_id = NEW.creation_receipt_id
       OR existing.seed_fingerprint = NEW.seed_fingerprint
       OR existing.binding_fingerprint = NEW.binding_fingerprint
)
BEGIN
    SELECT RAISE(ABORT, 'contextual runtime manifests cannot be replaced');
END;

DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_ready_transition;
CREATE TRIGGER contextual_revisit_runtime_manifest_ready_transition
BEFORE UPDATE ON contextual_revisit_runtime_manifest
WHEN OLD.state <> 'pending_index'
  OR NEW.state <> 'ready'
  OR NEW.creation_receipt_id <> OLD.creation_receipt_id
  OR NEW.association_id <> OLD.association_id
  OR NEW.context_cue_id <> OLD.context_cue_id
  OR NEW.need_cue_id <> OLD.need_cue_id
  OR NEW.seed_fingerprint <> OLD.seed_fingerprint
  OR NEW.binding_fingerprint <> OLD.binding_fingerprint
BEGIN
    SELECT RAISE(ABORT, 'contextual runtime manifest permits only one ready transition');
END;

DROP TRIGGER IF EXISTS contextual_revisit_runtime_manifest_no_delete;
CREATE TRIGGER contextual_revisit_runtime_manifest_no_delete
BEFORE DELETE ON contextual_revisit_runtime_manifest
BEGIN
    SELECT RAISE(ABORT, 'contextual runtime manifest is immutable');
END;

-- V22 adds a narrowly scoped, three-stage HMAC authorization for controlled query
-- rewrites.  It stores neither query text, entity/number/date values, source
-- text, answer text nor raw vectors.  Opaque fingerprints bind the guard to
-- the exact cue vectors and embedding-space metadata used by its manifest;
-- the second commitment additionally covers the canonical pending lifecycle
-- binding (receipt, source roles, validity window, and cue identities).
-- The final commitment covers the ready manifest fingerprint (including its
-- publication checkpoint and canonical contract), so a raw database writer
-- cannot replace those public hashes after Q1.
-- The real UDF-backed guards are installed by
-- Database._ensure_v18_schema()/_ensure_v19_schema()/_ensure_v20_schema()
-- /_ensure_v21_schema()/_ensure_v22_schema()
-- alongside V17; this readable block stays in the skipped bootstrap region so
-- a partial schema.sql replay cannot weaken its pending-manifest-only
-- insertion boundary.
CREATE TABLE IF NOT EXISTS contextual_restricted_rewrite_guard(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    creation_receipt_id INTEGER NOT NULL UNIQUE,
    association_id INTEGER NOT NULL,
    domain TEXT NOT NULL CHECK(length(domain) BETWEEN 1 AND 256),
    context_scope_hash TEXT NOT NULL CHECK(length(context_scope_hash) BETWEEN 72 AND 512),
    rewrite_commitment TEXT NOT NULL CHECK(length(rewrite_commitment) BETWEEN 72 AND 512),
    binding_commitment TEXT NOT NULL CHECK(length(binding_commitment) BETWEEN 72 AND 512),
    manifest_binding_commitment TEXT NOT NULL DEFAULT '' CHECK(length(manifest_binding_commitment) <= 512),
    ready_manifest_commitment TEXT NOT NULL DEFAULT '' CHECK(length(ready_manifest_commitment) <= 512),
    context_cue_vector_fingerprint TEXT NOT NULL DEFAULT '' CHECK(length(context_cue_vector_fingerprint) <= 512),
    need_cue_vector_fingerprint TEXT NOT NULL DEFAULT '' CHECK(length(need_cue_vector_fingerprint) <= 512),
    model_id TEXT NOT NULL DEFAULT '' CHECK(length(model_id) <= 256),
    embedding_space_id TEXT NOT NULL DEFAULT '' CHECK(length(embedding_space_id) <= 256),
    dimension INTEGER NOT NULL DEFAULT 0 CHECK(dimension >= 0),
    dtype TEXT NOT NULL DEFAULT '' CHECK(length(dtype) <= 32),
    commitment_key_id TEXT NOT NULL CHECK(length(commitment_key_id) BETWEEN 1 AND 256),
    grammar_version TEXT NOT NULL CHECK(length(grammar_version) BETWEEN 1 AND 256),
    seed_fingerprint TEXT NOT NULL CHECK(length(seed_fingerprint) BETWEEN 72 AND 512),
    signature_version TEXT NOT NULL CHECK(length(signature_version) BETWEEN 1 AND 256),
    created_at TEXT NOT NULL,
    guard_fingerprint TEXT NOT NULL UNIQUE CHECK(length(guard_fingerprint) BETWEEN 72 AND 512),
    FOREIGN KEY(creation_receipt_id) REFERENCES contextual_creation_receipt(id),
    FOREIGN KEY(association_id) REFERENCES association(id)
);

CREATE INDEX IF NOT EXISTS contextual_restricted_rewrite_guard_lookup_idx
    ON contextual_restricted_rewrite_guard(
        association_id, domain, context_scope_hash, rewrite_commitment,
        commitment_key_id, grammar_version, signature_version
    );
CREATE INDEX IF NOT EXISTS contextual_restricted_rewrite_guard_root_lookup_idx
    ON contextual_restricted_rewrite_guard(
        domain, context_scope_hash, rewrite_commitment,
        commitment_key_id, grammar_version, signature_version, id
    );


-- FTS is a retrieval index, not a new memory node.  The trigram tokenizer is
-- language-agnostic and supports Chinese/Japanese/Korean substrings as well as
-- Latin names.  External-content tables keep Episode and Source authoritative.
CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(
    text,
    source_key,
    content='episode',
    content_rowid='id',
    tokenize='trigram'
);

CREATE VIRTUAL TABLE IF NOT EXISTS source_fts USING fts5(
    raw_text,
    content='source',
    content_rowid='id',
    tokenize='trigram'
);

CREATE VIRTUAL TABLE IF NOT EXISTS episode_bigram_fts USING fts5(tokens);
CREATE VIRTUAL TABLE IF NOT EXISTS source_bigram_fts USING fts5(tokens);

CREATE TRIGGER IF NOT EXISTS episode_fts_insert AFTER INSERT ON episode BEGIN
    INSERT INTO episode_fts(rowid, text, source_key)
    VALUES (new.id, new.text, new.source_key);
END;

CREATE TRIGGER IF NOT EXISTS episode_bigram_fts_insert AFTER INSERT ON episode BEGIN
    INSERT INTO episode_bigram_fts(rowid, tokens)
    VALUES (new.id, memory_bigram_tokens(new.text || ' ' || new.source_key));
END;

CREATE TRIGGER IF NOT EXISTS episode_fts_delete AFTER DELETE ON episode BEGIN
    INSERT INTO episode_fts(episode_fts, rowid, text, source_key)
    VALUES ('delete', old.id, old.text, old.source_key);
END;

CREATE TRIGGER IF NOT EXISTS episode_bigram_fts_delete AFTER DELETE ON episode BEGIN
    DELETE FROM episode_bigram_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS episode_fts_update AFTER UPDATE OF text, source_key ON episode BEGIN
    INSERT INTO episode_fts(episode_fts, rowid, text, source_key)
    VALUES ('delete', old.id, old.text, old.source_key);
    INSERT INTO episode_fts(rowid, text, source_key)
    VALUES (new.id, new.text, new.source_key);
END;

CREATE TRIGGER IF NOT EXISTS episode_bigram_fts_update AFTER UPDATE OF text, source_key ON episode BEGIN
    DELETE FROM episode_bigram_fts WHERE rowid = old.id;
    INSERT INTO episode_bigram_fts(rowid, tokens)
    VALUES (new.id, memory_bigram_tokens(new.text || ' ' || new.source_key));
END;

CREATE TRIGGER IF NOT EXISTS source_fts_insert AFTER INSERT ON source BEGIN
    INSERT INTO source_fts(rowid, raw_text) VALUES (new.id, new.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS source_bigram_fts_insert AFTER INSERT ON source BEGIN
    INSERT INTO source_bigram_fts(rowid, tokens)
    VALUES (new.id, memory_bigram_tokens(new.raw_text));
END;

CREATE TRIGGER IF NOT EXISTS source_fts_delete AFTER DELETE ON source BEGIN
    INSERT INTO source_fts(source_fts, rowid, raw_text)
    VALUES ('delete', old.id, old.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS source_bigram_fts_delete AFTER DELETE ON source BEGIN
    DELETE FROM source_bigram_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS source_fts_update AFTER UPDATE OF raw_text ON source BEGIN
    INSERT INTO source_fts(source_fts, rowid, raw_text)
    VALUES ('delete', old.id, old.raw_text);
    INSERT INTO source_fts(rowid, raw_text) VALUES (new.id, new.raw_text);
END;

CREATE TRIGGER IF NOT EXISTS source_bigram_fts_update AFTER UPDATE OF raw_text ON source BEGIN
    DELETE FROM source_bigram_fts WHERE rowid = old.id;
    INSERT INTO source_bigram_fts(rowid, tokens)
    VALUES (new.id, memory_bigram_tokens(new.raw_text));
END;
