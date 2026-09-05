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
    display_text TEXT NOT NULL DEFAULT '',
    source_request_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(domain, cue_kind, model_id, text_hash)
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
