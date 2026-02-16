-- Idempotent migration: adds structured columns to temporal_facts
-- Run with: sqlite3 memory.sqlite < migrations/001_temporal_facts_extensions.sql

ALTER TABLE temporal_facts ADD COLUMN relation_type TEXT;
ALTER TABLE temporal_facts ADD COLUMN object_entity_id TEXT;
ALTER TABLE temporal_facts ADD COLUMN object_value TEXT;
ALTER TABLE temporal_facts ADD COLUMN confidence REAL DEFAULT 0.7;
ALTER TABLE temporal_facts ADD COLUMN is_active INTEGER DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_temporal_entity_rel_active ON temporal_facts(entity_id, relation_type, is_active);
CREATE INDEX IF NOT EXISTS idx_temporal_valid_from ON temporal_facts(valid_from);
