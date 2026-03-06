-- Migration: media_doc + embedding tables (cloud / local split)
--
-- Safe to run against:
--   (a) a fresh database  → creates all tables
--   (b) a database with the old media_docs schema (single `embedding` column)
--       → adds the new columns, drops the legacy one
--
-- Idempotent: every statement uses IF EXISTS / IF NOT EXISTS guards.
-- ─────────────────────────────────────────────────────────────────────────────

-- 3. Column migrations (only needed when upgrading from the legacy schema) ─────
--    If the tables already existed with a single `embedding` column, this block:
--      • copies legacy data into embedding_cloud
--      • adds embedding_local
--      • drops the old embedding column
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'media_emb_description',
        'media_emb_details',
        'media_emb_hybrid'
    ]
    LOOP
        -- Add new columns if missing
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = tbl AND column_name = 'embedding_cloud'
        ) THEN
            EXECUTE format('ALTER TABLE %I ADD COLUMN embedding_cloud VECTOR(1536)', tbl);
        END IF;

        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = tbl AND column_name = 'embedding_local'
        ) THEN
            EXECUTE format('ALTER TABLE %I ADD COLUMN embedding_local VECTOR(384)', tbl);
        END IF;

        -- Copy legacy embedding → embedding_cloud, then drop it
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = tbl AND column_name = 'embedding'
        ) THEN
            EXECUTE format(
                'UPDATE %I SET embedding_cloud = embedding WHERE embedding_cloud IS NULL',
                tbl
            );
            EXECUTE format('ALTER TABLE %I DROP COLUMN embedding', tbl);
        END IF;
    END LOOP;
END;
$$;

-- 4. Indexes ───────────────────────────────────────────────────────────────────
--    Uncomment after the tables are populated (IVFFlat requires data to train).
-- CREATE INDEX ON media_emb_description USING ivfflat (embedding_cloud vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_description USING ivfflat (embedding_local vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_details     USING ivfflat (embedding_cloud vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_details     USING ivfflat (embedding_local vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_hybrid      USING ivfflat (embedding_cloud vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_hybrid      USING ivfflat (embedding_local vector_cosine_ops) WITH (lists = 100);