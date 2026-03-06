-- Central project data table (flat columns, no JSONB noise)
CREATE TABLE IF NOT EXISTS media_doc (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    description TEXT,           -- Descriptions field from scraper
    detail      TEXT,           -- Details field from scraper
    image       TEXT,           -- image_href from scraper
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION trg_fn_set_updated_at_media_doc()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_updated_at_media_doc ON media_doc;
CREATE TRIGGER trg_set_updated_at_media_doc
  BEFORE UPDATE ON media_doc
  FOR EACH ROW EXECUTE FUNCTION trg_fn_set_updated_at_media_doc();

-- Enable pgvector (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- Embedding table: description-only context
-- embedding_cloud: cloud model (e.g. text-embedding-3-small → 1536 dims)
-- embedding_local: local/vLLM model (e.g. BAAI/bge-small-en-v1.5 → 384 dims)
CREATE TABLE IF NOT EXISTS media_emb_description (
    media_doc_id     TEXT PRIMARY KEY REFERENCES media_doc(id) ON DELETE CASCADE,
    context          TEXT NOT NULL,
    embedding_cloud  VECTOR(1536),
    embedding_local  VECTOR(384)
);

-- Embedding table: detail-only context
CREATE TABLE IF NOT EXISTS media_emb_details (
    media_doc_id     TEXT PRIMARY KEY REFERENCES media_doc(id) ON DELETE CASCADE,
    context          TEXT NOT NULL,
    embedding_cloud  VECTOR(1536),
    embedding_local  VECTOR(384)
);

-- Embedding table: hybrid context (description + detail concatenated)
CREATE TABLE IF NOT EXISTS media_emb_hybrid (
    media_doc_id     TEXT PRIMARY KEY REFERENCES media_doc(id) ON DELETE CASCADE,
    context          TEXT NOT NULL,
    embedding_cloud  VECTOR(1536),
    embedding_local  VECTOR(384)
);

-- IVFFlat indexes for ANN cosine search (uncomment after loading enough data)
-- CREATE INDEX ON media_emb_description USING ivfflat (embedding_cloud vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_description USING ivfflat (embedding_local vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_details     USING ivfflat (embedding_cloud vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_details     USING ivfflat (embedding_local vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_hybrid      USING ivfflat (embedding_cloud vector_cosine_ops) WITH (lists = 100);
-- CREATE INDEX ON media_emb_hybrid      USING ivfflat (embedding_local vector_cosine_ops) WITH (lists = 100);
