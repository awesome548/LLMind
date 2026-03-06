CREATE TABLE IF NOT EXISTS raw_projects (
    id TEXT PRIMARY KEY,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_projects_metadata_gin ON raw_projects USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_raw_projects_metadata_url ON raw_projects ((metadata ->> 'url'));

CREATE OR REPLACE FUNCTION set_updated_at_raw_projects()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_set_updated_at_raw_projects ON raw_projects;

CREATE TRIGGER trg_set_updated_at_raw_projects
BEFORE UPDATE ON raw_projects
FOR EACH ROW
EXECUTE FUNCTION set_updated_at_raw_projects();