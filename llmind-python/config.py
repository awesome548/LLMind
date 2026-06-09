from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised configuration loaded from environment / .env file.

    All values can be overridden via environment variables (case-insensitive).
    Required secrets (supabase_url, supabase_key) default to "" so the app
    can start without them; validation happens at the call site where they are
    actually used.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_table: str = "media_docs"
    supabase_raw_table: str = "raw_projects"
    supabase_media_doc_table: str = "media_doc"
    supabase_emb_description_table: str = "media_emb_description"
    supabase_emb_details_table: str = "media_emb_details"
    supabase_emb_hybrid_table: str = "media_emb_hybrid"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_node_model: str = "gpt-5-mini-2025-08-07"

    # ── Related-project retrieval ─────────────────────────────────────────────
    supabase_match_function: str = "match_media_docs"
    supabase_match_count: int = 5
    supabase_similarity_threshold: float = 0.0

    # ── Prompt budget (bound node-generation prompts to fit small contexts) ────
    # Each node-generation call is stateless (a fresh system+user message, no
    # accumulation across calls). The taxonomy is serialized as a small FOCUSED
    # view (root + aspects + the focus aspect's options), so it stays bounded as
    # the tree grows. Per-project descriptions are the main variable cost — cap
    # them; lower for very small contexts. prompt_max_taxonomy_chars is a final
    # safety cap on the focused view.
    prompt_max_project_chars: int = 400   # per related-project description
    prompt_max_taxonomy_chars: int = 1800  # safety cap on the focused taxonomy

    # ── vLLM (local embeddings) ───────────────────────────────────────────────
    vllm_base_url: str = "http://100.73.44.12:8001/v1"
    vllm_model: str = "qwen"
    vllm_embed_model: str = "BAAI/bge-small-en-v1.5"

    # ── Vector store ──────────────────────────────────────────────────────────
    # "supabase" → pgvector via Supabase RPC; "local" → offline npz index
    # (built by build_local_index.py, searched with the local embedding model).
    vector_store: str = "supabase"
    local_index_path: Path = Path("data") / "local_index.npz"

    # ── Design-space projection ───────────────────────────────────────────────
    # Frozen reducer + background surface for the design-space visualisation.
    # Anchored to ``data/`` (like local_index_path) rather than data_dir so it
    # sits next to the index it is fit from and is independent of DATA_DIR.
    projection_dir: Path = Path("data") / "projection"

    # ── Scraper ───────────────────────────────────────────────────────────────
    base_url: str = "https://awards.mediaarchitecture.org"
    default_listing_url: str = "https://awards.mediaarchitecture.org/mab/projects/"

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    analysis_dir: Path = Path("analysis")
    plots_dir: Path = Path("plots")
    taxonomy_dir: Path = Path("taxonomy")

    # ── Pipeline tuning ───────────────────────────────────────────────────────
    # for Details field 
    min_words: int = 20
    max_words: int = 600
    default_hist_bins: int = 50
    default_embed_batch_size: int = 180
    default_fetch_batch_size: int = 1000
    max_examples: int = 5


settings = Settings()
