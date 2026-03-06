"""Pipeline-wide constants derived from central settings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from config import settings
from utils.modes import BackendMode, ContentMode

# ── Filter thresholds ────────────────────────────────────────────────────────
MIN_WORDS = settings.min_words
MAX_WORDS = settings.max_words
DEFAULT_HIST_BINS = settings.default_hist_bins
DEFAULT_EMBED_BATCH_SIZE = settings.default_embed_batch_size
DEFAULT_FETCH_BATCH_SIZE = settings.default_fetch_batch_size
MAX_EXAMPLES = settings.max_examples

# ── Noise keys stripped during cleaning ─────────────────────────────────────
METADATA_NOISE_KEYS: frozenset[str] = frozenset({"url", "html_main"})

# ── Local output dirs ────────────────────────────────────────────────────────
DATA_DIR: Path = settings.data_dir
ANALYSIS_DIR: Path = settings.analysis_dir
PLOTS_DIR: Path = settings.plots_dir

OUTPUT_PATH = DATA_DIR / "cleaned_media_architecture.json"
ANALYSIS_PATH = ANALYSIS_DIR / "analysis_summary.json"
MIN_EXAMPLES_PATH = ANALYSIS_DIR / "min_detail_examples.json"

# ── Embedding / model config ─────────────────────────────────────────────────
EMBED_MODEL: str = settings.openai_embed_model
VLLM_BASE_URL: str = settings.vllm_base_url
VLLM_EMBED_MODEL: str = settings.vllm_embed_model

# ── Supabase tables ──────────────────────────────────────────────────────────
SUPABASE_TABLE: str = settings.supabase_table          # legacy (kept for compat)
SUPABASE_RAW_TABLE: str = settings.supabase_raw_table
SUPABASE_MEDIA_DOC_TABLE: str = settings.supabase_media_doc_table

# Embedding table per content mode
EMB_TABLE_MAP: Dict[ContentMode, str] = {
    ContentMode.description: settings.supabase_emb_description_table,
    ContentMode.details: settings.supabase_emb_details_table,
    ContentMode.hybrid: settings.supabase_emb_hybrid_table,
}

# Embedding column per backend (cloud vs local)
EMB_COLUMN_MAP: Dict[BackendMode, str] = {
    BackendMode.openai: "embedding_cloud",
    BackendMode.vllm: "embedding_local",
}

# Word-count filtering and analysis use the Details field
EMBED_CONTENT_FIELD: str = "Details"
