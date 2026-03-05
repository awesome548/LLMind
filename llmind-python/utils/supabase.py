# Supabase
from supabase import create_client, Client
from typing import Any, Dict, List, Optional, Protocol, Tuple, Literal
from pathlib import Path
import os
import json


# =============================
# Supabase helpers
# =============================

def _get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)

def _table_name() -> str:
    return os.getenv("SUPABASE_TABLE") or "media_docs"

def _load_selected_ids(path: Optional[Path]) -> List[str]:
    """Load list of ids from JSON file. Returns [] if not found/invalid/None."""
    if not path:
        return []
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
    except Exception:
        pass
    return []

def _fetch_rows_from_supabase_by_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch rows for given ids.
    Expect table schema: id (text), content (text), metadata (jsonb), embedding (vector)
    Returns mapping id -> dict(row)
    """
    if not ids:
        return {}
    sb = _get_supabase()
    tbl = _table_name()
    # Supabase PostgREST supports .in_ for filtering by a list
    res = sb.table(tbl).select("id, content, metadata").in_("id", ids).execute()
    if getattr(res, "error", None):
        raise RuntimeError(f"Supabase error: {res.error}")
    rows = res.data or []
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[str(r.get("id"))] = dict(r)
    return out

def _fetch_all_rows_from_supabase(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetch all rows from the table with simple paging.
    """
    sb = _get_supabase()
    tbl = _table_name()
    start = 0
    page = []
    all_rows: List[Dict[str, Any]] = []
    while True:
        # range is inclusive
        res = sb.table(tbl).select("id, content, metadata").range(start, start + limit - 1).execute()
        if getattr(res, "error", None):
            raise RuntimeError(f"Supabase error: {res.error}")
        page = res.data or []
        all_rows.extend(page)
        if len(page) < limit:
            break
        start += limit
    return all_rows


def _artefacts_from_rows(
    rows: List[Dict[str, Any]],
    *,
    mode: Literal["details_only", "both"] = "details_only",
    preserve_order_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert supabase rows -> artefacts list.
    - Each row: { id, content, metadata{ Name, Descriptions, Details, Image } }
    - mode 'details_only' uses Details or falls back to content
    - mode 'both' combines Descriptions + Details
    - If preserve_order_ids is provided, artefacts are sorted by that order and filtered to the listed ids.
    """
    # Build mapping for ordering if provided
    order_index: Dict[str, int] = {}
    if preserve_order_ids:
        order_index = {str(v): i for i, v in enumerate(preserve_order_ids)}

    artefacts: List[Dict[str, Any]] = []
    for r in rows:
        _id = str(r.get("id"))
        meta = r.get("metadata") or {}
        details = meta.get("Details")
        descriptions = meta.get("Descriptions")
        content = r.get("content")

        if mode == "both":
            combined = "; ".join([s for s in [descriptions, details] if s])
            text = combined or content
        else:
            text = details or content

        if not text:
            continue

        artefacts.append({
            "ID": _id,
            "Description": text,
        })

    # If we need to preserve order from ids file
    if preserve_order_ids:
        # keep only ids present in preserve_order_ids and sort
        artefacts = [a for a in artefacts if a["ID"] in order_index]
        artefacts.sort(key=lambda a: order_index.get(a["ID"], 10**9))

    return artefacts


def build_artefacts(
    *,
    source: Literal["selected", "all_supabase"] = "selected",
    ids_file: Optional[Path] = None,
    mode: Literal["details_only", "both"] = "details_only",
) -> List[Dict[str, Any]]:
    """
    Build artefacts from Supabase.

    - source = "selected": load ids from file, fetch matching rows from Supabase, preserve the ids order.
    - source = "all_supabase": fetch all rows from Supabase (ignores ids_file).

    Returns list of { "ID": <id>, "Description": <text> }.
    """
    if source == "selected":
        if not ids_file:
            raise ValueError("ids_file must be provided when source is 'selected'.")
        ids = _load_selected_ids(ids_file)
        if not ids:
            return []
        rows_map = _fetch_rows_from_supabase_by_ids(ids)
        # preserve order based on ids file
        rows = [rows_map[i] for i in ids if i in rows_map]
        return _artefacts_from_rows(rows, mode=mode, preserve_order_ids=ids)

    elif source == "all_supabase":
        rows = _fetch_all_rows_from_supabase()
        return _artefacts_from_rows(rows, mode=mode)

    else:
        raise ValueError("Invalid source. Use 'selected' or 'all_supabase'.")