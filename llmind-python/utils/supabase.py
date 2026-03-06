"""All Supabase operations: client, upsert, fetch, and artefact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

import typer
from supabase import Client, create_client

from config import settings
from utils.modes import BackendMode, ContentMode


# ── Client ────────────────────────────────────────────────────────────────────

def get_supabase_client() -> Client:
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(settings.supabase_url, settings.supabase_key)


# ── Error helpers ─────────────────────────────────────────────────────────────

def _format_supabase_error(error: Any) -> str:
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        return error.get("message", str(error))
    return str(getattr(error, "message", error))


def _is_missing_table_error(error: Any) -> bool:
    code = str(getattr(error, "code", "")).lower()
    message = _format_supabase_error(error).lower()
    return code == "pgrst116" or "does not exist" in message or "not found" in message


def _raise_on_error(response: Any, table: str, op: str) -> None:
    error = getattr(response, "error", None)
    if not error:
        return
    if _is_missing_table_error(error):
        raise RuntimeError(f"Supabase table '{table}' does not exist or is inaccessible.")
    raise RuntimeError(f"Supabase {op} failed for table '{table}': {_format_supabase_error(error)}")


def assert_table_exists(table: str, *, client: Optional[Client] = None) -> None:
    """Check table is accessible. Use for health checks / startup validation only."""
    sb = client or get_supabase_client()
    try:
        response = sb.table(table).select("count", count="exact").limit(0).execute()
    except Exception as exc:
        raise RuntimeError(f"Unable to verify Supabase table '{table}'.") from exc
    error = getattr(response, "error", None)
    if error:
        msg = _format_supabase_error(error)
        if _is_missing_table_error(error):
            raise RuntimeError(f"Supabase table '{table}' does not exist or is inaccessible.")
        raise RuntimeError(f"Supabase error while validating table '{table}': {msg}")


# ── Generic upsert ────────────────────────────────────────────────────────────

def upsert_rows(
    table: str,
    rows: List[Dict[str, Any]],
    *,
    on_conflict: str = "id",
    client: Optional[Client] = None,
) -> Any:
    """Upsert rows into table, updating on conflict. Duplicates are never silently ignored."""
    sb = client or get_supabase_client()
    response = sb.table(table).upsert(rows, on_conflict=on_conflict, ignore_duplicates=False).execute()
    _raise_on_error(response, table, "upsert")
    return response


# ── ID helper ─────────────────────────────────────────────────────────────────

def _project_id_from_url(url: str) -> str:
    if "project/" in url:
        return url.split("project/")[-1].rstrip("/")
    return str(uuid4())


# ── Upsert operations ─────────────────────────────────────────────────────────

def upsert_raw_to_supabase(records: list) -> None:
    """Upsert scraped ProjectRecord list into raw_projects table."""
    client = get_supabase_client()
    rows = [
        {"id": _project_id_from_url(r.url), "metadata": r.model_dump()}
        for r in records
    ]
    upsert_rows(settings.supabase_raw_table, rows, on_conflict="id", client=client)
    typer.secho(
        f"Upserted {len(rows)} records to '{settings.supabase_raw_table}'.",
        fg=typer.colors.GREEN,
    )


def upsert_media_doc(records: List[Dict[str, Any]]) -> None:
    """Upsert cleaned records into the central media_doc table (flat columns)."""
    client = get_supabase_client()
    rows = [
        {
            "id": r["id"],
            "name": r.get("Name"),
            "description": r.get("Descriptions"),
            "detail": r.get("Details"),
            "image": r.get("image_href"),
        }
        for r in records
    ]
    upsert_rows(settings.supabase_media_doc_table, rows, on_conflict="id", client=client)
    typer.secho(
        f"Upserted {len(rows)} records to '{settings.supabase_media_doc_table}'.",
        fg=typer.colors.GREEN,
    )


# ── Fetch operations ──────────────────────────────────────────────────────────

def fetch_raw_records(
    table: str,
    *,
    batch_size: int = 1000,
) -> List[Dict[str, Any]]:
    """Fetch all rows from a raw projects table, flattening metadata into each record."""
    client = get_supabase_client()
    records: List[Dict[str, Any]] = []
    offset = 0
    while True:
        resp = (
            client.table(table)
            .select("id, metadata")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        for row in rows:
            meta = row.get("metadata") or {}
            records.append({"id": row["id"], **meta})
        offset += len(rows)
    return records


def fetch_embeddings(
    table: str,
    *,
    embed_mode: BackendMode = BackendMode.openai,
    batch_size: int = 1000,
) -> Tuple[List[str], List[Dict[str, Any]], List[List[float]]]:
    """Fetch all rows from an embedding table via pagination.

    embed_mode selects the column: openai → embedding_cloud, vllm → embedding_local.
    Rows where the selected column is NULL are skipped.
    Returns (ids, metas, embs) where ids are media_doc_id values.
    """
    emb_col = "embedding_cloud" if embed_mode == BackendMode.openai else "embedding_local"
    client = get_supabase_client()
    ids: List[str] = []
    metas: List[Dict[str, Any]] = []
    embs: List[List[float]] = []
    offset = 0

    while True:
        resp = (
            client.table(table)
            .select(f"media_doc_id, {emb_col}")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        for row in rows:
            embedding = row.get(emb_col)
            if embedding is None:
                continue
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            ids.append(row["media_doc_id"])
            metas.append({})
            embs.append(embedding)
        offset += len(rows)

    return ids, metas, embs


def fetch_projects_by_ids(
    ids: List[str],
    table: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch projects from media_doc by a list of IDs (flat columns)."""
    table = table or settings.supabase_media_doc_table
    client = get_supabase_client()
    resp = (
        client.table(table)
        .select("id, name, description, detail, image")
        .in_("id", ids)
        .execute()
    )
    return list(resp.data or [])


# ── Artefact helpers (used by generate_taxonomy) ──────────────────────────────

def _load_selected_ids(path: Optional[Path]) -> List[str]:
    """Load list of ids from a JSON file. Returns [] if missing/invalid."""
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


def _artefacts_from_rows(
    rows: List[Dict[str, Any]],
    *,
    content_mode: ContentMode = ContentMode.details,
    preserve_order_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert media_doc rows into artefacts for taxonomy/retrieval use.

    Returns list of { "ID": <id>, "Description": <text> }.
    """
    order_index: Dict[str, int] = (
        {str(v): i for i, v in enumerate(preserve_order_ids)} if preserve_order_ids else {}
    )

    artefacts: List[Dict[str, Any]] = []
    for r in rows:
        _id = str(r.get("id"))
        desc = (r.get("description") or "").strip()
        detail = (r.get("detail") or "").strip()

        if content_mode == ContentMode.description:
            text = desc
        elif content_mode == ContentMode.hybrid:
            text = "; ".join(s for s in [desc, detail] if s)
        else:  # details (default)
            text = detail

        if not text:
            continue

        artefacts.append({"ID": _id, "Description": text})

    if preserve_order_ids:
        artefacts = [a for a in artefacts if a["ID"] in order_index]
        artefacts.sort(key=lambda a: order_index.get(a["ID"], 10**9))

    return artefacts


def build_artefacts(
    *,
    source: Literal["selected", "all_supabase"] = "selected",
    ids_file: Optional[Path] = None,
    content_mode: ContentMode = ContentMode.details,
    max_projects: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build artefacts from media_doc for taxonomy generation.

    - source = "selected": load ids from file, fetch matching rows, preserve order.
    - source = "all_supabase": fetch all rows (or up to ``max_projects`` when provided).

    Returns list of { "ID": <id>, "Description": <text> }.
    """
    if source == "selected":
        if not ids_file:
            raise ValueError("ids_file must be provided when source is 'selected'.")
        ids = _load_selected_ids(ids_file)
        if not ids:
            return []
        client = get_supabase_client()
        res = (
            client.table(settings.supabase_media_doc_table)
            .select("id, name, description, detail, image")
            .in_("id", ids)
            .execute()
        )
        if getattr(res, "error", None):
            raise RuntimeError(f"Supabase error: {res.error}")
        rows_map = {str(r["id"]): dict(r) for r in (res.data or [])}
        rows = [rows_map[i] for i in ids if i in rows_map]
        return _artefacts_from_rows(rows, content_mode=content_mode, preserve_order_ids=ids)

    elif source == "all_supabase":
        client = get_supabase_client()
        if max_projects is not None:
            capped = max(0, max_projects)
            if capped == 0:
                return []
            res = (
                client.table(settings.supabase_media_doc_table)
                .select("id, name, description, detail, image")
                .range(0, capped - 1)
                .execute()
            )
            if getattr(res, "error", None):
                raise RuntimeError(f"Supabase error: {res.error}")
            return _artefacts_from_rows(res.data or [], content_mode=content_mode)

        all_rows: List[Dict[str, Any]] = []
        start = 0
        limit = 1000
        while True:
            res = (
                client.table(settings.supabase_media_doc_table)
                .select("id, name, description, detail, image")
                .range(start, start + limit - 1)
                .execute()
            )
            if getattr(res, "error", None):
                raise RuntimeError(f"Supabase error: {res.error}")
            page = res.data or []
            all_rows.extend(page)
            if len(page) < limit:
                break
            start += limit
        return _artefacts_from_rows(all_rows, content_mode=content_mode)

    else:
        raise ValueError("Invalid source. Use 'selected' or 'all_supabase'.")
