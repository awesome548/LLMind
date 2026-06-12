"""In-memory async job store for long-running generation.

A local LLM generation takes ~50-80s. Holding that on one HTTP request is
fragile (proxy/connection timeouts, lost on reload) and gives no progress
feedback. Instead the client submits a job, gets a ``job_id`` immediately, and
polls ``GET /api/jobs/{job_id}`` with short requests until it is ``done`` or
``error``.

Jobs run on a dedicated thread pool (separate from FastAPI's request threadpool)
and are pruned after a TTL. State is process-local — fine for a single-process
dev/prototype server; a multi-worker deployment would need a shared store.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llmind-job")
_jobs: Dict[str, Dict[str, Any]] = {}
_keyed: Dict[str, str] = {}  # dedup key → job_id of the in-flight run
_lock = threading.Lock()
_TTL_SECONDS = 1800  # drop finished jobs after 30 minutes


def _prune_locked() -> None:
    now = time.time()
    stale = [
        jid
        for jid, job in _jobs.items()
        if job["status"] in ("done", "error") and now - job["updated"] > _TTL_SECONDS
    ]
    for jid in stale:
        _jobs.pop(jid, None)


def submit(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """Run ``fn(*args, **kwargs)`` on the job pool; return a job id to poll."""
    job_id = uuid.uuid4().hex
    now = time.time()
    with _lock:
        _prune_locked()
        _jobs[job_id] = {
            "status": "pending",
            "result": None,
            "detail": None,
            "created": now,
            "updated": now,
        }

    def _run() -> None:
        try:
            result = fn(*args, **kwargs)
            with _lock:
                if job_id in _jobs:
                    _jobs[job_id].update(status="done", result=result, updated=time.time())
        except Exception as exc:  # noqa: BLE001 — recorded as job error, surfaced on poll
            detail = str(exc)
            cause = getattr(exc, "__cause__", None)
            if cause:
                detail = f"{detail} (cause: {cause})"
            with _lock:
                if job_id in _jobs:
                    _jobs[job_id].update(status="error", detail=detail, updated=time.time())

    _executor.submit(_run)
    return job_id


def submit_keyed(key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """``submit()``, deduplicated by ``key``: while a job with the same key is
    still pending, its ``job_id`` is returned instead of starting duplicate
    work. Guards expensive idempotent jobs that several clients request at
    once — e.g. two browser sessions both annotating the same taxonomy, which
    would otherwise race the local LLM through identical judgments."""
    with _lock:
        existing = _keyed.get(key)
        if existing and _jobs.get(existing, {}).get("status") == "pending":
            return existing
    job_id = submit(fn, *args, **kwargs)
    with _lock:
        _keyed[key] = job_id
    return job_id


def get(job_id: str) -> Optional[Dict[str, Any]]:
    """Snapshot of a job's state, or ``None`` if unknown/expired."""
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None
