from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from backend import jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Poll a job: ``{status: pending|done|error, result, detail}``."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found or expired.",
        )
    return {"status": job["status"], "result": job["result"], "detail": job["detail"]}
