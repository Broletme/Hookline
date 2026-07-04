"""
storage.py — Supabase persistence layer for Hookline.

Responsibilities:
  1. CRUD on the `jobs` table (create, read, update status/transcript/clips).
  2. Upload cut clip files to Supabase Storage and return their public URLs.

Uses the supabase-py client with the service-role key so RLS is bypassed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from supabase import Client, create_client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
CLIPS_BUCKET = os.getenv("SUPABASE_CLIPS_BUCKET", "clips")

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client


# ── Job table helpers ──────────────────────────────────────────────────────────

def create_job(job_id: str, youtube_url: str) -> dict[str, Any]:
    """Insert a new job row and return it."""
    sb = _get_client()
    row = {
        "id": job_id,
        "youtube_url": youtube_url,
        "status": "queued",
    }
    result = sb.table("jobs").insert(row).execute()
    return result.data[0]


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return the job row or None if not found."""
    sb = _get_client()
    result = sb.table("jobs").select("*").eq("id", job_id).execute()
    return result.data[0] if result.data else None


def update_job(job_id: str, **fields: Any) -> None:
    """Patch arbitrary fields on a job row."""
    sb = _get_client()
    sb.table("jobs").update(fields).eq("id", job_id).execute()


def set_status(job_id: str, status: str, error: str | None = None) -> None:
    """Convenience wrapper to update only status (and optionally error)."""
    payload: dict[str, Any] = {"status": status}
    if error is not None:
        payload["error"] = error
    update_job(job_id, **payload)


# ── Clip upload ────────────────────────────────────────────────────────────────

def upload_clip(
    job_id: str,
    clip_path: Path,
    clip_index: int,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    Upload a cut clip MP4 to Supabase Storage and return a clip descriptor dict.

    Parameters
    ----------
    job_id      : str   — job UUID
    clip_path   : Path  — local MP4 file to upload
    clip_index  : int   — 0-based rank
    candidate   : dict  — scorer output dict with start/end/score/reason

    Returns
    -------
    dict with keys: index, start, end, score, reason, url
    """
    sb = _get_client()

    # Storage path: clips/<job_id>/<filename>
    storage_key = f"{job_id}/{clip_path.name}"

    with open(clip_path, "rb") as f:
        sb.storage.from_(CLIPS_BUCKET).upload(
            path=storage_key,
            file=f,
            file_options={"content-type": "video/mp4", "upsert": "true"},
        )

    public_url = (
        sb.storage.from_(CLIPS_BUCKET).get_public_url(storage_key)
    )

    return {
        "index": clip_index,
        "start": candidate["start"],
        "end": candidate["end"],
        "score": candidate["score"],
        "reason": candidate["reason"],
        "url": public_url,
    }


def save_clips_to_job(job_id: str, clips: list[dict[str, Any]]) -> None:
    """Persist the clip descriptor list into the job's `clips` JSONB column."""
    update_job(job_id, clips=clips, status="done")
