"""
storage.py — Supabase storage and database helpers.
"""

import os
from pathlib import Path

from supabase import create_client, Client


def _get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def upload_clip(local_path: Path, job_id: str, clip_index: int) -> str:
    """Upload a clip file to Supabase Storage and return its public URL.

    Args:
        local_path: Absolute path to the local clip file.
        job_id: UUID string identifying the job (used as a storage path prefix).
        clip_index: Zero-based index of the clip (used in the filename).

    Returns:
        Public URL string for the uploaded file.
    """
    bucket = os.environ.get("SUPABASE_CLIPS_BUCKET", "clips")
    client = _get_client()

    storage_path = f"{job_id}/clip_{clip_index}.mp4"

    with open(local_path, "rb") as f:
        client.storage.from_(bucket).upload(
            path=storage_path,
            file=f,
            file_options={"content-type": "video/mp4", "upsert": "true"},
        )

    # Build the public URL (works for buckets set to Public)
    public_url = (
        f"{os.environ['SUPABASE_URL']}/storage/v1/object/public/{bucket}/{storage_path}"
    )
    return public_url


def save_job_status(job_id: str, status: str, data: dict) -> None:
    """Upsert a job record in the Supabase `jobs` table.

    Args:
        job_id: UUID string for the job.
        status: Current pipeline stage (e.g. "downloading", "done", "failed").
        data: Additional fields to merge into the record (e.g. clips, error).
    """
    client = _get_client()

    record = {
        "id": job_id,
        "status": status,
        **data,
    }

    client.table("jobs").upsert(record).execute()
