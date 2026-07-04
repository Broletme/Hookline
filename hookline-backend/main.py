"""
main.py — FastAPI application for Hookline.

Endpoints
---------
POST /jobs          Accept a YouTube URL, create a DB job, kick off pipeline.
GET  /jobs/{id}     Poll job status; returns clips when done.

Pipeline stages (run in a FastAPI BackgroundTask):
  1. download   — yt-dlp + ffmpeg audio extract
  2. transcribe — Groq Whisper large-v3
  3. score      — Groq LLM clip candidate selection
  4. clip       — ffmpeg segment cutting
  5. upload     — Supabase Storage + DB write
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Must happen before any module-level os.getenv() calls.

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from clipper import cut_clips
from download import download_video
from scorer import score_transcript
from storage import (
    create_job,
    get_job,
    save_clips_to_job,
    set_status,
    update_job,
    upload_clip,
)
from transcribe import transcribe_audio

# ── Config ────────────────────────────────────────────────────────────────────
# No persistent WORK_DIR — each job gets a tempfile.TemporaryDirectory that is
# automatically deleted once clips are uploaded to Supabase Storage.
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hookline API",
    description="Extract viral clip candidates from YouTube videos.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class CreateJobRequest(BaseModel):
    youtube_url: str  # plain str so we accept youtu.be short links etc.


class ClipResponse(BaseModel):
    index: int
    start: float
    end: float
    score: int
    reason: str
    url: str


class JobResponse(BaseModel):
    id: str
    youtube_url: str
    status: str
    error: str | None = None
    clips: list[ClipResponse] | None = None


# ── Pipeline ──────────────────────────────────────────────────────────────────
async def process_job(job_id: str, youtube_url: str) -> None:
    """
    Full pipeline orchestrator.  Runs as a FastAPI BackgroundTask so the
    POST /jobs response is immediate.

    All intermediate files (video, audio, clips) are written to a
    tempfile.TemporaryDirectory that is automatically deleted once clips
    are uploaded to Supabase Storage — nothing persists locally.

    Any unhandled exception updates the job status to "error".
    """
    try:
        with tempfile.TemporaryDirectory(prefix=f"hookline_{job_id}_") as tmp:
            tmp_path = Path(tmp)

            # 1. Download ──────────────────────────────────────────────────────
            set_status(job_id, "downloading")
            video_path, audio_path = download_video(youtube_url, tmp_path)

            # 2. Transcribe ────────────────────────────────────────────────────
            set_status(job_id, "transcribing")
            segments = transcribe_audio(audio_path)
            update_job(job_id, transcript=segments)

            # Infer total duration from last segment end time.
            total_duration: float | None = None
            if segments:
                total_duration = segments[-1]["end"]

            # 3. Score ─────────────────────────────────────────────────────────
            set_status(job_id, "scoring")
            candidates = score_transcript(segments, total_duration=total_duration)

            if not candidates:
                # Edge case: LLM found no suitable clips (very short / music video).
                save_clips_to_job(job_id, [])
                return

            # 4. Cut clips ─────────────────────────────────────────────────────
            set_status(job_id, "clipping")
            clips_dir = tmp_path / "clips"
            clip_paths = cut_clips(video_path, candidates, clips_dir)

            # 5. Upload to Supabase Storage ────────────────────────────────────
            # All clips are uploaded before the TemporaryDirectory context exits
            # and local files are wiped.
            set_status(job_id, "uploading")
            clip_descriptors = []
            for idx, (clip_path, candidate) in enumerate(zip(clip_paths, candidates)):
                descriptor = upload_clip(job_id, clip_path, idx, candidate)
                clip_descriptors.append(descriptor)

        # TemporaryDirectory is deleted here — video, audio, and clip files
        # are gone; only the Supabase Storage copies remain.

        # 6. Persist and mark done ─────────────────────────────────────────────
        save_clips_to_job(job_id, clip_descriptors)

    except Exception as exc:  # noqa: BLE001
        set_status(job_id, "error", error=str(exc))
        raise


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/jobs", response_model=JobResponse, status_code=202)
async def create_job_endpoint(
    body: CreateJobRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """
    Create a new processing job and return immediately.
    Poll GET /jobs/{id} to track progress.
    """
    job_id = str(uuid.uuid4())
    try:
        job = create_job(job_id, body.youtube_url)
    except Exception as exc:
        error_msg = str(exc)
        if "Could not find the table 'public.jobs'" in error_msg:
            raise HTTPException(
                status_code=500,
                detail="Database table 'jobs' is missing. Please run schema.sql in your Supabase SQL editor."
            )
        raise HTTPException(status_code=500, detail=f"Database error: {error_msg}")

    background_tasks.add_task(process_job, job_id, body.youtube_url)

    return JobResponse(
        id=job["id"],
        youtube_url=job["youtube_url"],
        status=job["status"],
    )


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_endpoint(job_id: str) -> JobResponse:
    """Return current job state including clips once processing is complete."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    clips = None
    if job.get("clips"):
        clips = [ClipResponse(**c) for c in job["clips"]]

    return JobResponse(
        id=job["id"],
        youtube_url=job["youtube_url"],
        status=job["status"],
        error=job.get("error"),
        clips=clips,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
