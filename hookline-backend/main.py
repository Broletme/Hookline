"""
main.py — Hookline FastAPI application.

Pipeline stages per job:
  queued → downloading → transcribing → scoring → clipping → uploading → done
  Any stage can transition to: failed

CORS: Origins are loaded from the CORS_ORIGINS env var (comma-separated).
Fallback: http://localhost:3000
Without this middleware the browser will silently fail with "Failed to fetch"
because Next.js runs on a different origin to FastAPI.
"""

import os
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env before importing modules that read env vars at call time
load_dotenv()

from clipper import cut_clip
from download import download_video
from scorer import score_clips
from storage import save_job_status, upload_clip
from transcribe import format_transcript_for_llm, transcribe_audio

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Hookline API", version="1.0.0")

_raw_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateJobRequest(BaseModel):
    youtube_url: str


class CreateJobResponse(BaseModel):
    job_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/jobs", response_model=CreateJobResponse, status_code=202)
async def create_job(
    body: CreateJobRequest,
    background_tasks: BackgroundTasks,
) -> CreateJobResponse:
    """Create a new processing job and start the pipeline in the background."""
    job_id = str(uuid.uuid4())

    save_job_status(
        job_id,
        "queued",
        {"youtube_url": body.youtube_url},
    )

    background_tasks.add_task(process_job, job_id, body.youtube_url)

    return CreateJobResponse(job_id=job_id)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    """Return the current state of a job from Supabase."""
    from storage import _get_client

    client = _get_client()
    result = client.table("jobs").select("*").eq("id", job_id).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    return result.data[0]


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------


def process_job(job_id: str, youtube_url: str) -> None:
    """Run the full Hookline pipeline for a single job.

    Runs inside a temporary directory that is cleaned up automatically on exit.
    Supabase is updated at every stage so the frontend can show live progress.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        try:
            # 1. Download
            save_job_status(job_id, "downloading", {"youtube_url": youtube_url})
            video_path, audio_path = download_video(youtube_url, tmp)

            # 2. Transcribe
            save_job_status(job_id, "transcribing", {"youtube_url": youtube_url})
            transcript_data = transcribe_audio(audio_path)
            segments = transcript_data["segments"]
            full_text = transcript_data["full_text"]

            # Derive video duration from last segment end time
            video_duration = segments[-1]["end"] if segments else 0.0
            formatted_transcript = format_transcript_for_llm(segments)

            # 3. Score
            save_job_status(
                job_id,
                "scoring",
                {
                    "youtube_url": youtube_url,
                    "transcript": transcript_data,
                },
            )
            scored_clips = score_clips(formatted_transcript, video_duration)

            if not scored_clips:
                raise ValueError("LLM returned zero valid clip candidates.")

            # 4. Cut clips
            save_job_status(
                job_id,
                "clipping",
                {
                    "youtube_url": youtube_url,
                    "transcript": transcript_data,
                },
            )
            clip_paths: list[tuple[int, Path, dict]] = []
            for i, clip in enumerate(scored_clips):
                out_path = tmp / f"clip_{i}.mp4"
                cut_clip(video_path, clip["start"], clip["end"], out_path)
                clip_paths.append((i, out_path, clip))

            # 5. Upload clips
            save_job_status(
                job_id,
                "uploading",
                {
                    "youtube_url": youtube_url,
                    "transcript": transcript_data,
                },
            )
            final_clips = []
            for i, out_path, clip in clip_paths:
                url = upload_clip(out_path, job_id, i)
                final_clips.append(
                    {
                        "url": url,
                        "start": clip["start"],
                        "end": clip["end"],
                        "score": clip["score"],
                        "reason": clip["reason"],
                    }
                )

            # 6. Done
            save_job_status(
                job_id,
                "done",
                {
                    "youtube_url": youtube_url,
                    "transcript": transcript_data,
                    "clips": final_clips,
                },
            )

        except Exception as exc:  # noqa: BLE001
            save_job_status(
                job_id,
                "failed",
                {
                    "youtube_url": youtube_url,
                    "error": str(exc),
                },
            )
            # Re-raise so uvicorn logs the traceback for debugging
            raise
