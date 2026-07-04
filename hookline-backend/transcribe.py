"""
transcribe.py — Transcribe audio using Groq's Whisper-large-v3.
"""

import os
from pathlib import Path

import time

from groq import Groq


_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds, doubles each attempt


def transcribe_audio(audio_path: Path) -> dict:
    """Transcribe audio and return structured segments.

    Args:
        audio_path: Path to the audio file (WAV, mono 16 kHz recommended).

    Returns:
        dict with keys:
          - "full_text": complete transcript string
          - "segments": list of {"start": float, "end": float, "text": str}
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            with open(audio_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=(audio_path.name, audio_file),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                )
            break  # success
        except Exception as exc:
            last_err = exc
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAY * (2 ** attempt)
                print(f"[transcribe] Groq error (attempt {attempt+1}/{_MAX_RETRIES}), retrying in {wait}s: {exc}")
                time.sleep(wait)
    else:
        raise RuntimeError(f"Transcription failed after {_MAX_RETRIES} attempts: {last_err}") from last_err

    # The response object has .text and .segments
    segments = []
    for seg in response.segments or []:
        segments.append(
            {
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": seg.get("text", "").strip(),
            }
        )

    return {
        "full_text": response.text,
        "segments": segments,
    }


def format_transcript_for_llm(segments: list[dict]) -> str:
    """Format transcript segments into a timestamped string for the LLM.

    Example output line:
        [12.4s - 15.1s] And that's why I believe this changes everything.

    Args:
        segments: List of segment dicts with "start", "end", "text" keys.

    Returns:
        Multi-line string suitable for inclusion in an LLM prompt.
    """
    lines = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"]
        lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")
    return "\n".join(lines)
