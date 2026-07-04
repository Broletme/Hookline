"""
transcribe.py — Transcribe audio using Groq-hosted Whisper large-v3.

Returns a list of segment dicts, each with word-level timestamps:
[
    {
        "id": 0,
        "start": 0.0,
        "end": 4.3,
        "text": "Welcome to today's episode...",
        "words": [
            {"word": "Welcome", "start": 0.0, "end": 0.4},
            ...
        ]
    },
    ...
]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WHISPER_MODEL = "whisper-large-v3"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def transcribe_audio(audio_path: Path) -> list[dict[str, Any]]:
    """
    Send *audio_path* to Groq Whisper and return a list of segments
    with word-level timestamps.

    Parameters
    ----------
    audio_path : Path
        Path to a 16 kHz mono WAV file produced by download.py.

    Returns
    -------
    list[dict]  — segment objects (see module docstring)
    """
    client = _get_client()

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=(audio_path.name, f, "audio/wav"),
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    segments: list[dict[str, Any]] = []
    raw_segs = getattr(response, "segments", []) or []

    for i, seg in enumerate(raw_segs):
        # Groq returns Pydantic-like objects; normalise to plain dicts.
        seg_dict: dict[str, Any] = {
            "id": i,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": [],
        }

        raw_words = getattr(seg, "words", []) or []
        for w in raw_words:
            seg_dict["words"].append(
                {
                    "word": getattr(w, "word", ""),
                    "start": float(getattr(w, "start", seg.start)),
                    "end": float(getattr(w, "end", seg.end)),
                }
            )

        segments.append(seg_dict)

    return segments


def segments_to_plain_text(segments: list[dict[str, Any]]) -> str:
    """Collapse segment list into a single plain-text string for LLM input."""
    return " ".join(s["text"] for s in segments).strip()


def segments_to_timestamped_text(segments: list[dict[str, Any]]) -> str:
    """
    Produce a compact timestamped transcript suitable for the scoring LLM.
    Format: [MM:SS] segment text
    """
    lines = []
    for seg in segments:
        start_s = int(seg["start"])
        mm, ss = divmod(start_s, 60)
        lines.append(f"[{mm:02d}:{ss:02d}] {seg['text']}")
    return "\n".join(lines)
