"""
transcribe.py — Transcribe audio using Groq's Whisper-large-v3.
"""

import os
from pathlib import Path

from groq import Groq


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

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, audio_file),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

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
