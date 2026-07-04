"""
scorer.py — Identify viral clip candidates using a Groq LLM.

Takes the timestamped transcript produced by transcribe.py and asks
an LLM to return JSON describing 30–90 second segments that work as
standalone hooks.

Output schema (list of ClipCandidate dicts):
[
    {
        "start": 12.4,       # float, seconds from video start
        "end":   58.1,       # float, seconds from video start
        "score": 87,         # int 0–100, viral likelihood
        "reason": "Opens with a shocking statistic then resolves a curiosity gap."
    },
    ...
]
Sorted descending by score. Max 8 candidates returned.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from groq import Groq

from transcribe import segments_to_timestamped_text

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SCORER_MODEL = "llama-3.3-70b-versatile"
MAX_CLIPS = 8
MIN_DURATION = 25   # seconds — allow slight slack below 30 s
MAX_DURATION = 95   # seconds — allow slight slack above 90 s

_SYSTEM_PROMPT = """\
You are a viral content strategist with deep expertise in short-form video.
Your job is to identify the strongest 30–90 second clips from a video transcript
that would perform well as standalone social media content.

For each candidate clip, evaluate:
• Strong hook / opening line that grabs attention in the first 3 seconds
• Complete thought or story arc (beginning, tension, resolution)
• Emotional peak — surprise, curiosity, humour, or inspiration
• Curiosity gap — makes the viewer want to know more
• Quotable or shareable moment
• Self-contained — understandable without the rest of the video

Return ONLY a valid JSON array (no markdown, no prose, no ```).
Each element must have exactly these fields:
  "start"  : number (seconds, float)
  "end"    : number (seconds, float)
  "score"  : integer 0-100 (viral likelihood)
  "reason" : string (1–2 sentences explaining why this clip is strong)

Rules:
- Clip duration must be between 30 and 90 seconds.
- Return between 1 and 8 clips, sorted by score descending.
- Timestamps must map exactly to segment boundaries in the transcript.
- Do NOT fabricate timestamps that aren't in the transcript.
"""

_USER_TEMPLATE = """\
Here is the timestamped transcript. Each line starts with [MM:SS] indicating
the segment's start time in the video.

---
{transcript}
---

Identify the best viral clip candidates and return the JSON array.
"""

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _extract_json(text: str) -> list[dict[str, Any]]:
    """
    Robustly extract a JSON array from LLM output that may contain
    surrounding prose or markdown fences.
    """
    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences.
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Last resort: find first '[' … ']' block.
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"Could not extract JSON array from LLM response:\n{text[:500]}")


def _validate_clips(
    raw: list[dict[str, Any]],
    total_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Validate and normalise clip candidates."""
    valid = []
    for item in raw:
        try:
            start = float(item["start"])
            end = float(item["end"])
            score = max(0, min(100, int(item["score"])))
            reason = str(item.get("reason", "")).strip()
            duration = end - start

            if duration < MIN_DURATION or duration > MAX_DURATION:
                continue
            if start < 0:
                continue
            if total_duration and end > total_duration + 5:
                continue

            valid.append(
                {"start": start, "end": end, "score": score, "reason": reason}
            )
        except (KeyError, TypeError, ValueError):
            continue

    # Sort descending by score, cap at MAX_CLIPS.
    valid.sort(key=lambda c: c["score"], reverse=True)
    return valid[:MAX_CLIPS]


def score_transcript(
    segments: list[dict[str, Any]],
    total_duration: float | None = None,
) -> list[dict[str, Any]]:
    """
    Given word-timestamped *segments* from transcribe.py, return a list
    of clip candidate dicts sorted descending by viral score.

    Parameters
    ----------
    segments : list[dict]
        Transcript segments as returned by transcribe.transcribe_audio().
    total_duration : float | None
        Total video duration in seconds, used to sanity-check end times.

    Returns
    -------
    list[dict]  — validated ClipCandidate objects
    """
    transcript_text = segments_to_timestamped_text(segments)

    client = _get_client()
    response = client.chat.completions.create(
        model=SCORER_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(transcript=transcript_text)},
        ],
        temperature=0.2,   # low temp for deterministic JSON
        max_tokens=2048,
    )

    raw_text = response.choices[0].message.content or ""
    raw_clips = _extract_json(raw_text)
    return _validate_clips(raw_clips, total_duration)
