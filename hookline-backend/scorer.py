"""
scorer.py — Score transcript segments for viral clip potential using a Groq LLM.
"""

import json
import os
import re

from groq import Groq

_SYSTEM_PROMPT = """You are a social-media content strategist who specialises in identifying
short-form viral clips from long-form video transcripts.

Given a timestamped transcript, identify 3-6 segments that would work as
standalone hooks for TikTok / Instagram Reels / YouTube Shorts.

Each segment MUST:
- Be 30–90 seconds long (use start/end times from the transcript).
- Have a strong opening line that works without prior context.
- Be self-contained — no unresolved references like "as I mentioned earlier."
- Feature at least one of: emotional peak, surprising/controversial statement,
  a complete mini-story arc, or a clear actionable insight.
- Have start and end times that exactly match timestamp values that appear in
  the provided transcript. Do NOT invent timestamps outside the transcript range.

Return ONLY a valid JSON array — no markdown fences, no explanation, no preamble.
Format:
[
  {"start": <float>, "end": <float>, "score": <int 1-10>, "reason": "<why this clip works>"},
  ...
]
"""


def score_clips(transcript_text: str, video_duration: float) -> list[dict]:
    """Use an LLM to identify and score the best viral clip candidates.

    Args:
        transcript_text: Formatted transcript string from format_transcript_for_llm().
        video_duration: Total video duration in seconds (used for context).

    Returns:
        List of dicts: [{"start": float, "end": float, "score": int, "reason": str}]

    Raises:
        ValueError: If the LLM returns unparseable JSON.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    user_message = (
        f"Video duration: {video_duration:.1f} seconds\n\n"
        f"Transcript:\n{transcript_text}"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content.strip()

    # Strip any accidental markdown code fences the model might have added
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        clips = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON. Raw response:\n{raw}\n\nError: {exc}"
        ) from exc

    if not isinstance(clips, list):
        raise ValueError(
            f"LLM JSON was not a list. Got: {type(clips).__name__}"
        )

    # Normalise and validate each clip entry
    validated = []
    for item in clips:
        try:
            validated.append(
                {
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "score": int(item["score"]),
                    "reason": str(item.get("reason", "")),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            # Skip malformed entries rather than crashing the whole job
            print(f"[scorer] Skipping malformed clip entry {item!r}: {exc}")

    return validated
