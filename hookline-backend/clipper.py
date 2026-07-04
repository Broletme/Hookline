"""
clipper.py — Cut scored clip segments out of the source video using ffmpeg.

Each clip is written as an H.264/AAC MP4 suitable for web playback.
Returns a list of local Path objects, one per clip.

TODO (future): add vertical 9:16 reframe (center-crop → face-detect crop)
               and burn-in captions from Whisper word timestamps.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def cut_clips(
    video_path: Path,
    clip_candidates: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    """
    Cut *clip_candidates* from *video_path* and write MP4 files to *out_dir*.

    Parameters
    ----------
    video_path : Path
        Source video produced by download.py.
    clip_candidates : list[dict]
        Output of scorer.score_transcript() — each dict has "start", "end",
        "score", "reason".
    out_dir : Path
        Directory in which to write clip files (created if missing).

    Returns
    -------
    list[Path]  — clip file paths, in the same order as *clip_candidates*.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []

    for idx, clip in enumerate(clip_candidates):
        start: float = clip["start"]
        end: float = clip["end"]
        duration = end - start
        score: int = clip["score"]

        # Filename encodes rank and score for easy sorting in the frontend.
        out_path = out_dir / f"clip_{idx:02d}_score{score}.mp4"

        _ffmpeg_cut(video_path, out_path, start, duration)
        clip_paths.append(out_path)

    return clip_paths


def _ffmpeg_cut(
    src: Path,
    dst: Path,
    start: float,
    duration: float,
) -> None:
    """
    Run ffmpeg to extract a segment from *src* into *dst*.

    Uses:
    - `-ss` before `-i` for fast seek (keyframe-accurate start).
    - `-t` to limit duration.
    - libx264 + aac re-encode to guarantee clean cuts and web compatibility.
    - `-movflags +faststart` for progressive download / instant playback.
    """
    cmd = [
        "ffmpeg", "-y",
        # Fast seek to keyframe before start (input-side flag).
        "-ss", str(start),
        "-i", str(src),
        # Then fine-seek to exact frame (output-side flag).
        "-ss", "0",
        "-t", str(duration),
        # Video: H.264 baseline, CRF 23 (good quality, small file).
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-crf", "23",
        "-preset", "fast",
        # Audio: AAC stereo 128 kbps.
        "-c:a", "aac",
        "-b:a", "128k",
        "-ac", "2",
        # Container flags.
        "-movflags", "+faststart",
        str(dst),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg clip cut failed for {dst.name}:\n{result.stderr[-2000:]}"
        )


# ---------------------------------------------------------------------------
# TODO: Vertical reframe (9:16)
# ---------------------------------------------------------------------------
# def reframe_vertical(clip_path: Path, out_path: Path) -> Path:
#     """
#     Reframe a landscape clip to 9:16 vertical format.
#     Step 1: simple center-crop (immediate).
#     Step 2 (future): face/speaker-tracking crop using mediapipe or similar.
#     """
#     raise NotImplementedError("Vertical reframe is planned for a future release.")


# ---------------------------------------------------------------------------
# TODO: Caption burn-in
# ---------------------------------------------------------------------------
# def burn_captions(clip_path: Path, words: list[dict], out_path: Path) -> Path:
#     """
#     Burn word-level captions from Whisper timestamps into the clip.
#     Uses ffmpeg drawtext or an ASS subtitle file.
#     """
#     raise NotImplementedError("Caption burn-in is planned for a future release.")
