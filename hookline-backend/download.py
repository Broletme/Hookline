"""
download.py — Download a YouTube video and extract its audio track.

Returns (video_path, audio_path) as pathlib.Path objects inside *out_dir*.
The caller is responsible for providing (and later cleaning up) *out_dir*;
typically this is a tempfile.TemporaryDirectory managed by main.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yt_dlp


def download_video(youtube_url: str, out_dir: Path) -> tuple[Path, Path]:
    """
    Download the best quality MP4 for *youtube_url* and extract a mono
    16 kHz WAV audio track for Whisper transcription.

    Parameters
    ----------
    youtube_url : str   URL of the YouTube video to download.
    out_dir     : Path  Directory in which to write the video and audio files.
                        Must already exist (callers use tempfile.TemporaryDirectory).

    Returns
    -------
    video_path : Path   path to the downloaded .mp4
    audio_path : Path   path to the extracted .wav
    """
    video_path = out_dir / "video.mp4"
    audio_path = out_dir / "audio.wav"

    ydl_opts = {
        # Best single-file progressive MP4 up to 1080p; avoids merge step.
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(video_path),
        # Merge into MP4 even when separate streams are selected.
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        # Don't re-download if a previous run was interrupted and left a file.
        "nooverwrites": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # yt-dlp may append part of the title to the filename; find it.
    if not video_path.exists():
        candidates = list(out_dir.glob("*.mp4"))
        if not candidates:
            raise FileNotFoundError(
                f"yt-dlp did not produce an MP4 in {out_dir}"
            )
        video_path = candidates[0]

    # Extract mono 16 kHz WAV — optimal for Whisper.
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )

    return video_path, audio_path
