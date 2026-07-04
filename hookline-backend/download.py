"""
download.py — Download a YouTube video and extract its audio.

IMPORTANT: Pin yt-dlp frequently (pip install -U yt-dlp) because YouTube's
anti-bot JS challenge is patched continuously in yt-dlp hotfixes. If downloads
fail with "Requested format is not available", update yt-dlp first before
chasing JS runtime flags — that's almost always a stale yt-dlp issue.
"""

import subprocess
from pathlib import Path

import yt_dlp


def download_video(youtube_url: str, out_dir: Path) -> tuple[Path, Path]:
    """Download a YouTube video and extract WAV audio from it.

    Args:
        youtube_url: The full YouTube video URL.
        out_dir: Directory where both the video and audio files will be saved.

    Returns:
        A tuple of (video_path, audio_path).
    """
    video_path = out_dir / "video.mp4"
    audio_path = out_dir / "audio.mp3"

    # --- Download video -------------------------------------------------------
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(video_path),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # Fallback: yt-dlp sometimes alters the filename (e.g. appending video ID).
    if not video_path.exists():
        mp4_files = list(out_dir.glob("*.mp4"))
        if not mp4_files:
            raise FileNotFoundError(
                f"yt-dlp finished but no .mp4 file found in {out_dir}. "
                "Try updating yt-dlp: pip install -U yt-dlp"
            )
        video_path = mp4_files[0]

    # --- Extract audio --------------------------------------------------------
    # MP3 at 32 kbps mono — ~0.24 MB/min, well under Groq's 25 MB API limit.
    # PCM WAV is ~1.9 MB/min and would exceed the limit for videos over ~13 min.
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "32k",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed:\n{result.stderr}"
        )

    return video_path, audio_path
