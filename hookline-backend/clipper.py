"""
clipper.py — Cut video clips using ffmpeg stream copy.

NOTE: -c copy is fast because it avoids re-encoding, but it snaps to the
nearest keyframe. If clip boundaries look off by a few seconds in testing,
switch to re-encoding:
    ffmpeg -y -ss {start} -i {video} -to {duration} -c:v libx264 -c:a aac {out}
For v1 the keyframe-snap behaviour is acceptable.
"""

import subprocess
from pathlib import Path


def cut_clip(
    video_path: Path,
    start: float,
    end: float,
    out_path: Path,
) -> None:
    """Cut a segment from a video file using ffmpeg stream copy.

    Args:
        video_path: Path to the source video file.
        start: Start time in seconds.
        end: End time in seconds.
        out_path: Destination path for the output clip.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-ss", str(start),
        "-to", str(end),
        "-c", "copy",
        "-movflags", "+faststart",  # move moov atom to front — required for browser playback
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg clip extraction failed for {start}s-{end}s:\n{result.stderr}"
        )
