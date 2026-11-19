"""ffmpeg helpers: probing, muxing audio, stitching clips, frames."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"{args[0]} failed: {proc.stderr.strip()[-600:]}")
    return proc.stdout


def probe(path: str | Path) -> dict:
    out = _run(["ffprobe", "-v", "error", "-show_entries",
                "format=duration:stream=codec_type,width,height,r_frame_rate",
                "-of", "json", str(path)])
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "width": int(video["width"]) if video else None,
        "height": int(video["height"]) if video else None,
    }


def duration(path: str | Path) -> float:
    return probe(path)["duration"]


def last_frame(video: str | Path, out: str | Path) -> Path:
    _run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.25", "-i", str(video),
          "-update", "1", "-frames:v", "1", "-q:v", "2", str(out)])
    if not Path(out).is_file():
        raise MediaError("Could not extract the last frame")
    return Path(out)


def mux_audio(video: str | Path, audio: str | Path, out: str | Path,
              mode: str = "replace", new_volume: float = 1.0, old_volume: float = 1.0) -> Path:
    """Put audio on a video. mode='replace' drops existing sound, 'mix' blends both.
    The result keeps the video's length (audio is padded or cut)."""
    info = probe(video)
    dur = info["duration"]
    if mode == "mix" and info["has_audio"]:
        fc = (f"[0:a]volume={old_volume}[a0];[1:a]volume={new_volume},apad[a1];"
              f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[a]")
        args = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio),
                "-filter_complex", fc, "-map", "0:v", "-map", "[a]"]
    else:
        args = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio),
                "-filter_complex", f"[1:a]volume={new_volume},apad[a]", "-map", "0:v", "-map", "[a]"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", str(out)]
    _run(args)
    return Path(out)


def concat_videos(paths: list[str | Path], out: str | Path, fps: int = 24) -> Path:
    """Join clips into one video. Clips are scaled and padded to the first clip's size;
    clips without sound get silence so the result always has an audio track."""
    if not paths:
        raise MediaError("No clips to join")
    first = probe(paths[0])
    w, h = first["width"], first["height"]
    w, h = w - w % 2, h - h % 2
    args = ["ffmpeg", "-v", "error", "-y"]
    for p in paths:
        args += ["-i", str(p)]
    parts, labels = [], []
    for i, p in enumerate(paths):
        info = probe(p)
        parts.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                     f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v{i}]")
        if info["has_audio"]:
            parts.append(f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo[a{i}]")
        else:
            parts.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{info['duration']:.3f}[a{i}]")
        labels.append(f"[v{i}][a{i}]")
    fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(paths)}:v=1:a=1[v][a]"
    args += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(out)]
    _run(args)
    return Path(out)


def concat_audio(paths: list[str | Path], out: str | Path) -> Path:
    if len(paths) == 1:
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(paths[0]), str(out)])
        return Path(out)
    args = ["ffmpeg", "-v", "error", "-y"]
    for p in paths:
        args += ["-i", str(p)]
    fc = "".join(f"[{i}:a]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=0:a=1[a]"
    _run(args + ["-filter_complex", fc, "-map", "[a]", str(out)])
    return Path(out)


def loop_video_to(video: str | Path, seconds: float, out: str | Path) -> Path:
    """Repeat a clip (forward, then backward) until it lasts `seconds`."""
    fc = "[0:v]split[f][b];[b]reverse[r];[f][r]concat=n=2:v=1:a=0,loop=loop=-1:size=32767[v]"
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-filter_complex", fc, "-map", "[v]",
          "-t", f"{seconds:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
          "-pix_fmt", "yuv420p", str(out)])
    return Path(out)


def pad_audio(audio: str | Path, seconds: float, out: str | Path) -> Path:
    """Add `seconds` of silence at the end (talking-head models trim the tail)."""
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-af", f"apad=pad_dur={seconds:.3f}", str(out)])
    return Path(out)


def hold_last_frame(video: str | Path, seconds: float, out: str | Path) -> Path:
    """Make a clip at least `seconds` long by holding its last frame."""
    extra = max(0.0, seconds - duration(video))
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf",
          f"tpad=stop_mode=clone:stop_duration={extra:.3f},format=yuv420p", "-an",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(out)])
    return Path(out)


def silence_points(audio: str | Path, noise_db: int = -35, min_silence: float = 0.15) -> list[float]:
    """Midpoints of quiet gaps in the audio, in seconds."""
    proc = subprocess.run(["ffmpeg", "-v", "info", "-i", str(audio), "-af",
                           f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
                          capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", proc.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def plan_segments(total: float, quiet: list[float], max_len: float, min_len: float = 1.0) -> list[tuple[float, float]]:
    """Cut [0, total] into pieces no longer than max_len, preferring quiet points."""
    segs, start = [], 0.0
    while total - start > max_len:
        options = [q for q in quiet if start + min_len <= q <= start + max_len]
        cut = max(options) if options else start + max_len
        segs.append((start, cut))
        start = cut
    segs.append((start, total))
    return segs


def split_audio(audio: str | Path, max_len: float, out_dir: str | Path) -> list[Path]:
    """Split speech into pieces <= max_len seconds, cutting at pauses where possible."""
    total = duration(audio)
    if total <= max_len:
        return [Path(audio)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (s, e) in enumerate(plan_segments(total, silence_points(audio), max_len)):
        p = out_dir / f"{Path(audio).stem}-part{i}.wav"
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-ss", f"{s:.3f}", "-to", f"{e:.3f}", str(p)])
        parts.append(p)
    return parts


def fit_to_audio(video: str | Path, audio: str | Path, out: str | Path) -> Path:
    """Video with exactly the audio's length and sound (holds the last frame if short)."""
    need = duration(audio)
    if duration(video) < need:
        video = hold_last_frame(video, need, Path(out).with_suffix(".hold.mp4"))
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "192k", "-t", f"{need:.3f}", str(out)])
    return Path(out)


def trim_audio(audio: str | Path, seconds: float, out: str | Path) -> Path:
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-t", f"{seconds:.3f}", str(out)])
    return Path(out)


def trim_silence(audio: str | Path, out: str | Path, threshold_db: float = -42, pad: float = 0.08) -> Path:
    """Cut silence from the start and end of a speech clip (TTS output often has a second or
    two of dead air), keeping `pad` seconds of natural room either side."""
    f = (f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_silence={pad},"
         f"areverse,silenceremove=start_periods=1:start_threshold={threshold_db}dB:start_silence={pad},areverse")
    _run(["ffmpeg", "-v", "error", "-y", "-i", str(audio), "-af", f, str(out)])
    return Path(out)
