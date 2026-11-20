"""Final edit of finished films with Remotion: grade + package each film, render it, and
encode a delivery file (H.264, 1080x1920, 24 fps, AAC, +faststart).

    python scripts/render_films.py storyboards/voice-notes/*.json [--out out/voice-notes]

Needs every clip of a film rendered first (make_series.py); films that are not ready are skipped.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_film import Film  # noqa: E402

REMOTION = ROOT / "remotion"


class _NoStudio:
    colab = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storyboards", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "out" / "voice-notes"))
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in a.storyboards:
        sb = json.loads(Path(p).read_text())
        f = Film(sb, False, True, studio=_NoStudio())
        missing = [i + 1 for i in range(len(f.shots)) if not f.shot_ready(i)]
        if missing:
            print(f"{f.slug}: skipped, clips missing for shots {missing}", flush=True)
            continue
        t0 = time.time()
        props = f.remotion_package(REMOTION / "public")
        raw = REMOTION / "out" / f"{f.slug}.mp4"
        raw.parent.mkdir(exist_ok=True)
        subprocess.run(["npx", "remotion", "render", "src/index.ts", "VoiceNote", str(raw), f"--props={props}",
                        "--log=error"], cwd=REMOTION, check=True)
        final = out_dir / f"{int(sb.get('episode', 0)):02d}-{f.slug}.mp4"
        part = final.with_suffix(".part.mp4")  # renamed only when complete: no half-written deliverables
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "slow",
                        "-crf", "19", "-tune", "grain", "-profile:v", "high", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", str(part)],
                       check=True)
        part.replace(final)
        raw.unlink(missing_ok=True)
        print(f"FILM {final} ({final.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
