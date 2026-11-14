"""Generate a video with Wan 2.2 on your own Colab GPU, from a Claude Code session.

Examples:
  python scripts/colab_video.py --image start.png --prompt "she turns and smiles" --moves "Dolly In" --seconds 4
  python scripts/colab_video.py --prompt "a red tram in Lisbon at sunrise" --variants 2   # makes the first frame too
  python scripts/colab_video.py status
  python scripts/colab_video.py stop          # stop the runtime (stops by itself after 15 idle minutes)

The result is saved to the Reelwright library and its path is printed (JSON),
so it can be used further (extend, add sound, lip-sync, join clips, ...).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reelwright import media, presets  # noqa: E402
from reelwright.jobs import Job  # noqa: E402
from reelwright.studio import Studio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="generate", choices=["generate", "status", "stop"])
    ap.add_argument("--image", help="start frame (optional; without it a first frame is made from the prompt)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--moves", default="", help='camera moves, comma separated, e.g. "Dolly In,Orbit Left"')
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--quality", choices=["Standard", "High"], default="Standard")
    ap.add_argument("--sound", default="", help="optional sound prompt (EzAudio)")
    ap.add_argument("--stop-after", action="store_true", help="stop the Colab runtime when done")
    a = ap.parse_args()

    studio = Studio()
    if a.command == "status":
        print(studio.colab.status())
        return
    if a.command == "stop":
        print(studio.colab.stop())
        return

    moves = [m.strip() for m in a.moves.split(",") if m.strip()]
    unknown = [m for m in moves if m not in presets.CAMERA_MOVES]
    if unknown:
        sys.exit(f"Unknown camera moves {unknown}. Choose from: {', '.join(presets.CAMERA_MOVES)}")
    job = Job(id="cli", title="colab video")
    job.log = lambda msg: print("·", msg, flush=True)  # live progress
    try:
        items = studio.create_video(job, a.image, a.prompt, moves, "wan_colab", a.seconds, a.variants,
                                    None if a.seed < 0 else a.seed, a.quality, sound=a.sound)
    finally:
        if a.stop_after:
            print("·", studio.colab.stop(), flush=True)
    out = []
    for it in items:
        info = media.probe(it.path)
        out.append({"path": it.path, "seconds": round(info["duration"], 2),
                    "size": f"{info['width']}x{info['height']}", "audio": info["has_audio"], "seed": it.meta.get("seed")})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
