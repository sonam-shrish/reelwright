"""Make several short films/reels in one go, sharing one Colab runtime.

    python scripts/make_series.py storyboards/connection/*.json [--scale 1.5] [--colab-spaces image,edit]

Order (so each model is loaded only once):
  1. stills for every film: Z-Image first (portraits + plain shots), then Qwen-Image-Edit
     (shots with the recurring character). Music runs on Hugging Face at the same time.
  2. every video clip with Wan 2.2 on Colab (the model stays loaded).
  3. edit each film here (captions, mix, 1080x1920 for reels).
Progress is saved per film, so re-running resumes. Colab is stopped at the end.
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_film import Film  # noqa: E402

from reelwright.studio import Studio  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storyboards", nargs="+")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--keep-colab", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="Wan size: 1.0 = 480p class, 1.5 = 720p class")
    ap.add_argument("--colab-spaces", default="edit", help='Spaces to run on Colab ("" = Hugging Face)')
    ap.add_argument("--only", default="", help="stills | clips | edit (default: everything)")
    a = ap.parse_args()
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("reelwright.colab").setLevel(logging.INFO)

    studio = Studio()
    studio.colab_space_keys = {k.strip() for k in a.colab_spaces.split(",") if k.strip()}
    studio.colab.video_scale = a.scale
    films = [Film(json.loads(Path(p).read_text()), a.draft, True, studio) for p in a.storyboards]
    t0 = time.time()
    errors: list[str] = []

    def guarded(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
            print(f"[{name}] FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)

    def sounds():
        for f in films:
            guarded(f"{f.slug} sound", f.sound_track)

    def stills():
        for which in ("plain", "character"):
            for f in films:
                guarded(f"{f.slug} stills ({which})", lambda f=f: f.stills(which))
            if which == "plain" and studio.colab_space_keys and studio.colab.is_running():
                studio.colab.call("space_stop", timeout=90, retries=2)  # free the GPU for the next model

    try:
        if a.only in ("", "stills"):
            snd = threading.Thread(target=sounds)
            snd.start()
            stills()
            snd.join()
        if a.only in ("", "clips"):
            for f in films:
                if all(f"shot{i + 1}_still" in f.state for i in range(len(f.shots))):
                    q: queue.Queue = queue.Queue()
                    for i in range(len(f.shots)):
                        q.put(i)
                    q.put(None)
                    guarded(f"{f.slug} clips", lambda f=f, q=q: f.colab_track(q))
    finally:
        if not a.keep_colab:
            try:
                print("[colab]", studio.colab.stop(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("[colab] could not stop:", e, flush=True)
    done = []
    if a.only in ("", "edit"):
        for f in films:
            if all(f"shot{i + 1}_clip" in f.state for i in range(len(f.shots))):
                guarded(f"{f.slug} edit", lambda f=f: done.append(f.edit()))
    for p in done:
        print("FILM", p, flush=True)
    print(f"total {int((time.time() - t0) / 60)} min; {len(done)}/{len(films)} films", flush=True)
    if errors:
        print("Not finished (re-run to resume):\n  " + "\n  ".join(errors), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
