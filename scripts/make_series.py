"""Make several short films/reels in one go, spread over several Colab runtimes.

    python scripts/make_series.py storyboards/voice-notes/*.json --runtimes 3 --scale 1.5 --upscale 2

Order (each model loads once per runtime):
  1. plain stills (Z-Image on Hugging Face), music, ambience, sound effects and voices
     (Qwen3-TTS + IndexTTS-2 on Hugging Face) for every film, in parallel with:
  2. character stills (Qwen-Image-Edit) shared out over the Colab runtimes,
  3. every video clip (Wan 2.2, SageAttention where it works) shared out over the runtimes,
  4. edit each film here (trims, sound mix, subtitles, film look, 1080x1920).
Progress is saved per film, so re-running resumes. All runtimes are stopped at the end
(after --grace minutes, so quick fixes can reuse the loaded models).
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

from reelwright import config  # noqa: E402
from reelwright.colab import ColabRunner  # noqa: E402
from reelwright.studio import Studio  # noqa: E402


def session_name(k: int) -> str:
    return config.COLAB_SESSION if k == 0 else f"{config.COLAB_SESSION}-{k + 1}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storyboards", nargs="+")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--runtimes", type=int, default=1, help="Colab runtimes to use in parallel")
    ap.add_argument("--gpu", default=config.COLAB_GPU, help='GPU preference list, e.g. "G4,A100"')
    ap.add_argument("--scale", type=float, default=1.0, help="Wan size: 1.0 = 480p class, 1.5 = 720p class")
    ap.add_argument("--upscale", type=int, default=1, help="2 = Real-ESRGAN x2 on Colab after each clip")
    ap.add_argument("--colab-spaces", default="edit", help='Spaces to run on Colab ("" = Hugging Face)')
    ap.add_argument("--only", default="", help="prep (Hugging Face parts only) | stills | clips | edit")
    ap.add_argument("--grace", type=float, default=0, help="minutes to keep runtimes up after the work")
    ap.add_argument("--keep-colab", action="store_true")
    ap.add_argument("--no-sound", action="store_true", help="skip music/voices (made separately)")
    a = ap.parse_args()
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("reelwright.colab").setLevel(logging.INFO)

    keys = {k.strip() for k in a.colab_spaces.split(",") if k.strip()}
    studios = []
    for k in range(max(1, a.runtimes)):
        runner = ColabRunner(session=session_name(k), gpu=a.gpu, workdir=config.DATA_DIR / "colab" / f"r{k + 1}")
        runner.video_scale, runner.video_upscale = a.scale, a.upscale
        st = Studio(colab=runner)
        st.colab_space_keys = keys
        studios.append(st)
    films = [Film(json.loads(Path(p).read_text()), a.draft, True, studios[0]) for p in a.storyboards]
    t0 = time.time()
    errors: list[str] = []
    timings: dict[str, list[float]] = {"clip": [], "still": []}

    def say(msg):
        print(f"[{int(time.time() - t0):5d}s] {msg}", flush=True)

    def guarded(name, fn):
        try:
            fn()
            return True
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:300]}")
            say(f"[{name}] FAILED {type(e).__name__}: {str(e)[:300]}")
            return False

    def pool(jobs: list, kind: str):
        """Share (film, shot) jobs out over the runtimes; each runtime works through the queue."""
        q: queue.Queue = queue.Queue()
        for j in jobs:
            q.put(j)

        def worker(k: int, st: Studio):
            time.sleep(20 * k)  # stagger runtime starts: the Colab CLI keeps one shared session file
            while True:
                try:
                    f, i = q.get_nowait()
                except queue.Empty:
                    return
                t = time.time()
                fn = f.still_one if kind == "still" else f.clip_one
                if guarded(f"{f.slug} shot {i + 1} {kind} on {st.colab.session}", lambda: fn(i, st)):
                    timings[kind].append(time.time() - t)
                    say(f"{kind} {f.slug} #{i + 1} done on {st.colab.session} ({time.time() - t:.0f} s)")

        threads = [threading.Thread(target=worker, args=(k, st))
                   for k, st in enumerate(studios[:max(1, min(len(studios), len(jobs)))])]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    def sounds():
        for f in films:
            guarded(f"{f.slug} sound", f.sound_track)

    try:
        if a.only == "voices":  # voices, sound effects and plain stills; music comes after retime.py
            snd = threading.Thread(target=lambda: [guarded(f"{f.slug} voices", lambda f=f: f.sound_track(music=False))
                                                   for f in films])
            snd.start()
            for f in films:
                guarded(f"{f.slug} stills (plain)", lambda f=f: f.stills("plain"))
            snd.join()
        if a.only == "prep":  # everything that runs on Hugging Face only (no Colab)
            snd = threading.Thread(target=sounds)
            snd.start()
            for f in films:
                guarded(f"{f.slug} stills (plain)", lambda f=f: f.stills("plain"))
            snd.join()
        if a.only in ("", "stills"):
            snd = threading.Thread(target=(lambda: None) if a.no_sound else sounds)
            snd.start()
            for f in films:
                guarded(f"{f.slug} stills (plain)", lambda f=f: f.stills("plain"))
            jobs = [(f, i) for f in films for i, s in enumerate(f.shots)
                    if (s.get("character") or s.get("with")) and f.still_key(i) not in f.state]
            if jobs:
                pool(jobs, "still")
            snd.join()
        if a.only in ("", "clips"):
            jobs = [(f, i) for f in films for i in range(len(f.shots))
                    if f.clip_key(i) not in f.state and f.still_key(i) in f.state]
            if jobs:
                pool(jobs, "clip")
    finally:
        if not a.keep_colab and a.only not in ("prep", "voices"):
            if a.grace > 0:
                say(f"keeping runtimes up for {a.grace:.0f} min for quick fixes")
                time.sleep(60 * a.grace)
            for st in studios:
                try:
                    say(f"[{st.colab.session}] {st.colab.stop()}")
                except Exception as e:  # noqa: BLE001
                    say(f"[{st.colab.session}] could not stop: {e}")
    done = []
    if a.only in ("", "edit"):
        for f in films:
            if all(f.clip_key(i) in f.state for i in range(len(f.shots))):
                guarded(f"{f.slug} edit", lambda f=f: done.append(f.edit()))
    for p in done:
        print("FILM", p, flush=True)
    for kind, ts in timings.items():
        if ts:
            say(f"{kind}s: {len(ts)}, average {sum(ts) / len(ts):.0f} s each (wall clock per runtime)")
    say(f"total {int((time.time() - t0) / 60)} min; {len(done)}/{len(films)} films; {len(studios)} runtime(s)")
    if errors:
        print("Not finished (re-run to resume):\n  " + "\n  ".join(errors), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
