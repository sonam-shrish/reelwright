"""Make a short film from a storyboard file, with video on your Colab GPU.

    python scripts/make_film.py storyboards/look-up.json [--draft] [--keep-colab]

Pipeline (three things run at the same time):
  * Colab:  start the A100 and load Wan 2.2 right away, then render each shot as soon
            as its still is ready (one at a time, the model stays loaded).
  * Spaces: stills (Z-Image; Qwen-Image-Edit for a consistent character), then
            music (MiniMax Music 3), ambience (EzAudio) and narration (Qwen3-TTS:
            the first line designs the voice, later lines clone it).
  * Here:   join the shots, mix and master the sound, fade in/out.
Progress is saved after every step, so re-running resumes where it stopped.
The Colab runtime is always stopped at the end (unless --keep-colab).

Storyboard format (JSON):
{
  "title": "Look Up", "slug": "look-up", "look": "Cinematic",
  "character": {"description": "...", "seed": 1},          # optional, for consistent people
  "shots": [{"still": "...", "motion": "...", "moves": ["Dolly In"], "character": false,
             "seed": 1, "seconds": 5}],
  "music": {"description": "...", "seed": 1},               # instrumental, sized to the film
  "ambience": "city traffic, light rain",                    # optional
  "voice": {"description": "a warm young woman ...",
            "lines": [{"text": "...", "at": 20.0}]}         # optional narration
}
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reelwright import config, media  # noqa: E402
from reelwright.jobs import Job  # noqa: E402
from reelwright.studio import Studio  # noqa: E402


class Film:
    def __init__(self, sb: dict, draft: bool, keep_colab: bool):
        self.sb, self.draft, self.keep_colab = sb, draft, keep_colab
        self.out = config.DATA_DIR / "films" / sb.get("slug", "film")
        self.out.mkdir(parents=True, exist_ok=True)
        self.progress = self.out / "progress.json"
        self.state = json.loads(self.progress.read_text()) if self.progress.exists() else {}
        self.lock = threading.Lock()
        self.studio = Studio()
        self.t0 = time.time()
        self.errors: list[str] = []

    # ---------------------------------------------------------------- helpers
    def say(self, who: str, msg: str):
        print(f"[{int(time.time() - self.t0):4d}s] {who}: {msg}", flush=True)

    def job(self, who: str) -> Job:
        j = Job(id=who, title=who)
        j.log = lambda m: self.say(who, m)
        return j

    def save(self, key: str, value):
        with self.lock:
            self.state[key] = str(value)
            self.progress.write_text(json.dumps(self.state, indent=1))
        self.say("✓", f"{key}")

    def guard(self, name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"{name}: {type(e).__name__}: {e}")
            self.say(name, f"FAILED {type(e).__name__}: {str(e)[:300]}")

    @property
    def shots(self):
        return self.sb["shots"]

    def duration(self) -> float:
        return sum(float(s.get("seconds", 5)) for s in self.shots)

    # ---------------------------------------------------------------- tracks
    def stills_track(self, clip_queue: queue.Queue):
        q = "Standard" if self.draft else "High"
        ch = self.sb.get("character")
        if ch and any(s.get("character") for s in self.shots) and "character_ref" not in self.state:
            item = self.studio.create_images(self.job("character"), f"Portrait of {ch['description']}, "
                                             "soft natural light, photorealistic", self.sb.get("look", "Cinematic"),
                                             "3:4", 1, ch.get("seed", 1))[0]
            self.save("character_ref", item.path)
        for i, s in enumerate(self.shots):
            key = f"shot{i + 1}_still"
            if key not in self.state:
                who = f"shot {i + 1} still"
                if s.get("character"):
                    item = self.studio.character_scene(self.job(who), [self.state["character_ref"]], s["still"],
                                                       "16:9", q, s.get("seed"))[0]
                else:
                    item = self.studio.create_images(self.job(who), s["still"], self.sb.get("look", "Cinematic"),
                                                     "16:9", 1, s.get("seed"))[0]
                self.save(key, item.path)
            clip_queue.put(i)
        clip_queue.put(None)

    def colab_track(self, clip_queue: queue.Queue):
        colab = self.studio.colab
        pending = [i for i in range(len(self.shots)) if f"shot{i + 1}_clip" not in self.state]
        if not pending:
            while clip_queue.get() is not None:
                pass
            return
        colab.prewarm(lambda m: self.say("colab", m))  # loads the model while stills are made
        q = "Standard" if self.draft else "High"
        while (i := clip_queue.get()) is not None:
            key = f"shot{i + 1}_clip"
            if key in self.state:
                continue
            s = self.shots[i]
            item = self.studio.create_video(self.job(f"shot {i + 1} clip"), self.state[f"shot{i + 1}_still"],
                                            s["motion"], s.get("moves", []), "wan_colab",
                                            float(s.get("seconds", 5)), 1, s.get("seed"), q)[0]
            self.save(key, item.path)

    def sound_track(self):
        dur = self.duration()
        m = self.sb.get("music")
        if m and "music" not in self.state:
            self.save("music", self.studio.music_file(self.job("music"), m["description"], dur, instrumental=True,
                                                      seed=m.get("seed")))
        if self.sb.get("ambience") and "ambience" not in self.state:
            self.save("ambience", self.studio.sound_file(self.job("ambience"), self.sb["ambience"], dur))
        v = self.sb.get("voice")
        if v:
            first = None
            for i, line in enumerate(v["lines"]):
                key = f"voice{i + 1}"
                if key not in self.state:
                    if first is None:
                        path = self.studio.designed_speech_file(self.job(f"voice {i + 1}"), line["text"], v["description"])
                    else:
                        path = self.studio.cloned_speech_file(self.job(f"voice {i + 1}"), line["text"], first[0], first[1])
                    self.save(key, path)
                if first is None:
                    first = (self.state[key], line["text"])

    # ---------------------------------------------------------------- edit
    def edit(self) -> Path:
        clips = [self.state[f"shot{i + 1}_clip"] for i in range(len(self.shots))]
        picture = media.concat_videos(clips, self.out / "picture.mp4", fps=16)
        dur = media.duration(picture)
        inputs, parts, labels = [], [], []
        if "music" in self.state:
            inputs.append(self.state["music"])
            parts.append(f"[{len(inputs) - 1}:a]volume=0.75,apad,atrim=0:{dur:.2f}[m]")
            labels.append("[m]")
        if "ambience" in self.state:
            inputs.append(self.state["ambience"])
            parts.append(f"[{len(inputs) - 1}:a]volume=0.4,apad,atrim=0:{dur:.2f},afade=t=out:st={max(0, dur - 4):.2f}:d=3[a]")
            labels.append("[a]")
        for i, line in enumerate((self.sb.get("voice") or {}).get("lines", [])):
            inputs.append(self.state[f"voice{i + 1}"])
            ms = int(float(line.get("at", 0)) * 1000)
            parts.append(f"[{len(inputs) - 1}:a]volume=1.5,adelay={ms}|{ms},apad,atrim=0:{dur:.2f}[v{i}]")
            labels.append(f"[v{i}]")
        final = self.out / f"{self.sb.get('slug', 'film')}.mp4"
        vf = f"fade=t=in:d=0.6,fade=t=out:st={dur - 1.2:.2f}:d=1.2"
        if labels:
            fc = (";".join(parts) + ";" + "".join(labels) +
                  f"amix=inputs={len(labels)}:duration=first:normalize=0,afade=t=in:d=0.5,"
                  f"afade=t=out:st={dur - 1.5:.2f}:d=1.5,alimiter=limit=0.89:level=false,"
                  f"loudnorm=I=-16:TP=-1.5:LRA=11[out]")
            args = ["ffmpeg", "-v", "error", "-y"]
            for p in inputs:
                args += ["-i", p]
            args += ["-i", str(picture), "-filter_complex", fc + f";[{len(inputs)}:v]{vf}[vid]",
                     "-map", "[vid]", "-map", "[out]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        else:
            args = ["ffmpeg", "-v", "error", "-y", "-i", str(picture), "-vf", vf, "-an"]
        args += ["-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-t", f"{dur:.2f}", str(final)]
        subprocess.run(args, check=True)
        return final

    # ---------------------------------------------------------------- run
    def run(self) -> Path:
        clip_queue: queue.Queue = queue.Queue()
        stills = threading.Thread(target=self.guard, args=("stills", lambda: self.stills_track(clip_queue)))
        colab = threading.Thread(target=self.guard, args=("colab", lambda: self.colab_track(clip_queue)))
        sound = threading.Thread(target=self.guard, args=("sound", self.sound_track))
        # With Spaces on Colab, stills/voices and Wan share one GPU: do them first, then the video.
        sequential = bool(self.studio.colab_space_keys)
        try:
            stills.start()
            sound.start()
            if sequential:
                stills.join()
                sound.join()
                if self.errors:
                    raise SystemExit("Not finished (re-run to resume):\n  " + "\n  ".join(self.errors))
            colab.start()
            stills.join()
            if self.errors:  # stills failed: let the Colab track finish what it has, then stop
                clip_queue.put(None)
            colab.join()
            sound.join()
        finally:
            if not self.keep_colab:
                try:
                    self.say("colab", self.studio.colab.stop())
                except Exception as e:  # noqa: BLE001
                    self.say("colab", f"could not stop: {e}")
        if self.errors:
            raise SystemExit("Not finished (re-run to resume):\n  " + "\n  ".join(self.errors))
        final = self.edit()
        item = self.studio.library.add(final, f"Film · {self.sb.get('title', 'Untitled')}", self.sb.get("title", ""),
                                       {"shots": len(self.shots), "draft": self.draft})
        info = media.probe(final)
        self.say("done", f"{item.path}  ({info['duration']:.1f} s, {info['width']}x{info['height']}, "
                         f"total time {int((time.time() - self.t0) / 60)} min)")
        return Path(item.path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("storyboard")
    ap.add_argument("--draft", action="store_true", help="Standard quality (about twice as fast)")
    ap.add_argument("--keep-colab", action="store_true", help="leave the Colab runtime running afterwards")
    ap.add_argument("--colab-spaces", default="",
                    help='also run these Spaces on Colab instead of ZeroGPU, e.g. "edit,voice_design,image" or "all"')
    a = ap.parse_args()
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("reelwright.colab").setLevel(logging.INFO)  # show Colab Space progress
    film = Film(json.loads(Path(a.storyboard).read_text()), a.draft, a.keep_colab)
    if a.colab_spaces:
        film.studio.colab_space_keys = {k.strip() for k in a.colab_spaces.split(",") if k.strip()}
    film.run()


if __name__ == "__main__":
    main()
