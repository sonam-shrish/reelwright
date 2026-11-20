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
            "lines": [{"text": "...", "at": 20.0}]},        # optional narration
  "aspect": "9:16",                                          # optional, default 16:9
  "reel": true,             # export 1080x1920 30 fps, no fades (hook on frame 1, loops cleanly)
  "texts": [{"text": "...", "start": 0, "end": 3}]           # optional on-screen captions
}
"""
import argparse
import json
import math
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


CAPTION_FONTS = ["/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf", "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def caption_png(text: str, out: Path, w: int, h: int):
    """Reel-style caption: big white bold text with a dark outline and soft shadow, centred in the
    upper-middle of the frame (clear of Instagram's top bar and bottom caption/buttons)."""
    import textwrap
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    font_file = next((f for f in CAPTION_FONTS if Path(f).exists()), None)
    size = round(w * 0.08)
    font = ImageFont.truetype(font_file, size) if font_file else ImageFont.load_default(size=size)
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=18) or [""]
    line_h = round(size * 1.22)
    y0 = round(h * 0.19) - line_h * len(lines) // 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(shadow)
    for i, ln in enumerate(lines):
        x = (w - d.textlength(ln, font=font)) / 2
        y = y0 + i * line_h
        ds.text((x, y + 6), ln, font=font, fill=(0, 0, 0, 170))
        d.text((x, y), ln, font=font, fill=(255, 255, 255, 255), stroke_width=max(2, size // 18),
               stroke_fill=(0, 0, 0, 230))
    Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(10)), img).save(out)


# Film look for reels: 1080x1920 at 24 fps, soft contrast with lifted blacks, warm highlights and
# cooler shadows, gentle halation around highlights, vignette and moving grain.
FILM_LOOK = ("scale=1080:1920:flags=lanczos,setsar=1,fps=24,format=gbrp,"
             "eq=contrast=1.03:saturation=0.86:gamma=1.02,"
             "curves=r='0/0.035 0.5/0.52 1/0.97':g='0/0.03 0.5/0.5 1/0.955':b='0/0.055 0.5/0.48 1/0.92',"
             "split[fl_base][fl_hi];[fl_hi]curves=all='0/0 0.72/0 1/1',gblur=sigma=16,"
             "colorchannelmixer=rr=1:gg=0.5:bb=0.3[fl_glow];"
             "[fl_base][fl_glow]blend=all_mode=screen:all_opacity=0.16,"
             "vignette=angle=PI/5,format=yuv420p,noise=c0s=8:c0f=t+u:c1s=3:c1f=t+u:c2s=3:c2f=t+u")

# How a line sounds: a phone call, a voice note played from a phone speaker, a radio, a room.
VOICE_FX = {
    "phone": "highpass=f=320,lowpass=f=3300,acompressor=threshold=0.1:ratio=4,",
    "voicenote": "highpass=f=180,lowpass=f=7500,acompressor=threshold=0.15:ratio=3,aecho=0.8:0.5:18:0.12,",
    "radio": "highpass=f=250,lowpass=f=5000,acompressor=threshold=0.08:ratio=6,",
    "room": "aecho=0.8:0.6:35:0.18,",
    "whisper": "volume=0.8,",
}
SUBTITLE_FONTS = ["/usr/share/fonts/opentype/inter/Inter-Medium.otf", "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def subtitle_png(text: str, out: Path, w: int, h: int):
    """Film-style subtitle: medium white text with a soft shadow, low-centre, kept above
    Instagram's caption and buttons area."""
    import textwrap
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    font_file = next((f for f in SUBTITLE_FONTS if Path(f).exists()), None)
    size = round(w * 0.042)
    font = ImageFont.truetype(font_file, size) if font_file else ImageFont.load_default(size=size)
    lines = textwrap.wrap(text, width=34)[:3]
    line_h = round(size * 1.3)
    y0 = round(h * 0.70) - line_h * len(lines)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(shadow)
    for i, ln in enumerate(lines):
        x = (w - d.textlength(ln, font=font)) / 2
        y = y0 + i * line_h
        ds.text((x, y + 3), ln, font=font, fill=(0, 0, 0, 200), stroke_width=3, stroke_fill=(0, 0, 0, 160))
        d.text((x, y), ln, font=font, fill=(250, 248, 240, 255))
    Image.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(4)), img).save(out)


class Film:
    def __init__(self, sb: dict, draft: bool, keep_colab: bool, studio: Studio | None = None):
        self.sb, self.draft, self.keep_colab = sb, draft, keep_colab
        self.out = config.DATA_DIR / "films" / sb.get("slug", "film")
        self.out.mkdir(parents=True, exist_ok=True)
        self.progress = self.out / "progress.json"
        self.state = json.loads(self.progress.read_text()) if self.progress.exists() else {}
        self.lock = threading.Lock()
        self.studio = studio or Studio()
        self.aspect = sb.get("aspect", "16:9")
        self.t0 = time.time()
        self.errors: list[str] = []
        self._migrate_keys()
        try:
            self._migrate_clip_keys()
        except KeyError:
            pass

    # ---------------------------------------------------------------- asset keys
    # Assets are saved under a key made from what they depend on (not their position), so
    # adding, removing or reordering shots and lines in a storyboard reuses everything else.
    @staticmethod
    def _h(*parts) -> str:
        import hashlib
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def ref_key(self, name: str) -> str:
        p = self.sb["people"][name]
        return f"ref_{name}_{self._h(p['look'], p.get('seed', 1), self.sb.get('style', ''))}"

    def voice_key(self, name: str) -> str:
        c = self.sb["cast"][name]
        return f"voice_{name}_{self._h(c['voice'], c['sample'], c.get('language', 'English'))}"

    def still_key(self, i: int) -> str:
        s = self.shots[i]
        refs = [self.ref_key(n) for n in s.get("with", [])] or (["character"] if s.get("character") else [])
        return f"still_{self._h(s['still'], s.get('seed'), refs, self.sb.get('style', ''), self.aspect)}"

    def clip_key(self, i: int) -> str:
        # Clip length is not part of the key: retiming a film changes it a little, and any clip
        # that covers the shot's trim is usable (remotion_package checks and adapts).
        s = self.shots[i]
        return f"clip_{self._h(self.still_key(i), s['motion'], s.get('moves', []), s.get('seed'))}"

    def _legacy_clip_keys(self, i: int) -> list[str]:
        s = self.shots[i]
        return [f"clip_{self._h(self.still_key(i), s['motion'], s.get('moves', []), secs / 4, s.get('seed'))}"
                for secs in range(8, 21)]  # 2.0 .. 5.0 s in quarter seconds

    def _migrate_clip_keys(self):
        changed = False
        for i in range(len(self.shots)):
            new = self.clip_key(i)
            if new in self.state:
                continue
            for old in self._legacy_clip_keys(i):
                if old in self.state:
                    self.state[new] = self.state[old]
                    changed = True
                    break
        if changed:
            self.progress.write_text(json.dumps(self.state, indent=1))

    def sfx_key(self, i: int) -> str:
        return f"sfx_{self._h(self.shots[i]['sfx'])}"

    def line_key(self, i: int) -> str:
        ln = self.sb["lines"][i]
        return f"line_{self._h(self.voice_key(ln['who']), ln['text'], ln.get('emotion'), ln.get('strength', 0.8), ln.get('engine', ''))}"

    def end_card(self) -> float:
        return float(self.sb.get("end_card_seconds", 0))

    def music_length(self) -> float:
        """Music runs from its start to the end of the film (incl. end card), in MiniMax's 5 s steps."""
        m = self.sb["music"]
        return max(5.0, math.ceil((self.cut_duration() + self.end_card() - float(m.get("start", 0))) / 5) * 5)

    def music_key(self) -> str:
        m = self.sb["music"]
        return f"music_{self._h(m['description'], m.get('seed'), self.music_length())}"

    def ambience_key(self) -> str:
        return f"ambience_{self._h(self.sb['ambience'], round(self.cut_duration() + self.end_card()))}"

    def line_audio(self, i: int) -> str:
        """Line i's speech with the silence at both ends cut off (cached next to the progress file)."""
        src = Path(self.state[self.line_key(i)])
        out = self.out / "lines" / f"{self.line_key(i)}.wav"
        if not out.exists():
            out.parent.mkdir(exist_ok=True)
            media.trim_silence(src, out)
            if media.duration(out) < 0.3:  # trimmed too much (very quiet line): keep the original
                out.write_bytes(src.read_bytes())
        return str(out)

    def _migrate_keys(self):
        """Older progress files used positions (shot3_still, line2...): copy them to content keys."""
        st, changed = self.state, False
        # Only for progress files that have never been migrated: once content keys exist, old
        # position keys must never be mapped again (positions change when a storyboard is edited).
        if any(k.startswith(("still_", "line_", "clip_", "sfx_")) for k in st):
            return
        moved: list[str] = []

        def move(old, new):
            nonlocal changed
            if old in st and new not in st:
                st[new] = st[old]
                moved.append(old)
                changed = True
        try:
            for i, s in enumerate(self.shots):
                move(f"shot{i + 1}_still", self.still_key(i))
                move(f"shot{i + 1}_clip", self.clip_key(i))
                if s.get("sfx"):
                    move(f"shot{i + 1}_sfx", self.sfx_key(i))
            for name in self.sb.get("people") or {}:
                move(f"ref_{name}", self.ref_key(name))
            for name in self.sb.get("cast") or {}:
                move(f"voice_ref_{name}", self.voice_key(name))
            for i in range(len(self.sb.get("lines", []))):
                move(f"line{i + 1}", self.line_key(i))
        except KeyError:
            return
        for k in moved:  # drop the position keys so they can never be re-applied
            st.pop(k, None)
        if changed:
            self.progress.write_text(json.dumps(st, indent=1))

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
    def stills(self, which: str = "all", clip_queue: queue.Queue | None = None):
        """Make the stills. which: "plain" (character portrait + shots without the character,
        all Z-Image), "character" (shots with the character, Qwen-Image-Edit) or "all"."""
        q = "Standard" if self.draft else "High"
        ch = self.sb.get("character")
        style = self.sb.get("style", "")
        if which in ("plain", "all"):
            # "people": several recurring characters, each with a reference portrait (Z-Image)
            for name, person in (self.sb.get("people") or {}).items():
                if self.ref_key(name) not in self.state:
                    item = self.studio.create_images(self.job(f"{self.slug} {name}"),
                                                     f"Candid portrait photo of {person['look']}, natural window light, "
                                                     f"plain background, {style}", "None", "3:4", 1,
                                                     person.get("seed", 1))[0]
                    self.save(self.ref_key(name), item.path)
        if which in ("plain", "all") and ch and any(s.get("character") for s in self.shots) \
                and "character_ref" not in self.state:
            item = self.studio.create_images(self.job(f"{self.slug} character"), f"Portrait of {ch['description']}, "
                                             "soft natural light, photorealistic", self.sb.get("look", "Cinematic"),
                                             "3:4", 1, ch.get("seed", 1))[0]
            self.save("character_ref", item.path)
        for i, s in enumerate(self.shots):
            wanted = which == "all" or (which == "character") == bool(s.get("character") or s.get("with"))
            if wanted:
                self.still_one(i)
            if clip_queue is not None:
                clip_queue.put(i)
        if clip_queue is not None:
            clip_queue.put(None)

    def still_one(self, i: int, studio: Studio | None = None):
        key = self.still_key(i)
        if key in self.state:
            return
        st, s = studio or self.studio, self.shots[i]
        q = "Standard" if self.draft else "High"
        who = f"{self.slug} shot {i + 1} still"
        style = self.sb.get("style", "")
        prompt = f"{s['still']}, {style}" if style else s["still"]
        if s.get("with"):  # named people from "people", 1-3 of them
            refs = [self.state[self.ref_key(n)] for n in s["with"]]
            item = st.character_scene(self.job(who), refs, prompt, self.aspect, q, s.get("seed"),
                                      finish="Photorealistic." if style else "Photorealistic, cinematic lighting.")[0]
        elif s.get("character"):
            item = st.character_scene(self.job(who), [self.state["character_ref"]], prompt, self.aspect, q,
                                      s.get("seed"))[0]
        else:
            item = st.create_images(self.job(who), prompt, self.sb.get("look", "Cinematic"), self.aspect, 1,
                                    s.get("seed"))[0]
        self.save(key, item.path)

    @property
    def slug(self) -> str:
        return self.sb.get("slug", "film")

    def stills_track(self, clip_queue: queue.Queue):
        self.stills("all", clip_queue)

    def colab_track(self, clip_queue: queue.Queue):
        colab = self.studio.colab
        pending = [i for i in range(len(self.shots)) if self.clip_key(i) not in self.state]
        if not pending:
            while clip_queue.get() is not None:
                pass
            return
        colab.prewarm(lambda m: self.say("colab", m))  # loads the model while stills are made
        q = "Standard" if self.draft else "High"
        while (i := clip_queue.get()) is not None:
            key = self.clip_key(i)
            if key in self.state:
                continue
            s = self.shots[i]
            item = self.studio.create_video(self.job(f"shot {i + 1} clip"), self.state[self.still_key(i)],
                                            s["motion"], s.get("moves", []), "wan_colab",
                                            float(s.get("seconds", 5)), 1, s.get("seed"), q)[0]
            self.save(key, item.path)

    def clip_one(self, i: int, studio: Studio | None = None):
        """Render shot i's clip (used by make_series to spread shots over several Colab runtimes)."""
        key = self.clip_key(i)
        if key in self.state:
            return
        s = self.shots[i]
        q = "Standard" if self.draft else "High"
        item = (studio or self.studio).create_video(self.job(f"{self.slug} shot {i + 1} clip"),
                                                    self.state[self.still_key(i)], s["motion"],
                                                    s.get("moves", []), "wan_colab", float(s.get("seconds", 5)),
                                                    1, s.get("seed"), q)[0]
        self.save(key, item.path)

    def shot_lengths(self) -> list[float]:
        """Length of each shot in the cut (after "trim": [in, out])."""
        out = []
        for s in self.shots:
            a, b = s.get("trim", [0, float(s.get("seconds", 5))])
            out.append(float(b) - float(a))
        return out

    def cut_duration(self) -> float:
        return sum(self.shot_lengths())

    def sound_track(self, music: bool = True):
        """music=False: voices and effects only (music length depends on the retimed story)."""
        dur = self.cut_duration()
        m = self.sb.get("music") if music else None
        if m and self.music_key() not in self.state:
            self.save(self.music_key(), self.studio.music_file(self.job(f"{self.slug} music"), m["description"],
                                                               self.music_length(), instrumental=True,
                                                               seed=m.get("seed")))
        if music and self.sb.get("ambience") and self.ambience_key() not in self.state:
            self.save(self.ambience_key(), self.studio.sound_file(self.job(f"{self.slug} ambience"),
                                                                  self.sb["ambience"], dur + self.end_card()))
        # sound effects / foley per shot (EzAudio), laid under that shot
        for i, (s, length) in enumerate(zip(self.shots, self.shot_lengths())):
            if s.get("sfx") and self.sfx_key(i) not in self.state:  # 6 s, trimmed to the shot in the mix
                self.save(self.sfx_key(i), self.studio.sound_file(self.job(f"{self.slug} shot {i + 1} sfx"),
                                                                  s["sfx"], 6.0))
        # cast: design each voice once (Qwen3-TTS), then every line in that voice with its own
        # emotion (IndexTTS-2), so a character sounds the same but can laugh, break or whisper
        for name, c in (self.sb.get("cast") or {}).items():
            key = self.voice_key(name)
            if key not in self.state:
                self.save(key, self.studio.designed_speech_file(self.job(f"{self.slug} voice {name}"),
                                                                c["sample"], c["voice"], c.get("language", "English")))
        for i, line in enumerate(self.sb.get("lines", [])):
            key = self.line_key(i)
            if key not in self.state:
                ref = self.state[self.voice_key(line["who"])]
                cast = (self.sb.get("cast") or {})[line["who"]]
                lang = cast.get("language", "English")
                if lang not in ("English", "Chinese") or line.get("engine") == "qwen":
                    # IndexTTS-2 only speaks English and Chinese: clone the designed voice with Qwen3-TTS
                    path = self.studio.cloned_speech_file(self.job(f"{self.slug} line {i + 1}"), line["text"], ref,
                                                          cast["sample"], lang)
                else:
                    path = self.studio.emotional_speech_file(self.job(f"{self.slug} line {i + 1}"), line["text"],
                                                             ref, line.get("emotion"), float(line.get("strength", 0.8)))
                self.save(key, path)
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
    def line_starts(self) -> list[float]:
        """Planned start ("at") of each line, pushed later if it would overlap the line before."""
        starts, end = [], 0.0
        for i, line in enumerate(self.sb.get("lines", [])):
            a = max(float(line.get("at", 0)), end + 0.15 if i else 0.0)
            starts.append(a)
            end = a + media.duration(self.line_audio(i))
        return starts

    # ---------------------------------------------------------------- Remotion package
    def mix_audio(self, out: Path, total: float) -> Path:
        """The whole soundtrack (music, ambience, per-shot effects, voices with their phone/radio
        sound) as one mastered WAV of `total` seconds; the music rings out over the end card."""
        lines = self.sb.get("lines", [])
        line_at = self.line_starts() if lines else []
        lengths = self.shot_lengths()
        starts = [sum(lengths[:i]) for i in range(len(self.shots))]
        inputs, parts, labels = [], [], []
        m = self.sb.get("music")
        if m and self.music_key() in self.state:
            inputs.append(self.state[self.music_key()])
            ms = int(float(m.get("start", 0)) * 1000)
            parts.append(f"[{len(inputs) - 1}:a]volume={float(m.get('volume', 0.3))},afade=t=in:d={float(m.get('fade_in', 1.0))},"
                         f"adelay={ms}|{ms},apad,atrim=0:{total:.2f}[m]")
            labels.append("[m]")
        if self.sb.get("ambience") and self.ambience_key() in self.state:
            story = self.cut_duration()
            inputs.append(self.state[self.ambience_key()])
            parts.append(f"[{len(inputs) - 1}:a]volume=0.35,apad,atrim=0:{total:.2f},"
                         f"afade=t=out:st={max(0.0, story - 0.5):.2f}:d=1.2[a]")
            labels.append("[a]")
        for i, s in enumerate(self.shots):
            if s.get("sfx") and self.sfx_key(i) in self.state:
                inputs.append(self.state[self.sfx_key(i)])
                ms, ln = int(starts[i] * 1000), lengths[i]
                parts.append(f"[{len(inputs) - 1}:a]volume={float(s.get('sfx_volume', 0.6))},atrim=0:{ln + 0.4:.2f},"
                             f"afade=t=in:d=0.08,afade=t=out:st={max(0.0, ln - 0.1):.2f}:d=0.5,"
                             f"adelay={ms}|{ms},apad,atrim=0:{total:.2f}[s{i}]")
                labels.append(f"[s{i}]")
        for i, line in enumerate(lines):
            inputs.append(self.line_audio(i))
            ms = int(line_at[i] * 1000)
            fx = VOICE_FX.get(line.get("fx", ""), "")
            parts.append(f"[{len(inputs) - 1}:a]{fx}volume={float(line.get('volume', 1.4))},adelay={ms}|{ms},"
                         f"apad,atrim=0:{total:.2f}[l{i}]")
            labels.append(f"[l{i}]")
        fc = (";".join(parts) + ";" + "".join(labels) +
              f"amix=inputs={len(labels)}:duration=first:normalize=0,afade=t=in:d=0.03,"
              f"afade=t=out:st={total - 1.2:.2f}:d=1.2,alimiter=limit=0.89:level=false,"
              f"loudnorm=I=-14:TP=-1.5:LRA=11[out]")
        args = ["ffmpeg", "-v", "error", "-y"]
        for p_ in inputs:
            args += ["-i", p_]
        args += ["-filter_complex", fc, "-map", "[out]", "-ar", "48000", "-ac", "2", "-t", f"{total:.3f}", str(out)]
        subprocess.run(args, check=True)
        return out

    def remotion_package(self, public: Path, fps: int = 24, clips: list[str] | None = None) -> Path:
        """Write everything Remotion needs into public/<slug>/ and return the props JSON path.
        clips: optional stand-in clip per shot (for testing the template before real clips exist)."""
        import shutil
        missing = [what for what, ok in [
            ("music", not self.sb.get("music") or self.music_key() in self.state),
            ("ambience", not self.sb.get("ambience") or self.ambience_key() in self.state),
            ("sound effects", all(self.sfx_key(i) in self.state for i, s in enumerate(self.shots) if s.get("sfx"))),
            ("voice lines", all(self.line_key(i) in self.state for i in range(len(self.sb.get("lines", [])))))]
            if not ok]
        if missing:  # never cut a film with a silent gap: make the sound first (make_series --only prep)
            raise RuntimeError(f"{self.slug}: missing {', '.join(missing)}")
        d = public / self.slug
        d.mkdir(parents=True, exist_ok=True)
        story = self.cut_duration()
        story_frames = round(story * fps)
        total_frames = story_frames + round(self.end_card() * fps)
        shots, frame = [], 0
        for i, (s, ln) in enumerate(zip(self.shots, self.shot_lengths())):
            src = (clips[i] if clips else self.state[self.clip_key(i)])
            a = float(s.get("trim", [0, ln])[0])
            n = round((sum(self.shot_lengths()[:i + 1])) * fps) - frame  # frame-exact, no drift
            have = media.duration(src)
            if a + n / fps > have:  # clip a little short for this trim: start earlier instead of freezing
                a = max(0.0, have - n / fps)
            out = d / f"shot{i + 1:02d}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", src, "-t", f"{n / fps + 0.25:.3f}",
                            "-an", "-vf", FILM_LOOK, "-c:v", "libx264", "-crf", "14", "-preset", "medium",
                            "-pix_fmt", "yuv420p", str(out)], check=True)
            if media.duration(out) < n / fps:  # source clip too short: hold its last frame
                held = media.hold_last_frame(out, n / fps + 0.1, d / f"shot{i + 1:02d}_h.mp4")
                shutil.move(str(held), out)
            shots.append({"src": f"{self.slug}/{out.name}", "from": frame, "duration": n,
                          "push": float(s.get("push", 0.025))})
            frame += n
        audio = self.mix_audio(d / "mix.wav", total_frames / fps)
        lines, cast = [], self.sb.get("cast") or {}
        for i, (line, at) in enumerate(zip(self.sb.get("lines", []), self.line_starts())):
            src = d / f"line{i + 1:02d}.wav"
            shutil.copy(self.line_audio(i), src)
            who = line["who"]
            lines.append({"from": round(at * fps), "duration": max(1, round(media.duration(src) * fps)),
                          "text": line.get("subtitle_text", line["text"]), "who": who,
                          "label": cast.get(who, {}).get("label", who.replace("_", " ").title()),
                          "fx": line.get("fx", ""), "src": f"{self.slug}/{src.name}"})
        first = shots[0]["duration"] if shots else fps * 3
        props = {"series": self.sb.get("series", "VOICE NOTES"), "episode": int(self.sb.get("episode", 1)),
                 "title": self.sb.get("title", ""), "tagline": self.sb.get("tagline", ""),
                 "handle": self.sb.get("handle", "@_babelground"), "fps": fps, "durationInFrames": total_frames,
                 "storyFrames": story_frames, "audio": f"{self.slug}/{audio.name}", "shots": shots, "lines": lines,
                 "titleFrom": 16, "titleDuration": max(36, min(80, first - 22))}
        path = d / "props.json"
        path.write_text(json.dumps(props, indent=1, ensure_ascii=False))
        return path

    def trimmed_clips(self) -> list[str]:
        out = []
        for i, s in enumerate(self.shots):
            clip = self.state[self.clip_key(i)]
            if "trim" in s:
                a, b = (float(v) for v in s["trim"])
                t = self.out / f"shot{i + 1}_trim.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", clip, "-t", f"{b - a:.3f}",
                                "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(t)], check=True)
                clip = str(t)
            out.append(clip)
        return out

    def edit(self) -> Path:
        clips = self.trimmed_clips()
        picture = media.concat_videos(clips, self.out / "picture.mp4", fps=16)
        dur = media.duration(picture)
        lines = self.sb.get("lines", [])
        line_at = self.line_starts() if lines else []
        if lines:  # the last line must finish on picture: hold the last frame if needed
            need = line_at[-1] + media.duration(self.line_audio(len(lines) - 1)) + 0.7
            if need > dur:
                picture = media.hold_last_frame(picture, need, self.out / "picture_hold.mp4")
                dur = media.duration(picture)
        inputs, parts, labels = [], [], []
        if self.sb.get("music") and self.music_key() in self.state:
            m = self.sb.get("music", {})
            inputs.append(self.state[self.music_key()])
            ms = int(float(m.get("start", 0)) * 1000)
            parts.append(f"[{len(inputs) - 1}:a]volume={float(m.get('volume', 0.75))},afade=t=in:d={float(m.get('fade_in', 0.05))},"
                         f"adelay={ms}|{ms},apad,atrim=0:{dur:.2f}[m]")
            labels.append("[m]")
        starts = [sum(self.shot_lengths()[:i]) for i in range(len(self.shots))]
        for i, s in enumerate(self.shots):
            if s.get("sfx") and self.sfx_key(i) in self.state:
                inputs.append(self.state[self.sfx_key(i)])
                ms = int(starts[i] * 1000)
                ln = self.shot_lengths()[i]
                parts.append(f"[{len(inputs) - 1}:a]volume={float(s.get('sfx_volume', 0.6))},atrim=0:{ln + 0.4:.2f},"
                             f"afade=t=in:d=0.08,afade=t=out:st={max(0.0, ln - 0.1):.2f}:d=0.5,"
                             f"adelay={ms}|{ms},apad,atrim=0:{dur:.2f}[s{i}]")
                labels.append(f"[s{i}]")
        for i, line in enumerate(lines):
            inputs.append(self.line_audio(i))
            ms = int(line_at[i] * 1000)
            fx = VOICE_FX.get(line.get("fx", ""), "")
            parts.append(f"[{len(inputs) - 1}:a]{fx}volume={float(line.get('volume', 1.4))},adelay={ms}|{ms},"
                         f"apad,atrim=0:{dur:.2f}[l{i}]")
            labels.append(f"[l{i}]")
        if self.sb.get("ambience") and self.ambience_key() in self.state:
            inputs.append(self.state[self.ambience_key()])
            parts.append(f"[{len(inputs) - 1}:a]volume=0.4,apad,atrim=0:{dur:.2f},afade=t=out:st={max(0, dur - 4):.2f}:d=3[a]")
            labels.append("[a]")
        for i, line in enumerate((self.sb.get("voice") or {}).get("lines", [])):
            inputs.append(self.state[f"voice{i + 1}"])
            ms = int(float(line.get("at", 0)) * 1000)
            parts.append(f"[{len(inputs) - 1}:a]volume=1.5,adelay={ms}|{ms},apad,atrim=0:{dur:.2f}[v{i}]")
            labels.append(f"[v{i}]")
        final = self.out / f"{self.slug}.mp4"
        reel = bool(self.sb.get("reel"))
        if reel and self.sb.get("film_look"):  # graded like film, 24 fps, grain; ends on a cut
            vf = FILM_LOOK
            a_in, a_out = 0.03, float(self.sb.get("audio_tail", 0.6))
        elif reel:  # no fade from black: the hook is on frame 1 and the end loops back into the start
            vf = "scale=1080:1920:flags=lanczos,setsar=1,fps=30"
            a_in, a_out = 0.03, 0.4
        else:
            vf = f"fade=t=in:d=0.6,fade=t=out:st={dur - 1.2:.2f}:d=1.2"
            a_in, a_out = 0.5, 1.5
        texts = [dict(t, style="caption") for t in self.sb.get("texts", [])]
        if self.sb.get("subtitles"):
            for i, line in enumerate(lines):
                if line.get("subtitle", True):
                    a = line_at[i]
                    texts.append({"text": line.get("subtitle_text", line["text"]), "start": a, "style": "subtitle",
                                  "end": min(dur, a + media.duration(self.line_audio(i)) + 0.15)})
        args = ["ffmpeg", "-v", "error", "-y"]
        for p in inputs:
            args += ["-i", p]
        args += ["-i", str(picture)]
        vid_in = len(inputs)
        for k, t in enumerate(texts):
            png = self.out / f"text{k + 1}.png"
            size = (1080, 1920) if reel else (media.probe(picture)["width"], media.probe(picture)["height"])
            (subtitle_png if t["style"] == "subtitle" else caption_png)(t["text"], png, *size)
            args += ["-loop", "1", "-t", f"{dur:.2f}", "-i", str(png)]
        vparts = [f"[{vid_in}:v]{vf}[v0]"]
        for k, t in enumerate(texts):
            a, b = float(t["start"]), min(float(t["end"]), dur)
            vparts.append(f"[{vid_in + 1 + k}:v]format=rgba,fade=t=in:st={a:.2f}:d=0.2:alpha=1,"
                          f"fade=t=out:st={max(a, b - 0.2):.2f}:d=0.2:alpha=1[t{k}]")
            vparts.append(f"[v{k}][t{k}]overlay=0:0:enable='between(t,{a:.2f},{b:.2f})'[v{k + 1}]")
        fc = ";".join(vparts)
        if labels:
            fc += (";" + ";".join(parts) + ";" + "".join(labels) +
                   f"amix=inputs={len(labels)}:duration=first:normalize=0,afade=t=in:d={a_in},"
                   f"afade=t=out:st={dur - a_out:.2f}:d={a_out},alimiter=limit=0.89:level=false,"
                   f"loudnorm=I=-14:TP=-1.5:LRA=11[out]")
            args += ["-filter_complex", fc, "-map", f"[v{len(texts)}]", "-map", "[out]",
                     "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        else:
            args += ["-filter_complex", fc, "-map", f"[v{len(texts)}]", "-an"]
        args += ["-c:v", "libx264", "-crf", "17", "-preset", "slow", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-t", f"{dur:.2f}", str(final)]
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
