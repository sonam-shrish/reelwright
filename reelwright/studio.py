"""Everything Reelwright can make. Each method runs inside a background job and
returns the new library items. Models run on Hugging Face Spaces."""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from . import config, media, presets
from .library import Item, Library, UsageMeter
from .spaces import LocalFile, SpaceCaller, SpaceError, fetch, file_refs, save_data_uri, to_data_uri


def _handle(path):
    return LocalFile(str(path))


def _num(x: Any) -> int:
    """Gradio sometimes returns {'value': n, '__type__': 'update'} instead of n."""
    if isinstance(x, dict):
        x = x.get("value")
    return int(x)


NSFW_PLACEHOLDER_SIZE = (1024, 1024)  # Z-Image Space's nsfw.png


class Studio:
    def __init__(self, data_dir: Path | None = None, token: str | None = None,
                 spaces: dict | None = None, caller_factory: Callable[[str], Any] | None = None,
                 handle: Callable[[Any], Any] = _handle, colab=None):
        self.data = Path(data_dir or config.DATA_DIR)
        self.tmp = self.data / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.library = Library(self.data)
        self.usage = UsageMeter(self.data)
        self.token = token if token is not None else config.HF_TOKEN
        self.spaces = spaces or config.SPACES
        self._handle = handle
        self._colab = colab
        # Spaces (by key) that run on the Colab GPU instead of ZeroGPU
        self.colab_space_keys = set(config.COLAB_SPACE_KEYS)
        self._callers: dict[str, Any] = {}
        self._factory = caller_factory or (
            lambda space: SpaceCaller(space, self.token, on_timing=self.usage.record))

    # ------------------------------------------------------------ plumbing
    def caller(self, key: str):
        if key not in self._callers:
            if self.colab_space_keys and (key in self.colab_space_keys or "all" in self.colab_space_keys):
                from .colab import ColabSpaceCaller
                self._callers[key] = ColabSpaceCaller(self.colab, self.spaces[key])
            else:
                self._callers[key] = self._factory(self.spaces[key])
        return self._callers[key]

    def _fetch_first(self, result: Any, ext: str, key: str, last: bool = False) -> Path:
        refs = file_refs(result)
        if not refs:
            raise SpaceError("failed", f"No file came back. Raw result: {str(result)[:200]}", self.spaces[key])
        # Download over the same connection as the job, so it reaches the same replica.
        client = getattr(self.caller(key), "_client", None)
        http = getattr(client, "http", None)
        return fetch(refs[-1] if last else refs[0], self.tmp, None if http else self.token, default_ext=ext, http=http)

    def _tmp(self, ext: str) -> Path:
        return self.tmp / f"{uuid.uuid4().hex[:12]}{ext}"

    # ------------------------------------------------------------ images
    def create_images(self, job, prompt: str, look: str = "None", aspect: str = "16:9",
                      count: int = 1, seed: int | None = None, quality: str = "Standard") -> list[Item]:
        if not prompt.strip():
            raise SpaceError("invalid", "Please describe the image.")
        full = presets.styled(prompt, look)
        res = presets.image_resolution(aspect, quality)
        items = []
        for i in range(max(1, int(count))):
            s = presets.seed_for(seed, i)
            want = tuple(int(v) for v in res.split(" ")[0].split("x"))
            for attempt in range(3):
                job.log(f"Image {i + 1}/{count}: Z-Image-Turbo, {res}, seed {s}")
                r = self.caller("image").call("/generate", prompt=full, resolution=res, seed=s, steps=8,
                                              shift=3.0, random_seed=False, gallery_images=[])
                path = self._fetch_first(r[0] if isinstance(r, (list, tuple)) else r, ".png", "image", last=True)
                with Image.open(path) as im:
                    if im.size == want or im.size != NSFW_PLACEHOLDER_SIZE:
                        break
                # The Space's safety checker sometimes flags harmless prompts and returns its
                # "maybe not safe" placeholder (1024x1024 whatever size was asked): try another seed.
                job.log("Z-Image returned its safety placeholder; retrying with another seed")
                s += 7919
            else:
                raise SpaceError("blocked", "Z-Image's safety filter rejected this prompt three times; "
                                            "rephrase it.")
            items.append(self.library.add(path, f"Image · {look if look != 'None' else 'plain'} · {aspect}",
                                          prompt, {"seed": s, "look": look, "aspect": aspect, "model": "Z-Image-Turbo"}))
        return items

    def edit_image(self, job, images: list[str], prompt: str, aspect: str = "Keep",
                   quality: str = "Standard", seed: int | None = None, title: str = "Edit") -> list[Item]:
        images = [p for p in (images or []) if p]
        if not images:
            raise SpaceError("invalid", "Add at least one image.")
        if not prompt.strip():
            raise SpaceError("invalid", "Describe the change you want.")
        with Image.open(images[0]) as im:
            size = im.size
        w, h = presets.edit_dims(None if aspect == "Keep" else aspect, size)
        steps = 40 if quality == "High" else 24
        s = presets.seed_for(seed, 0)
        job.log(f"Qwen-Image-Edit: {len(images)} image(s), {w}x{h}, {steps} steps, seed {s}")
        r = self.caller("edit").call("/infer", images=[{"image": self._handle(p), "caption": None} for p in images],
                                     prompt=prompt, seed=s, randomize_seed=False, true_guidance_scale=4.0,
                                     num_inference_steps=steps, height=h, width=w, rewrite_prompt=False)
        path = self._fetch_first(r[0] if isinstance(r, (list, tuple)) else r, ".png", "edit")
        return [self.library.add(path, title, prompt, {"seed": s, "model": "Qwen-Image-Edit-2511", "inputs": len(images)})]

    def character_scene(self, job, references: list[str], scene: str, aspect: str = "9:16",
                        quality: str = "Standard", seed: int | None = None,
                        finish: str = "Photorealistic, cinematic lighting.") -> list[Item]:
        refs = [p for p in (references or []) if p]
        if not refs:
            raise SpaceError("invalid", "Add 1 to 3 reference images of the character.")
        if len(refs) == 1:
            keep = ("Keep the exact same character from the reference image: the same face, hair, body and "
                    "outfit, fully consistent identity.")
        else:
            keep = (f"Keep the exact same {len(refs)} people from the reference images, one person per image: "
                    "each keeps their own face, hair, body and outfit, fully consistent identities.")
        prompt = f"{keep} New scene: {scene.strip()}. {finish}".strip()
        return self.edit_image(job, refs[:3], prompt, aspect=aspect, quality=quality, seed=seed, title="Character scene")

    def _lora_edit(self, job, image: str, adapter: str, prompt: str, seed: int | None, title: str) -> list[Item]:
        if not image:
            raise SpaceError("invalid", "Add an image.")
        s = presets.seed_for(seed, 0)
        job.log(f"Qwen-Image-Edit LoRA '{adapter}', seed {s}")
        r = self.caller("edit_lora").call("/edit_image", images_b64_json=json.dumps([to_data_uri(image)]),
                                          prompt=prompt, lora_adapter=adapter, seed=s, randomize_seed=False,
                                          guidance_scale=1.0, steps=4)
        while isinstance(r, (list, tuple)) and r:
            r = r[0]
        if isinstance(r, str):
            try:
                r = json.loads(r)
            except json.JSONDecodeError:
                pass
        if not isinstance(r, dict) or r.get("status") != "success" or not r.get("image"):
            msg = r.get("message") if isinstance(r, dict) else str(r)[:200]
            kind = "blocked" if isinstance(r, dict) and r.get("status") == "blocked" else "failed"
            raise SpaceError(kind, msg or "No image came back", self.spaces["edit_lora"])
        path = save_data_uri(r["image"], self.tmp)
        return [self.library.add(path, title, prompt, {"seed": s, "adapter": adapter})]

    def change_angle(self, job, image: str, angle: str, seed: int | None = None) -> list[Item]:
        return self._lora_edit(job, image, "Multiple-Angles", presets.ANGLES[angle], seed, f"Angle · {angle}")

    def apply_effect(self, job, image: str, effect: str, prompt: str = "", seed: int | None = None) -> list[Item]:
        adapter, default = presets.EFFECTS[effect]
        return self._lora_edit(job, image, adapter, prompt.strip() or default, seed, f"Effect · {effect}")

    # ------------------------------------------------------------ video
    def _wan(self, job, image: str, prompt: str, moves, seconds: float, seed: int, quality: str) -> Path:
        steps = 8 if quality == "High" else 4
        fluid = 32 if quality == "High" else 16  # the Space wants a number, despite its API docs
        job.log(f"Wan 2.2: {seconds:.1f}s, {steps} steps, seed {seed}")
        r = self.caller("wan").call("/generate_video", input_image=self._handle(image), last_image=None,
                                    prompt=prompt, steps=steps, negative_prompt=presets.wan_negative(moves),
                                    duration_seconds=seconds, seed=seed, randomize_seed=False,
                                    frame_multiplier=fluid)
        return self._fetch_first(r[0] if isinstance(r, (list, tuple)) else r, ".mp4", "wan")

    def _ltx(self, job, image: str, prompt: str, seconds: float, seed: int, quality: str) -> Path:
        # Same size table the Space uses (its own size endpoint is flaky).
        with Image.open(image) as im:
            w, h = presets.ltx_dims(im.size, quality)
        if quality == "High" and seconds > 5:
            job.log("Note: LTX has a 75 s GPU limit per clip; long high-resolution clips may be stopped")
        job.log(f"LTX-2.3: {seconds:.1f}s at {w}x{h}, seed {seed}")
        r = self.caller("ltx").call("/generate_video", input_image=self._handle(image), prompt=prompt,
                                    duration=seconds, enhance_prompt=False, seed=seed, randomize_seed=False,
                                    height=h, width=w)
        return self._fetch_first(r[0] if isinstance(r, (list, tuple)) else r, ".mp4", "ltx")

    @property
    def colab(self):
        if self._colab is None:
            from .colab import ColabRunner
            self._colab = ColabRunner()
        return self._colab

    def _clip(self, job, model, image, prompt, moves, seconds, seed, quality) -> Path:
        vp = presets.video_prompt(prompt, moves)
        if model == "wan_colab":
            return self.colab.generate_video(image, vp, seconds, seed, steps=8 if quality == "High" else 4,
                                             negative_prompt=presets.wan_negative(moves), say=job.log)
        if model == "ltx":
            return self._ltx(job, image, vp, seconds, seed, quality)
        return self._wan(job, image, vp, moves, seconds, seed, quality)

    def create_video(self, job, image: str | None, prompt: str, moves=(), model: str = "wan",
                     seconds: float = 4.0, variants: int = 1, seed: int | None = None,
                     quality: str = "Standard", aspect: str = "16:9", look: str = "None",
                     sound: str = "") -> list[Item]:
        if not image and not prompt.strip():
            raise SpaceError("invalid", "Add a start image or describe the shot.")
        seconds = max(1.0, min(float(seconds), config.VIDEO_MAX_SECONDS))
        if not image:
            job.log("No start image: creating the first frame")
            image = self.create_images(job, prompt, look, aspect, 1, seed, quality)[0].path
        items = []
        for i in range(max(1, int(variants))):
            s = presets.seed_for(seed, i)
            job.log(f"Variant {i + 1}/{variants}")
            path = self._clip(job, model, image, prompt, moves, seconds, s, quality)
            if sound.strip():
                path = self._with_sound(job, path, sound, mode="mix" if model == "ltx" else "replace")
            label = presets.MODEL_LABELS.get(model, model)
            items.append(self.library.add(path, f"Video · {label} · {', '.join(moves) or 'auto camera'}", prompt,
                                          {"seed": s, "model": label, "moves": list(moves), "seconds": seconds,
                                           "start_image": str(image)}))
        return items

    def extend_video(self, job, video: str, prompt: str, moves=(), model: str = "wan",
                     seconds: float = 4.0, quality: str = "Standard", seed: int | None = None) -> list[Item]:
        if not video:
            raise SpaceError("invalid", "Pick a video to extend.")
        frame = media.last_frame(video, self._tmp(".png"))
        job.log("Continuing from the last frame")
        s = presets.seed_for(seed, 0)
        new = self._clip(job, model, str(frame), prompt, moves, max(1.0, min(float(seconds), config.VIDEO_MAX_SECONDS)), s, quality)
        job.log("Joining the clips")
        joined = media.concat_videos([video, new], self._tmp(".mp4"))
        return [self.library.add(joined, "Video · extended", prompt, {"seed": s, "source": video})]

    def stitch(self, job, videos: list[str]) -> list[Item]:
        videos = [v for v in (videos or []) if v]
        if len(videos) < 2:
            raise SpaceError("invalid", "Pick at least two clips to join.")
        job.log(f"Joining {len(videos)} clips")
        out = media.concat_videos(videos, self._tmp(".mp4"))
        return [self.library.add(out, f"Sequence · {len(videos)} shots", "", {"sources": videos})]

    # ------------------------------------------------------------ audio
    def speech_file(self, job, script: str, voice: str | None = None, expressiveness: float = 0.5) -> Path:
        chunks = presets.split_text(script, config.TTS_MAX_CHARS)
        if not chunks:
            raise SpaceError("invalid", "Write the words to speak.")
        parts = []
        for i, chunk in enumerate(chunks):
            job.log(f"Speech {i + 1}/{len(chunks)}: Chatterbox{' (cloned voice)' if voice else ''}")
            r = self.caller("tts").call("/generate_tts_audio", text_input=chunk,
                                        audio_prompt_path_input=self._handle(voice) if voice else None,
                                        exaggeration_input=float(expressiveness), temperature_input=0.8,
                                        seed_num_input=0, cfgw_input=0.5, vad_trim_input=False)
            parts.append(self._fetch_first(r, ".wav", "tts"))
        return media.concat_audio(parts, self._tmp(".wav")) if len(parts) > 1 else parts[0]

    def designed_speech_file(self, job, text: str, voice_description: str, language: str = "English") -> Path:
        """Speech in a voice described in words, incl. emotion and delivery (Qwen3-TTS VoiceDesign).
        Each call designs a new voice: use cloned_speech_file for more lines in the same voice."""
        if not text.strip():
            raise SpaceError("invalid", "Write the words to speak.")
        job.log(f"Qwen3-TTS voice design: {voice_description[:60]}")
        r = self.caller("voice_design").call("/generate_voice_design", text=text, language=language,
                                             voice_description=voice_description)
        return self._fetch_first(r, ".wav", "voice_design")

    def cloned_speech_file(self, job, text: str, reference: str, reference_text: str, language: str = "English") -> Path:
        """Speech in the voice of a reference clip (Qwen3-TTS voice clone)."""
        job.log("Qwen3-TTS voice clone")
        r = self.caller("voice_design").call("/generate_voice_clone", ref_audio=self._handle(reference),
                                             ref_text=reference_text, target_text=text, language=language,
                                             use_xvector_only=False, model_size="1.7B")
        return self._fetch_first(r, ".wav", "voice_design")

    def emotional_speech_file(self, job, text: str, reference: str, emotion: dict | str | None = None,
                              strength: float = 0.8) -> Path:
        """Speech in the voice of a reference clip with a chosen emotion (IndexTTS-2), e.g.
        emotion={"sad": 0.5, "calm": 0.3}. Pair with designed_speech_file for the reference."""
        if not text.strip():
            raise SpaceError("invalid", "Write the words to speak.")
        vec = presets.emotion_vector(emotion)
        job.log(f"IndexTTS-2: {emotion or 'calm'}")
        kw = {f"vec{i + 1}": v for i, v in enumerate(vec)}
        r = self.caller("emotion_tts").call("/gen_single", emo_control_method="Use emotion vectors",
                                            prompt=self._handle(reference), text=text, emo_ref_path=None,
                                            emo_weight=float(strength), emo_text="", emo_random=False,
                                            max_text_tokens_per_segment=120, **kw)
        return self._fetch_first(r, ".wav", "emotion_tts")

    def speech(self, job, script: str, voice: str | None = None, expressiveness: float = 0.5) -> list[Item]:
        path = self.speech_file(job, script, voice, expressiveness)
        return [self.library.add(path, "Voice", script, {"cloned": bool(voice)})]

    def sound_file(self, job, prompt: str, seconds: float) -> Path:
        if not prompt.strip():
            raise SpaceError("invalid", "Describe the sound or music.")
        total = max(1.0, float(seconds))
        n = math.ceil(total / config.SOUND_MAX_SECONDS)
        parts = []
        for i in range(n):
            length = int(min(config.SOUND_MAX_SECONDS, math.ceil(total - i * config.SOUND_MAX_SECONDS)))
            job.log(f"Sound {i + 1}/{n}: EzAudio, {length}s")
            r = self.caller("sound").call("/generate_audio", text=prompt, length=max(1, length), guidance_scale=5.0,
                                          guidance_rescale=0.75, ddim_steps=50, eta=1.0, random_seed=0,
                                          randomize_seed=True)
            parts.append(self._fetch_first(r, ".wav", "sound"))
        return media.concat_audio(parts, self._tmp(".wav")) if len(parts) > 1 else parts[0]

    def sound(self, job, prompt: str, seconds: float = 10) -> list[Item]:
        return [self.library.add(self.sound_file(job, prompt, seconds), "Sound", prompt, {"seconds": seconds})]

    def music_file(self, job, description: str, seconds: float = 60, instrumental: bool = False,
                   lyrics: str = "", seed: int | None = None) -> Path:
        """A song from MiniMax Music 3. Without lyrics the Space writes them (or none when
        instrumental); with lyrics it writes the style/vocals/arrangement around them."""
        if not description.strip() and not lyrics.strip():
            raise SpaceError("invalid", "Describe the music, or give lyrics.")
        step = 5
        secs = int(max(config.MUSIC_MIN_SECONDS, min(config.MUSIC_MAX_SECONDS, math.ceil(float(seconds) / step) * step)))
        s = presets.seed_for(seed, 0)
        base = {"description": description.strip(), "instrumental": bool(instrumental), "title": ""}
        gen = dict(duration=secs, seed=s, randomize_seed=False, headroom=0, steps=30, guidance=1.7)
        caller = self.caller("music")
        if lyrics.strip():
            job.log("MiniMax Music 3: writing the style, vocals and arrangement around your lyrics")
            r = caller.call("/compose_assist", raw_state={**base, "mode": "studio", "lyrics": lyrics,
                                                          "assist": "prompt", "assist_prompt": description.strip()},
                            duration=secs)
            state = r[0] if isinstance(r, (list, tuple)) else r
            if not isinstance(state, dict) or not state.get("global_meta"):
                raise SpaceError("failed", f"Could not prepare the song: {str(r)[:200]}", self.spaces["music"])
            state["lyrics"] = lyrics
            job.log(f"MiniMax Music 3: up to {secs}s with your lyrics, seed {s}")
            r = caller.call("/studio_generate", state=state, **gen)
        else:
            job.log(f"MiniMax Music 3: up to {secs}s{' instrumental' if instrumental else ''}, seed {s} "
                    "(it writes the lyrics first)")
            r = caller.call("/simple_generate", state={**base, "mode": "simple"}, **gen)
        return self._fetch_first(r, ".wav", "music")

    def music(self, job, description: str, seconds: float = 60, instrumental: bool = False, lyrics: str = "",
              seed: int | None = None) -> list[Item]:
        path = self.music_file(job, description, seconds, instrumental, lyrics, seed)
        title = "Music · instrumental" if instrumental else ("Song · your lyrics" if lyrics.strip() else "Song")
        return [self.library.add(path, title, description or lyrics[:200],
                                 {"model": "MiniMax Music 3", "seconds": seconds, "instrumental": instrumental})]

    def _with_sound(self, job, video: str | Path, prompt: str, mode: str = "replace", volume: float = 1.0,
                    engine: str = "effects") -> Path:
        dur = media.duration(video)
        if engine == "music":  # score the clip: instrumental music at least as long as the video
            audio = self.music_file(job, prompt, max(config.MUSIC_MIN_SECONDS, dur), instrumental=True)
        else:
            audio = self.sound_file(job, prompt, dur)
        job.log("Adding the sound to the video")
        return media.mux_audio(video, audio, self._tmp(".mp4"), mode=mode, new_volume=volume, old_volume=1.0)

    def add_sound(self, job, video: str, prompt: str, mode: str = "replace", volume: float = 1.0,
                  engine: str = "effects") -> list[Item]:
        if not video:
            raise SpaceError("invalid", "Pick a video.")
        out = self._with_sound(job, video, prompt, mode, volume, engine)
        title = "Video · with music" if engine == "music" else "Video · with sound"
        return [self.library.add(out, title, prompt, {"source": video, "mode": mode, "engine": engine})]

    # ------------------------------------------------------------ talking
    def talk(self, job, portrait: str, script: str = "", audio: str | None = None, voice: str | None = None,
             engine: str = "omni", motion: str = "", emotion: str = "Neutral", quality: str = "Standard",
             expressiveness: float = 0.5) -> list[Item]:
        if not portrait:
            raise SpaceError("invalid", "Add a portrait image.")
        speech = audio or str(self.speech_file(job, script, voice, expressiveness))
        if engine == "moda":
            job.log(f"MoDA talking head ({emotion})")
            padded = media.pad_audio(speech, 0.8, self._tmp(".wav"))  # MoDA trims the tail
            r = self.caller("avatar_fast").call("/generate_motion", source_image_path=self._handle(portrait),
                                                driving_audio_path=self._handle(padded), emotion_name=emotion,
                                                cfg_scale=1.2)
            video = self._finish_talk(job, self._fetch_first(r, ".mp4", "avatar_fast"), speech)
            return [self.library.add(video, "Talking · MoDA", script, {"engine": "MoDA", "portrait": portrait})]

        # OmniAvatar only uses the first 5 s of audio: longer speech is cut at
        # pauses into short parts, each part becomes a clip, and the clips are joined.
        steps = 12 if quality == "High" else 4
        caller = self.caller("avatar")
        try:  # the Space keeps per-session temp folders
            caller.client().predict(api_name="/start_session")
        except Exception:
            pass
        parts = media.split_audio(speech, config.OMNIAVATAR_MAX_AUDIO, self.tmp / "parts")
        text = motion.strip() or ("A realistic video of a person speaking naturally, moving their head "
                                  "and eyes, with subtle hand movements that match the speech.")
        clips = []
        for i, part in enumerate(parts):
            job.log(f"OmniAvatar part {i + 1}/{len(parts)} ({media.duration(part):.1f}s of speech, {steps} steps)")
            r = caller.call("/infer_scene", image_path=self._handle(portrait), audio_path=self._handle(part),
                            text=text, num_steps=steps)
            clips.append(media.fit_to_audio(self._fetch_first(r, ".mp4", "avatar"), part, self._tmp(".mp4")))
        if len(clips) > 1:
            job.log(f"Joining {len(clips)} parts")
        video = clips[0] if len(clips) == 1 else media.concat_videos(clips, self._tmp(".mp4"))
        return [self.library.add(video, "Talking · OmniAvatar", script,
                                 {"engine": "OmniAvatar", "portrait": portrait, "parts": len(clips)})]

    def _finish_talk(self, job, video: str | Path, speech: str | Path) -> Path:
        """Make sure every word is heard: hold the last frame if the model cut
        the clip short, then put the clean speech track on it."""
        need = media.duration(speech) + 0.3
        if media.duration(video) < need:
            job.log("The model ended early; holding the last frame so no words are cut")
            video = media.hold_last_frame(video, need, self._tmp(".mp4"))
        return media.mux_audio(video, speech, self._tmp(".mp4"), mode="replace")

    def lipsync(self, job, video: str, script: str = "", audio: str | None = None, voice: str | None = None,
                expressiveness: float = 0.5) -> list[Item]:
        if not video:
            raise SpaceError("invalid", "Add a video with a visible face.")
        speech = audio or str(self.speech_file(job, script, voice, expressiveness))
        audio = str(media.pad_audio(speech, 0.8, self._tmp(".wav")))
        vdur, adur = media.duration(video), media.duration(audio)
        if adur > vdur + 0.1:
            job.log(f"Video is shorter than the speech ({vdur:.1f}s < {adur:.1f}s): looping it")
            video = str(media.loop_video_to(video, adur, self._tmp(".mp4")))
        job.log("LatentSync lip-sync")
        r = self.caller("lipsync").call("/generate_lip_sync_video", input_video_path=self._handle(video),
                                        input_audio_path=self._handle(audio))
        out = self._finish_talk(job, self._fetch_first(r, ".mp4", "lipsync"), speech)
        return [self.library.add(out, "Lip-sync · LatentSync", script, {"source": video})]
