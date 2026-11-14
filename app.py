"""Reelwright: a personal AI video studio built on open models (Hugging Face Spaces).

Run locally:  HF_TOKEN=... python app.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr

from reelwright import config, presets
from reelwright.jobs import JobQueue
from reelwright.studio import Studio

studio = Studio()
queue = JobQueue(studio.data, workers=int(os.environ.get("REELWRIGHT_WORKERS", "2")))

CONSENT_TEXT = "I have the consent of every real person shown (or the person is fictional)."

# ------------------------------------------------------------------ helpers


def _seed(x) -> int | None:
    try:
        x = int(x)
    except (TypeError, ValueError):
        return None
    return None if x < 0 else x


def _fmt_secs(s: int) -> str:
    return f"{s // 60}:{s % 60:02d}"


def _status(job) -> str:
    last = job.logs[-1].split("  ", 1)[-1] if job.logs else ""
    if job.status == "queued":
        return f"⏳ **Queued** · {last}"
    if job.status == "running":
        return f"⚙️ **Working** · {_fmt_secs(job.seconds())} · {last}"
    if job.status == "done":
        return f"✅ **Done** in {_fmt_secs(job.seconds())} · saved to the Library"
    return f"❌ **Failed** · {job.error}"


def _gallery(paths):
    return [(p, Path(p).stem[-6:]) for p in paths]


def _follow(job):
    """Yield (status, gallery, paths) while a job runs, then its results."""
    while job.active:
        yield _status(job), gr.skip(), gr.skip()
        time.sleep(1.0)
    paths = [i.path for i in job.items if i.kind in ("image", "video")]
    yield _status(job), (_gallery(paths) if paths else gr.skip()), (paths if paths else gr.skip())


def _follow_audio(job):
    while job.active:
        yield _status(job), gr.skip()
        time.sleep(1.0)
    audio = next((i.path for i in job.items if i.kind == "audio"), None)
    yield _status(job), (audio if audio else gr.skip())


def _error(msg, n=3):
    return (f"❌ {msg}",) + tuple(gr.skip() for _ in range(n - 1))


def _pick(paths, evt: gr.SelectData):
    if paths and evt.index is not None and evt.index < len(paths):
        return paths[evt.index]
    return None


def _chosen(selected, paths):
    return selected or (paths[0] if paths else None)


def _usage_md() -> str:
    u = studio.usage.today()
    mins = u["total_seconds"] / 60
    return (f"**GPU calls today:** about {mins:.1f} of {config.DAILY_GPU_BUDGET_MIN:.0f} min "
            f"(estimate from call times; it overestimates)")


def _files(files) -> list[str]:
    out = []
    for f in files or []:
        out.append(f if isinstance(f, str) else getattr(f, "name", None) or f.get("path"))
    return [p for p in out if p]


# ------------------------------------------------------------------ actions

def run_images(prompt, look, aspect, count, quality, seed):
    if not (prompt or "").strip():
        yield _error("Describe the image first.")
        return
    job = queue.submit(f"Image · {prompt[:40]}", lambda j: studio.create_images(
        j, prompt, look, aspect, int(count), _seed(seed), quality))
    yield from _follow(job)


def run_video(image, prompt, moves, model_label, seconds, variants, quality, aspect, look, sound, seed):
    if not image and not (prompt or "").strip():
        yield _error("Add a start image or describe the shot.")
        return
    model = presets.VIDEO_MODELS[model_label]
    job = queue.submit(f"Video · {prompt[:40] or 'from image'}", lambda j: studio.create_video(
        j, image, prompt or "", moves or [], model, float(seconds), int(variants), _seed(seed), quality,
        aspect, look, sound or ""))
    yield from _follow(job)


def run_extend(selected, paths, prompt, moves, model_label, seconds, quality, seed):
    video = _chosen(selected, paths)
    if not video or not video.endswith(".mp4"):
        yield _error("Generate or pick a video first, then press Extend.")
        return
    job = queue.submit("Extend video", lambda j: studio.extend_video(
        j, video, prompt or "", moves or [], presets.VIDEO_MODELS[model_label], float(seconds), quality, _seed(seed)))
    yield from _follow(job)


def run_edit(image, extra, prompt, aspect, quality, seed):
    images = [image] + _files(extra) if image else _files(extra)
    if not images:
        yield _error("Add an image to edit.")
        return
    job = queue.submit(f"Edit · {prompt[:40]}", lambda j: studio.edit_image(
        j, images, prompt or "", aspect, quality, _seed(seed)))
    yield from _follow(job)


def run_angle(image, angle, seed):
    if not image:
        yield _error("Add an image first.")
        return
    job = queue.submit(f"Angle · {angle}", lambda j: studio.change_angle(j, image, angle, _seed(seed)))
    yield from _follow(job)


def run_effect(image, effect, prompt, seed):
    if not image:
        yield _error("Add an image first.")
        return
    job = queue.submit(f"Effect · {effect}", lambda j: studio.apply_effect(j, image, effect, prompt or "", _seed(seed)))
    yield from _follow(job)


def run_character(r1, r2, r3, scene, aspect, quality, consent, seed):
    refs = [r for r in (r1, r2, r3) if r]
    if not refs:
        yield _error("Add at least one reference image.")
        return
    if not consent:
        yield _error("Please confirm consent first.")
        return
    if not (scene or "").strip():
        yield _error("Describe the new scene.")
        return
    job = queue.submit(f"Character · {scene[:40]}", lambda j: studio.character_scene(
        j, refs, scene, aspect, quality, _seed(seed)))
    yield from _follow(job)


def run_talk(portrait, script, audio, voice, engine_label, emotion, motion, quality, expressiveness, consent):
    if not portrait:
        yield _error("Add a portrait.")
        return
    if not consent:
        yield _error("Please confirm consent first.")
        return
    if not audio and not (script or "").strip():
        yield _error("Write a script or upload speech audio.")
        return
    engine = presets.TALK_ENGINES[engine_label]
    job = queue.submit(f"Talking · {(script or 'audio')[:40]}", lambda j: studio.talk(
        j, portrait, script or "", audio, voice, engine, motion or "", emotion, quality, float(expressiveness)))
    yield from _follow(job)


def run_lipsync(video, script, audio, voice, expressiveness, consent):
    if not video:
        yield _error("Add a video with a visible face.")
        return
    if not consent:
        yield _error("Please confirm consent first.")
        return
    if not audio and not (script or "").strip():
        yield _error("Write a script or upload speech audio.")
        return
    job = queue.submit("Lip-sync video", lambda j: studio.lipsync(j, video, script or "", audio, voice, float(expressiveness)))
    yield from _follow(job)


def run_sound(prompt, seconds):
    if not (prompt or "").strip():
        yield f"❌ Describe the sound or music.", gr.skip()
        return
    job = queue.submit(f"Sound · {prompt[:40]}", lambda j: studio.sound(j, prompt, float(seconds)))
    yield from _follow_audio(job)


def run_voice(script, voice, expressiveness):
    if not (script or "").strip():
        yield f"❌ Write the words to speak.", gr.skip()
        return
    job = queue.submit(f"Voice · {script[:40]}", lambda j: studio.speech(j, script, voice, float(expressiveness)))
    yield from _follow_audio(job)


def run_add_sound(video, prompt, mode, volume):
    if not video:
        yield _error("Add a video.")
        return
    if not (prompt or "").strip():
        yield _error("Describe the sound or music.")
        return
    job = queue.submit(f"Add sound · {prompt[:40]}", lambda j: studio.add_sound(
        j, video, prompt, "mix" if mode.startswith("Mix") else "replace", float(volume)))
    yield from _follow(job)


def run_stitch(labels):
    paths = [lbl.split(" | ")[-1] for lbl in (labels or [])]
    if len(paths) < 2:
        yield _error("Pick at least two clips, in the order you want them.")
        return
    job = queue.submit(f"Join {len(paths)} clips", lambda j: studio.stitch(j, paths))
    yield from _follow(job)


# ------------------------------------------------------------------ library & jobs

def library_view(kind_label):
    kind = {"Images": "image", "Videos": "video", "Sounds": "audio"}.get(kind_label)
    items = studio.library.items(kind)
    visual = [i for i in items if i.kind in ("image", "video")]
    audio = [i for i in items if i.kind == "audio"]
    paths = [i.path for i in visual]
    audio_choices = [f"{i.caption()} | {i.path}" for i in audio]
    video_choices = [f"{i.caption()} | {i.path}" for i in studio.library.items("video")]
    return ([(i.path, i.caption()) for i in visual], paths,
            gr.update(choices=audio_choices, value=None),
            gr.update(choices=video_choices, value=[]))


def library_select(paths, evt: gr.SelectData):
    path = _pick(paths, evt)
    item = studio.library.find_by_path(path) if path else None
    if not item:
        return None, "", None
    meta = json.dumps(item.meta, indent=1, default=str)[:1500]
    info = f"**{item.title}** · {item.created.replace('T', ' ')}\n\n{item.prompt or '_no prompt_'}\n\n```json\n{meta}\n```"
    return path, info, path


def audio_pick(label):
    return label.split(" | ")[-1] if label else None


def jobs_table():
    rows = [[j.title, j.status, _fmt_secs(j.seconds()), (j.error or (j.logs[-1].split('  ', 1)[-1] if j.logs else ""))[:160]]
            for j in queue.list()[:50]]
    return rows or [["(no jobs yet)", "", "", ""]], _usage_md()


# ------------------------------------------------------------------ MCP tools
# These are the only public API endpoints, so they are what MCP clients see as
# tools. Each one runs as a normal background job (visible in the Jobs tab) and
# waits for it. Files can be passed as URLs or base64 and come back as files.

MCP_TOOLS: set[str] = set()


def _wait(title: str, fn) -> str:
    """Run as a background job, wait, and return the result file (one output, so MCP
    clients get a proper file/image back rather than a text dump)."""
    job = queue.submit(f"MCP · {title}", fn)
    while job.active:
        time.sleep(1.0)
    if job.status != "done" or not job.items:
        raise gr.Error(job.error or "Nothing was produced.")
    return job.items[0].path


def _moves(text: str) -> list[str]:
    moves = [m.strip() for m in (text or "").split(",") if m.strip()]
    bad = [m for m in moves if m not in presets.CAMERA_MOVES]
    if bad:
        raise gr.Error(f"Unknown camera moves {bad}. Choose up to 3 of: {', '.join(presets.CAMERA_MOVES)}")
    return moves[:3]


def create_image(prompt: str, look: str = "Cinematic", aspect: str = "16:9", quality: str = "Standard",
                 seed: int = -1):
    """Create an image from a text prompt with Z-Image-Turbo.

    Args:
        prompt: What the image shows.
        look: Visual style. One of: None, Cinematic, Photoreal, Fashion Editorial, Vintage Film, Film Noir, Anime, 3D Animation, Product Shot, Fantasy, Cyberpunk.
        aspect: Aspect ratio: 16:9, 9:16, 1:1, 4:3, 3:4 or 21:9.
        quality: Standard (about 1.3 megapixels) or High (up to 2K).
        seed: Random seed; -1 for random.
    """
    return _wait(f"Image · {prompt[:30]}", lambda j: studio.create_images(j, prompt, look, aspect, 1, _seed(seed), quality))


def create_video(prompt: str, start_image: str | None = None, camera_moves: str = "Dolly In",
                 model: str = "wan", seconds: float = 4.0, quality: str = "Standard", sound: str = "",
                 seed: int = -1):
    """Create a short video. Animates start_image, or first creates a start frame from the prompt.

    Args:
        prompt: What happens in the shot.
        start_image: Optional start frame (URL or file). Without it, a first frame is generated from the prompt.
        camera_moves: Up to 3 camera moves, comma separated, applied in order, e.g. "Dolly In, Orbit Left". Options: Static, Dolly In, Dolly Out, Push-in Close-up, Crash Zoom In, Crash Zoom Out, Orbit Left, Orbit Right, 360 Orbit, Crane Up, Crane Down, Tilt Up, Tilt Down, Pan Left, Pan Right, Whip Pan, Tracking Shot, Handheld, FPV Drone, Aerial Pullback, Bullet Time, Dutch Roll, Low Angle Hero, Snorricam, Rack Focus.
        model: "wan" (Wan 2.2, best consistency), "ltx" (LTX-2.3, adds its own audio) or "wan_colab" (Wan 2.2 on your Colab GPU; only when the app runs in a signed-in Claude session).
        seconds: Length, 1 to 10 seconds.
        quality: Standard or High (more steps, slower).
        sound: Optional description of sound or music to add.
        seed: Random seed; -1 for random.
    """
    if model not in presets.MODEL_LABELS:
        raise gr.Error("model must be wan, ltx or wan_colab")
    moves = _moves(camera_moves)
    return _wait(f"Video · {prompt[:30]}", lambda j: studio.create_video(
        j, start_image, prompt, moves, model, float(seconds), 1, _seed(seed), quality, sound=sound or ""))


def extend_video(video: str, prompt: str, camera_moves: str = "", model: str = "wan", seconds: float = 4.0):
    """Continue a video from its last frame and return the joined, longer clip.

    Args:
        video: The video to continue (URL or file).
        prompt: What happens next.
        camera_moves: Optional camera moves, comma separated (see create_video).
        model: "wan", "ltx" or "wan_colab".
        seconds: Length of the new part, 1 to 10 seconds.
    """
    moves = _moves(camera_moves)
    return _wait("Extend video", lambda j: studio.extend_video(j, video, prompt, moves, model, float(seconds)))


def edit_image(image: str, instruction: str, reference_image: str | None = None, aspect: str = "Keep",
               quality: str = "Standard"):
    """Edit an image with words (Qwen-Image-Edit-2511). An optional second image can be referenced, e.g. an outfit or product.

    Args:
        image: The image to edit (URL or file).
        instruction: The change to make, e.g. "make it night with neon signs".
        reference_image: Optional second image to use in the edit.
        aspect: Keep, 16:9, 9:16, 1:1, 4:3, 3:4 or 21:9.
        quality: Standard or High.
    """
    images = [image] + ([reference_image] if reference_image else [])
    return _wait(f"Edit · {instruction[:30]}", lambda j: studio.edit_image(j, images, instruction, aspect, quality))


def change_camera_angle(image: str, angle: str = "Rotate left 45°"):
    """Show the same scene from a different camera angle.

    Args:
        image: The image (URL or file).
        angle: One of: Rotate left 45°, Rotate right 45°, Top-down view, Wide-angle, Close-up, Move forward, Move left, Move right, Move down.
    """
    if angle not in presets.ANGLES:
        raise gr.Error(f"angle must be one of: {', '.join(presets.ANGLES)}")
    return _wait(f"Angle · {angle}", lambda j: studio.change_angle(j, image, angle))


def apply_effect(image: str, effect: str = "Upscale to 2K", instruction: str = ""):
    """Apply a one-click effect to an image.

    Args:
        image: The image (URL or file).
        effect: One of: Upscale to 2K, Relight, Flat light (de-light), Unblur, Ultra-realistic portrait, Anything to real, Anime, Polaroid, Noir comic, Cinematic flat log.
        instruction: Optional instruction; a sensible default is used if empty.
    """
    if effect not in presets.EFFECTS:
        raise gr.Error(f"effect must be one of: {', '.join(presets.EFFECTS)}")
    return _wait(f"Effect · {effect}", lambda j: studio.apply_effect(j, image, effect, instruction or ""))


def character_scene(reference_image: str, scene: str, aspect: str = "9:16", consent_confirmed: bool = False,
                    reference_image_2: str | None = None):
    """Put the same character into a new scene, keeping face, hair, body and outfit consistent.

    Args:
        reference_image: A clear image of the character (URL or file).
        scene: The new scene, e.g. "walking on a rainy street at night".
        aspect: 16:9, 9:16, 1:1, 4:3, 3:4 or 21:9.
        consent_confirmed: Must be true: every real person shown has consented (or the person is fictional).
        reference_image_2: Optional second reference of the same character.
    """
    if not consent_confirmed:
        raise gr.Error("Set consent_confirmed=true: every real person shown must have consented (or be fictional).")
    refs = [reference_image] + ([reference_image_2] if reference_image_2 else [])
    return _wait(f"Character · {scene[:30]}", lambda j: studio.character_scene(j, refs, scene, aspect))


def talking_video(portrait: str, script: str, consent_confirmed: bool = False, engine: str = "omni",
                  voice_sample: str | None = None, body_language: str = ""):
    """Make a portrait speak a script (text-to-speech plus talking-head video).

    Args:
        portrait: Image with a clearly visible face (URL or file).
        script: The words to speak.
        consent_confirmed: Must be true: the person has consented (or is fictional).
        engine: "omni" (OmniAvatar: best, slower; long speech is split into parts under 5 s) or "moda" (MoDA: fast).
        voice_sample: Optional 5-15 s audio sample of a voice to clone.
        body_language: Optional, OmniAvatar only, e.g. "nods, small hand gestures".
    """
    if not consent_confirmed:
        raise gr.Error("Set consent_confirmed=true: the person must have consented (or be fictional).")
    if engine not in ("omni", "moda"):
        raise gr.Error("engine must be omni or moda")
    return _wait(f"Talking · {script[:30]}", lambda j: studio.talk(j, portrait, script, None, voice_sample, engine,
                                                                  body_language or ""))


def text_to_speech(script: str, voice_sample: str | None = None, expressiveness: float = 0.5):
    """Turn text into speech with Chatterbox, optionally cloning a voice.

    Args:
        script: The words to speak (long scripts are split and joined).
        voice_sample: Optional 5-15 s audio sample of a voice to clone.
        expressiveness: 0.25 (flat) to 2.0 (very expressive); 0.5 is neutral.
    """
    return _wait(f"Voice · {script[:30]}", lambda j: studio.speech(j, script, voice_sample, float(expressiveness)))


def generate_sound(prompt: str, seconds: float = 10):
    """Generate sound effects, ambience or music with EzAudio (any length; made in 10 s parts).

    Args:
        prompt: Describe the sound or music, e.g. "calm cinematic piano with soft strings".
        seconds: Length in seconds.
    """
    return _wait(f"Sound · {prompt[:30]}", lambda j: studio.sound(j, prompt, float(seconds)))


def add_sound_to_video(video: str, prompt: str, mode: str = "replace", volume: float = 1.0):
    """Generate sound or music that fits a video and put it on the video.

    Args:
        video: The video (URL or file).
        prompt: Describe the sound or music.
        mode: "replace" (drop existing sound) or "mix" (blend with existing sound).
        volume: Volume of the new sound, 0.1 to 2.0.
    """
    return _wait(f"Add sound · {prompt[:30]}", lambda j: studio.add_sound(
        j, video, prompt, "mix" if mode == "mix" else "replace", float(volume)))


# ------------------------------------------------------------------ UI

CSS = """
#rw-header h1 {font-size: 2.1rem; margin-bottom: 0; letter-spacing: .02em}
#rw-header p {opacity: .75; margin-top: .2rem}
.rw-status {min-height: 2.2rem}
footer {display: none !important}
"""

with gr.Blocks(title="Reelwright") as demo:
    with gr.Row(elem_id="rw-header"):
        with gr.Column(scale=3):
            gr.Markdown("# 🎞️ Reelwright\nYour personal AI film studio: images, camera moves, characters, talking scenes and sound, all from open models.")
        with gr.Column(scale=2):
            usage_md = gr.Markdown(_usage_md())

    with gr.Tabs() as tabs:
        # ---------------------------------------------------------- IMAGE
        with gr.Tab("🖼️ Image", id="image"):
            with gr.Row():
                with gr.Column(scale=2):
                    img_prompt = gr.Textbox(label="Describe the image", lines=4,
                                            placeholder="A lone astronaut in a sunflower field at golden hour, wind in the flowers")
                    with gr.Row():
                        img_look = gr.Dropdown(list(presets.LOOKS), value="Cinematic", label="Look")
                        img_aspect = gr.Radio(presets.ASPECTS, value="16:9", label="Aspect")
                    with gr.Row():
                        img_count = gr.Slider(1, 4, value=2, step=1, label="How many")
                        img_quality = gr.Radio(["Standard", "High"], value="Standard", label="Resolution")
                        img_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    img_go = gr.Button("Generate images", variant="primary")
                    gr.Examples([["A rain-soaked neon alley in Tokyo at night, a woman in a yellow raincoat walking away"],
                                 ["Macro shot of a hummingbird drinking from a red flower, morning dew"],
                                 ["A vintage red sports car parked on a cliff road above the sea at sunset"]],
                                inputs=[img_prompt], label="Ideas")
                with gr.Column(scale=3):
                    img_status = gr.Markdown(elem_classes="rw-status")
                    img_gallery = gr.Gallery(label="Results", columns=2, height=520, object_fit="contain")
                    with gr.Row():
                        img_to_video = gr.Button("🎬 Animate")
                        img_to_edit = gr.Button("✏️ Edit / angles")
                        img_to_char = gr.Button("🧑 Use as character")
                        img_to_talk = gr.Button("🗣️ Make it talk")
            img_paths, img_sel = gr.State([]), gr.State(None)

        # ---------------------------------------------------------- VIDEO
        with gr.Tab("🎬 Video", id="video"):
            with gr.Row():
                with gr.Column(scale=2):
                    vid_image = gr.Image(type="filepath", label="Start frame (optional: without one, it is created from the prompt)", height=240)
                    vid_prompt = gr.Textbox(label="What happens in the shot", lines=3,
                                            placeholder="She turns toward the camera and smiles as the wind lifts her hair")
                    vid_moves = gr.Dropdown(list(presets.CAMERA_MOVES), multiselect=True, max_choices=3, value=["Dolly In"],
                                            label="Camera moves (up to 3, applied in order)")
                    with gr.Row():
                        vid_model = gr.Radio(list(presets.VIDEO_MODELS), value=list(presets.VIDEO_MODELS)[0], label="Model")
                        vid_quality = gr.Radio(["Standard", "High"], value="Standard", label="Quality")
                    with gr.Row():
                        vid_seconds = gr.Slider(1, config.VIDEO_MAX_SECONDS, value=4, step=0.5, label="Length (seconds)")
                        vid_variants = gr.Slider(1, 4, value=1, step=1, label="Variants")
                    vid_sound = gr.Textbox(label="Sound (optional)", placeholder="gentle wind, distant waves, soft piano")
                    with gr.Accordion("More", open=False):
                        with gr.Row():
                            vid_look = gr.Dropdown(list(presets.LOOKS), value="Cinematic", label="Look of the created first frame")
                            vid_aspect = gr.Radio(presets.ASPECTS, value="16:9", label="Aspect of the created first frame")
                        vid_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    vid_go = gr.Button("Generate video", variant="primary")
                    with gr.Accordion("Colab GPU (for the 'on my Colab GPU' model)", open=False):
                        gr.Markdown("Runs Wan 2.2 on your own Colab A100. Only works while this app runs inside a "
                                    "Claude Code session that is signed in to Colab. The first clip per runtime "
                                    "downloads the model (several minutes); later clips are fast. The runtime "
                                    "stops itself after 15 idle minutes.")
                        colab_status = gr.Markdown()
                        with gr.Row():
                            colab_check = gr.Button("Check Colab")
                            colab_stop = gr.Button("Stop Colab runtime", variant="stop")
                with gr.Column(scale=3):
                    vid_status = gr.Markdown(elem_classes="rw-status")
                    vid_gallery = gr.Gallery(label="Results", columns=2, height=520, object_fit="contain")
                    with gr.Row():
                        vid_extend = gr.Button("➕ Extend selected (uses the prompt and moves on the left)")
                        vid_to_sound = gr.Button("🔊 Add sound")
                        vid_to_lipsync = gr.Button("👄 Lip-sync")
            vid_paths, vid_sel = gr.State([]), gr.State(None)

        # ---------------------------------------------------------- EDIT
        with gr.Tab("✏️ Edit & Angles", id="edit"):
            with gr.Row():
                with gr.Column(scale=2):
                    ed_image = gr.Image(type="filepath", label="Image", height=300)
                    with gr.Tabs():
                        with gr.Tab("Edit with words"):
                            ed_extra = gr.File(file_count="multiple", file_types=["image"],
                                               label="Extra reference images (optional, e.g. a product or an outfit)")
                            ed_prompt = gr.Textbox(label="What should change", lines=3,
                                                   placeholder="Change the background to a snowy mountain village at dusk")
                            with gr.Row():
                                ed_aspect = gr.Dropdown(["Keep"] + presets.ASPECTS, value="Keep", label="Output shape")
                                ed_quality = gr.Radio(["Standard", "High"], value="Standard", label="Quality")
                            ed_go = gr.Button("Apply edit", variant="primary")
                        with gr.Tab("Camera angle"):
                            ed_angle = gr.Radio(list(presets.ANGLES), value="Rotate left 45°", label="New angle")
                            ed_angle_go = gr.Button("Change angle", variant="primary")
                        with gr.Tab("Effects"):
                            ed_effect = gr.Dropdown(list(presets.EFFECTS), value="Upscale to 2K", label="Effect")
                            ed_effect_prompt = gr.Textbox(label="Instruction (optional, a default is used)")
                            ed_effect_go = gr.Button("Apply effect", variant="primary")
                    ed_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                with gr.Column(scale=3):
                    ed_status = gr.Markdown(elem_classes="rw-status")
                    ed_gallery = gr.Gallery(label="Results", columns=2, height=520, object_fit="contain")
                    with gr.Row():
                        ed_again = gr.Button("↩️ Keep editing this one")
                        ed_to_video = gr.Button("🎬 Animate")
            ed_paths, ed_sel = gr.State([]), gr.State(None)

        # ---------------------------------------------------------- CHARACTER
        with gr.Tab("🧑 Character", id="character"):
            gr.Markdown("Keep one character consistent across scenes. Add 1 to 3 clear photos or renders of the same character, then describe a new scene.")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        ch_r1 = gr.Image(type="filepath", label="Reference 1", height=200)
                        ch_r2 = gr.Image(type="filepath", label="Reference 2 (optional)", height=200)
                        ch_r3 = gr.Image(type="filepath", label="Reference 3 (optional)", height=200)
                    ch_scene = gr.Textbox(label="New scene", lines=3,
                                          placeholder="Standing on a rooftop at night in a long black coat, city lights behind")
                    with gr.Row():
                        ch_aspect = gr.Radio(presets.ASPECTS, value="9:16", label="Aspect")
                        ch_quality = gr.Radio(["Standard", "High"], value="Standard", label="Quality")
                    ch_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    ch_consent = gr.Checkbox(label=CONSENT_TEXT)
                    ch_go = gr.Button("Create scene", variant="primary")
                with gr.Column(scale=3):
                    ch_status = gr.Markdown(elem_classes="rw-status")
                    ch_gallery = gr.Gallery(label="Results", columns=2, height=520, object_fit="contain")
                    with gr.Row():
                        ch_to_video = gr.Button("🎬 Animate")
                        ch_to_talk = gr.Button("🗣️ Make it talk")
            ch_paths, ch_sel = gr.State([]), gr.State(None)

        # ---------------------------------------------------------- TALK
        with gr.Tab("🗣️ Talking", id="talk"):
            with gr.Row():
                with gr.Column(scale=2):
                    tk_portrait = gr.Image(type="filepath", label="Portrait (face clearly visible)", height=260)
                    tk_script = gr.Textbox(label="Script", lines=4, placeholder="Hi! Welcome to my channel. Today I want to show you something special.")
                    with gr.Accordion("Use your own audio or clone a voice", open=False):
                        tk_audio = gr.Audio(type="filepath", label="Speech audio (replaces the script)")
                        tk_voice = gr.Audio(type="filepath", label="Voice sample to clone (5-15 s, optional)")
                        tk_expr = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Expressiveness of the voice")
                    with gr.Row():
                        tk_engine = gr.Radio(list(presets.TALK_ENGINES), value=list(presets.TALK_ENGINES)[0], label="Engine")
                        tk_quality = gr.Radio(["Standard", "High"], value="Standard", label="Quality (OmniAvatar)")
                    with gr.Row():
                        tk_motion = gr.Textbox(label="Body language (OmniAvatar)", placeholder="speaks warmly, nods, small hand gestures")
                        tk_emotion = gr.Dropdown(presets.EMOTIONS, value="Neutral", label="Emotion (MoDA)")
                    tk_consent = gr.Checkbox(label=CONSENT_TEXT)
                    tk_go = gr.Button("Make it talk", variant="primary")
                    with gr.Accordion("Lip-sync an existing video instead (LatentSync)", open=False):
                        ls_video = gr.Video(label="Video with a visible face")
                        ls_go = gr.Button("Lip-sync this video (uses the script, audio and voice above)")
                with gr.Column(scale=3):
                    tk_status = gr.Markdown(elem_classes="rw-status")
                    tk_gallery = gr.Gallery(label="Results", columns=1, height=560, object_fit="contain")
            tk_paths, tk_sel = gr.State([]), gr.State(None)

        # ---------------------------------------------------------- SOUND
        with gr.Tab("🔊 Sound", id="sound"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Sound effects and music")
                    sd_prompt = gr.Textbox(label="Describe it", lines=2, placeholder="calm cinematic piano with soft strings")
                    sd_seconds = gr.Slider(1, 60, value=10, step=1, label="Length (seconds)")
                    sd_go = gr.Button("Generate sound", variant="primary")
                    sd_status = gr.Markdown(elem_classes="rw-status")
                    sd_audio = gr.Audio(label="Result", type="filepath")
                    gr.Markdown("### Voice only")
                    vo_script = gr.Textbox(label="Words to speak", lines=3)
                    vo_voice = gr.Audio(type="filepath", label="Voice sample to clone (optional)")
                    vo_expr = gr.Slider(0.25, 2.0, value=0.5, step=0.05, label="Expressiveness")
                    vo_go = gr.Button("Generate voice")
                    vo_status = gr.Markdown(elem_classes="rw-status")
                    vo_audio = gr.Audio(label="Voice", type="filepath")
                with gr.Column():
                    gr.Markdown("### Add sound to a video")
                    as_video = gr.Video(label="Video")
                    as_prompt = gr.Textbox(label="Sound or music", placeholder="footsteps on marble, a clock ticking, quiet room tone")
                    with gr.Row():
                        as_mode = gr.Radio(["Replace the sound", "Mix with existing sound"], value="Replace the sound", label="Mode")
                        as_volume = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Volume of the new sound")
                    as_go = gr.Button("Add sound", variant="primary")
                    as_status = gr.Markdown(elem_classes="rw-status")
                    as_gallery = gr.Gallery(label="Result", columns=1, height=420, object_fit="contain")
            as_paths = gr.State([])

        # ---------------------------------------------------------- LIBRARY
        with gr.Tab("📚 Library", id="library"):
            with gr.Row():
                lib_kind = gr.Radio(["All", "Images", "Videos", "Sounds"], value="All", label="Show")
                lib_refresh = gr.Button("Refresh")
            with gr.Row():
                with gr.Column(scale=3):
                    lib_gallery = gr.Gallery(label="Everything you made", columns=4, height=560, object_fit="contain")
                with gr.Column(scale=2):
                    lib_info = gr.Markdown("Select something to see its details.")
                    lib_file = gr.File(label="Download")
                    with gr.Row():
                        lib_to_video = gr.Button("🎬 Animate")
                        lib_to_edit = gr.Button("✏️ Edit")
                        lib_to_talk = gr.Button("🗣️ Talk")
                        lib_to_sound = gr.Button("🔊 Add sound")
                    lib_audio_pick = gr.Dropdown([], label="Sounds and voices")
                    lib_audio = gr.Audio(label="Play", type="filepath")
            gr.Markdown("### Join clips into one sequence")
            with gr.Row():
                st_clips = gr.Dropdown([], multiselect=True, label="Clips, in order")
                st_go = gr.Button("Join clips", variant="primary")
            st_status = gr.Markdown(elem_classes="rw-status")
            st_gallery = gr.Gallery(label="Sequence", columns=1, height=420, object_fit="contain")
            lib_paths, lib_sel, st_paths = gr.State([]), gr.State(None), gr.State([])

        # ---------------------------------------------------------- JOBS
        with gr.Tab("⏱️ Jobs", id="jobs"):
            gr.Markdown("Every generation runs in the background. You can start several and switch tabs; results land in the Library.")
            jobs_df = gr.Dataframe(headers=["Job", "Status", "Time", "Details"], interactive=False, wrap=True)
            jobs_usage = gr.Markdown()

    # ------------------------------------------------------------- wiring
    def go_to(tab_id):
        return gr.Tabs(selected=tab_id)

    def wire(button, fn, inputs, status, gallery, paths):
        button.click(fn, inputs, [status, gallery, paths])

    wire(img_go, run_images, [img_prompt, img_look, img_aspect, img_count, img_quality, img_seed], img_status, img_gallery, img_paths)
    wire(vid_go, run_video, [vid_image, vid_prompt, vid_moves, vid_model, vid_seconds, vid_variants, vid_quality,
                             vid_aspect, vid_look, vid_sound, vid_seed], vid_status, vid_gallery, vid_paths)
    wire(vid_extend, run_extend, [vid_sel, vid_paths, vid_prompt, vid_moves, vid_model, vid_seconds, vid_quality, vid_seed],
         vid_status, vid_gallery, vid_paths)
    wire(ed_go, run_edit, [ed_image, ed_extra, ed_prompt, ed_aspect, ed_quality, ed_seed], ed_status, ed_gallery, ed_paths)
    wire(ed_angle_go, run_angle, [ed_image, ed_angle, ed_seed], ed_status, ed_gallery, ed_paths)
    wire(ed_effect_go, run_effect, [ed_image, ed_effect, ed_effect_prompt, ed_seed], ed_status, ed_gallery, ed_paths)
    wire(ch_go, run_character, [ch_r1, ch_r2, ch_r3, ch_scene, ch_aspect, ch_quality, ch_consent, ch_seed],
         ch_status, ch_gallery, ch_paths)
    wire(tk_go, run_talk, [tk_portrait, tk_script, tk_audio, tk_voice, tk_engine, tk_emotion, tk_motion, tk_quality,
                           tk_expr, tk_consent], tk_status, tk_gallery, tk_paths)
    wire(ls_go, run_lipsync, [ls_video, tk_script, tk_audio, tk_voice, tk_expr, tk_consent], tk_status, tk_gallery, tk_paths)
    wire(as_go, run_add_sound, [as_video, as_prompt, as_mode, as_volume], as_status, as_gallery, as_paths)
    wire(st_go, run_stitch, [st_clips], st_status, st_gallery, st_paths)
    colab_check.click(lambda: studio.colab.status(), None, [colab_status])
    colab_stop.click(lambda: studio.colab.stop(), None, [colab_status])
    sd_go.click(run_sound, [sd_prompt, sd_seconds], [sd_status, sd_audio])
    vo_go.click(run_voice, [vo_script, vo_voice, vo_expr], [vo_status, vo_audio])

    for gal, paths, sel in [(img_gallery, img_paths, img_sel), (vid_gallery, vid_paths, vid_sel),
                            (ed_gallery, ed_paths, ed_sel), (ch_gallery, ch_paths, ch_sel), (tk_gallery, tk_paths, tk_sel)]:
        gal.select(_pick, [paths], [sel])

    def send(button, sel, paths, target, tab_id):
        button.click(lambda s, p: (_chosen(s, p), go_to(tab_id)), [sel, paths], [target, tabs])

    send(img_to_video, img_sel, img_paths, vid_image, "video")
    send(img_to_edit, img_sel, img_paths, ed_image, "edit")
    send(img_to_char, img_sel, img_paths, ch_r1, "character")
    send(img_to_talk, img_sel, img_paths, tk_portrait, "talk")
    send(vid_to_sound, vid_sel, vid_paths, as_video, "sound")
    send(vid_to_lipsync, vid_sel, vid_paths, ls_video, "talk")
    send(ed_to_video, ed_sel, ed_paths, vid_image, "video")
    send(ed_again, ed_sel, ed_paths, ed_image, "edit")
    send(ch_to_video, ch_sel, ch_paths, vid_image, "video")
    send(ch_to_talk, ch_sel, ch_paths, tk_portrait, "talk")

    lib_outputs = [lib_gallery, lib_paths, lib_audio_pick, st_clips]
    lib_refresh.click(library_view, [lib_kind], lib_outputs)
    lib_kind.change(library_view, [lib_kind], lib_outputs)
    lib_gallery.select(library_select, [lib_paths], [lib_sel, lib_info, lib_file])
    lib_audio_pick.change(audio_pick, [lib_audio_pick], [lib_audio])

    def lib_send(button, target, tab_id, kinds):
        def fn(path):
            if not path or Path(path).suffix.lower() not in kinds:
                return gr.skip(), gr.skip()
            return path, go_to(tab_id)
        button.click(fn, [lib_sel], [target, tabs])

    lib_send(lib_to_video, vid_image, "video", {".png", ".jpg", ".jpeg", ".webp"})
    lib_send(lib_to_edit, ed_image, "edit", {".png", ".jpg", ".jpeg", ".webp"})
    lib_send(lib_to_talk, tk_portrait, "talk", {".png", ".jpg", ".jpeg", ".webp"})
    lib_send(lib_to_sound, as_video, "sound", {".mp4", ".webm", ".mov"})

    timer = gr.Timer(3)
    timer.tick(jobs_table, None, [jobs_df, jobs_usage])
    timer.tick(_usage_md, None, [usage_md])
    demo.load(library_view, [lib_kind], lib_outputs)
    demo.load(jobs_table, None, [jobs_df, jobs_usage])

    # ------------------------------------------------------------- MCP / API endpoints (no UI)
    with gr.Column(visible=False):
        quality = lambda: gr.Dropdown(["Standard", "High"], value="Standard")  # noqa: E731
        models = lambda: gr.Dropdown(list(presets.MODEL_LABELS), value="wan")  # noqa: E731
        img = lambda: gr.Image(type="filepath")  # noqa: E731
        vid = lambda: gr.Video()  # noqa: E731
        aud = lambda: gr.Audio(type="filepath")  # noqa: E731
        endpoints = [
            (create_image, [gr.Textbox(), gr.Dropdown(list(presets.LOOKS), value="Cinematic"),
                            gr.Dropdown(presets.ASPECTS, value="16:9"), quality(), gr.Number(-1, precision=0)], img()),
            (create_video, [gr.Textbox(), img(), gr.Textbox("Dolly In"), models(), gr.Number(4.0), quality(),
                            gr.Textbox(""), gr.Number(-1, precision=0)], vid()),
            (extend_video, [vid(), gr.Textbox(), gr.Textbox(""), models(), gr.Number(4.0)], vid()),
            (edit_image, [img(), gr.Textbox(), img(), gr.Dropdown(["Keep"] + presets.ASPECTS, value="Keep"), quality()], img()),
            (change_camera_angle, [img(), gr.Dropdown(list(presets.ANGLES), value="Rotate left 45°")], img()),
            (apply_effect, [img(), gr.Dropdown(list(presets.EFFECTS), value="Upscale to 2K"), gr.Textbox("")], img()),
            (character_scene, [img(), gr.Textbox(), gr.Dropdown(presets.ASPECTS, value="9:16"), gr.Checkbox(False), img()], img()),
            (talking_video, [img(), gr.Textbox(), gr.Checkbox(False), gr.Dropdown(["omni", "moda"], value="omni"), aud(),
                             gr.Textbox("")], vid()),
            (text_to_speech, [gr.Textbox(), aud(), gr.Number(0.5)], aud()),
            (generate_sound, [gr.Textbox(), gr.Number(10)], aud()),
            (add_sound_to_video, [vid(), gr.Textbox(), gr.Dropdown(["replace", "mix"], value="replace"), gr.Number(1.0)], vid()),
        ]
        for fn, inputs, output in endpoints:
            trigger = gr.Button(visible=False)
            trigger.click(fn, inputs, output, api_name=fn.__name__)
            MCP_TOOLS.add(fn.__name__)

    # Only the tools above are public; the UI's own events stay private (not MCP tools).
    for block_fn in demo.fns.values():
        if (block_fn.api_name or "").lstrip("/") not in MCP_TOOLS:
            block_fn.api_visibility = "private"


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[str(config.DATA_DIR), str(studio.data)],
        theme=gr.themes.Soft(primary_hue="amber", neutral_hue="zinc"),
        css=CSS,
        auth=("reelwright", config.APP_PASSWORD) if config.APP_PASSWORD else None,
        mcp_server=True,  # tools at /gradio_api/mcp/ (see README)
    )
