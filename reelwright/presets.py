"""Camera moves, looks, aspect ratios, angles and effects, and prompt building."""
from __future__ import annotations

import random
import re

# Camera moves are written into the video prompt. Up to 3 can be stacked.
CAMERA_MOVES: dict[str, str] = {
    "Static": "locked-off static camera, no camera movement",
    "Dolly In": "slow smooth dolly-in, the camera glides forward toward the subject",
    "Dolly Out": "smooth dolly-out, the camera glides backward away from the subject, revealing the scene",
    "Push-in Close-up": "slow push-in ending on an intimate close-up of the face",
    "Crash Zoom In": "sudden fast crash zoom into the subject",
    "Crash Zoom Out": "sudden fast crash zoom out from the subject",
    "Orbit Left": "the camera orbits smoothly around the subject from right to left in a wide arc",
    "Orbit Right": "the camera orbits smoothly around the subject from left to right in a wide arc",
    "360 Orbit": "the camera circles a full 360 degrees around the subject",
    "Crane Up": "the camera cranes up smoothly, rising high above the subject",
    "Crane Down": "the camera cranes down from high above toward the subject",
    "Tilt Up": "the camera tilts up slowly",
    "Tilt Down": "the camera tilts down slowly",
    "Pan Left": "the camera pans slowly to the left",
    "Pan Right": "the camera pans slowly to the right",
    "Whip Pan": "a fast whip pan with strong motion blur",
    "Tracking Shot": "the camera tracks alongside the moving subject at the same speed",
    "Handheld": "handheld camera with natural subtle shake, documentary feel",
    "FPV Drone": "fast FPV drone shot swooping and diving through the scene",
    "Aerial Pullback": "aerial drone shot pulling back and rising to reveal the whole landscape",
    "Bullet Time": "bullet time: the action freezes in mid-air while the camera sweeps around the subject",
    "Dutch Roll": "the camera slowly rolls into a tilted dutch angle",
    "Low Angle Hero": "low-angle hero shot looking up at the subject as the camera slowly rises",
    "Snorricam": "body-mounted camera locked on the subject's face while the background moves",
    "Rack Focus": "rack focus shifting from the foreground to the subject",
}
SHAKY_MOVES = {"Handheld", "FPV Drone", "Whip Pan", "Crash Zoom In", "Crash Zoom Out"}

LOOKS: dict[str, str] = {
    "None": "",
    "Cinematic": "cinematic film still, anamorphic lens, dramatic lighting, rich color grading, shallow depth of field",
    "Photoreal": "ultra-realistic photograph, natural light, 35mm lens, sharp fine detail",
    "Fashion Editorial": "high-fashion editorial photograph, studio strobe lighting, magazine cover quality",
    "Vintage Film": "1970s 35mm film photo, natural film grain, warm faded colors",
    "Film Noir": "black and white film noir, hard shadows, venetian blind light",
    "Anime": "high-quality anime illustration, clean line art, vibrant colors",
    "3D Animation": "stylized 3D animated film still, soft global illumination, expressive characters",
    "Product Shot": "studio product photography, softbox lighting, seamless clean background",
    "Fantasy": "epic fantasy concept art, volumetric light, highly detailed",
    "Cyberpunk": "neon cyberpunk night, rain-soaked streets, reflections, moody haze",
}

ASPECTS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
_ZIMAGE = {
    "Standard": {"16:9": "1536x864 ( 16:9 )", "9:16": "864x1536 ( 9:16 )", "1:1": "1280x1280 ( 1:1 )",
                 "4:3": "1472x1104 ( 4:3 )", "3:4": "1104x1472 ( 3:4 )", "21:9": "1680x720 ( 21:9 )"},
    "High": {"16:9": "2048x1152 ( 16:9 )", "9:16": "1152x2048 ( 9:16 )", "1:1": "1536x1536 ( 1:1 )",
             "4:3": "1728x1296 ( 4:3 )", "3:4": "1296x1728 ( 3:4 )", "21:9": "2016x864 ( 21:9 )"},
}
_EDIT_DIMS = {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (1024, 1024),
              "4:3": (1152, 864), "3:4": (864, 1152), "21:9": (1536, 656)}

# Camera-angle changes for a still image (Qwen "Multiple-Angles" adapter, which
# was trained on Chinese instructions, so both languages are sent).
ANGLES: dict[str, str] = {
    "Rotate left 45°": "Rotate the camera 45 degrees to the left. 将镜头向左旋转45度。",
    "Rotate right 45°": "Rotate the camera 45 degrees to the right. 将镜头向右旋转45度。",
    "Top-down view": "Turn the camera to a top-down view. 将镜头转为俯视。",
    "Wide-angle": "Turn the camera to a wide-angle lens. 将镜头转为广角镜头。",
    "Close-up": "Turn the camera to a close-up. 将镜头转为特写镜头。",
    "Move forward": "Move the camera forward. 将镜头向前移动。",
    "Move left": "Move the camera to the left. 将镜头向左移动。",
    "Move right": "Move the camera to the right. 将镜头向右移动。",
    "Move down": "Move the camera down. 将镜头向下移动。",
}

# One-click effects: (adapter on the LoRA Space, default instruction).
EFFECTS: dict[str, tuple[str, str]] = {
    "Upscale to 2K": ("Upscaler", "Upscale this image to 2K resolution with sharp, clean detail."),
    "Relight": ("Any-light", "Relight the scene with soft golden-hour light coming from the left."),
    "Flat light (de-light)": ("Studio-DeLight", "Remove the strong lighting and shadows; soft, even, neutral light."),
    "Unblur": ("Unblur-Anything", "Remove the blur and make the image sharp and clear."),
    "Ultra-realistic portrait": ("Ultra-Realistic-Portrait", "Make this portrait ultra realistic with natural skin texture."),
    "Anything to real": ("Anything2Real", "Make this look like a real photograph."),
    "Anime": ("Photo-to-Anime", "Transform this image into anime style."),
    "Polaroid": ("Polaroid-Photo", "Turn this into a vintage polaroid photo."),
    "Noir comic": ("Noir-Comic-Book", "Turn this into a noir comic book panel."),
    "Cinematic flat log": ("Cinematic-FlatLog", "Convert to a cinematic flat log color profile."),
}

VIDEO_MODELS = {
    "Wan 2.2 (best consistency)": "wan",
    "Wan 2.2 on my Colab GPU (no ZeroGPU quota)": "wan_colab",
    "LTX-2.3 (adds its own audio)": "ltx",
}
MODEL_LABELS = {"wan": "Wan 2.2", "wan_colab": "Wan 2.2 · Colab", "ltx": "LTX-2.3"}
TALK_ENGINES = {
    "OmniAvatar (best, slower)": "omni",
    "MoDA (fast)": "moda",
}
EMOTIONS = ["Neutral", "Happiness", "Sadness", "Surprise", "Anger", "Fear", "Disgust", "Contempt"]

_WAN_NEGATIVE = "blurry, low quality, chaotic, deformed, watermark, bad anatomy, distorted face, flicker"


def seed_for(seed: int | None, i: int) -> int:
    if seed is None or seed < 0:
        return random.randint(0, 2**31 - 1)
    return int(seed) + i


def styled(prompt: str, look: str) -> str:
    extra = LOOKS.get(look, "")
    prompt = prompt.strip()
    return f"{prompt}. {extra}" if extra else prompt


def video_prompt(prompt: str, moves: list[str] | tuple = ()) -> str:
    moves = [m for m in (moves or []) if m in CAMERA_MOVES][:3]
    base = prompt.strip() or "The scene comes alive with natural, realistic motion"
    if moves:
        cam = ", then ".join(CAMERA_MOVES[m] for m in moves)
        base = f"{base}. Camera: {cam}"
    return f"{base}. Cinematic, smooth motion, high detail, consistent characters."


def wan_negative(moves: list[str] | tuple = ()) -> str:
    neg = _WAN_NEGATIVE
    if not any(m in SHAKY_MOVES for m in (moves or [])):
        neg += ", shaky camera view point"
    if moves and "Static" not in moves:
        neg += ", static frozen frame"
    return neg


def image_resolution(aspect: str, quality: str = "Standard") -> str:
    return _ZIMAGE.get(quality, _ZIMAGE["Standard"]).get(aspect, _ZIMAGE["Standard"]["16:9"])


def edit_dims(aspect: str | None, source_size: tuple[int, int] | None = None, target_pixels: int = 1024 * 1024) -> tuple[int, int]:
    """(width, height) for the Qwen editor: an aspect preset, or the source image's shape."""
    if aspect in _EDIT_DIMS:
        return _EDIT_DIMS[aspect]
    if source_size:
        w, h = source_size
        scale = (target_pixels / (w * h)) ** 0.5
        return max(256, int(w * scale) // 16 * 16), max(256, int(h * scale) // 16 * 16)
    return _EDIT_DIMS["1:1"]


_LTX_SIZES = {"Standard": {"16:9": (768, 512), "9:16": (512, 768), "1:1": (768, 768)},
              "High": {"16:9": (1536, 1024), "9:16": (1024, 1536), "1:1": (1024, 1024)}}


def ltx_dims(size: tuple[int, int], quality: str = "Standard") -> tuple[int, int]:
    """(width, height) for LTX-2.3, using the Space's own table and nearest-aspect rule."""
    w, h = size
    ratio = w / h
    aspect = min({"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}.items(), key=lambda kv: abs(ratio - kv[1]))[0]
    return _LTX_SIZES.get(quality, _LTX_SIZES["Standard"])[aspect]


def split_text(text: str, max_chars: int) -> list[str]:
    """Split a script into chunks at sentence boundaries, each <= max_chars."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        while len(s) > max_chars:  # very long sentence: cut at a space
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


# IndexTTS-2 emotion sliders, in the Space's order (vec1..vec8).
TTS_EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def emotion_vector(emotion: dict | str | None) -> list[float]:
    """{"sad": 0.6, "calm": 0.3} or "sad" -> the 8 IndexTTS-2 sliders (each 0..1, total kept <= 1.2)."""
    if not emotion:
        return [0.0] * 7 + [0.6]
    if isinstance(emotion, str):
        emotion = {emotion: 0.7}
    vec = [max(0.0, min(1.0, float(emotion.get(k, 0.0)))) for k in TTS_EMOTIONS]
    unknown = set(emotion) - set(TTS_EMOTIONS)
    if unknown:
        raise ValueError(f"unknown emotion(s) {sorted(unknown)}; use {TTS_EMOTIONS}")
    total = sum(vec)
    if total > 1.2:
        vec = [v * 1.2 / total for v in vec]
    return vec
