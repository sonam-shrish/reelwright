"""Settings: which Spaces Reelwright calls and where it keeps files.

Every Space can be swapped with an environment variable, e.g.
REELWRIGHT_SPACE_WAN=someone/other-wan-space
"""
import os
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_DIR = Path(os.environ.get("REELWRIGHT_DATA", Path.home() / "reelwright-data"))
# Daily ZeroGPU budget shown in the app (owner's Pro plan; check the account page).
DAILY_GPU_BUDGET_MIN = float(os.environ.get("REELWRIGHT_GPU_BUDGET_MIN", "40"))
# Optional password for the web app (useful if it is ever hosted publicly).
APP_PASSWORD = os.environ.get("REELWRIGHT_PASSWORD")

_DEFAULT_SPACES = {
    "image": "Tongyi-MAI/Z-Image-Turbo",
    "wan": "Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom",
    "ltx": "Lightricks/LTX-2-3",
    "edit": "Qwen/Qwen-Image-Edit-2511",
    "edit_lora": "prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast",
    "tts": "ResembleAI/Chatterbox",
    "sound": "OpenSound/EzAudio",
    "avatar": "alexnasa/OmniAvatar",
    "avatar_fast": "multimodalart/MoDA-fast-talking-head",
    "lipsync": "fffiloni/LatentSync",
}

SPACES = {k: os.environ.get(f"REELWRIGHT_SPACE_{k.upper()}", v) for k, v in _DEFAULT_SPACES.items()}

# Google Colab backend (only inside a Claude Code session; see reelwright/colab.py)
COLAB_GPU = os.environ.get("REELWRIGHT_COLAB_GPU", "A100")
COLAB_SESSION = os.environ.get("REELWRIGHT_COLAB_SESSION", "reelwright")
COLAB_WAN_MODEL = os.environ.get("REELWRIGHT_COLAB_WAN_MODEL", "TestOrganizationPleaseIgnore/WAMU_v3_WAN2.2_I2V_LIGHTNING")
COLAB_IDLE_STOP_MIN = float(os.environ.get("REELWRIGHT_COLAB_IDLE_STOP_MIN", "15"))

# Spaces that need the official gradio_client (per-session state); all others use REST.
GRADIO_CLIENT_SPACES = {SPACES["avatar"]}

# Limits read from the Spaces' own sliders and code (2026-11-13).
OMNIAVATAR_MAX_AUDIO = 4.8  # the Space keeps only the first 5 s of audio
VIDEO_MAX_SECONDS = 10.0
SOUND_MAX_SECONDS = 10
TTS_MAX_CHARS = 280
