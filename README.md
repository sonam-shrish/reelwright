---
title: Reelwright
emoji: 🎞️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Personal AI film studio on open models
---

# Reelwright

A personal AI film studio built only from open models that run on Hugging Face Spaces. Personal use, for a very small circle. No videos of real people without their consent.

## What it can do

| Tab | What you get | Model (Space) |
|---|---|---|
| 🖼️ Image | Images from a prompt, 11 looks, 6 aspect ratios, up to 2K | Z-Image-Turbo |
| 🎬 Video | Image-to-video or text-to-video, 25 camera moves (stack up to 3), 1-10 s, up to 4 variants, optional sound, **Extend** a clip from its last frame | Wan 2.2 Lightning, LTX-2.3 |
| ✏️ Edit & Angles | Edit with words (with extra reference images), 9 camera-angle changes, 10 one-click effects (upscale to 2K, relight, unblur, ...) | Qwen-Image-Edit-2511 (+ LoRAs) |
| 🧑 Character | Keep one character consistent across new scenes from 1-3 references | Qwen-Image-Edit-2511 |
| 🗣️ Talking | A portrait speaks a script (or your audio), optional voice cloning; lip-sync an existing video | OmniAvatar, MoDA, LatentSync, Chatterbox |
| 🔊 Sound | Sound effects and music (any length), voice only, add sound to a video (replace or mix) | EzAudio, Chatterbox |
| 📚 Library | Everything you made, details, download, send to other tabs, join clips into one sequence | ffmpeg |
| ⏱️ Jobs | Every generation runs in the background; status, errors and an estimate of today's GPU use | |

Results flow between tabs: an image can be animated, edited, used as a character or made to talk with one click.

## Run it

```bash
pip install -r requirements.txt   # and ffmpeg
export HF_TOKEN=hf_...            # your Hugging Face token (ZeroGPU quota is used per token)
python app.py                     # http://localhost:7860
```

Settings (environment variables): `REELWRIGHT_DATA` (where results are kept, default `~/reelwright-data`), `REELWRIGHT_PASSWORD` (adds a login), `REELWRIGHT_GPU_BUDGET_MIN` (daily budget shown in the app), `REELWRIGHT_SPACE_<NAME>` (swap a model Space, e.g. `REELWRIGHT_SPACE_WAN`).

## Video on your own Colab GPU (inside a Claude Code session)
Wan 2.2 (14B, Lightning, 4 steps) can run on your Google Colab compute units instead of ZeroGPU.

```bash
bash scripts/colab_setup.sh                 # once per container: installs the Colab CLI
bash scripts/colab_signin.sh                # once per container: prints a Google link
bash scripts/colab_signin.sh <CODE>         # paste the code shown after approving
python scripts/colab_video.py --image start.png --prompt "she turns and smiles" --moves "Dolly In" --seconds 4
python scripts/colab_video.py stop          # stop the runtime (it also stops after 15 idle minutes)
```

In the app, pick the model "Wan 2.2 on my Colab GPU" in the Video tab. Results go to the Library like any other clip.
Tested 2026-11-14 on an A100 80 GB: first clip per runtime downloads the 69 GB model (about 3.5 min) and loads it (25 s); each 4 s clip (832x480, 65 frames) then takes about 40 s. Settings: `REELWRIGHT_COLAB_GPU` (default A100), `REELWRIGHT_COLAB_WAN_MODEL`, `REELWRIGHT_COLAB_IDLE_STOP_MIN`.
It only works where the Colab sign-in lives (a Claude session or your own machine), not from a hosted Space.

## Deploy as a private Space
Create a private Gradio Space on free CPU hardware, push this repo to it, and add `HF_TOKEN` as a Space secret. The UI itself uses no GPU; only the model calls use your ZeroGPU quota. A Space's disk is temporary, so download what you want to keep.

## Tests
```bash
python -m pytest -q                         # 40 unit tests, no GPU, real ffmpeg
python scripts/smoke_live.py face.png       # every feature against the real Spaces (uses ~8 min of GPU)
```

## Known limits (tested 2026-11-13)
- Quality is limited by the open models. Higgsfield runs paid models (Kling, Veo, Seedance); open models do not match them on every shot. Camera moves are written into the prompt; none of the open Spaces offers true camera-path control.
- Free ZeroGPU Spaces have queues and daily quotas. OmniAvatar costs about 1.5 GPU-minutes per 5 s of speech; Qwen edits about 40 s; Wan clips about 20-30 s.
- OmniAvatar only takes 5 s of audio per call, so longer speech is split at pauses and joined (a small cut can be visible between parts). Its output is 400x720.
- Several Spaces run on more than one replica and lose uploaded files when a job lands on another replica. Reelwright uploads over the job's own connection, retries fast failures with a fresh upload, and falls back to inline images.
