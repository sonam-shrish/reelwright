# Plan

Last updated: 2026-11-13. **Version 1 is built** (see README.md). The original plan follows; the README's "Known limits" section has what testing found.

## Decisions
- Personal use only, for a very small circle. Quality of results comes first.
- Open models only (no paid video APIs).
- Compute: Hugging Face Pro with ZeroGPU (about 40 min/day as stated by the owner; third-party sources say 25 min, check the account page). Hugging Face Jobs is a paid option later. Colab can only be used interactively from Claude Code, so it is not part of the automatic pipeline.
- Hosting: develop and preview in a Claude Code cloud session, then deploy a private Hugging Face Space on free CPU hardware (Gradio first). Whether free CPU Spaces allow Gradio/Docker on this account is not confirmed. Finished videos go to a permanent place (Drive or a private HF dataset), because Space disks are temporary.
- Safety: no videos of real people without their consent. Test faces so far were fictional and AI-generated.
- UI framework: Gradio for v1, with the generation logic in separate Python code so a different front end can be added later.

## Pipeline
1. Strong first frame (image model).
2. Image-to-video.
3. Several seeds, the owner picks the best.
4. Upscale and smooth (no good Space found; Real-ESRGAN and RIFE scripts are the fallback).
5. Sound: EzAudio for effects and music for now, Chatterbox for speech, lip-sync.

Camera control is prompt templates plus Qwen angle edits for now. No camera-control Space was found.

## Spaces that worked (call with `gradio_client` and `HF_TOKEN`)
| Task | Space | Endpoint | Test result |
|---|---|---|---|
| Image-to-video | `Lightricks/LTX-2-3` | `/generate_video(input_image, prompt, duration, enhance_prompt, seed, randomize_seed, height, width)` | about 30 s for 2 s at 512x768, 24 fps, has audio. The broccoli's face drifted in one test. |
| Image-to-video | `Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom` | `/generate_video(input_image, last_image=None, prompt, steps=4, duration_seconds=3.5, seed, ...)` | 29 s, 624x832 at 16 fps. Kept identity and scene well. `last_image=None` must be passed. |
| Image editing, camera angle | `prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast` | `/edit_image(images_b64_json, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps)` | 21 s. Adapter `Multiple-Angles` worked well. Image goes in as a JSON list of base64 data URIs. |
| Sound effects, music try | `OpenSound/EzAudio` | `/generate_audio(text, length, guidance_scale, guidance_rescale, ddim_steps, eta, random_seed, randomize_seed)` | 4 to 8 s for 10 s of audio. Owner says it sounds good. |
| Speech (TTS) | `ResembleAI/Chatterbox` | `/generate_tts_audio(text_input, audio_prompt_path_input, ...)` | 6 s. Pass `audio_prompt_path_input=None`. Voice cloning untested. |
| Lip-sync, image + audio | `alexnasa/OmniAvatar` | `/infer_scene(image_path, audio_path, text, num_steps)` | 97 s, 400x720. **First choice**, rated great by the owner. |
| Lip-sync, image + audio | `multimodalart/MoDA-fast-talking-head` | `/generate_motion(source_image_path, driving_audio_path, emotion_name, cfg_scale)` | 17 s, 832x1248. Not rated. |
| Lip-sync, existing video | `fffiloni/LatentSync` | `/generate_lip_sync_video(input_video_path, input_audio_path)` | 40 s, 560x832, output cut to 3.2 s. Rated fine. Backup. |

## Checked and dropped
- `amisima/Wan_2.2_I2V_Studio`: dozens of UI-only endpoints, no clean generation API.
- `Wan-AI/Wan2.2-S2V`, `fffiloni/Meigen-MultiTalk`: only on basic CPU hardware.
- `Skywork/skyreels-a1-talking-head`: error state.
- Not found: camera-motion Space, good FLUX Space, good upscaler or frame-smoothing Space.

## Lessons for the job runner
- Every Space needs its exact `*.hf.space` host allowed in the sandbox (get it from `https://huggingface.co/api/spaces/<owner>/<name>`, field `host`).
- LTX was flaky: early attempts failed within 3 s, and downloading the finished video returned 403 once (possibly several replicas). Use retries and a fallback download of the file URL.
- `/on_image_upload` on LTX returns `{'value': n}` objects, not numbers.
- No ZeroGPU quota errors appeared in testing, but the remaining quota cannot be read. Record the time of every call.

## Version 1 scope (proposed, not confirmed)
1. First-frame image, then image-to-video (Wan Lightning and LTX).
2. Camera presets as prompt templates, plus Qwen angle edits for keyframes.
3. Lip-sync with OmniAvatar, with Chatterbox for the speech.
4. Sound effects with EzAudio.

Later: character consistency with reference images, upscaling and smoothing, a dedicated music Space (name pending, calendar reminder set for 2026-11-14), a paid GPU backend if ZeroGPU runs short.

## Open questions
- Project name: **Reelwright** (chosen 2026-11-13).
- GitHub repo: not created yet. The session cannot create repos, so the owner creates an empty one and it gets attached.
- Whether to start building v1.
