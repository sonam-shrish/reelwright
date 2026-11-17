# Reelwright: standing instructions from the owner

Read this before making any video.

## Creative rules
- **Videos must not look AI-generated.** Make real short films: mixed shot sizes
  (wide, medium, close-up, inserts, cutaways), shorter cuts, motivated camera,
  ordinary-looking people and real locations, no glossy "AI" polish.
  Finish with a film look in post (grain, grade, halation, gentle vignette).
- **Use the whole toolset every time:** designed voices with emotion (Qwen3-TTS voice
  design and clone, IndexTTS-2 emotion), dialogue or voice-over, ambience and foley
  (EzAudio), a music score (MiniMax Music 3). Subtitles, not big meme captions.
- **Be original.** Every new series must be completely new and different from the
  earlier ones (no slogan reels).
- Consent rule: no real people without their consent. Fictional people only.

## Speed setup (video)
- 3 Colab A100 runtimes in parallel, SageAttention, Wan 2.2 at 8 steps.
- Render at 720p class (`--scale 1.5`) and upscale to 1080x1920 with an AI upscaler.
- Always stop Colab runtimes when done (they cost compute units while on).

## Publishing
- Instagram account `_babelground` via Buffer; reels are marked AI-generated.
- **Never delete Buffer posts unless the owner clearly says so. When unsure, ask.**
- Keep finished videos in Google Drive (Reelwright/Films, copied via a Colab Drive
  mount, which needs the owner's sign-in click) and in the public HF dataset
  `sonam-shrish/reelwright-media` (`scripts/host_reels.py`).
- Never use `GITHUB_TOKEN`.
