"""Run every Reelwright feature once against the real Spaces (uses GPU quota).

    HF_TOKEN=... python scripts/smoke_live.py [face_image.png]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reelwright import media  # noqa: E402
from reelwright.jobs import Job  # noqa: E402
from reelwright.studio import Studio  # noqa: E402

studio = Studio()
face = sys.argv[1] if len(sys.argv) > 1 else None
results, state = {}, {}


def step(name, fn):
    job = Job(id=name, title=name)
    t = time.time()
    try:
        items = fn(job)
        info = []
        for it in items:
            d = {"kind": it.kind, "path": it.path}
            if it.kind in ("video", "audio"):
                p = media.probe(it.path)
                d.update(seconds=round(p["duration"], 2), audio=p["has_audio"], size=f"{p['width']}x{p['height']}")
            info.append(d)
        results[name] = {"ok": True, "secs": round(time.time() - t), "items": info}
        state[name] = [it.path for it in items]
    except Exception as e:  # noqa: BLE001
        results[name] = {"ok": False, "secs": round(time.time() - t), "error": f"{type(e).__name__}: {e}"[:400],
                         "log": job.logs[-3:]}
    print(name, json.dumps(results[name]), flush=True)


step("image", lambda j: studio.create_images(j, "A lighthouse keeper in a yellow raincoat on a stormy cliff at dusk",
                                             "Cinematic", "16:9", 1, 11))
img = (state.get("image") or [None])[0]
step("video_wan_sound", lambda j: studio.create_video(j, img, "The keeper looks out at the crashing waves, rain blowing",
                                                       ["Orbit Left"], "wan", 3, 1, 3, sound="stormy wind and crashing waves"))
step("video_ltx", lambda j: studio.create_video(j, img, "Waves crash against the rocks below", ["Crane Up"], "ltx", 2, 1, 4))
wan = (state.get("video_wan_sound") or [None])[0]
step("extend", lambda j: studio.extend_video(j, wan, "He turns and walks back toward the lighthouse", ["Tracking Shot"], "wan", 2))
step("stitch", lambda j: studio.stitch(j, [p for p in [wan, (state.get("video_ltx") or [None])[0]] if p]))
step("edit", lambda j: studio.edit_image(j, [img], "Make it a clear starry night with the lighthouse beam on"))
step("angle", lambda j: studio.change_angle(j, img, "Close-up"))
step("effect_upscale", lambda j: studio.apply_effect(j, img, "Upscale to 2K"))
step("sound_12s", lambda j: studio.sound(j, "calm cinematic piano with soft strings", 12))
step("voice", lambda j: studio.speech(j, "Every storm passes. The light stays on. That is the job."))
if face:
    step("character", lambda j: studio.character_scene(j, [face], "standing on the lighthouse balcony in the rain, holding a lantern", "9:16"))
    step("talk_omni", lambda j: studio.talk(j, face, "Every storm passes. The light stays on.", engine="omni"))
    step("talk_moda", lambda j: studio.talk(j, face, "Every storm passes. The light stays on.", engine="moda", emotion="Happiness"))
    talk = (state.get("talk_moda") or [None])[0]
    step("lipsync", lambda j: studio.lipsync(j, talk, "And tomorrow, the sea will be calm again."))
step("add_sound_mix", lambda j: studio.add_sound(j, (state.get("video_ltx") or [None])[0], "seagulls calling", "mix", 0.8))

ok = sum(r["ok"] for r in results.values())
print(f"\nSUMMARY {ok}/{len(results)} passed; GPU-call time today: {studio.usage.today()['total_seconds'] / 60:.1f} min")
