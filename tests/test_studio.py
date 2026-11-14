import json

import pytest

from reelwright import media, presets
from reelwright.jobs import Job, JobQueue
from reelwright.spaces import SpaceError, to_data_uri
from reelwright.studio import Studio


class FakeCaller:
    def __init__(self, key, files, calls, overrides):
        self.key, self.f, self.calls, self.overrides = key, files, calls, overrides

    def client(self):
        return self

    def predict(self, api_name, **kw):  # used for OmniAvatar's /start_session
        self.calls.append((self.key, api_name, kw))

    def call(self, api_name, **kw):
        self.calls.append((self.key, api_name, kw))
        if (self.key, api_name) in self.overrides:
            return self.overrides[(self.key, api_name)](kw)
        f = self.f
        return {
            ("image", "/generate"): ([{"image": {"path": f["img"], "url": None}, "caption": None}], "1", 1),
            ("wan", "/generate_video"): ({"video": {"path": f["vid_silent"]}}, {"path": f["vid_silent"]}, 5),
            ("ltx", "/on_image_upload"): ({"value": 512, "__type__": "update"}, {"value": 768, "__type__": "update"}),
            ("ltx", "/generate_video"): ({"path": f["vid_audio"]}, 5),
            ("edit", "/infer"): ([{"image": {"path": f["img"]}, "caption": None}], 1),
            ("edit_lora", "/edit_image"): {"image": to_data_uri(f["img"]), "seed": 1, "status": "success"},
            ("tts", "/generate_tts_audio"): {"path": f["wav"]},
            ("voice_design", "/generate_voice_design"): {"path": f["wav"]},
            ("voice_design", "/generate_voice_clone"): {"path": f["wav"]},
            ("sound", "/generate_audio"): {"path": f["wav"]},
            # MiniMax returns a player-status dict with a relative url first, then the real FileData
            ("music", "/simple_generate"): ({"cmd": "done", "url": "/gradio_api/file=/tmp/x.wav"}, "done",
                                            {"path": f["long_wav"], "meta": {"_type": "gradio.FileData"}}, 7, None, None),
            ("music", "/studio_generate"): ({"cmd": "done", "url": "/gradio_api/file=/tmp/x.wav"}, "done",
                                            {"path": f["long_wav"], "meta": {"_type": "gradio.FileData"}}, 7, None),
            ("music", "/compose_assist"): ({"mode": "studio", "lyrics": "model lyrics", "global_meta": "Basic Attributes: ...",
                                            "vocals": "v", "arrangement": "a"}, "ready"),
            ("avatar", "/infer_scene"): {"path": f["vid_silent"]},
            ("avatar_fast", "/generate_motion"): {"video": {"path": f["vid_audio"]}, "subtitles": None},
            ("lipsync", "/generate_lip_sync_video"): {"path": f["vid_silent"]},
        }[(self.key, api_name)]


@pytest.fixture
def studio(tmp_path, media_files):
    calls, overrides = [], {}
    spaces = {k: k for k in ["image", "wan", "ltx", "edit", "edit_lora", "tts", "voice_design", "sound", "music",
                             "avatar", "avatar_fast", "lipsync"]}
    s = Studio(data_dir=tmp_path / "data", token="t", spaces=spaces,
               caller_factory=lambda key: FakeCaller(key, media_files, calls, overrides),
               handle=lambda p: {"path": str(p)})
    s.calls, s.overrides = calls, overrides
    return s


def job():
    return Job(id="j", title="t")


def apis(studio):
    return [(k, a) for k, a, _ in studio.calls]


def test_create_images_uses_seeds_and_saves(studio):
    items = studio.create_images(job(), "a red fox", look="Cinematic", aspect="9:16", count=2, seed=5)
    assert [i.kind for i in items] == ["image", "image"]
    seeds = [kw["seed"] for k, a, kw in studio.calls if a == "/generate"]
    assert seeds == [5, 6]
    kw = studio.calls[0][2]
    assert kw["resolution"] == "864x1536 ( 9:16 )" and "cinematic" in kw["prompt"].lower()
    assert len(studio.library.items()) == 2


def test_text_to_video_creates_first_frame_then_video_with_sound(studio):
    items = studio.create_video(job(), None, "a fox runs", moves=["Dolly In", "Orbit Left"],
                                model="wan", seconds=30, variants=1, seed=1, sound="forest wind")
    assert apis(studio)[:2] == [("image", "/generate"), ("wan", "/generate_video")]
    wan_kw = next(kw for k, a, kw in studio.calls if k == "wan")
    assert wan_kw["duration_seconds"] == 10.0  # clamped to the Space's limit
    assert "dolly-in" in wan_kw["prompt"] and "orbits" in wan_kw["prompt"]
    assert wan_kw["last_image"] is None
    assert wan_kw["frame_multiplier"] == 16  # a number: the Space rejects the string "16"
    info = media.probe(items[0].path)
    assert items[0].kind == "video" and info["has_audio"]


def test_ltx_uses_the_spaces_size_table(studio, media_files):
    studio.create_video(job(), media_files["img"], "waves", model="ltx", seconds=3, variants=2, seed=9)
    gens = [kw for k, a, kw in studio.calls if k == "ltx" and a == "/generate_video"]
    assert not any(a == "/on_image_upload" for k, a in apis(studio))
    assert len(gens) == 2 and (gens[0]["width"], gens[0]["height"]) == (768, 512)  # 640x360 is 16:9
    assert [g["seed"] for g in gens] == [9, 10]


def test_extend_joins_clips(studio, media_files):
    item = studio.extend_video(job(), media_files["vid_audio"], "continue", model="wan", seconds=2)[0]
    assert media.duration(item.path) == pytest.approx(2 + 1.5, abs=0.3)
    assert media.probe(item.path)["has_audio"]


def test_stitch_needs_two_and_keeps_audio(studio, media_files):
    with pytest.raises(SpaceError):
        studio.stitch(job(), [media_files["vid_audio"]])
    item = studio.stitch(job(), [media_files["vid_audio"], media_files["vid_silent"]])[0]
    assert media.probe(item.path)["has_audio"]
    assert media.duration(item.path) == pytest.approx(3.5, abs=0.3)


def test_character_scene_prompt_and_size(studio, media_files):
    studio.character_scene(job(), [media_files["img"]], "walking on a beach", aspect="9:16")
    kw = next(kw for k, a, kw in studio.calls if a == "/infer")
    assert "same character" in kw["prompt"] and (kw["width"], kw["height"]) == (768, 1344)
    assert kw["rewrite_prompt"] is False and len(kw["images"]) == 1


def test_edit_keeps_source_aspect(studio, media_files):
    studio.edit_image(job(), [media_files["img"]], "make it night")
    kw = next(kw for k, a, kw in studio.calls if a == "/infer")
    assert kw["width"] > kw["height"] and kw["width"] % 16 == 0 and kw["height"] % 16 == 0


def test_angle_and_effect_send_base64(studio, media_files):
    item = studio.change_angle(job(), media_files["img"], "Rotate left 45°")[0]
    kw = studio.calls[-1][2]
    assert kw["lora_adapter"] == "Multiple-Angles" and json.loads(kw["images_b64_json"])[0].startswith("data:image/png")
    assert item.kind == "image"
    studio.apply_effect(job(), media_files["img"], "Upscale to 2K")
    assert studio.calls[-1][2]["lora_adapter"] == "Upscaler"


def test_lora_refusal_is_reported(studio, media_files):
    studio.overrides[("edit_lora", "/edit_image")] = lambda kw: {"status": "blocked", "message": "unsafe", "image": ""}
    with pytest.raises(SpaceError) as e:
        studio.change_angle(job(), media_files["img"], "Close-up")
    assert e.value.kind == "blocked"


def test_designed_voice_then_clone_for_more_lines(studio, media_files):
    first = studio.designed_speech_file(job(), "Look up.", "a warm young woman, gentle, hopeful")
    studio.cloned_speech_file(job(), "Someone is right there.", str(first), "Look up.")
    design = next(kw for k, a, kw in studio.calls if a == "/generate_voice_design")
    clone = next(kw for k, a, kw in studio.calls if a == "/generate_voice_clone")
    assert design["voice_description"].startswith("a warm") and design["language"] == "English"
    assert clone["ref_text"] == "Look up." and clone["target_text"] == "Someone is right there."
    assert clone["ref_audio"]["path"] == str(first)


def test_long_script_is_split_and_joined(studio):
    script = " ".join(["This is one fairly long sentence about the scene."] * 15)
    item = studio.speech(job(), script)[0]
    n = len([1 for k, a, kw in studio.calls if k == "tts"])
    assert n == len(presets.split_text(script, 280)) and n >= 3
    assert media.duration(item.path) == pytest.approx(n * 1.0, abs=0.3)


def test_long_sound_is_generated_in_parts(studio):
    studio.sound(job(), "rain", seconds=25)
    lengths = [kw["length"] for k, a, kw in studio.calls if k == "sound"]
    assert lengths == [10, 10, 5]


def test_music_from_description_picks_the_real_file(studio):
    item = studio.music(job(), "cozy lo-fi beat", seconds=23, instrumental=True, seed=3)[0]
    kw = next(kw for k, a, kw in studio.calls if a == "/simple_generate")
    assert kw["state"] == {"description": "cozy lo-fi beat", "instrumental": True, "title": "", "mode": "simple"}
    assert kw["duration"] == 25 and kw["seed"] == 3  # rounded up to the Space's 5 s steps
    assert item.kind == "audio" and media.duration(item.path) == pytest.approx(4.0, abs=0.2)


def test_music_with_lyrics_writes_style_then_keeps_the_lyrics(studio):
    studio.music(job(), "warm soul ballad", seconds=60, lyrics="[verse]\nHello again")
    assert [a for k, a in apis(studio) if k == "music"] == ["/compose_assist", "/studio_generate"]
    compose = next(kw for k, a, kw in studio.calls if a == "/compose_assist")
    assert compose["raw_state"]["assist"] == "prompt" and compose["raw_state"]["lyrics"] == "[verse]\nHello again"
    gen = next(kw for k, a, kw in studio.calls if a == "/studio_generate")
    assert gen["state"]["lyrics"] == "[verse]\nHello again" and gen["state"]["global_meta"].startswith("Basic")


def test_add_music_to_video_is_instrumental_and_video_length(studio, media_files):
    item = studio.add_sound(job(), media_files["vid_audio"], "tense synth score", engine="music")[0]
    kw = next(kw for k, a, kw in studio.calls if a == "/simple_generate")
    assert kw["state"]["instrumental"] is True and kw["duration"] == 5  # 2 s clip -> 5 s minimum
    assert media.duration(item.path) == pytest.approx(2.0, abs=0.2) and item.title == "Video · with music"


def test_talk_omni_adds_audio_track(studio, media_files):
    item = studio.talk(job(), media_files["img"], script="Hello there.", engine="omni")[0]
    assert ("avatar", "/start_session") in apis(studio) and ("avatar", "/infer_scene") in apis(studio)
    assert media.probe(item.path)["has_audio"]


def test_talk_moda_with_uploaded_audio_skips_tts(studio, media_files):
    studio.talk(job(), media_files["img"], audio=media_files["wav"], engine="moda", emotion="Happiness")
    assert not any(k == "tts" for k, a in apis(studio))
    kw = next(kw for k, a, kw in studio.calls if k == "avatar_fast")
    assert kw["emotion_name"] == "Happiness"


def test_lipsync_loops_short_video_and_keeps_every_word(studio, media_files):
    item = studio.lipsync(job(), media_files["vid_silent"], audio=media_files["long_wav"])[0]
    kw = next(kw for k, a, kw in studio.calls if k == "lipsync")
    assert media.duration(kw["input_video_path"]["path"]) == pytest.approx(4.8, abs=0.2)  # 4 s + 0.8 s pad
    # the fake model returns a 1.5 s clip: the result still lasts for all of the speech
    assert media.duration(item.path) >= 4.2 and media.probe(item.path)["has_audio"]


def test_moda_gets_padded_audio_and_keeps_every_word(studio, media_files):
    studio.overrides[("avatar_fast", "/generate_motion")] = lambda kw: {"video": {"path": media_files["vid_silent"]}}
    item = studio.talk(job(), media_files["img"], audio=media_files["long_wav"], engine="moda")[0]
    kw = next(kw for k, a, kw in studio.calls if a == "/generate_motion")
    assert media.duration(kw["driving_audio_path"]["path"]) == pytest.approx(4.8, abs=0.1)
    assert media.duration(item.path) == pytest.approx(4.3, abs=0.2)  # 1.5 s clip held to the speech length


def test_omni_splits_long_speech_into_parts_under_5s(studio, media_files):
    item = studio.talk(job(), media_files["img"], audio=media_files["vlong_wav"], engine="omni")[0]
    parts = [kw["audio_path"]["path"] for k, a, kw in studio.calls if a == "/infer_scene"]
    assert len(parts) == 3 and all(media.duration(p) <= 4.85 for p in parts)
    assert media.duration(item.path) == pytest.approx(11.0, abs=0.3) and media.probe(item.path)["has_audio"]
    assert item.meta["parts"] == 3


def test_plan_segments_prefers_pauses():
    assert media.plan_segments(11.0, [3.9, 7.5, 10.0], 4.8) == [(0.0, 3.9), (3.9, 7.5), (7.5, 11.0)]
    assert media.plan_segments(10.0, [], 4.8) == [(0.0, 4.8), (4.8, 9.6), (9.6, 10.0)]
    assert media.plan_segments(3.0, [1.0], 4.8) == [(0.0, 3.0)]


def test_job_queue_reports_friendly_errors(tmp_path):
    q = JobQueue(tmp_path)

    def boom(j):
        raise SpaceError("quota", "You have exceeded your GPU quota", "x/y")

    j = q.submit("t", boom)
    q._pool.shutdown(wait=True)
    assert j.status == "failed" and "GPU quota" in j.error
    assert (tmp_path / "jobs.jsonl").read_text().count('"failed"') == 1
