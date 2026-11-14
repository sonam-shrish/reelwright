import httpx
import pytest

from reelwright import presets
from reelwright.library import Library, UsageMeter
from reelwright.spaces import SpaceCaller, SpaceError, classify, fetch, file_refs


@pytest.mark.parametrize("msg,kind", [
    ("You have exceeded your GPU quota (60s requested vs. 12s left)", "quota"),
    ("ZeroGPU worker error", "quota"),
    ("Queue is full! Please try again", "busy"),
    ("Could not fetch config for https://x.hf.space", "offline"),
    ("The upstream Gradio app has raised an exception", "failed"),
    ("Value: 16 is not in the list of choices: [16, 32, 64, 128]", "invalid"),
    ("CancelledError: ", "failed"),
])
def test_classify(msg, kind):
    assert classify(RuntimeError(msg)).kind == kind


def test_classify_client_side_type_error_is_invalid():
    assert classify(TypeError("No value provided for required argument: last_image")).kind == "invalid"


class FlakyClient:
    def __init__(self, errors):
        self.errors = errors

    def predict(self, api_name, **kw):
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def test_caller_retries_transient_then_succeeds():
    timings = []
    errors = [RuntimeError("upstream Gradio app has raised an exception")]
    c = SpaceCaller("a/b", client_factory=lambda s, t: FlakyClient(errors), sleep=lambda s: None,
                    on_timing=lambda *a: timings.append(a))
    assert c.call("/x") == "ok"
    assert [t[3] for t in timings] == [False, True]


def test_retry_gets_fresh_copy_of_file_arguments():
    seen = []

    class MutatingClient:
        def predict(self, api_name, **kw):
            seen.append(kw["image"]["path"])
            kw["image"]["path"] = "/tmp/gradio/server-side.png"  # what gradio_client does on upload
            if len(seen) == 1:
                raise RuntimeError("CancelledError")
            return "ok"

    c = SpaceCaller("a/b", client_factory=lambda s, t: MutatingClient(), sleep=lambda s: None)
    assert c.call("/x", image={"path": "/local/img.png"}) == "ok"
    assert seen == ["/local/img.png", "/local/img.png"]


def test_caller_does_not_retry_quota():
    errors = [RuntimeError("You have exceeded your GPU quota"), None]
    c = SpaceCaller("a/b", client_factory=lambda s, t: FlakyClient(errors), sleep=lambda s: None)
    with pytest.raises(SpaceError) as e:
        c.call("/x")
    assert e.value.kind == "quota" and "GPU quota" in e.value.user_message()
    assert errors == [None]  # only one attempt


def test_file_refs_handles_galleries_and_tuples():
    r = ([{"image": {"path": "/tmp/a.png", "url": "https://x/a.png"}, "caption": None}], "seed", 3)
    assert [x["url"] for x in file_refs(r)] == ["https://x/a.png"]
    r2 = ({"video": {"path": "/tmp/v.mp4", "url": "https://x/v.mp4"}, "subtitles": None}, 5)
    assert file_refs(r2)[0]["url"].endswith("v.mp4")


def test_fetch_retries_after_403(tmp_path):
    hits = []

    def handler(request):
        hits.append(request.headers.get("authorization"))
        return httpx.Response(403) if len(hits) == 1 else httpx.Response(200, content=b"video-bytes")

    http = httpx.Client(transport=httpx.MockTransport(handler))
    p = fetch({"url": "https://x.hf.space/file=/tmp/out.mp4", "path": "/tmp/out.mp4"}, tmp_path, token="tok",
              sleep=lambda s: None, http=http)
    assert p.suffix == ".mp4" and p.read_bytes() == b"video-bytes"
    assert hits == ["Bearer tok", "Bearer tok"]


def test_fetch_rejects_html_pages(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, text="<!doctype html>", headers={"content-type": "text/html"})
        return httpx.Response(200, content=b"RIFF", headers={"content-type": "audio/wav"})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    p = fetch({"url": "https://x/file=/a.wav", "path": "/a.wav"}, tmp_path, sleep=lambda s: None, http=http)
    assert p.read_bytes() == b"RIFF" and len(calls) == 2


def test_presets():
    assert presets.split_text("One. Two! Three?", 9) == ["One. Two!", "Three?"]
    assert all(len(c) <= 50 for c in presets.split_text("word " * 100, 50))
    assert "shaky camera" not in presets.wan_negative(["Handheld"])
    assert "shaky camera" in presets.wan_negative(["Dolly In"])
    assert presets.edit_dims(None, (1920, 1080)) == (1360, 768)
    assert presets.ltx_dims((480, 640)) == (512, 768) and presets.ltx_dims((1000, 1000), "High") == (1024, 1024)
    assert len(presets.video_prompt("x", ["Dolly In", "Orbit Left", "Crane Up", "Tilt Up"]).split(", then ")) == 3


def test_library_and_usage(tmp_path):
    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    lib = Library(tmp_path / "lib")
    a = lib.add(src, "first")
    b = lib.add(src, "second")
    assert [i.title for i in lib.items()] == ["second", "first"] and lib.get(a.id).title == "first"
    assert lib.find_by_path(b.path).id == b.id
    u = UsageMeter(tmp_path / "lib")
    u.record("s/1", "/x", 30, True)
    u.record("s/1", "/x", 12.5, False)
    assert u.today()["total_seconds"] == pytest.approx(42.5)
