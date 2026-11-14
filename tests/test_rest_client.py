import json

import httpx
import pytest

from reelwright.spaces import LocalFile, RemoteError, RestClient, SpaceCaller, classify

INFO = {"named_endpoints": {
    "/gen": {"parameters": [
        {"parameter_name": "image", "parameter_has_default": False},
        {"parameter_name": "audio", "parameter_has_default": True, "parameter_default": None},
        {"parameter_name": "prompt", "parameter_has_default": True, "parameter_default": "default prompt"},
        {"parameter_name": "steps", "parameter_has_default": True, "parameter_default": 4},
    ]},
}}


def make_server(results):
    """results: list of SSE bodies to return for successive /call/gen/<id> requests."""
    state = {"uploads": 0, "calls": [], "results": list(results)}

    def handler(request: httpx.Request):
        url = str(request.url)
        if "huggingface.co/api/spaces" in url:
            return httpx.Response(200, json={"host": "https://me-space.hf.space"})
        if url.endswith("/gradio_api/info"):
            return httpx.Response(200, json=INFO)
        if url.endswith("/gradio_api/upload"):
            state["uploads"] += 1
            return httpx.Response(200, json=[f"/tmp/gradio/up{state['uploads']}/voice.wav"])
        if request.method == "POST" and "/gradio_api/call/gen" in url:
            state["calls"].append(json.loads(request.content)["data"])
            return httpx.Response(200, json={"event_id": "e1"})
        if request.method == "GET" and "/gradio_api/call/gen/e1" in url:
            return httpx.Response(200, text=state["results"].pop(0))
        return httpx.Response(404)

    return state, httpx.Client(transport=httpx.MockTransport(handler))


OK = 'event: generating\ndata: null\n\nevent: complete\ndata: [{"path": "/tmp/out.mp4", "url": "https://me-space.hf.space/gradio_api/file=/tmp/out.mp4"}, 42]\n\n'
ERR = "event: error\ndata: null\n\n"


@pytest.fixture
def files(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG fake")
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"RIFF fake")
    return str(img), str(wav)


def test_builds_ordered_data_and_uploads_files(files):
    img, wav = files
    state, http = make_server([OK])
    c = RestClient("me/space", http=http)
    assert c.base == "https://me-space.hf.space/gradio_api"
    out = c.predict("/gen", image=LocalFile(img), audio=LocalFile(wav), steps=8)
    data = state["calls"][0]
    assert data[0]["path"] == "/tmp/gradio/up1/voice.wav" and data[1]["path"] == "/tmp/gradio/up2/voice.wav"
    assert data[2] == "default prompt" and data[3] == 8
    assert out[1] == 42
    # result URLs are rebuilt from the server path
    assert out[0]["url"] == "https://me-space.hf.space/gradio_api/file=/tmp/out.mp4"


def test_fast_failures_reupload_then_switch_images_to_inline(files):
    img, wav = files
    state, http = make_server([ERR, ERR, ERR, OK])
    c = RestClient("me/space", http=http)
    c.predict("/gen", image=LocalFile(img), audio=LocalFile(wav))
    assert len(state["calls"]) == 4
    assert all("path" in d[0] and d[0]["path"] for d in state["calls"][:3])  # uploaded 3 times
    last = state["calls"][3]
    assert last[0]["url"].startswith("data:image/png;base64,") and last[0]["path"] is None
    assert last[1]["path"].endswith("voice.wav")  # audio is always uploaded


def test_broken_result_url_is_rebuilt(files):
    img, _ = files
    body = 'event: complete\ndata: [{"path": "/tmp/gradio/x/audio.wav", "url": "https://me-space.hf.space/gradio_a/gradio_api/file=/tmp/gradio/x/audio.wav"}]\n\n'
    _, http = make_server([body])
    out = RestClient("me/space", http=http).predict("/gen", image=LocalFile(img))
    assert out["url"] == "https://me-space.hf.space/gradio_api/file=/tmp/gradio/x/audio.wav"


def test_persistent_error_gives_up_and_is_classified(files):
    img, _ = files
    state, http = make_server([ERR] * 6)
    c = RestClient("me/space", http=http)
    with pytest.raises(RemoteError) as e:
        c.predict("/gen", image=LocalFile(img))
    assert len(state["calls"]) == 6 and classify(e.value).kind == "failed"


def test_missing_required_argument_is_invalid():
    _, http = make_server([])
    c = RestClient("me/space", http=http)
    with pytest.raises(ValueError):
        c.predict("/gen", prompt="x")
    assert classify(ValueError("No value provided for required argument: image")).kind == "invalid"


def test_caller_retries_remote_errors_with_a_fresh_client(files):
    img, _ = files
    made = []

    def factory(space, token):
        state, http = make_server([ERR] if not made else [OK])
        made.append(state)
        return RestClient(space, http=http)

    caller = SpaceCaller("me/space", client_factory=factory, sleep=lambda s: None)
    assert caller.call("/gen", image=LocalFile(img))[1] == 42
    assert len(made) == 2
