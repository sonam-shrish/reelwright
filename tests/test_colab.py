import json
import subprocess
from pathlib import Path

import pytest

from reelwright import colab
from reelwright.colab import ColabError, ColabRunner
from reelwright.jobs import Job


class FakeCLI:
    """Pretends to be `colab`: records commands and answers like the real CLI."""

    def __init__(self, video, running=False, results=None):
        self.cmds, self.running, self.video = [], running, video
        self.results = list(results or [])

    def __call__(self, args, capture_output, text, timeout, stdin):
        cmd = args[3]  # [cli, --auth, oauth2, <command>, ...]
        self.cmds.append(args[3:])
        out = ""
        if cmd == "sessions":
            out = "reelwright  A100  RUNNING" if self.running else "[colab] No active sessions found on server."
        elif cmd == "new":
            self.running = True
            out = "[colab] Session READY."
        elif cmd == "stop":
            self.running = False
        elif cmd == "exec":
            script = Path(args[args.index("-f") + 1]).read_text()
            assert "def rw_call" in script  # the worker is sent along
            out = "noise\nRW_RESULT " + json.dumps(self.results.pop(0)) + "\n"
        elif cmd == "download":
            Path(args[-1]).write_bytes(Path(self.video).read_bytes())
        return subprocess.CompletedProcess(args, 0, out, "")


@pytest.fixture
def signed_in(tmp_path, monkeypatch):
    tok = tmp_path / "token.json"
    tok.write_text("{}")
    monkeypatch.setattr(colab, "TOKEN_PATH", tok)
    monkeypatch.setattr(colab, "find_cli", lambda: "/fake/colab")


def test_generate_starts_runtime_sets_up_and_downloads(tmp_path, signed_in, media_files):
    fake = FakeCLI(media_files["vid_silent"], results=[
        {"status": "ready", "cached": False, "mode": "gpu", "download_s": 300, "load_s": 60},
        {"status": "done", "path": "/content/rw_out/a.mp4", "gen_s": 20.5, "size": "832x480", "frames": 65}])
    r = ColabRunner(run=fake, workdir=tmp_path, gpu="A100")
    logs = []
    out = r.generate_video(media_files["img"], "a fox", 4, 7, say=logs.append)
    assert out.is_file() and out.stat().st_size > 0
    assert [c[0] for c in fake.cmds] == ["sessions", "new", "exec", "exec", "download"]
    assert fake.cmds[1][-3:] == ["--gpu", "A100", "--high-mem"]
    assert any("Model ready" in m for m in logs) and r.last_used > 0


def test_reuses_running_runtime(tmp_path, signed_in, media_files):
    fake = FakeCLI(media_files["vid_silent"], running=True, results=[
        {"status": "ready", "cached": True}, {"status": "done", "path": "/content/rw_out/b.mp4"}])
    ColabRunner(run=fake, workdir=tmp_path).generate_video(media_files["img"], "x", 2, 1)
    assert "new" not in [c[0] for c in fake.cmds]


def test_worker_error_is_reported(tmp_path, signed_in, media_files):
    fake = FakeCLI(media_files["vid_silent"], running=True, results=[
        {"status": "error", "error": "OutOfMemoryError: CUDA out of memory", "trace": "..."}])
    with pytest.raises(ColabError) as e:
        ColabRunner(run=fake, workdir=tmp_path).generate_video(media_files["img"], "x", 2, 1)
    assert "out of memory" in e.value.user_message()


def test_not_signed_in_gives_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr(colab, "TOKEN_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(colab, "find_cli", lambda: "/fake/colab")
    with pytest.raises(ColabError) as e:
        ColabRunner(run=lambda *a, **k: None, workdir=tmp_path).is_running()
    assert "colab_signin.sh" in str(e.value)


def test_worker_resize_and_frames_match_the_space():
    ns: dict = {}
    exec(colab.WORKER_SRC, ns)
    from PIL import Image
    assert ns["_resize"](Image.new("RGB", (1536, 864))).size == (832, 480)  # 16:9 is wider than 832:480, so cropped
    assert ns["_resize"](Image.new("RGB", (480, 640))).size == (624, 832)
    assert ns["_resize"](Image.new("RGB", (500, 500))).size == (640, 640)
    assert ns["_num_frames"](4.0) == 65 and (ns["_num_frames"](3.5) - 1) % 4 == 0


def test_studio_routes_colab_model(tmp_path, media_files):
    from reelwright.studio import Studio

    class FakeRunner:
        calls = []

        def generate_video(self, image, prompt, seconds, seed, steps, negative_prompt, say):
            self.calls.append((image, prompt, seconds, seed, steps))
            say("generating on fake colab")
            return Path(media_files["vid_silent"])

    runner = FakeRunner()
    s = Studio(data_dir=tmp_path / "d", token="t", spaces={}, caller_factory=lambda k: None, colab=runner)
    item = s.create_video(Job(id="j", title="t"), media_files["img"], "a fox runs", ["Dolly In"], "wan_colab",
                          4, 1, 3, quality="High")[0]
    assert runner.calls[0][2:] == (4.0, 3, 8) and "dolly-in" in runner.calls[0][1]
    assert item.title.startswith("Video · Wan 2.2 · Colab")
