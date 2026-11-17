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
    monkeypatch.setattr(colab, "KEYRING_PATH", tmp_path / "no-keyring.cfg")
    monkeypatch.setattr(colab, "find_cli", lambda: "/fake/colab")


def runner(fake, tmp_path, **kw):
    r = ColabRunner(run=fake, workdir=tmp_path, **kw)
    r.poll_every = 0
    return r


def test_generate_starts_runtime_polls_and_downloads(tmp_path, signed_in, media_files):
    fake = FakeCLI(media_files["vid_silent"], results=[
        {"status": "started", "id": "x"},
        {"status": "running", "stage": "loading model"},
        {"status": "done", "path": "/content/rw_out/a.mp4", "gen_s": 20.5, "size": "832x480", "frames": 65,
         "setup": {"cached": False, "mode": "gpu", "download_s": 300, "load_s": 60}}])
    r = runner(fake, tmp_path, gpu="A100")
    logs = []
    out = r.generate_video(media_files["img"], "a fox", 4, 7, say=logs.append)
    assert out.is_file() and out.stat().st_size > 0
    assert [c[0] for c in fake.cmds] == ["sessions", "new", "exec", "exec", "exec", "download"]
    assert fake.cmds[1][-3:] == ["--gpu", "A100", "--high-mem"]
    assert any("loading model" in m for m in logs) and any("Model loaded" in m for m in logs) and r.last_used > 0


def test_reuses_running_runtime(tmp_path, signed_in, media_files):
    fake = FakeCLI(media_files["vid_silent"], running=True, results=[
        {"status": "started", "id": "x"}, {"status": "done", "path": "/content/rw_out/b.mp4"}])
    runner(fake, tmp_path).generate_video(media_files["img"], "x", 2, 1)
    assert "new" not in [c[0] for c in fake.cmds]
    assert [c[0] for c in fake.cmds].count("exec") == 2  # start + one poll


def test_worker_error_is_reported(tmp_path, signed_in, media_files):
    fake = FakeCLI(media_files["vid_silent"], running=True, results=[
        {"status": "started", "id": "x"},
        {"status": "error", "error": "OutOfMemoryError: CUDA out of memory", "trace": "..."}])
    with pytest.raises(ColabError) as e:
        runner(fake, tmp_path).generate_video(media_files["img"], "x", 2, 1)
    assert "out of memory" in e.value.user_message()


def test_worker_background_jobs_start_once_and_report(monkeypatch):
    import time as _t
    ns: dict = {}
    exec(colab.WORKER_SRC, ns)
    ns["RW"].clear()
    ns["RW"].update(pipe=object(), model_id="m/x")
    ns["rw_generate"] = lambda job: {"status": "done", "path": "/content/rw_out/z.mp4", "gen_s": 1.0}
    ns["_background"].__globals__["rw_generate"] = ns["rw_generate"]
    assert ns["rw_start"]("j1", "m/x", {"seed": 1})["status"] == "started"
    assert ns["rw_start"]("j1", "m/x", {"seed": 1}).get("duplicate") is True  # a retried start is ignored
    for _ in range(50):
        if ns["rw_poll"]("j1")["status"] == "done":
            break
        _t.sleep(0.02)
    assert ns["rw_poll"]("j1")["path"].endswith("z.mp4")
    assert ns["rw_poll"]("nope")["status"] == "error"


def test_not_signed_in_gives_clear_message(tmp_path, monkeypatch):
    monkeypatch.setattr(colab, "TOKEN_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(colab, "KEYRING_PATH", tmp_path / "missing.cfg")
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
    assert ns["_resize"](Image.new("RGB", (864, 1536)), 1.5).size == (720, 1248)  # HD vertical
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


def test_sign_in_kept_in_keyring_counts(tmp_path, monkeypatch, media_files):
    monkeypatch.setattr(colab, "TOKEN_PATH", tmp_path / "gone.json")  # the CLI deletes it after moving it
    kr = tmp_path / "keyring_pass.cfg"
    kr.write_text("x")
    monkeypatch.setattr(colab, "KEYRING_PATH", kr)
    monkeypatch.setattr(colab, "find_cli", lambda: "/fake/colab")
    assert ColabRunner(run=FakeCLI(media_files["vid_silent"]), workdir=tmp_path).is_running() is False


def test_expired_sign_in_is_reported(tmp_path, signed_in):
    def ask_for_code(args, **kw):
        return subprocess.CompletedProcess(args, 1, "To authorize colab-cli, visit ...\nEnter the authorization code: ", "")
    with pytest.raises(ColabError) as e:
        ColabRunner(run=ask_for_code, workdir=tmp_path).is_running()
    assert "expired" in str(e.value)


def test_hung_poll_is_retried(tmp_path, signed_in, media_files):
    fake = FakeCLI(media_files["vid_silent"], running=True, results=[
        {"status": "started", "id": "x"}, {"status": "done", "path": "/content/rw_out/c.mp4"}])
    hung = {"n": 0}

    def run(args, **kw):
        if args[3] == "exec" and "poll" in Path(args[args.index("-f") + 1]).read_text().splitlines()[-1] and hung["n"] == 0:
            hung["n"] += 1
            raise subprocess.TimeoutExpired(args, kw["timeout"])
        return fake(args, **kw)

    out = runner(run, tmp_path).generate_video(media_files["img"], "x", 2, 1)
    assert out.is_file() and hung["n"] == 1


def test_run_space_starts_calls_and_downloads(tmp_path, signed_in, media_files):
    from reelwright.spaces import LocalFile
    fake = FakeCLI(media_files["wav"], running=True, results=[
        {"status": "starting", "stage": "queued"},
        {"status": "starting", "stage": "installing its requirements (own virtualenv)"},
        {"status": "ready", "stage": "ready"},
        {"status": "started", "id": "j"},
        {"status": "done", "result": [{"path": "/content/rw_out/ab-audio.wav", "meta": {"_type": "gradio.FileData"}}, "ok"],
         "files": ["/content/rw_out/ab-audio.wav"]}])
    r = runner(fake, tmp_path)
    logs = []
    out = r.run_space("Qwen/Qwen3-TTS", "/generate_voice_clone", {"ref_audio": LocalFile(media_files["wav"]),
                                                                  "target_text": "hi"}, logs.append)
    assert isinstance(out, tuple) and out[1] == "ok"
    assert Path(out[0]["path"]).is_file() and str(tmp_path) in out[0]["path"]  # downloaded locally
    sent = [Path(c[c.index("-f") + 1]) for c in fake.cmds if c[0] == "exec"]
    assert any("installing" in m for m in logs)


def test_studio_can_route_a_space_to_colab(tmp_path):
    from reelwright.colab import ColabSpaceCaller
    from reelwright.studio import Studio
    s = Studio(data_dir=tmp_path / "d", token="t", spaces={"voice_design": "Qwen/Qwen3-TTS", "image": "x/y"},
               caller_factory=lambda space: f"hf:{space}", colab=object())
    s.colab_space_keys = {"voice_design"}
    assert isinstance(s.caller("voice_design"), ColabSpaceCaller) and s.caller("image") == "hf:x/y"


def test_connection_error_is_retried(tmp_path, signed_in, media_files, monkeypatch):
    monkeypatch.setattr(colab.time, "sleep", lambda s: None)
    fake = FakeCLI(media_files["vid_silent"], running=True, results=[
        {"status": "started", "id": "x"}, {"status": "done", "path": "/content/rw_out/d.mp4"}])
    failed = {"n": 0}

    def run(args, **kw):
        if args[3] == "exec" and failed["n"] == 0:
            failed["n"] += 1
            return subprocess.CompletedProcess(args, 1, "", "requests.exceptions.ReadTimeout: read timed out")
        return fake(args, **kw)

    out = runner(run, tmp_path).generate_video(media_files["img"], "x", 2, 1)
    assert out.is_file() and failed["n"] == 1


def test_gpu_preference_list_falls_back(tmp_path, signed_in, media_files):
    fake = FakeCLI(media_files["vid_silent"], results=[{"status": "started", "id": "x"},
                                                      {"status": "done", "path": "/content/rw_out/e.mp4"}])

    def run(args, **kw):
        if args[3] == "new" and "G4" in args:
            return subprocess.CompletedProcess(args, 0, "[colab] Backend rejected accelerator 'G4'.", "")
        return fake(args, **kw)

    r = runner(run, tmp_path, gpu="G4,A100")
    r.generate_video(media_files["img"], "x", 2, 1)
    news = [c for c in fake.cmds if c[0] == "new"]
    assert r.active_gpu == "A100" and news[-1][news[-1].index("--gpu") + 1] == "A100"


def test_session_name_must_match_exactly(tmp_path, signed_in, media_files):
    def run(args, **kw):
        return subprocess.CompletedProcess(args, 0, "[reelwright-2] m-s-x | Hardware: A100", "")
    assert ColabRunner(session="reelwright", run=run, workdir=tmp_path).is_running() is False
    assert ColabRunner(session="reelwright-2", run=run, workdir=tmp_path).is_running() is True
