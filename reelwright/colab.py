"""Run generations on your own Google Colab GPU (inside a Claude Code session).

Uses Google's Colab CLI (`google-colab-cli`). One runtime is kept warm while you
work and stopped automatically after it has been idle, so compute units are not
wasted. The model stays loaded in the runtime between clips.

Setup once per container:  bash scripts/colab_setup.sh
Sign in (once per container):  bash scripts/colab_signin.sh   (then paste the code)
"""
from __future__ import annotations

import json
import re
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from . import config

log = logging.getLogger("reelwright.colab")

WORKER_SRC = (Path(__file__).parent / "colab_worker.py").read_text()
TOKEN_PATH = Path.home() / ".config" / "colab-cli" / "token.json"
# With keyrings.alt installed, the CLI moves the sign-in into this file and deletes token.json.
KEYRING_PATH = Path.home() / ".local" / "share" / "python_keyring" / "keyring_pass.cfg"


def signed_in() -> bool:
    return TOKEN_PATH.exists() or KEYRING_PATH.exists()


# Network hiccups between this sandbox and the runtime (worth retrying; jobs are idempotent)
TRANSIENT_HINTS = ("Timeout", "ConnectionError", "Connection was lost", "Connection reset", "RemoteDisconnected", "ProtocolError",
                   "502", "503", "504")


# GPUs Colab only offers in one shape: never request --high-mem for these.
STANDARD_ONLY_GPUS = {"G4", "L4"}


class ColabError(RuntimeError):
    def user_message(self) -> str:
        return f"Colab: {self}"


def find_cli() -> str | None:
    for cand in (os.environ.get("REELWRIGHT_COLAB_CLI"), shutil.which("colab"),
                 str(Path.home() / ".reelwright" / "colab-venv" / "bin" / "colab")):
        if cand and Path(cand).is_file():
            return cand
    return None


class ColabSpaceCaller:
    """Same interface as spaces.SpaceCaller, but runs the Space on your Colab GPU."""

    def __init__(self, runner: "ColabRunner", space_id: str):
        self.runner, self.space_id = runner, space_id

    def call(self, api_name: str, **kwargs):
        return self.runner.run_space(self.space_id, api_name, kwargs, lambda m: log.info("%s", m))

    def client(self):  # for code that calls client().predict(...)
        return self

    def predict(self, api_name: str, **kwargs):
        return self.call(api_name, **kwargs)


class ColabRunner:
    def __init__(self, session: str | None = None, gpu: str | None = None, high_mem: bool = True,
                 model_id: str | None = None, idle_stop_minutes: float | None = None,
                 run: Callable = subprocess.run, workdir: Path | None = None):
        self.session = session or config.COLAB_SESSION
        self.gpu = gpu or config.COLAB_GPU
        self.high_mem = high_mem
        self.model_id = model_id or config.COLAB_WAN_MODEL
        self.idle_stop = 60 * (idle_stop_minutes if idle_stop_minutes is not None else config.COLAB_IDLE_STOP_MIN)
        self._run = run
        self.workdir = Path(workdir or config.DATA_DIR / "colab")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # one Colab job at a time
        self.last_used = 0.0
        self.started_at = 0.0
        self._watchdog: threading.Thread | None = None
        self.poll_every = 15.0
        self.video_scale = config.COLAB_VIDEO_SCALE  # 1.5 = 720p class (slower, sharper)
        self.video_upscale = config.COLAB_VIDEO_UPSCALE  # 2 = Real-ESRGAN x2 on Colab after generating
        self.active_gpu = ""

    # ---------------------------------------------------------------- CLI
    def _cli(self, args: list[str], timeout: float = 600) -> str:
        cli = find_cli()
        if not cli:
            raise ColabError("the Colab CLI is not installed here. Run: bash scripts/colab_setup.sh")
        if not signed_in():
            raise ColabError("not signed in to Google Colab in this session. Run: bash scripts/colab_signin.sh")
        try:
            proc = self._run([cli, "--auth", "oauth2", *args], capture_output=True, text=True,
                             timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise ColabError(f"'colab {args[0]}' took longer than {int(timeout)} s") from e
        out = (proc.stdout or "") + (proc.stderr or "")
        if "authorization code" in out.lower():  # the stored sign-in was rejected or expired
            raise ColabError("the Colab sign-in has expired. Run: bash scripts/colab_signin.sh")
        if proc.returncode != 0:
            raise ColabError(f"'colab {args[0]}' failed: {out.strip()[-600:]}")
        return out

    # ---------------------------------------------------------------- runtime
    def is_running(self) -> bool:
        out = self._cli(["sessions"], timeout=120)
        # exact name: "reelwright" must not match "reelwright-2"
        return re.search(rf"(?<![\w-]){re.escape(self.session)}(?![\w-])", out) is not None

    def start(self, say: Callable[[str], None] = lambda m: None):
        if self.is_running():
            return
        # gpu may be a preference list like "G4,A100": try each (with and without high RAM),
        # because a GPU type can be refused (no entitlement) or have no free machines.
        errors = []
        for gpu in [g.strip() for g in self.gpu.split(",") if g.strip()]:
            # G4 has one shape (about 176 GB RAM already); asking for high RAM gets a misleading
            # "no quota or entitlement" refusal, so only A100/T4/CPU get the high-RAM request.
            for high_mem in ([True, False] if self.high_mem and gpu.upper() not in STANDARD_ONLY_GPUS else [False]):
                say(f"Starting a Colab {gpu} runtime" + (" (high RAM)" if high_mem else ""))
                args = ["new", "-s", self.session, "--gpu", gpu] + (["--high-mem"] if high_mem else [])
                try:
                    out = self._cli(args, timeout=900)
                except ColabError as e:
                    errors.append(f"{gpu}{' high-mem' if high_mem else ''}: {str(e)[-200:]}")
                    continue
                if "READY" not in out:
                    errors.append(f"{gpu}{' high-mem' if high_mem else ''}: {out.strip()[-200:]}")
                    continue
                self.active_gpu = gpu
                self.started_at = time.time()
                self._start_watchdog()
                return
        raise ColabError("could not start a Colab runtime: " + " | ".join(errors))

    def stop(self) -> str:
        with self._lock:
            try:
                if not self.is_running():
                    return "No Colab runtime is running."
                self._cli(["stop", "-s", self.session], timeout=300)
                return "Colab runtime stopped."
            finally:
                self.started_at = 0.0

    def status(self) -> str:
        if not find_cli():
            return "Colab: not set up in this session (run scripts/colab_setup.sh)."
        if not signed_in():
            return "Colab: not signed in (run scripts/colab_signin.sh)."
        try:
            running = self.is_running()
        except ColabError as e:
            return f"Colab: could not check ({e})."
        if not running:
            return f"Colab: no runtime running. One starts ({self.gpu}) when you pick the Colab model."
        idle = int((time.time() - self.last_used) / 60) if self.last_used else 0
        return (f"Colab: {self.gpu} runtime **running** (idle {idle} min; stops itself after "
                f"{int(self.idle_stop / 60)} idle min). Uses compute units while running.")

    def _start_watchdog(self):
        if self._watchdog and self._watchdog.is_alive():
            return

        def loop():
            while True:
                time.sleep(60)
                if not self.last_used or not self.started_at:
                    continue
                if time.time() - self.last_used > self.idle_stop and not self._lock.locked():
                    try:
                        log.info("Stopping idle Colab runtime")
                        self.stop()
                    except Exception:
                        log.exception("idle stop failed")
                    return

        self._watchdog = threading.Thread(target=loop, daemon=True, name="colab-idle-stop")
        self._watchdog.start()

    # ---------------------------------------------------------------- worker
    def call(self, action: str, timeout: float, retries: int = 0, **kwargs) -> dict:
        """Run one worker action. The CLI's connection to the runtime sometimes hangs
        (the kernel never receives the code), so a timed-out call is retried."""
        for attempt in range(retries + 1):
            try:
                return self._call_once(action, timeout, **kwargs)
            except ColabError as e:
                msg = str(e)
                hung = "took longer than" in msg
                flaky = "'colab exec' failed" in msg and any(k in msg for k in TRANSIENT_HINTS)
                if not (hung or flaky) or attempt == retries:
                    raise
                log.warning("Colab %s call %s; retrying (%d/%d)", action, "hung" if hung else "lost its connection",
                            attempt + 1, retries)
                time.sleep(min(30, 5 * (attempt + 1)))

    def _call_once(self, action: str, timeout: float, **kwargs) -> dict:
        script = self.workdir / f"call-{uuid.uuid4().hex[:8]}.py"
        script.write_text(WORKER_SRC + "\n\nimport json as _j\n"
                          f"rw_call({action!r}, **_j.loads({json.dumps(json.dumps(kwargs))}))\n")
        env = ["--env", f"HF_TOKEN={config.HF_TOKEN}"] if config.HF_TOKEN else []  # faster model downloads
        try:
            out = self._cli(["exec", "-s", self.session, "--timeout", str(int(timeout)), *env, "-f", str(script)],
                            timeout=timeout + 30)
        finally:
            script.unlink(missing_ok=True)
        lines = [ln for ln in out.splitlines() if ln.startswith("RW_RESULT ")]
        if not lines:
            raise ColabError(f"no result came back from the runtime: {out.strip()[-800:]}")
        result = json.loads(lines[-1][len("RW_RESULT "):])
        if result.get("status") == "error":
            raise ColabError(result.get("error", "unknown error") + ("\n" + result.get("trace", "")[-600:]))
        return result

    def download(self, remote: str, dest: Path, retries: int = 3) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries + 1):
            try:
                self._cli(["download", "-s", self.session, remote, str(dest)], timeout=180)
                break
            except ColabError as e:
                if "took longer than" not in str(e) or attempt == retries:
                    raise
        if not dest.is_file() or dest.stat().st_size == 0:
            raise ColabError(f"download of {remote} failed")
        return dest

    # ---------------------------------------------------------------- any HF Space on Colab
    def _wait(self, check: Callable[[], dict], deadline_s: float, say, label: str) -> dict:
        deadline, last = time.time() + deadline_s, None
        while time.time() < deadline:
            res = check()
            if res["status"] in ("ready", "done"):
                return res
            if res["status"] == "error":
                raise ColabError(f"{label}: {res.get('error', 'failed')}")
            if res.get("stage") != last:
                last = res.get("stage")
                say(f"Colab {label}: {last}")
            time.sleep(self.poll_every)
        raise ColabError(f"{label} did not finish in time")

    def run_space(self, space_id: str, api_name: str, kwargs: dict,
                  say: Callable[[str], None] = lambda m: None) -> object:
        """Run a Hugging Face Space's endpoint on the Colab GPU instead of on ZeroGPU.
        LocalFile arguments are sent along; result files are downloaded and returned
        as local paths in the same result structure."""
        import base64
        from .spaces import LocalFile

        def pack(v):
            if isinstance(v, LocalFile):
                return {"__file_b64__": base64.b64encode(Path(v.path).read_bytes()).decode(), "name": Path(v.path).name}
            if isinstance(v, dict):
                return {k: pack(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [pack(x) for x in v]
            return v

        with self._lock:
            self.start(say)
            up = self.call("space_up", timeout=50, retries=6, space_id=space_id)
            if up["status"] != "ready":
                say(f"Starting {space_id} on Colab (first time per runtime: downloads, installs, loads)")
                self._wait(lambda: self.call("space_status", timeout=40, retries=6, space_id=space_id),
                           2400, say, space_id)
            job_id = uuid.uuid4().hex[:12]
            self.call("space_call", timeout=50, retries=6, job_id=job_id, space_id=space_id, api_name=api_name,
                      kwargs=pack(kwargs))
            res = self._wait(lambda: self.call("poll", timeout=40, retries=6, job_id=job_id), 1800, say,
                             f"{space_id} {api_name}")
            local = {}
            for remote in res.get("files", []):
                local[remote] = str(self.download(remote, self.workdir / Path(remote).name))
            self.last_used = time.time()

        def unpack(v):
            if isinstance(v, dict):
                if v.get("path") in local:
                    return {**v, "path": local[v["path"]]}
                return {k: unpack(x) for k, x in v.items()}
            if isinstance(v, list):
                return [unpack(x) for x in v]
            return v

        result = unpack(res.get("result"))
        return tuple(result) if isinstance(result, list) else result

    def prewarm(self, say: Callable[[str], None] = lambda m: None) -> None:
        """Start the runtime and begin loading the model in the background, so the
        first clip doesn't wait for the 69 GB download. Returns in about a minute."""
        with self._lock:
            self.start(say)
            res = self.call("warm", timeout=50, retries=6, model_id=self.model_id)
            say("Colab model already loaded" if res.get("status") == "ready"
                else "Colab is loading the model in the background")
            self.last_used = time.time()

    def generate_video(self, image: str, prompt: str, seconds: float, seed: int, steps: int = 4,
                       negative_prompt: str = "", say: Callable[[str], None] = lambda m: None) -> Path:
        import base64
        with self._lock:
            self.start(say)
            # The job runs in a thread on the runtime; we only make short start/poll calls,
            # because one long `colab exec` connection can hang through the sandbox proxy.
            job_id = uuid.uuid4().hex[:12]
            job = {"image_b64": base64.b64encode(Path(image).read_bytes()).decode(), "prompt": prompt,
                   "negative_prompt": negative_prompt, "seconds": seconds, "seed": seed, "steps": steps,
                   "scale": self.video_scale, "upscale": self.video_upscale}
            # Short limits: a hung connection is dropped after ~1 min and retried; the job keeps going on Colab.
            self.call("start", timeout=50, retries=6, job_id=job_id, model_id=self.model_id, job=job)
            say(f"Started on Colab: {seconds:.1f}s, {steps} steps, seed {seed}")
            # first clip per runtime also downloads the 69 GB model (~4 min); a clip is ~85 s at 8 steps
            deadline = time.time() + 2400 + 60 * seconds * steps / 4
            last_stage, res = None, None
            while time.time() < deadline:
                time.sleep(self.poll_every)
                res = self.call("poll", timeout=40, retries=6, job_id=job_id)
                if res["status"] == "done":
                    break
                if res.get("stage") != last_stage:
                    last_stage = res.get("stage")
                    say(f"Colab: {last_stage}" + (" (first clip per runtime downloads ~69 GB: several minutes)"
                                                  if last_stage == "loading model" else ""))
            else:
                raise ColabError("the Colab job did not finish in time")
            if res.get("setup") and not res["setup"].get("cached"):
                say(f"Model loaded on Colab ({res['setup'].get('mode')} mode; download "
                    f"{res['setup'].get('download_s')} s, load {res['setup'].get('load_s')} s)")
            say(f"Colab finished in {res.get('gen_s')} s ({res.get('size')}, {res.get('frames')} frames)")
            local = self.download(res["path"], self.workdir / f"{uuid.uuid4().hex[:12]}.mp4")
            self.last_used = time.time()
            return local
