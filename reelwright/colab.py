"""Run generations on your own Google Colab GPU (inside a Claude Code session).

Uses Google's Colab CLI (`google-colab-cli`). One runtime is kept warm while you
work and stopped automatically after it has been idle, so compute units are not
wasted. The model stays loaded in the runtime between clips.

Setup once per container:  bash scripts/colab_setup.sh
Sign in (once per container):  bash scripts/colab_signin.sh   (then paste the code)
"""
from __future__ import annotations

import json
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


class ColabError(RuntimeError):
    def user_message(self) -> str:
        return f"Colab: {self}"


def find_cli() -> str | None:
    for cand in (os.environ.get("REELWRIGHT_COLAB_CLI"), shutil.which("colab"),
                 str(Path.home() / ".reelwright" / "colab-venv" / "bin" / "colab")):
        if cand and Path(cand).is_file():
            return cand
    return None


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

    # ---------------------------------------------------------------- CLI
    def _cli(self, args: list[str], timeout: float = 600) -> str:
        cli = find_cli()
        if not cli:
            raise ColabError("the Colab CLI is not installed here. Run: bash scripts/colab_setup.sh")
        if not TOKEN_PATH.exists():
            raise ColabError("not signed in to Google Colab in this session. Run: bash scripts/colab_signin.sh")
        try:
            proc = self._run([cli, "--auth", "oauth2", *args], capture_output=True, text=True,
                             timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise ColabError(f"'colab {args[0]}' took longer than {int(timeout)} s") from e
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise ColabError(f"'colab {args[0]}' failed: {out.strip()[-600:]}")
        return out

    # ---------------------------------------------------------------- runtime
    def is_running(self) -> bool:
        out = self._cli(["sessions"], timeout=120)
        return "No active sessions" not in out and self.session in out

    def start(self, say: Callable[[str], None] = lambda m: None):
        if self.is_running():
            return
        say(f"Starting a Colab {self.gpu} runtime")
        args = ["new", "-s", self.session, "--gpu", self.gpu] + (["--high-mem"] if self.high_mem else [])
        self._cli(args, timeout=900)
        self.started_at = time.time()
        self._start_watchdog()

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
        if not TOKEN_PATH.exists():
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
    def call(self, action: str, timeout: float, **kwargs) -> dict:
        script = self.workdir / f"call-{uuid.uuid4().hex[:8]}.py"
        script.write_text(WORKER_SRC + "\n\nimport json as _j\n"
                          f"rw_call({action!r}, **_j.loads({json.dumps(json.dumps(kwargs))}))\n")
        env = ["--env", f"HF_TOKEN={config.HF_TOKEN}"] if config.HF_TOKEN else []  # faster model downloads
        try:
            out = self._cli(["exec", "-s", self.session, "--timeout", str(int(timeout)), *env, "-f", str(script)],
                            timeout=timeout + 120)
        finally:
            script.unlink(missing_ok=True)
        lines = [ln for ln in out.splitlines() if ln.startswith("RW_RESULT ")]
        if not lines:
            raise ColabError(f"no result came back from the runtime: {out.strip()[-800:]}")
        result = json.loads(lines[-1][len("RW_RESULT "):])
        if result.get("status") == "error":
            raise ColabError(result.get("error", "unknown error") + ("\n" + result.get("trace", "")[-600:]))
        return result

    def download(self, remote: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._cli(["download", "-s", self.session, remote, str(dest)], timeout=600)
        if not dest.is_file() or dest.stat().st_size == 0:
            raise ColabError(f"download of {remote} failed")
        return dest

    def generate_video(self, image: str, prompt: str, seconds: float, seed: int, steps: int = 4,
                       negative_prompt: str = "", say: Callable[[str], None] = lambda m: None) -> Path:
        import base64
        with self._lock:
            self.start(say)
            say("Loading Wan 2.2 on Colab (first clip per runtime downloads ~69 GB: several minutes)")
            setup = self.call("setup", timeout=3600, model_id=self.model_id)
            if not setup.get("cached"):
                say(f"Model ready on Colab ({setup.get('mode')} mode; download {setup.get('download_s')} s, "
                    f"load {setup.get('load_s')} s)")
            say(f"Generating on Colab: {seconds:.1f}s, {steps} steps, seed {seed}")
            job = {"image_b64": base64.b64encode(Path(image).read_bytes()).decode(), "prompt": prompt,
                   "negative_prompt": negative_prompt, "seconds": seconds, "seed": seed, "steps": steps}
            res = self.call("generate", timeout=1800, model_id=self.model_id, job=job)
            say(f"Colab finished in {res.get('gen_s')} s ({res.get('size')}, {res.get('frames')} frames)")
            local = self.download(res["path"], self.workdir / f"{uuid.uuid4().hex[:12]}.mp4")
            self.last_used = time.time()
            return local
