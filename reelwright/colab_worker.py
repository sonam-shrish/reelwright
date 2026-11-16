"""Runs INSIDE a Google Colab runtime (sent by reelwright/colab.py with `colab exec`).

The Colab kernel keeps its state between `colab exec` calls, so the model is
downloaded and loaded once per runtime and reused for every clip after that.
Each call prints one line starting with "RW_RESULT " (JSON) for the caller.
Do not import Reelwright modules here: only this file is sent to Colab.
"""
import base64
import copy
import gc
import io
import json
import os
import subprocess
import sys
import time
import traceback
import uuid

RW = globals().setdefault("RW", {})
OUT_DIR = "/content/rw_out"

MAX_DIM, MIN_DIM, SQUARE_DIM, MULTIPLE_OF = 832, 480, 640, 16
FPS, MIN_FRAMES, MAX_FRAMES = 16, 8, 160


def _emit(**payload):
    print("RW_RESULT " + json.dumps(payload), flush=True)


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def _patch_ftfy():
    """If diffusers was imported before ftfy was installed, hand ftfy to the Wan modules."""
    try:
        import ftfy
        import diffusers.pipelines.wan.pipeline_wan_i2v as wan_i2v
        if not hasattr(wan_i2v, "ftfy"):
            wan_i2v.ftfy = ftfy
    except Exception:
        pass


def rw_setup(model_id: str) -> dict:
    """Install what is missing, download the model and load it on the GPU (once).
    Guarded by a lock: a warm-up and a job can both ask for the model at once."""
    import threading
    lock = RW.setdefault("setup_lock", threading.Lock())
    with lock:
        return _setup_locked(model_id)


def _setup_locked(model_id: str) -> dict:
    if RW.get("model_id") == model_id and RW.get("pipe") is not None:
        return {"status": "ready", "cached": True, "mode": RW["mode"]}
    _stop_spaces()  # Space apps would hold GPU memory that Wan needs
    import importlib.util
    # Check without importing: diffusers decides at import time whether ftfy exists,
    # so missing packages must be installed before diffusers is imported.
    missing = [pkg for module, pkg in [("ftfy", "ftfy"), ("imageio_ffmpeg", "imageio-ffmpeg"),
                                       ("diffusers", "diffusers>=0.35")]
               if importlib.util.find_spec(module) is None]
    if missing:
        _pip(*missing)
    _patch_ftfy()
    import torch
    from diffusers import WanImageToVideoPipeline
    from huggingface_hub import snapshot_download

    if RW.get("pipe") is not None:  # a different model was loaded before
        RW["pipe"] = None
        gc.collect()
        torch.cuda.empty_cache()
    t0 = time.time()
    path = snapshot_download(model_id, max_workers=16)
    t1 = time.time()
    pipe = WanImageToVideoPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram >= 75:  # A100 80 GB / H100: everything on the GPU (fast)
        pipe.to("cuda")
        mode = "gpu"
    else:  # smaller GPUs: move each part to the GPU only while it runs
        pipe.enable_model_cpu_offload()
        mode = "offload"
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass
    RW.update(pipe=pipe, model_id=model_id, mode=mode, scheduler=copy.deepcopy(pipe.scheduler))
    return {"status": "ready", "cached": False, "mode": mode, "vram_gb": round(vram, 1),
            "download_s": round(t1 - t0), "load_s": round(time.time() - t1)}


def _resize(image, scale: float = 1.0):
    """Same rule as the Wan 2.2 Lightning Space: 832x480 / 480x832 / 640x640 class sizes.
    scale > 1 renders bigger (1.5 gives 720p class: 1248x720), slower but sharper."""
    from PIL import Image
    width, height = image.size
    if width == height:
        side = round(SQUARE_DIM * scale / MULTIPLE_OF) * MULTIPLE_OF
        return image.resize((side, side), Image.LANCZOS)
    aspect = width / height
    max_ar, min_ar = MAX_DIM / MIN_DIM, MIN_DIM / MAX_DIM
    img = image
    if aspect > max_ar:
        tw, th = MAX_DIM, MIN_DIM
        cw = int(round(height * max_ar))
        left = (width - cw) // 2
        img = image.crop((left, 0, left + cw, height))
    elif aspect < min_ar:
        tw, th = MIN_DIM, MAX_DIM
        ch = int(round(width / min_ar))
        top = (height - ch) // 2
        img = image.crop((0, top, width, top + ch))
    elif width > height:
        tw, th = MAX_DIM, int(round(MAX_DIM / aspect))
    else:
        th, tw = MAX_DIM, int(round(MAX_DIM * aspect))
    fw = max(MIN_DIM, min(MAX_DIM, round(tw / MULTIPLE_OF) * MULTIPLE_OF))
    fh = max(MIN_DIM, min(MAX_DIM, round(th / MULTIPLE_OF) * MULTIPLE_OF))
    if scale != 1.0:
        fw, fh = (round(v * scale / MULTIPLE_OF) * MULTIPLE_OF for v in (fw, fh))
    return img.resize((fw, fh), Image.LANCZOS)


def _num_frames(seconds: float) -> int:
    n = max(MIN_FRAMES, min(MAX_FRAMES, int(round(seconds * FPS))))
    return (n // 4) * 4 + 1  # Wan needs 4k+1 frames


def rw_generate(job: dict) -> dict:
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    _patch_ftfy()
    pipe = RW["pipe"]
    image = _resize(Image.open(io.BytesIO(base64.b64decode(job["image_b64"]))).convert("RGB"),
                    float(job.get("scale", 1.0)))
    frames = _num_frames(float(job.get("seconds", 3.5)))
    kwargs = dict(image=image, prompt=job["prompt"], negative_prompt=job.get("negative_prompt", ""),
                  height=image.height, width=image.width, num_frames=frames,
                  guidance_scale=float(job.get("guidance", 1.0)), guidance_scale_2=float(job.get("guidance", 1.0)),
                  num_inference_steps=int(job.get("steps", 4)))
    t0 = time.time()
    try:
        out = pipe(**kwargs, generator=torch.Generator(device="cuda").manual_seed(int(job["seed"])))
    except torch.OutOfMemoryError:
        # fall back to the memory-saving mode for the rest of this runtime
        gc.collect()
        torch.cuda.empty_cache()
        pipe.to("cpu")
        pipe.enable_model_cpu_offload()
        RW["mode"] = "offload"
        out = pipe(**kwargs, generator=torch.Generator(device="cuda").manual_seed(int(job["seed"])))
    gen_s = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{uuid.uuid4().hex[:12]}.mp4"
    export_to_video(out.frames[0], path, fps=FPS)
    gc.collect()
    torch.cuda.empty_cache()
    return {"status": "done", "path": path, "frames": frames, "size": f"{image.width}x{image.height}",
            "gen_s": round(gen_s, 1), "mode": RW["mode"]}


def _background(job_id: str, model_id: str, job: dict):
    """Runs in a thread inside the kernel, so the caller only needs short polling calls."""
    entry = RW["jobs"][job_id]
    try:
        if RW.get("pipe") is None or RW.get("model_id") != model_id:
            entry["stage"] = "loading model"
            entry["setup"] = rw_setup(model_id)
        entry["stage"] = "generating"
        entry["result"] = rw_generate(job)
        entry["status"] = "done"
    except Exception as e:  # noqa: BLE001
        entry.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])


def rw_start(job_id: str, model_id: str, job: dict) -> dict:
    import threading
    jobs = RW.setdefault("jobs", {})
    if job_id in jobs:  # a retried start must not run the job twice
        return {"status": "started", "id": job_id, "duplicate": True}
    if any(j["status"] == "running" for k, j in jobs.items() if k != "warm"):
        return {"status": "error", "error": "another job is still running on this runtime"}
    jobs[job_id] = {"status": "running", "stage": "starting", "started": time.time()}
    threading.Thread(target=_background, args=(job_id, model_id, job), daemon=True).start()
    return {"status": "started", "id": job_id}


def rw_warm(model_id: str) -> dict:
    """Start loading the model in the background and return at once (pre-warming)."""
    import threading
    if RW.get("pipe") is not None and RW.get("model_id") == model_id:
        return {"status": "ready"}
    warm = RW.setdefault("jobs", {}).get("warm")
    if warm and warm["status"] == "running":
        return {"status": "warming", "elapsed_s": round(time.time() - warm["started"])}
    entry = {"status": "running", "stage": "loading model", "started": time.time()}
    RW["jobs"]["warm"] = entry

    def run():
        try:
            entry["setup"] = rw_setup(model_id)
            entry["status"] = "done"
        except Exception as e:  # noqa: BLE001
            entry.update(status="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=run, daemon=True).start()
    return {"status": "warming", "elapsed_s": 0}


def rw_poll(job_id: str) -> dict:
    entry = RW.get("jobs", {}).get(job_id)
    if entry is None:
        return {"status": "error", "error": f"unknown job {job_id} (the runtime may have restarted)"}
    if entry["status"] == "done":
        return {"status": "done", **entry["result"], "setup": entry.get("setup")}
    if entry["status"] == "error":
        return {"status": "error", "error": entry["error"], "trace": entry.get("trace", "")}
    return {"status": "running", "stage": entry["stage"], "elapsed_s": round(time.time() - entry["started"])}


# ---------------------------------------------------------------- Hugging Face Spaces on Colab
# Any Gradio Space can run here: its code is downloaded, its requirements go into
# its own virtualenv (Spaces pin versions that clash with Colab), its app.py is
# started on a local port, and calls go to it through gradio_client on this machine.
# @spaces.GPU is a no-op outside Hugging Face, so the code runs unchanged.

SPACES_DIR = "/content/rw_spaces"


def _space_slug(space_id: str) -> str:
    return space_id.replace("/", "__")


def _free_wan():
    """Wan 2.2 takes ~69 GB of the GPU: unload it before a Space needs the GPU."""
    if RW.get("pipe") is not None:
        import torch
        RW["pipe"] = None
        RW["model_id"] = None
        gc.collect()
        torch.cuda.empty_cache()


# Settings some Spaces need off ZeroGPU (their defaults target H100/H200 hardware).
SPACE_ENV = {
    # flash_3 needs Hopper GPUs and is not a valid name in current diffusers; compile + warmup only slow the start
    "Tongyi-MAI/Z-Image-Turbo": {"ATTENTION_BACKEND": "native", "ENABLE_COMPILE": "false", "ENABLE_WARMUP": "false"},
}


def _stop_spaces():
    for sid, s in list(RW.get("spaces", {}).items()):
        proc = s.get("proc")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        RW["spaces"].pop(sid, None)


def _space_boot(space_id: str, port: int, entry: dict):
    import urllib.request
    from huggingface_hub import snapshot_download
    slug = _space_slug(space_id)
    root = f"{SPACES_DIR}/{slug}"
    venv = f"{SPACES_DIR}/{slug}-venv"
    try:
        entry["stage"] = "downloading the Space"
        snapshot_download(space_id, repo_type="space", local_dir=root)
        if not os.path.exists(f"{venv}/bin/python"):
            entry["stage"] = "installing its requirements (own virtualenv)"
            _pip("uv")
            subprocess.run([sys.executable, "-m", "uv", "venv", "--python", sys.executable, venv], check=True,
                           capture_output=True)
            req = f"{root}/requirements.txt"
            args = [sys.executable, "-m", "uv", "pip", "install", "--python", f"{venv}/bin/python", "gradio", "spaces"]
            if os.path.exists(req):
                args += ["-r", req]
            out = subprocess.run(args, capture_output=True, text=True)
            if out.returncode != 0:
                raise RuntimeError("requirements failed: " + out.stderr[-1200:])
        entry["stage"] = "starting the app (loads its models)"
        _free_wan()
        env = dict(os.environ, GRADIO_SERVER_NAME="127.0.0.1", GRADIO_SERVER_PORT=str(port),
                   GRADIO_ANALYTICS_ENABLED="False", PYTHONUNBUFFERED="1")
        env.update(SPACE_ENV.get(space_id, {}))
        log = open(f"{SPACES_DIR}/{slug}.log", "w")
        entry["proc"] = subprocess.Popen([f"{venv}/bin/python", "app.py"], cwd=root, env=env, stdout=log,
                                         stderr=subprocess.STDOUT)
        deadline = time.time() + 1800
        while time.time() < deadline:
            if entry["proc"].poll() is not None:
                raise RuntimeError("app.py exited: " + open(f"{SPACES_DIR}/{slug}.log").read()[-1500:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/gradio_api/info", timeout=5)
                entry.update(status="ready", stage="ready", url=f"http://127.0.0.1:{port}")
                return
            except Exception:
                time.sleep(5)
        raise RuntimeError("the app did not start within 30 minutes")
    except Exception as e:  # noqa: BLE001
        entry.update(status="error", error=f"{type(e).__name__}: {e}")


def rw_space_up(space_id: str) -> dict:
    """Start a Space on this runtime in the background (idempotent)."""
    import threading
    spaces = RW.setdefault("spaces", {})
    s = spaces.get(space_id)
    if s and s["status"] in ("ready", "starting"):
        if s["status"] == "ready" and s["proc"].poll() is not None:
            spaces.pop(space_id)
        else:
            return {"status": s["status"], "stage": s["stage"]}
    port = 7870 + len(spaces)
    entry = {"status": "starting", "stage": "queued", "started": time.time(), "port": port}
    spaces[space_id] = entry

    def run():
        _space_boot(space_id, port, entry)
        if entry["status"] == "starting":
            entry["status"] = "error"

    threading.Thread(target=run, daemon=True).start()
    return {"status": "starting", "stage": entry["stage"]}


def rw_space_status(space_id: str) -> dict:
    s = RW.get("spaces", {}).get(space_id)
    if not s:
        return {"status": "down"}
    out = {"status": s["status"], "stage": s["stage"], "elapsed_s": round(time.time() - s["started"])}
    if s["status"] == "error":
        out["error"] = s.get("error", "")
    return out


def _space_predict(job_id: str, space_id: str, api_name: str, kwargs: dict):
    import shutil
    entry = RW["jobs"][job_id]
    try:
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            _pip("gradio_client")
            from gradio_client import Client, handle_file
        in_dir = f"/content/rw_in/{job_id}"
        os.makedirs(in_dir, exist_ok=True)

        def inputs(v):  # {"__file_b64__": ..., "name": ...} -> uploaded file
            if isinstance(v, dict) and "__file_b64__" in v:
                p = f"{in_dir}/{v.get('name') or 'file'}"
                with open(p, "wb") as f:
                    f.write(base64.b64decode(v["__file_b64__"]))
                return handle_file(p)
            if isinstance(v, dict):
                return {k: inputs(x) for k, x in v.items()}
            if isinstance(v, list):
                return [inputs(x) for x in v]
            return v

        entry["stage"] = "running"
        client = Client(RW["spaces"][space_id]["url"], verbose=False)
        result = client.predict(api_name=api_name, **inputs(kwargs))
        os.makedirs(OUT_DIR, exist_ok=True)
        files = []

        def outputs(v):  # copy every result file to OUT_DIR and point the result there
            if isinstance(v, str) and os.path.isfile(v):
                dst = f"{OUT_DIR}/{uuid.uuid4().hex[:8]}-{os.path.basename(v)}"
                shutil.copy(v, dst)
                files.append(dst)
                return {"path": dst, "meta": {"_type": "gradio.FileData"}}
            if isinstance(v, dict):
                return {k: outputs(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [outputs(x) for x in v]
            return v

        entry["result"] = {"result": outputs(result), "files": files}
        entry["status"] = "done"
    except Exception as e:  # noqa: BLE001
        entry.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])


def rw_space_call(job_id: str, space_id: str, api_name: str, kwargs: dict) -> dict:
    import threading
    jobs = RW.setdefault("jobs", {})
    if job_id in jobs:
        return {"status": "started", "id": job_id, "duplicate": True}
    s = RW.get("spaces", {}).get(space_id)
    if not s or s["status"] != "ready":
        return {"status": "error", "error": f"{space_id} is not running on this runtime"}
    jobs[job_id] = {"status": "running", "stage": "starting", "started": time.time()}
    threading.Thread(target=_space_predict, args=(job_id, space_id, api_name, kwargs), daemon=True).start()
    return {"status": "started", "id": job_id}


def rw_call(action: str, **kw):
    """Entry point: one call per `colab exec`; always prints exactly one RW_RESULT line."""
    try:
        if action == "start":
            _emit(**rw_start(kw["job_id"], kw["model_id"], kw["job"]))
        elif action == "warm":
            _emit(**rw_warm(kw["model_id"]))
        elif action == "space_up":
            _emit(**rw_space_up(kw["space_id"]))
        elif action == "space_status":
            _emit(**rw_space_status(kw["space_id"]))
        elif action == "space_call":
            _emit(**rw_space_call(kw["job_id"], kw["space_id"], kw["api_name"], kw["kwargs"]))
        elif action == "space_stop":
            _stop_spaces()
            _emit(status="stopped")
        elif action == "poll":
            _emit(**rw_poll(kw["job_id"]))
        elif action == "setup":
            _emit(**rw_setup(kw["model_id"]))
        elif action == "generate":
            if RW.get("pipe") is None or RW.get("model_id") != kw["model_id"]:
                rw_setup(kw["model_id"])
            _emit(**rw_generate(kw["job"]))
        elif action == "status":
            import torch
            _emit(status="ok", model_id=RW.get("model_id"), mode=RW.get("mode"),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        else:
            _emit(status="error", error=f"unknown action {action}")
    except Exception as e:  # noqa: BLE001
        _emit(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
