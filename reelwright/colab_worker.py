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


def _resize(image):
    """Same rule as the Wan 2.2 Lightning Space: 832x480 / 480x832 / 640x640 class sizes."""
    from PIL import Image
    width, height = image.size
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)
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
    image = _resize(Image.open(io.BytesIO(base64.b64decode(job["image_b64"]))).convert("RGB"))
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


def rw_call(action: str, **kw):
    """Entry point: one call per `colab exec`; always prints exactly one RW_RESULT line."""
    try:
        if action == "start":
            _emit(**rw_start(kw["job_id"], kw["model_id"], kw["job"]))
        elif action == "warm":
            _emit(**rw_warm(kw["model_id"]))
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
