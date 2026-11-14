"""Calls to Hugging Face Spaces with retries, timing and clear error messages."""
from __future__ import annotations

import base64
import copy
import json
import logging
import mimetypes
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import httpx

log = logging.getLogger("reelwright")


class SpaceError(RuntimeError):
    """An error with a kind the app can explain to the user."""

    def __init__(self, kind: str, message: str, space: str | None = None):
        super().__init__(message)
        self.kind = kind  # quota | busy | offline | blocked | invalid | failed
        self.space = space

    def user_message(self) -> str:
        where = f" ({self.space})" if self.space else ""
        return {
            "quota": f"Out of GPU quota{where}. Your daily ZeroGPU time is used up or too low for this job. Try a shorter clip, fewer variants, or wait until the quota resets.",
            "busy": f"The Space is busy{where}. It was retried and is still busy; try again in a minute.",
            "offline": f"The Space is offline, sleeping or unreachable{where}. Try again later or pick another model.",
            "blocked": f"The Space refused this request{where}: {self}",
            "invalid": f"The request was not accepted{where}: {self}",
        }.get(self.kind, f"The Space failed{where}: {self}")


RETRYABLE = {"busy", "offline", "failed"}
_QUOTA = ("quota", "you have exceeded", "zerogpu", "gpu task aborted")
_BUSY = ("queue is full", "too many requests", "429", "rate limit")
_INVALID = ("not in the list of choices", "no value provided for required argument", "please upload",
            "please enter")
_OFFLINE = ("could not fetch config", "is sleeping", "paused", "503", "502", "connecterror",
            "connection", "timed out", "timeout", "remoteprotocolerror", "403 forbidden", "404")


def classify(exc: BaseException, space: str | None = None) -> SpaceError:
    if isinstance(exc, SpaceError):
        return exc
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(h in text for h in _QUOTA):
        return SpaceError("quota", str(exc) or type(exc).__name__, space)
    if any(h in text for h in _BUSY):
        return SpaceError("busy", str(exc) or type(exc).__name__, space)
    if (isinstance(exc, (TypeError, ValueError)) and "upstream" not in text) or any(h in text for h in _INVALID):
        return SpaceError("invalid", str(exc), space)
    if any(h in text for h in _OFFLINE):
        return SpaceError("offline", str(exc) or type(exc).__name__, space)
    return SpaceError("failed", str(exc) or type(exc).__name__, space)


@dataclass(frozen=True)
class LocalFile:
    """A file on this machine that should be sent to a Space."""
    path: str


_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class RemoteError(RuntimeError):
    """The Space itself reported an error."""


class RestClient:
    """A small client for Gradio's REST API (/gradio_api/call/...).

    Why not gradio_client: Spaces with several replicas lose uploaded files
    when the job runs on another replica. Here images are sent inline as
    base64 (no upload at all), other files are uploaded over the same kept-alive
    connection as the job, and a job that fails fast right after an upload is
    re-uploaded and retried."""

    FAST_FAIL_SECONDS = 8
    UPLOAD_RETRIES = 5

    def __init__(self, space_id: str, token: str | None = None, http: httpx.Client | None = None):
        self.space_id = space_id
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = http or httpx.Client(timeout=httpx.Timeout(900, connect=30), headers=headers,
                                         follow_redirects=True)
        self.base = self._host() + "/gradio_api"
        r = self.http.get(f"{self.base}/info")
        if r.status_code != 200:
            raise RemoteError(f"Could not fetch config ({r.status_code}) for {space_id}")
        self.endpoints = r.json().get("named_endpoints", {})

    def _host(self) -> str:
        try:
            r = self.http.get(f"https://huggingface.co/api/spaces/{self.space_id}")
            if r.status_code == 200 and r.json().get("host"):
                return r.json()["host"].rstrip("/")
        except httpx.HTTPError:
            pass
        sub = self.space_id.lower().replace("/", "-").replace("_", "-").replace(".", "-")
        return f"https://{sub}.hf.space"

    # -- inputs
    INLINE_AFTER = 3  # after this many fast failures, send images inline instead of uploading

    def _prepare(self, v: Any, uploads: list, inline_images: bool = False) -> Any:
        if isinstance(v, LocalFile):
            p = Path(v.path)
            mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            if inline_images and p.suffix.lower() in _IMAGE_EXT:
                return {"url": to_data_uri(p), "path": None, "orig_name": p.name, "mime_type": mime,
                        "meta": {"_type": "gradio.FileData"}}
            with open(p, "rb") as f:
                r = self.http.post(f"{self.base}/upload", files={"files": (p.name, f, mime)})
            if r.status_code != 200:
                raise RemoteError(f"Upload failed ({r.status_code}): {r.text[:200]}")
            uploads.append(p.name)
            return {"path": r.json()[0], "orig_name": p.name, "mime_type": mime, "meta": {"_type": "gradio.FileData"}}
        if isinstance(v, dict):
            return {k: self._prepare(x, uploads, inline_images) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._prepare(x, uploads, inline_images) for x in v]
        return v

    def _fix_urls(self, v: Any) -> Any:
        """Rebuild file URLs from the server path: some Spaces return broken URLs
        (e.g. '/gradio_a/gradio_api/file=...')."""
        if isinstance(v, dict):
            v = {k: self._fix_urls(x) for k, x in v.items()}
            if isinstance(v.get("path"), str) and v["path"].startswith("/") and "url" in v:
                v["url"] = f"{self.base}/file={v['path']}"
            return v
        if isinstance(v, list):
            return [self._fix_urls(x) for x in v]
        return v

    def predict(self, api_name: str, **kwargs) -> Any:
        ep = self.endpoints.get(api_name)
        if ep is None:
            raise ValueError(f"{self.space_id} has no endpoint {api_name}")
        for attempt in range(self.UPLOAD_RETRIES + 1):
            uploads: list = []
            data = []
            inline = attempt >= self.INLINE_AFTER
            for p in ep.get("parameters", []):
                name = p["parameter_name"]
                if name in kwargs:
                    data.append(self._prepare(kwargs[name], uploads, inline))
                elif p.get("parameter_has_default"):
                    data.append(p.get("parameter_default"))
                else:
                    raise ValueError(f"No value provided for required argument: {name}")
            start = time.monotonic()
            try:
                return self._run(api_name, data)
            except RemoteError:
                quick = time.monotonic() - start < self.FAST_FAIL_SECONDS
                if (uploads or inline) and quick and attempt < self.UPLOAD_RETRIES:
                    log.info("%s: fast failure after upload, re-uploading (likely another replica)", self.space_id)
                    continue
                raise

    def _run(self, api_name: str, data: list) -> Any:
        name = api_name.lstrip("/")
        r = self.http.post(f"{self.base}/call/{name}", json={"data": data})
        if r.status_code != 200:
            raise RemoteError(f"HTTP {r.status_code}: {r.text[:300]}")
        event_id = r.json()["event_id"]
        event = None
        with self.http.stream("GET", f"{self.base}/call/{name}/{event_id}") as s:
            if s.status_code != 200:
                raise RemoteError(f"HTTP {s.status_code} while waiting for the result")
            for line in s.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event in ("complete", "error"):
                    payload = line[5:].strip()
                    try:
                        value = json.loads(payload)
                    except json.JSONDecodeError:
                        value = payload
                    if event == "error":
                        raise RemoteError(str(value) if value else "The Space reported an error without details")
                    value = self._fix_urls(value)
                    if isinstance(value, list):
                        return value[0] if len(value) == 1 else tuple(value)
                    return value
        raise RemoteError("The connection to the Space closed before the result arrived")


class GradioClientAdapter:
    """Uses the official gradio_client. Needed for Spaces that rely on per-session
    state set up when the page loads (e.g. OmniAvatar), which the REST API skips."""

    http = None

    def __init__(self, space_id: str, token: str | None = None):
        from gradio_client import Client
        self.space_id = space_id
        self._c = Client(space_id, token=token, verbose=False, download_files=False)

    def _conv(self, v: Any) -> Any:
        from gradio_client import handle_file
        if isinstance(v, LocalFile):
            return handle_file(v.path)
        if isinstance(v, dict):
            return {k: self._conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._conv(x) for x in v]
        return v

    def predict(self, api_name: str, **kwargs) -> Any:
        return self._c.predict(api_name=api_name, **self._conv(kwargs))


def _default_client_factory(space_id: str, token: str | None):
    from . import config
    if space_id in config.GRADIO_CLIENT_SPACES:
        return GradioClientAdapter(space_id, token)
    return RestClient(space_id, token)


class SpaceCaller:
    """Calls one Space. Reconnects and retries on transient failures."""

    def __init__(self, space_id: str, token: str | None = None, retries: int = 3,
                 client_factory: Callable | None = None,
                 on_timing: Callable[[str, str, float, bool], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep, backoff=(3, 10, 20)):
        self.space_id = space_id
        self.token = token
        self.retries = retries
        self._factory = client_factory or _default_client_factory
        self._client = None
        self._on_timing = on_timing
        self._sleep = sleep
        self._backoff = backoff

    def client(self):
        if self._client is None:
            try:
                self._client = self._factory(self.space_id, self.token)
            except Exception as e:  # config fetch failures etc.
                raise classify(e, self.space_id) from e
        return self._client

    def call(self, api_name: str, **kwargs) -> Any:
        last: SpaceError | None = None
        for attempt in range(self.retries + 1):
            start = time.monotonic()
            try:
                # Fresh copy every attempt: gradio_client rewrites uploaded-file
                # dicts in place to server paths, which breaks a retry.
                result = self.client().predict(api_name=api_name, **copy.deepcopy(kwargs))
                self._timing(api_name, time.monotonic() - start, True)
                return result
            except Exception as e:  # noqa: BLE001 - converted below
                self._timing(api_name, time.monotonic() - start, False)
                last = classify(e, self.space_id)
                log.warning("%s %s attempt %d failed: %s", self.space_id, api_name, attempt + 1, last)
                if last.kind not in RETRYABLE or attempt == self.retries:
                    raise last from e
                self._reset()  # reconnect on the next attempt
                self._sleep(self._backoff[min(attempt, len(self._backoff) - 1)])
        raise last  # pragma: no cover

    def _reset(self):
        http = getattr(self._client, "http", None)
        if http is not None:
            try:
                http.close()
            except Exception:
                pass
        self._client = None

    def _timing(self, api_name: str, seconds: float, ok: bool):
        if self._on_timing:
            try:
                self._on_timing(self.space_id, api_name, seconds, ok)
            except Exception:  # never let metering break a call
                log.exception("timing callback failed")


# ---------------------------------------------------------------- files

def _is_file_ref(obj: Any) -> bool:
    return isinstance(obj, dict) and (isinstance(obj.get("url"), str) or isinstance(obj.get("path"), str)) \
        and not any(k in obj for k in ("image", "video", "audio"))


def file_refs(obj: Any) -> list:
    """All file references in a Space result, in order (galleries, tuples, dicts)."""
    found: list = []
    if _is_file_ref(obj):
        found.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(file_refs(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found.extend(file_refs(v))
    elif isinstance(obj, str) and (obj.startswith("http") or Path(obj).is_file()):
        found.append(obj)
    return found


def _guess_ext(ref: Any, url: str | None, default: str) -> str:
    for cand in (ref.get("orig_name") if isinstance(ref, dict) else None,
                 urlparse(url).path if url else None,
                 ref.get("path") if isinstance(ref, dict) else (ref if isinstance(ref, str) else None)):
        if cand:
            ext = Path(unquote(cand)).suffix
            if 1 < len(ext) <= 5:
                return ext.lower()
    if isinstance(ref, dict) and ref.get("mime_type"):
        ext = mimetypes.guess_extension(ref["mime_type"])
        if ext:
            return ext
    return default


def fetch(ref: Any, dest_dir: Path, token: str | None = None, default_ext: str = ".bin",
          retries: int = 8, sleep: Callable[[float], None] = time.sleep,
          http: httpx.Client | None = None) -> Path:
    """Download (or copy) one result file into dest_dir and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = None
    if isinstance(ref, dict):
        url = ref.get("url")
        local = ref.get("path")
        if not url and local and Path(local).is_file():
            url = None
        elif not url and local and local.startswith("http"):
            url = local
    elif isinstance(ref, str):
        local = ref
        url = ref if ref.startswith("http") else None
    else:
        raise SpaceError("failed", f"Unexpected result: {ref!r:.200}")

    dest = dest_dir / f"{uuid.uuid4().hex[:12]}{_guess_ext(ref, url, default_ext)}"
    if not url:
        if local and Path(local).is_file():
            shutil.copy(local, dest)
            return dest
        raise SpaceError("failed", f"Result file not found: {ref!r:.200}")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    own = http is None
    http = http or httpx.Client(timeout=180, follow_redirects=True)
    try:
        last = None
        for attempt in range(retries):
            try:
                with http.stream("GET", url, headers=headers) as r:
                    if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
                        last = "got a web page instead of the file"
                    elif r.status_code == 200:
                        with open(dest, "wb") as f:
                            for chunk in r.iter_bytes():
                                f.write(chunk)
                        if dest.stat().st_size > 0:
                            return dest
                        last = "empty file"
                    else:
                        last = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
            sleep(1.5)
        raise SpaceError("offline", f"Could not download the result ({last}) from {url}")
    finally:
        if own:
            http.close()


def save_data_uri(data_uri: str, dest_dir: Path) -> Path:
    """Save a base64 image (data URI or raw base64) returned by a Space."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    header, _, payload = data_uri.partition(",") if data_uri.startswith("data:") else ("", "", data_uri)
    ext = ".png"
    if header:
        mime = header[5:].split(";")[0]
        ext = mimetypes.guess_extension(mime) or ".png"
    dest = dest_dir / f"{uuid.uuid4().hex[:12]}{ext}"
    dest.write_bytes(base64.b64decode(payload))
    return dest


def to_data_uri(path: str | Path) -> str:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
