"""Background jobs: every generation runs as a job with a status and a log,
so the app stays responsive and several things can be queued at once."""
from __future__ import annotations

import datetime as dt
import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .library import Item
from .spaces import SpaceError
from .media import MediaError


@dataclass
class Job:
    id: str
    title: str
    status: str = "queued"  # queued | running | done | failed
    created: str = ""
    started: str = ""
    finished: str = ""
    logs: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    error: str = ""

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")

    def log(self, msg: str):
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"{stamp}  {msg}")

    def seconds(self) -> int:
        if not self.started:
            return 0
        end = dt.datetime.fromisoformat(self.finished) if self.finished else dt.datetime.now()
        return int((end - dt.datetime.fromisoformat(self.started)).total_seconds())


class JobQueue:
    def __init__(self, root: Path | None = None, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._history = Path(root) / "jobs.jsonl" if root else None

    def submit(self, title: str, fn: Callable[[Job], list[Item]]) -> Job:
        job = Job(id=uuid.uuid4().hex[:8], title=title, created=dt.datetime.now().isoformat(timespec="seconds"))
        job.log("Queued")
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn):
        job.status = "running"
        job.started = dt.datetime.now().isoformat(timespec="seconds")
        job.log("Started")
        try:
            items = fn(job) or []
            job.items = list(items)
            job.status = "done"
            job.log(f"Done: {len(job.items)} result(s)")
        except SpaceError as e:
            job.status, job.error = "failed", e.user_message()
            job.log(f"Failed: {job.error}")
        except MediaError as e:
            job.status, job.error = "failed", f"Video/audio processing failed: {e}"
            job.log(f"Failed: {job.error}")
        except Exception as e:  # noqa: BLE001
            msg = e.user_message() if hasattr(e, "user_message") else f"Unexpected error: {type(e).__name__}: {e}"
            job.status, job.error = "failed", msg
            job.log(job.error)
            job.log(traceback.format_exc(limit=4))
        finally:
            job.finished = dt.datetime.now().isoformat(timespec="seconds")
            self._save(job)

    def _save(self, job: Job):
        if not self._history:
            return
        row = {"id": job.id, "title": job.title, "status": job.status, "created": job.created,
               "seconds": job.seconds(), "error": job.error, "items": [i.id for i in job.items]}
        try:
            with self._lock, open(self._history, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
