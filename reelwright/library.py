"""The library of finished results, and the GPU-time meter."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

KINDS = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
         ".mp4": "video", ".webm": "video", ".mov": "video",
         ".wav": "audio", ".mp3": "audio", ".flac": "audio", ".m4a": "audio", ".ogg": "audio"}


def kind_of(path: str | Path) -> str:
    return KINDS.get(Path(path).suffix.lower(), "file")


@dataclass
class Item:
    id: str
    kind: str
    path: str
    title: str
    prompt: str = ""
    meta: dict = field(default_factory=dict)
    created: str = ""

    def caption(self) -> str:
        return f"{self.title} · {self.created[5:16].replace('T', ' ')}"


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.files = self.root / "files"
        self.files.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "library.jsonl"
        self._lock = threading.Lock()

    def add(self, src: str | Path, title: str, prompt: str = "", meta: dict | None = None) -> Item:
        src = Path(src)
        item_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        dest = self.files / f"{item_id}{src.suffix.lower()}"
        shutil.copy(src, dest)
        item = Item(id=item_id, kind=kind_of(dest), path=str(dest), title=title, prompt=prompt,
                    meta=meta or {}, created=dt.datetime.now().isoformat(timespec="seconds"))
        with self._lock, open(self.index, "a") as f:
            f.write(json.dumps(asdict(item)) + "\n")
        return item

    def items(self, kind: str | None = None, limit: int = 300) -> list[Item]:
        if not self.index.exists():
            return []
        out = []
        with self._lock:
            lines = self.index.read_text().splitlines()
        for line in reversed(lines):
            try:
                it = Item(**json.loads(line))
            except Exception:
                continue
            if Path(it.path).is_file() and (kind is None or it.kind == kind):
                out.append(it)
            if len(out) >= limit:
                break
        return out

    def get(self, item_id: str) -> Item | None:
        return next((i for i in self.items(limit=100000) if i.id == item_id), None)

    def find_by_path(self, path: str) -> Item | None:
        name = Path(path).name
        return next((i for i in self.items(limit=100000) if Path(i.path).name == name), None)


class UsageMeter:
    """Wall-clock time of Space calls per day. It overestimates GPU time
    (it includes queueing and upload), which is the safe direction."""

    def __init__(self, root: Path):
        self.path = Path(root) / "usage.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, space: str, api_name: str, seconds: float, ok: bool):
        row = {"day": dt.date.today().isoformat(), "space": space, "api": api_name,
               "seconds": round(seconds, 1), "ok": ok}
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def today(self) -> dict:
        day = dt.date.today().isoformat()
        per_space: dict[str, float] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("day") == day:
                    per_space[r["space"]] = per_space.get(r["space"], 0) + r["seconds"]
        return {"total_seconds": sum(per_space.values()), "per_space": per_space}
