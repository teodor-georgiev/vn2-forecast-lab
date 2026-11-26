"""Atomic progress snapshots for long-running VN2 experiments."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


class Progress:
    """Single-writer status file; readers never see a partially written JSON."""
    def __init__(self, path):
        self.path = Path(path)
        self.started = time.monotonic()
        self.stage_started = self.started
        self.data = {}

    def __enter__(self):
        self.update("initializing", 0, 1)
        return self

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".progress-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, allow_nan=False, indent=2)
                f.write("\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def update(self, stage, completed, total, **metrics):
        if not 0 <= completed <= total or total <= 0:
            raise ValueError("invalid progress counts")
        now = time.monotonic()
        if stage != self.data.get("stage"):
            self.stage_started = now
        elapsed = now - self.stage_started
        self.data = dict(metrics, status="running", stage=stage, completed=completed,
                         total=total, percent=round(100 * completed / total, 2),
                         elapsed_seconds=round(now - self.started, 2),
                         stage_eta_seconds=round(elapsed * (total - completed) / completed, 2)
                         if completed else None,
                         updated_at_unix=time.time())
        self._write()

    def __exit__(self, kind, value, traceback):
        self.data.update(status="failed" if kind else "completed",
                         updated_at_unix=time.time(), stage_eta_seconds=None,
                         elapsed_seconds=round(time.monotonic() - self.started, 2))
        if kind:
            self.data["error_type"] = kind.__name__
        self._write()
        return False
