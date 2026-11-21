"""Trip log (LEAD-T12, AL-04): every event, alert, spoken text, reply, intent and latency.

One JSON object per line in logs/trip_<YYYYMMDD_HHMMSS>.jsonl, local only (logs/ is
git-ignored, NF-13). Each line is the bus envelope as it went over the bus ({topic, ts, src,
data}), so timestamps and senders are kept. Core also writes its own `log.*` lines, for
example `log.alert_created` with the trigger signals, which alert.state does not carry.

A file starts with the first line written; a control.trip start begins a new file and a
control.trip stop ends the current one. Lines are flushed one by one, so a crash loses nothing.
camera.frame is never written, whatever the config says: no raw video is ever stored.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

log = logging.getLogger("core.triplog")

ROOT = Path(__file__).resolve().parents[2]
NEVER_LOGGED = frozenset({"camera.frame"})


class TripLog:
    def __init__(
        self,
        cfg: dict[str, Any],
        root: Path = ROOT,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.dir = root / cfg["dir"]
        self.topics = [t for t in cfg["topics"] if t not in NEVER_LOGGED]
        self.clock = clock
        self.path: Path | None = None
        self._file: IO[str] | None = None

    def write(self, envelope: dict[str, Any]) -> None:
        """Log one bus envelope; control.trip start and stop open and close trip files."""
        topic = envelope.get("topic", "")
        if topic in NEVER_LOGGED:
            return
        trip = envelope.get("data", {}).get("action") if topic == "control.trip" else None
        if trip == "start":
            self.close()
        self._line(envelope)
        if trip == "stop":
            self.close()

    def note(self, kind: str, data: dict[str, Any]) -> None:
        """Log core's own record, such as log.alert_created."""
        self._line({"topic": f"log.{kind}", "ts": self.clock(), "src": "core", "data": data})

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            log.info("trip log closed: %s", self.path)
        self._file = None

    def _line(self, record: dict[str, Any]) -> None:
        if self._file is None:
            self._open()
        assert self._file is not None
        self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._file.flush()

    def _open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.clock()))
        path = self.dir / f"trip_{stamp}.jsonl"
        n = 1
        while path.exists():  # two trips in the same second
            n += 1
            path = self.dir / f"trip_{stamp}_{n}.jsonl"
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        log.info("trip log: %s", path)
