"""Listen on the bus and check every topic against the contract (LEAD-T06).

    python scripts/smoke_test.py                 # 10 s, all periodic topics required
    python scripts/smoke_test.py --ignore bio.hr --ignore context.road

For each topic: measured rate against the expected rate in backend/schemas.py TOPICS, messages
that fail schema validation, and senders whose src is not a listed producer. Exits 1 when a
periodic topic is missing or under 70 % of its rate, or any message breaks the contract.
On-event topics are reported but never required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import websockets
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.bus.client import MAX_MESSAGE_BYTES, bus_url  # noqa: E402
from backend.schemas import TOPICS, Envelope, validate  # noqa: E402

MIN_RATE_SHARE = 0.7


@dataclass
class TopicReport:
    count: int = 0
    invalid: int = 0
    first_error: str = ""
    srcs: Counter[str] = field(default_factory=Counter)

    def record_error(self, error: str) -> None:
        self.invalid += 1
        if not self.first_error:
            self.first_error = error


def check(raw: str, reports: dict[str, TopicReport]) -> None:
    try:
        envelope = Envelope.model_validate(json.loads(raw))
    except (ValueError, ValidationError) as error:
        reports.setdefault("<bad envelope>", TopicReport()).record_error(str(error)[:200])
        return
    report = reports.setdefault(envelope.topic, TopicReport())
    report.count += 1
    report.srcs[envelope.src] += 1
    spec = TOPICS.get(envelope.topic)
    if spec is None:
        report.record_error("topic is not in the contract")
        return
    if envelope.src not in spec.producers:
        report.record_error(f"src {envelope.src!r} is not a producer {spec.producers}")
    try:
        validate(envelope.topic, envelope.data)
    except ValidationError as error:
        first = error.errors()[0]
        loc = ".".join(str(part) for part in first["loc"])
        report.record_error(f"{loc}: {first['msg']}")


def evaluate(
    reports: dict[str, TopicReport], seconds: float, ignore: set[str]
) -> tuple[list[str], bool]:
    """Return printable lines and whether the run passed."""
    lines = [f"{'topic':<18} {'expected':>9} {'measured':>9} {'msgs':>6}  status"]
    passed = True
    names = sorted(set(TOPICS) | set(reports))
    for name in names:
        report = reports.get(name, TopicReport())
        spec = TOPICS.get(name)
        expected = spec.rate_hz if spec else None
        rate = report.count / seconds
        if name in ignore:
            status = "ignored"
        elif report.invalid:
            status = f"FAIL {report.invalid} invalid: {report.first_error}"
            passed = False
        elif expected is None:
            status = "ok" if report.count else "-- (on event, not seen)"
        elif report.count == 0:
            status = "FAIL missing"
            passed = False
        elif rate < MIN_RATE_SHARE * expected:
            status = f"FAIL slow (< {MIN_RATE_SHARE:.0%})"
            passed = False
        else:
            status = "ok"
        exp = f"{expected:g} Hz" if expected else "event"
        src = ",".join(sorted(report.srcs))
        lines.append(f"{name:<18} {exp:>9} {rate:>7.1f}Hz {report.count:>6}  {status} {src}")
    return lines, passed


async def listen(url: str, seconds: float) -> dict[str, TopicReport]:
    reports: dict[str, TopicReport] = {}
    async with websockets.connect(url, max_size=MAX_MESSAGE_BYTES) as ws:
        await ws.send(json.dumps({"op": "sub", "topics": ["*"]}))
        loop = asyncio.get_running_loop()
        end = loop.time() + seconds
        while (left := end - loop.time()) > 0:
            try:
                raw = await asyncio.wait_for(ws.recv(), left)
            except TimeoutError:
                break
            check(raw if isinstance(raw, str) else raw.decode(), reports)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="DriveMate bus smoke test")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--url", default=bus_url())
    parser.add_argument("--ignore", action="append", default=[], help="topic to skip")
    args = parser.parse_args()
    try:
        reports = asyncio.run(listen(args.url, args.seconds))
    except OSError as error:
        print(f"cannot reach the broker at {args.url}: {error}")
        return 1
    lines, passed = evaluate(reports, args.seconds, set(args.ignore))
    print("\n".join(lines))
    print("SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
