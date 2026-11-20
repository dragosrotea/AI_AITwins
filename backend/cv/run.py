from __future__ import annotations

import argparse
import math
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class FakeDriverState:
    face_present: bool
    eye_closed_p: float
    eyes_closed_ms: int
    perclos_60s: float
    blink_rate_pm: float
    gaze_dispersion_30s: float
    yaw: float
    pitch: float
    roll: float
    yawn_p: float
    yawns_10m: int
    anger_p: float
    zoning_visual_p: float
    phone_in_hand: bool
    fps: float


def fake_sequences(sequence: str) -> Iterator[FakeDriverState]:
    base = FakeDriverState(
        face_present=True,
        eye_closed_p=0.02,
        eyes_closed_ms=0,
        perclos_60s=0.02,
        blink_rate_pm=18.0,
        gaze_dispersion_30s=3.0,
        yaw=1.0,
        pitch=-2.0,
        roll=0.0,
        yawn_p=0.02,
        yawns_10m=0,
        anger_p=0.02,
        zoning_visual_p=0.05,
        phone_in_hand=False,
        fps=30.0,
    )
    elapsed = 0
    while True:
        phase = (elapsed // 20) % 5
        requested = sequence
        if sequence == "all":
            requested = ("normal", "eyes_closing", "looking_down", "angry", "staring")[phase]

        state = base
        if requested == "eyes_closing":
            state = _replace(base, eye_closed_p=0.95, eyes_closed_ms=min(elapsed % 20 * 100, 1200), perclos_60s=0.18)
        elif requested == "looking_down":
            state = _replace(base, pitch=-28.0, gaze_dispersion_30s=2.0)
        elif requested == "angry":
            state = _replace(base, anger_p=0.92)
        elif requested == "staring":
            state = _replace(base, gaze_dispersion_30s=0.8, blink_rate_pm=5.0, zoning_visual_p=0.82)

        yield state
        elapsed += 1


def _replace(base: FakeDriverState, **changes: object) -> FakeDriverState:
    values = base.__dict__.copy()
    values.update(changes)
    return FakeDriverState(**values)


def fake_state(sequence: str = "normal") -> FakeDriverState:
    return next(fake_sequences(sequence))


def state_dict(state: FakeDriverState) -> dict[str, object]:
    return {
        "face_present": state.face_present,
        "eye_closed_p": state.eye_closed_p,
        "eyes_closed_ms": state.eyes_closed_ms,
        "perclos_60s": state.perclos_60s,
        "blink_rate_pm": state.blink_rate_pm,
        "gaze_dispersion_30s": state.gaze_dispersion_30s,
        "head": {"yaw": state.yaw, "pitch": state.pitch, "roll": state.roll},
        "yawn_p": state.yawn_p,
        "yawns_10m": state.yawns_10m,
        "anger_p": state.anger_p,
        "zoning_visual_p": state.zoning_visual_p,
        "phone_in_hand": state.phone_in_hand,
        "fps": state.fps,
    }


class CVPublisher:
    def __init__(self) -> None:
        self._publisher: object | None = None
        try:
            from backend.bus.client import ThreadPublisher

            self._publisher = ThreadPublisher("cv")
        except (ImportError, RuntimeError) as exc:
            self._broker_error = exc

    def start(self) -> None:
        if self._publisher is not None:
            start = getattr(self._publisher, "start", None)
            if callable(start):
                start()

    def publish(self, topic: str, data: dict[str, object]) -> None:
        if self._publisher is None:
            raise RuntimeError("Lead-owned backend/bus/client.py is unavailable")
        publish = getattr(self._publisher, "publish")
        publish(topic, data)

    def stop(self) -> None:
        if self._publisher is not None:
            stop = getattr(self._publisher, "stop", None)
            if callable(stop):
                stop()


def run_fake(sequence: str, broker_url: str) -> None:
    del broker_url
    publisher = CVPublisher()
    publisher.start()
    generator = fake_sequences(sequence)
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    interval = 1.0 / 10.0
    next_publish = time.monotonic()
    while running:
        state = next(generator)
        envelope = {"topic": "driver.state", "ts": time.time(), "src": "cv", "data": state_dict(state)}
        try:
            publisher.publish("driver.state", envelope["data"])
        except RuntimeError:
            print(envelope)
        next_publish += interval
        sleep_for = next_publish - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    publisher.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DriveMate computer-vision process")
    parser.add_argument("--fake", action="store_true", help="Publish synthetic driver state")
    parser.add_argument("--sequence", choices=("normal", "eyes_closing", "looking_down", "angry", "staring", "all"), default="all")
    parser.add_argument("--broker", default="ws://localhost:8765/bus")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fake:
        run_fake(args.sequence, args.broker)
        return 0

    raise RuntimeError("Live mode is not complete before CV-T05. Use --fake at CP0.")


if __name__ == "__main__":
    raise SystemExit(main())
