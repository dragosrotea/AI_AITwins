from __future__ import annotations

import pytest

from backend.config import load_config
from backend.core.vehicle import Injector
from backend.schemas import validate

CFG = load_config("default")["core"]["inject"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def drive(speed: float = 108.0) -> dict:
    return {
        "accel_long_g": 0.05,
        "accel_lat_g": 0.1,
        "accel_mag_g": 0.112,
        "speed_kmh": speed,
        "lat": 40.5,
        "lon": -3.4,
        "road": "motorway",
        "source": "replay",
    }


def make() -> tuple[Injector, FakeClock]:
    clock = FakeClock()
    return Injector(CFG, clock), clock


def test_no_event_passes_samples_through() -> None:
    injector, _ = make()
    assert injector.apply(drive()) == drive()


def test_harsh_brake_overlays_for_its_duration() -> None:
    injector, clock = make()
    injector.inject("harsh_brake")
    out = injector.apply(drive())
    assert out["accel_long_g"] == pytest.approx(0.05 + CFG["harsh_brake_g"])
    assert out["accel_mag_g"] > 0.4  # crosses the DB-02 harsh marker
    assert out["speed_kmh"] == 108.0  # speed left alone
    clock.advance(CFG["harsh_brake_s"] - 0.01)
    assert injector.apply(drive())["accel_long_g"] < -0.4
    clock.advance(0.02)
    assert injector.apply(drive())["accel_long_g"] == pytest.approx(0.05)


def test_crash_spike_then_ramp_to_zero_then_stopped() -> None:
    injector, clock = make()
    injector.inject("crash")
    spike = injector.apply(drive(108.0))
    assert spike["accel_long_g"] == -CFG["crash_g"]
    assert spike["accel_mag_g"] >= 4.0  # CR-01 crash candidate threshold
    clock.advance(CFG["crash_spike_s"] + 0.01)
    ramp = injector.apply(drive(110.0) | {"lat": 40.6})
    assert -1.5 < ramp["accel_long_g"] < -0.5  # 108 km/h to 0 in 3 s is about -1 g
    assert 0 < ramp["speed_kmh"] < 108.0
    assert ramp["lat"] == 40.5  # position frozen at the impact point
    clock.advance(CFG["crash_stop_s"] / 2 - CFG["crash_spike_s"])
    assert injector.apply(drive())["speed_kmh"] == pytest.approx(54.0, abs=1.5)
    clock.advance(CFG["crash_stop_s"])
    for _ in range(3):
        stopped = injector.apply(drive())
        clock.advance(10)
    assert stopped["speed_kmh"] == 0.0 and stopped["accel_mag_g"] == 0.0
    assert injector.crashed


def test_clear_resumes_driving() -> None:
    injector, clock = make()
    injector.inject("crash")
    injector.apply(drive())
    clock.advance(5)
    injector.clear()
    assert not injector.crashed
    assert injector.apply(drive()) == drive()


def test_samples_stay_inside_the_contract() -> None:
    injector, clock = make()
    for event in ("harsh_brake", "crash"):
        injector.inject(event)
        for _ in range(60):
            validate("vehicle.motion", injector.apply(drive()))
            clock.advance(0.1)
        injector.clear()
