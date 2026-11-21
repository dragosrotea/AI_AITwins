from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.config import load_config
from backend.core.vehicle import ROOT, Player, load_trip, load_trips
from backend.schemas import validate

RATE = 10.0


def write_trip(folder: Path, seconds: int = 20) -> Path:
    """A small trip in the real UAH column layout (11 accel columns, 12 GPS columns).

    Accelerometer at ~10 Hz with a duplicate timestamp; Z (col 7) = 0.01 * t, Y (col 6) = -0.2.
    GPS at 1 Hz from t=1 with a no-fix first row; speed = 10 * t km/h, lat = 40 + t/1000.
    """
    folder.mkdir(parents=True)
    acc = np.zeros((seconds * 10 + 1, 11))
    acc[:, 0] = np.arange(len(acc)) / 10
    acc[:, 6] = -0.2
    acc[:, 7] = 0.01 * acc[:, 0]
    acc = np.insert(acc, 5, acc[5], axis=0)  # duplicate timestamp, as in the real files
    gps = np.zeros((seconds, 12))
    gps[:, 0] = np.arange(1, seconds + 1)
    gps[:, 1] = 10 * gps[:, 0]
    gps[:, 2] = 40 + gps[:, 0] / 1000
    gps[:, 3] = -3.3
    gps[:, 7] = 90.0
    gps[0, 1], gps[0, 7] = 0.0, -1  # before the first fix
    np.savetxt(folder / "RAW_ACCELEROMETERS.txt", acc, fmt="%.3f")
    np.savetxt(folder / "RAW_GPS.txt", gps, fmt="%.6f")
    return folder


def test_load_resamples_to_even_grid_and_maps_columns(tmp_path: Path) -> None:
    trip = load_trip("t", write_trip(tmp_path / "x-NORMAL-MOTORWAY"), RATE, gps_lag_s=0.0)
    assert np.allclose(np.diff(trip.t), 0.1)
    assert trip.t[0] == 0.0
    # The grid starts at the first GPS fix (t=2 s in file time, the no-fix row is dropped).
    assert trip.long_g[0] == pytest.approx(0.02)  # Z col 7 at t=2 s
    assert trip.lat_g[0] == pytest.approx(-0.2)  # Y col 6
    assert trip.speed_kmh[0] == pytest.approx(20.0)
    assert trip.speed_kmh[10] == pytest.approx(30.0)  # interpolated 1 Hz -> 10 Hz
    assert trip.lat[5] == pytest.approx(40.0025)
    assert trip.road == "motorway"


def test_gps_lag_moves_speed_earlier(tmp_path: Path) -> None:
    folder = write_trip(tmp_path / "x-DROWSY-SECONDARY")
    trip = load_trip("t", folder, RATE, gps_lag_s=1.0)
    # GPS file time 2 s now lines up with accelerometer time 1 s.
    assert trip.long_g[0] == pytest.approx(0.01)
    assert trip.speed_kmh[0] == pytest.approx(20.0)
    assert trip.road == "secondary"


def test_samples_match_contract(tmp_path: Path) -> None:
    trip = load_trip("t", write_trip(tmp_path / "x-NORMAL-MOTORWAY"), RATE, 0.0)
    for i in range(len(trip)):
        sample = trip.sample(i)
        validate("vehicle.motion", sample)
    assert sample["source"] == "replay"
    assert sample["accel_mag_g"] == pytest.approx(np.hypot(sample["accel_long_g"], -0.2), abs=1e-3)


def make_player(tmp_path: Path, loop: bool = True) -> Player:
    trips = {
        name: load_trip(name, write_trip(tmp_path / f"{name}-MOTORWAY", seconds), RATE, 0.0)
        for name, seconds in (("normal", 20), ("drowsy", 10))
    }
    return Player(trips, "normal", loop)


def test_play_pause_and_seek(tmp_path: Path) -> None:
    player = make_player(tmp_path)
    first = player.step()
    assert player.index == 1
    player.control({"action": "pause"})
    held = [player.step() for _ in range(3)]
    assert player.index == 1 and held[0] == held[2]
    player.control({"action": "play"})
    player.step()
    assert player.index == 2
    player.control({"action": "seek", "segment": 5.0})
    assert player.position_s == pytest.approx(5.0)
    player.control({"action": "seek", "segment": 999})
    assert player.index == len(player.trip) - 1
    assert first["speed_kmh"] == pytest.approx(20.0)


def test_trip_switch_and_unknown_trip(tmp_path: Path) -> None:
    player = make_player(tmp_path)
    player.control({"action": "seek", "segment": 4.0})
    player.control({"action": "play", "trip": "drowsy"})
    assert player.trip.name == "drowsy" and player.index == 0
    player.control({"action": "seek", "trip": "normal", "segment": 3.0})
    assert player.trip.name == "normal" and player.position_s == pytest.approx(3.0)
    player.control({"action": "pause", "trip": "nope"})
    assert player.trip.name == "normal" and player.playing  # ignored entirely


def test_end_of_trip_loops_or_stops(tmp_path: Path) -> None:
    player = make_player(tmp_path, loop=True)
    player.index = len(player.trip) - 1
    player.step()
    assert player.index == 0 and player.playing
    player = make_player(tmp_path / "b", loop=False)
    player.index = len(player.trip) - 1
    player.step()
    assert player.index == len(player.trip) - 1 and not player.playing


def test_missing_dataset_loads_nothing(tmp_path: Path) -> None:
    cfg = load_config("default")["core"]["vehicle"]
    assert load_trips(cfg, root=tmp_path) == {}
    with pytest.raises(ValueError):
        Player({}, "normal")


REAL = ROOT / "data" / "UAH-DRIVESET-v1"


@pytest.mark.skipif(not REAL.exists(), reason="UAH-DriveSet not downloaded")
def test_real_trips_look_like_driving() -> None:
    trips = load_trips(load_config("default")["core"]["vehicle"])
    assert set(trips) == {"normal", "drowsy", "aggressive"}
    for trip in trips.values():
        assert 10 * 60 < trip.duration_s < 20 * 60
        assert 90 < trip.speed_kmh.max() < 160  # motorway, km/h
        assert -0.6 < trip.long_g.min() < -0.1  # real braking in g (docs/datasets.md)
        assert abs(trip.lat - 40.45).max() < 0.2  # around Alcalá de Henares
    # T07: the drowsy and aggressive picks are the trips with braking beyond 0.4 g.
    assert trips["drowsy"].long_g.min() < -0.4
    assert trips["aggressive"].long_g.min() < -0.4
