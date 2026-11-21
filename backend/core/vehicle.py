"""Vehicle replay from UAH-DriveSet (LEAD-T08): vehicle.motion at 10 Hz with source "replay".

Column meanings and units were checked against the files (docs/datasets.md):
- RAW_ACCELEROMETERS.txt, ~10 Hz: col 0 time (s), col 7 Kalman-filtered Z = longitudinal g
  (positive = speeding up, negative = braking), col 6 filtered Y = lateral g (positive when the
  GPS course turns clockwise, i.e. turning right).
- RAW_GPS.txt, 1 Hz: col 0 time (s), col 1 speed (km/h), cols 2-3 lat/lon (degrees),
  col 7 course (-1 before the first fix). GPS trails the accelerometer by about gps_lag_s.

Everything is resampled onto an even grid at rate_hz. The Player obeys control.replay
(play, pause, seek to `segment` seconds, switch `trip`). With no dataset on disk, core falls
back to the synthetic drive so main always runs.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("core.vehicle")

ROOT = Path(__file__).resolve().parents[2]

# Column indices (docs/datasets.md)
ACC_T, ACC_LAT, ACC_LONG = 0, 6, 7
GPS_T, GPS_SPEED, GPS_LAT, GPS_LON, GPS_COURSE = 0, 1, 2, 3, 7


@dataclass
class Trip:
    name: str
    road: str  # "motorway" or "secondary", from the folder name
    t: np.ndarray  # seconds from trip start, even steps
    long_g: np.ndarray
    lat_g: np.ndarray
    speed_kmh: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    def __len__(self) -> int:
        return len(self.t)

    def sample(self, i: int) -> dict[str, Any]:
        long_g, lat_g = float(self.long_g[i]), float(self.lat_g[i])
        return {
            "accel_long_g": round(long_g, 3),
            "accel_lat_g": round(lat_g, 3),
            "accel_mag_g": round(math.hypot(long_g, lat_g), 3),
            "speed_kmh": round(max(float(self.speed_kmh[i]), 0.0), 1),
            "lat": round(float(self.lat[i]), 6),
            "lon": round(float(self.lon[i]), 6),
            "road": self.road,
            "source": "replay",
        }


def _unique_times(data: np.ndarray) -> np.ndarray:
    """Keep the first row of each timestamp, sorted (the files have a few duplicates)."""
    _, first = np.unique(data[:, 0], return_index=True)
    return data[np.sort(first)]


def load_trip(name: str, folder: Path, rate_hz: float, gps_lag_s: float) -> Trip:
    acc = _unique_times(np.loadtxt(folder / "RAW_ACCELEROMETERS.txt", ndmin=2))
    gps = np.loadtxt(folder / "RAW_GPS.txt", ndmin=2)
    gps = _unique_times(gps[gps[:, GPS_COURSE] != -1])  # drop rows before the first fix
    if len(acc) < 2 or len(gps) < 2:
        raise ValueError(f"{folder}: not enough accelerometer or GPS rows")

    gps_t = gps[:, GPS_T] - gps_lag_s  # move GPS earlier so speed lines up with braking
    start = max(acc[0, ACC_T], gps_t[0])
    end = min(acc[-1, ACC_T], gps_t[-1])
    if end <= start:
        raise ValueError(f"{folder}: accelerometer and GPS do not overlap")
    grid = np.arange(start, end, 1.0 / rate_hz)

    road = "secondary" if "SECONDARY" in folder.name.upper() else "motorway"
    return Trip(
        name=name,
        road=road,
        t=grid - start,
        long_g=np.interp(grid, acc[:, ACC_T], acc[:, ACC_LONG]),
        lat_g=np.interp(grid, acc[:, ACC_T], acc[:, ACC_LAT]),
        speed_kmh=np.interp(grid, gps_t, gps[:, GPS_SPEED]),
        lat=np.interp(grid, gps_t, gps[:, GPS_LAT]),
        lon=np.interp(grid, gps_t, gps[:, GPS_LON]),
    )


def load_trips(cfg: dict[str, Any], root: Path = ROOT) -> dict[str, Trip]:
    """Load every configured trip that exists; log and skip the rest."""
    data_dir = root / cfg["data_dir"]
    trips: dict[str, Trip] = {}
    for name, rel in cfg["trips"].items():
        folder = data_dir / rel
        try:
            trips[name] = load_trip(name, folder, cfg["rate_hz"], cfg["gps_lag_s"])
        except (OSError, ValueError) as error:
            log.warning("replay trip %s unavailable: %s", name, error)
            continue
        trip = trips[name]
        log.info("replay trip %s: %s, %.1f min", name, rel, trip.duration_s / 60)
    return trips


class Player:
    """Replay position and controls. Call step() once per tick; it returns the next sample."""

    def __init__(self, trips: dict[str, Trip], default_trip: str, loop: bool = True) -> None:
        if not trips:
            raise ValueError("no replay trips loaded")
        self.trips = trips
        self.trip = trips[default_trip] if default_trip in trips else next(iter(trips.values()))
        self.loop = loop
        self.index = 0
        self.playing = True

    @property
    def position_s(self) -> float:
        return float(self.trip.t[self.index])

    def step(self) -> dict[str, Any]:
        """Current sample; advances one step while playing. Paused, it repeats the same sample
        with speed and acceleration held, so the chart keeps scrolling at 10 Hz."""
        sample = self.trip.sample(self.index)
        if self.playing:
            if self.index + 1 < len(self.trip):
                self.index += 1
            elif self.loop:
                self.index = 0
            else:
                self.playing = False
        return sample

    def seek(self, seconds: float) -> None:
        seconds = min(max(seconds, 0.0), self.trip.duration_s)
        # The tolerance absorbs float drift in the grid (4.9999999 for 5.0).
        self.index = int(np.searchsorted(self.trip.t, seconds - 1e-6, side="left"))
        self.index = min(self.index, len(self.trip) - 1)

    def control(self, data: dict[str, Any]) -> None:
        """Apply a control.replay message: {action: play|pause|seek, trip?, segment?}."""
        trip = data.get("trip")
        if trip:
            if trip not in self.trips:
                log.warning("control.replay: unknown trip %r (have %s)", trip, list(self.trips))
                return
            if self.trips[trip] is not self.trip:
                self.trip = self.trips[trip]
                self.index = 0
        action = data.get("action")
        if action == "play":
            self.playing = True
        elif action == "pause":
            self.playing = False
        elif action == "seek":
            self.seek(float(data.get("segment") or 0.0))
        log.info(
            "replay %s: trip %s at %.1f s (%s)",
            action,
            self.trip.name,
            self.position_s,
            "playing" if self.playing else "paused",
        )
