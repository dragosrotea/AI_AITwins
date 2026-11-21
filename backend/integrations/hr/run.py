"""Publish heart-rate data to the DriveMate bus (UI-T15).

Supports both simulated HR generation (--sim) and live Bluetooth LE streaming
from a Suunto watch or standard BLE chest strap advertising Heart Rate Service 0x180D.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import random
import sys
import time
from typing import Any

from backend.integrations.hr.parser import (
    HR_MEASUREMENT_UUID,
    HeartRateMeasurement,
    parse_heart_rate_measurement,
)

log = logging.getLogger("hr")

RATE_HZ = 1.0
DEFAULT_RESTING_BPM = 65.0


def heart_rate(elapsed_s: float, resting_bpm: float, scenario: str) -> float:
    """Return a deterministic, gently varying BPM for the requested test case."""
    if scenario == "stress":
        target = resting_bpm + 36.0
    elif scenario == "fatigue":
        target = resting_bpm * 0.85
    elif scenario == "normal":
        target = resting_bpm + 8.0
    else:
        # 20 s baseline, 65 s sustained stress, then 65 s low-HR monotony cue.
        phase = elapsed_s % 150.0
        target = (
            resting_bpm + 8.0
            if phase < 20.0
            else resting_bpm + 36.0
            if phase < 85.0
            else resting_bpm * 0.85
        )
    return round(max(35.0, target + 1.2 * math.sin(elapsed_s / 7.0)), 1)


async def run_sim(bus: Any, scenario: str) -> None:
    """Run simulated heart-rate publisher (UI-T05 / CP0 fake)."""
    resting_bpm = DEFAULT_RESTING_BPM

    async def on_trip(envelope: dict[str, Any]) -> None:
        nonlocal resting_bpm
        data = envelope.get("data", {})
        pretrip = data.get("pretrip") or {}
        value = pretrip.get("resting_hr")
        if data.get("action") == "start" and isinstance(value, (int, float)) and value > 0:
            resting_bpm = float(value)

    bus.subscribe(["control.trip"], on_trip)
    await bus.connect()
    loop = asyncio.get_running_loop()
    started = loop.time()
    next_tick = started

    while True:
        elapsed = loop.time() - started
        bpm = heart_rate(elapsed, resting_bpm, scenario)
        # Approximate RR intervals (ms) around the current mean interval.
        mean_rr = 60_000.0 / bpm
        rr_ms = [round(mean_rr + random.uniform(-18.0, 18.0), 1) for _ in range(5)]
        await bus.publish("bio.hr", {"bpm": bpm, "rr_ms": rr_ms, "source": "sim"})
        next_tick += 1.0 / RATE_HZ
        next_tick = max(next_tick, loop.time())
        await asyncio.sleep(next_tick - loop.time())


class HeartRatePublisher:
    """Buffers BLE HR measurements and publishes bio.hr at 1 Hz to the bus."""

    def __init__(self, bus: Any, source: str = "suunto", rate_hz: float = RATE_HZ) -> None:
        self.bus = bus
        self.source = source
        self.rate_hz = rate_hz
        self.latest_bpm: float | None = None
        self.pending_rr: list[float] = []
        self.last_packet_time: float = 0.0
        self.is_connected: bool = False
        self._lock = asyncio.Lock()

    def mark_connected(self) -> None:
        self.is_connected = True

    def mark_disconnected(self) -> None:
        self.is_connected = False

    def has_fresh_data(self, max_stale_s: float = 5.0) -> bool:
        """Check whether we have received fresh HR packets within max_stale_s."""
        if not self.is_connected or self.latest_bpm is None:
            return False
        return (time.time() - self.last_packet_time) <= max_stale_s

    def handle_raw_data(self, data: bytes | bytearray) -> HeartRateMeasurement:
        """Parse raw BLE measurement and buffer metrics."""
        measurement = parse_heart_rate_measurement(data)
        if measurement.bpm > 0:
            self.latest_bpm = measurement.bpm
            self.pending_rr.extend(measurement.rr_ms)
            self.last_packet_time = time.time()
        return measurement

    async def publish_tick(self) -> bool:
        """Publish bio.hr if fresh measurement is available. Returns True if published."""
        if not self.has_fresh_data():
            return False

        async with self._lock:
            bpm = self.latest_bpm
            if bpm is None or bpm <= 0:
                return False
            rr = self.pending_rr
            self.pending_rr = []

        payload = {
            "bpm": float(bpm),
            "rr_ms": rr,
            "source": self.source,
        }
        await self.bus.publish("bio.hr", payload)
        return True

    async def run_publish_loop(self, stop_event: asyncio.Event) -> None:
        """Publish bio.hr at exactly rate_hz (1 Hz) while stop_event is not set."""
        loop = asyncio.get_running_loop()
        interval = 1.0 / self.rate_hz
        next_tick = loop.time()

        while not stop_event.is_set():
            next_tick += interval
            sleep_s = max(0.0, next_tick - loop.time())
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_s)
                break
            except asyncio.TimeoutError:
                pass

            if self.has_fresh_data():
                try:
                    await self.publish_tick()
                except Exception as err:
                    log.warning("Failed to publish bio.hr: %s", err)


async def run_ble(
    bus: Any,
    address: str | None = None,
    source: str = "suunto",
    retry_interval_s: float = 3.0,
    stop_event: asyncio.Event | None = None,
    client_factory: Any = None,
) -> None:
    """Connect to a BLE Heart Rate monitor, subscribe to 0x2A37, and publish bio.hr at 1 Hz.

    Automatically reconnects upon disconnection or connection errors.
    """
    if stop_event is None:
        stop_event = asyncio.Event()

    if client_factory is None:
        try:
            from bleak import BleakClient

            client_factory = BleakClient
        except ImportError:
            raise RuntimeError(
                "Bleak is required for live BLE mode. Install with: pip install bleak"
            ) from None

    await bus.connect()

    publisher = HeartRatePublisher(bus=bus, source=source, rate_hz=RATE_HZ)
    publish_task = asyncio.create_task(
        publisher.run_publish_loop(stop_event=stop_event), name="hr-publish-loop"
    )

    target_address = address or os.environ.get("SUUNTO_BLE_ADDRESS") or os.environ.get("HR_BLE_ADDRESS")

    try:
        while not stop_event.is_set():
            if not target_address:
                log.info("No BLE device address specified. Scanning for HR device (0x180D)...")
                try:
                    from backend.integrations.hr.scan import find_hr_device

                    target_address = await find_hr_device(timeout=5.0)
                except Exception as scan_err:
                    log.warning("Scan error: %s", scan_err)

                if not target_address:
                    log.warning(
                        "No Heart Rate device found advertising 0x180D. Retrying scan in %s s...",
                        retry_interval_s,
                    )
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=retry_interval_s)
                        break
                    except asyncio.TimeoutError:
                        continue

            log.info("Connecting to BLE HR device %s...", target_address)
            disconnected_event = asyncio.Event()

            def on_disconnect(_client: Any) -> None:
                log.warning("BLE device %s disconnected.", target_address)
                publisher.mark_disconnected()
                disconnected_event.set()

            def on_notification(_sender: Any, data: bytearray | bytes) -> None:
                publisher.handle_raw_data(bytes(data))

            try:
                async with client_factory(
                    target_address, disconnected_callback=on_disconnect
                ) as client:
                    if not client.is_connected:
                        log.warning("Could not establish connection to %s.", target_address)
                        await asyncio.sleep(retry_interval_s)
                        continue

                    log.info(
                        "Connected to %s. Subscribing to Heart Rate Measurement (0x2A37)...",
                        target_address,
                    )
                    await client.start_notify(HR_MEASUREMENT_UUID, on_notification)
                    publisher.mark_connected()

                    wait_disc = asyncio.create_task(disconnected_event.wait())
                    wait_stop = asyncio.create_task(stop_event.wait())

                    _, pending = await asyncio.wait(
                        [wait_disc, wait_stop],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()

                    if client.is_connected:
                        try:
                            await client.stop_notify(HR_MEASUREMENT_UUID)
                        except Exception:
                            pass

            except Exception as conn_err:
                log.warning("BLE connection error (%s): %s", target_address, conn_err)
                publisher.mark_disconnected()

            if stop_event.is_set():
                break

            log.info("Waiting %s s before reconnecting to %s...", retry_interval_s, target_address)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=retry_interval_s)
                break
            except asyncio.TimeoutError:
                pass
    finally:
        stop_event.set()
        publish_task.cancel()
        try:
            await publish_task
        except asyncio.CancelledError:
            pass


async def run_fit_replay(bus: Any, fit_path: str, rate_hz: float = RATE_HZ) -> None:
    """Replay heart rate values from a Suunto .FIT workout file at rate_hz (1 Hz)."""
    from backend.integrations.hr.fit import parse_fit_heart_rate

    hr_values = parse_fit_heart_rate(fit_path)
    if not hr_values:
        raise ValueError(f"No valid heart rate records found in {fit_path}")

    log.info(
        "Loaded %d HR records from %s (min: %.1f, max: %.1f BPM)",
        len(hr_values),
        fit_path,
        min(hr_values),
        max(hr_values),
    )
    print(
        f"[hr] Loaded {len(hr_values)} Suunto HR records from {fit_path} "
        f"(BPM range: {min(hr_values):.0f} - {max(hr_values):.0f}). Replaying at {rate_hz} Hz..."
    )

    await bus.connect()
    loop = asyncio.get_running_loop()
    interval = 1.0 / rate_hz
    next_tick = loop.time()
    idx = 0

    while True:
        bpm = hr_values[idx]
        mean_rr = 60_000.0 / bpm
        rr_ms = [round(mean_rr + random.uniform(-10.0, 10.0), 1) for _ in range(3)]
        payload = {"bpm": float(bpm), "rr_ms": rr_ms, "source": "suunto"}
        await bus.publish("bio.hr", payload)

        idx = (idx + 1) % len(hr_values)
        next_tick += interval
        next_tick = max(next_tick, loop.time())
        await asyncio.sleep(next_tick - loop.time())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit",
        "--replay",
        dest="fit_path",
        default=None,
        help="Path to a Suunto .FIT workout file to replay at 1 Hz",
    )
    parser.add_argument(
        "--sim",
        "--fake",
        dest="sim",
        action="store_true",
        default=False,
        help="Run with simulated heart-rate data",
    )
    parser.add_argument(
        "--scenario",
        choices=("cycle", "normal", "stress", "fatigue"),
        default="cycle",
        help="Simulation scenario (default: cycle)",
    )
    parser.add_argument(
        "--address",
        "--device",
        dest="address",
        default=None,
        help="BLE MAC address or UUID of the heart rate monitor",
    )
    parser.add_argument(
        "--source",
        choices=("suunto", "strap", "sim"),
        default="suunto",
        help="Data source label (default: suunto)",
    )
    parser.add_argument(
        "--retry-interval",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting after disconnect",
    )
    args = parser.parse_args()

    if args.fit_path:
        from backend.bus.client import BusClient

        bus = BusClient("hr")
        try:
            asyncio.run(run_fit_replay(bus, args.fit_path))
        except KeyboardInterrupt:
            pass
    elif args.sim:
        from backend.bus.client import BusClient

        bus = BusClient("hr")
        try:
            asyncio.run(run_sim(bus, args.scenario))
        except KeyboardInterrupt:
            pass
    else:
        try:
            from bleak import BleakClient  # noqa: F401
        except ImportError:
            print(
                "Error: 'bleak' is required for live BLE mode, but is not installed.\n"
                "Install it using: pip install bleak\n"
                "Or run simulation mode using: python -m backend.integrations.hr.run --sim\n"
                "Or run FIT replay mode using: python -m backend.integrations.hr.run --fit <PATH.fit>",
                file=sys.stderr,
            )
            sys.exit(1)

        from backend.bus.client import BusClient

        bus = BusClient("hr")
        try:
            asyncio.run(
                run_ble(
                    bus,
                    address=args.address,
                    source=args.source,
                    retry_interval_s=args.retry_interval,
                )
            )
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
