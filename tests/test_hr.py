"""Heart-rate integration: simulation and Suunto .FIT replay (UI-T15)."""

from __future__ import annotations

import asyncio
import struct
import unittest
from typing import Any

from backend.integrations.hr.fit import parse_fit_heart_rate
from backend.integrations.hr.run import (
    DEFAULT_RESTING_BPM,
    heart_rate,
    run_fit_replay,
    run_sim,
)
from backend.schemas import validate


class MockBus:
    """Mock DriveMate BusClient for testing."""

    def __init__(self, src: str = "hr") -> None:
        self.src = src
        self.connected = False
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.subscriptions: list[tuple[list[str], Any]] = []

    async def connect(self) -> None:
        self.connected = True

    def subscribe(self, topics: list[str], handler: Any) -> None:
        self.subscriptions.append((topics, handler))

    async def publish(self, topic: str, data: dict[str, Any], ts: float | None = None) -> None:
        self.published.append((topic, data))


class TestSimMode(unittest.TestCase):
    """Test simulated heart rate generator preservation."""

    def test_heart_rate_scenarios(self) -> None:
        resting = DEFAULT_RESTING_BPM  # 65.0
        # Normal scenario: resting + 8.0 (+ variation)
        hr_normal = heart_rate(0.0, resting, "normal")
        self.assertAlmostEqual(hr_normal, 73.0, delta=2.0)

        # Stress scenario: resting + 36.0 (+ variation)
        hr_stress = heart_rate(0.0, resting, "stress")
        self.assertAlmostEqual(hr_stress, 101.0, delta=2.0)

        # Fatigue scenario: resting * 0.85 (+ variation)
        hr_fatigue = heart_rate(0.0, resting, "fatigue")
        self.assertAlmostEqual(hr_fatigue, 55.25, delta=2.0)

        # Cycle scenario
        hr_cycle_0 = heart_rate(10.0, resting, "cycle")
        self.assertAlmostEqual(hr_cycle_0, 73.0, delta=2.0)
        hr_cycle_50 = heart_rate(50.0, resting, "cycle")
        self.assertAlmostEqual(hr_cycle_50, 101.0, delta=2.0)

    def test_run_sim_publishes_and_updates_resting(self) -> None:
        async def run_test() -> None:
            bus = MockBus()
            task = asyncio.create_task(run_sim(bus, "normal"))
            await asyncio.sleep(0.05)

            self.assertTrue(bus.connected)
            self.assertGreaterEqual(len(bus.published), 1)
            topic, data = bus.published[0]
            self.assertEqual(topic, "bio.hr")
            self.assertEqual(data["source"], "sim")
            validate("bio.hr", data)

            # Test resting HR subscription update
            self.assertEqual(len(bus.subscriptions), 1)
            topics, handler = bus.subscriptions[0]
            self.assertIn("control.trip", topics)
            await handler({"data": {"action": "start", "pretrip": {"resting_hr": 50.0}}})

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_test())


class TestFitParserAndReplay(unittest.TestCase):
    """Test pure-Python FIT parser and replay publisher."""

    def test_fit_parser_with_synthetic_file(self) -> None:
        import tempfile

        # Minimal valid FIT file with message 20 (record), field 3 (heart rate = 82)
        def_msg = bytes([0x40, 0x00, 0x00, 20, 0, 1, 3, 1, 2])
        data_msg = bytes([0x00, 82])
        records_bytes = def_msg + data_msg
        header = struct.pack("<BBHI4sH", 14, 0x20, 0x0810, len(records_bytes), b".FIT", 0)
        full_fit = header + records_bytes

        with tempfile.NamedTemporaryFile(suffix=".fit") as tmp:
            tmp.write(full_fit)
            tmp.flush()

            hrs = parse_fit_heart_rate(tmp.name)
            self.assertEqual(hrs, [82.0])

    def test_fit_replay_publishes_to_bus(self) -> None:
        import tempfile

        def_msg = bytes([0x40, 0x00, 0x00, 20, 0, 1, 3, 1, 2])
        data_msg = bytes([0x00, 75])
        records_bytes = def_msg + data_msg
        header = struct.pack("<BBHI4sH", 14, 0x20, 0x0810, len(records_bytes), b".FIT", 0)
        full_fit = header + records_bytes

        with tempfile.NamedTemporaryFile(suffix=".fit") as tmp:
            tmp.write(full_fit)
            tmp.flush()

            async def run_test() -> None:
                bus = MockBus()
                task = asyncio.create_task(run_fit_replay(bus, tmp.name, rate_hz=100.0))
                await asyncio.sleep(0.05)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                self.assertTrue(bus.connected)
                self.assertGreaterEqual(len(bus.published), 1)
                topic, payload = bus.published[0]
                self.assertEqual(topic, "bio.hr")
                self.assertEqual(payload["bpm"], 75.0)
                self.assertEqual(payload["source"], "suunto")
                validate("bio.hr", payload)

            asyncio.run(run_test())

    def test_fit_parser_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            parse_fit_heart_rate("non_existent_file.fit")

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".fit") as tmp:
            tmp.write(b"not a valid fit file header")
            tmp.flush()
            with self.assertRaises(ValueError):
                parse_fit_heart_rate(tmp.name)


if __name__ == "__main__":
    unittest.main()
