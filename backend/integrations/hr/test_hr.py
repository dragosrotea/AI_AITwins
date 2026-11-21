"""Unit tests for Heart Rate integration (parser, scanner, publisher, sim)."""

from __future__ import annotations

import asyncio
import struct
import unittest
from typing import Any
from unittest.mock import MagicMock

from backend.integrations.hr.fit import parse_fit_heart_rate
from backend.integrations.hr.parser import (
    HR_MEASUREMENT_UUID,
    HR_SERVICE_UUID,
    HeartRateMeasurement,
    parse_heart_rate_measurement,
)
from backend.integrations.hr.run import (
    DEFAULT_RESTING_BPM,
    HeartRatePublisher,
    heart_rate,
    run_ble,
    run_fit_replay,
    run_sim,
)
from backend.integrations.hr.scan import (
    find_hr_device,
    is_hr_service_uuid,
    scan_for_hr_devices,
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


class TestHeartRateParser(unittest.TestCase):
    """Test Bluetooth GATT Heart Rate Measurement (0x2A37) parser."""

    def test_8bit_bpm_basic(self) -> None:
        # Flags: 0x00 (8-bit BPM, contact not supported, no energy, no RR)
        # BPM: 72
        raw = bytes([0x00, 72])
        m = parse_heart_rate_measurement(raw)
        self.assertEqual(m.bpm, 72.0)
        self.assertEqual(m.rr_ms, [])
        self.assertIsNone(m.energy_expended)
        self.assertFalse(m.sensor_contact_supported)
        self.assertIsNone(m.sensor_contact_detected)

    def test_16bit_bpm(self) -> None:
        # Flags: 0x01 (16-bit BPM, contact not supported, no energy, no RR)
        # BPM: 300 (0x012C -> 0x2C, 0x01)
        raw = bytes([0x01, 0x2C, 0x01])
        m = parse_heart_rate_measurement(raw)
        self.assertEqual(m.bpm, 300.0)
        self.assertEqual(m.rr_ms, [])
        self.assertIsNone(m.energy_expended)

    def test_8bit_bpm_with_energy(self) -> None:
        # Flags: 0x08 (8-bit BPM, energy expended present)
        # BPM: 85
        # Energy: 1000 kJ (0x03E8 -> 0xE8, 0x03)
        raw = bytes([0x08, 85, 0xE8, 0x03])
        m = parse_heart_rate_measurement(raw)
        self.assertEqual(m.bpm, 85.0)
        self.assertEqual(m.energy_expended, 1000)
        self.assertEqual(m.rr_ms, [])

    def test_8bit_bpm_with_single_rr(self) -> None:
        # Flags: 0x10 (8-bit BPM, RR present)
        # BPM: 60
        # RR: 1024 (1/1024 s = 1.0 s = 1000.0 ms -> 0x0400 -> 0x00, 0x04)
        raw = bytes([0x10, 60, 0x00, 0x04])
        m = parse_heart_rate_measurement(raw)
        self.assertEqual(m.bpm, 60.0)
        self.assertEqual(m.rr_ms, [1000.0])

    def test_8bit_bpm_with_multiple_rr(self) -> None:
        # Flags: 0x10 (8-bit BPM, RR present)
        # BPM: 75
        # RR1: 1024 -> 1000.0 ms (0x00, 0x04)
        # RR2: 819 -> round(819 * 1000 / 1024, 1) = 799.8 ms (0x0333 -> 0x33, 0x03)
        raw = bytes([0x10, 75, 0x00, 0x04, 0x33, 0x03])
        m = parse_heart_rate_measurement(raw)
        self.assertEqual(m.bpm, 75.0)
        self.assertEqual(m.rr_ms, [1000.0, 799.8])

    def test_16bit_bpm_with_energy_and_multiple_rr(self) -> None:
        # Flags: 0x19 (16-bit BPM | energy expended | RR present = 1 | 8 | 16 = 25 = 0x19)
        # BPM: 140 (0x008C -> 0x8C, 0x00)
        # Energy: 250 kJ (0x00FA -> 0xFA, 0x00)
        # RR1: 512 (500.0 ms -> 0x0200 -> 0x00, 0x02)
        # RR2: 1024 (1000.0 ms -> 0x0400 -> 0x00, 0x04)
        raw = bytes([0x19, 0x8C, 0x00, 0xFA, 0x00, 0x00, 0x02, 0x00, 0x04])
        m = parse_heart_rate_measurement(raw)
        self.assertEqual(m.bpm, 140.0)
        self.assertEqual(m.energy_expended, 250)
        self.assertEqual(m.rr_ms, [500.0, 1000.0])

    def test_sensor_contact_status(self) -> None:
        # Flags: bit 1 and 2
        # 0x00: unsupported
        m0 = parse_heart_rate_measurement(bytes([0x00, 70]))
        self.assertFalse(m0.sensor_contact_supported)
        self.assertIsNone(m0.sensor_contact_detected)

        # 0x02: unsupported (01 in bits 1-2)
        m1 = parse_heart_rate_measurement(bytes([0x02, 70]))
        self.assertFalse(m1.sensor_contact_supported)
        self.assertIsNone(m1.sensor_contact_detected)

        # 0x04: supported, contact NOT detected (10 in bits 1-2)
        m2 = parse_heart_rate_measurement(bytes([0x04, 70]))
        self.assertTrue(m2.sensor_contact_supported)
        self.assertFalse(m2.sensor_contact_detected)

        # 0x06: supported, contact DETECTED (11 in bits 1-2)
        m3 = parse_heart_rate_measurement(bytes([0x06, 70]))
        self.assertTrue(m3.sensor_contact_supported)
        self.assertTrue(m3.sensor_contact_detected)

    def test_schema_contract_compliance(self) -> None:
        raw = bytes([0x10, 78, 0x00, 0x04, 0x33, 0x03])
        m = parse_heart_rate_measurement(raw)
        payload = m.to_payload(source="suunto")
        # Ensure schema validates without extra fields
        model = validate("bio.hr", payload)
        self.assertEqual(model.bpm, 78.0)
        self.assertEqual(model.rr_ms, [1000.0, 799.8])
        self.assertEqual(model.source, "suunto")

    def test_malformed_packets(self) -> None:
        with self.assertRaises(ValueError):
            parse_heart_rate_measurement(b"")
        with self.assertRaises(ValueError):
            parse_heart_rate_measurement(bytes([0x00]))
        # 16-bit BPM indicated but only 1 byte follows
        with self.assertRaises(ValueError):
            parse_heart_rate_measurement(bytes([0x01, 75]))
        # Energy expended indicated but packet ends
        with self.assertRaises(ValueError):
            parse_heart_rate_measurement(bytes([0x08, 75, 0x01]))


class TestHeartRateScanner(unittest.TestCase):
    """Test BLE scanner functions for Heart Rate Service 0x180D."""

    def test_is_hr_service_uuid(self) -> None:
        self.assertTrue(is_hr_service_uuid("180d"))
        self.assertTrue(is_hr_service_uuid("180D"))
        self.assertTrue(is_hr_service_uuid("0x180D"))
        self.assertTrue(is_hr_service_uuid("0000180d-0000-1000-8000-00805f9b34fb"))
        self.assertTrue(is_hr_service_uuid("0000180D-0000-1000-8000-00805F9B34FB"))
        self.assertFalse(is_hr_service_uuid("180a"))
        self.assertFalse(is_hr_service_uuid("00001800-0000-1000-8000-00805f9b34fb"))

    def test_scan_and_find_with_mock_scanner(self) -> None:
        class MockDevice:
            def __init__(self, address: str, name: str) -> None:
                self.address = address
                self.name = name

        class MockAdv:
            def __init__(self, local_name: str, service_uuids: list[str], rssi: int) -> None:
                self.local_name = local_name
                self.service_uuids = service_uuids
                self.rssi = rssi

        class MockBleakScanner:
            @classmethod
            async def discover(cls, timeout: float = 5.0, return_adv: bool = True) -> dict[str, Any]:
                return {
                    "AA:BB:CC:DD:EE:01": (
                        MockDevice("AA:BB:CC:DD:EE:01", "Polar H10"),
                        MockAdv("Polar H10", ["0000180d-0000-1000-8000-00805f9b34fb"], -65),
                    ),
                    "AA:BB:CC:DD:EE:02": (
                        MockDevice("AA:BB:CC:DD:EE:02", "Suunto Vertical HR"),
                        MockAdv("Suunto Vertical HR", ["180d", "180a"], -55),
                    ),
                    "AA:BB:CC:DD:EE:03": (
                        MockDevice("AA:BB:CC:DD:EE:03", "Bluetooth Headset"),
                        MockAdv("Bluetooth Headset", ["110b"], -80),
                    ),
                }

        results = asyncio.run(scan_for_hr_devices(timeout=1.0, scanner_cls=MockBleakScanner))
        self.assertEqual(len(results), 3)
        hr_only = [r for r in results if r["offers_hr"]]
        self.assertEqual(len(hr_only), 2)

        # find_hr_device should prioritize Suunto
        selected = asyncio.run(find_hr_device(timeout=1.0, scanner_cls=MockBleakScanner))
        self.assertEqual(selected, "AA:BB:CC:DD:EE:02")


class TestHeartRatePublisher(unittest.TestCase):
    """Test HeartRatePublisher buffering, schema, and 1 Hz publishing."""

    def test_publisher_buffering_and_tick(self) -> None:
        bus = MockBus()
        pub = HeartRatePublisher(bus=bus, source="suunto", rate_hz=1.0)
        pub.mark_connected()

        # Handle raw measurement packet
        raw = bytes([0x10, 72, 0x00, 0x04])  # 72 BPM, 1000.0 ms RR
        pub.handle_raw_data(raw)
        self.assertTrue(pub.has_fresh_data())

        # Publish tick
        published = asyncio.run(pub.publish_tick())
        self.assertTrue(published)
        self.assertEqual(len(bus.published), 1)

        topic, payload = bus.published[0]
        self.assertEqual(topic, "bio.hr")
        self.assertEqual(payload, {"bpm": 72.0, "rr_ms": [1000.0], "source": "suunto"})
        # Validates against schema
        validate("bio.hr", payload)

        # Second tick without new packet: bpm retained, rr_ms empty
        published_2 = asyncio.run(pub.publish_tick())
        self.assertTrue(published_2)
        _, payload_2 = bus.published[1]
        self.assertEqual(payload_2, {"bpm": 72.0, "rr_ms": [], "source": "suunto"})

    def test_publisher_disconnected_stops_publishing(self) -> None:
        bus = MockBus()
        pub = HeartRatePublisher(bus=bus, source="suunto")
        pub.mark_connected()
        pub.handle_raw_data(bytes([0x00, 80]))
        self.assertTrue(pub.has_fresh_data())

        pub.mark_disconnected()
        self.assertFalse(pub.has_fresh_data())
        published = asyncio.run(pub.publish_tick())
        self.assertFalse(published)
        self.assertEqual(len(bus.published), 0)


class TestRunBle(unittest.TestCase):
    """Test run_ble with mocked BleakClient and reconnection."""

    def test_ble_publisher_lifecycle(self) -> None:
        bus = MockBus()
        stop_event = asyncio.Event()

        class MockBleakClient:
            def __init__(self, address: str, disconnected_callback: Any = None) -> None:
                self.address = address
                self.disconnected_callback = disconnected_callback
                self.is_connected = False
                self.notification_callback: Any = None

            async def __aenter__(self) -> MockBleakClient:
                self.is_connected = True
                return self

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                self.is_connected = False
                if self.disconnected_callback:
                    self.disconnected_callback(self)

            async def start_notify(self, char_uuid: str, callback: Any) -> None:
                self.notification_callback = callback
                # Send a sample packet
                packet = bytes([0x10, 75, 0x00, 0x04])
                callback(None, bytearray(packet))
                # Stop after receiving packet
                stop_event.set()

            async def stop_notify(self, char_uuid: str) -> None:
                pass

        async def run_test() -> None:
            await run_ble(
                bus=bus,
                address="AA:BB:CC:11:22:33",
                source="suunto",
                retry_interval_s=0.1,
                stop_event=stop_event,
                client_factory=MockBleakClient,
            )

        asyncio.run(run_test())
        self.assertTrue(bus.connected)


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
