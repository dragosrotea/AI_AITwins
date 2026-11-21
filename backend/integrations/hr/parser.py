"""Bluetooth GATT Heart Rate Measurement (UUID 0x2A37) parser.

Bluetooth SIG GATT Specifications:
- Service: Heart Rate Service (UUID 0x180D / 0000180d-0000-1000-8000-00805f9b34fb)
- Characteristic: Heart Rate Measurement (UUID 0x2A37 / 00002a37-0000-1000-8000-00805f9b34fb)

Flags Byte (Byte 0):
- Bit 0: Heart Rate Value Format (0 = UINT8 BPM, 1 = UINT16 BPM little-endian)
- Bit 1-2: Sensor Contact Status:
    00 / 01 = Feature not supported
    10 = Supported, contact not detected
    11 = Supported, contact detected
- Bit 3: Energy Expended Status (0 = Not present, 1 = Present as UINT16 little-endian in kJ)
- Bit 4: RR-Interval (0 = Not present, 1 = One or more UINT16 little-endian values in 1/1024 s)
- Bits 5-7: Reserved for future use

RR-Interval conversion:
- Raw unit: 1/1024 second.
- Milliseconds: raw * 1000.0 / 1024.0
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Literal

HR_SERVICE_16BIT = 0x180D
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"

HR_MEASUREMENT_16BIT = 0x2A37
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


@dataclass(frozen=True)
class HeartRateMeasurement:
    """Parsed Bluetooth Heart Rate Measurement characteristic value."""

    bpm: float
    rr_ms: list[float] = field(default_factory=list)
    energy_expended: int | None = None
    sensor_contact_detected: bool | None = None
    sensor_contact_supported: bool = False

    def to_payload(self, source: Literal["suunto", "strap", "sim"] = "suunto") -> dict[str, Any]:
        """Convert to the DriveMate bio.hr bus payload format."""
        return {
            "bpm": float(self.bpm),
            "rr_ms": list(self.rr_ms),
            "source": source,
        }


def parse_heart_rate_measurement(data: bytes | bytearray) -> HeartRateMeasurement:
    """Parse raw bytes from characteristic 0x2A37 into a HeartRateMeasurement.

    Args:
        data: Raw byte string received in the BLE notification.

    Returns:
        HeartRateMeasurement object containing bpm, rr_ms (in milliseconds),
        optional energy expended, and contact status.

    Raises:
        ValueError: If packet is empty, too short, or truncated for indicated fields.
    """
    if not data or len(data) < 2:
        raise ValueError("Heart Rate Measurement packet too short (at least 2 bytes required)")

    flags = data[0]
    is_16bit_bpm = bool(flags & 0x01)
    contact_bits = (flags >> 1) & 0x03
    contact_supported = bool(contact_bits & 0x02)
    contact_detected = bool(contact_bits == 0x03) if contact_supported else None
    has_energy = bool(flags & 0x08)
    has_rr = bool(flags & 0x10)

    offset = 1
    if is_16bit_bpm:
        if len(data) < offset + 2:
            raise ValueError("Packet truncated: expected 2 bytes for 16-bit BPM")
        (raw_bpm,) = struct.unpack_from("<H", data, offset)
        bpm = float(raw_bpm)
        offset += 2
    else:
        bpm = float(data[offset])
        offset += 1

    energy_expended: int | None = None
    if has_energy:
        if len(data) < offset + 2:
            raise ValueError("Packet truncated: expected 2 bytes for Energy Expended")
        (raw_energy,) = struct.unpack_from("<H", data, offset)
        energy_expended = int(raw_energy)
        offset += 2

    rr_ms: list[float] = []
    if has_rr:
        while offset + 2 <= len(data):
            (raw_rr,) = struct.unpack_from("<H", data, offset)
            offset += 2
            # Convert units of 1/1024 second to milliseconds
            ms = round(raw_rr * 1000.0 / 1024.0, 1)
            rr_ms.append(ms)

    return HeartRateMeasurement(
        bpm=bpm,
        rr_ms=rr_ms,
        energy_expended=energy_expended,
        sensor_contact_detected=contact_detected,
        sensor_contact_supported=contact_supported,
    )
