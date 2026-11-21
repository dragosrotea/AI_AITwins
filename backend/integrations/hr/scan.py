"""BLE scanner for devices advertising Heart Rate Service (UUID 0x180D).

Corresponds to task UI-T05 (SW-00).
Usage:
    python -m backend.integrations.hr.scan [--timeout SECONDS] [--all] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from backend.integrations.hr.parser import HR_SERVICE_UUID


def is_hr_service_uuid(uuid_str: str) -> bool:
    """Check if a UUID string matches Heart Rate Service 0x180D."""
    u = uuid_str.lower().strip()
    return u in ("180d", "0x180d", HR_SERVICE_UUID) or u.startswith("0000180d")


async def scan_for_hr_devices(
    timeout: float = 5.0,
    scanner_cls: Any = None,
) -> list[dict[str, Any]]:
    """Scan for BLE devices and identify those advertising Heart Rate Service 0x180D."""
    if scanner_cls is None:
        try:
            from bleak import BleakScanner

            scanner_cls = BleakScanner
        except ImportError:
            raise RuntimeError(
                "Bleak is required for BLE scanning. Install with: pip install bleak"
            ) from None

    results: list[dict[str, Any]] = []
    # discover with return_adv=True returns a mapping of device address to (BLEDevice, AdvertisementData)
    devices_and_adv = await scanner_cls.discover(timeout=timeout, return_adv=True)

    for device, adv in devices_and_adv.values():
        service_uuids = [str(u) for u in (adv.service_uuids or [])]
        offers_hr = any(is_hr_service_uuid(u) for u in service_uuids)
        name = adv.local_name or device.name or "Unknown"
        results.append({
            "address": device.address,
            "name": name,
            "rssi": adv.rssi,
            "offers_hr": offers_hr,
            "service_uuids": service_uuids,
        })

    # Sort: HR devices first, then highest RSSI
    results.sort(key=lambda x: (not x["offers_hr"], -(x["rssi"] or -999)))
    return results


async def find_hr_device(timeout: float = 5.0, scanner_cls: Any = None) -> str | None:
    """Find the address of the first BLE device advertising Heart Rate Service 0x180D.

    Prefers devices whose name contains 'suunto' if multiple are detected.
    """
    devices = await scan_for_hr_devices(timeout=timeout, scanner_cls=scanner_cls)
    hr_devices = [d for d in devices if d["offers_hr"]]
    if not hr_devices:
        return None
    for d in hr_devices:
        if "suunto" in d["name"].lower():
            return d["address"]
    return hr_devices[0]["address"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="Scan duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="List all discovered BLE devices, not just HR monitors",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    try:
        from bleak import BleakScanner  # noqa: F401
    except ImportError:
        print(
            "Error: 'bleak' is required for BLE scanning, but is not installed.\n"
            "Install it using: pip install bleak",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Scanning for BLE devices advertising Heart Rate Service (0x180D) for {args.timeout}s..."
    )
    try:
        devices = asyncio.run(scan_for_hr_devices(timeout=args.timeout))
    except KeyboardInterrupt:
        print("\nScan interrupted.")
        return
    except Exception as err:
        print(f"Scan failed: {err}", file=sys.stderr)
        sys.exit(1)

    hr_devices = [d for d in devices if d["offers_hr"]]

    if args.json:
        output_list = devices if args.all else hr_devices
        print(json.dumps(output_list, indent=2))
        return

    if hr_devices:
        print(f"\nFound {len(hr_devices)} device(s) advertising Heart Rate Service (0x180D):")
        print(f"{'Address':<38} {'Name':<25} {'RSSI':<6}")
        print("-" * 72)
        for d in hr_devices:
            rssi_str = f"{d['rssi']} dBm" if d["rssi"] is not None else "N/A"
            print(f"{d['address']:<38} {d['name']:<25} {rssi_str:<6}")
    else:
        print("\nNo devices advertising Heart Rate Service (0x180D) were found.")
        print(
            "Note: If testing with Suunto 9 Peak Pro, native 0x180D broadcast is not supported\n"
            "      on this model (see research report). Consider testing with a BLE chest strap,\n"
            "      or re-run with '--all' to inspect all detected Bluetooth advertisements."
        )

    if args.all and len(devices) > len(hr_devices):
        other_devices = [d for d in devices if not d["offers_hr"]]
        print(f"\nOther discovered BLE devices ({len(other_devices)}):")
        print(f"{'Address':<38} {'Name':<25} {'RSSI':<6}")
        print("-" * 72)
        for d in other_devices:
            rssi_str = f"{d['rssi']} dBm" if d["rssi"] is not None else "N/A"
            print(f"{d['address']:<38} {d['name']:<25} {rssi_str:<6}")


if __name__ == "__main__":
    main()
