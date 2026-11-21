"""Pure-Python parser for extracting heart rate records from Garmin/Suunto .FIT files.

Zero external dependencies required (uses only standard library struct).
"""

from __future__ import annotations

import struct
from pathlib import Path


def parse_fit_heart_rate(file_path: str | Path) -> list[float]:
    """Parse a .FIT file and extract all valid heart rate (BPM) values.

    Args:
        file_path: Path to the .FIT binary file.

    Returns:
        List of heart rate values in beats per minute (BPM) as floats.

    Raises:
        ValueError: If file is not a valid FIT format or has an invalid header.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"FIT file not found: {file_path}")

    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 14:
        raise ValueError("File too small to be a valid FIT file")

    header_len = data[0]
    if header_len < 12 or len(data) < header_len:
        raise ValueError(f"Invalid FIT header length: {header_len}")

    data_len = struct.unpack("<I", data[4:8])[0]
    magic = data[8:12]
    if magic != b".FIT":
        raise ValueError(f"Invalid FIT magic string: expected b'.FIT', got {magic!r}")

    pos = header_len
    end = min(len(data), header_len + data_len)

    definitions: dict[int, dict] = {}
    hr_records: list[float] = []

    while pos < end:
        h = data[pos]
        pos += 1

        # Compressed timestamp header (bit 7 set)
        if h & 0x80:
            local = (h >> 5) & 0x03
            if local not in definitions:
                break
            defn = definitions[local]
            msg_size = defn["total_size"]
            msg_data = data[pos : pos + msg_size]
            pos += msg_size

            # Message 20 is 'record'
            if defn["global_msg_num"] == 20:
                for f_num, _, f_off in defn["fields"]:
                    if f_num == 3 and f_off < len(msg_data):  # field 3 is heart_rate in FIT profile
                        hr = msg_data[f_off]
                        if 30 < hr < 250:
                            hr_records.append(float(hr))
            continue

        is_def = bool(h & 0x40)
        has_dev = bool(h & 0x20)
        local = h & 0x0F

        if is_def:
            if pos + 5 > end:
                break
            arch = data[pos + 1]
            endian = "<" if arch == 0 else ">"
            global_msg_num = struct.unpack_from(endian + "H", data, pos + 2)[0]
            num_fields = data[pos + 4]
            pos += 5

            fields = []
            f_offset = 0
            for _ in range(num_fields):
                if pos + 3 > end:
                    break
                f_num = data[pos]
                f_size = data[pos + 1]
                pos += 3
                fields.append((f_num, f_size, f_offset))
                f_offset += f_size

            if has_dev and pos < end:
                num_dev = data[pos]
                pos += 1
                for _ in range(num_dev):
                    if pos + 3 > end:
                        break
                    dev_size = data[pos + 1]
                    pos += 3
                    f_offset += dev_size

            definitions[local] = {
                "global_msg_num": global_msg_num,
                "endian": endian,
                "fields": fields,
                "total_size": f_offset,
            }
        else:
            if local not in definitions:
                break
            defn = definitions[local]
            msg_size = defn["total_size"]
            msg_data = data[pos : pos + msg_size]
            pos += msg_size

            if defn["global_msg_num"] == 20:  # record
                for f_num, _, f_off in defn["fields"]:
                    if f_num == 3 and f_off < len(msg_data):  # heart_rate
                        hr = msg_data[f_off]
                        if 30 < hr < 250:
                            hr_records.append(float(hr))

    return hr_records
