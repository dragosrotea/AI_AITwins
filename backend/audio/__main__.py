"""List devices and check the configured audio path without recording to disk."""

import argparse
import math
import struct
import time

import sounddevice as sd
import yaml

from .devices import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, AudioSettings, Microphone


def check_devices(settings: AudioSettings) -> None:
    input_device = sd.query_devices(settings.input_device, "input")
    output_device = sd.query_devices(settings.output_device, "output")
    print(f"Input: {input_device['name']}")
    print(f"Output: {output_device['name']}")

    print("Speak into the microphone for 3 seconds.", flush=True)
    with Microphone(settings) as mic:
        cursor = mic.cursor()
        time.sleep(3)
        frames = mic.read_since(cursor)
    if not frames or any(len(frame.pcm) != FRAME_SAMPLES * 2 for frame in frames):
        raise ValueError("No complete microphone frames captured")
    peak = max(
        abs(sample)
        for frame in frames
        for (sample,) in struct.iter_unpack("=h", frame.pcm)
    )
    print(f"Captured {len(frames)} frames, {SAMPLE_RATE} Hz mono, {FRAME_MS} ms each")
    print(f"Peak input: {100 * peak / 32768:.1f}% of full scale")

    rate = int(output_device["default_samplerate"])
    tone = b"".join(
        struct.pack("=h", int(1800 * math.sin(2 * math.pi * 440 * index / rate)))
        for index in range(rate // 4)
    )
    with sd.RawOutputStream(
        device=settings.output_device, samplerate=rate, channels=1, dtype="int16"
    ) as output:
        output.write(tone)
    print("Played a short test tone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-devices", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if args.list_devices:
            print(sd.query_devices())
        else:
            check_devices(AudioSettings.load())
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        sd.PortAudioError,
        yaml.YAMLError,
    ) as error:
        parser.exit(1, f"Audio setup failed: {error}\n")


if __name__ == "__main__":
    main()
