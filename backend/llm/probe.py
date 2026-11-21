"""Opt-in timing probe for one live provider request."""

import argparse
import asyncio
from time import perf_counter

from .client import request_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--timeout", type=float, default=10.0, metavar="SECONDS")
    args = parser.parse_args()
    started = perf_counter()
    result = asyncio.run(
        request_text("Reply with the single word hello.", args.timeout)
    )
    elapsed_ms = (perf_counter() - started) * 1000
    if result is None:
        parser.exit(
            1,
            "Live probe failed. Check local provider configuration and connectivity.\n",
        )
    print(f"Live provider call succeeded in {elapsed_ms:.0f} ms")


if __name__ == "__main__":
    main()
