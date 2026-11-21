"""Export backend/schemas.py as JSON Schema to docs/contract.json (LEAD-T04).

    python scripts/export_schema.py            # write docs/contract.json
    python scripts/export_schema.py --check    # exit 1 if docs/contract.json is stale

contract.json layout:
    version, envelope (ref), topics {topic: {model (ref), producers, rate_hz}}, $defs {models}
The dashboard generates src/types.ts from the $defs (e.g. with json-schema-to-typescript).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic.json_schema import models_json_schema

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.schemas import CONTRACT_VERSION, TOPICS, Envelope  # noqa: E402

OUT = ROOT / "docs" / "contract.json"
REF = "#/$defs/{model}"


def build_contract() -> dict:
    models = [Envelope] + [spec.model for spec in TOPICS.values()]
    unique = list(dict.fromkeys(models))
    _, schema = models_json_schema(
        [(m, "validation") for m in unique], ref_template=REF, title="DriveMate bus contract"
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DriveMate bus contract",
        "version": CONTRACT_VERSION,
        "description": "Generated from backend/schemas.py by scripts/export_schema.py. "
        "Do not edit by hand.",
        "envelope": {"$ref": REF.format(model=Envelope.__name__)},
        "topics": {
            topic: {
                "model": {"$ref": REF.format(model=spec.model.__name__)},
                "producers": list(spec.producers),
                "rate_hz": spec.rate_hz,
            }
            for topic, spec in TOPICS.items()
        },
        "$defs": schema["$defs"],
    }


def render() -> str:
    return json.dumps(build_contract(), indent=2, sort_keys=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the file is stale")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is stale: run python scripts/export_schema.py")
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(TOPICS)} topics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
