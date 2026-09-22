"""Write a reproducible execution environment manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from mace_odt.audit import environment_manifest, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = environment_manifest()
    write_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
