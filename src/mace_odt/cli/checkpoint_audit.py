"""Audit a released MACE-OFF23 checkpoint and its basic forward behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

from mace_odt.audit import (
    environment_manifest,
    find_checkpoint_candidates,
    forward_smoke_tests,
    model_metadata,
    module_inventory,
    sha256_file,
    state_shapes,
    top_level_parameter_counts,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    from mace.calculators import mace_off

    calc = mace_off(
        model=args.model,
        device=args.device,
        default_dtype=args.dtype,
    )
    model = calc.models[0]
    checkpoint_candidates = find_checkpoint_candidates(args.model)
    checkpoints = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in checkpoint_candidates
    ]
    payload = {
        "schema_version": 1,
        "experiment": "E1_checkpoint_architecture_audit",
        "requested_model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "environment": environment_manifest(),
        "checkpoint_files": checkpoints,
        "model_type": f"{type(model).__module__}.{type(model).__name__}",
        "metadata": model_metadata(model),
        "parameter_counts": top_level_parameter_counts(model),
        "modules": module_inventory(model),
        "state_dict": state_shapes(model),
        "smoke_tests": (
            None if args.skip_smoke else forward_smoke_tests(calc, dtype=args.dtype)
        ),
    }
    payload["gate_passed"] = bool(checkpoints) and (
        args.skip_smoke or payload["smoke_tests"]["passed"]
    )
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"gate_passed={payload['gate_passed']}")
    if not payload["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
