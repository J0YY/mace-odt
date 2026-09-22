"""Replace live MACE path maps with their exact supported coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from mace_odt.audit import evaluate, smoke_geometries, write_json
from mace_odt.quotient_module import (
    QuotientedScalarSymmetricContraction,
    quotient_random_equivalence,
)


def compare_outputs(
    reference: list[tuple[float, np.ndarray]],
    candidate: list[tuple[float, np.ndarray]],
) -> dict[str, Any]:
    records = []
    for atoms, (energy_ref, force_ref), (energy_new, force_new) in zip(
        smoke_geometries(), reference, candidate
    ):
        energy_error = abs(energy_new - energy_ref)
        force_error = float(np.max(np.abs(force_new - force_ref)))
        records.append(
            {
                "formula": atoms.get_chemical_formula(),
                "energy_absolute_error_eV": energy_error,
                "force_max_absolute_error_eV_per_A": force_error,
            }
        )
    return {
        "records": records,
        "max_energy_absolute_error_eV": max(
            record["energy_absolute_error_eV"] for record in records
        ),
        "max_force_absolute_error_eV_per_A": max(
            record["force_max_absolute_error_eV_per_A"] for record in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    reference = [evaluate(calc, atoms) for atoms in smoke_geometries()]
    layer_records: list[dict[str, Any]] = []
    staged_results: list[dict[str, Any]] = []

    for layer_index, product in enumerate(calc.models[0].products):
        native = product.symmetric_contractions
        reduced = QuotientedScalarSymmetricContraction(native)
        random_check = quotient_random_equivalence(native, reduced)
        product.symmetric_contractions = reduced
        candidate = [evaluate(calc, atoms) for atoms in smoke_geometries()]
        comparison = compare_outputs(reference, candidate)
        layer_records.append(
            {
                "layer": layer_index,
                "quotients": reduced.quotient_metadata,
                "random_feature_equivalence": random_check,
            }
        )
        staged_results.append(
            {
                "layers_replaced": list(range(layer_index + 1)),
                "comparison_to_native": comparison,
            }
        )

    max_random = max(
        record["random_feature_equivalence"]["max_absolute_error"]
        for record in layer_records
    )
    max_energy = max(
        stage["comparison_to_native"]["max_energy_absolute_error_eV"]
        for stage in staged_results
    )
    max_force = max(
        stage["comparison_to_native"]["max_force_absolute_error_eV_per_A"]
        for stage in staged_results
    )
    tolerance = 2e-10 if args.dtype == "float64" else 2e-5
    gate_passed = max(max_random, max_energy, max_force) <= tolerance
    payload = {
        "schema_version": 1,
        "experiment": "E2_live_path_quotient_integration",
        "requested_model": args.model,
        "dtype": args.dtype,
        "layer_records": layer_records,
        "staged_results": staged_results,
        "tolerance": tolerance,
        "gate_passed": gate_passed,
        "scope": (
            "The replacement changes path coordinates only. It does not reduce "
            "the 96-channel representation and is not an ODT truncation."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"gate_passed={gate_passed}")
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
