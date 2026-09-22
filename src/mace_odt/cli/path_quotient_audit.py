"""Audit exact repeated-slot path redundancy in a MACE-OFF23 checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mace_odt.audit import environment_manifest, write_json
from mace_odt.path_quotient import compute_path_quotient, random_null_perturbation_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    records: list[dict[str, Any]] = []
    all_passed = True

    for layer_index, product in enumerate(model.products):
        contractions = product.symmetric_contractions.contractions
        for output_index, contraction in enumerate(contractions):
            for order in range(1, contraction.correlation + 1):
                coupling = contraction.U_tensors(order).detach().cpu().numpy()
                quotient = compute_path_quotient(coupling, order)
                check = random_null_perturbation_check(
                    coupling=coupling,
                    quotient=quotient,
                    num_elements=contraction.weights_max.shape[0],
                    num_channels=contraction.num_features,
                )
                record = asdict(quotient)
                residual_tolerance = (
                    100.0
                    * np.finfo(coupling.dtype).eps
                    * max(coupling.shape)
                    * max(float(np.linalg.norm(coupling)), 1.0)
                )
                record.update(
                    {
                        "layer": layer_index,
                        "output_index": output_index,
                        "output_irrep": str(product.symmetric_contractions.irreps_out[output_index]),
                        "order": order,
                        "coupling_shape": list(coupling.shape),
                        "random_functional_check": check,
                        "gate_residual_tolerance": residual_tolerance,
                    }
                )
                records.append(record)
                all_passed = all_passed and quotient.null_residual <= residual_tolerance
                all_passed = (
                    all_passed and quotient.projector_residual <= residual_tolerance
                )
                all_passed = all_passed and check["status"] == "passed"

    payload = {
        "schema_version": 1,
        "experiment": "E2_exact_coupling_path_quotient",
        "requested_model": args.model,
        "environment": environment_manifest(),
        "records": records,
        "gate_passed": all_passed and bool(records),
        "scope": (
            "This verifies the serialized repeated-slot coupling maps. A converted "
            "full-checkpoint forward comparison remains a separate integration gate."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"gate_passed={payload['gate_passed']}")
    if not payload["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
