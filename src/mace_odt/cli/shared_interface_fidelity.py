"""Compare branch-only and all-consumer use of the same functional maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mace_odt.audit import evaluate, write_json
from mace_odt.cli.heldout_branch_fidelity import (
    distribution_summary,
    load_selected_geometries,
    paired_bootstrap,
    validate_manifest_source,
)
from mace_odt.cli.projected_branch_diagnostics import method_bases, retained_rank_ladder
from mace_odt.functional_projector import native_functional_map
from mace_odt.projected_product import ProjectedProductBlock, projected_product_context
from mace_odt.radial_interface import angular_blocks


def validate_branch_results(
    branch_results: dict, manifest: dict, source_sha256: str
) -> None:
    if branch_results.get("manifest_source_sha256") != source_sha256:
        raise ValueError("branch results do not match the current manifest source")
    expected_indices = sorted(record["index"] for record in manifest["selected"])
    if branch_results.get("configuration_count") != len(expected_indices):
        raise ValueError("branch results have a different configuration count")
    for method in branch_results.get("methods", []):
        observed_indices = sorted(
            record["index"] for record in method.get("per_configuration", [])
        )
        if observed_indices != expected_indices:
            raise ValueError("branch results use different configuration indices")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--radial-npz", type=Path, required=True)
    parser.add_argument("--environment-npz", type=Path, required=True)
    parser.add_argument("--branch-results", type=Path, required=True)
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--ranks", type=int, nargs="+", default=[48, 64, 80])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    manifest = json.loads(args.manifest.read_text())
    branch_results = json.loads(args.branch_results.read_text())
    source_sha256 = validate_manifest_source(args.xyz, manifest)
    validate_branch_results(branch_results, manifest, source_sha256)
    geometries = load_selected_geometries(args.xyz, manifest["selected"])
    radial = np.load(args.radial_npz)
    environment = np.load(args.environment_npz)
    if not np.array_equal(radial["atomic_numbers"], environment["atomic_numbers"]):
        raise ValueError("radial and environment artifacts use different species")
    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    model_numbers = model.atomic_numbers.detach().cpu().numpy()
    if not np.array_equal(model_numbers, radial["atomic_numbers"]):
        raise ValueError("the checkpoint does not match the radial artifact")

    atoms_list = [atoms for _, atoms in geometries]
    reference = [evaluate(calc, atoms) for atoms in atoms_list]
    support = int(radial["metric_factor_z0_l0"].shape[1])
    ranks = retained_rank_ladder(args.ranks, support)
    bases = method_bases(environment, support, len(model_numbers), seed=20260924)
    bases = {
        method: values
        for method, values in bases.items()
        if method in ("global_environment", "local_radial_svd")
    }
    native_product = model.products[0]
    angular_slices = {
        block.ell: slice(block.start, block.stop) for block in angular_blocks(model)
    }
    records = []
    for method, basis_by_block in bases.items():
        for retained in ranks:
            maps = {}
            for central in range(len(model_numbers)):
                for ell in range(4):
                    factor = np.asarray(
                        radial[f"metric_factor_z{central}_l{ell}"],
                        dtype=np.float64,
                    )
                    maps[(central, ell)] = native_functional_map(
                        factor, basis_by_block[(central, ell)][:, :retained]
                    )
            replacement = ProjectedProductBlock(
                native_product,
                maps,
                num_elements=len(model_numbers),
                angular_slices=angular_slices,
            )
            with projected_product_context(model, replacement):
                candidate = [evaluate(calc, atoms) for atoms in atoms_list]
            per_configuration = []
            for (metadata, atoms), (energy_ref, force_ref), (energy_new, force_new) in zip(
                geometries, reference, candidate
            ):
                energy_error = energy_new - energy_ref
                force_error = force_new - force_ref
                per_configuration.append(
                    {
                        "index": metadata["index"],
                        "config_type": metadata["config_type"],
                        "num_atoms": len(atoms),
                        "energy_error_eV": energy_error,
                        "energy_absolute_error_per_atom_eV": abs(energy_error) / len(atoms),
                        "force_rmse_eV_per_A": float(np.sqrt(np.mean(force_error**2))),
                        "force_max_absolute_error_eV_per_A": float(
                            np.max(np.abs(force_error))
                        ),
                    }
                )
            branch_record = next(
                item
                for item in branch_results["methods"]
                if item["method"] == method
                and item["multiplicity_rank_per_irrep"] == retained
            )
            shared_force_mean = float(
                np.mean([item["force_rmse_eV_per_A"] for item in per_configuration])
            )
            branch_force_mean = branch_record["force_rmse_eV_per_A"]["mean"]
            records.append(
                {
                    "method": method,
                    "multiplicity_rank_per_irrep": retained,
                    "shared_energy_absolute_error_per_atom_eV": distribution_summary(
                        [
                            item["energy_absolute_error_per_atom_eV"]
                            for item in per_configuration
                        ]
                    ),
                    "shared_force_rmse_eV_per_A": distribution_summary(
                        [item["force_rmse_eV_per_A"] for item in per_configuration]
                    ),
                    "shared_force_max_absolute_error_eV_per_A": distribution_summary(
                        [
                            item["force_max_absolute_error_eV_per_A"]
                            for item in per_configuration
                        ]
                    ),
                    "branch_only_force_rmse_mean_eV_per_A": branch_force_mean,
                    "shared_to_branch_force_rmse_mean_ratio": (
                        shared_force_mean / branch_force_mean
                        if branch_force_mean > 0.0
                        else None
                    ),
                    "per_configuration": per_configuration,
                }
            )
    paired = []
    for rank in ranks:
        global_record = next(
            item
            for item in records
            if item["method"] == "global_environment"
            and item["multiplicity_rank_per_irrep"] == rank
        )
        local_record = next(
            item
            for item in records
            if item["method"] == "local_radial_svd"
            and item["multiplicity_rank_per_irrep"] == rank
        )
        paired.append(
            {
                "multiplicity_rank_per_irrep": rank,
                "shared_energy_absolute_error_per_atom_eV": paired_bootstrap(
                    [
                        item["energy_absolute_error_per_atom_eV"]
                        for item in global_record["per_configuration"]
                    ],
                    [
                        item["energy_absolute_error_per_atom_eV"]
                        for item in local_record["per_configuration"]
                    ],
                    seed=20263000 + rank,
                ),
                "shared_force_rmse_eV_per_A": paired_bootstrap(
                    [
                        item["force_rmse_eV_per_A"]
                        for item in global_record["per_configuration"]
                    ],
                    [
                        item["force_rmse_eV_per_A"]
                        for item in local_record["per_configuration"]
                    ],
                    seed=20264000 + rank,
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "experiment": "E8_shared_interface_fidelity",
        "manifest": str(args.manifest),
        "manifest_source_sha256": source_sha256,
        "branch_results": str(args.branch_results),
        "configuration_count": len(geometries),
        "records": records,
        "paired_local_minus_global": paired,
        "claim_boundary": (
            "The projector is inserted before product zero, so both the first "
            "readout and every later consumer receive the altered features. This "
            "tests fidelity only and does not yet compile a faster architecture."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"configurations={len(geometries)}")


if __name__ == "__main__":
    main()
