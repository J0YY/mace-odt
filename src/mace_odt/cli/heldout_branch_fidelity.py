"""Evaluate frozen first-branch projectors on an official held-out subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mace_odt.audit import sha256_file, write_json
from mace_odt.cli.projected_branch_diagnostics import (
    density_graph,
    method_bases,
    retained_rank_ladder,
)
from mace_odt.first_branch import native_first_branch_node_energy
from mace_odt.functional_projector import (
    NativeFunctionalMap,
    apply_species_functional_maps,
    native_functional_map,
)


def distribution_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def paired_bootstrap(
    global_values: list[float],
    local_values: list[float],
    seed: int,
    replicates: int = 2000,
) -> dict[str, float]:
    """Return local minus global mean error with a paired percentile interval."""
    global_array = np.asarray(global_values, dtype=np.float64)
    local_array = np.asarray(local_values, dtype=np.float64)
    if global_array.shape != local_array.shape:
        raise ValueError("paired arrays must have equal shapes")
    differences = local_array - global_array
    rng = np.random.default_rng(seed)
    selections = rng.integers(
        0, len(differences), size=(replicates, len(differences))
    )
    means = np.mean(differences[selections], axis=1)
    return {
        "mean_local_minus_global": float(np.mean(differences)),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def load_selected_geometries(
    xyz: Path, selected_records: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], Any]]:
    from ase.io import iread

    by_index = {record["index"]: record for record in selected_records}
    selected = []
    for index, atoms in enumerate(iread(xyz, index=":")):
        if index in by_index:
            selected.append((by_index[index], atoms))
        if len(selected) == len(by_index):
            break
    if len(selected) != len(by_index):
        raise ValueError("the manifest contains indices outside the XYZ dataset")
    selected.sort(key=lambda item: item[0]["index"])
    return selected


def validate_manifest_source(xyz: Path, manifest: dict[str, Any]) -> str:
    """Require the evaluated XYZ file to match the frozen manifest exactly."""
    try:
        expected = manifest["source"]["xyz_sha256"]
    except KeyError as error:
        raise ValueError("the manifest does not declare source.xyz_sha256") from error
    actual = sha256_file(xyz)
    if actual != expected:
        raise ValueError(
            "the supplied XYZ file does not match the frozen manifest: "
            f"expected {expected}, found {actual}"
        )
    return actual


def build_maps(
    radial: Any,
    environment: Any,
    ranks: list[int],
    random_seeds: list[int],
    num_elements: int,
) -> dict[tuple[str, int], dict[tuple[int, int], NativeFunctionalMap]]:
    support = int(radial["metric_factor_z0_l0"].shape[1])
    ranks = retained_rank_ladder(ranks, support)
    basis_sets = method_bases(
        environment, support, num_elements, random_seeds[0]
    )
    basis_sets.pop("canonical_random")
    for seed in random_seeds:
        random_basis = method_bases(environment, support, num_elements, seed)[
            "canonical_random"
        ]
        basis_sets[f"canonical_random_seed_{seed}"] = random_basis
    maps = {}
    for method, bases in basis_sets.items():
        for retained in ranks:
            block_maps = {}
            for central in range(num_elements):
                for ell in range(4):
                    factor = np.asarray(
                        radial[f"metric_factor_z{central}_l{ell}"],
                        dtype=np.float64,
                    )
                    block_maps[(central, ell)] = native_functional_map(
                        factor, bases[(central, ell)][:, :retained]
                    )
            maps[(method, retained)] = block_maps
    return maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--radial-npz", type=Path, required=True)
    parser.add_argument("--environment-npz", type=Path, required=True)
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 48, 64, 80])
    parser.add_argument(
        "--random-seeds", type=int, nargs="+", default=[20260924, 20260925, 20260926]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    manifest = json.loads(args.manifest.read_text())
    source_sha256 = validate_manifest_source(args.xyz, manifest)
    geometries = load_selected_geometries(args.xyz, manifest["selected"])
    radial = np.load(args.radial_npz)
    environment = np.load(args.environment_npz)
    if not np.array_equal(radial["atomic_numbers"], environment["atomic_numbers"]):
        raise ValueError("radial and environment artifacts use different species")
    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    model_numbers = model.atomic_numbers.detach().cpu().numpy()
    if not np.array_equal(model_numbers, radial["atomic_numbers"]):
        raise ValueError("the checkpoint does not match the decomposition artifacts")

    maps = build_maps(
        radial,
        environment,
        args.ranks,
        args.random_seeds,
        len(model_numbers),
    )
    angular_slices = {
        0: slice(0, 1),
        1: slice(1, 4),
        2: slice(4, 9),
        3: slice(9, 16),
    }
    measurements: dict[tuple[str, int], list[dict[str, Any]]] = {
        key: [] for key in maps
    }
    for metadata, atoms in geometries:
        state = density_graph(model, calc, atoms)
        native_energy = float(state["native_energy"].detach().cpu())
        for key, block_maps in maps.items():
            projected_density = apply_species_functional_maps(
                state["density"],
                state["node_attrs"],
                angular_slices,
                block_maps,
            )
            projected_energy = native_first_branch_node_energy(
                model, projected_density, state["node_attrs"]
            ).sum()
            projected_forces = -torch.autograd.grad(
                projected_energy, state["positions"], retain_graph=True
            )[0]
            energy_error = float(projected_energy.detach().cpu()) - native_energy
            force_error = projected_forces - state["native_forces"]
            measurements[key].append(
                {
                    "index": metadata["index"],
                    "config_type": metadata["config_type"],
                    "num_atoms": len(atoms),
                    "energy_error_eV": energy_error,
                    "energy_absolute_error_per_atom_eV": abs(energy_error) / len(atoms),
                    "force_rmse_eV_per_A": float(
                        torch.sqrt(torch.mean(force_error**2)).detach().cpu()
                    ),
                    "force_max_absolute_error_eV_per_A": float(
                        force_error.abs().max().detach().cpu()
                    ),
                }
            )

    method_records = []
    for (method, rank), records in measurements.items():
        method_records.append(
            {
                "method": method,
                "multiplicity_rank_per_irrep": rank,
                "canonical_coordinate_count_per_species": 16 * rank,
                "energy_absolute_error_per_atom_eV": distribution_summary(
                    [record["energy_absolute_error_per_atom_eV"] for record in records]
                ),
                "force_rmse_eV_per_A": distribution_summary(
                    [record["force_rmse_eV_per_A"] for record in records]
                ),
                "force_max_absolute_error_eV_per_A": distribution_summary(
                    [record["force_max_absolute_error_eV_per_A"] for record in records]
                ),
                "per_configuration": records,
            }
        )
    method_records.sort(key=lambda item: (item["method"], item["multiplicity_rank_per_irrep"]))

    paired = []
    support = int(radial["metric_factor_z0_l0"].shape[1])
    for rank in retained_rank_ladder(args.ranks, support):
        global_records = measurements[("global_environment", rank)]
        local_records = measurements[("local_radial_svd", rank)]
        paired.append(
            {
                "multiplicity_rank_per_irrep": rank,
                "energy_absolute_error_per_atom_eV": paired_bootstrap(
                    [record["energy_absolute_error_per_atom_eV"] for record in global_records],
                    [record["energy_absolute_error_per_atom_eV"] for record in local_records],
                    seed=20261000 + rank,
                ),
                "force_rmse_eV_per_A": paired_bootstrap(
                    [record["force_rmse_eV_per_A"] for record in global_records],
                    [record["force_rmse_eV_per_A"] for record in local_records],
                    seed=20262000 + rank,
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "experiment": "E7_frozen_heldout_first_branch_fidelity",
        "manifest": str(args.manifest),
        "manifest_source_sha256": source_sha256,
        "radial_artifact": str(args.radial_npz),
        "environment_artifact": str(args.environment_npz),
        "configuration_count": len(geometries),
        "random_seeds": args.random_seeds,
        "methods": method_records,
        "paired_local_minus_global": paired,
        "claim_boundary": (
            "This measures fidelity to the frozen model's first branch. It does "
            "not measure DFT accuracy, full-model fidelity, or wall-time speedup."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"configurations={len(geometries)}")


if __name__ == "__main__":
    main()
