"""Run an end-to-end diagnostic rank ladder on the compiled first branch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mace_odt.audit import smoke_geometries, write_json
from mace_odt.first_branch import native_first_branch_node_energy
from mace_odt.functional_projector import (
    apply_species_functional_maps,
    native_functional_map,
)


def density_graph(model: Any, calc: Any, atoms: Any) -> dict[str, Any]:
    from mace.modules.utils import get_edge_vectors_and_lengths

    device = next(model.parameters()).device
    batch = calc._atoms_to_batch(atoms).to(device)
    data = batch.to_dict()
    dtype = next(model.parameters()).dtype
    for key, value in list(data.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            data[key] = value.to(dtype=dtype)
    positions = data["positions"].detach().clone().requires_grad_(True)
    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=data["edge_index"],
        shifts=data["shifts"],
    )
    edge_attrs = model.spherical_harmonics(vectors)
    edge_features, cutoff = model.radial_embedding(
        lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
    )
    node_features = model.node_embedding(data["node_attrs"])
    density, residual = model.interactions[0](
        node_attrs=data["node_attrs"],
        node_feats=node_features,
        edge_attrs=edge_attrs,
        edge_feats=edge_features,
        edge_index=data["edge_index"],
        cutoff=cutoff,
        first_layer=True,
    )
    if residual is not None:
        raise ValueError("the registered first interaction unexpectedly has a residual")
    native_nodes = native_first_branch_node_energy(model, density, data["node_attrs"])
    native_energy = native_nodes.sum()
    native_forces = -torch.autograd.grad(
        native_energy, positions, retain_graph=True
    )[0]
    return {
        "positions": positions,
        "node_attrs": data["node_attrs"],
        "density": density,
        "native_energy": native_energy,
        "native_forces": native_forces,
    }


def method_bases(
    environment: Any,
    support: int,
    num_elements: int,
    seed: int,
) -> dict[str, dict[tuple[int, int], np.ndarray]]:
    bases: dict[str, dict[tuple[int, int], np.ndarray]] = {
        "global_environment": {},
        "local_radial_svd": {},
        "canonical_random": {},
    }
    rng = np.random.default_rng(seed)
    for central in range(num_elements):
        for ell in range(4):
            bases["global_environment"][(central, ell)] = np.asarray(
                environment[f"gamma_eigenvectors_z{central}_l{ell}"],
                dtype=np.float64,
            )
            bases["local_radial_svd"][(central, ell)] = np.eye(support)
            bases["canonical_random"][(central, ell)] = np.linalg.qr(
                rng.normal(size=(support, support))
            )[0]
    return bases


def retained_rank_ladder(requested: list[int], support: int) -> list[int]:
    if support <= 0:
        raise ValueError("support must be positive")
    if any(rank < 0 for rank in requested):
        raise ValueError("ranks must be nonnegative")
    return sorted({min(rank, support) for rank in requested})


def complete_rank_ladder(requested: list[int], support: int) -> list[int]:
    return sorted(set(retained_rank_ladder(requested, support)) | {support})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radial-npz", type=Path, required=True)
    parser.add_argument("--environment-npz", type=Path, required=True)
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 48, 64, 80, 96])
    parser.add_argument("--random-seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    radial = np.load(args.radial_npz)
    environment = np.load(args.environment_npz)
    if not np.array_equal(radial["atomic_numbers"], environment["atomic_numbers"]):
        raise ValueError("radial and environment artifacts use different species")
    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    model_numbers = model.atomic_numbers.detach().cpu().numpy()
    if not np.array_equal(model_numbers, radial["atomic_numbers"]):
        raise ValueError("the live checkpoint does not match the decomposition artifacts")

    num_elements = len(model_numbers)
    support = int(radial["metric_factor_z0_l0"].shape[1])
    ranks = complete_rank_ladder(args.ranks, support)
    bases = method_bases(environment, support, num_elements, args.random_seed)
    angular_slices = {
        0: slice(0, 1),
        1: slice(1, 4),
        2: slice(4, 9),
        3: slice(9, 16),
    }
    geometry_states = [
        (atoms, density_graph(model, calc, atoms)) for atoms in smoke_geometries()
    ]
    method_records = []
    all_map_residuals = []
    for method, basis_by_block in bases.items():
        rank_records = []
        for retained in ranks:
            functional_maps = {}
            inverse_pair_residuals = []
            relative_idempotence_residuals = []
            for central in range(num_elements):
                for ell in range(4):
                    factor = np.asarray(
                        radial[f"metric_factor_z{central}_l{ell}"],
                        dtype=np.float64,
                    )
                    modes = basis_by_block[(central, ell)][:, :retained]
                    functional_map = native_functional_map(factor, modes)
                    functional_maps[(central, ell)] = functional_map
                    inverse_pair_residuals.append(
                        max(
                            functional_map.left_inverse_residual,
                            functional_map.inverse_pair_residual,
                        )
                    )
                    relative_idempotence_residuals.append(
                        functional_map.relative_idempotence_residual
                    )
            geometry_records = []
            for atoms, state in geometry_states:
                projected_density = apply_species_functional_maps(
                    state["density"],
                    state["node_attrs"],
                    angular_slices,
                    functional_maps,
                )
                projected_energy = native_first_branch_node_energy(
                    model, projected_density, state["node_attrs"]
                ).sum()
                projected_forces = -torch.autograd.grad(
                    projected_energy, state["positions"], retain_graph=True
                )[0]
                energy_error = projected_energy - state["native_energy"]
                force_error = projected_forces - state["native_forces"]
                geometry_records.append(
                    {
                        "formula": atoms.get_chemical_formula(),
                        "native_branch_energy_eV": float(
                            state["native_energy"].detach().cpu()
                        ),
                        "projected_branch_energy_eV": float(
                            projected_energy.detach().cpu()
                        ),
                        "energy_error_eV": float(energy_error.detach().cpu()),
                        "energy_absolute_error_eV": float(
                            energy_error.abs().detach().cpu()
                        ),
                        "force_rmse_eV_per_A": float(
                            torch.sqrt(torch.mean(force_error**2)).detach().cpu()
                        ),
                        "force_max_absolute_error_eV_per_A": float(
                            force_error.abs().max().detach().cpu()
                        ),
                    }
                )
            maximum_map_residual = max(inverse_pair_residuals)
            all_map_residuals.append(maximum_map_residual)
            rank_records.append(
                {
                    "multiplicity_rank_per_irrep": retained,
                    "canonical_coordinate_count_per_species": 16 * retained,
                    "maximum_map_algebra_residual": maximum_map_residual,
                    "maximum_relative_projector_idempotence_residual": max(
                        relative_idempotence_residuals
                    ),
                    "geometries": geometry_records,
                    "maximum_energy_absolute_error_eV": max(
                        record["energy_absolute_error_eV"]
                        for record in geometry_records
                    ),
                    "maximum_force_absolute_error_eV_per_A": max(
                        record["force_max_absolute_error_eV_per_A"]
                        for record in geometry_records
                    ),
                }
            )
        method_records.append({"method": method, "rank_ladder": rank_records})

    full_rank_records = [
        rank
        for method in method_records
        for rank in method["rank_ladder"]
        if rank["multiplicity_rank_per_irrep"] == support
    ]
    full_rank_error = max(
        max(
            record["maximum_energy_absolute_error_eV"],
            record["maximum_force_absolute_error_eV_per_A"],
        )
        for record in full_rank_records
    )
    gate_full_rank_records = [
        rank
        for method in method_records
        if method["method"] in ("global_environment", "local_radial_svd")
        for rank in method["rank_ladder"]
        if rank["multiplicity_rank_per_irrep"] == support
    ]
    gate_full_rank_error = max(
        max(
            record["maximum_energy_absolute_error_eV"],
            record["maximum_force_absolute_error_eV_per_A"],
        )
        for record in gate_full_rank_records
    )
    full_rank_tolerance = 5e-10 if args.dtype == "float64" else 5e-4
    gate_passed = (
        gate_full_rank_error <= full_rank_tolerance
        and max(all_map_residuals) <= 5e-5
    )
    payload = {
        "schema_version": 1,
        "experiment": "E6_projected_first_branch_diagnostics",
        "radial_artifact": str(args.radial_npz),
        "environment_artifact": str(args.environment_npz),
        "random_seed": args.random_seed,
        "methods": method_records,
        "full_rank_maximum_energy_or_force_error": full_rank_error,
        "full_rank_gate_methods": ["global_environment", "local_radial_svd"],
        "full_rank_gate_maximum_energy_or_force_error": gate_full_rank_error,
        "map_residual_tolerance": 5e-5,
        "full_rank_tolerance": full_rank_tolerance,
        "gate_passed": gate_passed,
        "scope": (
            "These two molecules are integration diagnostics only. They are not a "
            "held-out fidelity result and cannot establish a scientific advantage."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"gate_passed={gate_passed}")
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
