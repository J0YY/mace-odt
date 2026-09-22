"""Extract and validate the exact first-interaction radial function interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mace_odt.audit import smoke_geometries, write_json
from mace_odt.radial_interface import (
    angular_blocks,
    angular_factorization_check,
    pair_radial_coefficients,
    reconstruct_density_from_pairs,
    supported_metric_factor,
)


def molecule_reconstruction(model: Any, calc: Any, atoms: Any) -> dict[str, Any]:
    from mace.modules.utils import get_edge_vectors_and_lengths

    device = next(model.parameters()).device
    batch = calc._atoms_to_batch(atoms).to(device)
    data = batch.to_dict()
    model_dtype = next(model.parameters()).dtype
    for key, value in list(data.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            data[key] = value.to(dtype=model_dtype)
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
    native_density, native_sc = model.interactions[0](
        node_attrs=data["node_attrs"],
        node_feats=node_features,
        edge_attrs=edge_attrs,
        edge_feats=edge_features,
        edge_index=data["edge_index"],
        cutoff=cutoff,
        first_layer=True,
    )
    if native_sc is not None:
        raise ValueError("the selected first interaction unexpectedly has a residual path")
    reconstructed_density = reconstruct_density_from_pairs(
        model=model,
        node_attrs=data["node_attrs"],
        edge_index=data["edge_index"],
        edge_attrs=edge_attrs,
        edge_lengths=lengths,
    )
    native_features = model.products[0](
        native_density, native_sc, data["node_attrs"]
    )
    reconstructed_features = model.products[0](
        reconstructed_density, native_sc, data["node_attrs"]
    )
    native_node_energy = model.readouts[0](native_features).squeeze(-1)
    reconstructed_node_energy = model.readouts[0](reconstructed_features).squeeze(-1)
    scale = model.scale_shift.scale.to(dtype=model_dtype)
    native_energy = scale * native_node_energy.sum()
    reconstructed_energy = scale * reconstructed_node_energy.sum()
    native_forces = -torch.autograd.grad(
        native_energy, positions, retain_graph=True
    )[0]
    reconstructed_forces = -torch.autograd.grad(
        reconstructed_energy, positions
    )[0]
    density_difference = reconstructed_density - native_density
    force_difference = reconstructed_forces - native_forces
    return {
        "formula": atoms.get_chemical_formula(),
        "density_max_absolute_error": float(
            density_difference.abs().max().detach().cpu()
        ),
        "density_relative_l2_error": float(
            density_difference.norm().detach().cpu()
            / torch.clamp(native_density.norm().detach().cpu(), min=1.0)
        ),
        "branch_energy_absolute_error_eV": float(
            (reconstructed_energy - native_energy).abs().detach().cpu()
        ),
        "branch_force_max_absolute_error_eV_per_A": float(
            force_difference.abs().max().detach().cpu()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--radial-min", type=float, default=0.5)
    parser.add_argument("--radial-max", type=float, default=4.5)
    parser.add_argument("--quadrature-order", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    args = parser.parse_args()
    if args.quadrature_order < 4:
        raise ValueError("quadrature order must be at least four")

    from mace.calculators import mace_off

    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    if abs(float(model.r_max) - args.radial_max) > 1e-12:
        raise ValueError("the declared radial maximum must equal the checkpoint cutoff")

    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(
        args.quadrature_order
    )
    radial_span = args.radial_max - args.radial_min
    radii_np = args.radial_min + 0.5 * radial_span * (legendre_nodes + 1.0)
    radial_weights = 0.5 * legendre_weights
    comparison_order = args.quadrature_order // 2
    comparison_nodes, comparison_weights_raw = np.polynomial.legendre.leggauss(
        comparison_order
    )
    comparison_radii_np = args.radial_min + 0.5 * radial_span * (
        comparison_nodes + 1.0
    )
    comparison_weights = 0.5 * comparison_weights_raw
    num_elements = int(model.atomic_numbers.numel())
    species_weight = 1.0 / num_elements
    blocks = angular_blocks(model)
    summaries = []
    convergence_values = []
    arrays: dict[str, np.ndarray] = {
        "radii_angstrom": radii_np,
        "normalized_uniform_radial_weights": radial_weights,
        "atomic_numbers": model.atomic_numbers.detach().cpu().numpy(),
    }

    for central_index in range(num_elements):
        central = np.repeat(central_index, num_elements * args.quadrature_order)
        neighbors = np.repeat(np.arange(num_elements), args.quadrature_order)
        radii = np.tile(radii_np, num_elements)
        coefficients = pair_radial_coefficients(
            model,
            torch.as_tensor(central, device=device),
            torch.as_tensor(neighbors, device=device),
            torch.as_tensor(radii, dtype=dtype, device=device),
        )
        curves = coefficients.detach().cpu().numpy().reshape(
            num_elements, args.quadrature_order, len(blocks), -1
        )
        comparison_central = np.repeat(
            central_index, num_elements * comparison_order
        )
        comparison_neighbors = np.repeat(np.arange(num_elements), comparison_order)
        comparison_radii = np.tile(comparison_radii_np, num_elements)
        comparison_coefficients = pair_radial_coefficients(
            model,
            torch.as_tensor(comparison_central, device=device),
            torch.as_tensor(comparison_neighbors, device=device),
            torch.as_tensor(comparison_radii, dtype=dtype, device=device),
        )
        comparison_curves = comparison_coefficients.detach().cpu().numpy().reshape(
            num_elements, comparison_order, len(blocks), -1
        )
        arrays[f"curves_central_{central_index}"] = curves
        central_records = []
        for block_index, block in enumerate(blocks):
            table = curves[:, :, block_index, :].reshape(-1, curves.shape[-1])
            row_weights = np.repeat(
                species_weight * radial_weights[None, :], num_elements, axis=0
            ).reshape(-1)
            weighted_table = table * np.sqrt(row_weights[:, None])
            factor, singular_values, rank, tolerance = supported_metric_factor(
                weighted_table
            )
            arrays[f"metric_factor_z{central_index}_l{block.ell}"] = factor
            arrays[f"metric_singular_values_z{central_index}_l{block.ell}"] = (
                singular_values
            )
            metric = weighted_table.T @ weighted_table
            comparison_table = comparison_curves[:, :, block_index, :].reshape(
                -1, comparison_curves.shape[-1]
            )
            comparison_row_weights = np.repeat(
                species_weight * comparison_weights[None, :],
                num_elements,
                axis=0,
            ).reshape(-1)
            comparison_weighted_table = comparison_table * np.sqrt(
                comparison_row_weights[:, None]
            )
            comparison_metric = comparison_weighted_table.T @ comparison_weighted_table
            convergence = float(
                np.linalg.norm(metric - comparison_metric)
                / max(float(np.linalg.norm(metric)), np.finfo(np.float64).tiny)
            )
            convergence_values.append(convergence)
            factor_residual = np.linalg.norm(metric - factor @ factor.T)
            central_records.append(
                {
                    "ell": block.ell,
                    "supported_rank": rank,
                    "native_channels": int(curves.shape[-1]),
                    "rank_tolerance": tolerance,
                    "largest_singular_value": float(singular_values[0]),
                    "smallest_retained_singular_value": float(
                        singular_values[rank - 1]
                    ),
                    "factor_residual": float(factor_residual),
                    "quadrature_convergence_relative_metric_change": convergence,
                    "singular_values": singular_values.tolist(),
                }
            )
        summaries.append(
            {
                "central_index": central_index,
                "atomic_number": int(model.atomic_numbers[central_index]),
                "blocks": central_records,
            }
        )

    angular_checks = [
        angular_factorization_check(model, 1, 0, 1.1),
        angular_factorization_check(model, 0, 1, 1.4),
        angular_factorization_check(model, 3, 2, 2.0),
    ]
    molecule_checks = [
        molecule_reconstruction(model, calc, atoms) for atoms in smoke_geometries()
    ]
    max_angular = max(
        record["max_absolute_error"]
        for check in angular_checks
        for record in check["records"]
    )
    max_density = max(
        record["density_max_absolute_error"] for record in molecule_checks
    )
    max_energy = max(
        record["branch_energy_absolute_error_eV"] for record in molecule_checks
    )
    max_force = max(
        record["branch_force_max_absolute_error_eV_per_A"]
        for record in molecule_checks
    )
    tolerance = 2e-10 if args.dtype == "float64" else 2e-5
    quadrature_tolerance = 2e-6 if args.dtype == "float64" else 1e-4
    maximum_quadrature_change = max(convergence_values)
    gate_passed = (
        max(max_angular, max_density, max_energy, max_force) <= tolerance
        and maximum_quadrature_change <= quadrature_tolerance
    )
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz, **arrays)
    payload = {
        "schema_version": 1,
        "experiment": "E3_first_layer_radial_interface",
        "requested_model": args.model,
        "measure": {
            "neighbor_species": "equal",
            "radial_weight": "normalized_uniform_distance",
            "radial_min_angstrom": args.radial_min,
            "radial_max_angstrom": args.radial_max,
            "quadrature": "Gauss-Legendre",
            "quadrature_order": args.quadrature_order,
            "quadrature_comparison_order": comparison_order,
            "angular_measure": "uniform_sphere_with_component_normalized_harmonics",
        },
        "central_species": summaries,
        "angular_factorization_checks": angular_checks,
        "molecule_reconstruction_checks": molecule_checks,
        "tolerance": tolerance,
        "quadrature_convergence_tolerance": quadrature_tolerance,
        "maximum_quadrature_relative_metric_change": maximum_quadrature_change,
        "gate_passed": gate_passed,
        "npz": str(args.npz),
        "scope": (
            "This establishes the registered upstream function map and its "
            "metric. It does not select downstream-important modes."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.npz}")
    print(f"gate_passed={gate_passed}")
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
