"""Construct the mixed-order global environment and its certified rank ladder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from mace_odt.audit import write_json
from mace_odt.decomposition import symmetric_eigh_descending
from mace_odt.global_environment import (
    angular_ell_labels,
    angular_metric,
    canonical_environment_trace,
    coefficient_norm_squared,
    extract_equivariant_environment,
    multiplicity_blocks_from_native,
    native_slot_marginal,
    projected_metric_by_angular,
)


def spectrum_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(np.sum(values))
    cumulative = np.cumsum(values)
    ranks = {}
    for fraction in (0.9, 0.95, 0.99, 0.999):
        ranks[str(fraction)] = (
            int(np.searchsorted(cumulative, fraction * total) + 1)
            if total > 0.0
            else 0
        )
    return {
        "trace": total,
        "largest_eigenvalue": float(values[0]) if values.size else 0.0,
        "smallest_eigenvalue": float(values[-1]) if values.size else 0.0,
        "effective_rank_trace_over_max": (
            total / float(values[0]) if values.size and values[0] > 0.0 else 0.0
        ),
        "ranks_for_trace_fraction": ranks,
        "eigenvalues": values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radial-npz", type=Path, required=True)
    parser.add_argument("--branch-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument(
        "--beta-convention",
        choices=("equal", "coefficient-balanced"),
        default="equal",
    )
    parser.add_argument(
        "--uniform-ranks",
        type=int,
        nargs="+",
        default=[0, 4, 8, 16, 32, 48, 64, 80, 96],
    )
    args = parser.parse_args()
    if any(rank < 0 for rank in args.uniform_ranks):
        raise ValueError("uniform ranks must be nonnegative")

    radial = np.load(args.radial_npz)
    branch = np.load(args.branch_npz)
    radial_numbers = radial["atomic_numbers"]
    branch_numbers = branch["atomic_numbers"]
    if not np.array_equal(radial_numbers, branch_numbers):
        raise ValueError("radial and branch artifacts use different element orders")
    ell_labels = angular_ell_labels({0: 1, 1: 3, 2: 5, 3: 7})
    orders = {
        order: np.asarray(branch[f"coefficients_order_{order}"], dtype=np.float64)
        for order in (1, 2, 3)
    }
    if any(tensor.shape[2:] != (len(ell_labels),) * order for order, tensor in orders.items()):
        raise ValueError("branch angular axes do not match complete ell zero to three blocks")
    arrays: dict[str, np.ndarray] = {
        "atomic_numbers": branch_numbers,
        "angular_ell_labels": ell_labels,
    }
    beta_rows = []
    species_records = []
    all_gate_values: list[float] = []

    for central_index, atomic_number in enumerate(branch_numbers):
        factors = {
            ell: np.asarray(
                radial[f"metric_factor_z{central_index}_l{ell}"], dtype=np.float64
            )
            for ell in range(4)
        }
        metric = angular_metric(factors, ell_labels)
        order_records = []
        order_norms: dict[int, float] = {}
        order_environments: dict[int, np.ndarray] = {}
        for order in (1, 2, 3):
            coefficients = orders[order][central_index]
            norm_squared = coefficient_norm_squared(coefficients, metric)
            order_norms[order] = norm_squared
            slot_marginals = [
                native_slot_marginal(coefficients, metric, slot)
                for slot in range(order)
            ]
            summed = sum(slot_marginals)
            order_environments[order] = summed
            slot_reference = max(float(np.linalg.norm(slot_marginals[0])), 1e-300)
            slot_deviation = max(
                (
                    float(np.linalg.norm(item - slot_marginals[0])) / slot_reference
                    for item in slot_marginals
                ),
                default=0.0,
            )
            slot_trace_residuals = [
                abs(canonical_environment_trace(item, factors, ell_labels) - norm_squared)
                / max(abs(norm_squared), 1e-300)
                for item in slot_marginals
            ]
            order_blocks = multiplicity_blocks_from_native(
                summed, factors, ell_labels
            )
            order_spectra = {}
            for ell, block in order_blocks.items():
                values, _ = symmetric_eigh_descending(block)
                order_spectra[str(ell)] = spectrum_summary(values)
                arrays[f"gamma_order{order}_z{central_index}_l{ell}"] = block
            order_records.append(
                {
                    "order": order,
                    "coefficient_norm_squared_eV2": norm_squared,
                    "slot_symmetry_relative_residual": slot_deviation,
                    "slot_trace_relative_residuals": slot_trace_residuals,
                    "multiplicity_spectra": order_spectra,
                }
            )
            all_gate_values.extend([slot_deviation, *slot_trace_residuals])

        if args.beta_convention == "equal":
            beta = {1: 1.0, 2: 1.0, 3: 1.0}
        else:
            beta = {
                order: 1.0 / order_norms[order] for order in (1, 2, 3)
            }
        beta_rows.append([beta[1], beta[2], beta[3]])
        for record in order_records:
            record["beta"] = beta[record["order"]]
        aggregate_native = sum(
            beta[order] * order_environments[order] for order in (1, 2, 3)
        )
        aggregate = extract_equivariant_environment(
            aggregate_native, factors, ell_labels
        )
        expected_trace = sum(
            order * beta[order] * order_norms[order] for order in (1, 2, 3)
        )
        trace_relative_residual = abs(aggregate.full_trace - expected_trace) / max(
            abs(expected_trace), 1e-300
        )
        block_trace_relative_residual = abs(
            aggregate.full_trace - aggregate.block_trace
        ) / max(abs(aggregate.full_trace), 1e-300)
        aggregate_spectra = {}
        for ell in range(4):
            arrays[f"gamma_z{central_index}_l{ell}"] = aggregate.multiplicity_blocks[
                ell
            ]
            arrays[f"gamma_eigenvalues_z{central_index}_l{ell}"] = aggregate.eigenvalues[
                ell
            ]
            arrays[f"gamma_eigenvectors_z{central_index}_l{ell}"] = aggregate.eigenvectors[
                ell
            ]
            aggregate_spectra[str(ell)] = spectrum_summary(
                aggregate.eigenvalues[ell]
            )

        ladder = []
        full_weighted_norm = sum(
            beta[order] * order_norms[order] for order in (1, 2, 3)
        )
        for requested_rank in args.uniform_ranks:
            retained = {
                ell: min(requested_rank, factors[ell].shape[1]) for ell in range(4)
            }
            projectors = {}
            for ell in range(4):
                vectors = aggregate.eigenvectors[ell][:, : retained[ell]]
                projectors[ell] = vectors @ vectors.T
            projected_metric = projected_metric_by_angular(
                factors, projectors, ell_labels
            )
            projected_norms = {
                order: coefficient_norm_squared(
                    orders[order][central_index], projected_metric
                )
                for order in (1, 2, 3)
            }
            actual = sum(
                beta[order] * max(order_norms[order] - projected_norms[order], 0.0)
                for order in (1, 2, 3)
            )
            bound = sum(
                (2 * ell + 1)
                * float(np.sum(aggregate.eigenvalues[ell][retained[ell] :]))
                for ell in range(4)
            )
            inequality_scale = max(full_weighted_norm, expected_trace, 1.0)
            inequality_tolerance = 2e-10 * inequality_scale
            lower_passed = actual <= bound + inequality_tolerance
            upper_passed = bound <= 3.0 * actual + inequality_tolerance
            ladder.append(
                {
                    "uniform_multiplicity_rank": requested_rank,
                    "retained_multiplicity_ranks": {
                        str(ell): retained[ell] for ell in range(4)
                    },
                    "retained_canonical_coordinates": sum(
                        (2 * ell + 1) * retained[ell] for ell in range(4)
                    ),
                    "actual_coefficient_error_squared_eV2": actual,
                    "marginal_tail_bound_eV2": bound,
                    "actual_relative_to_full_coefficient_norm": (
                        actual / full_weighted_norm if full_weighted_norm > 0.0 else 0.0
                    ),
                    "bound_relative_to_full_coefficient_norm": (
                        bound / full_weighted_norm if full_weighted_norm > 0.0 else 0.0
                    ),
                    "lower_bound_inequality_passed": lower_passed,
                    "factor_three_inequality_passed": upper_passed,
                    "inequality_tolerance_eV2": inequality_tolerance,
                }
            )
            if not lower_passed or not upper_passed:
                all_gate_values.append(np.inf)

        psd_scale = max(
            max(float(values[0]) for values in aggregate.eigenvalues.values()), 1.0
        )
        psd_relative_violation = max(0.0, -aggregate.minimum_eigenvalue) / psd_scale
        all_gate_values.extend(
            [
                trace_relative_residual,
                block_trace_relative_residual,
                aggregate.relative_off_block_norm,
                aggregate.relative_magnetic_deviation,
                psd_relative_violation,
            ]
        )
        species_records.append(
            {
                "central_index": central_index,
                "atomic_number": int(atomic_number),
                "orders": order_records,
                "aggregate": {
                    "beta": {str(key): value for key, value in beta.items()},
                    "expected_trace_from_occurrence_weighted_norms": expected_trace,
                    "canonical_trace": aggregate.full_trace,
                    "trace_relative_residual": trace_relative_residual,
                    "block_trace_relative_residual": block_trace_relative_residual,
                    "relative_off_block_norm": aggregate.relative_off_block_norm,
                    "relative_magnetic_deviation": aggregate.relative_magnetic_deviation,
                    "minimum_raw_block_eigenvalue": aggregate.minimum_eigenvalue,
                    "psd_relative_violation": psd_relative_violation,
                    "multiplicity_spectra": aggregate_spectra,
                },
                "uniform_rank_ladder": ladder,
            }
        )

    diagnostic_tolerance = 2e-9
    gate_passed = max(all_gate_values, default=0.0) <= diagnostic_tolerance
    arrays["beta_by_central_species_and_order"] = np.asarray(beta_rows)
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz, **arrays)
    payload = {
        "schema_version": 1,
        "experiment": "E5_global_mixed_order_environment",
        "radial_artifact": str(args.radial_npz),
        "branch_artifact": str(args.branch_npz),
        "order_weight_convention": args.beta_convention,
        "order_weight_definition": (
            "beta_nu equals one for every order"
            if args.beta_convention == "equal"
            else "beta_z_nu equals the inverse full coefficient norm squared for that central species and order"
        ),
        "coefficient_norm": (
            "Independent-slot L2 coefficient norm under the frozen radial, species, "
            "and uniform-sphere measure. This is not an activation distribution."
        ),
        "algorithm_alignment": (
            "The radial factors orthogonalize the upstream function map as in Dooms "
            "Algorithm 1. Every native slot marginal contracts the complete sibling "
            "environment as in Algorithm 2. The same equivariant projector is then "
            "used in every repeated slot, which is the mixed-order specialization "
            "required here rather than verbatim independent tree cuts."
        ),
        "central_species": species_records,
        "diagnostic_tolerance": diagnostic_tolerance,
        "gate_passed": gate_passed,
        "npz": str(args.npz),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"wrote {args.npz}")
    print(f"gate_passed={gate_passed}")
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
