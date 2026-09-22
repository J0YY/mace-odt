"""Build a data-assisted gradient-sensitivity pilot for both MACE readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mace_odt.audit import find_checkpoint_candidates, sha256_file, write_json
from mace_odt.cli.heldout_branch_fidelity import (
    load_selected_geometries,
    validate_manifest_source,
)
from mace_odt.functional_projector import supported_left_inverse
from mace_odt.multi_consumer import (
    BlockKey,
    canonical_activation_gram,
    canonical_gradient_gram,
    deterministic_completed_eigensystem,
    descending_eigensystem,
    globally_trace_balanced_blocks,
    projector_distance,
    symmetric_psd,
)
from mace_odt.radial_interface import angular_blocks, pair_radial_coefficients


def _scale(model: Any) -> torch.Tensor:
    scale = model.scale_shift.scale
    if scale.numel() != 1:
        raise ValueError("the pilot currently requires one checkpoint head")
    return scale.reshape(())


def validate_radial_artifact(model: Any, radial: Any) -> dict[str, float]:
    """Bind every stored radial curve and metric factor to the live checkpoint."""
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    radii = np.asarray(radial["radii_angstrom"], dtype=np.float64)
    weights = np.asarray(
        radial["normalized_uniform_radial_weights"], dtype=np.float64
    )
    num_elements = int(model.atomic_numbers.numel())
    maximum_curve_error = 0.0
    maximum_metric_relative_error = 0.0
    for central in range(num_elements):
        central_indices = np.repeat(central, num_elements * len(radii))
        neighbor_indices = np.repeat(np.arange(num_elements), len(radii))
        pair_radii = np.tile(radii, num_elements)
        observed = pair_radial_coefficients(
            model,
            torch.as_tensor(central_indices, device=device),
            torch.as_tensor(neighbor_indices, device=device),
            torch.as_tensor(pair_radii, dtype=dtype, device=device),
        ).detach().cpu().numpy().reshape(num_elements, len(radii), 4, -1)
        expected = np.asarray(radial[f"curves_central_{central}"], dtype=np.float64)
        maximum_curve_error = max(
            maximum_curve_error, float(np.max(np.abs(observed - expected)))
        )
        row_weights = np.repeat(
            (weights / num_elements)[None, :], num_elements, axis=0
        ).reshape(-1)
        for ell in range(4):
            table = expected[:, :, ell, :].reshape(-1, expected.shape[-1])
            metric = (table * np.sqrt(row_weights[:, None])).T @ (
                table * np.sqrt(row_weights[:, None])
            )
            factor = np.asarray(
                radial[f"metric_factor_z{central}_l{ell}"], dtype=np.float64
            )
            reconstructed = factor @ factor.T
            relative = float(
                np.linalg.norm(reconstructed - metric)
                / max(np.linalg.norm(metric), np.finfo(np.float64).tiny)
            )
            maximum_metric_relative_error = max(
                maximum_metric_relative_error, relative
            )
    return {
        "maximum_curve_absolute_error": maximum_curve_error,
        "maximum_metric_relative_error": maximum_metric_relative_error,
    }


def branch_graph(model: Any, calc: Any, atoms: Any) -> dict[str, Any]:
    """Evaluate the two exact learned branches from the first density cut."""
    from mace.modules.utils import get_edge_vectors_and_lengths

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch = calc._atoms_to_batch(atoms).to(device)
    data = batch.to_dict()
    for key, value in list(data.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            data[key] = value.to(dtype=dtype)
    positions = data["positions"].detach().clone().requires_grad_(True)
    data["positions"] = positions
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
    density0, residual0 = model.interactions[0](
        node_attrs=data["node_attrs"],
        node_feats=node_features,
        edge_attrs=edge_attrs,
        edge_feats=edge_features,
        edge_index=data["edge_index"],
        cutoff=cutoff,
        first_layer=True,
    )
    if residual0 is not None:
        raise ValueError("the selected first interaction unexpectedly has a residual")
    features0 = model.products[0](density0, residual0, data["node_attrs"])
    density1, residual1 = model.interactions[1](
        node_attrs=data["node_attrs"],
        node_feats=features0,
        edge_attrs=edge_attrs,
        edge_feats=edge_features,
        edge_index=data["edge_index"],
        cutoff=cutoff,
        first_layer=False,
    )
    if residual1 is None:
        raise ValueError("the selected second interaction is missing its skip path")
    features1 = model.products[1](density1, residual1, data["node_attrs"])
    scale = _scale(model)
    linear_nodes = scale * model.readouts[0](features0).squeeze(-1)
    head_preactivations = model.readouts[1].linear_1(features1)
    nonlinear_nodes = scale * model.readouts[1](features1).squeeze(-1)
    if linear_nodes.ndim != 1 or nonlinear_nodes.shape != linear_nodes.shape:
        raise ValueError("readout outputs do not have one scalar per node")
    return {
        "data": data,
        "positions": positions,
        "density0": density0,
        "features0": features0,
        "density1": density1,
        "residual1": residual1,
        "features1": features1,
        "head_preactivations": head_preactivations,
        "linear_nodes": linear_nodes,
        "nonlinear_nodes": nonlinear_nodes,
    }


def checkpoint_trace_residuals(model: Any, state: dict[str, Any]) -> dict[str, float]:
    """Compare the manually exposed graph with the checkpoint forward graph."""
    captured: dict[str, Any] = {}
    handles = []

    def capture(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[name] = output

        return hook

    modules = {
        "interaction0": model.interactions[0],
        "product0": model.products[0],
        "interaction1": model.interactions[1],
        "product1": model.products[1],
        "readout0": model.readouts[0],
        "readout1": model.readouts[1],
    }
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture(name)))
        model(state["data"], compute_force=False)
    finally:
        for handle in handles:
            handle.remove()

    expected = {
        "interaction0_density": state["density0"],
        "product0": state["features0"],
        "interaction1_density": state["density1"],
        "interaction1_residual": state["residual1"],
        "product1": state["features1"],
        "readout0": state["linear_nodes"] / _scale(model),
        "readout1": state["nonlinear_nodes"] / _scale(model),
    }
    observed = {
        "interaction0_density": captured["interaction0"][0],
        "product0": captured["product0"],
        "interaction1_density": captured["interaction1"][0],
        "interaction1_residual": captured["interaction1"][1],
        "product1": captured["product1"],
        "readout0": captured["readout0"].squeeze(-1),
        "readout1": captured["readout1"].squeeze(-1),
    }
    return {
        name: float((observed[name] - tensor).abs().max().detach().cpu())
        for name, tensor in expected.items()
    }


def add_blocks(
    accumulator: dict[BlockKey, np.ndarray], contribution: dict[BlockKey, np.ndarray]
) -> None:
    for key, value in contribution.items():
        accumulator[key] += value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--radial-npz", type=Path, required=True)
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--convergence-ranks", type=int, nargs="+", default=[48, 64, 80])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    manifest = json.loads(args.manifest.read_text())
    source_sha256 = validate_manifest_source(args.xyz, manifest)
    geometries = load_selected_geometries(args.xyz, manifest["selected"])
    radial = np.load(args.radial_npz)
    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    checkpoint_candidates = find_checkpoint_candidates(args.model)
    checkpoint_hashes = {sha256_file(path) for path in checkpoint_candidates}
    if len(checkpoint_hashes) != 1:
        raise ValueError("could not identify one unique cached checkpoint digest")
    checkpoint_sha256 = checkpoint_hashes.pop()
    model_numbers = model.atomic_numbers.detach().cpu().numpy()
    if not np.array_equal(model_numbers, radial["atomic_numbers"]):
        raise ValueError("checkpoint and radial artifact species orders differ")
    radial_checkpoint_validation = validate_radial_artifact(model, radial)
    blocks = angular_blocks(model)
    angular_slices = {block.ell: slice(block.start, block.stop) for block in blocks}
    angular_dimensions = {block.ell: block.dimension for block in blocks}
    if set(angular_slices) != set(range(4)):
        raise ValueError("the registered pilot expects ell zero through three")
    num_elements = len(model_numbers)
    factors = {
        (central, ell): np.asarray(
            radial[f"metric_factor_z{central}_l{ell}"], dtype=np.float64
        )
        for central in range(num_elements)
        for ell in angular_slices
    }
    supports = {factor.shape[1] for factor in factors.values()}
    if len(supports) != 1:
        raise ValueError("all registered functional blocks must have equal support")
    support = supports.pop()
    left_inverses = {key: supported_left_inverse(value) for key, value in factors.items()}
    keys = sorted(factors)

    names = ("linear", "nonlinear", "total", "activation")
    accumulated = {
        name: {key: np.zeros((support, support), dtype=np.float64) for key in keys}
        for name in names
    }
    folded = {
        fold: {
            name: {key: np.zeros((support, support), dtype=np.float64) for key in keys}
            for name in ("linear", "nonlinear")
        }
        for fold in ("A", "B")
    }
    fold_counts = {"A": 0, "B": 0}
    fold_species_counts = {
        fold: {central: 0 for central in range(num_elements)} for fold in ("A", "B")
    }
    species_configuration_counts = {central: 0 for central in range(num_elements)}
    trace_checks = []
    gradient_checks = []

    for geometry_index, (metadata, atoms) in enumerate(geometries):
        state = branch_graph(model, calc, atoms)
        if geometry_index < 2:
            trace_checks.append(
                {
                    "index": metadata["index"],
                    "residuals": checkpoint_trace_residuals(model, state),
                }
            )
        normalizer = np.sqrt(len(atoms))
        linear_energy = state["linear_nodes"].sum() / normalizer
        nonlinear_energy = state["nonlinear_nodes"].sum() / normalizer
        linear_gradient = torch.autograd.grad(
            linear_energy, state["density0"], retain_graph=True
        )[0]
        nonlinear_gradient = torch.autograd.grad(
            nonlinear_energy, state["density0"], retain_graph=True
        )[0]
        total_gradient = torch.autograd.grad(
            linear_energy + nonlinear_energy, state["density0"], retain_graph=True
        )[0]
        gradient_residual = float(
            (total_gradient - linear_gradient - nonlinear_gradient)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        gradient_checks.append(gradient_residual)
        contributions = {
            "linear": canonical_gradient_gram(
                linear_gradient, state["data"]["node_attrs"], factors, angular_slices
            ),
            "nonlinear": canonical_gradient_gram(
                nonlinear_gradient, state["data"]["node_attrs"], factors, angular_slices
            ),
            "total": canonical_gradient_gram(
                total_gradient, state["data"]["node_attrs"], factors, angular_slices
            ),
            "activation": canonical_activation_gram(
                state["density0"],
                state["data"]["node_attrs"],
                left_inverses,
                angular_slices,
            ),
        }
        for key in contributions["activation"]:
            contributions["activation"][key] /= len(atoms)
        for name in names:
            add_blocks(accumulated[name], contributions[name])
        present_species = set(
            state["data"]["node_attrs"].argmax(dim=-1).detach().cpu().tolist()
        )
        for central in present_species:
            species_configuration_counts[int(central)] += 1
        fold = metadata["convergence_fold"]
        if fold not in folded:
            raise ValueError("manifest convergence_fold must be A or B")
        fold_counts[fold] += 1
        for central in present_species:
            fold_species_counts[fold][int(central)] += 1
        add_blocks(folded[fold]["linear"], contributions["linear"])
        add_blocks(folded[fold]["nonlinear"], contributions["nonlinear"])

    configuration_count = len(geometries)
    for name in names:
        for key in keys:
            accumulated[name][key] /= configuration_count
    for fold in folded:
        if fold_counts[fold] == 0:
            raise ValueError("both deterministic convergence folds must be nonempty")
        for name in folded[fold]:
            for key in keys:
                folded[fold][name][key] /= fold_counts[fold]

    psd_absolute_violations = []
    psd_relative_violations = []
    for family in [*accumulated.values(), *(item for fold in folded.values() for item in fold.values())]:
        for key in keys:
            family[key], absolute, relative = symmetric_psd(family[key])
            psd_absolute_violations.append(absolute)
            psd_relative_violations.append(relative)
    balanced, normalization = globally_trace_balanced_blocks(
        accumulated["linear"],
        accumulated["nonlinear"],
        angular_dimensions,
    )
    folded_balanced = {
        fold: globally_trace_balanced_blocks(
            values["linear"], values["nonlinear"], angular_dimensions
        )[0]
        for fold, values in folded.items()
    }

    manifest_sha256 = sha256_file(args.manifest)
    radial_sha256 = sha256_file(args.radial_npz)
    evaluation_indices_sha256 = manifest.get("evaluation_indices_sha256")
    if not evaluation_indices_sha256:
        raise ValueError("discovery manifest does not bind the excluded evaluation indices")
    arrays: dict[str, np.ndarray] = {
        "atomic_numbers": model_numbers,
        "provenance_schema": np.asarray("data_assisted_readout_gradient_v1"),
        "provenance_model": np.asarray(args.model),
        "provenance_dtype": np.asarray(args.dtype),
        "provenance_support": np.asarray(support, dtype=np.int64),
        "provenance_manifest_sha256": np.asarray(manifest_sha256),
        "provenance_source_sha256": np.asarray(source_sha256),
        "provenance_radial_sha256": np.asarray(radial_sha256),
        "provenance_checkpoint_sha256": np.asarray(checkpoint_sha256),
        "provenance_evaluation_indices_sha256": np.asarray(evaluation_indices_sha256),
    }
    block_records = []
    for central, ell in keys:
        record: dict[str, Any] = {
            "central_index": central,
            "atomic_number": int(model_numbers[central]),
            "ell": ell,
            "support_rank": support,
            "discovery_configurations_containing_species": species_configuration_counts[central],
            "spectra": {},
            "fold_projector_distances": {},
        }
        families = {
            "linear": accumulated["linear"][(central, ell)],
            "nonlinear": accumulated["nonlinear"][(central, ell)],
            "total_energy_gradient": accumulated["total"][(central, ell)],
            "activation_pca": accumulated["activation"][(central, ell)],
            "data_assisted_readout_gradient": balanced[(central, ell)],
        }
        for name, matrix in families.items():
            values, true_vectors = descending_eigensystem(matrix)
            _, basis, empirical_rank = deterministic_completed_eigensystem(
                matrix
            )
            arrays[f"gamma_{name}_z{central}_l{ell}"] = matrix
            arrays[f"gamma_eigenvalues_{name}_z{central}_l{ell}"] = values
            arrays[f"gamma_eigenvectors_{name}_z{central}_l{ell}"] = true_vectors
            arrays[f"basis_{name}_z{central}_l{ell}"] = basis
            record["spectra"][name] = {
                "trace": float(np.sum(values)),
                "largest_eigenvalue": float(values[0]),
                "minimum_eigenvalue": float(values[-1]),
                "empirical_rank": empirical_rank,
                "deterministic_prior_completion_count": support - empirical_rank,
            }
        fold_a_matrix = folded_balanced["A"][(central, ell)]
        fold_b_matrix = folded_balanced["B"][(central, ell)]
        _, fold_a_vectors, _ = deterministic_completed_eigensystem(fold_a_matrix)
        _, fold_b_vectors, _ = deterministic_completed_eigensystem(fold_b_matrix)
        fold_defined = (
            fold_species_counts["A"][central] > 0
            and fold_species_counts["B"][central] > 0
            and np.trace(fold_a_matrix) > np.finfo(np.float64).tiny
            and np.trace(fold_b_matrix) > np.finfo(np.float64).tiny
        )
        for rank in args.convergence_ranks:
            retained = min(rank, support)
            record["fold_projector_distances"][str(retained)] = (
                projector_distance(fold_a_vectors, fold_b_vectors, retained)
                if fold_defined
                else None
            )
        block_records.append(record)

    exactness_tolerance = 5e-9 if args.dtype == "float64" else 5e-4
    psd_relative_tolerance = 1e-10 if args.dtype == "float64" else 1e-5
    psd_absolute_tolerance = 1e-12 if args.dtype == "float64" else 1e-7
    maximum_trace_residual = max(
        residual
        for check in trace_checks
        for residual in check["residuals"].values()
    )
    maximum_gradient_residual = max(gradient_checks, default=0.0)
    maximum_absolute_psd_violation = max(psd_absolute_violations, default=0.0)
    maximum_relative_psd_violation = max(psd_relative_violations, default=0.0)
    psd_gate_passed = all(
        absolute <= psd_absolute_tolerance or relative <= psd_relative_tolerance
        for absolute, relative in zip(
            psd_absolute_violations, psd_relative_violations
        )
    )
    minimum_configurations_per_species_per_fold = 1
    fold_species_gate_passed = all(
        fold_species_counts[fold][central] >= minimum_configurations_per_species_per_fold
        for fold in ("A", "B")
        for central in range(num_elements)
    )
    gate_passed = (
        maximum_trace_residual <= exactness_tolerance
        and maximum_gradient_residual <= exactness_tolerance
        and psd_gate_passed
        and all(count > 0 for count in species_configuration_counts.values())
        and fold_species_gate_passed
        and radial_checkpoint_validation["maximum_curve_absolute_error"]
        <= exactness_tolerance
        and radial_checkpoint_validation["maximum_metric_relative_error"]
        <= exactness_tolerance
    )
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.npz, **arrays)
    output_npz_sha256 = sha256_file(args.npz)
    payload = {
        "schema_version": 1,
        "experiment": "T1_data_assisted_readout_gradient_environment",
        "method_name": "data_assisted_readout_gradient",
        "model": args.model,
        "dtype": args.dtype,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_source_sha256": source_sha256,
        "radial_artifact": str(args.radial_npz),
        "radial_artifact_sha256": radial_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "excluded_evaluation_indices_sha256": evaluation_indices_sha256,
        "configuration_count": configuration_count,
        "normalization": {
            "per_configuration_energy": "each branch energy is divided by sqrt(num_atoms)",
            "magnetic": "mean over every complete magnetic multiplet",
            "nodes": "sum within each configuration",
            "dataset": "equal mean over frozen discovery configurations",
            "consumer_balance": normalization,
        },
        "radial_checkpoint_validation": radial_checkpoint_validation,
        "exactness_checks": trace_checks,
        "maximum_checkpoint_trace_residual": maximum_trace_residual,
        "maximum_branch_gradient_sum_residual": maximum_gradient_residual,
        "maximum_absolute_psd_violation": maximum_absolute_psd_violation,
        "maximum_relative_psd_violation": maximum_relative_psd_violation,
        "psd_gate_passed": psd_gate_passed,
        "fold_configuration_counts": fold_counts,
        "fold_species_configuration_counts": fold_species_counts,
        "fold_species_gate_passed": fold_species_gate_passed,
        "minimum_configurations_per_species_per_fold": minimum_configurations_per_species_per_fold,
        "blocks": block_records,
        "exactness_tolerance": exactness_tolerance,
        "psd_absolute_tolerance": psd_absolute_tolerance,
        "psd_relative_tolerance": psd_relative_tolerance,
        "gate_passed": gate_passed,
        "npz": str(args.npz),
        "npz_sha256": output_npz_sha256,
        "claim_boundary": (
            "This is a discovery-data additive-gradient active-subspace pilot for "
            "the exact frozen linear and nonlinear readout energies at the first-density "
            "cut. It ranks unit additive perturbation directions. It is not the finite "
            "projector-loss Hessian, the exact checkpoint-only ODT coefficient "
            "environment, or a nonlinear fidelity certificate."
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
