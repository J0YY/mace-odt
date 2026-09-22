"""Evaluate activation-aware multi-consumer bases on the frozen test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mace_odt.audit import (
    evaluate,
    find_checkpoint_candidates,
    sha256_file,
    write_json,
)
from mace_odt.cli.heldout_branch_fidelity import (
    distribution_summary,
    load_selected_geometries,
    validate_manifest_source,
)
from mace_odt.cli.discovery_manifest import exact_structure_sha256
from mace_odt.cli.multi_consumer_environment import branch_graph
from mace_odt.cli.projected_branch_diagnostics import retained_rank_ladder
from mace_odt.functional_projector import native_functional_map
from mace_odt.first_branch import compile_first_branch
from mace_odt.projected_product import ProjectedProductBlock, projected_product_context
from mace_odt.radial_interface import angular_blocks


BlockKey = tuple[int, int]


def branch_diagnostics(model: Any, calc: Any, atoms: Any) -> dict[str, Any]:
    """Immediately detach the branch quantities needed for held-out scoring."""
    with torch.no_grad():
        state = branch_graph(model, calc, atoms)
        return {
            "linear_energy_eV": float(state["linear_nodes"].sum().cpu()),
            "nonlinear_energy_eV": float(state["nonlinear_nodes"].sum().cpu()),
            "head_preactivations": state["head_preactivations"].cpu().numpy().copy(),
        }


def load_bases(
    multi: Any,
    first_branch: Any,
    support: int,
    num_elements: int,
    random_seeds: list[int],
) -> dict[str, dict[BlockKey, np.ndarray]]:
    bases: dict[str, dict[BlockKey, np.ndarray]] = {
        "data_assisted_readout_gradient": {},
        "total_energy_gradient": {},
        "linear_readout_gradient": {},
        "nonlinear_readout_gradient": {},
        "activation_pca": {},
        "first_branch_global": {},
        "local_radial_svd": {},
    }
    for central in range(num_elements):
        for ell in range(4):
            key = (central, ell)
            bases["data_assisted_readout_gradient"][key] = np.asarray(
                multi[
                    f"basis_data_assisted_readout_gradient_z{central}_l{ell}"
                ],
                dtype=np.float64,
            )
            bases["total_energy_gradient"][key] = np.asarray(
                multi[f"basis_total_energy_gradient_z{central}_l{ell}"],
                dtype=np.float64,
            )
            bases["linear_readout_gradient"][key] = np.asarray(
                multi[f"basis_linear_z{central}_l{ell}"], dtype=np.float64
            )
            bases["nonlinear_readout_gradient"][key] = np.asarray(
                multi[f"basis_nonlinear_z{central}_l{ell}"], dtype=np.float64
            )
            bases["activation_pca"][key] = np.asarray(
                multi[f"basis_activation_pca_z{central}_l{ell}"],
                dtype=np.float64,
            )
            bases["first_branch_global"][key] = np.asarray(
                first_branch[f"gamma_eigenvectors_z{central}_l{ell}"],
                dtype=np.float64,
            )
            bases["local_radial_svd"][key] = np.eye(support)
    for seed in random_seeds:
        rng = np.random.default_rng(seed)
        name = f"canonical_random_seed_{seed}"
        bases[name] = {
            (central, ell): np.linalg.qr(rng.normal(size=(support, support)))[0]
            for central in range(num_elements)
            for ell in range(4)
        }
    expected = {(central, ell) for central in range(num_elements) for ell in range(4)}
    for name, blocks in bases.items():
        if set(blocks) != expected:
            raise ValueError(f"method {name} is missing functional blocks")
        if any(matrix.shape != (support, support) for matrix in blocks.values()):
            raise ValueError(f"method {name} contains an invalid basis shape")
        for matrix in blocks.values():
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"method {name} contains nonfinite basis entries")
            if np.linalg.norm(matrix.T @ matrix - np.eye(support)) > 1e-8:
                raise ValueError(f"method {name} contains a nonorthonormal basis")
    return bases


def paired_difference(
    baseline: list[float], candidate: list[float], seed: int, replicates: int = 10000
) -> dict[str, float]:
    """Bootstrap baseline minus candidate, so positive values favor candidate."""
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if baseline_array.shape != candidate_array.shape:
        raise ValueError("paired metric arrays must have equal shapes")
    differences = baseline_array - candidate_array
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(differences), size=(replicates, len(differences)))
    sampled = np.mean(differences[selections], axis=1)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(replicates, len(differences)))
    null_means = np.mean(signs * differences[None, :], axis=1)
    observed = abs(float(np.mean(differences)))
    permutation_p = (
        np.count_nonzero(np.abs(null_means) >= observed) + 1
    ) / (replicates + 1)
    return {
        "mean_baseline_minus_candidate": float(np.mean(differences)),
        "ci95_lower": float(np.quantile(sampled, 0.025)),
        "ci95_upper": float(np.quantile(sampled, 0.975)),
        "two_sided_sign_flip_p": float(permutation_p),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def holm_two(p_first: float, p_second: float) -> tuple[float, float]:
    values = [(0, p_first), (1, p_second)]
    values.sort(key=lambda item: item[1])
    adjusted = [0.0, 0.0]
    running = 0.0
    for position, (index, value) in enumerate(values):
        corrected = min(1.0, (2 - position) * value)
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted[0], adjusted[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--radial-npz", type=Path, required=True)
    parser.add_argument("--multi-consumer-npz", type=Path, required=True)
    parser.add_argument("--multi-consumer-json", type=Path, required=True)
    parser.add_argument("--first-branch-environment-npz", type=Path, required=True)
    parser.add_argument("--first-branch-environment-json", type=Path, required=True)
    parser.add_argument("--first-branch-npz", type=Path, required=True)
    parser.add_argument("--model", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--ranks", type=int, nargs="+", default=[48, 64, 80, 96])
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=[20260924, 20260925, 20260926, 20260927, 20260928],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    manifest = json.loads(args.manifest.read_text())
    discovery_manifest = json.loads(args.discovery_manifest.read_text())
    environment_metadata = json.loads(args.multi_consumer_json.read_text())
    first_environment_metadata = json.loads(
        args.first_branch_environment_json.read_text()
    )
    if discovery_manifest.get("experiment") != "T0_data_assisted_discovery_manifest":
        raise ValueError("unsupported discovery manifest")
    if environment_metadata.get("experiment") != "T1_data_assisted_readout_gradient_environment":
        raise ValueError("unsupported gradient environment metadata")
    if not environment_metadata.get("gate_passed"):
        raise ValueError("gradient environment numerical gate did not pass")
    source_sha256 = validate_manifest_source(args.xyz, manifest)
    if discovery_manifest.get("source", {}).get("xyz_sha256") != source_sha256:
        raise ValueError("discovery and evaluation manifests use different sources")
    if discovery_manifest.get("evaluation_manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("discovery manifest is not bound to this evaluation manifest")
    geometries = load_selected_geometries(args.xyz, manifest["selected"])
    discovery_geometries = load_selected_geometries(
        args.xyz, discovery_manifest["selected"]
    )
    evaluation_indices = [int(item["index"]) for item in manifest["selected"]]
    discovery_indices = [int(item["index"]) for item in discovery_manifest["selected"]]
    if len(set(evaluation_indices)) != len(evaluation_indices):
        raise ValueError("evaluation manifest contains duplicate indices")
    if len(set(discovery_indices)) != len(discovery_indices):
        raise ValueError("discovery manifest contains duplicate indices")
    if set(evaluation_indices) & set(discovery_indices):
        raise ValueError("discovery and evaluation indices overlap")
    discovery_hashes = set()
    for metadata, atoms in discovery_geometries:
        observed_hash = exact_structure_sha256(atoms)
        if metadata.get("exact_structure_sha256") != observed_hash:
            raise ValueError("discovery structure hash does not match the XYZ source")
        discovery_hashes.add(observed_hash)
    evaluation_hashes = {exact_structure_sha256(atoms) for _, atoms in geometries}
    if discovery_hashes & evaluation_hashes:
        raise ValueError("discovery and evaluation exact structure hashes overlap")
    radial = np.load(args.radial_npz)
    multi = np.load(args.multi_consumer_npz)
    first_branch = np.load(args.first_branch_environment_npz)
    branch_coefficients = np.load(args.first_branch_npz)
    for artifact, name in ((multi, "multi-consumer"), (first_branch, "first-branch")):
        if not np.array_equal(radial["atomic_numbers"], artifact["atomic_numbers"]):
            raise ValueError(f"{name} artifact uses a different species order")
    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    checkpoint_hashes = {
        sha256_file(path) for path in find_checkpoint_candidates(args.model)
    }
    if len(checkpoint_hashes) != 1:
        raise ValueError("could not identify one unique cached checkpoint digest")
    checkpoint_sha256 = checkpoint_hashes.pop()
    model_numbers = model.atomic_numbers.detach().cpu().numpy()
    if not np.array_equal(model_numbers, radial["atomic_numbers"]):
        raise ValueError("checkpoint and decomposition artifacts do not match")
    compiled_branch = compile_first_branch(model)
    branch_validation_errors = []
    if not np.array_equal(
        branch_coefficients["atomic_numbers"], model_numbers
    ):
        raise ValueError("first-branch artifact species order does not match checkpoint")
    for order in (1, 2, 3):
        observed = branch_coefficients[f"coefficients_order_{order}"]
        expected = compiled_branch.orders[order].detach().cpu().numpy()
        branch_validation_errors.append(float(np.max(np.abs(observed - expected))))
    branch_validation_errors.append(
        float(
            np.max(
                np.abs(
                    branch_coefficients["downstream_channel_weights"]
                    - compiled_branch.downstream_channel_weights.detach().cpu().numpy()
                )
            )
        )
    )
    branch_checkpoint_maximum_error = max(branch_validation_errors)
    branch_checkpoint_tolerance = 5e-10 if args.dtype == "float64" else 5e-5
    if branch_checkpoint_maximum_error > branch_checkpoint_tolerance:
        raise ValueError("first-branch artifact does not match the live checkpoint")
    support = int(radial["metric_factor_z0_l0"].shape[1])
    evaluation_indices_sha256 = hashlib.sha256(
        ",".join(map(str, sorted(item["index"] for item in manifest["selected"]))).encode()
    ).hexdigest()
    expected_provenance = {
        "provenance_schema": "data_assisted_readout_gradient_v1",
        "provenance_model": args.model,
        "provenance_dtype": args.dtype,
        "provenance_manifest_sha256": sha256_file(args.discovery_manifest),
        "provenance_source_sha256": source_sha256,
        "provenance_radial_sha256": sha256_file(args.radial_npz),
        "provenance_checkpoint_sha256": checkpoint_sha256,
        "provenance_evaluation_indices_sha256": evaluation_indices_sha256,
    }
    for key, expected in expected_provenance.items():
        if key not in multi or str(multi[key].item()) != expected:
            raise ValueError(f"multi-consumer artifact provenance mismatch for {key}")
    if int(multi["provenance_support"].item()) != support:
        raise ValueError("multi-consumer artifact support dimension mismatch")
    if environment_metadata.get("manifest_sha256") != sha256_file(args.discovery_manifest):
        raise ValueError("environment JSON and discovery manifest hashes differ")
    if environment_metadata.get("radial_artifact_sha256") != sha256_file(args.radial_npz):
        raise ValueError("environment JSON and radial artifact hashes differ")
    if environment_metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("environment JSON and checkpoint hashes differ")
    if environment_metadata.get("npz_sha256") != sha256_file(args.multi_consumer_npz):
        raise ValueError("environment JSON is not bound to the supplied NPZ")
    if first_environment_metadata.get("experiment") != "E5_global_mixed_order_environment":
        raise ValueError("unsupported first-branch environment metadata")
    if not first_environment_metadata.get("gate_passed"):
        raise ValueError("first-branch environment gate did not pass")
    if first_environment_metadata.get("npz_sha256") != sha256_file(
        args.first_branch_environment_npz
    ):
        raise ValueError("first-branch environment JSON and NPZ do not match")
    if first_environment_metadata.get("radial_artifact_sha256") != sha256_file(
        args.radial_npz
    ):
        raise ValueError("first-branch environment uses a different radial artifact")
    if first_environment_metadata.get("branch_artifact_sha256") != sha256_file(
        args.first_branch_npz
    ):
        raise ValueError("first-branch environment uses different branch coefficients")
    if str(first_branch["provenance_radial_sha256"].item()) != sha256_file(
        args.radial_npz
    ) or str(first_branch["provenance_branch_sha256"].item()) != sha256_file(
        args.first_branch_npz
    ):
        raise ValueError("first-branch NPZ provenance does not match its inputs")
    ranks = retained_rank_ladder(args.ranks, support)
    if support not in ranks:
        raise ValueError("the full supported rank is required as an exactness gate")
    bases = load_bases(multi, first_branch, support, len(model_numbers), args.random_seeds)
    block_layout = angular_blocks(model)
    angular_slices = {block.ell: slice(block.start, block.stop) for block in block_layout}
    atoms_list = [atoms for _, atoms in geometries]
    reference_full = [evaluate(calc, atoms) for atoms in atoms_list]
    reference_branch_values = [
        branch_diagnostics(model, calc, atoms) for atoms in atoms_list
    ]

    native_product = model.products[0]
    records = []
    for method, basis_by_block in bases.items():
        for retained in ranks:
            maps = {
                (central, ell): native_functional_map(
                    np.asarray(radial[f"metric_factor_z{central}_l{ell}"], dtype=np.float64),
                    basis_by_block[(central, ell)][:, :retained],
                )
                for central in range(len(model_numbers))
                for ell in angular_slices
            }
            replacement = ProjectedProductBlock(
                native_product,
                maps,
                num_elements=len(model_numbers),
                angular_slices=angular_slices,
            )
            started = time.perf_counter()
            with projected_product_context(model, replacement):
                candidate_full = [evaluate(calc, atoms) for atoms in atoms_list]
                candidate_branches = [
                    branch_diagnostics(model, calc, atoms) for atoms in atoms_list
                ]
            elapsed = time.perf_counter() - started
            per_configuration = []
            for (metadata, atoms), (energy_ref, force_ref), branch_ref, (energy_new, force_new), branch_new in zip(
                geometries,
                reference_full,
                reference_branch_values,
                candidate_full,
                candidate_branches,
            ):
                energy_error = energy_new - energy_ref
                force_error = force_new - force_ref
                linear_new = branch_new["linear_energy_eV"]
                nonlinear_new = branch_new["nonlinear_energy_eV"]
                preactivation_error = (
                    branch_new["head_preactivations"]
                    - branch_ref["head_preactivations"]
                )
                per_configuration.append(
                    {
                        "index": metadata["index"],
                        "config_type": metadata["config_type"],
                        "num_atoms": len(atoms),
                        "energy_absolute_error_eV": abs(energy_error),
                        "energy_absolute_error_per_atom_eV": abs(energy_error) / len(atoms),
                        "force_rmse_eV_per_A": float(np.sqrt(np.mean(force_error**2))),
                        "force_max_absolute_error_eV_per_A": float(np.max(np.abs(force_error))),
                        "linear_branch_absolute_error_per_atom_eV": abs(
                            linear_new - branch_ref["linear_energy_eV"]
                        )
                        / len(atoms),
                        "nonlinear_branch_absolute_error_per_atom_eV": abs(
                            nonlinear_new - branch_ref["nonlinear_energy_eV"]
                        )
                        / len(atoms),
                        "head_preactivation_rmse": float(
                            np.sqrt(np.mean(preactivation_error**2))
                        ),
                        "head_preactivation_unit_rmse": np.sqrt(
                            np.mean(preactivation_error**2, axis=0)
                        ).tolist(),
                    }
                )
            records.append(
                {
                    "method": method,
                    "multiplicity_rank_per_irrep": retained,
                    "canonical_coordinate_count_per_species": sum(
                        block.dimension * retained for block in block_layout
                    ),
                    "energy_absolute_error_per_atom_eV": distribution_summary(
                        [item["energy_absolute_error_per_atom_eV"] for item in per_configuration]
                    ),
                    "energy_absolute_error_eV": distribution_summary(
                        [item["energy_absolute_error_eV"] for item in per_configuration]
                    ),
                    "force_rmse_eV_per_A": distribution_summary(
                        [item["force_rmse_eV_per_A"] for item in per_configuration]
                    ),
                    "force_max_absolute_error_eV_per_A": distribution_summary(
                        [item["force_max_absolute_error_eV_per_A"] for item in per_configuration]
                    ),
                    "linear_branch_absolute_error_per_atom_eV": distribution_summary(
                        [item["linear_branch_absolute_error_per_atom_eV"] for item in per_configuration]
                    ),
                    "nonlinear_branch_absolute_error_per_atom_eV": distribution_summary(
                        [item["nonlinear_branch_absolute_error_per_atom_eV"] for item in per_configuration]
                    ),
                    "head_preactivation_rmse": distribution_summary(
                        [item["head_preactivation_rmse"] for item in per_configuration]
                    ),
                    "wall_seconds_total": elapsed,
                    "wall_seconds_per_configuration": elapsed / len(geometries),
                    "per_configuration": per_configuration,
                }
            )

    def unique_record(method: str, rank: int) -> dict[str, Any]:
        matches = [
            item
            for item in records
            if item["method"] == method and item["multiplicity_rank_per_irrep"] == rank
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one record for {method} at rank {rank}")
        return matches[0]

    if 64 not in ranks:
        raise ValueError("rank 64 is required for the preregistered primary comparison")
    candidate = unique_record("data_assisted_readout_gradient", 64)
    comparisons = []
    for offset, baseline_name in enumerate(("local_radial_svd", "first_branch_global")):
        baseline = unique_record(baseline_name, 64)
        comparison = {
            "candidate": "data_assisted_readout_gradient",
            "baseline": baseline_name,
            "rank": 64,
            "energy_absolute_error_per_atom_eV": paired_difference(
                [item["energy_absolute_error_per_atom_eV"] for item in baseline["per_configuration"]],
                [item["energy_absolute_error_per_atom_eV"] for item in candidate["per_configuration"]],
                seed=20265000 + offset,
            ),
            "force_rmse_eV_per_A": paired_difference(
                [item["force_rmse_eV_per_A"] for item in baseline["per_configuration"]],
                [item["force_rmse_eV_per_A"] for item in candidate["per_configuration"]],
                seed=20265100 + offset,
            ),
        }
        comparisons.append(comparison)
    adjusted = holm_two(
        comparisons[0]["force_rmse_eV_per_A"]["two_sided_sign_flip_p"],
        comparisons[1]["force_rmse_eV_per_A"]["two_sided_sign_flip_p"],
    )
    for comparison, corrected in zip(comparisons, adjusted):
        comparison["force_rmse_holm_adjusted_p"] = corrected

    linear_only = unique_record("linear_readout_gradient", 64)
    nonlinear_only = unique_record("nonlinear_readout_gradient", 64)
    consumer_ablation = {
        "candidate": "nonlinear_readout_gradient",
        "baseline": "linear_readout_gradient",
        "rank": 64,
        "force_rmse_eV_per_A": paired_difference(
            [item["force_rmse_eV_per_A"] for item in linear_only["per_configuration"]],
            [item["force_rmse_eV_per_A"] for item in nonlinear_only["per_configuration"]],
            seed=20265200,
        ),
        "nonlinear_branch_absolute_error_per_atom_eV": paired_difference(
            [
                item["nonlinear_branch_absolute_error_per_atom_eV"]
                for item in linear_only["per_configuration"]
            ],
            [
                item["nonlinear_branch_absolute_error_per_atom_eV"]
                for item in nonlinear_only["per_configuration"]
            ],
            seed=20265201,
        ),
        "head_preactivation_rmse": paired_difference(
            [item["head_preactivation_rmse"] for item in linear_only["per_configuration"]],
            [item["head_preactivation_rmse"] for item in nonlinear_only["per_configuration"]],
            seed=20265202,
        ),
    }

    full_rank_maximum = max(
        max(
            item["energy_absolute_error_per_atom_eV"]["maximum"],
            item["energy_absolute_error_eV"]["maximum"],
            item["force_max_absolute_error_eV_per_A"]["maximum"],
        )
        for item in records
        if item["multiplicity_rank_per_irrep"] == support
    )
    full_rank_tolerance = 5e-9 if args.dtype == "float64" else 5e-4
    full_rank_gate_passed = full_rank_maximum <= full_rank_tolerance
    local = unique_record("local_radial_svd", 64)
    global_first = unique_record("first_branch_global", 64)
    candidate_force = candidate["force_rmse_eV_per_A"]["mean"]
    candidate_energy = candidate["energy_absolute_error_per_atom_eV"]["mean"]
    local_energy_noninferiority_margin = 0.1 * local[
        "energy_absolute_error_per_atom_eV"
    ]["mean"]
    weak_feasibility = (
        full_rank_gate_passed
        and candidate_force < local["force_rmse_eV_per_A"]["mean"]
        and candidate_force < global_first["force_rmse_eV_per_A"]["mean"]
        and all(item["force_rmse_eV_per_A"]["ci95_lower"] > 0.0 for item in comparisons)
        and all(item["force_rmse_holm_adjusted_p"] < 0.05 for item in comparisons)
        and comparisons[0]["energy_absolute_error_per_atom_eV"]["ci95_lower"]
        >= -local_energy_noninferiority_margin
    )
    strong_feasibility = (
        full_rank_gate_passed
        and candidate_force <= 1e-3
        and candidate_energy <= 1e-4
    )
    no_go_active_subspace = all(
        unique_record("data_assisted_readout_gradient", rank)["force_rmse_eV_per_A"]["mean"]
        >= unique_record("local_radial_svd", rank)["force_rmse_eV_per_A"]["mean"]
        for rank in (64, 80)
        if rank in ranks
    ) and all(rank in ranks for rank in (64, 80))
    payload = {
        "schema_version": 1,
        "experiment": "T2_data_assisted_readout_gradient_fidelity",
        "manifest": str(args.manifest),
        "manifest_source_sha256": source_sha256,
        "configuration_count": len(geometries),
        "artifact_sha256": {
            "radial": sha256_file(args.radial_npz),
            "multi_consumer": sha256_file(args.multi_consumer_npz),
            "first_branch_environment": sha256_file(args.first_branch_environment_npz),
            "first_branch_coefficients": sha256_file(args.first_branch_npz),
        },
        "first_branch_checkpoint_maximum_error": branch_checkpoint_maximum_error,
        "first_branch_checkpoint_tolerance": branch_checkpoint_tolerance,
        "records": records,
        "primary_rank_64_comparisons": comparisons,
        "consumer_ablation_rank_64": consumer_ablation,
        "full_rank_maximum_energy_or_force_error": full_rank_maximum,
        "full_rank_tolerance": full_rank_tolerance,
        "full_rank_gate_passed": full_rank_gate_passed,
        "decision": {
            "weak_feasibility_passed": weak_feasibility,
            "strong_feasibility_passed": strong_feasibility,
            "active_subspace_no_go_triggered": no_go_active_subspace,
            "weak_rule": (
                "At rank 64 beat local radial SVD and first-branch global in mean "
                "force RMSE with positive paired intervals and Holm p below 0.05, "
                "with a paired energy interval inside a 10 percent noninferiority margin."
            ),
            "strong_rule": (
                "Using point estimates, rank-64 force RMSE <= 1e-3 eV/A and energy "
                "error <= 1e-4 eV/atom. The full-rank exactness gate is a prerequisite."
            ),
            "no_go_rule": (
                "Reject this active-subspace construction if it does not beat local "
                "radial SVD in mean force RMSE at both ranks 64 and 80."
            ),
        },
        "claim_boundary": (
            "All methods use the same encode-then-decode wrapper before product zero. "
            "This measures frozen-model fidelity and does not demonstrate a faster or "
            "smaller compiled evaluator."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"full_rank_gate_passed={full_rank_gate_passed}")
    if not full_rank_gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
