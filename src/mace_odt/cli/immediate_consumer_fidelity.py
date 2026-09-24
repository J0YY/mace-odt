"""Held-out fidelity of the exact immediate-consumer environment projectors.

Every intervention is inserted before product zero in the frozen checkpoint.
The same projected density therefore controls product zero, its first readout,
the second interaction message, and the second interaction skip.  No weights
are trained or refit here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mace_odt.audit import evaluate, find_checkpoint_candidates, sha256_file, write_json
from mace_odt.cli.discovery_manifest import exact_structure_sha256
from mace_odt.cli.heldout_branch_fidelity import (
    distribution_summary,
    load_selected_geometries,
    validate_manifest_source,
)
from mace_odt.cli.multi_consumer_environment import branch_graph
from mace_odt.cli.projected_branch_diagnostics import retained_rank_ladder
from mace_odt.functional_projector import NativeFunctionalMap, native_functional_map
from mace_odt.projected_product import ProjectedProductBlock, projected_product_context


BlockKey = tuple[int, int]
METHOD_EXACT_EQUAL = "exact_immediate_consumer_equal"
METHOD_EXACT_BALANCED = "exact_immediate_consumer_trace_balanced"
METHOD_EXACT = METHOD_EXACT_BALANCED
METHOD_LOCAL = "local_radial_svd"
METHOD_FIRST = "first_branch_global"
METHOD_DATA = "total_energy_gradient"
METHOD_REVERSE = "canonical_reverse"
NAMED_BASELINES = (METHOD_LOCAL, METHOD_FIRST, METHOD_DATA)
CONSUMERS = (
    "first_readout",
    "second_interaction_message",
    "second_interaction_skip",
)


def validate_frozen_protocol(
    protocol: Mapping[str, Any],
    environment: Mapping[str, Any],
    *,
    model: str,
    device: str,
    dtype: str,
    ranks: list[int],
    random_seeds: list[int],
) -> None:
    """Bind the evaluator to the declared T4 objective before loading structures."""
    if protocol.get("experiment") != "T4_exact_immediate_consumer_heldout_fidelity":
        raise ValueError("unsupported frozen fidelity protocol")
    if protocol.get("model") != f"MACE-OFF23-{model}":
        raise ValueError("frozen protocol and requested model differ")
    if protocol.get("dtype") != dtype or protocol.get("device") != device:
        raise ValueError("frozen protocol and requested numerical mode differ")
    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("frozen protocol omitted evaluation settings")
    if evaluation.get("ranks") != ranks:
        raise ValueError("frozen protocol and requested rank ladder differ")
    if evaluation.get("random_seeds") != random_seeds:
        raise ValueError("frozen protocol and requested random seeds differ")
    if evaluation.get("primary_exact_method") != METHOD_EXACT:
        raise ValueError("frozen protocol selected a different primary method")

    objective = protocol.get("environment_objective")
    if not isinstance(objective, Mapping):
        raise ValueError("frozen protocol omitted the exact environment objective")
    if environment.get("experiment") != objective.get("experiment"):
        raise ValueError("exact environment experiment differs from frozen protocol")
    if environment.get("exactness_tolerance") != objective.get(
        "exactness_tolerance"
    ):
        raise ValueError("exact environment tolerance differs from frozen protocol")
    if environment.get("required_consumers") != objective.get("required_consumers"):
        raise ValueError("exact environment consumers differ from frozen protocol")
    method = environment.get("method")
    expected_method = objective.get("method")
    if not isinstance(method, Mapping) or not isinstance(expected_method, Mapping):
        raise ValueError("exact environment method metadata is incomplete")
    for name, expected in expected_method.items():
        if method.get(name) != expected:
            raise ValueError(
                f"exact environment method field {name} differs from frozen protocol"
            )
    if environment.get("requested_model") != model:
        raise ValueError("exact environment and requested model differ")
    if environment.get("dtype") != dtype or environment.get("device") != device:
        raise ValueError("exact environment and requested numerical mode differ")
    if environment.get("rank_ladder") != ranks:
        raise ValueError("exact environment and requested rank ladder differ")


def _scalar(archive: Any, name: str) -> str:
    if name not in archive:
        raise ValueError(f"artifact omitted {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"artifact field {name} must be scalar")
    return str(value.item())


def _orthonormal_basis(value: object, support: int, name: str) -> np.ndarray:
    basis = np.asarray(value, dtype=np.float64)
    if basis.shape != (support, support) or not np.isfinite(basis).all():
        raise ValueError(f"{name} must be a finite full square basis")
    residual = np.linalg.norm(basis.T @ basis - np.eye(support))
    if residual > 1e-8:
        raise ValueError(f"{name} is not orthonormal")
    return basis


def load_basis_sets(
    exact: Any,
    first: Any,
    multi: Any,
    support: int,
    num_elements: int,
    random_seeds: list[int],
) -> dict[str, dict[BlockKey, np.ndarray]]:
    """Load the exact basis and frozen structural, learned, and sanity baselines."""
    result: dict[str, dict[BlockKey, np.ndarray]] = {
        METHOD_EXACT_EQUAL: {},
        METHOD_EXACT_BALANCED: {},
        METHOD_LOCAL: {},
        METHOD_FIRST: {},
        METHOD_DATA: {},
        METHOD_REVERSE: {},
    }
    identity = np.eye(support)
    reverse = identity[:, ::-1].copy()
    consumer_blocks = {
        consumer: {
            (central, ell): np.asarray(
                exact[f"gamma_{consumer}_z{central}_l{ell}"], dtype=np.float64
            )
            for central in range(num_elements)
            for ell in range(4)
        }
        for consumer in CONSUMERS
    }
    global_traces = {
        consumer: float(
            sum(
                (2 * ell + 1) * np.trace(blocks[(central, ell)])
                for central in range(num_elements)
                for ell in range(4)
            )
        )
        for consumer, blocks in consumer_blocks.items()
    }
    if any(value <= np.finfo(np.float64).tiny for value in global_traces.values()):
        raise ValueError("every immediate consumer must have positive global trace")

    def descending_basis(matrix: np.ndarray, name: str) -> np.ndarray:
        symmetric = 0.5 * (matrix + matrix.T)
        values, vectors = np.linalg.eigh(symmetric)
        scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
        if float(values[0]) < -1e-10 * scale:
            raise ValueError(f"{name} is materially indefinite")
        return vectors[:, np.argsort(values)[::-1]]

    for central in range(num_elements):
        for ell in range(4):
            key = (central, ell)
            suffix = f"z{central}_l{ell}"
            equal = sum(consumer_blocks[name][key] for name in CONSUMERS)
            balanced = sum(
                consumer_blocks[name][key] / global_traces[name]
                for name in CONSUMERS
            ) / len(CONSUMERS)
            result[METHOD_EXACT_EQUAL][key] = _orthonormal_basis(
                descending_basis(equal, f"equal {suffix}"),
                support,
                f"equal {suffix}",
            )
            result[METHOD_EXACT_BALANCED][key] = _orthonormal_basis(
                descending_basis(balanced, f"balanced {suffix}"),
                support,
                f"balanced {suffix}",
            )
            result[METHOD_LOCAL][key] = identity
            result[METHOD_FIRST][key] = _orthonormal_basis(
                first[f"gamma_eigenvectors_{suffix}"], support, f"first {suffix}"
            )
            result[METHOD_DATA][key] = _orthonormal_basis(
                multi[f"basis_total_energy_gradient_{suffix}"],
                support,
                f"total energy gradient {suffix}",
            )
            result[METHOD_REVERSE][key] = reverse
            for consumer in CONSUMERS:
                method = f"exact_consumer_{consumer}"
                result.setdefault(method, {})[key] = _orthonormal_basis(
                    descending_basis(consumer_blocks[consumer][key], f"{method} {suffix}"),
                    support,
                    f"{method} {suffix}",
                )
    for seed in random_seeds:
        name = f"canonical_random_seed_{seed}"
        generator = np.random.default_rng(seed)
        result[name] = {
            (central, ell): np.linalg.qr(
                generator.normal(size=(support, support))
            )[0]
            for central in range(num_elements)
            for ell in range(4)
        }
    expected = {
        (central, ell) for central in range(num_elements) for ell in range(4)
    }
    for method, blocks in result.items():
        if set(blocks) != expected:
            raise ValueError(f"method {method} has incomplete block coverage")
        for key, basis in blocks.items():
            _orthonormal_basis(basis, support, f"{method} {key}")
    return result


def build_maps(
    radial: Any,
    bases: Mapping[str, Mapping[BlockKey, np.ndarray]],
    ranks: tuple[int, ...],
    num_elements: int,
) -> dict[tuple[str, int], dict[BlockKey, NativeFunctionalMap]]:
    result = {}
    for method, blocks in bases.items():
        for rank in ranks:
            result[(method, rank)] = {
                (central, ell): native_functional_map(
                    np.asarray(
                        radial[f"metric_factor_z{central}_l{ell}"], dtype=np.float64
                    ),
                    blocks[(central, ell)][:, :rank],
                )
                for central in range(num_elements)
                for ell in range(4)
            }
    return result


def branch_values(model: Any, calc: Any, atoms: Any) -> dict[str, np.ndarray]:
    """Return detached tensors at every requested downstream checkpoint."""
    with torch.no_grad():
        state = branch_graph(model, calc, atoms)
    return {
        "first_readout": state["linear_nodes"].detach().cpu().numpy().copy(),
        "second_message": state["density1"].detach().cpu().numpy().copy(),
        "second_skip": state["residual1"].detach().cpu().numpy().copy(),
        "final_output": state["nonlinear_nodes"].detach().cpu().numpy().copy(),
    }


def tensor_fidelity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        candidate.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("candidate branch tensor is nonfinite or has changed shape")
    difference = candidate - reference
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(np.linalg.norm(difference) / denominator),
    }


def paired_difference(
    baseline: list[float], candidate: list[float], seed: int, replicates: int = 10000
) -> dict[str, float]:
    """Bootstrap baseline minus candidate, so positive values favor candidate."""
    baseline_array = np.asarray(baseline, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if baseline_array.shape != candidate_array.shape or baseline_array.ndim != 1:
        raise ValueError("paired metric arrays must be equal vectors")
    differences = baseline_array - candidate_array
    generator = np.random.default_rng(seed)
    selections = generator.integers(
        0, len(differences), size=(replicates, len(differences))
    )
    sampled = np.mean(differences[selections], axis=1)
    signs = generator.choice(
        np.asarray([-1.0, 1.0]), size=(replicates, len(differences))
    )
    null_means = np.mean(signs * differences[None, :], axis=1)
    observed = abs(float(np.mean(differences)))
    sign_flip_p = (
        np.count_nonzero(np.abs(null_means) >= observed) + 1
    ) / (replicates + 1)
    return {
        "mean_baseline_minus_exact": float(np.mean(differences)),
        "ci95_lower": float(np.quantile(sampled, 0.025)),
        "ci95_upper": float(np.quantile(sampled, 0.975)),
        "two_sided_sign_flip_p": float(sign_flip_p),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def holm_adjusted(values: Mapping[str, float]) -> dict[str, float]:
    """Return step-down Holm adjusted p-values for one named hypothesis family."""
    ordered = sorted(values.items(), key=lambda item: item[1])
    total = len(ordered)
    running = 0.0
    result: dict[str, float] = {}
    for position, (name, value) in enumerate(ordered):
        corrected = min(1.0, (total - position) * float(value))
        running = max(running, corrected)
        result[name] = running
    return result


def _record_lookup(
    records: list[dict[str, Any]], method: str, rank: int
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if record["method"] == method
        and record["multiplicity_rank_per_irrep"] == rank
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one record for {method} at rank {rank}")
    return matches[0]


def decision_summary(
    records: list[dict[str, Any]], ranks: tuple[int, ...], full_rank_gate: bool
) -> dict[str, Any]:
    """Apply the frozen rank-64 feasibility and rank-64/80 no-go rules."""
    if 64 not in ranks:
        return {
            "weak_feasibility_passed": False,
            "strong_feasibility_passed": False,
            "exact_beats_all_named_baselines_rank_64": False,
            "exact_no_go_triggered": False,
            "status": "rank_64_not_evaluated",
        }
    exact = _record_lookup(records, METHOD_EXACT, 64)
    comparisons = {}
    for offset, baseline in enumerate(NAMED_BASELINES):
        other = _record_lookup(records, baseline, 64)
        comparisons[baseline] = {
            "force_rmse_eV_per_A": paired_difference(
                [item["force_rmse_eV_per_A"] for item in other["per_configuration"]],
                [item["force_rmse_eV_per_A"] for item in exact["per_configuration"]],
                20261010 + offset,
            ),
            "energy_absolute_error_per_atom_eV": paired_difference(
                [
                    item["energy_absolute_error_per_atom_eV"]
                    for item in other["per_configuration"]
                ],
                [
                    item["energy_absolute_error_per_atom_eV"]
                    for item in exact["per_configuration"]
                ],
                20261020 + offset,
            ),
        }
        for metric_offset, metric in enumerate(
            (
                "first_readout_rmse",
                "second_message_rmse",
                "second_skip_rmse",
                "final_output_rmse",
            )
        ):
            comparisons[baseline][metric] = paired_difference(
                [item[metric] for item in other["per_configuration"]],
                [item[metric] for item in exact["per_configuration"]],
                20261030 + 10 * offset + metric_offset,
            )
    adjusted = holm_adjusted(
        {
            baseline: comparisons[baseline]["force_rmse_eV_per_A"][
                "two_sided_sign_flip_p"
            ]
            for baseline in NAMED_BASELINES
        }
    )
    for baseline in NAMED_BASELINES:
        comparisons[baseline]["force_rmse_holm_adjusted_p"] = adjusted[baseline]
    beats_all = all(
        exact["force_rmse_eV_per_A"]["mean"]
        < _record_lookup(records, baseline, 64)["force_rmse_eV_per_A"]["mean"]
        for baseline in NAMED_BASELINES
    )
    positive_force_intervals = all(
        comparisons[name]["force_rmse_eV_per_A"]["ci95_lower"] > 0.0
        for name in NAMED_BASELINES
    )
    significant_force_improvement = all(
        comparisons[name]["force_rmse_holm_adjusted_p"] < 0.05
        for name in NAMED_BASELINES
    )
    local_mean = _record_lookup(records, METHOD_LOCAL, 64)[
        "energy_absolute_error_per_atom_eV"
    ]["mean"]
    energy_noninferior = comparisons[METHOD_LOCAL][
        "energy_absolute_error_per_atom_eV"
    ]["ci95_lower"] >= -0.1 * local_mean
    weak = (
        full_rank_gate
        and beats_all
        and positive_force_intervals
        and significant_force_improvement
        and energy_noninferior
    )
    strong = (
        weak
        and exact["force_rmse_eV_per_A"]["mean"] <= 1e-3
        and exact["energy_absolute_error_per_atom_eV"]["mean"] <= 1e-4
    )
    no_go = False
    continue_not_compact = False
    rank_80_comparisons: dict[str, Any] = {}
    if 80 in ranks:
        exact_80 = _record_lookup(records, METHOD_EXACT, 80)
        for offset, baseline in enumerate((METHOD_LOCAL, METHOD_DATA)):
            other_80 = _record_lookup(records, baseline, 80)
            rank_80_comparisons[baseline] = {
                "force_rmse_eV_per_A": paired_difference(
                    [
                        item["force_rmse_eV_per_A"]
                        for item in other_80["per_configuration"]
                    ],
                    [
                        item["force_rmse_eV_per_A"]
                        for item in exact_80["per_configuration"]
                    ],
                    20261110 + offset,
                ),
                "energy_absolute_error_per_atom_eV": paired_difference(
                    [
                        item["energy_absolute_error_per_atom_eV"]
                        for item in other_80["per_configuration"]
                    ],
                    [
                        item["energy_absolute_error_per_atom_eV"]
                        for item in exact_80["per_configuration"]
                    ],
                    20261120 + offset,
                ),
            }
        rank_80_adjusted = holm_adjusted(
            {
                baseline: rank_80_comparisons[baseline]["force_rmse_eV_per_A"][
                    "two_sided_sign_flip_p"
                ]
                for baseline in (METHOD_LOCAL, METHOD_DATA)
            }
        )
        for baseline in (METHOD_LOCAL, METHOD_DATA):
            rank_80_comparisons[baseline]["force_rmse_holm_adjusted_p"] = (
                rank_80_adjusted[baseline]
            )
        rank_80_relative = all(
            exact_80["force_rmse_eV_per_A"]["mean"]
            < _record_lookup(records, baseline, 80)["force_rmse_eV_per_A"]["mean"]
            and rank_80_comparisons[baseline]["force_rmse_eV_per_A"]["ci95_lower"]
            > 0.0
            and rank_80_comparisons[baseline]["force_rmse_holm_adjusted_p"] < 0.05
            for baseline in (METHOD_LOCAL, METHOD_DATA)
        )
        rank_80_absolute = (
            exact_80["force_rmse_eV_per_A"]["mean"] <= 1e-3
            and exact_80["energy_absolute_error_per_atom_eV"]["mean"] <= 1e-4
        )
        continue_not_compact = (
            full_rank_gate and not strong and rank_80_relative and rank_80_absolute
        )
        no_go = full_rank_gate and all(
            _record_lookup(records, METHOD_EXACT, rank)["force_rmse_eV_per_A"][
                "mean"
            ]
            >= _record_lookup(records, baseline, rank)["force_rmse_eV_per_A"][
                "mean"
            ]
            for rank in (64, 80)
            for baseline in (METHOD_LOCAL, METHOD_DATA)
        )
    return {
        "rank_64_comparisons": comparisons,
        "rank_80_comparisons": rank_80_comparisons,
        "weak_feasibility_passed": weak,
        "strong_feasibility_passed": strong,
        "continue_not_compact_passed": continue_not_compact,
        "exact_beats_all_named_baselines_rank_64": beats_all,
        "exact_no_go_triggered": no_go,
        "weak_rule": (
            "At rank 64 the exact environment has lower mean force RMSE than local "
            "radial SVD, first-branch global, and total-energy-gradient, with positive "
            "paired 95 percent intervals and Holm-adjusted paired sign-flip p below "
            "0.05, while its paired energy interval is within 10 percent of the local "
            "mean error. Full rank exactness is required."
        ),
        "strong_rule": (
            "The weak paired rule passes, rank-64 mean force RMSE is at most 1e-3 "
            "eV/A, and mean absolute energy error is at most 1e-4 eV/atom."
        ),
        "no_go_rule": (
            "Trigger only if the exact environment fails to beat both local radial "
            "SVD and total-energy-gradient mean force RMSE at ranks 64 and 80."
        ),
    }


def _validate_disjoint_manifests(
    xyz: Path, evaluation: dict[str, Any], discovery: dict[str, Any]
) -> list[tuple[dict[str, Any], Any]]:
    source = validate_manifest_source(xyz, evaluation)
    if discovery.get("experiment") != "T0_data_assisted_discovery_manifest":
        raise ValueError("unsupported discovery manifest")
    if discovery.get("source", {}).get("xyz_sha256") != source:
        raise ValueError("discovery and evaluation sources differ")
    evaluation_geometries = load_selected_geometries(xyz, evaluation["selected"])
    discovery_geometries = load_selected_geometries(xyz, discovery["selected"])
    for metadata, atoms in (*evaluation_geometries, *discovery_geometries):
        if int(metadata.get("num_atoms", -1)) != len(atoms):
            raise ValueError("manifest atom count differs from the XYZ structure")
        if metadata.get("formula") != atoms.get_chemical_formula():
            raise ValueError("manifest formula differs from the XYZ structure")
    for metadata, atoms in discovery_geometries:
        if metadata.get("exact_structure_sha256") != exact_structure_sha256(atoms):
            raise ValueError("discovery structure hash differs from the XYZ structure")
    evaluation_indices = {int(item[0]["index"]) for item in evaluation_geometries}
    discovery_indices = {int(item[0]["index"]) for item in discovery_geometries}
    if len(evaluation_indices) != len(evaluation_geometries) or len(
        discovery_indices
    ) != len(discovery_geometries):
        raise ValueError("a frozen manifest contains duplicate indices")
    if evaluation_indices & discovery_indices:
        raise ValueError("discovery and evaluation indices overlap")
    evaluation_hashes = {
        exact_structure_sha256(atoms) for _, atoms in evaluation_geometries
    }
    discovery_hashes = {
        exact_structure_sha256(atoms) for _, atoms in discovery_geometries
    }
    if evaluation_hashes & discovery_hashes:
        raise ValueError("discovery and evaluation structures overlap exactly")
    return evaluation_geometries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--radial-npz", type=Path, required=True)
    parser.add_argument("--environment-npz", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path, required=True)
    parser.add_argument("--first-branch-environment-npz", type=Path, required=True)
    parser.add_argument("--first-branch-environment-json", type=Path, required=True)
    parser.add_argument("--multi-consumer-npz", type=Path, required=True)
    parser.add_argument("--multi-consumer-json", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--model", choices=("small",), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float64",), default="float64")
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 48, 64, 80, 96])
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=[20260924, 20260925, 20260926, 20260927, 20260928],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from mace.calculators import mace_off

    evaluation_manifest = json.loads(args.manifest.read_text())
    discovery_manifest = json.loads(args.discovery_manifest.read_text())
    protocol = json.loads(args.protocol_config.read_text())
    if discovery_manifest.get("evaluation_manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("discovery manifest is not bound to the evaluation manifest")
    environment_json = json.loads(args.environment_json.read_text())
    first_json = json.loads(args.first_branch_environment_json.read_text())
    multi_json = json.loads(args.multi_consumer_json.read_text())
    validate_frozen_protocol(
        protocol,
        environment_json,
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        ranks=args.ranks,
        random_seeds=args.random_seeds,
    )
    if environment_json.get("experiment") != "T3_exact_immediate_consumer_environment":
        raise ValueError("unsupported exact environment")
    if not environment_json.get("gate_passed"):
        raise ValueError("exact environment numerical gate did not pass")
    if first_json.get("experiment") != "E5_global_mixed_order_environment" or not first_json.get(
        "gate_passed"
    ):
        raise ValueError("first-branch environment is unsupported or failed")
    if multi_json.get("experiment") != "T1_data_assisted_readout_gradient_environment" or not multi_json.get(
        "gate_passed"
    ):
        raise ValueError("data-assisted environment is unsupported or failed")

    radial_sha = sha256_file(args.radial_npz)
    environment_sha = sha256_file(args.environment_npz)
    first_sha = sha256_file(args.first_branch_environment_npz)
    multi_sha = sha256_file(args.multi_consumer_npz)
    if environment_json.get("npz_sha256") != environment_sha:
        raise ValueError("exact environment JSON and NPZ differ")
    if first_json.get("npz_sha256") != first_sha:
        raise ValueError("first-branch environment JSON and NPZ differ")
    if multi_json.get("npz_sha256") != multi_sha:
        raise ValueError("data-assisted environment JSON and NPZ differ")
    if first_json.get("radial_artifact_sha256") != radial_sha:
        raise ValueError("first-branch environment uses a different radial artifact")
    if (
        multi_json.get("radial_artifact_sha256") != radial_sha
        or multi_json.get("manifest_sha256") != sha256_file(args.discovery_manifest)
        or multi_json.get("manifest_source_sha256") != sha256_file(args.xyz)
    ):
        raise ValueError("data-assisted environment provenance differs from inputs")
    provenance = environment_json.get("provenance", {})
    if provenance.get("radial_npz_sha256") != radial_sha or provenance.get(
        "manifest_sha256"
    ) != sha256_file(args.discovery_manifest):
        raise ValueError("exact environment provenance differs from supplied inputs")

    radial = np.load(args.radial_npz, allow_pickle=False)
    environment = np.load(args.environment_npz, allow_pickle=False)
    first = np.load(args.first_branch_environment_npz, allow_pickle=False)
    multi = np.load(args.multi_consumer_npz, allow_pickle=False)
    species = np.asarray(radial["atomic_numbers"])
    for name, artifact in (
        ("exact", environment),
        ("first", first),
        ("data assisted", multi),
    ):
        if not np.array_equal(species, artifact["atomic_numbers"]):
            raise ValueError(f"{name} artifact species order differs")
    if _scalar(environment, "provenance_radial_sha256") != radial_sha:
        raise ValueError("exact environment NPZ uses a different radial artifact")
    if _scalar(environment, "provenance_manifest_sha256") != sha256_file(
        args.discovery_manifest
    ):
        raise ValueError("exact environment NPZ uses a different discovery manifest")
    if _scalar(first, "provenance_radial_sha256") != radial_sha:
        raise ValueError("first-branch NPZ uses a different radial artifact")
    if (
        _scalar(multi, "provenance_radial_sha256") != radial_sha
        or _scalar(multi, "provenance_manifest_sha256")
        != sha256_file(args.discovery_manifest)
        or _scalar(multi, "provenance_source_sha256") != sha256_file(args.xyz)
    ):
        raise ValueError("data-assisted NPZ provenance differs from inputs")

    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    checkpoint_hashes = {
        sha256_file(path) for path in find_checkpoint_candidates(args.model)
    }
    if len(checkpoint_hashes) != 1:
        raise ValueError("could not identify one unique cached checkpoint")
    checkpoint_sha = checkpoint_hashes.pop()
    if provenance.get("checkpoint_sha256") != checkpoint_sha or _scalar(
        environment, "provenance_checkpoint_sha256"
    ) != checkpoint_sha:
        raise ValueError("exact environment and live checkpoint differ")
    if multi_json.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("data-assisted environment and live checkpoint differ")
    if _scalar(multi, "provenance_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("data-assisted NPZ and live checkpoint differ")
    if not np.array_equal(model.atomic_numbers.detach().cpu().numpy(), species):
        raise ValueError("checkpoint species order differs from artifacts")

    geometries = _validate_disjoint_manifests(
        args.xyz, evaluation_manifest, discovery_manifest
    )

    support = int(radial["metric_factor_z0_l0"].shape[1])
    ranks = retained_rank_ladder(args.ranks, support)
    if support not in ranks:
        raise ValueError("the full supported rank is required as an exactness gate")
    stored_ranks = tuple(int(value) for value in environment["rank_ladder"])
    if any(rank not in stored_ranks for rank in ranks):
        raise ValueError("requested rank was not frozen in the exact environment")
    bases = load_basis_sets(
        environment, first, multi, support, len(species), args.random_seeds
    )
    maps = build_maps(radial, bases, ranks, len(species))
    angular_slices = {
        0: slice(0, 1),
        1: slice(1, 4),
        2: slice(4, 9),
        3: slice(9, 16),
    }

    atoms_list = [atoms for _, atoms in geometries]
    reference_full = [evaluate(calc, atoms) for atoms in atoms_list]
    reference_branches = [branch_values(model, calc, atoms) for atoms in atoms_list]
    measurements: dict[tuple[str, int], list[dict[str, Any]]] = {
        key: [] for key in maps
    }
    native_product = model.products[0]
    for key, block_maps in maps.items():
        replacement = ProjectedProductBlock(
            native_product,
            block_maps,
            num_elements=len(species),
            angular_slices=angular_slices,
        )
        with projected_product_context(model, replacement):
            candidate_full = [evaluate(calc, atoms) for atoms in atoms_list]
            candidate_branches = [branch_values(model, calc, atoms) for atoms in atoms_list]
        for (metadata, atoms), (energy_ref, force_ref), branch_ref, (energy_new, force_new), branch_new in zip(
            geometries,
            reference_full,
            reference_branches,
            candidate_full,
            candidate_branches,
        ):
            energy_error = float(energy_new - energy_ref)
            force_difference = np.asarray(force_new) - np.asarray(force_ref)
            if not np.isfinite(energy_error) or not np.isfinite(force_difference).all():
                raise FloatingPointError("held-out energy or force became nonfinite")
            first_metrics = tensor_fidelity(
                branch_ref["first_readout"], branch_new["first_readout"]
            )
            message_metrics = tensor_fidelity(
                branch_ref["second_message"], branch_new["second_message"]
            )
            skip_metrics = tensor_fidelity(
                branch_ref["second_skip"], branch_new["second_skip"]
            )
            final_metrics = tensor_fidelity(
                branch_ref["final_output"], branch_new["final_output"]
            )
            measurements[key].append(
                {
                    "index": metadata["index"],
                    "config_type": metadata["config_type"],
                    "num_atoms": len(atoms),
                    "energy_absolute_error_eV": abs(energy_error),
                    "energy_absolute_error_per_atom_eV": abs(energy_error) / len(atoms),
                    "force_rmse_eV_per_A": float(np.sqrt(np.mean(force_difference**2))),
                    "force_max_absolute_error_eV_per_A": float(
                        np.max(np.abs(force_difference))
                    ),
                    "first_readout_rmse": first_metrics["rmse"],
                    "first_readout_maximum_absolute_error": first_metrics[
                        "maximum_absolute_error"
                    ],
                    "first_readout_relative_l2_error": first_metrics[
                        "relative_l2_error"
                    ],
                    "second_message_rmse": message_metrics["rmse"],
                    "second_message_maximum_absolute_error": message_metrics[
                        "maximum_absolute_error"
                    ],
                    "second_message_relative_l2_error": message_metrics[
                        "relative_l2_error"
                    ],
                    "second_skip_rmse": skip_metrics["rmse"],
                    "second_skip_maximum_absolute_error": skip_metrics[
                        "maximum_absolute_error"
                    ],
                    "second_skip_relative_l2_error": skip_metrics[
                        "relative_l2_error"
                    ],
                    "final_output_rmse": final_metrics["rmse"],
                    "final_output_maximum_absolute_error": final_metrics[
                        "maximum_absolute_error"
                    ],
                    "final_output_relative_l2_error": final_metrics[
                        "relative_l2_error"
                    ],
                }
            )

    records = []
    summary_metrics = (
        "energy_absolute_error_eV",
        "energy_absolute_error_per_atom_eV",
        "force_rmse_eV_per_A",
        "force_max_absolute_error_eV_per_A",
        "first_readout_rmse",
        "first_readout_maximum_absolute_error",
        "first_readout_relative_l2_error",
        "second_message_rmse",
        "second_message_maximum_absolute_error",
        "second_message_relative_l2_error",
        "second_skip_rmse",
        "second_skip_maximum_absolute_error",
        "second_skip_relative_l2_error",
        "final_output_rmse",
        "final_output_maximum_absolute_error",
        "final_output_relative_l2_error",
    )
    for (method, rank), configurations in measurements.items():
        record: dict[str, Any] = {
            "method": method,
            "multiplicity_rank_per_irrep": rank,
            "canonical_coordinate_count_per_species": 16 * rank,
            "per_configuration": configurations,
        }
        for metric in summary_metrics:
            record[metric] = distribution_summary(
                [item[metric] for item in configurations]
            )
        records.append(record)
    records.sort(key=lambda item: (item["method"], item["multiplicity_rank_per_irrep"]))

    full_rank_records = [
        record for record in records if record["multiplicity_rank_per_irrep"] == support
    ]
    full_rank_energy_force_maximum = max(
        max(
            record["energy_absolute_error_eV"]["maximum"],
            record["force_max_absolute_error_eV_per_A"]["maximum"],
        )
        for record in full_rank_records
    )
    full_rank_branch_maximum = max(
        max(
            record["first_readout_maximum_absolute_error"]["maximum"],
            record["second_message_maximum_absolute_error"]["maximum"],
            record["second_skip_maximum_absolute_error"]["maximum"],
            record["final_output_maximum_absolute_error"]["maximum"],
        )
        for record in full_rank_records
    )
    full_rank_tolerance = 5e-9
    full_rank_gate = full_rank_energy_force_maximum <= full_rank_tolerance
    decision = decision_summary(records, ranks, full_rank_gate)
    payload = {
        "schema_version": 1,
        "experiment": "T4_exact_immediate_consumer_fidelity",
        "requested_model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "configuration_count": len(geometries),
        "rank_ladder": list(ranks),
        "artifact_sha256": {
            "evaluation_manifest": sha256_file(args.manifest),
            "discovery_manifest": sha256_file(args.discovery_manifest),
            "radial": radial_sha,
            "exact_environment": environment_sha,
            "first_branch_environment": first_sha,
            "data_assisted_environment": multi_sha,
            "checkpoint": checkpoint_sha,
        },
        "data_usage": {
            "uses_reference_labels": False,
            "offset_fitting": False,
            "rank_selection_on_evaluation": False,
            "retraining": False,
        },
        "methods": sorted(bases),
        "records": records,
        "full_rank_maximum_energy_or_force_error": full_rank_energy_force_maximum,
        "full_rank_maximum_branch_absolute_error": full_rank_branch_maximum,
        "full_rank_tolerance": full_rank_tolerance,
        "full_rank_gate_passed": full_rank_gate,
        "decision": decision,
        "claim_boundary": (
            "This is a frozen-checkpoint held-out intervention test. It compares "
            "finite shared projectors using energy, force, immediate-consumer, and "
            "final-readout fidelity. It does not retrain the model and does not "
            "establish speedup, a smaller compiled evaluator, transfer across "
            "checkpoints, or physical interpretation."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"full_rank_gate_passed={full_rank_gate}")
    if not full_rank_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
