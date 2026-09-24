"""Build the exact immediate-consumer environment under a declared measure.

This file is deliberately a protocol and artifact wrapper. The tensor algebra
lives in :mod:`mace_odt.immediate_consumer`. The expected public API is::

    build_immediate_consumer_environment(
        *,
        model,
        metric_factors: Mapping[tuple[int, int], numpy.ndarray],
        message_measure: Mapping[str, object],
        order_weights: Mapping[int, float],
        consumer_weights: Mapping[str, float],
        checkpoint_sha256: str,
        exactness_tolerance: float,
    ) -> Mapping[str, object]

The returned mapping must contain ``atomic_numbers``, ``aggregate_blocks``,
``consumer_blocks``, ``checks``, and ``metadata``. Block mappings use
``(central_species_index, ell)`` keys and finite square float arrays. The three
required consumers are ``first_readout``, ``second_interaction_message``, and
``second_interaction_skip``. Consumer blocks are raw unweighted pullback Grams.
The aggregate must equal their declared consumer-weighted sum. Each check is a mapping with finite nonnegative
``value`` and ``tolerance`` entries. The required independent checks are
``dense_action_relative_residual`` and
``tied_factor_three_relative_residual``. Metadata must echo the message measure
and weights, declare the output metric, set ``coefficient_only=True``, and set
``uses_reference_labels=False``.

The CLI validates and serializes that contract. It does not read energies or
forces from the XYZ file and it never fits a rank, metric, or offset.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from mace_odt.audit import (
    environment_manifest,
    find_checkpoint_candidates,
    json_value,
    sha256_file,
)


REQUIRED_CONSUMERS = (
    "first_readout",
    "second_interaction_message",
    "second_interaction_skip",
)
REQUIRED_API_CHECKS = (
    "analytic_coefficient_reconstruction_relative_residual",
    "complete_irrep_metric_relative_residual",
    "dense_action_relative_residual",
    "readout_scale_relative_residual",
    "tied_factor_three_relative_residual",
)


def _three_nonnegative(values: list[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3 or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} requires three finite nonnegative values")
    if sum(values) <= 0.0:
        raise ValueError(f"{name} cannot be identically zero")
    return float(values[0]), float(values[1]), float(values[2])


def _sha256_argument(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _positive_finite(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return result


def _staged_json(path: Path, payload: dict[str, Any]) -> Path:
    """Stage a JSON file beside its destination without publishing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=json_value,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _staged_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays):
                raise ArithmeticError("staged NPZ has missing or unexpected arrays")
            for name, expected in arrays.items():
                observed = loaded[name]
                if observed.dtype != expected.dtype or not np.array_equal(
                    observed, expected
                ):
                    raise ArithmeticError(f"staged NPZ changed array {name}")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _mapping(value: object, name: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"immediate-consumer API returned non-mapping {name}")
    return value


def _block_key(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
    ):
        raise TypeError("environment block keys must be integer (species, ell) tuples")
    return value


def _validated_block(value: object, name: str) -> np.ndarray:
    block = np.asarray(value, dtype=np.float64)
    if (
        block.ndim != 2
        or block.shape[0] != block.shape[1]
        or block.shape[0] == 0
        or not np.isfinite(block).all()
    ):
        raise ValueError(f"{name} must be a finite nonempty square matrix")
    return block


def _validate_api_checks(
    raw: object, maximum_tolerance: float
) -> tuple[list[dict[str, Any]], bool]:
    checks = _mapping(raw, "checks")
    if not set(REQUIRED_API_CHECKS) <= set(checks):
        missing = sorted(set(REQUIRED_API_CHECKS) - set(checks))
        raise ValueError(f"immediate-consumer API omitted required checks: {missing}")
    records = []
    passed = True
    for name in sorted(checks):
        if not isinstance(name, str) or not name:
            raise TypeError("check names must be nonempty strings")
        record = _mapping(checks[name], f"check {name}")
        if set(record) != {"value", "tolerance"}:
            raise ValueError(f"check {name} must contain only value and tolerance")
        value = float(record["value"])
        tolerance = float(record["tolerance"])
        if (
            not np.isfinite(value)
            or value < 0.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or tolerance > maximum_tolerance
        ):
            raise ValueError(f"check {name} has an invalid or weakened tolerance")
        item_passed = value <= tolerance
        records.append(
            {
                "name": name,
                "value": value,
                "tolerance": tolerance,
                "passed": item_passed,
            }
        )
        passed = passed and item_passed
    return records, passed


def _rank_summary(values: np.ndarray, ranks: tuple[int, ...]) -> dict[str, Any]:
    trace = float(np.sum(values))
    result: dict[str, Any] = {
        "trace": trace,
        "largest_eigenvalue": float(values[0]),
        "smallest_eigenvalue": float(values[-1]),
        "ranks": {},
    }
    cumulative = np.cumsum(values)
    for rank in ranks:
        retained = float(cumulative[rank - 1])
        tail = max(trace - retained, 0.0)
        result["ranks"][str(rank)] = {
            "retained_trace": retained,
            "omitted_trace": tail,
            "retained_trace_fraction": retained / trace,
            "omitted_trace_fraction": tail / trace,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile the exact immediate-consumer coefficient environment."
    )
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--radial-npz", type=Path, required=True)
    parser.add_argument("--model", choices=("small",), default="small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[16, 32, 48, 64, 80, 96]
    )
    parser.add_argument(
        "--order-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0]
    )
    parser.add_argument(
        "--consumer-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0]
    )
    parser.add_argument(
        "--exactness-tolerance", type=_positive_finite, default=5e-9
    )
    parser.add_argument(
        "--expected-checkpoint-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-manifest-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-radial-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    args = parser.parse_args()

    ranks = tuple(args.ranks)
    if (
        not ranks
        or any(type(rank) is not int or rank <= 0 for rank in ranks)
        or tuple(sorted(set(ranks))) != ranks
    ):
        raise ValueError("ranks must be unique positive integers in increasing order")
    order_weight_values = _three_nonnegative(args.order_weights, "order weights")
    consumer_weight_values = _three_nonnegative(
        args.consumer_weights, "consumer weights"
    )
    order_weights = {
        order: order_weight_values[order - 1] for order in range(1, 4)
    }
    consumer_weights = dict(zip(REQUIRED_CONSUMERS, consumer_weight_values))
    if args.output.resolve() == args.npz.resolve():
        raise ValueError("JSON and NPZ outputs must be different paths")
    for path in (args.output, args.npz):
        if path.exists():
            raise FileExistsError(f"refusing to replace existing output: {path}")

    from mace_odt.cli.heldout_branch_fidelity import validate_manifest_source
    from mace_odt.cli.multi_consumer_environment import validate_radial_artifact

    manifest_sha256 = sha256_file(args.manifest)
    radial_sha256 = sha256_file(args.radial_npz)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError("discovery manifest differs from the frozen digest")
    if radial_sha256 != args.expected_radial_sha256:
        raise ValueError("radial artifact differs from the frozen digest")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("experiment") != "T0_data_assisted_discovery_manifest":
        raise ValueError("unsupported discovery manifest")
    if manifest.get("selection", {}).get("uses_reference_labels") is not False:
        raise ValueError("discovery manifest must explicitly exclude reference labels")
    source_sha256 = validate_manifest_source(args.xyz, manifest)

    from mace.calculators import mace_off
    import mace_odt.immediate_consumer as immediate_consumer

    builder = getattr(
        immediate_consumer, "build_immediate_consumer_environment", None
    )
    if not callable(builder):
        raise ImportError(
            "mace_odt.immediate_consumer must expose "
            "build_immediate_consumer_environment as documented in this module"
        )

    calc = mace_off(model=args.model, device=args.device, default_dtype=args.dtype)
    model = calc.models[0]
    candidates = find_checkpoint_candidates(args.model)
    candidate_hashes = {sha256_file(path) for path in candidates}
    if candidate_hashes != {args.expected_checkpoint_sha256}:
        raise ValueError("cached checkpoint set does not match the frozen digest")
    checkpoint_sha256 = args.expected_checkpoint_sha256

    with np.load(args.radial_npz, allow_pickle=False) as archive:
        radial = {name: archive[name].copy() for name in archive.files}
    model_numbers = model.atomic_numbers.detach().cpu().numpy()
    if not np.array_equal(radial.get("atomic_numbers"), model_numbers):
        raise ValueError("checkpoint and radial artifact species orders differ")
    metric_factors = {
        (central, ell): np.asarray(
            radial[f"metric_factor_z{central}_l{ell}"], dtype=np.float64
        )
        for central in range(len(model_numbers))
        for ell in range(4)
    }
    radii = np.asarray(radial["radii_angstrom"], dtype=np.float64)
    radial_weights = np.asarray(
        radial["normalized_uniform_radial_weights"], dtype=np.float64
    )
    if (
        radii.ndim != 1
        or radial_weights.shape != radii.shape
        or not np.isfinite(radii).all()
        or not np.isfinite(radial_weights).all()
        or np.any(radial_weights <= 0.0)
    ):
        raise ValueError("radial artifact does not define a finite positive quadrature")
    message_measure: dict[str, object] = {
        "schema": "uniform-species-radial-quadrature-o3-haar-v1",
        "radii_angstrom": radii,
        "radial_weights": radial_weights,
        "neighbor_species_weights": np.full(
            len(model_numbers), 1.0 / len(model_numbers), dtype=np.float64
        ),
        "conditional_source_species": "environment_block_central_species_index",
        "target_species_rule": "uniform_over_checkpoint_species",
        "checkpoint_edge_path": (
            "radial_embedding_to_interaction1_conv_tp_weights_times_cutoff"
        ),
        "angular_rule": (
            "analytic_o3_irrep_orthogonality_on_complete_multiplets_through_degree_6"
        ),
        "output_irrep_metric": "identity_on_every_complete_output_irrep",
        "edge_rule": "one_free_directed_edge_coefficient_lift",
        "topology_convention": "simple_directed_graph_without_parallel_edges",
        "integration_scope": "exact_for_frozen_discrete_radial_quadrature",
        "uses_discovery_geometries": False,
    }
    radial_checkpoint_validation = validate_radial_artifact(model, radial)
    if any(
        not np.isfinite(value) or value > args.exactness_tolerance
        for value in radial_checkpoint_validation.values()
    ):
        raise ArithmeticError("radial artifact failed live checkpoint validation")

    bundle = _mapping(
        builder(
            model=model,
            metric_factors=metric_factors,
            message_measure=message_measure,
            order_weights=order_weights,
            consumer_weights=consumer_weights,
            checkpoint_sha256=checkpoint_sha256,
            exactness_tolerance=args.exactness_tolerance,
        ),
        "environment bundle",
    )
    required_bundle_keys = {
        "atomic_numbers",
        "aggregate_blocks",
        "consumer_blocks",
        "checks",
        "metadata",
    }
    if set(bundle) != required_bundle_keys:
        raise ValueError(
            "immediate-consumer bundle has missing or unexpected keys: "
            f"{sorted(set(bundle) ^ required_bundle_keys)}"
        )
    if not np.array_equal(np.asarray(bundle["atomic_numbers"]), model_numbers):
        raise ValueError("immediate-consumer species order differs from checkpoint")

    metadata = _mapping(bundle["metadata"], "metadata")
    expected_output_metric = {
        "first_readout": {
            "dimension": 1,
            "metric": "euclidean_identity",
            "normalization": "checkpoint_scaled_node_output",
        },
        "second_interaction_message": {
            "dimension": 1536,
            "layout": [96, 16],
            "metric": "euclidean_identity_on_complete_irreps",
            "normalization": "one_free_directed_edge",
        },
        "second_interaction_skip": {
            "dimension": 96,
            "metric": "euclidean_identity",
            "normalization": "per_node_output",
        },
    }
    if (
        metadata.get("output_metric") != expected_output_metric
        or metadata.get("message_measure_schema") != message_measure["schema"]
        or metadata.get("order_weights") != order_weights
        or metadata.get("order_weight_scope") != "global_by_order"
        or metadata.get("consumer_weights") != consumer_weights
        or metadata.get("consumer_weight_scope") != "global_by_consumer"
        or metadata.get("consumer_blocks_weighted") is not False
        or metadata.get("order_weights_applied_to_consumer_blocks") is not True
        or metadata.get("analytic_product_coefficients") is not True
        or metadata.get("readout_checkpoint_scale_included") is not True
        or metadata.get("complete_output_irreps") is not True
        or metadata.get("coefficient_only") is not True
        or metadata.get("uses_reference_labels") is not False
        or metadata.get("uses_discovery_geometries") is not False
    ):
        raise ValueError(
            "metadata did not echo the frozen measure, weights, and label-free scope"
        )
    json.dumps(metadata, allow_nan=False, default=json_value)
    api_checks, api_checks_passed = _validate_api_checks(
        bundle["checks"], args.exactness_tolerance
    )

    raw_aggregate = _mapping(bundle["aggregate_blocks"], "aggregate_blocks")
    aggregate = {
        _block_key(key): _validated_block(value, f"aggregate block {key}")
        for key, value in raw_aggregate.items()
    }
    expected_keys = {
        (central, ell)
        for central in range(len(model_numbers))
        for ell in range(4)
    }
    if set(aggregate) != expected_keys:
        raise ValueError("aggregate environment has incomplete species or irrep coverage")

    raw_consumers = _mapping(bundle["consumer_blocks"], "consumer_blocks")
    if set(raw_consumers) != set(REQUIRED_CONSUMERS):
        raise ValueError("immediate-consumer environment has the wrong consumer family")
    consumers: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for name in REQUIRED_CONSUMERS:
        if re.fullmatch(r"[a-z0-9_]+", name) is None:
            raise ValueError("consumer name is not artifact-safe")
        raw_blocks = _mapping(raw_consumers[name], f"consumer {name}")
        blocks = {
            _block_key(key): _validated_block(value, f"{name} block {key}")
            for key, value in raw_blocks.items()
        }
        if set(blocks) != expected_keys:
            raise ValueError(f"consumer {name} has incomplete block coverage")
        consumers[name] = blocks

    arrays: dict[str, np.ndarray] = {
        "atomic_numbers": model_numbers,
        "provenance_checkpoint_sha256": np.asarray(checkpoint_sha256),
        "provenance_manifest_sha256": np.asarray(manifest_sha256),
        "provenance_source_sha256": np.asarray(source_sha256),
        "provenance_radial_sha256": np.asarray(radial_sha256),
        "rank_ladder": np.asarray(ranks, dtype=np.int64),
    }
    block_records = []
    numerical_checks = []
    for central, ell in sorted(expected_keys):
        combined = sum(
            consumer_weights[name] * consumers[name][(central, ell)]
            for name in REQUIRED_CONSUMERS
        )
        block = aggregate[(central, ell)]
        scale = max(float(np.linalg.norm(block)), np.finfo(np.float64).tiny)
        sum_residual = float(np.linalg.norm(block - combined) / scale)
        symmetry_residual = float(np.linalg.norm(block - block.T) / scale)
        symmetric = 0.5 * (block + block.T)
        raw_values, vectors = np.linalg.eigh(symmetric)
        spectral_scale = max(
            float(np.max(np.abs(raw_values))), np.finfo(np.float64).tiny
        )
        psd_relative_violation = max(0.0, -float(raw_values[0])) / spectral_scale
        values = np.maximum(raw_values[::-1], 0.0)
        vectors = vectors[:, ::-1]
        if max(ranks) != len(values) or any(rank > len(values) for rank in ranks):
            raise ValueError(
                "rank ladder must end at the common full environment dimension"
            )
        if float(np.sum(values)) <= np.finfo(np.float64).tiny:
            raise ValueError("every immediate-consumer block must have positive trace")
        numerical_checks.extend(
            [sum_residual, symmetry_residual, psd_relative_violation]
        )
        key_suffix = f"z{central}_l{ell}"
        arrays[f"gamma_{key_suffix}"] = block
        arrays[f"gamma_eigenvalues_{key_suffix}"] = values
        arrays[f"gamma_eigenvectors_{key_suffix}"] = vectors
        consumer_records = {}
        for name in REQUIRED_CONSUMERS:
            consumer_block = consumers[name][(central, ell)]
            consumer_scale = max(
                float(np.linalg.norm(consumer_block)), np.finfo(np.float64).tiny
            )
            consumer_symmetry = float(
                np.linalg.norm(consumer_block - consumer_block.T) / consumer_scale
            )
            consumer_values = np.linalg.eigvalsh(
                0.5 * (consumer_block + consumer_block.T)
            )
            consumer_spectral_scale = max(
                float(np.max(np.abs(consumer_values))), np.finfo(np.float64).tiny
            )
            consumer_psd = max(0.0, -float(consumer_values[0])) / consumer_spectral_scale
            numerical_checks.extend([consumer_symmetry, consumer_psd])
            arrays[f"gamma_{name}_{key_suffix}"] = consumer_block
            consumer_records[name] = {
                "aggregate_weight": consumer_weights[name],
                "trace": float(np.trace(consumer_block)),
                "symmetry_relative_residual": consumer_symmetry,
                "psd_relative_violation": consumer_psd,
            }
        block_records.append(
            {
                "central_index": central,
                "atomic_number": int(model_numbers[central]),
                "ell": ell,
                "dimension": len(values),
                "consumer_sum_relative_residual": sum_residual,
                "symmetry_relative_residual": symmetry_residual,
                "psd_relative_violation": psd_relative_violation,
                "consumers": consumer_records,
                "spectrum": _rank_summary(values, ranks),
            }
        )

    maximum_numerical_residual = max(numerical_checks, default=0.0)
    gate_passed = (
        api_checks_passed
        and maximum_numerical_residual <= args.exactness_tolerance
        and all(
            value <= args.exactness_tolerance
            for value in radial_checkpoint_validation.values()
        )
    )
    staged_npz = _staged_npz(args.npz, arrays)
    try:
        npz_sha256 = sha256_file(staged_npz)
        payload = {
            "schema_version": 1,
            "experiment": "T3_exact_immediate_consumer_environment",
            "requested_model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "rank_ladder": list(ranks),
            "provenance": {
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_candidates": [str(path) for path in candidates],
                "xyz": str(args.xyz),
                "xyz_sha256": source_sha256,
                "manifest": str(args.manifest),
                "manifest_sha256": manifest_sha256,
                "radial_npz": str(args.radial_npz),
                "radial_npz_sha256": radial_sha256,
                "immediate_consumer_source": str(Path(immediate_consumer.__file__)),
                "immediate_consumer_source_sha256": sha256_file(
                    Path(immediate_consumer.__file__)
                ),
                "runner_source": str(Path(__file__).resolve()),
                "runner_source_sha256": sha256_file(Path(__file__).resolve()),
            },
            "data_usage": {
                "xyz_bytes_hashed_only": True,
                "reference_energies_read": False,
                "reference_forces_read": False,
                "label_fitting": False,
                "rank_selection_from_evaluation_data": False,
                "formal_message_measure": message_measure["schema"],
                "radial_integration_scope": message_measure["integration_scope"],
            },
            "method": metadata,
            "required_consumers": list(REQUIRED_CONSUMERS),
            "radial_checkpoint_validation": radial_checkpoint_validation,
            "api_checks": api_checks,
            "maximum_numerical_residual": maximum_numerical_residual,
            "exactness_tolerance": args.exactness_tolerance,
            "blocks": block_records,
            "gate_passed": gate_passed,
            "npz": str(args.npz),
            "npz_sha256": npz_sha256,
            "environment": environment_manifest(),
            "claim_boundary": (
                "This is an exact immediate-consumer coefficient environment under "
                "the declared frozen discrete edge measure and output metric. It uses no DFT "
                "labels and does not certify continuous radial integration. It "
                "does not establish finite shared-projector fidelity, speedup, useful "
                "compression, cross-checkpoint recurrence, or physical interpretation."
            ),
        }
        staged_json = _staged_json(args.output, payload)
        try:
            os.replace(staged_npz, args.npz)
            try:
                os.replace(staged_json, args.output)
            except BaseException:
                args.npz.unlink(missing_ok=True)
                raise
        finally:
            staged_json.unlink(missing_ok=True)
    except BaseException:
        staged_npz.unlink(missing_ok=True)
        raise

    print(f"wrote {args.output}")
    print(f"wrote {args.npz}")
    print(f"gate_passed={gate_passed}")
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
