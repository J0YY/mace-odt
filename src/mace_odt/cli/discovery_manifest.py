"""Freeze a label-free discovery set disjoint from the held-out evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mace_odt.audit import sha256_file, write_json
from mace_odt.cli.dataset_manifest import deterministic_score, proportional_allocations


def exact_structure_sha256(atoms: Any) -> str:
    """Hash ordered Cartesian geometry bytes while ignoring attached labels."""
    digest = hashlib.sha256()
    arrays = (
        np.asarray(atoms.numbers, dtype="<i8"),
        np.asarray(atoms.positions, dtype="<f8"),
        np.asarray(atoms.cell.array, dtype="<f8"),
        np.asarray(atoms.pbc, dtype=np.uint8),
    )
    for array in arrays:
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from ase.io import iread

    evaluation = json.loads(args.evaluation_manifest.read_text())
    if evaluation.get("schema_version") != 1 or evaluation.get("experiment") != "E7_frozen_evaluation_manifest":
        raise ValueError("unsupported evaluation manifest schema or experiment")
    source_sha256 = sha256_file(args.xyz)
    expected_sha256 = evaluation.get("source", {}).get("xyz_sha256")
    if source_sha256 != expected_sha256:
        raise ValueError("evaluation manifest and discovery XYZ source do not match")
    evaluation_indices = {int(item["index"]) for item in evaluation["selected"]}
    if len(evaluation_indices) != len(evaluation["selected"]):
        raise ValueError("evaluation manifest contains duplicate indices")
    evaluation_hashes = {
        exact_structure_sha256(atoms)
        for index, atoms in enumerate(iread(args.xyz, index=":"))
        if index in evaluation_indices
    }
    if len(evaluation_hashes) != len(evaluation_indices):
        raise ValueError("the evaluation manifest contains missing or duplicate structures")

    groups: dict[str, list[dict[str, Any]]] = {}
    seen_hashes = set(evaluation_hashes)
    excluded_duplicate_count = 0
    dataset_size = 0
    for index, atoms in enumerate(iread(args.xyz, index=":")):
        dataset_size += 1
        if index in evaluation_indices:
            continue
        geometry_hash = exact_structure_sha256(atoms)
        if geometry_hash in seen_hashes:
            excluded_duplicate_count += 1
            continue
        seen_hashes.add(geometry_hash)
        config_type = str(atoms.info.get("config_type", "<missing>"))
        groups.setdefault(config_type, []).append(
            {
                "index": index,
                "config_type": config_type,
                "num_atoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
                "atomic_numbers": sorted({int(number) for number in atoms.numbers}),
                "exact_structure_sha256": geometry_hash,
                "selection_score": deterministic_score(args.seed, index),
            }
        )
    group_sizes = {name: len(records) for name, records in groups.items() if records}
    allocations = proportional_allocations(group_sizes, args.sample_size)
    selected = []
    for name, records in groups.items():
        ranked = sorted(records, key=lambda record: record["selection_score"])
        selected.extend(ranked[: allocations[name]])
    species_occurrences: dict[int, int] = {}
    for record in selected:
        for atomic_number in record["atomic_numbers"]:
            species_occurrences[atomic_number] = species_occurrences.get(atomic_number, 0) + 1
    fold_species_counts = {
        "A": {number: 0 for number in species_occurrences},
        "B": {number: 0 for number in species_occurrences},
    }
    fold_sizes = {"A": 0, "B": 0}
    rare_first = sorted(
        selected,
        key=lambda record: (
            min(species_occurrences[number] for number in record["atomic_numbers"]),
            record["selection_score"],
        ),
    )
    for record in rare_first:
        costs = {
            fold: sum(
                fold_species_counts[fold][number] / species_occurrences[number]
                for number in record["atomic_numbers"]
            )
            for fold in ("A", "B")
        }
        fold = min(
            ("A", "B"),
            key=lambda name: (
                costs[name],
                fold_sizes[name],
                0 if (int(record["selection_score"], 16) % 2 == 0) == (name == "A") else 1,
            ),
        )
        record["convergence_fold"] = fold
        fold_sizes[fold] += 1
        for atomic_number in record["atomic_numbers"]:
            fold_species_counts[fold][atomic_number] += 1
    selected.sort(key=lambda record: record["index"])
    for record in selected:
        record.pop("selection_score")
    if any(
        fold_species_counts[fold][number] == 0
        for fold in ("A", "B")
        for number in species_occurrences
    ):
        raise ValueError("unable to give both convergence folds every selected species")
    selected_hashes = {record["exact_structure_sha256"] for record in selected}
    if selected_hashes & evaluation_hashes:
        raise RuntimeError("discovery and evaluation structure hashes overlap")

    payload = {
        "schema_version": 1,
        "experiment": "T0_data_assisted_discovery_manifest",
        "source": {
            **evaluation["source"],
            "xyz_path": str(args.xyz),
            "xyz_sha256": source_sha256,
        },
        "evaluation_manifest": str(args.evaluation_manifest),
        "evaluation_manifest_sha256": sha256_file(args.evaluation_manifest),
        "evaluation_indices_sha256": hashlib.sha256(
            ",".join(map(str, sorted(evaluation_indices))).encode()
        ).hexdigest(),
        "selection": {
            "rule": (
                "Exclude held-out indices and byte-identical ordered Cartesian geometry "
                "hashes. Then take one "
                "item per config_type and allocate the remainder proportionally. Rank "
                "within groups by SHA-256 of the seed and original frame index. Assign "
                "convergence folds deterministically in rare-species-first order, always "
                "choosing the fold with lower normalized coverage of the record's species."
            ),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "stratification_field": "config_type",
            "uses_reference_labels": False,
        },
        "dataset": {
            "num_configurations": dataset_size,
            "eligible_group_sizes": group_sizes,
            "excluded_evaluation_count": len(evaluation_indices),
            "excluded_duplicate_count": excluded_duplicate_count,
        },
        "selected_group_counts": allocations,
        "convergence_fold_sizes": fold_sizes,
        "convergence_fold_species_counts": fold_species_counts,
        "selected": selected,
        "disjointness": {
            "evaluation_structure_hash_count": len(evaluation_hashes),
            "discovery_structure_hash_count": len(selected_hashes),
            "overlap_count": 0,
        },
        "claim_boundary": (
            "Selection uses configuration metadata and geometry only. Reference energies "
            "and forces are not read and the existing held-out structures are excluded."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"selected={len(selected)}")


if __name__ == "__main__":
    main()
