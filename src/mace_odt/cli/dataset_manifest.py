"""Freeze a metadata-stratified evaluation subset before rank evaluation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from mace_odt.audit import sha256_file, write_json


def deterministic_score(seed: int, index: int) -> str:
    return hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()


def proportional_allocations(
    group_sizes: dict[str, int], sample_size: int
) -> dict[str, int]:
    if sample_size < len(group_sizes):
        raise ValueError("sample size must include at least one item from every group")
    if sample_size > sum(group_sizes.values()):
        raise ValueError("sample size exceeds the dataset")
    allocations = {name: 1 for name in group_sizes}
    remaining = sample_size - len(group_sizes)
    capacities = {name: size - 1 for name, size in group_sizes.items()}
    total_capacity = sum(capacities.values())
    raw = {
        name: remaining * capacity / total_capacity if total_capacity else 0.0
        for name, capacity in capacities.items()
    }
    for name in allocations:
        addition = min(capacities[name], int(raw[name]))
        allocations[name] += addition
    unassigned = sample_size - sum(allocations.values())
    priority = sorted(
        allocations,
        key=lambda name: (raw[name] - int(raw[name]), capacities[name], name),
        reverse=True,
    )
    while unassigned:
        progressed = False
        for name in priority:
            if allocations[name] < group_sizes[name]:
                allocations[name] += 1
                unassigned -= 1
                progressed = True
                if not unassigned:
                    break
        if not progressed:
            raise RuntimeError("unable to complete the sample allocation")
    return allocations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--archive-id", required=True)
    parser.add_argument("--archive-md5", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from ase.io import iread

    groups: dict[str, list[dict[str, Any]]] = {}
    total_atoms = 0
    for index, atoms in enumerate(iread(args.xyz, index=":")):
        config_type = str(atoms.info.get("config_type", "<missing>"))
        record = {
            "index": index,
            "config_type": config_type,
            "num_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "selection_score": deterministic_score(args.seed, index),
        }
        groups.setdefault(config_type, []).append(record)
        total_atoms += len(atoms)
    group_sizes = {name: len(records) for name, records in groups.items()}
    allocations = proportional_allocations(group_sizes, args.sample_size)
    selected = []
    for name, records in groups.items():
        ranked = sorted(records, key=lambda record: record["selection_score"])
        selected.extend(ranked[: allocations[name]])
    selected.sort(key=lambda record: record["index"])
    for record in selected:
        record.pop("selection_score")

    payload = {
        "schema_version": 1,
        "experiment": "E7_frozen_evaluation_manifest",
        "source": {
            "repository_doi": "10.17863/CAM.107498",
            "archive_id": args.archive_id,
            "archive_md5": args.archive_md5,
            "source_url": args.source_url,
            "xyz_path": str(args.xyz),
            "xyz_sha256": sha256_file(args.xyz),
        },
        "selection": {
            "rule": (
                "One item per config_type, then proportional allocation by group "
                "size. Within each group choose the lowest SHA-256 score of seed "
                "and original frame index."
            ),
            "seed": args.seed,
            "sample_size": args.sample_size,
            "stratification_field": "config_type",
        },
        "dataset": {
            "num_configurations": sum(group_sizes.values()),
            "num_atoms": total_atoms,
            "group_sizes": group_sizes,
        },
        "selected_group_counts": allocations,
        "selected": selected,
        "claim_boundary": (
            "Selection uses metadata only and was frozen before projected model "
            "errors were evaluated. Reference labels are not used to fit subspaces."
        ),
    }
    write_json(args.output, payload)
    print(f"wrote {args.output}")
    print(f"configurations={sum(group_sizes.values())}")
    print(f"selected={len(selected)}")


if __name__ == "__main__":
    main()
