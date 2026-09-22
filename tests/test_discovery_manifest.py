from ase import Atoms

from mace_odt.cli.discovery_manifest import exact_structure_sha256


def test_structure_hash_ignores_labels_but_not_geometry() -> None:
    first = Atoms("OH2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    second = first.copy()
    second.info["energy"] = -123.0
    assert exact_structure_sha256(first) == exact_structure_sha256(second)
    second.positions[1, 0] += 1e-8
    assert exact_structure_sha256(first) != exact_structure_sha256(second)
