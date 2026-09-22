from pathlib import Path

import numpy as np

import mace_odt.audit as audit
from mace_odt.audit import forward_smoke_tests, json_value, rigid_rotation, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_bytes(b"mace-odt")
    assert sha256_file(path) == "5df96394c870a8bc9455a33b4b77a02fa999f05b87aee9f7d4364e1757f4f629"


def test_rigid_rotation_is_proper() -> None:
    rotation = rigid_rotation()
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-14)


def test_json_value_bounds_large_array() -> None:
    value = json_value(np.arange(100))
    assert value["shape"] == [100]


def test_json_value_numpy_scalar() -> None:
    assert json_value(np.bool_(True)) is True


def test_forward_smoke_tests_enforce_energy_tolerance_separately(monkeypatch) -> None:
    class FakeAtoms:
        def __init__(self) -> None:
            self.positions = np.zeros((2, 3))

        def copy(self):
            copied = FakeAtoms()
            copied.positions = self.positions.copy()
            return copied

        def get_chemical_formula(self) -> str:
            return "H"

        def __len__(self) -> int:
            return 2

        def __getitem__(self, item):
            del item
            return self.copy()

    evaluations = iter(
        [
            (0.0, np.zeros((2, 3))),
            (5.0e-8, np.zeros((2, 3))),
            (0.0, np.zeros((2, 3))),
            (0.0, np.zeros((2, 3))),
            (0.0, np.zeros((2, 3))),
            (0.0, np.zeros((2, 3))),
            (0.0, np.zeros((2, 3))),
        ]
    )
    monkeypatch.setattr(audit, "smoke_geometries", lambda: [FakeAtoms()])
    monkeypatch.setattr(audit, "evaluate", lambda calc, atoms: next(evaluations))

    result = forward_smoke_tests(calc=object())

    checks = {check["name"]: check for check in result["checks"]}
    assert checks["H_translation_energy"]["passed"] is False
    assert checks["H_translation_force"]["passed"] is True
    assert result["passed"] is False


def test_forward_smoke_tests_use_dtype_specific_tolerances(monkeypatch) -> None:
    class FakeAtoms:
        def __init__(self) -> None:
            self.positions = np.zeros((2, 3))

        def copy(self):
            return FakeAtoms()

        def get_chemical_formula(self) -> str:
            return "H"

        def __len__(self) -> int:
            return 2

        def __getitem__(self, item):
            del item
            return self.copy()

    monkeypatch.setattr(audit, "smoke_geometries", lambda: [FakeAtoms()])
    monkeypatch.setattr(
        audit,
        "evaluate",
        lambda calc, atoms: (0.0, np.zeros((2, 3))),
    )

    result = forward_smoke_tests(calc=object(), dtype="float32")

    assert result["passed"] is True
    assert result["declared_tolerances"]["energy"] == 5e-4
    assert result["declared_tolerances"]["force"] == 1e-3
    assert result["declared_tolerances"]["finite_difference_step"] == 1e-2
