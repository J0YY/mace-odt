"""Reproducible environment, checkpoint, and forward-pass audits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACKAGE_NAMES = (
    "ase",
    "e3nn",
    "mace-torch",
    "numpy",
    "opt-einsum",
    "scipy",
    "torch",
    "torch-ema",
    "torchmetrics",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: Iterable[str] = PACKAGE_NAMES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output or None


def environment_manifest() -> dict[str, Any]:
    torch_info: dict[str, Any] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "default_dtype": str(torch.get_default_dtype()),
        }
        if torch.cuda.is_available():
            torch_info["devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        torch_info = {"installed": False}

    return {
        "schema_version": 1,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": package_versions(),
        "torch": torch_info,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
        "git": {
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "status": command_output(["git", "status", "--short"]),
        },
    }


def json_value(value: Any) -> Any:
    """Convert common checkpoint metadata into bounded JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if hasattr(value, "detach") and hasattr(value, "shape"):
        array = value.detach().cpu()
        if array.numel() <= 32:
            return array.tolist()
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    text = str(value)
    return text if len(text) <= 1000 else text[:997] + "..."


def model_metadata(model: Any) -> dict[str, Any]:
    names = (
        "r_max",
        "num_interactions",
        "num_elements",
        "hidden_irreps",
        "MLP_irreps",
        "atomic_numbers",
        "correlation",
        "avg_num_neighbors",
        "heads",
        "atomic_energies_fn",
        "scale_shift",
    )
    metadata: dict[str, Any] = {}
    for name in names:
        if hasattr(model, name):
            metadata[name] = json_value(getattr(model, name))
    return metadata


def module_inventory(model: Any) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        direct_parameters = list(module.parameters(recurse=False))
        direct_buffers = list(module.named_buffers(recurse=False))
        inventory.append(
            {
                "name": name or "<root>",
                "type": f"{type(module).__module__}.{type(module).__name__}",
                "direct_parameter_count": int(
                    sum(parameter.numel() for parameter in direct_parameters)
                ),
                "direct_parameter_shapes": [
                    list(parameter.shape) for parameter in direct_parameters
                ],
                "direct_buffers": {
                    buffer_name: {
                        "shape": list(buffer.shape),
                        "dtype": str(buffer.dtype),
                    }
                    for buffer_name, buffer in direct_buffers
                },
            }
        )
    return inventory


def top_level_parameter_counts(model: Any) -> dict[str, int]:
    counts = {
        name: int(sum(parameter.numel() for parameter in child.parameters()))
        for name, child in model.named_children()
    }
    counts["<total>"] = int(sum(parameter.numel() for parameter in model.parameters()))
    counts["<trainable>"] = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return counts


def state_shapes(model: Any) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in model.state_dict().items()
    }


def find_checkpoint_candidates(model_name: str) -> list[Path]:
    roots = [
        Path.home() / ".cache" / "mace",
        Path.home() / ".cache" / "torch",
    ]
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        roots.extend([Path(xdg_cache), Path(xdg_cache) / "mace"])
    target = f"MACE-OFF23_{model_name}.model"
    found: list[Path] = []
    for root in roots:
        if root.exists():
            found.extend(root.rglob(target))
    return sorted(set(path.resolve() for path in found))


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    value: float
    tolerance: float
    units: str


def rigid_rotation() -> np.ndarray:
    axis = np.asarray([1.0, 2.0, -0.5], dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = 0.731
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def smoke_geometries() -> list[Any]:
    from ase import Atoms

    methane_scale = 1.09 / np.sqrt(3.0)
    methane = Atoms(
        symbols="CH4",
        positions=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
        * methane_scale,
    )
    water = Atoms(
        symbols="OH2",
        positions=[
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2399872, 0.927297, 0.0],
        ],
    )
    return [methane, water]


def evaluate(calc: Any, atoms: Any) -> tuple[float, np.ndarray]:
    candidate = atoms.copy()
    candidate.calc = calc
    energy = float(candidate.get_potential_energy())
    forces = np.asarray(candidate.get_forces(), dtype=float)
    return energy, forces


def forward_smoke_tests(calc: Any, dtype: str = "float64") -> dict[str, Any]:
    if dtype == "float64":
        energy_tolerance = 1e-8
        force_tolerance = 1e-7
        derivative_tolerance = 2e-4
        derivative_step = 1e-4
    elif dtype == "float32":
        energy_tolerance = 5e-4
        force_tolerance = 1e-3
        derivative_tolerance = 2e-2
        derivative_step = 1e-2
    else:
        raise ValueError(f"unsupported smoke-test dtype: {dtype}")
    results: list[CheckResult] = []
    examples: list[dict[str, Any]] = []

    for atoms in smoke_geometries():
        base_energy, base_forces = evaluate(calc, atoms)
        examples.append(
            {
                "formula": atoms.get_chemical_formula(),
                "energy_eV": base_energy,
                "force_norm_eV_per_A": float(np.linalg.norm(base_forces)),
            }
        )

        translated = atoms.copy()
        translated.positions += np.asarray([0.37, -0.19, 0.51])
        translated_energy, translated_forces = evaluate(calc, translated)
        translation_energy_error = abs(translated_energy - base_energy)
        translation_force_error = float(
            np.max(np.abs(translated_forces - base_forces))
        )
        results.append(
            CheckResult(
                f"{atoms.get_chemical_formula()}_translation_energy",
                translation_energy_error <= energy_tolerance,
                translation_energy_error,
                energy_tolerance,
                "eV",
            )
        )
        results.append(
            CheckResult(
                f"{atoms.get_chemical_formula()}_translation_force",
                translation_force_error <= force_tolerance,
                translation_force_error,
                force_tolerance,
                "eV_per_A",
            )
        )

        rotation = rigid_rotation()
        rotated = atoms.copy()
        rotated.positions = atoms.positions @ rotation.T
        rotated_energy, rotated_forces = evaluate(calc, rotated)
        expected_forces = base_forces @ rotation.T
        rotation_energy_error = abs(rotated_energy - base_energy)
        rotation_force_error = float(
            np.max(np.abs(rotated_forces - expected_forces))
        )
        results.append(
            CheckResult(
                f"{atoms.get_chemical_formula()}_rotation_energy",
                rotation_energy_error <= energy_tolerance,
                rotation_energy_error,
                energy_tolerance,
                "eV",
            )
        )
        results.append(
            CheckResult(
                f"{atoms.get_chemical_formula()}_rotation_force",
                rotation_force_error <= force_tolerance,
                rotation_force_error,
                force_tolerance,
                "eV_per_A",
            )
        )

        permutation = np.arange(len(atoms))[::-1]
        permuted = atoms[permutation]
        permuted_energy, permuted_forces = evaluate(calc, permuted)
        expected_permuted_forces = base_forces[permutation]
        permutation_energy_error = abs(permuted_energy - base_energy)
        permutation_force_error = float(
            np.max(np.abs(permuted_forces - expected_permuted_forces))
        )
        results.append(
            CheckResult(
                f"{atoms.get_chemical_formula()}_permutation_energy",
                permutation_energy_error <= energy_tolerance,
                permutation_energy_error,
                energy_tolerance,
                "eV",
            )
        )
        results.append(
            CheckResult(
                f"{atoms.get_chemical_formula()}_permutation_force",
                permutation_force_error <= force_tolerance,
                permutation_force_error,
                force_tolerance,
                "eV_per_A",
            )
        )

    finite_atoms = smoke_geometries()[0]
    finite_energy, finite_forces = evaluate(calc, finite_atoms)
    del finite_energy
    step = derivative_step
    plus = finite_atoms.copy()
    minus = finite_atoms.copy()
    plus.positions[1, 0] += step
    minus.positions[1, 0] -= step
    energy_plus, _ = evaluate(calc, plus)
    energy_minus, _ = evaluate(calc, minus)
    finite_difference_force = -(energy_plus - energy_minus) / (2.0 * step)
    derivative_error = abs(finite_difference_force - finite_forces[1, 0])
    results.append(
        CheckResult(
            "CH4_force_finite_difference",
            derivative_error <= derivative_tolerance,
            derivative_error,
            derivative_tolerance,
            "eV_per_A",
        )
    )

    return {
        "passed": all(result.passed for result in results),
        "checks": [asdict(result) for result in results],
        "examples": examples,
        "declared_tolerances": {
            "energy": energy_tolerance,
            "force": force_tolerance,
            "finite_difference_force": derivative_tolerance,
            "finite_difference_step": derivative_step,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_value) + "\n"
    )
