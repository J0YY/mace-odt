"""Exact path quotients for repeated symmetric contraction slots."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


def symmetrize_repeated_slots(tensor: np.ndarray, order: int) -> np.ndarray:
    """Average a coupling tensor over its repeated input slots.

    The final axis is the stored path axis. Any axes before the repeated slots
    are output equivariance axes and remain fixed.
    """
    if order < 1:
        raise ValueError("order must be positive")
    leading = tensor.ndim - order - 1
    if leading < 0:
        raise ValueError("tensor has too few axes for the declared order")
    repeated_axes = tuple(range(leading, leading + order))
    prefix = tuple(range(leading))
    path_axis = tensor.ndim - 1
    permutations = list(itertools.permutations(repeated_axes))
    return sum(
        np.transpose(tensor, axes=prefix + permutation + (path_axis,))
        for permutation in permutations
    ) / len(permutations)


@dataclass(frozen=True)
class PathQuotient:
    stored_dimension: int
    supported_dimension: int
    null_dimension: int
    tolerance: float
    singular_values: np.ndarray
    support_basis: np.ndarray
    null_basis: np.ndarray
    null_residual: float
    projector_residual: float


def compute_path_quotient(tensor: np.ndarray, order: int) -> PathQuotient:
    symmetrized = symmetrize_repeated_slots(tensor, order)
    matrix = symmetrized.reshape(-1, symmetrized.shape[-1])
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=True)
    leading_value = float(singular_values[0]) if singular_values.size else 0.0
    tolerance = leading_value * max(matrix.shape) * np.finfo(matrix.dtype).eps
    rank = int(np.sum(singular_values > tolerance))
    support_basis = right_vectors[:rank].T
    null_basis = right_vectors[rank:].T
    null_residual = float(np.linalg.norm(matrix @ null_basis))
    identity = np.eye(matrix.shape[1], dtype=matrix.dtype)
    resolved_identity = support_basis @ support_basis.T + null_basis @ null_basis.T
    projector_residual = float(np.linalg.norm(resolved_identity - identity))
    return PathQuotient(
        stored_dimension=matrix.shape[1],
        supported_dimension=rank,
        null_dimension=matrix.shape[1] - rank,
        tolerance=tolerance,
        singular_values=singular_values,
        support_basis=support_basis,
        null_basis=null_basis,
        null_residual=null_residual,
        projector_residual=projector_residual,
    )


def scalar_repeated_slot_response(
    coupling: np.ndarray,
    path_weights: np.ndarray,
    features: np.ndarray,
    element_one_hot: np.ndarray,
) -> np.ndarray:
    """Evaluate one scalar order block directly for an integration check."""
    order = coupling.ndim - 1
    if order not in (1, 2, 3):
        raise ValueError("the current direct oracle supports orders one through three")
    if order == 1:
        return np.einsum("ip,epc,bci,be->bc", coupling, path_weights, features, element_one_hot)
    if order == 2:
        return np.einsum(
            "ijp,epc,bci,bcj,be->bc",
            coupling,
            path_weights,
            features,
            features,
            element_one_hot,
        )
    return np.einsum(
        "ijkp,epc,bci,bcj,bck,be->bc",
        coupling,
        path_weights,
        features,
        features,
        features,
        element_one_hot,
    )


def random_null_perturbation_check(
    coupling: np.ndarray,
    quotient: PathQuotient,
    num_elements: int,
    num_channels: int,
    seed: int = 20260922,
) -> dict[str, Any]:
    if coupling.ndim - 1 not in (1, 2, 3):
        return {"status": "not_run", "reason": "non_scalar_or_unsupported_order"}
    rng = np.random.default_rng(seed)
    batch_size = 7
    input_dimension = coupling.shape[0]
    features = rng.normal(size=(batch_size, num_channels, input_dimension))
    element_indices = np.arange(batch_size) % num_elements
    elements = np.eye(num_elements)[element_indices]

    if quotient.null_dimension:
        null_direction = quotient.null_basis[:, 0]
        null_weights = np.zeros(
            (num_elements, quotient.stored_dimension, num_channels), dtype=coupling.dtype
        )
        null_weights[0, :, 0] = null_direction
        null_response = scalar_repeated_slot_response(
            coupling, null_weights, features, elements
        )
        null_response_norm = float(np.linalg.norm(null_response))
    else:
        null_response_norm = 0.0

    retained_direction = quotient.support_basis[:, 0]
    retained_weights = np.zeros(
        (num_elements, quotient.stored_dimension, num_channels), dtype=coupling.dtype
    )
    retained_weights[0, :, 0] = retained_direction
    retained_response = scalar_repeated_slot_response(
        coupling, retained_weights, features, elements
    )
    retained_response_norm = float(np.linalg.norm(retained_response))
    scale = max(retained_response_norm, 1.0)
    null_tolerance = (
        100.0
        * np.finfo(coupling.dtype).eps
        * max(coupling.shape)
        * max(float(np.linalg.norm(coupling)), 1.0)
    )
    passed = (
        null_response_norm <= null_tolerance * scale
        and retained_response_norm > np.sqrt(np.finfo(coupling.dtype).eps)
    )
    return {
        "status": "passed" if passed else "failed",
        "null_response_norm": null_response_norm,
        "retained_response_norm": retained_response_norm,
        "relative_null_response": null_response_norm / scale,
        "relative_null_tolerance": null_tolerance,
        "seed": seed,
    }
