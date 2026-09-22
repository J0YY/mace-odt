"""Reference ODT bridge and tied mixed-order projector mathematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def symmetric_eigh_descending(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    negative_floor = -100.0 * np.finfo(matrix.dtype).eps * max(
        1.0, float(np.linalg.norm(symmetric, ord=2))
    )
    if float(eigenvalues[-1]) < negative_floor:
        raise ValueError("operator is not positive semidefinite within tolerance")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return eigenvalues, eigenvectors


def supported_factor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    eigenvalues, eigenvectors = symmetric_eigh_descending(matrix)
    tolerance = (
        float(eigenvalues[0])
        * max(matrix.shape)
        * np.finfo(matrix.dtype).eps
        if eigenvalues.size
        else 0.0
    )
    keep = eigenvalues > tolerance
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    return factor, eigenvalues, tolerance


@dataclass(frozen=True)
class ValidCutResult:
    squared_singular_values: np.ndarray
    decoder: np.ndarray
    encoder: np.ndarray
    projector: np.ndarray
    approximation: np.ndarray
    residual_squared: float
    discarded_eigenvalue_sum: float
    upstream_isometry_residual: float


def valid_cut_decomposition(
    upstream: np.ndarray, downstream: np.ndarray, rank: int
) -> ValidCutResult:
    """Coordinate-explicit form of the canonical ODT cut calculation."""
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if upstream.ndim != 2 or downstream.ndim != 2:
        raise ValueError("upstream and downstream maps must be matrices")
    if upstream.shape[0] != downstream.shape[1]:
        raise ValueError("bond dimensions do not match")
    left_gram = upstream @ upstream.T
    right_gram = downstream.T @ downstream
    factor, _, _ = supported_factor(left_gram)
    if factor.shape[1] == 0:
        raise ValueError("the upstream support is empty")
    left_inverse = np.linalg.solve(factor.T @ factor, factor.T)
    canonical_upstream = left_inverse @ upstream
    isometry_residual = float(
        np.linalg.norm(canonical_upstream @ canonical_upstream.T - np.eye(factor.shape[1]))
    )
    canonical_environment = factor.T @ right_gram @ factor
    eigenvalues, eigenvectors = symmetric_eigh_descending(canonical_environment)
    retained = min(rank, factor.shape[1])
    modes = eigenvectors[:, :retained]
    decoder = factor @ modes
    encoder = modes.T @ left_inverse
    projector = decoder @ encoder
    approximation = downstream @ projector @ upstream
    original = downstream @ upstream
    residual_squared = float(np.linalg.norm(original - approximation) ** 2)
    discarded = float(np.sum(eigenvalues[retained:]))
    return ValidCutResult(
        squared_singular_values=eigenvalues,
        decoder=decoder,
        encoder=encoder,
        projector=projector,
        approximation=approximation,
        residual_squared=residual_squared,
        discarded_eigenvalue_sum=discarded,
        upstream_isometry_residual=isometry_residual,
    )


def slot_marginal(tensor: np.ndarray, slot: int) -> np.ndarray:
    moved = np.moveaxis(tensor, slot, 0)
    unfolding = moved.reshape(moved.shape[0], -1)
    return unfolding @ unfolding.T


def mixed_order_score(
    kernels: Mapping[int, np.ndarray], beta: Mapping[int, float]
) -> np.ndarray:
    dimensions = {tensor.shape[0] for tensor in kernels.values()}
    if len(dimensions) != 1:
        raise ValueError("all kernels must use the same primitive dimension")
    dimension = dimensions.pop()
    score = np.zeros((dimension, dimension), dtype=next(iter(kernels.values())).dtype)
    for order, tensor in kernels.items():
        if tensor.ndim != order or any(size != dimension for size in tensor.shape):
            raise ValueError(f"kernel order {order} has incompatible axes")
        if beta[order] <= 0:
            raise ValueError("order weights must be positive")
        for slot in range(order):
            score += beta[order] * slot_marginal(tensor, slot)
    return 0.5 * (score + score.T)


def apply_projector_all_slots(tensor: np.ndarray, projector: np.ndarray) -> np.ndarray:
    result = tensor
    for slot in range(tensor.ndim):
        result = np.tensordot(projector, result, axes=([1], [slot]))
        result = np.moveaxis(result, 0, slot)
    return result


@dataclass(frozen=True)
class MixedOrderResult:
    score: np.ndarray
    eigenvalues: np.ndarray
    projector: np.ndarray
    projected_kernels: dict[int, np.ndarray]
    actual_error_squared: float
    marginal_bound: float
    quasioptimal_ratio_bound: float
    inequality_tolerance: float


def mixed_order_decomposition(
    kernels: Mapping[int, np.ndarray], beta: Mapping[int, float], rank: int
) -> MixedOrderResult:
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    score = mixed_order_score(kernels, beta)
    eigenvalues, eigenvectors = symmetric_eigh_descending(score)
    retained = min(rank, score.shape[0])
    modes = eigenvectors[:, :retained]
    projector = modes @ modes.T
    projected = {
        order: apply_projector_all_slots(tensor, projector)
        for order, tensor in kernels.items()
    }
    actual_error = float(
        sum(
            beta[order] * np.linalg.norm(kernels[order] - projected[order]) ** 2
            for order in kernels
        )
    )
    bound = float(np.trace((np.eye(score.shape[0]) - projector) @ score))
    scale = max(1.0, actual_error, bound)
    tolerance = 500.0 * np.finfo(score.dtype).eps * scale * score.shape[0]
    if actual_error > bound + tolerance:
        raise AssertionError("the tied marginal upper bound failed")
    maximum_order = max(kernels)
    if bound > maximum_order * actual_error + tolerance:
        raise AssertionError("the mixed-order factor bound failed")
    return MixedOrderResult(
        score=score,
        eigenvalues=eigenvalues,
        projector=projector,
        projected_kernels=projected,
        actual_error_squared=actual_error,
        marginal_bound=bound,
        quasioptimal_ratio_bound=float(maximum_order),
        inequality_tolerance=tolerance,
    )
