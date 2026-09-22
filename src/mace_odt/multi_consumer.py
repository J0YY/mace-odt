"""Data-assisted additive-sensitivity Grams at the first MACE density."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


BlockKey = tuple[int, int]


def canonical_gradient_gram(
    native_gradient: torch.Tensor,
    node_attrs: torch.Tensor,
    metric_factors: Mapping[BlockKey, np.ndarray],
    angular_slices: Mapping[int, slice],
) -> dict[BlockKey, np.ndarray]:
    """Haar-twirl one scalar-output gradient in canonical function coordinates.

    The native gradient has axes ``[node, channel, angular]``. For each central
    species and irrep, the returned matrix is the sum over nodes of that
    species and the mean over magnetic coordinates of
    ``(C.T @ g) (C.T @ g).T``. Averaging over the full magnetic multiplet is
    the analytic O(3) twirl. Dataset averaging is deliberately left to the
    caller so every configuration can receive equal weight.
    """
    if native_gradient.ndim != 3:
        raise ValueError("native_gradient must have axes [node, channel, angular]")
    if node_attrs.ndim != 2 or node_attrs.shape[0] != native_gradient.shape[0]:
        raise ValueError("node_attrs do not match the gradient node axis")
    species = node_attrs.argmax(dim=-1)
    result: dict[BlockKey, np.ndarray] = {}
    for central in range(node_attrs.shape[1]):
        selected = species == central
        if not bool(torch.any(selected)):
            continue
        for ell, section in angular_slices.items():
            factor = torch.as_tensor(
                metric_factors[(central, ell)],
                dtype=native_gradient.dtype,
                device=native_gradient.device,
            )
            block = native_gradient[selected, :, section]
            if block.shape[1] != factor.shape[0]:
                raise ValueError("metric factor does not match the channel axis")
            canonical = torch.einsum("cs,ncm->nsm", factor, block)
            gram = torch.einsum("nsm,ntm->st", canonical, canonical)
            gram = gram / canonical.shape[2]
            result[(central, ell)] = gram.detach().cpu().numpy()
    return result


def canonical_activation_gram(
    density: torch.Tensor,
    node_attrs: torch.Tensor,
    left_inverses: Mapping[BlockKey, np.ndarray],
    angular_slices: Mapping[int, slice],
) -> dict[BlockKey, np.ndarray]:
    """Return the canonical activation covariance baseline for one geometry."""
    if density.ndim != 3:
        raise ValueError("density must have axes [node, channel, angular]")
    species = node_attrs.argmax(dim=-1)
    result: dict[BlockKey, np.ndarray] = {}
    for central in range(node_attrs.shape[1]):
        selected = species == central
        if not bool(torch.any(selected)):
            continue
        for ell, section in angular_slices.items():
            left_inverse = torch.as_tensor(
                left_inverses[(central, ell)],
                dtype=density.dtype,
                device=density.device,
            )
            block = density[selected, :, section]
            canonical = torch.einsum("sc,ncm->nsm", left_inverse, block)
            gram = torch.einsum("nsm,ntm->st", canonical, canonical)
            gram = gram / canonical.shape[2]
            result[(central, ell)] = gram.detach().cpu().numpy()
    return result


def symmetric_psd(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Symmetrize a sampled Gram and report absolute and relative violations."""
    symmetric = 0.5 * (
        np.asarray(matrix, dtype=np.float64)
        + np.asarray(matrix, dtype=np.float64).T
    )
    values, vectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(values))), np.finfo(np.float64).tiny)
    absolute_violation = max(0.0, -float(values[0]))
    relative_violation = absolute_violation / scale
    clipped = (vectors * np.maximum(values, 0.0)[None, :]) @ vectors.T
    return 0.5 * (clipped + clipped.T), absolute_violation, relative_violation


def trace_balanced_sum(
    first: np.ndarray,
    second: np.ndarray,
    first_weight: float = 0.5,
    second_weight: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Combine two PSD environments after independent trace normalization."""
    if first_weight < 0.0 or second_weight < 0.0:
        raise ValueError("consumer weights must be nonnegative")
    if first_weight + second_weight <= 0.0:
        raise ValueError("at least one consumer weight must be positive")
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("consumer environments must be equal square matrices")
    first_trace = max(float(np.trace(first)), 0.0)
    second_trace = max(float(np.trace(second)), 0.0)
    tiny = np.finfo(np.float64).tiny
    if first_trace <= tiny or second_trace <= tiny:
        raise ValueError("both consumers must have positive trace in every block")
    total_weight = first_weight + second_weight
    combined = (
        (first_weight / total_weight) * first / first_trace
        + (second_weight / total_weight) * second / second_trace
    )
    return 0.5 * (combined + combined.T), {
        "linear_trace": first_trace,
        "nonlinear_trace": second_trace,
        "linear_weight": first_weight / total_weight,
        "nonlinear_weight": second_weight / total_weight,
    }


def globally_trace_balanced_blocks(
    first: Mapping[BlockKey, np.ndarray],
    second: Mapping[BlockKey, np.ndarray],
    angular_dimensions: Mapping[int, int],
    first_weight: float = 0.5,
    second_weight: float = 0.5,
) -> tuple[dict[BlockKey, np.ndarray], dict[str, float]]:
    """Trace-normalize two consumer families once, then combine every block."""
    if set(first) != set(second):
        raise ValueError("consumer block families do not match")
    if first_weight < 0.0 or second_weight < 0.0:
        raise ValueError("consumer weights must be nonnegative")
    total_weight = first_weight + second_weight
    if total_weight <= 0.0:
        raise ValueError("at least one consumer weight must be positive")
    first_trace = float(
        sum(angular_dimensions[ell] * np.trace(first[(central, ell)])
            for central, ell in first)
    )
    second_trace = float(
        sum(angular_dimensions[ell] * np.trace(second[(central, ell)])
            for central, ell in second)
    )
    if first_trace <= 0.0 or second_trace <= 0.0:
        raise ValueError("both consumer families must have positive global trace")
    combined = {
        key: 0.5 * (
            (first_weight / total_weight) * first[key] / first_trace
            + (second_weight / total_weight) * second[key] / second_trace
            + (
                (first_weight / total_weight) * first[key] / first_trace
                + (second_weight / total_weight) * second[key] / second_trace
            ).T
        )
        for key in first
    }
    return combined, {
        "linear_global_trace": first_trace,
        "nonlinear_global_trace": second_trace,
        "linear_weight": first_weight / total_weight,
        "nonlinear_weight": second_weight / total_weight,
    }


def descending_eigensystem(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic descending eigensystem of a symmetric matrix."""
    matrix = 0.5 * (
        np.asarray(matrix, dtype=np.float64)
        + np.asarray(matrix, dtype=np.float64).T
    )
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def deterministic_completed_eigensystem(
    matrix: np.ndarray,
    relative_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Diagonalize the supported spectrum and complete its nullspace canonically."""
    values, vectors = descending_eigensystem(matrix)
    scale = max(float(values[0]), np.finfo(np.float64).tiny)
    empirical_rank = int(np.sum(values > relative_tolerance * scale))
    columns = [vectors[:, index].copy() for index in range(empirical_rank)]
    dimension = matrix.shape[0]
    for axis in range(dimension):
        candidate = np.eye(dimension)[:, axis].copy()
        for _ in range(2):
            for column in columns:
                candidate -= column * float(column @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-10:
            columns.append(candidate / norm)
        if len(columns) == dimension:
            break
    if len(columns) != dimension:
        raise RuntimeError("failed to construct a deterministic orthonormal completion")
    completed = np.column_stack(columns)
    completed, _ = np.linalg.qr(completed)
    if np.linalg.norm(completed.T @ completed - np.eye(dimension)) > 1e-10:
        raise RuntimeError("deterministic eigensystem completion is not orthonormal")
    return values, completed, empirical_rank


def projector_distance(
    left_vectors: np.ndarray, right_vectors: np.ndarray, rank: int
) -> float:
    """Normalized Frobenius distance between two rank-r invariant subspaces."""
    if rank < 0 or rank > min(left_vectors.shape[1], right_vectors.shape[1]):
        raise ValueError("rank is outside the supplied bases")
    if rank == 0:
        return 0.0
    left = left_vectors[:, :rank]
    right = right_vectors[:, :rank]
    left_projector = left @ left.T
    right_projector = right @ right.T
    return float(np.linalg.norm(left_projector - right_projector) / np.sqrt(2.0 * rank))
