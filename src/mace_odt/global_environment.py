"""Dooms-aligned environments for the tied mixed-order first branch."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping

import numpy as np
import torch

from mace_odt.decomposition import symmetric_eigh_descending


def _validated_output_metric(
    output_metric: np.ndarray, channels: int
) -> np.ndarray:
    metric = np.asarray(output_metric, dtype=np.float64)
    if metric.shape != (channels, channels) or not np.isfinite(metric).all():
        raise ValueError("the output metric must be a finite channel Gram")
    scale = max(float(np.linalg.norm(metric)), np.finfo(np.float64).tiny)
    symmetry_residual = float(np.linalg.norm(metric - metric.T) / scale)
    if symmetry_residual > 1e-10:
        raise ValueError("the output metric must be symmetric")
    metric = 0.5 * (metric + metric.T)
    eigenvalues = np.linalg.eigvalsh(metric)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1e-12 * spectral_scale:
        raise ValueError("the output metric must be positive semidefinite")
    return metric


def angular_ell_labels(block_dimensions: Mapping[int, int]) -> np.ndarray:
    labels: list[int] = []
    for ell, dimension in block_dimensions.items():
        if dimension != 2 * ell + 1:
            raise ValueError("an angular block must contain a complete irrep")
        labels.extend([ell] * dimension)
    return np.asarray(labels, dtype=np.int64)


def angular_metric(
    metric_factors: Mapping[int, np.ndarray], ell_labels: np.ndarray
) -> np.ndarray:
    """Return L[c,d,i] for every native angular coordinate i."""
    channel_counts = {factor.shape[0] for factor in metric_factors.values()}
    if len(channel_counts) != 1:
        raise ValueError("every irrep block must use the same native channel count")
    channels = channel_counts.pop()
    metric = np.empty((channels, channels, len(ell_labels)), dtype=np.float64)
    for index, ell_value in enumerate(ell_labels):
        factor = np.asarray(metric_factors[int(ell_value)], dtype=np.float64)
        metric[:, :, index] = factor @ factor.T
    return metric


def native_slot_marginal(
    coefficients: np.ndarray,
    metric_by_angular: np.ndarray,
    slot: int,
    output_metric: np.ndarray | None = None,
    pair_chunk: int = 512,
) -> np.ndarray:
    """Contract every slot but one, retaining all channel cross terms.

    The result has axes ``[channel, angular, channel, angular]``. This is the
    direct specialization of the Dooms environment contraction before the
    upstream isometric factor is absorbed.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    order = coefficients.ndim - 1
    if slot < 0 or slot >= order:
        raise ValueError("slot is outside the coefficient order")
    channels = coefficients.shape[0]
    angular = coefficients.shape[1]
    if any(size != angular for size in coefficients.shape[1:]):
        raise ValueError("every repeated slot must use one angular dimension")
    if metric_by_angular.shape != (channels, channels, angular):
        raise ValueError("the radial metric does not match the coefficient axes")
    if output_metric is None:
        output_metric = np.ones((channels, channels), dtype=np.float64)
    output_metric = _validated_output_metric(output_metric, channels)

    moved = np.moveaxis(coefficients, 1 + slot, 1)
    moved_t = torch.as_tensor(moved, dtype=torch.float64)
    metric_t = torch.as_tensor(metric_by_angular, dtype=torch.float64)
    output_t = torch.as_tensor(output_metric, dtype=torch.float64)
    result = torch.empty(
        (channels, angular, channels, angular), dtype=torch.float64
    )
    left_indices = torch.arange(channels).repeat_interleave(channels)
    right_indices = torch.arange(channels).repeat(channels)
    for start in range(0, channels * channels, pair_chunk):
        left = left_indices[start : start + pair_chunk]
        right = right_indices[start : start + pair_chunk]
        lhs = moved_t[left].reshape(len(left), angular, -1)
        rhs = moved_t[right]
        output_weights = output_t[left, right]
        remaining_slots = order - 1
        if remaining_slots == 1:
            rhs = rhs * metric_t[left, right, :][:, None, :]
        elif remaining_slots == 2:
            one_slot_metric = metric_t[left, right, :]
            rhs = rhs * one_slot_metric[:, None, :, None]
            rhs = rhs * one_slot_metric[:, None, None, :]
        elif remaining_slots > 2:
            raise ValueError("only orders one through three are supported")
        rhs = rhs * output_weights.reshape(
            (len(left),) + (1,) * (rhs.ndim - 1)
        )
        rhs = rhs.reshape(len(left), angular, -1)
        blocks = torch.bmm(lhs, rhs.transpose(1, 2))
        result[left, :, right, :] = blocks
    return result.numpy()


def coefficient_norm_squared(
    coefficients: np.ndarray,
    metric_by_angular: np.ndarray,
    output_metric: np.ndarray | None = None,
    pair_chunk: int = 512,
) -> float:
    """Compute the exact coefficient norm with every path cross term."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    channels = coefficients.shape[0]
    order = coefficients.ndim - 1
    if output_metric is None:
        output_metric = np.ones((channels, channels), dtype=np.float64)
    output_metric = _validated_output_metric(output_metric, channels)
    coefficients_t = torch.as_tensor(coefficients, dtype=torch.float64)
    metric_t = torch.as_tensor(metric_by_angular, dtype=torch.float64)
    output_t = torch.as_tensor(output_metric, dtype=torch.float64)
    left_indices = torch.arange(channels).repeat_interleave(channels)
    right_indices = torch.arange(channels).repeat(channels)
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, channels * channels, pair_chunk):
        left = left_indices[start : start + pair_chunk]
        right = right_indices[start : start + pair_chunk]
        product = coefficients_t[left] * coefficients_t[right]
        product = product * output_t[left, right].reshape(
            (len(left),) + (1,) * order
        )
        one_slot_metric = metric_t[left, right]
        for axis in range(order):
            shape = [len(left)] + [1] * order
            shape[1 + axis] = coefficients.shape[1 + axis]
            product = product * one_slot_metric.reshape(shape)
        total = total + torch.sum(product)
    return float(total)


def projected_metric_by_angular(
    metric_factors: Mapping[int, np.ndarray],
    projectors: Mapping[int, np.ndarray],
    ell_labels: np.ndarray,
) -> np.ndarray:
    """Return the native overlap induced by canonical orthogonal projectors."""
    projected: dict[int, np.ndarray] = {}
    for ell, factor in metric_factors.items():
        projector = projectors[ell]
        if projector.shape != (factor.shape[1], factor.shape[1]):
            raise ValueError("a canonical projector has the wrong support dimension")
        projected[ell] = factor @ projector @ factor.T
    channels = next(iter(metric_factors.values())).shape[0]
    result = np.empty((channels, channels, len(ell_labels)), dtype=np.float64)
    for index, ell_value in enumerate(ell_labels):
        result[:, :, index] = projected[int(ell_value)]
    return result


def canonical_marginal_block(
    native_marginal: np.ndarray,
    left_index: int,
    right_index: int,
    left_factor: np.ndarray,
    right_factor: np.ndarray,
) -> np.ndarray:
    native_block = native_marginal[:, left_index, :, right_index]
    return left_factor.T @ native_block @ right_factor


@dataclass(frozen=True)
class EquivariantEnvironment:
    multiplicity_blocks: dict[int, np.ndarray]
    eigenvalues: dict[int, np.ndarray]
    eigenvectors: dict[int, np.ndarray]
    full_trace: float
    block_trace: float
    relative_off_block_norm: float
    relative_magnetic_deviation: float
    minimum_eigenvalue: float


def multiplicity_blocks_from_native(
    native_environment: np.ndarray,
    metric_factors: Mapping[int, np.ndarray],
    ell_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Trace magnetic diagonal blocks and divide by their dimensions."""
    blocks: dict[int, list[np.ndarray]] = {ell: [] for ell in metric_factors}
    for angular_index, ell_value in enumerate(ell_labels):
        ell = int(ell_value)
        blocks[ell].append(
            canonical_marginal_block(
                native_environment,
                angular_index,
                angular_index,
                metric_factors[ell],
                metric_factors[ell],
            )
        )
    return {ell: sum(items) / len(items) for ell, items in blocks.items()}


def canonical_environment_trace(
    native_environment: np.ndarray,
    metric_factors: Mapping[int, np.ndarray],
    ell_labels: np.ndarray,
) -> float:
    blocks = multiplicity_blocks_from_native(
        native_environment, metric_factors, ell_labels
    )
    return float(
        sum((2 * ell + 1) * np.trace(block) for ell, block in blocks.items())
    )


def extract_equivariant_environment(
    native_environment: np.ndarray,
    metric_factors: Mapping[int, np.ndarray],
    ell_labels: np.ndarray,
) -> EquivariantEnvironment:
    """Validate and extract Gamma_mult from Gamma_mult tensor identity."""
    diagonal_blocks: dict[int, list[np.ndarray]] = {
        ell: [] for ell in metric_factors
    }
    all_block_norm_squared = 0.0
    off_block_norm_squared = 0.0
    full_trace = 0.0
    for left_index, left_ell_value in enumerate(ell_labels):
        left_ell = int(left_ell_value)
        left_factor = metric_factors[left_ell]
        for right_index, right_ell_value in enumerate(ell_labels):
            right_ell = int(right_ell_value)
            block = canonical_marginal_block(
                native_environment,
                left_index,
                right_index,
                left_factor,
                metric_factors[right_ell],
            )
            norm_squared = float(np.linalg.norm(block) ** 2)
            all_block_norm_squared += norm_squared
            if left_index == right_index:
                diagonal_blocks[left_ell].append(block)
                full_trace += float(np.trace(block))
            else:
                off_block_norm_squared += norm_squared

    multiplicity_blocks = {
        ell: sum(blocks) / len(blocks) for ell, blocks in diagonal_blocks.items()
    }
    deviation_squared = 0.0
    diagonal_norm_squared = 0.0
    for ell, blocks in diagonal_blocks.items():
        mean = multiplicity_blocks[ell]
        for block in blocks:
            deviation_squared += float(np.linalg.norm(block - mean) ** 2)
            diagonal_norm_squared += float(np.linalg.norm(block) ** 2)
    eigenvalues: dict[int, np.ndarray] = {}
    eigenvectors: dict[int, np.ndarray] = {}
    minimum = np.inf
    for ell, block in multiplicity_blocks.items():
        values, vectors = symmetric_eigh_descending(block)
        eigenvalues[ell] = values
        eigenvectors[ell] = vectors
        minimum = min(minimum, float(np.linalg.eigvalsh(0.5 * (block + block.T))[0]))
    block_trace = float(
        sum((2 * ell + 1) * np.trace(block) for ell, block in multiplicity_blocks.items())
    )
    scale = max(all_block_norm_squared, np.finfo(np.float64).tiny)
    diagonal_scale = max(diagonal_norm_squared, np.finfo(np.float64).tiny)
    return EquivariantEnvironment(
        multiplicity_blocks=multiplicity_blocks,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        full_trace=full_trace,
        block_trace=block_trace,
        relative_off_block_norm=float(np.sqrt(off_block_norm_squared / scale)),
        relative_magnetic_deviation=float(
            np.sqrt(deviation_squared / diagonal_scale)
        ),
        minimum_eigenvalue=minimum,
    )


def dense_canonical_kernel(
    coefficients: np.ndarray,
    metric_factors: Mapping[int, np.ndarray],
    ell_labels: np.ndarray,
) -> np.ndarray:
    """Materialize a small canonical kernel for independent test oracles."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    order = coefficients.ndim - 1
    support_dimensions = {factor.shape[1] for factor in metric_factors.values()}
    if len(support_dimensions) != 1:
        raise ValueError("the dense oracle requires equal support dimensions")
    support = support_dimensions.pop()
    angular = len(ell_labels)
    primitive = support * angular
    kernel = np.zeros((primitive,) * order, dtype=np.float64)
    for angular_indices in itertools.product(range(angular), repeat=order):
        canonical_values = np.zeros((support,) * order, dtype=np.float64)
        for native_channel in range(coefficients.shape[0]):
            term: np.ndarray | float = float(
                coefficients[(native_channel,) + angular_indices]
            )
            for angular_index in angular_indices:
                ell = int(ell_labels[angular_index])
                term = np.multiply.outer(
                    term, metric_factors[ell][native_channel]
                )
            canonical_values += term
        destination = tuple(
            slice(angular_index * support, (angular_index + 1) * support)
            for angular_index in angular_indices
        )
        kernel[destination] = canonical_values
    return kernel
