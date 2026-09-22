"""Registered first-layer radial interface for MACE-OFF23."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class AngularBlock:
    ell: int
    start: int
    stop: int

    @property
    def dimension(self) -> int:
        return self.stop - self.start


def angular_blocks(model: Any) -> list[AngularBlock]:
    blocks: list[AngularBlock] = []
    start = 0
    for multiplicity, irrep in model.spherical_harmonics.irreps_out:
        if multiplicity != 1:
            raise ValueError("the registered spherical harmonic basis must have multiplicity one")
        stop = start + irrep.dim
        blocks.append(AngularBlock(ell=int(irrep.l), start=start, stop=stop))
        start = stop
    interaction_dimension = int(model.interactions[0].irreps_out.dim)
    channels = int(model.interactions[0].irreps_out[0].mul)
    if interaction_dimension != channels * start:
        raise ValueError("interaction and angular layouts do not match")
    return blocks


def reference_direction(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    direction = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
    return direction / direction.norm()


def pair_radial_coefficients(
    model: Any,
    central_indices: torch.Tensor,
    neighbor_indices: torch.Tensor,
    radii: torch.Tensor,
) -> torch.Tensor:
    """Return F[z,z',r,ell,channel] for independent directed edges.

    Each row describes one source neighbor and one target central atom. The
    returned tensor has shape ``[n_pairs, n_ell, n_channels]``.
    """
    if central_indices.ndim != 1 or neighbor_indices.ndim != 1:
        raise ValueError("species index arrays must be one-dimensional")
    if radii.ndim == 2 and radii.shape[1] == 1:
        radii = radii[:, 0]
    if radii.ndim != 1:
        raise ValueError("radii must have shape [n_pairs] or [n_pairs, 1]")
    if not (len(central_indices) == len(neighbor_indices) == len(radii)):
        raise ValueError("pair arrays must have equal lengths")

    pair_count = len(radii)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    central_indices = central_indices.to(device=device, dtype=torch.long)
    neighbor_indices = neighbor_indices.to(device=device, dtype=torch.long)
    radii = radii.to(device=device, dtype=dtype)
    num_elements = int(model.atomic_numbers.numel())

    node_attrs = torch.zeros(
        (2 * pair_count, num_elements), dtype=dtype, device=device
    )
    row = torch.arange(pair_count, device=device)
    node_attrs[row, central_indices] = 1.0
    node_attrs[pair_count + row, neighbor_indices] = 1.0
    edge_index = torch.stack((pair_count + row, row), dim=0)
    directions = reference_direction(dtype, device).expand(pair_count, 3)

    node_features = model.node_embedding(node_attrs)
    edge_attrs = model.spherical_harmonics(directions)
    edge_features, cutoff = model.radial_embedding(
        radii[:, None], node_attrs, edge_index, model.atomic_numbers
    )
    density, _ = model.interactions[0](
        node_attrs=node_attrs,
        node_feats=node_features,
        edge_attrs=edge_attrs,
        edge_feats=edge_features,
        edge_index=edge_index,
        cutoff=cutoff,
        first_layer=True,
    )
    density = density[:pair_count]

    coefficients = []
    for block in angular_blocks(model):
        values = density[:, :, block.start : block.stop]
        harmonics = edge_attrs[:, block.start : block.stop]
        denominator = torch.sum(harmonics * harmonics, dim=1)
        coefficient = torch.einsum("ncm,nm->nc", values, harmonics)
        coefficient = coefficient / denominator[:, None]
        coefficients.append(coefficient)
    return torch.stack(coefficients, dim=1)


def reconstruct_density_from_pairs(
    model: Any,
    node_attrs: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attrs: torch.Tensor,
    edge_lengths: torch.Tensor,
) -> torch.Tensor:
    source = edge_index[0]
    target = edge_index[1]
    central_indices = node_attrs[target].argmax(dim=-1)
    neighbor_indices = node_attrs[source].argmax(dim=-1)
    coefficients = pair_radial_coefficients(
        model, central_indices, neighbor_indices, edge_lengths
    )
    channels = coefficients.shape[-1]
    angular_dimension = edge_attrs.shape[-1]
    edge_density = torch.zeros(
        (edge_attrs.shape[0], channels, angular_dimension),
        dtype=edge_attrs.dtype,
        device=edge_attrs.device,
    )
    for block_index, block in enumerate(angular_blocks(model)):
        edge_density[:, :, block.start : block.stop] = torch.einsum(
            "ec,em->ecm",
            coefficients[:, block_index, :],
            edge_attrs[:, block.start : block.stop],
        )
    density = torch.zeros(
        (node_attrs.shape[0], channels, angular_dimension),
        dtype=edge_density.dtype,
        device=edge_density.device,
    )
    density.index_add_(0, target, edge_density)
    return density


def angular_factorization_check(
    model: Any,
    central_index: int,
    neighbor_index: int,
    radius: float,
    direction_count: int = 32,
    seed: int = 20260922,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    directions_np = rng.normal(size=(direction_count, 3))
    directions_np /= np.linalg.norm(directions_np, axis=1, keepdims=True)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    directions = torch.as_tensor(directions_np, dtype=dtype, device=device)
    num_elements = int(model.atomic_numbers.numel())
    pair_count = direction_count
    attrs = torch.zeros((2 * pair_count, num_elements), dtype=dtype, device=device)
    attrs[:pair_count, central_index] = 1.0
    attrs[pair_count:, neighbor_index] = 1.0
    rows = torch.arange(pair_count, device=device)
    edge_index = torch.stack((pair_count + rows, rows), dim=0)
    radii = torch.full((pair_count, 1), radius, dtype=dtype, device=device)
    node_features = model.node_embedding(attrs)
    edge_attrs = model.spherical_harmonics(directions)
    edge_features, cutoff = model.radial_embedding(
        radii, attrs, edge_index, model.atomic_numbers
    )
    density, _ = model.interactions[0](
        node_attrs=attrs,
        node_feats=node_features,
        edge_attrs=edge_attrs,
        edge_feats=edge_features,
        edge_index=edge_index,
        cutoff=cutoff,
        first_layer=True,
    )
    density = density[:pair_count]
    records = []
    for block in angular_blocks(model):
        values = density[:, :, block.start : block.stop]
        harmonics = edge_attrs[:, block.start : block.stop]
        coefficient = torch.einsum("ncm,nm->c", values, harmonics)
        coefficient = coefficient / torch.sum(harmonics * harmonics)
        reconstructed = torch.einsum("c,nm->ncm", coefficient, harmonics)
        difference = values - reconstructed
        records.append(
            {
                "ell": block.ell,
                "max_absolute_error": float(difference.abs().max().detach().cpu()),
                "relative_l2_error": float(
                    difference.norm().detach().cpu()
                    / torch.clamp(values.norm().detach().cpu(), min=1.0)
                ),
            }
        )
    return {"records": records, "seed": seed, "direction_count": direction_count}


def supported_metric_factor(
    weighted_feature_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Factor a weighted function table directly without forming its Gram."""
    _, singular_values, right_vectors = np.linalg.svd(
        weighted_feature_matrix, full_matrices=False
    )
    tolerance = (
        float(singular_values[0])
        * max(weighted_feature_matrix.shape)
        * np.finfo(weighted_feature_matrix.dtype).eps
    )
    rank = int(np.sum(singular_values > tolerance))
    factor = right_vectors[:rank].T * singular_values[:rank][None, :]
    return factor, singular_values, rank, tolerance
