"""Native encoders and decoders for canonical functional subspaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


def supported_left_inverse(factor: np.ndarray) -> np.ndarray:
    """Return the inverse on the numerically declared column support."""
    factor = np.asarray(factor, dtype=np.float64)
    left_vectors, singular_values, right_vectors_t = np.linalg.svd(
        factor, full_matrices=False
    )
    tolerance = (
        float(singular_values[0])
        * max(factor.shape)
        * np.finfo(factor.dtype).eps
    )
    if np.any(singular_values <= tolerance):
        raise ValueError("the supplied factor contains columns outside its support")
    return (right_vectors_t.T / singular_values[None, :]) @ left_vectors.T


@dataclass(frozen=True)
class NativeFunctionalMap:
    encoder: np.ndarray
    decoder: np.ndarray
    projector: np.ndarray
    canonical_projector: np.ndarray
    retained_rank: int
    support_rank: int
    left_inverse_residual: float
    inverse_pair_residual: float
    idempotence_residual: float
    relative_idempotence_residual: float


def native_functional_map(
    metric_factor: np.ndarray, canonical_modes: np.ndarray
) -> NativeFunctionalMap:
    """Construct D, B, and P=BD from Equation 9 of the specification."""
    factor = np.asarray(metric_factor, dtype=np.float64)
    modes = np.asarray(canonical_modes, dtype=np.float64)
    if modes.ndim != 2 or modes.shape[0] != factor.shape[1]:
        raise ValueError("canonical modes do not match the metric support")
    orthogonality = np.linalg.norm(modes.T @ modes - np.eye(modes.shape[1]))
    if orthogonality > 1e-9:
        raise ValueError("canonical modes must have orthonormal columns")
    left_inverse = supported_left_inverse(factor)
    decoder = factor @ modes
    encoder = modes.T @ left_inverse
    projector = decoder @ encoder
    canonical_projector = modes @ modes.T
    support_identity = np.eye(factor.shape[1])
    retained_identity = np.eye(modes.shape[1])
    idempotence_residual = float(np.linalg.norm(projector @ projector - projector))
    return NativeFunctionalMap(
        encoder=encoder,
        decoder=decoder,
        projector=projector,
        canonical_projector=canonical_projector,
        retained_rank=modes.shape[1],
        support_rank=factor.shape[1],
        left_inverse_residual=float(
            np.linalg.norm(left_inverse @ factor - support_identity)
        ),
        inverse_pair_residual=float(
            np.linalg.norm(encoder @ decoder - retained_identity)
        ),
        idempotence_residual=idempotence_residual,
        relative_idempotence_residual=(
            idempotence_residual / max(float(np.linalg.norm(projector)), 1.0)
        ),
    )


def apply_species_equivariant_projectors(
    density: torch.Tensor,
    node_attrs: torch.Tensor,
    angular_slices: Mapping[int, slice],
    projectors: Mapping[tuple[int, int], np.ndarray],
) -> torch.Tensor:
    """Apply one central-species channel map to every full angular multiplet."""
    result = torch.empty_like(density)
    num_elements = node_attrs.shape[1]
    for ell, section in angular_slices.items():
        matrices = np.stack(
            [projectors[(central, ell)] for central in range(num_elements)], axis=0
        )
        matrices_t = torch.as_tensor(
            matrices, dtype=density.dtype, device=density.device
        )
        node_matrices = torch.einsum("nz,zcd->ncd", node_attrs, matrices_t)
        result[:, :, section] = torch.einsum(
            "ncd,ndm->ncm", node_matrices, density[:, :, section]
        )
    return result


def apply_species_functional_maps(
    density: torch.Tensor,
    node_attrs: torch.Tensor,
    angular_slices: Mapping[int, slice],
    maps: Mapping[tuple[int, int], NativeFunctionalMap],
) -> torch.Tensor:
    """Apply native encoder and decoder pairs without forming their product."""
    result = density.clone()
    num_elements = node_attrs.shape[1]
    for ell, section in angular_slices.items():
        encoders = np.stack(
            [maps[(central, ell)].encoder for central in range(num_elements)], axis=0
        )
        decoders = np.stack(
            [maps[(central, ell)].decoder for central in range(num_elements)], axis=0
        )
        encoders_t = torch.as_tensor(
            encoders, dtype=density.dtype, device=density.device
        )
        decoders_t = torch.as_tensor(
            decoders, dtype=density.dtype, device=density.device
        )
        node_encoders = torch.einsum("nz,zrc->nrc", node_attrs, encoders_t)
        node_decoders = torch.einsum("nz,zcr->ncr", node_attrs, decoders_t)
        encoded = torch.einsum(
            "nrc,ncm->nrm", node_encoders, density[:, :, section]
        )
        result[:, :, section] = torch.einsum(
            "ncr,nrm->ncm", node_decoders, encoded
        )
    return result
