"""Shared-interface product wrapper for full-model projector tests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping

import numpy as np
import torch

from mace_odt.functional_projector import NativeFunctionalMap


class ProjectedProductBlock(torch.nn.Module):
    """Apply central-species functional maps before one native product block."""

    def __init__(
        self,
        native: Any,
        maps: Mapping[tuple[int, int], NativeFunctionalMap],
        num_elements: int,
        angular_slices: Mapping[int, slice] | None = None,
    ) -> None:
        super().__init__()
        self.native = native
        self.num_elements = num_elements
        self.angular_slices = dict(
            angular_slices
            if angular_slices is not None
            else {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 9), 3: slice(9, 16)}
        )
        if set(self.angular_slices) != {ell for _, ell in maps}:
            raise ValueError("angular slices do not match the supplied functional maps")
        ordered_sections = sorted(
            self.angular_slices.values(), key=lambda section: int(section.start or 0)
        )
        expected_start = 0
        for section in ordered_sections:
            if section.step not in (None, 1):
                raise ValueError("angular slices must have unit stride")
            if section.start != expected_start or section.stop is None:
                raise ValueError("angular slices must be ordered, contiguous, and start at zero")
            expected_start = section.stop
        self.angular_dimension = expected_start
        for ell in self.angular_slices:
            encoders = np.stack(
                [maps[(central, ell)].encoder for central in range(num_elements)],
                axis=0,
            )
            decoders = np.stack(
                [maps[(central, ell)].decoder for central in range(num_elements)],
                axis=0,
            )
            self.register_buffer(
                f"encoders_{ell}", torch.as_tensor(encoders, dtype=torch.float64)
            )
            self.register_buffer(
                f"decoders_{ell}", torch.as_tensor(decoders, dtype=torch.float64)
            )

    def project(
        self, node_features: torch.Tensor, node_attrs: torch.Tensor
    ) -> torch.Tensor:
        if node_features.shape[-1] != self.angular_dimension:
            raise ValueError("node feature angular dimension does not match the slices")
        result = torch.empty_like(node_features)
        for ell, section in self.angular_slices.items():
            encoders = getattr(self, f"encoders_{ell}").to(
                dtype=node_features.dtype, device=node_features.device
            )
            decoders = getattr(self, f"decoders_{ell}").to(
                dtype=node_features.dtype, device=node_features.device
            )
            node_encoders = torch.einsum("nz,zrc->nrc", node_attrs, encoders)
            node_decoders = torch.einsum("nz,zcr->ncr", node_attrs, decoders)
            encoded = torch.einsum(
                "nrc,ncm->nrm", node_encoders, node_features[:, :, section]
            )
            result[:, :, section] = torch.einsum(
                "ncr,nrm->ncm", node_decoders, encoded
            )
        return result

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor | None,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.project(node_feats, node_attrs)
        return self.native(projected, sc, node_attrs)


@contextmanager
def projected_product_context(model: Any, replacement: torch.nn.Module):
    """Install product zero temporarily and restore it after success or failure."""
    native = model.products[0]
    model.products[0] = replacement
    try:
        yield
    finally:
        model.products[0] = native
