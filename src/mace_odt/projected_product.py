"""Shared-interface product wrapper for full-model projector tests."""

from __future__ import annotations

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
    ) -> None:
        super().__init__()
        self.native = native
        self.num_elements = num_elements
        self.angular_slices = {
            0: slice(0, 1),
            1: slice(1, 4),
            2: slice(4, 9),
            3: slice(9, 16),
        }
        for ell in range(4):
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
        blocks = []
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
            blocks.append(torch.einsum("ncr,nrm->ncm", node_decoders, encoded))
        return torch.cat(blocks, dim=-1)

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: torch.Tensor | None,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.project(node_feats, node_attrs)
        return self.native(projected, sc, node_attrs)
