"""Exact reduced-path evaluators for MACE scalar symmetric contractions."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch

from mace_odt.path_quotient import compute_path_quotient


def torch_symmetrize_repeated_slots(tensor: torch.Tensor, order: int) -> torch.Tensor:
    leading = tensor.ndim - order - 1
    if leading < 0:
        raise ValueError("tensor has too few axes for the declared order")
    repeated_axes = tuple(range(leading, leading + order))
    prefix = tuple(range(leading))
    path_axis = tensor.ndim - 1
    permutations = list(itertools.permutations(repeated_axes))
    return sum(
        tensor.permute(prefix + permutation + (path_axis,))
        for permutation in permutations
    ) / len(permutations)


def native_weight_for_order(contraction: Any, order: int) -> torch.Tensor:
    if order == contraction.correlation:
        return contraction.weights_max
    index = contraction.correlation - order - 1
    if index < 0 or index >= len(contraction.weights):
        raise IndexError(f"no native weights for order {order}")
    return contraction.weights[index]


def scalar_order_response(
    coupling: torch.Tensor,
    weights: torch.Tensor,
    features: torch.Tensor,
    element_one_hot: torch.Tensor,
) -> torch.Tensor:
    order = coupling.ndim - 1
    if order == 1:
        return torch.einsum(
            "ip,epc,bci,be->bc", coupling, weights, features, element_one_hot
        )
    if order == 2:
        return torch.einsum(
            "ijp,epc,bci,bcj,be->bc",
            coupling,
            weights,
            features,
            features,
            element_one_hot,
        )
    if order == 3:
        return torch.einsum(
            "ijkp,epc,bci,bcj,bck,be->bc",
            coupling,
            weights,
            features,
            features,
            features,
            element_one_hot,
        )
    raise ValueError("only scalar orders one through three are supported")


class QuotientedScalarSymmetricContraction(torch.nn.Module):
    """Drop only the exact path kernel while preserving the polynomial."""

    def __init__(self, native: Any) -> None:
        super().__init__()
        if len(native.contractions) != 1:
            raise ValueError("the current exact adapter requires one scalar output irrep")
        contraction = native.contractions[0]
        output_irrep = native.irreps_out[0].ir
        if output_irrep.l != 0:
            raise ValueError("the current exact adapter requires a scalar output")
        if contraction.correlation > 3:
            raise ValueError("the current exact adapter supports correlation at most three")

        self.correlation = int(contraction.correlation)
        self.num_features = int(contraction.num_features)
        self.irreps_in = native.irreps_in
        self.irreps_out = native.irreps_out
        self.quotient_metadata: list[dict[str, Any]] = []

        for order in range(1, self.correlation + 1):
            native_coupling = contraction.U_tensors(order).detach()
            symmetrized = torch_symmetrize_repeated_slots(native_coupling, order)
            quotient = compute_path_quotient(
                symmetrized.cpu().numpy(), order=order
            )
            support = torch.as_tensor(
                quotient.support_basis,
                dtype=symmetrized.dtype,
                device=symmetrized.device,
            )
            reduced_coupling = torch.tensordot(
                symmetrized, support, dims=([-1], [0])
            )
            native_weights = native_weight_for_order(contraction, order).detach()
            reduced_weights = torch.einsum(
                "pr,epc->erc", support, native_weights
            )
            self.register_buffer(f"coupling_{order}", reduced_coupling.contiguous())
            self.register_buffer(f"weights_{order}", reduced_weights.contiguous())
            self.quotient_metadata.append(
                {
                    "order": order,
                    "stored_dimension": quotient.stored_dimension,
                    "supported_dimension": quotient.supported_dimension,
                    "null_dimension": quotient.null_dimension,
                    "null_residual": quotient.null_residual,
                }
            )

    def forward(
        self, features: torch.Tensor, element_one_hot: torch.Tensor
    ) -> torch.Tensor:
        result = torch.zeros(
            (features.shape[0], features.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
        for order in range(1, self.correlation + 1):
            coupling = getattr(self, f"coupling_{order}")
            weights = getattr(self, f"weights_{order}")
            result = result + scalar_order_response(
                coupling, weights, features, element_one_hot
            )
        return result


def quotient_random_equivalence(
    native: Any,
    reduced: QuotientedScalarSymmetricContraction,
    seed: int = 20260922,
) -> dict[str, float]:
    contraction = native.contractions[0]
    rng = np.random.default_rng(seed)
    batch_size = 11
    input_dimension = int(contraction.U_tensors(1).shape[0])
    num_elements = int(contraction.weights_max.shape[0])
    features = torch.as_tensor(
        rng.normal(size=(batch_size, contraction.num_features, input_dimension)),
        dtype=contraction.weights_max.dtype,
        device=contraction.weights_max.device,
    )
    indices = torch.arange(batch_size, device=features.device) % num_elements
    elements = torch.nn.functional.one_hot(indices, num_classes=num_elements).to(
        dtype=features.dtype
    )
    with torch.no_grad():
        expected = native(features, elements)
        observed = reduced(features, elements)
    difference = observed - expected
    return {
        "max_absolute_error": float(difference.abs().max().cpu()),
        "relative_l2_error": float(
            difference.norm().cpu() / torch.clamp(expected.norm().cpu(), min=1.0)
        ),
    }
