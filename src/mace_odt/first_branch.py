"""Exact compiler for the bounded polynomial first energy branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from mace_odt.quotient_module import (
    native_weight_for_order,
    torch_symmetrize_repeated_slots,
)


@dataclass(frozen=True)
class FirstBranchCoefficients:
    """Composed scalar coefficients in the native density coordinates.

    ``orders[nu]`` has axes ``[central_species, channel, angular]^nu``, with
    one channel axis shared by all repeated angular slots. The downstream
    post-product map, scalar readout, and checkpoint scale are included.
    """

    orders: Mapping[int, torch.Tensor]
    constant: torch.Tensor
    downstream_channel_weights: torch.Tensor
    angular_dimension: int
    channels: int
    num_elements: int


def _checkpoint_scale(model: Any) -> torch.Tensor:
    scale = model.scale_shift.scale
    if scale.numel() != 1:
        raise ValueError("the first-branch compiler currently requires one model head")
    return scale.reshape(())


def native_first_branch_node_energy(
    model: Any,
    density: torch.Tensor,
    node_attrs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the checkpoint's first scalar branch, including its scale."""
    product = model.products[0]
    features = product(density, None, node_attrs)
    node_energy = model.readouts[0](features).squeeze(-1)
    if node_energy.ndim != 1:
        raise ValueError("the first readout must return one scalar per node")
    return _checkpoint_scale(model) * node_energy


def compile_first_branch(model: Any) -> FirstBranchCoefficients:
    """Compose the first MACE product and readout into symmetric coefficients."""
    product = model.products[0]
    native = product.symmetric_contractions
    if len(native.contractions) != 1:
        raise ValueError("the registered branch requires one scalar output irrep")
    contraction = native.contractions[0]
    correlation = int(contraction.correlation)
    if correlation != 3:
        raise ValueError("the registered MACE-OFF23 branch must have correlation three")

    channels = int(contraction.num_features)
    num_elements = int(contraction.weights_max.shape[0])
    dtype = contraction.weights_max.dtype
    device = contraction.weights_max.device
    basis = torch.eye(channels, dtype=dtype, device=device)
    downstream = model.readouts[0](product.linear(basis)).squeeze(-1)
    if downstream.shape != (channels,):
        raise ValueError("the post-product branch does not preserve the channel layout")
    downstream = downstream * _checkpoint_scale(model)

    orders: dict[int, torch.Tensor] = {}
    angular_dimension: int | None = None
    for order in range(1, correlation + 1):
        coupling = contraction.U_tensors(order).detach()
        coupling = torch_symmetrize_repeated_slots(coupling, order)
        if angular_dimension is None:
            angular_dimension = int(coupling.shape[0])
        if any(int(size) != angular_dimension for size in coupling.shape[:-1]):
            raise ValueError("every repeated primitive slot must use one angular space")
        weights = native_weight_for_order(contraction, order).detach()
        if weights.shape[0] != num_elements or weights.shape[-1] != channels:
            raise ValueError("path weight axes do not match the registered branch")
        coefficients = compose_order_coefficients(coupling, weights, downstream)
        orders[order] = coefficients.contiguous()

    if angular_dimension is None:
        raise ValueError("the branch has no polynomial orders")
    zero_density = torch.zeros(
        (num_elements, channels, angular_dimension), dtype=dtype, device=device
    )
    element_basis = torch.eye(num_elements, dtype=dtype, device=device)
    constant = native_first_branch_node_energy(
        model, zero_density, element_basis
    ).detach()
    return FirstBranchCoefficients(
        orders=orders,
        constant=constant,
        downstream_channel_weights=downstream.detach(),
        angular_dimension=angular_dimension,
        channels=channels,
        num_elements=num_elements,
    )


def compose_order_coefficients(
    symmetric_coupling: torch.Tensor,
    path_weights: torch.Tensor,
    downstream_channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Contract every path coefficient into one scalar branch tensor."""
    order = symmetric_coupling.ndim - 1
    if order == 1:
        return torch.einsum(
            "ip,zpc,c->zci",
            symmetric_coupling,
            path_weights,
            downstream_channel_weights,
        )
    if order == 2:
        return torch.einsum(
            "ijp,zpc,c->zcij",
            symmetric_coupling,
            path_weights,
            downstream_channel_weights,
        )
    if order == 3:
        return torch.einsum(
            "ijkp,zpc,c->zcijk",
            symmetric_coupling,
            path_weights,
            downstream_channel_weights,
        )
    raise ValueError("only correlation orders one through three are supported")


def evaluate_first_branch_coefficients(
    coefficients: FirstBranchCoefficients,
    density: torch.Tensor,
    node_attrs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the exact composed coefficients on aggregate density features."""
    if density.ndim != 3:
        raise ValueError("density must have axes [node, channel, angular]")
    if density.shape[1:] != (
        coefficients.channels,
        coefficients.angular_dimension,
    ):
        raise ValueError("density axes do not match the compiled coefficients")
    if node_attrs.shape != (density.shape[0], coefficients.num_elements):
        raise ValueError("node attributes do not match density or element count")

    result = node_attrs @ coefficients.constant
    result = result + torch.einsum(
        "zci,nci,nz->n", coefficients.orders[1], density, node_attrs
    )
    result = result + torch.einsum(
        "zcij,nci,ncj,nz->n",
        coefficients.orders[2],
        density,
        density,
        node_attrs,
    )
    result = result + torch.einsum(
        "zcijk,nci,ncj,nck,nz->n",
        coefficients.orders[3],
        density,
        density,
        density,
        node_attrs,
    )
    return result


def coefficient_symmetry_residual(tensor: torch.Tensor) -> float:
    """Return the largest repeated-slot permutation residual."""
    order = tensor.ndim - 2
    if order == 1:
        return 0.0
    angular_axes = tuple(range(2, tensor.ndim))
    residual = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    import itertools

    for permutation in itertools.permutations(angular_axes):
        candidate = tensor.permute((0, 1) + permutation)
        residual = torch.maximum(residual, (tensor - candidate).abs().max())
    return float(residual.detach().cpu())
