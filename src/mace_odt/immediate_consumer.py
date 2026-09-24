"""Exact matrix-free action at the first MACE product and its consumers.

The MACE-OFF23 small product is a correlation-three polynomial on a
``[node, channel, angular]`` density.  Materializing that vector-valued
polynomial, or the following equivariant convolution, would require tensors
with billions of entries.  This module therefore binds the frozen native
modules and exposes their exact forward and reverse actions without replacing
any learned operation.

The shared product output has three immediate uses: the first linear readout,
the second interaction's message path, and that interaction's residual path.
The returned message density and skip tensor keep those latter paths separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping, NamedTuple

import numpy as np
import torch


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ImmediateConsumerShape:
    """Declared native tensor dimensions for the bounded graph fragment."""

    channels: int
    angular_dimension: int
    num_elements: int
    edge_attribute_dimension: int
    edge_feature_dimension: int
    readout_dimension: int = 1

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ImmediateConsumerProvenance:
    """Checkpoint and module binding recorded when the action is compiled."""

    checkpoint_sha256: str
    module_sha256: str
    shape: ImmediateConsumerShape

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.checkpoint_sha256) is None:
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        if _SHA256.fullmatch(self.module_sha256) is None:
            raise ValueError("module_sha256 must be a lowercase SHA-256 digest")


class ImmediateConsumerOutput(NamedTuple):
    """Exact values at the product-zero fan-out."""

    product0: torch.Tensor
    readout0: torch.Tensor
    interaction1_density: torch.Tensor
    interaction1_skip: torch.Tensor


class ImmediateConsumerCotangent(NamedTuple):
    """Cotangents for the three immediate consumer outputs."""

    readout0: torch.Tensor
    interaction1_density: torch.Tensor
    interaction1_skip: torch.Tensor


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _module_sha256(
    product0: torch.nn.Module,
    readout0: torch.nn.Module,
    interaction1: torch.nn.Module,
    shape: ImmediateConsumerShape,
    readout_scale: torch.Tensor,
) -> str:
    """Hash all frozen arrays plus architecture fields used by this adapter."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(asdict(shape), sort_keys=True, separators=(",", ":")).encode()
    )
    scale = readout_scale.detach().cpu().contiguous()
    digest.update(str(scale.dtype).encode())
    digest.update(scale.numpy().tobytes())
    for label, module in (
        ("product0", product0),
        ("readout0", readout0),
        ("interaction1", interaction1),
    ):
        digest.update(label.encode())
        digest.update(_qualified_name(module).encode())
        for attribute in (
            "irreps_in",
            "irreps_out",
            "node_feats_irreps",
            "target_irreps",
            "edge_attrs_irreps",
            "edge_feats_irreps",
            "avg_num_neighbors",
            "use_sc",
        ):
            if hasattr(module, attribute):
                digest.update(attribute.encode())
                digest.update(repr(getattr(module, attribute)).encode())
        for name, tensor in module.state_dict().items():
            value = tensor.detach().cpu().resolve_conj().resolve_neg().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _dimension(value: Any, name: str) -> int:
    try:
        result = int(value.dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"cannot infer {name} dimension") from exc
    if result < 1:
        raise ValueError(f"{name} dimension must be positive")
    return result


def _floating_reference(module: torch.nn.Module) -> torch.Tensor | None:
    for value in module.parameters():
        if value.is_floating_point():
            return value
    for value in module.buffers():
        if value.is_floating_point():
            return value
    return None


class ImmediateConsumerAction(torch.nn.Module):
    """Frozen exact action from ``rho0`` through all consumers of ``product0``.

    This wrapper does not copy, approximate, or retrain the bound modules.  A
    provenance check hashes them on demand and rejects any mutation relative to
    the checkpoint-bound construction.
    """

    def __init__(
        self,
        product0: torch.nn.Module,
        readout0: torch.nn.Module,
        interaction1: torch.nn.Module,
        shape: ImmediateConsumerShape,
        checkpoint_sha256: str,
        readout_scale: torch.Tensor | float = 1.0,
    ) -> None:
        super().__init__()
        if not all(
            isinstance(module, torch.nn.Module)
            for module in (product0, readout0, interaction1)
        ):
            raise TypeError("product0, readout0, and interaction1 must be modules")
        if _SHA256.fullmatch(checkpoint_sha256) is None:
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
        self.product0 = product0
        self.readout0 = readout0
        self.interaction1 = interaction1
        self.shape = shape
        reference = _floating_reference(product0)
        if reference is None:
            raise TypeError("product0 must expose floating parameters or buffers")
        scale = torch.as_tensor(
            readout_scale, dtype=reference.dtype, device=reference.device
        )
        if scale.numel() != 1 or not torch.isfinite(scale).all():
            raise ValueError("readout_scale must be one finite scalar")
        self.register_buffer("readout_scale", scale.reshape(()).detach().clone())
        self.provenance = ImmediateConsumerProvenance(
            checkpoint_sha256=checkpoint_sha256,
            module_sha256=_module_sha256(
                product0, readout0, interaction1, shape, self.readout_scale
            ),
            shape=shape,
        )

    def validate_provenance(self, checkpoint_sha256: str) -> None:
        """Reject a different checkpoint digest or any changed bound array."""
        if checkpoint_sha256 != self.provenance.checkpoint_sha256:
            raise ValueError("checkpoint provenance mismatch")
        observed = _module_sha256(
            self.product0,
            self.readout0,
            self.interaction1,
            self.shape,
            self.readout_scale,
        )
        if observed != self.provenance.module_sha256:
            raise ValueError("bound immediate-consumer modules have changed")

    def _validate_common(
        self,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
        nodes: int,
        reference: torch.Tensor,
    ) -> None:
        shape = self.shape
        if node_attrs.shape != (nodes, shape.num_elements):
            raise ValueError("node_attrs shape does not match the compiled fragment")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge]")
        edges = int(edge_index.shape[1])
        if edge_attrs.shape != (edges, shape.edge_attribute_dimension):
            raise ValueError("edge_attrs shape does not match the compiled fragment")
        if edge_feats.shape != (edges, shape.edge_feature_dimension):
            raise ValueError("edge_feats shape does not match the compiled fragment")
        if edge_index.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("edge_index must be integral")
        if edges and (
            int(edge_index.min().detach().cpu()) < 0
            or int(edge_index.max().detach().cpu()) >= nodes
        ):
            raise ValueError("edge_index contains an invalid node")
        floating = (node_attrs, edge_attrs, edge_feats)
        if cutoff is not None:
            if (
                cutoff.ndim not in (1, 2)
                or cutoff.shape[0] != edges
                or (cutoff.ndim == 2 and cutoff.shape[1] != 1)
            ):
                raise ValueError("cutoff must have one value or row per edge")
            floating = floating + (cutoff,)
        for value in floating:
            if not value.is_floating_point():
                raise TypeError("all feature inputs must be floating point")
            if value.dtype != reference.dtype or value.device != reference.device:
                raise TypeError("all fragment inputs must share dtype and device")
        if edge_index.device != reference.device:
            raise TypeError("edge_index must be on the feature device")
        for module in (self.product0, self.readout0, self.interaction1):
            parameter = _floating_reference(module)
            if parameter is not None and (
                parameter.dtype != reference.dtype or parameter.device != reference.device
            ):
                raise TypeError("bound modules and fragment inputs differ in dtype or device")

    def _validate_product(self, product0: torch.Tensor, nodes: int) -> None:
        if product0.shape != (nodes, self.shape.channels):
            raise ValueError("product0 output does not have the declared channel shape")

    def forward_from_product(
        self,
        product0: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> ImmediateConsumerOutput:
        """Apply the three exact consumers to an already computed product output."""
        if product0.ndim != 2 or not product0.is_floating_point():
            raise ValueError("product0 must be a floating [node, channel] tensor")
        nodes = int(product0.shape[0])
        self._validate_product(product0, nodes)
        self._validate_common(
            node_attrs,
            edge_attrs,
            edge_feats,
            edge_index,
            cutoff,
            nodes,
            product0,
        )
        readout = self.readout0(product0)
        density, skip = self.interaction1(
            node_attrs=node_attrs,
            node_feats=product0,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=edge_index,
            cutoff=cutoff,
            first_layer=False,
        )
        if readout.shape != (nodes, self.shape.readout_dimension):
            raise ValueError("readout0 output does not match the declared shape")
        if density.shape != (
            nodes,
            self.shape.channels,
            self.shape.angular_dimension,
        ):
            raise ValueError("interaction1 density does not match the declared shape")
        if skip is None or skip.shape != (nodes, self.shape.channels):
            raise ValueError("interaction1 skip does not match the declared shape")
        return ImmediateConsumerOutput(product0, readout, density, skip)

    def forward(
        self,
        density0: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
    ) -> ImmediateConsumerOutput:
        """Evaluate native product0 and all three of its immediate consumers."""
        if density0.ndim != 3 or not density0.is_floating_point():
            raise ValueError("density0 must be a floating [node, channel, angular] tensor")
        nodes = int(density0.shape[0])
        if density0.shape[1:] != (
            self.shape.channels,
            self.shape.angular_dimension,
        ):
            raise ValueError("density0 shape does not match the compiled fragment")
        self._validate_common(
            node_attrs,
            edge_attrs,
            edge_feats,
            edge_index,
            cutoff,
            nodes,
            density0,
        )
        product0 = self.product0(density0, None, node_attrs)
        self._validate_product(product0, nodes)
        return self.forward_from_product(
            product0, node_attrs, edge_attrs, edge_feats, edge_index, cutoff
        )

    @staticmethod
    def _validate_cotangent(
        value: torch.Tensor, expected: torch.Tensor, name: str
    ) -> None:
        if value.shape != expected.shape:
            raise ValueError(f"{name} cotangent shape differs from its output")
        if value.dtype != expected.dtype or value.device != expected.device:
            raise TypeError(f"{name} cotangent differs in dtype or device")

    def vjp_from_product(
        self,
        product0: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
        cotangent: ImmediateConsumerCotangent,
        *,
        create_graph: bool = False,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """Exact matrix-free adjoint of all three consumers at ``product0``."""
        if not product0.requires_grad:
            raise ValueError("product0 must require gradients for a VJP")
        output = self.forward_from_product(
            product0, node_attrs, edge_attrs, edge_feats, edge_index, cutoff
        )
        expected = ImmediateConsumerCotangent(
            output.readout0, output.interaction1_density, output.interaction1_skip
        )
        for name, value, target in zip(cotangent._fields, cotangent, expected):
            self._validate_cotangent(value, target, name)
        gradient = torch.autograd.grad(
            outputs=(
                output.readout0,
                output.interaction1_density,
                output.interaction1_skip,
            ),
            inputs=product0,
            grad_outputs=tuple(cotangent),
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=False,
        )[0]
        return gradient

    def vjp_from_density(
        self,
        density0: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: torch.Tensor | None,
        cotangent: ImmediateConsumerCotangent,
        *,
        create_graph: bool = False,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """Exact matrix-free adjoint through product0 back to ``density0``."""
        if not density0.requires_grad:
            raise ValueError("density0 must require gradients for a VJP")
        output = self(
            density0, node_attrs, edge_attrs, edge_feats, edge_index, cutoff
        )
        expected = ImmediateConsumerCotangent(
            output.readout0, output.interaction1_density, output.interaction1_skip
        )
        for name, value, target in zip(cotangent._fields, cotangent, expected):
            self._validate_cotangent(value, target, name)
        return torch.autograd.grad(
            outputs=(
                output.readout0,
                output.interaction1_density,
                output.interaction1_skip,
            ),
            inputs=density0,
            grad_outputs=tuple(cotangent),
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=False,
        )[0]


def compile_mace_off23_immediate_consumer(
    model: Any, checkpoint_sha256: str
) -> ImmediateConsumerAction:
    """Bind the registered two-layer scalar MACE-OFF23 architecture strictly."""
    interactions = tuple(getattr(model, "interactions", ()))
    products = tuple(getattr(model, "products", ()))
    readouts = tuple(getattr(model, "readouts", ()))
    if len(interactions) != 2 or len(products) != 2 or len(readouts) != 2:
        raise TypeError("the registered immediate fragment requires exactly two layers")
    product0, readout0, interaction1 = products[0], readouts[0], interactions[1]
    expected_classes = {
        "product0": "mace.modules.blocks.EquivariantProductBasisBlock",
        "readout0": "mace.modules.blocks.LinearReadoutBlock",
        "interaction1": "mace.modules.blocks.RealAgnosticResidualInteractionBlock",
    }
    observed_classes = {
        "product0": _qualified_name(product0),
        "readout0": _qualified_name(readout0),
        "interaction1": _qualified_name(interaction1),
    }
    if observed_classes != expected_classes:
        raise TypeError(
            "registered immediate fragment module classes differ: "
            f"{observed_classes}"
        )
    if bool(getattr(product0, "use_sc", True)):
        raise TypeError("product0 must not have a residual input")
    contractions = tuple(
        getattr(getattr(product0, "symmetric_contractions", None), "contractions", ())
    )
    if len(contractions) != 1 or int(getattr(contractions[0], "correlation", -1)) != 3:
        raise TypeError("product0 must be one scalar correlation-three contraction")
    channels = _dimension(interaction1.node_feats_irreps, "product channel")
    target = _dimension(interaction1.target_irreps, "interaction target")
    if target % channels:
        raise TypeError("interaction1 target does not form a rectangular field")
    if int(getattr(contractions[0], "num_features", -1)) != channels:
        raise TypeError("product0 contraction and interaction1 channel counts differ")
    num_elements = int(getattr(model, "atomic_numbers").numel())
    readout_linear = getattr(readout0, "linear", None)
    if readout_linear is None:
        raise TypeError("readout0 must expose its linear map")
    if _dimension(readout_linear.irreps_in, "readout input") != channels:
        raise TypeError("readout0 and product0 channel counts differ")
    shape = ImmediateConsumerShape(
        channels=channels,
        angular_dimension=target // channels,
        num_elements=num_elements,
        edge_attribute_dimension=_dimension(
            interaction1.edge_attrs_irreps, "edge attribute"
        ),
        edge_feature_dimension=_dimension(
            interaction1.edge_feats_irreps, "edge feature"
        ),
        readout_dimension=_dimension(readout_linear.irreps_out, "readout output"),
    )
    scale = getattr(getattr(model, "scale_shift", None), "scale", None)
    if scale is None or scale.numel() != 1:
        raise TypeError("the registered model must expose one checkpoint scale")
    return ImmediateConsumerAction(
        product0,
        readout0,
        interaction1,
        shape,
        checkpoint_sha256,
        readout_scale=scale,
    )


def _product0_coefficients(model: Any) -> dict[int, torch.Tensor]:
    """Return native-channel coefficients before product0's final linear map."""
    from mace_odt.quotient_module import (
        native_weight_for_order,
        torch_symmetrize_repeated_slots,
    )

    contraction = model.products[0].symmetric_contractions.contractions[0]
    result: dict[int, torch.Tensor] = {}
    for order in range(1, int(contraction.correlation) + 1):
        coupling = torch_symmetrize_repeated_slots(
            contraction.U_tensors(order).detach(), order
        )
        weights = native_weight_for_order(contraction, order).detach()
        if order == 1:
            coefficients = torch.einsum("ip,zpc->zci", coupling, weights)
        elif order == 2:
            coefficients = torch.einsum("ijp,zpc->zcij", coupling, weights)
        elif order == 3:
            coefficients = torch.einsum("ijkp,zpc->zcijk", coupling, weights)
        else:
            raise ValueError("only product orders one through three are supported")
        result[order] = coefficients.contiguous()
    return result


def _evaluate_product0_coefficients(
    orders: Mapping[int, torch.Tensor],
    density: torch.Tensor,
    node_attrs: torch.Tensor,
) -> torch.Tensor:
    result = torch.einsum("zci,nci,nz->nc", orders[1], density, node_attrs)
    result = result + torch.einsum(
        "zcij,nci,ncj,nz->nc", orders[2], density, density, node_attrs
    )
    result = result + torch.einsum(
        "zcijk,nci,ncj,nck,nz->nc",
        orders[3],
        density,
        density,
        density,
        node_attrs,
    )
    return result


def _relative_residual(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual - expected)
    denominator = torch.clamp(torch.linalg.vector_norm(expected), min=1.0)
    return float((numerator / denominator).detach().cpu())


def _numpy_key(*values: torch.Tensor | None) -> bytes:
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"none")
            continue
        array = value.detach().cpu().contiguous()
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.numpy().tobytes())
    return digest.digest()


def _one_edge_context(
    model: Any,
    source_species: int,
    target_species: int,
    radius: float,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    reference = next(model.parameters())
    device, dtype = reference.device, reference.dtype
    num_elements = int(model.atomic_numbers.numel())
    attrs = torch.zeros((2, num_elements), dtype=dtype, device=device)
    attrs[0, source_species] = 1.0
    attrs[1, target_species] = 1.0
    edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    unit = direction.to(dtype=dtype, device=device)
    unit = unit / torch.linalg.vector_norm(unit)
    vector = unit[None, :] * float(radius)
    lengths = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    edge_attrs = model.spherical_harmonics(vector)
    edge_feats, cutoff = model.radial_embedding(
        lengths, attrs, edge_index, model.atomic_numbers
    )
    return attrs, edge_attrs, edge_feats, edge_index, cutoff


def _message_pullback(
    action: ImmediateConsumerAction,
    context: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    generator: torch.Generator,
) -> tuple[torch.Tensor, float, float]:
    """Return L*L for one directed edge and independent algebra checks."""
    attrs, edge_attrs, edge_feats, edge_index, cutoff = context
    channels = action.shape.channels
    reference = edge_attrs
    zero = torch.zeros(channels, dtype=reference.dtype, device=reference.device)

    def message(source: torch.Tensor) -> torch.Tensor:
        product = torch.stack((source, torch.zeros_like(source)), dim=0)
        output = action.forward_from_product(
            product, attrs, edge_attrs, edge_feats, edge_index, cutoff
        )
        return output.interaction1_density[1].reshape(-1)

    jacobian = torch.func.jacfwd(message)(zero)
    if jacobian.shape != (
        action.shape.channels * action.shape.angular_dimension,
        channels,
    ):
        raise ValueError("message Jacobian has an unexpected shape")
    gram = jacobian.T @ jacobian
    x = torch.randn(
        channels, generator=generator, dtype=zero.dtype, device=zero.device
    )
    y = torch.randn(
        channels, generator=generator, dtype=zero.dtype, device=zero.device
    )
    response = message(x)
    pullback = torch.dot(x, gram @ x)
    pullback_residual = float(
        torch.abs(torch.dot(response, response) - pullback)
        / torch.clamp(torch.abs(torch.dot(response, response)), min=1.0)
    )
    linearity = _relative_residual(message(x + y), message(x) + message(y))
    return gram, pullback_residual, linearity


def build_immediate_consumer_environment(
    *,
    model: Any,
    metric_factors: Mapping[tuple[int, int], np.ndarray],
    message_measure: Mapping[str, object],
    order_weights: Mapping[int, float],
    consumer_weights: Mapping[str, float],
    checkpoint_sha256: str,
    exactness_tolerance: float,
) -> Mapping[str, object]:
    """Compile the exact degree-three immediate-consumer coefficient objective.

    The message metric is the declared one-free-directed-edge structural measure.
    It is exact for the supplied finite radial quadrature. It is not a molecular
    activation distribution and it uses no reference energy or force labels.
    """
    from mace_odt.global_environment import (
        angular_ell_labels,
        angular_metric,
        canonical_environment_trace,
        coefficient_norm_squared,
        extract_equivariant_environment,
        native_slot_marginal,
        projected_metric_by_angular,
    )

    required_consumers = (
        "first_readout",
        "second_interaction_message",
        "second_interaction_skip",
    )
    if set(order_weights) != {1, 2, 3}:
        raise ValueError("order_weights must declare orders one through three")
    if set(consumer_weights) != set(required_consumers):
        raise ValueError("consumer_weights must declare the three immediate consumers")
    if any(not np.isfinite(value) or value < 0.0 for value in order_weights.values()):
        raise ValueError("order weights must be finite and nonnegative")
    if any(
        not np.isfinite(value) or value < 0.0 for value in consumer_weights.values()
    ):
        raise ValueError("consumer weights must be finite and nonnegative")
    if sum(order_weights.values()) <= 0.0 or sum(consumer_weights.values()) <= 0.0:
        raise ValueError("weight families cannot be identically zero")
    if not np.isfinite(exactness_tolerance) or exactness_tolerance <= 0.0:
        raise ValueError("exactness_tolerance must be positive and finite")
    if message_measure.get("schema") != "uniform-species-radial-quadrature-o3-haar-v1":
        raise ValueError("unsupported message measure schema")

    action = compile_mace_off23_immediate_consumer(model, checkpoint_sha256)
    action.validate_provenance(checkpoint_sha256)
    shape = action.shape
    atomic_numbers = model.atomic_numbers.detach().cpu().numpy()
    expected_factor_keys = {
        (central, ell)
        for central in range(shape.num_elements)
        for ell in range(4)
    }
    if set(metric_factors) != expected_factor_keys:
        raise ValueError("metric factors have incomplete species or irrep coverage")
    factors = {
        key: np.asarray(value, dtype=np.float64) for key, value in metric_factors.items()
    }
    supports = {value.shape[1] for value in factors.values()}
    if len(supports) != 1:
        raise ValueError("all canonical multiplicity supports must have equal width")
    support = supports.pop()
    if any(
        value.ndim != 2
        or value.shape[0] != shape.channels
        or not np.isfinite(value).all()
        for value in factors.values()
    ):
        raise ValueError("metric factors must be finite channel-by-support arrays")

    radii = np.asarray(message_measure.get("radii_angstrom"), dtype=np.float64)
    radial_weights = np.asarray(message_measure.get("radial_weights"), dtype=np.float64)
    neighbor_weights = np.asarray(
        message_measure.get("neighbor_species_weights"), dtype=np.float64
    )
    if (
        radii.ndim != 1
        or not len(radii)
        or radial_weights.shape != radii.shape
        or neighbor_weights.shape != (shape.num_elements,)
        or not np.isfinite(radii).all()
        or not np.isfinite(radial_weights).all()
        or not np.isfinite(neighbor_weights).all()
        or np.any(radii <= 0.0)
        or np.any(radial_weights <= 0.0)
        or np.any(neighbor_weights < 0.0)
        or not np.isclose(radial_weights.sum(), 1.0, rtol=1e-10, atol=1e-12)
        or not np.isclose(neighbor_weights.sum(), 1.0, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("message measure weights and supports are invalid")

    orders_t = _product0_coefficients(model)
    generator = torch.Generator(device="cpu").manual_seed(20260924)
    reference = next(model.parameters())
    density = torch.randn(
        shape.num_elements,
        shape.channels,
        shape.angular_dimension,
        generator=generator,
        dtype=reference.dtype,
        device="cpu",
    ).to(reference.device)
    attrs = torch.eye(
        shape.num_elements, dtype=reference.dtype, device=reference.device
    )
    raw = _evaluate_product0_coefficients(orders_t, density, attrs)
    compiled_product = action.product0.linear(raw)
    native_product = action.product0(density, None, attrs)
    reconstruction_residual = _relative_residual(compiled_product, native_product)

    identity = torch.eye(
        shape.channels, dtype=reference.dtype, device=reference.device
    )
    product_linear = action.product0.linear(identity).detach()
    readout_response = (
        action.readout0(identity).reshape(shape.channels, -1)
        * action.readout_scale
    ).detach()
    readout_gram = readout_response @ readout_response.T
    skip_grams: list[torch.Tensor] = []
    pullback_residuals: list[float] = []
    linearity_residuals: list[float] = []
    for central in range(shape.num_elements):
        onehot = torch.zeros(
            (shape.channels, shape.num_elements),
            dtype=reference.dtype,
            device=reference.device,
        )
        onehot[:, central] = 1.0
        response = action.interaction1.skip_tp(identity, onehot).detach()
        if response.shape != (shape.channels, shape.channels):
            raise ValueError("skip response has an unexpected shape")
        skip_grams.append(response @ response.T)

    message_grams = [
        torch.zeros_like(readout_gram) for _ in range(shape.num_elements)
    ]
    cache: dict[bytes, tuple[torch.Tensor, float, float]] = {}
    z_direction = torch.tensor([0.0, 0.0, 1.0])
    x_direction = torch.tensor([1.0, 0.0, 0.0])
    angular_check: float | None = None
    for central in range(shape.num_elements):
        for neighbor, species_weight in enumerate(neighbor_weights):
            if species_weight == 0.0:
                continue
            for radius, radial_weight in zip(radii, radial_weights):
                context = _one_edge_context(
                    model, central, neighbor, float(radius), z_direction
                )
                key = _numpy_key(context[1], context[2], context[4])
                if key not in cache:
                    cache[key] = _message_pullback(action, context, generator)
                gram, pullback_residual, linearity = cache[key]
                message_grams[central] = message_grams[central] + (
                    float(species_weight * radial_weight) * gram
                )
                pullback_residuals.append(pullback_residual)
                linearity_residuals.append(linearity)
                if angular_check is None:
                    rotated_context = _one_edge_context(
                        model, central, neighbor, float(radius), x_direction
                    )
                    rotated, rotated_pullback, rotated_linearity = _message_pullback(
                        action, rotated_context, generator
                    )
                    angular_check = float(
                        torch.linalg.vector_norm(rotated - gram)
                        / torch.clamp(torch.linalg.vector_norm(gram), min=1.0)
                    )
                    pullback_residuals.append(rotated_pullback)
                    linearity_residuals.append(rotated_linearity)

    output_metrics: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in required_consumers
    }
    for central in range(shape.num_elements):
        native_grams = {
            "first_readout": readout_gram,
            "second_interaction_message": message_grams[central],
            "second_interaction_skip": skip_grams[central],
        }
        for name, gram in native_grams.items():
            pulled = product_linear @ gram @ product_linear.T
            output_metrics[name][central] = pulled.detach().cpu().numpy()

    ell_labels = angular_ell_labels({0: 1, 1: 3, 2: 5, 3: 7})
    if len(ell_labels) != shape.angular_dimension:
        raise ValueError("registered angular layout differs from ell zero through three")
    orders = {
        order: tensor.detach().cpu().numpy() for order, tensor in orders_t.items()
    }
    consumer_blocks: dict[str, dict[tuple[int, int], np.ndarray]] = {
        name: {} for name in required_consumers
    }
    norms: dict[tuple[str, int, int], float] = {}
    trace_residuals: list[float] = []
    equivariance_residuals: list[float] = []
    for name in required_consumers:
        for central in range(shape.num_elements):
            local_factors = {ell: factors[(central, ell)] for ell in range(4)}
            radial_metric = angular_metric(local_factors, ell_labels)
            native_environment = np.zeros(
                (
                    shape.channels,
                    shape.angular_dimension,
                    shape.channels,
                    shape.angular_dimension,
                ),
                dtype=np.float64,
            )
            expected_trace = 0.0
            for order in (1, 2, 3):
                coefficients = orders[order][central]
                output_metric = output_metrics[name][central]
                norm = coefficient_norm_squared(
                    coefficients, radial_metric, output_metric=output_metric
                )
                norms[(name, central, order)] = norm
                slot_sum = sum(
                    native_slot_marginal(
                        coefficients,
                        radial_metric,
                        slot,
                        output_metric=output_metric,
                    )
                    for slot in range(order)
                )
                native_environment += float(order_weights[order]) * slot_sum
                expected_trace += float(order_weights[order]) * order * norm
            extracted = extract_equivariant_environment(
                native_environment, local_factors, ell_labels
            )
            trace_residuals.extend(
                [
                    abs(extracted.full_trace - expected_trace)
                    / max(abs(expected_trace), 1.0),
                    abs(extracted.block_trace - extracted.full_trace)
                    / max(abs(extracted.full_trace), 1.0),
                ]
            )
            equivariance_residuals.extend(
                [
                    extracted.relative_off_block_norm,
                    extracted.relative_magnetic_deviation,
                    max(0.0, -extracted.minimum_eigenvalue)
                    / max(
                        max(float(values[0]) for values in extracted.eigenvalues.values()),
                        1.0,
                    ),
                ]
            )
            for ell in range(4):
                consumer_blocks[name][(central, ell)] = extracted.multiplicity_blocks[
                    ell
                ]

    aggregate_blocks = {
        key: sum(
            float(consumer_weights[name]) * consumer_blocks[name][key]
            for name in required_consumers
        )
        for key in expected_factor_keys
    }

    factor_three_residuals: list[float] = []
    for central in range(shape.num_elements):
        local_factors = {ell: factors[(central, ell)] for ell in range(4)}
        radial_metric = angular_metric(local_factors, ell_labels)
        eigensystems = {}
        for ell in range(4):
            block = 0.5 * (
                aggregate_blocks[(central, ell)]
                + aggregate_blocks[(central, ell)].T
            )
            values, vectors = np.linalg.eigh(block)
            order = np.argsort(values)[::-1]
            eigensystems[ell] = (np.maximum(values[order], 0.0), vectors[:, order])
        for rank in sorted({1, max(1, support // 2), max(1, support - 1), support}):
            projectors = {
                ell: eigensystems[ell][1][:, :rank]
                @ eigensystems[ell][1][:, :rank].T
                for ell in range(4)
            }
            projected_metric = projected_metric_by_angular(
                local_factors, projectors, ell_labels
            )
            actual = 0.0
            for name in required_consumers:
                for polynomial_order in (1, 2, 3):
                    projected_norm = coefficient_norm_squared(
                        orders[polynomial_order][central],
                        projected_metric,
                        output_metric=output_metrics[name][central],
                    )
                    residual = max(
                        norms[(name, central, polynomial_order)] - projected_norm,
                        0.0,
                    )
                    actual += (
                        float(consumer_weights[name])
                        * float(order_weights[polynomial_order])
                        * residual
                    )
            bound = sum(
                (2 * ell + 1) * float(np.sum(eigensystems[ell][0][rank:]))
                for ell in range(4)
            )
            scale = max(actual, bound, 1.0)
            factor_three_residuals.extend(
                [max(0.0, actual - bound) / scale, max(0.0, bound - 3 * actual) / scale]
            )

    maximum_pullback = max(pullback_residuals, default=0.0)
    maximum_linearity = max(linearity_residuals, default=0.0)
    maximum_trace = max(trace_residuals, default=0.0)
    maximum_equivariance = max(
        [angular_check or 0.0, *equivariance_residuals], default=0.0
    )
    maximum_factor_three = max(factor_three_residuals, default=0.0)
    checks = {
        "analytic_coefficient_reconstruction_relative_residual": {
            "value": reconstruction_residual,
            "tolerance": exactness_tolerance,
        },
        "consumer_linearity_relative_residual": {
            "value": maximum_linearity,
            "tolerance": exactness_tolerance,
        },
        "output_pullback_relative_residual": {
            "value": maximum_pullback,
            "tolerance": exactness_tolerance,
        },
        "equivariant_block_relative_residual": {
            "value": maximum_equivariance,
            "tolerance": exactness_tolerance,
        },
        "environment_trace_relative_residual": {
            "value": maximum_trace,
            "tolerance": exactness_tolerance,
        },
        "tied_factor_three_relative_residual": {
            "value": maximum_factor_three,
            "tolerance": exactness_tolerance,
        },
    }
    metadata = {
        "output_metric": {
            "first_readout": {
                "dimension": 1,
                "metric": "euclidean_identity",
                "normalization": "checkpoint_scaled_node_output",
            },
            "second_interaction_message": {
                "dimension": shape.channels * shape.angular_dimension,
                "layout": [shape.channels, shape.angular_dimension],
                "metric": "euclidean_identity_on_complete_irreps",
                "normalization": "one_free_directed_edge",
            },
            "second_interaction_skip": {
                "dimension": shape.channels,
                "metric": "euclidean_identity",
                "normalization": "per_node_output",
            },
        },
        "message_measure_schema": message_measure["schema"],
        "message_measure_scope": "exact_finite_radial_quadrature_one_directed_edge",
        "message_unique_linear_maps": len(cache),
        "order_weights": dict(order_weights),
        "order_weight_scope": "global_by_order",
        "consumer_weights": dict(consumer_weights),
        "consumer_weight_scope": "global_by_consumer",
        "consumer_blocks_weighted": False,
        "order_weights_applied_to_consumer_blocks": True,
        "analytic_product_coefficients": True,
        "readout_checkpoint_scale_included": True,
        "complete_output_irreps": True,
        "coefficient_only": True,
        "uses_reference_labels": False,
        "uses_discovery_geometries": False,
    }
    return {
        "atomic_numbers": atomic_numbers,
        "aggregate_blocks": aggregate_blocks,
        "consumer_blocks": consumer_blocks,
        "checks": checks,
        "metadata": metadata,
    }


__all__ = [
    "ImmediateConsumerAction",
    "ImmediateConsumerCotangent",
    "ImmediateConsumerOutput",
    "ImmediateConsumerProvenance",
    "ImmediateConsumerShape",
    "build_immediate_consumer_environment",
    "compile_mace_off23_immediate_consumer",
]
