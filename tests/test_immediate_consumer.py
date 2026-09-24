import pytest
import torch

from mace_odt.immediate_consumer import (
    ImmediateConsumerAction,
    ImmediateConsumerCotangent,
    ImmediateConsumerShape,
)


class PolynomialProduct(torch.nn.Module):
    use_sc = False

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(1, channels + 1, dtype=torch.float64) / channels,
            requires_grad=False,
        )

    def forward(self, density, sc, node_attrs):
        assert sc is None
        species_scale = 1.0 + node_attrs[:, :1]
        return species_scale * (
            density.sum(dim=-1)
            + self.weight * density.square().sum(dim=-1)
            + 0.1 * density.pow(3).sum(dim=-1)
        )


class LinearReadout(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(channels, dtype=torch.float64)[None] + 0.25,
            requires_grad=False,
        )

    def forward(self, features):
        return features @ self.weight.T


class ResidualInteraction(torch.nn.Module):
    avg_num_neighbors = 2.0

    def __init__(self, channels: int, angular: int, elements: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(73)
        self.message = torch.nn.Parameter(
            torch.randn(channels, angular, channels, generator=generator, dtype=torch.float64),
            requires_grad=False,
        )
        self.skip = torch.nn.Parameter(
            torch.randn(elements, channels, channels, generator=generator, dtype=torch.float64),
            requires_grad=False,
        )

    def forward(
        self,
        *,
        node_attrs,
        node_feats,
        edge_attrs,
        edge_feats,
        edge_index,
        cutoff,
        first_layer,
    ):
        assert not first_layer
        edge_scale = edge_feats.sum(dim=-1)
        if cutoff is not None:
            edge_scale = edge_scale * cutoff.reshape(-1)
        edge_values = torch.einsum(
            "e,eca,eac->eca",
            edge_scale,
            edge_attrs[:, None, :].expand(-1, node_feats.shape[1], -1),
            torch.einsum("oac,ec->eao", self.message, node_feats[edge_index[0]]),
        )
        density = torch.zeros(
            node_feats.shape[0],
            node_feats.shape[1],
            edge_attrs.shape[1],
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        density.index_add_(0, edge_index[1], edge_values)
        selected = torch.einsum("nz,zoc,nc->no", node_attrs, self.skip, node_feats)
        return density, selected


def fixture():
    channels, angular, elements, radial = 3, 2, 2, 4
    shape = ImmediateConsumerShape(channels, angular, elements, angular, radial)
    action = ImmediateConsumerAction(
        PolynomialProduct(channels),
        LinearReadout(channels),
        ResidualInteraction(channels, angular, elements),
        shape,
        "a" * 64,
    )
    generator = torch.Generator().manual_seed(19)
    density = torch.randn(4, channels, angular, generator=generator, dtype=torch.float64)
    attrs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
    edge_attrs = torch.randn(5, angular, generator=generator, dtype=torch.float64)
    edge_feats = torch.randn(5, radial, generator=generator, dtype=torch.float64)
    cutoff = torch.rand(5, generator=generator, dtype=torch.float64)
    return action, density, attrs, edge_attrs, edge_feats, edge_index, cutoff


def test_exact_forward_exposes_all_three_product_consumers() -> None:
    action, density, attrs, edge_attrs, edge_feats, edge_index, cutoff = fixture()
    observed = action(density, attrs, edge_attrs, edge_feats, edge_index, cutoff)
    product = action.product0(density, None, attrs)
    readout = action.readout0(product)
    message, skip = action.interaction1(
        node_attrs=attrs,
        node_feats=product,
        edge_attrs=edge_attrs,
        edge_feats=edge_feats,
        edge_index=edge_index,
        cutoff=cutoff,
        first_layer=False,
    )
    torch.testing.assert_close(observed.product0, product)
    torch.testing.assert_close(observed.readout0, readout)
    torch.testing.assert_close(observed.interaction1_density, message)
    torch.testing.assert_close(observed.interaction1_skip, skip)


def test_matrix_free_vjps_match_independent_autograd() -> None:
    action, density, attrs, edge_attrs, edge_feats, edge_index, cutoff = fixture()
    density = density.requires_grad_(True)
    output = action(density, attrs, edge_attrs, edge_feats, edge_index, cutoff)
    generator = torch.Generator().manual_seed(91)
    cotangent = ImmediateConsumerCotangent(
        torch.randn(output.readout0.shape, generator=generator, dtype=torch.float64),
        torch.randn(
            output.interaction1_density.shape, generator=generator, dtype=torch.float64
        ),
        torch.randn(
            output.interaction1_skip.shape, generator=generator, dtype=torch.float64
        ),
    )
    scalar = (
        (output.readout0 * cotangent.readout0).sum()
        + (output.interaction1_density * cotangent.interaction1_density).sum()
        + (output.interaction1_skip * cotangent.interaction1_skip).sum()
    )
    expected_density = torch.autograd.grad(scalar, density, retain_graph=True)[0]
    observed_density = action.vjp_from_density(
        density,
        attrs,
        edge_attrs,
        edge_feats,
        edge_index,
        cutoff,
        cotangent,
    )
    torch.testing.assert_close(observed_density, expected_density)

    product = output.product0.detach().requires_grad_(True)
    product_output = action.forward_from_product(
        product, attrs, edge_attrs, edge_feats, edge_index, cutoff
    )
    product_scalar = (
        (product_output.readout0 * cotangent.readout0).sum()
        + (product_output.interaction1_density * cotangent.interaction1_density).sum()
        + (product_output.interaction1_skip * cotangent.interaction1_skip).sum()
    )
    expected_product = torch.autograd.grad(product_scalar, product, retain_graph=True)[0]
    observed_product = action.vjp_from_product(
        product,
        attrs,
        edge_attrs,
        edge_feats,
        edge_index,
        cutoff,
        cotangent,
    )
    torch.testing.assert_close(observed_product, expected_product)


def test_shape_and_provenance_checks_reject_mismatches() -> None:
    action, density, attrs, edge_attrs, edge_feats, edge_index, cutoff = fixture()
    action.validate_provenance("a" * 64)
    with pytest.raises(ValueError, match="checkpoint provenance"):
        action.validate_provenance("b" * 64)
    with torch.no_grad():
        action.readout0.weight.add_(1.0)
    with pytest.raises(ValueError, match="modules have changed"):
        action.validate_provenance("a" * 64)

    action, density, attrs, edge_attrs, edge_feats, edge_index, cutoff = fixture()
    with pytest.raises(ValueError, match="density0 shape"):
        action(
            density[:, :-1], attrs, edge_attrs, edge_feats, edge_index, cutoff
        )
    invalid_edges = edge_index.clone()
    invalid_edges[0, 0] = len(density)
    with pytest.raises(ValueError, match="invalid node"):
        action(
            density, attrs, edge_attrs, edge_feats, invalid_edges, cutoff
        )
    with pytest.raises(TypeError, match="share dtype"):
        action(
            density,
            attrs.float(),
            edge_attrs,
            edge_feats,
            edge_index,
            cutoff,
        )
