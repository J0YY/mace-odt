import torch

from mace_odt.first_branch import (
    FirstBranchCoefficients,
    coefficient_symmetry_residual,
    evaluate_first_branch_coefficients,
)


def test_coefficient_evaluator_uses_same_species_and_channel_in_every_slot() -> None:
    generator = torch.Generator().manual_seed(41)
    elements = 3
    channels = 4
    angular = 5
    batch = 7
    linear = torch.randn(elements, channels, angular, generator=generator)
    quadratic_raw = torch.randn(
        elements, channels, angular, angular, generator=generator
    )
    quadratic = 0.5 * (quadratic_raw + quadratic_raw.transpose(-1, -2))
    cubic_raw = torch.randn(
        elements, channels, angular, angular, angular, generator=generator
    )
    cubic = sum(
        cubic_raw.permute(0, 1, *permutation)
        for permutation in (
            (2, 3, 4),
            (2, 4, 3),
            (3, 2, 4),
            (3, 4, 2),
            (4, 2, 3),
            (4, 3, 2),
        )
    ) / 6.0
    constant = torch.randn(elements, generator=generator)
    compiled = FirstBranchCoefficients(
        orders={1: linear, 2: quadratic, 3: cubic},
        constant=constant,
        downstream_channel_weights=torch.ones(channels),
        angular_dimension=angular,
        channels=channels,
        num_elements=elements,
    )
    density = torch.randn(batch, channels, angular, generator=generator)
    indices = torch.arange(batch) % elements
    attrs = torch.nn.functional.one_hot(indices, num_classes=elements).to(
        dtype=density.dtype
    )
    observed = evaluate_first_branch_coefficients(compiled, density, attrs)
    expected = []
    for node in range(batch):
        z = int(indices[node])
        value = constant[z]
        for channel in range(channels):
            x = density[node, channel]
            value = value + torch.einsum("i,i->", linear[z, channel], x)
            value = value + torch.einsum(
                "ij,i,j->", quadratic[z, channel], x, x
            )
            value = value + torch.einsum(
                "ijk,i,j,k->", cubic[z, channel], x, x, x
            )
        expected.append(value)
    torch.testing.assert_close(observed, torch.stack(expected))
    assert coefficient_symmetry_residual(quadratic) == 0.0
    assert coefficient_symmetry_residual(cubic) < 1e-6
