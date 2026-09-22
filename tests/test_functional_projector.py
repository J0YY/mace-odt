import numpy as np
import torch

from mace_odt.functional_projector import (
    apply_species_equivariant_projectors,
    apply_species_functional_maps,
    native_functional_map,
)


def test_native_map_is_inverse_pair_and_supported_projector() -> None:
    rng = np.random.default_rng(12)
    factor = rng.normal(size=(7, 5))
    modes = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    result = native_functional_map(factor, modes)
    np.testing.assert_allclose(result.encoder @ result.decoder, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        result.projector @ result.projector, result.projector, atol=1e-12
    )
    np.testing.assert_allclose(result.projector, result.decoder @ result.encoder)


def test_species_projector_preserves_complete_angular_blocks() -> None:
    density = torch.arange(2 * 3 * 4, dtype=torch.float64).reshape(2, 3, 4)
    attrs = torch.eye(2, dtype=torch.float64)
    first = np.diag([1.0, 0.0, 0.0])
    second = np.diag([0.0, 1.0, 0.0])
    projectors = {
        (0, 0): first,
        (0, 1): first,
        (1, 0): second,
        (1, 1): second,
    }
    observed = apply_species_equivariant_projectors(
        density,
        attrs,
        {0: slice(0, 1), 1: slice(1, 4)},
        projectors,
    )
    expected = torch.zeros_like(density)
    expected[0, 0] = density[0, 0]
    expected[1, 1] = density[1, 1]
    torch.testing.assert_close(observed, expected)


def test_sequential_encoder_decoder_matches_projector_action() -> None:
    rng = np.random.default_rng(51)
    factor = rng.normal(size=(3, 3))
    modes = np.linalg.qr(rng.normal(size=(3, 2)))[0]
    first = native_functional_map(factor, modes)
    second = native_functional_map(factor @ np.diag([1.0, 2.0, 3.0]), modes)
    density = torch.randn(2, 3, 4, dtype=torch.float64)
    attrs = torch.eye(2, dtype=torch.float64)
    maps = {
        (0, 0): first,
        (0, 1): first,
        (1, 0): second,
        (1, 1): second,
    }
    projectors = {key: value.projector for key, value in maps.items()}
    slices = {0: slice(0, 1), 1: slice(1, 4)}
    sequential = apply_species_functional_maps(density, attrs, slices, maps)
    direct = apply_species_equivariant_projectors(
        density, attrs, slices, projectors
    )
    torch.testing.assert_close(sequential, direct, atol=1e-12, rtol=1e-12)


def test_functional_maps_respect_declared_slice_locations() -> None:
    factor = np.eye(3)
    first = native_functional_map(factor, np.eye(3)[:, :1])
    second = native_functional_map(factor, np.eye(3)[:, 1:2])
    density = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)
    attrs = torch.ones((1, 1), dtype=torch.float64)
    maps = {(0, 0): first, (0, 1): second}
    slices = {1: slice(1, 4), 0: slice(0, 1)}

    sequential = apply_species_functional_maps(density, attrs, slices, maps)
    direct = apply_species_equivariant_projectors(
        density,
        attrs,
        slices,
        {key: value.projector for key, value in maps.items()},
    )

    torch.testing.assert_close(sequential, direct, atol=1e-12, rtol=1e-12)
