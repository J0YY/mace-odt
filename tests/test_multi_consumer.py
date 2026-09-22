import numpy as np
import torch

from mace_odt.multi_consumer import (
    canonical_activation_gram,
    canonical_gradient_gram,
    deterministic_completed_eigensystem,
    globally_trace_balanced_blocks,
    projector_distance,
)


def test_canonical_gradient_gram_matches_explicit_magnetic_partial_trace() -> None:
    gradient = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 1.5, 2.0]],
            [[-2.0, 1.0, 0.0, 3.0], [1.0, 2.0, -0.5, 1.0]],
        ],
        dtype=torch.float64,
    )
    attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    factor0 = np.asarray([[2.0, 0.0], [0.0, 0.5]])
    factor1 = np.asarray([[1.0, 0.2], [0.0, 1.5]])
    factors = {
        (0, 0): factor0,
        (0, 1): factor0,
        (1, 0): factor1,
        (1, 1): factor1,
    }
    observed = canonical_gradient_gram(
        gradient, attrs, factors, {0: slice(0, 1), 1: slice(1, 4)}
    )
    for central, factor in enumerate((factor0, factor1)):
        for ell, section in {0: slice(0, 1), 1: slice(1, 4)}.items():
            block = gradient[central, :, section].numpy()
            canonical = factor.T @ block
            expected = canonical @ canonical.T / canonical.shape[1]
            np.testing.assert_allclose(observed[(central, ell)], expected, atol=1e-14)


def test_canonical_activation_gram_uses_supported_left_inverse() -> None:
    factor = np.asarray([[2.0, 0.0], [0.0, 0.5]])
    inverse = np.linalg.inv(factor)
    canonical = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
    density = torch.einsum(
        "cs,nsm->ncm", torch.as_tensor(factor), canonical
    )
    attrs = torch.ones((1, 1), dtype=torch.float64)
    observed = canonical_activation_gram(
        density,
        attrs,
        {(0, 1): inverse},
        {1: slice(0, 2)},
    )[(0, 1)]
    expected = np.einsum("nsm,ntm->st", canonical.numpy(), canonical.numpy()) / 2.0
    np.testing.assert_allclose(observed, expected, atol=1e-14)


def test_global_trace_balance_gives_equal_consumer_mass() -> None:
    first = {(0, 0): np.diag([4.0, 1.0]), (0, 1): np.diag([2.0, 1.0])}
    second = {(0, 0): np.diag([1.0, 3.0]), (0, 1): np.diag([5.0, 2.0])}
    combined, metadata = globally_trace_balanced_blocks(
        first, second, {0: 1, 1: 3}
    )
    first_mass = sum(
        {0: 1, 1: 3}[ell] * np.trace(first[(central, ell)])
        for central, ell in first
    )
    second_mass = sum(
        {0: 1, 1: 3}[ell] * np.trace(second[(central, ell)])
        for central, ell in second
    )
    expected = {
        key: 0.5 * first[key] / first_mass + 0.5 * second[key] / second_mass
        for key in first
    }
    for key in expected:
        np.testing.assert_allclose(combined[key], expected[key], atol=1e-14)
    assert metadata["linear_global_trace"] == first_mass
    assert metadata["nonlinear_global_trace"] == second_mass


def test_projector_distance_is_sign_and_basis_invariant() -> None:
    identity = np.eye(4)
    rotated_inside = identity.copy()
    rotated_inside[:, :2] = rotated_inside[:, :2] @ np.asarray(
        [[0.0, -1.0], [1.0, 0.0]]
    )
    assert projector_distance(identity, rotated_inside, 2) < 1e-14
    assert projector_distance(identity, identity[:, ::-1], 2) == 1.0


def test_rank_deficient_eigensystem_has_deterministic_axis_completion() -> None:
    matrix = np.diag([5.0, 2.0, 0.0, 0.0])
    values, vectors, rank = deterministic_completed_eigensystem(matrix)
    assert rank == 2
    np.testing.assert_allclose(values, [5.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(np.abs(vectors[:, 2:]), np.eye(4)[:, 2:], atol=1e-14)
