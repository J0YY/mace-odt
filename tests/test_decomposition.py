import numpy as np
import pytest

from mace_odt.decomposition import (
    mixed_order_decomposition,
    valid_cut_decomposition,
)


def test_valid_cut_matches_explicit_svd_and_tail() -> None:
    rng = np.random.default_rng(42)
    upstream = rng.normal(size=(7, 13))
    downstream = rng.normal(size=(9, 7))
    result = valid_cut_decomposition(upstream, downstream, rank=4)
    singular_values = np.linalg.svd(downstream @ upstream, compute_uv=False)
    np.testing.assert_allclose(
        result.squared_singular_values,
        singular_values[:7] ** 2,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        result.residual_squared,
        result.discarded_eigenvalue_sum,
        atol=1e-10,
        rtol=1e-10,
    )


def test_mixed_order_bound_uses_all_slots_and_eigenvalue_tail() -> None:
    rng = np.random.default_rng(19)
    linear = rng.normal(size=(6,))
    quadratic_raw = rng.normal(size=(6, 6))
    quadratic = 0.5 * (quadratic_raw + quadratic_raw.T)
    cubic_raw = rng.normal(size=(6, 6, 6))
    cubic = sum(
        np.transpose(cubic_raw, axes=permutation)
        for permutation in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    ) / 6.0
    kernels = {1: linear, 2: quadratic, 3: cubic}
    beta = {1: 0.7, 2: 1.1, 3: 0.4}
    result = mixed_order_decomposition(kernels, beta, rank=3)
    np.testing.assert_allclose(
        result.marginal_bound,
        np.sum(result.eigenvalues[3:]),
        atol=1e-10,
        rtol=1e-10,
    )
    assert result.actual_error_squared <= result.marginal_bound + result.inequality_tolerance
    assert result.marginal_bound <= 3.0 * result.actual_error_squared + result.inequality_tolerance


def test_decompositions_reject_negative_rank() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        valid_cut_decomposition(np.eye(2), np.eye(2), rank=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        mixed_order_decomposition({1: np.ones(2)}, {1: 1.0}, rank=-1)
