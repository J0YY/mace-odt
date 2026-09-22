import numpy as np

from mace_odt.radial_interface import supported_metric_factor


def test_supported_metric_factor_reconstructs_gram() -> None:
    rng = np.random.default_rng(5)
    left = rng.normal(size=(20, 3))
    right = rng.normal(size=(3, 7))
    table = left @ right
    factor, singular_values, rank, tolerance = supported_metric_factor(table)
    assert rank == 3
    assert singular_values[rank - 1] > tolerance
    np.testing.assert_allclose(
        factor @ factor.T, table.T @ table, atol=1e-10, rtol=1e-10
    )
