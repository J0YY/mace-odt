import numpy as np
import pytest

from mace_odt.decomposition import apply_projector_all_slots, slot_marginal
from mace_odt.global_environment import (
    angular_ell_labels,
    angular_metric,
    canonical_marginal_block,
    coefficient_norm_squared,
    dense_canonical_kernel,
    extract_equivariant_environment,
    native_slot_marginal,
    projected_metric_by_angular,
)


def test_factorized_marginals_and_norm_match_dense_canonical_oracle() -> None:
    rng = np.random.default_rng(31)
    channels = 3
    support = 2
    labels = angular_ell_labels({0: 1, 1: 3})
    factors = {
        0: rng.normal(size=(channels, support)),
        1: rng.normal(size=(channels, support)),
    }
    raw = rng.normal(size=(channels, len(labels), len(labels), len(labels)))
    coefficients = sum(
        np.transpose(raw, axes=(0,) + permutation)
        for permutation in (
            (1, 2, 3),
            (1, 3, 2),
            (2, 1, 3),
            (2, 3, 1),
            (3, 1, 2),
            (3, 2, 1),
        )
    ) / 6.0
    metric = angular_metric(factors, labels)
    dense = dense_canonical_kernel(coefficients, factors, labels)
    np.testing.assert_allclose(
        coefficient_norm_squared(coefficients, metric),
        np.linalg.norm(dense) ** 2,
        atol=1e-9,
        rtol=1e-10,
    )
    native = native_slot_marginal(coefficients, metric, slot=1, pair_chunk=4)
    canonical = np.empty((len(labels) * support,) * 2)
    for left_index, left_ell in enumerate(labels):
        for right_index, right_ell in enumerate(labels):
            block = canonical_marginal_block(
                native,
                left_index,
                right_index,
                factors[int(left_ell)],
                factors[int(right_ell)],
            )
            canonical[
                left_index * support : (left_index + 1) * support,
                right_index * support : (right_index + 1) * support,
            ] = block
    np.testing.assert_allclose(canonical, slot_marginal(dense, 1), atol=1e-9)

    directions = {
        ell: np.linalg.qr(rng.normal(size=(support, 1)))[0]
        for ell in factors
    }
    projectors = {ell: vector @ vector.T for ell, vector in directions.items()}
    projected_metric = projected_metric_by_angular(factors, projectors, labels)
    projected_norm = coefficient_norm_squared(coefficients, projected_metric)
    full_projector = np.zeros((len(labels) * support,) * 2)
    for index, ell in enumerate(labels):
        section = slice(index * support, (index + 1) * support)
        full_projector[section, section] = projectors[int(ell)]
    projected_dense = apply_projector_all_slots(dense, full_projector)
    np.testing.assert_allclose(
        projected_norm, np.linalg.norm(projected_dense) ** 2, atol=1e-9
    )


def test_vector_output_metric_matches_explicit_whitened_outputs() -> None:
    rng = np.random.default_rng(203)
    channels = 3
    angular = 2
    coefficients = rng.normal(size=(channels, angular, angular))
    coefficients = 0.5 * (coefficients + coefficients.swapaxes(1, 2))
    metric = rng.normal(size=(channels, channels, angular))
    metric = np.einsum("aci,bci->abi", metric, metric)
    response = rng.normal(size=(5, channels))
    output_metric = response.T @ response

    expected_norm = 0.0
    expected_marginal = np.zeros((angular, angular))
    for output in range(len(response)):
        for i in range(angular):
            for j in range(angular):
                for k in range(angular):
                    for left in range(channels):
                        for right in range(channels):
                            value = (
                                response[output, left]
                                * response[output, right]
                                * coefficients[left, i, k]
                                * coefficients[right, j, k]
                                * metric[left, right, k]
                            )
                            expected_marginal[i, j] += value
                            expected_norm += (
                                value * metric[left, right, i] if i == j else 0.0
                            )

    actual_marginal = native_slot_marginal(
        coefficients, metric, slot=0, output_metric=output_metric, pair_chunk=2
    )
    np.testing.assert_allclose(
        actual_marginal.sum(axis=(0, 2)), expected_marginal, atol=1e-9
    )
    np.testing.assert_allclose(
        coefficient_norm_squared(
            coefficients, metric, output_metric=output_metric, pair_chunk=2
        ),
        expected_norm,
        atol=1e-9,
    )


def test_output_metric_rejects_non_gram_matrices() -> None:
    coefficients = np.ones((2, 1), dtype=np.float64)
    metric = np.ones((2, 2, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="symmetric"):
        coefficient_norm_squared(
            coefficients,
            metric,
            output_metric=np.asarray([[1.0, 1.0], [0.0, 1.0]]),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        native_slot_marginal(
            coefficients,
            metric,
            slot=0,
            output_metric=np.asarray([[1.0, 0.0], [0.0, -1.0]]),
        )


def test_equivariant_extraction_traces_magnetic_indices() -> None:
    labels = angular_ell_labels({0: 1, 1: 3})
    factors = {0: np.eye(2), 1: np.eye(2)}
    gamma_0 = np.asarray([[3.0, 0.4], [0.4, 1.0]])
    gamma_1 = np.asarray([[2.0, -0.2], [-0.2, 0.7]])
    native = np.zeros((2, len(labels), 2, len(labels)))
    native[:, 0, :, 0] = gamma_0
    for index in range(1, 4):
        native[:, index, :, index] = gamma_1
    result = extract_equivariant_environment(native, factors, labels)
    np.testing.assert_allclose(result.multiplicity_blocks[0], gamma_0)
    np.testing.assert_allclose(result.multiplicity_blocks[1], gamma_1)
    np.testing.assert_allclose(result.full_trace, np.trace(gamma_0) + 3 * np.trace(gamma_1))
    np.testing.assert_allclose(result.full_trace, result.block_trace)
    assert result.relative_off_block_norm == 0.0
    assert result.relative_magnetic_deviation < 1e-15
