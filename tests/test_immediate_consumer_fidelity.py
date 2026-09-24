import numpy as np
import pytest

from mace_odt.cli.immediate_consumer_fidelity import (
    CONSUMERS,
    METHOD_DATA,
    METHOD_EXACT,
    METHOD_EXACT_BALANCED,
    METHOD_EXACT_EQUAL,
    METHOD_FIRST,
    METHOD_LOCAL,
    decision_summary,
    load_basis_sets,
    tensor_fidelity,
    validate_frozen_protocol,
)


class Archive(dict):
    pass


def synthetic_archives(support=4, elements=2):
    exact, first, multi = Archive(), Archive(), Archive()
    for central in range(elements):
        for ell in range(4):
            suffix = f"z{central}_l{ell}"
            exact[f"gamma_eigenvectors_{suffix}"] = np.eye(support)
            for offset, consumer in enumerate(
                (
                    "first_readout",
                    "second_interaction_message",
                    "second_interaction_skip",
                )
            ):
                exact[f"gamma_{consumer}_{suffix}"] = np.diag(
                    np.arange(support, 0, -1, dtype=np.float64) + offset
                )
            first[f"gamma_eigenvectors_{suffix}"] = np.eye(support)[:, ::-1]
            multi[f"basis_total_energy_gradient_{suffix}"] = np.roll(
                np.eye(support), 1, axis=1
            )
    return exact, first, multi


def test_load_basis_sets_includes_exact_and_frozen_baselines() -> None:
    exact, first, multi = synthetic_archives()
    bases = load_basis_sets(exact, first, multi, 4, 2, [17, 19])
    assert {
        METHOD_EXACT_EQUAL,
        METHOD_EXACT_BALANCED,
        METHOD_LOCAL,
        METHOD_FIRST,
        METHOD_DATA,
        "canonical_reverse",
        "canonical_random_seed_17",
        "canonical_random_seed_19",
        "exact_consumer_first_readout",
        "exact_consumer_second_interaction_message",
        "exact_consumer_second_interaction_skip",
    } == set(bases)
    assert all(len(blocks) == 8 for blocks in bases.values())
    for blocks in bases.values():
        for basis in blocks.values():
            np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)


def test_global_trace_balance_is_invariant_to_consumer_family_scale() -> None:
    exact, first, multi = synthetic_archives()
    rotation = np.asarray(
        [
            [0.8, -0.6, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    for central in range(2):
        for ell in range(4):
            suffix = f"z{central}_l{ell}"
            exact[f"gamma_second_interaction_message_{suffix}"] = (
                rotation @ np.diag([7.0, 4.0, 2.0, 1.0]) @ rotation.T
            )
    scaled = Archive(
        {
            name: (100.0 * value if "gamma_second_interaction_message_" in name else value)
            for name, value in exact.items()
        }
    )
    first_bases = load_basis_sets(exact, first, multi, 4, 2, [])
    scaled_bases = load_basis_sets(scaled, first, multi, 4, 2, [])
    for key in first_bases[METHOD_EXACT_BALANCED]:
        left = first_bases[METHOD_EXACT_BALANCED][key][:, :2]
        right = scaled_bases[METHOD_EXACT_BALANCED][key][:, :2]
        np.testing.assert_allclose(left @ left.T, right @ right.T, atol=1e-11)


def test_tensor_fidelity_reports_absolute_and_relative_errors() -> None:
    reference = np.asarray([[3.0, 4.0], [0.0, 0.0]])
    candidate = reference + 1.0
    metrics = tensor_fidelity(reference, candidate)
    assert metrics["rmse"] == 1.0
    assert metrics["maximum_absolute_error"] == 1.0
    assert metrics["relative_l2_error"] == 0.4


def record(method, rank, force, energy):
    configurations = [
        {
            "force_rmse_eV_per_A": force,
            "energy_absolute_error_per_atom_eV": energy,
            "first_readout_rmse": force,
            "second_message_rmse": force,
            "second_skip_rmse": force,
            "final_output_rmse": force,
        }
        for _ in range(8)
    ]
    return {
        "method": method,
        "multiplicity_rank_per_irrep": rank,
        "force_rmse_eV_per_A": {"mean": force},
        "energy_absolute_error_per_atom_eV": {"mean": energy},
        "per_configuration": configurations,
    }


def test_decision_rules_use_named_baselines_and_full_rank_gate() -> None:
    records = []
    for rank in (64, 80):
        records.extend(
            [
                record(METHOD_EXACT, rank, 5e-4, 5e-5),
                record(METHOD_LOCAL, rank, 2e-3, 6e-5),
                record(METHOD_FIRST, rank, 3e-3, 6e-5),
                record(METHOD_DATA, rank, 4e-3, 6e-5),
            ]
        )
    decision = decision_summary(records, (64, 80), True)
    assert decision["weak_feasibility_passed"]
    assert decision["strong_feasibility_passed"]
    assert not decision["continue_not_compact_passed"]
    assert decision["exact_beats_all_named_baselines_rank_64"]
    assert not decision["exact_no_go_triggered"]
    assert all(
        comparison["force_rmse_holm_adjusted_p"] < 0.05
        for comparison in decision["rank_64_comparisons"].values()
    )
    assert set(decision["rank_80_comparisons"]) == {METHOD_LOCAL, METHOD_DATA}
    assert all(
        comparison["force_rmse_holm_adjusted_p"] < 0.05
        for comparison in decision["rank_80_comparisons"].values()
    )

    failed_gate = decision_summary(records, (64, 80), False)
    assert not failed_gate["weak_feasibility_passed"]
    assert not failed_gate["strong_feasibility_passed"]
    assert not failed_gate["exact_no_go_triggered"]


def test_invalid_full_rank_replay_cannot_emit_no_go() -> None:
    records = []
    for rank in (64, 80):
        records.extend(
            [
                record(METHOD_EXACT, rank, 4e-3, 5e-5),
                record(METHOD_LOCAL, rank, 2e-3, 6e-5),
                record(METHOD_FIRST, rank, 3e-3, 6e-5),
                record(METHOD_DATA, rank, 1e-3, 6e-5),
            ]
        )
    assert decision_summary(records, (64, 80), True)["exact_no_go_triggered"]
    assert not decision_summary(records, (64, 80), False)["exact_no_go_triggered"]


def test_frozen_protocol_rejects_changed_environment_objective() -> None:
    method = {
        "consumer_weights": {
            "first_readout": 1.0,
            "second_interaction_message": 1.0,
            "second_interaction_skip": 1.0,
        },
        "order_weights": {"1": 1.0, "2": 1.0, "3": 1.0},
    }
    objective = {
        "experiment": "T3_exact_immediate_consumer_environment",
        "exactness_tolerance": 5e-9,
        "required_consumers": list(CONSUMERS),
        "method": method,
    }
    protocol = {
        "experiment": "T4_exact_immediate_consumer_heldout_fidelity",
        "model": "MACE-OFF23-small",
        "dtype": "float64",
        "device": "cpu",
        "environment_objective": objective,
        "evaluation": {
            "ranks": [64, 80, 96],
            "random_seeds": [17],
            "primary_exact_method": METHOD_EXACT,
        },
    }
    environment = {
        "experiment": objective["experiment"],
        "exactness_tolerance": objective["exactness_tolerance"],
        "required_consumers": objective["required_consumers"],
        "method": method,
        "requested_model": "small",
        "dtype": "float64",
        "device": "cpu",
        "rank_ladder": [64, 80, 96],
    }
    kwargs = {
        "model": "small",
        "device": "cpu",
        "dtype": "float64",
        "ranks": [64, 80, 96],
        "random_seeds": [17],
    }
    validate_frozen_protocol(protocol, environment, **kwargs)
    changed = dict(environment)
    changed["method"] = {
        **method,
        "consumer_weights": {
            **method["consumer_weights"],
            "first_readout": 2.0,
        },
    }
    with pytest.raises(ValueError, match="consumer_weights"):
        validate_frozen_protocol(protocol, changed, **kwargs)
