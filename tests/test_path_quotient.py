import numpy as np

from mace_odt.path_quotient import (
    compute_path_quotient,
    random_null_perturbation_check,
    symmetrize_repeated_slots,
)


def test_symmetrize_repeated_slots() -> None:
    tensor = np.zeros((2, 2, 2))
    tensor[0, 1, 0] = 2.0
    symmetrized = symmetrize_repeated_slots(tensor, order=2)
    assert symmetrized[0, 1, 0] == 1.0
    assert symmetrized[1, 0, 0] == 1.0


def test_path_quotient_finds_antisymmetric_null() -> None:
    coupling = np.zeros((2, 2, 2))
    coupling[0, 1, 0] = 1.0
    coupling[1, 0, 1] = 1.0
    quotient = compute_path_quotient(coupling, order=2)
    assert quotient.stored_dimension == 2
    assert quotient.supported_dimension == 1
    assert quotient.null_dimension == 1
    assert quotient.null_residual < 1e-14
    check = random_null_perturbation_check(
        coupling=coupling,
        quotient=quotient,
        num_elements=2,
        num_channels=3,
    )
    assert check["status"] == "passed"
