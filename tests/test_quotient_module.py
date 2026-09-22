import numpy as np
import torch

from mace_odt.quotient_module import (
    scalar_order_response,
    torch_symmetrize_repeated_slots,
)


def test_torch_symmetrization_preserves_repeated_input_response() -> None:
    generator = torch.Generator().manual_seed(8)
    coupling = torch.randn((4, 4, 4, 5), generator=generator, dtype=torch.float64)
    weights = torch.randn((2, 5, 3), generator=generator, dtype=torch.float64)
    features = torch.randn((7, 3, 4), generator=generator, dtype=torch.float64)
    indices = torch.arange(7) % 2
    elements = torch.nn.functional.one_hot(indices, 2).to(torch.float64)
    symmetrized = torch_symmetrize_repeated_slots(coupling, order=3)
    original = scalar_order_response(coupling, weights, features, elements)
    symmetric = scalar_order_response(symmetrized, weights, features, elements)
    np.testing.assert_allclose(
        original.detach().numpy(), symmetric.detach().numpy(), atol=1e-12, rtol=1e-12
    )
