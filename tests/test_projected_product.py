import numpy as np
import torch

from mace_odt.functional_projector import native_functional_map
from mace_odt.projected_product import ProjectedProductBlock


class IdentityProduct(torch.nn.Module):
    def forward(self, node_feats, sc, node_attrs):
        assert sc is None
        return node_feats


def test_projected_product_applies_maps_before_native_block() -> None:
    factor = np.eye(3)
    first_modes = np.asarray([[1.0], [0.0], [0.0]])
    second_modes = np.asarray([[0.0], [1.0], [0.0]])
    first = native_functional_map(factor, first_modes)
    second = native_functional_map(factor, second_modes)
    maps = {
        (central, ell): first if central == 0 else second
        for central in range(2)
        for ell in range(4)
    }
    wrapper = ProjectedProductBlock(IdentityProduct(), maps, num_elements=2)
    features = torch.randn(2, 3, 16, dtype=torch.float64)
    attrs = torch.eye(2, dtype=torch.float64)
    observed = wrapper(node_feats=features, sc=None, node_attrs=attrs)
    expected = torch.zeros_like(features)
    expected[0, 0] = features[0, 0]
    expected[1, 1] = features[1, 1]
    torch.testing.assert_close(observed, expected)
