from mace_odt.cli.dataset_manifest import (
    deterministic_score,
    proportional_allocations,
)


def test_proportional_allocation_is_complete_and_covers_every_group() -> None:
    sizes = {"large": 80, "medium": 15, "small": 5}
    allocation = proportional_allocations(sizes, 20)
    assert sum(allocation.values()) == 20
    assert all(1 <= allocation[name] <= sizes[name] for name in sizes)
    assert allocation["large"] > allocation["medium"] > allocation["small"]


def test_selection_score_is_repeatable_and_index_specific() -> None:
    assert deterministic_score(17, 4) == deterministic_score(17, 4)
    assert deterministic_score(17, 4) != deterministic_score(17, 5)
