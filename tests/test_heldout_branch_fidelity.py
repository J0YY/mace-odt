import pytest

from mace_odt.cli.heldout_branch_fidelity import (
    distribution_summary,
    paired_bootstrap,
    validate_manifest_source,
)
from mace_odt.cli.projected_branch_diagnostics import (
    complete_rank_ladder,
    retained_rank_ladder,
)
from mace_odt.cli.shared_interface_fidelity import validate_branch_results


def test_distribution_summary_uses_configuration_level_values() -> None:
    result = distribution_summary([1.0, 2.0, 3.0, 4.0])
    assert result["mean"] == 2.5
    assert result["median"] == 2.5
    assert result["maximum"] == 4.0


def test_paired_bootstrap_sign_is_local_minus_global() -> None:
    result = paired_bootstrap(
        global_values=[1.0, 2.0, 1.5],
        local_values=[2.0, 3.0, 2.5],
        seed=7,
        replicates=200,
    )
    assert result["mean_local_minus_global"] == 1.0
    assert result["ci95_lower"] == 1.0
    assert result["ci95_upper"] == 1.0


def test_manifest_source_validation_accepts_exact_file(tmp_path) -> None:
    xyz = tmp_path / "test.xyz"
    xyz.write_bytes(b"frozen dataset")
    manifest = {
        "source": {
            "xyz_sha256": (
                "88c08a34ebfe74a6a202127e9af097dea312b0c5c71eec87b79cba3854d17603"
            )
        }
    }
    assert validate_manifest_source(xyz, manifest) == manifest["source"]["xyz_sha256"]


def test_manifest_source_validation_rejects_different_file(tmp_path) -> None:
    xyz = tmp_path / "test.xyz"
    xyz.write_bytes(b"different dataset")
    manifest = {"source": {"xyz_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="does not match"):
        validate_manifest_source(xyz, manifest)


def test_rank_ladder_always_includes_full_support() -> None:
    assert complete_rank_ladder([4, 8, 200], support=12) == [4, 8, 12]
    assert retained_rank_ladder([4, 8, 200], support=12) == [4, 8, 12]
    with pytest.raises(ValueError, match="nonnegative"):
        complete_rank_ladder([-1], support=12)


def test_shared_results_must_match_manifest_indices_and_source() -> None:
    manifest = {"selected": [{"index": 2}, {"index": 7}]}
    valid = {
        "manifest_source_sha256": "abc",
        "configuration_count": 2,
        "methods": [
            {"per_configuration": [{"index": 7}, {"index": 2}]},
        ],
    }
    validate_branch_results(valid, manifest, "abc")
    invalid = {**valid, "manifest_source_sha256": "different"}
    with pytest.raises(ValueError, match="manifest source"):
        validate_branch_results(invalid, manifest, "abc")
