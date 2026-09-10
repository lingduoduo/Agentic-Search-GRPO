"""`reward_sides` is an additive two-way rollup over the four reward dimensions:
what the documents earned versus what the answer earned. It changes no value."""

from __future__ import annotations

import pytest

from src.model.post_training.reward import (
    REWARD_DIMENSIONS,
    REWARD_SIDES,
    group_reward_components,
    reward_sides,
)


def _components() -> dict[str, float]:
    # One distinct value per member key, so a mis-filed key changes a sum.
    members = [key for keys in REWARD_DIMENSIONS.values() for key in keys]
    return {key: float(i + 1) / 10 for i, key in enumerate(members)}


def test_sides_partition_the_dimensions():
    assert set(REWARD_SIDES) == {"retrieval", "generation"}
    assigned = [dim for dims in REWARD_SIDES.values() for dim in dims]
    assert sorted(assigned) == sorted(REWARD_DIMENSIONS)


def test_sides_sum_to_the_dimension_total():
    components = _components()
    dims = group_reward_components(components)
    sides = reward_sides(components)
    assert sides["retrieval"] == pytest.approx(
        dims["retrieval_quality"] + dims["search_efficiency"]
    )
    assert sides["generation"] == pytest.approx(
        dims["correctness"] + dims["citation_support"]
    )
    assert sum(sides.values()) == pytest.approx(sum(dims.values()))


def test_missing_keys_count_as_zero():
    assert reward_sides({}) == {"retrieval": 0.0, "generation": 0.0}
    assert reward_sides({"correctness": 1.0}) == {"retrieval": 0.0, "generation": 1.0}
