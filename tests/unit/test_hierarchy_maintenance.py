import copy
import math

import numpy as np
import pytest

from src.internal.document_index.hierarchy import HierarchyConfig
from src.internal.document_index.hierarchy_maintenance import (
    balance_levels,
    centroid,
    fit_reference,
    kmeans_partition,
    maintain_leaves,
)

DIMS = 8


def unit(v):
    v = np.asarray(v, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


def topic(axis, prefix, n):
    """n unit vectors near topic ``axis`` with small fixed offsets on axes 4-7."""
    vectors = {}
    for j in range(n):
        v = np.zeros(DIMS)
        v[axis] = 1.0
        v[4 + j % 4] = 0.05 * (1 + j % 2)
        vectors[f"{prefix}{j}"] = unit(v)
    return vectors


def leaf(gid, parent, members, vectors):
    return dict(
        id=gid,
        parent_id=parent,
        label=gid,
        document_ids=sorted(members),
        centroid=centroid(sorted(members), vectors),
    )


def two_leaves():
    vectors = {**topic(0, "a", 6), **topic(1, "b", 6)}
    leaves = [
        leaf("A", "p", [i for i in vectors if i.startswith("a")], vectors),
        leaf("B", "p", [i for i in vectors if i.startswith("b")], vectors),
    ]
    return vectors, leaves


def assign(leaves):
    return {i: g["id"] for g in leaves for i in g["document_ids"]}


def test_kmeans_partition_bounds_size_and_is_deterministic():
    vectors = {**topic(0, "a", 15), **topic(1, "b", 15)}
    members = sorted(vectors)
    first = kmeans_partition(members, vectors, 1, 7, 42)
    second = kmeans_partition(members, vectors, 1, 7, 42)
    buckets, degenerate, oversized = first
    assert first == second
    assert not degenerate and oversized
    assert max(len(b) for b in buckets) <= 7
    assert sorted(i for b in buckets for i in b) == members


def test_kmeans_partition_reports_degenerate_vectors():
    vectors = {str(i): unit([1, 0, 0, 0]) for i in range(4)}
    buckets, degenerate, oversized = kmeans_partition(
        sorted(vectors), vectors, 3, 10, 42
    )
    assert degenerate and not oversized
    assert sorted(i for b in buckets for i in b) == sorted(vectors)


def test_centroid_is_normalized_mean():
    vectors = {"x": unit([1, 0]), "y": unit([0, 1])}
    assert np.allclose(centroid(["x", "y"], vectors), unit([1, 1]))


def test_fit_reference_is_quantile_of_member_similarity():
    vectors = {**topic(0, "a", 4), **topic(1, "b", 4)}
    vectors["x"] = unit(
        [math.cos(math.pi / 3), 0, math.sin(math.pi / 3), 0, 0, 0, 0, 0]
    )
    a_members = [*topic(0, "a", 4), "x"]
    leaves = [
        dict(
            id="A",
            parent_id=None,
            label="A",
            document_ids=a_members,
            centroid=unit([1] + [0] * 7),
        ),
        leaf("B", None, list(topic(1, "b", 4)), vectors),
    ]
    pooled = [
        float(np.dot(vectors[i], g["centroid"]))
        for g in leaves
        for i in g["document_ids"]
    ]
    reference = fit_reference(leaves, vectors, 0.05)
    assert math.isclose(reference, float(np.quantile(pooled, 0.05)))
    tight = [s for s in pooled if s > 0.9]
    assert len(tight) == 8 and reference < min(tight)


def test_new_topic_spawns_leaf_and_forces_parent_refresh():
    vectors, leaves = two_leaves()
    reference = fit_reference(leaves, vectors, 0.05)
    new = topic(2, "n", 6)
    vectors.update(new)
    assignments = {**assign(leaves), **{i: "A" for i in new}}
    r = maintain_leaves(
        leaves,
        vectors,
        assignments,
        HierarchyConfig(min_cluster_size=5),
        reference,
        set(new),
    )
    assert [o["op"] for o in r.operations] == ["spawn"]
    spawned = [g for g in r.leaves if g["id"] not in {"A", "B"}]
    assert len(spawned) == 1
    assert (
        set(spawned[0]["document_ids"]) == set(new) and spawned[0]["parent_id"] == "p"
    )
    assert spawned[0]["id"].startswith("semantic-")
    assert r.operations[0]["group_ids"] == [spawned[0]["id"]]
    assert r.operations[0]["source_ids"] == ["A"]
    assert r.forced_description_ids == {"p"}
    assert {"A", "B"} <= {g["id"] for g in r.leaves}
    assert set(next(g for g in r.leaves if g["id"] == "A")["document_ids"]) == set(
        topic(0, "a", 6)
    )
    # Only spawned members below the reference stay pending (here n3 ties the
    # reference to within float rounding).
    assert r.pending_poor_fit_ids == sorted(
        i
        for i in spawned[0]["document_ids"]
        if float(np.dot(vectors[i], spawned[0]["centroid"])) < reference
    )
    assert len(r.pending_poor_fit_ids) < len(new)
    assert r.poor_fit_count == sum(
        float(np.dot(vectors[i], g["centroid"])) < reference
        for g in r.leaves
        for i in g["document_ids"]
    )
    assert not r.degenerate


def test_new_topic_spawns_despite_scattered_poor_fit_candidates():
    # Regression: one k-means cluster over all poor fits mixed the new topic
    # with unrelated poor fits, failed the fit check, and spawned nothing.
    vectors, leaves = two_leaves()
    reference = fit_reference(leaves, vectors, 0.05)
    new = topic(2, "n", 6)
    scattered = {
        "s0": unit([0, 0, 0, 1, 0, 0, 0, 0]),
        "s1": unit([1, 0, 0, 1, 0, 0, 0, 0]),
        "s2": unit([0, 1, 0, 1, 0, 0, 0, 0]),
    }
    vectors.update(new)
    vectors.update(scattered)
    candidates = set(new) | set(scattered)
    assignments = {**assign(leaves), **{i: "A" for i in candidates}}
    r = maintain_leaves(
        leaves,
        vectors,
        assignments,
        HierarchyConfig(min_cluster_size=5),
        reference,
        candidates,
    )
    spawns = [o for o in r.operations if o["op"] == "spawn"]
    assert len(spawns) == 1
    spawned = next(g for g in r.leaves if g["id"] == spawns[0]["group_ids"][0])
    assert set(spawned["document_ids"]) == set(new)
    assert set(scattered) <= set(r.pending_poor_fit_ids)


@pytest.mark.parametrize("seed", [0, 11, 29])
def test_spawn_returns_noise_member_to_pending(seed):
    # Regression: mean fit let one random doc ride along with a coherent new
    # topic, and spawned members left pending_poor_fit_ids for good.
    dims = 32
    rng = np.random.default_rng(seed)
    axes = [rng.normal(size=dims) for _ in range(4)]

    def near(axis):
        return unit(axis / np.linalg.norm(axis) + rng.normal(scale=0.12, size=dims))

    vectors, leaves = {}, []
    for t in range(3):
        ids = [f"t{t}_{j}" for j in range(40)]
        vectors.update({i: near(axes[t]) for i in ids})
        leaves.append(leaf(f"L{t}", "p", ids, vectors))
    reference = fit_reference(leaves, vectors, 0.05)
    new = {f"n{j}": near(axes[3]) for j in range(12)}
    noise = {f"s{j}": unit(rng.normal(size=dims)) for j in range(6)}
    vectors.update(new)
    vectors.update(noise)
    candidates = set(new) | set(noise)
    assignments = assign(leaves)
    for i in candidates:
        assignments[i] = max(leaves, key=lambda g: np.dot(vectors[i], g["centroid"]))[
            "id"
        ]
    r = maintain_leaves(
        leaves,
        vectors,
        assignments,
        HierarchyConfig(min_cluster_size=5),
        reference,
        candidates,
    )
    spawns = [o for o in r.operations if o["op"] == "spawn"]
    assert len(spawns) == 1
    spawned = next(g for g in r.leaves if g["id"] == spawns[0]["group_ids"][0])
    # A noise doc may ride along in the spawned leaf, but it stays a pending
    # poor fit; genuine members above the reference do not.
    assert set(noise) & set(spawned["document_ids"])
    assert set(noise) <= set(r.pending_poor_fit_ids)
    assert r.pending_poor_fit_ids == sorted(r.pending_poor_fit_ids)
    members = {i for g in r.leaves for i in g["document_ids"]}
    assert set(r.pending_poor_fit_ids) <= members
    for i in spawned["document_ids"]:
        fits = float(np.dot(vectors[i], spawned["centroid"])) >= reference
        assert (i in r.pending_poor_fit_ids) != fits


def test_unrelated_poor_fits_never_spawn_and_stay_pending():
    vectors, leaves = two_leaves()
    reference = fit_reference(leaves, vectors, 0.05)
    unrelated = {
        f"u{k}": unit(np.eye(DIMS)[2 + k % 6] + 0.3 * np.eye(DIMS)[(3 + k) % DIMS])
        for k in range(12)
    }
    vectors.update(unrelated)
    assignments = {**assign(leaves), **{i: "A" for i in unrelated}}
    r = maintain_leaves(
        leaves,
        vectors,
        assignments,
        HierarchyConfig(min_cluster_size=5),
        reference,
        set(unrelated),
    )
    assert "spawn" not in [o["op"] for o in r.operations]
    assert set(unrelated) <= set(r.pending_poor_fit_ids)


def test_effective_minimum_is_capped_by_max_cluster_size():
    vectors = {**topic(0, "a", 3), **topic(1, "b", 3)}
    leaves = [
        leaf("A", "p", list(topic(0, "a", 3)), vectors),
        leaf("B", "p", list(topic(1, "b", 3)), vectors),
    ]
    reference = fit_reference(leaves, vectors, 0.05)
    new = topic(2, "n", 3)
    vectors.update(new)
    assignments = {**assign(leaves), **{i: "A" for i in new}}
    config = HierarchyConfig(max_cluster_size=3, min_cluster_size=5)
    r = maintain_leaves(leaves, vectors, assignments, config, reference, set(new))
    assert [o["op"] for o in r.operations] == ["spawn"]
    assert {"A", "B"} <= {g["id"] for g in r.leaves}


def test_too_few_poor_fits_do_not_spawn():
    vectors, leaves = two_leaves()
    reference = fit_reference(leaves, vectors, 0.05)
    new = topic(2, "n", 3)
    vectors.update(new)
    assignments = {**assign(leaves), **{i: "A" for i in new}}
    r = maintain_leaves(
        leaves,
        vectors,
        assignments,
        HierarchyConfig(min_cluster_size=5),
        reference,
        set(new),
    )
    assert r.operations == []
    assert r.pending_poor_fit_ids == sorted(new)
    assert r.forced_description_ids == set()


def test_established_poor_fits_never_spawn():
    vectors, leaves = two_leaves()
    reference = fit_reference(leaves, vectors, 0.05)
    new = topic(2, "n", 6)
    vectors.update(new)
    assignments = {**assign(leaves), **{i: "A" for i in new}}
    r = maintain_leaves(
        leaves,
        vectors,
        assignments,
        HierarchyConfig(min_cluster_size=5),
        reference,
        set(),
    )
    assert "spawn" not in [o["op"] for o in r.operations]
    assert r.pending_poor_fit_ids == []


def test_drifted_leaf_splits_into_coherent_children():
    vectors, _ = two_leaves()
    leaves = [leaf("D", "p", sorted(vectors), vectors)]
    r = maintain_leaves(
        leaves, vectors, assign(leaves), HierarchyConfig(min_cluster_size=5), 0.9, set()
    )
    assert [o["op"] for o in r.operations] == ["split"]
    assert r.operations[0]["source_ids"] == ["D"]
    assert "D" not in {g["id"] for g in r.leaves}
    assert sorted(r.operations[0]["group_ids"]) == sorted(g["id"] for g in r.leaves)
    assert sorted(sorted(g["document_ids"]) for g in r.leaves) == [
        sorted(topic(0, "a", 6)),
        sorted(topic(1, "b", 6)),
    ]
    assert all(g["parent_id"] == "p" for g in r.leaves)
    assert r.poor_fit_count == 0


def test_undersized_leaf_merges_into_nearest_sibling():
    vectors, leaves = two_leaves()
    small = topic(0, "s", 2)
    vectors.update(small)
    leaves.append(leaf("S", "p", list(small), vectors))
    r = maintain_leaves(
        leaves, vectors, assign(leaves), HierarchyConfig(min_cluster_size=5), 0.9, set()
    )
    assert [o["op"] for o in r.operations] == ["merge"]
    assert r.operations[0]["group_ids"] == ["A"]
    assert sorted({g["id"] for g in r.leaves}) == ["A", "B"]
    assert set(small) <= set(
        next(g for g in r.leaves if g["id"] == "A")["document_ids"]
    )


def test_merge_prefers_same_parent():
    vectors, leaves = two_leaves()
    # "0C" fits as well as A and sorts first, so only the parent preference picks A.
    other = topic(0, "c", 6)
    vectors.update(other)
    leaves.append(leaf("0C", "q", list(other), vectors))
    small = topic(0, "s", 2)
    vectors.update(small)
    leaves.append(leaf("S", "p", list(small), vectors))
    r = maintain_leaves(
        leaves, vectors, assign(leaves), HierarchyConfig(min_cluster_size=5), 0.9, set()
    )
    assert [(o["op"], o["group_ids"]) for o in r.operations] == [("merge", ["A"])]


def test_dissimilar_undersized_leaf_does_not_merge():
    vectors, leaves = two_leaves()
    small = topic(2, "s", 2)
    vectors.update(small)
    leaves.append(leaf("S", "p", list(small), vectors))
    r = maintain_leaves(
        leaves, vectors, assign(leaves), HierarchyConfig(min_cluster_size=5), 0.9, set()
    )
    assert r.operations == []
    assert sorted(g["id"] for g in r.leaves) == ["A", "B", "S"]


def test_merge_skips_nearer_parent_target_that_fails_fit():
    vectors, leaves = two_leaves()
    leaves[0]["parent_id"] = "q"
    small = topic(0, "s", 2)
    vectors.update(small)
    leaves.append(leaf("S", "p", list(small), vectors))
    r = maintain_leaves(
        leaves, vectors, assign(leaves), HierarchyConfig(min_cluster_size=5), 0.9, set()
    )
    # Sibling B (same parent, other topic) fails the fit check; A fits.
    assert [(o["op"], o["group_ids"]) for o in r.operations] == [("merge", ["A"])]


def test_emptied_leaf_is_removed():
    vectors, leaves = two_leaves()
    leaves.append(
        dict(
            id="E",
            parent_id="p",
            label="E",
            document_ids=["a0"],
            centroid=unit([1] + [0] * 7),
        )
    )
    leaves[0]["document_ids"] = leaves[0]["document_ids"][1:]
    assignments = {**assign(leaves[:2]), "a0": "A"}
    r = maintain_leaves(
        leaves, vectors, assignments, HierarchyConfig(min_cluster_size=5), 0.9, set()
    )
    assert [(o["op"], o["source_ids"]) for o in r.operations] == [("remove", ["E"])]
    assert sorted(g["id"] for g in r.leaves) == ["A", "B"]


def test_tiny_semantic_set_is_stable():
    vectors = {"x": unit([1, 0, 0, 0]), "y": unit([0, 1, 0, 0])}
    leaves = [leaf("X", None, ["x"], vectors), leaf("Y", None, ["y"], vectors)]
    reference = fit_reference(leaves, vectors, 0.05)
    r = maintain_leaves(
        leaves,
        vectors,
        assign(leaves),
        HierarchyConfig(min_cluster_size=5),
        reference,
        {"x", "y"},
    )
    ops = [o["op"] for o in r.operations]
    assert "spawn" not in ops and "split" not in ops
    assert sorted(i for g in r.leaves for i in g["document_ids"]) == ["x", "y"]


def test_inputs_not_mutated():
    vectors, leaves = two_leaves()
    reference = fit_reference(leaves, vectors, 0.05)
    new = topic(2, "n", 6)
    vectors.update(new)
    assignments = {**assign(leaves), **{i: "A" for i in new}}
    candidates = set(new)
    before = copy.deepcopy((leaves, vectors, assignments, candidates))
    maintain_leaves(
        leaves,
        vectors,
        assignments,
        HierarchyConfig(min_cluster_size=5),
        reference,
        candidates,
    )
    assert (leaves, vectors, assignments, candidates) == before


def test_oversized_leaf_splits_when_small_bucket_cannot_fold():
    for na, nb, max_size in [(8, 3, 10), (12, 3, 14)]:
        vectors = {**topic(0, "a", na), **topic(1, "b", nb)}
        leaves = [leaf("L", "p", sorted(vectors), vectors)]
        config = HierarchyConfig(min_cluster_size=5, max_cluster_size=max_size)
        r = maintain_leaves(leaves, vectors, assign(leaves), config, 0.0, set())
        assert [o["op"] for o in r.operations] == ["split"]
        assert len(r.leaves) >= 2
        assert all(len(g["document_ids"]) <= max_size for g in r.leaves)
        assert sorted(i for g in r.leaves for i in g["document_ids"]) == sorted(vectors)


def test_split_folds_small_bucket_when_it_fits():
    vectors = {**topic(0, "a", 6), **topic(1, "b", 6), **topic(2, "c", 2)}
    leaves = [leaf("D", "p", sorted(vectors), vectors)]
    config = HierarchyConfig(min_cluster_size=5, target_cluster_size=5)
    r = maintain_leaves(leaves, vectors, assign(leaves), config, 0.9, set())
    assert [o["op"] for o in r.operations] == ["split"]
    assert sorted(len(g["document_ids"]) for g in r.leaves) == [6, 8]


def test_spawned_leaf_is_not_split_in_same_call():
    vectors, leaves = two_leaves()
    new = {**topic(2, "n", 7), **topic(3, "m", 3)}
    vectors.update(new)
    assignments = {**assign(leaves), **{i: "A" for i in new}}
    config = HierarchyConfig(min_cluster_size=3)
    r = maintain_leaves(leaves, vectors, assignments, config, 0.75, set(new))
    assert [o["op"] for o in r.operations] == ["spawn"]
    ids = {g["id"] for g in r.leaves}
    assert all(gid in ids for o in r.operations for gid in o["group_ids"])


def _level_fixture(n=9):
    vectors = {}
    leaves = []
    for k in range(n):
        v = np.zeros(DIMS)
        v[k % DIMS] = 1.0
        v[(k + 1) % DIMS] += 0.1 * (k // DIMS)
        vectors[f"d{k}"] = unit(v)
        leaves.append(leaf(f"L{k}", None, [f"d{k}"], vectors))
    return vectors, leaves


def _child_counts(groups):
    counts = {}
    for g in groups:
        counts[g["parent_id"]] = counts.get(g["parent_id"], 0) + 1
    return counts


def test_root_over_fanout_inserts_level_with_bounded_children():
    vectors, leaves = _level_fixture(9)
    before = copy.deepcopy(leaves)
    groups, ops, new_ids = balance_levels(leaves, vectors, 3, 42)
    assert leaves == before
    assert max(_child_counts(groups).values()) <= 3
    assert {g["id"] for g in leaves} <= {g["id"] for g in groups}
    assert any(o["op"] == "insert_level" for o in ops)
    assert new_ids and new_ids <= {g["id"] for g in groups}
    by_id = {g["id"]: g for g in groups}
    for g in groups:
        assert g["parent_id"] is None or g["parent_id"] in by_id
    for gid in new_ids:
        assert gid.startswith("semantic-level-")
        assert by_id[gid]["document_ids"] == []


def test_deep_overflow_repeats_until_bounded():
    vectors, leaves = _level_fixture(30)
    groups, _, _ = balance_levels(leaves, vectors, 2, 42)
    assert max(_child_counts(groups).values()) <= 2
    assert sum(1 for g in groups if g["document_ids"]) == 30


def test_disabled_fanout_changes_nothing_but_collapses_single_child():
    vectors, leaves = _level_fixture(4)
    leaves[0]["parent_id"] = "m"
    middle = dict(
        id="m",
        parent_id=None,
        label="m",
        document_ids=[],
        centroid=centroid(["d0"], vectors),
    )
    groups, ops, new_ids = balance_levels([middle] + leaves, vectors, None, 42)
    by_id = {g["id"]: g for g in groups}
    assert "m" not in by_id
    assert by_id["L0"]["parent_id"] is None
    assert [o["op"] for o in ops] == ["collapse_level"]
    assert not new_ids


def test_empty_intermediate_removed_and_no_dangling_parents():
    vectors, leaves = _level_fixture(3)
    empty = dict(id="e", parent_id="m", label="e", document_ids=[], centroid=[0.0])
    middle = dict(id="m", parent_id=None, label="m", document_ids=[], centroid=[0.0])
    groups, _, _ = balance_levels([middle, empty] + leaves, vectors, None, 42)
    ids = {g["id"] for g in groups}
    assert ids == {"L0", "L1", "L2"}
    assert all(g["parent_id"] is None or g["parent_id"] in ids for g in groups)


def test_levels_deterministic():
    vectors, leaves = _level_fixture(12)
    assert balance_levels(leaves, vectors, 3, 7) == balance_levels(
        leaves, vectors, 3, 7
    )


def test_identical_vectors_still_terminate_within_bound():
    vectors = {f"d{k}": unit([1, 0, 0, 0]) for k in range(10)}
    leaves = [leaf(f"L{k}", None, [f"d{k}"], vectors) for k in range(10)]
    groups, _, _ = balance_levels(leaves, vectors, 3, 42)
    assert max(_child_counts(groups).values()) <= 3


def test_inserted_level_id_never_collides_with_existing_group():
    vectors = {"a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.0, 1.0]}
    leaves = [
        dict(id=k, parent_id="old", label=k, document_ids=[k], centroid=v)
        for k, v in vectors.items()
    ]
    # An existing intermediate whose ID is the hash of the bucket {a, b}.
    from src.internal.document_index.hierarchy_maintenance import _hash

    old = dict(
        id="semantic-level-" + _hash(["a", "b"])[:20],
        parent_id=None,
        label="old",
        document_ids=[],
        centroid=[0.7, 0.7],
    )
    for leaf in leaves:
        leaf["parent_id"] = old["id"]
    groups, _, new_ids = balance_levels([old] + leaves, vectors, 2, 42)
    ids = [g["id"] for g in groups]
    assert len(ids) == len(set(ids))
    assert old["id"] not in new_ids
    by_id = {g["id"]: g for g in groups}
    for g in groups:
        seen = set()
        gid = g["id"]
        while gid is not None:
            assert gid not in seen
            seen.add(gid)
            gid = by_id[gid]["parent_id"]
