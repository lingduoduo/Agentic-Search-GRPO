"""Pure leaf-level planner for local semantic hierarchy maintenance.

Vectors and centroids are unit-normalized, so a dot product is a cosine
similarity. Every threshold is relative to ``reference``, a quantile of the
member-to-centroid similarities recorded when the leaves were built.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field

import numpy as np


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@dataclass
class MaintenanceResult:
    leaves: list[dict]
    operations: list[dict]
    forced_description_ids: set[str] = field(default_factory=set)
    poor_fit_count: int = 0
    pending_poor_fit_ids: list[str] = field(default_factory=list)
    degenerate: bool = False


def kmeans_partition(member_ids, vectors, count, max_size, random_state):
    """K-means, then split buckets over ``max_size`` into balanced chunks.

    Returns ``(buckets, degenerate, oversized)``; ``degenerate`` means sklearn
    warned, typically because duplicate vectors produced fewer distinct
    clusters; ``oversized`` means some k-means cluster exceeded ``max_size``.
    """
    members = sorted(member_ids)
    if not members:
        return [], False, False
    from sklearn.cluster import KMeans

    with warnings.catch_warnings(record=True) as caught:
        # Record even under -W error so degeneracy is reported, not raised.
        warnings.simplefilter("always")
        labels = KMeans(
            n_clusters=min(count, len(members)), random_state=random_state, n_init=10
        ).fit_predict(np.array([vectors[i] for i in members]))
    bucket_map = {}
    for member, label in zip(members, labels):
        bucket_map.setdefault(int(label), []).append(member)
    chunks = []
    oversized = False
    for label in sorted(bucket_map):
        bucket = bucket_map[label]
        parts = math.ceil(len(bucket) / max_size)
        oversized |= parts > 1
        chunks.extend(
            part.tolist()
            for part in np.array_split(np.array(bucket, dtype=object), parts)
        )
    return chunks, bool(caught), oversized


def centroid(member_ids, vectors):
    """Normalized mean of the members' vectors."""
    center = np.mean([vectors[i] for i in member_ids], axis=0)
    norm = np.linalg.norm(center)
    return (center / norm if norm else center).tolist()


def fit_reference(leaves, vectors, quantile):
    """``quantile`` of pooled member-to-own-centroid similarity over all leaves."""
    pooled = [
        float(np.dot(vectors[i], g["centroid"]))
        for g in leaves
        for i in g["document_ids"]
    ]
    return float(np.quantile(pooled, quantile))


def _new_leaf(parent_id, members, vectors):
    members = sorted(members)
    gid = "semantic-" + _hash(members)[:20]
    return dict(
        id=gid,
        parent_id=parent_id,
        label="Cluster " + gid[-8:],
        document_ids=members,
        centroid=centroid(members, vectors),
    )


def _similarity(doc_id, group, vectors):
    return float(np.dot(vectors[doc_id], group["centroid"]))


def _nearest(center, candidates):
    # Candidates arrive sorted by ID, so ties keep the smallest ID.
    return max(candidates, key=lambda g: float(np.dot(center, g["centroid"])))


def maintain_leaves(leaves, vectors, assignments, config, reference, candidate_ids):
    """Plan spawn, split, merge and remove operations over semantic leaves.

    ``assignments`` maps each semantic document to its already chosen leaf ID;
    only ``candidate_ids`` (new or previously pending poor fits) may spawn.
    """
    min_size = min(config.min_cluster_size, config.max_cluster_size)
    max_size = config.max_cluster_size
    operations = []
    forced = set()
    degenerate = False

    # 1. Rebuild memberships from assignments.
    members = {g["id"]: [] for g in leaves}
    for doc_id in sorted(assignments):
        members[assignments[doc_id]].append(doc_id)
    current = []
    for g in leaves:
        g = dict(g, document_ids=members[g["id"]])
        if g["document_ids"]:
            g["centroid"] = centroid(g["document_ids"], vectors)
        current.append(g)
    by_id = {g["id"]: g for g in current}
    owner = {i: by_id[assignments[i]] for i in assignments}

    # 2. Spawn coherent groups of poor-fit candidates.
    poor = sorted(
        i
        for i in candidate_ids
        if i in owner and _similarity(i, owner[i], vectors) < reference
    )
    spawned = set()
    spawned_ids = set()
    if len(poor) >= min_size:
        existing = sorted(
            (g for g in current if g["document_ids"]), key=lambda g: g["id"]
        )
        buckets, flag, _ = kmeans_partition(
            poor,
            vectors,
            math.ceil(len(poor) / config.target_cluster_size),
            max_size,
            config.random_state,
        )
        degenerate |= flag
        for bucket in buckets:
            if len(bucket) < min_size:
                continue
            center = centroid(bucket, vectors)
            fit = np.mean([float(np.dot(vectors[i], center)) for i in bucket])
            if fit < reference:
                continue
            parent_id = _nearest(center, existing)["parent_id"]
            if parent_id is not None:
                forced.add(parent_id)
            new = _new_leaf(parent_id, bucket, vectors)
            sources = sorted({owner[i]["id"] for i in bucket})
            for source in sources:
                g = by_id[source]
                g["document_ids"] = [i for i in g["document_ids"] if i not in bucket]
            current.append(new)
            spawned.update(bucket)
            spawned_ids.add(new["id"])
            operations.append(
                dict(
                    op="spawn",
                    group_ids=[new["id"]],
                    source_ids=sources,
                    reason=f"{len(poor)} poor-fit candidates below the fit reference",
                )
            )
        for g in current:
            if g["document_ids"]:
                g["centroid"] = centroid(g["document_ids"], vectors)

    # 3. Split oversized or drifted leaves.
    result = []
    for g in current:
        size = len(g["document_ids"])
        if g["id"] in spawned_ids:
            result.append(g)
            continue
        poor_fraction = (
            sum(_similarity(i, g, vectors) < reference for i in g["document_ids"])
            / size
            if size
            else 0.0
        )
        oversized = size > max_size
        drifted = (
            size >= 2 * min_size and poor_fraction >= config.split_poor_fit_fraction
        )
        if not (oversized or drifted):
            result.append(g)
            continue
        buckets, flag, _ = kmeans_partition(
            g["document_ids"],
            vectors,
            max(2, math.ceil(size / config.target_cluster_size)),
            max_size,
            config.random_state,
        )
        degenerate |= flag
        if flag and not oversized:
            result.append(g)
            continue
        # Fold undersized buckets, smallest first, into the nearest other bucket
        # when the result fits; a bucket that cannot fold stays its own child.
        kept = []
        while buckets:
            small = min(range(len(buckets)), key=lambda k: len(buckets[k]))
            bucket = buckets.pop(small)
            others = [b for b in buckets + kept if len(b) + len(bucket) <= max_size]
            if len(bucket) >= min_size or not others:
                kept.append(bucket)
                continue
            center = centroid(bucket, vectors)
            target = max(
                others, key=lambda b: float(np.dot(center, centroid(b, vectors)))
            )
            target.extend(bucket)
            target.sort()
        buckets = kept
        if len(buckets) < 2:
            result.append(g)
            continue
        children = [_new_leaf(g["parent_id"], b, vectors) for b in buckets]
        result.extend(children)
        reason = (
            f"size {size} exceeds max_cluster_size {max_size}"
            if oversized
            else f"poor-fit fraction {poor_fraction:.2f} reached {config.split_poor_fit_fraction}"
        )
        operations.append(
            dict(
                op="split",
                group_ids=[c["id"] for c in children],
                source_ids=[g["id"]],
                reason=reason,
            )
        )
    current = result

    # 4. Merge undersized leaves into the nearest leaf with room whose centroid
    # the members fit (mean similarity >= reference), same parent first.
    merged = set()
    for g in sorted(current, key=lambda g: (len(g["document_ids"]), g["id"])):
        size = len(g["document_ids"])
        if not 0 < size < min_size:
            continue
        fitting = sorted(
            (
                o
                for o in current
                if o is not g
                and o["document_ids"]
                and len(o["document_ids"]) + size <= max_size
                and np.mean([_similarity(i, o, vectors) for i in g["document_ids"]])
                >= reference
            ),
            key=lambda o: o["id"],
        )
        siblings = [o for o in fitting if o["parent_id"] == g["parent_id"]]
        if not (siblings or fitting):
            continue
        survivor = _nearest(g["centroid"], siblings or fitting)
        survivor["document_ids"] = sorted(survivor["document_ids"] + g["document_ids"])
        survivor["centroid"] = centroid(survivor["document_ids"], vectors)
        g["document_ids"] = []
        operations.append(
            dict(
                op="merge",
                group_ids=[survivor["id"]],
                source_ids=sorted([g["id"], survivor["id"]]),
                reason=f"size {size} below min_cluster_size {min_size}",
            )
        )
        merged.add(g["id"])

    # 5. Drop empty leaves; merged-away leaves are already reported.
    final = []
    for g in current:
        if g["document_ids"]:
            final.append(g)
        elif g["id"] not in merged:
            operations.append(
                dict(
                    op="remove",
                    group_ids=[],
                    source_ids=[g["id"]],
                    reason="leaf has no members",
                )
            )

    # 6. Final centroids and fit accounting.
    for g in final:
        g["centroid"] = centroid(g["document_ids"], vectors)
    poor_fit_count = sum(
        _similarity(i, g, vectors) < reference for g in final for i in g["document_ids"]
    )
    remaining = {i for g in final for i in g["document_ids"]}
    pending = [i for i in poor if i not in spawned and i in remaining]
    return MaintenanceResult(
        leaves=final,
        operations=operations,
        forced_description_ids=forced,
        poor_fit_count=int(poor_fit_count),
        pending_poor_fit_ids=pending,
        degenerate=degenerate,
    )


def _descendant_documents(gid, children, by_id):
    if not children.get(gid):
        return list(by_id[gid]["document_ids"])
    return [
        i
        for child in children[gid]
        for i in _descendant_documents(child, children, by_id)
    ]


def balance_levels(groups, vectors, max_fanout, random_state):
    """Collapse degenerate intermediates, then bound every parent's fanout.

    Returns ``(groups, operations, new_group_ids)``. Intermediates are groups
    with no documents of their own; a parent over ``max_fanout`` children gets
    new intermediate levels built by k-means over child centroids.
    """
    if max_fanout is not None and max_fanout < 2:
        raise ValueError("max_fanout must be at least 2")
    current = [dict(g) for g in groups]
    operations = []

    # 1. Remove childless intermediates and collapse single-child ones.
    while True:
        counts = {}
        for g in current:
            counts.setdefault(g["parent_id"], []).append(g["id"])
        target = next(
            (
                g
                for g in sorted(current, key=lambda g: g["id"])
                if not g["document_ids"] and len(counts.get(g["id"], [])) <= 1
            ),
            None,
        )
        if target is None:
            break
        kids = counts.get(target["id"], [])
        for g in current:
            if g["parent_id"] == target["id"]:
                g["parent_id"] = target["parent_id"]
        current = [g for g in current if g["id"] != target["id"]]
        operations.append(
            dict(
                op="collapse_level",
                group_ids=kids,
                source_ids=[target["id"]],
                reason=(
                    "intermediate has one child"
                    if kids
                    else "intermediate has no children"
                ),
            )
        )

    # 2. Insert levels under every parent over the fanout bound.
    new_ids = set()
    while max_fanout is not None:
        by_id = {g["id"]: g for g in current}
        children = {}
        for g in current:
            children.setdefault(g["parent_id"], []).append(g["id"])
        over = sorted(
            (p for p, kids in children.items() if len(kids) > max_fanout),
            key=lambda p: (p is not None, p or ""),
        )
        if not over:
            break
        parent = over[0]
        kids = sorted(children[parent])
        child_centroids = {
            k: centroid(_descendant_documents(k, children, by_id), vectors)
            for k in kids
        }
        buckets, _, _ = kmeans_partition(
            kids,
            child_centroids,
            math.ceil(len(kids) / max_fanout),
            max_fanout,
            random_state,
        )
        if len(buckets) >= len(kids):
            # No progress (degenerate centroids): chunk by ID order instead.
            parts = math.ceil(len(kids) / max_fanout)
            buckets = [
                p.tolist() for p in np.array_split(np.array(kids, dtype=object), parts)
            ]
        created = []
        moved = []
        for bucket in buckets:
            if len(bucket) < 2:
                continue
            bucket = sorted(bucket)
            gid = "semantic-level-" + _hash(bucket)[:20]
            for g in current:
                if g["id"] in bucket:
                    g["parent_id"] = gid
            documents = [
                i for k in bucket for i in _descendant_documents(k, children, by_id)
            ]
            current.append(
                dict(
                    id=gid,
                    parent_id=parent,
                    label="Topic " + gid[-8:],
                    document_ids=[],
                    centroid=centroid(documents, vectors),
                )
            )
            created.append(gid)
            moved.extend(bucket)
        new_ids.update(created)
        operations.append(
            dict(
                op="insert_level",
                group_ids=created,
                source_ids=sorted(moved),
                reason=f"{len(kids)} children exceed max_fanout {max_fanout}",
            )
        )
    return current, operations, new_ids
