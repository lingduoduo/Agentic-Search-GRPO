"""Deterministic, offline corpus hierarchy construction and snapshot maintenance."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .hierarchy_descriptions import (
    DESCRIPTION_POLICY_VERSION,
    assess_description,
    select_description_samples,
)
from .hierarchy_formats import first_sentence, normalize_document, split_passages


@dataclass
class HierarchyConfig:
    mode: str = "auto"
    target_cluster_size: int = 100
    max_cluster_size: int = 200
    n_clusters: int | None = None
    random_state: int = 42
    infer_layout: bool = False
    embedding_model: str = "intfloat/multilingual-e5-base"
    description_model: str | None = None
    description_sample_size: int = 12

    def __post_init__(self):
        if self.mode not in ("auto", "structure", "cluster"):
            raise ValueError("mode must be auto, structure or cluster")
        for name in (
            "target_cluster_size",
            "max_cluster_size",
            "n_clusters",
            "description_sample_size",
        ):
            value = getattr(self, name)
            if value is None and name == "n_clusters":
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.random_state) is not int or type(self.infer_layout) is not bool:
            raise ValueError("invalid random_state or infer_layout")
        if (
            not isinstance(self.embedding_model, str)
            or not self.embedding_model.strip()
        ):
            raise ValueError("embedding_model must be nonempty")
        if self.description_model is not None and (
            not isinstance(self.description_model, str)
            or not self.description_model.strip()
        ):
            raise ValueError("description_model must be nonempty or None")


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


@dataclass
class HierarchyArtifact:
    schema_version: int = 1
    config: dict = field(default_factory=dict)
    documents: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    assignments: dict = field(default_factory=dict)
    fingerprints: dict = field(default_factory=dict)
    vectors: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def validate(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported hierarchy schema_version")
        HierarchyConfig(**self.config)
        if not isinstance(self.documents, list) or not isinstance(self.groups, list):
            raise ValueError("documents and groups must be lists")
        for name in ("assignments", "fingerprints", "vectors", "diagnostics"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be an object")
        ids = []
        for doc in self.documents:
            if (
                not isinstance(doc, dict)
                or not isinstance(doc.get("id"), str)
                or not doc["id"]
            ):
                raise ValueError("invalid document id")
            for key in ("title", "contents", "url", "structure_source"):
                if not isinstance(doc.get(key), str):
                    raise ValueError(f"invalid document {key}")
            if (
                not isinstance(doc.get("metadata"), dict)
                or not isinstance(doc.get("sections"), list)
                or not isinstance(doc.get("structure_path"), list)
            ):
                raise ValueError("invalid normalized document")
            if any(not isinstance(p, str) or not p for p in doc["structure_path"]):
                raise ValueError("invalid structure_path")
            if any(
                not isinstance(s, dict)
                or not isinstance(s.get("heading"), str)
                or not isinstance(s.get("text"), str)
                for s in doc["sections"]
            ):
                raise ValueError("invalid sections")
            ids.append(doc["id"])
        id_set = set(ids)
        if (
            len(id_set) != len(ids)
            or set(self.assignments) != id_set
            or set(self.fingerprints) != id_set
        ):
            raise ValueError(
                "duplicate documents or incomplete assignments/fingerprints"
            )
        if any(
            not isinstance(v, str) or v != _hash(d)
            for d in self.documents
            for v in [self.fingerprints[d["id"]]]
        ):
            raise ValueError("invalid document fingerprint")
        groups = {}
        memberships = {}
        dim = None

        def check_vector(vector):
            nonlocal dim
            if (
                not isinstance(vector, list)
                or not vector
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in vector
                )
            ):
                raise ValueError("invalid finite vector")
            if dim is not None and len(vector) != dim:
                raise ValueError("inconsistent vector dimensions")
            dim = len(vector)

        for key, vector in self.vectors.items():
            if key not in id_set:
                raise ValueError("orphan vector")
            check_vector(vector)
        for group in self.groups:
            if (
                not isinstance(group, dict)
                or not isinstance(group.get("id"), str)
                or not group["id"]
                or group["id"] in groups
            ):
                raise ValueError("invalid or duplicate group id")
            for key in ("label", "description", "description_source"):
                if not isinstance(group.get(key), str) or not group[key].strip():
                    raise ValueError(f"invalid group {key}")
            if "parent_id" not in group or (
                group["parent_id"] is not None
                and not isinstance(group["parent_id"], str)
            ):
                raise ValueError("invalid parent_id")
            if not isinstance(group.get("document_ids"), list):
                raise ValueError("invalid membership")
            for docid in group["document_ids"]:
                if (
                    not isinstance(docid, str)
                    or docid not in id_set
                    or docid in memberships
                ):
                    raise ValueError("orphan or duplicate membership")
                memberships[docid] = group["id"]
            if "centroid" in group:
                check_vector(group["centroid"])
            groups[group["id"]] = group
        if memberships != self.assignments:
            raise ValueError("assignment membership disagreement")
        for gid in groups:
            seen = set()
            while gid is not None:
                if gid in seen or gid not in groups:
                    raise ValueError("group parent cycle or missing parent")
                seen.add(gid)
                gid = groups[gid]["parent_id"]
        descendants = _descendant_memberships(groups)
        for gid, group in groups.items():
            provenance = group.get("description_provenance")
            if provenance is not None:
                if not isinstance(provenance, dict):
                    raise ValueError("invalid description provenance")
                samples = provenance.get("document_ids")
                roles = provenance.get("roles")
                if (
                    not isinstance(samples, list)
                    or any(not isinstance(i, str) for i in samples)
                    or len(set(samples)) != len(samples)
                    or not set(samples) <= set(descendants[gid])
                    or not isinstance(roles, dict)
                    or set(roles) != set(samples)
                    or any(
                        not isinstance(v, list)
                        or not v
                        or any(not isinstance(r, str) or not r.strip() for r in v)
                        for v in roles.values()
                    )
                    or any(
                        not isinstance(provenance.get(k), str)
                        or not provenance[k].strip()
                        for k in ("selection_version", "generation_version")
                    )
                ):
                    raise ValueError("invalid description provenance")
            quality = group.get("description_quality")
            if quality is not None and (
                not isinstance(quality, dict)
                or type(quality.get("weak")) is not bool
                or not isinstance(quality.get("flags"), list)
                or any(not isinstance(f, str) for f in quality["flags"])
            ):
                raise ValueError("invalid description quality")
        json.dumps(asdict(self), allow_nan=False)
        return self

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    asdict(self),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        try:
            with open(path, encoding="utf-8") as stream:
                return cls(**json.load(stream)).validate()
        except (TypeError, KeyError, OverflowError) as error:
            raise ValueError(f"invalid hierarchy artifact: {error}") from error


def _descendant_memberships(groups):
    """Collect direct and descendant membership in one bottom-up traversal."""
    children = {gid: [] for gid in groups}
    roots = []
    for group in groups.values():
        if group["parent_id"] is None:
            roots.append(group["id"])
        else:
            children[group["parent_id"]].append(group["id"])
    result = {}
    stack = [(gid, False) for gid in roots]
    while stack:
        gid, visited = stack.pop()
        if not visited:
            stack.append((gid, True))
            stack.extend((child, False) for child in children[gid])
        else:
            members = list(groups[gid]["document_ids"])
            for child in children[gid]:
                members.extend(result[child])
            result[gid] = sorted(members)
    return result


def build_hierarchy(docs, config=None, *, embedder=None, describer=None, previous=None):
    if previous is not None:
        previous.validate()
    config = config or (
        HierarchyConfig(**previous.config) if previous else HierarchyConfig()
    )
    if not isinstance(config, HierarchyConfig):
        raise ValueError("config must be HierarchyConfig")
    config.__post_init__()
    for adapter, field_name in (
        (embedder, "embedding_model"),
        (describer, "description_model"),
    ):
        identity = getattr(adapter, "model_name", None)
        if identity is not None and identity != getattr(config, field_name):
            raise ValueError(
                f"adapter identity does not match config.{field_name}; declare the correct model for a full rebuild"
            )
    settings = asdict(config)
    if previous and asdict(HierarchyConfig(**previous.config)) != settings:
        raise ValueError(
            "hierarchy configuration changed; perform a full rebuild without previous"
        )
    normalized = [
        normalize_document(doc, infer_layout=config.infer_layout) for doc in docs
    ]
    normalized.sort(key=lambda d: d["id"])
    by_id = {d["id"]: d for d in normalized}
    if len(by_id) != len(normalized):
        raise ValueError("duplicate document ids")
    fingerprints = {d["id"]: _hash(d) for d in normalized}
    old_docs = {d["id"]: d for d in previous.documents} if previous else {}
    unchanged = {
        i for i in by_id if previous and previous.fingerprints.get(i) == fingerprints[i]
    }
    normalized = [
        copy.deepcopy(old_docs[d["id"]]) if d["id"] in unchanged else d
        for d in normalized
    ]
    by_id = {d["id"]: d for d in normalized}
    semantic = sorted(
        i
        for i, d in by_id.items()
        if config.mode == "cluster"
        or (config.mode == "auto" and not d["structure_path"])
    )
    semantic_set = set(semantic)
    vectors = (
        {
            i: copy.deepcopy(previous.vectors[i])
            for i in unchanged & semantic_set
            if i in previous.vectors
        }
        if previous
        else {}
    )
    pending = [i for i in semantic if i not in vectors]
    if pending:
        if embedder is None:
            raise ValueError("semantic documents require an embedder")
        encoded = np.asarray(
            embedder.encode(
                [
                    f"{by_id[i]['title']}\n{first_sentence(by_id[i]['contents'])}"
                    for i in pending
                ]
            ),
            dtype=float,
        )
        if (
            encoded.ndim != 2
            or encoded.shape[0] != len(pending)
            or encoded.shape[1] == 0
            or not np.isfinite(encoded).all()
        ):
            raise ValueError("embedder must return finite two-dimensional vectors")
        norms = np.linalg.norm(encoded, axis=1)
        if np.any(norms == 0):
            raise ValueError("embedder returned zero vector")
        encoded = encoded / norms[:, None]
        vectors.update({i: vector.tolist() for i, vector in zip(pending, encoded)})
    if len({len(v) for v in vectors.values()}) > 1:
        raise ValueError("embedding dimension changed; full rebuild required")
    groups = {}
    assignments = {}
    notices = []

    def add_group(gid, parent, label, members, centroid=None):
        group = dict(
            id=gid,
            parent_id=parent,
            label=label,
            description="",
            description_source="",
            document_ids=sorted(members),
        )
        if centroid is not None:
            group["centroid"] = centroid
        groups[gid] = group
        assignments.update({i: gid for i in members})

    for i, doc in by_id.items():
        if i in semantic_set:
            continue
        path = doc["structure_path"] or ["Documents"]
        parent = None
        for depth, label in enumerate(path):
            gid = "structure-" + _hash(path[: depth + 1])[:20]
            if gid not in groups:
                add_group(gid, parent, label, [])
            parent = gid
        groups[parent]["document_ids"].append(i)
        assignments[i] = parent
        if not doc["structure_path"]:
            notices.append("Unstructured documents use the Documents fallback group.")

    def partition(members, prefix=None):
        members = sorted(members)
        if not members:
            return
        count = min(
            len(members),
            config.n_clusters or math.ceil(len(members) / config.target_cluster_size),
        )
        from sklearn.cluster import KMeans

        with warnings.catch_warnings(record=True) as caught:
            labels = KMeans(
                n_clusters=count, random_state=config.random_state, n_init=10
            ).fit_predict(np.array([vectors[i] for i in members]))
        if caught:
            notices.append("Degenerate semantic vectors reduced distinct clusters.")
        bucket_map = {}
        for member, label in zip(members, labels):
            bucket_map.setdefault(int(label), []).append(member)
        buckets = [bucket_map[label] for label in sorted(bucket_map)]
        for bucket in buckets:
            if len(bucket) > config.max_cluster_size:
                notices.append(
                    "Oversized semantic cluster split into deterministic balanced chunks; routing quality may be reduced."
                )
            count_parts = math.ceil(len(bucket) / config.max_cluster_size)
            for part in np.array_split(np.array(bucket, dtype=object), count_parts):
                chunk = part.tolist()
                gid = "semantic-" + _hash(chunk)[:20]
                center = np.mean([vectors[i] for i in chunk], axis=0)
                norm = np.linalg.norm(center)
                center = center / norm if norm else center
                add_group(gid, prefix, "Cluster " + gid[-8:], chunk, center.tolist())

    if previous and semantic:
        old_parent_ids = {g["parent_id"] for g in previous.groups}
        old_semantic = [
            g
            for g in previous.groups
            if "centroid" in g and g["id"] not in old_parent_ids
        ]
        if old_semantic:
            buckets = {g["id"]: [] for g in old_semantic}
            for i in semantic:
                old_gid = previous.assignments.get(i) if i in unchanged else None
                if old_gid not in buckets:
                    old_gid = max(
                        old_semantic,
                        key=lambda g: float(np.dot(vectors[i], g["centroid"])),
                    )["id"]
                buckets[old_gid].append(i)
            for old in old_semantic:
                members = buckets[old["id"]]
                if len(members) > config.max_cluster_size:
                    partition(members)
                elif members:
                    center = np.mean([vectors[i] for i in members], axis=0)
                    norm = np.linalg.norm(center)
                    add_group(
                        old["id"],
                        None,
                        old["label"],
                        members,
                        (center / norm if norm else center).tolist(),
                    )
        else:
            partition(semantic)
    else:
        partition(semantic)
    old_groups = {g["id"]: g for g in previous.groups} if previous else {}

    generation_settings = dict(
        sample_size=config.description_sample_size,
        adapter=type(describer).__module__ + "." + type(describer).__qualname__
        if describer
        else "extractive",
    )
    # sample_size comes from config: it is passed explicitly to the describer.
    for name in ("model_name", "max_context_tokens", "max_new_tokens"):
        value = getattr(describer, name, None)
        if value is not None:
            generation_settings[name] = value
    generation_signature = _hash(generation_settings)
    memberships = _descendant_memberships(groups)
    old_memberships = _descendant_memberships(old_groups)
    for gid, group in groups.items():
        group["document_ids"].sort()
        members = memberships[gid]
        old = old_groups.get(gid)
        old_members = old_memberships.get(gid, [])
        if (
            old
            and members == old_members
            and all(i in unchanged for i in members)
            and (old.get("description_provenance") or {}).get("selection_version")
            == DESCRIPTION_POLICY_VERSION
            and old["description_provenance"].get("generation_signature")
            == generation_signature
        ):
            for key in (
                "description",
                "description_source",
                "description_provenance",
                "description_quality",
            ):
                if key in old:
                    group[key] = copy.deepcopy(old[key])
        else:
            documents = [copy.deepcopy(by_id[i]) for i in members]
            samples = select_description_samples(
                documents, vectors=vectors, limit=config.description_sample_size
            )
            selected = [copy.deepcopy(by_id[s["document_id"]]) for s in samples]
            provenance = dict(
                document_ids=[s["document_id"] for s in samples],
                roles={s["document_id"]: s["roles"] for s in samples},
                selection_version=DESCRIPTION_POLICY_VERSION,
                generation_version="custom-text-v1" if describer else "extractive-v1",
                warnings=[],
            )
            if describer and hasattr(describer, "describe_with_provenance"):
                result = describer.describe_with_provenance(
                    group["label"],
                    documents,
                    vectors=vectors,
                    sample_size=config.description_sample_size,
                )
                description = result["text"]
                provenance = {k: v for k, v in result.items() if k != "text"}
            elif describer:
                description = describer.describe(group["label"], selected)
            else:
                quota = max(1, (1600 - 2 * len(selected)) // max(1, len(selected)))
                contributing = [
                    (
                        d["id"],
                        (d["title"] or first_sentence(d["contents"])).strip()[:quota],
                    )
                    for d in selected
                ]
                contributing = [(i, t) for i, t in contributing if t]
                description = "; ".join(t for _, t in contributing) or group["label"]
                provenance["document_ids"] = [i for i, _ in contributing]
                provenance["roles"] = {
                    i: provenance["roles"][i] for i, _ in contributing
                }
            if not isinstance(description, str) or not description.strip():
                raise ValueError("describer must return nonempty text")
            group["description"] = description.strip()
            group["description_source"] = (
                getattr(describer, "description_source", "llm")
                if describer
                else "extractive"
            )
            provenance["generation_signature"] = generation_signature
            provenance["generation_settings"] = generation_settings
            provenance.setdefault("warnings", [])
            group["description_provenance"] = provenance
        if group["description_provenance"].get("warnings"):
            notices.append(
                "Description provenance records context or sampling coverage losses; inspect group provenance."
            )
        group["description_quality"] = assess_description(group["description"])
        if any(
            "coverage_limited:language" in roles
            for roles in group["description_provenance"]["roles"].values()
        ):
            notices.append(
                "Description sampling budget cannot cover every language stratum."
            )
    if any(g["description_source"] == "extractive" for g in groups.values()):
        notices.append(
            "Extractive fallback descriptions use document titles; routing quality is not evaluated."
        )
    added = len(set(by_id) - set(old_docs))
    deleted = len(set(old_docs) - set(by_id))
    changed = len(set(by_id) & set(old_docs) - unchanged)
    churn = (added + deleted + changed) / max(1, len(old_docs)) if previous else 0
    depths = []
    for g in groups.values():
        depth, parent = 1, g["parent_id"]
        while parent:
            depth += 1
            parent = groups[parent]["parent_id"]
        depths.append(depth)
    sizes = [len(g["document_ids"]) for g in groups.values()]
    nonempty_sizes = [size for size in sizes if size]
    imbalance = max(nonempty_sizes) / min(nonempty_sizes) if nonempty_sizes else 0.0
    passage_settings = {"max_words": 200, "max_chars": 1200}
    max_document_passages = 0
    max_section_passages = 0
    for document in normalized:
        section_counts = [
            len(split_passages(section["text"], **passage_settings))
            for section in document["sections"]
        ]
        max_document_passages = max(max_document_passages, sum(section_counts))
        max_section_passages = max(max_section_passages, max(section_counts, default=0))
    if max_document_passages > 60:
        notices.append(
            "A document exceeds 60 passages using the report's estimate settings; inspect section boundaries and retrieval fanout. Server passage settings may differ."
        )
    if imbalance >= 5:
        notices.append(
            "Direct group membership is imbalanced (largest/smallest nonempty group >= 5); inspect routing granularity. This diagnostic threshold is not an optimality claim."
        )
    if any(size > config.max_cluster_size for size in sizes):
        notices.append(
            "A structural group exceeds max_cluster_size; inspect source hierarchy granularity. The hard cluster-size bound applies to semantic groups."
        )
    diagnostics = dict(
        weak_description_count=sum(
            g["description_quality"]["weak"] for g in groups.values()
        ),
        description_policy_version=DESCRIPTION_POLICY_VERSION,
        document_count=len(by_id),
        group_count=len(groups),
        group_sizes=sizes,
        max_group_size=max(sizes, default=0),
        group_size_max=max(sizes, default=0),
        group_size_min=min(sizes, default=0),
        group_size_mean=sum(sizes) / len(sizes) if sizes else 0.0,
        singleton_group_count=sum(size == 1 for size in sizes),
        imbalance_ratio=imbalance,
        passage_estimate_settings=passage_settings,
        estimated_max_passages_per_document=max_document_passages,
        estimated_max_passages_per_section=max_section_passages,
        max_depth=max(depths, default=0),
        empty_titles=sum(not d["title"].strip() for d in normalized),
        structure_sources=dict(Counter(d["structure_source"] for d in normalized)),
        section_sources=dict(
            Counter(d.get("section_source", "plain") for d in normalized)
        ),
        heuristic_documents=sum(
            d.get("section_source") == "layout" for d in normalized
        ),
        new_documents=added,
        deleted_documents=deleted,
        changed_documents=changed,
        reused_documents=len(unchanged),
        oversize_groups=sum(size > config.max_cluster_size for size in sizes),
        churn=churn,
        rebuild_recommended=bool(previous and churn >= 0.3),
        warnings=sorted(set(notices)),
        max_section_count=max((len(d["sections"]) for d in normalized), default=0),
    )
    if diagnostics["rebuild_recommended"]:
        diagnostics["warnings"].append(
            "Significant corpus churn: perform a full rebuild to reconsider semantic cluster boundaries."
        )
    return HierarchyArtifact(
        config=settings,
        documents=normalized,
        groups=sorted(groups.values(), key=lambda g: g["id"]),
        assignments=assignments,
        fingerprints=fingerprints,
        vectors=vectors,
        diagnostics=diagnostics,
    ).validate()
