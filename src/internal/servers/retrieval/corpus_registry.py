"""Corpus registry: resolve a corpus spec to a list of documents.

A spec is one of:
  - a registered name from data/corpora.json (e.g. "demo")
  - "all" — the union of every registered corpus. A registered corpus loaded
    more than once (e.g. "a,a") collapses its repeat, but an id shared by two
    *different* corpora is a collision and is rejected, since merging would
    conflate distinct documents under one id.
  - a comma-separated list of names (e.g. "demo,scifact")
  - a filesystem path to a .jsonl corpus (back-compat with --corpus_path)

``register_corpus`` is the write side: a tool that produces a corpus file
calls it so the corpus is reachable by name, not just by path.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = "data/corpora.json"


def _apply_source(doc: dict, corpus_source: str | None) -> dict:
    """Ensure a document carries a human-readable ``metadata.source``.

    Precedence: an explicit per-document source (a top-level ``source`` field or
    an existing ``metadata.source``) wins over the corpus-level default label
    from the manifest. A document with neither is returned untouched, so the web
    backend can still fall back to its provider label ("Local Retrieval").
    """
    meta = dict(doc.get("metadata") or {})
    source = doc.get("source") or meta.get("source") or corpus_source
    if source is None:
        return doc
    meta["source"] = source
    return {**doc, "metadata": meta}


def load_manifest(path: str = DEFAULT_MANIFEST_PATH) -> dict[str, dict]:
    """Load the corpus manifest; a missing file yields an empty manifest so
    path-only (--corpus_path) usage keeps working with no manifest present."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def register_corpus(
    name: str,
    *,
    corpus_path: str,
    doc_count: int,
    domain: str | None = None,
    source: str | None = None,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> dict:
    """Add or update *name* in the manifest, and return the entry written.

    Writing a corpus file is not enough to make ``--corpus <name>`` resolve; the
    manifest is the half that maps a name to a path. This is the write side of
    :func:`load_manifest`, kept here because the registry owns the entry format.

    Re-registering an existing name replaces that entry -- re-running a
    converter without ``--limit`` must correct the document count rather than
    leave a stale one behind -- and leaves every other entry untouched.
    """
    entry = {
        "path": corpus_path,
        "docs": doc_count,
        "domain": domain,
        "source": source or name,
    }

    manifest: dict = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as exc:
            # Overwriting would destroy hand-registered corpora that are only
            # unreachable because of a typo. Report it and change nothing.
            raise ValueError(
                f"Corpus manifest {manifest_path!r} could not be parsed: {exc}. "
                f"Fix the file (or move it aside) and re-run; it was left "
                f"unchanged."
            ) from exc

    manifest[name] = entry
    parent = os.path.dirname(manifest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(
        "Registered corpus %r -> %s (%d docs) in %s",
        name,
        corpus_path,
        doc_count,
        manifest_path,
    )
    return entry


def unregister_corpus(name: str, *, manifest_path: str = DEFAULT_MANIFEST_PATH) -> bool:
    """Remove *name* from the manifest; the corpus file is left alone.

    Returns whether an entry was removed. Like :func:`register_corpus`, an
    unparseable manifest is reported rather than overwritten.
    """
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Corpus manifest {manifest_path!r} could not be parsed: {exc}. "
            f"It was left unchanged."
        ) from exc
    if name not in manifest:
        return False
    del manifest[name]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("Unregistered corpus %r from %s", name, manifest_path)
    return True


def resolve_corpus_docs(spec: str, manifest: dict | None = None) -> list[dict]:
    # Imported here so reading the manifest (the web app's dataset list) does
    # not load scikit-learn through the demo server module.
    from src.internal.servers.retrieval.demo import _load_corpus

    if manifest is None:
        manifest = load_manifest()

    if not spec:
        raise ValueError(
            "Empty corpus spec; pass a corpus name, 'all', or a file path."
        )

    if spec == "all":
        names = list(manifest.keys())
        if not names:
            raise ValueError("No corpora registered in the manifest for 'all'.")
    else:
        candidate = [s.strip() for s in spec.split(",") if s.strip()]
        if candidate and all(n in manifest for n in candidate):
            names = candidate
        elif os.path.exists(spec):
            docs = [_apply_source(d, None) for d in _load_corpus(spec)]
            logger.info("Loaded corpus from path %s (%d docs)", spec, len(docs))
            return docs
        else:
            available = ", ".join(sorted(manifest)) or "(none)"
            raise ValueError(
                f"Unknown corpus spec {spec!r}. Available: {available}, or a file path."
            )

    docs: list[dict] = []
    id_source: dict[object, str] = {}
    loaded: list[str] = []
    for name in names:
        entry = manifest[name]
        path = entry["path"] if isinstance(entry, dict) else entry
        corpus_source = entry.get("source") if isinstance(entry, dict) else None
        if not os.path.exists(path):
            logger.warning(
                "Corpus %r file %r is missing; skipping (regenerate via beir_to_corpus.py)",
                name,
                path,
            )
            continue
        loaded.append(name)
        for doc in _load_corpus(path):
            doc_id = doc.get("id")
            if doc_id is not None and doc_id in id_source:
                prior = id_source[doc_id]
                if prior == name:
                    # Same corpus loaded more than once (e.g. "a,a") or an
                    # internal duplicate — collapse the repeat.
                    continue
                # Distinct corpora sharing an id would conflate unrelated docs
                # under one citation. Fail loud rather than silently drop one.
                raise ValueError(
                    f"Corpus id collision: id {doc_id!r} appears in both "
                    f"{prior!r} and {name!r}. Combining these corpora would "
                    f"conflate distinct documents; give them disjoint ids "
                    f"(namespace by corpus) before merging."
                )
            if doc_id is not None:
                id_source[doc_id] = name
            docs.append(_apply_source(doc, corpus_source))
    if not docs:
        raise ValueError(
            f"None of the requested corpora {names} have files on disk. "
            "Regenerate them via beir_to_corpus.py or provide a valid corpus path."
        )
    logger.info("Loaded corpora %s (%d docs)", loaded, len(docs))
    return docs
