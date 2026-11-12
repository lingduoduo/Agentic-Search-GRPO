"""Read-only knowledge-base API over the corpora registered in data/corpora.json.

``/api/datasets`` pages the registered corpora, ``/api/datasets/{name}`` is one
corpus, and ``/api/datasets/{name}/documents`` pages the documents in its .jsonl
file. All admin only: the responses carry server paths. Nothing here writes;
corpora are built and registered offline, and the retriever loads them at start.
"""

from __future__ import annotations

import json
import math
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.internal.configs import AppSettings
from src.internal.servers._auth import make_require_admin
from src.internal.servers.retrieval import corpus_registry
from src.internal.servers.tools.api import Paginator


class DatasetView(BaseModel):
    name: str
    path: str | None
    docs: int | None
    domain: str | None
    # The label results from this corpus cite.
    source: str
    # Its file exists on the server.
    available: bool
    # Loaded by the in-process retriever; None when a separate retrieval
    # server answers, since its corpus is not known here.
    served: bool | None


class DatasetPage(BaseModel):
    list: list[DatasetView]
    paginator: Paginator


class DatasetDocumentView(BaseModel):
    id: str | None
    title: str | None
    contents: str | None
    source: str | None
    acl: list[str]


class DatasetDocumentPage(BaseModel):
    list: list[DatasetDocumentView]
    paginator: Paginator


def _page(items: list, current_page: int, page_size: int) -> tuple[list, Paginator]:
    start = (current_page - 1) * page_size
    return items[start : start + page_size], Paginator(
        current_page=current_page,
        page_size=page_size,
        total_page=math.ceil(len(items) / page_size),
        total_record=len(items),
    )


def create_datasets_router(app_settings: AppSettings) -> APIRouter:
    router = APIRouter(
        prefix="/api/datasets",
        dependencies=[Depends(make_require_admin(app_settings))],
    )

    def manifest() -> dict:
        try:
            return corpus_registry.load_manifest()
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail="The corpus manifest could not be parsed."
            ) from exc

    def served_names(names: set[str]) -> set[str] | None:
        spec = app_settings.services.embedded_retrieval
        if spec is None:
            return None
        if spec == "all":
            return names
        return {name.strip() for name in spec.split(",")} & names

    def datasets() -> list[DatasetView]:
        entries = manifest()
        served = served_names(set(entries))
        views = []
        for name, entry in sorted(entries.items()):
            # An entry is a dict, or just a path in hand-written manifests.
            entry = entry if isinstance(entry, dict) else {"path": entry}
            path = entry.get("path")
            views.append(
                DatasetView(
                    name=name,
                    path=path,
                    docs=entry.get("docs"),
                    domain=entry.get("domain"),
                    source=entry.get("source") or name,
                    available=bool(path) and os.path.exists(path),
                    served=None if served is None else name in served,
                )
            )
        return views

    def dataset(name: str) -> DatasetView:
        for view in datasets():
            if view.name == name:
                return view
        raise HTTPException(status_code=404, detail="Dataset not found.")

    @router.get("", response_model=DatasetPage)
    def list_datasets(
        current_page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        search_word: str = "",
    ) -> DatasetPage:
        word = search_word.strip().lower()
        matches = [
            d
            for d in datasets()
            if word in f"{d.name} {d.source} {d.domain or ''}".lower()
        ]
        items, paginator = _page(matches, current_page, page_size)
        return DatasetPage(list=items, paginator=paginator)

    @router.get("/{name}", response_model=DatasetView)
    def get_dataset(name: str) -> DatasetView:
        return dataset(name)

    @router.get("/{name}/documents", response_model=DatasetDocumentPage)
    def list_documents(
        name: str,
        current_page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        search_word: str = "",
    ) -> DatasetDocumentPage:
        view = dataset(name)
        if not view.available or view.path is None:
            raise HTTPException(
                status_code=409, detail="The dataset's file is missing on the server."
            )
        word = search_word.strip().lower()
        documents = []
        with open(view.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                meta = doc.get("metadata") or {}
                title = doc.get("title")
                contents = doc.get("contents")
                if word and word not in f"{title or ''} {contents or ''}".lower():
                    continue
                documents.append(
                    DatasetDocumentView(
                        id=None if doc.get("id") is None else str(doc["id"]),
                        title=title,
                        contents=contents,
                        # Per-document source wins over the corpus label, as at
                        # search time.
                        source=doc.get("source") or meta.get("source") or view.source,
                        acl=list(meta.get("acl") or []),
                    )
                )
        items, paginator = _page(documents, current_page, page_size)
        return DatasetDocumentPage(list=items, paginator=paginator)

    return router
