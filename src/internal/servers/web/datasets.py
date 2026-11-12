"""Knowledge-base API: the registered corpora, and management of the ones kept
in the store.

A corpus is *registered* when data/corpora.json names it, which is how both
retrievers find it. It is *managed* when its documents live in the store
(``datasets`` / ``dataset_documents``): only managed datasets can be edited.
After every change a managed dataset's enabled documents are written to
``data/datasets/<name>.jsonl`` and registered under its name, so search reads
them the same way it reads any corpus. A registered corpus becomes managed by
importing it, which copies its documents into the store; its file is left alone.

When the web app serves retrieval itself (AGENTIC_SEARCH_EMBEDDED_RETRIEVAL),
``reload_retrieval`` rebuilds that index after a change to a corpus it serves. A
separate retrieval server picks changes up when it restarts.

All routes are admin only: the responses carry server paths.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
from collections.abc import Callable

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from pydantic import BaseModel, Field

from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.db.models import (
    DatasetDocumentRecord,
    DatasetRecord,
    DatasetUploadRecord,
)
from src.internal.servers._auth import make_require_admin
from src.internal.servers.retrieval import corpus_registry
from src.internal.servers.tools.api import Paginator
from src.internal.servers.web.dataset_files import (
    ALLOWED_EXTENSIONS,
    UploadError,
    extension,
    parse_upload,
)

logger = logging.getLogger(__name__)

#: Where managed datasets are exported for the retrievers.
DATASETS_DIR = "data/datasets"

# The name is the corpus key (``--corpus``, the manifest, the export file name),
# so it is fixed at creation and kept to characters safe in all three.
NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"

FILE_MISSING = "The dataset's file is missing on the server."

#: Upload limits, checked before anything is stored.
MAX_UPLOAD_FILES = 20
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


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
    # Its documents live in the store and can be edited.
    managed: bool
    description: str | None
    updated_at: str | None


class DatasetPage(BaseModel):
    list: list[DatasetView]
    paginator: Paginator


class DatasetDocumentView(BaseModel):
    # The corpus id search results cite.
    id: str | None
    # The store's id, for the document routes; None for an unmanaged corpus.
    document_id: str | None
    title: str | None
    contents: str | None
    source: str | None
    acl: list[str]
    # Disabled documents stay in the store but are not exported to search.
    enabled: bool
    updated_at: str | None


class DatasetDocumentPage(BaseModel):
    list: list[DatasetDocumentView]
    paginator: Paginator


class CreateDatasetRequest(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str = Field(default="", max_length=2000)
    domain: str | None = Field(default=None, max_length=200)
    source: str | None = Field(default=None, max_length=200)


class UpdateDatasetRequest(BaseModel):
    """Only the fields sent change; a blank domain or source clears it."""

    description: str | None = Field(default=None, max_length=2000)
    domain: str | None = Field(default=None, max_length=200)
    source: str | None = Field(default=None, max_length=200)


class UpdateDocumentRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    enabled: bool | None = None


class UploadFileView(BaseModel):
    filename: str
    # waiting -> parsing -> completed | error
    status: str
    error: str | None
    document_count: int


class DocumentBatchView(BaseModel):
    batch: str
    files: list[UploadFileView]
    # Every file is completed or error.
    done: bool


def _batch_view(uploads: list[DatasetUploadRecord]) -> DocumentBatchView:
    return DocumentBatchView(
        batch=uploads[0].batch,
        files=[
            UploadFileView(
                filename=u.filename,
                status=u.status,
                error=u.error,
                document_count=u.document_count,
            )
            for u in uploads
        ],
        done=all(u.status in ("completed", "error") for u in uploads),
    )


def _page(items: list, current_page: int, page_size: int) -> tuple[list, Paginator]:
    start = (current_page - 1) * page_size
    return items[start : start + page_size], _paginator(
        len(items), current_page, page_size
    )


def _paginator(total: int, current_page: int, page_size: int) -> Paginator:
    return Paginator(
        current_page=current_page,
        page_size=page_size,
        total_page=math.ceil(total / page_size),
        total_record=total,
    )


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _clean(value: str | None) -> str | None:
    """Trimmed text, or None for blank."""
    value = (value or "").strip()
    return value or None


def create_datasets_router(
    app_settings: AppSettings,
    store: AgenticSearchStore,
    reload_retrieval: Callable[[], None] | None = None,
) -> APIRouter:
    router = APIRouter(
        prefix="/api/datasets",
        dependencies=[Depends(make_require_admin(app_settings))],
    )
    # One change's export and manifest write happen together.
    publish_lock = threading.Lock()

    # Read when called, so tests can point them elsewhere.
    def manifest_path() -> str:
        return corpus_registry.DEFAULT_MANIFEST_PATH

    def export_path(name: str) -> str:
        return os.path.join(DATASETS_DIR, f"{name}.jsonl")

    def manifest() -> dict:
        try:
            return corpus_registry.load_manifest(manifest_path())
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

    def view(
        name: str,
        entry: dict | str | None,
        record: DatasetRecord | None,
        served: set[str] | None,
    ) -> DatasetView:
        # An entry is a dict, or just a path in hand-written manifests.
        entry = entry if isinstance(entry, dict) else {"path": entry}
        path = entry.get("path") or (export_path(name) if record else None)
        if record is not None:
            docs: int | None = store.count_dataset_documents(record.id)
            domain, source = record.domain, record.source
        else:
            docs = entry.get("docs")
            domain, source = entry.get("domain"), entry.get("source")
        return DatasetView(
            name=name,
            path=path,
            docs=docs,
            domain=domain,
            source=source or name,
            available=bool(path) and os.path.exists(path),
            served=None if served is None else name in served,
            managed=record is not None,
            description=record.description if record else None,
            updated_at=record.updated_at if record else None,
        )

    def datasets() -> list[DatasetView]:
        entries = manifest()
        records = {r.name: r for r in store.list_datasets()}
        names = set(entries) | set(records)
        served = served_names(names)
        return [
            view(name, entries.get(name), records.get(name), served)
            for name in sorted(names)
        ]

    def dataset(name: str) -> DatasetView:
        entries = manifest()
        record = store.get_dataset_by_name(name)
        if record is None and name not in entries:
            raise HTTPException(status_code=404, detail="Dataset not found.")
        return view(name, entries.get(name), record, served_names({name}))

    def managed(name: str) -> DatasetRecord:
        record = store.get_dataset_by_name(name)
        if record is not None:
            return record
        if name in manifest():
            raise HTTPException(
                status_code=409, detail="Import this dataset before changing it."
            )
        raise HTTPException(status_code=404, detail="Dataset not found.")

    def managed_document(
        name: str, document_id: str
    ) -> tuple[DatasetRecord, DatasetDocumentRecord]:
        record = managed(name)
        doc = store.get_dataset_document(record.id, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return record, doc

    def document_view(
        doc: DatasetDocumentRecord, record: DatasetRecord
    ) -> DatasetDocumentView:
        meta = doc.metadata or {}
        return DatasetDocumentView(
            id=doc.doc_id or doc.id,
            document_id=doc.id,
            title=doc.title,
            contents=doc.contents,
            # Per-document source wins over the dataset label, as at search time.
            source=meta.get("source") or record.source or record.name,
            acl=list(meta.get("acl") or []),
            enabled=doc.enabled,
            updated_at=doc.updated_at,
        )

    def refresh_search(name: str) -> None:
        """Rebuild the in-process index when it serves *name*."""
        if reload_retrieval is None or not served_names({name}):
            return
        try:
            reload_retrieval()
        except Exception:
            # Search keeps the previous index; the change itself is saved.
            logger.exception("Rebuilding in-process retrieval after %r failed", name)

    def publish(record: DatasetRecord) -> None:
        """Export the enabled documents and register the file under the name."""
        with publish_lock:
            documents = store.enabled_dataset_documents(record.id)
            path = export_path(record.name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            partial = f"{path}.partial"
            with open(partial, "w") as f:
                for doc in documents:
                    row = {
                        "id": doc.doc_id or doc.id,
                        "title": doc.title,
                        "contents": doc.contents,
                        "metadata": doc.metadata,
                    }
                    f.write(json.dumps(row) + "\n")
            os.replace(partial, path)
            corpus_registry.register_corpus(
                record.name,
                corpus_path=path,
                doc_count=len(documents),
                domain=record.domain,
                source=record.source,
                manifest_path=manifest_path(),
            )
        refresh_search(record.name)

    def unpublish(record: DatasetRecord) -> None:
        """Remove the export. An imported dataset's original file, if it is still
        there, is registered again, so deleting never strands a corpus that
        search or AGENTIC_SEARCH_EMBEDDED_RETRIEVAL was configured with."""
        original = record.imported_from
        path = export_path(record.name)
        restore = (
            original is not None
            and os.path.abspath(original) != os.path.abspath(path)
            and os.path.exists(original)
        )
        with publish_lock:
            if restore:
                corpus_registry.register_corpus(
                    record.name,
                    corpus_path=original,  # type: ignore[arg-type]
                    doc_count=len(_read_jsonl(original)),
                    domain=record.domain,
                    source=record.source,
                    manifest_path=manifest_path(),
                )
            else:
                corpus_registry.unregister_corpus(
                    record.name, manifest_path=manifest_path()
                )
            if os.path.exists(path):
                os.remove(path)
        refresh_search(record.name)

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

    @router.post("", response_model=DatasetView, status_code=201)
    def create_dataset(body: CreateDatasetRequest) -> DatasetView:
        if body.name in manifest():
            raise HTTPException(status_code=409, detail="That name is taken.")
        try:
            record = store.create_dataset(
                body.name,
                description=body.description.strip(),
                domain=_clean(body.domain),
                source=_clean(body.source),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="That name is taken.") from exc
        publish(record)
        return dataset(record.name)

    @router.get("/{name}", response_model=DatasetView)
    def get_dataset(name: str) -> DatasetView:
        return dataset(name)

    @router.patch("/{name}", response_model=DatasetView)
    def update_dataset(name: str, body: UpdateDatasetRequest) -> DatasetView:
        record = managed(name)
        changes = body.model_dump(exclude_unset=True)
        if "description" in changes:
            changes["description"] = (changes["description"] or "").strip()
        for key in ("domain", "source"):
            if key in changes:
                changes[key] = _clean(changes[key])
        store.update_dataset(record.id, **changes)
        publish(managed(name))
        return dataset(name)

    @router.delete("/{name}", status_code=204)
    def delete_dataset(name: str) -> Response:
        record = managed(name)
        store.delete_dataset(record.id)
        unpublish(record)
        return Response(status_code=204)

    @router.post("/{name}/import", response_model=DatasetView)
    def import_dataset(name: str) -> DatasetView:
        """Copy a registered corpus's documents into the store to manage it."""
        if store.get_dataset_by_name(name) is not None:
            raise HTTPException(
                status_code=409, detail="This dataset is already managed."
            )
        current = dataset(name)
        if not current.available or current.path is None:
            raise HTTPException(status_code=409, detail=FILE_MISSING)
        documents = _read_jsonl(current.path)
        entry = manifest().get(name)
        record = store.create_dataset(
            name,
            domain=current.domain,
            source=entry.get("source") if isinstance(entry, dict) else None,
            imported_from=current.path,
        )
        store.add_dataset_documents(record.id, documents)
        publish(record)
        return dataset(name)

    @router.get("/{name}/documents", response_model=DatasetDocumentPage)
    def list_documents(
        name: str,
        current_page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        search_word: str = "",
    ) -> DatasetDocumentPage:
        record = store.get_dataset_by_name(name)
        if record is not None:
            docs, total = store.list_dataset_documents(
                record.id,
                offset=(current_page - 1) * page_size,
                limit=page_size,
                search_word=search_word,
            )
            return DatasetDocumentPage(
                list=[document_view(doc, record) for doc in docs],
                paginator=_paginator(total, current_page, page_size),
            )
        current = dataset(name)
        if not current.available or current.path is None:
            raise HTTPException(status_code=409, detail=FILE_MISSING)
        word = search_word.strip().lower()
        documents = []
        for doc in _read_jsonl(current.path):
            meta = doc.get("metadata") or {}
            title = doc.get("title")
            contents = doc.get("contents")
            if word and word not in f"{title or ''} {contents or ''}".lower():
                continue
            documents.append(
                DatasetDocumentView(
                    id=None if doc.get("id") is None else str(doc["id"]),
                    document_id=None,
                    title=title,
                    contents=contents,
                    # Per-document source wins over the corpus label, as at
                    # search time.
                    source=doc.get("source") or meta.get("source") or current.source,
                    acl=list(meta.get("acl") or []),
                    enabled=True,
                    updated_at=None,
                )
            )
        items, paginator = _page(documents, current_page, page_size)
        return DatasetDocumentPage(list=items, paginator=paginator)

    def process_uploads(
        dataset_id: str, name: str, files: list[tuple[str, str, bytes]]
    ) -> None:
        """Parse each file into documents, then publish the dataset once."""
        added = 0
        for upload_id, filename, data in files:
            store.set_dataset_upload_status(upload_id, "parsing")
            try:
                documents = parse_upload(filename, data)
                store.add_dataset_documents(dataset_id, documents)
            except UploadError as exc:
                store.set_dataset_upload_status(upload_id, "error", error=str(exc))
                continue
            except Exception:
                logger.exception("Processing upload %r for %r failed", filename, name)
                store.set_dataset_upload_status(
                    upload_id, "error", error="The file could not be processed."
                )
                continue
            store.set_dataset_upload_status(
                upload_id, "completed", document_count=len(documents)
            )
            added += len(documents)
        record = store.get_dataset_by_name(name)
        # The dataset may have been deleted while the files were processed.
        if added and record is not None and record.id == dataset_id:
            publish(record)

    @router.post("/{name}/documents", response_model=DocumentBatchView, status_code=202)
    async def upload_documents(
        name: str,
        background: BackgroundTasks,
        files: list[UploadFile] = File(...),
    ) -> DocumentBatchView:
        """Accept files and process them in the background; poll the batch."""
        record = managed(name)
        if len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(
                status_code=422,
                detail=f"Upload at most {MAX_UPLOAD_FILES} files at a time.",
            )
        received: list[tuple[str, bytes]] = []
        for upload in files:
            filename = os.path.basename(upload.filename or "") or "upload"
            if extension(filename) not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=422,
                    detail=f"{filename}: use .jsonl, .txt or .md files.",
                )
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{filename} is over the 10 MB limit.",
                )
            received.append((filename, data))
        uploads = store.create_dataset_upload_batch(
            record.id, [filename for filename, _ in received]
        )
        background.add_task(
            process_uploads,
            record.id,
            name,
            [(u.id, f, d) for u, (f, d) in zip(uploads, received, strict=True)],
        )
        return _batch_view(uploads)

    @router.get("/{name}/documents/batch/{batch}", response_model=DocumentBatchView)
    def get_upload_batch(name: str, batch: str) -> DocumentBatchView:
        record = managed(name)
        uploads = store.list_dataset_upload_batch(record.id, batch)
        if not uploads:
            raise HTTPException(status_code=404, detail="Upload batch not found.")
        return _batch_view(uploads)

    @router.get("/{name}/documents/{document_id}", response_model=DatasetDocumentView)
    def get_document(name: str, document_id: str) -> DatasetDocumentView:
        record, doc = managed_document(name, document_id)
        return document_view(doc, record)

    @router.patch("/{name}/documents/{document_id}", response_model=DatasetDocumentView)
    def update_document(
        name: str, document_id: str, body: UpdateDocumentRequest
    ) -> DatasetDocumentView:
        record, _ = managed_document(name, document_id)
        title = None if body.title is None else _clean(body.title)
        store.update_dataset_document(
            record.id, document_id, title=title, enabled=body.enabled
        )
        publish(record)
        record, doc = managed_document(name, document_id)
        return document_view(doc, record)

    @router.delete("/{name}/documents/{document_id}", status_code=204)
    def delete_document(name: str, document_id: str) -> Response:
        record, _ = managed_document(name, document_id)
        store.delete_dataset_document(record.id, document_id)
        publish(record)
        return Response(status_code=204)

    return router
