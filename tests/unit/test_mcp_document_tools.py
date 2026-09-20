"""Tests for MCP document extraction input and TXT handling."""

import ast
import asyncio
import base64
import concurrent.futures
import io
import json
import os
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import pytest

from src.internal.mcp_server.tools import documents
from src.internal.mcp_server import document_parser_runtime
from tests.unit import _document_parser_probes as parser_probes

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_python_310_unit_test_requirements_include_tomli():
    """Python 3.10 test runs install the TOML parser used by this contract test."""
    requirements = Path("requirements-unit-test.txt").read_text().splitlines()

    assert any(requirement.startswith("tomli") for requirement in requirements)


def test_document_parser_extra_declares_all_optional_parsers():
    """Installing document extraction support provides every parser it needs."""
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    requirements = project["optional-dependencies"]["mcp-documents"]

    assert any(item.startswith("pypdf>=6.12.2") for item in requirements)
    assert any(item.startswith("psutil") for item in requirements)
    assert any(item.startswith("python-docx") for item in requirements)
    assert any(item.startswith("python-pptx") for item in requirements)


def test_unit_test_requirements_install_document_parsers():
    """A clean unit-test install exercises every supported parser format."""
    requirements = Path("requirements-unit-test.txt").read_text().splitlines()

    assert any(requirement.startswith("pypdf>=6.12.2") for requirement in requirements)
    assert any(requirement.startswith("psutil") for requirement in requirements)
    assert any(requirement.startswith("python-docx") for requirement in requirements)
    assert any(requirement.startswith("python-pptx") for requirement in requirements)


def _imports_documents(module_path: Path, from_module: str | None) -> bool:
    """Return whether a module explicitly imports the document tool module."""
    tree = ast.parse(module_path.read_text())
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == from_module
        and any(alias.name == "documents" for alias in node.names)
        for node in ast.walk(tree)
    )


def test_document_tool_is_registered_by_mcp_startup_modules():
    """Startup wiring imports the decorated tool rather than relying on test imports."""
    server_dir = Path(documents.__file__).resolve().parents[1]

    assert _imports_documents(server_dir / "api.py", "tools")
    assert _imports_documents(server_dir / "tools" / "__init__.py", None)


@pytest.mark.asyncio
async def test_extract_document_is_registered():
    """The running MCP server exposes document extraction to clients."""
    from src.internal.mcp_server.api import mcp_server

    names = {tool.name for tool in await mcp_server.list_tools()}
    assert "extract_document" in names


@pytest.mark.asyncio
async def test_extract_document_runs_sync_parser_in_a_worker_thread(monkeypatch):
    """The async tool offloads blocking extraction without changing its response."""
    calls: list[tuple[object, tuple[object, ...]]] = []

    async def worker_spy(function, *args):
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", worker_spy)

    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"alpha").decode()
    )

    assert result["document"]["text"] == "alpha"
    assert calls == [
        (
            documents._extract_document_sync,
            ("notes.txt", b"alpha", None, 1000),
        )
    ]


@pytest.mark.asyncio
async def test_extract_document_keeps_docx_extraction_memory_backed(monkeypatch):
    """DOCX parsing consumes bytes in memory and never creates a temporary file."""
    created: list[object] = []

    def named_temporary_file(*args, **kwargs):
        created.append((args, kwargs))
        raise AssertionError("document extraction must not create temporary files")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", named_temporary_file)

    result = await documents.extract_document("notes.docx", _docx_payload())

    assert result["document"]["paragraphs"] == ["paragraph text"]
    assert created == []


def _docx_payload() -> str:
    """Create a small DOCX payload with paragraph and table content."""
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("paragraph text")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "left"
    table.cell(0, 1).text = "right"
    output = io.BytesIO()
    document.save(output)
    return base64.b64encode(output.getvalue()).decode()


def _pptx_payload(slide_texts: tuple[str, ...] = ("slide text", "")) -> str:
    """Create a small PPTX payload with optional text on each slide."""
    pptx = pytest.importorskip("pptx")
    presentation = pptx.Presentation()
    for text in slide_texts:
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        if text:
            slide.shapes.add_textbox(0, 0, 100, 100).text = text
    output = io.BytesIO()
    presentation.save(output)
    return base64.b64encode(output.getvalue()).decode()


@pytest.mark.parametrize(
    ("raw", "total", "expected"),
    [
        ("1", 5, [0]),
        ("1-3,5", 5, [0, 1, 2, 4]),
        ("3,1,3", 5, [0, 2]),
        ("1-99", 3, [0, 1, 2]),
        ("99", 3, []),
    ],
)
def test_parse_page_range(raw: str, total: int, expected: list[int]):
    """PDF page ranges are one-based, sorted, deduplicated, and bounded."""
    assert documents.parse_page_range(raw, total) == expected


@pytest.mark.parametrize("page_range", ["", "0", "-1", "3-1", "1-", "1--2", "a"])
def test_parse_page_range_rejects_malformed_ranges(page_range: str):
    """Malformed PDF page ranges are rejected before extraction."""
    with pytest.raises(ValueError):
        documents.parse_page_range(page_range, 3)


def test_parse_page_range_rejects_empty_documents():
    """Page selection is invalid for a PDF with no pages."""
    with pytest.raises(ValueError):
        documents.parse_page_range("1", 0)


def _two_page_pdf() -> bytes:
    """Build a small text PDF without a non-PDF test dependency."""
    pypdf = pytest.importorskip("pypdf")
    generic = pypdf.generic
    writer = pypdf.PdfWriter()

    for text in ("alpha", "beta"):
        writer.add_blank_page(width=200, height=200)
        page = writer.pages[-1]
        resources = generic.DictionaryObject()
        font = generic.DictionaryObject(
            {
                generic.NameObject("/Type"): generic.NameObject("/Font"),
                generic.NameObject("/Subtype"): generic.NameObject("/Type1"),
                generic.NameObject("/BaseFont"): generic.NameObject("/Helvetica"),
            }
        )
        resources[generic.NameObject("/Font")] = generic.DictionaryObject(
            {generic.NameObject("/F1"): writer._add_object(font)}
        )
        page[generic.NameObject("/Resources")] = resources
        stream = generic.DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
        page[generic.NameObject("/Contents")] = writer._add_object(stream)

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_extract_pdf_returns_selected_labeled_pages():
    """PDF extraction returns bounded labeled text and page-count metadata."""
    result = documents._extract_pdf(_two_page_pdf(), "notes.pdf", "2")

    assert result == {
        "document": {
            "file_name": "notes.pdf",
            "file_type": "pdf",
            "text": "--- Page 2 ---\nbeta",
            "text_length": 19,
            "truncated": False,
            "total_pages": 2,
            "extracted_pages": 1,
        }
    }


def test_extract_pdf_uses_the_maintained_pypdf_package(monkeypatch):
    """PDF extraction lazily imports the maintained package name."""
    imported: list[str] = []
    real_import = document_parser_runtime.importlib.import_module

    def record_import(module_name: str):
        imported.append(module_name)
        return real_import(module_name)

    monkeypatch.setattr(
        document_parser_runtime.importlib, "import_module", record_import
    )

    document_parser_runtime.extract_pdf(_two_page_pdf(), "notes.pdf", "1")

    assert imported
    assert set(imported) == {"pypdf"}


def test_extract_pdf_reports_missing_optional_dependency(monkeypatch):
    """Absent PDF support explains how to install the document extra."""

    def missing_module(module_name: str):
        raise ImportError(module_name)

    monkeypatch.setattr(
        document_parser_runtime.importlib, "import_module", missing_module
    )

    with pytest.raises(ImportError, match=r"agentic-search\[mcp-documents\]"):
        document_parser_runtime._require_module("pypdf", "pypdf")


@pytest.mark.asyncio
async def test_extract_document_decodes_txt():
    """TXT content is decoded and returned with its metadata."""
    result = await documents.extract_document(
        "notes.TXT", base64.b64encode(b"alpha\nbeta").decode()
    )

    assert result == {
        "document": {
            "file_name": "notes.TXT",
            "file_type": "txt",
            "text": "alpha\nbeta",
            "text_length": 10,
            "truncated": False,
        }
    }


@pytest.mark.asyncio
async def test_extract_document_rejects_page_ranges_for_non_pdf_documents():
    """Page ranges apply only to PDF documents."""
    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"alpha").decode(), page_range="1"
    )

    assert result["document"] is None
    assert "page range" in result["error"].lower()


@pytest.mark.parametrize(
    ("file_name", "payload", "error_fragment"),
    [
        ("", "YQ==", "file name"),
        ("no-extension", "YQ==", "extension"),
        ("data.bin", "YQ==", "unsupported"),
        ("data.txt", "%%%", "base64"),
        ("data.txt", "", "empty"),
    ],
)
@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_input(
    file_name: str, payload: str, error_fragment: str
):
    """Malformed requests receive bounded error responses."""
    result = await documents.extract_document(file_name, payload)

    assert result["document"] is None
    assert error_fragment in result["error"].lower()


@pytest.mark.parametrize(
    "file_name",
    [
        "C:\\uploads\\notes.txt",
        "\\\\server\\share\\notes.txt",
        "https://example.com/notes.txt",
    ],
)
@pytest.mark.asyncio
async def test_extract_document_rejects_path_or_uri_file_names(file_name: str):
    """Document names must be simple basenames on every client platform."""
    result = await documents.extract_document(
        file_name, base64.b64encode(b"alpha").decode()
    )

    assert result["document"] is None
    assert "file name" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_utf8_txt():
    """TXT input must be UTF-8 rather than silently replacing bad bytes."""
    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"\xff").decode()
    )

    assert result["document"] is None
    assert "utf-8" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_input_larger_than_limit(monkeypatch):
    """Decoded input larger than the configured limit is rejected."""
    monkeypatch.setattr(documents, "MAX_INPUT_BYTES", 3)

    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"four").decode()
    )

    assert result["document"] is None
    assert "input limit" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_oversized_base64_before_decode(monkeypatch):
    """Encoded input is rejected before allocating a decoded byte payload."""
    assert hasattr(documents, "MAX_ENCODED_INPUT_CHARS")
    monkeypatch.setattr(documents, "MAX_ENCODED_INPUT_CHARS", 4)
    decode_called = False

    def forbidden_decode(*args, **kwargs):
        nonlocal decode_called
        decode_called = True
        raise AssertionError("oversized base64 must not be decoded")

    monkeypatch.setattr(documents.base64, "b64decode", forbidden_decode)

    result = await documents.extract_document("notes.txt", "YWFhYQ==")

    assert result["document"] is None
    assert "encoded input limit" in result["error"].lower()
    assert decode_called is False


@pytest.mark.asyncio
async def test_extract_document_rejects_oversized_filename_without_echoing_it():
    """Filename validation is bounded and never reflects attacker-controlled text."""
    attacker_text = "SENSITIVE-FILENAME-" + ("x" * 255)

    result = await documents.extract_document(
        f"{attacker_text}.txt", base64.b64encode(b"alpha").decode()
    )

    assert result["document"] is None
    assert attacker_text not in result["error"]
    assert len(json.dumps(result, ensure_ascii=False)) <= 50_000


@pytest.mark.asyncio
async def test_extract_document_rejects_oversized_page_range_without_echoing_it():
    """PDF option validation is bounded and never reflects attacker-controlled text."""
    attacker_text = "SENSITIVE-PAGE-RANGE-" + ("9" * 256)

    result = await documents.extract_document(
        "notes.pdf",
        base64.b64encode(_two_page_pdf()).decode(),
        page_range=attacker_text,
    )

    assert result["document"] is None
    assert attacker_text not in result["error"]
    assert len(json.dumps(result, ensure_ascii=False)) <= 50_000


@pytest.mark.asyncio
async def test_extract_document_truncates_text_larger_than_output_limit(monkeypatch):
    """TXT responses expose truncation while retaining the original length."""
    assert hasattr(documents, "MAX_RESPONSE_CHARS")
    monkeypatch.setattr(documents, "MAX_RESPONSE_CHARS", 112)

    result = await documents.extract_document(
        "notes.txt", base64.b64encode(b"four").decode()
    )

    assert result["document"]["text"] != "four"
    assert result["document"]["text_length"] == 4
    assert result["document"]["truncated"] is True
    assert len(json.dumps(result, ensure_ascii=False)) <= 112


@pytest.mark.asyncio
async def test_extract_document_extracts_docx_paragraphs_and_tables():
    """DOCX output includes nonblank paragraphs and JSON-compatible table cells."""
    result = await documents.extract_document("notes.docx", _docx_payload())

    assert result["document"] == {
        "file_name": "notes.docx",
        "file_type": "docx",
        "paragraphs": ["paragraph text"],
        "tables": [[["left", "right"]]],
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_extract_document_extracts_pptx_slide_text():
    """PPTX output groups shape text by slide and counts nonblank slides."""
    result = await documents.extract_document("slides.pptx", _pptx_payload())

    assert result["document"] == {
        "file_name": "slides.pptx",
        "file_type": "pptx",
        "slides": [{"slide": 1, "text": "slide text"}],
        "total_slides": 2,
        "nonblank_slides": 1,
        "returned_slides": 1,
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_csv_exact_row_limit_is_not_truncated():
    """A CSV at the requested row limit is complete rather than truncated."""
    payload = base64.b64encode(b"name,value\na,1\nb,2\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=2)

    assert result["document"]["rows"] == 2
    assert result["document"]["truncated"] is False


@pytest.mark.asyncio
async def test_csv_over_row_limit_is_truncated():
    """A CSV with an extra record reports truncation after the requested limit."""
    payload = base64.b64encode(b"name,value\na,1\nb,2\nc,3\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=2)

    assert result["document"]["rows"] == 2
    assert result["document"]["truncated"] is True


@pytest.mark.asyncio
async def test_extract_document_extracts_csv_columns_and_records():
    """CSV records retain headings and values as JSON-compatible dictionaries."""
    payload = base64.b64encode(b"name,value\na,1\n").decode()
    result = await documents.extract_document("rows.csv", payload)

    assert result["document"] == {
        "file_name": "rows.csv",
        "file_type": "csv",
        "columns": ["name", "value"],
        "records": [{"name": "a", "value": "1"}],
        "rows": 1,
        "truncated": False,
    }


@pytest.mark.parametrize("max_rows", [0, -1, documents.MAX_CSV_ROWS + 1])
@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_csv_row_limits(max_rows: int):
    """CSV row limits must be positive and within the configured maximum."""
    payload = base64.b64encode(b"name\na\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=max_rows)

    assert result["document"] is None
    assert "row limit" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_invalid_utf8_csv():
    """CSV input must be valid UTF-8 rather than silently replacing bytes."""
    result = await documents.extract_document(
        "rows.csv", base64.b64encode(b"name\n\xff\n").decode()
    )

    assert result["document"] is None
    assert "utf-8" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_rejects_duplicate_csv_headings():
    """Duplicate CSV headings cannot safely become dictionary keys."""
    payload = base64.b64encode(b"name,name\na,b\n").decode()
    result = await documents.extract_document("rows.csv", payload)

    assert result["document"] is None
    assert "duplicate" in result["error"].lower()


@pytest.mark.asyncio
async def test_extract_document_bounds_csv_records(monkeypatch):
    """CSV records stop before nested output can exceed the shared character budget."""
    assert hasattr(documents, "MAX_RESPONSE_CHARS")
    monkeypatch.setattr(documents, "MAX_RESPONSE_CHARS", 150)
    payload = base64.b64encode(b"name\nfirst\nsecond\n").decode()
    result = await documents.extract_document("rows.csv", payload, max_rows=2)

    assert len(result["document"]["records"]) < 2
    assert result["document"]["truncated"] is True
    assert len(json.dumps(result, ensure_ascii=False)) <= 150


@pytest.mark.asyncio
async def test_csv_headings_count_toward_complete_response_budget(monkeypatch):
    """CSV records cannot consume space already occupied by column headings."""
    assert hasattr(documents, "MAX_RESPONSE_CHARS")
    monkeypatch.setattr(documents, "MAX_RESPONSE_CHARS", 200)
    headings = "first_long_heading,second_long_heading"
    payload = base64.b64encode(f"{headings}\na,b\n".encode()).decode()

    result = await documents.extract_document("rows.csv", payload)

    assert result["document"]["columns"] == [
        "first_long_heading",
        "second_long_heading",
    ]
    assert result["document"]["records"] == []
    assert result["document"]["truncated"] is True
    assert len(json.dumps(result, ensure_ascii=False)) <= 200


@pytest.mark.asyncio
async def test_extract_document_bounds_docx_tables(monkeypatch):
    """DOCX table cells stop accumulating after the shared character budget."""
    assert hasattr(documents, "MAX_RESPONSE_CHARS")
    monkeypatch.setattr(documents, "MAX_RESPONSE_CHARS", 140)
    result = await documents.extract_document("notes.docx", _docx_payload())

    assert result["document"]["truncated"] is True
    assert len(json.dumps(result, ensure_ascii=False)) <= 140


@pytest.mark.asyncio
async def test_extract_document_bounds_pptx_slides(monkeypatch):
    """PPTX slide objects stop accumulating after the shared character budget."""
    assert hasattr(documents, "MAX_RESPONSE_CHARS")
    monkeypatch.setattr(documents, "MAX_RESPONSE_CHARS", 190)
    result = await documents.extract_document("slides.pptx", _pptx_payload())

    assert result["document"]["slides"] == []
    assert result["document"]["truncated"] is True
    assert len(json.dumps(result, ensure_ascii=False)) <= 190


@pytest.mark.asyncio
async def test_extract_document_reports_complete_pptx_slide_counts_after_truncation(
    monkeypatch,
):
    """PPTX metadata distinguishes all nonblank slides from returned slides."""
    assert hasattr(documents, "MAX_RESPONSE_CHARS")
    monkeypatch.setattr(documents, "MAX_RESPONSE_CHARS", 210)
    result = await documents.extract_document(
        "slides.pptx", _pptx_payload(("a", "b", ""))
    )

    assert result["document"]["total_slides"] == 3
    assert result["document"]["nonblank_slides"] == 2
    assert result["document"]["returned_slides"] == 1
    assert result["document"]["truncated"] is True


def _office_zip(entries: list[tuple[str, bytes]]) -> bytes:
    """Build a compressed Office-like archive for preflight tests."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def test_office_zip_preflight_rejects_too_many_entries(monkeypatch):
    """Office archives cannot exceed the explicit central-directory entry cap."""
    assert hasattr(documents, "MAX_OFFICE_ZIP_ENTRIES")
    assert hasattr(documents, "_preflight_office_zip")
    monkeypatch.setattr(documents, "MAX_OFFICE_ZIP_ENTRIES", 1)
    data = _office_zip([("one.xml", b"x"), ("two.xml", b"x")])

    with pytest.raises(ValueError, match="too many entries"):
        documents._preflight_office_zip(data)


def test_office_zip_preflight_rejects_aggregate_expansion(monkeypatch):
    """Aggregate uncompressed Office content is bounded before parser startup."""
    assert hasattr(documents, "MAX_OFFICE_UNCOMPRESSED_BYTES")
    assert hasattr(documents, "MAX_OFFICE_COMPRESSION_RATIO")
    assert hasattr(documents, "_preflight_office_zip")
    monkeypatch.setattr(documents, "MAX_OFFICE_UNCOMPRESSED_BYTES", 3)
    monkeypatch.setattr(documents, "MAX_OFFICE_COMPRESSION_RATIO", 10_000)
    data = _office_zip([("one.xml", b"four")])

    with pytest.raises(ValueError, match="expanded content limit"):
        documents._preflight_office_zip(data)


def test_office_zip_preflight_rejects_high_compression_ratio(monkeypatch):
    """Highly compressed Office entries are rejected as expansion bombs."""
    assert hasattr(documents, "MAX_OFFICE_COMPRESSION_RATIO")
    assert hasattr(documents, "_preflight_office_zip")
    monkeypatch.setattr(documents, "MAX_OFFICE_COMPRESSION_RATIO", 2)
    data = _office_zip([("one.xml", b"A" * 2_000)])

    with pytest.raises(ValueError, match="compression ratio"):
        documents._preflight_office_zip(data)


def _measure_parser_startup_seconds():
    """Wall time of a trivial parser round-trip — dominated by child startup.

    A forkserver child re-imports ``document_parser_runtime`` to unpickle the
    worker on every call, and forkserver does not preload it, so this cost
    recurs per call rather than being paid once. It is ~0.2s on a warm machine
    and over 2s on a cold or loaded one.
    """
    probe_queue = documents._PARSER_CONTEXT.Queue()
    started = time.monotonic()
    documents._run_parser_in_process(parser_probes.sleeping_parser, probe_queue, 0.0)
    elapsed = time.monotonic() - started
    probe_queue.get(timeout=10)
    return elapsed


def test_parser_timeout_terminates_the_worker_process(monkeypatch):
    """A timed-out parser process is killed and reaped before returning."""
    assert hasattr(documents, "_PARSER_CONTEXT")
    assert hasattr(documents, "ParserTimeoutError")
    assert hasattr(documents, "_run_parser_in_process")
    context = documents._PARSER_CONTEXT

    # The deadline covers child startup, not just the parser body, so a fixed
    # timeout races the child's import of the parser runtime. Below ~2.5s on a
    # cold machine the child is killed mid-import, before the probe publishes
    # its pid — and the assertion then fails on a missing pid for a reason that
    # has nothing to do with termination. Measure startup here and leave room
    # above it, so what is under test is the kill, not the machine's speed.
    deadline = _measure_parser_startup_seconds() + 1.0
    monkeypatch.setattr(documents, "PARSER_TIMEOUT_SECONDS", deadline)

    pid_queue = context.Queue()
    # Sleeps well past the deadline, so the parser is always still running when
    # the timeout fires.
    with pytest.raises(documents.ParserTimeoutError, match="timed out"):
        documents._run_parser_in_process(parser_probes.sleeping_parser, pid_queue, 30.0)

    pid = pid_queue.get(timeout=10)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_parser_processes_respect_the_concurrency_boundary(monkeypatch):
    """Concurrent parser requests never exceed the configured process count."""
    assert hasattr(documents, "_PARSER_CONTEXT")
    assert hasattr(documents, "_PARSER_SLOTS")
    assert hasattr(documents, "_run_parser_in_process")
    context = documents._PARSER_CONTEXT
    active = context.Value("i", 0)
    maximum = context.Value("i", 0)
    lock = context.Lock()
    monkeypatch.setattr(documents, "_PARSER_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(documents, "PARSER_TIMEOUT_SECONDS", 10.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                documents._run_parser_in_process,
                parser_probes.concurrency_probe_parser,
                active,
                maximum,
                lock,
                0.15,
            )
            for _ in range(2)
        ]
        for future in futures:
            future.result(timeout=15)

    assert maximum.value == 1


def test_parser_resource_failures_are_bounded_and_nonsensitive():
    """Resource exhaustion does not expose child-process exception details."""
    assert hasattr(documents, "ParserResourceError")
    assert hasattr(documents, "_run_parser_in_process")
    with pytest.raises(documents.ParserResourceError) as raised:
        documents._run_parser_in_process(parser_probes.resource_failure_parser)

    assert "SENSITIVE-PARSER-DETAIL" not in str(raised.value)
    assert len(str(raised.value)) <= 256


def test_parser_watchdog_fails_closed_when_process_metrics_are_denied():
    """An unavailable RSS/CPU sample cannot silently disable resource limits."""

    class FakePsutil:
        class NoSuchProcess(Exception):
            pass

        class AccessDenied(Exception):
            pass

        @staticmethod
        def Process(process_id):
            raise FakePsutil.AccessDenied(process_id)

    with pytest.raises(documents.ParserResourceError, match="unavailable"):
        documents._watchdog_limit_exceeded(FakePsutil, 123)


def test_parser_watchdog_terminates_a_process_over_the_rss_limit(monkeypatch):
    """Fallback platforms kill and reap a parser whose sampled RSS is excessive."""
    context = documents._PARSER_CONTEXT
    pid_reader, pid_writer = context.Pipe(duplex=False)
    monkeypatch.setattr(documents, "_USES_PARENT_RESOURCE_WATCHDOG", True)
    monkeypatch.setattr(documents, "PARSER_MEMORY_BYTES", 256 * 1024 * 1024)
    # Same trap #504 fixed for the sibling timeout test: the deadline covers
    # child startup, not just the parser body. Measured here, startup is the
    # entire cost -- 1.41s of a 1.42s run on an idle machine -- and #504
    # measured 2.0-2.5s on a loaded one. When startup outruns the budget the
    # timeout fires first and this test fails on ParserTimeoutError, matching
    # "timed out" instead of "resource limit": a failure that says nothing
    # about the watchdog. Derive the deadline so what is under test is the
    # kill, not the machine's speed.
    #
    # The probe sleeps 2.0s after allocating, so +3.0 leaves the watchdog the
    # whole window to notice. If it never fires, this now fails with DID NOT
    # RAISE -- the honest symptom -- rather than a timeout wearing its clothes.
    deadline = _measure_parser_startup_seconds() + 3.0
    monkeypatch.setattr(documents, "PARSER_TIMEOUT_SECONDS", deadline)

    with pytest.raises(documents.ParserResourceError, match="resource limit"):
        documents._run_parser_in_process(
            parser_probes.memory_hog_parser,
            pid_writer,
            384 * 1024 * 1024,
            2.0,
        )

    pid_writer.close()
    assert pid_reader.poll(10)
    pid = pid_reader.recv()
    pid_reader.close()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_complete_success_and_error_payloads_fit_response_cap():
    """The central response guard bounds both tool result variants."""
    success = await documents.extract_document(
        "notes.txt", base64.b64encode(b"x" * 100_000).decode()
    )
    error = await documents.extract_document(
        "notes.pdf",
        base64.b64encode(b"x").decode(),
        page_range="SENSITIVE-" + ("9" * 256),
    )

    assert len(json.dumps(success, ensure_ascii=False)) <= 50_000
    assert len(json.dumps(error, ensure_ascii=False)) <= 50_000


@pytest.mark.parametrize(
    ("extension", "module_name", "distribution_name"),
    [
        ("docx", "docx", "python-docx"),
        ("pptx", "pptx", "python-pptx"),
    ],
)
@pytest.mark.asyncio
async def test_extract_document_reports_missing_office_dependency(
    monkeypatch, extension: str, module_name: str, distribution_name: str
):
    """Missing optional office parsers return actionable installation errors."""

    def missing_module(name: str):
        if name == module_name:
            raise ImportError(name)
        return __import__(name)

    monkeypatch.setattr(
        document_parser_runtime.importlib, "import_module", missing_module
    )
    with pytest.raises(document_parser_runtime.ParserDependencyError) as raised:
        document_parser_runtime._require_module(module_name, distribution_name)

    assert distribution_name in str(raised.value)
