"""Extract text and source metadata without chunking or retrieval."""

import hashlib
import os
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentPage:
    """Nonempty text from one PDF page or an entire Markdown file."""

    document_id: str
    source: str
    page_number: int | None
    text: str


def extract_pdf(path: Path) -> list[DocumentPage]:
    """Extract nonblank physical pages; reject invalid or encrypted PDFs."""
    content = path.read_bytes()
    document_id = hashlib.sha256(content).hexdigest()
    page_number = None
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text().strip()
            if text:
                pages.append(DocumentPage(document_id, path.name, page_number, text))
        return pages
    except Exception as exc:
        # Add source context to parser failures; never return partial results.
        location = f", page {page_number}" if page_number is not None else ""
        raise ValueError(f"Cannot extract PDF '{path}'{location}: {exc}") from exc


def extract_markdown(path: Path) -> list[DocumentPage]:
    """Read UTF-8 Markdown as one record, preserving its markup."""
    content = path.read_bytes()
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Cannot decode Markdown '{path}' as UTF-8") from exc
    if not text:
        return []
    return [DocumentPage(hashlib.sha256(content).hexdigest(), path.name, None, text)]


def ingest_directory(path: Path) -> list[DocumentPage]:
    """Read PDFs/Markdown recursively in relative-path order; fail on errors."""
    if not path.is_dir():
        if not path.exists():
            raise FileNotFoundError(f"Ingestion directory does not exist: {path}")
        raise NotADirectoryError(f"Not an ingestion directory: {path}")

    def raise_walk_error(error: OSError) -> None:
        raise error

    files = []
    for root, _, names in os.walk(path, onerror=raise_walk_error):
        for name in names:
            candidate = Path(root) / name
            if candidate.suffix.lower() in {".pdf", ".md"}:
                files.append(candidate)

    pages = []
    for file in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        extract = extract_pdf if file.suffix.lower() == ".pdf" else extract_markdown
        source = file.relative_to(path).as_posix()
        pages.extend(replace(page, source=source) for page in extract(file))
    return pages
