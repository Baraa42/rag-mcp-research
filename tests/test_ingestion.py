"""Exercise ingestion using tiny, generated PDFs and Markdown files."""

import hashlib
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, StreamObject

from research_assistant.ingest.ingestion import (
    extract_markdown,
    extract_pdf,
    ingest_directory,
)


def write_pdf(path: Path, texts: list[str]) -> None:
    """Create simple ASCII text pages without a fixture-generation dependency."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in texts:
        page = writer.add_blank_page(width=300, height=300)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = StreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    writer.write(path)


@pytest.mark.parametrize("texts", [["First"], ["First", "Second", "Third"]])
def test_pdf_pages(tmp_path: Path, texts: list[str]) -> None:
    path = tmp_path / "paper.pdf"
    write_pdf(path, texts)
    pages = extract_pdf(path)
    assert [page.text for page in pages] == texts
    assert [page.page_number for page in pages] == list(range(1, len(texts) + 1))
    assert {page.source for page in pages} == {"paper.pdf"}
    assert {page.document_id for page in pages} == {
        hashlib.sha256(path.read_bytes()).hexdigest()
    }


def test_blank_pages_preserve_physical_numbers(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    write_pdf(path, ["", "Second", "   ", "Fourth"])
    assert [page.page_number for page in extract_pdf(path)] == [2, 4]


@pytest.mark.parametrize("texts", [[], [""], ["   ", ""]])
def test_empty_pdf(tmp_path: Path, texts: list[str]) -> None:
    path = tmp_path / "empty.pdf"
    write_pdf(path, texts)
    assert extract_pdf(path) == []


def test_markdown(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("  # Résumé\n\nSome **text**.\n", encoding="utf-8")
    (page,) = extract_markdown(path)
    assert page.text == "# Résumé\n\nSome **text**."
    assert page.source == "notes.md"
    assert page.page_number is None


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_empty_markdown(tmp_path: Path, text: str) -> None:
    path = tmp_path / "empty.md"
    path.write_text(text, encoding="utf-8")
    assert extract_markdown(path) == []


def test_empty_directory(tmp_path: Path) -> None:
    assert ingest_directory(tmp_path) == []


def test_recursive_order_and_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.md").write_text("last", encoding="utf-8")
    write_pdf(tmp_path / "nested/b.PDF", ["one", "two"])
    (tmp_path / "a.md").write_text("first", encoding="utf-8")
    (tmp_path / "ignored.txt").write_bytes(b"\xff")
    (tmp_path / "ignored.pdf.bak").write_bytes(b"invalid")
    pages = ingest_directory(tmp_path)
    assert [page.source for page in pages] == [
        "a.md",
        "nested/b.PDF",
        "nested/b.PDF",
        "z.md",
    ]
    assert [page.text for page in pages] == ["first", "one", "two", "last"]
    assert ingest_directory(tmp_path) == pages


@pytest.mark.parametrize("content", [b"", b"not a PDF", b"%PDF-1.7\ntruncated"])
def test_corrupted_pdf(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Cannot extract PDF.*broken.pdf"):
        extract_pdf(path)
    with pytest.raises(ValueError, match="broken.pdf"):
        ingest_directory(tmp_path)


@pytest.mark.parametrize("extension", ["md", "pdf"])
def test_stable_content_identifiers(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f"original.{extension}"
    if extension == "pdf":
        write_pdf(path, ["Original"])
    else:
        path.write_text("Original", encoding="utf-8")
    extract = extract_pdf if extension == "pdf" else extract_markdown
    original = extract(path)[0].document_id
    copied = tmp_path / f"renamed.{extension}"
    copied.write_bytes(path.read_bytes())
    assert extract(path)[0].document_id == original
    assert extract(copied)[0].document_id == original
    if extension == "pdf":
        write_pdf(path, ["Changed"])
    else:
        path.write_text("Changed", encoding="utf-8")
    assert extract(path)[0].document_id != original


def test_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="bad.md.*UTF-8"):
        extract_markdown(path)


def test_missing_files_and_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pdf(tmp_path / "missing.pdf")
    with pytest.raises(FileNotFoundError):
        extract_markdown(tmp_path / "missing.md")
    with pytest.raises(FileNotFoundError):
        ingest_directory(tmp_path / "missing")
    file = tmp_path / "file.md"
    file.touch()
    with pytest.raises(NotADirectoryError):
        ingest_directory(file)


def test_unreadable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def deny_read(self: Path) -> bytes:
        raise PermissionError(f"Permission denied: {self}")

    monkeypatch.setattr(Path, "read_bytes", deny_read)
    with pytest.raises(PermissionError, match="denied"):
        extract_pdf(tmp_path / "unreadable.pdf")
    with pytest.raises(PermissionError, match="denied"):
        extract_markdown(tmp_path / "unreadable.md")


def test_encrypted_pdf(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("password")
    writer.write(path)
    with pytest.raises(ValueError, match="Encrypted PDFs"):
        extract_pdf(path)
