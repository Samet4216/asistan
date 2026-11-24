"""Unit tests for DocumentLoader (PDF, Markdown, TXT page preservation).
"""

from pathlib import Path
import pytest
import pymupdf

from personal_ai.rag.document_loader import DocumentLoader, DocumentLoaderError


def test_load_text_file(tmp_path: Path):
    """Verify plain text loading and metadata extraction."""
    loader = DocumentLoader()
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Linear Quadratic Regulator optimizes control effort.", encoding="utf-8")

    doc = loader.load_file(txt_file)
    assert doc.file_name == "sample.txt"
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert "Linear Quadratic Regulator" in doc.pages[0].text
    assert len(doc.file_hash) == 64  # Valid SHA-256


def test_load_markdown_file(tmp_path: Path):
    """Verify Markdown file loading."""
    loader = DocumentLoader()
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Deep Learning\n\nConvolutional layers preserve spatial hierarchy.", encoding="utf-8")

    doc = loader.load_file(md_file)
    assert doc.file_name == "notes.md"
    assert "spatial hierarchy" in doc.full_text


def test_load_pdf_page_preservation(tmp_path: Path):
    """Verify multi-page PDF extracts exact 1-indexed page numbers."""
    pdf_path = tmp_path / "multipage_paper.pdf"

    # Create real 2-page PDF via PyMuPDF
    pdf_doc = pymupdf.open()
    p1 = pdf_doc.new_page()
    p1.insert_text((50, 50), "Page 1 Content: Extended Kalman Filter equations.")
    p2 = pdf_doc.new_page()
    p2.insert_text((50, 50), "Page 2 Content: YOLOv8 real-time inference latency.")
    pdf_doc.save(pdf_path)
    pdf_doc.close()

    loader = DocumentLoader()
    doc = loader.load_file(pdf_path)

    assert doc.total_pages == 2
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Extended Kalman Filter" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "YOLOv8" in doc.pages[1].text


def test_unsupported_file_format(tmp_path: Path):
    """Verify error on unsupported formats like .exe or .bin."""
    loader = DocumentLoader()
    bad_file = tmp_path / "data.bin"
    bad_file.write_bytes(b"\x00\x01\x02")

    with pytest.raises(DocumentLoaderError, match="Unsupported file format"):
        loader.load_file(bad_file)
