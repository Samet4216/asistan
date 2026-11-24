"""Unit tests for TextChunker (boundary preservation and metadata).
"""

import pytest

from personal_ai.rag.chunker import TextChunker
from personal_ai.rag.document_loader import DocumentPage, LoadedDocument
from pathlib import Path


def test_chunker_invalid_overlap():
    """Verify error when overlap is equal or greater than chunk_size."""
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_splits_paragraphs():
    """Verify splitting on natural paragraph boundaries."""
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)

    p1 = "Bu bir birinci paragraftır. Robotik sistemlerde geri besleme kritik önem taşır."
    p2 = "Bu bir ikinci paragraftır. PID katsayılarının optimizasyonu Ziegler-Nichols ile yapılabilir."
    p3 = "Bu bir üçüncü paragraftır. Durum uzayı temsili modern kontrol teorisinin temelidir."

    text = f"{p1}\n\n{p2}\n\n{p3}"
    page = DocumentPage(text=text, page_number=1, source_file="control.txt", file_hash="hash123")
    doc = LoadedDocument(file_path=Path("control.txt"), file_name="control.txt", file_hash="hash123", pages=[page], total_pages=1)

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.source_file == "control.txt"
        assert c.page_number == 1
        assert c.file_hash == "hash123"
        assert len(c.text) <= 200


def test_empty_page_handling():
    """Verify empty pages do not produce chunks."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    page = DocumentPage(text="   \n   ", page_number=2, source_file="empty.txt", file_hash="emptyhash")
    chunks = chunker.chunk_page(page)
    assert chunks == []
