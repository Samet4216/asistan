"""Document Chunker with boundary preservation and metadata tracking.
"""

from dataclasses import dataclass
from typing import List, Optional

from personal_ai.rag.document_loader import DocumentPage, LoadedDocument


@dataclass
class DocumentChunk:
    """Represents a coherent piece of text extracted from a document page."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    file_hash: str
    chunk_index: int
    char_start: int
    char_end: int


class TextChunker:
    """Recursively splits text into chunks of specified length with overlap."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " "]

    def chunk_document(self, doc: LoadedDocument) -> List[DocumentChunk]:
        """Splits an entire LoadedDocument into DocumentChunks, preserving page numbers."""
        chunks: List[DocumentChunk] = []
        global_idx = 0
        for page in doc.pages:
            page_chunks = self.chunk_page(page, start_index=global_idx)
            chunks.extend(page_chunks)
            global_idx += len(page_chunks)
        return chunks

    def chunk_page(self, page: DocumentPage, start_index: int = 0) -> List[DocumentChunk]:
        """Splits a single DocumentPage into chunks."""
        text = page.text
        if not text.strip():
            return []

        raw_splits = self._split_text(text)
        chunks: List[DocumentChunk] = []

        cur_pos = 0
        for idx, chunk_text in enumerate(raw_splits):
            chunk_len = len(chunk_text)
            char_start = cur_pos
            char_end = cur_pos + chunk_len
            cur_pos = max(0, char_end - self.chunk_overlap)

            chunk_id = f"{page.file_hash[:8]}_p{page.page_number}_c{start_index + idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    file_hash=page.file_hash,
                    chunk_index=start_index + idx,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Recursive chunk splitting algorithm."""
        splits: List[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + self.chunk_size, n)
            if end < n:
                # Find best splitting point
                best_break = -1
                for sep in self.separators:
                    pos = text.rfind(sep, start, end)
                    if pos != -1 and pos > start:
                        best_break = pos + len(sep)
                        break

                if best_break != -1:
                    end = best_break

            chunk = text[start:end].strip()
            if chunk:
                splits.append(chunk)

            if end >= n:
                break
            start = max(start + 1, end - self.chunk_overlap)

        return splits
