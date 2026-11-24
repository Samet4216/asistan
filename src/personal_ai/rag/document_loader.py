"""Document Loader supporting PDF, Markdown, and Plain Text with exact page tracking.
"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import List, Optional
import pymupdf


@dataclass
class DocumentPage:
    """Represents a single page or segment of a loaded document."""
    text: str
    page_number: int  # 1-indexed for human readability and accurate citation
    source_file: str
    file_hash: str


@dataclass
class LoadedDocument:
    """Represents a complete loaded document with page-level granularity."""
    file_path: Path
    file_name: str
    file_hash: str
    pages: List[DocumentPage] = field(default_factory=list)
    total_pages: int = 0

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


class DocumentLoaderError(Exception):
    """Raised when document loading or parsing fails."""
    pass


class DocumentLoader:
    """Handles parsing of PDF, Markdown, and TXT files with metadata extraction."""

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Calculates SHA-256 hash of a file for integrity and deduplication."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_file(self, file_path: str | Path) -> LoadedDocument:
        """Loads and parses a single file into a LoadedDocument."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise DocumentLoaderError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoaderError(
                f"Unsupported file format '{suffix}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        file_hash = self.compute_sha256(path)
        file_name = path.name

        if suffix == ".pdf":
            return self._load_pdf(path, file_name, file_hash)
        else:
            return self._load_text_file(path, file_name, file_hash)

    def _load_pdf(self, path: Path, file_name: str, file_hash: str) -> LoadedDocument:
        """Parses a PDF document, preserving exact 1-indexed page numbers."""
        pages: List[DocumentPage] = []
        try:
            doc = pymupdf.open(path)
            total_pages = len(doc)
            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                text = page.get_text("text").strip()
                if text:  # Store non-empty pages
                    pages.append(
                        DocumentPage(
                            text=text,
                            page_number=page_idx + 1,
                            source_file=file_name,
                            file_hash=file_hash,
                        )
                    )
            doc.close()
            return LoadedDocument(
                file_path=path,
                file_name=file_name,
                file_hash=file_hash,
                pages=pages,
                total_pages=total_pages,
            )
        except Exception as e:
            raise DocumentLoaderError(f"Failed to parse PDF '{path}': {e}") from e

    def _load_text_file(self, path: Path, file_name: str, file_hash: str) -> LoadedDocument:
        """Parses Markdown or plain text file into a single or multi-part DocumentPage."""
        try:
            text = path.read_text(encoding="utf-8").strip()
            pages = [
                DocumentPage(
                    text=text,
                    page_number=1,  # Text/MD documents treat entirety as page 1
                    source_file=file_name,
                    file_hash=file_hash,
                )
            ]
            return LoadedDocument(
                file_path=path,
                file_name=file_name,
                file_hash=file_hash,
                pages=pages,
                total_pages=1,
            )
        except UnicodeDecodeError:
            # Fallback for Windows-1254 (Turkish ANSI) or Latin-1 if UTF-8 fails
            try:
                text = path.read_text(encoding="cp1254").strip()
                pages = [
                    DocumentPage(
                        text=text,
                        page_number=1,
                        source_file=file_name,
                        file_hash=file_hash,
                    )
                ]
                return LoadedDocument(
                    file_path=path,
                    file_name=file_name,
                    file_hash=file_hash,
                    pages=pages,
                    total_pages=1,
                )
            except Exception as e:
                raise DocumentLoaderError(f"Failed to decode text file '{path}': {e}") from e
        except Exception as e:
            raise DocumentLoaderError(f"Failed to read file '{path}': {e}") from e
