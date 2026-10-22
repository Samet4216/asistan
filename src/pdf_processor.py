"""
PDF Processor & Chapter-Aware Chunking Module
Extracts clean text from digital PDFs, detects sections/chapters,
and applies intelligent fallback chunking.
"""

import os
import re
import sys
import io
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import yaml

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PDFProcessor:
    def __init__(self, pdf_dir: str = None, chunk_size: int = 800, chunk_overlap: int = 150):
        config = load_config()
        self.pdf_dir = Path(pdf_dir or config.get("paths", {}).get("pdf_folder", "./pdf")).resolve()
        self.chunk_size = chunk_size or config.get("rag", {}).get("chunk_size", 800)
        self.chunk_overlap = chunk_overlap or config.get("rag", {}).get("chunk_overlap", 150)
        
        # Regex patterns for academic/technical headings:
        # e.g., "1. Introduction", "1.1 Background", "SECTION 2", "I. INTRODUCTION", "Abstract", "References"
        self.heading_patterns = [
            re.compile(r"^(?:[0-9]{1,2}\.){1,3}\s+[A-Z][A-Za-z0-9\s_-]{2,50}$", re.MULTILINE),
            re.compile(r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+[A-Z][A-Za-z0-9\s_-]{2,50}$", re.MULTILINE),
            re.compile(r"^(?:CHAPTER|SECTION|BÖLÜM)\s+[0-9IVXLCDM]+[:\s]+[A-Za-z0-9\s_-]{2,50}$", re.MULTILINE | re.IGNORECASE),
            re.compile(r"^(?:Abstract|Introduction|Related Work|Methodology|Experiments|Results|Discussion|Conclusion|References)\b", re.MULTILINE | re.IGNORECASE)
        ]

    def extract_text(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text page by page from a PDF.
        Returns a list of dicts with page number and page text.
        """
        pages = []
        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            if text and text.strip():
                pages.append({
                    "page_num": page_idx + 1,
                    "text": text.strip()
                })
        doc.close()
        return pages

    def detect_chapters(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Detects chapters/sections based on heading regex patterns.
        Falls back to a single unified chapter if no clear headings are detected.
        """
        lines = full_text.splitlines()
        headings = []

        for idx, line in enumerate(lines):
            clean_line = line.strip()
            if not clean_line or len(clean_line) > 80:
                continue

            for pattern in self.heading_patterns:
                if pattern.match(clean_line):
                    headings.append((idx, clean_line))
                    break

        # If no headings found, fallback to 1 chapter
        if not headings:
            return [{"chapter_no": 1, "title": "Main Content", "text": full_text}]

        chapters = []
        for i in range(len(headings)):
            start_line, title = headings[i]
            end_line = headings[i + 1][0] if i + 1 < len(headings) else len(lines)
            chapter_text = "\n".join(lines[start_line:end_line]).strip()
            if chapter_text:
                chapters.append({
                    "chapter_no": i + 1,
                    "title": title,
                    "text": chapter_text
                })

        return chapters

    def chunk_text(self, text: str, max_chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Splits text into chunks of roughly max_chunk_size characters with overlap,
        respecting paragraph and sentence boundaries.
        """
        max_chunk = max_chunk_size or self.chunk_size
        ov = overlap or self.chunk_overlap

        if len(text) <= max_chunk:
            return [text]

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + max_chunk
            if end >= text_len:
                chunks.append(text[start:].strip())
                break

            # Find paragraph or sentence boundary
            split_idx = text.rfind("\n\n", start + ov, end)
            if split_idx == -1:
                split_idx = text.rfind(". ", start + ov, end)
            if split_idx == -1:
                split_idx = text.rfind("\n", start + ov, end)
            if split_idx == -1:
                split_idx = end
            else:
                split_idx += 1  # Include the punctuation/newline

            chunk = text[start:split_idx].strip()
            if chunk:
                chunks.append(chunk)

            start = max(split_idx - ov, start + 1)

        return chunks

    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        Processes all PDFs in self.pdf_dir.
        Returns flattened list of chunk dictionaries ready for embedding / ChromaDB.
        """
        all_chunks = []
        if not self.pdf_dir.exists():
            print(f"[!] PDF directory not found: {self.pdf_dir}")
            return all_chunks

        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[*] No PDF files found in {self.pdf_dir}. Add your academic PDFs here.")
            return all_chunks

        for pdf_path in pdf_files:
            pages = self.extract_text(pdf_path)
            full_text = "\n\n".join(p["text"] for p in pages)
            chapters = self.detect_chapters(full_text)

            for chap in chapters:
                sub_chunks = self.chunk_text(chap["text"])
                for sub_idx, chunk_text in enumerate(sub_chunks):
                    chunk_id = f"{pdf_path.stem}_chap{chap['chapter_no']}_part{sub_idx+1}"
                    all_chunks.append({
                        "id": chunk_id,
                        "source": pdf_path.name,
                        "type": "pdf",
                        "chapter_no": chap["chapter_no"],
                        "chapter_title": chap["title"],
                        "part_no": sub_idx + 1,
                        "content": chunk_text,
                        "char_count": len(chunk_text)
                    })

        return all_chunks

    def print_summary(self):
        chunks = self.process_all_pdfs()
        print("\n" + "="*50)
        print("📄 PDF PROCESSOR REPORT")
        print("="*50)
        print(f"📁 PDF Directory: {self.pdf_dir}")
        print(f"📦 Total Chunks Generated: {len(chunks)}")
        if chunks:
            print("\nÖrnek Chunklar:")
            for c in chunks[:3]:
                print(f" • [{c['id']}] ({c['chapter_title']}) - {c['char_count']} karakter")
        print("="*50 + "\n")


if __name__ == "__main__":
    processor = PDFProcessor()
    processor.print_summary()
