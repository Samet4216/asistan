"""
Single arXiv Paper Ingestion Helper.
Enables on-demand download, parsing, chunking, and ChromaDB indexing of new papers
directly from the Web UI or CLI.
"""

import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ingest.arxiv_fetch import ArxivFetcher
from ingest.paper_parser import PaperParser
from ingest.chunker import SectionChunker
from ingest.indexer import Indexer


def ingest_arxiv_paper(arxiv_id: str, title: str = "") -> Dict[str, Any]:
    """
    Downloads, parses, chunks, and indexes a single arXiv paper into ChromaDB.
    Returns status dict with title and number of chunks added.
    """
    clean_id = arxiv_id.strip()
    if "/" in clean_id:
        clean_id = clean_id.split("/")[-1]
    clean_id = clean_id.replace("abs/", "").replace(".pdf", "")

    fetcher = ArxivFetcher()
    parser = PaperParser()
    chunker = SectionChunker()
    indexer = Indexer()

    try:
        # 1. Fetch paper (prefer LaTeX, fallback to PDF)
        fetch_res = fetcher.fetch_paper(clean_id, prefer_latex=True)
        folder = fetch_res.get("folder")
        resolved_title = title or f"arXiv:{clean_id}"

        # 2. Parse paper
        parsed_doc = parser.parse_paper(folder, arxiv_id=clean_id, title=resolved_title)
        if not parsed_doc or not parsed_doc.get("sections"):
            pdf_path = fetch_res.get("pdf_path")
            if pdf_path and Path(pdf_path).exists():
                parsed_doc = parser._parse_pdf(Path(pdf_path), arxiv_id=clean_id, default_title=resolved_title)

        if not parsed_doc or not parsed_doc.get("sections"):
            return {
                "success": False,
                "error": f"Could not extract sections from paper {clean_id}."
            }

        # 3. Chunk sections
        chunks = chunker.chunk_paper(parsed_doc)
        if not chunks:
            return {
                "success": False,
                "error": "No valid text or equation chunks generated."
            }

        # 4. Index chunks into ChromaDB
        indexer.index_paper_chunks(chunks)

        actual_title = parsed_doc.get("title", resolved_title)
        return {
            "success": True,
            "arxiv_id": clean_id,
            "title": actual_title,
            "chunks_count": len(chunks)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    test_id = "1706.03762"  # Transformer
    print(f"[*] Testing single_ingest for {test_id}...")
    res = ingest_arxiv_paper(test_id, title="Attention Is All You Need")
    print("Result:", res)
