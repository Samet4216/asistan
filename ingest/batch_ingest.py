"""
Batch Paper & Repository Ingestion Pipeline (Faz 13).
Reads data/papers.csv, fetches papers, parses LaTeX/PDF, chunks sections,
indexes into ChromaDB 'paper_chunks' & 'code', and generates paper cards.
"""

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ingest.arxiv_fetch import ArxivFetcher
from ingest.paper_parser import PaperParser
from ingest.chunker import SectionChunker
from ingest.indexer import Indexer
from ingest.repo_link import RepoLinker


def run_batch_ingestion():
    csv_path = PROJECT_ROOT / "data" / "papers.csv"
    if not csv_path.exists():
        print(f"[!] papers.csv not found at {csv_path}")
        return

    fetcher = ArxivFetcher()
    parser = PaperParser()
    chunker = SectionChunker()
    indexer = Indexer()
    linker = RepoLinker(indexer)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        papers = list(reader)

    print(f"[*] Starting Batch Ingestion of {len(papers)} foundational papers...\n")

    for p in papers:
        arxiv_id = p["arxiv_id"].strip()
        title = p["title"].strip()
        year = int(p["year"].strip()) if p.get("year") else 2021
        repo_url = p.get("repo_url", "").strip()

        print(f"\n" + "="*60)
        print(f"[*] Processing: [{arxiv_id}] {title} ({year})")
        print("="*60)

        # 1. Fetch paper
        fetch_res = fetcher.fetch_paper(arxiv_id, prefer_latex=True)
        folder = fetch_res["folder"]

        # 2. Parse paper
        parsed_doc = parser.parse_paper(folder, arxiv_id=arxiv_id, title=title, year=year)
        if not parsed_doc or not parsed_doc.get("sections"):
            print(f"[!] Warning: No sections parsed for {arxiv_id}. Trying PDF fallback...")
            pdf_path = fetch_res.get("pdf_path")
            if pdf_path and pdf_path.exists():
                parsed_doc = parser._parse_pdf(pdf_path, arxiv_id=arxiv_id, default_title=title, year=year)

        # 3. Chunk sections
        chunks = chunker.chunk_paper(parsed_doc)
        print(f"[+] Produced {len(chunks)} text/equation chunks.")

        # 4. Index chunks into ChromaDB
        indexer.index_paper_chunks(chunks)

        # 5. Handle repository if present
        if repo_url:
            repo_name = repo_url.rstrip("/").split("/")[-1]
            local_repo_dir = config.REPOS_DIR / repo_name
            if local_repo_dir.exists():
                print(f"[*] Indexing code repository: {repo_name}...")
                linker.index_repo(local_repo_dir, repo_name, arxiv_id=arxiv_id)

        # Polite rate-limit delay
        time.sleep(2)

    print("\n" + "="*60)
    print("[*] Batch Ingestion Complete! ChromaDB Collection Stats:")
    for name, cnt in indexer.get_collection_stats().items():
        print(f" • {name}: {cnt} items")
    print("="*60)


if __name__ == "__main__":
    run_batch_ingestion()
