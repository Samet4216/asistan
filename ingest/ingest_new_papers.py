"""
Ingest targeted papers into ChromaDB RAG memory.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest.arxiv_fetch import ArxivFetcher
from ingest.paper_parser import PaperParser
from ingest.chunker import SectionChunker
from ingest.indexer import Indexer

TARGET_PAPERS = [
    {
        "arxiv_id": "1707.06347",
        "title": "Proximal Policy Optimization Algorithms",
        "year": 2017
    },
    {
        "arxiv_id": "1312.5602",
        "title": "Playing Atari with Deep Reinforcement Learning",
        "year": 2013
    },
    {
        "arxiv_id": "2305.18290",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "year": 2023
    },
    {
        "arxiv_id": "2205.14135",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "year": 2022
    },
    {
        "arxiv_id": "2310.11511",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "year": 2023
    },
    {
        "arxiv_id": "1412.6980",
        "title": "Adam: A Method for Stochastic Optimization",
        "year": 2014
    }
]

def main():
    fetcher = ArxivFetcher()
    parser = PaperParser()
    chunker = SectionChunker()
    indexer = Indexer()

    print(f"[*] Starting ingestion of {len(TARGET_PAPERS)} Neural Net & RL papers into RAG memory...\n")
    initial_stats = indexer.get_collection_stats()
    print(f"[*] Initial collection stats: {initial_stats}\n")

    for p in TARGET_PAPERS:
        arxiv_id = p["arxiv_id"]
        title = p["title"]
        year = p["year"]

        print(f"\n{'='*60}")
        print(f"[*] Processing: [{arxiv_id}] {title} ({year})")
        print(f"{'='*60}")

        # 1. Fetch paper (LaTeX source or PDF fallback)
        try:
            fetch_res = fetcher.fetch_paper(arxiv_id, prefer_latex=True)
            folder = fetch_res["folder"]
            print(f"[+] Download/Cache complete: {folder} (LaTeX: {fetch_res.get('has_latex')})")
        except Exception as e:
            print(f"[!] Fetch failed for {arxiv_id}: {e}")
            continue

        # 2. Parse paper
        try:
            parsed_doc = parser.parse_paper(folder, arxiv_id=arxiv_id, title=title, year=year)
            if not parsed_doc or not parsed_doc.get("sections"):
                pdf_path = fetch_res.get("pdf_path")
                if pdf_path and pdf_path.exists():
                    print(f"[*] Falling back to PDF parser for {arxiv_id}...")
                    parsed_doc = parser._parse_pdf(pdf_path, arxiv_id=arxiv_id, default_title=title, year=year)
        except Exception as e:
            print(f"[!] Parsing failed for {arxiv_id}: {e}")
            continue

        # 3. Chunk sections
        try:
            chunks = chunker.chunk_paper(parsed_doc)
            print(f"[+] Created {len(chunks)} text/equation chunks.")
        except Exception as e:
            print(f"[!] Chunking failed for {arxiv_id}: {e}")
            continue

        # 4. Index into ChromaDB
        try:
            if chunks:
                indexed_count = indexer.index_paper_chunks(chunks)
                print(f"[+] Successfully indexed {indexed_count} chunks into ChromaDB 'paper_chunks'.")
            else:
                print(f"[!] No chunks generated for {arxiv_id}.")
        except Exception as e:
            print(f"[!] Indexing failed for {arxiv_id}: {e}")
            continue

        time.sleep(1)

    print("\n" + "="*60)
    final_stats = indexer.get_collection_stats()
    print(f"[*] Ingestion Complete! Final collection stats: {final_stats}")
    print("="*60)

if __name__ == "__main__":
    main()
