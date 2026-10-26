"""
arXiv Paper Fetcher Module (Faz 3.2).
Downloads LaTeX source (e-print) or PDF fallback from arXiv,
unpacks archives into data/papers/<arxiv_id>/, and respects rate limits.
"""

import os
import sys
import time
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import requests

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class ArxivFetcher:
    def __init__(self, papers_dir: Path = None):
        self.papers_dir = Path(papers_dir or config.PAPERS_DIR).resolve()
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "User-Agent": "DL-Lab-Assistant/2.0 (mailto:academic-research@local.ai)"
        }

    def fetch_paper(self, arxiv_id: str, prefer_latex: bool = True) -> Dict[str, Any]:
        """
        Fetches an arXiv paper by ID.
        Checks local cache first.
        Tries LaTeX source (e-print) first, falls back to PDF.
        Returns metadata with paths.
        """
        clean_id = arxiv_id.strip()
        paper_folder = self.papers_dir / clean_id
        paper_folder.mkdir(parents=True, exist_ok=True)

        pdf_path = paper_folder / f"{clean_id}.pdf"
        tex_files = list(paper_folder.glob("*.tex"))

        if tex_files or pdf_path.exists():
            return {
                "arxiv_id": clean_id,
                "cached": True,
                "folder": paper_folder,
                "has_latex": len(tex_files) > 0,
                "pdf_path": pdf_path if pdf_path.exists() else None
            }

        # Polite delay before network request
        time.sleep(1.5)

        has_latex = False
        if prefer_latex:
            has_latex = self._download_and_extract_source(clean_id, paper_folder)

        # Fallback to PDF if no LaTeX extracted
        if not has_latex:
            print(f"[*] Downloading PDF for {clean_id}...")
            pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
            for attempt in range(1, 4):
                try:
                    resp = requests.get(pdf_url, headers=self.headers, timeout=30)
                    if resp.status_code == 200:
                        pdf_path.write_bytes(resp.content)
                        print(f"[+] PDF saved: {pdf_path.name}")
                        break
                    else:
                        print(f"[!] Failed to download PDF (HTTP {resp.status_code})")
                except Exception as e:
                    print(f"[!] PDF download attempt {attempt} failed: {e}")
                    time.sleep(2)

        tex_files = list(paper_folder.glob("*.tex"))
        return {
            "arxiv_id": clean_id,
            "cached": False,
            "folder": paper_folder,
            "has_latex": len(tex_files) > 0,
            "pdf_path": pdf_path if pdf_path.exists() else None
        }

    def _download_and_extract_source(self, arxiv_id: str, target_dir: Path) -> bool:
        """Downloads arXiv e-print source and extracts .tex files."""
        src_url = f"https://arxiv.org/e-print/{arxiv_id}"
        tar_path = target_dir / "source.tar.gz"

        for attempt in range(1, 4):
            try:
                print(f"[*] Attempting LaTeX source download for {arxiv_id} (Attempt {attempt}/3)...")
                resp = requests.get(src_url, headers=self.headers, timeout=30)
                if resp.status_code != 200:
                    print(f"[-] No source available (HTTP {resp.status_code}). Falling back to PDF.")
                    return False

                tar_path.write_bytes(resp.content)

                # Try extracting as tarfile
                extracted = False
                try:
                    with tarfile.open(tar_path, "r:*") as tar:
                        tar.extractall(path=target_dir)
                        extracted = True
                except tarfile.ReadError:
                    # Might be single gzipped file (.tex.gz)
                    try:
                        with gzip.open(tar_path, "rb") as gz:
                            decompressed = gz.read()
                            single_tex = target_dir / f"{arxiv_id}.tex"
                            single_tex.write_bytes(decompressed)
                            extracted = True
                    except Exception:
                        pass

                # Cleanup archive file
                if tar_path.exists():
                    tar_path.unlink()

                tex_files = list(target_dir.glob("*.tex"))
                if tex_files:
                    print(f"[+] LaTeX source extracted: {len(tex_files)} .tex files found.")
                    return True
                return False

            except Exception as e:
                print(f"[!] Source download attempt {attempt} error for {arxiv_id}: {e}")
                if tar_path.exists():
                    tar_path.unlink()
                time.sleep(2)

        return False


if __name__ == "__main__":
    fetcher = ArxivFetcher()
    print("[*] Testing arxiv_fetch.py with Coordinate Attention (2103.02907)...")
    res = fetcher.fetch_paper("2103.02907")
    print(f"Result: {res}")
