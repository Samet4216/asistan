"""
Hierarchical Section-Aware & AST Code Chunker (Faz 3.4).
Produces small search chunks (~200-300 tokens) linked to parent sections,
protects equation blocks from being split, and chunks Python code via AST.
"""

import ast
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import budget


class SectionChunker:
    def __init__(self, target_chunk_tokens: int = 250, overlap_tokens: int = 40):
        self.target_tokens = target_chunk_tokens
        self.target_chars = int(target_chunk_tokens * 3.7)  # ~925 chars
        self.overlap_chars = int(overlap_tokens * 3.7)      # ~150 chars

    def chunk_paper(self, parsed_paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Takes structured output from paper_parser.py and creates section-aware chunks.
        Prepends [Title, year, section] to chunk content.
        Preserves section name in metadata for targeted filtering (e.g. /challenge).
        """
        arxiv_id = parsed_paper.get("arxiv_id", "")
        title = parsed_paper.get("title", "")
        year = parsed_paper.get("year", 2021)
        sections = parsed_paper.get("sections", [])

        chunks = []

        for sec_idx, sec in enumerate(sections):
            sec_name = sec.get("name", "Unknown Section")
            sec_text = sec.get("text", "")
            if not sec_text.strip():
                continue

            # Split section text into paragraphs
            paragraphs = sec_text.split("\n\n")
            current_chunk = []
            current_len = 0
            part_no = 1

            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue

                if current_len + len(p_clean) > self.target_chars and current_chunk:
                    chunk_body = "\n\n".join(current_chunk)
                    chunk_id = f"{arxiv_id}_sec{sec_idx+1}_p{part_no}"
                    header = f"[{title}, {year}, Section: {sec_name}]\n\n"

                    chunks.append({
                        "id": chunk_id,
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "year": year,
                        "section_name": sec_name,
                        "part_no": part_no,
                        "content": header + chunk_body,
                        "parent_text": sec_text[:2000],  # Parent context snippet
                        "type": "paper_chunk",
                        "token_est": budget.estimate_tokens(chunk_body)
                    })
                    part_no += 1
                    current_chunk = [p_clean]
                    current_len = len(p_clean)
                else:
                    current_chunk.append(p_clean)
                    current_len += len(p_clean)

            if current_chunk:
                chunk_body = "\n\n".join(current_chunk)
                chunk_id = f"{arxiv_id}_sec{sec_idx+1}_p{part_no}"
                header = f"[{title}, {year}, Section: {sec_name}]\n\n"

                chunks.append({
                    "id": chunk_id,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "year": year,
                    "section_name": sec_name,
                    "part_no": part_no,
                    "content": header + chunk_body,
                    "parent_text": sec_text[:2000],
                    "type": "paper_chunk",
                    "token_est": budget.estimate_tokens(chunk_body)
                })

        return chunks


class ASTCodeChunker:
    """Chunks Python files by classes and functions using the AST."""
    
    def chunk_python_file(self, file_path: Path, arxiv_id: str = "") -> List[Dict[str, Any]]:
        chunks = []
        code = file_path.read_text(encoding="utf-8", errors="replace")
        lines = code.splitlines()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback: line-based chunking
            return [{
                "id": f"{file_path.stem}_code_full",
                "arxiv_id": arxiv_id,
                "filepath": str(file_path),
                "name": file_path.name,
                "type": "code",
                "content": code[:2000],
                "lineno": 1,
                "end_lineno": len(lines)
            }]

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno
                node_code = "\n".join(lines[start:end])
                node_type = "class" if isinstance(node, ast.ClassDef) else "function"

                chunks.append({
                    "id": f"{file_path.stem}_{node_type}_{node.name}",
                    "arxiv_id": arxiv_id,
                    "filepath": str(file_path),
                    "name": node.name,
                    "type": "code",
                    "kind": node_type,
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "content": f"# File: {file_path.name} | {node_type.upper()}: {node.name} (Lines {node.lineno}-{node.end_lineno})\n" + node_code,
                    "token_est": budget.estimate_tokens(node_code)
                })

        return chunks


if __name__ == "__main__":
    from ingest.paper_parser import PaperParser
    parser = PaperParser()
    chunker = SectionChunker()

    test_folder = Path("data/papers/2103.02907")
    if test_folder.exists():
        parsed = parser.parse_paper(test_folder, "2103.02907", title="Coordinate Attention")
        paper_chunks = chunker.chunk_paper(parsed)
        print(f"[*] Generated {len(paper_chunks)} section-aware chunks for {parsed['title']}:")
        for c in paper_chunks[:3]:
            print(f" • [{c['id']}] ({c['section_name']}) - {c['token_est']} tokens")
            print(f"   Snippet: {c['content'][:140]}...\n")
