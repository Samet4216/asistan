"""
Paper Parser Module (Faz 3.3).
Extracts structured sections from LaTeX sources (.tex) or fallback PDFs (via PyMuPDF4LLM).
Preserves mathematical equations ($...$, align, equation) and section hierarchies without corruption.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pymupdf4llm
import fitz

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class PaperParser:
    def __init__(self):
        # Math detection regexes
        self.eq_block_pattern = re.compile(
            r"(\\begin\{(?:equation|align|gather|multline)\*?\}.*?\\end\{(?:equation|align|gather|multline)\*?\}|\$\$[\s\S]*?\$\$)",
            re.DOTALL
        )

    def parse_paper(self, paper_folder: Path, arxiv_id: str, title: str = "", year: int = 2021) -> Dict[str, Any]:
        """
        Parses a paper folder. Uses LaTeX parser if .tex files exist,
        otherwise falls back to PDF parser.
        """
        tex_files = list(paper_folder.glob("*.tex"))
        # Prefer main / full .tex if multiple
        main_tex = None
        for tf in tex_files:
            if any(name in tf.name.lower() for name in ["full", "main", "paper", arxiv_id]):
                main_tex = tf
                break
        if not main_tex and tex_files:
            main_tex = max(tex_files, key=lambda f: f.stat().st_size)

        if main_tex:
            try:
                tex_raw = main_tex.read_text(encoding="utf-8", errors="ignore")
                if "includepdf" in tex_raw or len(tex_raw) < 500:
                    pdf_files = list(paper_folder.glob("*.pdf"))
                    if pdf_files:
                        return self._parse_pdf(pdf_files[0], arxiv_id, default_title=title, year=year)
            except Exception:
                pass
            return self._parse_latex(main_tex, arxiv_id, default_title=title, year=year)

        # Fallback: check for PDF
        pdf_files = list(paper_folder.glob("*.pdf"))
        if pdf_files:
            return self._parse_pdf(pdf_files[0], arxiv_id, default_title=title, year=year)

        return {"error": f"No valid .tex or .pdf found in {paper_folder}"}

    def _parse_latex(self, tex_path: Path, arxiv_id: str, default_title: str = "", year: int = 2021) -> Dict[str, Any]:
        """Parses LaTeX file keeping equations and section structure intact."""
        raw = tex_path.read_text(encoding="utf-8", errors="replace")

        # Strip line comments (% not preceded by backslash)
        clean_lines = []
        for line in raw.splitlines():
            stripped = re.sub(r"(?<!\\)%.*$", "", line)
            clean_lines.append(stripped)
        text = "\n".join(clean_lines)

        # Recursively expand \input{...} and \include{...}
        base_dir = tex_path.parent
        def replace_input(match):
            rel_file = match.group(1).strip()
            if not rel_file.endswith(".tex"):
                rel_file += ".tex"
            sub_path = base_dir / rel_file
            if sub_path.exists():
                try:
                    sub_raw = sub_path.read_text(encoding="utf-8", errors="replace")
                    clean_sub = [re.sub(r"(?<!\\)%.*$", "", l) for l in sub_raw.splitlines()]
                    return "\n".join(clean_sub)
                except Exception:
                    return ""
            return ""

        for _ in range(3):
            if "\\input{" in text or "\\include{" in text:
                text = re.sub(r"\\(?:input|include)\{([^}]+)\}", replace_input, text)
            else:
                break

        # Extract title
        title_match = re.search(r"\\title\{([^}]+)\}", text)
        title = title_match.group(1).replace("\n", " ").strip() if title_match else default_title

        # Extract abstract
        abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ""

        # Find sections
        section_splits = re.split(r"\\(?:sub)?section\*?\{([^}]+)\}", text)
        sections = []

        if abstract:
            sections.append({
                "name": "Abstract",
                "text": self._clean_latex_text(abstract),
                "equations": self.eq_block_pattern.findall(abstract)
            })

        # section_splits gives [preamble, sec1_title, sec1_content, sec2_title, sec2_content, ...]
        if len(section_splits) > 1:
            for i in range(1, len(section_splits), 2):
                sec_name = section_splits[i].strip()
                sec_content = section_splits[i + 1] if i + 1 < len(section_splits) else ""
                clean_sec_text = self._clean_latex_text(sec_content)
                eqs = self.eq_block_pattern.findall(sec_content)

                if clean_sec_text:
                    sections.append({
                        "name": sec_name,
                        "text": clean_sec_text,
                        "equations": eqs
                    })
        else:
            # Fallback if no sections detected
            sections.append({
                "name": "Main Body",
                "text": self._clean_latex_text(text),
                "equations": self.eq_block_pattern.findall(text)
            })

        return {
            "arxiv_id": arxiv_id,
            "title": title or arxiv_id,
            "year": year,
            "source_type": "latex",
            "sections": sections,
            "total_sections": len(sections)
        }

    def _parse_pdf(self, pdf_path: Path, arxiv_id: str, default_title: str = "", year: int = 2021) -> Dict[str, Any]:
        """Converts PDF to Markdown via PyMuPDF4LLM and splits into sections."""
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
        # Split by markdown headings # or ##
        parts = re.split(r"\n(#{1,3}\s+[^\n]+)\n", md_text)
        sections = []

        if len(parts) > 1:
            # First part is title / abstract
            sections.append({
                "name": "Overview",
                "text": parts[0].strip(),
                "equations": []
            })
            for i in range(1, len(parts), 2):
                sec_name = parts[i].lstrip("# ").strip()
                sec_content = parts[i + 1] if i + 1 < len(parts) else ""
                if sec_content.strip():
                    sections.append({
                        "name": sec_name,
                        "text": sec_content.strip(),
                        "equations": []
                    })
        else:
            sections.append({
                "name": "Main Document",
                "text": md_text.strip(),
                "equations": []
            })

        return {
            "arxiv_id": arxiv_id,
            "title": default_title or pdf_path.stem,
            "year": year,
            "source_type": "pdf",
            "sections": sections,
            "total_sections": len(sections)
        }

    def _clean_latex_text(self, latex_str: str) -> str:
        """Removes common LaTeX macros while preserving math and structure."""
        t = latex_str
        # Replace cite and ref with readable placeholders
        t = re.sub(r"\\cite\{([^}]+)\}", r"[\1]", t)
        t = re.sub(r"\\ref\{([^}]+)\}", r"(\1)", t)
        # Remove text formatting macros but keep inner text
        t = re.sub(r"\\(?:textbf|textit|emph|underline|text)\{([^}]+)\}", r"\1", t)
        # Remove label tags
        t = re.sub(r"\\label\{[^}]+\}", "", t)
        # Clean extra newlines
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()


if __name__ == "__main__":
    parser = PaperParser()
    test_folder = Path("data/papers/2103.02907")
    if test_folder.exists():
        print("[*] Testing paper_parser.py on Coordinate Attention (2103.02907)...")
        parsed = parser.parse_paper(test_folder, "2103.02907", title="Coordinate Attention")
        print(f"Title: {parsed.get('title')}")
        print(f"Total Sections: {parsed.get('total_sections')}")
        for s in parsed.get("sections", [])[:5]:
            print(f" • Section: [{s['name']}] (Length: {len(s['text'])} chars, Eqs: {len(s['equations'])})")
    else:
        print("[!] Test folder not found, run arxiv_fetch.py first.")
