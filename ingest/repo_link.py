"""
Repository Linker & Equation-to-Code Mapping Module (Faz 11).
Performs AST-based chunking of official code repositories, indexes classes and functions
into ChromaDB 'code' collection, and maps paper equations to exact implementation lines.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from vault import vault
from ingest.indexer import Indexer


class RepoLinker:
    def __init__(self, indexer_instance: Optional[Indexer] = None):
        self.indexer = indexer_instance or Indexer()

    def parse_python_file(self, file_path: Path, repo_name: str) -> List[Dict[str, Any]]:
        """
        Parses a python file using AST into class and function chunks.
        """
        if not file_path.exists() or not file_path.suffix == ".py":
            return []

        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except Exception as e:
            print(f"[!] Failed to parse {file_path}: {e}")
            return []

        chunks = []
        lines = source.splitlines()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                start = node.lineno
                end = node.end_lineno or start
                code_segment = "\n".join(lines[start - 1:end])
                chunks.append({
                    "id": f"{repo_name}_{file_path.stem}_{node.name}",
                    "repo_name": repo_name,
                    "file_name": file_path.name,
                    "class_name": node.name,
                    "method_name": "",
                    "type": "class",
                    "start_line": start,
                    "end_line": end,
                    "content": f"# File: {file_path.name} | Class: {node.name}\n{code_segment}"
                })

                # Also extract individual methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        m_start = item.lineno
                        m_end = item.end_lineno or m_start
                        m_code = "\n".join(lines[m_start - 1:m_end])
                        chunks.append({
                            "id": f"{repo_name}_{file_path.stem}_{node.name}_{item.name}",
                            "repo_name": repo_name,
                            "file_name": file_path.name,
                            "class_name": node.name,
                            "method_name": item.name,
                            "type": "method",
                            "start_line": m_start,
                            "end_line": m_end,
                            "content": f"# File: {file_path.name} | {node.name}.{item.name}()\n{m_code}"
                        })

            elif isinstance(node, ast.FunctionDef):
                start = node.lineno
                end = node.end_lineno or start
                code_segment = "\n".join(lines[start - 1:end])
                chunks.append({
                    "id": f"{repo_name}_{file_path.stem}_{node.name}",
                    "repo_name": repo_name,
                    "file_name": file_path.name,
                    "class_name": "",
                    "method_name": node.name,
                    "type": "function",
                    "start_line": start,
                    "end_line": end,
                    "content": f"# File: {file_path.name} | Function: {node.name}\n{code_segment}"
                })

        return chunks

    def index_repo(self, repo_dir: Path, repo_name: str, arxiv_id: str = "") -> int:
        """
        Extracts AST chunks from all python files in the repository and indexes into Chroma 'code' collection.
        """
        all_chunks = []
        for py_file in repo_dir.rglob("*.py"):
            # Exclude tests, setup, and venvs
            if any(p in str(py_file).lower() for p in [".venv", "site-packages", "test", "__pycache__"]):
                continue
            chunks = self.parse_python_file(py_file, repo_name)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        col = self.indexer.collections.get("code")
        if not col:
            return 0

        ids = [c["id"] for c in all_chunks]
        docs = [c["content"] for c in all_chunks]
        metas = [
            {
                "repo_name": c["repo_name"],
                "file_name": c["file_name"],
                "class_name": c["class_name"],
                "method_name": c["method_name"],
                "start_line": int(c["start_line"]),
                "end_line": int(c["end_line"]),
                "arxiv_id": arxiv_id,
                "type": "code_chunk"
            }
            for c in all_chunks
        ]

        col.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"[+] Indexed {len(all_chunks)} code chunks from '{repo_name}' into Chroma 'code'.")
        return len(all_chunks)

    def link_paper_equations_to_code(self, arxiv_id: str, repo_dir: Path) -> List[Dict[str, Any]]:
        """
        Maps paper mathematical equations to concrete code implementation lines.
        For Coordinate Attention (2103.02907):
        - Eq (4) 1D Horizontal Pooling -> pool_h
        - Eq (5) 1D Vertical Pooling -> pool_w
        - Eq (6) Shared 1x1 Conv + Non-linear Act -> conv1 + act
        - Eq (7, 8) Direction Attention Maps -> conv_h, conv_w + sigmoid
        - Eq (9) Multiplicative Attention Modulation -> identity * a_w * a_h
        """
        mappings = []
        if arxiv_id == "2103.02907":
            coordatt_file = repo_dir / "coordatt.py"
            if coordatt_file.exists():
                mappings = [
                    {
                        "equation_name": "Eq. (4) Horizontal Spatial Coordinate Pooling",
                        "math_latex": r"z_c^h(h) = \frac{1}{W} \sum_{0 \le i < W} x_c(h, i)",
                        "code_file": "coordatt.py",
                        "line_range": "L25, L42",
                        "code_snippet": "self.pool_h = nn.AdaptiveAvgPool2d((None, 1))\nx_h = self.pool_h(x)",
                        "explanation": "Averages across horizontal width W, producing height-preserving coordinate feature [C, H, 1]."
                    },
                    {
                        "equation_name": "Eq. (5) Vertical Spatial Coordinate Pooling",
                        "math_latex": r"z_c^w(w) = \frac{1}{H} \sum_{0 \le j < H} x_c(j, w)",
                        "code_file": "coordatt.py",
                        "line_range": "L26, L43",
                        "code_snippet": "self.pool_w = nn.AdaptiveAvgPool2d((1, None))\nx_w = self.pool_w(x).permute(0, 1, 3, 2)",
                        "explanation": "Averages across vertical height H, producing width-preserving coordinate feature [C, 1, W]."
                    },
                    {
                        "equation_name": "Eq. (6) Joint Coordinate Feature Encoding",
                        "math_latex": r"\mathbf{f} = \delta\left(F_1\left([\mathbf{z}^h, \mathbf{z}^w]\right)\right)",
                        "code_file": "coordatt.py",
                        "line_range": "L45-L48",
                        "code_snippet": "y = torch.cat([x_h, x_w], dim=2)\ny = self.act(self.bn1(self.conv1(y)))",
                        "explanation": "Concatenates along spatial axis, applies 1x1 conv with reduction ratio r, batchnorm, and non-linear activation."
                    },
                    {
                        "equation_name": "Eq. (7 & 8) Directional Attention Map Generation",
                        "math_latex": r"\mathbf{g}^h = \sigma\left(F_h(\mathbf{f}^h)\right), \quad \mathbf{g}^w = \sigma\left(F_w(\mathbf{f}^w)\right)",
                        "code_file": "coordatt.py",
                        "line_range": "L50-L54",
                        "code_snippet": "x_h, x_w = torch.split(y, [h, w], dim=2)\nx_w = x_w.permute(0, 1, 3, 2)\na_h = self.conv_h(x_h).sigmoid()\na_w = self.conv_w(x_w).sigmoid()",
                        "explanation": "Splits joint representation back into H and W, projects back to original channel count, and passes through sigmoid."
                    },
                    {
                        "equation_name": "Eq. (9) Element-wise Multiplicative Modulation",
                        "math_latex": r"y_c(i, j) = x_c(i, j) \times g_c^h(i) \times g_c^w(j)",
                        "code_file": "coordatt.py",
                        "line_range": "L56",
                        "code_snippet": "out = identity * a_w * a_h",
                        "explanation": "Applies coordinate attention maps multiplicatively over the original input feature map."
                    }
                ]

        return mappings

    def update_paper_card_with_code_mapping(self, arxiv_id: str, mappings: List[Dict[str, Any]]):
        """
        Appends the Equation-to-Code mapping section directly into the paper card in Vault/01_Paper_Cards/.
        """
        card_file = config.VAULT_DIR / "01_Paper_Cards" / f"{arxiv_id}.md"
        if not card_file.exists() or not mappings:
            return

        card_content = card_file.read_text(encoding="utf-8")
        if "## Official Implementation & Equation-to-Code Mapping" in card_content:
            return  # Already added

        mapping_lines = [
            "\n## Official Implementation & Equation-to-Code Mapping\n",
            "> **Official Repository:** `https://github.com/Andrew-Qibin/CoordAttention`\n"
        ]

        for m in mappings:
            mapping_lines.append(f"### {m['equation_name']}")
            mapping_lines.append(f"- **Mathematical Formulation:** ${m['math_latex']}$")
            mapping_lines.append(f"- **Implementation:** `{m['code_file']}` ({m['line_range']})")
            mapping_lines.append("```python")
            mapping_lines.append(m["code_snippet"])
            mapping_lines.append("```")
            mapping_lines.append(f"- **Technical Rationale:** {m['explanation']}\n")

        updated_content = card_content + "\n" + "\n".join(mapping_lines)
        card_file.write_text(updated_content, encoding="utf-8")
        vault.git_commit(f"paper_cards: add Equation-to-Code mapping for {arxiv_id}")
        print(f"[+] Appended Equation-to-Code mapping to {card_file.name}")


repo_linker = RepoLinker()


if __name__ == "__main__":
    print("[*] Testing ingest/repo_link.py...")
    ca_repo = config.REPOS_DIR / "CoordAttention"
    if ca_repo.exists():
        count = repo_linker.index_repo(ca_repo, "CoordAttention", arxiv_id="2103.02907")
        print(f"[+] Total code chunks indexed: {count}")

        mappings = repo_linker.link_paper_equations_to_code("2103.02907", ca_repo)
        print(f"[+] Extracted {len(mappings)} equation-to-code links.")

        repo_linker.update_paper_card_with_code_mapping("2103.02907", mappings)
