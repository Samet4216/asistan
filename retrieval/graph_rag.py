"""
GraphRAG: Knowledge Graph Augmented Retrieval Engine (Faz 13.0).
Traverses Obsidian Vault synaptic knowledge graphs, paper cards, and cross-concept wikilinks
to provide multi-hop semantic reasoning and cross-paper structural insights.
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
VAULT_DIR = PROJECT_ROOT / "Vault"


class GraphRAGEngine:
    """Extracts, indexes, and traverses Obsidian concept synapses and paper wikilinks."""

    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = vault_path or VAULT_DIR
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges_out: Dict[str, Set[str]] = {}
        self.edges_in: Dict[str, Set[str]] = {}
        self._build_graph()

    def _build_graph(self):
        """Scans Vault markdown files and constructs the bidirectional graph."""
        self.nodes.clear()
        self.edges_out.clear()
        self.edges_in.clear()

        if not self.vault_path.exists():
            return

        md_files = list(self.vault_path.rglob("*.md"))
        for f in md_files:
            stem = f.stem
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Extract title
            title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
            display_title = title_match.group(1).strip() if title_match else stem.replace("_", " ")

            # Extract tags from frontmatter
            tags = re.findall(r"-\s+([a-zA-Z0-9_-]+)", content[:400])

            # Extract wikilinks: [[target]] or [[target|label]]
            raw_links = re.findall(r"\[\[(.*?)\]\]", content)
            wikilinks = set()
            for wl in raw_links:
                clean_target = wl.split("|")[0].strip()
                if clean_target:
                    wikilinks.add(clean_target)

            # Extract mathematical summary if present
            math_formulas = re.findall(r"\$\$([\s\S]*?)\$\$", content)
            math_snippet = math_formulas[0].strip().replace("\n", " ") if math_formulas else ""

            # Extract first summary paragraph
            clean_lines = [
                l.strip() for l in content.split("\n")
                if l.strip() and not l.startswith("#") and not l.startswith("---") and not l.startswith("-")
            ]
            preview = " ".join(clean_lines[:2])[:250] if clean_lines else ""

            # Store node
            rel_folder = f.relative_to(self.vault_path).parts[0] if len(f.relative_to(self.vault_path).parts) > 1 else "Root"
            self.nodes[stem] = {
                "id": stem,
                "title": display_title,
                "folder": rel_folder,
                "preview": preview,
                "math": math_snippet,
                "tags": tags,
                "file_path": str(f)
            }
            self.edges_out[stem] = wikilinks

            if stem not in self.edges_in:
                self.edges_in[stem] = set()

        # Build reverse index (backlinks)
        for src, targets in self.edges_out.items():
            for tgt in targets:
                if tgt not in self.edges_in:
                    self.edges_in[tgt] = set()
                self.edges_in[tgt].add(src)

    def extract_seed_nodes(self, query: str, retrieved_chunks: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Identifies seed nodes in the knowledge graph matching the query or retrieved chunks."""
        q_lower = query.lower()
        matched_seeds = set()

        # 1. Match against node IDs and titles
        for node_id, data in self.nodes.items():
            nid_lower = node_id.lower().replace("_", " ")
            title_lower = data["title"].lower()

            # Word boundary or clean substring match
            if len(node_id) > 2 and (nid_lower in q_lower or title_lower in q_lower):
                matched_seeds.add(node_id)

            # Match arXiv IDs (e.g., 2305.18290, 1706.03762)
            if re.match(r"^\d{4}\.\d{4,5}$", node_id) and node_id in query:
                matched_seeds.add(node_id)

        # 2. Match against concepts in retrieved chunks
        if retrieved_chunks:
            for c in retrieved_chunks[:3]:
                c_content = c.get("content", "").lower()
                c_title = c.get("metadata", {}).get("title", "").lower()
                for node_id, data in self.nodes.items():
                    nid_lower = node_id.lower().replace("_", " ")
                    if len(node_id) > 4 and (nid_lower in c_title or nid_lower in c_content[:300]):
                        matched_seeds.add(node_id)

        # Fallback / entity normalization mapping for key DL topics
        topic_aliases = {
            "dpo": ["Direct_Preference_Optimization", "2305.18290", "Reward_Modeling"],
            "ppo": ["Policy_Gradient", "Reward_Modeling"],
            "flashattention": ["FlashAttention", "GPU_Memory_Hierarchy", "2205.14135"],
            "transformer": ["Transformer_Architecture", "Self_Attention", "1706.03762"],
            "coordinate attention": ["Coordinate_Attention_Tensor_Shapes_20261023_003658", "2103.02907"],
            "vit": ["Vision_Transformer", "2010.11929"],
            "resnet": ["Residual_Connections", "1512.03385"],
            "adam": ["Adam_Optimizer", "1412.6980"],
            "dqn": ["Deep_Q_Networks", "1312.5602"]
        }

        for kw, seeds in topic_aliases.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
                for s in seeds:
                    if s in self.nodes:
                        matched_seeds.add(s)

        return list(matched_seeds)[:5]

    def traverse_subgraph(self, seed_nodes: List[str], max_neighbors: int = 5) -> Dict[str, Any]:
        """
        Traverses 1-hop neighborhood across forward links and backlinks.
        Returns active subgraph with notes, connections, and cross-concept rationales.
        """
        if not seed_nodes:
            return {"active_nodes": [], "connections": [], "summary_count": 0}

        active_nodes = set(seed_nodes)
        connections = []

        for seed in seed_nodes:
            # Outgoing links
            out_neighbors = self.edges_out.get(seed, set())
            for n in list(out_neighbors)[:max_neighbors]:
                if n in self.nodes:
                    active_nodes.add(n)
                    connections.append({
                        "source": seed,
                        "target": n,
                        "type": "cites_or_extends",
                        "source_title": self.nodes[seed]["title"],
                        "target_title": self.nodes[n]["title"]
                    })

            # Incoming links (backlinks)
            in_neighbors = self.edges_in.get(seed, set())
            for n in list(in_neighbors)[:max_neighbors]:
                if n in self.nodes:
                    active_nodes.add(n)
                    connections.append({
                        "source": n,
                        "target": seed,
                        "type": "referenced_by",
                        "source_title": self.nodes[n]["title"],
                        "target_title": self.nodes[seed]["title"]
                    })

        # Collect node details
        detailed_nodes = []
        for n_id in active_nodes:
            node_data = self.nodes.get(n_id)
            if node_data:
                detailed_nodes.append(node_data)

        return {
            "active_nodes": detailed_nodes,
            "connections": connections,
            "summary_count": len(detailed_nodes)
        }

    def format_graph_context(self, subgraph: Dict[str, Any]) -> str:
        """Formats the traversed subgraph into an explicit, structured prompt block."""
        nodes = subgraph.get("active_nodes", [])
        connections = subgraph.get("connections", [])

        if not nodes:
            return ""

        lines = [
            "### 🕸️ [SİNAPTİK BİLGİ GRAFİĞİ BAĞLANTILARI (GraphRAG - Obsidian Vault)]",
            "Bu araştırma konusu Obsidian ikinci beyin grafiğimizde aşağıdaki kavram ve makale düğümleriyle doğrudan sinaptik bağa sahiptir:"
        ]

        # List nodes and their key formulas or definitions
        for n in nodes[:5]:
            title = n.get("title", n["id"])
            preview = n.get("preview", "")
            math = n.get("math", "")
            math_str = f" | Temel Denklem: $${math}$$" if math else ""
            lines.append(f"- **[[{title}]]** (`{n['id']}`): {preview}{math_str}")

        # List relationships
        if connections:
            lines.append("\n**🔗 Çapraz Kavram İlişkileri & Hiyerarşi:**")
            seen_conn = set()
            for c in connections[:6]:
                pair_key = (c["source_title"], c["target_title"])
                if pair_key not in seen_conn:
                    seen_conn.add(pair_key)
                    rel_type = "temel bağıntısına dayanır" if c["type"] == "cites_or_extends" else "tarafından referans verilir"
                    lines.append(f"  • `[[{c['source_title']}]]` $\\rightarrow$ `[[{c['target_title']}]]` ({rel_type})")

        lines.append(
            "\n*(Yönerge: Yanıtında sadece tek bir izole parçayı anlatma; yukarıdaki sinaptik grafiği kullanarak kavramların birbirini nasıl tamamladığını, "
            "tarihsel gelişimini veya mimari üstünlüklerini çapraz sentezle.)*\n"
        )
        return "\n".join(lines)


# Global GraphRAG engine instance
graph_rag_engine = GraphRAGEngine()


if __name__ == "__main__":
    print("[*] Testing GraphRAG Engine...")
    print(f"Total graph nodes parsed: {len(graph_rag_engine.nodes)}")
    test_query = "Compare PPO with DPO and reward modeling"
    seeds = graph_rag_engine.extract_seed_nodes(test_query)
    print(f"Seeds for '{test_query}': {seeds}")
    subgraph = graph_rag_engine.traverse_subgraph(seeds)
    print(f"Active nodes: {[n['id'] for n in subgraph['active_nodes']]}")
    print(f"Connections count: {len(subgraph['connections'])}")
    print("\nFormatted Context:\n")
    print(graph_rag_engine.format_graph_context(subgraph))
