"""
Paper Cards Generator Module (Faz 6).
Offline batch extraction of structured paper summary cards using JSON schemas.
Key mathematical equations are copied directly from the parser (never hallucinated by LLM).
Saves to Vault/01_Paper_Cards/<arxiv_id>.md with draft status.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import llm
import vault
from ingest.paper_parser import PaperParser
from ingest.indexer import Indexer


CARD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "problem": {"type": "string"},
        "method": {"type": "string"},
        "datasets": {"type": "array", "items": {"type": "string"}},
        "results": {"type": "string"},
        "limitations": {"type": "string"},
        "summary": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["problem", "method", "datasets", "results", "limitations", "summary", "keywords"]
}


class PaperCardGenerator:
    def __init__(self):
        self.parser = PaperParser()
        self.vault = vault.vault
        self.indexer = Indexer()

    def generate_card(self, paper_folder: Path, arxiv_id: str, title: str = "", year: int = 2021) -> Path:
        """
        Parses paper, runs LLM with JSON schema on key sections,
        attaches ground-truth equations, and saves to Vault/01_Paper_Cards/.
        """
        parsed = self.parser.parse_paper(paper_folder, arxiv_id, title=title, year=year)
        if "error" in parsed:
            raise ValueError(parsed["error"])

        # Filter relevant sections for card generation
        key_sections = []
        all_eqs = []
        for s in parsed.get("sections", []):
            name_lower = s["name"].lower()
            all_eqs.extend(s.get("equations", []))
            if any(k in name_lower for k in ["abstract", "intro", "method", "coordinate attention", "experiment", "result", "conclusion", "limitation"]):
                key_sections.append(f"### Section: {s['name']}\n{s['text'][:1500]}")

        context_text = "\n\n".join(key_sections)[:6000]

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a rigorous deep learning research assistant. "
                    "Analyze the provided academic paper text and output a structured research summary in JSON. "
                    "Be factual, concise, and technical. Do not guess numbers not present in the text."
                )
            },
            {
                "role": "user",
                "content": f"Paper Title: {parsed['title']}\nYear: {year}\n\nContent:\n{context_text}"
            }
        ]

        print(f"[*] Generating structured card for {arxiv_id} via Ollama...")
        raw_json = llm.generate(prompt_messages, schema=CARD_JSON_SCHEMA, max_tokens=700)
        
        try:
            card_data = json.loads(raw_json)
        except Exception:
            card_data = {
                "problem": "Failed to parse JSON",
                "method": raw_json[:300],
                "datasets": [],
                "results": "",
                "limitations": "",
                "summary": raw_json[:300],
                "keywords": []
            }

        # Build Markdown content
        eq_bullets = "\n".join([f"$$\n{eq.strip()}\n$$" for eq in all_eqs[:5]]) if all_eqs else "*No explicit equation blocks found in LaTeX.*"
        datasets_str = ", ".join(card_data.get("datasets", [])) or "N/A"
        keywords_str = ", ".join(card_data.get("keywords", [])) or "deep learning"

        body = (
            f"# Paper Card: {parsed['title']} ({year})\n\n"
            f"**arXiv ID:** `{arxiv_id}` | **Keywords:** {keywords_str}\n\n"
            f"## 🎯 Problem Statement\n{card_data.get('problem', '')}\n\n"
            f"## ⚙️ Proposed Method\n{card_data.get('method', '')}\n\n"
            f"## 📐 Key Equations (Extracted from source)\n{eq_bullets}\n\n"
            f"## 📊 Datasets & Empirical Results\n"
            f"- **Datasets:** {datasets_str}\n"
            f"- **Results:** {card_data.get('results', '')}\n\n"
            f"## ⚠️ Limitations & Failure Modes\n{card_data.get('limitations', '')}\n\n"
            f"## 📝 Summary\n{card_data.get('summary', '')}\n"
        )

        frontmatter = {
            "type": "paper_card",
            "arxiv_id": arxiv_id,
            "title": parsed["title"],
            "year": year,
            "status": "draft",
            "source": "model",
            "tags": ["paper-card", "deep-learning"] + card_data.get("keywords", [])
        }

        card_path = self.vault.write_note("01_Paper_Cards", f"{arxiv_id}", frontmatter, body)
        print(f"[+] Paper card written to: {card_path}")

        card_emb = self.indexer.embedder.embed_documents([body])
        self.indexer.collections["paper_cards"].upsert(
            ids=[f"card_{arxiv_id}"],
            documents=[body],
            embeddings=card_emb,
            metadatas=[{
                "arxiv_id": arxiv_id,
                "title": parsed["title"],
                "year": year,
                "type": "paper_card",
                "status": "draft"
            }]
        )

        print(f"[+] Paper card indexed in Chroma 'paper_cards'.")
        return card_path


if __name__ == "__main__":
    generator = PaperCardGenerator()
    test_p = Path("data/papers/2103.02907")
    if test_p.exists():
        c_path = generator.generate_card(test_p, "2103.02907", title="Coordinate Attention", year=2021)
        print(f"Card generated: {c_path}")
