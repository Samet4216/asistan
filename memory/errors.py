"""
Memory Module: Error History & Engineering Gotchas (Faz 8.1).
Parses Vault/00_Profile/Errors.md, syncs with ChromaDB 'errors' collection,
and injects relevant past failure mitigations within the 350-token budget.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import budget
from vault import vault
from ingest.indexer import Indexer


ERRORS_FILE = config.VAULT_DIR / "00_Profile" / "Errors.md"


class ErrorMemoryManager:
    def __init__(self, indexer_instance: Optional[Indexer] = None):
        self.indexer = indexer_instance or Indexer()

    def load_errors(self) -> List[Dict[str, Any]]:
        """Parses Vault/00_Profile/Errors.md into structured error entries."""
        if not ERRORS_FILE.exists():
            return []

        text = ERRORS_FILE.read_text(encoding="utf-8")
        # Split on markdown bullet points starting with - **[CODE]**:
        blocks = re.split(r'\n-\s+\*\*\[(.*?)\]\*\*:', text)
        errors = []

        if len(blocks) < 2:
            return []

        # blocks[0] is frontmatter and title; subsequent items are code, body, code, body...
        for i in range(1, len(blocks), 2):
            code = blocks[i].strip()
            body = blocks[i + 1].strip() if (i + 1) < len(blocks) else ""

            symptom_match = re.search(r'-\s*\*Symptom\*:\s*(.*?)(?=\n\s*-\s*\*|\Z)', body, re.DOTALL)
            context_match = re.search(r'-\s*\*Context\*:\s*(.*?)(?=\n\s*-\s*\*|\Z)', body, re.DOTALL)
            fix_match = re.search(r'-\s*\*Verified Fix\*:\s*(.*?)(?=\n\s*-\s*\*|\Z)', body, re.DOTALL)

            symptom = symptom_match.group(1).strip() if symptom_match else ""
            ctx = context_match.group(1).strip() if context_match else ""
            fix = fix_match.group(1).strip() if fix_match else ""

            errors.append({
                "id": f"error_{code.lower()}",
                "code": code,
                "symptom": symptom,
                "context": ctx,
                "mitigation": fix,
                "raw_text": f"[{code}] Symptom: {symptom} | Context: {ctx} | Fix: {fix}"
            })

        return errors

    def sync_to_chroma(self) -> int:
        """Indexes all error entries into ChromaDB collection 'errors'."""
        errors = self.load_errors()
        if not errors:
            return 0
        self.indexer.index_errors(errors)
        return len(errors)

    def get_relevant_errors(self, query: str, max_tokens: int = None) -> str:
        """
        Retrieves top relevant error records from ChromaDB and packs them into token budget.
        """
        max_budget = max_tokens or config.TOKEN_BUDGET.get("error_history", 350)
        col = self.indexer.collections.get("errors")
        if not col or col.count() == 0:
            self.sync_to_chroma()
            col = self.indexer.collections.get("errors")

        if not col or col.count() == 0:
            return ""

        try:
            results = col.query(
                query_texts=[query],
                n_results=min(3, col.count())
            )
            docs = results.get("documents", [[]])[0]
        except Exception:
            # Fallback to loading directly from file
            errors = self.load_errors()
            docs = [e["raw_text"] for e in errors[:3]]

        if not docs:
            return ""

        # Format and fit into token budget
        formatted_entries = []
        for d in docs:
            formatted_entries.append(f"* {d}")

        full_text = "\n".join(formatted_entries)
        # Verify budget
        est_tokens = budget.estimate_tokens(full_text)
        if est_tokens > max_budget:
            # Truncate to first 1-2 entries
            full_text = "\n".join(formatted_entries[:1])

        return full_text

    def add_error_entry(self, code: str, symptom: str, context: str, fix: str) -> None:
        """Appends a verified new error gotcha to Errors.md, syncs to Chroma, and commits to git."""
        new_block = (
            f"\n- **[{code}]**:\n"
            f"  - *Symptom*: {symptom}\n"
            f"  - *Context*: {context}\n"
            f"  - *Verified Fix*: {fix}\n"
        )
        if ERRORS_FILE.exists():
            existing = ERRORS_FILE.read_text(encoding="utf-8")
            ERRORS_FILE.write_text(existing + new_block, encoding="utf-8")
        else:
            ERRORS_FILE.write_text(f"---\ntype: profile\ntags: [profile, error-history]\n---\n# Personal Error History\n{new_block}", encoding="utf-8")

        self.sync_to_chroma()
        vault.git_commit(f"memory: record error [{code}]")


error_manager = ErrorMemoryManager()


if __name__ == "__main__":
    print("[*] Testing memory/errors.py...")
    count = error_manager.sync_to_chroma()
    print(f"[+] Loaded & Synced {count} errors to ChromaDB.")

    test_q = "I am getting CUDA out of memory when training transformer on RTX 4050"
    rel_errors = error_manager.get_relevant_errors(test_q)
    print("\n[+] Retrieved Relevant Error Context:")
    print(rel_errors)
