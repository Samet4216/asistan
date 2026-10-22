"""
Draft-to-Approve Workflow Module (Faz 8.3).
Prevents LLM hallucinations or unverified notes from polluting permanent memory.
Flow: Proposed Note -> Vault/Drafts/ -> User Review -> /approve -> Permanent Vault + ChromaDB Index.
"""

import sys
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from vault import vault
from ingest.indexer import Indexer


DRAFTS_DIR = config.VAULT_DIR / "Drafts"


class ApprovalManager:
    def __init__(self, indexer_instance: Optional[Indexer] = None):
        self.indexer = indexer_instance or Indexer()
        DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    def create_draft(
        self,
        title: str,
        content: str,
        category: str = "Theory_Notes",
        tags: List[str] = None,
        source: str = "assistant"
    ) -> Path:
        """
        Creates a draft markdown file in Vault/Drafts/ pending user approval.
        category options: Theory_Notes, Experiment_Logs, Code_Reviews, Paper_Cards
        """
        safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title.strip()).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_title}_{timestamp}.md"

        frontmatter = {
            "title": title,
            "category": category,
            "status": "draft",
            "source": source,
            "tags": tags or ["lab-note"],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Apply deterministic wikilinks before writing
        linked_content = vault.format_wikilinks(content)
        draft_path = vault.write_note("Drafts", filename, frontmatter, linked_content)
        vault.git_commit(f"draft: proposed note '{title}'")
        return draft_path

    def list_drafts(self) -> List[Dict[str, Any]]:
        """Lists all unapproved markdown files in Vault/Drafts/."""
        drafts = []
        for file_path in DRAFTS_DIR.glob("*.md"):
            try:
                raw = file_path.read_text(encoding="utf-8")
                meta = {}
                body = raw
                if raw.startswith("---"):
                    parts = raw.split("---", 2)
                    if len(parts) >= 3:
                        meta = yaml.safe_load(parts[1]) or {}
                        body = parts[2].strip()

                drafts.append({
                    "id": file_path.stem,
                    "filename": file_path.name,
                    "title": meta.get("title", file_path.stem),
                    "category": meta.get("category", "Theory_Notes"),
                    "status": meta.get("status", "draft"),
                    "created_at": meta.get("created_at", ""),
                    "path": file_path,
                    "preview": body[:150] + ("..." if len(body) > 150 else "")
                })
            except Exception:
                continue

        drafts.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return drafts

    def approve_draft(self, draft_id_or_filename: str) -> Optional[Path]:
        """
        Approves a draft:
        1. Moves file to permanent directory (e.g. 02_Theory_Notes, 03_Experiment_Logs).
        2. Updates frontmatter status to 'approved'.
        3. Indexes the note content into ChromaDB.
        4. Stages and commits to git.
        """
        # Locate file
        target_file = None
        for f in DRAFTS_DIR.glob("*.md"):
            if f.name == draft_id_or_filename or f.stem == draft_id_or_filename or draft_id_or_filename in f.stem:
                target_file = f
                break

        if not target_file or not target_file.exists():
            return None

        raw = target_file.read_text(encoding="utf-8")
        meta = {}
        body = raw
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                meta = yaml.safe_load(parts[1]) or {}
                body = parts[2].strip()

        category = meta.get("category", "Theory_Notes")
        cat_map = {
            "Theory_Notes": "02_Theory_Notes",
            "Experiment_Logs": "03_Experiment_Logs",
            "Code_Reviews": "04_Code_Reviews",
            "Paper_Cards": "01_Paper_Cards"
        }
        dest_folder = cat_map.get(category, "02_Theory_Notes")

        # Update metadata
        meta["status"] = "approved"
        meta["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        dest_dir = config.VAULT_DIR / dest_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_file = dest_dir / target_file.name

        # Write approved file
        frontmatter_str = yaml.dump(meta, default_flow_style=False, sort_keys=False).strip()
        final_content = f"---\n{frontmatter_str}\n---\n\n{body}\n"
        final_file.write_text(final_content, encoding="utf-8")

        # Delete draft
        target_file.unlink()

        # Index into ChromaDB
        self.indexer.index_vault_note(
            note_id=final_file.stem,
            content=f"Title: {meta.get('title', '')}\nCategory: {category}\n\n{body}",
            metadata={
                "category": category,
                "type": "vault_note",
                "approved_at": meta["approved_at"]
            },
            collection_name="experiments"
        )

        vault.git_commit(f"vault: approved note '{meta.get('title', final_file.stem)}' into {dest_folder}")
        return final_file

    def reject_draft(self, draft_id_or_filename: str) -> bool:
        """Deletes a draft proposal."""
        for f in DRAFTS_DIR.glob("*.md"):
            if f.name == draft_id_or_filename or f.stem == draft_id_or_filename or draft_id_or_filename in f.stem:
                f.unlink()
                vault.git_commit(f"vault: rejected draft '{f.stem}'")
                return True
        return False


approval_manager = ApprovalManager()


if __name__ == "__main__":
    print("[*] Testing memory/approval.py...")
    test_draft = approval_manager.create_draft(
        title="Coordinate Attention Tensor Shapes",
        content="In Coordinate Attention, pooled features have dimensions [C, H, 1] and [C, 1, W]. These avoid full 2D spatial collapse.",
        category="Theory_Notes",
        tags=["attention", "tensor-shapes"]
    )
    print(f"[+] Draft created at: {test_draft}")

    drafts = approval_manager.list_drafts()
    print(f"[+] Pending Drafts ({len(drafts)}):")
    for d in drafts:
        print(f" - [{d['id']}] {d['title']} ({d['category']})")

    approved = approval_manager.approve_draft(test_draft.stem)
    print(f"[+] Approved note saved to: {approved}")
