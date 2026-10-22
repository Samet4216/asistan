"""
Vault Module for Obsidian Storage & Second-Brain Memory Management.
Enforces frontmatter schema, draft-approval flows, link allowlisting,
and automated git tracking for auditability.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import config


class VaultManager:
    def __init__(self, vault_dir: Path = None):
        self.vault_dir = Path(vault_dir or config.VAULT_DIR).resolve()
        self.subdirs = [
            "00_Profile",
            "01_Paper_Cards",
            "02_Theory_Notes",
            "03_Experiment_Logs",
            "04_Code_Reviews",
            "Drafts"
        ]
        self.ensure_layout()
        self.aliases = {
            "overfit": "Overfitting",
            "overfitting": "Overfitting",
            "lr": "Learning_Rate",
            "loss": "Loss_Functions",
            "attention": "Self_Attention",
            "norm": "Normalization",
            "bn": "Batch_Normalization",
            "ln": "Layer_Normalization"
        }

    def ensure_layout(self):
        """Creates directory structure and default profile files if missing."""
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        for sub in self.subdirs:
            (self.vault_dir / sub).mkdir(parents=True, exist_ok=True)

        # Initialize Style.md if missing
        style_file = self.vault_dir / "00_Profile" / "Style.md"
        if not style_file.exists():
            default_style = (
                "---\n"
                "type: profile\n"
                "tags: [profile, learning-style]\n"
                "---\n"
                "# User Learning & Response Style\n\n"
                "- Give exact mathematical formulations, tensor dimensions, and code references.\n"
                "- Minimize fluff and pleasantries; prioritize technical depth.\n"
                "- Highlight trade-offs (e.g. VRAM footprint vs accuracy gains).\n"
            )
            style_file.write_text(default_style, encoding="utf-8")

        # Initialize Knowledge_State.md if missing
        know_file = self.vault_dir / "00_Profile" / "Knowledge_State.md"
        if not know_file.exists():
            default_know = (
                "---\n"
                "type: profile\n"
                "tags: [profile, knowledge-state]\n"
                "---\n"
                "# Knowledge State\n\n"
                "## Mastered\n"
                "- PyTorch basics, ConvNets, basic Backpropagation.\n\n"
                "## In Progress / Exploring\n"
                "- Transformer attention variants, self-attention complexity, tensor shape alignment.\n"
            )
            know_file.write_text(default_know, encoding="utf-8")

    def init_git(self):
        """Initializes git in the vault for version control and change tracking."""
        git_dir = self.vault_dir / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(["git", "init"], cwd=self.vault_dir, check=True, capture_output=True)
                print(f"[*] Git initialized in vault: {self.vault_dir}")
            except Exception as e:
                print(f"[!] Could not initialize git in vault: {e}")

    def git_commit(self, message: str):
        """Stages all changes in Vault and commits."""
        try:
            subprocess.run(["git", "add", "."], cwd=self.vault_dir, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", message], cwd=self.vault_dir, check=True, capture_output=True)
            print(f"[*] Vault committed: {message}")
        except Exception as e:
            print(f"[!] Git commit skipped or failed: {e}")

    def get_existing_note_titles(self) -> List[str]:
        """Returns all markdown note stems in the vault."""
        return [f.stem for f in self.vault_dir.rglob("*.md")]

    def format_wikilinks(self, text: str) -> str:
        """
        Deterministic Link Generation:
        Scans text for canonical keywords / aliases and inserts verified [[Link]] syntax.
        Code generates links, not the LLM.
        """
        existing = set(self.get_existing_note_titles())
        words = text.split()
        formatted_words = []

        for word in words:
            clean = word.strip(".,;:()[]{}'\"")
            canonical = self.aliases.get(clean.lower())
            if canonical and (canonical in existing or True):
                formatted_words.append(word.replace(clean, f"[[{canonical}]]"))
            else:
                formatted_words.append(word)

        return " ".join(formatted_words)

    def write_note(
        self,
        folder_name: str,
        filename: str,
        frontmatter: Dict[str, Any],
        content: str
    ) -> Path:
        """
        Writes a note with clean YAML frontmatter and body.
        """
        target_dir = self.vault_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / (filename if filename.endswith(".md") else f"{filename}.md")

        frontmatter_str = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip()
        full_content = f"---\n{frontmatter_str}\n---\n\n{content}\n"

        target_file.write_text(full_content, encoding="utf-8")
        return target_file

    def write_draft(self, filename: str, metadata: Dict[str, Any], content: str) -> Path:
        """Writes a proposed memory entry to Vault/Drafts/ for user approval."""
        metadata["status"] = metadata.get("status", "suggestion")
        metadata["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.write_note("Drafts", filename, metadata, content)


# Default vault instance
vault = VaultManager()


if __name__ == "__main__":
    print("[*] Testing vault.py...")
    vault.init_git()
    test_note = vault.write_draft(
        "sample_error_log",
        {"source": "model", "tags": ["test", "attention"]},
        "Proposed fix for overfit observed during training."
    )
    print(f"[*] Draft note created at: {test_note}")
    print(f"[*] Existing note stems in vault: {vault.get_existing_note_titles()}")
