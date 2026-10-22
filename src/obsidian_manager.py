"""
Obsidian Manager for SametBeyin Vault
Handles scanning notes, extracting wikilinks, and updating memory notes
according to the second-brain protocol in BEYIN.md.
"""

import os
import re
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import yaml

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")



def load_config(config_path: str = "config.yaml") -> dict:
    """Loads YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ObsidianManager:
    def __init__(self, vault_path: str = None):
        if vault_path is None:
            config = load_config()
            vault_path = config.get("paths", {}).get("obsidian_vault", "./SametBeyin")
        self.vault_path = Path(vault_path).resolve()
        
        # Locate Hafiza folder dynamically (handles emoji variations like 🔮 Hafiza)
        self.hafiza_path = self._find_folder_by_keyword("Hafiza")

    def _find_folder_by_keyword(self, keyword: str) -> Path:
        """Finds a subfolder inside the vault containing the keyword, even if it has emojis."""
        if not self.vault_path.exists():
            return None
        for item in self.vault_path.iterdir():
            if item.is_dir() and keyword.lower() in item.name.lower():
                return item
        return self.vault_path / keyword

    def scan_notes(self) -> List[Dict[str, Any]]:
        """
        Scans all .md files in the vault (excluding .obsidian and hidden folders).
        Extracts content, title, tags, and wikilinks [[...]].
        """
        notes = []
        wikilink_pattern = re.compile(r"\[\[(.*?)\]\]")
        tag_pattern = re.compile(r"(?:^|\s)#([a-zA-Z0-9_-]+)")

        if not self.vault_path.exists():
            print(f"[!] Vault path not found: {self.vault_path}")
            return notes

        for root, dirs, files in os.walk(self.vault_path):
            # Skip hidden and system folders like .obsidian
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if not file.endswith(".md"):
                    continue

                full_path = Path(root) / file
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    print(f"[!] Error reading {file}: {e}")
                    continue

                # Extract wikilinks [[Target|Display]] or [[Target]]
                raw_links = wikilink_pattern.findall(content)
                cleaned_links = [link.split("|")[0].strip() for link in raw_links]

                # Extract tags
                tags = tag_pattern.findall(content)

                rel_path = full_path.relative_to(self.vault_path)

                notes.append({
                    "title": file[:-3],
                    "filename": file,
                    "relative_path": str(rel_path).replace("\\", "/"),
                    "full_path": str(full_path),
                    "wikilinks": cleaned_links,
                    "tags": list(set(tags)),
                    "content": content,
                    "char_count": len(content),
                    "word_count": len(content.split())
                })

        return notes

    def update_last_session(self, ai_name: str, summary: str, done_list: List[str], open_topics: str = "None", mood: str = "Focused"):
        """
        Updates 'Son-Oturum.md' inside the Hafiza folder according to BEYIN.md protocol.
        """
        if not self.hafiza_path or not self.hafiza_path.exists():
            print("[!] Hafiza folder not found.")
            return False

        son_oturum_file = self.hafiza_path / "Son-Oturum.md"
        today_str = datetime.now().strftime("%Y-%m-%d")

        done_bullets = "\n".join([f"- {item}" for item in done_list]) if done_list else "- Belirtilmedi."

        new_entry = (
            f"## Oturum: {today_str}\n"
            f"**AI:** {ai_name}\n"
            f"**Özet:** {summary}\n"
            f"**Yapılanlar:**\n{done_bullets}\n"
            f"**Açık Kalan:** {open_topics}\n"
            f"**Ruh Hali:** {mood}\n\n"
        )

        existing_content = ""
        if son_oturum_file.exists():
            with open(son_oturum_file, "r", encoding="utf-8") as f:
                existing_content = f.read()

        # Place new entry right below '# Son Oturum\n\n' if header exists
        if "# Son Oturum" in existing_content:
            header, rest = existing_content.split("# Son Oturum", 1)
            updated_content = f"# Son Oturum\n\n{new_entry}{rest.lstrip()}"
        else:
            updated_content = f"# Son Oturum\n\n{new_entry}{existing_content}"

        with open(son_oturum_file, "w", encoding="utf-8") as f:
            f.write(updated_content)

        # Also append to Gunluk.md
        self._append_to_daily_log(today_str, ai_name, summary)
        return True

    def _append_to_daily_log(self, date_str: str, ai_name: str, summary: str):
        """Appends a concise bullet to Gunluk.md."""
        gunluk_file = self.hafiza_path / "Gunluk.md"
        if not gunluk_file.exists():
            return
        
        entry_line = f"- **{date_str} ({ai_name}):** {summary}\n"
        with open(gunluk_file, "a", encoding="utf-8") as f:
            f.write(entry_line)

    def print_summary(self):
        """Prints a verification report of the vault."""
        notes = self.scan_notes()
        total_links = sum(len(n["wikilinks"]) for n in notes)
        
        print("\n" + "="*50)
        print("📊 OBSIDIAN VAULT SCAN REPORT (SametBeyin)")
        print("="*50)
        print(f"📁 Vault Path: {self.vault_path}")
        print(f"📝 Total Notes Scanned: {len(notes)}")
        print(f"🔗 Total Wikilinks Found: {total_links}")
        print("-" * 50)
        print("Not Listesi:")
        for n in notes:
            link_str = f"-> Bağlantılar: {', '.join(n['wikilinks'])}" if n['wikilinks'] else "(Bağlantısız)"
            print(f" • [{n['relative_path']}] ({n['word_count']} kelime) {link_str}")
        print("="*50 + "\n")


if __name__ == "__main__":
    manager = ObsidianManager()
    manager.print_summary()
