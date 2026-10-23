"""
Memory Module: Profile & Knowledge State Manager (Faz 8.2).
Maintains Vault/00_Profile/Style.md and Vault/00_Profile/Knowledge_State.md.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from vault import vault


STYLE_FILE = config.VAULT_DIR / "00_Profile" / "Style.md"
KNOWLEDGE_FILE = config.VAULT_DIR / "00_Profile" / "Knowledge_State.md"
ABOUT_FILE = config.VAULT_DIR / "00_Profile" / "About_Me.md"


class ProfileManager:
    def __init__(self):
        vault.ensure_layout()

    def get_about_me(self) -> str:
        """Reads personal researcher profile from Vault/00_Profile/About_Me.md."""
        if ABOUT_FILE.exists():
            try:
                content = ABOUT_FILE.read_text(encoding="utf-8")
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        return parts[2].strip()
                return content.strip()
            except Exception:
                pass
        return "Araştırmacı Profili: Derin Öğrenme ve Bilgisayarlı Görü çalışıyor."

    def append_about_me(self, fact: str):
        """Appends a new personal fact or project note to About_Me.md."""
        current = self.get_about_me()
        new_content = current.strip() + f"\n- {fact}\n"
        frontmatter = {
            "type": "profile",
            "tags": ["profile", "about-me"]
        }
        vault.write_note("00_Profile", "About_Me.md", frontmatter, new_content)
        vault.git_commit(f"profile: added personal fact '{fact[:30]}'")

    def get_style(self) -> str:
        """Reads style instructions from Vault/00_Profile/Style.md."""
        if STYLE_FILE.exists():
            content = STYLE_FILE.read_text(encoding="utf-8")
            # Strip frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    return parts[2].strip()
            return content.strip()
        return "Give exact mathematical formulations and tensor dimensions. Minimal prose, maximal technical depth."

    def update_style(self, guidelines: List[str]):
        """Rewrites Style.md with updated user guidelines."""
        frontmatter = {
            "type": "profile",
            "tags": ["profile", "learning-style"]
        }
        body_lines = ["# User Learning & Response Style\n"]
        for g in guidelines:
            body_lines.append(f"- {g}")
        content = "\n".join(body_lines) + "\n"

        vault.write_note("00_Profile", "Style.md", frontmatter, content)
        vault.git_commit("profile: update learning style preferences")

    def get_knowledge_state(self) -> Dict[str, List[str]]:
        """Parses Mastered and In Progress topics from Knowledge_State.md."""
        if not KNOWLEDGE_FILE.exists():
            return {"mastered": [], "exploring": []}

        content = KNOWLEDGE_FILE.read_text(encoding="utf-8")
        mastered = []
        exploring = []
        current_section = None

        for line in content.split("\n"):
            line = line.strip()
            if "## Mastered" in line:
                current_section = "mastered"
            elif "## In Progress" in line or "## Exploring" in line:
                current_section = "exploring"
            elif line.startswith("-") and current_section:
                item = line.lstrip("-").strip()
                if current_section == "mastered":
                    mastered.append(item)
                else:
                    exploring.append(item)

        return {"mastered": mastered, "exploring": exploring}

    def add_concept(self, concept: str, status: str = "mastered"):
        """Adds or updates a concept in Knowledge_State.md."""
        state = self.get_knowledge_state()
        mastered = set(state["mastered"])
        exploring = set(state["exploring"])

        if status == "mastered":
            exploring.discard(concept)
            mastered.add(concept)
        else:
            if concept not in mastered:
                exploring.add(concept)

        frontmatter = {
            "type": "profile",
            "tags": ["profile", "knowledge-state"]
        }
        body = (
            "# Knowledge State\n\n"
            "## Mastered\n" +
            "\n".join(f"- {c}" for c in sorted(mastered)) + "\n\n"
            "## In Progress / Exploring\n" +
            "\n".join(f"- {c}" for c in sorted(exploring)) + "\n"
        )

        vault.write_note("00_Profile", "Knowledge_State.md", frontmatter, body)
        vault.git_commit(f"profile: updated concept '{concept}' -> {status}")


profile_manager = ProfileManager()


if __name__ == "__main__":
    print("[*] Testing memory/profile.py...")
    print(f"Current Style:\n{profile_manager.get_style()}\n")
    state = profile_manager.get_knowledge_state()
    print(f"Knowledge State: Mastered={state['mastered']}, Exploring={state['exploring']}")
