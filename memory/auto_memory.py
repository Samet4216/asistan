"""
Autonomous User Memory & Profile Extraction Engine.
Monitors user messages in real-time, extracts personal attributes, projects,
skills, and learning preferences, and writes them permanently into Obsidian
Vault (Vault/00_Profile/About_Me.md & Vault/00_Profile/Style.md).
"""

import sys
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import llm
from vault import vault

ABOUT_FILE = config.VAULT_DIR / "00_Profile" / "About_Me.md"
STYLE_FILE = config.VAULT_DIR / "00_Profile" / "Style.md"

FIRST_PERSON_CUES = [
    "ben ", "benim ", "benim adım", "adım ", "okuyorum", "öğrenciyim",
    "mühendisim", "mühendisiyim", "çalışıyorum", "üzerinde çalışıyorum", "yapıyorum",
    "projem", "tezim", "bitirme projem", "kullanıyorum", "biliyorum", "ilgiliyim",
    "öğrenmek istiyorum", "aklında tut", "hafızana kaydet", "beni hatırla",
    "düşünüyorum", "planlıyorum", "yapmak istiyorum", "yapacağım", "fikrim var",
    "hedefim", "bana ... anlat", "kod ağırlıklı", "pratik anlat", "formül istemiyorum",
    "i am", "my name", "my project", "i'm working on", "i study", "remember that"
]


class AutoMemoryEngine:
    """Intelligent extractor and manager for user persona and personal memory."""

    def __init__(self):
        vault.ensure_layout()

    def has_personal_cues(self, text: str) -> bool:
        """Lightweight check to see if message contains potential user profile info."""
        lower = text.lower().strip()
        return any(cue in lower for cue in FIRST_PERSON_CUES)

    def extract_and_update(self, user_message: str) -> Optional[str]:
        """
        Analyzes the user message. If personal facts or preferences are found,
        updates About_Me.md and/or Style.md and returns a human-readable summary.
        """
        if not self.has_personal_cues(user_message):
            return None

        # 1. LLM-based structured extraction
        extraction_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a background personal memory extractor. Extract any new or updated personal "
                    "facts about the user from their message.\n"
                    "Output ONLY a valid JSON object with any of these keys (omit keys if not mentioned):\n"
                    "{\n"
                    '  "name": "User name if mentioned",\n'
                    '  "occupation_or_degree": "University, department, degree, or job role if mentioned",\n'
                    '  "projects": ["Specific project or thesis topic mentioned"],\n'
                    '  "skills_or_tools": ["Specific tools, frameworks, or languages mentioned"],\n'
                    '  "learning_preference": "Any requested explanation style (e.g. code-focused, practical, simplified)"\n'
                    "}\n"
                    "If no actual personal information about the user is stated, output {}."
                )
            },
            {"role": "user", "content": user_message}
        ]

        try:
            raw_response = "".join(list(llm.stream_generate(extraction_prompt, max_tokens=200, temperature=0.0)))
            # Clean markdown codeblocks
            clean_json = re.sub(r'```json\s*', '', raw_response)
            clean_json = re.sub(r'```\s*', '', clean_json).strip()
            
            data = json.loads(clean_json)
        except Exception:
            return None

        if not data or not isinstance(data, dict):
            return None

        # Check if at least one meaningful field is present
        has_data = any(data.get(k) for k in ["name", "occupation_or_degree", "projects", "skills_or_tools", "learning_preference"])
        if not has_data:
            return None

        # 2. Update Profile & Obsidian files
        return self._apply_profile_update(data, original_msg=user_message)

    def _apply_profile_update(self, data: Dict[str, Any], original_msg: str) -> str:
        """Applies extracted profile facts into About_Me.md and Style.md."""
        current_profile = self._parse_current_about_me()

        changes_summary = []

        # Name
        if data.get("name"):
            current_profile["name"] = data["name"].strip()
            changes_summary.append(f"İsim: {current_profile['name']}")

        # Occupation / Degree
        if data.get("occupation_or_degree"):
            current_profile["occupation"] = data["occupation_or_degree"].strip()
            changes_summary.append(f"Bölüm/Meslek: {current_profile['occupation']}")

        # Projects
        if data.get("projects"):
            new_projs = data["projects"] if isinstance(data["projects"], list) else [data["projects"]]
            for p in new_projs:
                p_clean = str(p).strip()
                if p_clean and p_clean not in current_profile["projects"]:
                    current_profile["projects"].append(p_clean)
                    changes_summary.append(f"Proje: {p_clean}")

        # Skills & Tools
        if data.get("skills_or_tools"):
            new_skills = data["skills_or_tools"] if isinstance(data["skills_or_tools"], list) else [data["skills_or_tools"]]
            for s in new_skills:
                s_clean = str(s).strip()
                if s_clean and s_clean not in current_profile["skills"]:
                    current_profile["skills"].append(s_clean)
                    changes_summary.append(f"Yetkinlik: {s_clean}")

        # Learning Preferences
        if data.get("learning_preference"):
            pref = str(data["learning_preference"]).strip()
            if pref and pref not in current_profile["preferences"]:
                current_profile["preferences"].append(pref)
                changes_summary.append(f"Anlatım Tarzı: {pref}")
                self._update_style_file(pref)

        # Write updated About_Me.md
        self._write_about_me(current_profile)

        summary_text = ", ".join(changes_summary)
        return summary_text if summary_text else "Profil güncellendi"

    def _parse_current_about_me(self) -> Dict[str, Any]:
        """Parses current fields from About_Me.md."""
        profile = {
            "name": "Kullanıcı / Araştırmacı",
            "occupation": "Mekatronik Mühendisliği & Derin Öğrenme Araştırmacısı",
            "hardware": "ASUS TUF A15 (AMD Ryzen CPU + NVIDIA RTX 4050 6GB Laptop GPU + 16GB RAM)",
            "projects": [
                "Hafif dikkat mekanizmaları ve model optimizasyonu (YOLO, Coordinate Attention)",
                "Yerel, gizlilik odaklı derin öğrenme asistanı"
            ],
            "skills": ["Python", "PyTorch", "Derin Öğrenme"],
            "preferences": []
        }

        if not ABOUT_FILE.exists():
            return profile

        try:
            content = ABOUT_FILE.read_text(encoding="utf-8")
            # Name extract
            m_name = re.search(r'\*\*Kullanıcı(?:\s*/\s*Araştırmacı)?:\*\*\s*(.+)', content)
            if m_name and m_name.group(1).strip() not in ("Araştırmacı", "Kullanıcı"):
                profile["name"] = m_name.group(1).strip()

            # Occupation extract
            m_occ = re.search(r'\*\*(?:Bölüm / Meslek|Uzmanlık Alanı):\*\*\s*(.+)', content)
            if m_occ:
                profile["occupation"] = m_occ.group(1).strip()

            # Hardware extract
            m_hw = re.search(r'\*\*Donanım:\*\*\s*(.+)', content)
            if m_hw:
                profile["hardware"] = m_hw.group(1).strip()

        except Exception:
            pass

        return profile

    def _write_about_me(self, profile: Dict[str, Any]):
        """Renders and writes structured About_Me.md to Obsidian Vault."""
        frontmatter = {
            "type": "profile",
            "tags": ["profile", "about-me", "second-brain"]
        }

        projs_str = "\n".join(f"  - {p}" for p in profile["projects"]) if profile["projects"] else "  - Henüz proje eklenmedi"
        skills_str = ", ".join(profile["skills"]) if profile["skills"] else "Belirtilmedi"
        prefs_str = "\n".join(f"  - {pr}" for pr in profile["preferences"]) if profile["preferences"] else "  - Standart akademik ve pratik derin öğrenme yaklaşımı"

        body = f"""# Araştırmacı Profili (Benim Hakkımda)

### 👤 Kimlik & Akademik Durum
- **Kullanıcı / Araştırmacı:** {profile['name']}
- **Bölüm / Meslek:** {profile['occupation']}
- **Teknik Araçlar & Diller:** {skills_str}

### 💻 Donanım & Laboratuvar Ortamı
- **Donanım:** {profile['hardware']}
- **Çalışma Modu:** Yerel CUDA & Gizlilik Odaklı İkinci Beyin

### 🚀 Aktif Projeler & İlgi Alanları
{projs_str}

### 🎯 Tercih Edilen Öğrenme & Anlatım Tarzı
{prefs_str}
"""
        vault.write_note("00_Profile", "About_Me.md", frontmatter, body)
        vault.git_commit(f"profile: auto-updated memory for user '{profile['name']}'")

    def _update_style_file(self, new_pref: str):
        """Appends new style preference to Style.md."""
        try:
            frontmatter = {"type": "profile", "tags": ["profile", "style"]}
            current = ""
            if STYLE_FILE.exists():
                current = STYLE_FILE.read_text(encoding="utf-8")
                if "---" in current:
                    parts = current.split("---", 2)
                    if len(parts) >= 3:
                        current = parts[2].strip()

            updated = current.strip() + f"\n- {new_pref}\n"
            vault.write_note("00_Profile", "Style.md", frontmatter, updated)
        except Exception:
            pass


auto_memory = AutoMemoryEngine()


if __name__ == "__main__":
    print("[*] Testing auto_memory engine...")
    test_msg = "Benim adım Samet, Mekatronik Mühendisliği 4. sınıf öğrencisiyim. İHA için termal kamera ile nesne tespiti projesi yapıyorum, bana kod ağırlıklı anlat."
    print("Has cues:", auto_memory.has_personal_cues(test_msg))
    res = auto_memory.extract_and_update(test_msg)
    print("Update Result:", res)
    print("New About_Me.md:\n", ABOUT_FILE.read_text(encoding="utf-8"))
