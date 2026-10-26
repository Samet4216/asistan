"""
Reinforcement Learning & Continuous Preference Alignment Engine (Faz 9.0).
Enables the assistant to learn the user's personal communication style, technical
depth preferences, and feedback signals (Explicit 👍/👎 and Implicit Conversational
Critiques).

Persists:
1. Dynamic Reinforcement Policy in Vault/00_Profile/Reinforcement_Policy.md & Style.md
2. Direct Preference Optimization (DPO) dataset in data/rl_preferences/preferences.jsonl
3. Metrics & Cumulative Rewards in data/rl_preferences/learning_metrics.json
"""

import sys
import os
import io
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import llm
from vault import vault

RL_DIR = PROJECT_ROOT / "data" / "rl_preferences"
RL_DIR.mkdir(parents=True, exist_ok=True)

PREFERENCES_FILE = RL_DIR / "preferences.jsonl"
METRICS_FILE = RL_DIR / "learning_metrics.json"

STYLE_FILE = config.VAULT_DIR / "00_Profile" / "Style.md"
POLICY_FILE = config.VAULT_DIR / "00_Profile" / "Reinforcement_Policy.md"

# Implicit feedback trigger phrases
POSITIVE_CUES = [
    "harika", "süper", "mükemmel", "tam istediğim gibi", "çok iyi", "eline sağlık",
    "işte bu", "tam aradığım", "çok net oldu", "güzel özetledin", "aynen böyle",
    "harika anlattın", "teşekkürler tam", "perfect", "great", "exactly what i wanted"
]

CRITIQUE_CUES = [
    "çok uzun", "kısa kes", "kısa tut", "daha kısa", "özet geç", "çok uzattın",
    "daha çok kod", "kod ver", "kod ağırlıklı", "formül istemiyorum", "formülleri açıkla",
    "teoriyi geç", "pratik anlat", "mekatronik açısından", "donanım açısından",
    "öyle değil", "bunu sevmedim", "anlamadım", "daha basit", "daha sade",
    "böyle anlatma", "bir dahakine", "bundan sonra", "too long", "more code", "simpler"
]


class RLFeedbackEngine:
    """Continuous Reinforcement Learning & Style Adaptation Engine."""

    def __init__(self):
        vault.ensure_layout()
        self._ensure_metrics_file()

    def _ensure_metrics_file(self):
        if not METRICS_FILE.exists():
            default_metrics = {
                "total_feedbacks": 0,
                "positive_count": 0,
                "negative_count": 0,
                "cumulative_reward": 0.0,
                "learned_axioms_count": 0,
                "last_updated": datetime.now().isoformat()
            }
            METRICS_FILE.write_text(json.dumps(default_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_metrics(self) -> Dict[str, Any]:
        """Loads cumulative RL metrics."""
        try:
            if METRICS_FILE.exists():
                return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {
            "total_feedbacks": 0,
            "positive_count": 0,
            "negative_count": 0,
            "cumulative_reward": 0.0,
            "learned_axioms_count": 0,
            "last_updated": datetime.now().isoformat()
        }

    def _update_metrics(self, reward: float):
        metrics = self.get_metrics()
        metrics["total_feedbacks"] += 1
        if reward > 0:
            metrics["positive_count"] += 1
        elif reward < 0:
            metrics["negative_count"] += 1
        metrics["cumulative_reward"] = round(metrics.get("cumulative_reward", 0.0) + reward, 2)
        metrics["last_updated"] = datetime.now().isoformat()
        metrics["learned_axioms_count"] = len(self.get_active_axioms())
        try:
            METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def record_feedback(
        self,
        query: str,
        response: str,
        reward: float,
        style_tag: Optional[str] = None,
        user_comment: Optional[str] = None,
        source: str = "explicit"
    ) -> Dict[str, Any]:
        """
        Records an explicit or implicit user preference entry (Reward: +1.0 / -1.0).
        Persists into preferences.jsonl (DPO format candidate).
        If negative or constructive, triggers policy adaptation.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "reward": float(reward),
            "style_tag": style_tag or "general",
            "user_comment": user_comment or "",
            "query": query,
            "response": response[:1200]
        }

        # Append to JSONL
        try:
            with open(PREFERENCES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[!] Error writing to preferences.jsonl: {e}")

        # Update metrics
        self._update_metrics(reward)

        # Trigger Style Policy Evolution
        learned_axiom = None
        if style_tag or user_comment or reward != 0:
            learned_axiom = self._evolve_policy_from_feedback(query, response, reward, style_tag, user_comment)

        return {
            "success": True,
            "reward": reward,
            "learned_axiom": learned_axiom,
            "metrics": self.get_metrics()
        }

    def detect_conversational_critique(
        self,
        user_msg: str,
        last_query: Optional[str] = None,
        last_response: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyzes the user's ongoing chat message. If it contains praise, criticism,
        or a steering instruction regarding the assistant's previous answer, extracts
        the reward signal and updates the reinforcement style policy.
        """
        lower = user_msg.lower().strip()

        # Check positive cues
        is_positive = any(re.search(r'\b' + re.escape(cue) + r'\b', lower) for cue in POSITIVE_CUES)
        
        # Check critique/steering cues
        matched_critique = None
        for cue in CRITIQUE_CUES:
            if cue in lower:
                matched_critique = cue
                break

        if not is_positive and not matched_critique:
            return None

        # Determine reward and target preference
        reward = 1.0 if is_positive else -0.8
        style_tag = "conversational_praise" if is_positive else "conversational_critique"

        if matched_critique:
            if any(k in matched_critique for k in ["kısa", "uzun", "özet"]):
                style_tag = "conciseness"
            elif any(k in matched_critique for k in ["kod"]):
                style_tag = "code_preference"
            elif any(k in matched_critique for k in ["formül", "teori"]):
                style_tag = "math_theory_preference"
            elif any(k in matched_critique for k in ["mekatronik", "donanım"]):
                style_tag = "engineering_context"

        # Record this implicit preference
        res = self.record_feedback(
            query=last_query or "Previous Query",
            response=last_response or "",
            reward=reward,
            style_tag=style_tag,
            user_comment=user_msg,
            source="implicit_conversation"
        )
        return res

    def _evolve_policy_from_feedback(
        self,
        query: str,
        response: str,
        reward: float,
        style_tag: Optional[str],
        user_comment: Optional[str]
    ) -> Optional[str]:
        """
        Translates the user feedback into a concrete, non-redundant style axiom.
        Updates both Vault/00_Profile/Reinforcement_Policy.md and Style.md.
        """
        # Quick heuristic mappings for standard buttons to avoid latency
        axiom_map = {
            "concise": "Cevapları doğrudan konuya girerek, gereksiz giriş-çıkış nezaket cümleleri olmadan öz ve net tut.",
            "code_heavy": "Açıklamalarda her zaman çalışan PyTorch/Python kod blokları ve açık tensör boyutları (B, C, H, W) ver.",
            "math_depth": "Matematiksel formülasyonları, optimizasyon hedeflerini ve kaybın analitik türevlerini detaylandır.",
            "robotics": "Kavramları mekatronik, gömülü sistemler ve robotik kontrol (sensör füzyonu, motor kontrol, gerçek zamanlı kısıtlar) perspektifiyle bağdaştır.",
            "perfect": "Mevcut anlatım tonu, teknik derinliği ve kod/açıklama dengesi ideal; bu üslubu sürdür."
        }

        new_axiom = None
        if style_tag in axiom_map:
            new_axiom = axiom_map[style_tag]
        elif user_comment:
            # Generate axiom using LLM
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI Style Alignment Engine. The user provided feedback on an assistant's response.\n"
                        "Synthesize the user's preference into a SINGLE concise, imperative Turkish guideline (Axiom).\n"
                        "Example inputs: 'çok uzattın biraz daha özet geç' -> 'Cevapları gereksiz teorik detaylara boğmadan özet ve pratik tut.'\n"
                        "Example inputs: 'daha çok kod ve tensör boyutları görmek istiyorum' -> 'Teorik açıklamaları minimalde tutup zengin PyTorch kodları ve tensör boyutları ver.'\n"
                        "Output ONLY the single Turkish guideline sentence, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"Feedback/Critique: {user_comment}\nContext Tag: {style_tag or 'N/A'}"
                }
            ]
            try:
                candidate = "".join(list(llm.stream_generate(prompt, max_tokens=80, temperature=0.1))).strip()
                if candidate and len(candidate) > 10:
                    new_axiom = candidate.strip('"\'-* ')
            except Exception as e:
                print(f"[!] Error generating style axiom: {e}")

        if not new_axiom:
            return None

        # Add to policy file
        self.add_style_axiom(new_axiom, weight=reward)
        return new_axiom

    def add_style_axiom(self, axiom: str, weight: float = 1.0):
        """Adds or reinforces an axiom in Reinforcement_Policy.md and Style.md."""
        axioms = self.get_active_axioms()
        
        # Deduplicate or promote
        cleaned_axiom = axiom.strip().rstrip(".") + "."
        if cleaned_axiom not in axioms:
            axioms.append(cleaned_axiom)

        # Keep top 8 most refined axioms to prevent prompt bloat
        if len(axioms) > 8:
            axioms = axioms[-8:]

        # Update Vault/00_Profile/Reinforcement_Policy.md
        policy_frontmatter = {
            "type": "profile",
            "tags": ["profile", "reinforcement-learning", "style-policy"],
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        policy_body = f"""# 🎯 Dynamic Reinforcement Style Policy

Bu belge, Samet ile yapılan sohbetler ve verilen geri bildirimler (👍/👎, tarz yönlendirmeleri, örtük düzeltmeler) sonucunda pekiştirilmiş aktif model üslup kurallarını içerir. Her yanıtta bu kurallar öncelikli olarak uygulanır.

## 📌 Aktif Pekiştirilmiş Tarz Kuralları (Learned Axioms)

"""
        for a in axioms:
            policy_body += f"- {a}\n"

        policy_body += f"""
---
*Son Güncelleme:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
*Toplam Geri Bildirim Sinyali:* `{self.get_metrics().get('total_feedbacks', 0)}`
"""
        vault.write_note("00_Profile", "Reinforcement_Policy.md", policy_frontmatter, policy_body)

        # Update Style.md for backwards-compatibility with profile manager
        style_frontmatter = {
            "type": "profile",
            "tags": ["profile", "style", "rl-aligned"]
        }
        style_body = "# User Learning & Response Style (RL-Aligned)\n\n"
        for a in axioms:
            style_body += f"- {a}\n"
        
        vault.write_note("00_Profile", "Style.md", style_frontmatter, style_body)
        vault.git_commit(f"rl: reinforced style policy '{cleaned_axiom[:35]}'")

    def get_active_axioms(self) -> List[str]:
        """Reads active learned axioms from Reinforcement_Policy.md or Style.md."""
        target_file = POLICY_FILE if POLICY_FILE.exists() else STYLE_FILE
        if not target_file.exists():
            return [
                "Cevapları doğrudan konuya girerek, gereksiz giriş-çıkış nezaket cümleleri olmadan öz ve net tut.",
                "Açıklamalarda her zaman çalışan PyTorch/Python kod blokları ve açık tensör boyutları ver.",
                "Teknik terimleri (roll, pitch, yaw, attention, gradient) orijinal İngilizce halleriyle koru."
            ]

        try:
            content = target_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2]
            
            axioms = []
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    clean_item = line.lstrip("- ").strip()
                    if clean_item and not clean_item.startswith("*"):
                        axioms.append(clean_item)
            return axioms or [
                "Cevapları doğrudan konuya girerek, gereksiz nezaket cümleleri olmadan öz ve net tut.",
                "Teknik derinlik, net tensör boyutları ve çalışan kod blokları sun."
            ]
        except Exception:
            return ["Öz, net, teknik derinlikli ve kod ağırlıklı anlat."]

    def get_style_prompt_injection(self) -> str:
        """
        Formats active learned axioms into a direct, high-priority system prompt injection.
        """
        axioms = self.get_active_axioms()
        if not axioms:
            return ""

        lines = [
            "### 🎯 KULLANICIYA ÖZEL PEKİŞTİRİLMİŞ TARZ VE TERCİHLER (RL POLICY):",
            "Kullanıcı ile geçmiş etkileşimlerden ve geri bildirimlerden öğrenilmiş aşağıdaki tarz kurallarına KESİNLİKLE uy:"
        ]
        for a in axioms:
            lines.append(f"• {a}")
        lines.append("Bu kurallar senin yanıt biçimini, dil tonunu ve detay seviyeni doğrudan belirlemelidir.")
        return "\n".join(lines)


# Global singleton instance
rl_engine = RLFeedbackEngine()


if __name__ == "__main__":
    print("[*] Testing RLFeedbackEngine...")
    print("Initial Metrics:", rl_engine.get_metrics())
    print("Active Axioms:", rl_engine.get_active_axioms())
    
    # Test recording feedback
    res = rl_engine.record_feedback(
        query="Coordinate Attention nasıl çalışır?",
        response="Coordinate Attention 1D pooling kullanır...",
        reward=1.0,
        style_tag="code_heavy",
        user_comment="Harika, kodları çok iyi verdin aynen böyle devam et."
    )
    print("Recorded Feedback:", res)
    print("Injected Prompt:\n", rl_engine.get_style_prompt_injection())
