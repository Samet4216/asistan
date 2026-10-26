"""
Reinforcement Learning & Continuous Preference Alignment Engine (Faz 9.0).
Enables the assistant to learn the user's personal communication style, technical
depth preferences, and feedback signals (Explicit 👍/👎 and Implicit Conversational
Critiques).

Persists:
1. Dynamic Reinforcement Policy in Vault/00_Profile/Reinforcement_Policy.md & Style.md
2. Direct Preference Optimization (DPO) dataset in data/rl_preferences/preferences.jsonl
3. Gold Exemplars for Few-Shot In-Context Self-Training in data/rl_preferences/gold_exemplars.json
4. Metrics & Cumulative Rewards in data/rl_preferences/learning_metrics.json
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
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
GOLD_EXEMPLARS_FILE = RL_DIR / "gold_exemplars.json"

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
    "kod verme", "kod istemedim", "kodsuz anlat", "formülleri açıkla",
    "teoriyi geç", "pratik anlat", "mekatronik açısından", "donanım açısından",
    "öyle değil", "bunu sevmedim", "anlamadım", "daha basit", "daha sade",
    "yanlış bilgi", "hatalı", "uydurma", "böyle anlatma", "bir dahakine",
    "bundan sonra", "too long", "simpler", "no code", "don't give code"
]


class RLFeedbackEngine:
    """Continuous Reinforcement Learning, Style Adaptation & Self-Training Engine."""

    def __init__(self):
        vault.ensure_layout()
        self._ensure_metrics_file()
        self._ensure_gold_exemplars_file()

    def _ensure_metrics_file(self):
        if not METRICS_FILE.exists():
            default_metrics = {
                "total_feedbacks": 0,
                "positive_count": 0,
                "negative_count": 0,
                "cumulative_reward": 0.0,
                "learned_axioms_count": 0,
                "gold_exemplars_count": 0,
                "last_updated": datetime.now().isoformat()
            }
            METRICS_FILE.write_text(json.dumps(default_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    def _ensure_gold_exemplars_file(self):
        if not GOLD_EXEMPLARS_FILE.exists():
            GOLD_EXEMPLARS_FILE.write_text("[]", encoding="utf-8")

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
            "gold_exemplars_count": 0,
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
        metrics["gold_exemplars_count"] = len(self.get_gold_exemplars())
        try:
            METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def save_gold_exemplar(self, query: str, response: str, score: float = 1.0):
        """Saves a high-reward turn as a golden few-shot exemplar for in-context self-training."""
        q_clean = query.strip()
        r_clean = response.strip()
        if len(q_clean) < 5 or len(r_clean) < 30:
            return

        exemplars = self.get_gold_exemplars()
        # Avoid duplicate queries
        for ex in exemplars:
            if ex.get("query", "").strip().lower() == q_clean.lower():
                ex["response"] = r_clean
                ex["timestamp"] = datetime.now().isoformat()
                ex["score"] = score
                break
        else:
            exemplars.append({
                "query": q_clean,
                "response": r_clean,
                "timestamp": datetime.now().isoformat(),
                "score": score
            })

        # Keep top 15 gold exemplars to maintain quality
        exemplars = exemplars[-15:]
        try:
            GOLD_EXEMPLARS_FILE.write_text(json.dumps(exemplars, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"[!] Error saving gold exemplar: {e}")

    def get_gold_exemplars(self, query: Optional[str] = None, max_items: int = 2) -> List[Dict[str, Any]]:
        """Retrieves best matching gold exemplars for few-shot prompt injection."""
        try:
            if not GOLD_EXEMPLARS_FILE.exists():
                return []
            data = json.loads(GOLD_EXEMPLARS_FILE.read_text(encoding="utf-8"))
            if not query:
                return data

            # Simple token overlap retrieval for in-context injection
            q_tokens = set(re.findall(r'\b\w{3,}\b', query.lower()))
            scored = []
            for ex in data:
                ex_q = ex.get("query", "").lower()
                ex_tokens = set(re.findall(r'\b\w{3,}\b', ex_q))
                overlap = len(q_tokens.intersection(ex_tokens))
                scored.append((overlap, ex))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:max_items] if item[0] > 0] or data[-max_items:]
        except Exception:
            return []

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
        If positive, saves as gold exemplar for self-training.
        Triggers policy adaptation and fine-tuning synchronization.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "reward": float(reward),
            "style_tag": style_tag or ("positive" if reward > 0 else "negative"),
            "user_comment": user_comment or "",
            "query": query,
            "response": response[:1600]
        }

        # Append to JSONL
        try:
            with open(PREFERENCES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[!] Error writing to preferences.jsonl: {e}")

        # Update metrics
        self._update_metrics(reward)

        # Trigger Style Policy Evolution & Self-Training Sync
        learned_axiom = self._evolve_policy_from_feedback(query, response, reward, entry["style_tag"], user_comment)

        # Also notify conversation collector for local dataset building
        try:
            from memory.conversation_collector import collector
            collector.record_interaction(
                user_text=query,
                assistant_text=response,
                reward=reward,
                mode="feedback"
            )
        except Exception:
            pass

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
                style_tag = "no_code_rule"
            elif any(k in matched_critique for k in ["formül", "teori"]):
                style_tag = "math_theory_preference"
            elif any(k in matched_critique for k in ["mekatronik", "donanım"]):
                style_tag = "engineering_context"
            elif any(k in matched_critique for k in ["yanlış", "hatalı", "uydurma"]):
                style_tag = "fact_correction"

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
        Updates Vault/00_Profile/Reinforcement_Policy.md and gold exemplars.
        """
        new_axiom = None

        # Standard button heuristic mappings
        axiom_map = {
            "concise": "Cevapları doğrudan konuya girerek, gereksiz giriş-çıkış nezaket cümleleri olmadan öz ve net tut.",
            "math_depth": "Matematiksel formülasyonları, optimizasyon hedeflerini ve kaybın analitik türevlerini ($$...$$) detaylandır.",
            "robotics": "Kavramları mekatronik, gömülü sistemler ve robotik kontrol perspektifiyle bağdaştır.",
            "perfect": "Mevcut anlatım tonu, matematiksel derinliği ve doğrudan anlatım standardı ideal; bu üslubu sürdür.",
            "positive": "Açıklamaları pedagojik derinlikte, açık formüllerle ($$...$$) ve doğrudan konuya girerek sürdür.",
            "no_code_rule": "Kullanıcı açıkça istemediği sürece ASLA kod bloğu üretme; kavramları formüller ve mimari mantıkla anlat.",
            "fact_correction": "Doğrulanmamış veya spekülatif bilgiler üretme; kesinlik taşımayan durumlarda dürüstçe kaynak yetersizliğini belirt."
        }

        if style_tag in axiom_map:
            new_axiom = axiom_map[style_tag]

        # Check negative feedback heuristics
        elif reward < 0:
            resp_has_code = "```" in response
            user_asked_code = any(kw in query.lower() for kw in ["kod", "code", "python", "script", "yaz"])
            if resp_has_code and not user_asked_code:
                new_axiom = "Kullanıcı açıkça kod istemediği sürece ASLA kod bloğu üretme; sadece matematik ve mimariyle anlat."
            elif any(w in response.lower() for w in ["muhtemelen", "tahmince", "fatih terim", "spikerleri"]):
                new_axiom = "Doğrulanmamış spekülatif bilgi üretme; kesin kaynak yoksa doğrudan 'Yerel ve güncel kaynaklarda bulunamadı' de."
            else:
                new_axiom = "Yüzeysel veya varsayımsal açıklamalardan kaçın; doğrudan kanıtlanmış literatür, net tensörler ve formüllerle yanıtla."

        # Check positive feedback heuristics
        elif reward > 0:
            self.save_gold_exemplar(query, response, score=reward)
            new_axiom = "Mevcut pedagojik derinlik, açık matematik ($$...$$) ve doğrudan anlatım tarzı ideal; bu standardı koru."

        # If custom user comment provided, synthesize with LLM
        if user_comment and len(user_comment.strip()) > 3:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI Style Alignment Engine. The user provided critique/feedback on an AI response.\n"
                        "Synthesize the user's preference into a SINGLE concise, imperative Turkish guideline (Axiom).\n"
                        "STRICT RULE: The user strictly forbids unsolicited code blocks ('ben istemeden kod verme'). NEVER generate a rule asking for code unless explicitly instructed.\n"
                        "Output ONLY the single Turkish guideline sentence, nothing else."
                    )
                },
                {
                    "role": "user",
                    "content": f"Feedback/Critique: {user_comment}\nContext Query: {query[:200]}"
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
        
        cleaned_axiom = axiom.strip().rstrip(".") + "."
        if cleaned_axiom not in axioms:
            axioms.append(cleaned_axiom)

        # Keep top 7 most refined axioms to prevent prompt bloat
        if len(axioms) > 7:
            axioms = axioms[-7:]

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
*Altın Örnekler (Gold Exemplars):* `{len(self.get_gold_exemplars())}`
"""
        vault.write_note("00_Profile", "Reinforcement_Policy.md", policy_frontmatter, policy_body)

        # Update Style.md for backwards-compatibility
        style_frontmatter = {
            "type": "profile",
            "tags": ["profile", "style", "rl-aligned"]
        }
        style_body = "# User Learning & Response Style (RL-Aligned)\n\n"
        for a in axioms:
            style_body += f"- {a}\n"
        
        vault.write_note("00_Profile", "Style.md", style_frontmatter, style_body)

    def get_active_axioms(self) -> List[str]:
        """Reads active learned axioms from Reinforcement_Policy.md or Style.md."""
        target_file = POLICY_FILE if POLICY_FILE.exists() else STYLE_FILE
        if not target_file.exists():
            return [
                "Cevapları doğrudan konuya girerek, gereksiz nezaket cümleleri olmadan öz ve net tut.",
                "Kullanıcı açıkça istemediği sürece ASLA kod bloğu üretme; matematiksel formüller ($$...$$) ve mimari analizle açıkla.",
                "Teknik terimleri (roll, pitch, yaw, attention, gradient) orijinal teknik halleriyle koru."
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
                        if not any(bad in clean_item.lower() for bad in ["error", "httpconnection", "traceback"]):
                            axioms.append(clean_item)
            return axioms or [
                "Cevapları doğrudan konuya girerek, gereksiz nezaket cümleleri olmadan öz ve net tut.",
                "Kullanıcı açıkça kod istemediği sürece ASLA kod bloğu üretme; teorik ve matematiksel anlatım yap."
            ]
        except Exception:
            return ["Cevapları doğrudan konuya girerek öz, net ve pedagojik derinlikte sun."]

    def get_style_prompt_injection(self, query: str = "", mode: str = "general") -> str:
        """
        Formats active learned axioms and gold exemplars into a high-priority system prompt injection.
        Enforces continuous in-context self-training without latency.
        """
        axioms = self.get_active_axioms()
        lines = [
            "### 🎯 KULLANICIYA ÖZEL PEKİŞTİRİLMİŞ TARZ VE TERCİHLER (RL POLICY):",
            "Kullanıcı ile geçmiş etkileşimlerden ve geri bildirimlerden öğrenilmiş aşağıdaki tarz kurallarına KESİNLİKLE uy:"
        ]
        for a in axioms:
            lines.append(f"• {a}")
        
        lines.append("• KULLANICI KURALI: Kullanıcı açıkça istemediği sürece ASLA kod bloğu verme. Matematiksel formüller ($$...$$) ve kavramsal analizle açıkla.")

        # In-Context Few-Shot Self-Training: Inject relevant gold exemplars if available
        if query:
            matching_exemplars = self.get_gold_exemplars(query, max_items=1)
            if matching_exemplars:
                lines.append("\n### ⭐ GEÇMİŞTE PEKİŞTİRİLMİŞ ALTIN ÖRNEK (GOLD EXEMPLAR):")
                lines.append(f"Kullanıcı Sorusuna İdeal Yaklaşım Referansı:\nSoru: {matching_exemplars[0]['query']}\nİdeal Yanıt Özeti:\n{matching_exemplars[0]['response'][:400]}...")

        lines.append("\nBu kurallar senin yanıt biçimini, dil tonunu ve detay seviyeni doğrudan belirlemelidir.")
        return "\n".join(lines)


# Global singleton instance
rl_engine = RLFeedbackEngine()


if __name__ == "__main__":
    print("[*] Testing RLFeedbackEngine...")
    print("Initial Metrics:", rl_engine.get_metrics())
    print("Active Axioms:", rl_engine.get_active_axioms())
