"""
Conversation Collector & Fine-Tuning Dataset Generator (Faz 12.0).
Captures real user-assistant interactions, filters high-value dialogs,
and formats them for both:
1. Few-shot In-Context Ollama Modelfile Persona (Instant 0-VRAM model creation).
2. QLoRA Fine-Tuning JSONL (ShareGPT / Alpaca format for RTX 4050 local training).
3. Continuous Self-Training synchronization with DPO dataset and Modelfile.
"""

import sys
import io
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

CONV_DIR = PROJECT_ROOT / "data" / "my_conversations"
CONV_DIR.mkdir(parents=True, exist_ok=True)

RAW_CHAT_FILE = CONV_DIR / "all_conversations.jsonl"
TRAIN_SHAREGPT_FILE = CONV_DIR / "train_sharegpt.jsonl"
MODELFILE_PATH = PROJECT_ROOT / "Modelfile.samet"


class ConversationCollector:
    """Collects user conversations and transforms them into personal training data."""

    def __init__(self):
        self._ensure_files()

    def _ensure_files(self):
        if not RAW_CHAT_FILE.exists():
            RAW_CHAT_FILE.touch()

    def record_interaction(
        self,
        user_text: str = "",
        assistant_text: str = "",
        reward: float = 1.0,
        mode: str = "chat",
        user_message: str = "",
        assistant_response: str = ""
    ):
        """Saves a conversation turn to local dataset."""
        user_clean = (user_message or user_text or "").strip()
        asst_clean = (assistant_response or assistant_text or "").strip()
        
        if len(user_clean) < 2 or len(asst_clean) < 5:
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_clean,
            "assistant": asst_clean,
            "reward": float(reward),
            "mode": mode
        }

        try:
            with open(RAW_CHAT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[!] Error recording conversation: {e}")

    def get_all_interactions(self) -> List[Dict[str, Any]]:
        """Reads all recorded interactions."""
        interactions = []
        if RAW_CHAT_FILE.exists():
            try:
                with open(RAW_CHAT_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            interactions.append(json.loads(line))
            except Exception:
                pass
        return interactions

    def build_training_dataset(self) -> Dict[str, Any]:
        """
        Builds ShareGPT format dataset from positive and neutral interactions.
        """
        interactions = self.get_all_interactions()
        # Also include positive preferences from rl_preferences
        rl_file = PROJECT_ROOT / "data" / "rl_preferences" / "preferences.jsonl"
        if rl_file.exists():
            try:
                with open(rl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            d = json.loads(line)
                            if float(d.get("reward", 0.0)) >= 0:
                                q = d.get("query", "").strip()
                                r = d.get("response", "").strip()
                                if q and r and not any(i.get("user") == q for i in interactions):
                                    interactions.append({
                                        "timestamp": d.get("timestamp", datetime.now().isoformat()),
                                        "user": q,
                                        "assistant": r,
                                        "reward": d.get("reward", 1.0),
                                        "mode": "chat"
                                    })
            except Exception:
                pass

        sharegpt_data = []
        for item in interactions:
            u = item.get("user", "")
            a = item.get("assistant", "")
            # Filter out error traces or trivial inputs
            if "LLM STREAM ERROR" in u or "LLM STREAM ERROR" in a:
                continue
            if len(u) < 3 or len(a) < 10:
                continue

            sharegpt_data.append({
                "conversations": [
                    {"from": "human", "value": u},
                    {"from": "gpt", "value": a}
                ]
            })

        # Save to ShareGPT JSONL
        with open(TRAIN_SHAREGPT_FILE, "w", encoding="utf-8") as f:
            for item in sharegpt_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return {
            "total_dialogs": len(sharegpt_data),
            "file_path": str(TRAIN_SHAREGPT_FILE)
        }

    def generate_ollama_modelfile(self, base_model: str = "qwen2.5:7b-instruct-q4_K_M") -> str:
        """
        Creates a custom Ollama Modelfile with Samet's personal system prompt,
        learned active axioms, and real few-shot conversation examples.
        """
        from memory.user_profile import profile_manager
        from memory.rl_feedback import rl_engine
        about_me = profile_manager.get_about_me()
        axioms = rl_engine.get_active_axioms()
        interactions = self.get_all_interactions()

        # Pick top pristine interactions as few-shot examples (Prioritize verified Gold Exemplars)
        from memory.rl_feedback import rl_engine
        gold_list = rl_engine.get_gold_exemplars()
        clean_examples = []
        
        for g in reversed(gold_list):
            q = g.get("query", "").replace('"', '\\"').replace("\n", " ").strip()
            r = g.get("response", "").replace('"', '\\"').replace("\n", " ").strip()
            if q and r and len(q) < 160 and len(r) < 600 and "" not in r and "Özür dilerim" not in r:
                clean_examples.append((q, r))
                if len(clean_examples) >= 3:
                    break

        if len(clean_examples) < 3:
            for it in reversed(interactions):
                if float(it.get("reward", 0)) >= 0.8:
                    u = it.get("user", "").replace('"', '\\"').replace("\n", " ").strip()
                    a = it.get("assistant", "").replace('"', '\\"').replace("\n", " ").strip()
                    if u and a and "" not in a and "Özür dilerim" not in a and len(u) < 160 and len(a) < 600:
                        clean_examples.append((u, a))
                        if len(clean_examples) >= 3:
                            break

        few_shot_lines = []
        for u, a in clean_examples:
            few_shot_lines.append(f'MESSAGE user "{u}"')
            few_shot_lines.append(f'MESSAGE assistant "{a}"')

        few_shot_block = "\n".join(few_shot_lines)

        axiom_lines = "\n".join([f"- {ax}" for ax in axioms])

        modelfile_content = f"""FROM {base_model}

# Çıkarım Parametreleri (Dengeli, akıcı, deterministik)
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 8192

# Samet'e Özel Kişiselleştirilmiş Sistem İstemcisi (Pekiştirilmiş Tarz)
SYSTEM \"\"\"Sen Samet'in kişisel yapay zeka araştırma ortağı ve sinir ağları mühendisi asistanısın.
Samet'i çok iyi tanıyorsun:
{about_me}

PEKİŞTİRİLMİŞ TARZ VE ÖĞRENİLMİŞ KURALLAR:
{axiom_lines}
- KULLANICI KURALI: Samet açıkça kod istemediği sürece ASLA kod bloğu üretme. Teorik motivasyon, mimari analiz ve matematiksel formüller ($$...$$) üzerinden açıkla.
- Bilmediğin veya doğrulanmamış konularda spekülasyon yapma, dürüstçe kaynak yetersizliğini belirt.
- Türkçe'yi akıcı, teknik derinlikli ve doğrudan anlatımla konuş.
\"\"\"

{few_shot_block}
"""
        MODELFILE_PATH.write_text(modelfile_content, encoding="utf-8")
        return str(MODELFILE_PATH)

    def sync_self_training(self) -> Dict[str, Any]:
        """
        Consolidates reinforcement feedback into active self-training artifacts:
        1. Builds ShareGPT training dataset (train_sharegpt.jsonl).
        2. Exports DPO dataset (dpo_training_dataset.jsonl).
        3. Updates Modelfile.samet with latest axioms and gold exemplars.
        """
        sharegpt_res = self.build_training_dataset()
        from scripts.export_dpo_dataset import export_dpo_dataset
        dpo_path = export_dpo_dataset(min_feedback=1)
        modelfile_path = self.generate_ollama_modelfile(base_model=getattr(config, "MODEL", "qwen2.5:7b-instruct-q4_K_M"))
        return {
            "sharegpt_dialogs": sharegpt_res.get("total_dialogs", 0),
            "dpo_path": dpo_path,
            "modelfile_path": modelfile_path
        }


collector = ConversationCollector()


if __name__ == "__main__":
    print("[*] Testing ConversationCollector & Self-Training Sync...")
    res = collector.sync_self_training()
    print("Self-training synchronized:", res)
