"""
Teacher Correction & Ground-Truth Alignment Engine (Faz 14.0).
Allows the user to correct flawed or hallucinated model answers by providing
the true ground-truth answer.

Persists:
1. Direct Preference Optimization (DPO) pairs (chosen: user_truth, rejected: model_flaw)
2. Gold-Standard Notes in Vault/02_Theory_Notes/Teacher_{slug}.md
3. Upserts verified truth chunks into ChromaDB ('paper_chunks') with top priority
4. Reinforcement Policy Golden Axioms in Vault/00_Profile/Reinforcement_Policy.md
"""

import sys
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ingest.indexer import Indexer
from memory.rl_feedback import rl_engine

RL_DIR = PROJECT_ROOT / "data" / "rl_preferences"
PREFERENCES_FILE = RL_DIR / "preferences.jsonl"
DPO_TRAIN_FILE = RL_DIR / "dpo_training_dataset.jsonl"


class TeacherCorrectionEngine:
    """Handles teacher-driven ground-truth recording, DPO pair creation, and ChromaDB sealing."""

    def __init__(self):
        self.indexer = Indexer()

    def record_correction(
        self,
        query: str,
        wrong_response: str,
        ground_truth: str,
        critique: str = ""
    ) -> Dict[str, Any]:
        """
        Records the user's correct answer, updates DPO pairs, seals to Obsidian Vault,
        indexes into ChromaDB, and updates the RL policy.
        """
        clean_q = query.strip()
        clean_truth = ground_truth.strip()
        clean_wrong = wrong_response.strip()

        if not clean_truth:
            return {"success": False, "error": "Doğru cevap metni boş olamaz."}

        # 1. Append to Preferences JSONL with high reward (+2.0)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": "teacher_correction",
            "reward": 2.0,
            "style_tag": "ground_truth",
            "user_comment": critique or "Kullanıcı doğru cevabı öğretti.",
            "query": clean_q,
            "response": clean_wrong[:1000],
            "chosen_ground_truth": clean_truth
        }

        try:
            with open(PREFERENCES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[!] Error writing preferences: {e}")

        # 2. Append direct DPO training pair: {"prompt": q, "chosen": truth, "rejected": wrong}
        dpo_pair = {
            "prompt": clean_q,
            "chosen": clean_truth,
            "rejected": clean_wrong[:1200]
        }
        try:
            with open(DPO_TRAIN_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(dpo_pair, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[!] Error writing DPO dataset: {e}")

        # 3. Create Permanent Gold-Standard Obsidian Note
        slug_raw = re.sub(r'[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ\s_-]', '', clean_q).strip()
        slug = "_".join(slug_raw.split()[:5]).capitalize() or "Verified_Concept"
        filename = f"Teacher_{slug}.md"
        target_dir = config.VAULT_DIR / "02_Theory_Notes"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        counter = 1
        while target_path.exists():
            filename = f"Teacher_{slug}_{counter}.md"
            target_path = target_dir / filename
            counter += 1

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        note_content = f"""---
title: "{clean_q}"
type: teacher_ground_truth
date: {now_str}
status: verified_ground_truth
source: user_teacher
tags:
  - ground-truth
  - teacher-correction
  - verified-reference
---

# 🎓 Doğrulanmış Referans: {clean_q}

> [!IMPORTANT] 💡 Kullanıcı Tarafından Öğretilen Doğru Bilgi (Ground Truth)
> Bu konu hakkında yapay zekanın önceki cevabı yetersiz veya hatalı bulunmuş ve kullanıcı (öğretmen) tarafından doğrulanmış gerçek bilgi sisteme mühürlenmiştir. Gelecek tüm aramalarda bu referans en yüksek öncelikle kullanılır.

## 📌 Doğrulanmış Gerçek Cevap:
{clean_truth}

---
*Önceki Hatalı/Eksik Yanıt:*
> {clean_wrong[:300]}...
"""
        try:
            target_path.write_text(note_content, encoding="utf-8")
        except Exception as e:
            print(f"[!] Error saving teacher note: {e}")

        # 4. Upsert into ChromaDB 'paper_chunks' so it is immediately retrieved in future queries!
        chunk_id = f"teacher_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        truth_chunk = [{
            "id": chunk_id,
            "arxiv_id": "teacher_verified",
            "title": f"Doğrulanmış Bilgi: {clean_q}",
            "year": 2026,
            "section_name": "Kullanıcı Doğrulaması (Ground Truth)",
            "part_no": 1,
            "content": f"[DOĞRULANMIŞ REFERANS]: Soru: '{clean_q}'\nDoğru Açıklama: {clean_truth}"
        }]
        try:
            self.indexer.index_paper_chunks(truth_chunk)
            # Clear BM25 cache in searcher
            from retrieval.search import searcher
            searcher._bm25_cache.clear()
        except Exception as e:
            print(f"[!] Error indexing teacher chunk in ChromaDB: {e}")

        # 5. Persist Golden Axiom in Reinforcement Policy
        axiom = f"'{clean_q[:60]}' sorulduğunda kullanıcının öğrettiği doğrulanmış yanıtı ('{clean_truth[:80]}...') esas al; kanıt olmayan konularda asla uydurma yapma, dürüstçe 'Bilmiyorum' de."
        rl_engine.add_style_axiom(axiom)

        return {
            "success": True,
            "filename": filename,
            "chunk_id": chunk_id,
            "learned_axiom": axiom
        }


# Global instance
teacher_engine = TeacherCorrectionEngine()
