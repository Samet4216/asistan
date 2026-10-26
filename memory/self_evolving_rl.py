"""
Real-Time Autonomous Self-Improving Reinforcement Learning Engine.
Evaluates every model generation on the fly across multiple objective dimensions,
computes reward signals, generates self-critiques, and autonomously refines
the Reinforcement Learning Policy in Obsidian (Reinforcement_Policy.md).
"""

import sys
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from memory.rl_feedback import rl_engine
from vault import vault

AUTONOMOUS_METRICS_FILE = PROJECT_ROOT / "data" / "rl_preferences" / "self_evolution_metrics.json"


class AutonomousSelfEvolvingRL:
    """Real-Time Self-Refining Reinforcement Learning Engine."""

    def __init__(self):
        self._ensure_metrics_file()

    def _ensure_metrics_file(self):
        if not AUTONOMOUS_METRICS_FILE.exists():
            default_metrics = {
                "total_self_evaluations": 0,
                "average_quality_score": 0.0,
                "total_self_axioms_evolved": 0,
                "history": [],
                "last_evolution": datetime.now().isoformat()
            }
            AUTONOMOUS_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            AUTONOMOUS_METRICS_FILE.write_text(
                json.dumps(default_metrics, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

    def get_evolution_metrics(self) -> Dict[str, Any]:
        """Loads autonomous evolution telemetry."""
        try:
            if AUTONOMOUS_METRICS_FILE.exists():
                return json.loads(AUTONOMOUS_METRICS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {
            "total_self_evaluations": 0,
            "average_quality_score": 0.0,
            "total_self_axioms_evolved": 0
        }

    def evaluate_and_evolve(
        self,
        query: str,
        response: str,
        retrieved_context: str = "",
        mode: str = "chat",
        active_axioms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Performs real-time multi-dimensional self-criticism of the model's answer.
        Returns:
            {
                "score": float, # 0.0 to 100.0
                "reward": float, # -1.0 to 1.0
                "dimension_scores": dict,
                "critique": str,
                "self_evolved_axiom": str or None,
                "timestamp": str
            }
        """
        if not response:
            return {"score": 0.0, "reward": -1.0, "critique": "Boş yanıt", "self_evolved_axiom": None}

        axioms = active_axioms or rl_engine.get_active_axioms()
        score = 80.0  # Base benchmark score
        dimension_scores = {
            "style_alignment": 85.0,
            "language_cleanliness": 100.0,
            "technical_depth": 80.0,
            "completeness": 90.0
        }
        critique_notes = []
        new_axiom = None

        resp_lower = response.lower()
        q_lower = query.lower()

        # 1. Language Cleanliness Check (Zero Chinese / Zero Hallucinated Non-words)
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', response))
        has_broken_words = any(bw in resp_lower for bw in ["non-profiş", "soyutları", "onur olmuş"])
        if has_chinese or has_broken_words:
            dimension_scores["language_cleanliness"] = 10.0
            critique_notes.append("⚠️ Yabancı/bozuk kelime veya Çince karakter sızıntısı tespit edildi.")
            score -= 40.0
            new_axiom = "Türkçe dilbilgisini kusursuz koru; araya asla yabancı veya uydurma sözcük karıştırma."

        # 2. Style Alignment Check (Against Active Axioms)
        wants_concise = any("öz ve net" in a.lower() or "kısa" in a.lower() for a in axioms)
        preamble_fluff = [
            "tabii ki", "memnuniyetle", "sevgili samet", "harika bir soru",
            "bu konuda sana yardımcı olmaktan mutluluk duyarım", "öncelikle belirtmek gerekir ki"
        ]
        has_preamble = any(resp_lower.startswith(pf) or f"\n{pf}" in resp_lower[:120] for pf in preamble_fluff)

        if wants_concise and has_preamble:
            dimension_scores["style_alignment"] -= 25.0
            critique_notes.append("Öz & net kuralına rağmen başlangıçta gereksiz dolgu/nezaket cümlesi kullanıldı.")
            score -= 15.0
            new_axiom = "Cevap başlangıcında 'Tabii ki', 'Memnuniyetle' gibi dolgu ifadeleri kullanmadan doğrudan konunun özüne gir."
        elif wants_concise and len(response.split()) <= 180 and not has_preamble:
            dimension_scores["style_alignment"] = 98.0
            critique_notes.append("✅ Öz ve net anlatım kuralına kusursuz uyuldu.")
            score += 10.0

        # 3. Technical Depth & Tensor Quality (if technical query)
        is_tech = any(k in q_lower for k in [
            "yolo", "attention", "transformer", "ppo", "dpo", "loss", "mimarisi",
            "pytorch", "tensör", "kod", "nasıl çalışır", "batch", "gradient"
        ])
        if is_tech:
            has_code = "```" in response
            has_tensor_shapes = bool(re.search(r'\[[BNC0-9\s,\-_HW]+\]', response))
            has_latex = "$" in response or "$$" in response

            if has_code and has_tensor_shapes:
                dimension_scores["technical_depth"] = 96.0
                critique_notes.append("✅ Çalışan kod blokları ve açık tensör boyutları başarıyla verildi.")
                score += 10.0
            elif has_code and not has_tensor_shapes:
                dimension_scores["technical_depth"] -= 15.0
                critique_notes.append("Kod bloğu var ancak katman giriş/çıkış tensör boyutları [B, C, H, W] eksik.")
                if not new_axiom:
                    new_axiom = "Teknik kod bloklarında tensör boyutlarını [B, C, H, W] mutlaka satır içi yorum olarak belirt."
            elif not has_code and any("kod" in a.lower() for a in axioms):
                dimension_scores["technical_depth"] -= 20.0
                critique_notes.append("Pekiştirilmiş kod tercihi olmasına rağmen yanıtta kod bloğu bulunamadı.")
                score -= 10.0

        # 4. Completeness & Coherence
        ends_abruptly = not response.strip().endswith((".", "!", "?", "```", "}", "]", '"', "'"))
        if ends_abruptly:
            dimension_scores["completeness"] = 50.0
            critique_notes.append("⚠️ Yanıt son cümlesi tamamlanmadan kesilmiş görünüyor.")
            score -= 20.0
            new_axiom = "Yanıtları daima eksiksiz ve mantıksal bir sonuç cümlesiyle tamamla; asla yarıda kesme."

        # Cap score between 10.0 and 100.0
        final_score = max(10.0, min(100.0, round(score, 1)))
        normalized_reward = round((final_score - 50.0) / 50.0, 2)  # -1.0 to 1.0

        if not critique_notes:
            critique_notes.append("✅ Yanıt biçimi, dil kalitesi ve kullanıcı hedefleriyle tam uyumlu.")

        # If a self-evolved axiom is found and valid, integrate into Reinforcement_Policy.md
        if new_axiom and new_axiom not in axioms:
            rl_engine.add_style_axiom(new_axiom, weight=normalized_reward)
            self._increment_evolved_axioms()

        # Update autonomous metrics
        self._record_evaluation(final_score)

        return {
            "score": final_score,
            "reward": normalized_reward,
            "dimension_scores": dimension_scores,
            "critique": " • ".join(critique_notes),
            "self_evolved_axiom": new_axiom,
            "timestamp": datetime.now().isoformat()
        }

    def _increment_evolved_axioms(self):
        try:
            metrics = self.get_evolution_metrics()
            metrics["total_self_axioms_evolved"] = metrics.get("total_self_axioms_evolved", 0) + 1
            metrics["last_evolution"] = datetime.now().isoformat()
            AUTONOMOUS_METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _record_evaluation(self, score: float):
        try:
            metrics = self.get_evolution_metrics()
            n = metrics.get("total_self_evaluations", 0)
            old_avg = metrics.get("average_quality_score", 80.0)
            new_n = n + 1
            new_avg = round(((old_avg * n) + score) / new_n, 1)

            metrics["total_self_evaluations"] = new_n
            metrics["average_quality_score"] = new_avg
            metrics["last_evaluation_score"] = score
            metrics["last_evaluation_time"] = datetime.now().isoformat()

            AUTONOMOUS_METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


# Global singleton instance
self_evolving_rl = AutonomousSelfEvolvingRL()
