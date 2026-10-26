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
        Performs genuine, multi-dimensional dynamic evaluation of the model's answer:
        1. Relevance & Direct Fulfillment (0-35 points) - Penalizes evasion/dodging
        2. Information Density & Substance (0-25 points) - Penalizes fluff & hallucinations
        3. Domain / Mode Excellence (0-25 points) - Checks LaTeX, Citations, or Conversational warmth
        4. Coherence & Language Quality (0-15 points) - Checks endings, grammar, and clean Turkish
        Total Score: 0 to 100 points, translated to Reward: -1.0 to +1.0.
        """
        if not response or len(response.strip()) < 5:
            return {
                "score": 0.0,
                "reward": -1.0,
                "dimension_scores": {"relevance": 0, "substance": 0, "domain": 0, "coherence": 0},
                "critique": "Yanıt boş veya eksik üretildi.",
                "self_evolved_axiom": None,
                "timestamp": datetime.now().isoformat()
            }

        axioms = active_axioms or rl_engine.get_active_axioms()
        resp_lower = response.lower().strip()
        q_lower = query.lower().strip()
        critique_notes = []
        new_axiom = None

        # -------------------------------------------------------------
        # DIMENSION 1: Soru Uyumu & Doğrudan Yanıtlama (0 - 35 Puan)
        # -------------------------------------------------------------
        relevance_score = 30.0

        # Kaçamak / Geçiştirme tespiti (Evasion Detection):
        # Soruyu doğrudan yanıtlamak yerine karşı soru sorup topu taca atan kalıplar
        evasion_patterns = [
            "belirtir misiniz", "belirtmeniz", "spesifik olarak", "hangi yıllar",
            "hangi yıl veya durum", "daha fazla bilgi verin", "netleştirirseniz",
            "yeterli bilgiye sahip olabilmek için", "detaylı bilgiye sahip olmak için",
            "nasıl bir bilgi istediğinizi", "yardımcı olacak şekilde bilgi"
        ]
        has_evasion = any(ep in resp_lower for ep in evasion_patterns)
        
        # Soru doğrudan bir bilgi talebi içeriyor mu?
        is_direct_demand = any(w in q_lower for w in [
            "yaz", "ver", "nedir", "kimdir", "kadro", "kodro", "liste", "nasıl",
            "açıkla", "kaç", "değeri", "nerede", "tarihi"
        ])

        if has_evasion and is_direct_demand:
            # Model somut bilgi vermek yerine karşı soru sorup kaçmış!
            relevance_score -= 20.0
            critique_notes.append("⚠️ Doğrudan yanıt vermek yerine karşı sorularla geçiştirildi.")
            new_axiom = "Kullanıcı somut bir bilgi istediğinde kaçamak cevaplar vermeden doğrudan yanıtı sun."
        else:
            # Soru terimlerinin yanıtta yer alma oranı (Selamlaşma hariç)
            is_greeting_q = any(g in q_lower for g in ["selam", "merhaba", "naber", "nasılsın", "günaydın", "iyi akşamlar"])
            if not is_greeting_q:
                q_tokens = [w for w in re.findall(r'\b\w{3,}\b', q_lower) if w not in ("bir", "icin", "için", "olan", "nelerdir", "nedir", "yaz", "ver")]
                if q_tokens:
                    matched_tokens = sum(1 for tok in q_tokens if tok in resp_lower)
                    overlap_ratio = matched_tokens / len(q_tokens)
                    if overlap_ratio >= 0.5:
                        relevance_score += 5.0
                    elif overlap_ratio < 0.2:
                        relevance_score -= 10.0
                        critique_notes.append("⚠️ Sorudaki anahtar kavramlar yanıtta yeterince ele alınmadı.")

        relevance_score = max(5.0, min(35.0, relevance_score))

        # -------------------------------------------------------------
        # DIMENSION 2: Bilgi Yoğunluğu & Boş Dolgu / Halüsinasyon (0 - 25 Puan)
        # -------------------------------------------------------------
        substance_score = 20.0

        # Boş dolgu / yapay nezaket kalıpları
        fluff_patterns = [
            "tabii ki", "memnuniyetle", "öncelikle belirtmek gerekir ki",
            "büyük kulüplerin", "spikerleri tarafından açıklanmış", "genellikle uzun talimatta",
            "bu konuda yardımcı olmaktan", "böyle bir yapıdır"
        ]
        fluff_count = sum(1 for fp in fluff_patterns if fp in resp_lower)
        if fluff_count >= 2:
            substance_score -= 10.0
            critique_notes.append("Gereksiz dolgu cümleleri tespit edildi.")
        elif fluff_count == 1:
            substance_score -= 4.0

        # Halüsinasyon / uydurma söz öbeği kontrolü
        hallucination_indicators = ["spikerleri tarafından", "uzun talimatta ve tecrübeli", "onur olmuş", "non-profiş"]
        if any(hi in resp_lower for hi in hallucination_indicators):
            substance_score -= 8.0
            critique_notes.append("⚠️ Uydurma / bağlamsız söz öbeği (halüsinasyon) tespit edildi.")

        # Çok kısa veya yetersiz yanıt cezası
        words_count = len(response.split())
        if words_count < 12 and not any(g in q_lower for g in ["selam", "merhaba", "naber"]):
            substance_score -= 8.0
            critique_notes.append("Yanıt çok sığ ve yetersiz uzunlukta kaldı.")

        substance_score = max(5.0, min(25.0, substance_score))

        # -------------------------------------------------------------
        # DIMENSION 3: Mod Mükemmelliği (Akademik vs Sohbet) (0 - 25 Puan)
        # -------------------------------------------------------------
        domain_score = 18.0

        if mode == "research" or "araştırma" in mode.lower():
            # Araştırma Modu: Atıf, LaTeX formül ve tensör denetimi
            has_citations = bool(re.search(r'\[[SW]\d+\]', response))
            has_latex = "$" in response or "$$" in response
            has_tensors = bool(re.search(r'\[[BNC0-9\s,\-_HW]+\]', response))
            has_verbatim_quote = '"' in response

            academic_points = 0.0
            if has_citations:
                academic_points += 8.0
            else:
                critique_notes.append("Akademik kaynak atfı ([S1]..[S5]) kullanılmadı.")

            if has_latex:
                academic_points += 7.0
            if has_tensors:
                academic_points += 5.0
            if has_verbatim_quote:
                academic_points += 5.0

            domain_score = min(25.0, academic_points)
            if domain_score >= 20.0:
                critique_notes.append("✅ Kaynak atıfları, LaTeX ve mimari analiz eksiksiz.")
        else:
            # Sohbet Modu: Samimiyet, akıcılık, gereksiz formül kirliliği olmaması
            has_unsolicited_formulas = "$$" in response and not any(w in q_lower for w in ["formül", "hesap", "matematik", "loss"])
            has_warmth = any(w in resp_lower for w in ["nasılsın", "gün", "nasıl gidiyor", "harika", "dostum", "sevindim", "kolay gelsin", "keyifler"])
            has_robotic_intro = any(rb in resp_lower for rb in ["bir yapay zeka", "dil modeli olarak", "asistan olarak"])

            if has_unsolicited_formulas:
                domain_score -= 8.0
                critique_notes.append("Sohbette gereksiz matematiksel şablon kullanıldı.")
            if has_robotic_intro:
                domain_score -= 8.0
                critique_notes.append("Robotik asistan kalıpları kullanıldı.")
            if has_warmth:
                domain_score += 7.0
                critique_notes.append("✅ Samimi ve doğal sohbet üslubu korundu.")

        domain_score = max(5.0, min(25.0, domain_score))

        # -------------------------------------------------------------
        # DIMENSION 4: Dil Temizliği, Dilbilgisi & Bütünlük (0 - 15 Puan)
        # -------------------------------------------------------------
        coherence_score = 15.0

        # Çince veya yabancı karakter kontrolü
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', response))
        if has_chinese:
            coherence_score -= 12.0
            critique_notes.append("⚠️ Yabancı/Çince karakter sızıntısı tespit edildi.")

        # Cümlenin yarıda kesilmesi kontrolü
        ends_cleanly = response.strip().endswith((".", "!", "?", "```", "}", "]", '"', "'", "*"))
        if not ends_cleanly:
            coherence_score -= 8.0
            critique_notes.append("⚠️ Yanıt son cümlesi tamamlanmadan kesilmiş.")

        coherence_score = max(2.0, min(15.0, coherence_score))

        # -------------------------------------------------------------
        # TOPLAM PUAN VE ÖDÜL HESAPLAMA (Dinamik ve Gerçekçi)
        # -------------------------------------------------------------
        total_score = round(relevance_score + substance_score + domain_score + coherence_score, 1)
        total_score = max(10.0, min(99.0, total_score))
        
        # Ödül: 50 baz olmak üzere -1.0 ile +1.0 arasında dengeli
        normalized_reward = round((total_score - 50.0) / 50.0, 2)

        dimension_scores = {
            "relevance": round(relevance_score, 1),
            "substance": round(substance_score, 1),
            "domain": round(domain_score, 1),
            "coherence": round(coherence_score, 1)
        }

        if not critique_notes:
            if total_score >= 80:
                critique_notes.append("✅ Yanıt akıcı, doğrudan ve amaca uygun.")
            else:
                critique_notes.append("Yanıt kabul edilebilir düzeyde.")

        # Otonom kural evrimi
        if new_axiom and new_axiom not in axioms:
            rl_engine.add_style_axiom(new_axiom, weight=normalized_reward)
            self._increment_evolved_axioms()

        # Telemetriyi kaydet
        self._record_evaluation(total_score)

        return {
            "score": total_score,
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
