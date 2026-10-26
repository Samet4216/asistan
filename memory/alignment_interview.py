"""
Cognitive Alignment Interview Engine (Faz 10.0).
Interactive interview mode triggered by `/align` or "🧠 Zihniyetimi Öğret".
Interviews Samet about his engineering mindset, project motivations ("why"),
trade-off priorities (latency vs accuracy, hardware limits), and embeds these
mental models directly into the Reinforcement Learning policy and profile.
"""

import sys
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
from memory.user_profile import profile_manager
from memory.rl_feedback import rl_engine


class AlignmentInterviewEngine:
    """Orchestrates structured cognitive alignment interviews."""

    def __init__(self):
        vault.ensure_layout()

    def generate_next_question(
        self,
        answered_questions: List[Dict[str, str]],
        current_topic: Optional[str] = None
    ) -> str:
        """
        Generates a fresh, topic-specific interview question based on the user's active topic,
        prior dialogue turns, and engineering goals. Never repeats generic boilerplate templates.
        """
        about_me = profile_manager.get_about_me()
        axioms = rl_engine.get_active_axioms()
        
        history_summary = ""
        if answered_questions:
            history_summary = "Önceki Sorulan Sorular ve Samet'in Yanıtları:\n"
            for q_data in answered_questions:
                history_summary += f"- Soru: {q_data.get('question')}\n  Samet'in Yanıtı: {q_data.get('answer')}\n"

        topic_instruction = ""
        if current_topic:
            topic_instruction = (
                f"\n[AKTİF ODAK KONUSU]: '{current_topic}'\n"
                f"Sorunu KESİNLİKLE bu konu ('{current_topic}') etrafında kurgula! "
                f"Kullanıcının bu konudaki tasarım tercihlerini, neden bu yöntemi seçtiğini, "
                f"hangi teknik takasları (trade-off) göze aldığını sor. Sakın alakasız konulara sapma!"
            )
        else:
            topic_instruction = (
                "\nKullanıcının şu anki genel araştırma yönelimine, donanım optimizasyonu (RTX 4050 6GB) "
                "veya mühendislik yaklaşımına odaklanan özgün ve taze bir soru sor."
            )

        prompt = [
            {
                "role": "system",
                "content": (
                    "Sen Samet'in yapay zeka laboratuvarındaki en yakın kıdemli mühendislik partnerisin.\n"
                    "Görevin Samet ile 'Bilişsel Zihniyet Hizalama Röportajı' (Cognitive Alignment Interview) yapmak.\n"
                    "Amacın Samet'in mühendislik kararlarını nasıl aldığını, bu teknolojileri NE AMAÇLA ve NEDEN tercih ettiğini "
                    "anlamak.\n"
                    f"{topic_instruction}\n\n"
                    f"KULLANICI PROFİLİ:\n{about_me}\n\n"
                    f"MEVCUT BİLİNEN TARZ KURALLARI:\n" + "\n".join([f"- {a}" for a in axioms]) + "\n\n"
                    f"{history_summary}\n"
                    "KURALLAR:\n"
                    "- Kalıpsal, ezber veya tekrarlayan şablon cümleler KULLANMA.\n"
                    "- Sabit olarak her defasında aynı 'termal kamera İHA' sorusunu TEKRARLAMA.\n"
                    "- Doğrudan konunun kalbine inen, Samet'in karar felsefesini sorgulayan TEK BİR net Türkçe soru sor.\n"
                    "- Sadece soruyu üret, giriş nezaket cümleleri kurma."
                )
            },
            {
                "role": "user",
                "content": f"Aktif konudaki zihniyet ve tercih sorusunu sor. Konu: {current_topic or 'Genel Mühendislik'}"
            }
        ]

        try:
            q = "".join(list(llm.stream_generate(prompt, max_tokens=150, temperature=0.7))).strip()
            q_clean = q.strip('"\'-')
            if len(q_clean) > 20:
                return q_clean
        except Exception:
            pass

        # Dynamic fallback based on topic keywords
        if current_topic and any(k in current_topic.lower() for k in ["ppo", "dpo", "rl", "reinforcement", "ödül"]):
            return f"PPO ve DPO optimizasyonunda senin için belirleyici mühendislik kararı nedir: Bir ödül modeli eğitmenin getirdiği karmaşıklıktan kaçınmak mı, yoksa çevrimiçi simülasyonda dinamik politika keşfi mi?"
        elif current_topic and any(k in current_topic.lower() for k in ["attention", "transformer", "flash"]):
            return f"Dikkat mekanizmalarında senin için asıl donanım kısıtı nedir: Bellek bant genişliği (HBM IO) mi yoksa RTX 4050'nin 6GB VRAM sınırı mı? Hangisini önceliklendiriyorsun?"
        else:
            return f"Bu projede/mimaride senin için asıl vazgeçilmez başarı kriteri nedir: Donanım kısıtlarında minimum gecikme ve hafiflik mi, yoksa ne pahasına olursa olsun maksimum doğruluk mu?"

    def process_answer(
        self,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Analyzes Samet's answer to extract his underlying decision heuristic / project purpose,
        reinforces it into RL Policy and Profile, and returns a synthesis confirmation.
        """
        extraction_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a Cognitive Mindset Extractor. Analyze the user's interview answer.\n"
                    "Extract:\n"
                    "1. 'heuristic': A single, imperative Turkish rule expressing HOW Samet thinks/decides "
                    "(e.g., 'Uygulamada 30+ FPS gerçek zamanlı hız, %1-2 mAP doğruluk artışından her zaman daha önceliklidir.')\n"
                    "2. 'project_why': A 1-sentence explanation of WHY and FOR WHAT PURPOSE he is doing this project.\n"
                    "3. 'confirmation': A warm, concise 1-2 sentence response validating this mindset and confirming it is permanently memorized.\n"
                    "Output ONLY valid JSON in this exact structure:\n"
                    "{\n"
                    '  "heuristic": "...",\n'
                    '  "project_why": "...",\n'
                    '  "confirmation": "..."\n'
                    "}"
                )
            },
            {
                "role": "user",
                "content": f"Interview Question: {question}\nSamet's Answer: {answer}"
            }
        ]

        try:
            raw = "".join(list(llm.stream_generate(extraction_prompt, max_tokens=250, temperature=0.2))).strip()
            clean_json = re.sub(r'```json\s*', '', raw)
            clean_json = re.sub(r'```\s*', '', clean_json).strip()
            data = json.loads(clean_json)
        except Exception:
            data = {
                "heuristic": f"Samet'in Karar İlkesi: {answer[:80]}",
                "project_why": answer[:100],
                "confirmation": "Düşünce tarzını ve bu kararın arkasındaki temel mühendislik amacını zihnime kaydettim."
            }

        heuristic = data.get("heuristic", "").strip()
        project_why = data.get("project_why", "").strip()
        confirmation = data.get("confirmation", "").strip()

        # 1. Reinforce as active RL style axiom
        if heuristic:
            rl_engine.add_style_axiom(f"Zihniyet İlkesi: {heuristic}", weight=1.0)
            rl_engine.record_feedback(
                query=question,
                response=answer,
                reward=1.0,
                style_tag="mindset_alignment",
                user_comment=heuristic,
                source="alignment_interview"
            )

        # 2. Append project why to About_Me if meaningful
        if project_why and len(project_why) > 15:
            profile_manager.append_about_me(f"Proje Amacı & Zihniyeti: {project_why}")

        return {
            "heuristic": heuristic,
            "project_why": project_why,
            "confirmation": confirmation
        }


# Singleton instance
alignment_interviewer = AlignmentInterviewEngine()


if __name__ == "__main__":
    print("[*] Testing AlignmentInterviewEngine...")
    q1 = alignment_interviewer.generate_next_question([])
    print("Sample Question 1:", q1)
    
    sample_ans = "Benim için İHA'da 30 FPS üzerinde gerçek zamanlı çalışması şart, çünkü hızlı manevrada hedefi kaçırmamamız lazım. %2 daha yüksek doğruluk için 15 FPS'e düşen ağır modelleri kesinlikle istemiyorum."
    res = alignment_interviewer.process_answer(q1, sample_ans)
    print("Extracted Mindset:", res)
