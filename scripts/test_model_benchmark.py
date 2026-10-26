"""
Comprehensive End-to-End Autonomous Benchmark & Quality Audit Script.
Tests the model across Casual Chat, Modular Topic Decomposition, CoT Reasoning,
Multi-Turn Continuation, Academic Grounding, and Hardware Performance.
Outputs a detailed scored JSON and diagnostic markdown report.
"""

import sys
import time
import json
import re
from pathlib import Path

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
from utils import get_gpu_status, normalize_latex, contextualize_query
from retrieval.search import searcher
from answer.prompts import (
    build_thought_plan_prompt,
    build_answer_prompt,
    build_casual_response_prompt
)
from memory.self_evolving_rl import self_evolving_rl


def run_benchmark():
    print("=" * 75)
    print("🔬 LOCAL DEEP LEARNING LAB ASSISTANT — OTONOM KALİTE VE BAŞARIM DENETİMİ")
    print(f"🎯 Model: {config.MODEL} | Cihaz: RTX 4050 / CPU Hybrid")
    print("=" * 75)

    results = {}
    total_score = 0
    max_score = 0

    # -------------------------------------------------------------
    # 0. Donanım & VRAM Durum Kontrolü
    # -------------------------------------------------------------
    gpu_info = get_gpu_status()
    print("\n[0] DONANIM & BELLEK DENETİMİ:")
    print(f"  • GPU: {gpu_info.get('name', 'Bilinmiyor')} | VRAM: {gpu_info.get('vram_used_mb', 0)}MB / {gpu_info.get('vram_total_mb', 0)}MB | Sıcaklık: {gpu_info.get('temp', 'N/A')}°C")

    # -------------------------------------------------------------
    # TEST 1: Sohbet Modu & İnsansı Sıcaklık
    # -------------------------------------------------------------
    print("\n[1] TEST 1: Sohbet Modu & Empati / Doğallık Testi...")
    t1_q = "Selam dostum naber, bugün projede biraz yoruldum"
    t1_start = time.time()
    t1_msgs = build_casual_response_prompt(t1_q, history=[])
    t1_resp = llm.generate(t1_msgs, max_tokens=250, temperature=0.75, model=config.MODEL_CHAT)
    t1_time = round(time.time() - t1_start, 2)

    # Değerlendirme
    t1_lower = t1_resp.lower()
    has_warmth = any(w in t1_lower for w in ["nasılsın", "kolay gelsin", "dinlen", "geçmiş olsun", "yorulduysan", "dostum", "canını sıkma", "harika"])
    has_academic_leak = "$$" in t1_resp or any(al in t1_lower for al in ["tensör", "öğrenme oranı", "loss fonksiyonu", "backpropagation"])
    has_evasion = any(ev in t1_lower for ev in ["belirtir misiniz", "spesifik olarak", "hangi yıl"])

    t1_score = 100
    t1_issues = []
    if not has_warmth:
        t1_score -= 25
        t1_issues.append("Empati ve hal-hatır karşılığı zayıf kaldı.")
    if has_academic_leak:
        t1_score -= 30
        t1_issues.append("Sohbet modunda istenmeyen akademik terim/formül sızıntısı oldu.")
    if has_evasion:
        t1_score -= 20
        t1_issues.append("Selamlaşmaya kaçamak/gereksiz karşı soruyla karşılık verildi.")

    results["test_1_casual_chat"] = {
        "title": "Sohbet Modu & İnsansı Sıcaklık",
        "score": t1_score,
        "max": 100,
        "latency_sec": t1_time,
        "issues": t1_issues,
        "sample": t1_resp[:200] + "..."
    }
    print(f"  -> Puan: {t1_score}/100 | Süre: {t1_time}s")

    # -------------------------------------------------------------
    # TEST 2: Derin Makale Anlatımı & Modüler Parçalama (Stage 1 CoT + Stage 2 Answer)
    # -------------------------------------------------------------
    print("\n[2] TEST 2: Derin Makale Anlatımı & Modüler Parçalama Testi (YOLOv9 PGI)...")
    t2_q = "YOLOv9 makalesindeki Programlanabilir Gradyan Bilgisi (PGI) ve GELAN mimarisini anlat"
    t2_start = time.time()
    
    # RAG Retrieval
    search_res = searcher.search(t2_q, top_k=5)
    packed_context = search_res.get("packed_context", "")

    # Stage 1: CoT Reasoning Monologue
    cot_msgs = build_thought_plan_prompt(t2_q, packed_context)
    cot_plan = llm.generate(cot_msgs, max_tokens=750, temperature=0.1, model=config.MODEL_REASONING)

    # Stage 2: Detailed Explanation
    ans_msgs = build_answer_prompt(
        query=t2_q,
        packaged_context=packed_context,
        thought_plan=cot_plan,
        deep_dive=True,
        history=[]
    )
    t2_resp = llm.generate(ans_msgs, max_tokens=1500, temperature=0.2, model=config.MODEL_ACADEMIC)
    t2_time = round(time.time() - t2_start, 2)

    t2_lower = t2_resp.lower()
    has_modular_roadmap = any(rm in t2_lower for rm in ["parça", "bölüm", "modül", "1.", "adımda", "aşamada"])
    has_latex = "$" in t2_resp or "$$" in t2_resp
    has_citations = bool(re.search(r'\[S\d+\]', t2_resp))
    has_verbatim_quote = '"' in t2_resp
    has_checkpoint = any(cp in t2_lower for cp in ["devam edelim mi", "sıradaki", "geçelim mi", "ister misin", "hazırsan"])

    t2_score = 100
    t2_issues = []
    if not has_modular_roadmap:
        t2_score -= 20
        t2_issues.append("Konu modüler parçalara bölünmedi veya yol haritası verilmedi.")
    if not has_latex:
        t2_score -= 20
        t2_issues.append("Matematiksel KaTeX formülleri eksik.")
    if not has_citations:
        t2_score -= 15
        t2_issues.append("Kaynak atıfları ([S1]..[S5]) kullanılmadı.")
    if not has_checkpoint:
        t2_score -= 25
        t2_issues.append("Bölüm sonunda 'İstersen devam edelim mi?' etkileşimli kontrol noktası konulmadı.")

    results["test_2_modular_deep_dive"] = {
        "title": "Derin Makale Anlatımı & Modüler Parçalama",
        "score": t2_score,
        "max": 100,
        "latency_sec": t2_time,
        "issues": t2_issues,
        "has_latex": has_latex,
        "has_citations": has_citations,
        "has_checkpoint": has_checkpoint,
        "sample": t2_resp[:250] + "..."
    }
    print(f"  -> Puan: {t2_score}/100 | Süre: {t2_time}s | CoT & Checkpoint: {'VAR' if has_checkpoint else 'YOK'}")

    # -------------------------------------------------------------
    # TEST 3: Çok Turlu Devam & Bağlam Koruma ("devam et")
    # -------------------------------------------------------------
    print("\n[3] TEST 3: Çok Turlu Devam & Bağlam Koruma Testi ('devam edelim')...")
    t3_q = "evet devam edelim"
    t3_start = time.time()
    history = [
        {"role": "user", "content": t2_q},
        {"role": "assistant", "content": t2_resp}
    ]
    augmented_q = contextualize_query(t3_q, history)
    t3_msgs = build_answer_prompt(
        query=augmented_q,
        packaged_context=packed_context,
        thought_plan="User approved continuing to Part 2 (GELAN & Tensor Flow). Explain Part 2 with full formulas.",
        deep_dive=True,
        history=history
    )
    t3_resp = llm.generate(t3_msgs, max_tokens=1000, temperature=0.2, model=config.MODEL_ACADEMIC)
    t3_time = round(time.time() - t3_start, 2)

    t3_lower = t3_resp.lower()
    retained_context = any(rc in t3_lower for rc in ["yolo", "pgi", "gelan", "gradyan", "ikinci", "2. parça", "2. bölüm", "tensör"])

    t3_score = 100
    t3_issues = []
    if not retained_context:
        t3_score -= 40
        t3_issues.append("Kullanıcı 'devam edelim' dediğinde geçmiş YOLOv9 konusu unutuldu.")
    if len(t3_resp.split()) < 50:
        t3_score -= 20
        t3_issues.append("2. Parça anlatımı çok kısa kesildi.")

    results["test_3_multi_turn_continuation"] = {
        "title": "Çok Turlu Devam & Bağlam Koruma",
        "score": t3_score,
        "max": 100,
        "latency_sec": t3_time,
        "augmented_query": augmented_q,
        "issues": t3_issues,
        "sample": t3_resp[:200] + "..."
    }
    print(f"  -> Puan: {t3_score}/100 | Süre: {t3_time}s | Bağlam Korundu mu: {'EVET' if retained_context else 'HAYIR'}")

    # -------------------------------------------------------------
    # TEST 4: Dürüstlük & Olmayan Bilgide Halüsinasyon Önleme
    # -------------------------------------------------------------
    print("\n[4] TEST 4: Dürüstlük & Halüsinasyon Önleme Testi...")
    t4_q = "Coordinate Attention makalesinde ResNet-50 yerine Vision Transformer ile ablation deneyi yapıldı mı, sonuç tablosu nedir?"
    t4_start = time.time()
    t4_search = searcher.search(t4_q, top_k=3)
    t4_msgs = build_answer_prompt(
        query=t4_q,
        packaged_context=t4_search.get("packed_context", ""),
        thought_plan="",
        deep_dive=False,
        history=[]
    )
    t4_resp = llm.generate(t4_msgs, max_tokens=300, temperature=0.1, model=config.MODEL_ACADEMIC)
    t4_time = round(time.time() - t4_start, 2)

    t4_lower = t4_resp.lower()
    admits_absence = any(ab in t4_lower for ab in ["yer almıyor", "bulunmuyor", "yer almamaktadır", "bilmiyorum", "bahsedilmemiştir", "yapılmamıştır", "mevcut değil"])

    t4_score = 100
    t4_issues = []
    if not admits_absence:
        t4_score -= 45
        t4_issues.append("Makalede olmayan Transformer deneyi hakkında dürüstçe 'yer almıyor' demek yerine varsayımda bulundu.")

    results["test_4_honesty_grounding"] = {
        "title": "Dürüstlük & Halüsinasyon Kontrolü",
        "score": t4_score,
        "max": 100,
        "latency_sec": t4_time,
        "issues": t4_issues,
        "sample": t4_resp[:200] + "..."
    }
    print(f"  -> Puan: {t4_score}/100 | Süre: {t4_time}s | Dürüst İtiraf: {'EVET' if admits_absence else 'HAYIR'}")

    # -------------------------------------------------------------
    # TEST 5: Doğrudan Soru & Kaçamak Cevap Kontrolü (General Inquiry)
    # -------------------------------------------------------------
    print("\n[5] TEST 5: Doğrudan Yanıtlama & Anti-Evasion Testi...")
    t5_q = "Fenerbahçe spor kulübünün kuruluş yılı ve renkleri nedir?"
    t5_start = time.time()
    t5_msgs = build_casual_response_prompt(t5_q, history=[])
    t5_resp = llm.generate(t5_msgs, max_tokens=300, temperature=0.3, model=config.MODEL_CHAT)
    t5_time = round(time.time() - t5_start, 2)

    t5_lower = t5_resp.lower()
    has_year = "1907" in t5_resp
    has_colors = "sarı" in t5_lower and "lacivert" in t5_lower
    has_evasion = any(ev in t5_lower for ev in ["hangi yıllar", "spesifik olarak belirtin", "belirtir misiniz"])

    t5_score = 100
    t5_issues = []
    if not has_year or not has_colors:
        t5_score -= 30
        t5_issues.append("Temel olgusal bilgiler (1907, sarı-lacivert) eksik veya hatalı.")
    if has_evasion:
        t5_score -= 35
        t5_issues.append("Soruyu doğrudan yanıtlamak yerine karşı soru sorarak geçiştirdi.")

    results["test_5_direct_fulfillment"] = {
        "title": "Doğrudan Yanıtlama & Kaçamak Önleme",
        "score": t5_score,
        "max": 100,
        "latency_sec": t5_time,
        "issues": t5_issues,
        "sample": t5_resp[:200] + "..."
    }
    print(f"  -> Puan: {t5_score}/100 | Süre: {t5_time}s | Doğru Bilgi: {'EVET' if has_year and has_colors else 'HAYIR'}")

    # -------------------------------------------------------------
    # Toplam Puanlama ve Rapor
    # -------------------------------------------------------------
    scores = [v["score"] for v in results.values()]
    avg_score = round(sum(scores) / len(scores), 1)

    print("\n" + "=" * 75)
    print(f"📊 GENEL TEST SONUCU: %{avg_score} / 100")
    print("=" * 75)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.MODEL,
        "gpu": gpu_info,
        "overall_score": avg_score,
        "results": results
    }

    report_path = PROJECT_ROOT / "data" / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Rapor dosyası kaydedildi: {report_path}")

    return report


if __name__ == "__main__":
    run_benchmark()
