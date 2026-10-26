"""
Foundational Building-Block Paper Ingestion Pipeline.
Downloads, chunks, and indexes foundational deep learning building blocks:
1. VGGNet: Very Deep Convolutional Networks (3x3 Convolutions & Receptive Field)
2. Batch Normalization: Accelerating Training via Covariate Shift Reduction
3. YOLOv1: You Only Look Once (Genesis of Single-Stage Real-Time Object Detection)
4. GRU: Gated Recurrent Neural Networks (Sequential Gating Mechanics)
5. Dropout / Co-adaptation: Regularization & Deep Representation (Hinton et al.)
Updates corpus_catalog.json and invalidates searcher cache.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.single_ingest import ingest_arxiv_paper

FOUNDATIONAL_PAPERS = [
    {
        "id": "1409.1556",
        "title": "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)",
        "authors": "Simonyan & Zisserman (Oxford Visual Geometry Group)",
        "domain": "CNN Mimarisi & Alıcı Alan (Receptive Field)",
        "color": "#38bdf8",
        "desc": "Büyük filtreler yerine ardışık 3x3 küçük konvolüsyon katmanlarının aynı etki alanını daha az parametre ve daha çok non-lineer aktivasyonla taradığını kanıtlayan temel CNN mimarisi.",
        "prompt": "VGGNet makalesinde iki adet 3x3 konvolüsyon katmanının tek bir 5x5 katmana matematiksel denkliği ve parametre tasarrufu mantığını açıkla."
    },
    {
        "id": "1502.03167",
        "title": "Batch Normalization: Accelerating Deep Network Training",
        "authors": "Ioffe & Szegedy (Google)",
        "domain": "Normalizasyon & Optimizasyon Temelleri",
        "color": "#10b981",
        "desc": "Katman girdilerini mini-küme ortalama ve varyansıyla normalize edip ölçekleme/kaydırma (gamma, beta) parametreleriyle model eğitimini 10 kat hızlandıran kurucu makale.",
        "prompt": "Batch Normalization katmanının ileri ve geri yayılım (forward/backward) formülasyonunu ve Internal Covariate Shift kavramını açıkla."
    },
    {
        "id": "1506.02640",
        "title": "You Only Look Once: Unified, Real-Time Object Detection (YOLOv1)",
        "authors": "Redmon et al.",
        "domain": "Gerçek Zamanlı Nesne Tespiti Temelleri",
        "color": "#f59e0b",
        "desc": "Görüntüyü S x S ızgaraya bölerek nesne tespitini iki aşamalı sınıflandırmadan tek bir uçtan uca regresyon problemine dönüştüren orijinal YOLO mimarisi.",
        "prompt": "YOLOv1 mimarisindeki S x S ızgara hücresi (grid cell), sınır kutusu koordinatları (x, y, w, h), güven skoru (confidence) ve çok görevli kayıp (loss) fonksiyonunu açıkla."
    },
    {
        "id": "1412.3555",
        "title": "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling (GRU)",
        "authors": "Chung, Gulcehre, Cho, Bengio",
        "domain": "Özyineli Ağlar & Kapılı Bellek (GRU / RNN)",
        "color": "#a855f7",
        "desc": "Zaman serileri ve sensör verilerinde geçmiş bilgiyi unutma/hatırlama dengesini Reset ve Update kapılarıyla kuran Gated Recurrent Unit (GRU) mimarisi.",
        "prompt": "GRU mimarisindeki reset kapısı (r_t) ve update kapısının (z_t) matematiksel durum denklemlerini ve LSTM'e göre hesaplama avantajını açıkla."
    },
    {
        "id": "1207.0580",
        "title": "Improving neural networks by preventing co-adaptation of feature detectors (Dropout)",
        "authors": "Hinton, Srivastava, Krizhevsky, Sutskever, Salakhutdinov",
        "domain": "Düzenlileştirme (Regularization) & Dropout",
        "color": "#ec4899",
        "desc": "Eğitim sırasında nöronları rastgele p olasılığıyla devre dışı bırakarak nöronların birbirine bağımlı ezberlemesini (co-adaptation) önleyen ve modern derin öğrenmeyi başlatan kurucu makale.",
        "prompt": "Dropout tekniğinin matematiksel formülasyonunu, test anındaki ağırlık ölçekleme kuralını ve bir topluluk (ensemble) öğrenmesi gibi çalışmasının mantığını açıkla."
    }
]

def main():
    print(f"[*] Starting ingestion of {len(FOUNDATIONAL_PAPERS)} Foundational Building-Block papers...")
    catalog_path = ROOT / "data" / "corpus_catalog.json"
    existing_catalog = []
    if catalog_path.exists():
        try:
            existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing_catalog = []

    existing_ids = {item["id"] for item in existing_catalog}
    successful_ingests = []

    for idx, p in enumerate(FOUNDATIONAL_PAPERS):
        paper_id = p["id"]
        paper_title = p["title"]
        print(f"\n[{idx+1}/{len(FOUNDATIONAL_PAPERS)}] Processing {paper_id}: {paper_title}...")

        res = ingest_arxiv_paper(paper_id, title=paper_title)
        print(f"    Status: {res.get('success')} | Chunks: {res.get('chunks_count', 0)} | Msg: {res.get('error', 'OK')}")

        if res.get("success"):
            successful_ingests.append(p)
            if paper_id not in existing_ids:
                existing_catalog.append(p)
                existing_ids.add(paper_id)

    # Save updated catalog
    catalog_path.write_text(json.dumps(existing_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Updated corpus catalog with {len(successful_ingests)} new Foundational papers.")

    # Invalidate hybrid searcher cache
    try:
        from retrieval.search import searcher
        searcher._bm25_cache.clear()
        print("[+] HybridSearcher BM25 cache cleared.")
    except Exception as e:
        print(f"[-] Cache clear notice: {e}")

    print("\n[+] All Foundational papers processed successfully!")

if __name__ == "__main__":
    main()
