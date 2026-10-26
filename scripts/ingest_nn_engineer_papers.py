"""
Neural Network Engineer Specialist Paper Ingestion Pipeline.
Downloads, chunks, and indexes core deep learning engineering papers:
1. LoRA: Parameter-Efficient Fine-Tuning (PEFT & Low-Rank Adaptation)
2. Grad-CAM: Debugging & Visual Explainability (Heatmaps & Gradient Localization)
3. Kaiming Initialization: Gradient Flow & Weight Stability (He Init / PReLU)
4. CutMix: Advanced Data Augmentation & Regularization
5. Masked Autoencoders (MAE): Self-Supervised Vision Pre-training
6. SGDR: Cosine Annealing with Warm Restarts (Learning Rate Scheduling)
Updates corpus_catalog.json and invalidates searcher cache.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.single_ingest import ingest_arxiv_paper

NN_ENGINEER_PAPERS = [
    {
        "id": "2106.09685",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al. (Microsoft)",
        "domain": "Hafif Model Eğitimi & PEFT (LoRA)",
        "color": "#a855f7",
        "desc": "Büyük modellerin orijinal ağırlıklarını dondurup düşük rank matrisleri (A ve B) eğiterek 6GB VRAM donanımlarda bellek taşması (OOM) yaşamadan ince ayar (fine-tuning) yapma matematiği.",
        "prompt": "LoRA mimarisinde W = W_0 + (alpha/r) * B * A düşük rank matris ayrıştırmasının matematiksel mantığını ve rank (r), alfa katsayısı ayarlarını açıkla."
    },
    {
        "id": "1610.02391",
        "title": "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
        "authors": "Selvaraju et al.",
        "domain": "Model Hata Ayıklama & Açıklanabilirlik (XAI)",
        "color": "#06b6d4",
        "desc": "Son evrişim katmanının gradyanlarını kullanarak modelin hangi piksellere odaklandığını gösteren ısı haritaları (heatmaps) üreten hata ayıklama ve model doğrulama aracı.",
        "prompt": "Grad-CAM algoritmasında sınıf skoru gradyanlarının özellik haritası üzerinde küresel ortalama havuzlama (GAP) ile ağırlıklandırılması ve ReLU süzgeci formülünü açıkla."
    },
    {
        "id": "1502.01852",
        "title": "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet (Kaiming / He Init)",
        "authors": "He, Zhang, Ren, Sun (Microsoft Research)",
        "domain": "Ağırlık Başlatma & Gradyan Kararlılığı",
        "color": "#10b981",
        "desc": "ReLU ve PReLU katmanları için ağırlık varyansını sqrt(2/n_in) olarak başlatıp çok derin ağlarda loss patlamasını (NaN) ve gradyan sönmesini yok eden kurucu makale.",
        "prompt": "Kaiming / He Initialization formülasyonunun türetimini, Xavier/Glorot başlatmasından farkını ve derin ağlarda varyans korunumu ispatını açıkla."
    },
    {
        "id": "1905.04899",
        "title": "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features",
        "authors": "Yun et al. (Naver / Clova AI)",
        "domain": "İleri Seviye Veri Artırma & Regülerizasyon",
        "color": "#f59e0b",
        "desc": "Görüntülerin rastgele yama (patch) kesitlerini birbirinin üzerine yapıştırıp etiketleri alan oranına (lambda) göre ağırlıklandırarak aşırı ezberlemeyi (overfitting) kıran veri artırma tekniği.",
        "prompt": "CutMix tekniğinde yama koordinatlarının (r_x, r_y, r_w, r_h) Beta dağılımıyla belirlenmesi ve kayıp fonksiyonunun (loss) alan ağırlıklı karışım formülasyonunu açıkla."
    },
    {
        "id": "2111.06377",
        "title": "Masked Autoencoders Are Scalable Vision Learners (MAE)",
        "authors": "He et al. (FAIR / Meta)",
        "domain": "Kendi Kendine Denetimli Öğrenme (Self-Supervised / SSL)",
        "color": "#ec4899",
        "desc": "Görüntü yamalarının %75'ini gizleyip sadece görünür kısımlardan eksik pikselleri tamamlayarak etiketsiz devasa verilerden güçlü görsel temsil öğrenen ön-eğitim mimarisi.",
        "prompt": "Masked Autoencoder (MAE) mimarisinde asimetrik kodlayıcı-kod çözücü (encoder-decoder) tasarımını ve %75 yüksek maskeleme oranının neden optimal olduğunu açıkla."
    },
    {
        "id": "1608.03983",
        "title": "SGDR: Stochastic Gradient Descent with Warm Restarts (Cosine Annealing)",
        "authors": "Loshchilov & Hutter",
        "domain": "Öğrenme Oranı Zamanlayıcıları (LR Schedulers)",
        "color": "#3b82f6",
        "desc": "Öğrenme oranını kosinüs eğrisiyle periyodik olarak sıfırlayıp yeniden başlatarak modelin yerel minimum çukurlarından kurtulmasını ve genelleme performansını artıran optimizasyon rehberi.",
        "prompt": "SGDR Cosine Annealing algoritmasındaki periyot çarpanı (T_mult) ve minimum/maksimum öğrenme oranı denklemlerini açıkla."
    }
]

def main():
    print(f"[*] Starting ingestion of {len(NN_ENGINEER_PAPERS)} Neural Network Engineer specialist papers...")
    catalog_path = ROOT / "data" / "corpus_catalog.json"
    existing_catalog = []
    if catalog_path.exists():
        try:
            existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing_catalog = []

    existing_ids = {item["id"] for item in existing_catalog}
    successful_ingests = []

    for idx, p in enumerate(NN_ENGINEER_PAPERS):
        paper_id = p["id"]
        paper_title = p["title"]
        print(f"\n[{idx+1}/{len(NN_ENGINEER_PAPERS)}] Processing {paper_id}: {paper_title}...")

        res = ingest_arxiv_paper(paper_id, title=paper_title)
        print(f"    Status: {res.get('success')} | Chunks: {res.get('chunks_count', 0)} | Msg: {res.get('error', 'OK')}")

        if res.get("success"):
            successful_ingests.append(p)
            if paper_id not in existing_ids:
                existing_catalog.append(p)
                existing_ids.add(paper_id)

    # Save updated catalog
    catalog_path.write_text(json.dumps(existing_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Updated corpus catalog with {len(successful_ingests)} new NN Engineer papers.")

    # Invalidate hybrid searcher cache
    try:
        from retrieval.search import searcher
        searcher._bm25_cache.clear()
        print("[+] HybridSearcher BM25 cache cleared.")
    except Exception as e:
        print(f"[-] Cache clear notice: {e}")

    print("\n[+] All Neural Network Engineer papers processed successfully!")

if __name__ == "__main__":
    main()
