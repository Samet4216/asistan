"""
Batch Ingestion Script for Recommended Drone & Thermal AI Papers.
Downloads, chunks, and indexes high-priority papers into ChromaDB and updates corpus catalog.
"""

import sys
import json
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
from ingest.single_ingest import ingest_arxiv_paper

PAPERS_TO_ADD = [
    {
        "id": "2202.06456",
        "title": "SAHI: Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection",
        "authors": "Akyon et al.",
        "domain": "İHA Küçük Hedef Tespiti",
        "color": "#f97316",
        "desc": "Uzak ve birkaç piksellik küçük İHA hava hedeflerini dilimleme (slicing) ile kaçırmadan tespit eden açık kaynak standardı.",
        "prompt": "SAHI mimarisinin dilimleme tabanlı çıkarım (slicing aided hyper inference) mantığını ve küçük hedef tespitindeki üstünlüğünü açıkla."
    },
    {
        "id": "2203.11991",
        "title": "OSTrack: Joint Monocular 3D and 2D One-Stream Visual Tracking",
        "authors": "Ye et al.",
        "domain": "Hava Hedef Takip & Kilitlenme",
        "color": "#10b981",
        "desc": "Hedef şablonu ile arama bölgesini tek bir Transformer katmanında birleştiren yüksek FPS hedef takip mimarisi.",
        "prompt": "OSTrack mimarisinin tek akışlı (one-stream) dikkat mekanizmasıyla hedef takibini nasıl yaptığını ve Siamese modellere üstünlüğünü açıkla."
    },
    {
        "id": "2207.08051",
        "title": "CFT: Cross-Modality Feature Fusion Transformer for Multispectral Detection",
        "authors": "Qingyun et al.",
        "domain": "Termal Kamera & Sensör Füzyonu",
        "color": "#ef4444",
        "desc": "Termal kızılötesi ve RGB kamera kanallarını Transformer dikkat mekanizmasıyla kaynaştırarak gece-gündüz hedef tespitini maksimize eden mimari.",
        "prompt": "CFT mimarisinin termal ve görünür ışık özelliklerini çapraz dikkat (cross-attention) ile nasıl birleştirdiğini açıkla."
    },
    {
        "id": "2303.14189",
        "title": "FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization",
        "authors": "Vasu et al. (Apple)",
        "domain": "Gömülü Uç Yapay Zeka (Fast Edge AI)",
        "color": "#38bdf8",
        "desc": "Uç donanımlarda Vision Transformer doğruluğunu CNN hızında ve rekor gecikmeyle çalıştıran hibrit mimari.",
        "prompt": "FastViT mimarisindeki yapısal yeniden parametrelendirme (structural reparameterization) ve RepMixer bloğunun hız avantajını açıkla."
    },
    {
        "id": "2301.12900",
        "title": "DepGraph: Towards Any Structural Pruning",
        "authors": "Fang et al.",
        "domain": "Model Sıkıştırma & Budama (Pruning)",
        "color": "#a855f7",
        "desc": "YOLO ve konvolüsyonel modelleri doğruluğu bozmadan %40 küçülterek donanımda 30+ FPS'e ulaştıran yapısal budama kütüphanesi.",
        "prompt": "DepGraph algoritmasının katman bağımlılık grafiğini çıkararak yapısal budamayı (structural pruning) nasıl otomatikleştirdiğini açıkla."
    },
    {
        "id": "2006.05768",
        "title": "Deep Drone Acrobatics",
        "authors": "Kaufmann et al. (ETH Zurich / Scaramuzza)",
        "domain": "İHA Çevik Uçuş & Otonom Manevra",
        "color": "#ec4899",
        "desc": "Yüksek G kuvvetli hava muharebesi ve it dalaşında görsel sensörlerden doğrudan agresif uçuş komutları üreten derin öğrenme mimarisi.",
        "prompt": "Deep Drone Acrobatics makalesindeki sim-to-real transfer ve agresif yörünge izleme mantığını açıkla."
    }
]


def update_catalog():
    catalog_path = config.DATA_DIR / "corpus_catalog.json"
    existing = []
    if catalog_path.exists():
        try:
            existing = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing_ids = {p.get("id") for p in existing}
    added_count = 0

    for paper in PAPERS_TO_ADD:
        if paper["id"] not in existing_ids:
            existing.append(paper)
            added_count += 1

    catalog_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Corpus kataloğuna {added_count} yeni makale kartı eklendi.")


def main():
    print("=" * 65)
    print("🚀 İHA & TERMAL KAMERA & UÇ YAPAY ZEKA MAKALELERİNİ İNDİRME VE İNDEKSLENME")
    print("=" * 65)

    update_catalog()

    for idx, paper in enumerate(PAPERS_TO_ADD, 1):
        arxiv_id = paper["id"]
        title = paper["title"]
        print(f"\n[{idx}/{len(PAPERS_TO_ADD)}] İndiriliyor & İndeksleniyor: [{arxiv_id}] {title}")
        res = ingest_arxiv_paper(arxiv_id, title=title)
        if res.get("success"):
            print(f"  ✅ Başarılı! {res.get('chunks_count')} vektör parçası ChromaDB'ye kaydedildi.")
        else:
            print(f"  ⚠️ Hata: {res.get('error')}")

    print("\n✅ Tüm işlemler tamamlandı.")


if __name__ == "__main__":
    main()
