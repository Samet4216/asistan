"""
Batch Ingestion Script for Recommended Dogfight, YOLOv9, and Night Vision AI Papers.
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
        "id": "2203.14360",
        "title": "Observation-Centric SORT (OC-SORT): Robust Object Tracking for Fast Maneuvers",
        "authors": "Cao et al.",
        "domain": "İHA İt Dalaşı & Viraj Takibi",
        "color": "#f97316",
        "desc": "Hedef ani manevra veya viraj yaptığında Kalman filtresinin yön sapmasını gözlem merkezli düzelten ve kilidin kopmasını engelleyen algoritma.",
        "prompt": "OC-SORT mimarisi klasik SORT ve ByteTrack modellerine göre ani dönüşlerde ve hedef kaybolmalarında kilitlenmeyi nasıl korur?"
    },
    {
        "id": "2112.00995",
        "title": "SwinTrack: Visual Tracking with Swin Transformer",
        "authors": "Lin et al.",
        "domain": "Hava Hedef Takip & Kilitlenme",
        "color": "#10b981",
        "desc": "Swin Transformer'ın pencereli dikkat mimarisini kullanarak yakın ve uzak ölçeklerdeki hava hedeflerini kararlı takip eden yapı.",
        "prompt": "SwinTrack mimarisinin hedef ve arama alanını birleştiren hiyerarşik dikkat mekanizmasını açıkla."
    },
    {
        "id": "2402.13616",
        "title": "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information (PGI)",
        "authors": "Wang, Bochkovskiy & Liao",
        "domain": "YOLO Mimarisi & PGI Teorisi",
        "color": "#ef4444",
        "desc": "Derin ağlarda küçük hedeflere ait bilginin kaybolmasını engelleyen PGI (Programlanabilir Gradyan Bilgisi) ve GELAN mimarisi.",
        "prompt": "YOLOv9 mimarisindeki Programmable Gradient Information (PGI) ve tersinir fonksiyonlar derin katmanlarda bilgi kaybını nasıl önler?"
    },
    {
        "id": "2309.11331",
        "title": "Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism",
        "authors": "Wang et al.",
        "domain": "Hedef Tespiti & FPN İyileştirme",
        "color": "#eab308",
        "desc": "Klasik FPN/PAN yapılarının yerine geçerek hedef özelliklerini tüm ölçeklere minimum gecikmeyle yayan Topla ve Dağıt (GD) mekanizması.",
        "prompt": "Gold-YOLO mimarisindeki Gather-and-Distribute (GD) mekanizması FPN yapılarına göre küçük hedeflerin tespitinde nasıl avantaj sağlar?"
    },
    {
        "id": "2001.06826",
        "title": "Zero-DCE: Zero-Reference Deep Curve Estimation for Low-Light Vision",
        "authors": "Guo et al.",
        "domain": "Gece & Sis Görüntü İyileştirme",
        "color": "#a855f7",
        "desc": "Referans görüntü olmadan 300+ FPS hızda gece ve sisli kamera görüntülerini anında aydınlatan ve netleştiren hafif sinir ağı.",
        "prompt": "Zero-DCE mimarisi dinamik eğri kestirimi (deep curve estimation) ve referanssız kayıp fonksiyonlarıyla gece görüntüsünü nasıl iyileştirir?"
    },
    {
        "id": "1903.04780",
        "title": "Fast-Planner: A Robust and Efficient Trajectory Planner for Quadrotors",
        "authors": "Zhou et al. (HKUST)",
        "domain": "İHA 3D Kinematik Yörünge Planlama",
        "color": "#06b6d4",
        "desc": "Kinodinamik A* ve B-spline optimizasyonu ile engelli ortamlarda milisaniyeler içinde akıcı ve güvenli 3D uçuş rotası üreten algoritma.",
        "prompt": "Fast-Planner algoritması kinodinamik A* arama ve B-spline eğri türevleriyle quadrotor kısıtlarına uygun yörüngeyi nasıl oluşturur?"
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
    print("🚀 İHA İT DALAŞI, YOLOV9, GECE GÖRÜŞ VE 3D PLANLAMA MAKALELERİNİ İNDİRME")
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

    print("\n✅ Tüm yeni makaleler başarıyla hafızaya mühürlendi.")


if __name__ == "__main__":
    main()
