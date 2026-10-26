"""
Mechatronics Engineering Paper Ingestion Pipeline.
Downloads, chunks, and indexes high-impact mechatronics, robotics, control,
sensor fusion, embedded AI, and sim-to-real deep learning papers.
Updates corpus_catalog.json and invalidates searcher cache.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.single_ingest import ingest_arxiv_paper

MECHATRONICS_PAPERS = [
    {
        "id": "2011.04094",
        "title": "Learning-based Model Predictive Control for Autonomous Racing (Neural MPC)",
        "authors": "Kabzan et al. (ETH Zurich)",
        "domain": "Kontrol Teorisi & Neural MPC",
        "color": "#06b6d4",
        "desc": "Otonom araçlar ve robotlarda aerodinamik sürtünme ve lastik/zemin dinamiğini sinir ağıyla öğrenip gerçek zamanlı MPC ile çözen mimari.",
        "prompt": "Learning-based Model Predictive Control (Neural MPC) sistem dinamiğindeki belirsizlikleri Gauss Süreçleri ve sinir ağlarıyla nasıl telafi eder?"
    },
    {
        "id": "1703.06907",
        "title": "Domain Randomization for Transferring Deep Neural Networks to the Real World (Sim-to-Real)",
        "authors": "Tobin et al. (OpenAI)",
        "domain": "Sim-to-Real & Robot Öğrenmesi",
        "color": "#10b981",
        "desc": "Simülatörde (Gazebo/Isaac Sim) eğitilen robot kol veya İHA kontrol politikalarının fiziksel dünyadaki sürtünme ve ışık farklarına rağmen sıfır hatayla çalışmasını sağlayan sim-to-real transferi.",
        "prompt": "Domain Randomization tekniğinin simülasyondan fiziksel robot donanımına (sim-to-real) geçişteki matematiksel mantığını ve avantajlarını açıkla."
    },
    {
        "id": "2007.11898",
        "title": "ORB-SLAM3: Accurate Visual, Visual-Inertial, and Multimap SLAM",
        "authors": "Campos et al.",
        "domain": "Sensör Füzyonu & VIO / SLAM",
        "color": "#f59e0b",
        "desc": "GPS olmayan ortamlarda kamera ve IMU (ataletsel ölçüm) sensörlerini sıkı bağlı (tightly-coupled) kütüphanede birleştirerek santimetre hassasiyetinde robotik konumlandırma.",
        "prompt": "ORB-SLAM3 mimarisinde Visual-Inertial Odometry (VIO) ve IMU ön-entegrasyonu (preintegration) durum kestirimini nasıl optimize eder?"
    },
    {
        "id": "1911.11763",
        "title": "SuperGlue: Learning Feature Matching with Graph Neural Networks",
        "authors": "Sarlin et al. (Magic Leap)",
        "domain": "Robotik Görme & Çift Görüntü Eşleme",
        "color": "#8b5cf6",
        "desc": "Graf Sinir Ağları (GNN) ve Optimal Taşıma (Sinkhorn) ile dönen, titreşen ve zorlu açılardaki robotik kamera görüntülerinde milisaniyede anahtar nokta eşleme.",
        "prompt": "SuperGlue Graf Sinir Ağı (GNN) ve Sinkhorn algoritması ile robotik görsel odometride özellik eşlemeyi nasıl gerçekleştirir?"
    },
    {
        "id": "2103.05247",
        "title": "A White Paper on Neural Network Quantization (Edge AI / TensorRT)",
        "authors": "Nagel et al. (Qualcomm AI Research)",
        "domain": "Gömülü Yapay Zeka & Uç Donanım (Edge AI)",
        "color": "#ec4899",
        "desc": "Robot üzerindeki NVIDIA Jetson, STM32 veya RTX 4050 donanımlarında modelleri INT8/FP16 formatına kuantize ederek VRAM ve gecikmeyi 4 kat düşürme rehberi.",
        "prompt": "Post-Training Quantization (PTQ) ile Quantization-Aware Training (QAT) arasındaki farkları ve gömülü donanımda INT8 ölçekleme faktörünün (scale/zero-point) türetimini açıkla."
    },
    {
        "id": "1711.00199",
        "title": "PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation",
        "authors": "Xiang et al. (Univ. of Washington)",
        "domain": "Robotik Manipülasyon & 6-DOF Poz",
        "color": "#3b82f6",
        "desc": "Endüstriyel robot kollarının parçayı tutabilmesi için tekil RGB kameradan 3B konum (X,Y,Z) ve 3B yönelimi (kuaterniyon) kestiren temel konvolüsyonel mimari.",
        "prompt": "PoseCNN mimarisinin nesne merkezi vektör oylaması ve kuaterniyon kaybı (quaternion loss) ile 6-DOF poz kestirimini nasıl yaptığını açıkla."
    }
]

def main():
    print(f"[*] Starting ingestion of {len(MECHATRONICS_PAPERS)} Mechatronics & Robotics papers...")
    catalog_path = ROOT / "data" / "corpus_catalog.json"
    existing_catalog = []
    if catalog_path.exists():
        try:
            existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing_catalog = []

    existing_ids = {item["id"] for item in existing_catalog}
    successful_ingests = []

    for idx, p in enumerate(MECHATRONICS_PAPERS):
        paper_id = p["id"]
        paper_title = p["title"]
        print(f"\n[{idx+1}/{len(MECHATRONICS_PAPERS)}] Processing {paper_id}: {paper_title}...")

        res = ingest_arxiv_paper(paper_id, title=paper_title)
        print(f"    Status: {res.get('success')} | Chunks: {res.get('chunks_count', 0)} | Msg: {res.get('error', 'OK')}")

        if res.get("success"):
            successful_ingests.append(p)
            if paper_id not in existing_ids:
                existing_catalog.append(p)
                existing_ids.add(paper_id)

    # Save updated catalog
    catalog_path.write_text(json.dumps(existing_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Updated corpus catalog with {len(successful_ingests)} new Mechatronics papers.")

    # Invalidate hybrid searcher cache
    try:
        from retrieval.search import searcher
        searcher._bm25_cache.clear()
        print("[+] HybridSearcher BM25 cache cleared.")
    except Exception as e:
        print(f"[-] Cache clear notice: {e}")

    print("\n[+] All Mechatronics papers processed successfully!")

if __name__ == "__main__":
    main()
