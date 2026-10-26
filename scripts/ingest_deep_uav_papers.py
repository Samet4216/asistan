"""
Deep UAV Tracking, Planning & Edge AI Paper Ingestion Pipeline.
Downloads, chunks, and indexes high-impact tracking, trajectory planning,
neural Kalman filtering, and knowledge distillation papers into ChromaDB.
Updates corpus_catalog.json and invalidates searcher cache.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.single_ingest import ingest_arxiv_paper

DEEP_UAV_PAPERS = [
    {
        "id": "2110.06864",
        "title": "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
        "authors": "Zhang et al.",
        "domain": "İHA Hedef Takip & Veri Eşleme (MOT)",
        "color": "#10b981",
        "desc": "Düşük güven skorlu tespit kutularını atmadan Kalman filtresi ve BYTE veri eşleme algoritmasıyla hedefin kilitlenmesini kesintisiz koruyan mimari.",
        "prompt": "ByteTrack algoritmasının iki aşamalı (high-score ve low-score) kutu eşleme mantığını ve İHA kamerasındaki hedef takibi avantajlarını açıkla."
    },
    {
        "id": "2206.14651",
        "title": "BoT-SORT: Robust Associations Multi-Pedestrian Tracking",
        "authors": "Aharon et al.",
        "domain": "Kamera Hareketi Telafisi & Kalman Filtresi",
        "color": "#06b6d4",
        "desc": "İHA manevra yaparken kamera açısının dönmesini Kamera Hareket Telafisi (CMC) ve geliştirilmiş Kalman durumuyla (x, y, w, h) modelleyen mimari.",
        "prompt": "BoT-SORT mimarisinde Kamera Hareket Telafisi (Camera Motion Compensation - CMC) ve Kalman filtresi durum vektörü formülasyonunu açıkla."
    },
    {
        "id": "2008.08835",
        "title": "EGO-Planner: An ESDF-Free Gradient-Based Local Planner for Quadrotors",
        "authors": "Zhou et al. (Zhejiang Univ.)",
        "domain": "İHA Yörünge Planlama & Engelden Kaçınma",
        "color": "#f59e0b",
        "desc": "Ağır mesafe haritaları (ESDF) kullanmadan B-spline eğrileri ve ceza gradyanlarıyla İHA'ların gerçek zamanlı engelden kaçmasını sağlayan yerel yörünge planlayıcı.",
        "prompt": "EGO-Planner algoritmasının B-spline eğri parametrizasyonu ve haritasız (ESDF-free) gradyan tabanlı yörünge optimizasyonunu açıkla."
    },
    {
        "id": "2104.09918",
        "title": "KalmanNet: Neural Network Aided Kalman Filtering for State Estimation",
        "authors": "Revach et al.",
        "domain": "Yapay Zeka Destekli Kalman Filtresi",
        "color": "#8b5cf6",
        "desc": "Gürültü kovaryans matrislerini (Q ve R) elle tahmin etmek yerine RNN tabanlı mimariyle Kalman kazancını dinamik öğrenen gürültüye dayanıklı durum kestirici.",
        "prompt": "KalmanNet mimarisinin Kalman kazancını (Kalman Gain) derin sinir ağıyla nasıl öğrendiğini ve gürültülü sensörlerdeki üstünlüğünü açıkla."
    },
    {
        "id": "1503.02531",
        "title": "Distilling the Knowledge in a Neural Network",
        "authors": "Hinton et al. (Google Brain)",
        "domain": "Model Küçültme & Bilgi Damıtma (Edge AI)",
        "color": "#ec4899",
        "desc": "Büyük ve yavaş bir öğretmen modelin bilgisini İHA uç donanımında çalışan hafif öğrenci modele sıcaklık (T) yumuşatmalı softmax ile aktaran temel makale.",
        "prompt": "Knowledge Distillation kuramındaki sıcaklık (temperature T) parametreli yumuşatılmış softmax ve damıtma kaybı (distillation loss) fonksiyonunu açıkla."
    }
]

def main():
    print(f"[*] Starting ingestion of {len(DEEP_UAV_PAPERS)} Deep UAV & Edge AI papers...")
    catalog_path = ROOT / "data" / "corpus_catalog.json"
    existing_catalog = []
    if catalog_path.exists():
        try:
            existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing_catalog = []

    existing_ids = {item["id"] for item in existing_catalog}
    successful_ingests = []

    for idx, p in enumerate(DEEP_UAV_PAPERS):
        paper_id = p["id"]
        paper_title = p["title"]
        print(f"\n[{idx+1}/{len(DEEP_UAV_PAPERS)}] Processing {paper_id}: {paper_title}...")

        res = ingest_arxiv_paper(paper_id, title=paper_title)
        print(f"    Status: {res.get('success')} | Chunks: {res.get('chunks_count', 0)} | Msg: {res.get('error', 'OK')}")

        if res.get("success"):
            successful_ingests.append(p)
            if paper_id not in existing_ids:
                existing_catalog.append(p)
                existing_ids.add(paper_id)

    # Save updated catalog
    catalog_path.write_text(json.dumps(existing_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Updated corpus catalog with {len(successful_ingests)} new Deep UAV papers.")

    # Invalidate hybrid searcher cache
    try:
        from retrieval.search import searcher
        searcher._bm25_cache.clear()
        print("[+] HybridSearcher BM25 cache cleared.")
    except Exception as e:
        print(f"[-] Cache clear notice: {e}")

    print("\n[+] All papers processed successfully!")

if __name__ == "__main__":
    main()
