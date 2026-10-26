"""
Aviation & Combat Drone (Savaşan İHA) Paper Ingestion Pipeline.
Downloads, chunks, and indexes high-impact aerospace & UAV deep learning papers.
Updates corpus_catalog.json and invalidates searcher cache.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.single_ingest import ingest_arxiv_paper

AVIATION_PAPERS = [
    {
        "id": "2103.01955",
        "title": "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)",
        "authors": "Yu et al.",
        "domain": "Sürü İHA & İt Dalaşı (MARL)",
        "color": "#38bdf8",
        "desc": "Çoklu Savaşan İHA sürülerinde koordineli hava muharebesi, hava savunma delme ve 2v2/4v4 angajmanlar için çok ajanlı PPO mimarisi.",
        "prompt": "MAPPO mimarisinin çoklu İHA koordinasyonunda merkezi eğitim - dağıtık icra (CTDE) prensibini ve PPO'ya göre avantajlarını açıkla."
    },
    {
        "id": "2304.08069",
        "title": "RT-DETR: DETRs Beat YOLOs on Real-time Object Detection",
        "authors": "Lv et al. (Baidu)",
        "domain": "İHA Gerçek Zamanlı Görsel Tespit",
        "color": "#f59e0b",
        "desc": "Savaşan İHA görev bilgisayarlarında NMS gecikmesi olmadan gerçek zamanlı hava ve yer hedefi tespiti yapan uç transformatör mimarisi.",
        "prompt": "RT-DETR mimarisinin NMS gerektirmeyen tasarımı ve İHA üzerindeki gerçek zamanlı nesne tespiti hız avantajını açıkla."
    },
    {
        "id": "1812.11703",
        "title": "SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks",
        "authors": "Li et al.",
        "domain": "İHA Hedef Takip & Kilitlenme (VOT)",
        "color": "#10b981",
        "desc": "İHA kamerasından kilitlenilen hava araçlarının derin siyam ağları ve uzamsal farkındalık ile kare kare gerçek zamanlı takibi.",
        "prompt": "SiamRPN++ mimarisinin derin konvolüsyonel omurga (ResNet) ve derin çapraz korelasyon ile görsel hedef takibini nasıl yaptığını açıkla."
    },
    {
        "id": "1803.11485",
        "title": "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL",
        "authors": "Rashid et al. (Oxford)",
        "domain": "Sürü İHA Görev & Hedef Dağıtımı",
        "color": "#a855f7",
        "desc": "Sürü İHA'ların ortak hava görevinde monopolistik değer fonksiyonu ayrıştırması ile optimal hedef angajmanı paylaşımı.",
        "prompt": "QMIX algoritmasındaki monotonic değer çarpanlama (monotonicity constraint) şartını ve CTDE yapısını açıkla."
    },
    {
        "id": "1711.10561",
        "title": "Physics-Informed Neural Networks (PINNs): Nonlinear PDEs",
        "authors": "Raissi et al.",
        "domain": "Uçuş Dinamiği & Aerodinamik Modelleme",
        "color": "#ec4899",
        "desc": "Uçak ve İHA'ların Navier-Stokes aerodinamik akış ve 6-DOF uçuş dinamiği denklemlerini yapay sinir ağı kayıp fonksiyonuna gömen mimari.",
        "prompt": "PINNs mimarisinin kısmi diferansiyel denklemleri (PDE) otomatik türev (autodiff) ile sinir ağına nasıl entegre ettiğini açıkla."
    }
]

def main():
    print(f"[*] Starting ingestion of {len(AVIATION_PAPERS)} aviation/combat UAV papers...")
    catalog_path = ROOT / "data" / "corpus_catalog.json"
    existing_catalog = []
    if catalog_path.exists():
        try:
            existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing_catalog = []

    existing_ids = {item["id"] for item in existing_catalog}
    successful_ingests = []

    for idx, p in enumerate(AVIATION_PAPERS):
        paper_id = p["id"]
        paper_title = p["title"]
        print(f"\n[{idx+1}/{len(AVIATION_PAPERS)}] Processing {paper_id}: {paper_title}...")
        
        res = ingest_arxiv_paper(paper_id, title=paper_title)
        print(f"    Status: {res.get('success')} | Chunks: {res.get('chunks_count', 0)} | Msg: {res.get('error', 'OK')}")
        
        if res.get("success"):
            successful_ingests.append(p)
            if paper_id not in existing_ids:
                existing_catalog.append(p)
                existing_ids.add(paper_id)

    # Save updated catalog
    catalog_path.write_text(json.dumps(existing_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Updated corpus catalog with {len(successful_ingests)} new aviation papers.")

    # Invalidate hybrid searcher cache
    try:
        from retrieval.search import searcher
        searcher._bm25_cache.clear()
        print("[+] HybridSearcher BM25 cache cleared.")
    except Exception as e:
        print(f"[-] Cache clear notice: {e}")

    print("\n[+] Ingestion process finished successfully!")

if __name__ == "__main__":
    main()
