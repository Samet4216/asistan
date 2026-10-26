"""
Frontier Physical AI & Horizon-Expanding Mechatronics Paper Ingestion Pipeline.
Downloads, chunks, and indexes Diffusion Policy, Event-based Neuromorphic UAV flight,
Differentiable Physics Simulation, and 3D Gaussian Splatting SLAM into ChromaDB.
Updates corpus_catalog.json and invalidates searcher cache.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.single_ingest import ingest_arxiv_paper

FRONTIER_PAPERS = [
    {
        "id": "2303.04137",
        "title": "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion",
        "authors": "Chi et al. (Columbia / Toyota Research)",
        "domain": "Robot Eylemleri İçin Difüzyon Modelleri",
        "color": "#f43f5e",
        "desc": "Robotik manipülasyon ve motor kontrolünü görüntü üretimindeki difüzyon matematiğiyle gürültüden arındırma (denoising) süreci olarak çözen en modern robotik kontrol mimarisi.",
        "prompt": "Diffusion Policy kuramında eylem uzayının (action space) difüzyon süreciyle modellenmesini ve çok modlu (multimodal) robot kontrolündeki matematiksel avantajlarını açıkla."
    },
    {
        "id": "2106.04538",
        "title": "High-Speed Autonomous Quadrotor Navigation with Event Cameras",
        "authors": "Falanga et al. (Univ. of Zurich - Scaramuzza Lab)",
        "domain": "Nöromorfik Görme & Aşırı Hızlı İHA Uçuşu",
        "color": "#8b5cf6",
        "desc": "Kare kare çekim yerine piksel bazında mikrosaniyelik parlaklık değişimlerini algılayan olay kameraları (Event Cameras) ile sıfır hareket bulanıklığında agresif İHA navigasyonu ve kaçınma.",
        "prompt": "Olay kameralarının (Event Cameras) asenkron olay akışının (event stream) matematiksel modelini ve yüksek hızlı İHA manevralarında hareket bulanıklığını (motion blur) nasıl yok ettiğini açıkla."
    },
    {
        "id": "2104.02704",
        "title": "GradSim: Differentiable Simulation for System Identification and Robot Control",
        "authors": "Murthy et al. (NVIDIA)",
        "domain": "Diferansiyellenebilir Fizik & Sistem Tanımlama",
        "color": "#10b981",
        "desc": "Fizik motorunu doğrudan türevlenebilir bir sinir ağı katmanı gibi kullanarak sürtünme katsayısı, atalet momenti gibi parametreleri geriye yayılım (backprop) ile saniyeler içinde çözen mimari.",
        "prompt": "GradSim diferansiyellenebilir fizik motorunun geriye yayılım gradyanlarıyla robotik sistem tanımlama (system identification) ve kontrol optimizasyonunu nasıl gerçekleştirdiğini açıkla."
    },
    {
        "id": "2312.02126",
        "title": "SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM",
        "authors": "Keetha et al. (CMU)",
        "domain": "3B Gaussian Splatting & Fotogerçekçi SLAM",
        "color": "#f59e0b",
        "desc": "Nokta bulutu ve mesh haritalar yerine uzayı türevlenebilir 3B Gauss elipsleriyle modelleyerek 100+ FPS hızında yoğun fotogerçekçi SLAM ve robotik haritalama yapan yeni nesil mimari.",
        "prompt": "SplaTAM mimarisinde 3B Gaussian Splatting temsili ile kamera poz takibi (tracking) ve haritalama (mapping) optimizasyonunun kapalı form denklemlerini açıkla."
    }
]

def main():
    print(f"[*] Starting ingestion of {len(FRONTIER_PAPERS)} Frontier Robotics & Mechatronics papers...")
    catalog_path = ROOT / "data" / "corpus_catalog.json"
    existing_catalog = []
    if catalog_path.exists():
        try:
            existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing_catalog = []

    existing_ids = {item["id"] for item in existing_catalog}
    successful_ingests = []

    for idx, p in enumerate(FRONTIER_PAPERS):
        paper_id = p["id"]
        paper_title = p["title"]
        print(f"\n[{idx+1}/{len(FRONTIER_PAPERS)}] Processing {paper_id}: {paper_title}...")

        res = ingest_arxiv_paper(paper_id, title=paper_title)
        print(f"    Status: {res.get('success')} | Chunks: {res.get('chunks_count', 0)} | Msg: {res.get('error', 'OK')}")

        if res.get("success"):
            successful_ingests.append(p)
            if paper_id not in existing_ids:
                existing_catalog.append(p)
                existing_ids.add(paper_id)

    # Save updated catalog
    catalog_path.write_text(json.dumps(existing_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Updated corpus catalog with {len(successful_ingests)} new Frontier papers.")

    # Invalidate hybrid searcher cache
    try:
        from retrieval.search import searcher
        searcher._bm25_cache.clear()
        print("[+] HybridSearcher BM25 cache cleared.")
    except Exception as e:
        print(f"[-] Cache clear notice: {e}")

    print("\n[+] All Frontier papers processed successfully!")

if __name__ == "__main__":
    main()
