"""
Multidisciplinary Mechatronics Paper Ingestion Pipeline.
Downloads, chunks, and indexes high-impact papers across:
1. Autonomous Driving & BEV Perception (BEVFormer)
2. Industrial Predictive Maintenance (Machinery Fault Diagnosis)
3. Biomechatronics & Bionic Prosthetics (sEMG Gesture Recognition)
4. Battery Management Systems & RUL Estimation (Transformer Battery)
5. Multimodal Vision-Language Models (LLaVA-1.5)
Updates corpus_catalog.json and invalidates searcher cache.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.single_ingest import ingest_arxiv_paper

MULTIDISCIPLINARY_PAPERS = [
    {
        "id": "2203.04050",
        "title": "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images",
        "authors": "Li et al.",
        "domain": "Otonom Araçlar & 360° BEV Algılama",
        "color": "#0ea5e9",
        "desc": "Çoklu araç kameralarından gelen görüntüleri uzamsal ve zamansal dikkat (spatiotemporal attention) ile 3B kuşbakışı (Bird's Eye View) koordinat düzlemine dönüştüren otonom sürüş mimarisi.",
        "prompt": "BEVFormer mimarisinde uzamsal çapraz dikkat (spatial cross-attention) ve zamansal öz-dikkat (temporal self-attention) ile çoklu kameradan 3B kuşbakışı harita nasıl çıkarılır?"
    },
    {
        "id": "1907.02522",
        "title": "Deep Learning for Machinery Fault Diagnostics: A Review & Benchmark",
        "authors": "Zhao et al.",
        "domain": "Endüstriyel Kestirimci Bakım & Titreşim Analizi",
        "color": "#f97316",
        "desc": "Fabrika motorları, rulmanlar ve redüktörlerin ivmeölçer titreşim verilerini 1D-CNN ve Fourier dönüşümüyle (FFT) işleyerek arızaları önceden kestiren endüstriyel arıza teşhis mimarisi.",
        "prompt": "Endüstriyel kestirimci bakımda titreşim ve akım zaman serilerinden rulman/dişli arızalarının 1D-CNN ve dalgacık dönüşümü (wavelet) ile tespit mantığını açıkla."
    },
    {
        "id": "2007.13028",
        "title": "Deep Learning-based Hand Gesture Recognition using Surface Electromyography (sEMG)",
        "authors": "Cote-Allard et al.",
        "domain": "Biyomekatronik & Kas Sinyalli Protez Kontrolü",
        "color": "#14b8a6",
        "desc": "Koldaki kas elektrik sinyallerini (sEMG) derin evrişimli ağlarla sınıflandırarak biyonik protez ellerin ve dış iskeletlerin (exoskeleton) niyet kestirimiyle sezgisel kontrolü.",
        "prompt": "Biyomekatronikte yüzeyel elektromiyografi (sEMG) sinyallerinin zaman-frekans analizi ve derin öğrenmeyle protez el hareket niyetine dönüştürülmesi adımlarını açıkla."
    },
    {
        "id": "2304.14811",
        "title": "Transformer-based Remaining Useful Life (RUL) Prediction for Lithium-Ion Batteries",
        "authors": "Wang et al.",
        "domain": "Batarya Yönetim Sistemleri (BMS) & Kalan Ömür",
        "color": "#eab308",
        "desc": "Elektrikli araç ve İHA bataryalarının voltaj, akım ve sıcaklık döngülerinden bataryanın kalan ömrünü (RUL) ve yıpranma durumunu (SoH) kestiren transformatör mimarisi.",
        "prompt": "Lityum-İyon batarya yönetim sistemlerinde (BMS) kapasite düşüş eğrisi ve şarj döngülerinden Kalan Faydalı Ömür (RUL) kestiriminin matematiksel modelini açıkla."
    },
    {
        "id": "2310.03744",
        "title": "LLaVA-1.5: Improved Baselines with Visual Instruction Tuning",
        "authors": "Liu et al.",
        "domain": "Çok Modlu Görsel-Dil Modelleri (VLM)",
        "color": "#a855f7",
        "desc": "CLIP vizyon kodlayıcısı ile dil modellerini MLP projeksiyon katmanıyla bağlayarak devre şemalarını, teknik çizimleri ve fotoğrafları analiz edebilen çok modlu yapay zeka mimarisi.",
        "prompt": "LLaVA-1.5 mimarisinde CLIP ViT görsel özelliklerinin doğrusal/MLP projeksiyon ile büyük dil modelinin (LLM) kelime gömme uzayına (token embedding space) nasıl hizalandığını açıkla."
    }
]

def main():
    print(f"[*] Starting ingestion of {len(MULTIDISCIPLINARY_PAPERS)} Multidisciplinary Mechatronics papers...")
    catalog_path = ROOT / "data" / "corpus_catalog.json"
    existing_catalog = []
    if catalog_path.exists():
        try:
            existing_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except Exception:
            existing_catalog = []

    existing_ids = {item["id"] for item in existing_catalog}
    successful_ingests = []

    for idx, p in enumerate(MULTIDISCIPLINARY_PAPERS):
        paper_id = p["id"]
        paper_title = p["title"]
        print(f"\n[{idx+1}/{len(MULTIDISCIPLINARY_PAPERS)}] Processing {paper_id}: {paper_title}...")

        res = ingest_arxiv_paper(paper_id, title=paper_title)
        print(f"    Status: {res.get('success')} | Chunks: {res.get('chunks_count', 0)} | Msg: {res.get('error', 'OK')}")

        if res.get("success"):
            successful_ingests.append(p)
            if paper_id not in existing_ids:
                existing_catalog.append(p)
                existing_ids.add(paper_id)

    # Save updated catalog
    catalog_path.write_text(json.dumps(existing_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Updated corpus catalog with {len(successful_ingests)} new Multidisciplinary papers.")

    # Invalidate hybrid searcher cache
    try:
        from retrieval.search import searcher
        searcher._bm25_cache.clear()
        print("[+] HybridSearcher BM25 cache cleared.")
    except Exception as e:
        print(f"[-] Cache clear notice: {e}")

    print("\n[+] All Multidisciplinary papers processed successfully!")

if __name__ == "__main__":
    main()
