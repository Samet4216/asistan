"""
Bulk Indexing Utility for User Engineering Vault & Knowledge Base.

Indexes all paper cards, theory notes, and personal profiles from Vault/ and SametBeyin/
into the production ChromaDB vector store (data/chroma).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from personal_ai.rag.retriever import DocumentRetriever
from personal_ai.rag.document_loader import DocumentLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vault_indexer")


def index_user_vault() -> None:
    project_root = Path(__file__).parent.parent.resolve()
    retriever = DocumentRetriever(config_path=project_root / "configs" / "rag.yaml")

    target_dirs = [
        project_root / "Vault" / "01_Paper_Cards",
        project_root / "Vault" / "02_Theory_Notes",
        project_root / "Vault" / "00_Profile",
        project_root / "SametBeyin" / "🧠 Bilgi",
        project_root / "SametBeyin" / "🔮 Hafiza",
    ]

    total_files = 0
    total_chunks = 0
    failed_files = 0

    print("=================================================================")
    print("📚 KİŞİSEL MÜHENDİSLİK DOKÜMANLARI RAG İNDEKSLEME BAŞLATILIYOR")
    print(f"🎯 Hedef Vektör Veritabanı: {retriever.persist_directory}")
    print(f"📦 Hedef Koleksiyon: {retriever.default_collection}")
    print("=================================================================\n")

    for dir_path in target_dirs:
        if not dir_path.is_dir():
            logger.warning(f"Dizin bulunamadı, atlanıyor: {dir_path}")
            continue

        print(f"📁 Taranıyor: {dir_path.relative_to(project_root)}")
        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS and file_path.is_file():
                try:
                    num_chunks = retriever.ingest_file(file_path, overwrite=False)
                    if num_chunks > 0:
                        total_files += 1
                        total_chunks += num_chunks
                        print(f"  ✓ {file_path.name} -> {num_chunks} chunk indekslendi.")
                    elif num_chunks == 0:
                        print(f"  - {file_path.name} (Zaten indeksli, atlandı)")
                except Exception as e:
                    failed_files += 1
                    print(f"  ✗ {file_path.name} -> Hata: {e}")

    print("\n=================================================================")
    print(f"✅ İNDEKSLEME TAMAMLANDI!")
    print(f"📄 İndekslenen Yeni Dosyalar: {total_files}")
    print(f"🧩 Eklenen Yeni Chunk Sayısı: {total_chunks}")
    print(f"⚠️ Hata Alan Dosyalar: {failed_files}")
    print("=================================================================\n")

    # Doğrulama Test Sorgusu
    test_queries = [
        "BTU Mekatronik Robotik",
        "Transformer self attention mimarisi",
        "İHA güdüm ve hedef takip sistemleri",
    ]

    print("🔍 DOĞRULAMA TEST SORGULARI ÇALIŞTIRILIYOR:")
    for query in test_queries:
        res = retriever.retrieve(query, top_k=2)
        print(f"\nSoru: '{query}'")
        if res.has_context:
            print(f"  Kaynaklar: {', '.join(res.sources_summary)}")
            first_chunk = res.chunks[0]
            print(f"  En iyi eşleşme ({first_chunk.source_file}, skor: {first_chunk.distance:.3f}):")
            snippet = first_chunk.text.replace("\n", " ")[:140]
            print(f"  \"{snippet}...\"")
        else:
            print(f"  Uyarı: Bağlam bulunamadı ({res.abstention_message})")


if __name__ == "__main__":
    index_user_vault()
