"""
Local Document Ingestion Module.
Handles drag-and-drop file uploads (PDF, Markdown, TXT) from the Web UI:
1. Parses PDF with PyMuPDF4LLM or reads Markdown/TXT.
2. Chunks text into semantically cohesive sections.
3. Indexes chunks into ChromaDB (collection: 'paper_chunks').
4. Generates an Obsidian Card in Vault/01_Paper_Cards/ so it connects to the Knowledge Graph!
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from ingest.paper_parser import PaperParser
from ingest.chunker import SectionChunker
from ingest.indexer import Indexer


def ingest_uploaded_file(
    file_bytes: bytes,
    filename: str,
    custom_title: str = ""
) -> Dict[str, Any]:
    """
    Ingests an uploaded PDF, MD, or TXT document into ChromaDB and Obsidian.
    Returns status dict with title and number of indexed chunks.
    """
    if not file_bytes:
        return {"success": False, "error": "Boş dosya yüklendi."}

    # Ensure upload directory exists
    upload_dir = config.DATA_DIR / "uploaded_docs"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    clean_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_") or "doc"
    suffix = Path(filename).suffix.lower()
    save_filename = f"{clean_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    saved_path = upload_dir / save_filename

    # Save to disk
    with open(saved_path, "wb") as f:
        f.write(file_bytes)

    doc_id = clean_stem[:30]
    resolved_title = custom_title.strip() or filename.replace("_", " ").rsplit(".", 1)[0]

    parser = PaperParser()
    chunker = SectionChunker()
    indexer = Indexer()

    try:
        # 1. Parse based on file type
        if suffix == ".pdf":
            parsed_doc = parser._parse_pdf(saved_path, arxiv_id=doc_id, default_title=resolved_title)
        elif suffix in [".md", ".markdown", ".txt"]:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            # Split by markdown headers
            parts = re.split(r"\n(#{1,3}\s+[^\n]+)\n", raw_text)
            sections = []
            if len(parts) > 1:
                sections.append({"name": "Giriş / Genel Bakış", "text": parts[0].strip(), "equations": []})
                for i in range(1, len(parts), 2):
                    sec_name = parts[i].lstrip("# ").strip()
                    sec_content = parts[i + 1] if i + 1 < len(parts) else ""
                    if sec_content.strip():
                        sections.append({"name": sec_name, "text": sec_content.strip(), "equations": []})
            else:
                sections.append({"name": "Ana Metin", "text": raw_text.strip(), "equations": []})

            parsed_doc = {
                "arxiv_id": doc_id,
                "title": resolved_title,
                "year": datetime.now().year,
                "source_type": suffix.lstrip("."),
                "sections": sections,
                "total_sections": len(sections)
            }
        else:
            return {"success": False, "error": f"Desteklenmeyen dosya formatı: {suffix}. Lütfen PDF, MD veya TXT yükleyin."}

        if not parsed_doc or not parsed_doc.get("sections"):
            return {"success": False, "error": f"Dosyadan metin veya bölüm çıkarılamadı: {filename}"}

        # 2. Chunk sections
        chunks = chunker.chunk_paper(parsed_doc)
        if not chunks:
            return {"success": False, "error": "Bölümlerden geçerli metin parçaları üretilemedi."}

        # 3. Index chunks into ChromaDB
        indexer.index_paper_chunks(chunks)

        # 4. Create an Obsidian Note in Vault/01_Paper_Cards/ to integrate into Knowledge Graph
        vault_cards_dir = config.VAULT_DIR / "01_Paper_Cards"
        vault_cards_dir.mkdir(parents=True, exist_ok=True)
        card_file = vault_cards_dir / f"{doc_id}.md"

        sec_list_md = "\n".join([f"- **{s['name']}**" for s in parsed_doc.get("sections", [])[:8]])
        card_content = (
            f"---\n"
            f"type: user_uploaded_document\n"
            f"title: \"{resolved_title}\"\n"
            f"filename: \"{filename}\"\n"
            f"date_added: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"chunks_count: {len(chunks)}\n"
            f"tags: [local-upload, user-corpus, dl-lab]\n"
            f"---\n\n"
            f"# {resolved_title}\n\n"
            f"**Kaynak Dosya:** `{filename}` | **Eklenme Tarihi:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"## 📑 Çıkarılan Temel Bölümler\n"
            f"{sec_list_md}\n\n"
            f"## 🔗 Sinaptik Bağlantılar\n"
            f"- [[Knowledge_State]]\n"
            f"- [[Self_Attention]]\n"
        )
        try:
            card_file.write_text(card_content, encoding="utf-8")
        except Exception:
            pass

        return {
            "success": True,
            "doc_id": doc_id,
            "title": resolved_title,
            "filename": filename,
            "chunks_count": len(chunks),
            "file_path": str(saved_path)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("[*] Local file ingest module ready. Use ingest_uploaded_file(file_bytes, filename) from UI.")
