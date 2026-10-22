"""
Indexing & Hash-Guarded Vector/BM25 Store Module (Faz 3.5 & 5.1).
Integrates ChromaBGEEmbeddingFunction (768-dim) across all Chroma collections.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from retrieval.embed import embedder, chroma_bge_ef


class Indexer:
    def __init__(self, chroma_dir: Path = None):
        self.chroma_dir = Path(chroma_dir or config.CHROMA_DIR).resolve()
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.embedder = embedder
        self.ef = chroma_bge_ef

        # Initialize collections explicitly with 768-dim BGE embedding function
        self.collection_names = ["paper_chunks", "paper_cards", "code", "errors", "experiments"]
        self.collections = {
            name: self.client.get_or_create_collection(
                name=name,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"}
            )
            for name in self.collection_names
        }

        self.hash_log_file = self.chroma_dir / "file_hashes.json"
        self.hashes = self._load_hashes()

    def _load_hashes(self) -> Dict[str, str]:
        if self.hash_log_file.exists():
            try:
                return json.loads(self.hash_log_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_hashes(self):
        self.hash_log_file.write_text(json.dumps(self.hashes, indent=2), encoding="utf-8")

    def compute_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def index_paper_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        col = self.collections["paper_chunks"]
        ids = [c["id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "arxiv_id": str(c.get("arxiv_id", "")),
                "title": str(c.get("title", "")),
                "year": int(c.get("year", 2021)),
                "section_name": str(c.get("section_name", "")),
                "part_no": int(c.get("part_no", 1)),
                "type": "paper_chunk"
            }
            for c in chunks
        ]

        # ChromaBGEEmbeddingFunction encodes documents automatically
        col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[+] Upserted {len(chunks)} chunks into Chroma collection 'paper_chunks'.")

    def index_errors(self, errors: List[Dict[str, Any]]):
        if not errors:
            return
        col = self.collections["errors"]
        ids = [e["id"] for e in errors]
        documents = [f"[{e.get('code', '')}] Symptom: {e.get('symptom', '')} | Mitigation: {e.get('mitigation', '')}" for e in errors]
        metadatas = [
            {
                "code": str(e.get("code", "")),
                "type": "error_history",
                "tags": ",".join(e.get("tags", [])) if isinstance(e.get("tags"), list) else str(e.get("tags", ""))
            }
            for e in errors
        ]
        col.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[+] Upserted {len(errors)} error records into Chroma 'errors'.")

    def index_vault_note(self, note_id: str, content: str, metadata: Dict[str, Any], collection_name: str = "experiments"):
        if collection_name not in self.collections:
            collection_name = "experiments"
        col = self.collections[collection_name]
        col.upsert(
            ids=[note_id],
            documents=[content],
            metadatas=[{str(k): str(v) for k, v in metadata.items()}]
        )
        print(f"[+] Upserted note '{note_id}' into Chroma '{collection_name}'.")

    def reset_all_collections(self):
        """Wipes and re-initializes all collections to guarantee 768-dim consistency."""
        for name in self.collection_names:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"}
            )
        print("[*] All collections reset with 768-dim BGE schema.")

    def get_collection_stats(self) -> Dict[str, int]:
        return {name: col.count() for name, col in self.collections.items()}


if __name__ == "__main__":
    indexer = Indexer()
    print("[*] ChromaDB Collections Initialized:")
    for name, count in indexer.get_collection_stats().items():
        print(f" • {name}: {count} items")
