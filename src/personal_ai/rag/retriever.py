"""Document Retrieval & Prompt Augmentation Pipeline for RAG (Phase 2).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from personal_ai.rag.chunker import TextChunker
from personal_ai.rag.document_loader import DocumentLoader, LoadedDocument
from personal_ai.rag.vector_store import RetrievedChunk, VectorStore


@dataclass
class RetrievalResult:
    """Represents the outcome of a semantic context search."""
    query: str
    has_context: bool
    chunks: List[RetrievedChunk]
    context_text: str
    sources_summary: List[str]
    abstention_message: Optional[str] = None


class DocumentRetriever:
    """High-level facade orchestrating document parsing, vector indexing, and RAG retrieval."""

    def __init__(self, config_path: str | Path = "configs/rag.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.persist_directory = self.config.get("persist_directory", "data/chroma")
        self.default_collection = self.config.get("default_collection", "engineering_vault")
        self.chunk_size = int(self.config.get("chunk_size", 600))
        self.chunk_overlap = int(self.config.get("chunk_overlap", 100))
        self.embedding_model = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_device = self.config.get("embedding_device", "cpu")
        self.top_k = int(self.config.get("top_k", 4))
        self.distance_threshold = float(self.config.get("distance_threshold", 0.75))
        self.abstention_message = self.config.get(
            "abstention_message",
            "Yüklenen kişisel teknik dokümanlarda bu konuyla ilgili doğrudan bir bilgi bulunamadı."
        )

        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.vector_store = VectorStore(
            persist_directory=self.persist_directory,
            embedding_model_name=self.embedding_model,
            embedding_device=self.embedding_device,
        )

    def _load_config(self) -> Dict[str, Any]:
        """Loads RAG configuration from YAML or returns defaults."""
        if self.config_path.exists():
            try:
                data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
                return data.get("rag", {}) if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def ingest_file(
        self,
        file_path: str | Path,
        collection_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> int:
        """Parses, chunks, and indexes a single document into the vector store."""
        col_name = collection_name or self.default_collection
        loaded_doc = self.loader.load_file(file_path)
        chunks = self.chunker.chunk_document(loaded_doc)
        return self.vector_store.add_chunks(chunks, collection_name=col_name, overwrite=overwrite)

    def ingest_directory(
        self,
        directory_path: str | Path,
        collection_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """Scans a directory and ingests all supported PDF, Markdown, and TXT files."""
        dir_path = Path(directory_path).resolve()
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")

        results: Dict[str, int] = {}
        for ext in DocumentLoader.SUPPORTED_EXTENSIONS:
            for file_path in dir_path.glob(f"*{ext}"):
                try:
                    num_chunks = self.ingest_file(file_path, collection_name=collection_name, overwrite=overwrite)
                    results[file_path.name] = num_chunks
                except Exception as e:
                    results[file_path.name] = -1
        return results

    def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """Queries the vector store and builds structured citations or honest abstention."""
        col_name = collection_name or self.default_collection
        k = top_k or self.top_k
        thresh = distance_threshold or self.distance_threshold

        chunks = self.vector_store.search(
            query=query,
            collection_name=col_name,
            top_k=k,
            distance_threshold=thresh,
        )

        if not chunks:
            return RetrievalResult(
                query=query,
                has_context=False,
                chunks=[],
                context_text="",
                sources_summary=[],
                abstention_message=self.abstention_message,
            )

        context_lines: List[str] = ["=== BAĞLAM DOKÜMANLARI ==="]
        sources_summary: List[str] = []

        for chunk in chunks:
            citation = f"[Kaynak: {chunk.source_file}, Sayfa: {chunk.page_number}]"
            context_lines.append(f"\n{citation}\n{chunk.text.strip()}\n---")
            if citation not in sources_summary:
                sources_summary.append(citation)

        context_text = "\n".join(context_lines)
        return RetrievalResult(
            query=query,
            has_context=True,
            chunks=chunks,
            context_text=context_text,
            sources_summary=sources_summary,
            abstention_message=None,
        )

    def build_rag_prompt(self, user_query: str, retrieval: RetrievalResult) -> str:
        """Builds augmented prompt with strict zero-hallucination and citation instructions."""
        if not retrieval.has_context:
            # When no context is found, do not inject hallucinated context
            return user_query

        return (
            f"{retrieval.context_text}\n\n"
            f"KULLANICI SORUSU: {user_query}\n\n"
            f"YÖNERGE: Yukarıdaki bağlam dokümanlarından yararlanarak soruyu yanıtla. "
            f"Verdiğin kritik teknik bilgilerin yanına ilgili kaynağı ekle (örn: `[Kaynak: dosya.pdf, Sayfa: X]`). "
            f"Dokümanda doğrudan yer almayan çıkarımları spekülasyon olarak belirt."
        )
