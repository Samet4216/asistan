"""Persistent Vector Store management using ChromaDB and SentenceTransformers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from personal_ai.rag.chunker import DocumentChunk


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with similarity distance score."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    distance: float  # Cosine distance: smaller means more similar (0.0 = identical)
    metadata: Dict[str, Any]


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


class VectorStore:
    """Manages ChromaDB collections, embeddings, and similarity search."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_device: str = "cpu",
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Lazy load embedding model to save memory until first use
        self._embedding_model: Optional[SentenceTransformer] = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Loads SentenceTransformer on demand."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.embedding_device
            )
        return self._embedding_model

    def get_or_create_collection(self, collection_name: str = "engineering_vault"):
        """Returns or initializes a ChromaDB collection."""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def is_document_indexed(self, file_hash: str, collection_name: str = "engineering_vault") -> bool:
        """Checks if chunks matching the file_hash already exist in the collection."""
        col = self.get_or_create_collection(collection_name)
        results = col.get(where={"file_hash": file_hash}, limit=1)
        return len(results.get("ids", [])) > 0

    def delete_document(self, file_hash: str, collection_name: str = "engineering_vault") -> int:
        """Deletes all chunks associated with a specific file hash."""
        col = self.get_or_create_collection(collection_name)
        matches = col.get(where={"file_hash": file_hash})
        ids_to_del = matches.get("ids", [])
        if ids_to_del:
            col.delete(ids=ids_to_del)
        return len(ids_to_del)

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        collection_name: str = "engineering_vault",
        overwrite: bool = False,
    ) -> int:
        """Embeds and indexes document chunks. Skips if already indexed unless overwrite=True."""
        if not chunks:
            return 0

        file_hash = chunks[0].file_hash
        if self.is_document_indexed(file_hash, collection_name):
            if not overwrite:
                return 0  # Already indexed; prevent uncontrolled duplication
            self.delete_document(file_hash, collection_name)

        col = self.get_or_create_collection(collection_name)

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "file_hash": chunk.file_hash,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            for chunk in chunks
        ]

        col.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        collection_name: str = "engineering_vault",
        top_k: int = 4,
        distance_threshold: float = 0.75,
    ) -> List[RetrievedChunk]:
        """Performs cosine similarity search for a query and filters by distance threshold."""
        col = self.get_or_create_collection(collection_name)
        if col.count() == 0:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        results = col.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[RetrievedChunk] = []
        doc_ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(doc_ids, docs, metas, distances):
            if dist <= distance_threshold:
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        source_file=meta.get("source_file", "unknown"),
                        page_number=int(meta.get("page_number", 1)),
                        distance=float(dist),
                        metadata=meta,
                    )
                )

        return retrieved
