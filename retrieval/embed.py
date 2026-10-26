"""
Embedding Module for Hybrid Retrieval (Faz 5.1).
Loads BAAI/bge-base-en-v1.5 strictly on CPU to leave VRAM 100% dedicated to LLM.
Provides a ChromaDB-compliant EmbeddingFunction for seamless collection integration.
"""

import sys
from pathlib import Path
from typing import List, Any
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class EmbeddingModel:
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.EMBED_MODEL
        self.device = device or config.DEVICE
        self.query_prefix = config.EMBED_QUERY_PREFIX
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"[*] Loading embedding model '{self.model_name}' on {self.device}...")
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        prefixed = f"{self.query_prefix}{query}"
        embedding = self.model.encode(
            [prefixed],
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embedding[0].tolist()


# Global embedding instance
embedder = EmbeddingModel()


class ChromaBGEEmbeddingFunction:
    """ChromaDB-compatible wrapper for BGE embeddings."""
    def __init__(self, embedder_instance: EmbeddingModel = None):
        self.embedder = embedder_instance or embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        return [self.embedder.embed_query(q) for q in input]

    def name(self) -> str:
        return "bge-base-en-v1.5"




chroma_bge_ef = ChromaBGEEmbeddingFunction()
