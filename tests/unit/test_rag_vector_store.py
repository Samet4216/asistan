"""Unit tests for VectorStore with ChromaDB.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

from personal_ai.rag.chunker import DocumentChunk
from personal_ai.rag.vector_store import VectorStore


def test_vector_store_add_and_search_mocked(tmp_path: Path):
    """Verify chunk ingestion, deduplication, and search using mock embeddings."""
    store = VectorStore(persist_directory=tmp_path / "chroma")

    # Mock embedding model to return deterministic 384-dimensional vectors
    mock_model = MagicMock()
    # Query vector and document vector are close to each other
    mock_model.encode.side_effect = lambda texts, **kwargs: np.array([[0.1] * 384 for _ in texts])
    store._embedding_model = mock_model

    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            text="Kalman filtering is an optimal linear quadratic estimator.",
            source_file="kalman.pdf",
            page_number=3,
            file_hash="hash_abc",
            chunk_index=0,
            char_start=0,
            char_end=58,
        )
    ]

    # Ingest chunks
    added_count = store.add_chunks(chunks, collection_name="test_col")
    assert added_count == 1
    assert store.is_document_indexed("hash_abc", collection_name="test_col")

    # Second ingestion of same hash should be skipped (deduplication)
    duplicate_added = store.add_chunks(chunks, collection_name="test_col", overwrite=False)
    assert duplicate_added == 0

    # Search
    results = store.search(
        query="What is Kalman filter?",
        collection_name="test_col",
        top_k=2,
        distance_threshold=0.8,
    )

    assert len(results) == 1
    assert results[0].source_file == "kalman.pdf"
    assert results[0].page_number == 3
    assert "optimal linear quadratic" in results[0].text
