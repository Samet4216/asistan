"""Integration tests for RAG pipeline: Ingestion -> Retrieval -> Citations -> Abstention.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pymupdf

from personal_ai.rag.retriever import DocumentRetriever


def test_rag_pipeline_end_to_end_mocked(tmp_path: Path):
    """Verify document ingestion, search, page citation, and prompt augmentation."""
    # Create sample PDF with 2 pages
    pdf_path = tmp_path / "uav_control.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Quadcopter attitude dynamics are governed by Euler's equations.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "LQR controller weights Q and R balance state error and actuator effort.")
    doc.save(pdf_path)
    doc.close()

    # Configure custom retriever pointing to tmp directories
    chroma_dir = str(tmp_path / "chroma").replace("\\", "/")
    config_file = tmp_path / "rag_test.yaml"
    config_file.write_text(f"""
rag:
  persist_directory: "{chroma_dir}"
  default_collection: "uav_test_vault"
  chunk_size: 500
  chunk_overlap: 50
  distance_threshold: 0.50
  abstention_message: "Teknik dokümanlarda bilgi bulunamadı."
""", encoding="utf-8")

    retriever = DocumentRetriever(config_path=config_file)

    # Mock embeddings: document chunks get specific vectors
    def mock_encode(texts, **kwargs):
        vectors = []
        for t in texts:
            if "Quadcopter attitude" in t or "attitude" in t:
                # Close vector for attitude
                vectors.append([1.0, 0.0, 0.0] + [0.0] * 381)
            elif "LQR controller" in t or "LQR" in t:
                # Vector for LQR
                vectors.append([0.0, 1.0, 0.0] + [0.0] * 381)
            else:
                # Unrelated query vector (orthogonal / distant)
                vectors.append([0.0, 0.0, 1.0] + [0.0] * 381)
        return np.array(vectors)

    retriever.vector_store._embedding_model = MagicMock()
    retriever.vector_store._embedding_model.encode.side_effect = mock_encode

    # 1. Ingest PDF
    chunks_added = retriever.ingest_file(pdf_path)
    assert chunks_added == 2

    # 2. Query for LQR (matches Page 2)
    lqr_result = retriever.retrieve("LQR controller weights")
    assert lqr_result.has_context is True
    assert len(lqr_result.chunks) >= 1
    # Check citation
    assert "[Kaynak: uav_control.pdf, Sayfa: 2]" in lqr_result.context_text
    assert "LQR controller weights" in lqr_result.context_text

    # 3. Test Augmented Prompt Generation
    rag_prompt = retriever.build_rag_prompt("LQR matrisleri nasıl seçilir?", lqr_result)
    assert "BAĞLAM DOKÜMANLARI" in rag_prompt
    assert "[Kaynak: uav_control.pdf, Sayfa: 2]" in rag_prompt
    assert "KULLANICI SORUSU: LQR matrisleri nasıl seçilir?" in rag_prompt

    # 4. Zero-Hallucination Honest Abstention Test: query with no relevant matches
    unrelated_result = retriever.retrieve("What is photosynthesis in plant biology?")
    assert unrelated_result.has_context is False
    assert unrelated_result.abstention_message == "Teknik dokümanlarda bilgi bulunamadı."
    assert unrelated_result.context_text == ""
