"""
Cross-Encoder Reranker Module (Faz 5.3).
Reranks initial candidate chunks on CPU using cross-encoder/ms-marco-MiniLM-L-6-v2.
Filters out low-confidence chunks below threshold to prevent model hallucinations.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class Reranker:
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.RERANK_MODEL
        self.device = device or config.DEVICE
        self.threshold = config.RETRIEVAL_SCORE_THRESHOLD
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"[*] Loading cross-encoder reranker '{self.model_name}' on {self.device}...")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Reranks a list of candidate chunk dictionaries.
        Each candidate must have a 'content' key.
        Returns:
            (reranked_candidates, has_sufficient_evidence)
        """
        if not candidates:
            return [], False

        for c in candidates:
            if "\\sysname" in c.get("content", ""):
                c["content"] = c["content"].replace("\\sysname", "FlashAttention")

        pairs = [[query, c.get("content", "")] for c in candidates]
        scores = self.model.predict(pairs)

        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])

        # Sort descending by rerank score
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        top_candidates = ranked[:top_n]

        # Filter candidates meeting minimum relevance threshold
        # For MS MARCO cross-encoders, logits > -2.0 indicate genuine semantic relevance.
        # Negative extremes (-10 to -12) indicate completely unrelated content.
        valid_candidates = [c for c in top_candidates if c["rerank_score"] >= -2.0]
        has_sufficient_evidence = len(valid_candidates) > 0

        if not has_sufficient_evidence:
            return [], False

        return valid_candidates, True


# Global reranker instance
reranker = Reranker()


if __name__ == "__main__":
    print("[*] Testing rerank.py on CPU...")
    sample_query = "How is spatial pooling performed in Coordinate Attention?"
    sample_candidates = [
        {"id": "doc1", "content": "The weather in Ankara is cloudy today."},
        {"id": "doc2", "content": "Coordinate Attention uses two 1D spatial pooling kernels along X and Y axes."},
        {"id": "doc3", "content": "Standard convolutional layers have fixed receptive fields."}
    ]
    results, ok = reranker.rerank(sample_query, sample_candidates, top_n=2)
    print(f"Sufficient Evidence: {ok}")
    for r in results:
        print(f" • [{r['id']}] Score: {r['rerank_score']:.3f} | {r['content'][:60]}...")
