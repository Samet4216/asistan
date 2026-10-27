"""
Cross-Encoder Reranker Module (Faz 5.3).
Reranks initial candidate chunks on CPU using cross-encoder/ms-marco-MiniLM-L-6-v2.
Filters out low-confidence chunks below threshold to prevent model hallucinations.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
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
            try:
                self._model = CrossEncoder(self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
        original_query: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Reranks a list of candidate chunk dictionaries.
        Supports dual-language scoring (English search query + Turkish original query)
        to prevent language penalties on Turkish notes or textbooks.
        Applies ground-truth priority and diverse chunk coverage.
        """
        if not candidates:
            return [], False

        for c in candidates:
            if "\\sysname" in c.get("content", ""):
                c["content"] = c["content"].replace("\\sysname", "FlashAttention")

        pairs = [[query, c.get("content", "")] for c in candidates]
        scores = self.model.predict(pairs)

        # If original query differs (e.g. Turkish question vs English search query),
        # take the maximum score to ensure non-English passages are fairly scored.
        if original_query and original_query.strip().lower() != query.strip().lower():
            try:
                pairs_orig = [[original_query, c.get("content", "")] for c in candidates]
                scores_orig = self.model.predict(pairs_orig)
                scores = [max(float(s1), float(s2)) for s1, s2 in zip(scores, scores_orig)]
            except Exception:
                pass

        for i, cand in enumerate(candidates):
            score = float(scores[i])
            # Boost verified ground-truth teacher corrections and specialized BTÜ course compendiums
            meta = cand.get("metadata", {})
            arxiv_id = str(meta.get("arxiv_id", ""))
            cid = str(cand.get("id", ""))
            if arxiv_id == "teacher_verified" or "[DOĞRULANMIŞ REFERANS]" in cand.get("content", ""):
                score += 4.0
            elif cid.startswith("BTU-MEK-") or "BTU-MEK-" in arxiv_id:
                score += 2.5
            cand["rerank_score"] = score

        # Sort descending by rerank score
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Filter candidates meeting minimum relevance threshold
        # For MS MARCO cross-encoders, logits >= -2.0 indicate genuine semantic relevance.
        valid_candidates = [c for c in ranked if c["rerank_score"] >= -2.0]
        if not valid_candidates:
            return [], False

        # Source diversity: Limit chunks from the exact same paper and section to max 2
        # to ensure the prompt receives multi-perspective evidence rather than redundant repeats.
        diverse_top: List[Dict[str, Any]] = []
        section_counts: Dict[str, int] = {}

        for c in valid_candidates:
            meta = c.get("metadata", {})
            paper_id = meta.get("arxiv_id", c.get("id", "doc")).split("_")[0]
            sec_name = meta.get("section_name", "general")
            key = f"{paper_id}:{sec_name}"

            count = section_counts.get(key, 0)
            if count < 2 or len(valid_candidates) <= top_n:
                diverse_top.append(c)
                section_counts[key] = count + 1
            if len(diverse_top) >= top_n:
                break

        # Fallback if diversity filtering left too few results
        if len(diverse_top) < min(top_n, len(valid_candidates)):
            diverse_top = valid_candidates[:top_n]

        return diverse_top, True


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
