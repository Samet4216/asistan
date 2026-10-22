"""
Hybrid Search Module (Faz 5.2 & 5.4).
Combines Dense Vector Search (ChromaDB) + Sparse Lexical Search (BM25)
using Reciprocal Rank Fusion (RRF), followed by Cross-Encoder Reranking
and token-budgeted context packaging.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import budget
from ingest.indexer import Indexer
from retrieval.embed import embedder
from retrieval.rerank import reranker


class HybridSearcher:
    def __init__(self, indexer: Indexer = None):
        self.indexer = indexer or Indexer()
        self.embedder = embedder
        self.reranker = reranker

    def search(
        self,
        query: str,
        collection_names: List[str] = None,
        top_k: int = 5,
        section_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes hybrid search across specified collections.
        1. Dense Vector Search (top 20)
        2. Sparse BM25 Search (top 20)
        3. Reciprocal Rank Fusion (RRF)
        4. Cross-Encoder Reranking (top_k)
        5. Token Budget Context Packaging
        """
        cols = collection_names or ["paper_chunks", "paper_cards", "code", "errors"]
        all_candidates = []

        # 1. Fetch documents for BM25 and Chroma
        for col_name in cols:
            if col_name not in self.indexer.collections:
                continue
            col = self.indexer.collections[col_name]
            count = col.count()
            if count == 0:
                continue

            # Query vector store
            query_emb = self.embedder.embed_query(query)
            n_fetch = min(20, count)
            
            # Apply section filter if provided (for reverse-RAG / challenge mode)
            where_filter = None
            if section_filter and col_name == "paper_chunks":
                where_filter = {"section_name": {"$in": section_filter}}

            try:
                vector_results = col.query(
                    query_texts=[query],
                    n_results=n_fetch,
                    where=where_filter
                )
            except Exception:
                # If filter produces 0 matches, query without filter
                vector_results = col.query(
                    query_texts=[query],
                    n_results=n_fetch
                )


            v_ids = vector_results.get("ids", [[]])[0]
            v_docs = vector_results.get("documents", [[]])[0]
            v_metas = vector_results.get("metadatas", [[]])[0]

            # Build candidate list with initial vector ranks
            col_candidates = {}
            for rank, cid in enumerate(v_ids):
                col_candidates[cid] = {
                    "id": cid,
                    "content": v_docs[rank],
                    "metadata": v_metas[rank] if rank < len(v_metas) else {},
                    "collection": col_name,
                    "vector_rank": rank + 1,
                    "bm25_rank": 999
                }

            # 2. Sparse BM25 Search over the same collection
            all_data = col.get()
            all_ids = all_data.get("ids", [])
            all_docs = all_data.get("documents", [])
            all_metas = all_data.get("metadatas", [])

            if all_docs:
                tokenized_corpus = [doc.lower().split() for doc in all_docs]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)

                # Get top N by BM25 score
                bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_fetch]

                for rank, idx in enumerate(bm25_ranked_indices):
                    cid = all_ids[idx]
                    if cid in col_candidates:
                        col_candidates[cid]["bm25_rank"] = rank + 1
                    else:
                        col_candidates[cid] = {
                            "id": cid,
                            "content": all_docs[idx],
                            "metadata": all_metas[idx] if idx < len(all_metas) else {},
                            "collection": col_name,
                            "vector_rank": 999,
                            "bm25_rank": rank + 1
                        }

            # 3. Reciprocal Rank Fusion (RRF)
            # RRF Score = 1 / (60 + vector_rank) + 1 / (60 + bm25_rank)
            for c in col_candidates.values():
                rrf_score = (1.0 / (60.0 + c["vector_rank"])) + (1.0 / (60.0 + c["bm25_rank"]))
                c["rrf_score"] = rrf_score
                all_candidates.append(c)

        # Sort combined candidates by RRF score
        all_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_candidates = all_candidates[:20]

        # 4. Cross-Encoder Reranking
        reranked, has_sufficient_evidence = self.reranker.rerank(query, top_candidates, top_n=top_k)

        # 5. Pack Context with Budget
        packed_chunks = "\n\n---\n\n".join([
            f"[Source: {c['metadata'].get('title', c['id'])}, Section: {c['metadata'].get('section_name', 'N/A')}]\n{c['content']}"
            for c in reranked
        ])

        return {
            "query": query,
            "results": reranked,
            "has_sufficient_evidence": has_sufficient_evidence,
            "packed_context": packed_chunks,
            "total_candidates": len(all_candidates)
        }


# Global searcher instance
searcher = HybridSearcher()


if __name__ == "__main__":
    print("[*] Testing retrieval/search.py with hybrid RRF...")
    test_q = "How does Coordinate Attention capture position with 1D pooling?"
    res = searcher.search(test_q, top_k=3)
    print(f"\nQuery: {res['query']}")
    print(f"Has Sufficient Evidence: {res['has_sufficient_evidence']}")
    print(f"Total Combined Candidates: {res['total_candidates']}")
    print("\nTop Reranked Results:")
    for r in res["results"]:
        sec = r["metadata"].get("section_name", "N/A")
        print(f" • [{r['id']}] (Sec: {sec}) - Rerank Score: {r.get('rerank_score', 0):.3f}")
