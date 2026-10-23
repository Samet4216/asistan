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
from retrieval.query_transform import query_transformer


class HybridSearcher:
    def __init__(self, indexer: Indexer = None):
        self.indexer = indexer or Indexer()
        self.embedder = embedder
        self.reranker = reranker
        self.query_transformer = query_transformer

    def search(
        self,
        query: str,
        collection_names: List[str] = None,
        top_k: int = 5,
        section_filter: Optional[List[str]] = None,
        use_decomposition: bool = False,
        use_hyde: bool = False
    ) -> Dict[str, Any]:
        """
        Executes hybrid search across specified collections.
        Supports:
        - Sub-Query Decomposition for complex multi-part queries
        - HyDE (Hypothetical Document Embeddings) for bridging vocabulary gaps
        1. Dense Vector Search (top 20)
        2. Sparse BM25 Search (top 20)
        3. Reciprocal Rank Fusion (RRF)
        4. Cross-Encoder Reranking (top_k)
        5. Token Budget Context Packaging
        """
        cols = collection_names or ["paper_chunks", "paper_cards", "code", "errors"]

        # Determine sub-queries and hypothetical passage
        sub_queries = [query]
        if use_decomposition:
            sub_queries = self.query_transformer.decompose(query)

        hyde_passage = None
        if use_hyde:
            hyde_passage = self.query_transformer.generate_hyde(query)

        master_candidates: Dict[str, Dict[str, Any]] = {}

        for sq in sub_queries:
            # Query vector text is either the HyDE passage or the sub-query
            dense_query_text = hyde_passage if (use_hyde and hyde_passage) else sq

            for col_name in cols:
                if col_name not in self.indexer.collections:
                    continue
                col = self.indexer.collections[col_name]
                count = col.count()
                if count == 0:
                    continue

                n_fetch = min(20, count)

                # 1. Dense Vector Query
                where_filter = None
                if section_filter and col_name == "paper_chunks":
                    where_filter = {"section_name": {"$in": section_filter}}

                try:
                    vector_results = col.query(
                        query_texts=[dense_query_text],
                        n_results=n_fetch,
                        where=where_filter
                    )
                except Exception:
                    vector_results = col.query(
                        query_texts=[dense_query_text],
                        n_results=n_fetch
                    )

                v_ids = vector_results.get("ids", [[]])[0]
                v_docs = vector_results.get("documents", [[]])[0]
                v_metas = vector_results.get("metadatas", [[]])[0]

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

                # 2. Sparse BM25 Query
                all_data = col.get()
                all_ids = all_data.get("ids", [])
                all_docs = all_data.get("documents", [])
                all_metas = all_data.get("metadatas", [])

                if all_docs:
                    tokenized_corpus = [doc.lower().split() for doc in all_docs]
                    bm25 = BM25Okapi(tokenized_corpus)
                    tokenized_query = sq.lower().split()
                    bm25_scores = bm25.get_scores(tokenized_query)

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

                # 3. Accumulate Reciprocal Rank Fusion (RRF) Scores across sub-queries
                for cid, c in col_candidates.items():
                    rrf_contrib = (1.0 / (60.0 + c["vector_rank"])) + (1.0 / (60.0 + c["bm25_rank"]))
                    key = f"{col_name}:{cid}"
                    if key in master_candidates:
                        master_candidates[key]["rrf_score"] += rrf_contrib
                    else:
                        c["rrf_score"] = rrf_contrib
                        master_candidates[key] = c

        # Rank all accumulated candidates
        all_candidates = list(master_candidates.values())
        all_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        top_candidates = all_candidates[:25]

        # 4. Cross-Encoder Reranking
        reranked, has_sufficient_evidence = self.reranker.rerank(query, top_candidates, top_n=top_k)

        # 5. Pack Context with Budget
        packed_chunks = "\n\n---\n\n".join([
            f"[Source: {c['metadata'].get('title', c['id'])}, Section: {c['metadata'].get('section_name', 'N/A')}]\n{c['content']}"
            for c in reranked
        ])

        return {
            "query": query,
            "sub_queries": sub_queries,
            "hyde_passage": hyde_passage,
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
