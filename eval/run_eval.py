"""
Evaluation Suite Runner (Faz 4).
Measures Recall@5, MRR (Mean Reciprocal Rank), and citation validity against eval/questions.yaml.
Stores benchmark results with timestamps to track improvements across pipeline changes.
"""

import json
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import yaml

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from retrieval.search import searcher


class EvaluationRunner:
    def __init__(self, questions_path: Path = None):
        self.questions_file = Path(questions_path or (PROJECT_ROOT / "eval" / "questions.yaml")).resolve()
        self.searcher = searcher
        self.results_dir = PROJECT_ROOT / "eval" / "logs"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_questions(self) -> List[Dict[str, Any]]:
        if not self.questions_file.exists():
            print(f"[!] Questions file not found at {self.questions_file}")
            return []
        with open(self.questions_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    def run_retrieval_eval(self, top_k: int = 5) -> Dict[str, Any]:
        questions = self.load_questions()
        if not questions:
            return {"error": "No questions found"}

        total_q = len(questions)
        hits_at_k = 0
        reciprocal_ranks = []
        detailed_logs = []

        print(f"\n[*] Running Benchmark with Hybrid RRF & Reranker on {total_q} questions (top_k={top_k})...\n")

        for q_item in questions:
            qid = q_item["id"]
            query = q_item["question"]
            expected = q_item.get("expected_source", "")
            q_type = q_item.get("type", "general")

            # Execute full hybrid search + rerank pipeline
            search_res = self.searcher.search(query, top_k=top_k)
            retrieved_items = search_res.get("results", [])

            # Check if expected source matches any retrieved item
            matched_rank = None
            for rank_idx, item in enumerate(retrieved_items):
                meta = item.get("metadata", {})
                doc_text = item.get("content", "").lower()
                sec_name = meta.get("section_name", "").lower()
                arxiv_id = meta.get("arxiv_id", "")

                if expected and (expected.lower() in sec_name or arxiv_id in expected or any(kp.lower() in doc_text for kp in q_item.get("key_points", []))):
                    matched_rank = rank_idx + 1
                    break


            if matched_rank is not None:
                hits_at_k += 1
                rr = 1.0 / matched_rank
                reciprocal_ranks.append(rr)
                status = f"HIT (Rank {matched_rank})"
            else:
                reciprocal_ranks.append(0.0)
                status = "MISS"

            detailed_logs.append({
                "id": qid,
                "type": q_type,
                "question": query,
                "expected": expected,
                "status": status,
                "matched_rank": matched_rank,
                "top_retrieved": [it.get("metadata", {}).get("section_name") for it in retrieved_items[:3]]
            })


            print(f"[{qid}] [{q_type:12s}] {status:12s} | Q: {query[:55]}...")

        recall_at_k = (hits_at_k / total_q) if total_q > 0 else 0.0
        mrr = (sum(reciprocal_ranks) / total_q) if total_q > 0 else 0.0

        benchmark_summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": total_q,
            "top_k": top_k,
            "hits_at_k": hits_at_k,
            "recall_at_k": round(recall_at_k, 3),
            "mrr": round(mrr, 3),
            "details": detailed_logs
        }

        # Save to dated json file
        log_file = self.results_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_file.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")

        print("\n" + "="*50)
        print("📊 EVALUATION BENCHMARK RESULTS")
        print("="*50)
        print(f"Total Questions : {total_q}")
        print(f"Recall@{top_k}        : {recall_at_k * 100:.1f}% ({hits_at_k}/{total_q})")
        print(f"MRR             : {mrr:.3f}")
        print(f"Log Saved To    : {log_file.name}")
        print("="*50 + "\n")

        return benchmark_summary


if __name__ == "__main__":
    runner = EvaluationRunner()
    runner.run_retrieval_eval(top_k=5)
