"""Standardized Engineering Benchmark Evaluator (Phase 5).
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from personal_ai.runtime.ollama_client import OllamaClient


@dataclass
class BenchmarkItem:
    """Represents a single standardized evaluation prompt with grading criteria."""
    id: str
    category: str
    prompt: str
    rubric_keywords: List[str]
    negative_keywords: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Evaluation result for a single benchmark item."""
    item_id: str
    category: str
    prompt: str
    response: str
    latency_seconds: float
    rubric_score: float
    math_detected: bool
    code_detected: bool
    negative_penalties: int
    passed: bool


class BenchmarkRunner:
    """Executes evaluation benchmarks and grades model responses programmatically."""

    def __init__(self, benchmark_path: str | Path = "data/evaluation/dl_cv_benchmark.jsonl"):
        self.benchmark_path = Path(benchmark_path)

    def load_items(self) -> List[BenchmarkItem]:
        """Loads benchmark items from JSONL."""
        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_path}")

        items: List[BenchmarkItem] = []
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    items.append(
                        BenchmarkItem(
                            id=data["id"],
                            category=data.get("category", "general"),
                            prompt=data["prompt"],
                            rubric_keywords=data.get("rubric_keywords", []),
                            negative_keywords=data.get("negative_keywords", []),
                        )
                    )
        return items

    def grade_response(self, item: BenchmarkItem, response: str, latency: float) -> BenchmarkResult:
        """Automated rubric evaluation of response content, math formatting, and constraints."""
        norm_resp = response.lower()

        # 1. Rubric keyword coverage
        matches = 0
        for kw in item.rubric_keywords:
            if kw.lower() in norm_resp:
                matches += 1
        coverage_score = matches / len(item.rubric_keywords) if item.rubric_keywords else 1.0

        # 2. LaTeX Math detection ($...$ or $$...$$ or \[...\])
        math_detected = bool(re.search(r"\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", response, re.DOTALL))

        # 3. Code block detection (```...```)
        code_detected = "```" in response

        # 4. Negative constraint penalties (forbidden repetitive boilerplate)
        negative_penalties = 0
        for neg in item.negative_keywords:
            if neg.lower() in norm_resp:
                negative_penalties += 1

        # Calculate final rubric score (penalties subtract 0.25 each)
        final_score = max(0.0, coverage_score - (negative_penalties * 0.25))
        passed = (final_score >= 0.50) and (negative_penalties == 0)

        return BenchmarkResult(
            item_id=item.id,
            category=item.category,
            prompt=item.prompt,
            response=response,
            latency_seconds=round(latency, 2),
            rubric_score=round(final_score, 4),
            math_detected=math_detected,
            code_detected=code_detected,
            negative_penalties=negative_penalties,
            passed=passed,
        )

    def run_benchmark(
        self,
        client: OllamaClient,
        system_prompt: Optional[str] = None,
        max_items: Optional[int] = None,
        output_report_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Runs inference across benchmark suite and generates comprehensive report."""
        items = self.load_items()
        if max_items:
            items = items[:max_items]

        results: List[BenchmarkResult] = []
        start_all = time.time()

        for item in items:
            t0 = time.time()
            try:
                response = client.chat(
                    messages=[{"role": "user", "content": item.prompt}],
                    system_prompt=system_prompt,
                )
            except Exception as e:
                response = f"[Inference Error: {e}]"
            latency = time.time() - t0

            grade = self.grade_response(item, response, latency)
            results.append(grade)

        total_time = time.time() - start_all
        total_items = len(results)
        passed_count = sum(1 for r in results if r.passed)
        avg_score = sum(r.rubric_score for r in results) / total_items if total_items else 0.0
        avg_latency = sum(r.latency_seconds for r in results) / total_items if total_items else 0.0

        # Category breakdown
        category_breakdown: Dict[str, Dict[str, Any]] = {}
        for r in results:
            if r.category not in category_breakdown:
                category_breakdown[r.category] = {"total": 0, "passed": 0, "scores": []}
            category_breakdown[r.category]["total"] += 1
            if r.passed:
                category_breakdown[r.category]["passed"] += 1
            category_breakdown[r.category]["scores"].append(r.rubric_score)

        for cat, stats in category_breakdown.items():
            stats["average_score"] = round(sum(stats["scores"]) / len(stats["scores"]), 3)
            stats["pass_rate"] = f"{stats['passed'] / stats['total'] * 100:.1f}%"
            del stats["scores"]

        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": client.default_model,
            "total_questions": total_items,
            "passed_count": passed_count,
            "pass_rate": f"{passed_count / total_items * 100:.1f}%" if total_items else "0%",
            "average_score": round(avg_score, 4),
            "average_latency_seconds": round(avg_latency, 2),
            "total_benchmark_time_seconds": round(total_time, 2),
            "category_breakdown": category_breakdown,
            "results": [asdict(r) for r in results],
        }

        if output_report_path:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary
