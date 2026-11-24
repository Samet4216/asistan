"""
Teacher vs Student Distillation Benchmark Comparator.

Evaluates teacher and student models side-by-side across:
- Task Accuracy & Task Performance
- Inference Latency (ms/sample & tokens/sec)
- Memory Consumption (active parameters, VRAM allocation, weight size)
- Regression & Retention Analysis
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkItem:
    """A benchmark evaluation item."""
    id: str
    prompt: str
    expected_output: str
    task_category: str  # 'code', 'math', 'reasoning'
    evaluator: Optional[Callable[[str, str], bool]] = None


@dataclass
class ModelMetrics:
    """Evaluation telemetry for a single model."""
    name: str
    accuracy: float
    avg_latency_ms: float
    tokens_per_sec: float
    param_count: int
    model_size_mb: float
    peak_vram_mb: float
    per_task_results: Dict[str, bool] = field(default_factory=dict)


@dataclass
class DistillationComparisonReport:
    """Side-by-side comparison between teacher and distilled student."""
    teacher_metrics: ModelMetrics
    student_metrics: ModelMetrics
    speedup_factor: float
    compression_ratio: float
    vram_reduction_pct: float
    matched_tasks: List[str]
    regressed_tasks: List[str]
    improved_tasks: List[str]

    def to_markdown(self) -> str:
        """Render comparison report as clean markdown table."""
        lines = [
            "### Teacher vs Student Distillation Benchmark Report",
            "",
            "| Metric | Teacher Model | Distilled Student | Delta / Ratio |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Model Name** | `{self.teacher_metrics.name}` | `{self.student_metrics.name}` | - |",
            f"| **Accuracy** | {self.teacher_metrics.accuracy * 100:.1f}% | {self.student_metrics.accuracy * 100:.1f}% | {(self.student_metrics.accuracy - self.teacher_metrics.accuracy) * 100:+.1f}% |",
            f"| **Avg Latency** | {self.teacher_metrics.avg_latency_ms:.2f} ms | {self.student_metrics.avg_latency_ms:.2f} ms | **{self.speedup_factor:.2f}x faster** |",
            f"| **Tokens / Sec** | {self.teacher_metrics.tokens_per_sec:.1f} | {self.student_metrics.tokens_per_sec:.1f} | **{self.student_metrics.tokens_per_sec / max(self.teacher_metrics.tokens_per_sec, 0.01):.2f}x** |",
            f"| **Active Parameters** | {self.teacher_metrics.param_count:,} | {self.student_metrics.param_count:,} | **{self.compression_ratio:.2f}x compression** |",
            f"| **Model Weight Size** | {self.teacher_metrics.model_size_mb:.2f} MB | {self.student_metrics.model_size_mb:.2f} MB | {(1 - self.student_metrics.model_size_mb / max(self.teacher_metrics.model_size_mb, 0.01)) * 100:.1f}% smaller |",
            f"| **Peak VRAM** | {self.teacher_metrics.peak_vram_mb:.2f} MB | {self.student_metrics.peak_vram_mb:.2f} MB | {self.vram_reduction_pct:.1f}% saved |",
            "",
            "#### Task Breakdown",
            f"- **Matched / Retained Tasks ({len(self.matched_tasks)}):** {', '.join(self.matched_tasks) or 'None'}",
            f"- **Regressed Tasks ({len(self.regressed_tasks)}):** {', '.join(self.regressed_tasks) or 'None'}",
            f"- **Improved / New Pass Tasks ({len(self.improved_tasks)}):** {', '.join(self.improved_tasks) or 'None'}",
        ]
        return "\n".join(lines)


class DistillationComparator:
    """
    Executes benchmark comparison between a teacher and a student model.
    """

    def __init__(self, benchmark_items: Optional[List[BenchmarkItem]] = None) -> None:
        self.benchmark_items = benchmark_items or self._default_benchmark()

    def _default_benchmark(self) -> List[BenchmarkItem]:
        """Built-in lightweight benchmark items."""
        return [
            BenchmarkItem(
                id="math_deriv_poly",
                prompt="Derivative of x**3 + 2*x",
                expected_output="3*x**2 + 2",
                task_category="math",
            ),
            BenchmarkItem(
                id="math_quad_solve",
                prompt="Roots of x**2 - 4",
                expected_output="-2, 2",
                task_category="math",
            ),
            BenchmarkItem(
                id="code_python_syntax",
                prompt="def add(a, b): return a + b",
                expected_output="valid",
                task_category="code",
            ),
            BenchmarkItem(
                id="code_cpp_balanced",
                prompt="int main() { return 0; }",
                expected_output="valid",
                task_category="code",
            ),
            BenchmarkItem(
                id="reasoning_kv_cache",
                prompt="What is KV caching in transformers?",
                expected_output="avoids recomputing attention keys and values",
                task_category="reasoning",
            ),
        ]

    def _count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def _model_size_mb(self, model: nn.Module) -> float:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_bytes + buffer_bytes) / (1024 ** 2)

    def evaluate_model(
        self,
        model_name: str,
        inference_fn: Callable[[str], str],
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ) -> ModelMetrics:
        """
        Evaluate a model on the benchmark suite.

        Args:
            model_name: Identifier string.
            inference_fn: Function mapping prompt -> output text.
            model: Optional PyTorch model to measure parameters and size.
            device: 'cpu' or 'cuda'.
        """
        results: Dict[str, bool] = {}
        latencies_ms: List[float] = []
        total_tokens_gen = 0
        total_gen_time = 0.0

        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for item in self.benchmark_items:
            t0 = time.perf_counter()
            output = inference_fn(item.prompt)
            duration = time.perf_counter() - t0

            latencies_ms.append(duration * 1000.0)
            tokens = len(output.split())
            total_tokens_gen += tokens
            total_gen_time += duration

            # Evaluation check
            if item.evaluator is not None:
                passed = item.evaluator(output, item.expected_output)
            else:
                passed = item.expected_output.lower() in output.lower()
            results[item.id] = passed

        passed_count = sum(1 for v in results.values() if v)
        accuracy = passed_count / len(self.benchmark_items) if self.benchmark_items else 0.0
        avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        tokens_per_sec = total_tokens_gen / max(total_gen_time, 1e-4)

        param_count = self._count_parameters(model) if model is not None else 0
        model_size_mb = self._model_size_mb(model) if model is not None else 0.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available() and device == "cuda"
            else 0.0
        )

        return ModelMetrics(
            name=model_name,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            tokens_per_sec=tokens_per_sec,
            param_count=param_count,
            model_size_mb=model_size_mb,
            peak_vram_mb=peak_vram_mb,
            per_task_results=results,
        )

    def compare(
        self,
        teacher_metrics: ModelMetrics,
        student_metrics: ModelMetrics,
    ) -> DistillationComparisonReport:
        """
        Compare teacher metrics and student metrics, analyzing retention and regression.
        """
        speedup = (
            teacher_metrics.avg_latency_ms / max(student_metrics.avg_latency_ms, 1e-4)
            if student_metrics.avg_latency_ms > 0
            else 1.0
        )
        compression = (
            teacher_metrics.param_count / max(student_metrics.param_count, 1)
            if student_metrics.param_count > 0
            else 1.0
        )
        vram_reduction = 0.0
        if teacher_metrics.peak_vram_mb > 0:
            vram_reduction = (
                (teacher_metrics.peak_vram_mb - student_metrics.peak_vram_mb)
                / teacher_metrics.peak_vram_mb
                * 100.0
            )

        matched: List[str] = []
        regressed: List[str] = []
        improved: List[str] = []

        all_tasks = set(teacher_metrics.per_task_results.keys()) | set(student_metrics.per_task_results.keys())
        for task in sorted(all_tasks):
            t_pass = teacher_metrics.per_task_results.get(task, False)
            s_pass = student_metrics.per_task_results.get(task, False)

            if t_pass and s_pass:
                matched.append(task)
            elif t_pass and not s_pass:
                regressed.append(task)
            elif not t_pass and s_pass:
                improved.append(task)

        return DistillationComparisonReport(
            teacher_metrics=teacher_metrics,
            student_metrics=student_metrics,
            speedup_factor=speedup,
            compression_ratio=compression,
            vram_reduction_pct=vram_reduction,
            matched_tasks=matched,
            regressed_tasks=regressed,
            improved_tasks=improved,
        )
