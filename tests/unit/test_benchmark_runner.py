"""Unit tests for BenchmarkRunner (Phase 5).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from personal_ai.evaluation.benchmark_runner import BenchmarkItem, BenchmarkRunner
from personal_ai.runtime.ollama_client import OllamaClient


def test_grade_response_perfect():
    """Verify high score when all rubric keywords and LaTeX math are present."""
    runner = BenchmarkRunner()
    item = BenchmarkItem(
        id="test_01",
        category="architectural_analysis",
        prompt="Explain skip connections",
        rubric_keywords=["gradient", "identity", "F(x) + x"],
        negative_keywords=["Bölüm 1/N"],
    )

    response = "Residual connections define $y = F(x) + x$, allowing identity gradient flow directly."
    grade = runner.grade_response(item, response, latency=1.2)

    assert grade.rubric_score == 1.0
    assert grade.math_detected is True
    assert grade.negative_penalties == 0
    assert grade.passed is True


def test_grade_response_negative_boilerplate_penalty():
    """Verify penalty is applied when forbidden boilerplate (Bölüm 1/N) is detected."""
    runner = BenchmarkRunner()
    item = BenchmarkItem(
        id="test_02",
        category="architectural_analysis",
        prompt="Explain skip connections",
        rubric_keywords=["gradient", "identity", "F(x) + x"],
        negative_keywords=["Bölüm 1/N"],
    )

    response = "🎯 Bölüm 1/N: Yol Haritası. Residual connections define $F(x) + x$ with identity gradient."
    grade = runner.grade_response(item, response, latency=1.0)

    # Negative keyword should trigger penalty and fail the test
    assert grade.negative_penalties == 1
    assert grade.passed is False
    assert grade.rubric_score < 1.0


def test_run_benchmark_mocked(tmp_path: Path):
    """Verify benchmark run produces correct summary JSON structure."""
    bench_file = tmp_path / "bench.jsonl"
    bench_file.write_text(
        '{"id": "q1", "category": "dl", "prompt": "What is AdamW?", "rubric_keywords": ["adamw", "weight decay"], "negative_keywords": []}\n',
        encoding="utf-8"
    )

    runner = BenchmarkRunner(benchmark_path=bench_file)
    mock_client = MagicMock(spec=OllamaClient)
    mock_client.default_model = "qwen2.5:7b-instruct-q4_K_M"
    mock_client.chat.return_value = "AdamW decouples weight decay from gradient updates."

    report_path = tmp_path / "report.json"
    summary = runner.run_benchmark(
        client=mock_client,
        output_report_path=report_path,
    )

    assert summary["total_questions"] == 1
    assert summary["passed_count"] == 1
    assert summary["pass_rate"] == "100.0%"
    assert report_path.exists()
