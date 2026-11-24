"""
Unit Tests for Phase 8: Knowledge Distillation & Edge Models.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from personal_ai.training.distillation import (
    DistillationConfig,
    DistillationLoss,
    DistillationTrainer,
    TeacherSample,
    TeacherVerificationFilter,
)
from research.distillation.benchmark_comparator import (
    BenchmarkItem,
    DistillationComparator,
    ModelMetrics,
)
from research.scratch_transformer.architecture import ScratchCausalTransformer
from research.scratch_transformer.config import TransformerConfig


class DummyEdgeModel(nn.Module):
    """Minimal module for deterministic loss verification."""
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def test_distillation_loss_gradient_and_alpha():
    """Verify Hinton distillation loss computation and gradient isolation."""
    loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)

    batch_size, seq_len, vocab_size = 2, 4, 10
    student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=False)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss, metrics = loss_fn(student_logits, teacher_logits, labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
    assert "kd_loss" in metrics
    assert "ce_loss" in metrics
    assert "total_loss" in metrics

    # Backward pass
    loss.backward()
    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()


def test_distillation_loss_pure_soft_targets():
    """Verify alpha=0.0 produces purely soft KL loss scaled by T^2."""
    T = 3.0
    loss_fn = DistillationLoss(temperature=T, alpha=0.0)

    student_logits = torch.randn(2, 8, requires_grad=True)
    teacher_logits = torch.randn(2, 8, requires_grad=False)

    loss, metrics = loss_fn(student_logits, teacher_logits)
    assert metrics["ce_loss"] == 0.0
    assert abs(loss.item() - metrics["kd_loss"]) < 1e-5


def test_distillation_invalid_parameters():
    """Verify validation on temperature and alpha."""
    with pytest.raises(ValueError, match="Temperature must be positive"):
        DistillationLoss(temperature=-1.0)

    with pytest.raises(ValueError, match="Alpha must be in"):
        DistillationLoss(temperature=2.0, alpha=1.5)


def test_teacher_verification_filter_license():
    """Verify license gating on teacher samples."""
    v_filter = TeacherVerificationFilter()

    # Apache 2.0 should pass
    sample_valid = TeacherSample(
        prompt="Explain backprop",
        output="Backpropagation computes gradients using the chain rule.",
        task_type="general",
        teacher_model="qwen2.5-7b",
        license="Apache-2.0",
    )
    is_valid, msg = v_filter.verify_sample(sample_valid)
    assert is_valid, msg

    # Disallowed proprietary license should fail
    sample_invalid = TeacherSample(
        prompt="Explain backprop",
        output="Backpropagation computes gradients using the chain rule.",
        task_type="general",
        teacher_model="closed-source-ai",
        license="commercial-exclusive-no-distill",
    )
    is_valid, msg = v_filter.verify_sample(sample_invalid)
    assert not is_valid
    assert "not approved" in msg


def test_teacher_verification_filter_code():
    """Verify code syntax filtering for Python and C++ teacher outputs."""
    v_filter = TeacherVerificationFilter()

    # Valid Python code sample
    code_sample = TeacherSample(
        prompt="Write a factorial function",
        output="```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```",
        task_type="code",
        teacher_model="qwen2.5-7b",
        license="MIT",
    )
    is_valid, msg = v_filter.verify_sample(code_sample)
    assert is_valid, msg

    # Invalid Python syntax sample (syntax error)
    bad_code_sample = TeacherSample(
        prompt="Write a broken function",
        output="```python\ndef broken(\n    return 42\n```",
        task_type="code",
        teacher_model="qwen2.5-7b",
        license="MIT",
    )
    is_valid, msg = v_filter.verify_sample(bad_code_sample)
    assert not is_valid
    assert "syntax error" in msg.lower()


def test_teacher_verification_repetition_detector():
    """Verify rejection of degenerate repeating outputs."""
    v_filter = TeacherVerificationFilter()

    repetitive_text = "the loss decreases " * 50
    sample = TeacherSample(
        prompt="What happens during training?",
        output=repetitive_text,
        task_type="general",
        teacher_model="qwen2.5-7b",
        license="Apache-2.0",
    )
    is_valid, msg = v_filter.verify_sample(sample)
    assert not is_valid
    assert "degenerate repetition" in msg.lower()


def test_distillation_trainer_step():
    """Verify student model training step with teacher supervision."""
    vocab_size = 32
    student = DummyEdgeModel(vocab_size=vocab_size, hidden_dim=16)
    teacher = DummyEdgeModel(vocab_size=vocab_size, hidden_dim=32)

    config = DistillationConfig(
        temperature=2.0,
        alpha=0.5,
        learning_rate=1e-3,
        device="cpu",
    )
    trainer = DistillationTrainer(student_model=student, teacher_model=teacher, config=config)

    input_ids = torch.randint(0, vocab_size, (2, 8))
    labels = torch.randint(0, vocab_size, (2, 8))

    metrics = trainer.train_step(input_ids=input_ids, labels=labels)

    assert "total_loss" in metrics
    assert "kd_loss" in metrics
    assert "ce_loss" in metrics
    assert "tokens_per_sec" in metrics
    assert metrics["total_loss"] > 0.0


def test_distillation_benchmark_comparator():
    """Verify side-by-side benchmark comparison and markdown generation."""
    comparator = DistillationComparator()

    teacher_metrics = ModelMetrics(
        name="Teacher-7B",
        accuracy=1.0,
        avg_latency_ms=85.0,
        tokens_per_sec=25.0,
        param_count=7_000_000_000,
        model_size_mb=4200.0,
        peak_vram_mb=4500.0,
        per_task_results={
            "math_deriv_poly": True,
            "math_quad_solve": True,
            "code_python_syntax": True,
            "code_cpp_balanced": True,
            "reasoning_kv_cache": True,
        },
    )

    student_metrics = ModelMetrics(
        name="Student-Edge-0.5B",
        accuracy=0.8,
        avg_latency_ms=14.0,
        tokens_per_sec=140.0,
        param_count=500_000_000,
        model_size_mb=300.0,
        peak_vram_mb=800.0,
        per_task_results={
            "math_deriv_poly": True,
            "math_quad_solve": False,  # Regressed task
            "code_python_syntax": True,
            "code_cpp_balanced": True,
            "reasoning_kv_cache": True,
        },
    )

    report = comparator.compare(teacher_metrics, student_metrics)

    assert report.speedup_factor > 5.0
    assert report.compression_ratio == 14.0
    assert "math_quad_solve" in report.regressed_tasks
    assert len(report.matched_tasks) == 4

    md = report.to_markdown()
    assert "Teacher vs Student Distillation Benchmark Report" in md
    assert "faster" in md
    assert "math_quad_solve" in md
