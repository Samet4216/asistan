"""
Integration test for Phase 8: End-to-end Knowledge Distillation Pipeline.
"""

from __future__ import annotations

import torch
import pytest

from personal_ai.training.distillation import (
    DistillationConfig,
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


def test_end_to_end_distillation_and_comparison():
    """
    Simulate full distillation cycle:
    1. Filter & verify teacher dataset samples.
    2. Setup Teacher model (e.g. 4 layers, dim=128) and Edge Student model (2 layers, dim=64).
    3. Run distillation training loop with loss logging and VRAM checks.
    4. Execute comparative benchmark evaluation.
    """
    # Step 1: Verification Filter
    v_filter = TeacherVerificationFilter()
    sample = TeacherSample(
        prompt="Solve x + 5 = 12",
        output="The solution is x = 7, obtained by subtracting 5 from both sides.",
        task_type="math",
        teacher_model="qwen2.5-7b-instruct",
        license="Apache-2.0",
    )
    is_valid, msg = v_filter.verify_sample(sample)
    assert is_valid, msg

    # Step 2: Architecture setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64

    # Teacher: 4 layers, dim=128, 4 heads
    teacher_cfg = TransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=64,
    )
    teacher = ScratchCausalTransformer(teacher_cfg).to(device)

    # Edge Student: 2 layers, dim=64, 2 heads
    student_cfg = TransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=64,
    )
    student = ScratchCausalTransformer(student_cfg).to(device)

    # Step 3: Run Distillation Trainer
    distill_cfg = DistillationConfig(
        temperature=2.0,
        alpha=0.5,
        learning_rate=1e-3,
        batch_size=1,
        max_vram_mb=5000.0,
        device=device,
    )
    trainer = DistillationTrainer(student_model=student, teacher_model=teacher, config=distill_cfg)

    # Train for 3 steps
    dummy_input = torch.randint(0, vocab_size, (1, 16), device=device)
    dummy_labels = torch.randint(0, vocab_size, (1, 16), device=device)

    losses = []
    for _ in range(3):
        metrics = trainer.train_step(input_ids=dummy_input, labels=dummy_labels)
        losses.append(metrics["total_loss"])

    assert len(losses) == 3
    assert all(l > 0 for l in losses)

    # Step 4: Side-by-side benchmark comparison
    comparator = DistillationComparator()

    t_metrics = comparator.evaluate_model(
        model_name="Teacher-Transformer-4L",
        inference_fn=lambda prompt: "derivative is 3*x**2 + 2",
        model=teacher,
        device=device,
    )

    s_metrics = comparator.evaluate_model(
        model_name="Student-Edge-2L",
        inference_fn=lambda prompt: "derivative is 3*x**2 + 2",
        model=student,
        device=device,
    )

    report = comparator.compare(t_metrics, s_metrics)

    # Validate metrics
    assert s_metrics.param_count < t_metrics.param_count
    assert report.compression_ratio > 1.0
    md = report.to_markdown()
    assert "Teacher vs Student Distillation Benchmark Report" in md
    assert "Model Weight Size" in md
