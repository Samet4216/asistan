"""Integration test for Scratch Transformer pretraining loop (Phase 7).
"""

import pytest

from research.scratch_transformer.config import TransformerConfig
from research.scratch_transformer.trainer import ScratchTrainer


def test_scratch_pretraining_smoke_loop():
    """Verify pretraining steps, loss computation, tokens/sec, and GPU/CPU execution."""
    cfg = TransformerConfig(
        vocab_size=128,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        ffn_dim=170,
        max_seq_len=32,
    )

    trainer = ScratchTrainer(config=cfg)
    metrics = trainer.train_steps(steps=8, batch_size=2, seq_len=16, learning_rate=2e-3)

    assert metrics.steps == 8
    assert metrics.total_tokens_processed == 8 * 2 * 16  # 256 tokens
    assert metrics.tokens_per_second > 0.0
    assert metrics.initial_loss > 0.0
    assert metrics.final_loss > 0.0
    assert metrics.elapsed_seconds > 0.0
