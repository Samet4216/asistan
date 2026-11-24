"""
Integration Test for Phase 10: Sparse Mixture of Experts (MoE) Pretraining & Benchmark.
"""

from __future__ import annotations

import pytest
import torch

from research.moe.architecture import MoEConfig, SparseMoETransformer
from research.moe.comparator import MoEComparator
from research.scratch_transformer.architecture import ScratchCausalTransformer
from research.scratch_transformer.config import TransformerConfig


def test_moe_pretraining_and_collapse_defense():
    """
    Simulate full MoE pretraining step:
    1. Verify load balancing loss backpropagation.
    2. Check that routing entropy does not collapse.
    3. Benchmark against dense baseline on CUDA (if available) or CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64
    hidden_dim = 64

    # MoE Config
    moe_cfg = MoEConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=2,
        ffn_dim=128,
        num_experts=4,
        top_k=2,
        use_shared_expert=True,
        aux_loss_coef=0.02,
        max_seq_len=32,
    )
    moe_model = SparseMoETransformer(moe_cfg).to(device)
    optimizer = torch.optim.AdamW(moe_model.parameters(), lr=1e-3)

    # Synthetic batch: 2 sequences of 16 tokens
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    losses = []
    entropies = []

    # Run 4 training steps
    for step in range(4):
        optimizer.zero_grad()
        logits, total_loss, metrics = moe_model(input_ids, labels=labels)

        assert "ce_loss" in metrics
        assert "aux_loss" in metrics
        assert metrics["aux_loss"] > 0.0

        losses.append(metrics["total_loss"])
        entropies.append(metrics["routing_entropy"])

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(moe_model.parameters(), max_norm=1.0)
        optimizer.step()

    # Routing entropy must be healthy (well above collapse threshold of 0)
    assert all(e > 0.3 for e in entropies)

    # 4. Dense vs MoE Benchmark Comparison
    dense_cfg = TransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=2,
        ffn_dim=128,
        max_seq_len=32,
    )
    dense_model = ScratchCausalTransformer(dense_cfg).to(device)

    benchmark_res = MoEComparator.benchmark(
        dense_model=dense_model,
        moe_model=moe_model,
        batch_size=batch_size,
        seq_len=seq_len,
        num_runs=3,
        device=device,
    )

    assert benchmark_res.capacity_gain > 1.0
    assert not benchmark_res.collapsed
    md_report = benchmark_res.to_markdown()
    assert "Dense Baseline vs Sparse Mixture of Experts" in md_report
