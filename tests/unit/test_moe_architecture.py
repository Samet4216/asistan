"""
Unit Tests for Phase 10: Sparse Mixture of Experts (MoE) Architecture and Router.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from research.moe.architecture import (
    MoEConfig,
    SparseMoEBlock,
    SparseMoETransformer,
)
from research.moe.comparator import MoEComparator
from research.moe.router import TopKRouter
from research.scratch_transformer.architecture import ScratchCausalTransformer
from research.scratch_transformer.config import TransformerConfig


def test_topk_router_output_shapes_and_weights():
    """Verify router selects top_k experts and produces normalized weights."""
    hidden_dim = 32
    num_experts = 4
    top_k = 2
    router = TopKRouter(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)

    batch_tokens = 10
    x = torch.randn(batch_tokens, hidden_dim)

    out = router(x)

    assert out.expert_indices.shape == (batch_tokens, top_k)
    assert out.routing_weights.shape == (batch_tokens, top_k)
    # Weights should sum to 1.0 per token
    weight_sums = out.routing_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4)

    assert out.aux_loss.item() >= 0.0
    assert 0.0 <= out.routing_entropy <= 1.0
    assert len(out.expert_usage) == num_experts


def test_topk_router_invalid_top_k():
    """Verify error when top_k exceeds num_experts."""
    with pytest.raises(ValueError, match="cannot exceed num_experts"):
        TopKRouter(hidden_dim=32, num_experts=3, top_k=4)


def test_sparse_moe_block_forward_and_backward():
    """Verify forward and backward passes through SparseMoEBlock."""
    cfg = MoEConfig(
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        num_experts=4,
        top_k=2,
        use_shared_expert=True,
    )
    moe_block = SparseMoEBlock(cfg)

    x = torch.randn(2, 8, cfg.hidden_dim, requires_grad=True)
    out, router_out = moe_block(x)

    assert out.shape == x.shape
    assert router_out.aux_loss.requires_grad

    # Test backward pass
    loss = out.sum() + router_out.aux_loss
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_sparse_moe_transformer_parameter_efficiency():
    """Verify total parameters vs active parameters per token."""
    cfg = MoEConfig(
        vocab_size=64,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        num_experts=4,
        top_k=1,  # Only 1 expert active at a time
        use_shared_expert=False,
        max_seq_len=32,
    )
    moe_model = SparseMoETransformer(cfg)
    report = moe_model.parameter_efficiency_report()

    assert report["total_parameters"] > report["active_parameters"]
    assert report["sparsity_ratio"] > 0.0


def test_sparse_moe_transformer_training_step():
    """Verify forward and optimization step of SparseMoETransformer."""
    cfg = MoEConfig(
        vocab_size=32,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        num_experts=4,
        top_k=2,
        use_shared_expert=True,
        max_seq_len=16,
    )
    model = SparseMoETransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    logits, loss, metrics = model(tokens, labels=labels)

    assert logits.shape == (2, 8, cfg.vocab_size)
    assert "ce_loss" in metrics
    assert "aux_loss" in metrics
    assert "total_loss" in metrics

    loss.backward()
    optimizer.step()


def test_moe_comparator_benchmark():
    """Verify benchmark execution and report generation between Dense and MoE."""
    vocab_size = 32
    hidden_dim = 32

    dense_cfg = TransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=2,
        max_seq_len=16,
    )
    dense_model = ScratchCausalTransformer(dense_cfg)

    moe_cfg = MoEConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=2,
        num_experts=4,
        top_k=1,
        use_shared_expert=True,
        max_seq_len=16,
    )
    moe_model = SparseMoETransformer(moe_cfg)

    result = MoEComparator.benchmark(dense_model, moe_model, batch_size=1, seq_len=8, num_runs=2)

    assert result.capacity_gain > 1.0  # MoE has more total capacity
    assert result.compute_efficiency < 1.0  # Active compute is fraction of total

    md = result.to_markdown()
    assert "Dense Baseline vs Sparse Mixture of Experts" in md
    assert "Normalized Routing Entropy" in md
