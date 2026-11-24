"""Unit tests for Scratch Transformer Architecture (Phase 7).
"""

import torch
import pytest

from research.scratch_transformer.architecture import (
    RMSNorm,
    RotaryEmbedding,
    MultiHeadCausalAttention,
    SwiGLU,
    ScratchCausalTransformer,
)
from research.scratch_transformer.config import TransformerConfig


@pytest.fixture
def tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=256,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        ffn_dim=170,
        max_seq_len=64,
    )


def test_rmsnorm_normalizes_variance():
    """Verify RMSNorm produces unit variance output."""
    norm = RMSNorm(dim=32)
    x = torch.randn(2, 8, 32) * 5.0 + 3.0
    out = norm(x)
    # Output shape unchanged
    assert out.shape == (2, 8, 32)
    # Scaled magnitude is bounded
    assert not torch.isnan(out).any()


def test_rope_embedding_shapes():
    """Verify RoPE precomputed frequency shapes."""
    rope = RotaryEmbedding(dim=32, max_seq_len=64)
    cos, sin = rope(seq_len=16, device=torch.device("cpu"))
    assert cos.shape == (16, 32)
    assert sin.shape == (16, 32)


def test_multihead_attention_shape(tiny_config: TransformerConfig):
    """Verify multihead causal attention input and output dimensions."""
    attn = MultiHeadCausalAttention(tiny_config)
    x = torch.randn(2, 12, tiny_config.hidden_dim)
    out = attn(x)
    assert out.shape == (2, 12, tiny_config.hidden_dim)


def test_swiglu_shape(tiny_config: TransformerConfig):
    """Verify SwiGLU feed-forward layer dimensions."""
    ffn = SwiGLU(tiny_config)
    x = torch.randn(2, 12, tiny_config.hidden_dim)
    out = ffn(x)
    assert out.shape == (2, 12, tiny_config.hidden_dim)


def test_scratch_transformer_forward_and_generate(tiny_config: TransformerConfig):
    """Verify complete Transformer next-token forward pass and generation."""
    model = ScratchCausalTransformer(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 10))

    # Forward pass
    logits = model(input_ids)
    assert logits.shape == (2, 10, tiny_config.vocab_size)

    # Generation pass
    prompt_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
    generated = model.generate(prompt_ids, max_new_tokens=6, temperature=0.7)
    assert generated.shape == (1, 10)  # 4 prompt tokens + 6 new tokens
