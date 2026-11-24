"""Configuration for custom Transformer from scratch (Phase 7).
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Hyperparameters for custom causal language model."""
    vocab_size: int = 1024
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 684           # 8/3 * hidden_dim for SwiGLU
    max_seq_len: int = 256
    dropout: float = 0.05
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        return self.hidden_dim // self.num_heads

    def estimate_parameter_count(self) -> int:
        """Calculates total trainable parameter count."""
        # Embeddings: vocab_size * hidden_dim
        embed_params = self.vocab_size * self.hidden_dim
        # Layer: Attn (4 * hidden_dim * hidden_dim) + SwiGLU (3 * hidden_dim * ffn_dim) + 2 * RMSNorm (2 * hidden_dim)
        attn_params = 4 * self.hidden_dim * self.hidden_dim
        swiglu_params = 3 * self.hidden_dim * self.ffn_dim
        norm_params = 2 * self.hidden_dim
        layer_params = attn_params + swiglu_params + norm_params
        # Total
        total = embed_params + (self.num_layers * layer_params) + self.hidden_dim  # Final norm
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_dim  # LM head
        return total
