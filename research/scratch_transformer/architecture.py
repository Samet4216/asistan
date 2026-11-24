"""Pure PyTorch Causal Transformer from Scratch with Modern Architecture (Phase 7).

Includes:
- RMSNorm (Root Mean Square Layer Normalization)
- Rotary Position Embeddings (RoPE)
- Pre-LN Multi-Head Causal Self-Attention
- SwiGLU (Swish Gated Linear Unit) Feed-Forward Network
- Weight-tied Causal Language Model Head
- Autoregressive Generation with Temperature and Top-k Sampling
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.scratch_transformer.config import TransformerConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (standard in modern LLMs like LLaMA / Qwen)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class RotaryEmbedding(nn.Module):
    """Applies Rotary Position Embeddings (RoPE) to Query and Key tensors."""
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        # Compute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate sin and cos for rotary application
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates query or key states with precomputed cosine and sine."""
    # x shape: [batch, num_heads, seq_len, head_dim]
    # cos/sin shape: [seq_len, head_dim] -> unsqueeze to [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadCausalAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and Pre-LN."""
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.dropout_p = cfg.dropout

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=cfg.max_seq_len)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(seq_len, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Scaled dot-product attention with causal mask
        # Shapes: [batch, num_heads, seq_len, head_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (prevent attending to future tokens)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)

        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout_p > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.o_proj(context)


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit feed-forward layer."""
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block combining Causal Attention and SwiGLU."""
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_dim)
        self.attn = MultiHeadCausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_dim)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ScratchCausalTransformer(nn.Module):
    """Complete Causal Language Model constructed from scratch in PyTorch."""
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        # Weight tying: share embedding weights with LM output projection
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using standard Gaussian initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass computing next-token logits.

        Args:
            input_ids: LongTensor of shape [batch_size, seq_len]
        Returns:
            logits: FloatTensor of shape [batch_size, seq_len, vocab_size]
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 16,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> torch.Tensor:
        """Autoregressive generation loop with temperature and top-k filtering."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = (
                input_ids
                if input_ids.shape[1] <= self.cfg.max_seq_len
                else input_ids[:, -self.cfg.max_seq_len :]
            )
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

        return input_ids
