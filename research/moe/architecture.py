"""
Sparse Mixture of Experts (MoE) Architecture (Phase 10).

Integrates:
- Top-K sparse routing with auxiliary load-balancing loss
- SwiGLU expert feed-forward networks
- DeepSeek-style Shared Expert for universal token representations
- RMSNorm and Rotary Position Embeddings (RoPE)
- Telemetry for total vs active parameters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.moe.router import RouterOutput, TopKRouter
from research.scratch_transformer.config import TransformerConfig
from research.scratch_transformer.architecture import (
    MultiHeadCausalAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig(TransformerConfig):
    """Configuration for Sparse MoE Transformer."""
    vocab_size: int = 1024
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 684
    num_experts: int = 4
    top_k: int = 2
    use_shared_expert: bool = True
    aux_loss_coef: float = 0.01
    max_seq_len: int = 256
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})")


class SparseMoEBlock(nn.Module):
    """
    Sparse MoE Feed-Forward Block with Top-K routing and optional Shared Expert.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.use_shared_expert = config.use_shared_expert

        # Router
        self.router = TopKRouter(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            aux_loss_coef=config.aux_loss_coef,
        )

        # Routed Experts
        self.experts = nn.ModuleList([
            SwiGLU(config) for _ in range(config.num_experts)
        ])

        # Optional Shared Expert (runs for all tokens, DeepSeek style)
        if config.use_shared_expert:
            self.shared_expert = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, RouterOutput]:
        """
        Forward pass through Sparse MoE block.

        Args:
            x: Tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            output: Tensor of shape (batch, seq_len, hidden_dim)
            router_out: RouterOutput containing aux_loss and telemetry
        """
        batch, seq_len, hidden_dim = x.shape
        flat_x = x.view(-1, hidden_dim)  # (N, hidden_dim)
        num_tokens = flat_x.size(0)

        router_out = self.router(flat_x)
        # router_out.expert_indices: (N, top_k)
        # router_out.routing_weights: (N, top_k)

        # Output accumulator
        final_output = torch.zeros_like(flat_x)

        # Route tokens to experts
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert across any top_k slot
            mask = (router_out.expert_indices == expert_idx)  # (N, top_k)
            token_mask = mask.any(dim=-1)  # (N,)

            if not token_mask.any():
                continue

            selected_tokens = flat_x[token_mask]  # (M, hidden_dim)
            expert_out = self.experts[expert_idx](selected_tokens)  # (M, hidden_dim)

            # Extract routing weight for this expert
            # Where mask is True, get corresponding weight
            weights = torch.where(mask, router_out.routing_weights, torch.zeros_like(router_out.routing_weights))
            token_weights = weights.sum(dim=-1)[token_mask].unsqueeze(-1)  # (M, 1)

            final_output[token_mask] += expert_out * token_weights

        # Add shared expert output if configured
        if self.use_shared_expert:
            shared_out = self.shared_expert(flat_x)
            final_output = final_output + shared_out

        output = final_output.view(batch, seq_len, hidden_dim)
        return output, router_out


class MoETransformerLayer(nn.Module):
    """
    A single Transformer layer featuring Attention and Sparse MoE FFN.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = MultiHeadCausalAttention(config)
        self.moe_norm = RMSNorm(config.hidden_dim)
        self.moe = SparseMoEBlock(config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, RouterOutput]:
        # Pre-LN Causal Attention
        normed_x = self.attn_norm(x)
        attn_out = self.attn(normed_x)
        x = x + attn_out

        # Pre-LN Sparse MoE FFN
        normed_moe = self.moe_norm(x)
        moe_out, router_out = self.moe(normed_moe)
        x = x + moe_out

        return x, router_out


class SparseMoETransformer(nn.Module):
    """
    Full Causal Decoder Transformer utilizing Sparse Mixture of Experts.
    """

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.layers = nn.ModuleList([
            MoETransformerLayer(config) for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Forward pass computing logits, cumulative auxiliary loss, and metrics.

        Returns:
            logits: (batch, seq_len, vocab_size)
            total_loss: scalar loss (ce_loss + aux_loss) or aux_loss
            metrics: dict with breakdown
        """
        x = self.token_embeddings(input_ids)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        avg_entropy = 0.0

        for layer in self.layers:
            x, router_out = layer(x)
            total_aux_loss = total_aux_loss + router_out.aux_loss
            avg_entropy += router_out.routing_entropy

        avg_entropy /= len(self.layers)
        x = self.norm(x)
        logits = self.lm_head(x)

        metrics = {
            "aux_loss": total_aux_loss.item(),
            "routing_entropy": avg_entropy,
        }

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = ce_loss + total_aux_loss
            metrics["ce_loss"] = ce_loss.item()
            metrics["total_loss"] = total_loss.item()
        else:
            total_loss = total_aux_loss

        return logits, total_loss, metrics

    def parameter_efficiency_report(self) -> Dict[str, Any]:
        """
        Compute total parameters vs active parameters per token.
        """
        total_params = sum(p.numel() for p in self.parameters())

        # Parameters active during a forward pass for a single token:
        # Embedding + Attention layers + RMSNorms + Top-k Experts + Shared Expert + Head
        base_params = (
            self.token_embeddings.weight.numel()
            + sum(
                sum(p.numel() for p in layer.attn.parameters())
                + sum(p.numel() for p in layer.attn_norm.parameters())
                + sum(p.numel() for p in layer.moe_norm.parameters())
                + sum(p.numel() for p in layer.moe.router.parameters())
                for layer in self.layers
            )
            + sum(p.numel() for p in self.norm.parameters())
        )

        single_expert_params = sum(p.numel() for p in self.layers[0].moe.experts[0].parameters())
        shared_expert_params = (
            sum(p.numel() for p in self.layers[0].moe.shared_expert.parameters())
            if self.config.use_shared_expert
            else 0
        )

        active_expert_params_per_layer = (self.config.top_k * single_expert_params) + shared_expert_params
        active_params = base_params + (active_expert_params_per_layer * self.config.num_layers)

        sparsity_ratio = (total_params - active_params) / max(total_params, 1)

        return {
            "total_parameters": total_params,
            "active_parameters": active_params,
            "sparsity_ratio": sparsity_ratio,
            "top_k": self.config.top_k,
            "num_experts": self.config.num_experts,
            "use_shared_expert": self.config.use_shared_expert,
        }
