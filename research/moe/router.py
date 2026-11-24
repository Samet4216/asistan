"""
Top-K Gating Router with Load Balancing and Anti-Collapse Loss (Phase 10).

Implements Shazeer et al. (2017) and Fedus et al. (2021) Switch Transformer
routing with auxiliary load-balancing loss and routing entropy analysis.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RouterOutput:
    """Telemetry and routing tensors returned by TopKRouter."""
    expert_indices: torch.Tensor  # (batch * seq_len, top_k)
    routing_weights: torch.Tensor  # (batch * seq_len, top_k)
    aux_loss: torch.Tensor  # Scalar auxiliary load balancing loss
    routing_entropy: float  # Shannon entropy of expert allocation
    expert_usage: Dict[int, float]  # Fraction of tokens routed to each expert
    is_collapsed: bool  # True if routing collapsed to a single expert


class TopKRouter(nn.Module):
    """
    Top-k Gating Router with auxiliary load-balancing loss.

    Routes tokens to the top-k experts and computes Shazeer auxiliary loss:
    L_aux = alpha * num_experts * sum_i (f_i * P_i)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        noisy_gating: bool = True,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.noisy_gating = noisy_gating

        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        if noisy_gating:
            self.noise_linear = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """
        Route tokens to top-k experts.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim) or (total_tokens, hidden_dim)

        Returns:
            RouterOutput with indices, weights, aux_loss, entropy, and telemetry.
        """
        orig_shape = x.shape
        flat_x = x.view(-1, self.hidden_dim)  # (N, hidden_dim)
        num_tokens = flat_x.size(0)

        # 1. Compute routing logits
        logits = self.gate(flat_x)  # (N, num_experts)

        if self.noisy_gating and self.training:
            raw_noise_std = self.noise_linear(flat_x)
            noise_std = F.softplus(raw_noise_std)
            eps = torch.randn_like(logits)
            logits = logits + eps * noise_std

        # Softmax probabilities over all experts (used for routing weights and aux loss)
        gate_probs = F.softmax(logits, dim=-1)  # (N, num_experts)

        # 2. Select Top-K experts
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)  # (N, top_k)
        # Re-normalize weights among top-k so they sum to 1.0 per token
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Compute Load Balancing Auxiliary Loss (f_i * P_i)
        # f_i: fraction of tokens routed to expert i
        # P_i: average probability of expert i across all tokens
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)
        for k in range(self.top_k):
            tokens_per_expert.scatter_add_(
                0, topk_indices[:, k], torch.ones(num_tokens, device=x.device, dtype=torch.float32)
            )

        # Fraction of total routing slots (num_tokens * top_k)
        total_slots = num_tokens * self.top_k
        f = tokens_per_expert / max(total_slots, 1)  # (num_experts,)
        P = gate_probs.mean(dim=0)  # (num_experts,)

        aux_loss = self.aux_loss_coef * self.num_experts * torch.sum(f * P)

        # 4. Measure Routing Entropy and Collapse Diagnostics
        # Shannon entropy of f: H(f) = - sum f_i log(f_i)
        safe_f = f.clamp(min=1e-8)
        entropy = -torch.sum(safe_f * torch.log(safe_f)).item()
        max_possible_entropy = math.log(self.num_experts) if self.num_experts > 1 else 1.0
        normalized_entropy = entropy / max(max_possible_entropy, 1e-4)

        expert_usage = {i: float(f[i].item()) for i in range(self.num_experts)}
        # Collapse flagged if one expert handles > 85% of traffic when num_experts > 2
        is_collapsed = (max(expert_usage.values()) > 0.85) if self.num_experts >= 3 else False

        return RouterOutput(
            expert_indices=topk_indices,
            routing_weights=topk_weights,
            aux_loss=aux_loss,
            routing_entropy=normalized_entropy,
            expert_usage=expert_usage,
            is_collapsed=is_collapsed,
        )
