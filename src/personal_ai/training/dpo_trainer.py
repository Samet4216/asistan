"""
Direct Preference Optimization (DPO) Training Engine (Phase 9).

Implements Rafailov et al. (2023) Direct Preference Optimization:
L_DPO = - E_{(x, y_w, y_l)} [ log sigma( beta * ( log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x) ) ) ]

Eliminates the instability of training a separate reward model by directly
optimizing policy parameters on preference pairs under a KL divergence constraint.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Hyperparameters for Direct Preference Optimization."""
    beta: float = 0.1  # KL divergence penalty weight
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_vram_mb: float = 5000.0  # 6GB VRAM safety threshold
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"DPO beta must be strictly positive, got {self.beta}")


@dataclass
class PreferencePair:
    """A verified pairwise preference sample."""
    prompt: str
    chosen: str
    rejected: str
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Computes per-sample sum of log probabilities for label tokens.

    Args:
        logits: Tensor of shape (batch, seq_len, vocab_size)
        labels: Tensor of shape (batch, seq_len) with target token IDs

    Returns:
        Tensor of shape (batch,) containing sum of log probabilities
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Log-softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log prob of true labels
    mask = shift_labels != ignore_index
    # Replace ignore_index with 0 for valid gathering, then zero-out via mask
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    per_token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    # Apply mask and sum over sequence dimension
    per_sample_log_probs = (per_token_log_probs * mask.float()).sum(dim=-1)
    return per_sample_log_probs


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization Loss.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"Beta must be > 0, got {beta}")
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss and telemetry metrics.
        """
        # Log ratios
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps.detach() - reference_rejected_logps.detach()

        # Logit difference (implicit reward margin)
        logits = self.beta * (pi_logratios - ref_logratios)

        # DPO loss = - log sigma(logits)
        loss = -F.logsigmoid(logits).mean()

        # Implicit rewards for telemetry
        chosen_rewards = (self.beta * (policy_chosen_logps - reference_chosen_logps.detach())).detach()
        rejected_rewards = (self.beta * (policy_rejected_logps - reference_rejected_logps.detach())).detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": reward_margin,
            "accuracy": accuracy,
        }
        return loss, metrics


class DPOTrainer:
    """
    Executes DPO training on policy model while maintaining frozen reference model.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: Optional[DPOConfig] = None,
    ) -> None:
        self.config = config or DPOConfig()
        self.policy = policy_model.to(self.config.device)
        self.reference = reference_model.to(self.config.device)
        self.reference.eval()  # Freeze reference model
        for param in self.reference.parameters():
            param.requires_grad = False

        self.loss_fn = DPOLoss(beta=self.config.beta)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _check_vram(self) -> float:
        if torch.cuda.is_available() and self.config.device == "cuda":
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            if allocated_mb > self.config.max_vram_mb:
                raise RuntimeError(
                    f"VRAM budget exceeded during DPO training! Allocated: {allocated_mb:.2f} MB, Limit: {self.config.max_vram_mb} MB"
                )
            return allocated_mb
        return 0.0

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute a single DPO optimization step.
        """
        self.policy.train()
        self.optimizer.zero_grad()

        chosen_input_ids = chosen_input_ids.to(self.config.device)
        rejected_input_ids = rejected_input_ids.to(self.config.device)

        start_time = time.perf_counter()

        # Policy forward pass
        policy_chosen_out = self.policy(chosen_input_ids)
        policy_chosen_logits = policy_chosen_out.logits if hasattr(policy_chosen_out, "logits") else policy_chosen_out
        policy_chosen_logps = compute_log_probs_from_logits(policy_chosen_logits, chosen_input_ids)

        policy_rejected_out = self.policy(rejected_input_ids)
        policy_rejected_logits = policy_rejected_out.logits if hasattr(policy_rejected_out, "logits") else policy_rejected_out
        policy_rejected_logps = compute_log_probs_from_logits(policy_rejected_logits, rejected_input_ids)

        # Frozen reference forward pass (no grads)
        with torch.no_grad():
            ref_chosen_out = self.reference(chosen_input_ids)
            ref_chosen_logits = ref_chosen_out.logits if hasattr(ref_chosen_out, "logits") else ref_chosen_out
            ref_chosen_logps = compute_log_probs_from_logits(ref_chosen_logits, chosen_input_ids)

            ref_rejected_out = self.reference(rejected_input_ids)
            ref_rejected_logits = ref_rejected_out.logits if hasattr(ref_rejected_out, "logits") else ref_rejected_out
            ref_rejected_logps = compute_log_probs_from_logits(ref_rejected_logits, rejected_input_ids)

        # Compute DPO loss
        loss, metrics = self.loss_fn(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        elapsed = time.perf_counter() - start_time
        tokens = chosen_input_ids.numel() + rejected_input_ids.numel()
        tokens_per_sec = tokens / max(elapsed, 1e-4)

        metrics["tokens_per_sec"] = tokens_per_sec
        metrics["vram_mb"] = self._check_vram()
        return metrics
