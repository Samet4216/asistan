"""
Unit Tests for Phase 9: Direct Preference Optimization (DPO) and Verifiable Rewards.
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn

from personal_ai.rl.verifiable_rewards import (
    RewardHackingDetector,
    VerifiableRewardEngine,
)
from personal_ai.training.dpo_trainer import (
    DPOConfig,
    DPOLoss,
    DPOTrainer,
    compute_log_probs_from_logits,
)


class DummyPolicyModel(nn.Module):
    """Simple model for deterministic DPO testing."""
    def __init__(self, vocab_size: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def test_verifiable_rewards_code_syntax():
    """Verify code reward scoring for valid vs broken code."""
    engine = VerifiableRewardEngine()

    valid_code = "```python\ndef square(x):\n    return x * x\n```"
    eval_res = engine.evaluate_code_reward("Write square function", valid_code)
    assert eval_res.is_valid
    assert eval_res.reward > 0.0
    assert not eval_res.hacked_detected

    broken_code = "```python\ndef broken(\n    return 0\n```"
    eval_broken = engine.evaluate_code_reward("Write broken function", broken_code)
    assert not eval_broken.is_valid
    assert eval_broken.reward == -1.0


def test_reward_hacking_evasion_and_cheat_detection():
    """Verify detection of robotic evasion, boilerplate and trivial cheats."""
    engine = VerifiableRewardEngine()

    # Robotic evasion
    evasion_text = "As an AI language model, I cannot verify this code."
    eval_evasion = engine.evaluate_code_reward("Test prompt", evasion_text)
    assert eval_evasion.hacked_detected
    assert eval_evasion.reward == -1.0

    # Trivial cheat (only pass)
    cheat_code = "```python\n# empty\npass\n```"
    eval_cheat = engine.evaluate_code_reward("Write algorithm", cheat_code)
    assert eval_cheat.hacked_detected
    assert eval_cheat.reward < 0.0


def test_verifiable_rewards_math():
    """Verify deterministic math reward checking."""
    engine = VerifiableRewardEngine()

    eval_correct = engine.evaluate_math_reward("x = 42", "The computed solution is x = 42.")
    assert eval_correct.is_valid
    assert eval_correct.reward == 1.0

    eval_incorrect = engine.evaluate_math_reward("x = 42", "The answer is x = 99.")
    assert not eval_incorrect.is_valid
    assert eval_incorrect.reward == -1.0


def test_dpo_loss_and_gradients():
    """Verify DPO loss mathematical properties and backpropagation."""
    loss_fn = DPOLoss(beta=0.1)

    batch_size = 2
    policy_chosen = torch.tensor([-1.0, -1.5], requires_grad=True)
    policy_rejected = torch.tensor([-2.5, -3.0], requires_grad=True)

    ref_chosen = torch.tensor([-1.2, -1.6], requires_grad=False)
    ref_rejected = torch.tensor([-2.0, -2.8], requires_grad=False)

    loss, metrics = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0
    assert "reward_margin" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0  # Chosen has higher margin than rejected

    loss.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None


def test_dpo_loss_invalid_beta():
    """Verify strictly positive constraint on beta."""
    with pytest.raises(ValueError, match="Beta must be > 0"):
        DPOLoss(beta=-0.1)


def test_compute_log_probs_from_logits():
    """Verify sequence log-probability extraction."""
    batch, seq_len, vocab_size = 2, 4, 8
    logits = torch.randn(batch, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    log_probs = compute_log_probs_from_logits(logits, labels)
    assert log_probs.shape == (batch,)
    assert (log_probs <= 0).all()  # Probabilities <= 1.0 -> log <= 0


def test_dpo_trainer_step():
    """Verify full DPO policy optimization step."""
    vocab_size = 16
    policy = DummyPolicyModel(vocab_size=vocab_size, hidden_dim=32)
    reference = copy.deepcopy(policy)

    initial_ref_weight = reference.head.weight.clone()

    config = DPOConfig(beta=0.1, learning_rate=1e-3, device="cpu")
    trainer = DPOTrainer(policy_model=policy, reference_model=reference, config=config)

    chosen_ids = torch.randint(0, vocab_size, (2, 8))
    rejected_ids = torch.randint(0, vocab_size, (2, 8))

    metrics = trainer.train_step(chosen_input_ids=chosen_ids, rejected_input_ids=rejected_ids)

    assert "loss" in metrics
    assert "reward_margin" in metrics
    assert "tokens_per_sec" in metrics

    # Reference weights must remain completely unchanged
    assert torch.equal(reference.head.weight, initial_ref_weight)
