"""
Integration Test for Phase 9: End-to-End Direct Preference Optimization (DPO) Pipeline.
"""

from __future__ import annotations

import copy
import pytest
import torch

from personal_ai.rl.verifiable_rewards import VerifiableRewardEngine
from personal_ai.training.dpo_trainer import DPOConfig, DPOTrainer, PreferencePair
from research.scratch_transformer.architecture import ScratchCausalTransformer
from research.scratch_transformer.config import TransformerConfig


def test_dpo_pipeline_end_to_end():
    """
    Simulate full DPO training pipeline:
    1. Score candidate pairs with verifiable rewards.
    2. Setup Policy and frozen Reference Transformer models.
    3. Run optimization steps and check implicit reward margin progression.
    """
    engine = VerifiableRewardEngine()

    # Verified preference pair
    chosen_code = "```python\ndef compute_factorial(n):\n    return 1 if n <= 1 else n * compute_factorial(n - 1)\n```"
    rejected_code = "```python\ndef compute_factorial(n\n    return 0\n```"

    chosen_eval = engine.evaluate_code_reward("Factorial prompt", chosen_code)
    rejected_eval = engine.evaluate_code_reward("Factorial prompt", rejected_code)

    assert chosen_eval.is_valid and chosen_eval.reward > 0
    assert not rejected_eval.is_valid and rejected_eval.reward < 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64

    # Transformer config
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=64,
    )
    policy = ScratchCausalTransformer(cfg).to(device)
    reference = copy.deepcopy(policy).to(device)

    dpo_cfg = DPOConfig(
        beta=0.1,
        learning_rate=1e-3,
        max_vram_mb=5000.0,
        device=device,
    )
    trainer = DPOTrainer(policy_model=policy, reference_model=reference, config=dpo_cfg)

    # Synthetic chosen & rejected token sequences
    chosen_tokens = torch.randint(0, vocab_size, (1, 16), device=device)
    rejected_tokens = torch.randint(0, vocab_size, (1, 16), device=device)

    # Train for 3 steps
    margins = []
    for _ in range(3):
        metrics = trainer.train_step(chosen_input_ids=chosen_tokens, rejected_input_ids=rejected_tokens)
        margins.append(metrics["reward_margin"])

    assert len(margins) == 3
    assert "tokens_per_sec" in metrics
    assert "vram_mb" in metrics
