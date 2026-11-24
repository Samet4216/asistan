"""Pre-training Engine for Scratch Transformer (Phase 7).

Trains custom ScratchCausalTransformer on next-token prediction,
measuring loss reduction, tokens per second, and VRAM memory footprint.
"""

from dataclasses import dataclass
import time
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

from research.scratch_transformer.architecture import ScratchCausalTransformer
from research.scratch_transformer.config import TransformerConfig


@dataclass
class PretrainMetrics:
    """Stores performance and convergence metrics of a pretraining run."""
    steps: int
    initial_loss: float
    final_loss: float
    total_tokens_processed: int
    tokens_per_second: float
    peak_vram_mb: float
    elapsed_seconds: float


class ScratchTrainer:
    """Orchestrates pretraining on synthetic or domain text tokens."""

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScratchCausalTransformer(self.config).to(self.device)

    def train_steps(
        self,
        steps: int = 15,
        batch_size: int = 2,
        seq_len: int = 32,
        learning_rate: float = 1e-3,
    ) -> PretrainMetrics:
        """Runs pretraining for a fixed number of steps on synthetic next-token targets."""
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        initial_loss = 0.0
        final_loss = 0.0
        total_tokens = steps * batch_size * seq_len

        t0 = time.time()

        for step in range(steps):
            # Generate synthetic token sequences
            data = torch.randint(
                0, self.config.vocab_size, (batch_size, seq_len + 1), device=self.device
            )
            inputs = data[:, :-1]
            targets = data[:, 1:]

            optimizer.zero_grad()
            logits = self.model(inputs)

            # Reshape for cross entropy: [batch * seq, vocab_size] vs [batch * seq]
            loss = criterion(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = float(loss.item())
            final_loss = float(loss.item())

        elapsed = time.time() - t0
        tok_per_sec = total_tokens / max(elapsed, 1e-5)

        peak_vram = (
            torch.cuda.max_memory_allocated(device=self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )

        return PretrainMetrics(
            steps=steps,
            initial_loss=round(initial_loss, 4),
            final_loss=round(final_loss, 4),
            total_tokens_processed=total_tokens,
            tokens_per_second=round(tok_per_sec, 1),
            peak_vram_mb=round(peak_vram, 2),
            elapsed_seconds=round(elapsed, 3),
        )
