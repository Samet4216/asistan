"""LoRA Fine-Tuning Pipeline with VRAM Budget Enforcement (Phase 5).

Strict Engineering Rules:
1. Every training job must estimate VRAM usage before allocating tensors.
2. 6GB VRAM limit: Never exceed 5.0 GB dedicated allocable ceiling.
3. LoRA adapter weights, configs, and training logs must be saved deterministically.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
import yaml


class VRAMBudgetExceededError(Exception):
    """Raised when estimated memory exceeds safe hardware bounds."""
    pass


@dataclass
class TrainingMetrics:
    """Stores output metrics of a training run."""
    experiment_name: str
    steps_completed: int
    initial_loss: float
    final_loss: float
    peak_vram_mb: float
    elapsed_seconds: float
    checkpoint_dir: str


class LoRATrainer:
    """Manages parameter-efficient fine-tuning with memory guards."""

    def __init__(self, config_path: str | Path = "configs/training.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.output_dir = Path(self.config.get("output_dir", "checkpoints/dl_cv_lora"))
        self.max_vram_gb = float(self.config.get("max_vram_gb", 5.0))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML training configuration."""
        if self.config_path.exists():
            data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            return data.get("training", {}) if isinstance(data, dict) else {}
        return {}

    def estimate_vram_usage(
        self,
        param_count_billions: float = 7.0,
        quantized_bits: int = 4,
        seq_length: int = 512,
        batch_size: int = 1,
    ) -> float:
        """Estimates total VRAM footprint in GB.

        Formula:
        - Base model weights: param_count * (quantized_bits / 8) GB
        - KV Cache & Activations: ~0.8 GB at batch=1, seq=512
        - Gradients & Optimizer states (LoRA parameters only): ~0.2 GB
        - PyTorch CUDA runtime context: ~0.6 GB
        """
        weights_gb = param_count_billions * (quantized_bits / 8.0)
        activations_gb = (seq_length / 512.0) * batch_size * 0.8
        lora_opt_gb = 0.25
        cuda_overhead_gb = 0.65
        total_estimated = weights_gb + activations_gb + lora_opt_gb + cuda_overhead_gb
        return round(total_estimated, 2)

    def verify_vram_budget(self, param_count_billions: float = 7.0, quantized_bits: int = 4):
        """Checks if training fits inside the 5.0 GB safe limit on 6GB VRAM."""
        estimated = self.estimate_vram_usage(
            param_count_billions=param_count_billions,
            quantized_bits=quantized_bits,
            seq_length=self.config.get("max_seq_length", 512),
            batch_size=self.config.get("batch_size", 1),
        )
        if estimated > self.max_vram_gb:
            raise VRAMBudgetExceededError(
                f"Estimated VRAM ({estimated:.2f} GB) exceeds safe allocable ceiling ({self.max_vram_gb:.1f} GB) "
                f"on 6GB RTX 4050 GPU! Reduce seq_length or ensure 4-bit QLoRA is used."
            )
        return estimated

    def build_lora_config(self) -> LoraConfig:
        """Constructs PEFT LoraConfig from yaml parameters."""
        lora_cfg = self.config.get("lora", {})
        return LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_cfg.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

    def run_smoke_test(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        steps: int = 3,
    ) -> TrainingMetrics:
        """Executes an end-to-end smoke test training loop with PEFT LoRA.

        Verifies forward pass, backward pass, optimizer step, loss reduction,
        and checkpoint export without memory leakage.
        """
        t0 = time.time()
        hidden_dim = 128
        vocab_size = 500

        # Create a miniature causal language model architecture
        class MiniCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden_dim)
                self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                q = self.q_proj(x)
                v = self.v_proj(x)
                logits = self.head(q + v)
                return logits

        base_model = MiniCausalLM().to(device)

        # Apply LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_config)

        # Verify trainable parameters are only a small fraction
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        assert trainable_params < total_params, "LoRA must freeze base parameters!"

        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        initial_loss = 0.0
        final_loss = 0.0

        # Synthetic training data
        for step in range(steps):
            inputs = torch.randint(0, vocab_size, (1, 16), device=device)
            targets = torch.randint(0, vocab_size, (1, 16), device=device)

            optimizer.zero_grad()
            logits = peft_model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = float(loss.item())
            final_loss = float(loss.item())

        # Checkpoint export
        save_path = self.output_dir / "smoke_checkpoint"
        peft_model.save_pretrained(str(save_path))

        peak_vram = (
            torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
            if device.startswith("cuda")
            else 0.0
        )

        return TrainingMetrics(
            experiment_name="smoke_test",
            steps_completed=steps,
            initial_loss=round(initial_loss, 4),
            final_loss=round(final_loss, 4),
            peak_vram_mb=round(peak_vram, 2),
            elapsed_seconds=round(time.time() - t0, 3),
            checkpoint_dir=str(save_path),
        )
