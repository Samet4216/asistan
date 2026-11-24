"""Integration smoke test for LoRA training loop and checkpoint generation (Phase 5).
"""

from pathlib import Path
import torch
import pytest

from personal_ai.training.lora_trainer import LoRATrainer


def test_lora_training_smoke_test(tmp_path: Path):
    """Verify end-to-end forward, backward, optimizer step, and checkpoint export."""
    ckpt_dir_str = str(tmp_path / 'checkpoints').replace('\\', '/')
    cfg_file = tmp_path / "training.yaml"
    cfg_file.write_text(f"""
training:
  output_dir: "{ckpt_dir_str}"
  max_vram_gb: 5.0
  lora:
    r: 4
    lora_alpha: 8
    target_modules: ["q_proj", "v_proj"]
""", encoding="utf-8")

    trainer = LoRATrainer(config_path=cfg_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = trainer.run_smoke_test(device=device, steps=3)

    assert metrics.steps_completed == 3
    assert metrics.elapsed_seconds > 0.0

    # Verify adapter checkpoint files were written to disk
    ckpt_dir = Path(metrics.checkpoint_dir)
    assert ckpt_dir.exists()
    assert (ckpt_dir / "adapter_config.json").exists()
    assert (ckpt_dir / "adapter_model.safetensors").exists() or (ckpt_dir / "adapter_model.bin").exists()
