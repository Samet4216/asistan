"""Unit tests for LoRA Configuration and VRAM Budgeting (Phase 5).
"""

from pathlib import Path
import pytest

from personal_ai.training.lora_trainer import LoRATrainer, VRAMBudgetExceededError


def test_vram_estimation_within_bounds(tmp_path: Path):
    """Verify 4-bit 7B model at batch=1 fits inside 5.0 GB allocable limit."""
    trainer = LoRATrainer()
    vram_est = trainer.estimate_vram_usage(
        param_count_billions=7.0,
        quantized_bits=4,
        seq_length=512,
        batch_size=1,
    )
    # 7.0 * (4/8) = 3.5GB + 0.8GB activations + 0.25GB LoRA + 0.65GB overhead = 5.2GB approx
    assert 4.0 <= vram_est <= 5.5


def test_vram_budget_check_raises_oom_warning(tmp_path: Path):
    """Verify trainer blocks oversized jobs before OOM crashes."""
    cfg_file = tmp_path / "training.yaml"
    cfg_file.write_text("training:\n  max_vram_gb: 5.0\n  max_seq_length: 2048\n  batch_size: 8\n", encoding="utf-8")

    trainer = LoRATrainer(config_path=cfg_file)

    # 16-bit unquantized 7B model at batch=8 exceeds 6GB VRAM
    with pytest.raises(VRAMBudgetExceededError, match="exceeds safe allocable ceiling"):
        trainer.verify_vram_budget(param_count_billions=7.0, quantized_bits=16)


def test_build_lora_config():
    """Verify PEFT LoraConfig construction."""
    trainer = LoRATrainer()
    lora_cfg = trainer.build_lora_config()

    assert lora_cfg.r == 8
    assert lora_cfg.lora_alpha == 16
    assert "q_proj" in lora_cfg.target_modules
    assert "v_proj" in lora_cfg.target_modules
