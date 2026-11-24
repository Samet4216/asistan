"""Hardware & CUDA environment verification tests (Phase 0)."""

import pytest
import psutil
import torch


def test_system_ram_visibility():
    """Verify system RAM is accurately reported and accessible."""
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    assert total_gb >= 14.0, f"Expected at least 14 GB RAM, got {total_gb:.2f} GB"


def test_cuda_device_availability():
    """Verify PyTorch is compiled with CUDA and detects RTX 4050."""
    assert torch.cuda.is_available(), "PyTorch CUDA is not available! Must install CUDA-enabled build."
    device_count = torch.cuda.device_count()
    assert device_count >= 1, f"Expected at least 1 CUDA device, got {device_count}"
    
    device_name = torch.cuda.get_device_name(0)
    assert "RTX 4050" in device_name or "GeForce" in device_name, f"Unexpected GPU: {device_name}"


def test_vram_capacity():
    """Verify GPU VRAM is approximately 6 GB."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    assert 5.0 <= vram_gb <= 7.0, f"Expected ~6 GB VRAM, got {vram_gb:.2f} GB"


def test_cuda_tensor_smoke():
    """Verify basic tensor allocation and matrix multiplication on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    y = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    z = torch.matmul(x, y)
    assert z.shape == (512, 512)
    assert not torch.isnan(z).any()
    torch.cuda.empty_cache()
