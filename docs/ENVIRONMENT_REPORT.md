# Hardware & Environment Discovery Report (Phase 0)

**Date:** 2026-11-24\
**Auditor:** ML Systems Engineering Agent\
**Target Repository:** `c:\Users\24406601057\Desktop\Asistan`

---

## 1. System & OS Architecture
- **Operating System:** Windows 10 (Build 10.0.26300-SP0) 64-bit
- **Python Version:** 3.11.9 (64-bit)
- **Virtual Environment Path:** `c:\Users\24406601057\Desktop\Asistan\.venv`

## 2. Memory & Storage Metrics
- **Total Physical RAM:** 15.25 GB
- **Initial Available RAM:** ~1.99 GB (System paging observed under heavy browser/IDE load)
- **Disk Space (C:):** Total 475.86 GB, Free 213.51 GB (Adequate for checkpoints and datasets)

## 3. GPU & Accelerator Discovery
- **GPU Model:** NVIDIA GeForce RTX 4050 Laptop GPU
- **VRAM Total:** 6,141 MiB (~6.0 GB)
- **Driver Version:** 566.26
- **System CUDA Support:** CUDA 12.7
- **Inference Runtime:** Ollama (Port 11434 active)
- **Installed Ollama Models:**
  - `qwen2.5:7b-instruct-q4_K_M` (4.7 GB)
  - `deepseek-r1:7b` (4.7 GB)

## 4. PyTorch & CUDA Bridge Status
- **Initial State:** `torch 2.14.1+cpu` (`torch.cuda.is_available() == False`)
- **Remediation Action:** Upgrading virtual environment PyTorch to CUDA 12.1 enabled build (`torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`).
- **Post-Install Target:** Full CUDA tensor allocation and mixed-precision (FP16/BF16) verification.

## 5. Constraints & Guardrails
- **VRAM Allocation Budget:** Maximum 5.0 GB allocated to ML tasks to avoid Windows DWM/Display driver OOM crashes.
- **Inference Strategy:** 4-bit / 8-bit quantized execution for 7B models.
- **Training Strategy:** LoRA / QLoRA with rank $r \in [8, 16]$, small batch size ($b=1$ or $2$), gradient accumulation steps ($\ge 8$).
