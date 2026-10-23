---
title: GPU Memory Hierarchy & VRAM Allocation
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- gpu
- vram
- sram
- cuda
- hardware
---

# GPU Memory Hierarchy & Deep Learning Workstation Constraints

Deep learning workloads on consumer workstations (e.g., NVIDIA RTX 4050 6GB Laptop GPU) are constrained by GPU memory architecture:

## 📐 Memory Tier Bandwidth & Latency
1. **SRAM (On-Chip Register / Shared Memory):** $\approx 19 \text{ TB/s}$, tens of KB per Streaming Multiprocessor (SM). Extremely low latency.
2. **HBM / GDDR6 VRAM:** $\approx 200 - 900 \text{ GB/s}$, 6 GB total capacity. Holds model weights, activations, and KV cache.
3. **Host System RAM (DDR5):** $\approx 50 \text{ GB/s}$, 16 GB capacity. Used by Asistan for BGE embeddings and BM25.
4. **PCIe Bus (Gen 4 x8/x16):** $\approx 16 - 32 \text{ GB/s}$ transfer bottleneck between CPU and GPU.

## 🔗 Strategic Optimizations
- IO-Aware Tiling: [[FlashAttention]] and [[2205.14135]]
- Quantized LLMs: `qwen2.5:7b-instruct-q4_K_M` (~4.1 GB VRAM)
- CPU Offloading: Embeddings (`bge-base-en-v1.5`) & Reranker run entirely on CPU

